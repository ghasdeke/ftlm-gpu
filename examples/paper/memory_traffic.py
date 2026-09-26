"""Memory traffic of one Lanczos step on the GPU (paper, Section 3.4).

Usage:  python memory_traffic.py [out.json]

For the M = 0 sectors of the s = 2 icosahedron and the s = 1/2
icosidodecahedron (B = 4 chains), measures with CUDA events

  t_spmv   one matrix-free SpMV
  t_vec    the vector operations of one Lanczos step, launched as in
           GpuLanczos.block_lanczos: dot product <v, w> (2 passes over the
           B-column vectors), fused orthogonalization and norm (4 passes),
           rescaling (2 passes), and the tree reductions of the partial sums

and, for reference, the host wall time t_step of one step (difference of
runs with 26 and 6 steps), which additionally contains the host
synchronization of the recursion coefficients.  The times are related to
byte counts per step (element size e):

  vector operations   8 dim B e
  SpMV, requested     dim (4 [label, CLT] + 2 B e [own row, output])
                      + n_off (B e [gathered values] + 8 [CLT mask + base])
  SpMV, sectors       as 'requested', but every gather of B values and each
                      of the two CLT reads occupies whole 32-byte sectors
                      (no reuse of fetched sectors)
  SpMV, compulsory    dim (4 [label, CLT] + 2 B e) + |CLT|
                      every array read once (ideal cache)

n_off is the exact number of off-diagonal matrix elements of the sector.
The device-to-device copy bandwidth is measured for reference.  Requires
CuPy (see python/README).
"""

# ================================================================
# Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
# and Helmholtz-Zentrum Dresden-Rossendorf e.V.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================================================

import json
import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from ftlm_gpu import Model, enumerate_sector, sectors  # noqa: E402
from ftlm_gpu.gpu import REDUCE_BS, SPMV_BS, GpuLanczos  # noqa: E402

B = 4
CASES = [("ico_s2_M0", "clt", "double"), ("ico_s2_M0", "clt", "single"),
         ("ico_s2_M0", "clt", "half"), ("ico_s2_M0", "cr", "single"),
         ("icosid_M0", "clt", "double"), ("icosid_M0", "clt", "single"),
         ("icosid_M0", "clt", "half"), ("icosid_M0", "cr", "single")]
ELEM = {"double": 8, "single": 4, "half": 2}


def model_of(key):
    return Model.preset("icosid", s=0.5) if key == "icosid_M0" else Model.preset("ico", s=2.0)


def count_offdiag(m, A):
    """Exact number of off-diagonal elements (both flip directions) of sector A."""
    n_off = 0
    for i, j, _ in m.couplings:
        i, j = int(i), int(j)
        poly = np.ones(1, dtype=np.int64)
        for k in range(m.N):
            if k not in (i, j):
                poly = np.convolve(poly, np.ones(int(m.radix[k]), dtype=np.int64))
        for ai in range(int(m.two_s[i]) + 1):
            for aj in range(int(m.two_s[j]) + 1):
                rest = A - ai - aj
                if not 0 <= rest < poly.size:
                    continue
                cnt = int(poly[rest])
                if ai > 0 and aj < m.two_s[j]:
                    n_off += cnt
                if ai < m.two_s[i] and aj > 0:
                    n_off += cnt
    return n_off


def event_time(f, n=5):
    ts = []
    for _ in range(n):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        f()
        b.record()
        b.synchronize()
        ts.append(cp.cuda.get_elapsed_time(a, b) / 1e3)
    return float(np.median(ts))


def main():
    out_file = sys.argv[1] if len(sys.argv) > 1 else "memory_traffic.json"
    props = cp.cuda.runtime.getDeviceProperties(0)
    x = cp.empty(2 * 1024 ** 3 // 4, dtype=cp.float32)
    y = cp.empty_like(x)
    bw_copy = 2 * x.nbytes / event_time(lambda: cp.copyto(y, x), 7) / 1e9
    del x, y
    cp.get_default_memory_pool().free_all_blocks()
    info = {"device": props["name"].decode(), "l2_MB": props["l2CacheSize"] / 2 ** 20,
            "bw_copy_GBs": bw_copy,
            "bw_nominal_GBs": 2 * props["memoryClockRate"] * 1e3 * props["memoryBusWidth"] / 8 / 1e9,
            "B": B, "runs": []}
    print(f"{info['device']}: L2 {info['l2_MB']:.0f} MB, copy bandwidth {bw_copy:.0f} GB/s "
          f"(nominal {info['bw_nominal_GBs']:.0f} GB/s)")
    cache = {}
    for key, lookup, prec in CASES:
        if key not in cache:
            m = model_of(key)
            sec = sectors(m, True)[0]
            cache = {key: (m, sec, enumerate_sector(m, sec.A), count_offdiag(m, sec.A))}
        m, sec, basis, n_off = cache[key]
        eng = GpuLanczos(m, sec.A, lookup, prec, B, basis if lookup == "clt" else None)
        rng = np.random.default_rng(1)
        V0 = rng.standard_normal((sec.dim, B)).astype(np.float64 if prec == "double" else np.float32)
        eng.block_lanczos(V0, 3)                      # warm-up; leaves normalized vectors
        t_spmv = event_time(lambda: eng._spmv(eng.w, eng.v, B))
        # vector operations with coefficients that leave the vectors unchanged
        eng.d_alpha.fill(0)
        eng.d_beta_prev.fill(0)
        eng.d_beta.fill(1)
        ni, Bi = np.int32(sec.dim), np.int32(B)
        rb, bl = (eng.rblocks,), (eng.blocks,)

        def reduce_passes():
            src, buf, n = eng.partial, eng.partial2, eng.rblocks
            while n > 1:
                nb = (n + REDUCE_BS - 1) // REDUCE_BS
                eng.f["reduce"]((nb,), (REDUCE_BS,), (buf, src, np.int32(n), Bi))
                n = nb
                src, buf = buf, src

        def vec_ops():
            eng.f["dot"](rb, (REDUCE_BS,), (eng.partial, eng.v, eng.w, ni, Bi))
            reduce_passes()
            eng.f["ortho"](rb, (REDUCE_BS,), (eng.w, eng.v, eng.vp, eng.d_alpha, eng.d_beta_prev,
                                              eng.partial, ni, Bi, np.int32(1)))
            reduce_passes()
            eng.f["scale"](bl, (SPMV_BS,), (eng.w, eng.d_beta, ni, Bi))
        t_vec = event_time(vec_ops, 7)
        tt = {}
        for M in (6, 26):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            eng.block_lanczos(V0, M)
            tt[M] = time.perf_counter() - t0
        t_step = (tt[26] - tt[6]) / 20
        e, dim, clt = ELEM[prec], sec.dim, lookup == "clt"
        clt_bytes = 8 * ((int(m.D_full) + 31) // 32) if clt else 0
        vec = 8 * dim * B * e
        row = dim * (4 * clt + 2 * B * e)
        spmv_requested = row + n_off * (B * e + 8 * clt)
        spmv_sectors = row + n_off * (32 * math.ceil(B * e / 32) + 64 * clt)
        spmv_compulsory = row + clt_bytes
        r = dict(system=key, lookup=lookup, precision=prec, dim=int(dim),
                 offdiag_per_row=n_off / dim, t_spmv=t_spmv, t_vec=t_vec,
                 t_step_host=t_step, host_overhead=1 - (t_spmv + t_vec) / t_step,
                 spmv_fraction=t_spmv / (t_spmv + t_vec),
                 vec_GBs=vec / t_vec / 1e9,
                 spmv_noreuse_GB=spmv_requested / 1e9, spmv_noreuse_GBs=spmv_requested / t_spmv / 1e9,
                 spmv_sectors_GB=spmv_sectors / 1e9, spmv_sectors_GBs=spmv_sectors / t_spmv / 1e9,
                 spmv_compulsory_GB=spmv_compulsory / 1e9,
                 spmv_compulsory_GBs=spmv_compulsory / t_spmv / 1e9)
        print(f"{key:10s} {lookup} {prec:6s} SpMV {1e3 * t_spmv:7.1f} ms, vector ops {1e3 * t_vec:6.1f} ms "
              f"({r['vec_GBs']:.0f} GB/s), host step {1e3 * t_step:7.1f} ms; SpMV "
              f"{r['spmv_compulsory_GBs']:.0f} GB/s compulsory / {r['spmv_noreuse_GBs']:.0f} requested / "
              f"{r['spmv_sectors_GBs']:.0f} sectors")
        info["runs"].append(r)
        eng.free()
        del eng
        cp.get_default_memory_pool().free_all_blocks()
    with open(out_file, "w") as fh:
        json.dump(info, fh, indent=1)
    print("wrote", out_file)


if __name__ == "__main__":
    main()
