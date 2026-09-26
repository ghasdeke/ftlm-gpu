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
"""GPU backend: CuPy/NVRTC driver for the kernels in cuda/ftlm_kernels.cuh.

The host-side block Lanczos loop mirrors ftlm_gpu_mex.cu step by step
(same kernels, same launch configuration, same host arithmetic in the
compute precision), so that MATLAB and Python produce identical Lanczos
coefficients for identical start vectors.
"""

import ctypes
import functools
from pathlib import Path

import numpy as np

from . import basis as _basis

try:
    import cupy as cp
except ImportError:  # pragma: no cover - reported when the backend is used
    cp = None

MAX_SITES = 32
MAX_COUPLINGS = 512
MAX_B = 16
LUT_W = 16
MAX_TWO_S = 15
LUT_SIZE = LUT_W * LUT_W
SPMV_BS = 256
REDUCE_BS = 256
LOOKUP_CLT, LOOKUP_CR = 0, 1
BREAKDOWN_C = {"double": 64.0, "single": 64.0, "half": 8.0, "bfloat16": 8.0}

# storage type name in the kernels, compute dtype, bytes per entry, unit roundoff
PRECISIONS = {
    "double":   ("double",    np.float64, 8, 2.0 ** -53),
    "single":   ("float",     np.float32, 4, 2.0 ** -24),
    "half":     ("ftlm_fp16", np.float32, 2, 2.0 ** -11),
    "bfloat16": ("ftlm_bf16", np.float32, 2, 2.0 ** -8),
}


class FtlmConst(ctypes.Structure):
    """ctypes mirror of struct FtlmConst in cuda/ftlm_kernels.cuh."""
    _fields_ = [
        ("J_d", ctypes.c_double * MAX_COUPLINGS),
        ("hJ_d", ctypes.c_double * MAX_COUPLINGS),
        ("s_d", ctypes.c_double * MAX_SITES),
        ("J_f", ctypes.c_float * MAX_COUPLINGS),
        ("hJ_f", ctypes.c_float * MAX_COUPLINGS),
        ("s_f", ctypes.c_float * MAX_SITES),
        ("lut_raise_d", ctypes.c_double * LUT_SIZE),
        ("lut_lower_d", ctypes.c_double * LUT_SIZE),
        ("lut_raise_f", ctypes.c_float * LUT_SIZE),
        ("lut_lower_f", ctypes.c_float * LUT_SIZE),
        ("ci", ctypes.c_int * MAX_COUPLINGS),
        ("cj", ctypes.c_int * MAX_COUPLINGS),
        ("two_s", ctypes.c_int * MAX_SITES),
        ("radix", ctypes.c_int * MAX_SITES),
        ("power", ctypes.c_int * MAX_SITES),
        ("shift", ctypes.c_int * MAX_SITES),
        ("mask", ctypes.c_int * MAX_SITES),
        ("N", ctypes.c_int),
        ("n_coup", ctypes.c_int),
        ("A_total", ctypes.c_int),
        ("dcum_pstride", ctypes.c_int),
        ("dcum_astride", ctypes.c_int),
        ("dcum_size", ctypes.c_int),
        ("pad0", ctypes.c_int),
        ("pad1", ctypes.c_int),
    ]


def kernel_source_path():
    """Location of ftlm_kernels.cuh (installed package or source tree)."""
    here = Path(__file__).resolve().parent
    for cand in (here / "cuda" / "ftlm_kernels.cuh",
                 here.parents[1] / "cuda" / "ftlm_kernels.cuh"):
        if cand.is_file():
            return cand
    raise FileNotFoundError("cuda/ftlm_kernels.cuh not found")


@functools.lru_cache(maxsize=None)
def _module(prec):
    if cp is None:
        raise ImportError("the GPU backend requires CuPy (pip install cupy-cuda12x)")
    ts, _, _, _ = PRECISIONS[prec]
    tc = "double" if prec == "double" else "float"
    names = {
        "spmv_clt_0": f"ftlm_spmv<{ts}, {tc}, 0, 0>",
        "spmv_clt_1": f"ftlm_spmv<{ts}, {tc}, 0, 1>",
        "spmv_cr_0": f"ftlm_spmv<{ts}, {tc}, 1, 0>",
        "spmv_cr_1": f"ftlm_spmv<{ts}, {tc}, 1, 1>",
        "import_f": f"ftlm_import<{ts}, {tc}, float>",
        "import_d": f"ftlm_import<{ts}, {tc}, double>",
        "export_d": f"ftlm_export<{ts}, {tc}, double>",
        "dot": f"ftlm_dot_partial<{ts}, {tc}>",
        "reduce": f"ftlm_reduce_tree<{tc}>",
        "ortho": f"ftlm_ortho_norm_partial<{ts}, {tc}>",
        "scale": f"ftlm_scale<{ts}, {tc}>",
    }
    src = kernel_source_path().read_text(encoding="utf-8")
    mod = cp.RawModule(code=src, options=("-std=c++14",),
                       name_expressions=list(names.values()) + ["ftlm_const_layout"])
    # layout check of the ctypes mirror
    out = cp.zeros(8, dtype=cp.int32)
    mod.get_function("ftlm_const_layout")((1,), (1,), (out,))
    lay = out.get()
    exp = [ctypes.sizeof(FtlmConst), FtlmConst.J_f.offset, FtlmConst.lut_raise_d.offset,
           FtlmConst.ci.offset, FtlmConst.two_s.offset, FtlmConst.N.offset,
           FtlmConst.dcum_size.offset]
    if list(lay[:7]) != exp:
        raise RuntimeError(f"FtlmConst layout mismatch: device {lay[:7]}, host {exp}")
    funcs = {k: mod.get_function(v) for k, v in names.items()}
    return mod, funcs


def _fill_lut(h):
    for ts in range(MAX_TWO_S + 1):
        s = 0.5 * ts
        s_f = np.float32(0.5) * np.float32(ts)
        ss1f = s_f * (s_f + np.float32(1))
        for a in range(min(ts, LUT_W - 1) + 1):
            k = ts * LUT_W + a
            m = a - s
            mf = np.float32(a) - s_f
            if a < ts:
                h.lut_raise_d[k] = np.sqrt(s * (s + 1) - m * (m + 1))
                h.lut_raise_f[k] = np.sqrt(ss1f - mf * (mf + np.float32(1)))
            if a > 0:
                h.lut_lower_d[k] = np.sqrt(s * (s + 1) - m * (m - 1))
                h.lut_lower_f[k] = np.sqrt(ss1f - mf * (mf - np.float32(1)))


class GpuLanczos:
    """Matrix-free block Lanczos for one sector on the GPU.

    Parameters
    ----------
    model : Model
    A : int
        digit sum of the sector
    lookup : 'clt' | 'cr'
    precision : 'single' | 'double' | 'half' | 'bfloat16'
    B : int
        maximum number of chains per call (1..16)
    basis : int32 array, required for 'clt'
    """

    def __init__(self, model, A, lookup="clt", precision="single", B=8, basis=None):
        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision {precision!r}")
        if not 1 <= B <= MAX_B:
            raise ValueError(f"B = {B} outside [1, {MAX_B}]")
        self.prec = precision
        self.ts_name, self.tc, self.elem, self.u = PRECISIONS[precision]
        self.lookup = {"clt": LOOKUP_CLT, "cr": LOOKUP_CR}[lookup]
        self.B_max = B
        self.mod, self.f = _module(precision)

        h = FtlmConst()
        N = model.N
        h.N = N
        for k in range(N):
            h.two_s[k] = int(model.two_s[k])
            h.radix[k] = int(model.radix[k])
            h.s_d[k] = 0.5 * int(model.two_s[k])
            h.s_f[k] = np.float32(0.5) * np.float32(int(model.two_s[k]))
        C = model.couplings
        h.n_coup = C.shape[0]
        for c, (i, j, J) in enumerate(C):
            h.ci[c], h.cj[c] = int(i), int(j)
            h.J_d[c], h.hJ_d[c] = J, 0.5 * J
            h.J_f[c] = np.float32(J)
            h.hJ_f[c] = np.float32(0.5) * np.float32(J)
        _fill_lut(h)

        self.block_base = self.block_mask = self.basis = self.dcum = None
        self.dcum_size = 0
        if self.lookup == LOOKUP_CLT:
            if basis is None:
                basis = _basis.enumerate_sector(model, A)
            bb, bm = _basis.build_clt(basis, model.D_full)
            self.block_base = cp.asarray(bb)
            self.block_mask = cp.asarray(bm)
            self.basis = cp.asarray(np.asarray(basis, dtype=np.int32))
            self.dim = int(basis.size)
            for k in range(N):
                h.power[k] = int(model.power[k])
        else:
            if not model.cr_ok:
                raise ValueError("lookup 'cr': the packed state needs more than 64 bits")
            dcum, ps, as_, dim = _basis.cr_tables(model, A)
            # dynamic D_c table plus the static ladder-factor tables of ftlm_spmv
            lut_bytes = 2 * LUT_SIZE * (8 if precision == "double" else 4)
            if dcum.size * 4 + lut_bytes > 48 * 1024:
                raise ValueError("D_c table plus ladder-factor tables exceed 48 kB of shared memory")
            self.dcum = cp.asarray(dcum)
            self.dcum_size = int(dcum.size)
            self.dim = dim
            for k in range(N):
                h.shift[k] = int(model.shift[k])
                h.mask[k] = (1 << int(model.bits[k])) - 1
            h.A_total, h.dcum_pstride, h.dcum_astride = A, ps, as_
            h.dcum_size = self.dcum_size
        if not 1 <= self.dim < 2 ** 31:
            raise ValueError(f"dim = {self.dim} outside [1, 2^31-1]")
        self.uni = int(np.all(model.two_s == model.two_s[0]))
        self.const = h
        self.mod.get_global("c_p").copy_from_host(ctypes.addressof(h), ctypes.sizeof(h))

        n = self.dim
        self.sigma = np.sqrt(float(n)) if self.elem == 2 else 1.0
        vdt = np.float64 if precision == "double" else (
            np.float32 if precision == "single" else np.uint16)
        self.v = cp.empty(n * B, dtype=vdt)
        self.vp = cp.empty(n * B, dtype=vdt)
        self.w = cp.empty(n * B, dtype=vdt)
        self.tmp = cp.empty(n, dtype=np.float64)
        self.rblocks = (n + REDUCE_BS - 1) // REDUCE_BS
        self.blocks = (n + SPMV_BS - 1) // SPMV_BS
        self.partial = cp.empty(self.rblocks * B, dtype=self.tc)
        self.partial2 = cp.empty(((self.rblocks + REDUCE_BS - 1) // REDUCE_BS) * B, dtype=self.tc)
        self.d_alpha = cp.empty(MAX_B, dtype=self.tc)
        self.d_beta = cp.empty(MAX_B, dtype=self.tc)
        self.d_beta_prev = cp.zeros(MAX_B, dtype=self.tc)

    # ------------------------------------------------------------------
    def _spmv(self, W, V, B):
        n = np.int32(self.dim)
        if self.lookup == LOOKUP_CLT:
            self.f[f"spmv_clt_{self.uni}"]((self.blocks,), (SPMV_BS,),
                                           (W, V, self.block_base, self.block_mask,
                                            self.basis, 0, n, np.int32(B)))
        else:
            self.f[f"spmv_cr_{self.uni}"]((self.blocks,), (SPMV_BS,),
                                          (W, V, 0, 0, 0, self.dcum, n, np.int32(B)),
                                          shared_mem=self.dcum_size * 4)

    def _import(self, V0):
        """Import start vectors (dim x B array, or B x dim with chains as rows)."""
        V0 = np.asarray(V0)
        if V0.ndim == 1:
            V0 = V0[:, None]
        if V0.shape[0] != self.dim:
            raise ValueError(f"V0 has {V0.shape[0]} rows, expected {self.dim}")
        B = V0.shape[1]
        if not 1 <= B <= self.B_max:
            raise ValueError(f"B = {B} outside [1, {self.B_max}]")
        n = np.int32(self.dim)
        for b in range(B):
            col = V0[:, b]
            if col.dtype == np.float32:
                buf = self.tmp.view(np.float32)[: self.dim]
                buf.set(np.ascontiguousarray(col))
                self.f["import_f"]((self.blocks,), (SPMV_BS,),
                                   (self.v, buf, n, np.int32(B), np.int32(b), np.int32(1)))
            else:
                self.tmp.set(np.ascontiguousarray(col, dtype=np.float64))
                self.f["import_d"]((self.blocks,), (SPMV_BS,),
                                   (self.v, self.tmp, n, np.int32(B), np.int32(b), np.int32(1)))
        return B

    def spmv(self, V):
        """H V in the configured precision (host float64 in/out; for tests)."""
        B = self._import(V)
        self._spmv(self.w, self.v, B)
        out = np.empty((self.dim, B))
        for b in range(B):
            self.f["export_d"]((self.blocks,), (SPMV_BS,),
                               (self.tmp, self.w, np.int32(self.dim), np.int32(B),
                                np.int32(b), np.int32(1)))
            out[:, b] = self.tmp.get()
        return out

    def block_lanczos(self, V0, M_lz):
        """Run B independent Lanczos chains; returns (alpha, beta, nsteps).

        alpha, beta: (n_max x B) float64, zero beyond nsteps[b];
        beta[j, b] is the norm of the residual after step j.
        """
        tc = self.tc
        n = self.dim
        B = self._import(V0)
        M_lz = min(int(M_lz), n)
        ni, Bi = np.int32(n), np.int32(B)
        rb, bl = (self.rblocks,), (self.blocks,)
        sigma = tc(self.sigma)
        sigma2 = sigma * sigma
        one = tc(1)

        def reduce_to_host(dst):
            # repeated tree passes (same as reduce_partials in ftlm_gpu_mex.cu)
            src, buf, m = self.partial, self.partial2, self.rblocks
            while m > 1:
                nb = (m + REDUCE_BS - 1) // REDUCE_BS
                self.f["reduce"]((nb,), (REDUCE_BS,), (buf, src, np.int32(m), Bi))
                m = nb
                src, buf = buf, src
            dst[:B] = src[:B]
            return dst[:B].get()

        # normalize each chain to norm sigma
        self.f["dot"](rb, (REDUCE_BS,), (self.partial, self.v, self.v, ni, Bi))
        nrm2 = reduce_to_host(self.d_alpha)
        self.d_alpha[:B].set((sigma / np.sqrt(nrm2)).astype(tc))
        self.f["scale"](bl, (SPMV_BS,), (self.v, self.d_alpha, ni, Bi))
        self.vp.fill(0)

        AL = np.zeros((M_lz, B))
        BE = np.zeros((M_lz, B))
        nsteps = np.full(B, M_lz, dtype=np.int64)
        active = np.ones(B, dtype=bool)
        normT = np.zeros(B)
        beta_prev = np.zeros(B, dtype=tc)
        tol_fac = BREAKDOWN_C[self.prec] * self.u
        pv, pvp, pw = self.v, self.vp, self.w

        for j in range(M_lz):
            self._spmv(pw, pv, B)
            self.f["dot"](rb, (REDUCE_BS,), (self.partial, pv, pw, ni, Bi))
            alpha = (reduce_to_host(self.d_alpha) / sigma2).astype(tc)
            AL[j, active] = alpha[active]
            if sigma2 != one:
                self.d_alpha[:B].set(alpha)
            if j > 0:
                self.d_beta_prev[:B].set(beta_prev)
            self.f["ortho"](rb, (REDUCE_BS,),
                            (pw, pv, pvp, self.d_alpha, self.d_beta_prev, self.partial,
                             ni, Bi, np.int32(1 if j > 0 else 0)))
            beta = (np.sqrt(reduce_to_host(self.d_beta)) / sigma).astype(tc)
            if not (np.all(np.isfinite(alpha[active])) and np.all(np.isfinite(beta[active]))):
                raise FloatingPointError(
                    f"non-finite Lanczos coefficient in step {j + 1} (FP16 range exceeded? "
                    "Rescale the couplings so that |J| is of order 1)")
            scale = np.zeros(B, dtype=tc)
            for b in range(B):
                if not active[b]:
                    continue
                BE[j, b] = beta[b]
                tj = abs(float(alpha[b])) + float(beta[b]) + float(beta_prev[b])
                normT[b] = max(normT[b], tj)
                if float(beta[b]) <= tol_fac * normT[b]:
                    active[b] = False
                    nsteps[b] = j + 1
                else:
                    scale[b] = one / beta[b]
            if not active.any() or j == M_lz - 1:
                break
            self.d_beta[:B].set(scale)
            self.f["scale"](bl, (SPMV_BS,), (pw, self.d_beta, ni, Bi))
            pvp, pv, pw = pv, pw, pvp
            beta_prev = np.where(active, beta, tc(0)).astype(tc)

        cp.cuda.runtime.deviceSynchronize()
        self.v, self.vp, self.w = pv, pvp, pw
        n_max = int(nsteps.max())
        return AL[:n_max], BE[:n_max], nsteps

    def free(self):
        for name in ("v", "vp", "w", "tmp", "partial", "partial2", "block_base", "block_mask",
                     "basis", "dcum"):
            setattr(self, name, None)
        cp.get_default_memory_pool().free_all_blocks()
