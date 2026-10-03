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
"""Sector-resolved FTLM: driver, Ritz data, thermodynamic observables."""

import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

from . import basis as _basis
from .model import Model

DEFAULTS = dict(
    backend="gpu",          # 'gpu' | 'cpu' (explicit sparse matrix, small systems)
    precision="single",     # gpu: single|double|half|bfloat16; cpu: double|single
    lookup="clt",           # 'clt' | 'cr' (gpu only)
    use_cpu_reference=False,
    cpu_precision="double",
    only_M0=False,
    ed_thresh=1000,         # exact diagonalization for dim <= ed_thresh (if prod(2s+1) <= 2^31)
    seed=0,
    B_gpu=0,
    B_cpu=8,
    L2_cache_bytes=48e6,
    save_ritz=False,
    verbose=True,
)


def solve_tridiag(alpha, beta):
    """Ritz values and weights |s_k1|^2 of the Lanczos tridiagonal matrix."""
    alpha = np.asarray(alpha, dtype=float)
    n = alpha.size
    if n == 1:
        return alpha.copy(), np.ones(1)
    theta, S = eigh_tridiagonal(alpha, np.asarray(beta[: n - 1], dtype=float))
    return theta, S[0, :] ** 2


def observables(E, w, M, T):
    """Heat capacity, susceptibility (per g^2 mu_B^2) and Z_eff(T)."""
    E = np.asarray(E, dtype=float)
    w = np.asarray(w, dtype=float)
    M2 = np.asarray(M, dtype=float) ** 2
    T = np.atleast_1d(np.asarray(T, dtype=float))
    dE = E - E.min()
    C = np.zeros(T.size)
    chi = np.zeros(T.size)
    Z = np.zeros(T.size)
    for k, t in enumerate(T):
        b = 1.0 / t
        boltz = w * np.exp(-b * dE)
        z = boltz.sum()
        if z < 1e-250:
            continue
        e_avg = (dE * boltz).sum() / z
        C[k] = b * b * (((dE - e_avg) ** 2) * boltz).sum() / z
        chi[k] = b * (M2 * boltz).sum() / z
        Z[k] = z
    return C, chi, Z


def start_vectors(dim, n, seed, rng_state):
    """Normally distributed start vectors (dim x n), drawn chain by chain.

    seed = 0: generator seeded with the sector dimension; seed > 0: an
    independent stream per (dim, seed) (numpy SeedSequence).
    """
    if rng_state.get("gen") is None:
        ss = np.random.SeedSequence([dim] if seed == 0 else [dim, seed])
        rng_state["gen"] = np.random.Generator(np.random.PCG64(ss))
    return rng_state["gen"].standard_normal((n, dim)).T   # column b = chain b


# ---------------------------------------------------------------------------
# CPU reference backend (explicit sparse matrix; small and medium systems)
# ---------------------------------------------------------------------------
class CpuLanczos:
    """Block Lanczos with an explicit scipy.sparse Hamiltonian (reference)."""

    def __init__(self, model, A, precision="double", basis=None):
        if precision not in ("double", "single"):
            raise ValueError("CPU precision must be 'double' or 'single'")
        if basis is None:
            basis = _basis.enumerate_sector(model, A)
        self.dt = np.float64 if precision == "double" else np.float32
        self.u = 2.0 ** -53 if precision == "double" else 2.0 ** -24
        self.H = _basis.hamiltonian(model, basis).astype(self.dt)
        self.dim = basis.size

    def spmv(self, V):
        return (self.H @ np.asarray(V, dtype=self.dt)).astype(float)

    def block_lanczos(self, V0, M_lz):
        dt = self.dt
        V = np.array(V0, dtype=dt, copy=True)
        if V.ndim == 1:
            V = V[:, None]
        n, B = V.shape
        M_lz = min(int(M_lz), n)
        V /= np.sqrt((V * V).sum(axis=0))
        Vp = np.zeros_like(V)
        AL, BE = np.zeros((M_lz, B)), np.zeros((M_lz, B))
        nsteps = np.full(B, M_lz)
        active = np.ones(B, dtype=bool)
        normT = np.zeros(B)
        beta_prev = np.zeros(B, dtype=dt)
        tol_fac = 64.0 * self.u
        for j in range(M_lz):
            W = self.H @ V
            alpha = (V * W).sum(axis=0)
            AL[j, active] = alpha[active]
            W -= alpha * V
            if j > 0:
                W -= beta_prev * Vp
            beta = np.sqrt((W * W).sum(axis=0))
            for b in range(B):
                if not active[b]:
                    continue
                BE[j, b] = beta[b]
                normT[b] = max(normT[b], abs(alpha[b]) + beta[b] + beta_prev[b])
                if beta[b] <= tol_fac * normT[b]:
                    active[b] = False
                    nsteps[b] = j + 1
            if not active.any() or j == M_lz - 1:
                break
            scale = np.where(active, 1.0 / np.where(beta > 0, beta, 1), 0).astype(dt)
            Vp, V = V, W * scale
            beta_prev = np.where(active, beta, 0).astype(dt)
        n_max = int(nsteps.max())
        return AL[:n_max], BE[:n_max], nsteps

    def free(self):
        self.H = None


# ---------------------------------------------------------------------------
def _block_size(opts, dim, R_eff, elem, model):
    if opts["backend"] != "gpu":
        return min(opts["B_cpu"], R_eff)
    if opts["B_gpu"] == 0:
        B = 8 if 3 * dim * 8 * elem <= opts["L2_cache_bytes"] else 4
    else:
        B = opts["B_gpu"]
    B = min(B, R_eff)
    import cupy as cp
    free, _ = cp.cuda.runtime.memGetInfo()
    free = 0.95 * (free + cp.get_default_memory_pool().free_bytes())
    fixed = 8 * dim + (4 * dim + 8 * ((model.D_full + 31) // 32) if opts["lookup"] == "clt" else 0)
    while B > 1 and fixed + 3 * dim * B * elem > free:
        B //= 2
    if fixed + 3 * dim * B * elem > free:
        raise MemoryError(f"sector dim = {dim} does not fit into GPU memory")
    return B


def sector_ftlm(model, sec, opts, basis=None):
    """FTLM for one sector; returns dict with E, w (without multiplicity), B,
    t_lanczos, nsteps and optionally alpha/beta."""
    from .gpu import GpuLanczos, PRECISIONS
    dim = sec.dim
    R_eff = min(opts["R"], dim)
    M_eff = min(opts["M_lz"], dim)
    gpu = opts["backend"] == "gpu"
    elem = PRECISIONS[opts["precision"]][2]
    B = _block_size(opts, dim, R_eff, elem, model)
    if gpu:
        if opts["lookup"] == "clt" and basis is None:
            basis = _basis.enumerate_sector(model, sec.A)
        eng = GpuLanczos(model, sec.A, opts["lookup"], opts["precision"], B, basis)
    else:
        eng = CpuLanczos(model, sec.A, opts["precision"], basis)
    single_in = gpu and opts["precision"] != "double"
    rng_state = {}
    E, w, nsteps = [], [], np.zeros(R_eff, dtype=np.int64)
    AL_all = np.zeros((M_eff, R_eff)) if opts["save_ritz"] else None
    BE_all = np.zeros((M_eff, R_eff)) if opts["save_ritz"] else None
    t_lz = 0.0
    for r0 in range(0, R_eff, B):
        nb = min(B, R_eff - r0)
        V0 = start_vectors(dim, nb, opts["seed"], rng_state)
        if single_in:
            V0 = V0.astype(np.float32)
        t0 = time.perf_counter()
        AL, BE, ns = eng.block_lanczos(V0, M_eff)
        t_lz += time.perf_counter() - t0
        for b in range(nb):
            n_b = int(ns[b])
            theta, q1 = solve_tridiag(AL[:n_b, b], BE[:n_b, b])
            E.append(theta)
            w.append((dim / R_eff) * q1)
            nsteps[r0 + b] = n_b
            if opts["save_ritz"]:
                AL_all[:n_b, r0 + b] = AL[:n_b, b]
                BE_all[:n_b, r0 + b] = BE[:n_b, b]
    eng.free()
    out = dict(E=np.concatenate(E), w=np.concatenate(w), B=B, t_lanczos=t_lz, nsteps=nsteps)
    if opts["save_ritz"]:
        out.update(alpha=AL_all, beta=BE_all)
    return out


def _run_sectors(model, secs, opts, label=""):
    E, w, M = [], [], []
    info = dict(method=[], B=[], t_sec=[], ritz=[])
    t_start = time.perf_counter()
    t_lz = 0.0
    for sec in secs:
        t0 = time.perf_counter()
        if sec.dim <= opts["ed_thresh"] and model.clt_ok:   # ED needs int32 labels
            b = _basis.enumerate_sector(model, sec.A)
            e = np.linalg.eigvalsh(_basis.hamiltonian(model, b).toarray())
            ww = np.ones(sec.dim)
            info["method"].append("ED")
            info["B"].append(0)
            if opts["save_ritz"]:
                info["ritz"].append(dict(E=e, w=ww, method="ED"))
        else:
            r = sector_ftlm(model, sec, opts)
            e, ww = r["E"], r["w"]
            t_lz += r["t_lanczos"]
            info["method"].append(f"Lanczos B={r['B']}, R={min(opts['R'], sec.dim)}")
            info["B"].append(r["B"])
            if opts["save_ritz"]:
                r["method"] = "FTLM"
                info["ritz"].append(r)
        dt = time.perf_counter() - t0
        info["t_sec"].append(dt)
        E.append(e)
        w.append(sec.mult * ww)
        M.append(np.full(e.size, sec.M))
        if opts["verbose"]:
            print(f"Sector M={sec.M:4g}{label}: dim={sec.dim:10d}, "
                  f"{info['method'][-1]:<22s} t={dt:.2f}s", flush=True)
    info["t_wall"] = time.perf_counter() - t_start
    info["t_lanczos"] = t_lz
    if opts["verbose"]:
        print(f"Total wall time{label}: {info['t_wall']:.2f} s (Lanczos: {t_lz:.2f} s)")
    return np.concatenate(E), np.concatenate(w), np.concatenate(M), info


def run(opts=None, model=None, **kwargs):
    """Sector-FTLM thermodynamics.

    Parameters are given as a mapping ``opts`` and/or keyword arguments
    (same names as in the MATLAB input files): model keys (geometry,
    N_ring, s_val, spins, J, couplings, N_sites, index_base) or a
    :class:`Model` via ``model``; required R, M_lz, T_range; optional keys
    as in :data:`DEFAULTS`.  Returns a dict with T_range, C_T, chi_T,
    Z_eff, optional *_cpu reference results, sector diagnostics and
    timings.
    """
    o = dict(DEFAULTS)
    o.update(opts or {})
    o.update(kwargs)
    for key in ("R", "M_lz", "T_range"):
        if key not in o:
            raise ValueError(f"required input missing: {key}")
    o["T_range"] = np.atleast_1d(np.asarray(o["T_range"], dtype=float))
    if np.any(o["T_range"] <= 0):
        raise ValueError("T_range must be positive")
    if o["backend"] == "cpu" and o["precision"] not in ("double", "single"):
        raise ValueError("CPU precision must be 'double' or 'single'")
    if model is None:
        model = Model.from_options(o)
    if o["use_cpu_reference"] and not model.clt_ok:
        raise ValueError("use_cpu_reference requires prod(2 s_k + 1) <= 2^31 "
                         "(the CPU backend uses the basis array)")
    secs = _basis.sectors(model, o["only_M0"])
    if o["verbose"]:
        print(f"System:  {model.name}, N = {model.N}, {model.couplings.shape[0]} couplings, "
              f"spins = {sorted(set(model.spins.tolist()))}")
        print(f"FTLM:    R = {o['R']}, M_lz = {o['M_lz']}, backend = {o['backend']}, "
              f"precision = {o['precision']}, lookup = {o['lookup']}, "
              f"ed_thresh = {o['ed_thresh']}\n")

    E, w, M, info = _run_sectors(model, secs, o)
    C, chi, Z = observables(E, w, M, o["T_range"])
    res = dict(T_range=o["T_range"], C_T=C, chi_T=chi, Z_eff=Z,
               C_T_cpu=None, chi_T_cpu=None, Z_eff_cpu=None, t_wall_cpu=np.nan,
               sector_M=np.array([s.M for s in secs]),
               sector_A=np.array([s.A for s in secs]),
               sector_dims=np.array([s.dim for s in secs]),
               sector_mult=np.array([s.mult for s in secs]),
               sector_method=info["method"], sector_B=np.array(info["B"]),
               sector_t=np.array(info["t_sec"]),
               t_wall=info["t_wall"], t_lanczos=info["t_lanczos"],
               model=model, opts=o)
    if o["save_ritz"]:
        res["ritz"] = info["ritz"]
    if o["use_cpu_reference"]:
        ro = dict(o, backend="cpu", precision=o["cpu_precision"], lookup="clt")
        if o["verbose"]:
            print(f"\nCPU reference run (precision = {ro['precision']})...")
        E2, w2, M2, info2 = _run_sectors(model, secs, ro, " (CPU)")
        res["C_T_cpu"], res["chi_T_cpu"], res["Z_eff_cpu"] = observables(E2, w2, M2, o["T_range"])
        res["t_wall_cpu"] = info2["t_wall"]
        denom = np.abs(res["C_T_cpu"]).max()
        res["rel_err_C"] = np.abs(C - res["C_T_cpu"]).max() / denom if denom > 0 else 0.0
        if o["verbose"]:
            print(f"Max |C_T - C_T_cpu| / max|C_T_cpu| = {res['rel_err_C']:.3e}")
    return res
