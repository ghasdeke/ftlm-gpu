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
"""Correctness tests (run with ``python -m unittest discover python/tests``
or ``pytest python/tests``).  GPU tests are skipped without CuPy/GPU."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftlm_gpu import (Model, sectors, enumerate_sector, cr_tables, hamiltonian,  # noqa: E402
                      run, sector_ftlm, CpuLanczos, DEFAULTS, save_results)
from ftlm_gpu.ftlm import _run_sectors  # noqa: E402


def gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


HAVE_GPU = gpu_available()


def mixed_model():
    rs = np.random.default_rng(1)
    N = 7
    spins = [0.5, 1, 1.5, 0.5, 1, 2, 0.5]
    C = [(i, j, rs.standard_normal()) for i in range(N) for j in range(i + 1, N)]
    return Model(spins, C)


class TestBasis(unittest.TestCase):
    def test_enumeration_and_cr(self):
        m = mixed_model()
        labels = np.arange(m.D_full)
        digsum = np.zeros(m.D_full, dtype=int)
        tmp = labels.copy()
        for d in m.radix:
            digsum += tmp % d
            tmp //= d
        for sec in sectors(m):
            b = enumerate_sector(m, sec.A)
            np.testing.assert_array_equal(b, labels[digsum == sec.A])
            self.assertEqual(b.size, sec.dim)
            self.assertEqual(cr_tables(m, sec.A)[3], sec.dim)

    def test_presets(self):
        for g, nb in (("ico", 30), ("cubo", 24), ("cube", 12), ("dodeca", 30), ("icosid", 60)):
            self.assertEqual(Model.preset(g, s=0.5).couplings.shape[0], nb)
        m = Model.preset("ring", s=1, n_ring=5)
        np.testing.assert_array_equal(m.couplings[-1, :2], [4, 0])

    def test_validation(self):
        with self.assertRaises(ValueError):
            Model([0.5, 0.5], [(0, 0, 1.0)])
        with self.assertRaises(ValueError):
            Model([0.5, 0.7], [(0, 1, 1.0)])
        with self.assertRaises(ValueError):
            Model.preset("ico", spins=[0.5, 0.5])
        m = Model([1, 1, 1], [(1, 2, 1), (2, 1, 0.5), (2, 3, 0), (3, 1, 2)], index_base=1)
        np.testing.assert_array_equal(m.couplings, [[0, 1, 1.5], [2, 0, 2]])


class TestKernels(unittest.TestCase):
    def setUp(self):
        self.m = mixed_model()
        self.sec = sectors(self.m)[0]
        self.basis = enumerate_sector(self.m, self.sec.A)
        self.H = hamiltonian(self.m, self.basis)
        rs = np.random.default_rng(2)
        self.V = rs.standard_normal((self.sec.dim, 3))
        self.W0 = self.H @ self.V
        self.scale = (abs(self.H).sum(axis=0).max() * np.linalg.norm(self.V)
                      / np.linalg.norm(self.W0))

    def check(self, W, u, what):
        err = np.linalg.norm(W - self.W0) / np.linalg.norm(self.W0)
        self.assertLess(err, 50 * u * self.scale, f"{what}: {err:.2e}")

    def test_hamiltonian_symmetric(self):
        self.assertLess(abs(self.H - self.H.T).max(), 1e-12)

    def test_cpu_spmv(self):
        for prec, u in (("double", 2.0 ** -53), ("single", 2.0 ** -24)):
            eng = CpuLanczos(self.m, self.sec.A, prec, self.basis)
            self.check(eng.spmv(self.V), u, f"cpu/{prec}")

    @unittest.skipUnless(HAVE_GPU, "no GPU")
    def test_gpu_spmv_and_order(self):
        from ftlm_gpu.gpu import GpuLanczos
        for prec, u in (("double", 2.0 ** -53), ("single", 2.0 ** -24),
                        ("half", 2.0 ** -11), ("bfloat16", 2.0 ** -8)):
            Ws = {}
            for lk in ("clt", "cr"):
                eng = GpuLanczos(self.m, self.sec.A, lk, prec, 3, self.basis)
                Ws[lk] = eng.spmv(self.V)
                eng.free()
                self.check(Ws[lk], u, f"gpu/{lk}/{prec}")
            d = np.linalg.norm(Ws["clt"] - Ws["cr"]) / np.linalg.norm(self.W0)
            self.assertLess(d, 50 * u * self.scale, f"CLT/CR order ({prec})")

    @unittest.skipUnless(HAVE_GPU, "no GPU")
    def test_gpu_two_engines(self):
        # engines of the same precision share the constant memory of the
        # compiled module; each engine must use its own model/sector data
        from ftlm_gpu.gpu import GpuLanczos
        sec1 = sectors(self.m)[1]
        b1 = enumerate_sector(self.m, sec1.A)
        H1 = hamiltonian(self.m, b1)
        V1 = np.random.default_rng(3).standard_normal((sec1.dim, 2))
        for lk in ("clt", "cr"):
            e0 = GpuLanczos(self.m, self.sec.A, lk, "double", 3, self.basis)
            e1 = GpuLanczos(self.m, sec1.A, lk, "double", 3, b1)
            self.check(e0.spmv(self.V), 2.0 ** -53, f"first engine ({lk})")
            W1 = e1.spmv(V1)
            err = np.linalg.norm(W1 - H1 @ V1) / np.linalg.norm(H1 @ V1)
            self.assertLess(err, 1e-12, f"second engine ({lk})")
            e0.free()
            e1.free()


class TestLanczos(unittest.TestCase):
    def test_exact_quadrature(self):
        # N_L = dim: Gauss quadrature exact for the same start vectors
        m = mixed_model()
        sec = sectors(m)[-3]
        b = enumerate_sector(m, sec.A)
        E, U = np.linalg.eigh(hamiltonian(m, b).toarray())
        R = 5
        ss = np.random.SeedSequence([sec.dim])
        V = np.random.Generator(np.random.PCG64(ss)).standard_normal((R, sec.dim)).T
        c2 = (U.T @ V) ** 2 / (V ** 2).sum(axis=0)
        backends = ["cpu"] + (["gpu"] if HAVE_GPU else [])
        for be in backends:
            o = dict(DEFAULTS, R=R, M_lz=sec.dim, backend=be, precision="double",
                     B_gpu=2, B_cpu=2)
            r = sector_ftlm(m, sec, o, b)
            for beta in (0.1, 1.0, 5.0):
                exact = (c2.T @ np.exp(-beta * E)).sum()
                est = (R / sec.dim) * (r["w"] * np.exp(-beta * r["E"])).sum()
                self.assertLess(abs(est - exact), 1e-10 * exact, f"{be}, beta={beta}")

    def test_sum_rule_half_integer(self):
        be = "gpu" if HAVE_GPU else "cpu"
        for opts in (dict(geometry="ring", N_ring=9, s_val=0.5, J=1.0),
                     dict(spins=[0.5, 1, 0.5, 1, 0.5, 1],
                          couplings=[[1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1],
                                     [6, 1, 1], [1, 4, -0.3]])):
            res = run(opts, R=3, M_lz=20, T_range=[1e3, 1e9], backend=be, ed_thresh=20,
                      verbose=False)
            D = res["model"].D_full
            self.assertLess(abs(res["Z_eff"][-1] - D), 1e-6 * D)
        self.assertTrue(np.all(np.mod(run(dict(geometry="ring", N_ring=9, s_val=0.5, J=1.0),
                                          R=2, M_lz=5, T_range=[1.0], backend="cpu",
                                          verbose=False)["sector_M"], 1) == 0.5))

    def test_breakdown(self):
        m = Model.preset("ico", s=1.0)
        sec = [s for s in sectors(m) if s.M == 11][0]
        b = enumerate_sector(m, sec.A)
        n_dist = np.unique(np.round(np.linalg.eigvalsh(hamiltonian(m, b).toarray()), 8)).size
        variants = [("cpu", "double"), ("cpu", "single")]
        if HAVE_GPU:
            variants += [("gpu", "double"), ("gpu", "single"), ("gpu", "half")]
        for be, prec in variants:
            o = dict(DEFAULTS, R=12, M_lz=100, backend=be, precision=prec)
            r = sector_ftlm(m, sec, o, b)
            self.assertTrue(np.all(np.isfinite(r["E"])) and np.all(np.isfinite(r["w"])))
            self.assertLessEqual(r["nsteps"].max(), n_dist + 1, f"{be}/{prec}")
            self.assertAlmostEqual(r["w"].sum() / sec.dim, 1.0, places=5)

    @unittest.skipUnless(HAVE_GPU, "no GPU")
    def test_large_sector_reduction(self):
        # s = 1 icosahedron, M = 0: dim = 73,789 > 256^2 (two tree passes)
        from ftlm_gpu.gpu import GpuLanczos
        m = Model.preset("ico", s=1.0)
        sec = sectors(m, True)[0]
        b = enumerate_sector(m, sec.A)
        H = hamiltonian(m, b)
        V = np.random.default_rng(7).standard_normal((sec.dim, 2))
        nl = 4
        AL = np.zeros((nl, 2))
        BE = np.zeros((nl, 2))
        for c in range(2):
            v = V[:, c] / np.linalg.norm(V[:, c])
            vp = np.zeros(sec.dim)
            beta = 0.0
            for j in range(nl):
                w = H @ v
                alpha = v @ w
                w = w - alpha * v - beta * vp
                beta = np.linalg.norm(w)
                AL[j, c], BE[j, c] = alpha, beta
                vp, v = v, w / beta
        for prec, tol in (("double", 1e-12), ("single", 1e-5), ("half", 1e-2)):
            eng = GpuLanczos(m, sec.A, "clt", prec, 2, b)
            a, bt, _ = eng.block_lanczos(V if prec == "double" else V.astype(np.float32), nl)
            eng.free()
            err = max(np.abs(a[:nl] - AL).max(), np.abs(bt[:nl] - BE).max()) / np.abs(AL).max()
            self.assertLess(err, tol, prec)

    def test_cr_beyond_2_31(self):
        # prod(2 s_i + 1) = 2^32: the exact diagonalization of small sectors
        # (int32 basis) is not available; these sectors are treated by FTLM
        m = Model.preset("ring", s=0.5, n_ring=32)
        self.assertFalse(m.clt_ok)
        self.assertTrue(m.cr_ok)
        opts = dict(DEFAULTS, R=8, M_lz=40, T_range=np.array([1.0]), lookup="cr",
                    precision="double", verbose=False)
        with self.assertRaises(ValueError):     # CPU reference needs the basis array
            run(opts, model=m, use_cpu_reference=True)
        if not HAVE_GPU:
            return
        small = [sec for sec in sectors(m) if sec.dim <= opts["ed_thresh"]]
        self.assertEqual([sec.dim for sec in small], [496, 32, 1])
        E, w, M, info = _run_sectors(m, small, opts)
        self.assertTrue(all(meth.startswith("Lanczos") for meth in info["method"]))
        # ferromagnetic state E = N s^2 J = 8, one-magnon minimum 8 - 4 s J = 6
        self.assertAlmostEqual(max(E), 8.0, places=9)
        self.assertAlmostEqual(min(E[np.asarray(M) == 15]), 6.0, places=9)

    def test_save_npz(self):
        m = Model.preset("ring", s=0.5, n_ring=6)
        res = run(model=m, R=4, M_lz=10, T_range=[0.5, 1.0], backend="cpu",
                  save_ritz=True, verbose=False)
        with tempfile.TemporaryDirectory() as d:
            f = save_results(res, Path(d) / "r.npz")
            with np.load(f) as z:       # default allow_pickle=False
                self.assertEqual(list(z["sector_method"]), list(res["sector_method"]))
                np.testing.assert_allclose(z["C_T"], res["C_T"])
            with np.load(f, allow_pickle=True) as z:
                self.assertEqual(len(z["ritz"]), len(res["sector_method"]))
            save_results(res, Path(d) / "r.mat")

    def test_ftlm_vs_ed(self):
        # statistical test: stochastic error of FTLM with R_eff = min(R, dim)
        # random vectors stays below ~3 % for T >= 0.5 (larger at lower T)
        base = dict(spins=[1, 0.5] * 4, R=200, M_lz=60, T_range=np.linspace(0.5, 5, 25),
                    verbose=False,
                    couplings=[[1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1],
                               [6, 7, 1], [7, 8, 1], [8, 1, 1], [1, 3, 0.4], [5, 7, 0.4],
                               [2, 6, -0.2]])
        ed = run(base, ed_thresh=10 ** 4, backend="cpu")
        ft = run(base, ed_thresh=0, backend="gpu" if HAVE_GPU else "cpu")
        self.assertLess(np.abs(ft["C_T"] - ed["C_T"]).max() / ed["C_T"].max(), 0.05)
        self.assertLess(np.abs(ft["chi_T"] - ed["chi_T"]).max() / ed["chi_T"].max(), 0.05)


if __name__ == "__main__":
    unittest.main()
