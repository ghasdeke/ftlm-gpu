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
"""Spin model: local spins, pairwise couplings and basis encoding.

The Hamiltonian is

    H = sum_c  J_c  s_{i_c} . s_{j_c}

with an arbitrary list of pairwise couplings (i_c, j_c, J_c) and arbitrary
local spins s_k (integers or half-integers up to 15/2).  Site indices are
0-based in the Python API (``index_base=1`` accepts 1-based input, as in
the MATLAB front end and in TOML input files).
"""

import math

import numpy as np

MAX_SITES = 32
MAX_COUPLINGS = 512
MAX_TWO_S = 15

GEOMETRIES = ("ico", "cubo", "cube", "dodeca", "icosid", "ring")


# ---------------------------------------------------------------------------
# Predefined geometries (bond order identical to +ftlm/geometry.m)
# ---------------------------------------------------------------------------
def _edges_at_distance(V, d, tol):
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.linalg.norm(V[i] - V[j]) - d) < tol:
                bonds.append((i, j))
    return bonds


def geometry(name, n_ring=None):
    """Nearest-neighbor bonds of a predefined cluster.

    Returns ``(bonds, N, long_name, short_name)`` with 0-based bonds.
    """
    phi = (1 + math.sqrt(5)) / 2
    if name == "ico":
        V = [[0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
             [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
             [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]]
        bonds, N, long, short = _edges_at_distance(V, 2, 0.01), 12, "Icosahedron", "ico"
        assert len(bonds) == 30
    elif name == "cubo":
        V = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
             [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
             [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]]
        bonds, N, long, short = (_edges_at_distance(V, math.sqrt(2), 0.01), 12,
                                 "Cuboctahedron", "cubo")
        assert len(bonds) == 24
    elif name == "cube":
        V = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
        bonds, N, long, short = _edges_at_distance(V, 2, 0.01), 8, "Cube", "cube"
        assert len(bonds) == 12
    elif name == "dodeca":
        ip = 1 / phi
        V = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
             [0, ip, phi], [0, -ip, phi], [0, ip, -phi], [0, -ip, -phi],
             [ip, phi, 0], [-ip, phi, 0], [ip, -phi, 0], [-ip, -phi, 0],
             [phi, 0, ip], [phi, 0, -ip], [-phi, 0, ip], [-phi, 0, -ip]]
        bonds, N, long, short = (_edges_at_distance(V, 2 / phi, 0.1), 20,
                                 "Dodecahedron", "dodeca")
        assert len(bonds) == 30
    elif name == "icosid":
        sc = (1 + phi) / 2
        h = phi / 2
        polar = [[0, 0, phi], [0, 0, -phi], [phi, 0, 0], [-phi, 0, 0],
                 [0, phi, 0], [0, -phi, 0]]
        equat = [[.5, h, sc], [-.5, h, sc], [.5, -h, sc], [-.5, -h, sc],
                 [.5, h, -sc], [-.5, h, -sc], [.5, -h, -sc], [-.5, -h, -sc],
                 [h, sc, .5], [-h, sc, .5], [h, -sc, .5], [-h, -sc, .5],
                 [h, sc, -.5], [-h, sc, -.5], [h, -sc, -.5], [-h, -sc, -.5],
                 [sc, .5, h], [-sc, .5, h], [sc, -.5, h], [-sc, -.5, h],
                 [sc, .5, -h], [-sc, .5, -h], [sc, -.5, -h], [-sc, -.5, -h]]
        bonds, N, long, short = (_edges_at_distance(polar + equat, 1, 0.1), 30,
                                 "Icosidodecahedron", "icosid")
        assert len(bonds) == 60
    elif name == "ring":
        if n_ring is None or int(n_ring) != n_ring or n_ring < 3:
            raise ValueError("geometry 'ring' requires n_ring (integer >= 3)")
        N = int(n_ring)
        bonds = [(i, i + 1) for i in range(N - 1)] + [(N - 1, 0)]
        long, short = f"{N}-Ring", f"ring_{N}"
    else:
        raise ValueError(f"unknown geometry {name!r} (allowed: {', '.join(GEOMETRIES)})")
    return bonds, N, long, short


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Model:
    """Isotropic spin model with arbitrary couplings and local spins.

    Use :meth:`Model.preset` for the predefined clusters or construct
    directly from a coupling list::

        Model(spins=[0.5, 1, 0.5, 1], couplings=[(0, 1, 1.0), (1, 2, 1.0),
                                                 (2, 3, 1.0), (3, 0, 1.0),
                                                 (0, 2, -0.3)])

    Duplicate pairs (either orientation) are summed, zero couplings are
    dropped; the orientation of the first occurrence is kept.
    """

    def __init__(self, spins, couplings, index_base=0, name="Custom", short="custom",
                 geometry="custom"):
        spins = np.atleast_1d(np.asarray(spins, dtype=float))
        C = np.asarray(couplings, dtype=float)
        if C.ndim != 2 or C.shape[1] != 3 or C.shape[0] < 1 or not np.all(np.isfinite(C)):
            raise ValueError("couplings must be a K x 3 array (i, j, J) of finite numbers")
        ij = C[:, :2] - index_base
        if np.any(ij != np.round(ij)) or np.any(ij < 0):
            raise ValueError(f"coupling site indices must be integers >= {index_base}")
        if np.any(ij[:, 0] == ij[:, 1]):
            raise ValueError("couplings with i == j are not allowed")
        N = spins.size
        if not 1 <= N <= MAX_SITES:
            raise ValueError(f"N = {N} outside [1, {MAX_SITES}]")
        if ij.max() >= N:
            raise ValueError(f"coupling site index exceeds the number of sites N = {N}")
        two_s = np.round(2 * spins).astype(np.int64)
        if (np.any(np.abs(2 * spins - two_s) > 1e-12) or np.any(two_s < 1)
                or np.any(two_s > MAX_TWO_S)):
            raise ValueError("local spins must be integers or half-integers in [1/2, 15/2]")

        # merge duplicates (unordered pair), keep first orientation, drop zeros
        merged = {}
        order = []
        for (i, j), J in zip(ij.astype(np.int64), C[:, 2]):
            key = (min(i, j), max(i, j))
            if key in merged:
                merged[key][2] += J
            else:
                merged[key] = [int(i), int(j), float(J)]
                order.append(key)
        C = np.array([merged[k] for k in order if merged[k][2] != 0.0], dtype=float).reshape(-1, 3)
        if C.shape[0] == 0:
            raise ValueError("all couplings are zero")
        if C.shape[0] > MAX_COUPLINGS:
            raise ValueError(f"{C.shape[0]} couplings exceed the kernel limit {MAX_COUPLINGS}")

        self.spins = spins
        self.couplings = C
        self.name, self.short, self.geometry = name, short, geometry
        self.two_s = two_s
        self.radix = two_s + 1
        self.power = np.concatenate(([1], np.cumprod(self.radix[:-1]))).astype(np.int64)
        self.bits = np.array([math.ceil(math.log2(d)) for d in self.radix], dtype=np.int64)
        self.shift = np.concatenate(([0], np.cumsum(self.bits[:-1]))).astype(np.int64)

    @classmethod
    def preset(cls, geometry_name, s=None, J=1.0, spins=None, n_ring=None):
        """Predefined cluster with uniform nearest-neighbor coupling J."""
        bonds, N, long, short = geometry(geometry_name, n_ring)
        if spins is None:
            if s is None:
                raise ValueError("give the local spin s or a list of spins")
            spins = [s] * N
        if len(spins) != N:
            raise ValueError(f"{len(spins)} spins given, geometry {geometry_name!r} has {N} sites")
        C = [(i, j, J) for i, j in bonds]
        return cls(spins, C, name=long, short=short, geometry=geometry_name)

    @classmethod
    def from_options(cls, opts):
        """Model from an options mapping (same keys as the MATLAB input files).

        Keys: geometry, N_ring, s_val, spins, J, couplings, N_sites,
        index_base (default 1 for couplings given here, as in MATLAB).
        """
        couplings = opts.get("couplings")
        spins = opts.get("spins")
        s_val = opts.get("s_val")
        if couplings is not None and len(couplings) > 0:
            base = int(opts.get("index_base", 1))
            C = np.asarray(couplings, dtype=float)
            if spins is None:
                if s_val is None:
                    raise ValueError("give s_val or spins")
                n = opts.get("N_sites") or int(C[:, :2].max()) + 1 - base
                spins = [s_val] * int(n)
            return cls(spins, C, index_base=base)
        geom = opts.get("geometry")
        if not geom:
            raise ValueError("specify either a predefined geometry or couplings")
        J = opts.get("J")
        if J is None or not np.isscalar(J):
            raise ValueError("a predefined geometry requires a scalar coupling J")
        return cls.preset(geom, s=s_val, J=float(J), spins=spins, n_ring=opts.get("N_ring"))

    def __repr__(self):
        return (f"Model({self.name}, N={self.N}, spins={self.spins.tolist()}, "
                f"{self.couplings.shape[0]} couplings)")

    # -- derived quantities -------------------------------------------------
    @property
    def N(self):
        return self.spins.size

    @property
    def S_max(self):
        return float(self.spins.sum())

    @property
    def D_full(self):
        return int(np.prod(self.radix.astype(object)))

    @property
    def clt_ok(self):
        return self.D_full <= 2 ** 31

    @property
    def cr_ok(self):
        return int(self.bits.sum()) <= 64

    @property
    def tag(self):
        if np.all(self.spins == self.spins[0]):
            ts = int(self.two_s[0])
            s_str = f"{ts // 2}" if ts % 2 == 0 else f"{ts}o2"
            return f"{self.short}_s{s_str}"
        return f"{self.short}_mixed"
