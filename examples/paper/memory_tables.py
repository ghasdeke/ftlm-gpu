"""Memory estimates of Tables 1, 2 and 4 of the paper (analytic, no GPU needed).

Usage:  python memory_tables.py

Table 1  number of stored entries N_nz of the M = 0 sector Hamiltonian of
         the icosahedron (N = 12, N_B = 30) from Eq. (10) (all diagonal
         elements counted, although for integer s a few of them vanish)
         and the memory of a
         MATLAB-style 64-bit sparse matrix, 16 N_nz + 8 (D_0 + 1) bytes
Table 2  memory of the stored Hamiltonian (CS), the full lookup table, the
         compressed lookup table (CLT), the basis array and three FP64 work
         vectors for the M = 0 sector of the s = 2 icosahedron
Table 4  GPU memory budget of the CLT-based batched kernel (B = 4, FP32):
         CLT (D/32) 8 bytes, basis D_0 4 bytes, Lanczos workspace 3 B D_0 4 bytes

Sizes are in decimal units (1 MB = 1e6 bytes).  The paper rounds to about
three significant digits.
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

from fractions import Fraction


def sector_dims(n_sites, s):
    """dims[A] = number of product states of n_sites spins s with digit sum A
    (digits a_k = m_k + s in {0, ..., 2s}); M = A - n_sites s."""
    d = int(2 * s) + 1
    dims = [1]
    for _ in range(n_sites):
        new = [0] * (len(dims) + d - 1)
        for a, c in enumerate(dims):
            for k in range(d):
                new[a + k] += c
        dims = new
    return dims


def dim_M(n_sites, s, M):
    """Dimension of the sector with magnetization M (0 outside the range)."""
    A = M + n_sites * s
    if A.denominator != 1:
        return 0
    A = int(A)
    dims = sector_dims(n_sites, s)
    return dims[A] if 0 <= A < len(dims) else 0


def nnz(n_sites, s, n_bonds, M=Fraction(0)):
    """Eq. (10): all diagonal elements plus the two ladder-operator terms of each bond."""
    s = Fraction(s)
    m_values = [-s + k for k in range(int(2 * s) + 1)]
    a_range = [m for m in m_values if m <= s - 1]      # s_i^+ acts on m_i = a
    b_range = [m for m in m_values if m >= -s + 1]     # s_j^- acts on m_j = b
    off = sum(dim_M(n_sites - 2, s, M - a - b) for a in a_range for b in b_range)
    return dim_M(n_sites, s, M) + 2 * n_bonds * off


def fmt_bytes(b):
    return f"{b / 1e9:.3g} GB" if b >= 1e9 else f"{b / 1e6:.3g} MB"


def main():
    N, NB = 12, 30
    print("Table 1: icosahedron (N = 12, N_B = 30), M = 0 sector")
    print(f"{'s':>5} {'D_0':>13} {'N_nz':>16} {'memory':>10}")
    for s in (Fraction(1, 2), 1, Fraction(3, 2), 2, Fraction(5, 2), 3):
        s = Fraction(s)
        d0 = dim_M(N, s, Fraction(0))
        nz = nnz(N, s, NB)
        print(f"{str(s):>5} {d0:>13,} {nz:>16,} {fmt_bytes(16 * nz + 8 * (d0 + 1)):>10}")

    s = Fraction(2)
    d0 = dim_M(N, s, Fraction(0))
    d_full = (int(2 * s) + 1) ** N
    cs = 16 * nnz(N, s, NB) + 8 * (d0 + 1)
    full = 4 * d_full
    clt = d_full // 32 * 8
    basis = 4 * d0
    work = 3 * 8 * d0
    print("\nTable 2: s = 2 icosahedron, M = 0 sector (MB)")
    print(f"{'':>18} {'CS':>10} {'full lookup':>12} {'CLT':>8}")
    print(f"{'Hamiltonian':>18} {cs / 1e6:>10.0f} {'-':>12} {'-':>8}")
    print(f"{'lookup table':>18} {'-':>10} {full / 1e6:>12.0f} {clt / 1e6:>8.0f}")
    print(f"{'basis array':>18} {'-':>10} {basis / 1e6:>12.0f} {basis / 1e6:>8.0f}")
    print(f"{'work vectors (3x)':>18} {work / 1e6:>10.0f} {work / 1e6:>12.0f} {work / 1e6:>8.0f}")
    print(f"{'sum':>18} {(cs + work) / 1e6:>10.0f} {(full + basis + work) / 1e6:>12.0f} "
          f"{(clt + basis + work) / 1e6:>8.0f}")
    print(f"(FP32 work vectors: {work / 2e6:.0f} MB)")

    print("\nTable 4: GPU memory budget, M = 0 sector, B = 4, FP32 (MB)")
    print(f"{'system':>28} {'CLT':>8} {'basis':>8} {'Lanczos':>9} {'total':>8}")
    B = 4
    for name, n_sites, s in (("icosahedron, s = 3/2", 12, Fraction(3, 2)),
                             ("icosahedron, s = 2", 12, Fraction(2)),
                             ("icosidodecahedron, s = 1/2", 30, Fraction(1, 2))):
        d0 = dim_M(n_sites, s, Fraction(0))
        d_full = (int(2 * s) + 1) ** n_sites
        clt = -(-d_full // 32) * 8
        basis = 4 * d0
        lanczos = 3 * B * d0 * 4
        print(f"{name:>28} {clt / 1e6:>8.1f} {basis / 1e6:>8.1f} {lanczos / 1e6:>9.0f} "
              f"{(clt + basis + lanczos) / 1e6:>8.0f}")


if __name__ == "__main__":
    main()
