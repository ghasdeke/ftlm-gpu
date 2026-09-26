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
"""Magnetization sectors, basis enumeration, CLT and CR tables, explicit H.

Basis encoding (identical to the MATLAB front end and the CUDA kernels):
digit a_k = m_k + s_k in {0..2 s_k}, CLT label n = sum_k a_k P_k with
P_0 = 1, P_{k+1} = P_k (2 s_k + 1); a sector is fixed by the digit sum
A = S_max + M; the sector basis is ordered by increasing label.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Sector:
    A: int          # digit sum
    M: float        # magnetization quantum number
    mult: int       # 2 for M > 0 (sector -M has the same spectrum), else 1
    dim: int


def sectors(model, only_lowest=False):
    """Sectors with M >= 0, ordered by increasing M."""
    A_tot2 = int(model.two_s.sum())
    poly = np.array([1], dtype=object)
    for d in model.radix:
        poly = np.convolve(poly, np.ones(int(d), dtype=object))
    A_list = list(range((A_tot2 + 1) // 2, A_tot2 + 1))
    if only_lowest:
        A_list = A_list[:1]
    out = []
    for A in A_list:
        M = A - A_tot2 / 2
        out.append(Sector(A=A, M=M, mult=1 + (M > 0), dim=int(poly[A])))
    return out


def enumerate_sector(model, A):
    """Sorted int32 labels of the sector with digit sum A.

    Site-by-site construction: the sorted label lists of the first k sites,
    grouped by partial digit sum, are extended by the digit of site k (the
    most significant so far); only partial sums that can still reach A are
    kept.  Requires prod(2 s_k + 1) <= 2^31.
    """
    if not model.clt_ok:
        raise ValueError("prod(2 s_k + 1) exceeds 2^31: labels do not fit into int32; "
                         "use lookup='cr'")
    two_s = [int(t) for t in model.two_s]
    P = [int(p) for p in model.power]
    N = len(two_s)
    cap_after = [sum(two_s[k + 1:]) for k in range(N)]
    lists = {0: np.zeros(1, dtype=np.int32)}
    for k in range(N):
        lo = max(0, A - cap_after[k])
        hi = min(A, sum(two_s[:k + 1]))
        new = {}
        for s_new in range(lo, hi + 1):
            parts = [lists[s_new - a] + np.int32(a * P[k])
                     for a in range(two_s[k] + 1)
                     if (s_new - a) in lists and lists[s_new - a].size]
            new[s_new] = (np.concatenate(parts) if parts
                          else np.zeros(0, dtype=np.int32))
        lists = new
    return lists.get(A, np.zeros(0, dtype=np.int32))


def build_clt(basis, D_full):
    """Compressed lookup table: (block_base int32, block_mask uint32).

    32 labels per block; block_mask has bit j set iff label 32 b + j is in
    the sector, block_base is the sector index of the first in-sector label
    of the block (-1 for empty blocks).
    """
    n_blocks = (int(D_full) + 31) // 32
    states = np.asarray(basis, dtype=np.int64)
    blks = states >> 5
    bits = states & 31
    block_base = np.full(n_blocks, -1, dtype=np.int32)
    ub, first = np.unique(blks, return_index=True)
    block_base[ub] = first.astype(np.int32)
    block_mask = np.zeros(n_blocks, dtype=np.uint64)
    np.bitwise_or.at(block_mask, blks, np.left_shift(np.uint64(1), bits.astype(np.uint64)))
    return block_base, block_mask.astype(np.uint32)


def cr_tables(model, A):
    """Cumulative dimension table D_c(p, A', a) for combinatorial ranking.

    Returns (dcum int32, pstride, astride, dim); element (p, A', a) is at
    position p*pstride + A'*astride + a.
    """
    two_s = [int(t) for t in model.two_s]
    N = len(two_s)
    D = np.zeros((N + 1, A + 1), dtype=np.int64)
    D[0, 0] = 1
    for p in range(1, N + 1):
        for Ap in range(A + 1):
            q = np.arange(0, min(two_s[p - 1], Ap) + 1)
            D[p, Ap] = D[p - 1, Ap - q].sum()
    dim = int(D[N, A])
    if dim >= 2 ** 31:
        raise ValueError("sector dimension exceeds the int32 rank range")
    astride = max(two_s) + 1
    pstride = (A + 1) * astride
    dcum = np.zeros(N * pstride, dtype=np.int64)
    for p in range(N):
        for Ap in range(A + 1):
            base = p * pstride + Ap * astride
            acc = 0
            for a in range(two_s[p] + 1):
                dcum[base + a] = acc
                if Ap - a >= 0:
                    acc += D[p, Ap - a]
    return dcum.astype(np.int32), pstride, astride, dim


def digits(model, basis):
    """Digit matrix a[t, k] of the sector basis."""
    labels = np.asarray(basis, dtype=np.int64)
    a = np.empty((labels.size, model.N), dtype=np.int64)
    tmp = labels.copy()
    for k in range(model.N):
        a[:, k] = tmp % model.radix[k]
        tmp //= model.radix[k]
    return a


def hamiltonian(model, basis):
    """Explicit sparse Hamiltonian (CSR, float64) on a sorted sector basis."""
    labels = np.asarray(basis, dtype=np.int64)
    dim = labels.size
    a = digits(model, basis)
    s = model.spins
    m = a - s
    C = model.couplings
    diag = np.zeros(dim)
    for i, j, J in C:
        diag += J * m[:, int(i)] * m[:, int(j)]
    rows, cols, vals = [np.arange(dim)], [np.arange(dim)], [diag]
    P = model.power
    for i, j, J in C:
        i, j = int(i), int(j)
        hJ = 0.5 * J
        ri, rj = s[i] * (s[i] + 1), s[j] * (s[j] + 1)
        for di, dj in ((-1, 1), (1, -1)):
            sel = np.nonzero((a[:, i] + di >= 0) & (a[:, i] + di <= model.two_s[i])
                             & (a[:, j] + dj >= 0) & (a[:, j] + dj <= model.two_s[j]))[0]
            if sel.size == 0:
                continue
            mi, mj = m[sel, i], m[sel, j]
            # <out| ... |src>, src = out with (m_i + di, m_j + dj)
            ci = np.sqrt(ri - mi * (mi + di))
            cj = np.sqrt(rj - mj * (mj + dj))
            src = labels[sel] + di * P[i] + dj * P[j]
            col = np.searchsorted(labels, src)
            ok = (col < dim) & (labels[np.minimum(col, dim - 1)] == src)
            rows.append(sel[ok])
            cols.append(col[ok])
            vals.append((hJ * ci * cj)[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim, dim))
