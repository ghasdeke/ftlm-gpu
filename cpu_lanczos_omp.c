/*
 * cpu_lanczos_omp.c
 *
 * Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
 * and Helmholtz-Zentrum Dresden-Rossendorf e.V.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ================================================================
 * Matrix-free Heisenberg Lanczos on CPU with OpenMP (FP64).
 *
 * Direct CPU counterpart of cuda_lanczos_mfree.cu.
 * Strictly C89 compatible for MSVC.
 *
 * Compile (Windows / MSVC):
 *   mex cpu_lanczos_omp.c COMPFLAGS="$COMPFLAGS /openmp"
 *
 * Compile (Linux / GCC):
 *   mex cpu_lanczos_omp.c CFLAGS="$CFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp"
 * ================================================================
 */

#include "mex.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_SITES  32
#define MAX_BONDS  64
#define BLOCK_MAX  32   /* maximum block size for block_lanczos */
#define CLUT_BITS  5    /* 2^5 = 32 states per block */
#define CLUT_SIZE  32

/* ================================================================
 * Persistent state
 * ================================================================ */
static int    s_bonds[2 * MAX_BONDS];
static int    s_powers[MAX_SITES];
static int    s_N         = 0;
static int    s_d         = 0;
static int    s_nbonds    = 0;
static double s_s         = 0.0;
static double s_J         = 0.0;
static double s_ss1       = 0.0;
static int    s_d_minus_1 = 0;
static int    s_ntotal    = 0;

static int    *s_lookup   = NULL;
static int    *s_basis    = NULL;
static double *s_v        = NULL;
static double *s_vp       = NULL;
static double *s_w        = NULL;
static int     s_dim      = 0;
static int     s_init     = 0;

/* CLT data structures (compressed lookup) */
static int          *s_block_base = NULL;
static unsigned int *s_block_mask = NULL;
static int           s_clut_init  = 0;

static void cleanup(void)
{
    if (s_lookup)     { mxFree(s_lookup);     s_lookup     = NULL; }
    if (s_basis)      { mxFree(s_basis);      s_basis      = NULL; }
    if (s_v)          { mxFree(s_v);          s_v          = NULL; }
    if (s_vp)         { mxFree(s_vp);         s_vp         = NULL; }
    if (s_w)          { mxFree(s_w);          s_w          = NULL; }
    if (s_block_base) { mxFree(s_block_base); s_block_base = NULL; }
    if (s_block_mask) { mxFree(s_block_mask); s_block_mask = NULL; }
    s_dim       = 0;
    s_init      = 0;
    s_clut_init = 0;
}

/* ================================================================
 * Matrix-free SpMV:  w = H * v   (OpenMP-parallel)
 * ================================================================ */
static void heisenberg_spmv_omp(
    double       *w,
    const double *v,
    const int    *lookup,
    const int    *basis,
    int dim)
{
    int N       = s_N;
    int d       = s_d;
    int nbonds  = s_nbonds;
    double sv   = s_s;
    double J    = s_J;
    double ss1  = s_ss1;
    int dm1     = s_d_minus_1;
    int beta;

    #pragma omp parallel for schedule(static)
    for (beta = 0; beta < dim; beta++) {
        int state, tmp, k, b, si, sj, di, dj, state_a, idx_a;
        int digits[MAX_SITES];
        double v_beta, result, diag, mi, mj, mi_a, mj_a, coeff;

        state = basis[beta];

        tmp = state;
        for (k = 0; k < N; k++) {
            digits[k] = tmp % d;
            tmp /= d;
        }

        v_beta = v[beta];

        /* Diagonal part */
        diag = 0.0;
        for (b = 0; b < nbonds; b++) {
            mi = (double)digits[s_bonds[2*b]]     - sv;
            mj = (double)digits[s_bonds[2*b + 1]] - sv;
            diag += mi * mj;
        }
        result = J * diag * v_beta;

        /* Off-diagonal */
        for (b = 0; b < nbonds; b++) {
            si = s_bonds[2*b];
            sj = s_bonds[2*b + 1];
            di = digits[si];
            dj = digits[sj];
            mi = (double)di - sv;
            mj = (double)dj - sv;

            /* S+_i S-_j */
            if (di > 0 && dj < dm1) {
                mi_a = mi - 1.0;
                mj_a = mj + 1.0;
                coeff = 0.5 * J
                    * sqrt(ss1 - mi_a * (mi_a + 1.0))
                    * sqrt(ss1 - mj_a * (mj_a - 1.0));
                state_a = state - s_powers[si] + s_powers[sj];
                idx_a = lookup[state_a];
                if (idx_a >= 0) {
                    result += coeff * v[idx_a];
                }
            }

            /* S-_i S+_j */
            if (di < dm1 && dj > 0) {
                mi_a = mi + 1.0;
                mj_a = mj - 1.0;
                coeff = 0.5 * J
                    * sqrt(ss1 - mi_a * (mi_a - 1.0))
                    * sqrt(ss1 - mj_a * (mj_a + 1.0));
                state_a = state + s_powers[si] - s_powers[sj];
                idx_a = lookup[state_a];
                if (idx_a >= 0) {
                    result += coeff * v[idx_a];
                }
            }
        }

        w[beta] = result;
    }
}

/* ================================================================
 * BLAS-like operations with OpenMP
 * ================================================================ */

static double dot_omp(const double *x, const double *y, int n)
{
    double sum = 0.0;
    int i;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

static double nrm2_omp(const double *x, int n)
{
    double sum = 0.0;
    int i;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

static void axpy_omp(double alpha, const double *x, double *y, int n)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

static void scal_omp(double alpha, double *x, int n)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

/* ================================================================
 * Block SpMV:  W = H * V   (OpenMP-parallel, row-major)
 *
 * V, W: dim x B in row-major:  V[beta * B + b]
 * Amortizes lookup/basis accesses over B vectors.
 * For B=8 typically 2-3x faster than B individual SpMV calls.
 * ================================================================ */
static void heisenberg_spmv_block_omp(
    double       *W,
    const double *V,
    const int    *lookup,
    const int    *basis,
    int dim, int B)
{
    int N_loc   = s_N;
    int d       = s_d;
    int nbonds  = s_nbonds;
    double sv   = s_s;
    double Jv   = s_J;
    double ss1  = s_ss1;
    int dm1     = s_d_minus_1;
    int beta_idx;

    #pragma omp parallel for schedule(static)
    for (beta_idx = 0; beta_idx < dim; beta_idx++) {
        int state, tmp, k, b, si, sj, di, dj, state_a, idx_a, bv;
        int digits[MAX_SITES];
        double mi, mj, mi_a, mj_a, coeff, diag;
        int row_beta, row_a;

        row_beta = beta_idx * B;
        state = basis[beta_idx];

        tmp = state;
        for (k = 0; k < N_loc; k++) {
            digits[k] = tmp % d;
            tmp /= d;
        }

        /* Diagonal part - compute once */
        diag = 0.0;
        for (b = 0; b < nbonds; b++) {
            mi = (double)digits[s_bonds[2*b]]     - sv;
            mj = (double)digits[s_bonds[2*b + 1]] - sv;
            diag += mi * mj;
        }
        diag *= Jv;

        /* Diagonal contribution for all B vectors */
        for (bv = 0; bv < B; bv++) {
            W[row_beta + bv] = diag * V[row_beta + bv];
        }

        /* Off-diagonal */
        for (b = 0; b < nbonds; b++) {
            si = s_bonds[2*b];
            sj = s_bonds[2*b + 1];
            di = digits[si];
            dj = digits[sj];
            mi = (double)di - sv;
            mj = (double)dj - sv;

            /* S+_i S-_j */
            if (di > 0 && dj < dm1) {
                mi_a = mi - 1.0;
                mj_a = mj + 1.0;
                coeff = 0.5 * Jv
                    * sqrt(ss1 - mi_a * (mi_a + 1.0))
                    * sqrt(ss1 - mj_a * (mj_a - 1.0));
                state_a = state - s_powers[si] + s_powers[sj];
                idx_a = lookup[state_a];
                if (idx_a >= 0) {
                    row_a = idx_a * B;
                    for (bv = 0; bv < B; bv++) {
                        W[row_beta + bv] += coeff * V[row_a + bv];
                    }
                }
            }

            /* S-_i S+_j */
            if (di < dm1 && dj > 0) {
                mi_a = mi + 1.0;
                mj_a = mj - 1.0;
                coeff = 0.5 * Jv
                    * sqrt(ss1 - mi_a * (mi_a - 1.0))
                    * sqrt(ss1 - mj_a * (mj_a + 1.0));
                state_a = state + s_powers[si] - s_powers[sj];
                idx_a = lookup[state_a];
                if (idx_a >= 0) {
                    row_a = idx_a * B;
                    for (bv = 0; bv < B; bv++) {
                        W[row_beta + bv] += coeff * V[row_a + bv];
                    }
                }
            }
        }
    }
}

/* ================================================================
 * CPU CLT lookup (compressed lookup with popcount)
 *
 * Replaces lookup[state] with:
 *   block_base[state >> 5] + popcount(block_mask[state >> 5] & lower_bits)
 *
 * Uses GCC __builtin_popcount or MSVC __popcnt.
 * ================================================================ */
#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT32(x) ((int)__popcnt((unsigned int)(x)))
#else
#define POPCOUNT32(x) __builtin_popcount((unsigned int)(x))
#endif

static int clut_lookup_cpu(
    const int          *block_base,
    const unsigned int *block_mask,
    int state_a)
{
    int blk = state_a >> CLUT_BITS;
    int bit = state_a & (CLUT_SIZE - 1);
    unsigned int mask = block_mask[blk];

    if (!(mask & (1u << bit)))
        return -1;

    return block_base[blk] + POPCOUNT32(mask & ((1u << bit) - 1u));
}

/* ================================================================
 * Block SpMV with CLT:  W = H * V   (OpenMP-parallel, row-major)
 *
 * Identical to heisenberg_spmv_block_omp, but replaces
 * lookup[state_a] with clut_lookup_cpu().
 * ================================================================ */
static void heisenberg_spmv_block_clut_omp(
    double             *W,
    const double       *V,
    const int          *block_base,
    const unsigned int *block_mask,
    const int          *basis,
    int dim, int B)
{
    int N_loc   = s_N;
    int d       = s_d;
    int nbonds  = s_nbonds;
    double sv   = s_s;
    double Jv   = s_J;
    double ss1  = s_ss1;
    int dm1     = s_d_minus_1;
    int beta_idx;

    #pragma omp parallel for schedule(static)
    for (beta_idx = 0; beta_idx < dim; beta_idx++) {
        int state, tmp, k, b, si, sj, di, dj, state_a, idx_a, bv;
        int digits[MAX_SITES];
        double mi, mj, mi_a, mj_a, coeff, diag;
        int row_beta, row_a;

        row_beta = beta_idx * B;
        state = basis[beta_idx];

        tmp = state;
        for (k = 0; k < N_loc; k++) {
            digits[k] = tmp % d;
            tmp /= d;
        }

        /* Diagonal part - compute once */
        diag = 0.0;
        for (b = 0; b < nbonds; b++) {
            mi = (double)digits[s_bonds[2*b]]     - sv;
            mj = (double)digits[s_bonds[2*b + 1]] - sv;
            diag += mi * mj;
        }
        diag *= Jv;

        /* Diagonal contribution for all B vectors */
        for (bv = 0; bv < B; bv++) {
            W[row_beta + bv] = diag * V[row_beta + bv];
        }

        /* Off-diagonal */
        for (b = 0; b < nbonds; b++) {
            si = s_bonds[2*b];
            sj = s_bonds[2*b + 1];
            di = digits[si];
            dj = digits[sj];
            mi = (double)di - sv;
            mj = (double)dj - sv;

            /* S+_i S-_j */
            if (di > 0 && dj < dm1) {
                mi_a = mi - 1.0;
                mj_a = mj + 1.0;
                coeff = 0.5 * Jv
                    * sqrt(ss1 - mi_a * (mi_a + 1.0))
                    * sqrt(ss1 - mj_a * (mj_a - 1.0));
                state_a = state - s_powers[si] + s_powers[sj];
                idx_a = clut_lookup_cpu(block_base, block_mask, state_a);
                if (idx_a >= 0) {
                    row_a = idx_a * B;
                    for (bv = 0; bv < B; bv++) {
                        W[row_beta + bv] += coeff * V[row_a + bv];
                    }
                }
            }

            /* S-_i S+_j */
            if (di < dm1 && dj > 0) {
                mi_a = mi + 1.0;
                mj_a = mj - 1.0;
                coeff = 0.5 * Jv
                    * sqrt(ss1 - mi_a * (mi_a - 1.0))
                    * sqrt(ss1 - mj_a * (mj_a + 1.0));
                state_a = state + s_powers[si] - s_powers[sj];
                idx_a = clut_lookup_cpu(block_base, block_mask, state_a);
                if (idx_a >= 0) {
                    row_a = idx_a * B;
                    for (bv = 0; bv < B; bv++) {
                        W[row_beta + bv] += coeff * V[row_a + bv];
                    }
                }
            }
        }
    }
}

/* ================================================================
 * Block BLAS with OpenMP  (row-major: X[i*B + b])
 *
 * All operations work on B vectors simultaneously.
 * Manual reduction because MSVC only supports OpenMP 2.0
 * (no array reduction).
 * ================================================================ */

/* results[b] = sum_i X[i,b] * Y[i,b]  for b = 0..B-1 */
static void block_dot_omp(double *results,
                           const double *X, const double *Y,
                           int dim, int B)
{
    int bv, i;
    for (bv = 0; bv < B; bv++) results[bv] = 0.0;

    #pragma omp parallel
    {
        double local_s[BLOCK_MAX];
        for (bv = 0; bv < B; bv++) local_s[bv] = 0.0;

        #pragma omp for schedule(static)
        for (i = 0; i < dim; i++) {
            int row = i * B;
            for (bv = 0; bv < B; bv++) {
                local_s[bv] += X[row + bv] * Y[row + bv];
            }
        }

        #pragma omp critical
        {
            for (bv = 0; bv < B; bv++) results[bv] += local_s[bv];
        }
    }
}

/* norms[b] = ||X(:,b)||  for b = 0..B-1 */
static void block_nrm2_omp(double *norms, const double *X, int dim, int B)
{
    int bv, i;
    for (bv = 0; bv < B; bv++) norms[bv] = 0.0;

    #pragma omp parallel
    {
        double local_s[BLOCK_MAX];
        for (bv = 0; bv < B; bv++) local_s[bv] = 0.0;

        #pragma omp for schedule(static)
        for (i = 0; i < dim; i++) {
            int row = i * B;
            for (bv = 0; bv < B; bv++) {
                local_s[bv] += X[row + bv] * X[row + bv];
            }
        }

        #pragma omp critical
        {
            for (bv = 0; bv < B; bv++) norms[bv] += local_s[bv];
        }
    }

    for (bv = 0; bv < B; bv++) norms[bv] = sqrt(norms[bv]);
}

/* Y[:,b] -= alpha[b] * X[:,b]  for b = 0..B-1 */
static void block_axpy_neg_omp(const double *alphas,
                                const double *X, double *Y,
                                int dim, int B)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < dim; i++) {
        int row = i * B;
        int bv;
        for (bv = 0; bv < B; bv++) {
            Y[row + bv] -= alphas[bv] * X[row + bv];
        }
    }
}

/* X[:,b] *= factors[b]  for b = 0..B-1 */
static void block_scal_omp(const double *factors, double *X, int dim, int B)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < dim; i++) {
        int row = i * B;
        int bv;
        for (bv = 0; bv < B; bv++) {
            X[row + bv] *= factors[bv];
        }
    }
}

/* ================================================================
 * MEX Gateway
 * ================================================================ */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    char mode[32];
    mxGetString(prhs[0], mode, sizeof(mode));

    /* ==================== INIT ==================== */
    if (strcmp(mode, "init") == 0)
    {
        const int *in_lookup, *in_basis, *in_bonds;
        int lk_n, bs_n, n_entries, nb, k;

        if (s_init) cleanup();

        in_lookup = (const int *)mxGetData(prhs[1]);
        lk_n = (int)mxGetNumberOfElements(prhs[1]);

        in_basis = (const int *)mxGetData(prhs[2]);
        bs_n = (int)mxGetNumberOfElements(prhs[2]);

        in_bonds = (const int *)mxGetData(prhs[3]);
        n_entries = (int)mxGetNumberOfElements(prhs[3]);
        nb = n_entries / 2;

        s_N         = (int)mxGetScalar(prhs[4]);
        s_d         = (int)mxGetScalar(prhs[5]);
        s_s         = mxGetScalar(prhs[6]);
        s_J         = mxGetScalar(prhs[7]);
        s_dim       = (int)mxGetScalar(prhs[8]);
        s_ntotal    = (int)mxGetScalar(prhs[9]);

        s_nbonds    = nb;
        s_ss1       = s_s * (s_s + 1.0);
        s_d_minus_1 = s_d - 1;

        memcpy(s_bonds, in_bonds, 2 * nb * sizeof(int));

        s_powers[0] = 1;
        for (k = 1; k < s_N; k++)
            s_powers[k] = s_powers[k-1] * s_d;

        s_lookup = (int *)mxMalloc(lk_n * sizeof(int));
        mexMakeMemoryPersistent(s_lookup);
        memcpy(s_lookup, in_lookup, lk_n * sizeof(int));

        s_basis = (int *)mxMalloc(bs_n * sizeof(int));
        mexMakeMemoryPersistent(s_basis);
        memcpy(s_basis, in_basis, bs_n * sizeof(int));

        s_v  = (double *)mxMalloc(s_dim * sizeof(double));
        s_vp = (double *)mxMalloc(s_dim * sizeof(double));
        s_w  = (double *)mxMalloc(s_dim * sizeof(double));
        mexMakeMemoryPersistent(s_v);
        mexMakeMemoryPersistent(s_vp);
        mexMakeMemoryPersistent(s_w);

        s_init = 1;
        mexLock();
        mexAtExit(cleanup);

        #ifdef _OPENMP
        mexPrintf("  cpu_lanczos_omp: init OK (dim=%d, %d OpenMP threads)\n",
                  s_dim, omp_get_max_threads());
        #else
        mexPrintf("  cpu_lanczos_omp: init OK (dim=%d, single-threaded)\n",
                  s_dim);
        #endif
    }

    /* ==================== LANCZOS ==================== */
    else if (strcmp(mode, "lanczos") == 0)
    {
        const double *v0;
        int M_lz, n, nsteps, j;
        double *h_al, *h_be;
        double nrm, inv, al, be, inv_be;

        if (!s_init)
            mexErrMsgIdAndTxt("omp:run", "Call 'init' first!");

        v0   = mxGetPr(prhs[1]);
        M_lz = (int)mxGetScalar(prhs[2]);
        n    = s_dim;
        if (M_lz > n) M_lz = n;

        /* Copy v = v0 and normalize */
        memcpy(s_v, v0, n * sizeof(double));
        nrm = nrm2_omp(s_v, n);
        inv = 1.0 / nrm;
        scal_omp(inv, s_v, n);

        memset(s_vp, 0, n * sizeof(double));

        h_al = (double *)mxCalloc(M_lz, sizeof(double));
        h_be = (double *)mxCalloc(M_lz, sizeof(double));

        nsteps = M_lz;

        for (j = 0; j < M_lz; j++) {

            /* w = H * v  (MATRIX-FREE, OpenMP!) */
            heisenberg_spmv_omp(s_w, s_v, s_lookup, s_basis, n);

            /* alpha = v' * w */
            al = dot_omp(s_v, s_w, n);
            h_al[j] = al;

            /* w = w - alpha * v */
            axpy_omp(-al, s_v, s_w, n);

            /* w = w - beta_{j-1} * v_prev */
            if (j > 0) {
                axpy_omp(-h_be[j-1], s_vp, s_w, n);
            }

            /* beta = ||w|| */
            be = nrm2_omp(s_w, n);
            h_be[j] = be;

            if (be < 1e-14) { nsteps = j + 1; break; }

            if (j < M_lz - 1) {
                memcpy(s_vp, s_v, n * sizeof(double));
                inv_be = 1.0 / be;
                scal_omp(inv_be, s_w, n);
                memcpy(s_v, s_w, n * sizeof(double));
            }
        }

        plhs[0] = mxCreateDoubleMatrix(nsteps, 1, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(nsteps, 1, mxREAL);
        memcpy(mxGetPr(plhs[0]), h_al, nsteps * sizeof(double));
        memcpy(mxGetPr(plhs[1]), h_be, nsteps * sizeof(double));
        mxFree(h_al);
        mxFree(h_be);
    }

    /* ==================== BLOCK_LANCZOS ==================== */
    /*
     * B independent Lanczos runs in lockstep.
     * Each step performs ONE block SpMV over all B vectors,
     * amortizing lookup/basis accesses by a factor of B.
     *
     * Call: [AL, BE] = cpu_lanczos_omp('block_lanczos', V0, M_lz)
     *   V0:  dim x B  (B starting vectors, FP64)
     *   AL:  nsteps x B  (alpha coefficients per chain)
     *   BE:  nsteps x B  (beta coefficients per chain)
     *
     * Internally: row-major memory layout V[i*B+b] for optimal
     *             cache usage in the block SpMV.
     */
    else if (strcmp(mode, "block_lanczos") == 0)
    {
        const double *V0_cm;
        int M_lz, n, B, nsteps, j, bv;
        double *V_rm, *VP_rm, *W_rm;
        double *h_al, *h_be;
        double alphas[BLOCK_MAX], betas[BLOCK_MAX], betas_prev[BLOCK_MAX];
        double inv_sc[BLOCK_MAX], min_beta;

        if (!s_init)
            mexErrMsgIdAndTxt("omp:run", "Call 'init' first!");

        V0_cm = mxGetPr(prhs[1]);
        M_lz  = (int)mxGetScalar(prhs[2]);
        n     = s_dim;
        B     = (int)mxGetN(prhs[1]);

        if ((int)mxGetM(prhs[1]) != n)
            mexErrMsgIdAndTxt("omp:dim",
                "V0 must be dim x B! (expected %d rows, got %d)",
                n, (int)mxGetM(prhs[1]));
        if (B > BLOCK_MAX)
            mexErrMsgIdAndTxt("omp:block",
                "B=%d > BLOCK_MAX=%d!", B, BLOCK_MAX);
        if (M_lz > n) M_lz = n;

        /* Allocate work arrays (row-major: dim x B) */
        V_rm  = (double *)mxMalloc((size_t)n * B * sizeof(double));
        VP_rm = (double *)mxCalloc((size_t)n * B, sizeof(double));
        W_rm  = (double *)mxMalloc((size_t)n * B * sizeof(double));

        /* Transpose V0: column-major (MATLAB) -> row-major (intern) */
        {
            int i;
            for (i = 0; i < n; i++) {
                for (bv = 0; bv < B; bv++) {
                    V_rm[i * B + bv] = V0_cm[i + (size_t)n * bv];
                }
            }
        }

        /* Normalize each chain */
        block_nrm2_omp(betas, V_rm, n, B);
        for (bv = 0; bv < B; bv++) inv_sc[bv] = 1.0 / betas[bv];
        block_scal_omp(inv_sc, V_rm, n, B);

        /* Coefficient arrays (column-major for MATLAB output) */
        h_al = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
        h_be = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));

        nsteps = M_lz;

        for (j = 0; j < M_lz; j++) {

            /* W = H * V  (block SpMV: ONE iteration over the basis!) */
            heisenberg_spmv_block_omp(W_rm, V_rm,
                s_lookup, s_basis, n, B);

            /* alpha[j,b] = V(:,b)' * W(:,b) */
            block_dot_omp(alphas, V_rm, W_rm, n, B);
            for (bv = 0; bv < B; bv++)
                h_al[j + (size_t)M_lz * bv] = alphas[bv];

            /* W -= alpha .* V */
            block_axpy_neg_omp(alphas, V_rm, W_rm, n, B);

            /* W -= beta_prev .* VP */
            if (j > 0)
                block_axpy_neg_omp(betas_prev, VP_rm, W_rm, n, B);

            /* beta[j,b] = ||W(:,b)|| */
            block_nrm2_omp(betas, W_rm, n, B);
            for (bv = 0; bv < B; bv++) {
                h_be[j + (size_t)M_lz * bv] = betas[bv];
                betas_prev[bv] = betas[bv];
            }

            /* Convergence: stop when ALL chains have converged */
            min_beta = betas[0];
            for (bv = 1; bv < B; bv++)
                if (betas[bv] < min_beta) min_beta = betas[bv];
            if (min_beta < 1e-14) { nsteps = j + 1; break; }

            if (j < M_lz - 1) {
                /* VP = V */
                memcpy(VP_rm, V_rm, (size_t)n * B * sizeof(double));
                /* V = W / beta */
                for (bv = 0; bv < B; bv++)
                    inv_sc[bv] = 1.0 / betas[bv];
                block_scal_omp(inv_sc, W_rm, n, B);
                memcpy(V_rm, W_rm, (size_t)n * B * sizeof(double));
            }
        }

        /* Output: AL (nsteps x B), BE (nsteps x B) */
        plhs[0] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        {
            double *out_al = mxGetPr(plhs[0]);
            double *out_be = mxGetPr(plhs[1]);
            for (bv = 0; bv < B; bv++) {
                memcpy(out_al + (size_t)nsteps * bv,
                       h_al + (size_t)M_lz * bv,
                       nsteps * sizeof(double));
                memcpy(out_be + (size_t)nsteps * bv,
                       h_be + (size_t)M_lz * bv,
                       nsteps * sizeof(double));
            }
        }

        mxFree(h_al);
        mxFree(h_be);
        mxFree(V_rm);
        mxFree(VP_rm);
        mxFree(W_rm);

        #ifdef _OPENMP
        mexPrintf("  block_lanczos: %d chains x %d steps, %d threads\n",
                  B, nsteps, omp_get_max_threads());
        #endif
    }

    /* ==================== INIT_CLUT ==================== */
    /*
     * init_clut(block_base, block_mask, basis, bonds_flat,
     *           N, d, s, J, dim)
     *
     * block_base: int32   [n_blocks x 1]
     * block_mask: uint32  [n_blocks x 1]
     * basis:      int32   [dim x 1]
     *
     * Stores CLT instead of the full lookup.  Keeps the
     * persistent Lanczos vectors (s_v, s_vp, s_w).
     */
    else if (strcmp(mode, "init_clut") == 0)
    {
        const int *in_bb, *in_bm, *in_basis_c, *in_bonds;
        int bb_n, bm_n, bs_n, n_entries, nb, k;

        if (s_init) cleanup();

        in_bb = (const int *)mxGetData(prhs[1]);
        bb_n  = (int)mxGetNumberOfElements(prhs[1]);

        in_bm = (const int *)mxGetData(prhs[2]);  /* uint32 in MATLAB */
        bm_n  = (int)mxGetNumberOfElements(prhs[2]);

        in_basis_c = (const int *)mxGetData(prhs[3]);
        bs_n = (int)mxGetNumberOfElements(prhs[3]);

        in_bonds = (const int *)mxGetData(prhs[4]);
        n_entries = (int)mxGetNumberOfElements(prhs[4]);
        nb = n_entries / 2;

        s_N         = (int)mxGetScalar(prhs[5]);
        s_d         = (int)mxGetScalar(prhs[6]);
        s_s         = mxGetScalar(prhs[7]);
        s_J         = mxGetScalar(prhs[8]);
        s_dim       = (int)mxGetScalar(prhs[9]);

        s_nbonds    = nb;
        s_ss1       = s_s * (s_s + 1.0);
        s_d_minus_1 = s_d - 1;

        memcpy(s_bonds, in_bonds, 2 * nb * sizeof(int));

        s_powers[0] = 1;
        for (k = 1; k < s_N; k++)
            s_powers[k] = s_powers[k-1] * s_d;

        /* Copy CLT arrays */
        s_block_base = (int *)mxMalloc(bb_n * sizeof(int));
        mexMakeMemoryPersistent(s_block_base);
        memcpy(s_block_base, in_bb, bb_n * sizeof(int));

        s_block_mask = (unsigned int *)mxMalloc(bm_n * sizeof(unsigned int));
        mexMakeMemoryPersistent(s_block_mask);
        memcpy(s_block_mask, (const unsigned int *)in_bm,
               bm_n * sizeof(unsigned int));

        /* Copy basis */
        s_basis = (int *)mxMalloc(bs_n * sizeof(int));
        mexMakeMemoryPersistent(s_basis);
        memcpy(s_basis, in_basis_c, bs_n * sizeof(int));

        /* Lanczos vectors (for single Lanczos, optional) */
        s_v  = (double *)mxMalloc(s_dim * sizeof(double));
        s_vp = (double *)mxMalloc(s_dim * sizeof(double));
        s_w  = (double *)mxMalloc(s_dim * sizeof(double));
        mexMakeMemoryPersistent(s_v);
        mexMakeMemoryPersistent(s_vp);
        mexMakeMemoryPersistent(s_w);

        s_init      = 1;
        s_clut_init = 1;
        mexLock();
        mexAtExit(cleanup);

        #ifdef _OPENMP
        mexPrintf("  cpu_lanczos_omp: init_clut OK (dim=%d, CLT %d+%d entries, %d threads)\n",
                  s_dim, bb_n, bm_n, omp_get_max_threads());
        #else
        mexPrintf("  cpu_lanczos_omp: init_clut OK (dim=%d, single-threaded)\n",
                  s_dim);
        #endif
    }

    /* ==================== BLOCK_LANCZOS_CLUT ==================== */
    /*
     * B independent Lanczos runs with CLT-based SpMV.
     * Identical to block_lanczos, but uses heisenberg_spmv_block_clut_omp.
     *
     * Call: [AL, BE] = cpu_lanczos_omp('block_lanczos_clut', V0, M_lz)
     */
    else if (strcmp(mode, "block_lanczos_clut") == 0)
    {
        const double *V0_cm;
        int M_lz, n, B, nsteps, j, bv;
        double *V_rm, *VP_rm, *W_rm;
        double *h_al, *h_be;
        double alphas[BLOCK_MAX], betas[BLOCK_MAX], betas_prev[BLOCK_MAX];
        double inv_sc[BLOCK_MAX], min_beta;

        if (!s_clut_init)
            mexErrMsgIdAndTxt("omp:run",
                "Call 'init_clut' first for block_lanczos_clut!");

        V0_cm = mxGetPr(prhs[1]);
        M_lz  = (int)mxGetScalar(prhs[2]);
        n     = s_dim;
        B     = (int)mxGetN(prhs[1]);

        if ((int)mxGetM(prhs[1]) != n)
            mexErrMsgIdAndTxt("omp:dim",
                "V0 must be dim x B! (expected %d rows, got %d)",
                n, (int)mxGetM(prhs[1]));
        if (B > BLOCK_MAX)
            mexErrMsgIdAndTxt("omp:block",
                "B=%d > BLOCK_MAX=%d!", B, BLOCK_MAX);
        if (M_lz > n) M_lz = n;

        /* Allocate work arrays (row-major: dim x B) */
        V_rm  = (double *)mxMalloc((size_t)n * B * sizeof(double));
        VP_rm = (double *)mxCalloc((size_t)n * B, sizeof(double));
        W_rm  = (double *)mxMalloc((size_t)n * B * sizeof(double));

        /* Transpose V0: column-major (MATLAB) -> row-major (intern) */
        {
            int i;
            for (i = 0; i < n; i++) {
                for (bv = 0; bv < B; bv++) {
                    V_rm[i * B + bv] = V0_cm[i + (size_t)n * bv];
                }
            }
        }

        /* Normalize each chain */
        block_nrm2_omp(betas, V_rm, n, B);
        for (bv = 0; bv < B; bv++) inv_sc[bv] = 1.0 / betas[bv];
        block_scal_omp(inv_sc, V_rm, n, B);

        /* Coefficient arrays (column-major for MATLAB output) */
        h_al = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
        h_be = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));

        nsteps = M_lz;

        for (j = 0; j < M_lz; j++) {

            /* W = H * V  (block SpMV with CLT!) */
            heisenberg_spmv_block_clut_omp(W_rm, V_rm,
                s_block_base, s_block_mask, s_basis, n, B);

            /* alpha[j,b] = V(:,b)' * W(:,b) */
            block_dot_omp(alphas, V_rm, W_rm, n, B);
            for (bv = 0; bv < B; bv++)
                h_al[j + (size_t)M_lz * bv] = alphas[bv];

            /* W -= alpha .* V */
            block_axpy_neg_omp(alphas, V_rm, W_rm, n, B);

            /* W -= beta_prev .* VP */
            if (j > 0)
                block_axpy_neg_omp(betas_prev, VP_rm, W_rm, n, B);

            /* beta[j,b] = ||W(:,b)|| */
            block_nrm2_omp(betas, W_rm, n, B);
            for (bv = 0; bv < B; bv++) {
                h_be[j + (size_t)M_lz * bv] = betas[bv];
                betas_prev[bv] = betas[bv];
            }

            /* Convergence: stop when ALL chains have converged */
            min_beta = betas[0];
            for (bv = 1; bv < B; bv++)
                if (betas[bv] < min_beta) min_beta = betas[bv];
            if (min_beta < 1e-14) { nsteps = j + 1; break; }

            if (j < M_lz - 1) {
                /* VP = V */
                memcpy(VP_rm, V_rm, (size_t)n * B * sizeof(double));
                /* V = W / beta */
                for (bv = 0; bv < B; bv++)
                    inv_sc[bv] = 1.0 / betas[bv];
                block_scal_omp(inv_sc, W_rm, n, B);
                memcpy(V_rm, W_rm, (size_t)n * B * sizeof(double));
            }
        }

        /* Output: AL (nsteps x B), BE (nsteps x B) */
        plhs[0] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        {
            double *out_al = mxGetPr(plhs[0]);
            double *out_be = mxGetPr(plhs[1]);
            for (bv = 0; bv < B; bv++) {
                memcpy(out_al + (size_t)nsteps * bv,
                       h_al + (size_t)M_lz * bv,
                       nsteps * sizeof(double));
                memcpy(out_be + (size_t)nsteps * bv,
                       h_be + (size_t)M_lz * bv,
                       nsteps * sizeof(double));
            }
        }

        mxFree(h_al);
        mxFree(h_be);
        mxFree(V_rm);
        mxFree(VP_rm);
        mxFree(W_rm);

        #ifdef _OPENMP
        mexPrintf("  block_lanczos_clut: %d chains x %d steps, %d threads\n",
                  B, nsteps, omp_get_max_threads());
        #endif
    }

    /* ==================== CLEANUP ==================== */
    else if (strcmp(mode, "cleanup") == 0)
    {
        cleanup();
        mexUnlock();
    }

    /* ==================== INFO ==================== */
    else if (strcmp(mode, "info") == 0)
    {
        #ifdef _OPENMP
        mexPrintf("cpu_lanczos_omp: OpenMP active, max_threads = %d\n",
                  omp_get_max_threads());
        #else
        mexPrintf("cpu_lanczos_omp: OpenMP NOT available\n");
        #endif
        mexPrintf("  init = %d, dim = %d\n", s_init, s_dim);
        if (nlhs >= 1) {
            #ifdef _OPENMP
            plhs[0] = mxCreateDoubleScalar((double)omp_get_max_threads());
            #else
            plhs[0] = mxCreateDoubleScalar(1.0);
            #endif
        }
    }

    /* ==================== SET_THREADS ==================== */
    else if (strcmp(mode, "set_threads") == 0)
    {
        int nt;
        nt = (int)mxGetScalar(prhs[1]);
        #ifdef _OPENMP
        omp_set_num_threads(nt);
        mexPrintf("  OpenMP threads -> %d\n", nt);
        #else
        mexPrintf("  OpenMP not available, request ignored.\n");
        #endif
        if (nlhs >= 1) {
            #ifdef _OPENMP
            plhs[0] = mxCreateDoubleScalar((double)omp_get_max_threads());
            #else
            plhs[0] = mxCreateDoubleScalar(1.0);
            #endif
        }
    }

    else {
        mexErrMsgIdAndTxt("omp:mode",
            "Unknown mode '%s'. Use 'init','init_clut','lanczos','block_lanczos','block_lanczos_clut','cleanup','info','set_threads'.",
            mode);
    }
}
