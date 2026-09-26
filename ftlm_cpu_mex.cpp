/*
 * ftlm_cpu_mex.cpp
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
 * MATLAB MEX gateway: matrix-free block Lanczos on the CPU (OpenMP).
 *
 * CPU counterpart of ftlm_gpu_mex.cu with the compressed lookup table
 * (CLT): identical model (arbitrary couplings, mixed spins), identical
 * basis order, block Lanczos over B chains in lockstep, row-major
 * vector layout V[i*B + b], FP64 or FP32 arithmetic.
 *
 * Modes:
 *   ftlm_cpu_mex('init', cfg)                 cfg from ftlm.kernel_config
 *                                              (lookup = 'clt',
 *                                               precision = 'double'|'single')
 *   [AL, BE, nsteps] = ftlm_cpu_mex('block_lanczos', V0, M_lz)
 *   W = ftlm_cpu_mex('spmv', V)               (host double in/out; tests)
 *   n = ftlm_cpu_mex('info')                  OpenMP thread count
 *   ftlm_cpu_mex('set_threads', n)
 *   ftlm_cpu_mex('cleanup')
 *
 * Breakdown: chain b is frozen once beta_j <= 64 u ||T_j||_1 (see
 * ftlm_gpu_mex.cu).
 *
 * Reductions (dot products, norms) are summed per thread and the thread
 * partials are combined in thread order, so results are deterministic
 * for a fixed number of threads.
 *
 * Lifetime: after the first 'init' the MEX file stays locked in memory
 * for the rest of the MATLAB session ('cleanup' frees all buffers but
 * does not unlock).  Unloading a DLL whose MSVC OpenMP worker threads
 * are still alive crashes MATLAB; the permanent lock prevents that.
 * To rebuild after use, restart MATLAB first.
 *
 * Compile (Windows / MSVC):
 *   mex ftlm_cpu_mex.cpp COMPFLAGS="$COMPFLAGS /openmp"
 * Compile (Linux / GCC):
 *   mex ftlm_cpu_mex.cpp CXXFLAGS="$CXXFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp"
 * ================================================================
 */

#include "mex.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT32(x) ((int)__popcnt((unsigned int)(x)))
#else
#define POPCOUNT32(x) __builtin_popcount((unsigned int)(x))
#endif

#define MAX_SITES      32
#define MAX_COUPLINGS  512
#define MAX_B          32
#define MAX_TWO_S      15
#define LUT_W          16
#define LUT_SIZE       (LUT_W * LUT_W)
#define CLT_BITS       5
#define CLT_BS         32
#define MAX_THREADS    256
#define BREAKDOWN_C    64.0   /* beta <= c u ||T||_1: invariant subspace */

/* ================================================================
 * Persistent state
 * ================================================================ */
static struct {
    int     init;
    int     prec;               /* 0 = double, 1 = single */
    int     N, n_coup, dim, B_max;
    int     two_s[MAX_SITES], radix[MAX_SITES], power[MAX_SITES];
    int     ci[MAX_COUPLINGS], cj[MAX_COUPLINGS];
    double  J_d[MAX_COUPLINGS], hJ_d[MAX_COUPLINGS], s_d[MAX_SITES];
    float   J_f[MAX_COUPLINGS], hJ_f[MAX_COUPLINGS], s_f[MAX_SITES];
    double  raise_d[LUT_SIZE], lower_d[LUT_SIZE];
    float   raise_f[LUT_SIZE], lower_f[LUT_SIZE];
    int          *basis;
    int          *block_base;
    unsigned int *block_mask;
} g = { 0 };

static void cleanup(void)
{
    if (g.basis)      { mxFree(g.basis);      g.basis      = NULL; }
    if (g.block_base) { mxFree(g.block_base); g.block_base = NULL; }
    if (g.block_mask) { mxFree(g.block_mask); g.block_mask = NULL; }
    g.dim  = 0;
    g.init = 0;
}

/* precision-dependent parameter access */
template <typename T> struct Par;
template <> struct Par<double> {
    static double J (int c) { return g.J_d[c]; }
    static double hJ(int c) { return g.hJ_d[c]; }
    static double s (int k) { return g.s_d[k]; }
    static const double *raise() { return g.raise_d; }
    static const double *lower() { return g.lower_d; }
};
template <> struct Par<float> {
    static float J (int c) { return g.J_f[c]; }
    static float hJ(int c) { return g.hJ_f[c]; }
    static float s (int k) { return g.s_f[k]; }
    static const float *raise() { return g.raise_f; }
    static const float *lower() { return g.lower_f; }
};

static inline float  ftlm_sqrt(float x)  { return sqrtf(x); }
static inline double ftlm_sqrt(double x) { return sqrt(x); }

static inline int clt_lookup(int label)
{
    int blk = label >> CLT_BITS;
    int bit = label & (CLT_BS - 1);
    unsigned int mask = g.block_mask[blk];
    if (!(mask & (1u << bit)))
        return -1;
    return g.block_base[blk] + POPCOUNT32(mask & ((1u << bit) - 1u));
}

/* ================================================================
 * Block SpMV with CLT:  W = H V   (row-wise gather, OpenMP)
 * ================================================================ */
template <typename T>
static void spmv(T *W, const T *V, int B)
{
    const int N = g.N, nc = g.n_coup, dim = g.dim;
    const T *raise = Par<T>::raise();
    const T *lower = Par<T>::lower();
    int t;

    #pragma omp parallel for schedule(static)
    for (t = 0; t < dim; t++) {
        int digits[MAX_SITES];
        int label = g.basis[t];
        int tmp = label, k, c, b;
        for (k = 0; k < N; k++) {
            digits[k] = tmp % g.radix[k];
            tmp /= g.radix[k];
        }

        T diag = (T)0;
        for (c = 0; c < nc; c++) {
            T mi = (T)digits[g.ci[c]] - Par<T>::s(g.ci[c]);
            T mj = (T)digits[g.cj[c]] - Par<T>::s(g.cj[c]);
            diag += Par<T>::J(c) * (mi * mj);
        }
        size_t row = (size_t)t * B;
        for (b = 0; b < B; b++)
            W[row + b] = diag * V[row + b];

        for (c = 0; c < nc; c++) {
            int i = g.ci[c], j = g.cj[c];
            int ai = digits[i], aj = digits[j];
            int ti = g.two_s[i], tj = g.two_s[j];

            if (ai > 0 && aj < tj) {        /* source (a_i - 1, a_j + 1) */
                T coeff = Par<T>::hJ(c) * lower[ti * LUT_W + ai] * raise[tj * LUT_W + aj];
                int idx = clt_lookup(label - g.power[i] + g.power[j]);
                if (idx >= 0) {
                    size_t r = (size_t)idx * B;
                    for (b = 0; b < B; b++) W[row + b] += coeff * V[r + b];
                }
            }
            if (ai < ti && aj > 0) {        /* source (a_i + 1, a_j - 1) */
                T coeff = Par<T>::hJ(c) * raise[ti * LUT_W + ai] * lower[tj * LUT_W + aj];
                int idx = clt_lookup(label + g.power[i] - g.power[j]);
                if (idx >= 0) {
                    size_t r = (size_t)idx * B;
                    for (b = 0; b < B; b++) W[row + b] += coeff * V[r + b];
                }
            }
        }
    }
}

/* ================================================================
 * Block BLAS (row-major X[i*B + b]), deterministic reductions
 * ================================================================ */
static int n_threads_max(void)
{
#ifdef _OPENMP
    int n = omp_get_max_threads();
    return n > MAX_THREADS ? MAX_THREADS : n;
#else
    return 1;
#endif
}

/* res[b] = sum_i X[i,b] * Y[i,b]
 *
 * Each thread sums a contiguous index range in chunks of CHUNK values and
 * combines the chunk sums by cascade (pairwise) summation, so that the
 * rounding error grows like (CHUNK + log2 dim) u rather than dim u.  The
 * thread results are added in thread order (deterministic for a fixed
 * number of threads). */
#define CHUNK 256
#define CASCADE_LEVELS 48
template <typename T>
static void block_dot(T *res, const T *X, const T *Y, int dim, int B)
{
    static T part[MAX_THREADS * MAX_B];
    int nt = n_threads_max(), nt_used = 1;
    memset(part, 0, sizeof(part));

    #pragma omp parallel num_threads(nt)
    {
        int tid = 0, nth = 1, b, L;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nth = omp_get_num_threads();
        #pragma omp single
        nt_used = nth;
#endif
        long long lo = (long long)dim * tid / nth;
        long long hi = (long long)dim * (tid + 1) / nth;
        T lvl[CASCADE_LEVELS][MAX_B];
        T s[MAX_B], tot[MAX_B];
        unsigned long long cnt = 0;
        for (long long c0 = lo; c0 < hi; c0 += CHUNK) {
            long long c1 = (c0 + CHUNK < hi) ? c0 + CHUNK : hi;
            for (b = 0; b < B; b++) s[b] = (T)0;
            for (long long i = c0; i < c1; i++) {
                size_t row = (size_t)i * B;
                for (b = 0; b < B; b++) s[b] += X[row + b] * Y[row + b];
            }
            /* binary-counter cascade: merge partial sums of equal size */
            unsigned long long k = cnt;
            L = 0;
            while (k & 1ULL) {
                for (b = 0; b < B; b++) s[b] += lvl[L][b];
                k >>= 1;
                L++;
            }
            for (b = 0; b < B; b++) lvl[L][b] = s[b];
            cnt++;
        }
        for (b = 0; b < B; b++) tot[b] = (T)0;
        for (L = 0; L < CASCADE_LEVELS; L++)
            if ((cnt >> L) & 1ULL)
                for (b = 0; b < B; b++) tot[b] += lvl[L][b];
        for (b = 0; b < B; b++) part[tid * MAX_B + b] = tot[b];
    }
    for (int b = 0; b < B; b++) {
        T acc = (T)0;
        for (int k = 0; k < nt_used; k++) acc += part[k * MAX_B + b];
        res[b] = acc;
    }
}

/* Y[:,b] -= a[b] * X[:,b] */
template <typename T>
static void block_axpy_neg(const T *a, const T *X, T *Y, int dim, int B)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < dim; i++) {
        size_t row = (size_t)i * B;
        for (int b = 0; b < B; b++) Y[row + b] -= a[b] * X[row + b];
    }
}

/* X[:,b] *= f[b] */
template <typename T>
static void block_scal(const T *f, T *X, int dim, int B)
{
    int i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < dim; i++) {
        size_t row = (size_t)i * B;
        for (int b = 0; b < B; b++) X[row + b] *= f[b];
    }
}

/* ================================================================
 * Import / export (MATLAB column-major <-> row-major interleaved)
 * ================================================================ */
template <typename T>
static int import_vectors(T *dst, const mxArray *mV)
{
    int n = (int)mxGetM(mV), B = (int)mxGetN(mV);
    if (n != g.dim || B < 1 || B > MAX_B || (!mxIsDouble(mV) && !mxIsSingle(mV)))
        mexErrMsgIdAndTxt("ftlm_cpu:V", "V is %d x %d, expected single/double %d x (1..%d).",
                          n, B, g.dim, MAX_B);
    int i;
    if (mxIsDouble(mV)) {
        const double *src = mxGetPr(mV);
        #pragma omp parallel for schedule(static)
        for (i = 0; i < n; i++)
            for (int b = 0; b < B; b++)
                dst[(size_t)i * B + b] = (T)src[i + (size_t)b * n];
    } else {
        const float *src = (const float *)mxGetData(mV);
        #pragma omp parallel for schedule(static)
        for (i = 0; i < n; i++)
            for (int b = 0; b < B; b++)
                dst[(size_t)i * B + b] = (T)src[i + (size_t)b * n];
    }
    return B;
}

template <typename T>
static void do_spmv(mxArray *plhs[], const mxArray *mV)
{
    int n = g.dim, B = (int)mxGetN(mV);
    T *V = (T *)mxMalloc((size_t)n * (B > 0 ? B : 1) * sizeof(T));
    T *W = (T *)mxMalloc((size_t)n * (B > 0 ? B : 1) * sizeof(T));
    import_vectors<T>(V, mV);
    spmv<T>(W, V, B);
    plhs[0] = mxCreateDoubleMatrix(n, B, mxREAL);
    double *out = mxGetPr(plhs[0]);
    for (int i = 0; i < n; i++)
        for (int b = 0; b < B; b++)
            out[i + (size_t)b * n] = (double)W[(size_t)i * B + b];
    mxFree(V);
    mxFree(W);
}

/* ================================================================
 * Block Lanczos (B independent chains in lockstep)
 * ================================================================ */
template <typename T>
static void do_block_lanczos(int nlhs, mxArray *plhs[], const mxArray *mV, int M_lz)
{
    const int n = g.dim;
    const int B = (int)mxGetN(mV);
    if (M_lz > n) M_lz = n;
    if (M_lz < 1) mexErrMsgIdAndTxt("ftlm_cpu:M_lz", "M_lz must be >= 1.");

    T *pv  = (T *)mxMalloc((size_t)n * B * sizeof(T));
    T *pvp = (T *)mxCalloc((size_t)n * B, sizeof(T));
    T *pw  = (T *)mxMalloc((size_t)n * B * sizeof(T));
    import_vectors<T>(pv, mV);

    T alpha[MAX_B], beta[MAX_B], beta_prev[MAX_B], sc[MAX_B];
    int    nsteps[MAX_B], active[MAX_B];
    double normT[MAX_B];
    const double u = (g.prec == 0) ? ldexp(1.0, -53) : ldexp(1.0, -24);
    const double tol_fac = BREAKDOWN_C * u;

    /* normalize each chain */
    block_dot<T>(sc, pv, pv, n, B);
    for (int b = 0; b < B; b++) sc[b] = (T)1 / ftlm_sqrt(sc[b]);
    block_scal<T>(sc, pv, n, B);

    double *h_AL = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
    double *h_BE = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
    for (int b = 0; b < B; b++) {
        nsteps[b] = M_lz; active[b] = 1; normT[b] = 0.0; beta_prev[b] = (T)0;
    }

    for (int j = 0; j < M_lz; j++) {
        spmv<T>(pw, pv, B);

        block_dot<T>(alpha, pv, pw, n, B);
        for (int b = 0; b < B; b++)
            if (active[b]) h_AL[j + (size_t)b * M_lz] = (double)alpha[b];

        block_axpy_neg<T>(alpha, pv, pw, n, B);
        if (j > 0) block_axpy_neg<T>(beta_prev, pvp, pw, n, B);

        block_dot<T>(beta, pw, pw, n, B);
        int n_active = 0;
        for (int b = 0; b < B; b++) {
            beta[b] = ftlm_sqrt(beta[b]);
            if (!active[b]) { sc[b] = (T)0; continue; }
            h_BE[j + (size_t)b * M_lz] = (double)beta[b];
            double tj = fabs((double)alpha[b]) + (double)beta[b] + (double)beta_prev[b];
            if (tj > normT[b]) normT[b] = tj;
            if ((double)beta[b] <= tol_fac * normT[b]) {
                active[b] = 0; nsteps[b] = j + 1; sc[b] = (T)0;
            } else {
                n_active++; sc[b] = (T)1 / beta[b];
            }
        }
        if (n_active == 0 || j == M_lz - 1) break;

        block_scal<T>(sc, pw, n, B);
        T *tmp = pvp; pvp = pv; pv = pw; pw = tmp;     /* vp <- v, v <- w */
        for (int b = 0; b < B; b++) beta_prev[b] = active[b] ? beta[b] : (T)0;
    }

    int n_max = 0;
    for (int b = 0; b < B; b++) if (nsteps[b] > n_max) n_max = nsteps[b];
    plhs[0] = mxCreateDoubleMatrix(n_max, B, mxREAL);
    if (nlhs > 1) plhs[1] = mxCreateDoubleMatrix(n_max, B, mxREAL);
    if (nlhs > 2) plhs[2] = mxCreateDoubleMatrix(1, B, mxREAL);
    for (int b = 0; b < B; b++) {
        memcpy(mxGetPr(plhs[0]) + (size_t)b * n_max, h_AL + (size_t)b * M_lz,
               n_max * sizeof(double));
        if (nlhs > 1)
            memcpy(mxGetPr(plhs[1]) + (size_t)b * n_max, h_BE + (size_t)b * M_lz,
                   n_max * sizeof(double));
        if (nlhs > 2) mxGetPr(plhs[2])[b] = (double)nsteps[b];
    }
    mxFree(h_AL); mxFree(h_BE);
    mxFree(pv); mxFree(pvp); mxFree(pw);
}

/* ================================================================
 * cfg helpers
 * ================================================================ */
static const mxArray *field(const mxArray *s, const char *name)
{
    const mxArray *f = mxGetField(s, 0, name);
    if (!f) mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.%s is missing.", name);
    return f;
}

static int int_array_field(const mxArray *s, const char *name, int *out, int n_max)
{
    const mxArray *f = field(s, name);
    int n = (int)mxGetNumberOfElements(f);
    if (n > n_max)
        mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.%s has %d > %d entries.", name, n, n_max);
    if (mxIsInt32(f) || mxIsUint32(f)) {
        memcpy(out, mxGetData(f), n * sizeof(int));
    } else if (mxIsDouble(f)) {
        const double *p = mxGetPr(f);
        for (int k = 0; k < n; k++) out[k] = (int)p[k];
    } else {
        mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.%s must be int32 or double.", name);
    }
    return n;
}

static void *copy_int_array(const mxArray *f, const char *name, int *n_out)
{
    if (!mxIsInt32(f) && !mxIsUint32(f))
        mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.%s must be a host int32/uint32 array.", name);
    size_t n = mxGetNumberOfElements(f);
    void *p = mxMalloc((n > 0 ? n : 1) * 4);
    mexMakeMemoryPersistent(p);
    memcpy(p, mxGetData(f), n * 4);
    *n_out = (int)n;
    return p;
}

static void fill_lut(void)
{
    memset(g.raise_d, 0, sizeof(g.raise_d)); memset(g.lower_d, 0, sizeof(g.lower_d));
    memset(g.raise_f, 0, sizeof(g.raise_f)); memset(g.lower_f, 0, sizeof(g.lower_f));
    for (int ts = 0; ts <= MAX_TWO_S; ts++) {
        double s = 0.5 * ts;
        float  s_f = 0.5f * (float)ts, ss1f = s_f * (s_f + 1.0f);
        for (int a = 0; a <= ts && a < LUT_W; a++) {
            int k = ts * LUT_W + a;
            double m = a - s;
            float  mf = (float)a - s_f;
            if (a < ts) {
                g.raise_d[k] = sqrt(s * (s + 1.0) - m * (m + 1.0));
                g.raise_f[k] = sqrtf(ss1f - mf * (mf + 1.0f));
            }
            if (a > 0) {
                g.lower_d[k] = sqrt(s * (s + 1.0) - m * (m - 1.0));
                g.lower_f[k] = sqrtf(ss1f - mf * (mf - 1.0f));
            }
        }
    }
}

/* ================================================================
 * MEX gateway
 * ================================================================ */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char mode[32];
    if (nrhs < 1 || mxGetString(prhs[0], mode, sizeof(mode)) != 0)
        mexErrMsgIdAndTxt("ftlm_cpu:mode", "First argument must be a mode string.");

    if (strcmp(mode, "init") == 0) {
        if (nrhs < 2 || !mxIsStruct(prhs[1]))
            mexErrMsgIdAndTxt("ftlm_cpu:init", "Usage: ftlm_cpu_mex('init', cfg)");
        const mxArray *cfg = prhs[1];
        if (g.init) cleanup();

        char buf[32];
        if (mxGetString(field(cfg, "lookup"), buf, sizeof(buf)) != 0 || strcmp(buf, "clt") != 0)
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "The CPU kernel supports cfg.lookup = 'clt' only.");
        if (mxGetString(field(cfg, "precision"), buf, sizeof(buf)) != 0)
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.precision must be a char vector.");
        if      (strcmp(buf, "double") == 0) g.prec = 0;
        else if (strcmp(buf, "single") == 0) g.prec = 1;
        else mexErrMsgIdAndTxt("ftlm_cpu:cfg", "CPU precision must be 'double' or 'single'.");

        g.N = (int)mxGetScalar(field(cfg, "N"));
        if (g.N < 1 || g.N > MAX_SITES)
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "N = %d outside [1, %d].", g.N, MAX_SITES);
        double dim_d = mxGetScalar(field(cfg, "dim"));
        if (dim_d < 1 || dim_d > 2147483647.0)
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "dim = %g outside [1, 2^31-1].", dim_d);
        g.dim = (int)dim_d;

        if (int_array_field(cfg, "two_s", g.two_s, MAX_SITES) != g.N ||
            int_array_field(cfg, "power", g.power, MAX_SITES) != g.N)
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.two_s and cfg.power need N entries.");
        for (int k = 0; k < g.N; k++) {
            if (g.two_s[k] < 1 || g.two_s[k] > MAX_TWO_S)
                mexErrMsgIdAndTxt("ftlm_cpu:cfg", "two_s(%d) = %d outside [1, %d].",
                                  k + 1, g.two_s[k], MAX_TWO_S);
            g.radix[k] = g.two_s[k] + 1;
            g.s_d[k] = 0.5 * g.two_s[k];
            g.s_f[k] = 0.5f * (float)g.two_s[k];
        }

        int nci = int_array_field(cfg, "ci", g.ci, MAX_COUPLINGS);
        int ncj = int_array_field(cfg, "cj", g.cj, MAX_COUPLINGS);
        const mxArray *fJ = field(cfg, "J");
        if (nci != ncj || nci != (int)mxGetNumberOfElements(fJ) || !mxIsDouble(fJ))
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "cfg.ci, cfg.cj, cfg.J (double) must have equal length.");
        g.n_coup = nci;
        const double *J = mxGetPr(fJ);
        for (int c = 0; c < nci; c++) {
            if (g.ci[c] < 0 || g.ci[c] >= g.N || g.cj[c] < 0 || g.cj[c] >= g.N || g.ci[c] == g.cj[c])
                mexErrMsgIdAndTxt("ftlm_cpu:cfg", "invalid coupling %d.", c + 1);
            g.J_d[c] = J[c];  g.hJ_d[c] = 0.5 * J[c];
            g.J_f[c] = (float)J[c];  g.hJ_f[c] = 0.5f * (float)J[c];
        }
        fill_lut();

        int nb = 0, nm = 0, ns = 0;
        g.block_base = (int *)copy_int_array(field(cfg, "block_base"), "block_base", &nb);
        g.block_mask = (unsigned int *)copy_int_array(field(cfg, "block_mask"), "block_mask", &nm);
        g.basis      = (int *)copy_int_array(field(cfg, "basis"), "basis", &ns);
        if (nb != nm || ns != g.dim) {
            cleanup();
            mexErrMsgIdAndTxt("ftlm_cpu:cfg", "inconsistent CLT/basis sizes.");
        }
        g.init = 1;
        if (!mexIsLocked()) mexLock();
        mexAtExit(cleanup);
    }
    else if (strcmp(mode, "block_lanczos") == 0 || strcmp(mode, "spmv") == 0) {
        if (!g.init) mexErrMsgIdAndTxt("ftlm_cpu:run", "Call 'init' first.");
        bool lz = (strcmp(mode, "block_lanczos") == 0);
        if (nrhs < (lz ? 3 : 2))
            mexErrMsgIdAndTxt("ftlm_cpu:run", "Missing arguments.");
        if (lz) {
            int M_lz = (int)mxGetScalar(prhs[2]);
            if (g.prec == 0) do_block_lanczos<double>(nlhs, plhs, prhs[1], M_lz);
            else             do_block_lanczos<float>(nlhs, plhs, prhs[1], M_lz);
        } else {
            if (g.prec == 0) do_spmv<double>(plhs, prhs[1]);
            else             do_spmv<float>(plhs, prhs[1]);
        }
    }
    /* Frees all persistent buffers; the MEX file stays locked (see header). */
    else if (strcmp(mode, "cleanup") == 0) {
        cleanup();
    }
    else if (strcmp(mode, "info") == 0) {
        plhs[0] = mxCreateDoubleScalar((double)n_threads_max());
    }
    else if (strcmp(mode, "set_threads") == 0) {
        if (nrhs < 2) mexErrMsgIdAndTxt("ftlm_cpu:run", "Usage: ftlm_cpu_mex('set_threads', n)");
#ifdef _OPENMP
        omp_set_num_threads((int)mxGetScalar(prhs[1]));
#endif
        plhs[0] = mxCreateDoubleScalar((double)n_threads_max());
    }
    else {
        mexErrMsgIdAndTxt("ftlm_cpu:mode",
            "Unknown mode '%s'. Use 'init', 'block_lanczos', 'spmv', 'info', 'set_threads', 'cleanup'.",
            mode);
    }
}
