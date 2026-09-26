/*
 * ftlm_gpu_mex.cu
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
 * MATLAB MEX gateway: matrix-free block Lanczos on the GPU.
 *
 * One MEX file for both state-to-index strategies (CLT, CR) and all
 * storage precisions (double, single, half, bfloat16).  The device
 * code lives in cuda/ftlm_kernels.cuh (shared with the Python package).
 *
 * Modes:
 *   ftlm_gpu_mex('init', cfg)
 *       cfg: struct built by ftlm.kernel_config (see +ftlm/kernel_config.m)
 *   [AL, BE, nsteps] = ftlm_gpu_mex('block_lanczos', V0, M_lz)
 *       V0:  dim x B start vectors (gpuArray or host; single or double)
 *       AL, BE: M_lz x B Lanczos coefficients (double); entries beyond
 *               nsteps(b) are zero.  BE(j,b) is beta_j (norm after step j).
 *       nsteps: 1 x B number of valid steps per chain
 *   W = ftlm_gpu_mex('spmv', V)
 *       H * V in the configured precision (host double in/out; for tests)
 *   info = ftlm_gpu_mex('info')
 *   ftlm_gpu_mex('cleanup')
 *
 * Breakdown handling: chain b is frozen (and its coefficients no longer
 * recorded) as soon as beta_j <= c * u * ||T_j||_1, with u the unit
 * roundoff of the storage precision, ||T_j||_1 the running Gershgorin
 * bound of the tridiagonal matrix, c = 64 for FP64/FP32 and c = 8 for
 * the 16-bit formats.  At an exhausted Krylov space the computed beta is
 * observed at 4-16 u ||T||; stopping at beta = eps changes Ritz values
 * and weights only at O(eps^2 / gap^2).  The loop ends when all chains
 * are frozen or M_lz steps are done.
 *
 * Compile:
 *   mexcuda ftlm_gpu_mex.cu
 * ================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include "cuda/ftlm_kernels.cuh"

#define FTLM_BREAKDOWN_C      64.0   /* FP64, FP32   */
#define FTLM_BREAKDOWN_C_16BIT 8.0   /* FP16, BF16   */

enum { PREC_DOUBLE = 0, PREC_SINGLE = 1, PREC_HALF = 2, PREC_BF16 = 3 };

/* ================================================================
 * Persistent state
 * ================================================================ */
static struct {
    bool          init;
    int           prec;
    int           lookup;
    int           dim;
    int           B_max;
    int           dcum_size;
    int           uniform;       /* all local spins equal: register fast path */
    size_t        elem;          /* bytes per stored vector entry */
    double        sigma;         /* norm of stored Lanczos vectors */
    double        u;             /* unit roundoff of storage type */
    void         *d_v, *d_vp, *d_w, *d_tmp;
    void         *d_partial, *d_alpha, *d_beta, *d_beta_prev;
    int          *d_block_base;
    unsigned int *d_block_mask;
    int          *d_basis;
    int          *d_dcum;
} g = { false };

static void ftlm_free(void **p) { if (*p) { cudaFree(*p); *p = NULL; } }

static void cleanup_all(void)
{
    ftlm_free(&g.d_v);  ftlm_free(&g.d_vp);  ftlm_free(&g.d_w);
    ftlm_free(&g.d_tmp);
    ftlm_free(&g.d_partial);  ftlm_free(&g.d_alpha);
    ftlm_free(&g.d_beta);     ftlm_free(&g.d_beta_prev);
    ftlm_free((void **)&g.d_block_base);
    ftlm_free((void **)&g.d_block_mask);
    ftlm_free((void **)&g.d_basis);
    ftlm_free((void **)&g.d_dcum);
    g.init = false;
}

static void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        cleanup_all();
        mexErrMsgIdAndTxt("ftlm_gpu:cuda", "%s failed: %s", what,
                          cudaGetErrorString(err));
    }
}

static void *dev_alloc(size_t bytes, const char *what)
{
    void *p = NULL;
    if (bytes == 0) bytes = 1;
    cuda_check(cudaMalloc(&p, bytes), what);
    return p;
}

/* ================================================================
 * cfg struct access
 * ================================================================ */
static const mxArray *field(const mxArray *s, const char *name, bool required)
{
    const mxArray *f = mxGetField(s, 0, name);
    if (!f && required)
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.%s is missing.", name);
    return f;
}

static double scalar_field(const mxArray *s, const char *name)
{
    return mxGetScalar(field(s, name, true));
}

static void string_field(const mxArray *s, const char *name, char *buf, int n)
{
    const mxArray *f = field(s, name, true);
    if (!mxIsChar(f) || mxGetString(f, buf, n) != 0)
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.%s must be a char vector.", name);
}

/* copy numeric field (any class) to int buffer */
static int int_array_field(const mxArray *s, const char *name, int *out, int n_max)
{
    const mxArray *f = field(s, name, true);
    int n = (int)mxGetNumberOfElements(f);
    if (n > n_max)
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.%s has %d > %d entries.", name, n, n_max);
    if (mxIsInt32(f) || mxIsUint32(f)) {
        memcpy(out, mxGetData(f), n * sizeof(int));
    } else if (mxIsDouble(f)) {
        const double *p = mxGetPr(f);
        for (int k = 0; k < n; k++) out[k] = (int)p[k];
    } else {
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.%s must be int32 or double.", name);
    }
    return n;
}

/* upload int32/uint32 array (host or gpuArray) to a new device buffer */
static void *upload_int_array(const mxArray *f, const char *name, int *n_out)
{
    if (mxIsGPUArray(f)) {
        const mxGPUArray *ga = mxGPUCreateFromMxArray(f);
        mxClassID cid = mxGPUGetClassID(ga);
        if (cid != mxINT32_CLASS && cid != mxUINT32_CLASS) {
            mxGPUDestroyGPUArray(ga);
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.%s must be int32/uint32.", name);
        }
        size_t n = (size_t)mxGPUGetNumberOfElements(ga);
        void *d = dev_alloc(n * 4, name);
        cuda_check(cudaMemcpy(d, mxGPUGetDataReadOnly(ga), n * 4,
                              cudaMemcpyDeviceToDevice), name);
        mxGPUDestroyGPUArray(ga);
        if (n_out) *n_out = (int)n;
        return d;
    }
    if (!mxIsInt32(f) && !mxIsUint32(f))
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.%s must be int32/uint32.", name);
    size_t n = mxGetNumberOfElements(f);
    void *d = dev_alloc(n * 4, name);
    cuda_check(cudaMemcpy(d, mxGetData(f), n * 4, cudaMemcpyHostToDevice), name);
    if (n_out) *n_out = (int)n;
    return d;
}

/* ================================================================
 * Constant-memory setup
 * ================================================================ */
static void fill_lut(FtlmConst *h)
{
    for (int ts = 0; ts <= FTLM_MAX_TWO_S; ts++) {
        double s    = 0.5 * ts;
        float  s_f  = 0.5f * (float)ts;
        float  ss1f = s_f * (s_f + 1.0f);
        for (int a = 0; a <= ts && a < FTLM_LUT_W; a++) {
            int    k = ts * FTLM_LUT_W + a;
            double m = a - s;
            float  mf = (float)a - s_f;
            if (a < ts) {    /* raise: m -> m + 1 */
                h->lut_raise_d[k] = sqrt(s * (s + 1.0) - m * (m + 1.0));
                h->lut_raise_f[k] = sqrtf(ss1f - mf * (mf + 1.0f));
            }
            if (a > 0) {     /* lower: m -> m - 1 */
                h->lut_lower_d[k] = sqrt(s * (s + 1.0) - m * (m - 1.0));
                h->lut_lower_f[k] = sqrtf(ss1f - mf * (mf - 1.0f));
            }
        }
    }
}

static void setup_constants(const mxArray *cfg)
{
    static FtlmConst h;
    memset(&h, 0, sizeof(h));

    h.N = (int)scalar_field(cfg, "N");
    if (h.N < 1 || h.N > FTLM_MAX_SITES)
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "N = %d outside [1, %d].", h.N, FTLM_MAX_SITES);

    if (int_array_field(cfg, "two_s", h.two_s, FTLM_MAX_SITES) != h.N)
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "numel(cfg.two_s) must equal N.");
    g.uniform = 1;
    for (int k = 0; k < h.N; k++) {
        if (h.two_s[k] != h.two_s[0]) g.uniform = 0;
        if (h.two_s[k] < 1 || h.two_s[k] > FTLM_MAX_TWO_S)
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "two_s(%d) = %d outside [1, %d].",
                              k + 1, h.two_s[k], FTLM_MAX_TWO_S);
        h.radix[k] = h.two_s[k] + 1;
        h.s_d[k]   = 0.5 * h.two_s[k];
        h.s_f[k]   = 0.5f * (float)h.two_s[k];
    }

    int nci = int_array_field(cfg, "ci", h.ci, FTLM_MAX_COUPLINGS);
    int ncj = int_array_field(cfg, "cj", h.cj, FTLM_MAX_COUPLINGS);
    const mxArray *fJ = field(cfg, "J", true);
    int nJ = (int)mxGetNumberOfElements(fJ);
    if (nci != ncj || nci != nJ || !mxIsDouble(fJ))
        mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.ci, cfg.cj, cfg.J (double) must have equal length.");
    h.n_coup = nci;
    const double *J = mxGetPr(fJ);
    for (int c = 0; c < nci; c++) {
        if (h.ci[c] < 0 || h.ci[c] >= h.N || h.cj[c] < 0 || h.cj[c] >= h.N || h.ci[c] == h.cj[c])
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "invalid coupling %d: (%d, %d).",
                              c + 1, h.ci[c], h.cj[c]);
        h.J_d[c]  = J[c];
        h.hJ_d[c] = 0.5 * J[c];
        h.J_f[c]  = (float)J[c];
        h.hJ_f[c] = 0.5f * (float)J[c];
    }

    if (g.lookup == FTLM_LOOKUP_CLT) {
        if (int_array_field(cfg, "power", h.power, FTLM_MAX_SITES) != h.N)
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "numel(cfg.power) must equal N.");
    } else {
        if (int_array_field(cfg, "shift", h.shift, FTLM_MAX_SITES) != h.N)
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "numel(cfg.shift) must equal N.");
        for (int k = 0; k < h.N; k++) {
            int bits = 0;
            while ((1 << bits) < h.radix[k]) bits++;
            h.mask[k] = (1 << bits) - 1;
        }
        h.A_total      = (int)scalar_field(cfg, "A_total");
        h.dcum_pstride = (int)scalar_field(cfg, "dcum_pstride");
        h.dcum_astride = (int)scalar_field(cfg, "dcum_astride");
        h.dcum_size    = g.dcum_size;
    }

    fill_lut(&h);
    cuda_check(cudaMemcpyToSymbol(c_p, &h, sizeof(FtlmConst)), "cudaMemcpyToSymbol");
}

/* ================================================================
 * Templated device-side driver
 * ================================================================ */
template <typename TS, typename TC>
static void launch_spmv(TS *W, const TS *V, int B)
{
    int blocks = (g.dim + FTLM_SPMV_BS - 1) / FTLM_SPMV_BS;
    size_t shmem = (size_t)g.dcum_size * sizeof(int);
    if (g.lookup == FTLM_LOOKUP_CLT) {
        if (g.uniform)
            ftlm_spmv<TS, TC, FTLM_LOOKUP_CLT, 1><<<blocks, FTLM_SPMV_BS>>>(
                W, V, g.d_block_base, g.d_block_mask, g.d_basis, NULL, g.dim, B);
        else
            ftlm_spmv<TS, TC, FTLM_LOOKUP_CLT, 0><<<blocks, FTLM_SPMV_BS>>>(
                W, V, g.d_block_base, g.d_block_mask, g.d_basis, NULL, g.dim, B);
    } else {
        if (g.uniform)
            ftlm_spmv<TS, TC, FTLM_LOOKUP_CR, 1><<<blocks, FTLM_SPMV_BS, shmem>>>(
                W, V, NULL, NULL, NULL, g.d_dcum, g.dim, B);
        else
            ftlm_spmv<TS, TC, FTLM_LOOKUP_CR, 0><<<blocks, FTLM_SPMV_BS, shmem>>>(
                W, V, NULL, NULL, NULL, g.d_dcum, g.dim, B);
    }
}

/* host <-> device helper for the B-vector coefficient arrays */
template <typename TC>
static void to_dev(void *d, const TC *h, int B)
{
    cuda_check(cudaMemcpy(d, h, B * sizeof(TC), cudaMemcpyHostToDevice), "H2D");
}
template <typename TC>
static void to_host(TC *h, const void *d, int B)
{
    cuda_check(cudaMemcpy(h, d, B * sizeof(TC), cudaMemcpyDeviceToHost), "D2H");
}

static inline float  ftlm_sqrt(float x)  { return sqrtf(x); }
static inline double ftlm_sqrt(double x) { return sqrt(x); }

/* Import a dim x B column-major array (single or double) into d_v.
 * gpuArray input is read in place; host input is uploaded column by
 * column through a staging buffer of dim doubles, so that no second
 * dim x B copy is held in device memory. */
template <typename TS, typename TC>
static int import_vectors(const mxArray *mV)
{
    int n, B;
    int blocks = (g.dim + FTLM_SPMV_BS - 1) / FTLM_SPMV_BS;
    if (mxIsGPUArray(mV)) {
        const mxGPUArray *ga = mxGPUCreateFromMxArray(mV);
        const mwSize *dims = mxGPUGetDimensions(ga);
        n = (int)dims[0];
        B = (mxGPUGetNumberOfDimensions(ga) > 1) ? (int)dims[1] : 1;
        mxClassID cid = mxGPUGetClassID(ga);
        if (n != g.dim || B < 1 || B > g.B_max ||
            (cid != mxDOUBLE_CLASS && cid != mxSINGLE_CLASS)) {
            mxGPUDestroyGPUArray(ga);
            mexErrMsgIdAndTxt("ftlm_gpu:V",
                "V is %d x %d, expected single/double %d x (1..%d).", n, B, g.dim, g.B_max);
        }
        if (cid == mxDOUBLE_CLASS)
            ftlm_import<TS, TC, double><<<blocks, FTLM_SPMV_BS>>>(
                (TS *)g.d_v, (const double *)mxGPUGetDataReadOnly(ga), n, B, 0, B);
        else
            ftlm_import<TS, TC, float><<<blocks, FTLM_SPMV_BS>>>(
                (TS *)g.d_v, (const float *)mxGPUGetDataReadOnly(ga), n, B, 0, B);
        cuda_check(cudaDeviceSynchronize(), "import kernel");
        mxGPUDestroyGPUArray(ga);
    } else {
        n = (int)mxGetM(mV);
        B = (int)mxGetN(mV);
        if (n != g.dim || B < 1 || B > g.B_max || (!mxIsDouble(mV) && !mxIsSingle(mV)))
            mexErrMsgIdAndTxt("ftlm_gpu:V",
                "V is %d x %d, expected single/double %d x (1..%d).", n, B, g.dim, g.B_max);
        size_t in_elem = mxIsDouble(mV) ? 8 : 4;
        const char *src = (const char *)mxGetData(mV);
        for (int b = 0; b < B; b++) {
            cuda_check(cudaMemcpy(g.d_tmp, src + (size_t)b * n * in_elem, (size_t)n * in_elem,
                                  cudaMemcpyHostToDevice), "V0 upload");
            if (in_elem == 8)
                ftlm_import<TS, TC, double><<<blocks, FTLM_SPMV_BS>>>(
                    (TS *)g.d_v, (const double *)g.d_tmp, n, B, b, 1);
            else
                ftlm_import<TS, TC, float><<<blocks, FTLM_SPMV_BS>>>(
                    (TS *)g.d_v, (const float *)g.d_tmp, n, B, b, 1);
        }
    }
    cuda_check(cudaGetLastError(), "import kernel");
    return B;
}

template <typename TS, typename TC>
static void do_spmv(int nlhs, mxArray *plhs[], const mxArray *mV)
{
    int B = import_vectors<TS, TC>(mV);
    int blocks = (g.dim + FTLM_SPMV_BS - 1) / FTLM_SPMV_BS;
    launch_spmv<TS, TC>((TS *)g.d_w, (const TS *)g.d_v, B);
    cuda_check(cudaGetLastError(), "spmv kernel");
    plhs[0] = mxCreateDoubleMatrix(g.dim, B, mxREAL);
    for (int b = 0; b < B; b++) {
        ftlm_export<TS, TC, double><<<blocks, FTLM_SPMV_BS>>>(
            (double *)g.d_tmp, (const TS *)g.d_w, g.dim, B, b, 1);
        cuda_check(cudaGetLastError(), "export kernel");
        cuda_check(cudaMemcpy(mxGetPr(plhs[0]) + (size_t)b * g.dim, g.d_tmp,
                              (size_t)g.dim * 8, cudaMemcpyDeviceToHost), "W download");
    }
}

template <typename TS, typename TC>
static void do_block_lanczos(int nlhs, mxArray *plhs[], const mxArray *mV, int M_lz)
{
    const int n = g.dim;
    int B = import_vectors<TS, TC>(mV);
    if (M_lz > n) M_lz = n;
    if (M_lz < 1) mexErrMsgIdAndTxt("ftlm_gpu:M_lz", "M_lz must be >= 1.");

    const int blocks  = (n + FTLM_SPMV_BS - 1) / FTLM_SPMV_BS;
    const int rblocks = (n + FTLM_REDUCE_BS - 1) / FTLM_REDUCE_BS;
    const TC  sigma   = (TC)g.sigma;
    const TC  sigma2  = sigma * sigma;

    /* Normalize each chain to norm sigma */
    {
        TC nrm2[FTLM_MAX_B], sc[FTLM_MAX_B];
        ftlm_dot_partial<TS, TC><<<rblocks, FTLM_REDUCE_BS>>>(
            (TC *)g.d_partial, (const TS *)g.d_v, (const TS *)g.d_v, n, B);
        ftlm_reduce_partial<TC><<<1, B>>>((TC *)g.d_alpha, (const TC *)g.d_partial,
                                          rblocks, B);
        to_host<TC>(nrm2, g.d_alpha, B);
        for (int b = 0; b < B; b++)
            sc[b] = sigma / ftlm_sqrt(nrm2[b]);
        to_dev<TC>(g.d_alpha, sc, B);
        ftlm_scale<TS, TC><<<blocks, FTLM_SPMV_BS>>>((TS *)g.d_v, (const TC *)g.d_alpha, n, B);
    }
    cuda_check(cudaMemset(g.d_vp, 0, (size_t)n * B * g.elem), "memset vp");

    double *h_AL = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
    double *h_BE = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
    int     nsteps[FTLM_MAX_B];
    int     active[FTLM_MAX_B];
    double  normT[FTLM_MAX_B];
    TC      h_alpha[FTLM_MAX_B], h_beta[FTLM_MAX_B], h_beta_prev[FTLM_MAX_B];
    TC      h_beta_sq[FTLM_MAX_B], h_scale[FTLM_MAX_B];
    for (int b = 0; b < B; b++) {
        nsteps[b] = M_lz; active[b] = 1; normT[b] = 0.0;
        h_beta_prev[b] = (TC)0;
    }
    const double tol_fac = (g.elem == 2 ? FTLM_BREAKDOWN_C_16BIT : FTLM_BREAKDOWN_C) * g.u;

    TS *pv = (TS *)g.d_v, *pvp = (TS *)g.d_vp, *pw = (TS *)g.d_w;

    for (int j = 0; j < M_lz; j++) {
        /* W = H V */
        launch_spmv<TS, TC>(pw, pv, B);

        /* alpha = <v, w> / sigma^2 */
        ftlm_dot_partial<TS, TC><<<rblocks, FTLM_REDUCE_BS>>>(
            (TC *)g.d_partial, pv, pw, n, B);
        ftlm_reduce_partial<TC><<<1, B>>>((TC *)g.d_alpha, (const TC *)g.d_partial,
                                          rblocks, B);
        to_host<TC>(h_alpha, g.d_alpha, B);
        for (int b = 0; b < B; b++) {
            h_alpha[b] = h_alpha[b] / sigma2;
            if (active[b]) h_AL[j + (size_t)b * M_lz] = (double)h_alpha[b];
        }
        if (sigma2 != (TC)1) to_dev<TC>(g.d_alpha, h_alpha, B);

        /* w -= alpha v + beta_prev vp,  ||w||^2 */
        if (j > 0) to_dev<TC>(g.d_beta_prev, h_beta_prev, B);
        ftlm_ortho_norm_partial<TS, TC><<<rblocks, FTLM_REDUCE_BS>>>(
            pw, pv, pvp, (const TC *)g.d_alpha, (const TC *)g.d_beta_prev,
            (TC *)g.d_partial, n, B, (j > 0) ? 1 : 0);
        ftlm_reduce_partial<TC><<<1, B>>>((TC *)g.d_beta, (const TC *)g.d_partial,
                                          rblocks, B);
        to_host<TC>(h_beta_sq, g.d_beta, B);

        int n_active = 0;
        for (int b = 0; b < B; b++) {
            h_beta[b] = ftlm_sqrt(h_beta_sq[b]) / sigma;
            if (!active[b]) { h_scale[b] = (TC)0; continue; }
            h_BE[j + (size_t)b * M_lz] = (double)h_beta[b];
            double tj = fabs((double)h_alpha[b]) + (double)h_beta[b]
                      + (double)h_beta_prev[b];
            if (tj > normT[b]) normT[b] = tj;
            if ((double)h_beta[b] <= tol_fac * normT[b]) {
                /* invariant subspace reached: freeze chain */
                active[b] = 0;
                nsteps[b] = j + 1;
                h_scale[b] = (TC)0;
            } else {
                n_active++;
                h_scale[b] = (TC)1 / h_beta[b];
            }
        }
        if (n_active == 0 || j == M_lz - 1) break;

        /* v_next = w / beta (frozen chains are set to zero) */
        to_dev<TC>(g.d_beta, h_scale, B);
        ftlm_scale<TS, TC><<<blocks, FTLM_SPMV_BS>>>(pw, (const TC *)g.d_beta, n, B);

        /* pointer swap: vp <- v, v <- w */
        TS *tmp = pvp; pvp = pv; pv = pw; pw = tmp;
        for (int b = 0; b < B; b++)
            h_beta_prev[b] = active[b] ? h_beta[b] : (TC)0;
    }
    cuda_check(cudaDeviceSynchronize(), "block_lanczos");
    g.d_v = pv; g.d_vp = pvp; g.d_w = pw;

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
    mxFree(h_AL);
    mxFree(h_BE);
}

/* ================================================================
 * MEX gateway
 * ================================================================ */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char mode[32];
    if (nrhs < 1 || mxGetString(prhs[0], mode, sizeof(mode)) != 0)
        mexErrMsgIdAndTxt("ftlm_gpu:mode", "First argument must be a mode string.");
    mxInitGPU();

    if (strcmp(mode, "init") == 0) {
        if (nrhs < 2 || !mxIsStruct(prhs[1]))
            mexErrMsgIdAndTxt("ftlm_gpu:init", "Usage: ftlm_gpu_mex('init', cfg)");
        const mxArray *cfg = prhs[1];
        if (g.init) cleanup_all();

        char buf[32];
        string_field(cfg, "lookup", buf, sizeof(buf));
        if      (strcmp(buf, "clt") == 0) g.lookup = FTLM_LOOKUP_CLT;
        else if (strcmp(buf, "cr")  == 0) g.lookup = FTLM_LOOKUP_CR;
        else mexErrMsgIdAndTxt("ftlm_gpu:cfg", "cfg.lookup must be 'clt' or 'cr'.");

        string_field(cfg, "precision", buf, sizeof(buf));
        if      (strcmp(buf, "double")   == 0) { g.prec = PREC_DOUBLE; g.elem = 8; g.u = ldexp(1.0, -53); }
        else if (strcmp(buf, "single")   == 0) { g.prec = PREC_SINGLE; g.elem = 4; g.u = ldexp(1.0, -24); }
        else if (strcmp(buf, "half")     == 0) { g.prec = PREC_HALF;   g.elem = 2; g.u = ldexp(1.0, -11); }
        else if (strcmp(buf, "bfloat16") == 0) { g.prec = PREC_BF16;   g.elem = 2; g.u = ldexp(1.0, -8); }
        else mexErrMsgIdAndTxt("ftlm_gpu:cfg",
                 "cfg.precision must be 'double', 'single', 'half' or 'bfloat16'.");

        double dim_d = scalar_field(cfg, "dim");
        if (dim_d < 1 || dim_d > 2147483647.0)
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "dim = %g outside [1, 2^31-1].", dim_d);
        g.dim   = (int)dim_d;
        g.B_max = (int)scalar_field(cfg, "B");
        if (g.B_max < 1 || g.B_max > FTLM_MAX_B)
            mexErrMsgIdAndTxt("ftlm_gpu:cfg", "B = %d outside [1, %d].", g.B_max, FTLM_MAX_B);

        /* FP16 storage needs O(1) vector entries: store vectors with norm
         * sqrt(dim) instead of 1 (applied for both 16-bit formats). */
        g.sigma = (g.prec == PREC_HALF || g.prec == PREC_BF16) ? sqrt((double)g.dim) : 1.0;

        g.dcum_size = 0;
        if (g.lookup == FTLM_LOOKUP_CLT) {
            int nb = 0, nm = 0, ns = 0;
            g.d_block_base = (int *)upload_int_array(field(cfg, "block_base", true), "block_base", &nb);
            g.d_block_mask = (unsigned int *)upload_int_array(field(cfg, "block_mask", true), "block_mask", &nm);
            g.d_basis      = (int *)upload_int_array(field(cfg, "basis", true), "basis", &ns);
            if (nb != nm || ns != g.dim)
                mexErrMsgIdAndTxt("ftlm_gpu:cfg", "inconsistent CLT/basis sizes.");
        } else {
            g.d_dcum = (int *)upload_int_array(field(cfg, "dcum", true), "dcum", &g.dcum_size);
            if ((size_t)g.dcum_size * sizeof(int) > 48 * 1024)
                mexErrMsgIdAndTxt("ftlm_gpu:cfg",
                    "D_c table (%d entries) exceeds 48 kB of shared memory.", g.dcum_size);
        }
        setup_constants(cfg);

        size_t vec_bytes = (size_t)g.dim * g.B_max * g.elem;
        size_t tmp_bytes = (size_t)g.dim * 8;   /* one staged column (import/export) */
        g.d_v  = dev_alloc(vec_bytes, "vector v");
        g.d_vp = dev_alloc(vec_bytes, "vector vp");
        g.d_w  = dev_alloc(vec_bytes, "vector w");
        g.d_tmp = dev_alloc(tmp_bytes, "staging buffer");
        size_t tc = (g.prec == PREC_DOUBLE) ? 8 : 4;
        int rblocks = (g.dim + FTLM_REDUCE_BS - 1) / FTLM_REDUCE_BS;
        g.d_partial   = dev_alloc((size_t)rblocks * g.B_max * tc, "partial");
        g.d_alpha     = dev_alloc(FTLM_MAX_B * tc, "alpha");
        g.d_beta      = dev_alloc(FTLM_MAX_B * tc, "beta");
        g.d_beta_prev = dev_alloc(FTLM_MAX_B * tc, "beta_prev");

        g.init = true;
        if (!mexIsLocked()) mexLock();
        mexAtExit(cleanup_all);
    }
    else if (strcmp(mode, "block_lanczos") == 0 || strcmp(mode, "spmv") == 0) {
        if (!g.init) mexErrMsgIdAndTxt("ftlm_gpu:run", "Call 'init' first.");
        bool lz = (strcmp(mode, "block_lanczos") == 0);
        if (nrhs < (lz ? 3 : 2))
            mexErrMsgIdAndTxt("ftlm_gpu:run", lz
                ? "Usage: [AL, BE, nsteps] = ftlm_gpu_mex('block_lanczos', V0, M_lz)"
                : "Usage: W = ftlm_gpu_mex('spmv', V)");
        int M_lz = lz ? (int)mxGetScalar(prhs[2]) : 0;
        switch (g.prec) {
        case PREC_DOUBLE:
            if (lz) do_block_lanczos<double, double>(nlhs, plhs, prhs[1], M_lz);
            else    do_spmv<double, double>(nlhs, plhs, prhs[1]);
            break;
        case PREC_SINGLE:
            if (lz) do_block_lanczos<float, float>(nlhs, plhs, prhs[1], M_lz);
            else    do_spmv<float, float>(nlhs, plhs, prhs[1]);
            break;
        case PREC_HALF:
            if (lz) do_block_lanczos<ftlm_fp16, float>(nlhs, plhs, prhs[1], M_lz);
            else    do_spmv<ftlm_fp16, float>(nlhs, plhs, prhs[1]);
            break;
        case PREC_BF16:
            if (lz) do_block_lanczos<ftlm_bf16, float>(nlhs, plhs, prhs[1], M_lz);
            else    do_spmv<ftlm_bf16, float>(nlhs, plhs, prhs[1]);
            break;
        }
    }
    else if (strcmp(mode, "info") == 0) {
        const char *names[] = {"init", "precision", "lookup", "dim", "B_max",
                               "sigma", "unit_roundoff", "sizeof_const"};
        plhs[0] = mxCreateStructMatrix(1, 1, 8, names);
        const char *pn[] = {"double", "single", "half", "bfloat16"};
        mxSetField(plhs[0], 0, "init", mxCreateLogicalScalar(g.init));
        mxSetField(plhs[0], 0, "precision", mxCreateString(g.init ? pn[g.prec] : ""));
        mxSetField(plhs[0], 0, "lookup",
                   mxCreateString(g.init ? (g.lookup == FTLM_LOOKUP_CLT ? "clt" : "cr") : ""));
        mxSetField(plhs[0], 0, "dim", mxCreateDoubleScalar(g.init ? g.dim : 0));
        mxSetField(plhs[0], 0, "B_max", mxCreateDoubleScalar(g.init ? g.B_max : 0));
        mxSetField(plhs[0], 0, "sigma", mxCreateDoubleScalar(g.init ? g.sigma : 0));
        mxSetField(plhs[0], 0, "unit_roundoff", mxCreateDoubleScalar(g.init ? g.u : 0));
        mxSetField(plhs[0], 0, "sizeof_const", mxCreateDoubleScalar((double)sizeof(FtlmConst)));
    }
    else if (strcmp(mode, "cleanup") == 0) {
        cleanup_all();
        if (mexIsLocked()) mexUnlock();
    }
    else {
        mexErrMsgIdAndTxt("ftlm_gpu:mode",
            "Unknown mode '%s'. Use 'init', 'block_lanczos', 'spmv', 'info', 'cleanup'.", mode);
    }
}
