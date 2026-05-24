/*
 * cuda_lanczos_clut_block.cu
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
 * Matrix-free Heisenberg BLOCK Lanczos on GPU (FP32)
 * with COMPRESSED lookup table (block bitmap).
 *
 * BLOCK LANCZOS with INTERLEAVED MEMORY LAYOUT:
 *   Processes B independent Krylov chains simultaneously.
 *   For each output basis state the state arithmetic
 *   (digit decomposition, bond iteration, CLT lookups) is
 *   performed ONCE and amortized over B input vectors.
 *
 * INTERLEAVED LAYOUT (key optimization):
 *   Vectors are stored as V[idx * B + b] instead of
 *   V[idx + b * dim].  The B values for a given index then lie
 *   in a contiguous 4*B-byte region:
 *
 *     Column-major:    V[idx], V[idx+dim], ..., V[idx+(B-1)*dim]
 *       => B separate cache lines per lookup (stride = dim)
 *
 *     Interleaved:     V[idx*B], V[idx*B+1], ..., V[idx*B+(B-1)]
 *       => 1 cache line per lookup (B<=8: 32 bytes)
 *
 *   For 2*N_b = 60 off-diagonal lookups per thread and B = 8:
 *     Column-major:  60 * 8 = 480 cache-line loads
 *     Interleaved:   60 * 1 =  60 cache-line loads
 *   => 8x less random-access traffic for the dominant
 *      bandwidth-limited part of the kernel.
 *
 * POINTER SWAP (instead of cudaMemcpy D2D):
 *   The Lanczos vector exchange (vp <- v, v <- w/beta) is
 *   realized by a simple pointer swap instead of two memory
 *   copies of dim*B*4 bytes each.
 *   For s=2, B=8: saves 2*20M*8*4 = 1.28 GB per step,
 *   i.e. ~128 GB over 100 Lanczos steps.
 *
 * MATLAB interface:
 *   V0_gpu is passed as column-major [dim x B].
 *   The init section transposes it internally to interleaved.
 *   AL, BE are returned as column-major [M_lz x B].
 *
 * Modes:
 *   cuda_lanczos_clut_block('init', block_base_gpu, block_mask_gpu,
 *                            basis_gpu, bonds_flat, N, d, s, J,
 *                            dim, B_batch)
 *   [AL, BE] = cuda_lanczos_clut_block('block_lanczos',
 *                                       V0_gpu, M_lz)
 *   cuda_lanczos_clut_block('cleanup')
 *
 * Compile:
 *   mexcuda cuda_lanczos_clut_block.cu -lcublas
 * ================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string.h>

#define MAX_SITES 32
#define MAX_BONDS 64
#define BLOCK_BITS 5         /* log2(32) */
#define CLT_BS    32
#define MAX_B     16         /* maximum block size */

/* ================================================================
 * Constant memory: model parameters (identical for all threads)
 * ================================================================ */
__constant__ int    c_bonds[2 * MAX_BONDS];
__constant__ int    c_powers[MAX_SITES];
__constant__ int    c_N;
__constant__ int    c_d;
__constant__ int    c_nbonds;
__constant__ float  c_s;
__constant__ float  c_J;
__constant__ float  c_ss1;
__constant__ int    c_d_minus_1;

/* ================================================================
 * Persistent state
 * ================================================================ */
static cublasHandle_t s_blH          = NULL;
static int           *s_d_block_base = NULL;
static unsigned int  *s_d_block_mask = NULL;
static int           *s_d_basis      = NULL;
static float         *s_d_v          = NULL;   /* dim * B_batch, interleaved */
static float         *s_d_vp         = NULL;   /* dim * B_batch, interleaved */
static float         *s_d_w          = NULL;   /* dim * B_batch, interleaved */
static int            s_dim          = 0;
static int            s_B_batch      = 0;
static bool           s_init         = false;

void cleanup(void) {
    if (s_d_block_base) cudaFree(s_d_block_base);
    if (s_d_block_mask) cudaFree(s_d_block_mask);
    if (s_d_basis)      cudaFree(s_d_basis);
    if (s_d_v)          cudaFree(s_d_v);
    if (s_d_vp)         cudaFree(s_d_vp);
    if (s_d_w)          cudaFree(s_d_w);
    if (s_blH)          cublasDestroy(s_blH);
    s_d_block_base = NULL; s_d_block_mask = NULL;
    s_d_basis = NULL;
    s_d_v = NULL; s_d_vp = NULL; s_d_w = NULL;
    s_blH = NULL; s_init = false;
}

/* ================================================================
 * Device: compressed lookup (identical to cuda_lanczos_clut.cu)
 * ================================================================ */
__device__ __forceinline__ int clut_lookup(
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    int state_a)
{
    int blk = state_a >> BLOCK_BITS;
    int bit = state_a & (CLT_BS - 1);
    unsigned int mask = __ldg(&block_mask[blk]);

    if (!(mask & (1u << bit)))
        return -1;

    unsigned int lower = mask & ((1u << bit) - 1u);
    return __ldg(&block_base[blk]) + __popc(lower);
}

/* ================================================================
 * CUDA kernel:  W = H * V  (block SpMV, INTERLEAVED layout)
 *
 * Memory layout: V[idx * B + b], W[idx * B + b]
 *
 * For each CLT lookup, B values are loaded from a contiguous
 * memory region (1 cache line for B <= 8).
 * ================================================================ */
__global__ void heisenberg_clut_block_spmv(
    float              * __restrict__ W,
    const float        * __restrict__ V,
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    const int          * __restrict__ basis,
    int dim,
    int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;

    int state = __ldg(&basis[t]);

    /* --- Digit decomposition (ONCE for all B vectors) --- */
    int digits[MAX_SITES];
    {
        int tmp = state;
        for (int k = 0; k < c_N; k++) {
            digits[k] = tmp % c_d;
            tmp /= c_d;
        }
    }

    /* --- Accumulators for B vectors --- */
    float result[MAX_B];
    for (int b = 0; b < B; b++)
        result[b] = 0.0f;

    /* --- Diagonal part: J * sum_<ij> mi*mj --- */
    float diag = 0.0f;
    for (int bd = 0; bd < c_nbonds; bd++) {
        float mi = (float)digits[c_bonds[2*bd]]     - c_s;
        float mj = (float)digits[c_bonds[2*bd + 1]] - c_s;
        diag += mi * mj;
    }
    float diag_coeff = c_J * diag;

    /* Diagonal read: V[t * B + b] — B values are contiguous */
    int t_base = t * B;
    for (int b = 0; b < B; b++)
        result[b] = diag_coeff * V[t_base + b];

    /* --- Off-diagonal: spin-flip terms --- */
    float ss1 = c_ss1;
    for (int bd = 0; bd < c_nbonds; bd++) {
        int si = c_bonds[2*bd];
        int sj = c_bonds[2*bd + 1];
        int di = digits[si];
        int dj = digits[sj];
        float mi = (float)di - c_s;
        float mj = (float)dj - c_s;

        /* S+_i S-_j */
        if (di > 0 && dj < c_d_minus_1) {
            float mi_a = mi - 1.0f;
            float mj_a = mj + 1.0f;
            float coeff = 0.5f * c_J
                * sqrtf(ss1 - mi_a * (mi_a + 1.0f))
                * sqrtf(ss1 - mj_a * (mj_a - 1.0f));
            int state_a = state - c_powers[si] + c_powers[sj];
            int idx_a = clut_lookup(block_base, block_mask, state_a);
            if (idx_a >= 0) {
                /* INTERLEAVED: B values are contiguous */
                int a_base = idx_a * B;
                for (int b = 0; b < B; b++)
                    result[b] += coeff * V[a_base + b];
            }
        }

        /* S-_i S+_j */
        if (di < c_d_minus_1 && dj > 0) {
            float mi_a = mi + 1.0f;
            float mj_a = mj - 1.0f;
            float coeff = 0.5f * c_J
                * sqrtf(ss1 - mi_a * (mi_a - 1.0f))
                * sqrtf(ss1 - mj_a * (mj_a + 1.0f));
            int state_a = state + c_powers[si] - c_powers[sj];
            int idx_a = clut_lookup(block_base, block_mask, state_a);
            if (idx_a >= 0) {
                int a_base = idx_a * B;
                for (int b = 0; b < B; b++)
                    result[b] += coeff * V[a_base + b];
            }
        }
    }

    /* --- Write results (interleaved) --- */
    for (int b = 0; b < B; b++)
        W[t_base + b] = result[b];
}

/* ================================================================
 * Helper kernel: transpose column-major -> interleaved
 *
 * Src: [dim x B] column-major, i.e. Src[row + col * dim]
 * Dst: [dim x B] interleaved, i.e. Dst[row * B + col]
 * ================================================================ */
__global__ void transpose_col2interleaved(
    float       * __restrict__ dst,
    const float * __restrict__ src,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int b = 0; b < B; b++)
        dst[t * B + b] = src[t + b * dim];
}

/* ================================================================
 * Helper kernel: transpose interleaved -> column-major
 * (only for debugging/export, not in the hot path)
 * ================================================================ */
__global__ void transpose_interleaved2col(
    float       * __restrict__ dst,
    const float * __restrict__ src,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int b = 0; b < B; b++)
        dst[t + b * dim] = src[t * B + b];
}

/* ================================================================
 * Helper kernel: BLAS substitute for interleaved layout
 *
 * Fused Lanczos update for all B chains in one kernel:
 *   For each index t and each vector b:
 *     1. alpha[b] += v[t*B+b] * w[t*B+b]      (dot product)
 *     2. w[t*B+b] -= alpha_in[b] * v[t*B+b]   (orthogonalization)
 *     3. w[t*B+b] -= beta_prev[b] * vp[t*B+b] (if j > 0)
 *     4. nrm2[b]  += w[t*B+b]^2               (norm computation)
 *
 * Uses block reduction for the B dot products and norms.
 * ================================================================ */

/* Warp reduction */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

#define FUSED_BS 256

/*
 * Phase 1: compute alpha = dot(v, w) per vector b.
 * Partial sums per block -> partial_alpha[gridDim.x * B]
 */
__global__ void fused_dot_partial(
    float       * __restrict__ partial,
    const float * __restrict__ V,
    const float * __restrict__ W,
    int dim, int B)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        if (t < dim)
            sum = V[t * B + b] * W[t * B + b];

        /* Block reduction */
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial[blockIdx.x * B + b] = sdata[0];
    }
}

/*
 * Phase 2: reduce partial sums -> final values [B]
 * n_blocks = number of blocks from phase 1
 */
__global__ void reduce_partial(
    float       * __restrict__ result,
    const float * __restrict__ partial,
    int n_blocks, int B)
{
    int b = threadIdx.x;
    if (b >= B) return;
    float sum = 0.0f;
    for (int i = 0; i < n_blocks; i++)
        sum += partial[i * B + b];
    result[b] = sum;
}

/*
 * Fused orthogonalization + norm:
 *   w -= alpha[b] * v  (and optionally: w -= beta_prev[b] * vp)
 *   Simultaneously computes ||w||^2 per vector b.
 *
 * use_vp: 0 = no vp term (j == 0), 1 = vp term (j > 0)
 */
__global__ void fused_ortho_norm_partial(
    float       * __restrict__ W,
    const float * __restrict__ V,
    const float * __restrict__ Vp,
    const float * __restrict__ alpha,
    const float * __restrict__ beta_prev,
    float       * __restrict__ partial_nrm,
    int dim, int B, int use_vp)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float w_val = 0.0f;
        if (t < dim) {
            int idx = t * B + b;
            w_val = W[idx] - alpha[b] * V[idx];
            if (use_vp)
                w_val -= beta_prev[b] * Vp[idx];
            W[idx] = w_val;
        }

        /* Block reduction for ||w||^2 */
        sdata[tid] = w_val * w_val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial_nrm[blockIdx.x * B + b] = sdata[0];
    }
}

/*
 * Scaling: w[t*B+b] *= scale[b]  for all t, b
 *
 * Used both for the initial normalization (scale = 1/||v||)
 * and for the Lanczos normalization (scale = 1/beta).
 */
__global__ void scale_interleaved(
    float       * __restrict__ W,
    const float * __restrict__ scale,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    int base = t * B;
    for (int b = 0; b < B; b++)
        W[base + b] *= scale[b];
}

/* ================================================================
 * Shared init code: bonds, powers, constants
 * ================================================================ */
static void init_common(int *h_bonds, int nb, int N, int d,
                        float sv, float J)
{
    int h_pw[MAX_SITES];
    h_pw[0] = 1;
    for (int k = 1; k < N; k++) h_pw[k] = h_pw[k-1] * d;

    cudaMemcpyToSymbol(c_bonds,     h_bonds, 2*nb*sizeof(int));
    cudaMemcpyToSymbol(c_powers,    h_pw,    N*sizeof(int));
    cudaMemcpyToSymbol(c_N,         &N,  sizeof(int));
    cudaMemcpyToSymbol(c_d,         &d,  sizeof(int));
    cudaMemcpyToSymbol(c_nbonds,    &nb, sizeof(int));
    cudaMemcpyToSymbol(c_s,         &sv, sizeof(float));
    cudaMemcpyToSymbol(c_J,         &J,  sizeof(float));
    float ss1 = sv * (sv + 1.0f);
    cudaMemcpyToSymbol(c_ss1,       &ss1, sizeof(float));
    int dm1 = d - 1;
    cudaMemcpyToSymbol(c_d_minus_1, &dm1, sizeof(int));
}

/* ================================================================
 * MEX Gateway
 * ================================================================ */

/* Persistent GPU buffers for reduction kernels */
static float *s_d_partial  = NULL;   /* partial reductions */
static float *s_d_alpha    = NULL;   /* alpha[B] on GPU */
static float *s_d_beta     = NULL;   /* beta[B] on GPU */
static float *s_d_beta_prev = NULL;  /* beta_prev[B] on GPU */
static float *s_d_tmp_col  = NULL;   /* scratch buffer for transposition */
static int    s_n_reduce_blocks = 0;

static void cleanup_all(void) {
    if (s_d_block_base) cudaFree(s_d_block_base);
    if (s_d_block_mask) cudaFree(s_d_block_mask);
    if (s_d_basis)      cudaFree(s_d_basis);
    if (s_d_v)          cudaFree(s_d_v);
    if (s_d_vp)         cudaFree(s_d_vp);
    if (s_d_w)          cudaFree(s_d_w);
    if (s_d_partial)    cudaFree(s_d_partial);
    if (s_d_alpha)      cudaFree(s_d_alpha);
    if (s_d_beta)       cudaFree(s_d_beta);
    if (s_d_beta_prev)  cudaFree(s_d_beta_prev);
    if (s_d_tmp_col)    cudaFree(s_d_tmp_col);
    if (s_blH)          cublasDestroy(s_blH);
    s_d_block_base = NULL; s_d_block_mask = NULL;
    s_d_basis = NULL;
    s_d_v = NULL; s_d_vp = NULL; s_d_w = NULL;
    s_d_partial = NULL; s_d_alpha = NULL;
    s_d_beta = NULL; s_d_beta_prev = NULL;
    s_d_tmp_col = NULL;
    s_blH = NULL; s_init = false;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    mxInitGPU();
    char mode[32];
    mxGetString(prhs[0], mode, sizeof(mode));

    /* ============================================================
     * INIT
     * ============================================================ */
    if (strcmp(mode, "init") == 0)
    {
        if (s_init) cleanup_all();

        const mxGPUArray *g_bb = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_bm = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_bs = mxGPUCreateFromMxArray(prhs[3]);

        int *h_bonds  = (int *)mxGetData(prhs[4]);
        int n_entries = (int)mxGetNumberOfElements(prhs[4]);
        int nb        = n_entries / 2;
        int N         = (int)mxGetScalar(prhs[5]);
        int d         = (int)mxGetScalar(prhs[6]);
        float sv      = (float)mxGetScalar(prhs[7]);
        float J       = (float)mxGetScalar(prhs[8]);
        s_dim         = (int)mxGetScalar(prhs[9]);
        s_B_batch     = (int)mxGetScalar(prhs[10]);

        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        init_common(h_bonds, nb, N, d, sv, J);

        int bb_n = (int)mxGPUGetNumberOfElements(g_bb);
        int bm_n = (int)mxGPUGetNumberOfElements(g_bm);
        int bs_n = (int)mxGPUGetNumberOfElements(g_bs);

        cudaMalloc(&s_d_block_base, bb_n * sizeof(int));
        cudaMalloc(&s_d_block_mask, bm_n * sizeof(unsigned int));
        cudaMalloc(&s_d_basis,      bs_n * sizeof(int));

        cudaMemcpy(s_d_block_base, mxGPUGetDataReadOnly(g_bb),
                   bb_n * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_block_mask, mxGPUGetDataReadOnly(g_bm),
                   bm_n * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_basis, mxGPUGetDataReadOnly(g_bs),
                   bs_n * sizeof(int), cudaMemcpyDeviceToDevice);

        /* Vectors: dim * B_batch, INTERLEAVED layout */
        size_t vec_bytes = (size_t)s_dim * s_B_batch * sizeof(float);
        cudaMalloc(&s_d_v,  vec_bytes);
        cudaMalloc(&s_d_vp, vec_bytes);
        cudaMalloc(&s_d_w,  vec_bytes);

        /* Reduction buffers */
        s_n_reduce_blocks = (s_dim + FUSED_BS - 1) / FUSED_BS;
        cudaMalloc(&s_d_partial, (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,       s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev,  s_B_batch * sizeof(float));

        /* Scratch buffer for column-major -> interleaved transposition */
        cudaMalloc(&s_d_tmp_col, vec_bytes);

        cublasCreate(&s_blH);

        mxGPUDestroyGPUArray(g_bb);
        mxGPUDestroyGPUArray(g_bm);
        mxGPUDestroyGPUArray(g_bs);

        s_init = true;
        mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * BLOCK_LANCZOS: B independent Krylov chains
     *
     * Interleaved layout + pointer swap + fused BLAS kernels
     * ============================================================ */
    else if (strcmp(mode, "block_lanczos") == 0)
    {
        if (!s_init)
            mexErrMsgIdAndTxt("clut_block:run", "Call 'init' first!");

        const mxGPUArray *g_V0 = mxGPUCreateFromMxArray(prhs[1]);
        const mwSize *dims_V0  = mxGPUGetDimensions(g_V0);
        int n  = (int)dims_V0[0];
        int B  = (mxGPUGetNumberOfDimensions(g_V0) > 1) ? (int)dims_V0[1] : 1;
        int M_lz = (int)mxGetScalar(prhs[2]);

        if (n != s_dim)
            mexErrMsgIdAndTxt("clut_block:dim",
                "V0 has %d rows, expected %d!", n, s_dim);
        if (B > s_B_batch)
            mexErrMsgIdAndTxt("clut_block:B",
                "B = %d exceeds B_batch = %d!", B, s_B_batch);
        if (M_lz > n) M_lz = n;

        int spmv_blocks = (n + 255) / 256;
        int reduce_blocks = (n + FUSED_BS - 1) / FUSED_BS;

        /* V0 is column-major [dim x B].
         * Transpose to interleaved [dim * B]. */
        cudaMemcpy(s_d_tmp_col, mxGPUGetDataReadOnly(g_V0),
                   (size_t)n * B * sizeof(float), cudaMemcpyDeviceToDevice);
        mxGPUDestroyGPUArray(g_V0);

        transpose_col2interleaved<<<spmv_blocks, 256>>>(
            s_d_v, s_d_tmp_col, n, B);

        /* Normalize each of the B vectors (interleaved).
         * Use fused_dot_partial + reduce_partial for the norm. */
        {
            /* Compute ||v_b||^2 via dot(v, v) */
            fused_dot_partial<<<reduce_blocks, FUSED_BS>>>(
                s_d_partial, s_d_v, s_d_v, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);

            /* alpha[b] = ||v_b||^2, need 1/sqrt of it */
            float h_nrm2[MAX_B];
            cudaMemcpy(h_nrm2, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);

            float h_inv[MAX_B];
            for (int b = 0; b < B; b++)
                h_inv[b] = 1.0f / sqrtf(h_nrm2[b]);

            /* v[t*B+b] *= 1/||v_b|| */
            cudaMemcpy(s_d_alpha, h_inv, B * sizeof(float),
                       cudaMemcpyHostToDevice);
            scale_interleaved<<<spmv_blocks, 256>>>(
                s_d_v, s_d_alpha, n, B);
        }

        cudaMemset(s_d_vp, 0, (size_t)n * B * sizeof(float));

        /* Output arrays: AL[M_lz x B], BE[M_lz x B] */
        double *h_AL = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
        double *h_BE = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));

        /* Host buffers for alpha/beta per step */
        float h_alpha[MAX_B];
        float h_beta[MAX_B];
        float h_beta_prev[MAX_B];

        int nsteps = M_lz;
        memset(h_beta_prev, 0, sizeof(h_beta_prev));

        /* Pointers for swap */
        float *ptr_v  = s_d_v;
        float *ptr_vp = s_d_vp;
        float *ptr_w  = s_d_w;

        for (int j = 0; j < M_lz; j++) {

            /* === W = H * V (block SpMV with CLT, interleaved) === */
            heisenberg_clut_block_spmv<<<spmv_blocks, 256>>>(
                ptr_w, ptr_v, s_d_block_base, s_d_block_mask,
                s_d_basis, n, B);

            /* === alpha[b] = dot(v_b, w_b) === */
            fused_dot_partial<<<reduce_blocks, FUSED_BS>>>(
                s_d_partial, ptr_v, ptr_w, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);

            /* alpha -> host (for output) */
            cudaMemcpy(h_alpha, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (int b = 0; b < B; b++)
                h_AL[j + (size_t)b * M_lz] = (double)h_alpha[b];

            /* === w -= alpha*v - beta_prev*vp, compute ||w||^2 === */
            if (j > 0) {
                cudaMemcpy(s_d_beta_prev, h_beta_prev, B * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            fused_ortho_norm_partial<<<reduce_blocks, FUSED_BS>>>(
                ptr_w, ptr_v, ptr_vp,
                s_d_alpha, s_d_beta_prev,
                s_d_partial,
                n, B, (j > 0) ? 1 : 0);

            reduce_partial<<<1, B>>>(s_d_beta, s_d_partial,
                                     reduce_blocks, B);

            /* beta -> host */
            float h_beta_sq[MAX_B];
            cudaMemcpy(h_beta_sq, s_d_beta, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            int all_converged = 1;
            for (int b = 0; b < B; b++) {
                h_beta[b] = sqrtf(h_beta_sq[b]);
                h_BE[j + (size_t)b * M_lz] = (double)h_beta[b];
                if (h_beta[b] >= 1e-6f)
                    all_converged = 0;
            }

            if (all_converged) { nsteps = j + 1; break; }

            if (j < M_lz - 1) {
                /* === Normalization: w *= 1/beta === */
                float h_inv_beta[MAX_B];
                for (int b = 0; b < B; b++)
                    h_inv_beta[b] = 1.0f / h_beta[b];
                cudaMemcpy(s_d_beta, h_inv_beta, B * sizeof(float),
                           cudaMemcpyHostToDevice);
                scale_interleaved<<<spmv_blocks, 256>>>(
                    ptr_w, s_d_beta, n, B);

                /* === POINTER SWAP: vp <- v, v <- w === */
                /* Saves 2x cudaMemcpy D2D of dim*B*4 bytes each! */
                float *tmp = ptr_vp;
                ptr_vp = ptr_v;
                ptr_v  = ptr_w;
                ptr_w  = tmp;

                /* Remember beta_prev for the next step */
                memcpy(h_beta_prev, h_beta, B * sizeof(float));
            }
        }

        cudaDeviceSynchronize();

        /* Reset pointers (for the next call) */
        s_d_v  = ptr_v;
        s_d_vp = ptr_vp;
        s_d_w  = ptr_w;

        /* Return: AL [nsteps x B], BE [nsteps x B] */
        plhs[0] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        double *out_AL = mxGetPr(plhs[0]);
        double *out_BE = mxGetPr(plhs[1]);

        for (int b = 0; b < B; b++) {
            memcpy(out_AL + (size_t)b * nsteps,
                   h_AL  + (size_t)b * M_lz,
                   nsteps * sizeof(double));
            memcpy(out_BE + (size_t)b * nsteps,
                   h_BE  + (size_t)b * M_lz,
                   nsteps * sizeof(double));
        }
        mxFree(h_AL);
        mxFree(h_BE);
    }

    /* ============================================================
     * CLEANUP
     * ============================================================ */
    else if (strcmp(mode, "cleanup") == 0)
    {
        cleanup_all();
        mexUnlock();
    }
}
