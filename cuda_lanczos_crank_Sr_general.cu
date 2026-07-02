/*
 * cuda_lanczos_crank_Sr_general.cu
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
 * Heisenberg BLOCK Lanczos for ARBITRARY s >= 1/2
 * (s = 1/2, 1, 3/2, 2) with COMBINATORIAL RANKING after
 * Schnack/Hage/Schmidt (2007), GPU adapted.
 *
 * Generalization of cuda_lanczos_crank_Sr.cu (s = 1/2 only):
 * instead of a bit encoding of the spins a DIGIT encoding is used.
 * Each site carries d in {0, 1, ..., 2s}, where d = s - m_z.
 * The state is stored as a packed uint64,
 * BITS_PER_DIGIT = ceil(log2(2s+1)):
 *
 *   s = 1/2  -> 1 bit / site,  N <= 64
 *   s = 1    -> 2 bits / site, N <= 32
 *   s = 3/2  -> 2 bits / site, N <= 32
 *   s = 2    -> 3 bits / site, N <= 21
 *
 * RANKING (Schnack recursion):
 *     D(m, A) = sum_{k=0}^{2s} D(m-1, A-k),   D(0, 0) = 1
 *   Index of x = (d_{N-1}, ..., d_0), A_total = sum d_p:
 *     rank(x) = sum_{pos=N-1..0} D_cum(pos, A_k, d_pos)
 *   with D_cum(pos, A, d) = sum_{n=0}^{d-1} D(pos, A - n)
 *   and A_k = A_total - sum_{p>pos} d_p.
 *
 * GPU OPTIMIZATIONS (as crank_Sr s=1/2, but even more divergence-free):
 *   1. No CLT, no basis array -> VRAM savings.
 *   2. D_cum table in SHARED MEMORY (constant memory serializes
 *      divergent accesses up to 32x).
 *   3. Ranking loop has UNIFORM length N (no __ffs-based
 *      iteration over set bits required -> even more uniform
 *      than the s=1/2 case).
 *   4. Unranking: linear scan over (2s+1) <= 5 digit values,
 *      uniform, branchless via ternary select.
 *   5. Block SpMV with up to MAX_B Krylov vectors per thread.
 *   6. Clebsch-Gordan factors plus/minus in SHARED MEMORY LUTs.
 *   7. Per bond two contributions (S+_i S-_j and S-_i S+_j) separately,
 *      each with the correct m-dependent matrix elements.
 *
 * INTERFACE (MATLAB):
 *   cuda_lanczos_crank_Sr_general('init', N, two_s, A_total,
 *                                  bonds_flat, J, dim, B)
 *     N         : number of sites (int32, <= 32)
 *     two_s     : 2s (int32, 1..4 supported)
 *     A_total   : sum of all digits in the sector = N*s - Sz
 *                 (determines the Sz sector)
 *     bonds_flat: bond list [i0,j0, i1,j1, ...] 0-based (int32)
 *     J         : coupling constant (double); H = +J sum s_i . s_j
 *     dim       : sector dimension D(N, A_total) (double)
 *     B         : batch size for block Lanczos (double)
 *
 *   [AL, BE] = cuda_lanczos_crank_Sr_general('block_lanczos', V0, M_lz)
 *   cuda_lanczos_crank_Sr_general('cleanup')
 *
 * Compile:
 *   mexcuda cuda_lanczos_crank_Sr_general.cu
 * ================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>

#define MAX_SITES    32
#define MAX_BONDS    64
#define MAX_B        16
#define MAX_TWO_S     4    /* s up to and including 2 */
#define MAX_A_TOTAL  48    /* allows e.g. N=24, s=1; N=16, s=3/2; N=12, s=2 */

/* ================================================================
 * Constant memory
 * ================================================================ */
__constant__ int   c_bonds[2 * MAX_BONDS];
__constant__ int   c_N;
__constant__ int   c_nbonds;
__constant__ int   c_two_s;           /* 2s */
__constant__ int   c_A_total;
__constant__ int   c_bits_per_digit;
__constant__ unsigned long long c_digit_mask;
__constant__ float c_J;
__constant__ float c_half_J;

/* D(pos, A): Schnack recursion.
 * Access: c_D[pos * D_STRIDE + A] (used on host, not in the kernel). */
#define D_STRIDE      (MAX_A_TOTAL + 1)
#define D_TABLE_SIZE  ((MAX_SITES + 1) * D_STRIDE)

/* D_cum(pos, A, d) = sum_{n=0}^{d-1} D(pos, A - n),  d = 0..2s.
 * Access: c_Dcum[pos*DCUM_ASTRIDE + A*DCUM_DSTRIDE + d]
 * Size (worst case): 33 * 49 * 5 ints = ~32 kB. Fits into shared memory. */
#define DCUM_DSTRIDE    (MAX_TWO_S + 1)
#define DCUM_ASTRIDE    (D_STRIDE * DCUM_DSTRIDE)
#define DCUM_TABLE_SIZE ((MAX_SITES + 1) * DCUM_ASTRIDE)
__constant__ int c_Dcum[DCUM_TABLE_SIZE];

/* Clebsch-Gordan factors:
 *   plus[d]  = sqrt(d * (2s - d + 1))       for S+|d> -> |d-1>
 *   minus[d] = sqrt((2s - d) * (d + 1))     for S-|d> -> |d+1>
 *   mval[d]  = s - d                         (z-component of the digit)
 */
__constant__ float c_plus[MAX_TWO_S + 1];
__constant__ float c_minus[MAX_TWO_S + 1];
__constant__ float c_mval[MAX_TWO_S + 1];

/* ================================================================
 * Persistent state
 * ================================================================ */
static float  *s_d_v          = NULL;
static float  *s_d_vp         = NULL;
static float  *s_d_w          = NULL;
static float  *s_d_partial    = NULL;
static float  *s_d_alpha      = NULL;
static float  *s_d_beta       = NULL;
static float  *s_d_beta_prev  = NULL;
static float  *s_d_tmp_col    = NULL;
static int     s_dim          = 0;
static int     s_B_batch      = 0;
static int     s_n_reduce_blocks = 0;
static bool    s_init         = false;

static void cleanup_all(void) {
    if (s_d_v)         cudaFree(s_d_v);
    if (s_d_vp)        cudaFree(s_d_vp);
    if (s_d_w)         cudaFree(s_d_w);
    if (s_d_partial)   cudaFree(s_d_partial);
    if (s_d_alpha)     cudaFree(s_d_alpha);
    if (s_d_beta)      cudaFree(s_d_beta);
    if (s_d_beta_prev) cudaFree(s_d_beta_prev);
    if (s_d_tmp_col)   cudaFree(s_d_tmp_col);
    s_d_v = NULL; s_d_vp = NULL; s_d_w = NULL;
    s_d_partial = NULL; s_d_alpha = NULL;
    s_d_beta = NULL; s_d_beta_prev = NULL;
    s_d_tmp_col = NULL;
    s_init = false;
}

/* ================================================================
 * Digit helpers (branchless, purely arithmetic)
 * ================================================================ */
__device__ __forceinline__ int get_digit(
    unsigned long long state, int pos, int bits, unsigned long long mask)
{
    return (int)((state >> (pos * bits)) & mask);
}

__device__ __forceinline__ unsigned long long set_digit(
    unsigned long long state, int pos, int bits, unsigned long long mask, int val)
{
    state &= ~(mask << (pos * bits));
    state |= ((unsigned long long)val) << (pos * bits);
    return state;
}

/* ================================================================
 * Ranking (Schnack, GPU adapted)
 *
 * The loop always runs exactly N times -- uniform across the warp,
 * no data dependence of the iteration count.  D_cum is in shared
 * memory, divergent indices are served in parallel.
 * ================================================================ */
__device__ __forceinline__ int crank_rank_gen(
    unsigned long long state,
    const int *s_Dcum,
    int N, int bits, unsigned long long mask, int A_total)
{
    int rank = 0;
    int Ak   = A_total;
    for (int pos = N - 1; pos >= 0; pos--) {
        int d = get_digit(state, pos, bits, mask);
        rank += s_Dcum[pos * DCUM_ASTRIDE + Ak * DCUM_DSTRIDE + d];
        Ak   -= d;
    }
    return rank;
}

/* Unranking: per position a linear scan over d = 0..2s.
 * Branchless via ternary: loop count uniform = 2s,
 * no warp divergence. */
__device__ __forceinline__ unsigned long long crank_unrank_gen(
    int rank,
    const int *s_Dcum,
    int N, int bits, int two_s, int A_total)
{
    unsigned long long state = 0ULL;
    int rem = rank;
    int Ak  = A_total;

    for (int pos = N - 1; pos >= 0; pos--) {
        int base = pos * DCUM_ASTRIDE + Ak * DCUM_DSTRIDE;

        /* find the largest d in {0, ..., two_s} with D_cum[..][d] <= rem.
         * D_cum is monotonically non-decreasing in d -> ternary select stays correct. */
        int d_found = 0;
        for (int d = 1; d <= two_s; d++) {
            int val = s_Dcum[base + d];
            d_found = (val <= rem) ? d : d_found;
        }

        rem  -= s_Dcum[base + d_found];
        Ak   -= d_found;
        state |= ((unsigned long long)d_found) << (pos * bits);
    }
    return state;
}

/* ================================================================
 * Block SpMV (matrix-free, combinatorial ranking, arbitrary s)
 * ================================================================ */
__global__ void heisenberg_crank_general_block_spmv(
    float       * __restrict__ W,
    const float * __restrict__ V,
    int dim, int B)
{
    /* ---- Shared memory: D_cum + LUTs ---- */
    __shared__ int   s_Dcum[DCUM_TABLE_SIZE];
    __shared__ float s_plus [MAX_TWO_S + 1];
    __shared__ float s_minus[MAX_TWO_S + 1];
    __shared__ float s_mval [MAX_TWO_S + 1];

    for (int i = threadIdx.x; i < DCUM_TABLE_SIZE; i += blockDim.x)
        s_Dcum[i] = c_Dcum[i];
    if ((int)threadIdx.x <= MAX_TWO_S) {
        int d = (int)threadIdx.x;
        s_plus [d] = c_plus [d];
        s_minus[d] = c_minus[d];
        s_mval [d] = c_mval [d];
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;

    /* Fetch parameters from constant memory once */
    const int  N       = c_N;
    const int  two_s   = c_two_s;
    const int  A_total = c_A_total;
    const int  bits    = c_bits_per_digit;
    const unsigned long long mask = c_digit_mask;
    const float J      = c_J;
    const float halfJ  = c_half_J;
    const int  nb      = c_nbonds;

    /* Reconstruct state from rank t */
    unsigned long long state =
        crank_unrank_gen(t, s_Dcum, N, bits, two_s, A_total);

    /* FP32 accumulators for up to B Krylov vectors */
    float result[MAX_B];
    for (int b = 0; b < B; b++) result[b] = 0.0f;

    /* -----------------------------------------------------------------
     * Diagonal:  H_diag = J * sum_bonds  m_i * m_j
     * ----------------------------------------------------------------- */
    float diag_coeff = 0.0f;
    for (int bd = 0; bd < nb; bd++) {
        int si = c_bonds[2*bd];
        int sj = c_bonds[2*bd + 1];
        int di = get_digit(state, si, bits, mask);
        int dj = get_digit(state, sj, bits, mask);
        diag_coeff += s_mval[di] * s_mval[dj];
    }
    diag_coeff *= J;

    int t_base = t * B;
    for (int b = 0; b < B; b++)
        result[b] = diag_coeff * V[t_base + b];

    /* -----------------------------------------------------------------
     * Off-diagonal:  (J/2) * sum_bonds ( S+_i S-_j  +  S-_i S+_j )
     *
     * In row alpha = t one needs columns beta that connect to alpha via
     *   S+_i S-_j : d_i^beta = d_i^alpha + 1,  d_j^beta = d_j^alpha - 1
     *   S-_i S+_j : d_i^beta = d_i^alpha - 1,  d_j^beta = d_j^alpha + 1
     * Matrix element:
     *   H_{alpha,beta} = (J/2) * plus(d_i^beta) * minus(d_j^beta)        (S+ S- case)
     *                  = (J/2) * minus(d_i^beta) * plus(d_j^beta)        (S- S+ case)
     * ----------------------------------------------------------------- */
    for (int bd = 0; bd < nb; bd++) {
        int si = c_bonds[2*bd];
        int sj = c_bonds[2*bd + 1];
        int di = get_digit(state, si, bits, mask);
        int dj = get_digit(state, sj, bits, mask);

        /* --- S+_i S-_j contribution (beta has (di+1, dj-1)) --- */
        if (di < two_s && dj >= 1) {
            unsigned long long sb = state;
            sb = set_digit(sb, si, bits, mask, di + 1);
            sb = set_digit(sb, sj, bits, mask, dj - 1);
            int rb = crank_rank_gen(sb, s_Dcum, N, bits, mask, A_total);
            float coeff = halfJ * s_plus[di + 1] * s_minus[dj - 1];
            int rb_base = rb * B;
            for (int b = 0; b < B; b++)
                result[b] += coeff * V[rb_base + b];
        }

        /* --- S-_i S+_j contribution (beta has (di-1, dj+1)) --- */
        if (di >= 1 && dj < two_s) {
            unsigned long long sb = state;
            sb = set_digit(sb, si, bits, mask, di - 1);
            sb = set_digit(sb, sj, bits, mask, dj + 1);
            int rb = crank_rank_gen(sb, s_Dcum, N, bits, mask, A_total);
            float coeff = halfJ * s_minus[di - 1] * s_plus[dj + 1];
            int rb_base = rb * B;
            for (int b = 0; b < B; b++)
                result[b] += coeff * V[rb_base + b];
        }
    }

    for (int b = 0; b < B; b++)
        W[t_base + b] = result[b];
}

/* ================================================================
 * Helper kernels (identical to the s=1/2 original)
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

#define FUSED_BS 256

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
 * Host helpers: D and D_cum tables
 * ================================================================ */
static int bits_for_two_s(int two_s)
{
    if (two_s <= 1) return 1;   /* s = 1/2: 1 bit */
    if (two_s <= 3) return 2;   /* s = 1 or 3/2: 2 bits */
    if (two_s <= 7) return 3;   /* s = 2 or 5/2: 3 bits */
    return 4;
}

/* ================================================================
 * MEX Gateway
 * ================================================================ */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    mxInitGPU();
    char mode[32];
    mxGetString(prhs[0], mode, sizeof(mode));

    /* ============================================================
     * INIT
     *   prhs[1] = N         (int32/double)
     *   prhs[2] = two_s     (int32/double; 2s, e.g. 1 for s=1/2)
     *   prhs[3] = A_total   (int32/double; sum of the digits)
     *   prhs[4] = bonds_flat(int32, [i0,j0, i1,j1,...])
     *   prhs[5] = J         (double)
     *   prhs[6] = dim       (double)
     *   prhs[7] = B_batch   (double)
     * ============================================================ */
    if (strcmp(mode, "init") == 0)
    {
        if (s_init) cleanup_all();

        int N       = (int)mxGetScalar(prhs[1]);
        int two_s   = (int)mxGetScalar(prhs[2]);
        int A_total = (int)mxGetScalar(prhs[3]);

        int *h_bonds  = (int *)mxGetData(prhs[4]);
        int n_entries = (int)mxGetNumberOfElements(prhs[4]);
        int nb        = n_entries / 2;

        float J       = (float)mxGetScalar(prhs[5]);
        s_dim         = (int)mxGetScalar(prhs[6]);
        s_B_batch     = (int)mxGetScalar(prhs[7]);

        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("crank_Sr_gen:B",
                "B_batch=%d > MAX_B=%d", s_B_batch, MAX_B);
        }
        if (two_s < 1 || two_s > MAX_TWO_S) {
            mexErrMsgIdAndTxt("crank_Sr_gen:two_s",
                "two_s=%d outside [1, %d]", two_s, MAX_TWO_S);
        }
        if (N < 1 || N > MAX_SITES) {
            mexErrMsgIdAndTxt("crank_Sr_gen:N",
                "N=%d outside [1, %d]", N, MAX_SITES);
        }
        if (A_total < 0 || A_total > MAX_A_TOTAL) {
            mexErrMsgIdAndTxt("crank_Sr_gen:A_total",
                "A_total=%d outside [0, %d]", A_total, MAX_A_TOTAL);
        }
        if (A_total > N * two_s) {
            mexErrMsgIdAndTxt("crank_Sr_gen:A_total",
                "A_total=%d > N*two_s=%d", A_total, N*two_s);
        }

        int bits = bits_for_two_s(two_s);
        if (N * bits > 64) {
            mexErrMsgIdAndTxt("crank_Sr_gen:state",
                "N*bits=%d > 64: state does not fit into uint64", N*bits);
        }
        unsigned long long mask = (1ULL << bits) - 1ULL;

        /* ---- Fill constant memory: basic parameters ---- */
        cudaMemcpyToSymbol(c_bonds,  h_bonds, 2*nb*sizeof(int));
        cudaMemcpyToSymbol(c_N,      &N,       sizeof(int));
        cudaMemcpyToSymbol(c_nbonds, &nb,      sizeof(int));
        cudaMemcpyToSymbol(c_two_s,  &two_s,   sizeof(int));
        cudaMemcpyToSymbol(c_A_total,&A_total, sizeof(int));
        cudaMemcpyToSymbol(c_bits_per_digit, &bits, sizeof(int));
        cudaMemcpyToSymbol(c_digit_mask, &mask, sizeof(unsigned long long));
        cudaMemcpyToSymbol(c_J,      &J,       sizeof(float));
        float hJ = 0.5f * J;
        cudaMemcpyToSymbol(c_half_J, &hJ,      sizeof(float));

        /* ---- D table (on host) ---- */
        int *h_D = (int *)mxCalloc(D_TABLE_SIZE, sizeof(int));
        h_D[0 * D_STRIDE + 0] = 1;   /* D(0, 0) = 1 */
        for (int m = 1; m <= N; m++) {
            for (int A = 0; A <= A_total; A++) {
                long long sum = 0;
                for (int k = 0; k <= two_s; k++) {
                    if (A - k >= 0)
                        sum += h_D[(m - 1) * D_STRIDE + (A - k)];
                }
                if (sum > 2000000000LL) {
                    mxFree(h_D);
                    mexErrMsgIdAndTxt("crank_Sr_gen:overflow",
                        "D(%d,%d)=%lld overflows int", m, A, sum);
                }
                h_D[m * D_STRIDE + A] = (int)sum;
            }
        }

        int dim_check = h_D[N * D_STRIDE + A_total];
        if (dim_check != s_dim) {
            mxFree(h_D);
            mexErrMsgIdAndTxt("crank_Sr_gen:dim",
                "D(%d,%d)=%d != dim=%d", N, A_total, dim_check, s_dim);
        }

        /* ---- D_cum table ---- */
        int *h_Dcum = (int *)mxCalloc(DCUM_TABLE_SIZE, sizeof(int));
        for (int m = 0; m <= N; m++) {
            for (int A = 0; A <= A_total; A++) {
                int base = m * DCUM_ASTRIDE + A * DCUM_DSTRIDE;
                h_Dcum[base + 0] = 0;
                for (int d = 1; d <= two_s; d++) {
                    int prev = h_Dcum[base + d - 1];
                    int dval = 0;
                    int Aix  = A - (d - 1);
                    if (Aix >= 0)
                        dval = h_D[m * D_STRIDE + Aix];
                    h_Dcum[base + d] = prev + dval;
                }
            }
        }
        cudaMemcpyToSymbol(c_Dcum, h_Dcum,
                           DCUM_TABLE_SIZE * sizeof(int));
        mxFree(h_D);
        mxFree(h_Dcum);

        /* ---- Clebsch-Gordan LUTs + m values ---- */
        float h_plus [MAX_TWO_S + 1] = {0};
        float h_minus[MAX_TWO_S + 1] = {0};
        float h_mval [MAX_TWO_S + 1] = {0};
        for (int d = 0; d <= MAX_TWO_S; d++) {
            if (d <= two_s) {
                /* plus[d] = <d-1|S+|d>; only meaningful for d >= 1 */
                h_plus [d] = (d >= 1)
                    ? sqrtf((float)d * (float)(two_s - d + 1))
                    : 0.0f;
                /* minus[d] = <d+1|S-|d>; only meaningful for d <= two_s-1 */
                h_minus[d] = (d <= two_s - 1)
                    ? sqrtf((float)(two_s - d) * (float)(d + 1))
                    : 0.0f;
                /* mval[d] = s - d */
                h_mval [d] = 0.5f * (float)(two_s - 2 * d);
            }
        }
        cudaMemcpyToSymbol(c_plus,  h_plus,  sizeof(h_plus));
        cudaMemcpyToSymbol(c_minus, h_minus, sizeof(h_minus));
        cudaMemcpyToSymbol(c_mval,  h_mval,  sizeof(h_mval));

        /* ---- Vector VRAM ---- */
        size_t vec_bytes = (size_t)s_dim * s_B_batch * sizeof(float);
        cudaMalloc(&s_d_v,  vec_bytes);
        cudaMalloc(&s_d_vp, vec_bytes);
        cudaMalloc(&s_d_w,  vec_bytes);

        s_n_reduce_blocks = (s_dim + FUSED_BS - 1) / FUSED_BS;
        cudaMalloc(&s_d_partial,
                   (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,       s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev,  s_B_batch * sizeof(float));
        cudaMalloc(&s_d_tmp_col,    vec_bytes);

        s_init = true;
        if (!mexIsLocked()) mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * BLOCK_LANCZOS
     *   prhs[1] = V0  (gpuArray, n x B, float)
     *   prhs[2] = M_lz
     * ============================================================ */
    else if (strcmp(mode, "block_lanczos") == 0)
    {
        if (!s_init)
            mexErrMsgIdAndTxt("crank_Sr_gen:run", "Call 'init' first!");

        const mxGPUArray *g_V0 = mxGPUCreateFromMxArray(prhs[1]);
        const mwSize *dims_V0  = mxGPUGetDimensions(g_V0);
        int n = (int)dims_V0[0];
        int B = (mxGPUGetNumberOfDimensions(g_V0) > 1)
                ? (int)dims_V0[1] : 1;
        int M_lz = (int)mxGetScalar(prhs[2]);

        if (n != s_dim)
            mexErrMsgIdAndTxt("crank_Sr_gen:dim",
                "V0 has %d rows, expected %d", n, s_dim);
        if (B > s_B_batch)
            mexErrMsgIdAndTxt("crank_Sr_gen:B",
                "B=%d > B_batch=%d", B, s_B_batch);
        if (M_lz > n) M_lz = n;

        int spmv_blocks   = (n + 255) / 256;
        int reduce_blocks = (n + FUSED_BS - 1) / FUSED_BS;

        /* V0 -> interleaved */
        cudaMemcpy(s_d_tmp_col, mxGPUGetDataReadOnly(g_V0),
                   (size_t)n * B * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        mxGPUDestroyGPUArray(g_V0);

        transpose_col2interleaved<<<spmv_blocks, 256>>>(
            s_d_v, s_d_tmp_col, n, B);

        /* Normalization */
        {
            fused_dot_partial<<<reduce_blocks, FUSED_BS>>>(
                s_d_partial, s_d_v, s_d_v, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);
            float h_nrm2[MAX_B];
            cudaMemcpy(h_nrm2, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            float h_inv[MAX_B];
            for (int b = 0; b < B; b++)
                h_inv[b] = 1.0f / sqrtf(h_nrm2[b]);
            cudaMemcpy(s_d_alpha, h_inv, B * sizeof(float),
                       cudaMemcpyHostToDevice);
            scale_interleaved<<<spmv_blocks, 256>>>(
                s_d_v, s_d_alpha, n, B);
        }

        cudaMemset(s_d_vp, 0, (size_t)n * B * sizeof(float));

        double *h_AL = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
        double *h_BE = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));

        float h_alpha[MAX_B];
        float h_beta[MAX_B];
        float h_beta_prev[MAX_B];
        int nsteps = M_lz;
        memset(h_beta_prev, 0, sizeof(h_beta_prev));

        float *ptr_v  = s_d_v;
        float *ptr_vp = s_d_vp;
        float *ptr_w  = s_d_w;

        for (int j = 0; j < M_lz; j++) {

            /* W = H * V */
            heisenberg_crank_general_block_spmv<<<spmv_blocks, 256>>>(
                ptr_w, ptr_v, n, B);

            /* alpha = dot(v, w) */
            fused_dot_partial<<<reduce_blocks, FUSED_BS>>>(
                s_d_partial, ptr_v, ptr_w, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);
            cudaMemcpy(h_alpha, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (int b = 0; b < B; b++)
                h_AL[j + (size_t)b * M_lz] = (double)h_alpha[b];

            /* Orthogonalization + norm */
            if (j > 0) {
                cudaMemcpy(s_d_beta_prev, h_beta_prev,
                           B * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            fused_ortho_norm_partial<<<reduce_blocks, FUSED_BS>>>(
                ptr_w, ptr_v, ptr_vp,
                s_d_alpha, s_d_beta_prev,
                s_d_partial,
                n, B, (j > 0) ? 1 : 0);

            reduce_partial<<<1, B>>>(s_d_beta, s_d_partial,
                                     reduce_blocks, B);

            float h_beta_sq[MAX_B];
            cudaMemcpy(h_beta_sq, s_d_beta, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            int all_converged = 1;
            for (int b = 0; b < B; b++) {
                h_beta[b] = sqrtf(h_beta_sq[b]);
                h_BE[j + (size_t)b * M_lz] = (double)h_beta[b];
                if (h_beta[b] >= 1e-6f) all_converged = 0;
            }
            if (all_converged) { nsteps = j + 1; break; }

            if (j < M_lz - 1) {
                float h_inv_beta[MAX_B];
                for (int b = 0; b < B; b++)
                    h_inv_beta[b] = 1.0f / h_beta[b];
                cudaMemcpy(s_d_beta, h_inv_beta,
                           B * sizeof(float),
                           cudaMemcpyHostToDevice);
                scale_interleaved<<<spmv_blocks, 256>>>(
                    ptr_w, s_d_beta, n, B);

                float *tmp = ptr_vp;
                ptr_vp = ptr_v;
                ptr_v  = ptr_w;
                ptr_w  = tmp;

                memcpy(h_beta_prev, h_beta, B * sizeof(float));
            }
        }

        cudaDeviceSynchronize();
        s_d_v  = ptr_v;
        s_d_vp = ptr_vp;
        s_d_w  = ptr_w;

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
        if (mexIsLocked()) mexUnlock();
    }
    else {
        mexErrMsgIdAndTxt("crank_Sr_gen:mode",
            "Unknown mode: '%s'. Use 'init', 'block_lanczos', 'cleanup'.",
            mode);
    }
}
