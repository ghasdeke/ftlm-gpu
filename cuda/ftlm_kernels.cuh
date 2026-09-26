/*
 * ftlm_kernels.cuh
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
 * Device code of the matrix-free block Lanczos for isotropic spin
 * Hamiltonians
 *
 *     H = sum_c J_c  s_{i_c} . s_{j_c}
 *
 * with an arbitrary list of pairwise couplings (i_c, j_c, J_c) and
 * arbitrary local spins s_k (k = 0..N-1, mixed spins allowed).
 *
 * This header contains DEVICE CODE ONLY.  It is shared by
 *   - the MATLAB MEX gateway  ftlm_gpu_mex.cu      (compiled by nvcc)
 *   - the Python package      ftlm_gpu             (compiled by NVRTC
 *                                                   through CuPy)
 * and therefore includes no host or standard-library headers.
 *
 * Basis encoding (identical for both lookup strategies)
 * ----------------------------------------------------
 *   Local digit     a_k = m_k + s_k  in {0, ..., 2 s_k}
 *   CLT label       n   = sum_k a_k P_k,   P_0 = 1, P_{k+1} = P_k d_k,
 *                   d_k = 2 s_k + 1  (mixed radix; int32 labels)
 *   CR packed state x   = sum_k a_k 2^{shift_k}  (uint64 bit fields)
 *   Sector          A   = sum_k a_k = S_max + M,  S_max = sum_k s_k
 * The sector basis is ordered by increasing label n.  The combinatorial
 * rank (CR) reproduces exactly this order, so CLT and CR kernels act on
 * identically ordered vectors.
 *
 * Template parameters
 * -------------------
 *   TS      storage type of the Lanczos vectors
 *           (double, float, ftlm_fp16, ftlm_bf16)
 *   TC      compute / accumulation type (double for TS = double,
 *           float otherwise)
 *   LOOKUP  FTLM_LOOKUP_CLT (compressed lookup table) or
 *           FTLM_LOOKUP_CR  (combinatorial ranking)
 *
 * Vector layout: interleaved, V[idx * B + b] for B chains.
 * ================================================================
 */

#ifndef FTLM_KERNELS_CUH
#define FTLM_KERNELS_CUH

#define FTLM_MAX_SITES      32
#define FTLM_MAX_COUPLINGS  512     /* >= N(N-1)/2 for N = 32 */
#define FTLM_MAX_B          16      /* maximum block size (GPU) */
#define FTLM_LUT_W          16      /* digits per spin value: 2s+1 <= 16 */
#define FTLM_MAX_TWO_S      15      /* s <= 15/2 */
#define FTLM_LUT_SIZE       (FTLM_LUT_W * FTLM_LUT_W)
#define FTLM_CLT_BITS       5
#define FTLM_CLT_BS         32
#define FTLM_REDUCE_BS      256
#define FTLM_SPMV_BS        256

#define FTLM_LOOKUP_CLT     0
#define FTLM_LOOKUP_CR      1

typedef long long           ftlm_i64;
typedef unsigned long long  ftlm_u64;

/* ================================================================
 * Model parameters in constant memory
 *
 * Layout note: the Python package mirrors this struct with
 * ctypes.Structure (same field order and types).  Keep both in sync.
 * All accesses inside a warp use warp-uniform indices (site or
 * coupling loop counters), i.e. constant-memory broadcasts.
 * ================================================================ */
struct FtlmConst {
    /* coupling constants, compute-precision copies */
    double J_d [FTLM_MAX_COUPLINGS];    /* J_c                        */
    double hJ_d[FTLM_MAX_COUPLINGS];    /* J_c / 2                    */
    double s_d [FTLM_MAX_SITES];        /* local spin s_k             */
    float  J_f [FTLM_MAX_COUPLINGS];
    float  hJ_f[FTLM_MAX_COUPLINGS];
    float  s_f [FTLM_MAX_SITES];
    /* Clebsch-Gordan factors, indexed [two_s * FTLM_LUT_W + a]:
     *   raise(a) = sqrt(s(s+1) - m(m+1)),  lower(a) = sqrt(s(s+1) - m(m-1)),
     *   m = a - s.  Copied to shared memory by each thread block. */
    double lut_raise_d[FTLM_LUT_SIZE];
    double lut_lower_d[FTLM_LUT_SIZE];
    float  lut_raise_f[FTLM_LUT_SIZE];
    float  lut_lower_f[FTLM_LUT_SIZE];
    /* couplings (0-based site indices) */
    int    ci[FTLM_MAX_COUPLINGS];
    int    cj[FTLM_MAX_COUPLINGS];
    /* sites */
    int    two_s[FTLM_MAX_SITES];       /* 2 s_k                      */
    int    radix[FTLM_MAX_SITES];       /* d_k = 2 s_k + 1  (CLT)     */
    int    power[FTLM_MAX_SITES];       /* P_k              (CLT)     */
    int    shift[FTLM_MAX_SITES];       /* bit offset       (CR)      */
    int    mask [FTLM_MAX_SITES];       /* (1 << bits_k) - 1 (CR)     */
    /* scalars */
    int    N;
    int    n_coup;
    int    A_total;                     /* digit sum of the sector (CR) */
    int    dcum_pstride;                /* D_c index: p*pstride + A*astride + a */
    int    dcum_astride;
    int    dcum_size;                   /* number of int entries of D_c */
    int    pad0;
    int    pad1;
};

__constant__ FtlmConst c_p;

/* ================================================================
 * Storage types and conversions
 *
 * FP16 / BF16 conversions are written without cuda_fp16.h /
 * cuda_bf16.h so that the header compiles unchanged under NVRTC.
 * ================================================================ */
struct ftlm_fp16 { unsigned short x; };
struct ftlm_bf16 { unsigned short x; };

template <typename TS, typename TC> struct FtlmConv;

template <> struct FtlmConv<double, double> {
    static __device__ __forceinline__ double load (double v) { return v; }
    static __device__ __forceinline__ double store(double v) { return v; }
};
template <> struct FtlmConv<float, float> {
    static __device__ __forceinline__ float load (float v) { return v; }
    static __device__ __forceinline__ float store(float v) { return v; }
};
template <> struct FtlmConv<ftlm_fp16, float> {
    static __device__ __forceinline__ float load(ftlm_fp16 v) {
        float f;
        asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v.x));
        return f;
    }
    static __device__ __forceinline__ ftlm_fp16 store(float f) {
        ftlm_fp16 v;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(v.x) : "f"(f));
        return v;
    }
};
template <> struct FtlmConv<ftlm_bf16, float> {
    static __device__ __forceinline__ float load(ftlm_bf16 v) {
        return __uint_as_float(((unsigned int)v.x) << 16);
    }
    static __device__ __forceinline__ ftlm_bf16 store(float f) {
        /* round to nearest even (finite inputs) */
        unsigned int u = __float_as_uint(f);
        u += 0x7FFFu + ((u >> 16) & 1u);
        ftlm_bf16 v;
        v.x = (unsigned short)(u >> 16);
        return v;
    }
};

/* Compute-precision selection of the constant-memory copies */
template <typename TC> struct FtlmPar;
template <> struct FtlmPar<double> {
    static __device__ __forceinline__ double J (int c) { return c_p.J_d[c]; }
    static __device__ __forceinline__ double hJ(int c) { return c_p.hJ_d[c]; }
    static __device__ __forceinline__ double s (int k) { return c_p.s_d[k]; }
    static __device__ __forceinline__ double raise(int i) { return c_p.lut_raise_d[i]; }
    static __device__ __forceinline__ double lower(int i) { return c_p.lut_lower_d[i]; }
};
template <> struct FtlmPar<float> {
    static __device__ __forceinline__ float J (int c) { return c_p.J_f[c]; }
    static __device__ __forceinline__ float hJ(int c) { return c_p.hJ_f[c]; }
    static __device__ __forceinline__ float s (int k) { return c_p.s_f[k]; }
    static __device__ __forceinline__ float raise(int i) { return c_p.lut_raise_f[i]; }
    static __device__ __forceinline__ float lower(int i) { return c_p.lut_lower_f[i]; }
};

/* ================================================================
 * State-to-index maps
 * ================================================================ */

/* CLT: compressed lookup table (occupancy bitmask + prefix counts) */
__device__ __forceinline__ int ftlm_clt_lookup(
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    int label)
{
    int blk = label >> FTLM_CLT_BITS;
    int bit = label & (FTLM_CLT_BS - 1);
    unsigned int mask = __ldg(&block_mask[blk]);
    if (!(mask & (1u << bit)))
        return -1;
    unsigned int lower = mask & ((1u << bit) - 1u);
    return __ldg(&block_base[blk]) + __popc(lower);
}

/* Site data for the two kernel variants: UNI = 1 if all sites carry the
 * same spin (values of site 0 kept in registers, as in the v1 kernels),
 * UNI = 0 for mixed spins (per-site values from constant memory). */
template <int UNI> struct FtlmSites {
    int ts0, bits0;
    unsigned int mask0;
    __device__ __forceinline__ FtlmSites() {
        ts0   = c_p.two_s[0];
        mask0 = (unsigned int)c_p.mask[0];
        bits0 = __popc(mask0);
    }
    __device__ __forceinline__ int two_s(int k) const { return UNI ? ts0 : c_p.two_s[k]; }
    __device__ __forceinline__ int radix(int k) const { return UNI ? ts0 + 1 : c_p.radix[k]; }
    __device__ __forceinline__ int shift(int k) const { return UNI ? k * bits0 : c_p.shift[k]; }
    __device__ __forceinline__ unsigned int mask(int k) const {
        return UNI ? mask0 : (unsigned int)c_p.mask[k];
    }
};

/* CR: rank of a packed state (fixed-length loop over all N positions) */
template <int UNI>
__device__ __forceinline__ int ftlm_cr_rank(ftlm_u64 x, const int *s_dcum,
                                            const FtlmSites<UNI> &st)
{
    const int N  = c_p.N;
    const int ps = c_p.dcum_pstride;
    const int as = c_p.dcum_astride;
    int rank = 0;
    int Ak   = c_p.A_total;
    for (int p = N - 1; p >= 0; p--) {
        int a = (int)((x >> st.shift(p)) & (ftlm_u64)st.mask(p));
        rank += s_dcum[p * ps + Ak * as + a];
        Ak   -= a;
    }
    return rank;
}

/* CR: inverse map (per position a branch-free scan over the digits) */
template <int UNI>
__device__ __forceinline__ ftlm_u64 ftlm_cr_unrank(int rank, const int *s_dcum,
                                                   const FtlmSites<UNI> &st)
{
    const int N  = c_p.N;
    const int ps = c_p.dcum_pstride;
    const int as = c_p.dcum_astride;
    ftlm_u64 x = 0ULL;
    int rem = rank;
    int Ak  = c_p.A_total;
    for (int p = N - 1; p >= 0; p--) {
        int base  = p * ps + Ak * as;
        int a_max = st.two_s(p);
        int a_found = 0;
        for (int a = 1; a <= a_max; a++) {
            int val = s_dcum[base + a];
            a_found = (val <= rem) ? a : a_found;
        }
        rem -= s_dcum[base + a_found];
        Ak  -= a_found;
        x   |= ((ftlm_u64)a_found) << st.shift(p);
    }
    return x;
}

/* ================================================================
 * Block SpMV:  W = H V   (row-wise gather, one thread per output row)
 *
 * For each output basis state the state arithmetic (digit
 * decomposition, coupling loop, state-to-index lookups) is performed
 * once and amortized over the B vectors of the block.
 *
 * UNI = 1: uniform local spin (register fast path), UNI = 0: mixed spins.
 * Dynamic shared memory: D_c table (CR only), c_p.dcum_size ints.
 * ================================================================ */
template <typename TS, typename TC, int LOOKUP, int UNI>
__global__ void ftlm_spmv(
    TS                 * __restrict__ W,
    const TS           * __restrict__ V,
    const int          * __restrict__ block_base,   /* CLT */
    const unsigned int * __restrict__ block_mask,   /* CLT */
    const int          * __restrict__ basis,        /* CLT */
    const int          * __restrict__ dcum,         /* CR  */
    int dim, int B)
{
    typedef FtlmConv<TS, TC> CV;
    typedef FtlmPar<TC>      PP;

    __shared__ TC s_raise[FTLM_LUT_SIZE];
    __shared__ TC s_lower[FTLM_LUT_SIZE];
    extern __shared__ int s_dcum[];

    for (int i = threadIdx.x; i < FTLM_LUT_SIZE; i += blockDim.x) {
        s_raise[i] = PP::raise(i);
        s_lower[i] = PP::lower(i);
    }
    if (LOOKUP == FTLM_LOOKUP_CR) {
        for (int i = threadIdx.x; i < c_p.dcum_size; i += blockDim.x)
            s_dcum[i] = dcum[i];
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;

    const int N  = c_p.N;
    const int nc = c_p.n_coup;
    const FtlmSites<UNI> st;
    const TC s0 = PP::s(0);

    /* --- Digit decomposition (once for all B vectors) ---
     * CLT: digits of the mixed-radix label in a local array;
     * CR:  digits are read directly from the packed state (register). */
    int digits[FTLM_MAX_SITES];
    int      label  = 0;     /* CLT */
    ftlm_u64 packed = 0ULL;  /* CR  */
    if (LOOKUP == FTLM_LOOKUP_CLT) {
        label = __ldg(&basis[t]);
        int tmp = label;
        for (int k = 0; k < N; k++) {
            int d = st.radix(k);
            digits[k] = tmp % d;
            tmp /= d;
        }
    } else {
        packed = ftlm_cr_unrank<UNI>(t, s_dcum, st);
    }
#define FTLM_DIGIT(k) ((LOOKUP == FTLM_LOOKUP_CLT) ? digits[k] \
    : (int)((packed >> st.shift(k)) & (ftlm_u64)st.mask(k)))

    TC result[FTLM_MAX_B];

    /* --- Diagonal part: sum_c J_c m_i m_j --- */
    TC diag = (TC)0;
    for (int c = 0; c < nc; c++) {
        int i = c_p.ci[c];
        int j = c_p.cj[c];
        TC mi = (TC)FTLM_DIGIT(i) - (UNI ? s0 : PP::s(i));
        TC mj = (TC)FTLM_DIGIT(j) - (UNI ? s0 : PP::s(j));
        diag += PP::J(c) * (mi * mj);
    }

    ftlm_i64 t_base = (ftlm_i64)t * B;
    for (int b = 0; b < B; b++)
        result[b] = diag * CV::load(V[t_base + b]);

    /* --- Off-diagonal part: (J_c/2)(S+_i S-_j + S-_i S+_j) ---
     * Gather: the source state differs from the output state by
     * (a_i - 1, a_j + 1) or (a_i + 1, a_j - 1).  The matrix element is
     * evaluated from the digits of the output state (H is symmetric). */
    for (int c = 0; c < nc; c++) {
        int i  = c_p.ci[c];
        int j  = c_p.cj[c];
        int ai = FTLM_DIGIT(i);
        int aj = FTLM_DIGIT(j);
        int ti = st.two_s(i);
        int tj = st.two_s(j);

        /* source (a_i - 1, a_j + 1) */
        if (ai > 0 && aj < tj) {
            TC coeff = PP::hJ(c)
                * s_lower[ti * FTLM_LUT_W + ai]
                * s_raise[tj * FTLM_LUT_W + aj];
            int idx;
            if (LOOKUP == FTLM_LOOKUP_CLT)
                idx = ftlm_clt_lookup(block_base, block_mask,
                                      label - c_p.power[i] + c_p.power[j]);
            else
                idx = ftlm_cr_rank<UNI>(packed - (1ULL << st.shift(i))
                                               + (1ULL << st.shift(j)), s_dcum, st);
            if (idx >= 0) {
                ftlm_i64 a_base = (ftlm_i64)idx * B;
                for (int b = 0; b < B; b++)
                    result[b] += coeff * CV::load(V[a_base + b]);
            }
        }

        /* source (a_i + 1, a_j - 1) */
        if (ai < ti && aj > 0) {
            TC coeff = PP::hJ(c)
                * s_raise[ti * FTLM_LUT_W + ai]
                * s_lower[tj * FTLM_LUT_W + aj];
            int idx;
            if (LOOKUP == FTLM_LOOKUP_CLT)
                idx = ftlm_clt_lookup(block_base, block_mask,
                                      label + c_p.power[i] - c_p.power[j]);
            else
                idx = ftlm_cr_rank<UNI>(packed + (1ULL << st.shift(i))
                                               - (1ULL << st.shift(j)), s_dcum, st);
            if (idx >= 0) {
                ftlm_i64 a_base = (ftlm_i64)idx * B;
                for (int b = 0; b < B; b++)
                    result[b] += coeff * CV::load(V[a_base + b]);
            }
        }
    }

    for (int b = 0; b < B; b++)
        W[t_base + b] = CV::store(result[b]);
#undef FTLM_DIGIT
}

/* ================================================================
 * Vector import / export (column-major <-> interleaved)
 *
 * src/dst hold nb columns (column-major, dim rows) that correspond to
 * chains b0 .. b0+nb-1 of the interleaved block with B chains.
 * ================================================================ */
template <typename TS, typename TC, typename TIN>
__global__ void ftlm_import(TS * __restrict__ dst, const TIN * __restrict__ src,
                            int dim, int B, int b0, int nb)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int c = 0; c < nb; c++)
        dst[(ftlm_i64)t * B + b0 + c] =
            FtlmConv<TS, TC>::store((TC)src[t + (ftlm_i64)c * dim]);
}

template <typename TS, typename TC, typename TOUT>
__global__ void ftlm_export(TOUT * __restrict__ dst, const TS * __restrict__ src,
                            int dim, int B, int b0, int nb)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int c = 0; c < nb; c++)
        dst[t + (ftlm_i64)c * dim] =
            (TOUT)FtlmConv<TS, TC>::load(src[(ftlm_i64)t * B + b0 + c]);
}

/* ================================================================
 * Lanczos BLAS substitutes (interleaved layout, B chains)
 * ================================================================ */

/* partial[blk * B + b] = sum over the block of V[t,b] * W[t,b] */
template <typename TS, typename TC>
__global__ void ftlm_dot_partial(TC * __restrict__ partial,
                                 const TS * __restrict__ V,
                                 const TS * __restrict__ W,
                                 int dim, int B)
{
    typedef FtlmConv<TS, TC> CV;
    __shared__ TC sdata[FTLM_REDUCE_BS];
    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        TC sum = (TC)0;
        if (t < dim) {
            ftlm_i64 idx = (ftlm_i64)t * B + b;
            sum = CV::load(V[idx]) * CV::load(W[idx]);
        }
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

/* result[b] = sum_i partial[i * B + b] */
template <typename TC>
__global__ void ftlm_reduce_partial(TC * __restrict__ result,
                                    const TC * __restrict__ partial,
                                    int n_blocks, int B)
{
    int b = threadIdx.x;
    if (b >= B) return;
    TC sum = (TC)0;
    for (int i = 0; i < n_blocks; i++)
        sum += partial[i * B + b];
    result[b] = sum;
}

/* W -= alpha[b] V (- beta_prev[b] Vp), partial sums of ||W||^2.
 * The norm is taken of the value as stored (after rounding to TS). */
template <typename TS, typename TC>
__global__ void ftlm_ortho_norm_partial(TS       * __restrict__ W,
                                        const TS * __restrict__ V,
                                        const TS * __restrict__ Vp,
                                        const TC * __restrict__ alpha,
                                        const TC * __restrict__ beta_prev,
                                        TC       * __restrict__ partial,
                                        int dim, int B, int use_vp)
{
    typedef FtlmConv<TS, TC> CV;
    __shared__ TC sdata[FTLM_REDUCE_BS];
    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        TC w_val = (TC)0;
        if (t < dim) {
            ftlm_i64 idx = (ftlm_i64)t * B + b;
            w_val = CV::load(W[idx]) - alpha[b] * CV::load(V[idx]);
            if (use_vp)
                w_val -= beta_prev[b] * CV::load(Vp[idx]);
            TS w_st = CV::store(w_val);
            W[idx]  = w_st;
            w_val   = CV::load(w_st);
        }
        sdata[tid] = w_val * w_val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial[blockIdx.x * B + b] = sdata[0];
    }
}

/* W[t,b] *= scale[b] */
template <typename TS, typename TC>
__global__ void ftlm_scale(TS * __restrict__ W, const TC * __restrict__ scale,
                           int dim, int B)
{
    typedef FtlmConv<TS, TC> CV;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    ftlm_i64 base = (ftlm_i64)t * B;
    for (int b = 0; b < B; b++)
        W[base + b] = CV::store(CV::load(W[base + b]) * scale[b]);
}

/* Layout check for the Python ctypes mirror of FtlmConst:
 * out[0] = sizeof(FtlmConst), out[1..] = selected field offsets. */
__global__ void ftlm_const_layout(int *out)
{
    FtlmConst *p = (FtlmConst *)0;
    out[0] = (int)sizeof(FtlmConst);
    out[1] = (int)(ftlm_i64)(&p->J_f[0]);
    out[2] = (int)(ftlm_i64)(&p->lut_raise_d[0]);
    out[3] = (int)(ftlm_i64)(&p->ci[0]);
    out[4] = (int)(ftlm_i64)(&p->two_s[0]);
    out[5] = (int)(ftlm_i64)(&p->N);
    out[6] = (int)(ftlm_i64)(&p->dcum_size);
}

#endif /* FTLM_KERNELS_CUH */
