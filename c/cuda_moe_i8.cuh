/* A prompt chunk's activations in int8 for the integer tensor cores (m16n8k32 s8.s8.s32): the rows of
 * x a layer's routed experts read, and the SwiGLU hidden rows their down reads, quantized once a
 * chunk where the f16 tiles (hcm_moe) turn them into f16 (hcm_half_rows_k, hcm_half_k).
 *
 * The arithmetic is gq_quant_act_ref's (ggml_quants.h) and bdx_quant_block's (cuda_bd.cuh), bit for
 * bit: per block of 32 values d = amax / 127 and id = 127 / amax (0 for a block of zeros), each code
 * round-to-nearest-even of v * id held to -127 .. 127, and the codes' sums per 16. So a row's codes,
 * scales and sums are the ones the CPU's integer form and the GEMV's (bd_gemv_i8_k) give it; a tile
 * kernel's integer dots over them are exact, as theirs are. Inputs are finite, and a block's maximum
 * zero or past 127 / FLT_MAX (so 127 / amax is finite), as for those two.
 *
 * The layout is planar, for tiles that load codes by cp.async:
 *   q: rows x K int8, row r at q + r * K (K % 32 == 0, so a row is a whole number of 16-byte chunks);
 *   d: rows x K/32 float scales;
 *   s: rows x K/16 int32 code sums (the formats with a minimum, Q4_1, Q5_1, Q2_K, Q4_K, Q5_K, read them).
 * cmi_x_bytes(rows, K) bytes in one buffer at cmi_x_at's offsets (each plane on a 256-byte boundary).
 *
 * The quantization: a warp's lane takes 4 values (one float4), 8 lanes a block of 32, a warp 4 blocks;
 * the block's maximum and the sums by shuffles inside the 8 lanes (a maximum and an integer sum are
 * exact in any order), the codes stored as one word a lane. Nothing here is wired into the prefill
 * yet: hcm_moe keeps its f16 tiles; a tile kernel on these rows comes with its own switch. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef CMI_PDL_WAIT
#ifdef COLI_PDL_WAIT
#define CMI_PDL_WAIT() COLI_PDL_WAIT()
#else
#define CMI_PDL_WAIT() ((void)0)
#endif
#endif

/* past a decode step's rows (the GEMV and bd_mmg's 1 .. 8): the rows a chunk quantizes this way */
#define CMI_MIN_ROWS 9

struct CmiX { int8_t *q; float *d; int32_t *s; int K; };

__host__ __device__ constexpr size_t cmi_al256(size_t n) { return (n + 255) / 256 * 256; }
__host__ __device__ constexpr size_t cmi_q_bytes(size_t rows, int K) { return cmi_al256(rows * (size_t)K); }
__host__ __device__ constexpr size_t cmi_d_bytes(size_t rows, int K) { return cmi_al256(rows * (size_t)(K / 32) * 4); }
__host__ __device__ constexpr size_t cmi_s_bytes(size_t rows, int K) { return cmi_al256(rows * (size_t)(K / 16) * 4); }
/* the three planes of rows of K values, in one buffer */
__host__ __device__ constexpr size_t cmi_x_bytes(size_t rows, int K) { return cmi_q_bytes(rows, K) + cmi_d_bytes(rows, K) + cmi_s_bytes(rows, K); }
/* the planes in a buffer of cmi_x_bytes(rows, K) at base (256-byte aligned) */
__host__ __device__ inline CmiX cmi_x_at(void *base, size_t rows, int K) {
    char *b = (char *)base;
    CmiX X = {(int8_t *)b, (float *)(b + cmi_q_bytes(rows, K)), (int32_t *)(b + cmi_q_bytes(rows, K) + cmi_d_bytes(rows, K)), K};
    return X;
}
/* the shapes cmi_quant_* take: whole blocks of 32, float4 rows (the row stride too) */
__host__ __device__ constexpr bool cmi_shape_ok(int K, int x_row) { return K > 0 && K % 32 == 0 && x_row >= K && x_row % 4 == 0; }

/* lane l of a group of 8 holds values 4 l .. 4 l + 3 of a block of 32 (v): the block's codes into
 * its word (byte c the code of value 4 l + c); the block's scale, and its sums of 16 (lanes 0-3, 4-7)
 * returned to every lane of the group. All 32 lanes of the warp call it together. */
__device__ __forceinline__ uint32_t cmi_quant4(float4 v, float *d, int *s16) {
    float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float id = a > 0.f ? __fdiv_rn(127.f, a) : 0.f;
    *d = __fdiv_rn(a, 127.f);
    int c[4];
    c[0] = __float2int_rn(__fmul_rn(v.x, id)); c[1] = __float2int_rn(__fmul_rn(v.y, id));
    c[2] = __float2int_rn(__fmul_rn(v.z, id)); c[3] = __float2int_rn(__fmul_rn(v.w, id));
    uint32_t w = 0;
    int sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        c[k] = c[k] > 127 ? 127 : c[k] < -127 ? -127 : c[k];
        sum += c[k];
        w |= ((uint32_t)c[k] & 255u) << (8 * k);
    }
    /* the sums of 16: lanes 0-3 and 4-7 of the group */
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    *s16 = sum;
    return w;
}

/* S rows of K values (row r at x + r * x_row floats) into X's planes: a warp 4 blocks of 32, a row's
 * blocks in order; grid-stride over the S * K / 128 groups (the last group of a row may hold fewer
 * blocks when K % 128 != 0: those lanes load nothing and take no part in another row's block) */
__global__ static void __launch_bounds__(256) cmi_quant_rows_k(const float *__restrict__ x, int x_row, int S, CmiX X) {
    CMI_PDL_WAIT();
    const int lane = threadIdx.x & 31, nb = X.K / 32, gpr = (nb + 3) / 4;   /* groups of 4 blocks a row */
    const long long ng = (long long)S * gpr, wpb = blockDim.x / 32;
    for (long long g = (long long)blockIdx.x * wpb + threadIdx.x / 32; g < ng; g += (long long)gridDim.x * wpb) {
        const int r = (int)(g / gpr), b = (int)(g % gpr) * 4 + (lane >> 3), l = lane & 7;
        const bool live = b < nb;   /* uniform across the 8 lanes of a block: the shuffles stay inside it */
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (live) v = reinterpret_cast<const float4 *>(x + (size_t)r * x_row + 32 * b)[l];
        float d; int s16;
        const uint32_t w = cmi_quant4(v, &d, &s16);
        if (!live) continue;
        reinterpret_cast<uint32_t *>(X.q + (size_t)r * X.K + 32 * b)[l] = w;
        if (l == 0) X.d[(size_t)r * nb + b] = d;
        if ((l & 3) == 0) X.s[(size_t)r * 2 * nb + 2 * b + (l >> 2)] = s16;
    }
}

/* the hidden rows of a chunk's slots from their gate and up (each n rows of I, rows g_row floats
 * apart; u may be g + I for an interleaved [g | u] row) through SwiGLU, silu(g) * u as CC_EW_SWIGLU and
 * hcm_body's GLU write it, and into X's planes: the epilogue a gate and up tile without its own
 * quantization would hand to down, and the bits of cmi_quant_rows_k on those float rows */
__device__ __forceinline__ float cmi_swiglu(float g, float u) { return (g / (1.f + expf(-g))) * u; }
__global__ static void __launch_bounds__(256) cmi_quant_swiglu_k(const float *__restrict__ g, const float *__restrict__ u, int g_row, int n, CmiX X) {
    CMI_PDL_WAIT();
    const int lane = threadIdx.x & 31, nb = X.K / 32, gpr = (nb + 3) / 4;
    const long long ng = (long long)n * gpr, wpb = blockDim.x / 32;
    for (long long gi = (long long)blockIdx.x * wpb + threadIdx.x / 32; gi < ng; gi += (long long)gridDim.x * wpb) {
        const int r = (int)(gi / gpr), b = (int)(gi % gpr) * 4 + (lane >> 3), l = lane & 7;
        const bool live = b < nb;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (live) {
            const float4 a = reinterpret_cast<const float4 *>(g + (size_t)r * g_row + 32 * b)[l];
            const float4 c = reinterpret_cast<const float4 *>(u + (size_t)r * g_row + 32 * b)[l];
            v = make_float4(cmi_swiglu(a.x, c.x), cmi_swiglu(a.y, c.y), cmi_swiglu(a.z, c.z), cmi_swiglu(a.w, c.w));
        }
        float d; int s16;
        const uint32_t w = cmi_quant4(v, &d, &s16);
        if (!live) continue;
        reinterpret_cast<uint32_t *>(X.q + (size_t)r * X.K + 32 * b)[l] = w;
        if (l == 0) X.d[(size_t)r * nb + b] = d;
        if ((l & 3) == 0) X.s[(size_t)r * 2 * nb + 2 * b + (l >> 2)] = s16;
    }
}

/* the launches: 0 for a shape they do not take (nothing launched), else 1 (launch errors are the
 * stream's, as with any launch) */
static inline unsigned cmi_grid(long long groups) {
    long long b = (groups + 7) / 8;   /* 8 warps a block */
    return (unsigned)(b < 1 ? 1 : b > 65535 * 4 ? 65535 * 4 : b);
}
static inline int cmi_quant_rows(const float *x, int x_row, int S, CmiX X, cudaStream_t st) {
    if (S <= 0 || !cmi_shape_ok(X.K, x_row) || ((uintptr_t)x & 15)) return 0;
    cmi_quant_rows_k<<<cmi_grid((long long)S * ((X.K / 32 + 3) / 4)), 256, 0, st>>>(x, x_row, S, X);
    return 1;
}
static inline int cmi_quant_swiglu(const float *g, const float *u, int g_row, int n, CmiX X, cudaStream_t st) {
    if (n <= 0 || !cmi_shape_ok(X.K, g_row) || ((uintptr_t)g & 15) || ((uintptr_t)u & 15)) return 0;
    cmi_quant_swiglu_k<<<cmi_grid((long long)n * ((X.K / 32 + 3) / 4)), 256, 0, st>>>(g, u, g_row, n, X);
    return 1;
}
