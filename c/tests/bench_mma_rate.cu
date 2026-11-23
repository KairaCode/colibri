/* The tensor cores' sustained rate per mma.sync form, on this card: the question a prompt chunk's
 * matmuls ask before their f16 tiles (hcm_*: m16n8k16 f16 operands, f32 sums) move to int8
 * (m16n8k32 s8 operands, s32 sums). A GeForce card runs f16 products with f32 sums at half the
 * rate of f16 sums, and its int8 at twice the f16 sums' rate; a workstation or datacenter card
 * runs f16 with f32 sums at the full rate. Which one this card is decides whether int8 tiles can
 * be worth their per-block scales. Self-contained: no backend, no weights.
 *
 * Forms: f16.f16.f32 m16n8k16 (hcm_mma's), f16.f16.f16 m16n8k16, s8.s8.s32 m16n8k16 (bd_mma16's)
 * and s8.s8.s32 m16n8k32 (bd_mma32's). Two loops each:
 *   reg  - the operands in registers, 16 independent sums a warp: the issue rate alone;
 *   ldsm - hcm_body's warp tile: per k step 4 A fragments (64 rows) and 4 B fragments (32 outputs)
 *          by ldmatrix.x4 from a swizzled shared tile, then 4 x 4 products. f16 k16 and s8 k32 read
 *          the same 512 bytes a fragment, so the int8 step does twice the products per byte.
 * Each runs one full wave (the blocks the card holds at once) at 4, 8 and 16 warps a multiprocessor
 * for about 60 ms, best of 5; the clock is the kernel's own (clock64 over %globaltimer in block 0), so
 * the products a clock and multiprocessor do not depend on boost. sm_80 and later.
 *
 *   make mma-rate-bench               (or: nvcc -O3 -std=c++17 -arch=native tests/bench_mma_rate.cu -o bench_mma_rate)
 *   ./bench_mma_rate [ms]             ms: each run's target length, default 60 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("FAIL %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

enum { F_F16F32 = 0, F_F16F16 = 1, F_S8K16 = 2, F_S8K32 = 3, NFORM = 4 };
static const char *const form_name[NFORM] = {"f16.f16.f32 m16n8k16", "f16.f16.f16 m16n8k16", "s8.s8.s32   m16n8k16", "s8.s8.s32   m16n8k32"};
static const int form_k[NFORM] = {16, 16, 16, 32};
static const int form_int[NFORM] = {0, 0, 1, 1};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define MR_BODY 1
#else
#define MR_BODY 0
#endif

/* one accumulator: four 32-bit words (f32 or s32 sums), two for f16 sums */
template <int F>
__device__ __forceinline__ void mr_mma(uint32_t (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if MR_BODY
    if (F == F_F16F32)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else if (F == F_F16F16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else if (F == F_S8K16)   /* A two words, B one: the k16 int8 form */
        asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    else
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c; (void)a; (void)b0; (void)b1;
#endif
}

#if MR_BODY
__device__ __forceinline__ void mr_ldm4(uint32_t (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
/* hcm_swz: row r's 16-byte chunk c of a tile of 64-byte rows, chunks xor (r / 2) & 3 */
__device__ __forceinline__ unsigned mr_swz(int r, int c) { return (unsigned)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }
#endif

#define MR_NACC 16
#define MR_STAGES 4
/* a stage: 64 rows of A and 32 of B, 64 bytes each (one k step of f16 k16 or s8 k32 is 32 bytes a row,
 * hcm's tile has two k16 steps a row: here a stage serves two k steps, its chunks 0-1 and 2-3) */
#define MR_STAGE_BYTES ((64 + 32) * 64)

/* out: a word a thread (the sums, so nothing is dead); block 0's clock64 and %globaltimer (ns) deltas at out_clk[0..1] */
template <int F, int LDSM>
__global__ void __launch_bounds__(512) mr_k(uint32_t *out, long long *out_clk, int iters, uint32_t seed) {
#if MR_BODY
    __shared__ __align__(128) unsigned char sm[LDSM ? MR_STAGES * MR_STAGE_BYTES : 16];
    const int lane = threadIdx.x & 31;
    unsigned long long g0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    long long t0 = clock64();
    uint32_t c[MR_NACC][4];
    #pragma unroll
    for (int j = 0; j < MR_NACC; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0;
    /* small values of either sign: f16 sums stay finite, s32 wraps harmlessly */
    const uint32_t h = F >= F_S8K16 ? 0x01FF02FEu : 0x1C00A000u;
    if (!LDSM) {
        uint32_t a[4] = {h ^ seed ^ (uint32_t)lane, h + (uint32_t)lane, h ^ 0x80008000u, h}, b0 = h ^ (seed >> 3), b1 = h + 7u;
        for (int it = 0; it < iters; it++) {
            #pragma unroll
            for (int j = 0; j < MR_NACC; j++) mr_mma<F>(c[j], a, b0, b1);
        }
    } else {
        for (int i = threadIdx.x; i < (int)sizeof sm / 4; i += blockDim.x) reinterpret_cast<uint32_t *>(sm)[i] = h ^ (uint32_t)(i * 2654435761u) ^ seed;
        __syncthreads();
        const unsigned base = (unsigned)__cvta_generic_to_shared(sm);
        /* ldmatrix.x4's addresses as hcm's: lanes 0-15 rows 0-15 at the k chunk, lanes 16-31 the next chunk */
        const int lr = lane & 15, lc = lane >> 4;
        for (int it = 0; it < iters; it++) {
            const unsigned st = base + (unsigned)((it >> 1) & (MR_STAGES - 1)) * MR_STAGE_BYTES;
            const int kc = (it & 1) * 2;
            uint32_t A[4][4], B[2][4];
            #pragma unroll
            for (int i = 0; i < 4; i++) mr_ldm4(A[i], st + mr_swz(16 * i + lr, kc + lc));
            #pragma unroll
            for (int i = 0; i < 2; i++) mr_ldm4(B[i], st + 64 * 64 + mr_swz(16 * i + lr, kc + lc));
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) mr_mma<F>(c[4 * i + j], A[i], B[j >> 1][(j & 1) * 2], B[j >> 1][(j & 1) * 2 + 1]);
            }
        }
    }
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < MR_NACC; j++) s += c[j][0] ^ c[j][1] ^ c[j][2] ^ c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    long long t1 = clock64();
    unsigned long long g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    if (blockIdx.x == 0 && threadIdx.x == 0) { out_clk[0] = t1 - t0; out_clk[1] = (long long)(g1 - g0); }
#else
    (void)out; (void)out_clk; (void)iters; (void)seed;
#endif
}

struct MrRes { double tops, mhz, per_clk_sm; int warps; };

template <int F, int LDSM>
static MrRes mr_run(int sms, double target_ms, uint32_t *dout, long long *dclk) {
    MrRes best = {0, 0, 0, 0};
    const int k = form_k[F];
    for (int warps : {4, 8, 16}) {
        const int threads = 128;
        int bps = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, mr_k<F, LDSM>, threads, 0));
        int want = warps / 4;
        if (bps < want) continue;
        const int blocks = sms * want;
        /* the iterations for about target_ms, from a short probe */
        /* doubled until a run takes 5 ms (a short probe is all launch overhead, and a run scaled from
         * it can outlast a display card's watchdog), then scaled to target_ms */
        int iters = 256;
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        mr_k<F, LDSM><<<blocks, threads>>>(dout, dclk, iters, 1u);
        CK(cudaDeviceSynchronize());
        float ms = 0;
        for (;;) {
            CK(cudaEventRecord(e0));
            mr_k<F, LDSM><<<blocks, threads>>>(dout, dclk, iters, 2u);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms >= 5.f || iters >= 1 << 24) break;
            iters *= 2;
        }
        double scale = target_ms / ms;
        iters = (int)(iters * (scale < 40.0 ? scale : 40.0));
        if (iters < 256) iters = 256;
        double bt = 0, bmhz = 0;
        for (int r = 0; r < 5; r++) {
            CK(cudaEventRecord(e0));
            mr_k<F, LDSM><<<blocks, threads>>>(dout, dclk, iters, 3u + (uint32_t)r);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&ms, e0, e1));
            long long clk[2] = {0, 0}; CK(cudaMemcpy(clk, dclk, sizeof clk, cudaMemcpyDeviceToHost));
            /* 2 x 16 x 8 x k operations a product, MR_NACC products a warp an iteration */
            const double ops = 2.0 * 16 * 8 * k * MR_NACC * (double)iters * (double)blocks * (threads / 32);
            const double tops = ops / (ms * 1e-3) / 1e12;
            if (tops > bt) { bt = tops; bmhz = clk[1] > 0 ? (double)clk[0] * 1e3 / (double)clk[1] : 0; }
        }
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        if (bt > best.tops) { best.tops = bt; best.mhz = bmhz; best.warps = warps; best.per_clk_sm = bmhz > 0 ? bt * 1e12 / 2.0 / (bmhz * 1e6) / sms : 0; }
    }
    return best;
}

int main(int argc, char **argv) {
    double target_ms = argc > 1 ? atof(argv[1]) : 60.0;
    if (!(target_ms > 0) || target_ms > 200) target_ms = 60.0;   /* a run past ~2 s meets Windows' display watchdog */
    int dev = 0, n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { printf("SKIP: no CUDA device\n"); return 0; }
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    if (p.major < 8) { printf("SKIP: compute capability %d.%d, mma.sync's int8 and ldmatrix forms need 8.0\n", p.major, p.minor); return 0; }
    printf("%s, sm_%d%d, %d multiprocessors; each form's best of 4, 8 and 16 warps a multiprocessor, one wave, ~%.0f ms a run\n",
           p.name, p.major, p.minor, p.multiProcessorCount, target_ms);
    uint32_t *dout; long long *dclk;
    CK(cudaMalloc(&dout, (size_t)p.multiProcessorCount * 512 * 4 * 4)); CK(cudaMalloc(&dclk, 2 * sizeof(long long)));
    MrRes r[2][NFORM];
    r[0][0] = mr_run<F_F16F32, 0>(p.multiProcessorCount, target_ms, dout, dclk);
    r[0][1] = mr_run<F_F16F16, 0>(p.multiProcessorCount, target_ms, dout, dclk);
    r[0][2] = mr_run<F_S8K16, 0>(p.multiProcessorCount, target_ms, dout, dclk);
    r[0][3] = mr_run<F_S8K32, 0>(p.multiProcessorCount, target_ms, dout, dclk);
    r[1][0] = mr_run<F_F16F32, 1>(p.multiProcessorCount, target_ms, dout, dclk);
    r[1][1] = mr_run<F_F16F16, 1>(p.multiProcessorCount, target_ms, dout, dclk);
    r[1][2] = mr_run<F_S8K16, 1>(p.multiProcessorCount, target_ms, dout, dclk);
    r[1][3] = mr_run<F_S8K32, 1>(p.multiProcessorCount, target_ms, dout, dclk);
    int bad = 0;
    for (int m = 0; m < 2; m++) {
        printf("%s\n", m ? "ldsm (hcm's warp tile: ldmatrix.x4 fragments from shared memory, 4 x 4 products a k step)" : "reg (operands in registers)");
        for (int f = 0; f < NFORM; f++) {
            if (r[m][f].tops <= 0) { printf("  %s: no run (occupancy)\n", form_name[f]); bad++; continue; }
            printf("  %s: %7.1f %s  at %4.0f MHz, %5.0f MACs a clock a multiprocessor (%d warps)\n", form_name[f], r[m][f].tops,
                   form_int[f] ? "TOPS  " : "TFLOPS", r[m][f].mhz, r[m][f].per_clk_sm, r[m][f].warps);
        }
        if (r[m][0].tops > 0)
            printf("  ratio to f16.f16.f32: f16 sums %.2fx, s8 k16 %.2fx, s8 k32 %.2fx\n", r[m][1].tops / r[m][0].tops, r[m][2].tops / r[m][0].tops,
                   r[m][3].tops / r[m][0].tops);
    }
    CK(cudaFree(dout)); CK(cudaFree(dclk));
    printf(bad ? "FAILED: %d forms did not run\n" : "OK\n", bad);
    return bad != 0;
}
