/* Parity test for the native-layout Q4_0 Metal kernels (default; opt out with
 * BN_METAL_DISABLE_Q4_NATIVE).
 *
 * The native kernel reads the GGUF 18-byte Q4_0 block directly (no repacked
 * GPU copy) and dots it with fp32 activations. We validate it against a manual
 * fp32 dequant+dot reference (exactly what the kernel should compute), and we
 * also run the default repacked path against the same reference so a failure
 * pinpoints which path is wrong. Synthetic weights -> no model needed. */
#include "quant.h"
#include "gguf.h"
#include "gpu_backend.h"
#include "gpu_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef BN_ENABLE_METAL
int main(void) { printf("SKIP (built without BN_ENABLE_METAL)\n"); return 0; }
#else

static uint32_t rng = 1234567u;
static int rnd(int n) { rng = rng * 1664525u + 1013904223u; return (int)((rng >> 8) % (unsigned)n); }
static float rndf(void) { rng = rng * 1664525u + 1013904223u; return ((float)(rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; }

/* Build a synthetic Q4_0 weight: rows x cols (cols % 32 == 0). */
static void fill_q4_0(BnBlockQ4_0 *blocks, int rows, int cols) {
    int bpr = cols / 32;
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < bpr; b++) {
            BnBlockQ4_0 *blk = &blocks[(size_t)r * bpr + b];
            blk->d = bn_fp32_to_fp16(0.01f + 0.001f * (float)((r * 7 + b * 3) % 23));
            for (int i = 0; i < 16; i++)
                blk->qs[i] = (uint8_t)(rnd(16) | (rnd(16) << 4));
        }
}

/* Exact reference: dequantize Q4_0 (nibble-8)*d, dot with fp32 x. */
static void ref_matvec(float *out, const BnBlockQ4_0 *blocks,
                       int rows, int cols, const float *x) {
    int bpr = cols / 32;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int b = 0; b < bpr; b++) {
            const BnBlockQ4_0 *blk = &blocks[(size_t)r * bpr + b];
            float d = bn_fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            for (int i = 0; i < 16; i++) {
                int lo = (blk->qs[i] & 0x0F) - 8;
                int hi = (blk->qs[i] >> 4)  - 8;
                acc += d * ((float)lo * xb[i] + (float)hi * xb[i + 16]);
            }
        }
        out[r] = acc;
    }
}

static int run_case(int native, int rows, int cols) {
    /* Native is the default; force the repacked path with the opt-out env. */
    if (native) unsetenv("BN_METAL_DISABLE_Q4_NATIVE");
    else        setenv("BN_METAL_DISABLE_Q4_NATIVE", "1", 1);

    int bpr = cols / 32;
    BnBlockQ4_0 *blocks = calloc((size_t)rows * bpr, sizeof(BnBlockQ4_0));
    float *x = malloc((size_t)cols * sizeof(float));
    float *ref = calloc(rows, sizeof(float));
    float *got = calloc(rows, sizeof(float));
    int8_t *xq = calloc(cols > rows ? cols : rows, 1);
    fill_q4_0(blocks, rows, cols);
    for (int j = 0; j < cols; j++) x[j] = rndf();
    ref_matvec(ref, blocks, rows, cols, x);

    BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
    if (!gpu) { printf("  SKIP (no Metal device)\n"); free(blocks);free(x);free(ref);free(got);free(xq); return 0; }

    size_t sz = (size_t)rows * bpr * sizeof(BnBlockQ4_0);
    void *wbuf = gpu->buffer_create(gpu->ctx, blocks, sz, BN_GGUF_TENSOR_Q4_0, rows, cols);
    int fail = 0;
    if (!wbuf) { printf("  buffer_create failed\n"); fail = 1; }
    else {
        int rc = gpu->matvec(gpu->ctx, got, wbuf, x, rows, cols, BN_GGUF_TENSOR_Q4_0);
        if (rc != 0) { printf("  matvec rc=%d\n", rc); fail = 1; }
        else {
            float max_abs = 0.0f;
            for (int r = 0; r < rows; r++) {
                float a = fabsf(got[r] - ref[r]);
                if (a > max_abs) max_abs = a;
            }
            /* Reference uses fp32 activations. The native kernel also uses fp32
             * x -> near-exact (fp16-scale rounding only). The repacked path
             * q8-quantizes x, so it differs from the exact reference by the
             * activation-quant error -- gate it loosely (it is not under test
             * here; it's a cross-check that both compute the right thing). */
            float tol = native ? 1e-3f : 5e-2f;
            int ok = max_abs < tol;
            printf("  %-8s rows=%d cols=%d  max_abs=%.2e (tol %.0e)  %s\n",
                   native ? "native" : "repack", rows, cols, max_abs, tol,
                   ok ? "PASS" : "FAIL");
            fail = !ok;
        }
        gpu->buffer_destroy(gpu->ctx, wbuf);
    }
    bn_gpu_metal_destroy(gpu);
    free(blocks); free(x); free(ref); free(got); free(xq);
    return fail;
}

/* GEMM (mul_mm) parity: out[t*rows + r] = W[r,:] . X[t,:]. cols%256==0,
 * rows>=64, n_tokens>=32 so the native mul_mm GEMM pipeline engages (else it
 * falls back to per-token native matvec, which this also exercises safely). */
static int run_matmul_case(int native, int rows, int cols, int n_tokens) {
    if (native) unsetenv("BN_METAL_DISABLE_Q4_NATIVE");
    else        setenv("BN_METAL_DISABLE_Q4_NATIVE", "1", 1);

    int bpr = cols / 32;
    BnBlockQ4_0 *blocks = calloc((size_t)rows * bpr, sizeof(BnBlockQ4_0));
    float *X = malloc((size_t)n_tokens * cols * sizeof(float));
    float *ref = calloc((size_t)n_tokens * rows, sizeof(float));
    float *got = calloc((size_t)n_tokens * rows, sizeof(float));
    fill_q4_0(blocks, rows, cols);
    for (int t = 0; t < n_tokens; t++)
        for (int j = 0; j < cols; j++) X[t * cols + j] = rndf();
    for (int t = 0; t < n_tokens; t++)
        ref_matvec(ref + (size_t)t * rows, blocks, rows, cols, X + (size_t)t * cols);

    BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
    if (!gpu) { printf("  SKIP (no Metal device)\n"); free(blocks);free(X);free(ref);free(got); return 0; }
    int fail = 0;
    if (!gpu->matmul) { printf("  SKIP (no matmul vtable)\n"); }
    else {
        size_t sz = (size_t)rows * bpr * sizeof(BnBlockQ4_0);
        void *wbuf = gpu->buffer_create(gpu->ctx, blocks, sz, BN_GGUF_TENSOR_Q4_0, rows, cols);
        if (!wbuf) { printf("  buffer_create failed\n"); fail = 1; }
        else {
            int rc = gpu->matmul(gpu->ctx, got, wbuf, X, rows, cols, n_tokens, BN_GGUF_TENSOR_Q4_0);
            if (rc != 0) { printf("  matmul rc=%d\n", rc); fail = 1; }
            else {
                float max_abs = 0.0f;
                for (int i = 0; i < n_tokens * rows; i++) {
                    float a = fabsf(got[i] - ref[i]);
                    if (a > max_abs) max_abs = a;
                }
                float tol = native ? 1e-2f : 5e-2f; /* half-accum GEMM -> looser */
                int ok = max_abs < tol;
                printf("  %-8s rows=%d cols=%d tok=%d  max_abs=%.2e (tol %.0e)  %s\n",
                       native ? "native" : "repack", rows, cols, n_tokens, max_abs, tol,
                       ok ? "PASS" : "FAIL");
                fail = !ok;
            }
            gpu->buffer_destroy(gpu->ctx, wbuf);
        }
    }
    bn_gpu_metal_destroy(gpu);
    free(blocks); free(X); free(ref); free(got);
    return fail;
}

int main(void) {
    printf("=== Q4_0 Metal matvec parity (native vs repacked, vs fp32 ref) ===\n");
    int fails = 0;
    int dims[][2] = { {64, 256}, {128, 512}, {320, 1024}, {7, 128} };
    for (int repack = 0; repack <= 1; repack++)
        for (int i = 0; i < (int)(sizeof dims / sizeof dims[0]); i++)
            fails += run_case(repack /*native when 1*/, dims[i][0], dims[i][1]);
    printf("=== Q4_0 Metal mul_mm GEMM parity (n_tokens>=32) ===\n");
    int mm[][3] = { {128, 256, 40}, {256, 512, 64}, {64, 256, 32} };
    for (int repack = 0; repack <= 1; repack++)
        for (int i = 0; i < (int)(sizeof mm / sizeof mm[0]); i++)
            fails += run_matmul_case(repack, mm[i][0], mm[i][1], mm[i][2]);
    printf(fails ? "FAILED (%d)\n" : "ALL Q4_0 NATIVE PARITY PASS\n", fails);
    return fails ? 1 : 0;
}
#endif
