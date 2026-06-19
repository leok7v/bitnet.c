/* SSM recurrent-state carry across bn_prefill calls (requires a hybrid GGUF).
 *
 * The agent multi-turn loop rewinds transient (thinking) tokens via
 * snapshot/restore + KV-truncate + delta-prefill. This gate proves that the
 * underlying primitive reproduces a fresh full prefill on every backend:
 *   A  split prefill      [0,N)+[N,M)             == prefill [0,M)
 *   B  rewind/restore      snapshot, run throwaway, restore+truncate, [N,M)
 *   C  chained 2 "turns"   snapshot AFTER a restore+prefill (the agent pattern)
 *
 * The GPU SSM buffers are per-context, so each from-scratch run is zero-seeded
 * (set_recurrent_state with a zeroed buffer) to make the comparison valid.
 * CPU is bit-exact; Metal carries ~1e-2 fp noise, so argmax must match and the
 * logit max-abs delta must stay under TOL.
 *
 * Usage: ./test_rewind <model.gguf> [--metal]   (needs a real hybrid model) */
#include "model.h"
#include "session.h"
#include "generate.h"
#include "gguf.h"
#include "threadpool.h"
#ifdef BN_ENABLE_METAL
#include "gpu_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TOL 0.1

static int argmax(const float *v, int n) {
    int a = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[a]) a = i;
    return a;
}

static int check(const char *be, const char *name, const float *L,
                 const float *ref, int vocab, int amax_ref) {
    int a = argmax(L, vocab);
    double maxabs = 0;
    for (int i = 0; i < vocab; i++) {
        double d = fabs((double)L[i] - (double)ref[i]);
        if (d > maxabs) maxabs = d;
    }
    int ok = (a == amax_ref) && (maxabs < TOL);
    printf("  [%s] %-16s argmax ref=%d got=%d max_abs=%.5g  %s\n",
           be, name, amax_ref, a, maxabs, ok ? "PASS" : "*** FAIL ***");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--metal]\n", argv[0]);
        return 2;
    }
    int use_metal = 0;
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--metal")) use_metal = 1;
    const char *be = use_metal ? "metal" : "cpu";

    BnGGUFFile *gf = bn_gguf_open_file(argv[1]);
    if (!gf) { fprintf(stderr, "gguf open failed\n"); return 1; }
    BnModel model;
    if (bn_model_load(&model, gf, 4096, 0, 0) != 0) {
        fprintf(stderr, "model load failed\n"); return 1;
    }
    bn_model_set_thread_pool(&model, bn_tp_create(4), 1);
    if (model.config.full_attn_interval <= 0 || model.config.ssm_inner_size <= 0) {
        printf("SKIP: %s is not a hybrid SSM model\n", argv[1]);
        return 0;
    }
#ifdef BN_ENABLE_METAL
    if (use_metal) {
        BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
        const BnMappedFile *mf = bn_gguf_primary_file(gf);
        if (gpu && mf && mf->is_mmap && mf->data)
            bn_gpu_metal_set_mmap_range(gpu, mf->data, mf->size);
        if (!gpu || bn_model_upload_weights(&model, gpu) != 0 ||
            (gpu->init_activations &&
             gpu->init_activations(gpu->ctx, &model.config) != 0)) {
            fprintf(stderr, "metal attach failed\n"); return 1;
        }
    }
#else
    if (use_metal) { printf("SKIP: built without BN_ENABLE_METAL\n"); return 0; }
#endif

    int vocab = model.config.vocab_size;
    int M = 48, N = 20, N1 = 16, K = 34;
    int toks[64], transient[16];
    for (int i = 0; i < M; i++) toks[i] = (i * 1013 + 7) % vocab;
    for (int i = 0; i < K - N; i++) transient[i] = (i * 577 + 99) % vocab;
    size_t snb = bn_session_recurrent_state_bytes(&model);
    void *zero = calloc(1, snb), *snap = malloc(snb), *snap1 = malloc(snb);
    if (!zero || !snap || !snap1) { fprintf(stderr, "oom\n"); return 1; }

    #define ZERO_SEED(s) do { \
        bn_session_set_recurrent_state((s), &model, zero, snb); \
        bn_session_kv_truncate((s), 0); } while (0)

    /* reference: zero-seeded fresh full prefill [0,M) */
    BnSession *s = bn_session_create(&model, NULL);
    ZERO_SEED(s);
    float *ref = malloc((size_t)vocab * sizeof(float));
    memcpy(ref, bn_prefill(&model, s, toks, M, 0, 0),
           (size_t)vocab * sizeof(float));
    int amax_ref = argmax(ref, vocab);
    bn_session_free(s, NULL);
    int fail = 0;

    /* A: split prefill [0,N)+[N,M) */
    s = bn_session_create(&model, NULL);
    ZERO_SEED(s);
    bn_prefill(&model, s, toks, N, 0, 0);
    fail += check(be, "split-prefill",
                  bn_prefill(&model, s, toks + N, M - N, N, 0),
                  ref, vocab, amax_ref);
    bn_session_free(s, NULL);

    /* B: snapshot, throwaway transient, restore + truncate, real [N,M) */
    s = bn_session_create(&model, NULL);
    ZERO_SEED(s);
    bn_prefill(&model, s, toks, N, 0, 0);
    bn_session_get_recurrent_state(s, &model, snap, snb);
    bn_prefill(&model, s, transient, K - N, N, 0);
    bn_session_set_recurrent_state(s, &model, snap, snb);
    bn_session_kv_truncate(s, N);
    fail += check(be, "rewind/restore",
                  bn_prefill(&model, s, toks + N, M - N, N, 0),
                  ref, vocab, amax_ref);
    bn_session_free(s, NULL);

    /* C: chained 2-turn -- snapshot AFTER a restore+prefill (agent pattern) */
    s = bn_session_create(&model, NULL);
    ZERO_SEED(s);
    bn_prefill(&model, s, toks, N1, 0, 0);
    bn_session_get_recurrent_state(s, &model, snap1, snb);
    bn_prefill(&model, s, transient, K - N, N1, 0);
    bn_session_set_recurrent_state(s, &model, snap1, snb);
    bn_session_kv_truncate(s, N1);
    fail += check(be, "chained-2-turn",
                  bn_prefill(&model, s, toks + N1, M - N1, N1, 0),
                  ref, vocab, amax_ref);
    bn_session_free(s, NULL);

    printf("%s\n", fail ? "*** test_rewind FAILED ***" : "All rewind checks passed!");
    return fail ? 1 : 0;
}
