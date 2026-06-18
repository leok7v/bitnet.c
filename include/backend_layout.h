#ifndef BN_BACKEND_LAYOUT_H
#define BN_BACKEND_LAYOUT_H

#include "gpu_backend.h"
#include "backend_model.h"
#include "model_config.h"
#include "model_weights.h"
#include "quant.h"
#include "sh_arena.h"

typedef enum {
    BN_BACKEND_LAYOUT_OK = 0,
    BN_BACKEND_LAYOUT_NO_GPU,
    BN_BACKEND_LAYOUT_NO_BUFFER_CREATE,
    BN_BACKEND_LAYOUT_NO_BUFFER_CREATE_BIASED,
    BN_BACKEND_LAYOUT_MISSING_WEIGHT,
    BN_BACKEND_LAYOUT_I2S_NOT_STACKABLE,
    BN_BACKEND_LAYOUT_TYPE_MISMATCH,
    BN_BACKEND_LAYOUT_COL_MISMATCH,
    BN_BACKEND_LAYOUT_ZERO_SIZE,
    BN_BACKEND_LAYOUT_ALLOC_FAILED,
    BN_BACKEND_LAYOUT_BIAS_UNSUPPORTED,
} BnBackendLayoutReason;

typedef struct {
    size_t q4_repack_bytes;
    size_t q4k_scale_bytes;
    size_t q6k_weight_bytes;
    size_t q8_scale_bytes;
} BnBackendLayoutPreparedStats;

const char *bn_backend_layout_reason_string(BnBackendLayoutReason reason);

BnBackendLayoutReason bn_backend_layout_stackable_reason(const BnQWeight *a,
                                                         const BnQWeight *b);

int bn_backend_layout_stackable(const BnQWeight *a, const BnQWeight *b);

BnBackendLayoutReason bn_backend_layout_stacked2_reason(const BnGPUBackend *gpu,
                                                        const BnQWeight *a,
                                                        const BnQWeight *b);

BnBackendLayoutReason bn_backend_layout_biased_qweight_reason(const BnGPUBackend *gpu,
                                                              const BnQWeight *w,
                                                              const float *bias);

BnBackendLayoutReason bn_backend_layout_stacked3_qkv_reason(const BnGPUBackend *gpu,
                                                            const BnQWeight *q,
                                                            const BnQWeight *k,
                                                            const BnQWeight *v,
                                                            const float *q_bias,
                                                            const float *k_bias,
                                                            const float *v_bias,
                                                            int q_bias_fused,
                                                            int k_bias_fused,
                                                            int v_bias_fused);

void *bn_backend_layout_upload_stacked2(BnGPUBackend *gpu,
                                        const BnQWeight *a,
                                        const BnQWeight *b);

void *bn_backend_layout_upload_biased_qweight(BnGPUBackend *gpu,
                                              const BnQWeight *w,
                                              const float *bias);

void *bn_backend_layout_upload_stacked3_qkv(BnGPUBackend *gpu,
                                            const BnQWeight *q,
                                            const BnQWeight *k,
                                            const BnQWeight *v,
                                            const float *q_bias,
                                            const float *k_bias,
                                            const float *v_bias,
                                            int q_bias_fused,
                                            int k_bias_fused,
                                            int v_bias_fused);

/* skip_q4_0_repack: when nonzero, the CPU SIMD Q4_0 repack is neither sized nor
 * built (a GPU backend reads native Q4_0 blocks). CPU Q4_0 matmul still works
 * from raw blocks, so this is safe even if the GPU is later unavailable. Other
 * prepared kinds (Q4_K/Q6_K/Q8_0) are unaffected. */
size_t bn_backend_layout_prepared_qweights_size(const BnConfig *config,
                                                const BnWeights *weights,
                                                int skip_q4_0_repack,
                                                BnBackendLayoutPreparedStats *stats);

void bn_backend_layout_prepare_qweights(BnBackendModel *backend,
                                        const BnConfig *config,
                                        const BnWeights *weights,
                                        int skip_q4_0_repack,
                                        SHArena *arena,
                                        BnBackendLayoutPreparedStats *stats);

#endif // BN_BACKEND_LAYOUT_H
