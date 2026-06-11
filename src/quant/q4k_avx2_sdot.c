#include "quant_ctx.h"
#include "simd_helpers.h"
#include <immintrin.h>
#include <string.h>

void bn_quant_q4k_avx2_sdot_range(void *ctx, int row_start, int row_end) {
    BnKQuantSdotCtx *c = (BnKQuantSdotCtx *)ctx;
    int cols = c->W->cols;
    int n_blocks_per_row = cols / BN_QK_K;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;
    const int8_t *x_q = c->x_q;
    const float *x_d = c->x_d;
    const int16_t *x_bsums = c->x_bsums;

    const __m256i mask_lo = _mm256_set1_epi8(0xF);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        __m256 row_acc = _mm256_setzero_ps();
        float row_corr = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const BnBlockQ4K *blk = &blocks[(size_t)row * n_blocks_per_row + b];
            _mm_prefetch((const char *)(blk + 1), _MM_HINT_T0);
            float d    = bn_fp16_to_fp32(blk->d);
            float dmin = bn_fp16_to_fp32(blk->dmin);
            float dx   = x_d[b];
            const uint8_t *qs = blk->qs;
            const int8_t *xb = x_q + b * BN_QK_K;
            const int16_t *bsums = x_bsums + b * 16;

            uint32_t utmp[3];
            memcpy(utmp, blk->scales, 12);
            uint32_t m_lo = utmp[1] & kmask1;
            uint32_t m_hi = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[0] &= kmask1;
            const uint8_t *sc = (const uint8_t *)utmp;
            uint8_t mins[8];
            memcpy(mins, &m_lo, 4);
            memcpy(mins + 4, &m_hi, 4);

            __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
            __m128i bs_lo = _mm256_castsi256_si128(q8sums);
            __m128i bs_hi = _mm256_extracti128_si256(q8sums, 1);
            __m128i bs_paired = _mm_hadd_epi16(bs_lo, bs_hi);
            __m128i mins_16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mins));
            __m128i corr128 = _mm_madd_epi16(mins_16, bs_paired);
            __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
            __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
            int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

            __m256i sumi_v = _mm256_setzero_si256();
            for (int p = 0; p < 4; p++) {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(qs + p * 32));
                __m256i lo = _mm256_and_si256(raw, mask_lo);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);
                __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb + p * 64));
                __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + p * 64 + 32));
                __m256i plo = _mm256_maddubs_epi16(lo, xv0);
                __m256i phi = _mm256_maddubs_epi16(hi, xv1);
                plo = _mm256_madd_epi16(_mm256_set1_epi16((int16_t)sc[2 * p]), plo);
                phi = _mm256_madd_epi16(_mm256_set1_epi16((int16_t)sc[2 * p + 1]), phi);
                sumi_v = _mm256_add_epi32(sumi_v, _mm256_add_epi32(plo, phi));
            }

            row_acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi_v),
                                       _mm256_set1_ps(dx * d), row_acc);
            row_corr += dx * dmin * (float)bsum_corr;
        }
        c->out[row] = bn_avx2_hsum_ps(row_acc) - row_corr;
    }
}

// Reordered matmul: row → token_tile → block → token_in_tile.
// Accumulates all blocks into local acc[] per token tile before
// writing to the scattered output, reducing cache misses.
#define Q4K_TILE_T 16

void bn_quant_q4k_avx2_sdot_matmul_range(void *ctx, int row_start, int row_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnBlockQ4K *blocks = (const BnBlockQ4K *)c->W->data;

    const __m256i mask_lo = _mm256_set1_epi8(0xF);

    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int row = row_start; row < row_end; row++) {
        for (int t0 = 0; t0 < n_tokens; t0 += Q4K_TILE_T) {
            int tile_n = t0 + Q4K_TILE_T <= n_tokens ? Q4K_TILE_T : n_tokens - t0;
            float acc[Q4K_TILE_T] = {0};

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ4K *blk = &blocks[(size_t)row * n_bpr + b];
                float d    = bn_fp16_to_fp32(blk->d);
                float dmin = bn_fp16_to_fp32(blk->dmin);
                const uint8_t *qs = blk->qs;

                uint32_t utmp[3];
                memcpy(utmp, blk->scales, 12);
                uint32_t m_lo_w = utmp[1] & kmask1;
                uint32_t m_hi_w = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[0] &= kmask1;
                const uint8_t *sc = (const uint8_t *)utmp;
                uint8_t mins[8];
                memcpy(mins, &m_lo_w, 4);
                memcpy(mins + 4, &m_hi_w, 4);

                __m256i w_lo[4], w_hi[4];
                {
                    const uint8_t *qp = qs;
                    for (int p = 0; p < 4; p++) {
                        __m256i raw = _mm256_loadu_si256((const __m256i *)qp);
                        w_lo[p] = _mm256_and_si256(raw, mask_lo);
                        w_hi[p] = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo);
                        qp += 32;
                    }
                }

                __m256i sc_v[8];
                for (int p = 0; p < 8; p++)
                    sc_v[p] = _mm256_set1_epi16((int16_t)sc[p]);

                __m128i mins_16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)mins));

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    const int8_t *xb = c->x_q + (size_t)t * cols + b * BN_QK_K;
                    float dx = c->x_d[(size_t)t * n_bpr + b];
                    const int16_t *bsums = c->x_bsums + ((size_t)t * n_bpr + b) * 16;

                    __m256i q8sums = _mm256_loadu_si256((const __m256i *)bsums);
                    __m128i bs_lo = _mm256_castsi256_si128(q8sums);
                    __m128i bs_hi = _mm256_extracti128_si256(q8sums, 1);
                    __m128i bs_paired = _mm_hadd_epi16(bs_lo, bs_hi);
                    __m128i corr128 = _mm_madd_epi16(mins_16, bs_paired);
                    __m128i ch1 = _mm_hadd_epi32(corr128, corr128);
                    __m128i ch2 = _mm_hadd_epi32(ch1, ch1);
                    int32_t bsum_corr = _mm_cvtsi128_si32(ch2);

                    __m256i sumi_v = _mm256_setzero_si256();
                    for (int p = 0; p < 4; p++) {
                        __m256i xv0 = _mm256_loadu_si256((const __m256i *)(xb + p * 64));
                        __m256i xv1 = _mm256_loadu_si256((const __m256i *)(xb + p * 64 + 32));
                        __m256i plo = _mm256_maddubs_epi16(w_lo[p], xv0);
                        __m256i phi = _mm256_maddubs_epi16(w_hi[p], xv1);
                        plo = _mm256_madd_epi16(sc_v[2 * p], plo);
                        phi = _mm256_madd_epi16(sc_v[2 * p + 1], phi);
                        sumi_v = _mm256_add_epi32(sumi_v, _mm256_add_epi32(plo, phi));
                    }

                    int32_t sumi = bn_avx2_hsum_epi32(sumi_v);
                    acc[ti] += dx * (d * (float)sumi - dmin * (float)bsum_corr);
                }
            }

            for (int ti = 0; ti < tile_n; ti++)
                c->out[(size_t)(t0 + ti) * rows + row] = acc[ti];
        }
    }
}

void bn_quant_q4k_avx2_x8_matmul_range(void *ctx, int group_start, int group_end) {
    BnKQuantMatmulCtx *c = (BnKQuantMatmulCtx *)ctx;
    int cols = c->cols;
    int rows = c->W->rows;
    int n_bpr = cols / BN_QK_K;
    int n_tokens = c->n_tokens;
    const BnPreparedWeight *prepared =
        (c->prepared && c->prepared->kind == BN_PREPARED_WEIGHT_Q4_K_SCALES &&
         c->prepared->aux) ? c->prepared : NULL;
    if (!prepared) {
        bn_quant_q4k_avx2_sdot_matmul_range(ctx, group_start * 8, group_end * 8);
        return;
    }

    const BnBlockQ4Kx8 *blocks = (const BnBlockQ4Kx8 *)prepared->aux;
    const __m128i deltamask =
        _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    const __m128i scalemask =
        _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    const __m256i finalpermutemask =
        _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    for (int g = group_start; g < group_end; g++) {
        int row0 = g * 8;
        if (row0 >= rows) break;
        const BnBlockQ4Kx8 *group_blocks = blocks + (size_t)g * n_bpr;

#define Q4K_X8_TILE_T 4
        for (int t0 = 0; t0 < n_tokens; t0 += Q4K_X8_TILE_T) {
            int tile_n = t0 + Q4K_X8_TILE_T <= n_tokens
                ? Q4K_X8_TILE_T : n_tokens - t0;
            __m256 acc_row[Q4K_X8_TILE_T];
            __m256 acc_min_rows[Q4K_X8_TILE_T];
            for (int ti = 0; ti < Q4K_X8_TILE_T; ti++) {
                acc_row[ti] = _mm256_setzero_ps();
                acc_min_rows[ti] = _mm256_setzero_ps();
            }

            for (int b = 0; b < n_bpr; b++) {
                const BnBlockQ4Kx8 *blk = &group_blocks[b];
                const __m256 col_scale_f32 = _mm256_cvtph_ps(
                    _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)blk->d),
                                     deltamask));
                const __m256 col_dmin_f32 = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *)blk->dmin));

                const int8_t *xb_tile[Q4K_X8_TILE_T];
                __m256 row_scale_f32[Q4K_X8_TILE_T];
                __m256i q8s_tile[Q4K_X8_TILE_T];
                __m256i iacc_b[Q4K_X8_TILE_T];
                __m256i iacc_min_b[Q4K_X8_TILE_T];

                for (int ti = 0; ti < tile_n; ti++) {
                    int t = t0 + ti;
                    xb_tile[ti] =
                        c->x_q + (size_t)t * cols + b * BN_QK_K;
                    float dx = c->x_d[(size_t)t * n_bpr + b];
                    row_scale_f32[ti] = _mm256_set1_ps(dx);
                    const int16_t *bsums =
                        c->x_bsums + ((size_t)t * n_bpr + b) * 16;
                    const __m256i q8sums =
                        _mm256_loadu_si256((const __m256i *)bsums);
                    __m256i q8s = _mm256_castsi128_si256(
                        _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                       _mm256_extracti128_si256(q8sums, 1)));
                    q8s_tile[ti] = _mm256_permute2f128_si256(q8s, q8s, 0);
                    iacc_b[ti] = _mm256_setzero_si256();
                    iacc_min_b[ti] = _mm256_setzero_si256();
                }

                for (int sb = 0; sb < BN_QK_K / 64; sb++) {
                    const uint8_t *qsp = blk->qs + sb * 256;
                    const __m256i rhs_raw_vec_0123_0 =
                        _mm256_loadu_si256((const __m256i *)(qsp));
                    const __m256i rhs_raw_vec_4567_0 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 32));
                    const __m256i rhs_raw_vec_0123_1 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 64));
                    const __m256i rhs_raw_vec_4567_1 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 96));
                    const __m256i rhs_raw_vec_0123_2 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 128));
                    const __m256i rhs_raw_vec_4567_2 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 160));
                    const __m256i rhs_raw_vec_0123_3 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 192));
                    const __m256i rhs_raw_vec_4567_3 =
                        _mm256_loadu_si256((const __m256i *)(qsp + 224));

                    const __m256i rhs_vec_0123_00 =
                        _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 =
                        _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 =
                        _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 =
                        _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 =
                        _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 =
                        _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 =
                        _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 =
                        _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    const __m256i rhs_vec_0123_10 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 =
                        _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    uint32_t utmp_0[4], utmp_1[4];
                    memcpy(utmp_0, blk->scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) |
                                (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) |
                                (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    memcpy(utmp_1, blk->scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) |
                                (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) |
                                (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    const __m128i mins_and_scales_0 =
                        _mm_set_epi32((int)utmp_0[3], (int)utmp_0[2],
                                      (int)utmp_0[1], (int)utmp_0[0]);
                    const __m128i scales_rearrange_0 =
                        _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    const __m256i scales_0 =
                        _mm256_cvtepu8_epi16(scales_rearrange_0);

                    const __m128i mins_and_scales_1 =
                        _mm_set_epi32((int)utmp_1[3], (int)utmp_1[2],
                                      (int)utmp_1[1], (int)utmp_1[0]);
                    const __m128i scales_rearrange_1 =
                        _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    const __m256i scales_1 =
                        _mm256_cvtepu8_epi16(scales_rearrange_1);

                    const __m256i mins_01 = _mm256_cvtepu8_epi16(
                        _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                                          _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    for (int ti = 0; ti < tile_n; ti++) {
                        const int8_t *xb = xb_tile[ti];
                        __m256i lhs_vec_00 = _mm256_castsi128_si256(
                            _mm_loadu_si128((const __m128i *)(xb + sb * 64)));
                        __m256i lhs_vec_01 = _mm256_castsi128_si256(
                            _mm_loadu_si128((const __m128i *)(xb + 16 + sb * 64)));
                        __m256i lhs_vec_10 = _mm256_castsi128_si256(
                            _mm_loadu_si128((const __m128i *)(xb + 32 + sb * 64)));
                        __m256i lhs_vec_11 = _mm256_castsi128_si256(
                            _mm_loadu_si128((const __m128i *)(xb + 48 + sb * 64)));

                        lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                        lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                        lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                        lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                        __m256i iacc_0 = _mm256_setzero_si256();
                        __m256i iacc_1 = _mm256_setzero_si256();

                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00, _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177), rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01, _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177), rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02, _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177), rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03, _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                        iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177), rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));
                        iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10, _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177), rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11, _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177), rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12, _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177), rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13, _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                        iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177), rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));
                        iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                        const __m256i iacc_sb =
                            _mm256_add_epi32(iacc_0, iacc_1);
                        const __m256i q8s_sb =
                            _mm256_shuffle_epi32(q8s_tile[ti], 0);
                        const __m256i iacc_min_sb =
                            _mm256_madd_epi16(q8s_sb, mins_01);
                        q8s_tile[ti] = _mm256_bsrli_epi128(q8s_tile[ti], 4);

                        iacc_b[ti] =
                            _mm256_add_epi32(iacc_b[ti], iacc_sb);
                        iacc_min_b[ti] =
                            _mm256_add_epi32(iacc_min_b[ti], iacc_min_sb);
                    }
                }

                for (int ti = 0; ti < tile_n; ti++) {
                    acc_row[ti] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_b[ti]),
                        _mm256_mul_ps(col_scale_f32, row_scale_f32[ti]),
                        acc_row[ti]);
                    acc_min_rows[ti] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_min_b[ti]),
                        _mm256_mul_ps(col_dmin_f32, row_scale_f32[ti]),
                        acc_min_rows[ti]);
                }
            }

            for (int ti = 0; ti < tile_n; ti++) {
                __m256 row = _mm256_permutevar8x32_ps(acc_row[ti],
                                                      finalpermutemask);
                _mm256_storeu_ps(c->out + (size_t)(t0 + ti) * rows + row0,
                                 _mm256_sub_ps(row, acc_min_rows[ti]));
            }
        }
#undef Q4K_X8_TILE_T
    }
}
