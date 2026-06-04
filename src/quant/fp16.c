#include "quant.h"
#include <math.h>
#include <string.h>

// --- FP16 <-> FP32 conversion ---

float bn_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & BN_FP16_SIGN_MASK) << 16;
    uint32_t exp  = (h >> 10) & BN_FP16_EXP_MASK;
    uint32_t mant = h & BN_FP16_MANT_MASK;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // +/-0
        } else {
            // Subnormal: normalize by shifting mantissa left until hidden bit appears.
            exp = 1;
            while (!(mant & BN_FP16_HIDDEN_BIT)) { mant <<= 1; exp--; }
            mant &= BN_FP16_MANT_MASK;
            f = sign | ((uint32_t)(exp + BN_FP16_EXP_REBIAS) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | BN_FP32_EXP_INF | ((uint32_t)mant << 13);  // Inf/NaN
    } else {
        f = sign | ((uint32_t)(exp + BN_FP16_EXP_REBIAS) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

// --- BF16 -> FP32 conversion ---
// BF16 is the upper 16 bits of an IEEE 754 float32.
// Conversion: zero-pad the low 16 bits.

float bn_bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

uint16_t bn_fp32_to_fp16(float val) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(val) * scale_to_inf) * scale_to_zero;

    uint32_t w;
    memcpy(&w, &val, sizeof(w));

    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    const uint32_t base_bits = (bias >> 1) + 0x07800000u;
    float base_bias;
    memcpy(&base_bias, &base_bits, sizeof(base_bias));
    base += base_bias;

    uint32_t bits;
    memcpy(&bits, &base, sizeof(bits));
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) |
                      (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}
