#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <arm_neon.h>

#define FAST_EXP_FP16_LOG2E_F16 1.442695f16   // log2(e)
#define FAST_EXP_FP16_LN2_F16 0.693147f16     // ln(2)
#define FAST_EXP_FP16_P2_F16 0.5f16           // ~ 1/2
#define FAST_EXP_FP16_P3_F16 0.166667f16      // ~ 1/6
#define FAST_EXP_FP16_ONE_F16 1.0f16

// Input clamp range for FP16 exp().
#define FAST_EXP_FP16_CLAMP_MAX_F16 11.0f16
#define FAST_EXP_FP16_CLAMP_MIN_F16 (-18.0f16)

// Range reduction / FP16 exponent construction parameters.
#define FAST_EXP_FP16_N_MAX 15
#define FAST_EXP_FP16_EXP_BIAS 15
#define FAST_EXP_FP16_EXP_MIN 0
#define FAST_EXP_FP16_EXP_MAX 30
#define FAST_EXP_FP16_EXP_SHIFT 10

// --------------------------------------------------------
// Fast Exp for FP16 (float16x8_t)
// Algorithm: range reduction + 3rd-order polynomial approximation.
// --------------------------------------------------------
static inline float16x8_t fast_exp_fp16_neon(float16x8_t x) {
    // ----------------------------------------------------
    // 1) Constants
    // ----------------------------------------------------
    const float16x8_t log2e = vdupq_n_f16(FAST_EXP_FP16_LOG2E_F16);
    const float16x8_t ln2 = vdupq_n_f16(FAST_EXP_FP16_LN2_F16);

    // Polynomial coefficients (low-order approximation is sufficient for FP16).
    const float16x8_t p2 = vdupq_n_f16(FAST_EXP_FP16_P2_F16);
    const float16x8_t p3 = vdupq_n_f16(FAST_EXP_FP16_P3_F16);
    const float16x8_t one = vdupq_n_f16(FAST_EXP_FP16_ONE_F16);

    // ----------------------------------------------------
    // 2) Input clamping (critical for FP16)
    // ----------------------------------------------------
    // FP16 max finite is ~65504. exp(11.1) ≈ 66171 (would overflow to Inf).
    // exp(-18) ≈ 1.5e-8 (close to FP16 precision floor).
    // Clamp to a safe range to keep computations stable.
    const float16x8_t max_val = vdupq_n_f16(FAST_EXP_FP16_CLAMP_MAX_F16);
    const float16x8_t min_val = vdupq_n_f16(FAST_EXP_FP16_CLAMP_MIN_F16);
    
    x = vminq_f16(x, max_val);
    x = vmaxq_f16(x, min_val);

    // ----------------------------------------------------
    // 3) Range reduction: exp(x) = 2^n * exp(r)
    // ----------------------------------------------------
    // n = round(x * log2(e))
    const float16x8_t z = vmulq_f16(x, log2e);

    // Convert to integer exponent.
    // NOTE: FP16 cannot represent 2^16 as a finite value (it overflows to Inf),
    // so we clamp n <= 15 and recompute r accordingly.
    int16x8_t n = vcvtaq_s16_f16(z);
    n = vminq_s16(n, vdupq_n_s16(FAST_EXP_FP16_N_MAX));

    // r = x - n * ln(2)  (use FMA form to reduce error)
    const float16x8_t n_float = vcvtq_f16_s16(n);
    const float16x8_t r = vfmsq_f16(x, n_float, ln2);

    // ----------------------------------------------------
    // 4) Polynomial approximation
    // ----------------------------------------------------
    // exp(r) ≈ 1 + r + r^2/2 + r^3/6
    // Horner form: 1 + r * (1 + r * (p2 + r * p3))
    
    // poly = p2 + r * p3
    float16x8_t poly = vfmaq_f16(p2, r, p3);
    
    // poly = 1 + r * poly
    poly = vfmaq_f16(one, poly, r);
    
    // poly = 1 + r * poly
    poly = vfmaq_f16(one, poly, r);

    // ----------------------------------------------------
    // 5) Reconstruct result: exp(x) ≈ poly(r) * 2^n
    // ----------------------------------------------------
    // Build 2^n as an FP16 value by constructing the exponent bits directly.
    // This intentionally only produces normal values (and 0 for underflow).
    //
    // WARNING:
    // - Do NOT shift/reinterpret a signed int16 directly. For n < -15 that can
    //   set the FP16 sign bit and yield negative/NaN results.
    // - FP16 exponent all-ones (31) encodes Inf/NaN, so we must avoid 31.
    //
    // FP16 layout: sign:1 | exponent:5 | mantissa:10, exponent bias = 15
    int16x8_t exp_int = vaddq_s16(n, vdupq_n_s16(FAST_EXP_FP16_EXP_BIAS));  // n + bias
    exp_int = vmaxq_s16(exp_int, vdupq_n_s16(FAST_EXP_FP16_EXP_MIN));       // underflow -> 0
    exp_int = vminq_s16(exp_int, vdupq_n_s16(FAST_EXP_FP16_EXP_MAX));       // avoid Inf/NaN

    // Shift exponent into place (<< 10 for FP16 mantissa bits).
    const int16x8_t exp_shifted = vshlq_n_s16(exp_int, FAST_EXP_FP16_EXP_SHIFT);

    // Reinterpret as FP16 with sign=0, mantissa=0.
    const float16x8_t two_pow_n = vreinterpretq_f16_s16(exp_shifted);
    
    // Final multiply.
    return vmulq_f16(poly, two_pow_n);
}

static inline float16x8_t silu_fp16_neon(float16x8_t x) {
    float16x8_t one = vdupq_n_f16(1.0f16);
    
    // 1) Compute sigmoid(x) in FP16:
    // sigmoid(x) = 1 / (1 + exp(-x))
    float16x8_t neg_x = vnegq_f16(x);
    
    // Use our FP16 fast exp implementation.
    // Note: fast_exp_fp16_neon() clamps its input to a safe FP16 range.
    // For SiLU, large positive x means exp(-x) should be ~0; clamping on the
    // negative side (e.g. -18) keeps the value tiny in FP16 as intended.
    float16x8_t exp_val = fast_exp_fp16_neon(neg_x);
    
    float16x8_t den = vaddq_f16(one, exp_val);
    
    // 2) Reciprocal via Newton–Raphson:
    // recp ≈ 1 / den
    float16x8_t recp = vrecpeq_f16(den);
    // One iteration is typically enough for FP16 accuracy.
    recp = vmulq_f16(vrecpsq_f16(den, recp), recp);
    
    // 3) SiLU(x) = x * sigmoid(x)
    return vmulq_f16(x, recp);
}

static inline float16x8_t silu_mul_fp16_neon(float16x8_t gate, float16x8_t up) {
    return vmulq_f16(silu_fp16_neon(gate), up);
}

static inline float16x8_t add_fp16_neon(float16x8_t a, float16x8_t b) {
    return vaddq_f16(a, b);
}

static inline void fp16_bits_to_fp32_neon(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t bits = vld1q_u16(src + i);
        const float16x8_t half = vreinterpretq_f16_u16(bits);
        const float32x4_t lo = vcvt_f32_f16(vget_low_f16(half));
        const float32x4_t hi = vcvt_f32_f16(vget_high_f16(half));
        vst1q_f32(dst + i, lo);
        vst1q_f32(dst + i + 4, hi);
    }

    if (i < n) {
        uint16_t tmp_in[8] = {};
        float tmp_out[8];
        const size_t r = n - i;
        for (size_t j = 0; j < r; ++j) {
            tmp_in[j] = src[i + j];
        }

        const uint16x8_t bits = vld1q_u16(tmp_in);
        const float16x8_t half = vreinterpretq_f16_u16(bits);
        const float32x4_t lo = vcvt_f32_f16(vget_low_f16(half));
        const float32x4_t hi = vcvt_f32_f16(vget_high_f16(half));
        vst1q_f32(tmp_out, lo);
        vst1q_f32(tmp_out + 4, hi);

        for (size_t j = 0; j < r; ++j) {
            dst[i + j] = tmp_out[j];
        }
    }
}

static inline void fp32_to_fp16_bits_neon(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t lo = vld1q_f32(src + i);
        const float32x4_t hi = vld1q_f32(src + i + 4);
        const float16x8_t half = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst + i, vreinterpretq_u16_f16(half));
    }

    if (i < n) {
        float tmp_in[8] = {};
        uint16_t tmp_out[8];
        const size_t r = n - i;
        for (size_t j = 0; j < r; ++j) {
            tmp_in[j] = src[i + j];
        }

        const float32x4_t lo = vld1q_f32(tmp_in);
        const float32x4_t hi = vld1q_f32(tmp_in + 4);
        const float16x8_t half = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(tmp_out, vreinterpretq_u16_f16(half));

        for (size_t j = 0; j < r; ++j) {
            dst[i + j] = tmp_out[j];
        }
    }
}

#endif
