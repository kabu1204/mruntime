#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <arm_neon.h>

#define FAST_EXP_FP16_LOG2E_F16 1.442695f16   // log2(e)
#define FAST_EXP_FP16_LN2_F16 0.693147f16     // ln(2)
#define FAST_EXP_FP16_P2_F16 0.5f16           // ~ 1/2
#define FAST_EXP_FP16_P3_F16 0.166667f16      // ~ 1/6
#define FAST_EXP_FP16_ONE_F16 1.0f16

// Input clamp range for FP32 exp() to avoid overflow/underflow in exp construction.
#define FAST_EXP_FP32_CLAMP_MAX_F32 88.3762626647949f
#define FAST_EXP_FP32_CLAMP_MIN_F32 (-88.3762626647949f)

// Range-reduction constants for exp(x) = 2^n * exp(r), with r around 0.
#define FAST_EXP_FP32_LOG2E_F32 1.44269504088896341f
#define FAST_EXP_FP32_LN2_HI_F32 0.693359375f
#define FAST_EXP_FP32_LN2_LO_F32 (-2.12194440e-4f)

// Cephes polynomial coefficients for exp(r) over reduced range.
#define FAST_EXP_FP32_P0_F32 1.9875691500e-4f
#define FAST_EXP_FP32_P1_F32 1.3981999507e-3f
#define FAST_EXP_FP32_P2_F32 8.3334519073e-3f
#define FAST_EXP_FP32_P3_F32 4.1665795894e-2f
#define FAST_EXP_FP32_P4_F32 1.6666665459e-1f
#define FAST_EXP_FP32_P5_F32 5.0000001201e-1f

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
// Fast Exp for FP32 (float32x4_t)
// Algorithm: range reduction + polynomial approximation.
// --------------------------------------------------------
static inline float32x4_t fast_exp_fp32_neon(float32x4_t x) {
    const float32x4_t max_val = vdupq_n_f32(FAST_EXP_FP32_CLAMP_MAX_F32);
    const float32x4_t min_val = vdupq_n_f32(FAST_EXP_FP32_CLAMP_MIN_F32);
    x = vminq_f32(x, max_val);
    x = vmaxq_f32(x, min_val);

    const float32x4_t log2e = vdupq_n_f32(FAST_EXP_FP32_LOG2E_F32);
    const float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t fx = vfmaq_f32(half, x, log2e);  // x * log2(e) + 0.5

    // floor(fx) without requiring dedicated floor intrinsics.
    int32x4_t n = vcvtq_s32_f32(fx);
    float32x4_t n_float = vcvtq_f32_s32(n);
    const uint32x4_t gt_mask = vcgtq_f32(n_float, fx);
    n = vsubq_s32(n, vreinterpretq_s32_u32(vandq_u32(gt_mask, vdupq_n_u32(1))));
    n_float = vcvtq_f32_s32(n);

    const float32x4_t ln2_hi = vdupq_n_f32(FAST_EXP_FP32_LN2_HI_F32);
    const float32x4_t ln2_lo = vdupq_n_f32(FAST_EXP_FP32_LN2_LO_F32);
    float32x4_t r = vfmsq_f32(x, n_float, ln2_hi);
    r = vfmsq_f32(r, n_float, ln2_lo);

    const float32x4_t p0 = vdupq_n_f32(FAST_EXP_FP32_P0_F32);
    const float32x4_t p1 = vdupq_n_f32(FAST_EXP_FP32_P1_F32);
    const float32x4_t p2 = vdupq_n_f32(FAST_EXP_FP32_P2_F32);
    const float32x4_t p3 = vdupq_n_f32(FAST_EXP_FP32_P3_F32);
    const float32x4_t p4 = vdupq_n_f32(FAST_EXP_FP32_P4_F32);
    const float32x4_t p5 = vdupq_n_f32(FAST_EXP_FP32_P5_F32);
    const float32x4_t one = vdupq_n_f32(1.0f);

    const float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t poly = vfmaq_f32(p1, p0, r);
    poly = vfmaq_f32(p2, poly, r);
    poly = vfmaq_f32(p3, poly, r);
    poly = vfmaq_f32(p4, poly, r);
    poly = vfmaq_f32(p5, poly, r);
    poly = vfmaq_f32(vaddq_f32(one, r), poly, r2);

    // Build 2^n from exponent bits (FP32 exponent bias = 127, mantissa bits = 23).
    int32x4_t exp_bits = vaddq_s32(n, vdupq_n_s32(127));
    exp_bits = vshlq_n_s32(exp_bits, 23);
    const float32x4_t two_pow_n = vreinterpretq_f32_s32(exp_bits);

    return vmulq_f32(poly, two_pow_n);
}

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

// Apply rotary position embedding to one [head_dim] vector in-place.
// Layout: x[0:half_dim] is real, x[half_dim:2*half_dim] is imag.
// cos_sin is interleaved: [cos0, sin0, cos1, sin1, ...] for this position.
// Returns the number of rotary pairs processed (multiple of 8).
static inline size_t rope_fp16_bits_inplace_neon(uint16_t* x, const float* cos_sin, size_t half_dim) {
    size_t i = 0;
    for (; i + 8 <= half_dim; i += 8) {
        const float32x4x2_t cs0 = vld2q_f32(cos_sin + i * 2);
        const float32x4x2_t cs1 = vld2q_f32(cos_sin + (i + 4) * 2);

        const uint16x8_t x0u = vld1q_u16(x + i);
        const uint16x8_t x1u = vld1q_u16(x + half_dim + i);
        const float16x8_t x0h = vreinterpretq_f16_u16(x0u);
        const float16x8_t x1h = vreinterpretq_f16_u16(x1u);

        const float32x4_t x0_lo = vcvt_f32_f16(vget_low_f16(x0h));
        const float32x4_t x0_hi = vcvt_f32_f16(vget_high_f16(x0h));
        const float32x4_t x1_lo = vcvt_f32_f16(vget_low_f16(x1h));
        const float32x4_t x1_hi = vcvt_f32_f16(vget_high_f16(x1h));

        const float32x4_t out0_lo = vfmsq_f32(vmulq_f32(x0_lo, cs0.val[0]), x1_lo, cs0.val[1]);
        const float32x4_t out1_lo = vfmaq_f32(vmulq_f32(x0_lo, cs0.val[1]), x1_lo, cs0.val[0]);
        const float32x4_t out0_hi = vfmsq_f32(vmulq_f32(x0_hi, cs1.val[0]), x1_hi, cs1.val[1]);
        const float32x4_t out1_hi = vfmaq_f32(vmulq_f32(x0_hi, cs1.val[1]), x1_hi, cs1.val[0]);

        const float16x8_t out0h = vcombine_f16(vcvt_f16_f32(out0_lo), vcvt_f16_f32(out0_hi));
        const float16x8_t out1h = vcombine_f16(vcvt_f16_f32(out1_lo), vcvt_f16_f32(out1_hi));

        vst1q_u16(x + i, vreinterpretq_u16_f16(out0h));
        vst1q_u16(x + half_dim + i, vreinterpretq_u16_f16(out1h));
    }
    return i;
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

// RMSNorm: out = (x / rms(x)) * weight, where rms(x) = sqrt(mean(x^2) + eps)
// Processes one token row of hidden_size elements.
static inline void rmsnorm_fp16_neon(
    const uint16_t* input,
    const uint16_t* weight,
    uint16_t* output,
    size_t hidden_size,
    float eps
) {
    // Phase 1: Vectorized sum-of-squares with 4 accumulators for ILP
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    // Process 32 elements per iteration (4x8) for better ILP
    for (; i + 32 <= hidden_size; i += 32) {
        // Load 4 groups of 8 fp16 values
        const float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(input + i));
        const float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(input + i + 8));
        const float16x8_t h2 = vreinterpretq_f16_u16(vld1q_u16(input + i + 16));
        const float16x8_t h3 = vreinterpretq_f16_u16(vld1q_u16(input + i + 24));

        // Convert and accumulate x*x
        float32x4_t x0_lo = vcvt_f32_f16(vget_low_f16(h0));
        float32x4_t x0_hi = vcvt_f32_f16(vget_high_f16(h0));
        acc0 = vfmaq_f32(acc0, x0_lo, x0_lo);
        acc0 = vfmaq_f32(acc0, x0_hi, x0_hi);

        float32x4_t x1_lo = vcvt_f32_f16(vget_low_f16(h1));
        float32x4_t x1_hi = vcvt_f32_f16(vget_high_f16(h1));
        acc1 = vfmaq_f32(acc1, x1_lo, x1_lo);
        acc1 = vfmaq_f32(acc1, x1_hi, x1_hi);

        float32x4_t x2_lo = vcvt_f32_f16(vget_low_f16(h2));
        float32x4_t x2_hi = vcvt_f32_f16(vget_high_f16(h2));
        acc2 = vfmaq_f32(acc2, x2_lo, x2_lo);
        acc2 = vfmaq_f32(acc2, x2_hi, x2_hi);

        float32x4_t x3_lo = vcvt_f32_f16(vget_low_f16(h3));
        float32x4_t x3_hi = vcvt_f32_f16(vget_high_f16(h3));
        acc3 = vfmaq_f32(acc3, x3_lo, x3_lo);
        acc3 = vfmaq_f32(acc3, x3_hi, x3_hi);
    }

    // Process remaining 8-element groups
    for (; i + 8 <= hidden_size; i += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(input + i));
        float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(h));
        acc0 = vfmaq_f32(acc0, x_lo, x_lo);
        acc0 = vfmaq_f32(acc0, x_hi, x_hi);
    }

    // Horizontal reduction
    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float32x4_t sum0123 = vaddq_f32(sum01, sum23);
    float sum_sq = vaddvq_f32(sum0123);

    // Handle tail (scalar) - use temp buffer approach for consistency
    if (i < hidden_size) {
        uint16_t tmp_in[8] = {};
        float tmp_f[8];
        const size_t r = hidden_size - i;
        for (size_t j = 0; j < r; ++j) tmp_in[j] = input[i + j];
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(tmp_in));
        vst1q_f32(tmp_f, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(tmp_f + 4, vcvt_f32_f16(vget_high_f16(h)));
        for (size_t j = 0; j < r; ++j) sum_sq += tmp_f[j] * tmp_f[j];
    }

    // Compute inv_rms
    const float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
    const float inv_rms = 1.0f / rms;
    const float32x4_t v_inv_rms = vdupq_n_f32(inv_rms);

    // Phase 2: Normalize and scale by weight
    i = 0;
    for (; i + 8 <= hidden_size; i += 8) {
        const float16x8_t h_in = vreinterpretq_f16_u16(vld1q_u16(input + i));
        const float16x8_t h_w = vreinterpretq_f16_u16(vld1q_u16(weight + i));

        float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(h_in));
        float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(h_in));
        float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(h_w));
        float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(h_w));

        float32x4_t out_lo = vmulq_f32(vmulq_f32(x_lo, v_inv_rms), w_lo);
        float32x4_t out_hi = vmulq_f32(vmulq_f32(x_hi, v_inv_rms), w_hi);

        const float16x8_t h_out = vcombine_f16(vcvt_f16_f32(out_lo), vcvt_f16_f32(out_hi));
        vst1q_u16(output + i, vreinterpretq_u16_f16(h_out));
    }

    // Handle tail
    if (i < hidden_size) {
        uint16_t tmp_in[8] = {}, tmp_w[8] = {}, tmp_out[8];
        float f_in[8], f_w[8], f_out[8];
        const size_t r = hidden_size - i;
        for (size_t j = 0; j < r; ++j) { tmp_in[j] = input[i + j]; tmp_w[j] = weight[i + j]; }

        const float16x8_t h_in = vreinterpretq_f16_u16(vld1q_u16(tmp_in));
        const float16x8_t h_w = vreinterpretq_f16_u16(vld1q_u16(tmp_w));
        vst1q_f32(f_in, vcvt_f32_f16(vget_low_f16(h_in)));
        vst1q_f32(f_in + 4, vcvt_f32_f16(vget_high_f16(h_in)));
        vst1q_f32(f_w, vcvt_f32_f16(vget_low_f16(h_w)));
        vst1q_f32(f_w + 4, vcvt_f32_f16(vget_high_f16(h_w)));

        for (size_t j = 0; j < 8; ++j) f_out[j] = f_in[j] * inv_rms * f_w[j];

        const float16x8_t h_out = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(f_out)), vcvt_f16_f32(vld1q_f32(f_out + 4)));
        vst1q_u16(tmp_out, vreinterpretq_u16_f16(h_out));
        for (size_t j = 0; j < r; ++j) output[i + j] = tmp_out[j];
    }
}

#endif
