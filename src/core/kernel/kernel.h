#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include "mruntime/dtype.h"
#include "neon.h"

namespace mruntime {

struct FlashAttentionArgs {
    int batch_size;
    int num_heads;
    int q_sequence_length;
    int kv_sequence_length;
    int qk_head_size;
    int v_head_size;
    int q_block_size;
    int kv_block_size;
    float scale;
    const float* query;
    const float* key;
    const float* value;
    float* output;
    int strides_q[4];
    int strides_k[4];
    int strides_v[4];
    int strides_out[4];
};

struct FlashAttentionStridedContext {
    const FlashAttentionArgs* args = nullptr;
    size_t task_count = 0;
    float* buffer = nullptr;
    size_t buffer_size_per_task = 0;  // bytes
};

void FlashAttentionStrided(void* argptr, size_t task_id);

inline void silu_fp16_neon(const __fp16* x, __fp16* output, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;

    // Software-pipelined main loop:
    // preload next vector while computing/storing current one.
    if (n >= 16) {
        float16x8_t vx = vld1q_f16(x);
        for (; i + 16 <= n; i += 8) {
            const float16x8_t vx_next = vld1q_f16(x + i + 8);
            const float16x8_t vy = ::silu_fp16_neon(vx);
            vst1q_f16(output + i, vy);
            vx = vx_next;
        }
        const float16x8_t vy = ::silu_fp16_neon(vx);
        vst1q_f16(output + i, vy);
        i += 8;
    } else {
        for (; i + 8 <= n; i += 8) {
            const float16x8_t vx = vld1q_f16(x + i);
            const float16x8_t vy = ::silu_fp16_neon(vx);
            vst1q_f16(output + i, vy);
        }
    }

    // Tail: still route through the same vector kernel.
    if (i < n) {
        __fp16 tmp_in[8] = {};
        __fp16 tmp_out[8];
        const size_t r = n - i;
        for (size_t j = 0; j < r; ++j) {
            tmp_in[j] = x[i + j];
        }
        const float16x8_t vx = vld1q_f16(tmp_in);
        const float16x8_t vy = ::silu_fp16_neon(vx);
        vst1q_f16(tmp_out, vy);
        for (size_t j = 0; j < r; ++j) {
            output[i + j] = tmp_out[j];
        }
    }
#else
    (void)x;
    (void)output;
    (void)n;
#endif
}
inline void fast_exp_fp16_neon(const __fp16* x, __fp16* output, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;

    // Software-pipelined main loop:
    // preload next vector while computing/storing current one.
    if (n >= 16) {
        float16x8_t vx = vld1q_f16(x);
        for (; i + 16 <= n; i += 8) {
            const float16x8_t vx_next = vld1q_f16(x + i + 8);
            const float16x8_t vy = ::fast_exp_fp16_neon(vx);
            vst1q_f16(output + i, vy);
            vx = vx_next;
        }
        const float16x8_t vy = ::fast_exp_fp16_neon(vx);
        vst1q_f16(output + i, vy);
        i += 8;
    } else {
        for (; i + 8 <= n; i += 8) {
            const float16x8_t vx = vld1q_f16(x + i);
            const float16x8_t vy = ::fast_exp_fp16_neon(vx);
            vst1q_f16(output + i, vy);
        }
    }

    // Tail: still route through the same vector kernel.
    if (i < n) {
        __fp16 tmp_in[8] = {};
        __fp16 tmp_out[8];
        const size_t r = n - i;
        for (size_t j = 0; j < r; ++j) {
            tmp_in[j] = x[i + j];
        }
        const float16x8_t vx = vld1q_f16(tmp_in);
        const float16x8_t vy = ::fast_exp_fp16_neon(vx);
        vst1q_f16(tmp_out, vy);
        for (size_t j = 0; j < r; ++j) {
            output[i + j] = tmp_out[j];
        }
    }
#else
    (void)x;
    (void)output;
    (void)n;
#endif
}

inline void silu_mul_fp16_neon(const __fp16* gate, const __fp16* up, __fp16* output, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;

    // Software-pipelined main loop:
    // preload next vectors while computing/storing current one.
    if (n >= 16) {
        float16x8_t vg = vld1q_f16(gate);
        float16x8_t vu = vld1q_f16(up);
        for (; i + 16 <= n; i += 8) {
            const float16x8_t vg_next = vld1q_f16(gate + i + 8);
            const float16x8_t vu_next = vld1q_f16(up + i + 8);
            const float16x8_t vy = ::silu_mul_fp16_neon(vg, vu);
            vst1q_f16(output + i, vy);
            vg = vg_next;
            vu = vu_next;
        }
        const float16x8_t vy = ::silu_mul_fp16_neon(vg, vu);
        vst1q_f16(output + i, vy);
        i += 8;
    } else {
        for (; i + 8 <= n; i += 8) {
            const float16x8_t vg = vld1q_f16(gate + i);
            const float16x8_t vu = vld1q_f16(up + i);
            const float16x8_t vy = ::silu_mul_fp16_neon(vg, vu);
            vst1q_f16(output + i, vy);
        }
    }

    // Tail: still route through the same vector kernel.
    if (i < n) {
        __fp16 tmp_g[8] = {};
        __fp16 tmp_u[8] = {};
        __fp16 tmp_out[8];
        const size_t r = n - i;
        for (size_t j = 0; j < r; ++j) {
            tmp_g[j] = gate[i + j];
            tmp_u[j] = up[i + j];
        }
        const float16x8_t vg = vld1q_f16(tmp_g);
        const float16x8_t vu = vld1q_f16(tmp_u);
        const float16x8_t vy = ::silu_mul_fp16_neon(vg, vu);
        vst1q_f16(tmp_out, vy);
        for (size_t j = 0; j < r; ++j) {
            output[i + j] = tmp_out[j];
        }
    }
#else
    (void)gate;
    (void)up;
    (void)output;
    (void)n;
#endif
}

inline void add_fp16_neon(const __fp16* a, const __fp16* b, __fp16* output, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;

    // Software-pipelined main loop:
    // preload next vectors while computing/storing current one.
    if (n >= 16) {
        float16x8_t va = vld1q_f16(a);
        float16x8_t vb = vld1q_f16(b);
        for (; i + 16 <= n; i += 8) {
            const float16x8_t va_next = vld1q_f16(a + i + 8);
            const float16x8_t vb_next = vld1q_f16(b + i + 8);
            const float16x8_t vy = ::add_fp16_neon(va, vb);
            vst1q_f16(output + i, vy);
            va = va_next;
            vb = vb_next;
        }
        const float16x8_t vy = ::add_fp16_neon(va, vb);
        vst1q_f16(output + i, vy);
        i += 8;
    } else {
        for (; i + 8 <= n; i += 8) {
            const float16x8_t va = vld1q_f16(a + i);
            const float16x8_t vb = vld1q_f16(b + i);
            const float16x8_t vy = ::add_fp16_neon(va, vb);
            vst1q_f16(output + i, vy);
        }
    }

    // Tail: still route through the same vector kernel.
    if (i < n) {
        __fp16 tmp_a[8] = {};
        __fp16 tmp_b[8] = {};
        __fp16 tmp_out[8];
        const size_t r = n - i;
        for (size_t j = 0; j < r; ++j) {
            tmp_a[j] = a[i + j];
            tmp_b[j] = b[i + j];
        }
        const float16x8_t va = vld1q_f16(tmp_a);
        const float16x8_t vb = vld1q_f16(tmp_b);
        const float16x8_t vy = ::add_fp16_neon(va, vb);
        vst1q_f16(tmp_out, vy);
        for (size_t j = 0; j < r; ++j) {
            output[i + j] = tmp_out[j];
        }
    }
#else
    (void)a;
    (void)b;
    (void)output;
    (void)n;
#endif
}

}  // namespace mruntime