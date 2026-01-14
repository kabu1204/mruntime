#include "mruntime/qwen2_ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "mruntime/dtype.h"
#include "kai_gemm.h"

namespace mruntime {

// ============================================================================
// GEMM
// ============================================================================

bool qwen2_has_kai_fp16() {
    return kai_has_fp16();
}

size_t qwen2_packed_weight_size_fp16(size_t N, size_t K) {
    // Size needed for KleidiAI packed format
    // This is an approximation; actual size comes from KleidiAI
    // For now, allocate extra space to be safe
    return (N * K + N * 16) * sizeof(uint16_t);
}

void qwen2_pack_weight_fp16(
    const uint16_t* B,
    uint16_t* packed,
    size_t N, size_t K
) {
    // Transpose B from NxK to KxN for KleidiAI
    std::vector<uint16_t> b_kxn(K * N);
    for (size_t row = 0; row < N; ++row) {
        for (size_t col = 0; col < K; ++col) {
            b_kxn[col * N + row] = B[row * K + col];
        }
    }

    // Pack using KleidiAI
    KaiPackedRhsFp16 result = kai_pack_rhs_fp16_kxn_with_zero_bias(b_kxn.data(), N, K);
    std::memcpy(packed, result.rhs_packed.data(), result.rhs_packed.size() * sizeof(uint16_t));
}

void qwen2_gemm_fp16(
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    size_t M, size_t N, size_t K,
    const uint16_t* packed_B,
    PThreadPool* pool
) {
    // KleidiAI fast path (Arm64 with FP16)
    if (qwen2_has_kai_fp16() && packed_B != nullptr) {
        kai_matmul_fp16_packed_rhs(
            M, N, K,
            A,
            K * sizeof(uint16_t),
            packed_B,
            C,
            N * sizeof(uint16_t)
        );
        return;
    }

    // Fallback: pack on-the-fly if we have KleidiAI but no pre-packed weights
    if (qwen2_has_kai_fp16()) {
        // Transpose B from NxK to KxN
        std::vector<uint16_t> b_kxn(K * N);
        for (size_t row = 0; row < N; ++row) {
            for (size_t col = 0; col < K; ++col) {
                b_kxn[col * N + row] = B[row * K + col];
            }
        }

        KaiPackedRhsFp16 packed = kai_pack_rhs_fp16_kxn_with_zero_bias(b_kxn.data(), N, K);
        kai_matmul_fp16_packed_rhs(
            M, N, K,
            A,
            K * sizeof(uint16_t),
            packed.rhs_packed.data(),
            C,
            N * sizeof(uint16_t)
        );
        return;
    }

    // Scalar fallback (no KleidiAI)
    if (M == 0 || N == 0) return;

    constexpr size_t tile_n = 128;
    const size_t n_tiles = (N + tile_n - 1) / tile_n;
    const size_t task_count = M * n_tiles;

    auto worker = [&](size_t task_id) {
        const size_t m = task_id / n_tiles;
        const size_t tile = task_id - m * n_tiles;
        const size_t n0 = tile * tile_n;
        const size_t n1 = std::min(n0 + tile_n, N);

        for (size_t n = n0; n < n1; ++n) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a = fp16_bits_to_float(A[m * K + k]);
                float b = fp16_bits_to_float(B[n * K + k]);
                acc += a * b;
            }
            C[m * N + n] = float_to_fp16_bits(acc);
        }
    };

    if (pool) {
        pool->parallelize_1d(task_count, worker);
    } else {
        for (size_t i = 0; i < task_count; ++i) worker(i);
    }
}

// ============================================================================
// RMSNorm
// ============================================================================

void qwen2_rmsnorm_fp16(
    const uint16_t* input,
    const uint16_t* weight,
    uint16_t* output,
    size_t num_tokens,
    size_t hidden_size,
    float eps,
    PThreadPool* pool
) {
    auto worker = [&](size_t t) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float v = fp16_bits_to_float(input[t * hidden_size + i]);
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        float inv_rms = 1.0f / rms;

        for (size_t i = 0; i < hidden_size; ++i) {
            float x = fp16_bits_to_float(input[t * hidden_size + i]);
            float w = fp16_bits_to_float(weight[i]);
            output[t * hidden_size + i] = float_to_fp16_bits(x * inv_rms * w);
        }
    };

    if (pool) {
        pool->parallelize_1d(num_tokens, worker);
    } else {
        for (size_t i = 0; i < num_tokens; ++i) worker(i);
    }
}

// ============================================================================
// RoPE
// ============================================================================

void qwen2_rope_fp16(
    uint16_t* Q,
    uint16_t* K,
    size_t batch,
    size_t seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t position_offset,
    float theta,
    PThreadPool* pool
) {
    const size_t half_dim = head_dim / 2;

    auto apply_rope = [&](uint16_t* data, size_t num_heads) {
        const size_t total_heads = batch * seq_len * num_heads;
        auto worker = [&](size_t idx) {
            size_t tmp = idx;
            size_t h = tmp % num_heads;
            tmp /= num_heads;
            size_t s = tmp % seq_len;
            size_t b = tmp / seq_len;

            size_t pos = position_offset + s;
            const size_t base = ((b * seq_len + s) * num_heads + h) * head_dim;

            for (size_t i = 0; i < half_dim; ++i) {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * freq;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                float x0 = fp16_bits_to_float(data[base + i]);
                float x1 = fp16_bits_to_float(data[base + i + half_dim]);

                data[base + i] = float_to_fp16_bits(x0 * cos_val - x1 * sin_val);
                data[base + i + half_dim] = float_to_fp16_bits(x0 * sin_val + x1 * cos_val);
            }
        };

        if (pool) {
            pool->parallelize_1d(total_heads, worker);
        } else {
            for (size_t i = 0; i < total_heads; ++i) worker(i);
        }
    };

    apply_rope(Q, num_q_heads);
    apply_rope(K, num_kv_heads);
}

// ============================================================================
// Flash Attention
// ============================================================================

void qwen2_flash_attention_fp16(
    const uint16_t* Q,
    const uint16_t* K,
    const uint16_t* V,
    uint16_t* O,
    size_t batch,
    size_t num_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    float scale,
    bool causal,
    PThreadPool* pool
) {
    // Strides for [batch, num_heads, seq_len, head_dim] layout
    const size_t q_stride0 = num_heads * q_len * head_dim;
    const size_t q_stride1 = q_len * head_dim;
    const size_t q_stride2 = head_dim;

    const size_t k_stride0 = num_heads * kv_len * head_dim;
    const size_t k_stride1 = kv_len * head_dim;
    const size_t k_stride2 = head_dim;

    const size_t v_stride0 = num_heads * kv_len * head_dim;
    const size_t v_stride1 = kv_len * head_dim;
    const size_t v_stride2 = head_dim;

    const size_t o_stride0 = num_heads * q_len * head_dim;
    const size_t o_stride1 = q_len * head_dim;
    const size_t o_stride2 = head_dim;

    // For causal: Q corresponds to the last q_len positions in K/V
    const size_t base_position = (kv_len >= q_len) ? (kv_len - q_len) : 0;

    const size_t task_count = batch * num_heads * q_len;
    auto worker = [&](size_t task_id) {
        size_t tmp = task_id;
        const size_t qi = tmp % q_len;
        tmp /= q_len;
        const size_t h = tmp % num_heads;
        const size_t b = tmp / num_heads;

        size_t max_k = kv_len - 1;
        if (causal && kv_len >= q_len) {
            max_k = base_position + qi;
            if (max_k >= kv_len) max_k = kv_len - 1;
        }

        static thread_local std::vector<float> scores;
        static thread_local std::vector<float> acc;
        if (scores.size() < kv_len) scores.resize(kv_len);
        if (acc.size() < head_dim) acc.resize(head_dim);
        std::fill(acc.begin(), acc.begin() + head_dim, 0.0f);

        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t ki = 0; ki <= max_k; ++ki) {
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                size_t q_idx = b * q_stride0 + h * q_stride1 + qi * q_stride2 + d;
                size_t k_idx = b * k_stride0 + h * k_stride1 + ki * k_stride2 + d;
                dot += fp16_bits_to_float(Q[q_idx]) * fp16_bits_to_float(K[k_idx]);
            }
            float s = dot * scale;
            scores[ki] = s;
            max_score = std::max(max_score, s);
        }

        float sum_exp = 0.0f;
        for (size_t ki = 0; ki <= max_k; ++ki) {
            float e = std::exp(scores[ki] - max_score);
            sum_exp += e;
            for (size_t d = 0; d < head_dim; ++d) {
                size_t v_idx = b * v_stride0 + h * v_stride1 + ki * v_stride2 + d;
                acc[d] += e * fp16_bits_to_float(V[v_idx]);
            }
        }

        float inv_sum = 1.0f / sum_exp;
        for (size_t d = 0; d < head_dim; ++d) {
            size_t o_idx = b * o_stride0 + h * o_stride1 + qi * o_stride2 + d;
            O[o_idx] = float_to_fp16_bits(acc[d] * inv_sum);
        }
    };

    if (pool) {
        pool->parallelize_1d(task_count, worker);
    } else {
        for (size_t i = 0; i < task_count; ++i) worker(i);
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

void qwen2_silu_fp16(
    const uint16_t* input,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
) {
    auto worker = [&](size_t i) {
        float x = fp16_bits_to_float(input[i]);
        float result = x / (1.0f + std::exp(-x));
        output[i] = float_to_fp16_bits(result);
    };

    if (pool) {
        pool->parallelize_1d(n, worker);
    } else {
        for (size_t i = 0; i < n; ++i) worker(i);
    }
}

void qwen2_mul_fp16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
) {
    auto worker = [&](size_t i) {
        float av = fp16_bits_to_float(a[i]);
        float bv = fp16_bits_to_float(b[i]);
        output[i] = float_to_fp16_bits(av * bv);
    };

    if (pool) {
        pool->parallelize_1d(n, worker);
    } else {
        for (size_t i = 0; i < n; ++i) worker(i);
    }
}

void qwen2_silu_mul_fp16(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
) {
    auto worker = [&](size_t i) {
        float g = fp16_bits_to_float(gate[i]);
        float u = fp16_bits_to_float(up[i]);
        float silu_g = g / (1.0f + std::exp(-g));
        output[i] = float_to_fp16_bits(silu_g * u);
    };

    if (pool) {
        pool->parallelize_1d(n, worker);
    } else {
        for (size_t i = 0; i < n; ++i) worker(i);
    }
}

void qwen2_add_fp16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
) {
    auto worker = [&](size_t i) {
        float av = fp16_bits_to_float(a[i]);
        float bv = fp16_bits_to_float(b[i]);
        output[i] = float_to_fp16_bits(av + bv);
    };

    if (pool) {
        pool->parallelize_1d(n, worker);
    } else {
        for (size_t i = 0; i < n; ++i) worker(i);
    }
}

// ============================================================================
// Embedding
// ============================================================================

void qwen2_embedding_lookup_fp16(
    const uint16_t* weight,
    const int32_t* token_ids,
    uint16_t* output,
    size_t num_tokens,
    size_t vocab_size,
    size_t hidden_size
) {
    const size_t row_bytes = hidden_size * sizeof(uint16_t);

    for (size_t t = 0; t < num_tokens; ++t) {
        int32_t token_id = token_ids[t];
        assert(token_id >= 0 && static_cast<size_t>(token_id) < vocab_size);
        (void)vocab_size;  // Used only in assert

        const uint16_t* src = weight + static_cast<size_t>(token_id) * hidden_size;
        uint16_t* dst = output + t * hidden_size;
        std::memcpy(dst, src, row_bytes);
    }
}

// ============================================================================
// Softmax
// ============================================================================

void qwen2_softmax_fp16(
    const uint16_t* input,
    uint16_t* output,
    size_t num_tokens,
    size_t vocab_size,
    PThreadPool* pool
) {
    auto worker = [&](size_t t) {
        const uint16_t* row_in = input + t * vocab_size;
        uint16_t* row_out = output + t * vocab_size;

        // Find max
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, fp16_bits_to_float(row_in[i]));
        }

        // Compute exp and sum
        static thread_local std::vector<float> tmp;
        if (tmp.size() < vocab_size) tmp.resize(vocab_size);

        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float v = fp16_bits_to_float(row_in[i]);
            float e = std::exp(v - max_val);
            tmp[i] = e;
            sum += e;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < vocab_size; ++i) {
            row_out[i] = float_to_fp16_bits(tmp[i] * inv_sum);
        }
    };

    if (pool) {
        pool->parallelize_1d(num_tokens, worker);
    } else {
        for (size_t i = 0; i < num_tokens; ++i) worker(i);
    }
}

// ============================================================================
// Copy/Layout Utilities
// ============================================================================

void qwen2_copy_to_kv_cache_fp16(
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* k_cache,
    uint16_t* v_cache,
    size_t batch,
    size_t seq_len,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t position_offset
) {
    // Input layout:  [batch, seq_len, num_kv_heads, head_dim]
    // Cache layout:  [batch, num_kv_heads, max_seq_len, head_dim]

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_kv_heads; ++h) {
                // Source index in [batch, seq_len, num_kv_heads, head_dim]
                size_t src_idx = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
                // Dest index in [batch, num_kv_heads, max_seq_len, head_dim]
                size_t dst_idx = ((b * num_kv_heads + h) * max_seq_len + (position_offset + s)) * head_dim;

                std::memcpy(k_cache + dst_idx, k + src_idx, head_dim * sizeof(uint16_t));
                std::memcpy(v_cache + dst_idx, v + src_idx, head_dim * sizeof(uint16_t));
            }
        }
    }
}

void qwen2_transpose_bshd_to_bhsd_fp16(
    const uint16_t* input,
    uint16_t* output,
    size_t B, size_t S, size_t H, size_t D,
    PThreadPool* pool
) {
    // Input:  [B, S, H, D]
    // Output: [B, H, S, D]

    const size_t task_count = B * H * S;
    auto worker = [&](size_t task_id) {
        size_t tmp = task_id;
        size_t s = tmp % S;
        tmp /= S;
        size_t h = tmp % H;
        size_t b = tmp / H;

        // Source: [b, s, h, :] in BSHD layout
        const uint16_t* src = input + ((b * S + s) * H + h) * D;
        // Dest: [b, h, s, :] in BHSD layout
        uint16_t* dst = output + ((b * H + h) * S + s) * D;

        std::memcpy(dst, src, D * sizeof(uint16_t));
    };

    if (pool) {
        pool->parallelize_1d(task_count, worker);
    } else {
        for (size_t i = 0; i < task_count; ++i) worker(i);
    }
}

}  // namespace mruntime
