#include "mruntime/qwen2_ops.h"

#include <_types/_uint16_t.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "mruntime/dtype.h"
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include "kernel/kernel.h"
#endif
#include "kai_gemm.h"

namespace mruntime {

void fp16_bits_to_fp32(const uint16_t* src, float* dst, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    ::fp16_bits_to_fp32_neon(src, dst, n);
    return;
#endif
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fp16_bits_to_float(src[i]);
    }
}

void fp32_to_fp16_bits(const float* src, uint16_t* dst, size_t n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    ::fp32_to_fp16_bits_neon(src, dst, n);
    return;
#endif
    for (size_t i = 0; i < n; ++i) {
        dst[i] = float_to_fp16_bits(src[i]);
    }
}

// ============================================================================
// GEMM
// ============================================================================

bool qwen2_has_kai_fp16() {
    return kai_has_fp16();
}

size_t qwen2_packed_weight_size_fp16(size_t N, size_t K) {
    // Size needed for KleidiAI packed format.
    // Prefer the exact size when KleidiAI FP16 is available.
    if (qwen2_has_kai_fp16()) {
        return kai_rhs_packed_size_fp16_kxn_with_zero_bias(N, K);
    }

    // Conservative fallback (unused when KleidiAI isn't active).
    return (N * K + N * 16) * sizeof(uint16_t);
}

void qwen2_pack_weight_fp16(
    const uint16_t* B,
    uint16_t* packed,
    size_t N, size_t K
) {
    if (!qwen2_has_kai_fp16()) return;

    // Transpose B from NxK to KxN for KleidiAI
    std::vector<uint16_t> b_kxn(K * N);
    for (size_t col = 0; col < K; ++col) {
        for (size_t row = 0; row < N; ++row) {
            b_kxn[col * N + row] = B[row * K + col];
        }
    }

    // Pack using KleidiAI
    const size_t packed_size = qwen2_packed_weight_size_fp16(N, K);
    kai_pack_rhs_fp16_kxn_with_zero_bias(b_kxn.data(), N, K, packed, packed_size);
}

void qwen2_gemm_fp16(
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    size_t M, size_t N, size_t K,
    const uint16_t* packed_B,
    PThreadPool* pool
) {
    // if (pool == nullptr && qwen2_has_kai_fp16() && packed_B != nullptr) {
    //     kai_matmul_fp16_packed_rhs(
    //         M, N, K,
    //         A,
    //         K * sizeof(uint16_t),
    //         packed_B,
    //         C,
    //         N * sizeof(uint16_t)
    //     );
    //     return;
    // }

    // KleidiAI multi-threaded fast path (Arm64 with FP16)
    if (qwen2_has_kai_fp16() && packed_B != nullptr) {
        const size_t n_step = kai_get_n_step_fp16();

        const size_t n_tiles = (N + n_step - 1) / n_step;
        if (pool == nullptr || pool->threads_count() <= 1 || n_tiles <= 1) {
            kai_matmul_fp16_packed_rhs(
                M, N, K,
                A, K * sizeof(uint16_t),
                packed_B,
                C, N * sizeof(uint16_t)
            );
            return;
        }

        // Tune this manually if false sharing for writing outputs is observed.
        constexpr size_t tiles_per_task = 2;

        const size_t task_count = (n_tiles + tiles_per_task - 1) / tiles_per_task;
        auto task_worker = [&](size_t task_id) {
            const size_t tile_begin = task_id * tiles_per_task;
            const size_t tile_end = std::min(n_tiles, tile_begin + tiles_per_task);

            const size_t n_start = tile_begin * n_step;
            const size_t n_end = std::min(N, tile_end * n_step);
            if (n_end <= n_start) return;

            kai_matmul_fp16_packed_rhs_stripe(
                n_start, n_end - n_start,
                M, N, K,
                A, K * sizeof(uint16_t),
                packed_B,
                C, N * sizeof(uint16_t)
            );
        };

        pool->parallelize_1d(task_count, task_worker);
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
    const float* rope_cos_sin,
    size_t rope_max_seq_len,
    PThreadPool* pool
) {
    assert(Q != nullptr);
    assert(K != nullptr);
    assert(rope_cos_sin != nullptr);
    assert(batch > 0);
    assert(seq_len > 0);
    assert(num_q_heads > 0);
    assert(num_kv_heads > 0);
    assert(head_dim > 0);
    assert(head_dim % 2 == 0);
    assert(rope_max_seq_len > 0);
    if (position_offset + seq_len > rope_max_seq_len) {
        assert(false && "qwen2_rope_fp16: position_offset + seq_len exceeds rope_max_seq_len");
        return;
    }

    const size_t half_dim = head_dim / 2;
    const size_t floats_per_pos = half_dim * 2;

    // Parallelize over (batch, seq_len). Each task handles all heads at one position.
    const size_t task_count = batch * seq_len;
    auto worker = [&](size_t task_id) {
        const size_t s = task_id % seq_len;
        const size_t b = task_id / seq_len;

        const size_t pos = position_offset + s;
        const float* cs = rope_cos_sin + pos * floats_per_pos;

        const size_t q_token_base = ((b * seq_len + s) * num_q_heads) * head_dim;
        const size_t k_token_base = ((b * seq_len + s) * num_kv_heads) * head_dim;

        auto apply_rope_heads = [&](uint16_t* data, size_t token_base, size_t num_heads) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t base = token_base + h * head_dim;
                size_t i = 0;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                i = ::rope_fp16_bits_inplace_neon(data + base, cs, half_dim);
#endif

                for (; i < half_dim; ++i) {
                    const float cos_val = cs[i * 2];
                    const float sin_val = cs[i * 2 + 1];
                    const float x0 = fp16_bits_to_float(data[base + i]);
                    const float x1 = fp16_bits_to_float(data[base + i + half_dim]);
                    data[base + i] = float_to_fp16_bits(x0 * cos_val - x1 * sin_val);
                    data[base + i + half_dim] = float_to_fp16_bits(x0 * sin_val + x1 * cos_val);
                }
            }
        };

        apply_rope_heads(Q, q_token_base, num_q_heads);
        apply_rope_heads(K, k_token_base, num_kv_heads);
    };

    if (pool) {
        pool->parallelize_1d(task_count, worker);
    } else {
        for (size_t i = 0; i < task_count; ++i) worker(i);
    }
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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    auto worker_range = [&](size_t start, size_t len) {
        // Runtime FP16 is stored as raw IEEE754 binary16 bits (uint16_t).
        // On Arm with FP16 vector arithmetic, this matches __fp16 in-memory
        // representation, so we can reinterpret and use the NEON wrapper.
        const __fp16* in = reinterpret_cast<const __fp16*>(input + start);
        __fp16* out = reinterpret_cast<__fp16*>(output + start);
        silu_fp16_neon(in, out, len);
    };

    if (pool) {
        constexpr size_t kGrain = 4096;  // elements
        const size_t task_count = (n + kGrain - 1) / kGrain;
        auto worker = [&](size_t task_id) {
            const size_t start = task_id * kGrain;
            const size_t len = std::min(kGrain, n - start);
            worker_range(start, len);
        };
        pool->parallelize_1d(task_count, worker);
    } else {
        worker_range(0, n);
    }
    return;
#endif

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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    auto worker_range = [&](size_t start, size_t len) {
        const __fp16* g = reinterpret_cast<const __fp16*>(gate + start);
        const __fp16* u = reinterpret_cast<const __fp16*>(up + start);
        __fp16* out = reinterpret_cast<__fp16*>(output + start);
        silu_mul_fp16_neon(g, u, out, len);
    };

    if (pool) {
        constexpr size_t kGrain = 4096;  // elements
        const size_t task_count = (n + kGrain - 1) / kGrain;
        auto worker = [&](size_t task_id) {
            const size_t start = task_id * kGrain;
            const size_t len = std::min(kGrain, n - start);
            worker_range(start, len);
        };
        pool->parallelize_1d(task_count, worker);
    } else {
        worker_range(0, n);
    }
    return;
#endif

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

void qwen2_silu_mul_interleaved_fp16(
    const uint16_t* gate_up,
    uint16_t* output,
    size_t num_tokens,
    size_t intermediate_size,
    PThreadPool* pool
) {
    const size_t row_stride = intermediate_size * 2;

    auto worker_token = [&](size_t t) {
        const uint16_t* gate = gate_up + t * row_stride;
        const uint16_t* up = gate + intermediate_size;
        uint16_t* out = output + t * intermediate_size;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        const __fp16* g = reinterpret_cast<const __fp16*>(gate);
        const __fp16* u = reinterpret_cast<const __fp16*>(up);
        __fp16* o = reinterpret_cast<__fp16*>(out);
        silu_mul_fp16_neon(g, u, o, intermediate_size);
        return;
#endif

        for (size_t i = 0; i < intermediate_size; ++i) {
            float g = fp16_bits_to_float(gate[i]);
            float u = fp16_bits_to_float(up[i]);
            float silu_g = g / (1.0f + std::exp(-g));
            out[i] = float_to_fp16_bits(silu_g * u);
        }
    };

    if (pool) {
        pool->parallelize_1d(num_tokens, worker_token);
    } else {
        for (size_t t = 0; t < num_tokens; ++t) worker_token(t);
    }
}

void qwen2_add_fp16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    auto worker_range = [&](size_t start, size_t len) {
        const __fp16* in_a = reinterpret_cast<const __fp16*>(a + start);
        const __fp16* in_b = reinterpret_cast<const __fp16*>(b + start);
        __fp16* out = reinterpret_cast<__fp16*>(output + start);
        add_fp16_neon(in_a, in_b, out, len);
    };

    if (pool) {
        constexpr size_t kGrain = 4096;  // elements
        const size_t task_count = (n + kGrain - 1) / kGrain;
        auto worker = [&](size_t task_id) {
            const size_t start = task_id * kGrain;
            const size_t len = std::min(kGrain, n - start);
            worker_range(start, len);
        };
        pool->parallelize_1d(task_count, worker);
    } else {
        worker_range(0, n);
    }
    return;
#endif

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

    if (S == 1) {
        std::memcpy(output, input, 
        B*H*D*sizeof(uint16_t));
        return;
    }

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
