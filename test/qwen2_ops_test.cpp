#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "mruntime/arena.h"
#include "mruntime/dtype.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/qwen2_generate.h"

using namespace mruntime;

TEST(ArenaTest, BasicAllocation) {
    Arena arena = create_arena(1024);
    ASSERT_NE(arena.base, nullptr);
    EXPECT_EQ(arena.capacity, 1024);
    EXPECT_EQ(arena.offset, 0);

    void* p1 = arena.alloc(64);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(p1, arena.base);

    void* p2 = arena.alloc(64);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);

    destroy_arena(arena);
    EXPECT_EQ(arena.base, nullptr);
}

TEST(ArenaTest, Reset) {
    Arena arena = create_arena(1024);

    arena.alloc(256);
    EXPECT_GT(arena.offset, 0);

    arena.reset();
    EXPECT_EQ(arena.offset, 0);

    destroy_arena(arena);
}

TEST(ArenaTest, Watermark) {
    Arena arena = create_arena(1024);

    arena.alloc(64);
    size_t mark = arena.watermark();
    EXPECT_GT(mark, 0);

    arena.alloc(128);
    EXPECT_GT(arena.offset, mark);

    arena.reset_to(mark);
    EXPECT_EQ(arena.offset, mark);

    destroy_arena(arena);
}

TEST(ArenaTest, Qwen2Arenas) {
    Qwen2Arenas arenas = create_qwen2_arenas(1024, 2048, 512);

    EXPECT_NE(arenas.weights.base, nullptr);
    EXPECT_NE(arenas.kv_cache.base, nullptr);
    EXPECT_NE(arenas.scratch.base, nullptr);

    EXPECT_GE(arenas.weights.capacity, 1024);
    EXPECT_GE(arenas.kv_cache.capacity, 2048);
    EXPECT_GE(arenas.scratch.capacity, 512);

    destroy_qwen2_arenas(arenas);
}

TEST(Qwen2OpsTest, RmsNorm) {
    const size_t hidden_size = 4;
    const size_t num_tokens = 2;

    // Create input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    std::vector<uint16_t> input(num_tokens * hidden_size);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = float_to_fp16_bits(static_cast<float>(i + 1));
    }

    // Weight: all 1.0
    std::vector<uint16_t> weight(hidden_size);
    for (size_t i = 0; i < hidden_size; ++i) {
        weight[i] = float_to_fp16_bits(1.0f);
    }

    std::vector<uint16_t> output(num_tokens * hidden_size);

    qwen2_rmsnorm_fp16(
        input.data(),
        weight.data(),
        output.data(),
        num_tokens,
        hidden_size,
        1e-6f,
        nullptr  // No thread pool
    );

    // Check output is normalized
    for (size_t t = 0; t < num_tokens; ++t) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float v = fp16_bits_to_float(output[t * hidden_size + i]);
            sum_sq += v * v;
        }
        // RMS should be approximately 1.0 after normalization
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size));
        EXPECT_NEAR(rms, 1.0f, 0.1f);
    }
}

namespace {

void rope_reference_fp16(
    uint16_t* data,
    size_t batch,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t position_offset,
    float theta
) {
    const size_t half_dim = head_dim / 2;

    std::vector<float> inv_freq(half_dim);
    const float two_over_head_dim = 2.0f / static_cast<float>(head_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) * two_over_head_dim);
    }

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t pos = position_offset + s;
            const float pos_f = static_cast<float>(pos);

            const size_t token_base = ((b * seq_len + s) * num_heads) * head_dim;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t base = token_base + h * head_dim;
                for (size_t i = 0; i < half_dim; ++i) {
                    const float angle = pos_f * inv_freq[i];
                    const float cos_val = std::cos(angle);
                    const float sin_val = std::sin(angle);

                    const float x0 = fp16_bits_to_float(data[base + i]);
                    const float x1 = fp16_bits_to_float(data[base + i + half_dim]);

                    data[base + i] = float_to_fp16_bits(x0 * cos_val - x1 * sin_val);
                    data[base + i + half_dim] = float_to_fp16_bits(x0 * sin_val + x1 * cos_val);
                }
            }
        }
    }
}

void flash_attention_gqa_reference_fp32(
    const uint16_t* Q,
    const uint16_t* K,
    const uint16_t* V,
    float* O,
    size_t batch,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t kv_stride,
    size_t head_dim,
    float scale,
    bool causal
) {
    ASSERT_NE(Q, nullptr);
    ASSERT_NE(K, nullptr);
    ASSERT_NE(V, nullptr);
    ASSERT_NE(O, nullptr);
    ASSERT_GT(batch, 0u);
    ASSERT_GT(num_q_heads, 0u);
    ASSERT_GT(num_kv_heads, 0u);
    ASSERT_GT(q_len, 0u);
    ASSERT_GT(kv_len, 0u);
    ASSERT_GE(kv_stride, kv_len);
    ASSERT_GT(head_dim, 0u);
    ASSERT_EQ(num_q_heads % num_kv_heads, 0u);

    const size_t heads_per_kv = num_q_heads / num_kv_heads;

    const size_t q_stride0 = num_q_heads * q_len * head_dim;
    const size_t q_stride1 = q_len * head_dim;
    const size_t q_stride2 = head_dim;

    const size_t k_stride0 = num_kv_heads * kv_stride * head_dim;
    const size_t k_stride1 = kv_stride * head_dim;
    const size_t k_stride2 = head_dim;

    const size_t v_stride0 = num_kv_heads * kv_stride * head_dim;
    const size_t v_stride1 = kv_stride * head_dim;
    const size_t v_stride2 = head_dim;

    const size_t o_stride0 = num_q_heads * q_len * head_dim;
    const size_t o_stride1 = q_len * head_dim;
    const size_t o_stride2 = head_dim;

    const size_t base_position = (kv_len >= q_len) ? (kv_len - q_len) : 0;

    std::vector<float> scores(kv_len);
    std::vector<float> acc(head_dim);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t q_h = 0; q_h < num_q_heads; ++q_h) {
            const size_t kv_h = q_h / heads_per_kv;
            for (size_t qi = 0; qi < q_len; ++qi) {
                size_t max_k = kv_len - 1;
                if (causal && kv_len >= q_len) {
                    max_k = base_position + qi;
                    if (max_k >= kv_len) max_k = kv_len - 1;
                }

                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t ki = 0; ki <= max_k; ++ki) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        const size_t q_idx = b * q_stride0 + q_h * q_stride1 + qi * q_stride2 + d;
                        const size_t k_idx = b * k_stride0 + kv_h * k_stride1 + ki * k_stride2 + d;
                        dot += fp16_bits_to_float(Q[q_idx]) * fp16_bits_to_float(K[k_idx]);
                    }
                    const float s = dot * scale;
                    scores[ki] = s;
                    max_score = std::max(max_score, s);
                }

                float sum_exp = 0.0f;
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (size_t ki = 0; ki <= max_k; ++ki) {
                    const float w = std::exp(scores[ki] - max_score);
                    sum_exp += w;
                    for (size_t d = 0; d < head_dim; ++d) {
                        const size_t v_idx = b * v_stride0 + kv_h * v_stride1 + ki * v_stride2 + d;
                        acc[d] += w * fp16_bits_to_float(V[v_idx]);
                    }
                }

                const float inv_sum = 1.0f / sum_exp;
                for (size_t d = 0; d < head_dim; ++d) {
                    const size_t o_idx = b * o_stride0 + q_h * o_stride1 + qi * o_stride2 + d;
                    O[o_idx] = acc[d] * inv_sum;
                }
            }
        }
    }
}

}  // namespace

TEST(Qwen2OpsTest, Rope) {
    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t num_q_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 4;
    const size_t position_offset = 0;
    const float theta = 10000.0f;

    Arena arena = create_arena(4096);
    const Qwen2RopeCache rope_cache = qwen2_init_rope_cache(head_dim, /*max_seq_len=*/seq_len, theta, arena);

    std::vector<uint16_t> Q(batch * seq_len * num_q_heads * head_dim);
    std::vector<uint16_t> K(batch * seq_len * num_kv_heads * head_dim);

    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = float_to_fp16_bits(static_cast<float>(i) * 0.125f - 1.0f);
    }
    for (size_t i = 0; i < K.size(); ++i) {
        K[i] = float_to_fp16_bits(static_cast<float>(i) * 0.25f + 0.5f);
    }

    const std::vector<uint16_t> Q_input = Q;
    const std::vector<uint16_t> K_input = K;

    std::vector<uint16_t> Q_expected = Q;
    std::vector<uint16_t> K_expected = K;

    rope_reference_fp16(Q_expected.data(), batch, seq_len, num_q_heads, head_dim, position_offset, theta);
    rope_reference_fp16(K_expected.data(), batch, seq_len, num_kv_heads, head_dim, position_offset, theta);

    qwen2_rope_fp16(
        Q.data(),
        K.data(),
        batch,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        position_offset,
        rope_cache.cos_sin,
        rope_cache.max_seq_len,
        nullptr
    );

    // pos=0 should be identity.
    for (size_t i = 0; i < num_q_heads * head_dim; ++i) {
        EXPECT_EQ(Q[i], Q_input[i]) << "Q i=" << i;
    }
    for (size_t i = 0; i < num_kv_heads * head_dim; ++i) {
        EXPECT_EQ(K[i], K_input[i]) << "K i=" << i;
    }

    // Full compare vs reference for the entire tensor.
    constexpr float kTol = 5e-3f;
    for (size_t i = 0; i < Q.size(); ++i) {
        EXPECT_NEAR(fp16_bits_to_float(Q[i]), fp16_bits_to_float(Q_expected[i]), kTol) << "Q i=" << i;
    }
    for (size_t i = 0; i < K.size(); ++i) {
        EXPECT_NEAR(fp16_bits_to_float(K[i]), fp16_bits_to_float(K_expected[i]), kTol) << "K i=" << i;
    }

    destroy_arena(arena);
}

TEST(Qwen2OpsTest, FlashAttentionGqaDecodeCausal) {
    const size_t batch = 1;
    const size_t num_q_heads = 4;
    const size_t num_kv_heads = 2;
    const size_t q_len = 1;
    const size_t kv_len = 7;
    const size_t kv_stride = 16;
    const size_t head_dim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<uint16_t> Q(batch * num_q_heads * q_len * head_dim);
    std::vector<uint16_t> K(batch * num_kv_heads * kv_stride * head_dim);
    std::vector<uint16_t> V(batch * num_kv_heads * kv_stride * head_dim);

    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = float_to_fp16_bits(static_cast<float>(i) * 0.013f - 0.2f);
    }

    for (size_t h = 0; h < num_kv_heads; ++h) {
        for (size_t t = 0; t < kv_stride; ++t) {
            for (size_t d = 0; d < head_dim; ++d) {
                const size_t idx = (h * kv_stride + t) * head_dim + d;
                if (t < kv_len) {
                    K[idx] = float_to_fp16_bits(0.05f * static_cast<float>(h + 1) +
                                                0.01f * static_cast<float>(t + 1) +
                                                0.001f * static_cast<float>(d));
                    V[idx] = float_to_fp16_bits(-0.03f * static_cast<float>(h + 1) +
                                                0.02f * static_cast<float>(t + 1) -
                                                0.001f * static_cast<float>(d));
                } else {
                    // Padding: should never be read.
                    K[idx] = float_to_fp16_bits(1000.0f);
                    V[idx] = float_to_fp16_bits(-1000.0f);
                }
            }
        }
    }

    std::vector<uint16_t> O(batch * num_q_heads * q_len * head_dim);
    std::vector<float> O_ref(batch * num_q_heads * q_len * head_dim);

    flash_attention_gqa_reference_fp32(
        Q.data(),
        K.data(),
        V.data(),
        O_ref.data(),
        batch,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        /*causal=*/true
    );

    qwen2_flash_attention_gqa_fp16(
        Q.data(),
        K.data(),
        V.data(),
        O.data(),
        batch,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        /*causal=*/true,
        /*pool=*/nullptr
    );

    constexpr float kTol = 5e-2f;
    for (size_t i = 0; i < O.size(); ++i) {
        const float got = fp16_bits_to_float(O[i]);
        ASSERT_TRUE(std::isfinite(got)) << "O[" << i << "] is not finite";
        EXPECT_NEAR(got, O_ref[i], kTol) << "i=" << i;
    }
}

TEST(Qwen2OpsTest, FlashAttentionGqaPrefillCausal) {
    const size_t batch = 1;
    const size_t num_q_heads = 4;
    const size_t num_kv_heads = 2;
    const size_t q_len = 3;
    const size_t kv_len = 9;
    const size_t kv_stride = 12;
    const size_t head_dim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<uint16_t> Q(batch * num_q_heads * q_len * head_dim);
    std::vector<uint16_t> K(batch * num_kv_heads * kv_stride * head_dim);
    std::vector<uint16_t> V(batch * num_kv_heads * kv_stride * head_dim);

    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = float_to_fp16_bits(static_cast<float>(i) * 0.007f - 0.15f);
    }

    for (size_t h = 0; h < num_kv_heads; ++h) {
        for (size_t t = 0; t < kv_stride; ++t) {
            for (size_t d = 0; d < head_dim; ++d) {
                const size_t idx = (h * kv_stride + t) * head_dim + d;
                if (t < kv_len) {
                    K[idx] = float_to_fp16_bits(0.03f * static_cast<float>(h + 1) +
                                                0.01f * static_cast<float>(t) +
                                                0.002f * static_cast<float>(d));
                    V[idx] = float_to_fp16_bits(0.02f * static_cast<float>(h + 1) -
                                                0.015f * static_cast<float>(t) +
                                                0.001f * static_cast<float>(d));
                } else {
                    K[idx] = float_to_fp16_bits(500.0f);
                    V[idx] = float_to_fp16_bits(-500.0f);
                }
            }
        }
    }

    std::vector<uint16_t> O(batch * num_q_heads * q_len * head_dim);
    std::vector<float> O_ref(batch * num_q_heads * q_len * head_dim);

    flash_attention_gqa_reference_fp32(
        Q.data(),
        K.data(),
        V.data(),
        O_ref.data(),
        batch,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        /*causal=*/true
    );

    qwen2_flash_attention_gqa_fp16(
        Q.data(),
        K.data(),
        V.data(),
        O.data(),
        batch,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        /*causal=*/true,
        /*pool=*/nullptr
    );

    constexpr float kTol = 5e-2f;
    for (size_t i = 0; i < O.size(); ++i) {
        const float got = fp16_bits_to_float(O[i]);
        ASSERT_TRUE(std::isfinite(got)) << "O[" << i << "] is not finite";
        EXPECT_NEAR(got, O_ref[i], kTol) << "i=" << i;
    }
}

TEST(Qwen2OpsTest, FlashAttentionGqaNonCausal) {
    const size_t batch = 1;
    const size_t num_q_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t q_len = 2;
    const size_t kv_len = 5;
    const size_t kv_stride = 7;
    const size_t head_dim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<uint16_t> Q(batch * num_q_heads * q_len * head_dim);
    std::vector<uint16_t> K(batch * num_kv_heads * kv_stride * head_dim);
    std::vector<uint16_t> V(batch * num_kv_heads * kv_stride * head_dim);

    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = float_to_fp16_bits(static_cast<float>(i) * 0.011f - 0.25f);
    }
    for (size_t t = 0; t < kv_stride; ++t) {
        for (size_t d = 0; d < head_dim; ++d) {
            const size_t idx = t * head_dim + d;
            if (t < kv_len) {
                K[idx] = float_to_fp16_bits(0.02f * static_cast<float>(t) - 0.01f * static_cast<float>(d));
                V[idx] = float_to_fp16_bits(0.03f * static_cast<float>(t) + 0.005f * static_cast<float>(d));
            } else {
                K[idx] = float_to_fp16_bits(250.0f);
                V[idx] = float_to_fp16_bits(-250.0f);
            }
        }
    }

    std::vector<uint16_t> O(batch * num_q_heads * q_len * head_dim);
    std::vector<float> O_ref(batch * num_q_heads * q_len * head_dim);

    flash_attention_gqa_reference_fp32(
        Q.data(),
        K.data(),
        V.data(),
        O_ref.data(),
        batch,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        /*causal=*/false
    );

    qwen2_flash_attention_gqa_fp16(
        Q.data(),
        K.data(),
        V.data(),
        O.data(),
        batch,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        /*causal=*/false,
        /*pool=*/nullptr
    );

    constexpr float kTol = 5e-2f;
    for (size_t i = 0; i < O.size(); ++i) {
        const float got = fp16_bits_to_float(O[i]);
        ASSERT_TRUE(std::isfinite(got)) << "O[" << i << "] is not finite";
        EXPECT_NEAR(got, O_ref[i], kTol) << "i=" << i;
    }
}

TEST(Qwen2OpsTest, SiluMul) {
    const size_t n = 4;

    std::vector<uint16_t> gate(n);
    std::vector<uint16_t> up(n);
    std::vector<uint16_t> output(n);

    // gate = [0, 1, 2, 3]
    // up = [1, 1, 1, 1]
    for (size_t i = 0; i < n; ++i) {
        gate[i] = float_to_fp16_bits(static_cast<float>(i));
        up[i] = float_to_fp16_bits(1.0f);
    }

    qwen2_silu_mul_fp16(gate.data(), up.data(), output.data(), n, nullptr);

    // SiLU(0) * 1 = 0
    EXPECT_NEAR(fp16_bits_to_float(output[0]), 0.0f, 0.01f);

    // SiLU(1) * 1 ≈ 0.731
    EXPECT_NEAR(fp16_bits_to_float(output[1]), 0.731f, 0.1f);
}

TEST(Qwen2OpsTest, EmbeddingLookup) {
    const size_t vocab_size = 10;
    const size_t hidden_size = 4;

    // Create embedding table
    std::vector<uint16_t> weight(vocab_size * hidden_size);
    for (size_t v = 0; v < vocab_size; ++v) {
        for (size_t h = 0; h < hidden_size; ++h) {
            // Each row has values [v*10, v*10+1, v*10+2, v*10+3]
            weight[v * hidden_size + h] = float_to_fp16_bits(static_cast<float>(v * 10 + h));
        }
    }

    // Look up tokens [2, 5]
    std::vector<int32_t> tokens = {2, 5};
    std::vector<uint16_t> output(tokens.size() * hidden_size);

    qwen2_embedding_lookup_fp16(
        weight.data(),
        tokens.data(),
        output.data(),
        tokens.size(),
        vocab_size,
        hidden_size
    );

    // Check output[0] = [20, 21, 22, 23]
    EXPECT_NEAR(fp16_bits_to_float(output[0]), 20.0f, 0.01f);
    EXPECT_NEAR(fp16_bits_to_float(output[1]), 21.0f, 0.01f);
    EXPECT_NEAR(fp16_bits_to_float(output[2]), 22.0f, 0.01f);
    EXPECT_NEAR(fp16_bits_to_float(output[3]), 23.0f, 0.01f);

    // Check output[1] = [50, 51, 52, 53]
    EXPECT_NEAR(fp16_bits_to_float(output[4]), 50.0f, 0.01f);
    EXPECT_NEAR(fp16_bits_to_float(output[5]), 51.0f, 0.01f);
    EXPECT_NEAR(fp16_bits_to_float(output[6]), 52.0f, 0.01f);
    EXPECT_NEAR(fp16_bits_to_float(output[7]), 53.0f, 0.01f);
}

TEST(Qwen2OpsTest, Argmax) {
    const size_t vocab_size = 5;

    std::vector<uint16_t> logits(vocab_size);
    logits[0] = float_to_fp16_bits(1.0f);
    logits[1] = float_to_fp16_bits(3.0f);  // max
    logits[2] = float_to_fp16_bits(2.0f);
    logits[3] = float_to_fp16_bits(-1.0f);
    logits[4] = float_to_fp16_bits(0.0f);

    int32_t result = qwen2_argmax_fp16(logits.data(), vocab_size);
    EXPECT_EQ(result, 1);
}

TEST(DTypeTest, Fp32ToFp16BitsBulkMatchesScalar) {
    const std::vector<float> src = {
        0.0f,
        -0.0f,
        1.0f,
        -1.0f,
        0.001f,
        123.456f,
        -789.0f,
        65504.0f,   // max finite FP16
        70000.0f,   // overflow -> +inf
        -70000.0f,  // overflow -> -inf
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        42.0f,
        -42.0f,
        3.1415926f,
        -2.7182818f,
        16.0f,  // tail coverage (17 elems)
    };

    std::vector<uint16_t> dst(src.size());
    fp32_to_fp16_bits(src.data(), dst.data(), dst.size());

    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_EQ(dst[i], float_to_fp16_bits(src[i])) << "i=" << i;
    }
}

TEST(DTypeTest, Fp16BitsToFp32BulkMatchesScalar) {
    const std::vector<uint16_t> src = {
        0x0000u,  // +0
        0x8000u,  // -0
        0x3C00u,  // +1
        0xBC00u,  // -1
        0x7BFFu,  // max finite
        0x0400u,  // min normal
        0x7C00u,  // +inf
        0xFC00u,  // -inf
        0x7E00u,  // NaN
        0x3555u,
        0xB555u,
        0x2A66u,
        0xAA66u,  // tail coverage (13 elems)
    };

    std::vector<float> dst(src.size());
    fp16_bits_to_fp32(src.data(), dst.data(), dst.size());

    for (size_t i = 0; i < src.size(); ++i) {
        const float expected = fp16_bits_to_float(src[i]);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(dst[i])) << "i=" << i;
            continue;
        }

        uint32_t expected_bits = 0;
        uint32_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
        std::memcpy(&actual_bits, &dst[i], sizeof(actual_bits));
        EXPECT_EQ(actual_bits, expected_bits) << "i=" << i;
    }
}
