#include <gtest/gtest.h>

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
