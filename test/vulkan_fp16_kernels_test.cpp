#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mruntime/dtype.h"
#include "mruntime/qwen2_ops.h"
#include "vk_buffer_arena.h"
#include "vk_context.h"
#include "vk_fp16_ops.h"
#include "vk_helpers.h"
#include "vk_kernel_runtime.h"

namespace {

struct TestContext {
    mruntime::vulkan::VkContext context;
    mruntime::vulkan::VkKernelRuntime runtime;
    mruntime::vulkan::VkFp16Ops fp16_ops;
    VkDeviceSize alignment;

    static TestContext Create() {
        TestContext tc;
        tc.context = mruntime::vulkan::VkContext::Create();
        tc.alignment = std::max<VkDeviceSize>(64, tc.context.min_storage_buffer_offset_alignment());
        tc.runtime = mruntime::vulkan::VkKernelRuntime::Create(tc.context);
        tc.fp16_ops = mruntime::vulkan::VkFp16Ops::Create(&tc.runtime);
        return tc;
    }
};

mruntime::vulkan::VkBufferArena make_arena(
    const TestContext& tc, VkDeviceSize capacity) {
    mruntime::vulkan::VkBufferArenaCreateInfo info;
    info.capacity_bytes = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    info.default_alignment = tc.alignment;
    return mruntime::vulkan::VkBufferArena::Create(
        tc.context.physical_device(), tc.context.device(), info);
}

void check_close(const char* test_name, const uint16_t* got_fp16, const float* expected,
                  uint32_t n, float tolerance) {
    for (uint32_t i = 0; i < n; ++i) {
        float got = mruntime::fp16_bits_to_float(got_fp16[i]);
        float diff = std::fabs(got - expected[i]);
        if (!std::isfinite(got) || diff > tolerance) {
            throw std::runtime_error(
                std::string(test_name) + " mismatch at i=" + std::to_string(i) +
                ": got=" + std::to_string(got) +
                ", expected=" + std::to_string(expected[i]) +
                ", diff=" + std::to_string(diff));
        }
    }
}

void check_exact_fp16(const char* test_name, const uint16_t* got, const uint16_t* expected,
                      uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        if (got[i] != expected[i]) {
            throw std::runtime_error(
                std::string(test_name) + " mismatch at i=" + std::to_string(i) +
                ": got=0x" + std::to_string(got[i]) +
                ", expected=0x" + std::to_string(expected[i]));
        }
    }
}

// ---- silu_mul_interleaved ----

void test_silu_mul_interleaved(TestContext& tc) {
    const uint32_t num_tokens = 4;
    const uint32_t intermediate_size = 512;
    const uint32_t gate_up_row = 2 * intermediate_size;
    const uint32_t gate_up_count = num_tokens * gate_up_row;
    const uint32_t out_count = num_tokens * intermediate_size;

    const VkDeviceSize gate_up_bytes = gate_up_count * sizeof(uint16_t);
    const VkDeviceSize out_bytes = out_count * sizeof(uint16_t);

    auto arena = make_arena(tc, gate_up_bytes + out_bytes + 2 * tc.alignment);

    const VkDeviceSize gate_up_offset = arena.alloc(gate_up_bytes);
    const VkDeviceSize out_offset = arena.alloc(out_bytes);

    uint16_t* gate_up_data = arena.host_ptr<uint16_t>(gate_up_offset);
    uint16_t* out_data = arena.host_ptr<uint16_t>(out_offset);

    // Fill gate_up with deterministic values.
    for (uint32_t i = 0; i < gate_up_count; ++i) {
        float val = -2.0f + 0.004f * static_cast<float>(i % 1024);
        gate_up_data[i] = mruntime::float_to_fp16_bits(val);
    }

    // CPU reference.
    std::vector<float> expected(out_count);
    for (uint32_t t = 0; t < num_tokens; ++t) {
        for (uint32_t j = 0; j < intermediate_size; ++j) {
            float gate = mruntime::fp16_bits_to_float(
                gate_up_data[t * gate_up_row + j]);
            float up = mruntime::fp16_bits_to_float(
                gate_up_data[t * gate_up_row + intermediate_size + j]);
            float silu = gate / (1.0f + std::exp(-gate));
            expected[t * intermediate_size + j] = silu * up;
        }
    }

    std::memset(out_data, 0, out_bytes);
    tc.fp16_ops.silu_mul_interleaved(
        arena.descriptor(gate_up_offset, gate_up_bytes),
        arena.descriptor(out_offset, out_bytes),
        intermediate_size, num_tokens);

    check_close("silu_mul_interleaved", out_data, expected.data(), out_count, 1e-2f);
    std::cout << "  silu_mul_interleaved PASSED\n";
}

// ---- rmsnorm ----

void test_rmsnorm(TestContext& tc) {
    struct RmsnormCase {
        uint32_t hidden_size;
        uint32_t num_tokens;
    };
    const std::vector<RmsnormCase> cases = {
        {256, 4},
        {896, 4},
        {896, 64},
    };

    for (const RmsnormCase& tc_case : cases) {
        const uint32_t hidden_size = tc_case.hidden_size;
        const uint32_t num_tokens = tc_case.num_tokens;
        const float eps = 1e-6f;
        const uint32_t input_count = num_tokens * hidden_size;

        const VkDeviceSize input_bytes = input_count * sizeof(uint16_t);
        const VkDeviceSize weight_bytes = hidden_size * sizeof(uint16_t);
        const VkDeviceSize out_bytes = input_count * sizeof(uint16_t);

        auto arena = make_arena(tc, input_bytes + weight_bytes + out_bytes + 3 * tc.alignment);

        const VkDeviceSize input_offset = arena.alloc(input_bytes);
        const VkDeviceSize weight_offset = arena.alloc(weight_bytes);
        const VkDeviceSize out_offset = arena.alloc(out_bytes);

        uint16_t* input_data = arena.host_ptr<uint16_t>(input_offset);
        uint16_t* weight_data = arena.host_ptr<uint16_t>(weight_offset);
        uint16_t* out_data = arena.host_ptr<uint16_t>(out_offset);

        for (uint32_t i = 0; i < input_count; ++i) {
            float val = -1.0f + 0.002f * static_cast<float>(i % 1000);
            input_data[i] = mruntime::float_to_fp16_bits(val);
        }
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float val = 0.5f + 0.001f * static_cast<float>(i);
            weight_data[i] = mruntime::float_to_fp16_bits(val);
        }

        // CPU reference.
        std::vector<float> expected(input_count);
        for (uint32_t t = 0; t < num_tokens; ++t) {
            float sum_sq = 0.0f;
            for (uint32_t i = 0; i < hidden_size; ++i) {
                float v = mruntime::fp16_bits_to_float(input_data[t * hidden_size + i]);
                sum_sq += v * v;
            }
            float rms_inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
            for (uint32_t i = 0; i < hidden_size; ++i) {
                float v = mruntime::fp16_bits_to_float(input_data[t * hidden_size + i]);
                float w = mruntime::fp16_bits_to_float(weight_data[i]);
                expected[t * hidden_size + i] = v * rms_inv * w;
            }
        }

        std::memset(out_data, 0, out_bytes);
        tc.fp16_ops.rmsnorm(
            arena.descriptor(input_offset, input_bytes),
            arena.descriptor(weight_offset, weight_bytes),
            arena.descriptor(out_offset, out_bytes),
            hidden_size, num_tokens, eps);

        check_close(
            ("rmsnorm(hidden=" + std::to_string(hidden_size) +
             ",tokens=" + std::to_string(num_tokens) + ")").c_str(),
            out_data,
            expected.data(),
            input_count,
            5e-3f);
        std::cout << "  rmsnorm(hidden=" << hidden_size
                  << ", tokens=" << num_tokens << ") PASSED\n";
    }

    {
        const uint32_t hidden_size = 897;
        const uint32_t num_tokens = 4;
        const float eps = 1e-6f;
        const uint32_t input_count = num_tokens * hidden_size;
        const VkDeviceSize input_bytes = input_count * sizeof(uint16_t);
        const VkDeviceSize weight_bytes = hidden_size * sizeof(uint16_t);
        const VkDeviceSize out_bytes = input_count * sizeof(uint16_t);

        auto arena = make_arena(tc, input_bytes + weight_bytes + out_bytes + 3 * tc.alignment);
        const VkDeviceSize input_offset = arena.alloc(input_bytes);
        const VkDeviceSize weight_offset = arena.alloc(weight_bytes);
        const VkDeviceSize out_offset = arena.alloc(out_bytes);

        bool threw = false;
        try {
            tc.fp16_ops.rmsnorm(
                arena.descriptor(input_offset, input_bytes),
                arena.descriptor(weight_offset, weight_bytes),
                arena.descriptor(out_offset, out_bytes),
                hidden_size,
                num_tokens,
                eps);
        } catch (const std::runtime_error&) {
            threw = true;
        }

        if (!threw) {
            throw std::runtime_error("rmsnorm(hidden=897): expected divisible-by-4 validation failure");
        }
        std::cout << "  rmsnorm(hidden=897, tokens=4) rejected as expected\n";
    }
}

// ---- rope ----

void test_rope(TestContext& tc) {
    const uint32_t batch = 1;
    const uint32_t seq_len = 4;
    const uint32_t num_q_heads = 4;
    const uint32_t num_kv_heads = 2;
    const uint32_t head_dim = 64;
    const uint32_t position_offset = 0;
    const uint32_t half_dim = head_dim / 2;

    const uint32_t q_count = batch * seq_len * num_q_heads * head_dim;
    const uint32_t k_count = batch * seq_len * num_kv_heads * head_dim;
    // cos_sin table: (position_offset + seq_len) positions, each head_dim floats
    const uint32_t max_pos = position_offset + seq_len;
    const uint32_t cos_sin_count = max_pos * head_dim;

    const VkDeviceSize q_bytes = q_count * sizeof(uint16_t);
    const VkDeviceSize k_bytes = k_count * sizeof(uint16_t);
    const VkDeviceSize cos_sin_bytes = cos_sin_count * sizeof(float);

    auto arena = make_arena(tc, q_bytes + k_bytes + cos_sin_bytes + 3 * tc.alignment);

    const VkDeviceSize q_offset = arena.alloc(q_bytes);
    const VkDeviceSize k_offset = arena.alloc(k_bytes);
    const VkDeviceSize cos_sin_offset = arena.alloc(cos_sin_bytes);

    uint16_t* q_data = arena.host_ptr<uint16_t>(q_offset);
    uint16_t* k_data = arena.host_ptr<uint16_t>(k_offset);
    float* cos_sin_data = arena.host_ptr<float>(cos_sin_offset);

    // Fill Q, K with deterministic values.
    for (uint32_t i = 0; i < q_count; ++i) {
        float val = -1.0f + 0.002f * static_cast<float>(i % 1000);
        q_data[i] = mruntime::float_to_fp16_bits(val);
    }
    for (uint32_t i = 0; i < k_count; ++i) {
        float val = 0.5f - 0.001f * static_cast<float>(i % 1000);
        k_data[i] = mruntime::float_to_fp16_bits(val);
    }

    // Fill cos/sin table with interleaved (cos, sin) pairs per position:
    // cos_sin[pos * head_dim + i*2] = cos, cos_sin[pos * head_dim + i*2 + 1] = sin
    for (uint32_t pos = 0; pos < max_pos; ++pos) {
        for (uint32_t i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            cos_sin_data[pos * head_dim + i * 2] = std::cos(angle);
            cos_sin_data[pos * head_dim + i * 2 + 1] = std::sin(angle);
        }
    }

    // Save copies for CPU reference.
    std::vector<uint16_t> q_copy(q_data, q_data + q_count);
    std::vector<uint16_t> k_copy(k_data, k_data + k_count);

    // CPU reference.
    std::vector<float> q_expected(q_count);
    std::vector<float> k_expected(k_count);

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
            uint32_t pos = position_offset + s;
            // Q heads
            for (uint32_t h = 0; h < num_q_heads; ++h) {
                uint32_t base = ((b * seq_len + s) * num_q_heads + h) * head_dim;
                for (uint32_t i = 0; i < half_dim; ++i) {
                    float x0 = mruntime::fp16_bits_to_float(q_copy[base + i]);
                    float x1 = mruntime::fp16_bits_to_float(q_copy[base + half_dim + i]);
                    float cos_val = cos_sin_data[pos * head_dim + i * 2];
                    float sin_val = cos_sin_data[pos * head_dim + i * 2 + 1];
                    q_expected[base + i] = x0 * cos_val - x1 * sin_val;
                    q_expected[base + half_dim + i] = x1 * cos_val + x0 * sin_val;
                }
            }
            // K heads
            for (uint32_t h = 0; h < num_kv_heads; ++h) {
                uint32_t base = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
                for (uint32_t i = 0; i < half_dim; ++i) {
                    float x0 = mruntime::fp16_bits_to_float(k_copy[base + i]);
                    float x1 = mruntime::fp16_bits_to_float(k_copy[base + half_dim + i]);
                    float cos_val = cos_sin_data[pos * head_dim + i * 2];
                    float sin_val = cos_sin_data[pos * head_dim + i * 2 + 1];
                    k_expected[base + i] = x0 * cos_val - x1 * sin_val;
                    k_expected[base + half_dim + i] = x1 * cos_val + x0 * sin_val;
                }
            }
        }
    }

    tc.fp16_ops.rope(
        arena.descriptor(q_offset, q_bytes),
        arena.descriptor(k_offset, k_bytes),
        arena.descriptor(cos_sin_offset, cos_sin_bytes),
        batch, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset);

    check_close("rope(Q)", q_data, q_expected.data(), q_count, 5e-3f);
    check_close("rope(K)", k_data, k_expected.data(), k_count, 5e-3f);
    std::cout << "  rope PASSED\n";
}

void test_qkv_bias_rope_cache_decode(TestContext& tc, bool has_bias) {
    const uint32_t num_q_heads = 6;
    const uint32_t num_kv_heads = 2;
    const uint32_t head_dim = 64;
    const uint32_t q_dim = num_q_heads * head_dim;
    const uint32_t kv_dim = num_kv_heads * head_dim;
    const uint32_t qkv_dim = q_dim + 2 * kv_dim;
    const uint32_t max_seq_len = 16;
    const uint32_t position_offset = 5;
    const uint32_t half_dim = head_dim / 2;
    const uint32_t cache_count = num_kv_heads * max_seq_len * head_dim;
    const uint32_t cos_sin_count = max_seq_len * head_dim;

    const VkDeviceSize qkv_bytes = qkv_dim * sizeof(uint16_t);
    const VkDeviceSize bias_bytes = qkv_dim * sizeof(uint16_t);
    const VkDeviceSize cos_sin_bytes = cos_sin_count * sizeof(float);
    const VkDeviceSize q_out_bytes = q_dim * sizeof(uint16_t);
    const VkDeviceSize cache_bytes = cache_count * sizeof(uint16_t);

    auto arena = make_arena(tc, qkv_bytes + bias_bytes + cos_sin_bytes + q_out_bytes + 2 * cache_bytes + 6 * tc.alignment);

    const VkDeviceSize qkv_offset = arena.alloc(qkv_bytes);
    const VkDeviceSize bias_offset = arena.alloc(bias_bytes);
    const VkDeviceSize cos_sin_offset = arena.alloc(cos_sin_bytes);
    const VkDeviceSize q_out_offset = arena.alloc(q_out_bytes);
    const VkDeviceSize k_cache_offset = arena.alloc(cache_bytes);
    const VkDeviceSize v_cache_offset = arena.alloc(cache_bytes);

    uint16_t* qkv_data = arena.host_ptr<uint16_t>(qkv_offset);
    uint16_t* bias_data = arena.host_ptr<uint16_t>(bias_offset);
    float* cos_sin_data = arena.host_ptr<float>(cos_sin_offset);
    uint16_t* q_out_data = arena.host_ptr<uint16_t>(q_out_offset);
    uint16_t* k_cache_data = arena.host_ptr<uint16_t>(k_cache_offset);
    uint16_t* v_cache_data = arena.host_ptr<uint16_t>(v_cache_offset);

    for (uint32_t i = 0; i < qkv_dim; ++i) {
        const float value = -0.35f + 0.0025f * static_cast<float>((i * 13u) % 257u);
        qkv_data[i] = mruntime::float_to_fp16_bits(value);
        const float bias = has_bias ? (-0.03f + 0.0004f * static_cast<float>((i * 7u) % 97u)) : 0.0f;
        bias_data[i] = mruntime::float_to_fp16_bits(bias);
    }

    for (uint32_t pos = 0; pos < max_seq_len; ++pos) {
        for (uint32_t i = 0; i < half_dim; ++i) {
            const float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(pos) * freq;
            cos_sin_data[pos * head_dim + i * 2] = std::cos(angle);
            cos_sin_data[pos * head_dim + i * 2 + 1] = std::sin(angle);
        }
    }

    const uint16_t cache_sentinel_k = mruntime::float_to_fp16_bits(11.0f);
    const uint16_t cache_sentinel_v = mruntime::float_to_fp16_bits(-7.0f);
    std::fill(q_out_data, q_out_data + q_dim, mruntime::float_to_fp16_bits(0.0f));
    std::fill(k_cache_data, k_cache_data + cache_count, cache_sentinel_k);
    std::fill(v_cache_data, v_cache_data + cache_count, cache_sentinel_v);

    std::vector<float> q_expected(q_dim, 0.0f);
    std::vector<float> k_expected(cache_count, mruntime::fp16_bits_to_float(cache_sentinel_k));
    std::vector<float> v_expected(cache_count, mruntime::fp16_bits_to_float(cache_sentinel_v));

    for (uint32_t head = 0; head < num_q_heads; ++head) {
        const uint32_t base = head * head_dim;
        for (uint32_t i = 0; i < half_dim; ++i) {
            const uint32_t idx0 = base + i;
            const uint32_t idx1 = base + half_dim + i;
            const float x0 = mruntime::fp16_bits_to_float(qkv_data[idx0]) + mruntime::fp16_bits_to_float(bias_data[idx0]);
            const float x1 = mruntime::fp16_bits_to_float(qkv_data[idx1]) + mruntime::fp16_bits_to_float(bias_data[idx1]);
            const float cos_val = cos_sin_data[position_offset * head_dim + i * 2];
            const float sin_val = cos_sin_data[position_offset * head_dim + i * 2 + 1];
            q_expected[idx0] = x0 * cos_val - x1 * sin_val;
            q_expected[idx1] = x1 * cos_val + x0 * sin_val;
        }
    }

    for (uint32_t head = 0; head < num_kv_heads; ++head) {
        const uint32_t src_base = q_dim + head * head_dim;
        const uint32_t dst_base = (head * max_seq_len + position_offset) * head_dim;
        for (uint32_t i = 0; i < half_dim; ++i) {
            const uint32_t src0 = src_base + i;
            const uint32_t src1 = src_base + half_dim + i;
            const float x0 = mruntime::fp16_bits_to_float(qkv_data[src0]) + mruntime::fp16_bits_to_float(bias_data[src0]);
            const float x1 = mruntime::fp16_bits_to_float(qkv_data[src1]) + mruntime::fp16_bits_to_float(bias_data[src1]);
            const float cos_val = cos_sin_data[position_offset * head_dim + i * 2];
            const float sin_val = cos_sin_data[position_offset * head_dim + i * 2 + 1];
            k_expected[dst_base + i] = x0 * cos_val - x1 * sin_val;
            k_expected[dst_base + half_dim + i] = x1 * cos_val + x0 * sin_val;
        }

        const uint32_t v_src_base = q_dim + kv_dim + head * head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) {
            const uint32_t src = v_src_base + d;
            v_expected[dst_base + d] =
                mruntime::fp16_bits_to_float(qkv_data[src]) + mruntime::fp16_bits_to_float(bias_data[src]);
        }
    }

    tc.fp16_ops.qkv_bias_rope_cache_decode(
        arena.descriptor(qkv_offset, qkv_bytes),
        has_bias ? arena.descriptor(bias_offset, bias_bytes) : arena.descriptor(qkv_offset, qkv_bytes),
        arena.descriptor(cos_sin_offset, cos_sin_bytes),
        arena.descriptor(q_out_offset, q_out_bytes),
        arena.descriptor(k_cache_offset, cache_bytes),
        arena.descriptor(v_cache_offset, cache_bytes),
        q_dim,
        kv_dim,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        position_offset,
        has_bias);

    const std::string label = has_bias ? "qkv_bias_rope_cache_decode(bias)" : "qkv_bias_rope_cache_decode(no_bias)";
    check_close((label + ":Q").c_str(), q_out_data, q_expected.data(), q_dim, 5e-3f);
    check_close((label + ":K").c_str(), k_cache_data, k_expected.data(), cache_count, 5e-3f);
    check_close((label + ":V").c_str(), v_cache_data, v_expected.data(), cache_count, 5e-3f);
    std::cout << "  " << label << " PASSED\n";
}

// ---- transpose BSHD -> BHSD ----

void test_transpose(TestContext& tc) {
    const uint32_t B = 2, S = 4, H = 4;
    const std::vector<uint32_t> head_dims = {64, 65};

    for (uint32_t D : head_dims) {
        const uint32_t total = B * S * H * D;
        const VkDeviceSize input_bytes = total * sizeof(uint16_t);
        const VkDeviceSize out_bytes = total * sizeof(uint16_t);

        auto arena = make_arena(tc, input_bytes + out_bytes + 2 * tc.alignment);

        const VkDeviceSize input_offset = arena.alloc(input_bytes);
        const VkDeviceSize out_offset = arena.alloc(out_bytes);

        uint16_t* input_data = arena.host_ptr<uint16_t>(input_offset);
        uint16_t* out_data = arena.host_ptr<uint16_t>(out_offset);

        for (uint32_t i = 0; i < total; ++i) {
            input_data[i] = mruntime::float_to_fp16_bits(static_cast<float>(i));
        }

        std::vector<uint16_t> expected(total);
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                for (uint32_t h = 0; h < H; ++h) {
                    for (uint32_t d = 0; d < D; ++d) {
                        uint32_t src_idx = ((b * S + s) * H + h) * D + d;
                        uint32_t dst_idx = ((b * H + h) * S + s) * D + d;
                        expected[dst_idx] = input_data[src_idx];
                    }
                }
            }
        }

        std::memset(out_data, 0, out_bytes);
        tc.fp16_ops.transpose_bshd_to_bhsd(
            arena.descriptor(input_offset, input_bytes),
            arena.descriptor(out_offset, out_bytes),
            B, S, H, D);

        check_exact_fp16(("transpose(D=" + std::to_string(D) + ")").c_str(), out_data, expected.data(), total);
        std::cout << "  transpose(D=" << D << ") PASSED\n";
    }
}

// ---- kv_cache_copy ----

void test_kv_cache_copy(TestContext& tc) {
    const uint32_t batch = 2;
    const uint32_t seq_len = 4;
    const uint32_t num_kv_heads = 2;
    const uint32_t max_seq_len = 32;
    const uint32_t position_offset = 3;
    const std::vector<uint32_t> head_dims = {64, 65};

    for (uint32_t head_dim : head_dims) {
        const uint32_t kv_count = batch * seq_len * num_kv_heads * head_dim;
        const uint32_t cache_count = batch * num_kv_heads * max_seq_len * head_dim;

        const VkDeviceSize kv_bytes = kv_count * sizeof(uint16_t);
        const VkDeviceSize cache_bytes = cache_count * sizeof(uint16_t);

        auto arena = make_arena(tc, 2 * kv_bytes + 2 * cache_bytes + 4 * tc.alignment);

        const VkDeviceSize k_in_offset = arena.alloc(kv_bytes);
        const VkDeviceSize v_in_offset = arena.alloc(kv_bytes);
        const VkDeviceSize k_cache_offset = arena.alloc(cache_bytes);
        const VkDeviceSize v_cache_offset = arena.alloc(cache_bytes);

        uint16_t* k_in_data = arena.host_ptr<uint16_t>(k_in_offset);
        uint16_t* v_in_data = arena.host_ptr<uint16_t>(v_in_offset);
        uint16_t* k_cache_data = arena.host_ptr<uint16_t>(k_cache_offset);
        uint16_t* v_cache_data = arena.host_ptr<uint16_t>(v_cache_offset);

        for (uint32_t i = 0; i < kv_count; ++i) {
            k_in_data[i] = mruntime::float_to_fp16_bits(1.0f + 0.01f * static_cast<float>(i));
            v_in_data[i] = mruntime::float_to_fp16_bits(-1.0f + 0.01f * static_cast<float>(i));
        }

        std::memset(k_cache_data, 0, cache_bytes);
        std::memset(v_cache_data, 0, cache_bytes);

        std::vector<uint16_t> k_cache_expected(cache_count, 0);
        std::vector<uint16_t> v_cache_expected(cache_count, 0);

        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t s = 0; s < seq_len; ++s) {
                for (uint32_t h = 0; h < num_kv_heads; ++h) {
                    uint32_t src_base = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
                    uint32_t cache_pos = position_offset + s;
                    uint32_t dst_base = ((b * num_kv_heads + h) * max_seq_len + cache_pos) * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        k_cache_expected[dst_base + d] = k_in_data[src_base + d];
                        v_cache_expected[dst_base + d] = v_in_data[src_base + d];
                    }
                }
            }
        }

        tc.fp16_ops.kv_cache_copy(
            arena.descriptor(k_in_offset, kv_bytes),
            arena.descriptor(v_in_offset, kv_bytes),
            arena.descriptor(k_cache_offset, cache_bytes),
            arena.descriptor(v_cache_offset, cache_bytes),
            batch, seq_len, num_kv_heads, head_dim, max_seq_len, position_offset);

        check_exact_fp16(("kv_cache_copy(K, head_dim=" + std::to_string(head_dim) + ")").c_str(),
                         k_cache_data, k_cache_expected.data(), cache_count);
        check_exact_fp16(("kv_cache_copy(V, head_dim=" + std::to_string(head_dim) + ")").c_str(),
                         v_cache_data, v_cache_expected.data(), cache_count);
        std::cout << "  kv_cache_copy(head_dim=" << head_dim << ") PASSED\n";
    }
}

void test_attention_decode_gqa(
    TestContext& tc,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t kv_len,
    uint32_t kv_stride
) {
    const std::string label =
        "attention_decode_gqa(qh=" + std::to_string(num_q_heads) +
        ",kvh=" + std::to_string(num_kv_heads) +
        ",d=" + std::to_string(head_dim) +
        ",kv_len=" + std::to_string(kv_len) + ")";

    const uint32_t q_count = num_q_heads * head_dim;
    const uint32_t kv_count = num_kv_heads * kv_stride * head_dim;
    const uint32_t out_count = q_count;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const VkDeviceSize q_bytes = q_count * sizeof(uint16_t);
    const VkDeviceSize kv_bytes = kv_count * sizeof(uint16_t);
    const VkDeviceSize out_bytes = out_count * sizeof(uint16_t);

    auto arena = make_arena(tc, q_bytes + 2 * kv_bytes + out_bytes + 4 * tc.alignment);

    const VkDeviceSize q_offset = arena.alloc(q_bytes);
    const VkDeviceSize k_offset = arena.alloc(kv_bytes);
    const VkDeviceSize v_offset = arena.alloc(kv_bytes);
    const VkDeviceSize out_offset = arena.alloc(out_bytes);

    uint16_t* q_data = arena.host_ptr<uint16_t>(q_offset);
    uint16_t* k_data = arena.host_ptr<uint16_t>(k_offset);
    uint16_t* v_data = arena.host_ptr<uint16_t>(v_offset);
    uint16_t* out_data = arena.host_ptr<uint16_t>(out_offset);

    for (uint32_t i = 0; i < q_count; ++i) {
        q_data[i] = mruntime::float_to_fp16_bits(-0.25f + 0.004f * static_cast<float>(i % 97));
    }

    for (uint32_t head = 0; head < num_kv_heads; ++head) {
        for (uint32_t seq = 0; seq < kv_stride; ++seq) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                const uint32_t idx = (head * kv_stride + seq) * head_dim + dim;
                if (seq < kv_len) {
                    const float k_val =
                        0.015f * static_cast<float>(head + 1) +
                        0.006f * static_cast<float>(seq % 41) -
                        0.002f * static_cast<float>(dim % 17);
                    const float v_val =
                        -0.020f * static_cast<float>(head + 1) +
                        0.005f * static_cast<float>(seq % 53) +
                        0.0015f * static_cast<float>(dim % 29);
                    k_data[idx] = mruntime::float_to_fp16_bits(k_val);
                    v_data[idx] = mruntime::float_to_fp16_bits(v_val);
                } else {
                    k_data[idx] = mruntime::float_to_fp16_bits(300.0f);
                    v_data[idx] = mruntime::float_to_fp16_bits(-300.0f);
                }
            }
        }
    }

    std::vector<uint16_t> expected_fp16(out_count);
    std::vector<float> expected(out_count);
    mruntime::qwen2_flash_attention_gqa_fp16(
        q_data,
        k_data,
        v_data,
        expected_fp16.data(),
        1,
        num_q_heads,
        num_kv_heads,
        1,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        true,
        nullptr);
    for (uint32_t i = 0; i < out_count; ++i) {
        expected[i] = mruntime::fp16_bits_to_float(expected_fp16[i]);
    }

    std::memset(out_data, 0, out_bytes);
    tc.fp16_ops.attention_decode_gqa(
        arena.descriptor(q_offset, q_bytes),
        arena.descriptor(k_offset, kv_bytes),
        arena.descriptor(v_offset, kv_bytes),
        arena.descriptor(out_offset, out_bytes),
        num_q_heads,
        num_kv_heads,
        kv_len,
        kv_stride,
        head_dim,
        scale);

    check_close(label.c_str(), out_data, expected.data(), out_count, 5e-2f);
    std::cout << "  " << label << " PASSED\n";
}

void test_attention_prefill_gqa(
    TestContext& tc,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t q_len,
    uint32_t head_dim,
    uint32_t kv_len,
    uint32_t kv_stride
) {
    const std::string label =
        "attention_prefill_gqa(qh=" + std::to_string(num_q_heads) +
        ",kvh=" + std::to_string(num_kv_heads) +
        ",q_len=" + std::to_string(q_len) +
        ",d=" + std::to_string(head_dim) +
        ",kv_len=" + std::to_string(kv_len) + ")";

    const uint32_t q_count = num_q_heads * q_len * head_dim;
    const uint32_t kv_count = num_kv_heads * kv_stride * head_dim;
    const uint32_t out_count = q_count;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const VkDeviceSize q_bytes = q_count * sizeof(uint16_t);
    const VkDeviceSize kv_bytes = kv_count * sizeof(uint16_t);
    const VkDeviceSize out_bytes = out_count * sizeof(uint16_t);

    auto arena = make_arena(tc, q_bytes + 2 * kv_bytes + out_bytes + 4 * tc.alignment);

    const VkDeviceSize q_offset = arena.alloc(q_bytes);
    const VkDeviceSize k_offset = arena.alloc(kv_bytes);
    const VkDeviceSize v_offset = arena.alloc(kv_bytes);
    const VkDeviceSize out_offset = arena.alloc(out_bytes);

    uint16_t* q_data = arena.host_ptr<uint16_t>(q_offset);
    uint16_t* k_data = arena.host_ptr<uint16_t>(k_offset);
    uint16_t* v_data = arena.host_ptr<uint16_t>(v_offset);
    uint16_t* out_data = arena.host_ptr<uint16_t>(out_offset);

    for (uint32_t i = 0; i < q_count; ++i) {
        q_data[i] = mruntime::float_to_fp16_bits(-0.15f + 0.003f * static_cast<float>(i % 131));
    }

    for (uint32_t head = 0; head < num_kv_heads; ++head) {
        for (uint32_t seq = 0; seq < kv_stride; ++seq) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                const uint32_t idx = (head * kv_stride + seq) * head_dim + dim;
                if (seq < kv_len) {
                    const float k_val =
                        0.012f * static_cast<float>(head + 1) +
                        0.004f * static_cast<float>(seq % 47) -
                        0.001f * static_cast<float>(dim % 19);
                    const float v_val =
                        -0.018f * static_cast<float>(head + 1) +
                        0.0035f * static_cast<float>(seq % 59) +
                        0.0012f * static_cast<float>(dim % 31);
                    k_data[idx] = mruntime::float_to_fp16_bits(k_val);
                    v_data[idx] = mruntime::float_to_fp16_bits(v_val);
                } else {
                    k_data[idx] = mruntime::float_to_fp16_bits(300.0f);
                    v_data[idx] = mruntime::float_to_fp16_bits(-300.0f);
                }
            }
        }
    }

    std::vector<uint16_t> expected_fp16(out_count);
    std::vector<float> expected(out_count);
    mruntime::qwen2_flash_attention_gqa_fp16(
        q_data,
        k_data,
        v_data,
        expected_fp16.data(),
        1,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale,
        true,
        nullptr);
    for (uint32_t i = 0; i < out_count; ++i) {
        expected[i] = mruntime::fp16_bits_to_float(expected_fp16[i]);
    }

    std::memset(out_data, 0, out_bytes);
    tc.fp16_ops.attention_prefill_gqa(
        arena.descriptor(q_offset, q_bytes),
        arena.descriptor(k_offset, kv_bytes),
        arena.descriptor(v_offset, kv_bytes),
        arena.descriptor(out_offset, out_bytes),
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        kv_stride,
        head_dim,
        scale);

    check_close(label.c_str(), out_data, expected.data(), out_count, 5e-2f);
    std::cout << "  " << label << " PASSED\n";
}

void run_all_tests() {
    TestContext tc = TestContext::Create();

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(tc.context.physical_device(), &properties);
    std::cout << "Using Vulkan device: " << properties.deviceName << "\n";

    test_silu_mul_interleaved(tc);
    test_rmsnorm(tc);
    test_qkv_bias_rope_cache_decode(tc, false);
    test_qkv_bias_rope_cache_decode(tc, true);
    test_rope(tc);
    test_transpose(tc);
    test_kv_cache_copy(tc);
    test_attention_decode_gqa(tc, 4, 2, 8, 5, 9);
    test_attention_decode_gqa(tc, 4, 2, 32, 37, 64);
    test_attention_decode_gqa(tc, 14, 2, 64, 129, 256);
    test_attention_prefill_gqa(tc, 4, 2, 5, 8, 5, 9);
    test_attention_prefill_gqa(tc, 4, 2, 8, 32, 37, 64);
    test_attention_prefill_gqa(tc, 14, 2, 3, 64, 129, 256);
}

}  // namespace

int main() {
    try {
        run_all_tests();
        std::cout << "vulkan_fp16_kernels_test PASSED\n";
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_fp16_kernels_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_fp16_kernels_test FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_fp16_kernels_test FAILED: " << error.what() << "\n";
        return 1;
    }
}
