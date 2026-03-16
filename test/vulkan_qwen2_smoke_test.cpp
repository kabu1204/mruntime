#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mruntime/arena.h"
#include "mruntime/dtype.h"
#include "mruntime/qwen2_forward.h"
#include "mruntime/qwen2_forward_vk.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/qwen2_weights.h"

#include "vk_helpers.h"

namespace {

constexpr float kLogitsTolerance = 2.0f;

uint32_t argmax_fp16(const uint16_t* values, uint32_t n) {
    uint32_t best_index = 0;
    float best_value = mruntime::fp16_bits_to_float(values[0]);
    for (uint32_t i = 1; i < n; ++i) {
        const float value = mruntime::fp16_bits_to_float(values[i]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    return best_index;
}

void check_close_fp16(
    const char* test_name,
    const uint16_t* got,
    const uint16_t* expected,
    uint32_t n,
    float tolerance
) {
    uint32_t max_i = 0;
    float max_g = 0.0f;
    float max_e = 0.0f;
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float g = mruntime::fp16_bits_to_float(got[i]);
        float e = mruntime::fp16_bits_to_float(expected[i]);
        float diff = std::fabs(g - e);
        if (!std::isfinite(g) || !std::isfinite(e)) {
            throw std::runtime_error(
                std::string(test_name) + " produced non-finite values at i=" + std::to_string(i));
        }
        if (diff > max_diff) {
            max_i = i;
            max_g = g;
            max_e = e;
            max_diff = diff;
        }
    }
    if (max_diff > tolerance) {
        throw std::runtime_error(
            std::string(test_name) + " mismatch at i=" + std::to_string(max_i) +
            ": got=" + std::to_string(max_g) +
            ", expected=" + std::to_string(max_e) +
            ", diff=" + std::to_string(max_diff));
    }
}

mruntime::Qwen2Weights make_tiny_weights(
    const mruntime::QwenConfig& cfg,
    mruntime::Arena& weights_arena
) {
    mruntime::Qwen2Weights weights = {};
    weights.num_layers = cfg.num_layers;
    weights.layers = weights_arena.alloc_array<mruntime::Qwen2LayerWeights>(cfg.num_layers);

    const size_t head_dim = cfg.head_dim();
    const size_t q_dim = cfg.num_attention_heads * head_dim;
    const size_t kv_dim = cfg.num_kv_heads * head_dim;
    const size_t qkv_dim = q_dim + 2 * kv_dim;

    auto fill = [](uint16_t* dst, size_t n, float base, float step) {
        for (size_t i = 0; i < n; ++i) {
            float v = base + step * static_cast<float>(i % 1024);
            dst[i] = mruntime::float_to_fp16_bits(v);
        }
    };

    // embed + final_norm + lm_head
    weights.embed_tokens = weights_arena.alloc_array<uint16_t>(cfg.vocab_size * cfg.hidden_size);
    fill(const_cast<uint16_t*>(weights.embed_tokens), cfg.vocab_size * cfg.hidden_size, -0.2f, 0.0001f);

    weights.final_norm = weights_arena.alloc_array<uint16_t>(cfg.hidden_size);
    fill(const_cast<uint16_t*>(weights.final_norm), cfg.hidden_size, 1.0f, 0.00005f);

    weights.lm_head = weights_arena.alloc_array<uint16_t>(cfg.vocab_size * cfg.hidden_size);
    fill(const_cast<uint16_t*>(weights.lm_head), cfg.vocab_size * cfg.hidden_size, 0.1f, 0.00007f);
    weights.lm_head_packed = nullptr;

    for (size_t layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        auto& layer = weights.layers[layer_idx];

        layer.input_norm = weights_arena.alloc_array<uint16_t>(cfg.hidden_size);
        fill(const_cast<uint16_t*>(layer.input_norm), cfg.hidden_size, 1.0f, 0.00003f);

        layer.qkv_proj = weights_arena.alloc_array<uint16_t>(qkv_dim * cfg.hidden_size);
        fill(const_cast<uint16_t*>(layer.qkv_proj), qkv_dim * cfg.hidden_size, -0.05f, 0.00002f);
        layer.qkv_bias = nullptr;

        layer.o_proj = weights_arena.alloc_array<uint16_t>(cfg.hidden_size * q_dim);
        fill(const_cast<uint16_t*>(layer.o_proj), cfg.hidden_size * q_dim, 0.02f, 0.00001f);

        layer.post_attn_norm = weights_arena.alloc_array<uint16_t>(cfg.hidden_size);
        fill(const_cast<uint16_t*>(layer.post_attn_norm), cfg.hidden_size, 1.0f, 0.00004f);

        layer.gate_up_proj = weights_arena.alloc_array<uint16_t>(2 * cfg.intermediate_size * cfg.hidden_size);
        fill(const_cast<uint16_t*>(layer.gate_up_proj), 2 * cfg.intermediate_size * cfg.hidden_size, -0.03f, 0.00001f);

        layer.down_proj = weights_arena.alloc_array<uint16_t>(cfg.hidden_size * cfg.intermediate_size);
        fill(const_cast<uint16_t*>(layer.down_proj), cfg.hidden_size * cfg.intermediate_size, 0.01f, 0.00001f);

        layer.qkv_proj_packed = nullptr;
        layer.o_proj_packed = nullptr;
        layer.gate_up_proj_packed = nullptr;
        layer.down_proj_packed = nullptr;
    }

    return weights;
}

void run_smoke() {
    mruntime::QwenConfig cfg;
    const size_t head_dim = 64;
    cfg.vocab_size = 128;
    cfg.hidden_size = head_dim * 4;
    cfg.num_layers = 2;
    cfg.num_attention_heads = 4;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 64;
    cfg.max_position_embeddings = 64;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    const size_t max_seq_len = 32;
    const size_t max_batch_tokens = prompt.size();

    const mruntime::Qwen2MemorySizes sizes =
        mruntime::qwen2_memory_sizes(cfg, max_seq_len, max_batch_tokens, true);

    mruntime::Qwen2Arenas arenas = mruntime::create_qwen2_arenas(
        sizes.weights_bytes,
        sizes.kv_cache_bytes,
        sizes.scratch_bytes
    );

    mruntime::Qwen2Weights weights = make_tiny_weights(cfg, arenas.weights);
    mruntime::Qwen2KVCache kv_cpu = mruntime::qwen2_init_kv_cache(cfg, arenas.kv_cache, max_seq_len);
    mruntime::Qwen2Scratch scratch_cpu = mruntime::qwen2_init_scratch(cfg, arenas.scratch, max_batch_tokens);

    const uint16_t* cpu_last = mruntime::qwen2_prefill(
        cfg,
        weights,
        kv_cpu,
        scratch_cpu,
        prompt.data(),
        prompt.size(),
        nullptr
    );
    std::vector<uint16_t> cpu_last_copy(cpu_last, cpu_last + cfg.vocab_size);

    const int32_t input_token = 7;
    const uint16_t* cpu_decode = mruntime::qwen2_decode(
        cfg,
        weights,
        kv_cpu,
        scratch_cpu,
        input_token,
        nullptr
    );
    std::vector<uint16_t> cpu_decode_copy(cpu_decode, cpu_decode + cfg.vocab_size);

    auto vk_state = mruntime::qwen2_vk_create(cfg, weights, max_seq_len, max_batch_tokens);
    mruntime::destroy_qwen2_arenas(arenas);

    const uint16_t* vk_last = mruntime::qwen2_prefill_vk(
        *vk_state,
        prompt.data(),
        prompt.size(),
        nullptr
    );
    if (argmax_fp16(vk_last, static_cast<uint32_t>(cfg.vocab_size)) !=
        argmax_fp16(cpu_last_copy.data(), static_cast<uint32_t>(cfg.vocab_size))) {
        throw std::runtime_error("prefill_logits argmax mismatch");
    }

    check_close_fp16(
        "prefill_logits",
        vk_last,
        cpu_last_copy.data(),
        static_cast<uint32_t>(cfg.vocab_size),
        kLogitsTolerance
    );

    const uint16_t* vk_decode = mruntime::qwen2_decode_vk(*vk_state, input_token, nullptr);
    if (argmax_fp16(vk_decode, static_cast<uint32_t>(cfg.vocab_size)) !=
        argmax_fp16(cpu_decode_copy.data(), static_cast<uint32_t>(cfg.vocab_size))) {
        throw std::runtime_error("decode_logits argmax mismatch");
    }
    check_close_fp16(
        "decode_logits",
        vk_decode,
        cpu_decode_copy.data(),
        static_cast<uint32_t>(cfg.vocab_size),
        kLogitsTolerance
    );

    std::cout << "vulkan_qwen2_smoke_test PASSED\n";
    vk_state.reset();
}

}  // namespace

int main() {
    try {
        run_smoke();
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_qwen2_smoke_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_qwen2_smoke_test FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_qwen2_smoke_test FAILED: " << error.what() << "\n";
        return 1;
    }
}
