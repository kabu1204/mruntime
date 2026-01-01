#include "mruntime/qwen_model.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mruntime {

QwenConfig QwenConfig::from_json(const std::string& json) {
    QwenConfig config;

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse Qwen config JSON: ") + e.what());
    }

    config.vocab_size = j.value("vocab_size", config.vocab_size);
    config.hidden_size = j.value("hidden_size", config.hidden_size);
    config.num_layers = j.value("num_hidden_layers", config.num_layers);
    config.num_attention_heads = j.value("num_attention_heads", config.num_attention_heads);
    config.num_kv_heads = j.value("num_key_value_heads", config.num_kv_heads);
    config.intermediate_size = j.value("intermediate_size", config.intermediate_size);
    config.max_position_embeddings = j.value("max_position_embeddings", config.max_position_embeddings);
    config.rms_norm_eps = j.value("rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = j.value("rope_theta", config.rope_theta);

    if (config.rope_theta == 0.0f) config.rope_theta = 10000.0f;
    return config;
}

void KVCache::allocate(const QwenConfig& config, size_t max_seq_len, DType dtype) {
    key_cache.clear();
    value_cache.clear();
    key_cache.reserve(config.num_layers);
    value_cache.reserve(config.num_layers);

    size_t head_dim = config.head_dim();

    for (size_t i = 0; i < config.num_layers; ++i) {
        key_cache.push_back(Tensor::zeros(
            Shape({1, config.num_kv_heads, max_seq_len, head_dim}), dtype));
        value_cache.push_back(Tensor::zeros(
            Shape({1, config.num_kv_heads, max_seq_len, head_dim}), dtype));
    }
    seq_len = 0;
}

void QwenEmbedding::load_weights(const SafeTensorsFile& file, const std::string& prefix) {
    // Load weights in native dtype (FP16/BF16) for memory efficiency
    // Backend handles mixed-precision computation
    embed_tokens_ = file.load_tensor_copy(prefix + ".weight");
}

Tensor QwenEmbedding::forward(Backend& backend, const std::vector<int>& token_ids) {
    size_t hidden_size = embed_tokens_.dim(1);
    Tensor output = Tensor::empty(Shape({1, token_ids.size(), hidden_size}), DType::FP32);
    backend.embedding_lookup(embed_tokens_, token_ids, output);
    return output;
}

void QwenRMSNorm::load_weights(const SafeTensorsFile& file, const std::string& name) {
    weight_ = file.load_tensor_copy(name + ".weight");
}

void QwenRMSNorm::forward(Backend& backend, const Tensor& input, Tensor& output, float eps) {
    backend.rmsnorm(input, weight_, output, eps);
}

QwenRotaryEmbedding::QwenRotaryEmbedding(const QwenConfig& config)
    : theta_(config.rope_theta) {}

void QwenRotaryEmbedding::apply(Backend& backend, Tensor& Q, Tensor& K, size_t position_offset) {
    backend.rope(Q, K, position_offset, theta_);
}

QwenAttention::QwenAttention(const QwenConfig& config)
    : config_(config), rope_(config) {}

void QwenAttention::load_weights(const SafeTensorsFile& file, const std::string& prefix) {
    q_proj_ = file.load_tensor_copy(prefix + ".q_proj.weight");
    k_proj_ = file.load_tensor_copy(prefix + ".k_proj.weight");
    v_proj_ = file.load_tensor_copy(prefix + ".v_proj.weight");
    o_proj_ = file.load_tensor_copy(prefix + ".o_proj.weight");

    // Qwen2.5 includes q/k/v biases (BF16 in the provided model). Convert to FP32 once.
    if (file.has_tensor(prefix + ".q_proj.bias")) {
        q_bias_ = file.load_tensor_copy(prefix + ".q_proj.bias", DType::FP32);
    }
    if (file.has_tensor(prefix + ".k_proj.bias")) {
        k_bias_ = file.load_tensor_copy(prefix + ".k_proj.bias", DType::FP32);
    }
    if (file.has_tensor(prefix + ".v_proj.bias")) {
        v_bias_ = file.load_tensor_copy(prefix + ".v_proj.bias", DType::FP32);
    }
}

Tensor QwenAttention::forward(
    Backend& backend,
    const Tensor& hidden_states,
    KVCache& kv_cache,
    size_t layer_idx,
    size_t position_offset
) {
    size_t batch = hidden_states.dim(0);
    size_t seq_len = hidden_states.dim(1);
    size_t hidden_size = config_.hidden_size;
    size_t num_heads = config_.num_attention_heads;
    size_t num_kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim();

    // Compute Q, K, V projections
    Tensor Q_proj = Tensor::empty(Shape({batch, seq_len, num_heads * head_dim}), DType::FP32);
    Tensor K_proj = Tensor::empty(Shape({batch, seq_len, num_kv_heads * head_dim}), DType::FP32);
    Tensor V_proj = Tensor::empty(Shape({batch, seq_len, num_kv_heads * head_dim}), DType::FP32);

    backend.gemm(hidden_states, q_proj_, Q_proj, 1.0f, 0.0f, false, true);
    backend.gemm(hidden_states, k_proj_, K_proj, 1.0f, 0.0f, false, true);
    backend.gemm(hidden_states, v_proj_, V_proj, 1.0f, 0.0f, false, true);

    auto add_bias = [](Tensor& out, const Tensor& bias) {
        if (bias.data() == nullptr || bias.numel() == 0) {
            return;
        }
        assert(out.dtype() == DType::FP32);
        assert(bias.dtype() == DType::FP32);
        assert(bias.ndim() == 1);
        const size_t out_features = bias.dim(0);
        assert(out.dim(out.ndim() - 1) == out_features);

        float* out_ptr = out.data_ptr<float>();
        const float* b = bias.data_ptr<float>();
        const size_t rows = out.numel() / out_features;
        for (size_t r = 0; r < rows; ++r) {
            float* row = out_ptr + r * out_features;
            for (size_t i = 0; i < out_features; ++i) {
                row[i] += b[i];
            }
        }
    };

    add_bias(Q_proj, q_bias_);
    add_bias(K_proj, k_bias_);
    add_bias(V_proj, v_bias_);

    // Reshape to [batch, seq_len, num_heads, head_dim] for RoPE
    // Note: from_buffer creates a non-owning view, but Q_proj/K_proj/V_proj stay alive
    Tensor Q = Tensor::from_buffer(Q_proj.data(), Shape({batch, seq_len, num_heads, head_dim}), DType::FP32);
    Tensor K = Tensor::from_buffer(K_proj.data(), Shape({batch, seq_len, num_kv_heads, head_dim}), DType::FP32);
    Tensor V = Tensor::from_buffer(V_proj.data(), Shape({batch, seq_len, num_kv_heads, head_dim}), DType::FP32);

    rope_.apply(backend, Q, K, position_offset);

    Tensor& key_cache = kv_cache.key_cache[layer_idx];
    Tensor& value_cache = kv_cache.value_cache[layer_idx];

    float* k_cache_ptr = key_cache.data_ptr<float>();
    float* v_cache_ptr = value_cache.data_ptr<float>();
    const float* k_ptr = K.data_ptr<float>();
    const float* v_ptr = V.data_ptr<float>();

    size_t kv_cache_stride = key_cache.dim(2) * head_dim;
    for (size_t h = 0; h < num_kv_heads; ++h) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t cache_offset = h * kv_cache_stride + (position_offset + s) * head_dim;
            size_t input_offset = (s * num_kv_heads + h) * head_dim;
            std::memcpy(k_cache_ptr + cache_offset, k_ptr + input_offset, head_dim * sizeof(float));
            std::memcpy(v_cache_ptr + cache_offset, v_ptr + input_offset, head_dim * sizeof(float));
        }
    }

    size_t total_seq_len = position_offset + seq_len;
    kv_cache.seq_len = total_seq_len;

    // Transpose Q from [batch, seq_len, num_heads, head_dim] to [batch, num_heads, seq_len, head_dim]
    Tensor Q_heads = Q.permute({0, 2, 1, 3});

    size_t heads_per_kv = num_heads / num_kv_heads;
    Tensor attn_output = Tensor::empty(Shape({batch, num_heads, seq_len, head_dim}), DType::FP32);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
        // Get K, V slices from cache - key_cache shape is [batch, num_kv_heads, max_seq_len, head_dim]
        Tensor K_cache_slice = key_cache.slice(1, kv_h, kv_h + 1);
        Tensor K_slice = Tensor::from_buffer(K_cache_slice.data(),
            Shape({batch, 1, key_cache.dim(2), head_dim}), DType::FP32);
        K_slice = K_slice.slice(2, 0, total_seq_len);

        Tensor V_cache_slice = value_cache.slice(1, kv_h, kv_h + 1);
        Tensor V_slice = Tensor::from_buffer(V_cache_slice.data(),
            Shape({batch, 1, value_cache.dim(2), head_dim}), DType::FP32);
        V_slice = V_slice.slice(2, 0, total_seq_len);

        for (size_t h_offset = 0; h_offset < heads_per_kv; ++h_offset) {
            size_t q_h = kv_h * heads_per_kv + h_offset;

            Tensor Q_head_slice = Q_heads.slice(1, q_h, q_h + 1);
            Tensor Q_head = Tensor::from_buffer(Q_head_slice.data(),
                Shape({batch, 1, seq_len, head_dim}), DType::FP32);

            Tensor out_slice = attn_output.slice(1, q_h, q_h + 1);
            Tensor out_head = Tensor::from_buffer(out_slice.data(),
                Shape({batch, 1, seq_len, head_dim}), DType::FP32);

            backend.flash_attention(Q_head, K_slice, V_slice, out_head, scale, true);
        }
    }

    // Transpose attn_output from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
    // then reshape to [batch, seq_len, num_heads * head_dim]
    Tensor attn_transposed = attn_output.permute({0, 2, 1, 3});
    Tensor attn_flat = Tensor::from_buffer(attn_transposed.data(),
        Shape({batch, seq_len, num_heads * head_dim}), DType::FP32);

    Tensor output = Tensor::empty(Shape({batch, seq_len, hidden_size}), DType::FP32);
    backend.gemm(attn_flat, o_proj_, output, 1.0f, 0.0f, false, true);

    return output;
}

QwenMLP::QwenMLP(const QwenConfig& config) : config_(config) {}

void QwenMLP::load_weights(const SafeTensorsFile& file, const std::string& prefix) {
    gate_proj_ = file.load_tensor_copy(prefix + ".gate_proj.weight");
    up_proj_ = file.load_tensor_copy(prefix + ".up_proj.weight");
    down_proj_ = file.load_tensor_copy(prefix + ".down_proj.weight");
}

Tensor QwenMLP::forward(Backend& backend, const Tensor& input) {
    size_t batch = input.dim(0);
    size_t seq_len = input.dim(1);
    size_t intermediate_size = config_.intermediate_size;
    size_t hidden_size = config_.hidden_size;

    Tensor gate = Tensor::empty(Shape({batch, seq_len, intermediate_size}), DType::FP32);
    Tensor up = Tensor::empty(Shape({batch, seq_len, intermediate_size}), DType::FP32);

    backend.gemm(input, gate_proj_, gate, 1.0f, 0.0f, false, true);
    backend.gemm(input, up_proj_, up, 1.0f, 0.0f, false, true);

    Tensor gate_activated = Tensor::empty(gate.shape(), DType::FP32);
    backend.silu(gate, gate_activated);

    Tensor gated = Tensor::empty(gate.shape(), DType::FP32);
    backend.elementwise_mul(gate_activated, up, gated);

    Tensor output = Tensor::empty(Shape({batch, seq_len, hidden_size}), DType::FP32);
    backend.gemm(gated, down_proj_, output, 1.0f, 0.0f, false, true);

    return output;
}

QwenBlock::QwenBlock(const QwenConfig& config)
    : config_(config), self_attn_(config), mlp_(config) {}

void QwenBlock::load_weights(const SafeTensorsFile& file, const std::string& prefix) {
    input_layernorm_.load_weights(file, prefix + ".input_layernorm");
    self_attn_.load_weights(file, prefix + ".self_attn");
    post_attention_layernorm_.load_weights(file, prefix + ".post_attention_layernorm");
    mlp_.load_weights(file, prefix + ".mlp");
}

Tensor QwenBlock::forward(
    Backend& backend,
    const Tensor& hidden_states,
    KVCache& kv_cache,
    size_t layer_idx,
    size_t position_offset
) {
    Tensor normed = Tensor::empty(hidden_states.shape(), DType::FP32);
    input_layernorm_.forward(backend, hidden_states, normed, config_.rms_norm_eps);

    Tensor attn_output = self_attn_.forward(backend, normed, kv_cache, layer_idx, position_offset);

    Tensor residual1 = Tensor::empty(hidden_states.shape(), DType::FP32);
    backend.add(hidden_states, attn_output, residual1);

    Tensor normed2 = Tensor::empty(residual1.shape(), DType::FP32);
    post_attention_layernorm_.forward(backend, residual1, normed2, config_.rms_norm_eps);

    Tensor mlp_output = mlp_.forward(backend, normed2);

    Tensor output = Tensor::empty(hidden_states.shape(), DType::FP32);
    backend.add(residual1, mlp_output, output);

    return output;
}

QwenModel::QwenModel(const QwenConfig& config) : config_(config) {
    layers_.reserve(config_.num_layers);
    for (size_t i = 0; i < config_.num_layers; ++i) {
        layers_.emplace_back(config_);
    }
}

void QwenModel::load_weights(const SafeTensorsFile& file) {
    embed_.load_weights(file, "model.embed_tokens");

    for (size_t i = 0; i < config_.num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        layers_[i].load_weights(file, prefix);
    }

    norm_.load_weights(file, "model.norm");

    // Handle tied embeddings: if lm_head.weight doesn't exist, use embed_tokens
    if (file.has_tensor("lm_head.weight")) {
        lm_head_ = file.load_tensor_copy("lm_head.weight");
    } else {
        // Tied embeddings - share weight with embedding layer
        lm_head_ = file.load_tensor_copy("model.embed_tokens.weight");
    }
}

Tensor QwenModel::forward(
    Backend& backend,
    const std::vector<int>& token_ids,
    KVCache& kv_cache
) {
    size_t position_offset = kv_cache.seq_len;

    Tensor hidden_states = embed_.forward(backend, token_ids);

    for (size_t i = 0; i < config_.num_layers; ++i) {
        hidden_states = layers_[i].forward(backend, hidden_states, kv_cache, i, position_offset);
    }

    Tensor normed = Tensor::empty(hidden_states.shape(), DType::FP32);
    norm_.forward(backend, hidden_states, normed, config_.rms_norm_eps);

    size_t batch = normed.dim(0);
    size_t seq_len = normed.dim(1);
    Tensor logits = Tensor::empty(Shape({batch, seq_len, config_.vocab_size}), DType::FP32);
    backend.gemm(normed, lm_head_, logits, 1.0f, 0.0f, false, true);

    return logits;
}

}  // namespace mruntime
