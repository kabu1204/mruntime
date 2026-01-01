#pragma once

#include <cstddef>
#include <string>

namespace mruntime {

struct QwenConfig {
    size_t vocab_size = 151936;
    size_t hidden_size = 896;
    size_t num_layers = 24;
    size_t num_attention_heads = 14;
    size_t num_kv_heads = 2;
    size_t intermediate_size = 4864;
    size_t max_position_embeddings = 32768;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    size_t head_dim() const { return hidden_size / num_attention_heads; }

    static QwenConfig from_json(const std::string& json);
};

}  // namespace mruntime
