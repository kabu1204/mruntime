#pragma once

#include <memory>
#include <vector>

#include "mruntime/backend.h"
#include "mruntime/qwen_config.h"
#include "mruntime/safetensors.h"
#include "mruntime/tensor.h"

namespace mruntime {

struct KVCache {
    std::vector<Tensor> key_cache;
    std::vector<Tensor> value_cache;
    size_t seq_len = 0;

    void allocate(const QwenConfig& config, size_t max_seq_len, DType dtype = DType::FP32);
    void reset() { seq_len = 0; }
};

class QwenEmbedding {
public:
    void load_weights(const SafeTensorsFile& file, const std::string& prefix);
    Tensor forward(Backend& backend, const std::vector<int>& token_ids);

private:
    Tensor embed_tokens_;
};

class QwenRMSNorm {
public:
    void load_weights(const SafeTensorsFile& file, const std::string& name);
    void forward(Backend& backend, const Tensor& input, Tensor& output, float eps);

private:
    Tensor weight_;
};

class QwenRotaryEmbedding {
public:
    explicit QwenRotaryEmbedding(const QwenConfig& config);
    void apply(Backend& backend, Tensor& Q, Tensor& K, size_t position_offset);

private:
    float theta_;
};

class QwenAttention {
public:
    explicit QwenAttention(const QwenConfig& config);
    void load_weights(const SafeTensorsFile& file, const std::string& prefix);
    Tensor forward(
        Backend& backend,
        const Tensor& hidden_states,
        KVCache& kv_cache,
        size_t layer_idx,
        size_t position_offset
    );

private:
    QwenConfig config_;
    Tensor q_proj_, k_proj_, v_proj_, o_proj_;
    // Qwen2.5 uses biases for q/k/v projections (o_proj has no bias).
    // Store these as FP32 for compute convenience (they're tiny).
    Tensor q_bias_, k_bias_, v_bias_;
    QwenRotaryEmbedding rope_;
};

class QwenMLP {
public:
    explicit QwenMLP(const QwenConfig& config);
    void load_weights(const SafeTensorsFile& file, const std::string& prefix);
    Tensor forward(Backend& backend, const Tensor& input);

private:
    QwenConfig config_;
    Tensor gate_proj_, up_proj_, down_proj_;
};

class QwenBlock {
public:
    explicit QwenBlock(const QwenConfig& config);
    void load_weights(const SafeTensorsFile& file, const std::string& prefix);
    Tensor forward(
        Backend& backend,
        const Tensor& hidden_states,
        KVCache& kv_cache,
        size_t layer_idx,
        size_t position_offset
    );

private:
    QwenConfig config_;
    QwenRMSNorm input_layernorm_;
    QwenAttention self_attn_;
    QwenRMSNorm post_attention_layernorm_;
    QwenMLP mlp_;
};

class QwenModel {
public:
    explicit QwenModel(const QwenConfig& config);
    void load_weights(const SafeTensorsFile& file);
    Tensor forward(
        Backend& backend,
        const std::vector<int>& token_ids,
        KVCache& kv_cache
    );
    const QwenConfig& config() const { return config_; }

private:
    QwenConfig config_;
    QwenEmbedding embed_;
    std::vector<QwenBlock> layers_;
    QwenRMSNorm norm_;
    Tensor lm_head_;
};

}  // namespace mruntime
