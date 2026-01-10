#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mruntime/backend.h"
#include "mruntime/qwen_model.h"

namespace mruntime {

struct GenerationConfig {
    size_t max_new_tokens = 100;
    // Default to deterministic generation (greedy).
    // Sampling can be enabled by setting `greedy=false` (and using temperature/top-k/top-p).
    float temperature = 1.0f;
    size_t top_k = 50;
    float top_p = 1.0f;
    int eos_token_id = 151643;
    bool greedy = true;
    uint64_t seed = 42;
};

class InferenceEngine {
public:
    InferenceEngine(QwenModel& model, Backend& backend);

    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config
    );

private:
    int sample(const Tensor& logits, const GenerationConfig& config);

    QwenModel& model_;
    Backend& backend_;
    KVCache kv_cache_;

    uint64_t rng_state_;
    float rand_float();
};

}  // namespace mruntime
