#pragma once

#include <cstddef>
#include <cstdint>

#include "mruntime/pthreadpool_raii.h"
#include "mruntime/qwen_config.h"
#include "mruntime/qwen2_weights.h"

namespace mruntime {

// ============================================================================
// Generation Configuration
// ============================================================================

struct Qwen2GenerateConfig {
    size_t max_new_tokens = 100;
    float temperature = 1.0f;
    size_t top_k = 50;
    float top_p = 1.0f;
    int32_t eos_token_id = 151643;  // Qwen2 default
    bool greedy = true;
    uint64_t seed = 42;
};

// ============================================================================
// Generation Functions
// ============================================================================

// Full generation: prompt tokens -> output tokens
// Returns number of generated tokens (including prompt)
// output_tokens buffer must have space for at least prompt_len + max_new_tokens
size_t qwen2_generate(
    const QwenConfig& cfg,
    const Qwen2Weights& weights,
    Qwen2KVCache& kv_cache,
    Qwen2Scratch& scratch,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    int32_t* output_tokens,
    const Qwen2GenerateConfig& gen_cfg,
    PThreadPool* pool
);

// ============================================================================
// Sampling Functions
// ============================================================================

// Sample from logits, returns token id
// logits: [vocab_size] - logits for a single position
int32_t qwen2_sample(
    const uint16_t* logits,
    size_t vocab_size,
    const Qwen2GenerateConfig& cfg,
    uint64_t* rng_state
);

// Greedy argmax (fast path for greedy decoding)
int32_t qwen2_argmax_fp16(
    const uint16_t* logits,
    size_t vocab_size
);

}  // namespace mruntime
