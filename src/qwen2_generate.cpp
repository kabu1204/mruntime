#include "mruntime/qwen2_generate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "mruntime/dtype.h"
#include "mruntime/qwen2_forward.h"

namespace mruntime {

// ============================================================================
// RNG Helper
// ============================================================================

namespace {

float rand_float(uint64_t* rng_state) {
    // PCG-style RNG
    *rng_state = (*rng_state) * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = ((*rng_state >> 18) ^ *rng_state) >> 27;
    uint32_t rot = *rng_state >> 59;
    uint32_t result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return static_cast<float>(result) / static_cast<float>(UINT32_MAX);
}

}  // namespace

// ============================================================================
// Sampling
// ============================================================================

int32_t qwen2_argmax_fp16(
    const uint16_t* logits,
    size_t vocab_size
) {
    float best = -std::numeric_limits<float>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i < vocab_size; ++i) {
        float v = fp16_bits_to_float(logits[i]);
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }

    return static_cast<int32_t>(best_idx);
}

int32_t qwen2_sample(
    const uint16_t* logits,
    size_t vocab_size,
    const Qwen2GenerateConfig& cfg,
    uint64_t* rng_state
) {
    // Fast path: greedy decoding
    if (cfg.greedy || cfg.temperature == 0.0f) {
        return qwen2_argmax_fp16(logits, vocab_size);
    }

    // Find max for numerical stability
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, fp16_bits_to_float(logits[i]));
    }

    // Compute scaled exp (softmax numerator)
    std::vector<float> probs(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((fp16_bits_to_float(logits[i]) - max_logit) / cfg.temperature);
    }

    // Top-k filtering
    std::vector<size_t> sorted_indices;
    if (cfg.top_k > 0 && cfg.top_k < vocab_size) {
        std::vector<size_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + cfg.top_k, indices.end(),
            [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

        sorted_indices.assign(indices.begin(), indices.begin() + cfg.top_k);

        // Zero out non-top-k probs
        std::vector<float> filtered_probs(vocab_size, 0.0f);
        for (size_t i = 0; i < cfg.top_k; ++i) {
            filtered_probs[indices[i]] = probs[indices[i]];
        }
        probs = std::move(filtered_probs);
    } else {
        sorted_indices.resize(vocab_size);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
    }

    // Top-p (nucleus) filtering
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        float total = 0.0f;
        for (size_t idx : sorted_indices) {
            total += probs[idx];
        }
        if (total > 0.0f) {
            float cumsum = 0.0f;
            size_t keep = 0;
            for (; keep < sorted_indices.size(); ++keep) {
                size_t idx = sorted_indices[keep];
                cumsum += probs[idx] / total;
                if (cumsum >= cfg.top_p) {
                    ++keep;
                    break;
                }
            }
            if (keep == 0) keep = 1;
            for (size_t j = keep; j < sorted_indices.size(); ++j) {
                probs[sorted_indices[j]] = 0.0f;
            }
        }
    }

    // Normalize
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f) {
        // Fallback to greedy
        return qwen2_argmax_fp16(logits, vocab_size);
    }
    for (float& p : probs) {
        p /= sum;
    }

    // Sample
    float r = rand_float(rng_state);
    float cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return static_cast<int32_t>(i);
        }
    }

    return static_cast<int32_t>(vocab_size - 1);
}

// ============================================================================
// Prefill
// ============================================================================

const uint16_t* qwen2_prefill(
    const QwenConfig& cfg,
    const Qwen2Weights& weights,
    Qwen2KVCache& kv_cache,
    Qwen2Scratch& scratch,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    PThreadPool* pool
) {
    if (prompt_len == 0) {
        return nullptr;
    }

    // Reset KV cache for new sequence
    qwen2_reset_kv_cache(kv_cache);

    size_t processed = 0;
    const uint16_t* last_logits = nullptr;

    // Process prompt in chunks to stay within scratch.max_tokens
    while (processed < prompt_len) {
        const size_t chunk = std::min(scratch.max_tokens, prompt_len - processed);

        uint16_t* logits = qwen2_forward(
            cfg,
            weights,
            kv_cache,
            scratch,
            prompt_tokens + processed,
            chunk,
            pool
        );

        processed += chunk;
        last_logits = logits + (chunk - 1) * cfg.vocab_size;
    }

    return last_logits;
}

// ============================================================================
// Decode Step
// ============================================================================

int32_t qwen2_decode_step(
    const QwenConfig& cfg,
    const Qwen2Weights& weights,
    Qwen2KVCache& kv_cache,
    Qwen2Scratch& scratch,
    int32_t input_token,
    const Qwen2GenerateConfig& gen_cfg,
    uint64_t* rng_state,
    PThreadPool* pool
) {
    // Forward pass for single token
    uint16_t* logits = qwen2_forward(
        cfg,
        weights,
        kv_cache,
        scratch,
        &input_token,
        1,
        pool
    );

    // Sample next token from logits
    return qwen2_sample(logits, cfg.vocab_size, gen_cfg, rng_state);
}

// ============================================================================
// Full Generation
// ============================================================================

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
) {
    if (prompt_len == 0) {
        return 0;
    }

    uint64_t rng_state = gen_cfg.seed;

    // Copy prompt to output
    for (size_t i = 0; i < prompt_len; ++i) {
        output_tokens[i] = prompt_tokens[i];
    }

    // Prefill: process prompt tokens
    const uint16_t* last_logits = qwen2_prefill(
        cfg,
        weights,
        kv_cache,
        scratch,
        prompt_tokens,
        prompt_len,
        pool
    );

    // Sample first new token
    int32_t next_token = qwen2_sample(last_logits, cfg.vocab_size, gen_cfg, &rng_state);
    output_tokens[prompt_len] = next_token;
    size_t total_len = prompt_len + 1;

    // Check for EOS
    if (next_token == gen_cfg.eos_token_id) {
        return total_len;
    }

    // Autoregressive decoding
    for (size_t i = 1; i < gen_cfg.max_new_tokens; ++i) {
        next_token = qwen2_decode_step(
            cfg,
            weights,
            kv_cache,
            scratch,
            next_token,
            gen_cfg,
            &rng_state,
            pool
        );

        output_tokens[total_len] = next_token;
        total_len++;

        if (next_token == gen_cfg.eos_token_id) {
            break;
        }
    }

    return total_len;
}

}  // namespace mruntime
