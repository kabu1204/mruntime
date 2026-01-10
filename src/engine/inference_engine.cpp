#include "mruntime/inference_engine.h"
#include "mruntime/logging.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace mruntime {

InferenceEngine::InferenceEngine(QwenModel& model, Backend& backend)
    : model_(model), backend_(backend), rng_state_(42) {
        MRUNTIME_LOG_DEBUG("InferenceEngine initialized");
    }

float InferenceEngine::rand_float() {
    rng_state_ = rng_state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = ((rng_state_ >> 18) ^ rng_state_) >> 27;
    uint32_t rot = rng_state_ >> 59;
    uint32_t result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return static_cast<float>(result) / static_cast<float>(UINT32_MAX);
}

std::vector<int> InferenceEngine::generate(
    const std::vector<int>& prompt_tokens,
    const GenerationConfig& config
) {
    rng_state_ = config.seed;

    const QwenConfig& model_config = model_.config();
    size_t max_seq_len = prompt_tokens.size() + config.max_new_tokens;
    kv_cache_.allocate(model_config, max_seq_len);

    std::vector<int> output_tokens = prompt_tokens;

    if (prompt_tokens.empty()) {
        return output_tokens;
    }

    // NOTE: We prefill the KV-cache one token at a time to match the
    // incremental decoding path (at the cost of speed).
    Tensor logits;
    logits = model_.forward(backend_, {prompt_tokens[0]}, kv_cache_);
    for (size_t i = 1; i < prompt_tokens.size(); ++i) {
        logits = model_.forward(backend_, {prompt_tokens[i]}, kv_cache_);
    }

    for (size_t i = 0; i < config.max_new_tokens; ++i) {
        // Sample/select next token from the current logits (which correspond to
        // the last token in the context).
        int next_token = sample(logits, config);
        MRUNTIME_LOG_INFO("next_token: {}", next_token);
        output_tokens.push_back(next_token);

        if (next_token == config.eos_token_id) {
            break;
        }

        // Advance the KV-cache with the selected token, and get logits for the
        // next step.
        logits = model_.forward(backend_, {next_token}, kv_cache_);
    }

    return output_tokens;
}

int InferenceEngine::sample(const Tensor& logits, const GenerationConfig& config) {
    assert(logits.ndim() == 3);
    assert(logits.dim(0) == 1);
    assert(logits.dim(1) >= 1);

    const size_t vocab_size = logits.dim(2);
    const size_t seq_len = logits.dim(1);
    const size_t base = (seq_len - 1) * vocab_size;
    const void* logits_data = logits.data();
    const DType logits_dtype = logits.dtype();

    auto logit_at = [&](size_t i) -> float {
        return load_scalar_as_fp32(logits_data, logits_dtype, base + i);
    };

    if (config.greedy || config.temperature == 0.0f) {
        float best = -std::numeric_limits<float>::infinity();
        size_t best_idx = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            float v = logit_at(i);
            if (v > best) {
                best = v;
                best_idx = i;
            }
        }
        return static_cast<int>(best_idx);
    }

    std::vector<float> probs(vocab_size);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logit_at(i));
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = (logit_at(i) - max_logit) / config.temperature;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(probs[i]);
    }

    std::vector<size_t> sorted_indices;

    if (config.top_k > 0 && config.top_k < vocab_size) {
        std::vector<size_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + config.top_k, indices.end(),
            [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

        sorted_indices.assign(indices.begin(), indices.begin() + config.top_k);

        std::vector<float> filtered_probs(vocab_size, 0.0f);
        for (size_t i = 0; i < config.top_k; ++i) {
            filtered_probs[indices[i]] = probs[indices[i]];
        }
        probs = std::move(filtered_probs);
    } else {
        sorted_indices.resize(vocab_size);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
    }

    if (config.top_p > 0.0f && config.top_p < 1.0f) {
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
                if (cumsum >= config.top_p) {
                    ++keep;  // keep this token too
                    break;
                }
            }
            if (keep == 0) keep = 1;
            for (size_t j = keep; j < sorted_indices.size(); ++j) {
                probs[sorted_indices[j]] = 0.0f;
            }
        }
    }

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f) {
        // Fallback to greedy if filtering zeroed everything.
        float best = -std::numeric_limits<float>::infinity();
        size_t best_idx = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            float v = logit_at(i);
            if (v > best) {
                best = v;
                best_idx = i;
            }
        }
        return static_cast<int>(best_idx);
    }
    for (float& p : probs) {
        p /= sum;
    }

    float r = rand_float();
    float cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return static_cast<int>(i);
        }
    }

    return static_cast<int>(vocab_size - 1);
}

}  // namespace mruntime
