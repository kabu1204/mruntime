#include "mruntime/qwen2_generate_vk.h"

#include <algorithm>
#include <cassert>

#include "mruntime/qwen2_forward_vk.h"

#include "src/qwen2_vk_state.h"

namespace mruntime {

size_t qwen2_generate_vk(
    Qwen2VkState& state,
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

    for (size_t i = 0; i < prompt_len; ++i) {
        output_tokens[i] = prompt_tokens[i];
    }

    const uint16_t* last_logits = qwen2_prefill_vk(
        state,
        prompt_tokens,
        prompt_len,
        pool
    );

    int32_t next_token = qwen2_sample(last_logits, state.cfg.vocab_size, gen_cfg, &rng_state);
    output_tokens[prompt_len] = next_token;
    size_t total_len = prompt_len + 1;

    if (next_token == gen_cfg.eos_token_id) {
        return total_len;
    }

    for (size_t i = 1; i < gen_cfg.max_new_tokens; ++i) {
        const uint16_t* logits = qwen2_decode_vk(state, next_token, pool);
        next_token = qwen2_sample(logits, state.cfg.vocab_size, gen_cfg, &rng_state);

        output_tokens[total_len] = next_token;
        total_len++;

        if (next_token == gen_cfg.eos_token_id) {
            break;
        }
    }

    return total_len;
}

}  // namespace mruntime

