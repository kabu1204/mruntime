#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mruntime/pthreadpool_raii.h"
#include "mruntime/qwen_config.h"
#include "mruntime/qwen2_weights.h"

#if !defined(MRUNTIME_ENABLE_VULKAN)
#error "mruntime: Vulkan support is not enabled. Reconfigure with -DMRUNTIME_ENABLE_VULKAN=ON."
#endif

namespace mruntime {

struct Qwen2VkState;

struct Qwen2VkStateDeleter {
    void operator()(Qwen2VkState* state) const noexcept;
};

using Qwen2VkStatePtr = std::unique_ptr<Qwen2VkState, Qwen2VkStateDeleter>;

Qwen2VkStatePtr qwen2_vk_create(
    const QwenConfig& cfg,
    const Qwen2Weights& cpu_weights,
    size_t max_seq_len,
    size_t max_batch_tokens
);

void qwen2_vk_reset_kv_cache(Qwen2VkState& state);

// Enable/disable Vulkan dispatch timing enrichment in trace output (GPU kernel timestamps + CPU submit/wait).
// Note: Has no effect unless `TraceCollector` is enabled.
void qwen2_vk_set_timing_enabled(Qwen2VkState& state, bool enabled);

uint16_t* qwen2_forward_vk(
    Qwen2VkState& state,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
);

const uint16_t* qwen2_prefill_vk(
    Qwen2VkState& state,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    PThreadPool* pool
);

uint16_t* qwen2_decode_vk(
    Qwen2VkState& state,
    int32_t input_token,
    PThreadPool* pool
);

}  // namespace mruntime
