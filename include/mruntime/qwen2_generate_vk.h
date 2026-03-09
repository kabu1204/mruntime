#pragma once

#include <cstddef>
#include <cstdint>

#include "mruntime/qwen2_forward_vk.h"
#include "mruntime/qwen2_generate.h"

namespace mruntime {

size_t qwen2_generate_vk(
    Qwen2VkState& state,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    int32_t* output_tokens,
    const Qwen2GenerateConfig& gen_cfg,
    PThreadPool* pool
);

}  // namespace mruntime

