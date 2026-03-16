#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mruntime/qwen_config.h"
#include "mruntime/qwen2_weights.h"

#include "vk_buffer_arena.h"
#include "vk_context.h"
#include "vk_fp16_ops.h"
#include "vk_kernel_runtime.h"

namespace mruntime {

struct Qwen2VkState {
    QwenConfig cfg;
    size_t max_seq_len = 0;
    size_t max_batch_tokens = 0;

    std::vector<uint16_t> embed_tokens_cpu;

    Qwen2Weights weights = {};
    std::vector<Qwen2LayerWeights> layer_weights;

    Qwen2KVCache kv_cache = {};
    Qwen2Scratch scratch = {};
    uint16_t* decode_partial_out = nullptr;
    float* decode_partial_stats = nullptr;

    vulkan::VkContext vk;
    vulkan::VkKernelRuntime runtime;
    vulkan::VkFp16Ops fp16_ops;

    vulkan::VkBufferArena weights_arena;
    vulkan::VkBufferArena kv_arena;
    vulkan::VkBufferArena scratch_arena;

    VkDeviceSize alignment = 0;
};

}  // namespace mruntime
