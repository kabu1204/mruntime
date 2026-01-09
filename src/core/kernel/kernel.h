#pragma once

#include <cstddef>
#include <cstdint>

namespace mruntime {

struct FlashAttentionArgs {
    int batch_size;
    int num_heads;
    int q_sequence_length;
    int kv_sequence_length;
    int qk_head_size;
    int v_head_size;
    int q_block_size;
    int kv_block_size;
    float scale;
    const float* query;
    const float* key;
    const float* value;
    float* output;
    int strides_q[4];
    int strides_k[4];
    int strides_v[4];
    int strides_out[4];
};

struct FlashAttentionStridedContext {
    const FlashAttentionArgs* args = nullptr;
    size_t task_count = 0;
    float* buffer = nullptr;
    size_t buffer_size_per_task = 0;  // bytes
};

void FlashAttentionStrided(void* argptr, size_t task_id);

}  // namespace mruntime