#pragma once

#include <cstddef>
#include <cstdint>

#include "mruntime/pthreadpool_raii.h"
#include "mruntime/qwen_config.h"
#include "mruntime/qwen2_weights.h"

namespace mruntime {

// ============================================================================
// Qwen2 Forward Pass - Free Functions
// ============================================================================

// Full forward pass: tokens -> logits
// Returns pointer to logits in scratch buffer [num_tokens, vocab_size]
// Note: Only the last token's logits are typically used for generation
uint16_t* qwen2_forward(
    const QwenConfig& cfg,
    const Qwen2Weights& weights,
    Qwen2KVCache& kv_cache,
    Qwen2Scratch& scratch,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
);

// ============================================================================
// Layer-Level Functions (for debugging/profiling)
// ============================================================================

// Single transformer layer forward pass
// Input: hidden_in [num_tokens, hidden_size]
// Output: hidden_out [num_tokens, hidden_size]
void qwen2_layer_forward(
    const QwenConfig& cfg,
    const Qwen2LayerWeights& layer,
    uint16_t* k_cache,            // This layer's K cache [num_kv_heads, max_seq_len, head_dim]
    uint16_t* v_cache,            // This layer's V cache [num_kv_heads, max_seq_len, head_dim]
    size_t kv_seq_len,            // Current KV cache length (before this forward)
    size_t max_seq_len,           // Maximum sequence length (cache dimension)
    const uint16_t* hidden_in,    // [num_tokens, hidden_size]
    uint16_t* hidden_out,         // [num_tokens, hidden_size]
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
);

// Attention block only
// Input: normed_input [num_tokens, hidden_size]
// Output: attn_output [num_tokens, hidden_size]
void qwen2_attention(
    const QwenConfig& cfg,
    const Qwen2LayerWeights& layer,
    uint16_t* k_cache,            // [num_kv_heads, max_seq_len, head_dim]
    uint16_t* v_cache,            // [num_kv_heads, max_seq_len, head_dim]
    size_t kv_seq_len,            // Current KV cache length
    size_t max_seq_len,           // Maximum sequence length (cache dimension)
    const uint16_t* normed_input, // [num_tokens, hidden_size]
    uint16_t* attn_output,        // [num_tokens, hidden_size]
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
);

// MLP block only
// Input: normed_input [num_tokens, hidden_size]
// Output: mlp_output [num_tokens, hidden_size]
void qwen2_mlp(
    const QwenConfig& cfg,
    const Qwen2LayerWeights& layer,
    const uint16_t* normed_input, // [num_tokens, hidden_size]
    uint16_t* mlp_output,         // [num_tokens, hidden_size]
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
);

}  // namespace mruntime
