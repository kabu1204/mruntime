#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "mruntime/arena.h"
#include "mruntime/qwen_config.h"

namespace mruntime {

class SafeTensorsFile;

// Per-layer weight pointers (all FP16)
struct Qwen2LayerWeights {
    const uint16_t* input_norm;         // [hidden_size]
    const uint16_t* q_proj;             // [num_heads * head_dim, hidden_size]
    const uint16_t* k_proj;             // [num_kv_heads * head_dim, hidden_size]
    const uint16_t* v_proj;             // [num_kv_heads * head_dim, hidden_size]
    const uint16_t* o_proj;             // [hidden_size, num_heads * head_dim]
    const uint16_t* q_bias;             // [num_heads * head_dim] (may be nullptr)
    const uint16_t* k_bias;             // [num_kv_heads * head_dim] (may be nullptr)
    const uint16_t* v_bias;             // [num_kv_heads * head_dim] (may be nullptr)
    const uint16_t* post_attn_norm;     // [hidden_size]
    const uint16_t* gate_proj;          // [intermediate_size, hidden_size]
    const uint16_t* up_proj;            // [intermediate_size, hidden_size]
    const uint16_t* down_proj;          // [hidden_size, intermediate_size]

    // Pre-packed weights for KleidiAI (may be nullptr if not packed)
    const uint16_t* q_proj_packed;
    const uint16_t* k_proj_packed;
    const uint16_t* v_proj_packed;
    const uint16_t* o_proj_packed;
    const uint16_t* gate_proj_packed;
    const uint16_t* up_proj_packed;
    const uint16_t* down_proj_packed;
};

// All model weights (pointers into weights arena)
struct Qwen2Weights {
    const uint16_t* embed_tokens;       // [vocab_size, hidden_size]
    const uint16_t* final_norm;         // [hidden_size]
    const uint16_t* lm_head;            // [vocab_size, hidden_size]

    Qwen2LayerWeights* layers;          // [num_layers]
    size_t num_layers;
};

// KV cache structure (pointers into kv_cache arena)
struct Qwen2KVCache {
    uint16_t* k_cache;                  // [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
    uint16_t* v_cache;                  // [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
    size_t seq_len;                     // Current sequence length in cache
    size_t max_seq_len;                 // Maximum sequence length

    // Helper to get per-layer cache pointers
    uint16_t* k_layer(size_t layer, size_t num_kv_heads, size_t head_dim) const {
        return k_cache + layer * num_kv_heads * max_seq_len * head_dim;
    }
    uint16_t* v_layer(size_t layer, size_t num_kv_heads, size_t head_dim) const {
        return v_cache + layer * num_kv_heads * max_seq_len * head_dim;
    }
};

// Scratch buffers for forward pass (pointers into scratch arena)
struct Qwen2Scratch {
    uint16_t* hidden;           // [max_tokens, hidden_size]
    uint16_t* residual;         // [max_tokens, hidden_size]
    uint16_t* normed;           // [max_tokens, hidden_size]
    uint16_t* q_proj;           // [max_tokens, num_heads * head_dim]
    uint16_t* k_proj;           // [max_tokens, num_kv_heads * head_dim]
    uint16_t* v_proj;           // [max_tokens, num_kv_heads * head_dim]
    uint16_t* q_transposed;     // [max_tokens * num_heads * head_dim] for BHSD layout
    uint16_t* attn_out;         // [max_tokens, num_heads * head_dim]
    uint16_t* gate;             // [max_tokens, intermediate_size]
    uint16_t* up;               // [max_tokens, intermediate_size]
    uint16_t* mlp_out;          // [max_tokens, hidden_size]
    uint16_t* logits;           // [max_tokens, vocab_size]

    size_t max_tokens;          // Maximum tokens per forward pass
};

// ============================================================================
// Memory Size Calculation
// ============================================================================

struct Qwen2MemorySizes {
    size_t weights_bytes;
    size_t kv_cache_bytes;
    size_t scratch_bytes;
    size_t packed_weights_bytes;  // Additional space for pre-packed weights (optional)
};

// Calculate memory requirements
Qwen2MemorySizes qwen2_memory_sizes(
    const QwenConfig& cfg,
    size_t max_seq_len,
    size_t max_batch_tokens = 32  // Max tokens per forward call
);

// ============================================================================
// Initialization Functions
// ============================================================================

// Load weights from SafeTensors file into arena.
// BF16 weights are converted to FP16 during load.
// Returns Qwen2Weights with pointers into the arena.
Qwen2Weights qwen2_load_weights(
    const QwenConfig& cfg,
    const SafeTensorsFile& file,
    Arena& weights_arena,
    bool pack_for_kai = true    // Pre-pack weights for KleidiAI
);

// Initialize KV cache in arena
Qwen2KVCache qwen2_init_kv_cache(
    const QwenConfig& cfg,
    Arena& kv_arena,
    size_t max_seq_len
);

// Initialize scratch buffers in arena
Qwen2Scratch qwen2_init_scratch(
    const QwenConfig& cfg,
    Arena& scratch_arena,
    size_t max_tokens = 32
);

// Reset KV cache for new sequence (keeps memory allocated)
inline void qwen2_reset_kv_cache(Qwen2KVCache& kv) {
    kv.seq_len = 0;
}

// Reset scratch buffers (keeps memory allocated)
inline void qwen2_reset_scratch(Arena& scratch_arena) {
    scratch_arena.reset();
}

}  // namespace mruntime
