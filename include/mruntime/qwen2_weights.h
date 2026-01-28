#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "mruntime/arena.h"
#include "mruntime/qwen_config.h"

namespace mruntime {

class SafeTensorsFile;

// Precomputed RoPE cos/sin cache for a single model configuration.
// cos_sin layout: [max_seq_len, half_dim, 2] interleaved (cos, sin).
struct Qwen2RopeCache {
    const float* inv_freq = nullptr;  // [half_dim]
    const float* cos_sin = nullptr;   // [max_seq_len * half_dim * 2]
    size_t max_seq_len = 0;
    size_t head_dim = 0;
    size_t half_dim = 0;
    float theta = 0.0f;
};

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
    // Fused (gate + up) projection weights: [2 * intermediate_size, hidden_size].
    // Row-major, with first intermediate_size rows = gate, second = up.
    const uint16_t* gate_up_proj;
    const uint16_t* down_proj;          // [hidden_size, intermediate_size]

    // Pre-packed weights for KleidiAI (may be nullptr if not packed)
    const uint16_t* q_proj_packed;
    const uint16_t* k_proj_packed;
    const uint16_t* v_proj_packed;
    const uint16_t* o_proj_packed;
    // Pre-packed concatenation of gate+up along N: [2*intermediate_size, hidden_size].
    // When present, qwen2_mlp can compute both projections in one GEMM.
    const uint16_t* gate_up_proj_packed;
    const uint16_t* down_proj_packed;
};

// All model weights (pointers into weights arena)
struct Qwen2Weights {
    const uint16_t* embed_tokens;       // [vocab_size, hidden_size]
    const uint16_t* final_norm;         // [hidden_size]
    const uint16_t* lm_head;            // [vocab_size, hidden_size]
    const uint16_t* lm_head_packed;     // KleidiAI packed RHS (may be nullptr)

    Qwen2LayerWeights* layers;          // [num_layers]
    size_t num_layers;
};

// KV cache structure (pointers into kv_cache arena)
struct Qwen2KVCache {
    uint16_t* k_cache;                  // [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
    uint16_t* v_cache;                  // [num_layers, batch, num_kv_heads, max_seq_len, head_dim]
    size_t seq_len;                     // Current sequence length in cache
    size_t max_seq_len;                 // Maximum sequence length
    Qwen2RopeCache rope;                // Precomputed RoPE cos/sin table

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
    uint16_t* up;               // [max_tokens, 2 * intermediate_size] (gate+up fused GEMM output; first half is gate)
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

// Precompute RoPE tables in `arena` and return a cache referencing them.
Qwen2RopeCache qwen2_init_rope_cache(
    size_t head_dim,
    size_t max_seq_len,
    float theta,
    Arena& arena
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
