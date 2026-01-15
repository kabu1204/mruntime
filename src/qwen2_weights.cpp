#include "mruntime/qwen2_weights.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

#include "mruntime/dtype.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/safetensors.h"
#include "mruntime/trace.h"

namespace mruntime {

// ============================================================================
// Memory Size Calculation
// ============================================================================

Qwen2MemorySizes qwen2_memory_sizes(
    const QwenConfig& cfg,
    size_t max_seq_len,
    size_t max_batch_tokens
) {
    Qwen2MemorySizes sizes = {};
    const size_t elem_size = sizeof(uint16_t);  // FP16

    // Weight sizes
    size_t embed_size = cfg.vocab_size * cfg.hidden_size * elem_size;
    size_t final_norm_size = cfg.hidden_size * elem_size;
    size_t lm_head_size = cfg.vocab_size * cfg.hidden_size * elem_size;

    size_t head_dim = cfg.head_dim();
    size_t q_proj_size = cfg.num_attention_heads * head_dim * cfg.hidden_size * elem_size;
    size_t kv_proj_size = cfg.num_kv_heads * head_dim * cfg.hidden_size * elem_size;
    size_t o_proj_size = cfg.hidden_size * cfg.num_attention_heads * head_dim * elem_size;
    size_t gate_proj_size = cfg.intermediate_size * cfg.hidden_size * elem_size;
    size_t up_proj_size = cfg.intermediate_size * cfg.hidden_size * elem_size;
    size_t down_proj_size = cfg.hidden_size * cfg.intermediate_size * elem_size;
    size_t norm_size = cfg.hidden_size * elem_size;

    // Bias sizes (optional, may be zero if not present)
    size_t q_bias_size = cfg.num_attention_heads * head_dim * elem_size;
    size_t kv_bias_size = cfg.num_kv_heads * head_dim * elem_size;

    size_t layer_weights_size = 2 * norm_size + q_proj_size + 2 * kv_proj_size + o_proj_size +
                                gate_proj_size + up_proj_size + down_proj_size +
                                q_bias_size + 2 * kv_bias_size;

    // Add space for layer pointers array
    size_t layer_pointers_size = cfg.num_layers * sizeof(Qwen2LayerWeights);

    sizes.weights_bytes = embed_size + final_norm_size + lm_head_size +
                          cfg.num_layers * layer_weights_size + layer_pointers_size;

    // KV cache: [num_layers, batch=1, num_kv_heads, max_seq_len, head_dim] for both K and V
    size_t kv_per_layer = cfg.num_kv_heads * max_seq_len * head_dim * elem_size;
    sizes.kv_cache_bytes = 2 * cfg.num_layers * kv_per_layer;  // K and V

    // Scratch space for forward pass
    size_t hidden_buf = max_batch_tokens * cfg.hidden_size * elem_size;
    size_t q_proj_buf = max_batch_tokens * cfg.num_attention_heads * head_dim * elem_size;
    size_t kv_proj_buf = max_batch_tokens * cfg.num_kv_heads * head_dim * elem_size;
    size_t attn_out_buf = max_batch_tokens * cfg.num_attention_heads * head_dim * elem_size;
    size_t gate_buf = max_batch_tokens * cfg.intermediate_size * elem_size;
    size_t logits_buf = max_batch_tokens * cfg.vocab_size * elem_size;

    sizes.scratch_bytes = hidden_buf * 3 +  // hidden, residual, normed
                          q_proj_buf * 2 +  // q_proj, q_transposed
                          kv_proj_buf * 2 + // k_proj, v_proj
                          attn_out_buf +
                          gate_buf * 2 +    // gate, up
                          hidden_buf +      // mlp_out
                          logits_buf;

    // Packed weights (optional, for KleidiAI)
    // Each packed weight is roughly the same size as unpacked
    sizes.packed_weights_bytes = cfg.num_layers * (
        qwen2_packed_weight_size_fp16(cfg.num_attention_heads * head_dim, cfg.hidden_size) +
        2 * qwen2_packed_weight_size_fp16(cfg.num_kv_heads * head_dim, cfg.hidden_size) +
        qwen2_packed_weight_size_fp16(cfg.hidden_size, cfg.num_attention_heads * head_dim) +
        qwen2_packed_weight_size_fp16(cfg.intermediate_size, cfg.hidden_size) +
        qwen2_packed_weight_size_fp16(cfg.intermediate_size, cfg.hidden_size) +
        qwen2_packed_weight_size_fp16(cfg.hidden_size, cfg.intermediate_size)
    );
    sizes.packed_weights_bytes += qwen2_packed_weight_size_fp16(cfg.vocab_size, cfg.hidden_size);

    return sizes;
}

// ============================================================================
// Internal Helper: Load tensor data into arena
// ============================================================================

namespace {

// Copy tensor data from SafeTensors file into arena, converting BF16 to FP16 if needed
uint16_t* load_tensor_to_arena(
    const SafeTensorsFile& file,
    const std::string& name,
    Arena& arena,
    size_t expected_numel
) {
    if (!file.has_tensor(name)) {
        return nullptr;
    }

    const TensorInfo& info = file.tensor_info(name);

    // Calculate numel from shape
    size_t numel = 1;
    for (size_t dim : info.shape) {
        numel *= dim;
    }
    assert(numel == expected_numel);
    (void)expected_numel;  // Suppress unused warning in release

    uint16_t* dst = arena.alloc_array<uint16_t>(numel);

    // Get raw pointer to mmap'd data (zero-copy access)
    const void* src = file.tensor_data(name);

    if (info.dtype == DType::FP16) {
        // Direct copy for FP16
        std::memcpy(dst, src, numel * sizeof(uint16_t));
    } else if (info.dtype == DType::BF16) {
        // Convert BF16 to FP16
        const uint16_t* src_bf16 = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < numel; ++i) {
            float f = bf16_to_float(src_bf16[i]);
            dst[i] = float_to_fp16_bits(f);
        }
    } else if (info.dtype == DType::FP32) {
        // Convert FP32 to FP16
        const float* src_fp32 = static_cast<const float*>(src);
        for (size_t i = 0; i < numel; ++i) {
            dst[i] = float_to_fp16_bits(src_fp32[i]);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for tensor: " + name);
    }

    return dst;
}

// Try to load bias (may not exist)
uint16_t* try_load_bias(
    const SafeTensorsFile& file,
    const std::string& name,
    Arena& arena,
    size_t expected_numel
) {
    if (!file.has_tensor(name)) {
        return nullptr;
    }
    return load_tensor_to_arena(file, name, arena, expected_numel);
}

}  // namespace

// ============================================================================
// Weight Loading
// ============================================================================

Qwen2Weights qwen2_load_weights(
    const QwenConfig& cfg,
    const SafeTensorsFile& file,
    Arena& weights_arena,
    bool pack_for_kai
) {
    Qwen2Weights weights = {};
    weights.num_layers = cfg.num_layers;

    size_t head_dim = cfg.head_dim();

    // Load embedding
    weights.embed_tokens = load_tensor_to_arena(
        file, "model.embed_tokens.weight", weights_arena,
        cfg.vocab_size * cfg.hidden_size
    );

    // Load final norm
    weights.final_norm = load_tensor_to_arena(
        file, "model.norm.weight", weights_arena,
        cfg.hidden_size
    );

    // Load lm_head (may be tied to embed_tokens)
    if (file.has_tensor("lm_head.weight")) {
        weights.lm_head = load_tensor_to_arena(
            file, "lm_head.weight", weights_arena,
            cfg.vocab_size * cfg.hidden_size
        );
    } else {
        // Tied embeddings - reuse embed_tokens
        weights.lm_head = weights.embed_tokens;
    }

    // Pack lm_head for KleidiAI.
    weights.lm_head_packed = nullptr;
    if (pack_for_kai) {
        TRACE_SCOPE("kai_pack_lm_head");
        const size_t packed_size_bytes = qwen2_packed_weight_size_fp16(cfg.vocab_size, cfg.hidden_size);
        uint16_t* packed = weights_arena.alloc_array<uint16_t>(packed_size_bytes / sizeof(uint16_t));
        qwen2_pack_weight_fp16(weights.lm_head, packed, cfg.vocab_size, cfg.hidden_size);
        weights.lm_head_packed = packed;
    }

    // Allocate layer weights array
    weights.layers = weights_arena.alloc_array<Qwen2LayerWeights>(cfg.num_layers);

    // Load each layer
    for (size_t i = 0; i < cfg.num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        Qwen2LayerWeights& layer = weights.layers[i];

        // Input norm
        layer.input_norm = load_tensor_to_arena(
            file, prefix + ".input_layernorm.weight", weights_arena,
            cfg.hidden_size
        );

        // Attention projections
        layer.q_proj = load_tensor_to_arena(
            file, prefix + ".self_attn.q_proj.weight", weights_arena,
            cfg.num_attention_heads * head_dim * cfg.hidden_size
        );
        layer.k_proj = load_tensor_to_arena(
            file, prefix + ".self_attn.k_proj.weight", weights_arena,
            cfg.num_kv_heads * head_dim * cfg.hidden_size
        );
        layer.v_proj = load_tensor_to_arena(
            file, prefix + ".self_attn.v_proj.weight", weights_arena,
            cfg.num_kv_heads * head_dim * cfg.hidden_size
        );
        layer.o_proj = load_tensor_to_arena(
            file, prefix + ".self_attn.o_proj.weight", weights_arena,
            cfg.hidden_size * cfg.num_attention_heads * head_dim
        );

        // Biases (optional)
        layer.q_bias = try_load_bias(
            file, prefix + ".self_attn.q_proj.bias", weights_arena,
            cfg.num_attention_heads * head_dim
        );
        layer.k_bias = try_load_bias(
            file, prefix + ".self_attn.k_proj.bias", weights_arena,
            cfg.num_kv_heads * head_dim
        );
        layer.v_bias = try_load_bias(
            file, prefix + ".self_attn.v_proj.bias", weights_arena,
            cfg.num_kv_heads * head_dim
        );

        // Post-attention norm
        layer.post_attn_norm = load_tensor_to_arena(
            file, prefix + ".post_attention_layernorm.weight", weights_arena,
            cfg.hidden_size
        );

        // MLP projections
        layer.gate_proj = load_tensor_to_arena(
            file, prefix + ".mlp.gate_proj.weight", weights_arena,
            cfg.intermediate_size * cfg.hidden_size
        );
        layer.up_proj = load_tensor_to_arena(
            file, prefix + ".mlp.up_proj.weight", weights_arena,
            cfg.intermediate_size * cfg.hidden_size
        );
        layer.down_proj = load_tensor_to_arena(
            file, prefix + ".mlp.down_proj.weight", weights_arena,
            cfg.hidden_size * cfg.intermediate_size
        );

        // Initialize packed pointers to nullptr
        layer.q_proj_packed = nullptr;
        layer.k_proj_packed = nullptr;
        layer.v_proj_packed = nullptr;
        layer.o_proj_packed = nullptr;
        layer.gate_proj_packed = nullptr;
        layer.up_proj_packed = nullptr;
        layer.down_proj_packed = nullptr;

        // Pack weights for KleidiAI if requested
        if (pack_for_kai) {
            TRACE_SCOPE("kai_pack_weights");
            // Q projection: [num_heads * head_dim, hidden_size]
            size_t q_packed_size = qwen2_packed_weight_size_fp16(
                cfg.num_attention_heads * head_dim, cfg.hidden_size);
            uint16_t* q_packed = weights_arena.alloc_array<uint16_t>(q_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.q_proj, q_packed,
                cfg.num_attention_heads * head_dim, cfg.hidden_size);
            layer.q_proj_packed = q_packed;

            // K projection: [num_kv_heads * head_dim, hidden_size]
            size_t k_packed_size = qwen2_packed_weight_size_fp16(
                cfg.num_kv_heads * head_dim, cfg.hidden_size);
            uint16_t* k_packed = weights_arena.alloc_array<uint16_t>(k_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.k_proj, k_packed,
                cfg.num_kv_heads * head_dim, cfg.hidden_size);
            layer.k_proj_packed = k_packed;

            // V projection: [num_kv_heads * head_dim, hidden_size]
            size_t v_packed_size = qwen2_packed_weight_size_fp16(
                cfg.num_kv_heads * head_dim, cfg.hidden_size);
            uint16_t* v_packed = weights_arena.alloc_array<uint16_t>(v_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.v_proj, v_packed,
                cfg.num_kv_heads * head_dim, cfg.hidden_size);
            layer.v_proj_packed = v_packed;

            // O projection: [hidden_size, num_heads * head_dim]
            size_t o_packed_size = qwen2_packed_weight_size_fp16(
                cfg.hidden_size, cfg.num_attention_heads * head_dim);
            uint16_t* o_packed = weights_arena.alloc_array<uint16_t>(o_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.o_proj, o_packed,
                cfg.hidden_size, cfg.num_attention_heads * head_dim);
            layer.o_proj_packed = o_packed;

            // Gate projection: [intermediate_size, hidden_size]
            size_t gate_packed_size = qwen2_packed_weight_size_fp16(
                cfg.intermediate_size, cfg.hidden_size);
            uint16_t* gate_packed = weights_arena.alloc_array<uint16_t>(gate_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.gate_proj, gate_packed,
                cfg.intermediate_size, cfg.hidden_size);
            layer.gate_proj_packed = gate_packed;

            // Up projection: [intermediate_size, hidden_size]
            size_t up_packed_size = qwen2_packed_weight_size_fp16(
                cfg.intermediate_size, cfg.hidden_size);
            uint16_t* up_packed = weights_arena.alloc_array<uint16_t>(up_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.up_proj, up_packed,
                cfg.intermediate_size, cfg.hidden_size);
            layer.up_proj_packed = up_packed;

            // Down projection: [hidden_size, intermediate_size]
            size_t down_packed_size = qwen2_packed_weight_size_fp16(
                cfg.hidden_size, cfg.intermediate_size);
            uint16_t* down_packed = weights_arena.alloc_array<uint16_t>(down_packed_size / sizeof(uint16_t));
            qwen2_pack_weight_fp16(layer.down_proj, down_packed,
                cfg.hidden_size, cfg.intermediate_size);
            layer.down_proj_packed = down_packed;
        }
    }

    return weights;
}

// ============================================================================
// KV Cache Initialization
// ============================================================================

Qwen2KVCache qwen2_init_kv_cache(
    const QwenConfig& cfg,
    Arena& kv_arena,
    size_t max_seq_len
) {
    Qwen2KVCache kv = {};
    kv.max_seq_len = max_seq_len;
    kv.seq_len = 0;

    size_t head_dim = cfg.head_dim();
    size_t kv_per_layer = cfg.num_kv_heads * max_seq_len * head_dim;

    // Allocate K and V caches for all layers
    kv.k_cache = kv_arena.alloc_array<uint16_t>(cfg.num_layers * kv_per_layer);
    kv.v_cache = kv_arena.alloc_array<uint16_t>(cfg.num_layers * kv_per_layer);

    // Zero-initialize
    std::memset(kv.k_cache, 0, cfg.num_layers * kv_per_layer * sizeof(uint16_t));
    std::memset(kv.v_cache, 0, cfg.num_layers * kv_per_layer * sizeof(uint16_t));

    return kv;
}

// ============================================================================
// Scratch Buffer Initialization
// ============================================================================

Qwen2Scratch qwen2_init_scratch(
    const QwenConfig& cfg,
    Arena& scratch_arena,
    size_t max_tokens
) {
    Qwen2Scratch scratch = {};
    scratch.max_tokens = max_tokens;

    size_t head_dim = cfg.head_dim();

    scratch.hidden = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.hidden_size);
    scratch.residual = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.hidden_size);
    scratch.normed = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.hidden_size);
    scratch.q_proj = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.num_attention_heads * head_dim);
    scratch.k_proj = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.num_kv_heads * head_dim);
    scratch.v_proj = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.num_kv_heads * head_dim);
    scratch.q_transposed = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.num_attention_heads * head_dim);
    scratch.attn_out = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.num_attention_heads * head_dim);
    scratch.gate = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.intermediate_size);
    scratch.up = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.intermediate_size);
    scratch.mlp_out = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.hidden_size);
    scratch.logits = scratch_arena.alloc_array<uint16_t>(max_tokens * cfg.vocab_size);

    return scratch;
}

}  // namespace mruntime
