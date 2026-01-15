#include "mruntime/qwen2_forward.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mruntime/dtype.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/trace.h"

namespace mruntime {

// ============================================================================
// Internal Helper: Add bias to projection output
// ============================================================================

namespace {

void add_bias_fp16(
    uint16_t* output,           // [num_tokens, out_features]
    const uint16_t* bias,       // [out_features] (may be nullptr)
    size_t num_tokens,
    size_t out_features
) {
    if (bias == nullptr) return;

    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t i = 0; i < out_features; ++i) {
            float v = fp16_bits_to_float(output[t * out_features + i]);
            v += fp16_bits_to_float(bias[i]);
            output[t * out_features + i] = float_to_fp16_bits(v);
        }
    }
}

}  // namespace

// ============================================================================
// MLP Forward
// ============================================================================

void qwen2_mlp(
    const QwenConfig& cfg,
    const Qwen2LayerWeights& layer,
    const uint16_t* normed_input,
    uint16_t* mlp_output,
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("mlp", "layer");

    size_t hidden_size = cfg.hidden_size;
    size_t intermediate_size = cfg.intermediate_size;

    // Gate projection: [num_tokens, hidden_size] @ [intermediate, hidden]^T -> [num_tokens, intermediate]
    {
        TRACE_SCOPE_CAT("gate_proj", "gemm");
        qwen2_gemm_fp16(
            normed_input,
            layer.gate_proj,
            scratch.gate,
            num_tokens, intermediate_size, hidden_size,
            layer.gate_proj_packed,
            pool
        );
    }

    // Up projection: [num_tokens, hidden_size] @ [intermediate, hidden]^T -> [num_tokens, intermediate]
    {
        TRACE_SCOPE_CAT("up_proj", "gemm");
        qwen2_gemm_fp16(
            normed_input,
            layer.up_proj,
            scratch.up,
            num_tokens, intermediate_size, hidden_size,
            layer.up_proj_packed,
            pool
        );
    }

    // Fused SiLU + elementwise multiply: out = silu(gate) * up
    {
        TRACE_SCOPE_CAT("silu_mul", "elementwise");
        qwen2_silu_mul_fp16(
            scratch.gate,
            scratch.up,
            scratch.gate,  // Reuse gate buffer for intermediate result
            num_tokens * intermediate_size,
            pool
        );
    }

    // Down projection: [num_tokens, intermediate] @ [hidden, intermediate]^T -> [num_tokens, hidden]
    {
        TRACE_SCOPE_CAT("down_proj", "gemm");
        qwen2_gemm_fp16(
            scratch.gate,
            layer.down_proj,
            mlp_output,
            num_tokens, hidden_size, intermediate_size,
            layer.down_proj_packed,
            pool
        );
    }
}

// ============================================================================
// Attention Forward
// ============================================================================

void qwen2_attention(
    const QwenConfig& cfg,
    const Qwen2LayerWeights& layer,
    uint16_t* k_cache,
    uint16_t* v_cache,
    size_t kv_seq_len,
    size_t max_seq_len,
    const uint16_t* normed_input,
    uint16_t* attn_output,
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("attention", "layer");

    const size_t hidden_size = cfg.hidden_size;
    const size_t num_heads = cfg.num_attention_heads;
    const size_t num_kv_heads = cfg.num_kv_heads;
    const size_t head_dim = cfg.head_dim();
    const size_t position_offset = kv_seq_len;
    if (position_offset > max_seq_len || num_tokens > (max_seq_len - position_offset)) {
        throw std::runtime_error("qwen2_attention: KV cache overflow (kv_seq_len + num_tokens > max_seq_len)");
    }

    // Q projection: [num_tokens, hidden] @ [num_heads*head_dim, hidden]^T -> [num_tokens, num_heads*head_dim]
    {
        TRACE_SCOPE_CAT("q_proj", "gemm");
        qwen2_gemm_fp16(
            normed_input,
            layer.q_proj,
            scratch.q_proj,
            num_tokens, num_heads * head_dim, hidden_size,
            layer.q_proj_packed,
            pool
        );
        add_bias_fp16(scratch.q_proj, layer.q_bias, num_tokens, num_heads * head_dim);
    }

    // K projection: [num_tokens, hidden] @ [num_kv_heads*head_dim, hidden]^T -> [num_tokens, num_kv_heads*head_dim]
    {
        TRACE_SCOPE_CAT("k_proj", "gemm");
        qwen2_gemm_fp16(
            normed_input,
            layer.k_proj,
            scratch.k_proj,
            num_tokens, num_kv_heads * head_dim, hidden_size,
            layer.k_proj_packed,
            pool
        );
        add_bias_fp16(scratch.k_proj, layer.k_bias, num_tokens, num_kv_heads * head_dim);
    }

    // V projection: [num_tokens, hidden] @ [num_kv_heads*head_dim, hidden]^T -> [num_tokens, num_kv_heads*head_dim]
    {
        TRACE_SCOPE_CAT("v_proj", "gemm");
        qwen2_gemm_fp16(
            normed_input,
            layer.v_proj,
            scratch.v_proj,
            num_tokens, num_kv_heads * head_dim, hidden_size,
            layer.v_proj_packed,
            pool
        );
        add_bias_fp16(scratch.v_proj, layer.v_bias, num_tokens, num_kv_heads * head_dim);
    }

    // Apply RoPE to Q and K
    {
        TRACE_SCOPE_CAT("rope", "attention");
        // Q/K are in [num_tokens, num_heads, head_dim] layout (treat num_tokens as batch*seq_len with seq_len=1 for each)
        // Actually Q is [1, num_tokens, num_heads, head_dim] for RoPE
        qwen2_rope_fp16(
            scratch.q_proj,
            scratch.k_proj,
            1,  // batch
            num_tokens,  // seq_len
            num_heads,
            num_kv_heads,
            head_dim,
            position_offset,
            cfg.rope_theta,
            pool
        );
    }

    // Copy K, V to cache
    {
        TRACE_SCOPE_CAT("kv_cache_copy", "attention");
        // scratch.k_proj is [num_tokens, num_kv_heads, head_dim] (implicit batch=1)
        // k_cache is [num_kv_heads, max_seq_len, head_dim]
        qwen2_copy_to_kv_cache_fp16(
            scratch.k_proj,
            scratch.v_proj,
            k_cache,
            v_cache,
            1,  // batch
            num_tokens,
            num_kv_heads,
            head_dim,
            max_seq_len,
            position_offset
        );
    }

    // Total sequence length after adding new tokens
    size_t total_seq_len = position_offset + num_tokens;

    // Transpose Q from [1, num_tokens, num_heads, head_dim] to [1, num_heads, num_tokens, head_dim]
    {
        TRACE_SCOPE_CAT("q_transpose", "attention");
        qwen2_transpose_bshd_to_bhsd_fp16(
            scratch.q_proj,
            scratch.q_transposed,
            1, num_tokens, num_heads, head_dim,
            pool
        );
    }

    // Flash attention with grouped query attention
    {
        TRACE_SCOPE_CAT("flash_attention", "attention");
        // Q: [1, num_heads, num_tokens, head_dim]
        // K: [1, num_kv_heads, total_seq_len, head_dim] (from cache)
        // V: [1, num_kv_heads, total_seq_len, head_dim] (from cache)
        // Output: [1, num_heads, num_tokens, head_dim]

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const size_t heads_per_kv = num_heads / num_kv_heads;

        // Process each KV head group
        for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            // Get K, V slices from cache
            const uint16_t* k_slice = k_cache + kv_h * max_seq_len * head_dim;
            const uint16_t* v_slice = v_cache + kv_h * max_seq_len * head_dim;

            for (size_t h_offset = 0; h_offset < heads_per_kv; ++h_offset) {
                size_t q_h = kv_h * heads_per_kv + h_offset;

                // Q slice: [1, 1, num_tokens, head_dim]
                const uint16_t* q_slice = scratch.q_transposed + q_h * num_tokens * head_dim;

                // Output slice: [1, 1, num_tokens, head_dim]
                uint16_t* out_slice = scratch.attn_out + q_h * num_tokens * head_dim;

                // Flash attention for this head
                qwen2_flash_attention_fp16(
                    q_slice,
                    k_slice,
                    v_slice,
                    out_slice,
                    1,  // batch
                    1,  // num_heads (processing one at a time)
                    num_tokens,  // q_len
                    total_seq_len,  // kv_len
                    head_dim,
                    scale,
                    true,  // causal
                    pool
                );
            }
        }
    }

    // Transpose attention output from [1, num_heads, num_tokens, head_dim] to [1, num_tokens, num_heads, head_dim]
    {
        TRACE_SCOPE_CAT("attn_transpose", "attention");
        // Then reshape to [num_tokens, num_heads * head_dim] for output projection
        // Note: scratch.attn_out is already in [num_heads, num_tokens, head_dim] layout
        // We need [num_tokens, num_heads, head_dim] for output projection

        // Transpose BHSD -> BSHD
        uint16_t* attn_transposed = scratch.q_proj;  // Reuse q_proj buffer
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t = 0; t < num_tokens; ++t) {
                const uint16_t* src = scratch.attn_out + h * num_tokens * head_dim + t * head_dim;
                uint16_t* dst = attn_transposed + t * num_heads * head_dim + h * head_dim;
                std::memcpy(dst, src, head_dim * sizeof(uint16_t));
            }
        }
    }

    // Output projection: [num_tokens, num_heads*head_dim] @ [hidden, num_heads*head_dim]^T -> [num_tokens, hidden]
    {
        TRACE_SCOPE_CAT("o_proj", "gemm");
        uint16_t* attn_transposed = scratch.q_proj;  // Same buffer as above
        qwen2_gemm_fp16(
            attn_transposed,
            layer.o_proj,
            attn_output,
            num_tokens, hidden_size, num_heads * head_dim,
            layer.o_proj_packed,
            pool
        );
    }
}

// ============================================================================
// Layer Forward
// ============================================================================

void qwen2_layer_forward(
    const QwenConfig& cfg,
    const Qwen2LayerWeights& layer,
    uint16_t* k_cache,
    uint16_t* v_cache,
    size_t kv_seq_len,
    size_t max_seq_len,
    const uint16_t* hidden_in,
    uint16_t* hidden_out,
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
) {
    const size_t hidden_size = cfg.hidden_size;

    // Input LayerNorm
    qwen2_rmsnorm_fp16(
        hidden_in,
        layer.input_norm,
        scratch.normed,
        num_tokens,
        hidden_size,
        cfg.rms_norm_eps,
        pool
    );

    // Self-attention
    qwen2_attention(
        cfg,
        layer,
        k_cache,
        v_cache,
        kv_seq_len,
        max_seq_len,
        scratch.normed,
        scratch.residual,  // Use residual buffer for attention output
        scratch,
        num_tokens,
        pool
    );

    // First residual connection: hidden_out = hidden_in + attn_out
    qwen2_add_fp16(
        hidden_in,
        scratch.residual,
        scratch.hidden,  // Store in hidden buffer
        num_tokens * hidden_size,
        pool
    );

    // Post-attention LayerNorm
    qwen2_rmsnorm_fp16(
        scratch.hidden,
        layer.post_attn_norm,
        scratch.normed,
        num_tokens,
        hidden_size,
        cfg.rms_norm_eps,
        pool
    );

    // MLP
    qwen2_mlp(
        cfg,
        layer,
        scratch.normed,
        scratch.mlp_out,
        scratch,
        num_tokens,
        pool
    );

    // Second residual connection: hidden_out = residual1 + mlp_out
    qwen2_add_fp16(
        scratch.hidden,
        scratch.mlp_out,
        hidden_out,
        num_tokens * hidden_size,
        pool
    );
}

// ============================================================================
// Full Forward Pass
// ============================================================================

uint16_t* qwen2_forward(
    const QwenConfig& cfg,
    const Qwen2Weights& weights,
    Qwen2KVCache& kv_cache,
    Qwen2Scratch& scratch,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("qwen2_forward", "forward");

    assert(num_tokens > 0);
    assert(num_tokens <= scratch.max_tokens);
    if (num_tokens > scratch.max_tokens) {
        throw std::runtime_error("qwen2_forward: num_tokens exceeds scratch.max_tokens; increase max_batch_tokens");
    }

    const size_t hidden_size = cfg.hidden_size;
    const size_t head_dim = cfg.head_dim();
    const size_t position_offset = kv_cache.seq_len;

    // Embedding lookup
    {
        TRACE_SCOPE_CAT("embedding", "forward");
        qwen2_embedding_lookup_fp16(
            weights.embed_tokens,
            token_ids,
            scratch.hidden,
            num_tokens,
            cfg.vocab_size,
            hidden_size
        );
    }

    // Process each transformer layer
    uint16_t* current_hidden = scratch.hidden;
    uint16_t* next_hidden = scratch.residual;

    for (size_t layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        TRACE_SCOPE_CAT("layer", "forward");

        const Qwen2LayerWeights& layer = weights.layers[layer_idx];

        uint16_t* k_cache_layer = kv_cache.k_layer(layer_idx, cfg.num_kv_heads, head_dim);
        uint16_t* v_cache_layer = kv_cache.v_layer(layer_idx, cfg.num_kv_heads, head_dim);

        // Input LayerNorm
        {
            TRACE_SCOPE_CAT("input_norm", "norm");
            qwen2_rmsnorm_fp16(
                current_hidden,
                layer.input_norm,
                scratch.normed,
                num_tokens,
                hidden_size,
                cfg.rms_norm_eps,
                pool
            );
        }

        // Self-attention
        qwen2_attention(
            cfg,
            layer,
            k_cache_layer,
            v_cache_layer,
            position_offset,
            kv_cache.max_seq_len,
            scratch.normed,
            scratch.attn_out,  // Reuse attn_out as temp storage for attention output
            scratch,
            num_tokens,
            pool
        );

        // First residual: next_hidden = current_hidden + attn_out
        {
            TRACE_SCOPE_CAT("residual_add", "elementwise");
            // Reinterpret attn_out as [num_tokens, hidden_size] for the add
            qwen2_add_fp16(
                current_hidden,
                scratch.attn_out,
                next_hidden,
                num_tokens * hidden_size,
                pool
            );
        }

        // Post-attention LayerNorm
        {
            TRACE_SCOPE_CAT("post_attn_norm", "norm");
            qwen2_rmsnorm_fp16(
                next_hidden,
                layer.post_attn_norm,
                scratch.normed,
                num_tokens,
                hidden_size,
                cfg.rms_norm_eps,
                pool
            );
        }

        // MLP
        qwen2_mlp(
            cfg,
            layer,
            scratch.normed,
            scratch.mlp_out,
            scratch,
            num_tokens,
            pool
        );

        // Second residual: current_hidden = next_hidden + mlp_out
        {
            TRACE_SCOPE_CAT("residual_add", "elementwise");
            qwen2_add_fp16(
                next_hidden,
                scratch.mlp_out,
                current_hidden,
                num_tokens * hidden_size,
                pool
            );
        }
    }

    // Update KV cache sequence length
    kv_cache.seq_len = position_offset + num_tokens;

    // Final LayerNorm
    {
        TRACE_SCOPE_CAT("final_norm", "norm");
        qwen2_rmsnorm_fp16(
            current_hidden,
            weights.final_norm,
            scratch.normed,
            num_tokens,
            hidden_size,
            cfg.rms_norm_eps,
            pool
        );
    }

    // LM head projection: [num_tokens, hidden] @ [vocab, hidden]^T -> [num_tokens, vocab]
    {
        TRACE_SCOPE_ARGS_CAT(
            "lm_head",
            "gemm",
            ::mruntime::trace_arg("m", static_cast<int64_t>(num_tokens)),
            ::mruntime::trace_arg("n", static_cast<int64_t>(cfg.vocab_size)),
            ::mruntime::trace_arg("k", static_cast<int64_t>(hidden_size))
        );
        qwen2_gemm_fp16(
            scratch.normed,
            weights.lm_head,
            scratch.logits,
            num_tokens, cfg.vocab_size, hidden_size,
            nullptr,  // No packed weights for lm_head (too large)
            pool
        );
    }

    return scratch.logits;
}

}  // namespace mruntime
