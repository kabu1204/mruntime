#include "mruntime/qwen2_forward_vk.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mruntime/dtype.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/trace.h"

#include "src/qwen2_vk_state.h"

namespace mruntime {

namespace {

VkDeviceSize choose_alignment(const vulkan::VkContext& ctx) {
    return std::max<VkDeviceSize>(64, ctx.min_storage_buffer_offset_alignment());
}

template <class T>
T* arena_alloc_array(vulkan::VkBufferArena& arena, size_t count) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(T);
    const VkDeviceSize offset = arena.alloc(bytes);
    T* ptr = arena.host_ptr<T>(offset);
    if (ptr == nullptr) {
        throw std::runtime_error("Vulkan arena allocation returned null mapped pointer");
    }
    return ptr;
}

VkDescriptorBufferInfo descriptor_for_ptr(
    const vulkan::VkBufferArena& arena,
    const void* ptr,
    VkDeviceSize bytes
) {
    if (ptr == nullptr) {
        throw std::runtime_error("descriptor_for_ptr: ptr is null");
    }
    if (arena.mapped() == nullptr) {
        throw std::runtime_error("descriptor_for_ptr: arena is not mapped");
    }
    const auto* base = static_cast<const uint8_t*>(arena.mapped());
    const auto* p = static_cast<const uint8_t*>(ptr);
    if (p < base) {
        throw std::runtime_error("descriptor_for_ptr: ptr is before arena base");
    }
    const VkDeviceSize offset = static_cast<VkDeviceSize>(p - base);
    if (offset + bytes > arena.capacity()) {
        throw std::runtime_error("descriptor_for_ptr: range out of arena bounds");
    }

    VkDescriptorBufferInfo info = {};
    info.buffer = arena.buffer();
    info.offset = offset;
    info.range = bytes;
    return info;
}

void add_bias_fp16(
    uint16_t* output,
    const uint16_t* bias,
    size_t num_tokens,
    size_t out_features
) {
    if (bias == nullptr) {
        return;
    }

    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t i = 0; i < out_features; ++i) {
            float v = fp16_bits_to_float(output[t * out_features + i]);
            v += fp16_bits_to_float(bias[i]);
            output[t * out_features + i] = float_to_fp16_bits(v);
        }
    }
}

void init_rope_cache_into_kv_arena(
    const QwenConfig& cfg,
    size_t max_seq_len,
    vulkan::VkBufferArena& kv_arena,
    Qwen2KVCache& out_kv
) {
    const size_t head_dim = cfg.head_dim();
    if (head_dim == 0 || (head_dim % 2) != 0) {
        throw std::runtime_error("qwen2_vk_create: invalid head_dim for RoPE");
    }
    if (max_seq_len == 0) {
        throw std::runtime_error("qwen2_vk_create: max_seq_len must be > 0");
    }
    if (!(cfg.rope_theta > 0.0f)) {
        throw std::runtime_error("qwen2_vk_create: rope_theta must be > 0");
    }

    Qwen2RopeCache rope = {};
    rope.head_dim = head_dim;
    rope.half_dim = head_dim / 2;
    rope.max_seq_len = max_seq_len;
    rope.theta = cfg.rope_theta;

    float* inv_freq = arena_alloc_array<float>(kv_arena, rope.half_dim);
    float* cos_sin = arena_alloc_array<float>(kv_arena, rope.max_seq_len * rope.half_dim * 2);

    const float two_over_head_dim = 2.0f / static_cast<float>(head_dim);
    for (size_t i = 0; i < rope.half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(i) * two_over_head_dim);
    }

    for (size_t pos = 0; pos < rope.max_seq_len; ++pos) {
        float* cs = cos_sin + pos * rope.half_dim * 2;
        const float pos_f = static_cast<float>(pos);
        for (size_t i = 0; i < rope.half_dim; ++i) {
            const float angle = pos_f * inv_freq[i];
            cs[i * 2] = std::cos(angle);
            cs[i * 2 + 1] = std::sin(angle);
        }
    }

    rope.inv_freq = inv_freq;
    rope.cos_sin = cos_sin;
    out_kv.rope = rope;
}

void qwen2_mlp_vk(
    Qwen2VkState& state,
    const Qwen2LayerWeights& layer,
    const uint16_t* normed_input,
    uint16_t* mlp_output,
    Qwen2Scratch& scratch,
    size_t num_tokens
) {
    TRACE_SCOPE_CAT("mlp_vk", "layer");

    const size_t hidden_size = state.cfg.hidden_size;
    const size_t intermediate_size = state.cfg.intermediate_size;

    // gate+up projection: [num_tokens, hidden] @ [2*intermediate, hidden]^T -> [num_tokens, 2*intermediate]
    {
        TRACE_SCOPE_CAT("gate_up_proj_vk", "gemm");
        state.fp16_ops.gemm(
            descriptor_for_ptr(state.scratch_arena, normed_input, num_tokens * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, layer.gate_up_proj,
                               2 * intermediate_size * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, scratch.up,
                               num_tokens * intermediate_size * 2 * sizeof(uint16_t)),
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(intermediate_size * 2),
            static_cast<uint32_t>(hidden_size)
        );
    }

    // out = silu(gate) * up, interleaved (gate|up) input.
    {
        TRACE_SCOPE_CAT("silu_mul_vk", "elementwise");
        state.fp16_ops.silu_mul_interleaved(
            descriptor_for_ptr(state.scratch_arena, scratch.up,
                               num_tokens * intermediate_size * 2 * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, scratch.gate, num_tokens * intermediate_size * sizeof(uint16_t)),
            static_cast<uint32_t>(intermediate_size),
            static_cast<uint32_t>(num_tokens)
        );
    }

    // down projection: [num_tokens, intermediate] @ [hidden, intermediate]^T -> [num_tokens, hidden]
    {
        TRACE_SCOPE_CAT("down_proj_vk", "gemm");
        state.fp16_ops.gemm(
            descriptor_for_ptr(state.scratch_arena, scratch.gate, num_tokens * intermediate_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, layer.down_proj, hidden_size * intermediate_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, mlp_output, num_tokens * hidden_size * sizeof(uint16_t)),
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(intermediate_size)
        );
    }
}

void qwen2_attention_vk(
    Qwen2VkState& state,
    const Qwen2LayerWeights& layer,
    uint16_t* k_cache,
    uint16_t* v_cache,
    size_t kv_seq_len,
    size_t max_seq_len,
    const float* rope_cos_sin,
    const uint16_t* normed_input,
    uint16_t* attn_output,
    Qwen2Scratch& scratch,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("attention_vk", "layer");

    const size_t hidden_size = state.cfg.hidden_size;
    const size_t num_heads = state.cfg.num_attention_heads;
    const size_t num_kv_heads = state.cfg.num_kv_heads;
    const size_t head_dim = state.cfg.head_dim();

    const size_t q_dim = num_heads * head_dim;
    const size_t kv_dim = num_kv_heads * head_dim;
    const size_t qkv_dim = q_dim + 2 * kv_dim;
    const size_t position_offset = kv_seq_len;

    if (position_offset > max_seq_len || num_tokens > (max_seq_len - position_offset)) {
        throw std::runtime_error("qwen2_attention_vk: KV cache overflow (kv_seq_len + num_tokens > max_seq_len)");
    }

    // QKV projection: [num_tokens, hidden] @ [qkv_dim, hidden]^T -> [num_tokens, qkv_dim]
    {
        TRACE_SCOPE_CAT("qkv_proj_vk", "gemm");
        state.fp16_ops.gemm(
            descriptor_for_ptr(state.scratch_arena, normed_input, num_tokens * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, layer.qkv_proj, qkv_dim * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, scratch.qkv_out, num_tokens * qkv_dim * sizeof(uint16_t)),
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(qkv_dim),
            static_cast<uint32_t>(hidden_size)
        );
        add_bias_fp16(scratch.qkv_out, layer.qkv_bias, num_tokens, qkv_dim);
    }

    // Split fused QKV output into separate Q, K, V buffers.
    {
        TRACE_SCOPE_CAT("qkv_split_vk", "elementwise");
        for (size_t t = 0; t < num_tokens; ++t) {
            const uint16_t* src = scratch.qkv_out + t * qkv_dim;
            std::memcpy(scratch.q_proj + t * q_dim, src, q_dim * sizeof(uint16_t));
            std::memcpy(scratch.k_proj + t * kv_dim, src + q_dim, kv_dim * sizeof(uint16_t));
            std::memcpy(scratch.v_proj + t * kv_dim, src + q_dim + kv_dim, kv_dim * sizeof(uint16_t));
        }
    }

    // RoPE in-place on Q and K.
    {
        TRACE_SCOPE_CAT("rope_vk", "attention");
        const VkDeviceSize rope_bytes =
            static_cast<VkDeviceSize>(max_seq_len) * (head_dim / 2) * 2 * sizeof(float);
        state.fp16_ops.rope(
            descriptor_for_ptr(state.scratch_arena, scratch.q_proj, num_tokens * q_dim * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, scratch.k_proj, num_tokens * kv_dim * sizeof(uint16_t)),
            descriptor_for_ptr(state.kv_arena, rope_cos_sin, rope_bytes),
            1,  // batch
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(num_kv_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(position_offset)
        );
    }

    // Copy K/V into the KV cache.
    {
        TRACE_SCOPE_CAT("kv_cache_copy_vk", "attention");
        const VkDeviceSize kv_in_bytes = num_tokens * kv_dim * sizeof(uint16_t);
        const VkDeviceSize cache_bytes =
            static_cast<VkDeviceSize>(num_kv_heads) * max_seq_len * head_dim * sizeof(uint16_t);

        VkDescriptorBufferInfo k_cache_desc = descriptor_for_ptr(state.kv_arena, k_cache, cache_bytes);
        k_cache_desc.range = 0;  // VK_WHOLE_SIZE (also covers V cache when suballocated after K).

        state.fp16_ops.kv_cache_copy(
            descriptor_for_ptr(state.scratch_arena, scratch.k_proj, kv_in_bytes),
            descriptor_for_ptr(state.scratch_arena, scratch.v_proj, kv_in_bytes),
            k_cache_desc,
            descriptor_for_ptr(state.kv_arena, v_cache, cache_bytes),
            1,  // batch
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(num_kv_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(max_seq_len),
            static_cast<uint32_t>(position_offset)
        );
    }

    const size_t total_seq_len = position_offset + num_tokens;

    // Transpose Q from BSHD -> BHSD for flash attention.
    {
        TRACE_SCOPE_CAT("q_transpose_vk", "attention");
        state.fp16_ops.transpose_bshd_to_bhsd(
            descriptor_for_ptr(state.scratch_arena, scratch.q_proj, num_tokens * q_dim * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, scratch.q_transposed, num_tokens * q_dim * sizeof(uint16_t)),
            1,
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(head_dim)
        );
    }

    // Flash attention (CPU) with GQA.
    {
        TRACE_SCOPE_CAT("flash_attention_cpu", "attention");
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        qwen2_flash_attention_gqa_fp16(
            scratch.q_transposed,  // Q: [1, num_heads, q_len, head_dim]
            k_cache,               // K: [1, num_kv_heads, max_seq_len, head_dim]
            v_cache,               // V: [1, num_kv_heads, max_seq_len, head_dim]
            scratch.attn_out,      // O: [1, num_heads, q_len, head_dim]
            1,
            num_heads,
            num_kv_heads,
            num_tokens,
            total_seq_len,
            max_seq_len,
            head_dim,
            scale,
            true,
            pool
        );
    }

    // Convert attention output from BHSD -> BSHD using the existing BSHD->BHSD transpose shader
    // with swapped S/H dimensions.
    {
        TRACE_SCOPE_CAT("attn_transpose_vk", "attention");
        state.fp16_ops.transpose_bshd_to_bhsd(
            descriptor_for_ptr(state.scratch_arena, scratch.attn_out, num_tokens * q_dim * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, scratch.q_proj, num_tokens * q_dim * sizeof(uint16_t)),
            1,
            static_cast<uint32_t>(num_heads),   // S := H_original
            static_cast<uint32_t>(num_tokens),  // H := S_original
            static_cast<uint32_t>(head_dim)
        );
    }

    // Output projection: [num_tokens, q_dim] @ [hidden, q_dim]^T -> [num_tokens, hidden]
    {
        TRACE_SCOPE_CAT("o_proj_vk", "gemm");
        state.fp16_ops.gemm(
            descriptor_for_ptr(state.scratch_arena, scratch.q_proj, num_tokens * q_dim * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, layer.o_proj, hidden_size * q_dim * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, attn_output, num_tokens * hidden_size * sizeof(uint16_t)),
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(q_dim)
        );
    }
}

uint16_t* qwen2_forward_hidden_vk(
    Qwen2VkState& state,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
) {
    const QwenConfig& cfg = state.cfg;
    Qwen2Scratch& scratch = state.scratch;
    Qwen2KVCache& kv_cache = state.kv_cache;
    const Qwen2Weights& weights = state.weights;

    const size_t hidden_size = cfg.hidden_size;
    const size_t head_dim = cfg.head_dim();
    const size_t position_offset = kv_cache.seq_len;

    // Embedding lookup (CPU) into Vulkan-mapped scratch.
    {
        TRACE_SCOPE_CAT("embedding_cpu", "forward");
        qwen2_embedding_lookup_fp16(
            weights.embed_tokens,
            token_ids,
            scratch.hidden,
            num_tokens,
            cfg.vocab_size,
            hidden_size
        );
    }

    uint16_t* current_hidden = scratch.hidden;
    uint16_t* next_hidden = scratch.residual;

    for (size_t layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        TRACE_SCOPE_CAT("layer", "forward_vk");

        const Qwen2LayerWeights& layer = weights.layers[layer_idx];

        uint16_t* k_cache_layer = kv_cache.k_layer(layer_idx, cfg.num_kv_heads, head_dim);
        uint16_t* v_cache_layer = kv_cache.v_layer(layer_idx, cfg.num_kv_heads, head_dim);

        // Input LayerNorm (Vulkan).
        {
            TRACE_SCOPE_CAT("input_norm_vk", "norm");
            state.fp16_ops.rmsnorm(
                descriptor_for_ptr(state.scratch_arena, current_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.weights_arena, layer.input_norm, hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, scratch.normed, num_tokens * hidden_size * sizeof(uint16_t)),
                static_cast<uint32_t>(hidden_size),
                static_cast<uint32_t>(num_tokens),
                cfg.rms_norm_eps
            );
        }

        qwen2_attention_vk(
            state,
            layer,
            k_cache_layer,
            v_cache_layer,
            position_offset,
            kv_cache.max_seq_len,
            kv_cache.rope.cos_sin,
            scratch.normed,
            scratch.attn_out,
            scratch,
            num_tokens,
            pool
        );

        // First residual: next_hidden = current_hidden + attn_out
        {
            TRACE_SCOPE_CAT("residual_add_vk", "elementwise");
            state.fp16_ops.add(
                descriptor_for_ptr(state.scratch_arena, current_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, scratch.attn_out, num_tokens * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, next_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
                static_cast<uint32_t>(num_tokens * hidden_size)
            );
        }

        // Post-attention LayerNorm (Vulkan).
        {
            TRACE_SCOPE_CAT("post_attn_norm_vk", "norm");
            state.fp16_ops.rmsnorm(
                descriptor_for_ptr(state.scratch_arena, next_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.weights_arena, layer.post_attn_norm, hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, scratch.normed, num_tokens * hidden_size * sizeof(uint16_t)),
                static_cast<uint32_t>(hidden_size),
                static_cast<uint32_t>(num_tokens),
                cfg.rms_norm_eps
            );
        }

        qwen2_mlp_vk(
            state,
            layer,
            scratch.normed,
            scratch.mlp_out,
            scratch,
            num_tokens
        );

        // Second residual: current_hidden = next_hidden + mlp_out
        {
            TRACE_SCOPE_CAT("residual_add_vk", "elementwise");
            state.fp16_ops.add(
                descriptor_for_ptr(state.scratch_arena, next_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, scratch.mlp_out, num_tokens * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, current_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
                static_cast<uint32_t>(num_tokens * hidden_size)
            );
        }
    }

    kv_cache.seq_len = position_offset + num_tokens;
    return current_hidden;
}

uint16_t* forward_chunk_last_logits_vk(
    Qwen2VkState& state,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("qwen2_forward_chunk_last_logits_vk", "forward");

    assert(num_tokens > 0);
    assert(num_tokens <= state.scratch.max_tokens);
    if (num_tokens > state.scratch.max_tokens) {
        throw std::runtime_error(
            "qwen2_forward_chunk_last_logits_vk: num_tokens exceeds scratch.max_tokens; increase max_batch_tokens"
        );
    }

    const size_t hidden_size = state.cfg.hidden_size;
    const uint16_t* current_hidden = qwen2_forward_hidden_vk(state, token_ids, num_tokens, pool);
    const uint16_t* last_hidden = current_hidden + (num_tokens - 1) * hidden_size;

    // Final LayerNorm (last token only).
    {
        TRACE_SCOPE_CAT("final_norm_vk", "norm");
        state.fp16_ops.rmsnorm(
            descriptor_for_ptr(state.scratch_arena, last_hidden, hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, state.weights.final_norm, hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, state.scratch.normed, hidden_size * sizeof(uint16_t)),
            static_cast<uint32_t>(hidden_size),
            1,
            state.cfg.rms_norm_eps
        );
    }

    // LM head projection (last token only): prefer GEMV for decode-like M=1 when available.
    {
        TRACE_SCOPE_ARGS_CAT(
            "lm_head_vk",
            "gemm_or_gemv",
            ::mruntime::trace_arg("m", 1),
            ::mruntime::trace_arg("n", static_cast<int64_t>(state.cfg.vocab_size)),
            ::mruntime::trace_arg("k", static_cast<int64_t>(hidden_size))
        );

        if ((hidden_size & 3u) == 0u) {
            state.fp16_ops.gemv(
                descriptor_for_ptr(state.scratch_arena, state.scratch.normed, hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.weights_arena, state.weights.lm_head,
                                   state.cfg.vocab_size * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, state.scratch.logits,
                                   state.cfg.vocab_size * sizeof(uint16_t)),
                static_cast<uint32_t>(state.cfg.vocab_size),
                static_cast<uint32_t>(hidden_size)
            );
        } else {
            state.fp16_ops.gemm(
                descriptor_for_ptr(state.scratch_arena, state.scratch.normed, hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.weights_arena, state.weights.lm_head,
                                   state.cfg.vocab_size * hidden_size * sizeof(uint16_t)),
                descriptor_for_ptr(state.scratch_arena, state.scratch.logits,
                                   state.cfg.vocab_size * sizeof(uint16_t)),
                1,
                static_cast<uint32_t>(state.cfg.vocab_size),
                static_cast<uint32_t>(hidden_size)
            );
        }
    }

    return state.scratch.logits;
}

void forward_chunk_no_logits_vk(
    Qwen2VkState& state,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("qwen2_forward_chunk_no_logits_vk", "forward");

    assert(num_tokens > 0);
    assert(num_tokens <= state.scratch.max_tokens);
    if (num_tokens > state.scratch.max_tokens) {
        throw std::runtime_error(
            "qwen2_forward_chunk_no_logits_vk: num_tokens exceeds scratch.max_tokens; increase max_batch_tokens"
        );
    }

    (void)qwen2_forward_hidden_vk(state, token_ids, num_tokens, pool);
}

size_t weights_bytes_needed_for_vk(
    const QwenConfig& cfg,
    const Qwen2Weights& cpu_weights
) {
    const size_t hidden_size = cfg.hidden_size;
    const size_t head_dim = cfg.head_dim();
    const size_t q_dim = cfg.num_attention_heads * head_dim;
    const size_t kv_dim = cfg.num_kv_heads * head_dim;
    const size_t qkv_dim = q_dim + 2 * kv_dim;

    size_t bytes = 0;
    bytes += hidden_size * sizeof(uint16_t);                      // final_norm
    bytes += cfg.vocab_size * hidden_size * sizeof(uint16_t);     // lm_head

    for (size_t i = 0; i < cfg.num_layers; ++i) {
        const Qwen2LayerWeights& layer = cpu_weights.layers[i];
        bytes += hidden_size * sizeof(uint16_t);                  // input_norm
        bytes += qkv_dim * hidden_size * sizeof(uint16_t);        // qkv_proj
        if (layer.qkv_bias != nullptr) {
            bytes += qkv_dim * sizeof(uint16_t);                  // qkv_bias
        }
        bytes += hidden_size * q_dim * sizeof(uint16_t);          // o_proj
        bytes += hidden_size * sizeof(uint16_t);                  // post_attn_norm
        bytes += 2 * cfg.intermediate_size * hidden_size * sizeof(uint16_t);  // gate_up_proj
        bytes += hidden_size * cfg.intermediate_size * sizeof(uint16_t);      // down_proj
    }

    return bytes;
}

}  // namespace

void Qwen2VkStateDeleter::operator()(Qwen2VkState* state) const noexcept {
    delete state;
}

Qwen2VkStatePtr qwen2_vk_create(
    const QwenConfig& cfg,
    const Qwen2Weights& cpu_weights,
    size_t max_seq_len,
    size_t max_batch_tokens
) {
    if (max_seq_len == 0) {
        throw std::runtime_error("qwen2_vk_create: max_seq_len must be > 0");
    }
    if (max_batch_tokens == 0) {
        throw std::runtime_error("qwen2_vk_create: max_batch_tokens must be > 0");
    }
    if (cpu_weights.layers == nullptr || cpu_weights.num_layers != cfg.num_layers) {
        throw std::runtime_error("qwen2_vk_create: cpu_weights.layers is null or num_layers mismatch");
    }
    if (cpu_weights.final_norm == nullptr || cpu_weights.lm_head == nullptr) {
        throw std::runtime_error("qwen2_vk_create: cpu_weights is missing required tensors");
    }

    Qwen2VkStatePtr state(new Qwen2VkState());
    state->cfg = cfg;
    state->max_seq_len = max_seq_len;
    state->max_batch_tokens = max_batch_tokens;

    state->vk = vulkan::VkContext::Create();
    state->alignment = choose_alignment(state->vk);
    state->runtime = vulkan::VkKernelRuntime::Create(state->vk);
    state->fp16_ops = vulkan::VkFp16Ops::Create(&state->runtime);

    // Allocate Vulkan arenas (host-visible coherent for correctness).
    const Qwen2MemorySizes sizes = qwen2_memory_sizes(cfg, max_seq_len, max_batch_tokens);

    vulkan::VkBufferArenaCreateInfo kv_info;
    kv_info.capacity_bytes = sizes.kv_cache_bytes + 4 * state->alignment;
    kv_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    kv_info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    kv_info.default_alignment = state->alignment;
    state->kv_arena = vulkan::VkBufferArena::Create(
        state->vk.physical_device(), state->vk.device(), kv_info);

    vulkan::VkBufferArenaCreateInfo scratch_info;
    scratch_info.capacity_bytes = sizes.scratch_bytes + 4 * state->alignment;
    scratch_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    scratch_info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    scratch_info.default_alignment = state->alignment;
    state->scratch_arena = vulkan::VkBufferArena::Create(
        state->vk.physical_device(), state->vk.device(), scratch_info);

    const size_t weights_bytes = weights_bytes_needed_for_vk(cfg, cpu_weights);
    vulkan::VkBufferArenaCreateInfo weights_info;
    weights_info.capacity_bytes = static_cast<VkDeviceSize>(weights_bytes) + 4 * state->alignment;
    weights_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    weights_info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    weights_info.default_alignment = state->alignment;
    state->weights_arena = vulkan::VkBufferArena::Create(
        state->vk.physical_device(), state->vk.device(), weights_info);

    // Upload weights to weights_arena.
    state->weights.num_layers = cfg.num_layers;
    state->layer_weights.resize(cfg.num_layers);
    state->weights.layers = state->layer_weights.data();

    const size_t head_dim = cfg.head_dim();
    const size_t q_dim = cfg.num_attention_heads * head_dim;
    const size_t kv_dim = cfg.num_kv_heads * head_dim;
    const size_t qkv_dim = q_dim + 2 * kv_dim;

    // final_norm
    {
        const size_t count = cfg.hidden_size;
        uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, count);
        std::memcpy(dst, cpu_weights.final_norm, count * sizeof(uint16_t));
        state->weights.final_norm = dst;
    }

    // lm_head
    {
        const size_t count = cfg.vocab_size * cfg.hidden_size;
        uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, count);
        std::memcpy(dst, cpu_weights.lm_head, count * sizeof(uint16_t));
        state->weights.lm_head = dst;
        state->weights.lm_head_packed = nullptr;
    }

    // embed_tokens: avoid uploading if untied; otherwise reuse lm_head.
    if (cpu_weights.embed_tokens == cpu_weights.lm_head) {
        state->weights.embed_tokens = state->weights.lm_head;
    } else {
        state->weights.embed_tokens = cpu_weights.embed_tokens;
    }

    for (size_t i = 0; i < cfg.num_layers; ++i) {
        const Qwen2LayerWeights& src = cpu_weights.layers[i];
        Qwen2LayerWeights& dst_layer = state->layer_weights[i];

        // input_norm
        {
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, cfg.hidden_size);
            std::memcpy(dst, src.input_norm, cfg.hidden_size * sizeof(uint16_t));
            dst_layer.input_norm = dst;
        }

        // qkv_proj
        {
            const size_t count = qkv_dim * cfg.hidden_size;
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, count);
            std::memcpy(dst, src.qkv_proj, count * sizeof(uint16_t));
            dst_layer.qkv_proj = dst;
        }

        // qkv_bias (optional)
        if (src.qkv_bias != nullptr) {
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, qkv_dim);
            std::memcpy(dst, src.qkv_bias, qkv_dim * sizeof(uint16_t));
            dst_layer.qkv_bias = dst;
        } else {
            dst_layer.qkv_bias = nullptr;
        }

        // o_proj
        {
            const size_t count = cfg.hidden_size * q_dim;
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, count);
            std::memcpy(dst, src.o_proj, count * sizeof(uint16_t));
            dst_layer.o_proj = dst;
        }

        // post_attn_norm
        {
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, cfg.hidden_size);
            std::memcpy(dst, src.post_attn_norm, cfg.hidden_size * sizeof(uint16_t));
            dst_layer.post_attn_norm = dst;
        }

        // gate_up_proj
        {
            const size_t count = static_cast<size_t>(2) * cfg.intermediate_size * cfg.hidden_size;
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, count);
            std::memcpy(dst, src.gate_up_proj, count * sizeof(uint16_t));
            dst_layer.gate_up_proj = dst;
        }

        // down_proj
        {
            const size_t count = cfg.hidden_size * cfg.intermediate_size;
            uint16_t* dst = arena_alloc_array<uint16_t>(state->weights_arena, count);
            std::memcpy(dst, src.down_proj, count * sizeof(uint16_t));
            dst_layer.down_proj = dst;
        }

        dst_layer.qkv_proj_packed = nullptr;
        dst_layer.o_proj_packed = nullptr;
        dst_layer.gate_up_proj_packed = nullptr;
        dst_layer.down_proj_packed = nullptr;
    }

    // Initialize KV cache in kv_arena: allocate K, then V, then rope tables.
    {
        state->kv_cache.max_seq_len = max_seq_len;
        state->kv_cache.seq_len = 0;

        const size_t kv_per_layer = cfg.num_kv_heads * max_seq_len * head_dim;
        const size_t total_count = cfg.num_layers * kv_per_layer;

        state->kv_cache.k_cache = arena_alloc_array<uint16_t>(state->kv_arena, total_count);
        state->kv_cache.v_cache = arena_alloc_array<uint16_t>(state->kv_arena, total_count);

        std::memset(state->kv_cache.k_cache, 0, total_count * sizeof(uint16_t));
        std::memset(state->kv_cache.v_cache, 0, total_count * sizeof(uint16_t));

        init_rope_cache_into_kv_arena(cfg, max_seq_len, state->kv_arena, state->kv_cache);
    }

    // Initialize scratch in scratch_arena.
    {
        state->scratch.max_tokens = max_batch_tokens;

        const size_t q_dim_elems = cfg.num_attention_heads * head_dim;
        const size_t kv_dim_elems = cfg.num_kv_heads * head_dim;
        const size_t qkv_dim_elems = q_dim_elems + 2 * kv_dim_elems;

        state->scratch.hidden = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.hidden_size);
        state->scratch.residual = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.hidden_size);
        state->scratch.normed = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.hidden_size);
        state->scratch.qkv_out = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * qkv_dim_elems);
        state->scratch.q_proj = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * q_dim_elems);
        state->scratch.k_proj = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * kv_dim_elems);
        state->scratch.v_proj = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * kv_dim_elems);
        state->scratch.q_transposed = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * q_dim_elems);
        state->scratch.attn_out = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * q_dim_elems);
        state->scratch.gate = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.intermediate_size);
        state->scratch.up = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.intermediate_size * 2);
        state->scratch.mlp_out = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.hidden_size);
        state->scratch.logits = arena_alloc_array<uint16_t>(state->scratch_arena, max_batch_tokens * cfg.vocab_size);
    }

    return state;
}

void qwen2_vk_reset_kv_cache(Qwen2VkState& state) {
    state.kv_cache.seq_len = 0;
}

uint16_t* qwen2_forward_vk(
    Qwen2VkState& state,
    const int32_t* token_ids,
    size_t num_tokens,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("qwen2_forward_vk", "forward");

    assert(num_tokens > 0);
    assert(num_tokens <= state.scratch.max_tokens);
    if (num_tokens > state.scratch.max_tokens) {
        throw std::runtime_error("qwen2_forward_vk: num_tokens exceeds scratch.max_tokens; increase max_batch_tokens");
    }

    const size_t hidden_size = state.cfg.hidden_size;
    const uint16_t* current_hidden = qwen2_forward_hidden_vk(state, token_ids, num_tokens, pool);

    // Final LayerNorm.
    {
        TRACE_SCOPE_CAT("final_norm_vk", "norm");
        state.fp16_ops.rmsnorm(
            descriptor_for_ptr(state.scratch_arena, current_hidden, num_tokens * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, state.weights.final_norm, hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, state.scratch.normed, num_tokens * hidden_size * sizeof(uint16_t)),
            static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(num_tokens),
            state.cfg.rms_norm_eps
        );
    }

    // LM head projection: [num_tokens, hidden] @ [vocab, hidden]^T -> [num_tokens, vocab]
    {
        TRACE_SCOPE_ARGS_CAT(
            "lm_head_vk",
            "gemm",
            ::mruntime::trace_arg("m", static_cast<int64_t>(num_tokens)),
            ::mruntime::trace_arg("n", static_cast<int64_t>(state.cfg.vocab_size)),
            ::mruntime::trace_arg("k", static_cast<int64_t>(hidden_size))
        );
        state.fp16_ops.gemm(
            descriptor_for_ptr(state.scratch_arena, state.scratch.normed, num_tokens * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.weights_arena, state.weights.lm_head,
                               state.cfg.vocab_size * hidden_size * sizeof(uint16_t)),
            descriptor_for_ptr(state.scratch_arena, state.scratch.logits,
                               num_tokens * state.cfg.vocab_size * sizeof(uint16_t)),
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(state.cfg.vocab_size),
            static_cast<uint32_t>(hidden_size)
        );
    }

    return state.scratch.logits;
}

const uint16_t* qwen2_prefill_vk(
    Qwen2VkState& state,
    const int32_t* prompt_tokens,
    size_t prompt_len,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("qwen2_prefill_vk", "forward");

    if (prompt_len == 0) {
        return nullptr;
    }

    qwen2_vk_reset_kv_cache(state);

    size_t processed = 0;
    const uint16_t* last_logits = nullptr;

    while (processed < prompt_len) {
        const size_t chunk = std::min(state.scratch.max_tokens, prompt_len - processed);
        const bool is_last_chunk = (processed + chunk == prompt_len);

        if (is_last_chunk) {
            last_logits = forward_chunk_last_logits_vk(
                state,
                prompt_tokens + processed,
                chunk,
                pool
            );
        } else {
            forward_chunk_no_logits_vk(
                state,
                prompt_tokens + processed,
                chunk,
                pool
            );
        }

        processed += chunk;
    }

    return last_logits;
}

uint16_t* qwen2_decode_vk(
    Qwen2VkState& state,
    int32_t input_token,
    PThreadPool* pool
) {
    TRACE_SCOPE_CAT("qwen2_decode_vk", "forward");

    return qwen2_forward_vk(
        state,
        &input_token,
        1,
        pool
    );
}

}  // namespace mruntime
