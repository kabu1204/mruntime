#pragma once

#include <cstddef>
#include <cstdint>

#include "mruntime/pthreadpool_raii.h"

namespace mruntime {

// ============================================================================
// Qwen2 Core Operations - All FP16, no dtype dispatch
// ============================================================================
// All operations work with raw FP16 (uint16_t) pointers.
// Dimensions are passed explicitly - no shape inference.
// PThreadPool* can be nullptr for single-threaded execution.

// ============ GEMM ============
// C = A @ B^T  (A: [M, K], B: [N, K], C: [M, N])
// Uses KleidiAI fast path on Arm64 when available.
//
// Parameters:
//   A: LHS matrix [M, K] row-major
//   B: RHS matrix [N, K] row-major (will be transposed internally)
//   C: Output matrix [M, N] row-major
//   M, N, K: Dimensions
//   packed_B: Pre-packed RHS from qwen2_pack_weight_fp16 (nullptr to pack on-the-fly)
//   pool: Thread pool (nullptr for single-threaded)
void qwen2_gemm_fp16(
    const uint16_t* A,
    const uint16_t* B,
    uint16_t* C,
    size_t M, size_t N, size_t K,
    const uint16_t* packed_B,
    PThreadPool* pool
);

// Returns true when KleidiAI FP16 kernels are available at runtime.
bool qwen2_has_kai_fp16();

// Get size needed for packed weight buffer
size_t qwen2_packed_weight_size_fp16(size_t N, size_t K);

// Pack weight matrix for repeated GEMM (call once at load time)
// B: [N, K] row-major weight
// packed: Output buffer (must be >= qwen2_packed_weight_size_fp16 bytes)
void qwen2_pack_weight_fp16(
    const uint16_t* B,
    uint16_t* packed,
    size_t N, size_t K
);

// ============ RMSNorm ============
// out[i] = (x[i] / rms(x)) * weight[i]
// where rms(x) = sqrt(mean(x^2) + eps)
void qwen2_rmsnorm_fp16(
    const uint16_t* input,      // [num_tokens, hidden_size]
    const uint16_t* weight,     // [hidden_size]
    uint16_t* output,           // [num_tokens, hidden_size]
    size_t num_tokens,
    size_t hidden_size,
    float eps,
    PThreadPool* pool
);

// ============ RoPE ============
// Apply rotary position embeddings in-place to Q and K.
// Layout: [batch, seq_len, num_heads, head_dim]
void qwen2_rope_fp16(
    uint16_t* Q,
    uint16_t* K,
    size_t batch,
    size_t seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t position_offset,
    const float* rope_cos_sin,     // [rope_max_seq_len, head_dim/2, 2] interleaved (cos, sin)
    size_t rope_max_seq_len,
    PThreadPool* pool
);

// ============ Flash Attention ============
// O = softmax(Q @ K^T / sqrt(d)) @ V
// Layout: [batch, num_heads, seq_len, head_dim]
void qwen2_flash_attention_fp16(
    const uint16_t* Q,      // [batch, num_heads, q_len, head_dim]
    const uint16_t* K,      // [batch, num_heads, kv_len, head_dim]
    const uint16_t* V,      // [batch, num_heads, kv_len, head_dim]
    uint16_t* O,            // [batch, num_heads, q_len, head_dim]
    size_t batch,
    size_t num_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    float scale,
    bool causal,
    PThreadPool* pool
);

// ============ Activation Functions ============

// SiLU: out = x * sigmoid(x)
void qwen2_silu_fp16(
    const uint16_t* input,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
);

// Element-wise multiply: out = a * b
void qwen2_mul_fp16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
);

// Fused SiLU + Mul: out = silu(gate) * up
// More efficient than separate silu + mul calls
void qwen2_silu_mul_fp16(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
);

// Fused SiLU + Mul for interleaved (gate|up) layout:
// gate_up: [num_tokens, 2 * intermediate_size], where each row is [gate..., up...]
// output:  [num_tokens, intermediate_size]
void qwen2_silu_mul_interleaved_fp16(
    const uint16_t* gate_up,
    uint16_t* output,
    size_t num_tokens,
    size_t intermediate_size,
    PThreadPool* pool
);

// Element-wise add: out = a + b
void qwen2_add_fp16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* output,
    size_t n,
    PThreadPool* pool
);

// ============ Embedding ============
// Look up token embeddings
void qwen2_embedding_lookup_fp16(
    const uint16_t* weight,     // [vocab_size, hidden_size]
    const int32_t* token_ids,   // [num_tokens]
    uint16_t* output,           // [num_tokens, hidden_size]
    size_t num_tokens,
    size_t vocab_size,
    size_t hidden_size
);

// ============ Softmax (for final logits) ============
// Softmax over the last dimension
void qwen2_softmax_fp16(
    const uint16_t* input,      // [num_tokens, vocab_size]
    uint16_t* output,           // [num_tokens, vocab_size]
    size_t num_tokens,
    size_t vocab_size,
    PThreadPool* pool
);

// ============ Copy/Layout Utilities ============

// Copy K/V to cache: updates cache[:, :, pos:pos+seq_len, :]
// Input layout:  [batch, seq_len, num_kv_heads, head_dim]
// Cache layout:  [batch, num_kv_heads, max_seq_len, head_dim]
void qwen2_copy_to_kv_cache_fp16(
    const uint16_t* k,          // [batch, seq_len, num_kv_heads, head_dim]
    const uint16_t* v,
    uint16_t* k_cache,          // [batch, num_kv_heads, max_seq_len, head_dim]
    uint16_t* v_cache,
    size_t batch,
    size_t seq_len,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t position_offset
);

// Transpose [B, S, H, D] -> [B, H, S, D]
void qwen2_transpose_bshd_to_bhsd_fp16(
    const uint16_t* input,
    uint16_t* output,
    size_t B, size_t S, size_t H, size_t D,
    PThreadPool* pool
);

}  // namespace mruntime
