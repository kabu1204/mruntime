#include "vk_fp16_ops.h"

#include <stdexcept>

#include "fp16_add_spv.h"
#include "fp16_attention_decode_gqa_spv.h"
#include "fp16_attention_prefill_gqa_spv.h"
#include "fp16_gemm_spv.h"
#include "fp16_gemm_prefill_wide_spv.h"
#include "fp16_gemv_rows4_vec4_spv.h"
#include "fp16_kv_cache_copy_spv.h"
#include "fp16_mul_spv.h"
#include "fp16_qkv_bias_rope_cache_decode_spv.h"
#include "fp16_qkv_bias_split_spv.h"
#include "fp16_rmsnorm_spv.h"
#include "fp16_rope_spv.h"
#include "fp16_silu_mul_interleaved_spv.h"
#include "fp16_transpose_spv.h"

namespace mruntime::vulkan {

namespace {

KernelCreateInfo make_kernel_create_info(
    const uint8_t* spirv,
    size_t spirv_size,
    uint32_t storage_buffer_count,
    uint32_t push_constant_size,
    uint32_t required_subgroup_size = 0
) {
    KernelCreateInfo info;
    info.spirv = spirv;
    info.spirv_size = spirv_size;
    info.storage_buffer_count = storage_buffer_count;
    info.push_constant_size = push_constant_size;
    info.required_subgroup_size = required_subgroup_size;
    return info;
}

constexpr uint32_t kPointwiseChunkWidth = 8;

void validate_runtime(VkKernelRuntime* runtime) {
    if (runtime == nullptr) {
        throw std::runtime_error("VkFp16Ops::Create: runtime is null");
    }
}

}  // namespace

VkFp16Ops VkFp16Ops::Create(VkKernelRuntime* runtime) {
    validate_runtime(runtime);

    VkFp16Ops ops;
    ops.runtime_ = runtime;
    ops.add_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16AddSpv, shaders::kFp16AddSpvSize, 3, sizeof(uint32_t)));
    ops.mul_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16MulSpv, shaders::kFp16MulSpvSize, 3, sizeof(uint32_t)));
    ops.silu_mul_interleaved_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16SiluMulInterleavedSpv,
            shaders::kFp16SiluMulInterleavedSpvSize,
            2,
            2 * sizeof(uint32_t)));
    ops.rmsnorm_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16RmsnormSpv,
            shaders::kFp16RmsnormSpvSize,
            3,
            2 * sizeof(uint32_t) + sizeof(float)));
    ops.qkv_bias_split_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16QkvBiasSplitSpv,
            shaders::kFp16QkvBiasSplitSpvSize,
            5,
            4 * sizeof(uint32_t)));
    ops.qkv_bias_rope_cache_decode_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16QkvBiasRopeCacheDecodeSpv,
            shaders::kFp16QkvBiasRopeCacheDecodeSpvSize,
            6,
            8 * sizeof(uint32_t)));
    ops.rope_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16RopeSpv, shaders::kFp16RopeSpvSize, 3, 6 * sizeof(uint32_t)));
    ops.transpose_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16TransposeSpv, shaders::kFp16TransposeSpvSize, 2, 4 * sizeof(uint32_t)));
    ops.kv_cache_copy_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16KvCacheCopySpv,
            shaders::kFp16KvCacheCopySpvSize,
            4,
            6 * sizeof(uint32_t)));
    ops.attention_decode_gqa_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16AttentionDecodeGqaSpv,
            shaders::kFp16AttentionDecodeGqaSpvSize,
            4,
            5 * sizeof(uint32_t) + sizeof(float)));
    ops.attention_prefill_gqa_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16AttentionPrefillGqaSpv,
            shaders::kFp16AttentionPrefillGqaSpvSize,
            4,
            6 * sizeof(uint32_t) + sizeof(float)));
    ops.gemv_rows4_vec4_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16GemvRows4Vec4Spv,
            shaders::kFp16GemvRows4Vec4SpvSize,
            3,
            2 * sizeof(uint32_t)));
    ops.gemm_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16GemmSpv, shaders::kFp16GemmSpvSize, 3, 3 * sizeof(uint32_t)));
    ops.gemm_prefill_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16GemmPrefillWideSpv,
            shaders::kFp16GemmPrefillWideSpvSize,
            3,
            3 * sizeof(uint32_t)));

    return ops;
}

void VkFp16Ops::add(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& out,
    uint32_t n,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::add: runtime is null");
    }
    if (n == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {a, b, out};
    const uint32_t push_constants = n;
    runtime_->dispatch_1d(
        add_kernel_,
        buffers,
        3,
        (n + kPointwiseChunkWidth - 1u) / kPointwiseChunkWidth,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::mul(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& out,
    uint32_t n,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::mul: runtime is null");
    }
    if (n == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {a, b, out};
    const uint32_t push_constants = n;
    runtime_->dispatch_1d(
        mul_kernel_,
        buffers,
        3,
        (n + kPointwiseChunkWidth - 1u) / kPointwiseChunkWidth,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::silu_mul_interleaved(
    const VkDescriptorBufferInfo& gate_up,
    const VkDescriptorBufferInfo& out,
    uint32_t intermediate_size,
    uint32_t num_tokens,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::silu_mul_interleaved: runtime is null");
    }
    if (num_tokens == 0 || intermediate_size == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[2] = {gate_up, out};
    struct {
        uint32_t intermediate_size;
        uint32_t num_tokens;
    } push_constants = {intermediate_size, num_tokens};

    const uint32_t total_elements = num_tokens * intermediate_size;
    runtime_->dispatch_1d(
        silu_mul_interleaved_kernel_,
        buffers,
        2,
        (total_elements + kPointwiseChunkWidth - 1u) / kPointwiseChunkWidth,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::rmsnorm(
    const VkDescriptorBufferInfo& input,
    const VkDescriptorBufferInfo& weight,
    const VkDescriptorBufferInfo& output,
    uint32_t hidden_size,
    uint32_t num_tokens,
    float eps,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::rmsnorm: runtime is null");
    }
    if (num_tokens == 0 || hidden_size == 0) {
        return;
    }
    if ((hidden_size & 3u) != 0u) {
        throw std::runtime_error("VkFp16Ops::rmsnorm: hidden_size must be divisible by 4");
    }

    const VkDescriptorBufferInfo buffers[3] = {input, weight, output};
    struct {
        uint32_t hidden_size;
        uint32_t num_tokens;
        float eps;
    } push_constants = {hidden_size, num_tokens, eps};

    runtime_->dispatch_1d(
        rmsnorm_kernel_,
        buffers,
        3,
        num_tokens * kRmsnormLocalSizeX,
        kRmsnormLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::qkv_bias_split(
    const VkDescriptorBufferInfo& qkv,
    const VkDescriptorBufferInfo& bias,
    const VkDescriptorBufferInfo& q_out,
    const VkDescriptorBufferInfo& k_out,
    const VkDescriptorBufferInfo& v_out,
    uint32_t num_tokens,
    uint32_t q_dim,
    uint32_t kv_dim,
    bool has_bias,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::qkv_bias_split: runtime is null");
    }
    if (num_tokens == 0 || q_dim == 0 || kv_dim == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[5] = {qkv, bias, q_out, k_out, v_out};
    struct {
        uint32_t num_tokens;
        uint32_t q_dim;
        uint32_t kv_dim;
        uint32_t has_bias;
    } push_constants = {num_tokens, q_dim, kv_dim, has_bias ? 1u : 0u};

    const uint32_t qkv_dim = q_dim + 2u * kv_dim;
    runtime_->dispatch_1d(
        qkv_bias_split_kernel_,
        buffers,
        5,
        (num_tokens * qkv_dim + kPointwiseChunkWidth - 1u) / kPointwiseChunkWidth,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        -1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::rope(
    const VkDescriptorBufferInfo& q,
    const VkDescriptorBufferInfo& k,
    const VkDescriptorBufferInfo& rope_cos_sin,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t position_offset,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::rope: runtime is null");
    }
    if (batch_size == 0 || seq_len == 0 || head_dim == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {q, k, rope_cos_sin};
    struct {
        uint32_t batch;
        uint32_t seq_len;
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        uint32_t position_offset;
    } push_constants = {batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset};

    const uint32_t half_dim = head_dim / 2;
    const uint32_t total = batch_size * seq_len * (num_q_heads + num_kv_heads) * half_dim;
    runtime_->dispatch_1d(
        rope_kernel_,
        buffers,
        3,
        total,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        -1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::qkv_bias_rope_cache_decode(
    const VkDescriptorBufferInfo& qkv,
    const VkDescriptorBufferInfo& bias,
    const VkDescriptorBufferInfo& rope_cos_sin,
    const VkDescriptorBufferInfo& q_out,
    const VkDescriptorBufferInfo& k_cache,
    const VkDescriptorBufferInfo& v_cache,
    uint32_t q_dim,
    uint32_t kv_dim,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len,
    uint32_t position_offset,
    bool has_bias,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::qkv_bias_rope_cache_decode: runtime is null");
    }
    if (q_dim == 0 || kv_dim == 0 || num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0) {
        return;
    }
    if ((head_dim & 1u) != 0u) {
        throw std::runtime_error("VkFp16Ops::qkv_bias_rope_cache_decode: head_dim must be even");
    }
    if (q_dim != num_q_heads * head_dim) {
        throw std::runtime_error("VkFp16Ops::qkv_bias_rope_cache_decode: q_dim mismatch");
    }
    if (kv_dim != num_kv_heads * head_dim) {
        throw std::runtime_error("VkFp16Ops::qkv_bias_rope_cache_decode: kv_dim mismatch");
    }
    if (position_offset >= max_seq_len) {
        throw std::runtime_error("VkFp16Ops::qkv_bias_rope_cache_decode: position_offset out of range");
    }

    const VkDescriptorBufferInfo buffers[6] = {qkv, bias, rope_cos_sin, q_out, k_cache, v_cache};
    struct {
        uint32_t q_dim;
        uint32_t kv_dim;
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        uint32_t max_seq_len;
        uint32_t position_offset;
        uint32_t has_bias;
    } push_constants = {
        q_dim,
        kv_dim,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        position_offset,
        has_bias ? 1u : 0u,
    };

    const uint32_t half_dim = head_dim / 2u;
    const uint32_t total = (num_q_heads + num_kv_heads) * half_dim + kv_dim;
    runtime_->dispatch_1d(
        qkv_bias_rope_cache_decode_kernel_,
        buffers,
        6,
        total,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        -1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::transpose_bshd_to_bhsd(
    const VkDescriptorBufferInfo& input,
    const VkDescriptorBufferInfo& output,
    uint32_t B,
    uint32_t S,
    uint32_t H,
    uint32_t D,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::transpose_bshd_to_bhsd: runtime is null");
    }
    if (B == 0 || S == 0 || H == 0 || D == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[2] = {input, output};
    struct {
        uint32_t B;
        uint32_t S;
        uint32_t H;
        uint32_t D;
    } push_constants = {B, S, H, D};

    runtime_->dispatch_1d(
        transpose_kernel_,
        buffers,
        2,
        B * H * S,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::kv_cache_copy(
    const VkDescriptorBufferInfo& k_in,
    const VkDescriptorBufferInfo& v_in,
    const VkDescriptorBufferInfo& k_cache,
    const VkDescriptorBufferInfo& v_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len,
    uint32_t position_offset,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::kv_cache_copy: runtime is null");
    }
    if (batch_size == 0 || seq_len == 0 || num_kv_heads == 0 || head_dim == 0) {
        return;
    }

    VkDescriptorBufferInfo buffers[4] = {k_in, v_in, k_cache, v_cache};
    buffers[2].range = 0;

    struct {
        uint32_t batch;
        uint32_t seq_len;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        uint32_t max_seq_len;
        uint32_t position_offset;
    } push_constants = {batch_size, seq_len, num_kv_heads, head_dim, max_seq_len, position_offset};

    runtime_->dispatch_1d(
        kv_cache_copy_kernel_,
        buffers,
        4,
        batch_size * seq_len * num_kv_heads,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::attention_decode_gqa(
    const VkDescriptorBufferInfo& q,
    const VkDescriptorBufferInfo& k,
    const VkDescriptorBufferInfo& v,
    const VkDescriptorBufferInfo& out,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len,
    uint32_t kv_stride,
    uint32_t head_dim,
    float scale,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::attention_decode_gqa: runtime is null");
    }
    if (num_q_heads == 0 || num_kv_heads == 0 || kv_len == 0 || head_dim == 0) {
        return;
    }
    if (kv_len > kv_stride) {
        throw std::runtime_error("VkFp16Ops::attention_decode_gqa: kv_len must be <= kv_stride");
    }
    if ((num_q_heads % num_kv_heads) != 0u) {
        throw std::runtime_error("VkFp16Ops::attention_decode_gqa: num_q_heads must be divisible by num_kv_heads");
    }
    if (head_dim > 512u) {
        throw std::runtime_error("VkFp16Ops::attention_decode_gqa: head_dim must be <= 512");
    }

    const VkDescriptorBufferInfo buffers[4] = {q, k, v, out};
    struct {
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t kv_len;
        uint32_t kv_stride;
        uint32_t head_dim;
        float scale;
    } push_constants = {num_q_heads, num_kv_heads, kv_len, kv_stride, head_dim, scale};

    runtime_->dispatch_1d(
        attention_decode_gqa_kernel_,
        buffers,
        4,
        num_q_heads * kAttentionGqaLocalSizeX,
        kAttentionGqaLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        -1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::attention_prefill_gqa(
    const VkDescriptorBufferInfo& q,
    const VkDescriptorBufferInfo& k,
    const VkDescriptorBufferInfo& v,
    const VkDescriptorBufferInfo& out,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t q_len,
    uint32_t kv_len,
    uint32_t kv_stride,
    uint32_t head_dim,
    float scale,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::attention_prefill_gqa: runtime is null");
    }
    if (num_q_heads == 0 || num_kv_heads == 0 || q_len == 0 || kv_len == 0 || head_dim == 0) {
        return;
    }
    if (q_len > kv_len) {
        throw std::runtime_error("VkFp16Ops::attention_prefill_gqa: q_len must be <= kv_len");
    }
    if (kv_len > kv_stride) {
        throw std::runtime_error("VkFp16Ops::attention_prefill_gqa: kv_len must be <= kv_stride");
    }
    if ((num_q_heads % num_kv_heads) != 0u) {
        throw std::runtime_error("VkFp16Ops::attention_prefill_gqa: num_q_heads must be divisible by num_kv_heads");
    }
    if (head_dim > 512u) {
        throw std::runtime_error("VkFp16Ops::attention_prefill_gqa: head_dim must be <= 512");
    }

    const VkDescriptorBufferInfo buffers[4] = {q, k, v, out};
    struct {
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t q_len;
        uint32_t kv_len;
        uint32_t kv_stride;
        uint32_t head_dim;
        float scale;
    } push_constants = {num_q_heads, num_kv_heads, q_len, kv_len, kv_stride, head_dim, scale};

    runtime_->dispatch_2d(
        attention_prefill_gqa_kernel_,
        buffers,
        4,
        num_q_heads,
        q_len,
        1,
        1,
        &push_constants,
        sizeof(push_constants),
        -1,
        VK_NULL_HANDLE,
        batch
    );
}

void VkFp16Ops::gemm(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    VkQueryPool query_pool,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::gemm: runtime is null");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {a, b, c};
    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } push_constants = {M, N, K};

    constexpr uint32_t tile_M = 64;
    constexpr uint32_t tile_N = 64;
    runtime_->dispatch_2d(
        gemm_kernel_,
        buffers,
        3,
        N,
        M,
        tile_N,
        tile_M,
        &push_constants,
        sizeof(push_constants),
        2,
        query_pool,
        batch
    );
}

void VkFp16Ops::gemm_prefill(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    VkQueryPool query_pool,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::gemm_prefill: runtime is null");
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }
    if ((K & 3u) != 0u) {
        gemm(a, b, c, M, N, K, query_pool, batch);
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {a, b, c};
    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } push_constants = {M, N, K};

    constexpr uint32_t tile_M = 64;
    constexpr uint32_t tile_N = 128;
    runtime_->dispatch_2d(
        gemm_prefill_kernel_,
        buffers,
        3,
        tile_N * ((N + tile_N - 1u) / tile_N),
        tile_M * ((M + tile_M - 1u) / tile_M),
        tile_N,
        tile_M,
        &push_constants,
        sizeof(push_constants),
        2,
        query_pool,
        batch
    );
}

void VkFp16Ops::gemv(
    const VkDescriptorBufferInfo& x,
    const VkDescriptorBufferInfo& w,
    const VkDescriptorBufferInfo& y,
    uint32_t N,
    uint32_t K,
    VkQueryPool query_pool,
    VkDispatchBatch* batch
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::gemv: runtime is null");
    }
    if (N == 0 || K == 0) {
        return;
    }
    if ((K & 3u) != 0u) {
        throw std::runtime_error("VkFp16Ops::gemv: K must be divisible by 4");
    }

    const VkDescriptorBufferInfo buffers[3] = {x, w, y};
    struct {
        uint32_t N;
        uint32_t K;
    } push_constants = {N, K};

    const uint64_t group_count_x =
        (static_cast<uint64_t>(N) + kGemvRows4Vec4RowsPerWg - 1) / kGemvRows4Vec4RowsPerWg;
    const uint64_t element_count64 = group_count_x * kGemvRows4Vec4LocalSizeX;
    if (element_count64 > UINT32_MAX) {
        throw std::runtime_error("VkFp16Ops::gemv: N too large");
    }

    runtime_->dispatch_1d(
        gemv_rows4_vec4_kernel_,
        buffers,
        3,
        static_cast<uint32_t>(element_count64),
        kGemvRows4Vec4LocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2,
        query_pool,
        batch
    );
}

}  // namespace mruntime::vulkan
