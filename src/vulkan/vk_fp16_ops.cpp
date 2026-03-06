#include "vk_fp16_ops.h"

#include <stdexcept>

#include "fp16_add_spv.h"
#include "fp16_attention_decode_gqa_spv.h"
#include "fp16_gemv_rows4_vec4_spv.h"
#include "fp16_gemm_spv.h"
#include "fp16_kv_cache_copy_spv.h"
#include "fp16_mul_spv.h"
#include "fp16_rmsnorm_spv.h"
#include "fp16_rope_spv.h"
#include "fp16_silu_mul_interleaved_spv.h"
#include "fp16_transpose_spv.h"

namespace mruntime::vulkan {

namespace {

KernelCreateInfo make_kernel_create_info(
    const uint8_t* spirv, size_t spirv_size,
    uint32_t storage_buffer_count, uint32_t push_constant_size) {
    KernelCreateInfo info;
    info.spirv = spirv;
    info.spirv_size = spirv_size;
    info.storage_buffer_count = storage_buffer_count;
    info.push_constant_size = push_constant_size;
    return info;
}

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

    // add / mul: 3 buffers, 1 uint push constant
    ops.add_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16AddSpv, shaders::kFp16AddSpvSize, 3, sizeof(uint32_t)));
    ops.mul_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16MulSpv, shaders::kFp16MulSpvSize, 3, sizeof(uint32_t)));

    // silu_mul_interleaved: 2 buffers (gate_up, out), 2 uint push constants
    ops.silu_mul_interleaved_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16SiluMulInterleavedSpv, shaders::kFp16SiluMulInterleavedSpvSize,
                                2, 2 * sizeof(uint32_t)));

    // rmsnorm: 3 buffers (input, weight, out), push = {uint, uint, float}
    ops.rmsnorm_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16RmsnormSpv, shaders::kFp16RmsnormSpvSize,
                                3, 2 * sizeof(uint32_t) + sizeof(float)));

    // rope: 3 buffers (q RW, k RW, cos_sin RO), push = 6 uint
    ops.rope_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16RopeSpv, shaders::kFp16RopeSpvSize,
                                3, 6 * sizeof(uint32_t)));

    // transpose: 2 buffers (input, out), push = 4 uint
    ops.transpose_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16TransposeSpv, shaders::kFp16TransposeSpvSize,
                                2, 4 * sizeof(uint32_t)));

    // kv_cache_copy: 4 buffers (k_in, v_in, k_cache, v_cache), push = 6 uint
    ops.kv_cache_copy_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16KvCacheCopySpv, shaders::kFp16KvCacheCopySpvSize,
                                4, 6 * sizeof(uint32_t)));

    // decode GQA attention: 4 buffers (Q, K, V, O), push = 5 uint + 1 float
    ops.attention_decode_gqa_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(
            shaders::kFp16AttentionDecodeGqaSpv,
            shaders::kFp16AttentionDecodeGqaSpvSize,
            4,
            5 * sizeof(uint32_t) + sizeof(float)));

    // gemv (rows4 + vec4): 3 buffers (x, W, y), push = {uint N, uint K}
    ops.gemv_rows4_vec4_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16GemvRows4Vec4Spv, shaders::kFp16GemvRows4Vec4SpvSize,
                                3, 2 * sizeof(uint32_t)));

    // gemm: 3 buffers (A, B, C), push = {uint M, uint N, uint K}
    ops.gemm_kernel_ = runtime->get_or_create_kernel(
        make_kernel_create_info(shaders::kFp16GemmSpv, shaders::kFp16GemmSpvSize,
                                3, 3 * sizeof(uint32_t)));

    return ops;
}

void VkFp16Ops::add(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& out,
    uint32_t n
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
        n,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2
    );
}

void VkFp16Ops::mul(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& out,
    uint32_t n
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
        n,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2
    );
}

void VkFp16Ops::silu_mul_interleaved(
    const VkDescriptorBufferInfo& gate_up,
    const VkDescriptorBufferInfo& out,
    uint32_t intermediate_size,
    uint32_t num_tokens
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

    runtime_->dispatch_1d(
        silu_mul_interleaved_kernel_,
        buffers,
        2,
        num_tokens * intermediate_size,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        1
    );
}

void VkFp16Ops::rmsnorm(
    const VkDescriptorBufferInfo& input,
    const VkDescriptorBufferInfo& weight,
    const VkDescriptorBufferInfo& output,
    uint32_t hidden_size,
    uint32_t num_tokens,
    float eps
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::rmsnorm: runtime is null");
    }
    if (num_tokens == 0 || hidden_size == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {input, weight, output};

    struct {
        uint32_t hidden_size;
        uint32_t num_tokens;
        float eps;
    } push_constants = {hidden_size, num_tokens, eps};

    // Dispatch exactly num_tokens workgroups (one per token).
    // Pass element_count = num_tokens * kLocalSizeX so dispatch_1d computes
    // ceil(num_tokens * 256 / 256) = num_tokens workgroups.
    runtime_->dispatch_1d(
        rmsnorm_kernel_,
        buffers,
        3,
        num_tokens * kLocalSizeX,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2
    );
}

void VkFp16Ops::rope(
    const VkDescriptorBufferInfo& q,
    const VkDescriptorBufferInfo& k,
    const VkDescriptorBufferInfo& rope_cos_sin,
    uint32_t batch,
    uint32_t seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t position_offset
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::rope: runtime is null");
    }
    if (batch == 0 || seq_len == 0 || head_dim == 0) {
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
    } push_constants = {batch, seq_len, num_q_heads, num_kv_heads, head_dim, position_offset};

    const uint32_t half_dim = head_dim / 2;
    const uint32_t total = batch * seq_len * (num_q_heads + num_kv_heads) * half_dim;

    runtime_->dispatch_1d(
        rope_kernel_,
        buffers,
        3,
        total,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        -1
    );
}

void VkFp16Ops::transpose_bshd_to_bhsd(
    const VkDescriptorBufferInfo& input,
    const VkDescriptorBufferInfo& output,
    uint32_t B,
    uint32_t S,
    uint32_t H,
    uint32_t D
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

    // One invocation per (b, h, s) triple.
    const uint32_t total_tasks = B * H * S;

    runtime_->dispatch_1d(
        transpose_kernel_,
        buffers,
        2,
        total_tasks,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        1
    );
}

void VkFp16Ops::kv_cache_copy(
    const VkDescriptorBufferInfo& k_in,
    const VkDescriptorBufferInfo& v_in,
    const VkDescriptorBufferInfo& k_cache,
    const VkDescriptorBufferInfo& v_cache,
    uint32_t batch,
    uint32_t seq_len,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len,
    uint32_t position_offset
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::kv_cache_copy: runtime is null");
    }
    if (batch == 0 || seq_len == 0 || num_kv_heads == 0 || head_dim == 0) {
        return;
    }

    VkDescriptorBufferInfo buffers[4] = {k_in, v_in, k_cache, v_cache};
    // Host reads often follow (e.g. CPU attention over the KV cache). Ensure we emit a host-read
    // barrier that covers both K and V cache ranges when they are suballocated from the same
    // VkBufferArena (V is typically allocated after K). We do this by making K's descriptor range
    // VK_WHOLE_SIZE for the barrier selection below.
    buffers[2].range = 0;

    struct {
        uint32_t batch;
        uint32_t seq_len;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        uint32_t max_seq_len;
        uint32_t position_offset;
    } push_constants = {batch, seq_len, num_kv_heads, head_dim, max_seq_len, position_offset};

    const uint32_t total = batch * seq_len * num_kv_heads;

    runtime_->dispatch_1d(
        kv_cache_copy_kernel_,
        buffers,
        4,
        total,
        kLocalSizeX,
        &push_constants,
        sizeof(push_constants),
        2
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
    float scale
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
        num_q_heads * kGemvRows4Vec4LocalSizeX,
        kGemvRows4Vec4LocalSizeX,
        &push_constants,
        sizeof(push_constants),
        -1
    );
}

void VkFp16Ops::gemm(
    const VkDescriptorBufferInfo& a,
    const VkDescriptorBufferInfo& b,
    const VkDescriptorBufferInfo& c,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    VkQueryPool query_pool
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
        N,  // width  = columns
        M,  // height = rows
        tile_N,
        tile_M,
        &push_constants,
        sizeof(push_constants),
        2,  // host-read buffer index = C
        query_pool
    );
}

void VkFp16Ops::gemv(
    const VkDescriptorBufferInfo& x,
    const VkDescriptorBufferInfo& w,
    const VkDescriptorBufferInfo& y,
    uint32_t N,
    uint32_t K,
    VkQueryPool query_pool
) const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("VkFp16Ops::gemv: runtime is null");
    }
    if (N == 0 || K == 0) {
        return;
    }

    const VkDescriptorBufferInfo buffers[3] = {x, w, y};

    struct {
        uint32_t N;
        uint32_t K;
    } push_constants = {N, K};

    if ((K & 3u) != 0u) {
        throw std::runtime_error("VkFp16Ops::gemv: K must be divisible by 4");
    }

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
        2,  // host-read buffer index = y
        query_pool
    );
}

}  // namespace mruntime::vulkan
