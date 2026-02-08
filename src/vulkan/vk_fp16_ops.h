#pragma once

#include <cstddef>
#include <cstdint>

#include "vk_kernel_runtime.h"

namespace mruntime::vulkan {

class VkFp16Ops {
  public:
    VkFp16Ops() = default;
    static VkFp16Ops Create(VkKernelRuntime* runtime);

    VkFp16Ops(const VkFp16Ops&) = delete;
    VkFp16Ops& operator=(const VkFp16Ops&) = delete;

    VkFp16Ops(VkFp16Ops&&) noexcept = default;
    VkFp16Ops& operator=(VkFp16Ops&&) noexcept = default;

    void add(
        const VkDescriptorBufferInfo& a,
        const VkDescriptorBufferInfo& b,
        const VkDescriptorBufferInfo& out,
        uint32_t n
    ) const;

    void mul(
        const VkDescriptorBufferInfo& a,
        const VkDescriptorBufferInfo& b,
        const VkDescriptorBufferInfo& out,
        uint32_t n
    ) const;

    void silu_mul_interleaved(
        const VkDescriptorBufferInfo& gate_up,
        const VkDescriptorBufferInfo& out,
        uint32_t intermediate_size,
        uint32_t num_tokens
    ) const;

    void rmsnorm(
        const VkDescriptorBufferInfo& input,
        const VkDescriptorBufferInfo& weight,
        const VkDescriptorBufferInfo& output,
        uint32_t hidden_size,
        uint32_t num_tokens,
        float eps
    ) const;

    void rope(
        const VkDescriptorBufferInfo& q,
        const VkDescriptorBufferInfo& k,
        const VkDescriptorBufferInfo& rope_cos_sin,
        uint32_t batch,
        uint32_t seq_len,
        uint32_t num_q_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t position_offset
    ) const;

    void transpose_bshd_to_bhsd(
        const VkDescriptorBufferInfo& input,
        const VkDescriptorBufferInfo& output,
        uint32_t B,
        uint32_t S,
        uint32_t H,
        uint32_t D
    ) const;

    void kv_cache_copy(
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
    ) const;

    // C = A @ B^T.  A:[M,K], B:[N,K], C:[M,N] — all row-major FP16.
    void gemm(
        const VkDescriptorBufferInfo& a,
        const VkDescriptorBufferInfo& b,
        const VkDescriptorBufferInfo& c,
        uint32_t M,
        uint32_t N,
        uint32_t K
    ) const;

  private:
    static constexpr uint32_t kLocalSizeX = 256;

    VkKernelRuntime* runtime_ = nullptr;
    VkKernel add_kernel_ = {};
    VkKernel mul_kernel_ = {};
    VkKernel silu_mul_interleaved_kernel_ = {};
    VkKernel rmsnorm_kernel_ = {};
    VkKernel rope_kernel_ = {};
    VkKernel transpose_kernel_ = {};
    VkKernel kv_cache_copy_kernel_ = {};
    VkKernel gemm_kernel_ = {};
};

}  // namespace mruntime::vulkan
