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

  private:
    static constexpr uint32_t kLocalSizeX = 256;

    VkKernelRuntime* runtime_ = nullptr;
    VkKernel add_kernel_ = {};
    VkKernel mul_kernel_ = {};
};

}  // namespace mruntime::vulkan
