#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace mruntime::vulkan {

struct VkContextCreateInfo {
    bool enable_validation = false;
};

// Minimal compute-only Vulkan context:
// - Instance + device
// - One compute queue
// - One command pool
//
// No fallback path: if required extensions/features are missing, Create() throws.
class VkContext {
  public:
    VkContext() = default;
    static VkContext Create(const VkContextCreateInfo& info = {});

    ~VkContext();

    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    VkContext(VkContext&& other) noexcept;
    VkContext& operator=(VkContext&& other) noexcept;

    VkInstance instance() const noexcept { return instance_; }
    VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    VkDevice device() const noexcept { return device_; }
    VkPipelineCache pipeline_cache() const noexcept { return pipeline_cache_; }

    uint32_t queue_family_index() const noexcept { return queue_family_index_; }
    VkQueue queue() const noexcept { return queue_; }

    VkCommandPool command_pool() const noexcept { return command_pool_; }

    VkCommandBuffer command_buffer() const noexcept { return command_buffer_; }
    VkFence fence() const noexcept { return fence_; }

    VkDeviceSize min_storage_buffer_offset_alignment() const noexcept {
        return min_storage_buffer_offset_alignment_;
    }

  private:
    void reset() noexcept;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;

    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = UINT32_MAX;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkDeviceSize min_storage_buffer_offset_alignment_ = 0;
};

}  // namespace mruntime::vulkan
