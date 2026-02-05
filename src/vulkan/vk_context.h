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

    uint32_t queue_family_index() const noexcept { return queue_family_index_; }
    VkQueue queue() const noexcept { return queue_; }

    VkCommandPool command_pool() const noexcept { return command_pool_; }

    VkDeviceSize min_storage_buffer_offset_alignment() const noexcept {
        return min_storage_buffer_offset_alignment_;
    }

  private:
    void reset() noexcept;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = UINT32_MAX;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDeviceSize min_storage_buffer_offset_alignment_ = 0;
};

}  // namespace mruntime::vulkan

