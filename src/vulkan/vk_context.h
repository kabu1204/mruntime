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

    VkQueryPool timestamp_query_pool() const noexcept { return timestamp_query_pool_; }
    float timestamp_period_ns() const noexcept { return timestamp_period_ns_; }
    uint32_t timestamp_valid_bits() const noexcept { return timestamp_valid_bits_; }

    VkDeviceSize min_storage_buffer_offset_alignment() const noexcept {
        return min_storage_buffer_offset_alignment_;
    }

    bool supports_calibrated_timestamps() const noexcept;

    // Sample calibrated timestamps from the device and a host time domain.
    // Returns false when calibrated timestamps are unsupported or no suitable host time domain is available.
    //
    // `out_host_ns` is in nanoseconds for the supported host time domains (CLOCK_MONOTONIC / CLOCK_MONOTONIC_RAW).
    bool calibrated_timestamps_sample(uint64_t* out_device_ticks, uint64_t* out_host_ns, uint64_t* out_max_dev_ns) const;

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
    VkQueryPool timestamp_query_pool_ = VK_NULL_HANDLE;
    VkDeviceSize min_storage_buffer_offset_alignment_ = 0;

    float timestamp_period_ns_ = 0.0f;
    uint32_t timestamp_valid_bits_ = 0;

    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR get_time_domains_ = nullptr;
    PFN_vkGetCalibratedTimestampsKHR get_calibrated_timestamps_ = nullptr;
    VkTimeDomainKHR calibrated_host_domain_ = static_cast<VkTimeDomainKHR>(0);
    bool calibrated_host_domain_is_ns_ = false;
};

}  // namespace mruntime::vulkan
