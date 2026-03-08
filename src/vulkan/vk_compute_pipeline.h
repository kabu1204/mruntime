#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vk_context.h"

namespace mruntime::vulkan {

struct ComputePipelineCreateInfo {
    const uint8_t* spirv = nullptr;
    size_t spirv_size = 0;
    uint32_t storage_buffer_count = 0;
    uint32_t push_constant_size = 0;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
};

struct VkDispatchTraceInfo {
    bool enable_timing_trace = false;

    // Calibrated timestamp mapping (best-effort).
    // When `has_calibrated_timestamps` is true, `calibrated_device_ticks` and
    // `calibrated_trace_base_us` define the mapping anchor.
    bool has_calibrated_timestamps = false;
    uint64_t calibrated_device_ticks = 0;
    int64_t calibrated_trace_base_us = 0;
    uint64_t calibrated_max_dev_ns = 0;
};

class VkComputePipeline {
  public:
    static constexpr uint32_t kBatchDescriptorSetCapacity = 2048;

    VkComputePipeline() = default;
    static VkComputePipeline Create(VkDevice device, const ComputePipelineCreateInfo& info);

    ~VkComputePipeline();

    VkComputePipeline(const VkComputePipeline&) = delete;
    VkComputePipeline& operator=(const VkComputePipeline&) = delete;

    VkComputePipeline(VkComputePipeline&& other) noexcept;
    VkComputePipeline& operator=(VkComputePipeline&& other) noexcept;

    VkPipeline pipeline() const noexcept { return pipeline_; }
    VkPipelineLayout pipeline_layout() const noexcept { return pipeline_layout_; }
    VkDescriptorSetLayout descriptor_set_layout() const noexcept { return descriptor_set_layout_; }
    uint32_t batch_descriptor_set_capacity() const noexcept {
        return static_cast<uint32_t>(descriptor_sets_.size());
    }

    void dispatch_and_wait(
        const VkContext& ctx,
        const VkDescriptorBufferInfo* buffers,
        uint32_t buffer_count,
        uint32_t group_count_x,
        uint32_t group_count_y,
        uint32_t group_count_z,
        const void* push_constants,
        uint32_t push_constants_size,
        VkBuffer host_read_buffer = VK_NULL_HANDLE,
        VkDeviceSize host_read_offset = 0,
        VkDeviceSize host_read_size = 0,
        VkQueryPool query_pool = VK_NULL_HANDLE,
        const VkDispatchTraceInfo* trace = nullptr
    ) const;

    void record_dispatch(
        const VkContext& ctx,
        VkCommandBuffer command_buffer,
        uint32_t descriptor_set_index,
        const VkDescriptorBufferInfo* buffers,
        uint32_t buffer_count,
        uint32_t group_count_x,
        uint32_t group_count_y,
        uint32_t group_count_z,
        const void* push_constants,
        uint32_t push_constants_size,
        VkQueryPool query_pool = VK_NULL_HANDLE,
        uint32_t query_index = UINT32_MAX
    ) const;

  private:
    void destroy() noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t storage_buffer_count_ = 0;
    uint32_t push_constant_size_ = 0;

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptor_sets_;
    VkDescriptorUpdateTemplate descriptor_update_template_ = VK_NULL_HANDLE;
};

}  // namespace mruntime::vulkan
