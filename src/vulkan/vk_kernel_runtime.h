#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "vk_compute_pipeline.h"
#include "vk_context.h"

namespace mruntime::vulkan {

class VkKernelRuntime;

struct VkDispatchBatchHostBarrier {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

class VkDispatchBatch {
  public:
    VkDispatchBatch() = default;

    VkDispatchBatch(const VkDispatchBatch&) = delete;
    VkDispatchBatch& operator=(const VkDispatchBatch&) = delete;

    VkDispatchBatch(VkDispatchBatch&&) noexcept = default;
    VkDispatchBatch& operator=(VkDispatchBatch&&) noexcept = default;

    bool recording() const noexcept { return recording_; }

  private:
    friend class VkKernelRuntime;

    struct TraceRecord {
        uint32_t query_index = 0;
        uint32_t group_count_x = 0;
        uint32_t group_count_y = 0;
        uint32_t group_count_z = 0;
    };

    const VkKernelRuntime* runtime_ = nullptr;
    bool recording_ = false;
    bool enable_trace_ = false;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    uint32_t next_query_index_ = 0;
    VkDispatchTraceInfo trace_info_ = {};
    int64_t record_begin_us_ = 0;
    std::vector<TraceRecord> trace_records_;
    std::unordered_map<const VkComputePipeline*, uint32_t> descriptor_set_use_counts_;
};

struct KernelCreateInfo {
    const uint8_t* spirv = nullptr;
    size_t spirv_size = 0;
    uint32_t storage_buffer_count = 0;
    uint32_t push_constant_size = 0;
};

struct VkKernel {
    // Non-owning pointer into VkKernelRuntime's internal pipeline cache.
    // Contract: any VkKernel handle is valid only while the originating
    // VkKernelRuntime instance remains alive.
    const VkComputePipeline* pipeline = nullptr;
    uint32_t storage_buffer_count = 0;
    uint32_t push_constant_size = 0;

    explicit operator bool() const noexcept { return pipeline != nullptr; }
};

// Tiny compute-kernel runtime for Vulkan:
// - Caches pipelines by SPIR-V blob + binding/push-constant layout.
// - Provides simple 1D/2D dispatch helpers.
class VkKernelRuntime {
  public:
    VkKernelRuntime() = default;
    static VkKernelRuntime Create(const VkContext& context);

    VkKernelRuntime(const VkKernelRuntime&) = delete;
    VkKernelRuntime& operator=(const VkKernelRuntime&) = delete;

    VkKernelRuntime(VkKernelRuntime&&) noexcept = default;
    VkKernelRuntime& operator=(VkKernelRuntime&&) noexcept = default;

    VkKernel get_or_create_kernel(const KernelCreateInfo& info);

    void set_timing_enabled(bool enabled) noexcept { timing_enabled_ = enabled; }
    bool timing_enabled() const noexcept { return timing_enabled_; }

    VkDispatchBatch begin_batch() const;
    void finish_batch(
        VkDispatchBatch* batch,
        const VkDispatchBatchHostBarrier* host_barriers,
        uint32_t host_barrier_count
    ) const;

    void dispatch_1d(
        VkKernel kernel,
        const VkDescriptorBufferInfo* buffers,
        uint32_t buffer_count,
        uint32_t element_count,
        uint32_t local_size_x,
        const void* push_constants,
        uint32_t push_constants_size,
        int32_t host_read_buffer_index = -1,
        VkQueryPool query_pool = VK_NULL_HANDLE,
        VkDispatchBatch* batch = nullptr
    ) const;

    void dispatch_2d(
        VkKernel kernel,
        const VkDescriptorBufferInfo* buffers,
        uint32_t buffer_count,
        uint32_t width,
        uint32_t height,
        uint32_t local_size_x,
        uint32_t local_size_y,
        const void* push_constants,
        uint32_t push_constants_size,
        int32_t host_read_buffer_index = -1,
        VkQueryPool query_pool = VK_NULL_HANDLE,
        VkDispatchBatch* batch = nullptr
    ) const;

  private:
    void maybe_refresh_calibration() const;

    struct PipelineCacheKey {
        std::vector<uint8_t> spirv;
        uint32_t storage_buffer_count = 0;
        uint32_t push_constant_size = 0;

        bool operator==(const PipelineCacheKey& other) const noexcept;
    };

    struct PipelineCacheKeyHash {
        size_t operator()(const PipelineCacheKey& key) const noexcept;
    };

    const VkContext* context_ = nullptr;
    std::unordered_map<PipelineCacheKey, std::unique_ptr<VkComputePipeline>, PipelineCacheKeyHash> pipeline_cache_;

    bool timing_enabled_ = false;

    struct CalibrationCache {
        bool valid = false;
        uint64_t device_ticks = 0;
        uint64_t host_ns = 0;
        uint64_t max_dev_ns = 0;
        int64_t trace_base_us = 0;
        std::chrono::steady_clock::time_point sampled_at = {};
    };
    mutable CalibrationCache calibration_cache_;
};

}  // namespace mruntime::vulkan
