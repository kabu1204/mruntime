#include "vk_kernel_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mruntime/trace.h"
#include "vk_command.h"
#include "vk_helpers.h"

namespace mruntime::vulkan {

namespace {

constexpr size_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr size_t kFnvPrime = 1099511628211ull;
constexpr uint32_t kStackBufferCapacity = 8;

size_t fnv1a_hash_bytes(const uint8_t* data, size_t size) {
    size_t hash = kFnvOffsetBasis;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<size_t>(data[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

void validate_kernel(const VkKernel& kernel) {
    if (!kernel) {
        throw std::runtime_error("VkKernelRuntime: kernel is null");
    }
}

void validate_dispatch_args(
    const VkKernel& kernel,
    const VkDescriptorBufferInfo* buffers,
    uint32_t buffer_count,
    const void* push_constants,
    uint32_t push_constants_size
) {
    if (buffers == nullptr) {
        throw std::runtime_error("VkKernelRuntime: buffers is null");
    }
    if (buffer_count != kernel.storage_buffer_count) {
        throw std::runtime_error("VkKernelRuntime: buffer_count mismatch");
    }
    if (push_constants_size != kernel.push_constant_size) {
        throw std::runtime_error("VkKernelRuntime: push_constants_size mismatch");
    }
    if (push_constants_size > 0 && push_constants == nullptr) {
        throw std::runtime_error("VkKernelRuntime: push_constants is null");
    }
}

VkDeviceSize descriptor_range_or_whole(VkDeviceSize range) {
    return range == 0 ? VK_WHOLE_SIZE : range;
}

VkDescriptorBufferInfo normalize_descriptor_range(VkDescriptorBufferInfo info) {
    info.range = descriptor_range_or_whole(info.range);
    return info;
}

void select_host_barrier_target(
    const VkDescriptorBufferInfo* buffers,
    uint32_t buffer_count,
    int32_t host_read_buffer_index,
    VkBuffer* out_buffer,
    VkDeviceSize* out_offset,
    VkDeviceSize* out_size
) {
    if (host_read_buffer_index < 0) {
        *out_buffer = VK_NULL_HANDLE;
        *out_offset = 0;
        *out_size = 0;
        return;
    }

    const uint32_t index = static_cast<uint32_t>(host_read_buffer_index);
    if (index >= buffer_count) {
        throw std::runtime_error("VkKernelRuntime: host_read_buffer_index out of range");
    }

    *out_buffer = buffers[index].buffer;
    *out_offset = buffers[index].offset;
    *out_size = descriptor_range_or_whole(buffers[index].range);
}

struct NormalizedBuffers {
    std::array<VkDescriptorBufferInfo, kStackBufferCapacity> stack = {};
    std::vector<VkDescriptorBufferInfo> heap;
    VkDescriptorBufferInfo* data = nullptr;
};

NormalizedBuffers normalize_buffers(const VkDescriptorBufferInfo* buffers, uint32_t buffer_count) {
    NormalizedBuffers out;
    if (buffer_count <= kStackBufferCapacity) {
        out.data = out.stack.data();
    } else {
        out.heap.resize(buffer_count);
        out.data = out.heap.data();
    }

    for (uint32_t i = 0; i < buffer_count; ++i) {
        out.data[i] = normalize_descriptor_range(buffers[i]);
    }
    return out;
}

void emit_kernel_trace(
    const VkContext& ctx,
    VkQueryPool query_pool,
    const VkDispatchTraceInfo& trace_info,
    uint32_t query_index,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z,
    int64_t submit_end_us,
    int64_t trace_begin_us,
    int64_t trace_end_us
) {
    const float timestamp_period_ns = ctx.timestamp_period_ns();
    const uint32_t valid_bits = ctx.timestamp_valid_bits();
    if (query_pool == VK_NULL_HANDLE || timestamp_period_ns <= 0.0f || valid_bits == 0) {
        return;
    }

    uint64_t timestamps[2] = {};
    VkResult result = vkGetQueryPoolResults(
        ctx.device(),
        query_pool,
        query_index,
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    );
    if (result == VK_NOT_READY) {
        return;
    }
    if (result != VK_SUCCESS) {
        vk_check(result, "vkGetQueryPoolResults");
    }

    const uint64_t start_ticks = timestamps[0];
    const uint64_t end_ticks = timestamps[1];

    uint64_t delta_ticks = 0;
    if (valid_bits >= 64) {
        delta_ticks = end_ticks - start_ticks;
    } else {
        const uint64_t mask = (uint64_t{1} << valid_bits) - 1;
        delta_ticks = (end_ticks - start_ticks) & mask;
    }

    const double kernel_dur_us_d =
        (static_cast<double>(delta_ticks) * static_cast<double>(timestamp_period_ns)) / 1000.0;
    const int64_t kernel_dur_us = std::max<int64_t>(0, static_cast<int64_t>(kernel_dur_us_d));

    int64_t kernel_start_us = submit_end_us;
    bool calibrated = false;

    if (trace_info.has_calibrated_timestamps && trace_info.calibrated_trace_base_us != 0) {
        uint64_t dticks = 0;
        if (valid_bits >= 64) {
            dticks = start_ticks - trace_info.calibrated_device_ticks;
        } else {
            const uint64_t mask = (uint64_t{1} << valid_bits) - 1;
            dticks = (start_ticks - trace_info.calibrated_device_ticks) & mask;
        }

        const double delta_us =
            (static_cast<double>(dticks) * static_cast<double>(timestamp_period_ns)) / 1000.0;
        kernel_start_us = trace_info.calibrated_trace_base_us + static_cast<int64_t>(delta_us);
        calibrated = true;
    }

    if (kernel_start_us < trace_begin_us || kernel_start_us > trace_end_us) {
        kernel_start_us = submit_end_us;
        calibrated = false;
    }

    const int64_t queue_delay_us = std::max<int64_t>(0, kernel_start_us - submit_end_us);
    ::mruntime::trace_complete_at(
        "vk.kernel",
        "vulkan.gpu",
        kernel_start_us,
        kernel_dur_us,
        {
            ::mruntime::trace_arg("calibrated", calibrated ? 1 : 0),
            ::mruntime::trace_arg("max_dev_ns", static_cast<int64_t>(trace_info.calibrated_max_dev_ns)),
            ::mruntime::trace_arg("queue_delay_us", queue_delay_us),
            ::mruntime::trace_arg("gx", static_cast<int64_t>(group_count_x)),
            ::mruntime::trace_arg("gy", static_cast<int64_t>(group_count_y)),
            ::mruntime::trace_arg("gz", static_cast<int64_t>(group_count_z)),
        }
    );
}

}  // namespace

VkKernelRuntime VkKernelRuntime::Create(const VkContext& context) {
    if (context.device() == VK_NULL_HANDLE) {
        throw std::runtime_error("VkKernelRuntime::Create: context has null device");
    }

    VkKernelRuntime runtime;
    runtime.context_ = &context;
    return runtime;
}

VkKernel VkKernelRuntime::get_or_create_kernel(const KernelCreateInfo& info) {
    if (context_ == nullptr) {
        throw std::runtime_error("VkKernelRuntime::get_or_create_kernel: runtime not initialized");
    }
    if (info.spirv == nullptr || info.spirv_size == 0) {
        throw std::runtime_error("VkKernelRuntime::get_or_create_kernel: empty SPIR-V");
    }
    if (info.storage_buffer_count == 0) {
        throw std::runtime_error("VkKernelRuntime::get_or_create_kernel: storage_buffer_count must be > 0");
    }

    PipelineCacheKey key;
    key.spirv.assign(info.spirv, info.spirv + info.spirv_size);
    key.storage_buffer_count = info.storage_buffer_count;
    key.push_constant_size = info.push_constant_size;

    auto it = pipeline_cache_.find(key);
    if (it == pipeline_cache_.end()) {
        ComputePipelineCreateInfo create_info;
        create_info.spirv = key.spirv.data();
        create_info.spirv_size = key.spirv.size();
        create_info.storage_buffer_count = key.storage_buffer_count;
        create_info.push_constant_size = key.push_constant_size;
        create_info.pipeline_cache = context_->pipeline_cache();

        auto pipeline = std::make_unique<VkComputePipeline>(VkComputePipeline::Create(context_->device(), create_info));
        it = pipeline_cache_.emplace(std::move(key), std::move(pipeline)).first;
    }

    VkKernel kernel;
    kernel.pipeline = it->second.get();
    kernel.storage_buffer_count = info.storage_buffer_count;
    kernel.push_constant_size = info.push_constant_size;
    return kernel;
}

void VkKernelRuntime::maybe_refresh_calibration() const {
    if (context_ == nullptr) {
        return;
    }
    if (!context_->supports_calibrated_timestamps()) {
        calibration_cache_.valid = false;
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (calibration_cache_.valid) {
        const auto age = now - calibration_cache_.sampled_at;
        if (age <= std::chrono::milliseconds(100)) {
            return;
        }
    }

    CalibrationCache best;
    uint64_t best_max_dev_ns = std::numeric_limits<uint64_t>::max();

    for (int attempt = 0; attempt < 3; ++attempt) {
        const int64_t t0 = ::mruntime::TraceCollector::instance().now_us();
        uint64_t device_ticks = 0;
        uint64_t host_ns = 0;
        uint64_t max_dev_ns = 0;

        bool ok = false;
        try {
            ok = context_->calibrated_timestamps_sample(&device_ticks, &host_ns, &max_dev_ns);
        } catch (const VulkanError& e) {
            if (e.result() == VK_ERROR_EXTENSION_NOT_PRESENT) {
                calibration_cache_.valid = false;
                return;
            }
            throw;
        }
        const int64_t t1 = ::mruntime::TraceCollector::instance().now_us();

        if (!ok) {
            continue;
        }

        if (max_dev_ns < best_max_dev_ns) {
            best.valid = true;
            best.device_ticks = device_ticks;
            best.host_ns = host_ns;
            best.max_dev_ns = max_dev_ns;
            best.trace_base_us = (t0 + t1) / 2;
            best.sampled_at = now;
            best_max_dev_ns = max_dev_ns;
        }

        if (max_dev_ns <= 50'000) {
            break;
        }
    }

    if (best.valid) {
        calibration_cache_ = best;
    } else {
        calibration_cache_.valid = false;
    }
}

VkDispatchBatch VkKernelRuntime::begin_batch() const {
    if (context_ == nullptr) {
        throw std::runtime_error("VkKernelRuntime::begin_batch: runtime not initialized");
    }

    VkDispatchBatch batch;
    batch.runtime_ = this;
    batch.recording_ = true;
    batch.enable_trace_ = timing_enabled_ && ::mruntime::TraceCollector::instance().is_enabled();
    batch.record_begin_us_ = batch.enable_trace_ ? ::mruntime::TraceCollector::instance().now_us() : 0;

    if (batch.enable_trace_) {
        maybe_refresh_calibration();
        batch.trace_info_.enable_timing_trace = true;
        if (calibration_cache_.valid) {
            batch.trace_info_.calibrated_device_ticks = calibration_cache_.device_ticks;
            batch.trace_info_.calibrated_trace_base_us = calibration_cache_.trace_base_us;
            batch.trace_info_.calibrated_max_dev_ns = calibration_cache_.max_dev_ns;
            batch.trace_info_.has_calibrated_timestamps = (calibration_cache_.max_dev_ns <= 1'000'000);
        }
        batch.query_pool_ = context_->timestamp_query_pool();
    }

    VkCommandBuffer cb = context_->command_buffer();
    vk_check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
    begin_command_buffer(cb);
    if (batch.query_pool_ != VK_NULL_HANDLE && context_->timestamp_query_count() > 0) {
        vkCmdResetQueryPool(cb, batch.query_pool_, 0, context_->timestamp_query_count());
    }

    return batch;
}

void VkKernelRuntime::finish_batch(
    VkDispatchBatch* batch,
    const VkDispatchBatchHostBarrier* host_barriers,
    uint32_t host_barrier_count
) const {
    if (batch == nullptr || !batch->recording_) {
        return;
    }
    if (batch->runtime_ != this || context_ == nullptr) {
        throw std::runtime_error("VkKernelRuntime::finish_batch: invalid batch/runtime");
    }

    VkCommandBuffer cb = context_->command_buffer();
    for (uint32_t i = 0; i < host_barrier_count; ++i) {
        const VkDispatchBatchHostBarrier& barrier = host_barriers[i];
        if (barrier.buffer != VK_NULL_HANDLE && barrier.size > 0) {
            cmd_buffer_barrier_to_host_read(cb, barrier.buffer, barrier.offset, barrier.size);
        }
    }

    const int64_t record_end_us = batch->enable_trace_ ? ::mruntime::TraceCollector::instance().now_us() : 0;
    end_command_buffer(cb);

    const VkFence fence = context_->fence();
    vk_check(vkResetFences(context_->device(), 1, &fence), "vkResetFences");

    VkCommandBufferSubmitInfo cmd_info = {};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = cb;

    VkSubmitInfo2 submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;

    vk_check(vkQueueSubmit2(context_->queue(), 1, &submit, fence), "vkQueueSubmit2");
    const int64_t submit_end_us = batch->enable_trace_ ? ::mruntime::TraceCollector::instance().now_us() : 0;
    vk_check(vkWaitForFences(context_->device(), 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    const int64_t wait_end_us = batch->enable_trace_ ? ::mruntime::TraceCollector::instance().now_us() : 0;

    if (batch->enable_trace_) {
        ::mruntime::trace_complete_at(
            "vk.batch_record",
            "vulkan.cpu",
            batch->record_begin_us_,
            std::max<int64_t>(0, record_end_us - batch->record_begin_us_),
            {}
        );
        ::mruntime::trace_complete_at(
            "vk.batch_submit",
            "vulkan.cpu",
            record_end_us,
            std::max<int64_t>(0, submit_end_us - record_end_us),
            {}
        );
        ::mruntime::trace_complete_at(
            "vk.batch_wait",
            "vulkan.cpu",
            submit_end_us,
            std::max<int64_t>(0, wait_end_us - submit_end_us),
            {}
        );

        if (batch->query_pool_ != VK_NULL_HANDLE) {
            for (const VkDispatchBatch::TraceRecord& record : batch->trace_records_) {
                emit_kernel_trace(
                    *context_,
                    batch->query_pool_,
                    batch->trace_info_,
                    record.query_index,
                    record.group_count_x,
                    record.group_count_y,
                    record.group_count_z,
                    submit_end_us,
                    batch->record_begin_us_,
                    wait_end_us
                );
            }
        }
    }

    batch->recording_ = false;
    batch->trace_records_.clear();
    batch->descriptor_set_use_counts_.clear();
    batch->next_query_index_ = 0;
}

void VkKernelRuntime::dispatch_1d(
    VkKernel kernel,
    const VkDescriptorBufferInfo* buffers,
    uint32_t buffer_count,
    uint32_t element_count,
    uint32_t local_size_x,
    const void* push_constants,
    uint32_t push_constants_size,
    int32_t host_read_buffer_index,
    VkQueryPool query_pool,
    VkDispatchBatch* batch
) const {
    if (context_ == nullptr) {
        throw std::runtime_error("VkKernelRuntime::dispatch_1d: runtime not initialized");
    }
    validate_kernel(kernel);
    validate_dispatch_args(kernel, buffers, buffer_count, push_constants, push_constants_size);

    if (local_size_x == 0) {
        throw std::runtime_error("VkKernelRuntime::dispatch_1d: local_size_x must be > 0");
    }
    if (element_count == 0) {
        return;
    }

    NormalizedBuffers normalized = normalize_buffers(buffers, buffer_count);
    const uint32_t group_count_x = (element_count + local_size_x - 1) / local_size_x;

    if (batch != nullptr && batch->recording_) {
        if (batch->runtime_ != this) {
            throw std::runtime_error("VkKernelRuntime::dispatch_1d: batch belongs to another runtime");
        }
        uint32_t descriptor_set_index = batch->descriptor_set_use_counts_[kernel.pipeline]++;
        if (descriptor_set_index >= kernel.pipeline->batch_descriptor_set_capacity()) {
            throw std::runtime_error("VkKernelRuntime::dispatch_1d: batch descriptor set capacity exceeded");
        }
        if (batch->query_pool_ != VK_NULL_HANDLE && batch->next_query_index_ + 1 >= context_->timestamp_query_count()) {
            throw std::runtime_error("VkKernelRuntime::dispatch_1d: timestamp query capacity exceeded");
        }

        const uint32_t query_index = (batch->query_pool_ != VK_NULL_HANDLE) ? batch->next_query_index_ : UINT32_MAX;
        kernel.pipeline->record_dispatch(
            *context_,
            context_->command_buffer(),
            descriptor_set_index,
            normalized.data,
            buffer_count,
            group_count_x,
            1,
            1,
            push_constants,
            push_constants_size,
            batch->query_pool_,
            query_index
        );
        if (query_index != UINT32_MAX) {
            batch->trace_records_.push_back({query_index, group_count_x, 1, 1});
            batch->next_query_index_ += 2;
        }
        cmd_buffer_barrier_compute_to_compute(context_->command_buffer());
        return;
    }

    VkBuffer host_read_buffer = VK_NULL_HANDLE;
    VkDeviceSize host_read_offset = 0;
    VkDeviceSize host_read_size = 0;
    select_host_barrier_target(
        normalized.data,
        buffer_count,
        host_read_buffer_index,
        &host_read_buffer,
        &host_read_offset,
        &host_read_size
    );

    const bool enable_trace = timing_enabled_ && ::mruntime::TraceCollector::instance().is_enabled();

    VkQueryPool effective_query_pool = query_pool;
    if (effective_query_pool == VK_NULL_HANDLE && enable_trace) {
        effective_query_pool = context_->timestamp_query_pool();
    }

    VkDispatchTraceInfo trace_info;
    const VkDispatchTraceInfo* trace_ptr = nullptr;
    if (enable_trace) {
        maybe_refresh_calibration();

        trace_info.enable_timing_trace = true;
        if (calibration_cache_.valid) {
            trace_info.calibrated_device_ticks = calibration_cache_.device_ticks;
            trace_info.calibrated_trace_base_us = calibration_cache_.trace_base_us;
            trace_info.calibrated_max_dev_ns = calibration_cache_.max_dev_ns;
            trace_info.has_calibrated_timestamps = (calibration_cache_.max_dev_ns <= 1'000'000);
        }

        trace_ptr = &trace_info;
    }

    kernel.pipeline->dispatch_and_wait(
        *context_,
        normalized.data,
        buffer_count,
        group_count_x,
        1,
        1,
        push_constants,
        push_constants_size,
        host_read_buffer,
        host_read_offset,
        host_read_size,
        effective_query_pool,
        trace_ptr
    );
}

void VkKernelRuntime::dispatch_2d(
    VkKernel kernel,
    const VkDescriptorBufferInfo* buffers,
    uint32_t buffer_count,
    uint32_t width,
    uint32_t height,
    uint32_t local_size_x,
    uint32_t local_size_y,
    const void* push_constants,
    uint32_t push_constants_size,
    int32_t host_read_buffer_index,
    VkQueryPool query_pool,
    VkDispatchBatch* batch
) const {
    if (context_ == nullptr) {
        throw std::runtime_error("VkKernelRuntime::dispatch_2d: runtime not initialized");
    }
    validate_kernel(kernel);
    validate_dispatch_args(kernel, buffers, buffer_count, push_constants, push_constants_size);

    if (local_size_x == 0 || local_size_y == 0) {
        throw std::runtime_error("VkKernelRuntime::dispatch_2d: local sizes must be > 0");
    }
    if (width == 0 || height == 0) {
        return;
    }

    NormalizedBuffers normalized = normalize_buffers(buffers, buffer_count);
    const uint32_t group_count_x = (width + local_size_x - 1) / local_size_x;
    const uint32_t group_count_y = (height + local_size_y - 1) / local_size_y;

    if (batch != nullptr && batch->recording_) {
        if (batch->runtime_ != this) {
            throw std::runtime_error("VkKernelRuntime::dispatch_2d: batch belongs to another runtime");
        }
        uint32_t descriptor_set_index = batch->descriptor_set_use_counts_[kernel.pipeline]++;
        if (descriptor_set_index >= kernel.pipeline->batch_descriptor_set_capacity()) {
            throw std::runtime_error("VkKernelRuntime::dispatch_2d: batch descriptor set capacity exceeded");
        }
        if (batch->query_pool_ != VK_NULL_HANDLE && batch->next_query_index_ + 1 >= context_->timestamp_query_count()) {
            throw std::runtime_error("VkKernelRuntime::dispatch_2d: timestamp query capacity exceeded");
        }

        const uint32_t query_index = (batch->query_pool_ != VK_NULL_HANDLE) ? batch->next_query_index_ : UINT32_MAX;
        kernel.pipeline->record_dispatch(
            *context_,
            context_->command_buffer(),
            descriptor_set_index,
            normalized.data,
            buffer_count,
            group_count_x,
            group_count_y,
            1,
            push_constants,
            push_constants_size,
            batch->query_pool_,
            query_index
        );
        if (query_index != UINT32_MAX) {
            batch->trace_records_.push_back({query_index, group_count_x, group_count_y, 1});
            batch->next_query_index_ += 2;
        }
        cmd_buffer_barrier_compute_to_compute(context_->command_buffer());
        return;
    }

    VkBuffer host_read_buffer = VK_NULL_HANDLE;
    VkDeviceSize host_read_offset = 0;
    VkDeviceSize host_read_size = 0;
    select_host_barrier_target(
        normalized.data,
        buffer_count,
        host_read_buffer_index,
        &host_read_buffer,
        &host_read_offset,
        &host_read_size
    );

    const bool enable_trace = timing_enabled_ && ::mruntime::TraceCollector::instance().is_enabled();

    VkQueryPool effective_query_pool = query_pool;
    if (effective_query_pool == VK_NULL_HANDLE && enable_trace) {
        effective_query_pool = context_->timestamp_query_pool();
    }

    VkDispatchTraceInfo trace_info;
    const VkDispatchTraceInfo* trace_ptr = nullptr;
    if (enable_trace) {
        maybe_refresh_calibration();

        trace_info.enable_timing_trace = true;
        if (calibration_cache_.valid) {
            trace_info.calibrated_device_ticks = calibration_cache_.device_ticks;
            trace_info.calibrated_trace_base_us = calibration_cache_.trace_base_us;
            trace_info.calibrated_max_dev_ns = calibration_cache_.max_dev_ns;
            trace_info.has_calibrated_timestamps = (calibration_cache_.max_dev_ns <= 1'000'000);
        }

        trace_ptr = &trace_info;
    }

    kernel.pipeline->dispatch_and_wait(
        *context_,
        normalized.data,
        buffer_count,
        group_count_x,
        group_count_y,
        1,
        push_constants,
        push_constants_size,
        host_read_buffer,
        host_read_offset,
        host_read_size,
        effective_query_pool,
        trace_ptr
    );
}

bool VkKernelRuntime::PipelineCacheKey::operator==(const PipelineCacheKey& other) const noexcept {
    return storage_buffer_count == other.storage_buffer_count &&
           push_constant_size == other.push_constant_size &&
           spirv == other.spirv;
}

size_t VkKernelRuntime::PipelineCacheKeyHash::operator()(const PipelineCacheKey& key) const noexcept {
    size_t hash = fnv1a_hash_bytes(key.spirv.data(), key.spirv.size());
    hash ^= static_cast<size_t>(key.storage_buffer_count) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    hash ^= static_cast<size_t>(key.push_constant_size) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash;
}

}  // namespace mruntime::vulkan
