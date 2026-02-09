#include "vk_kernel_runtime.h"

#include <algorithm>
#include <stdexcept>

namespace mruntime::vulkan {

namespace {

constexpr size_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr size_t kFnvPrime = 1099511628211ull;

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

        auto pipeline = std::make_unique<VkComputePipeline>(VkComputePipeline::Create(context_->device(), create_info));
        it = pipeline_cache_.emplace(std::move(key), std::move(pipeline)).first;
    }

    VkKernel kernel;
    kernel.pipeline = it->second.get();
    kernel.storage_buffer_count = info.storage_buffer_count;
    kernel.push_constant_size = info.push_constant_size;
    return kernel;
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
    VkQueryPool query_pool
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

    std::vector<VkDescriptorBufferInfo> normalized_buffers(buffer_count);
    for (uint32_t i = 0; i < buffer_count; ++i) {
        normalized_buffers[i] = normalize_descriptor_range(buffers[i]);
    }

    const uint32_t group_count_x = (element_count + local_size_x - 1) / local_size_x;

    VkBuffer host_read_buffer = VK_NULL_HANDLE;
    VkDeviceSize host_read_offset = 0;
    VkDeviceSize host_read_size = 0;
    select_host_barrier_target(
        normalized_buffers.data(),
        static_cast<uint32_t>(normalized_buffers.size()),
        host_read_buffer_index,
        &host_read_buffer,
        &host_read_offset,
        &host_read_size
    );

    kernel.pipeline->dispatch_and_wait(
        *context_,
        normalized_buffers.data(),
        static_cast<uint32_t>(normalized_buffers.size()),
        group_count_x,
        1,
        1,
        push_constants,
        push_constants_size,
        host_read_buffer,
        host_read_offset,
        host_read_size,
        query_pool
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
    VkQueryPool query_pool
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

    std::vector<VkDescriptorBufferInfo> normalized_buffers(buffer_count);
    for (uint32_t i = 0; i < buffer_count; ++i) {
        normalized_buffers[i] = normalize_descriptor_range(buffers[i]);
    }

    const uint32_t group_count_x = (width + local_size_x - 1) / local_size_x;
    const uint32_t group_count_y = (height + local_size_y - 1) / local_size_y;

    VkBuffer host_read_buffer = VK_NULL_HANDLE;
    VkDeviceSize host_read_offset = 0;
    VkDeviceSize host_read_size = 0;
    select_host_barrier_target(
        normalized_buffers.data(),
        static_cast<uint32_t>(normalized_buffers.size()),
        host_read_buffer_index,
        &host_read_buffer,
        &host_read_offset,
        &host_read_size
    );

    kernel.pipeline->dispatch_and_wait(
        *context_,
        normalized_buffers.data(),
        static_cast<uint32_t>(normalized_buffers.size()),
        group_count_x,
        group_count_y,
        1,
        push_constants,
        push_constants_size,
        host_read_buffer,
        host_read_offset,
        host_read_size,
        query_pool
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
