#include "vk_fp16_ops.h"

#include <stdexcept>

#include "fp16_add_spv.h"
#include "fp16_mul_spv.h"

namespace mruntime::vulkan {

namespace {

KernelCreateInfo make_kernel_create_info(const uint8_t* spirv, size_t spirv_size) {
    KernelCreateInfo info;
    info.spirv = spirv;
    info.spirv_size = spirv_size;
    info.storage_buffer_count = 3;
    info.push_constant_size = sizeof(uint32_t);
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
    ops.add_kernel_ = runtime->get_or_create_kernel(make_kernel_create_info(shaders::kFp16AddSpv, shaders::kFp16AddSpvSize));
    ops.mul_kernel_ = runtime->get_or_create_kernel(make_kernel_create_info(shaders::kFp16MulSpv, shaders::kFp16MulSpvSize));
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

}  // namespace mruntime::vulkan
