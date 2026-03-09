#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mruntime/dtype.h"
#include "vk_buffer_arena.h"
#include "vk_context.h"
#include "vk_fp16_ops.h"
#include "vk_helpers.h"
#include "vk_kernel_runtime.h"

namespace {

void run_fp16_add_mul_test() {
    mruntime::vulkan::VkContext context = mruntime::vulkan::VkContext::Create();

    VkPhysicalDevice physical_device = context.physical_device();
    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    std::cout << "Using Vulkan device: " << properties.deviceName << "\n";

    const VkDevice device = context.device();
    const VkDeviceSize alignment = std::max<VkDeviceSize>(64, context.min_storage_buffer_offset_alignment());

    mruntime::vulkan::VkKernelRuntime runtime = mruntime::vulkan::VkKernelRuntime::Create(context);
    mruntime::vulkan::VkFp16Ops fp16_ops = mruntime::vulkan::VkFp16Ops::Create(&runtime);

    const std::vector<uint32_t> test_sizes = {2048, 2049};
    for (const uint32_t n : test_sizes) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(uint16_t);

        mruntime::vulkan::VkBufferArenaCreateInfo arena_create_info;
        arena_create_info.capacity_bytes = 3 * bytes + 3 * alignment;
        arena_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        arena_create_info.memory_properties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        arena_create_info.default_alignment = alignment;

        mruntime::vulkan::VkBufferArena arena =
            mruntime::vulkan::VkBufferArena::Create(physical_device, device, arena_create_info);

        const VkDeviceSize a_offset = arena.alloc(bytes);
        const VkDeviceSize b_offset = arena.alloc(bytes);
        const VkDeviceSize out_offset = arena.alloc(bytes);

        uint16_t* a_data = arena.host_ptr<uint16_t>(a_offset);
        uint16_t* b_data = arena.host_ptr<uint16_t>(b_offset);
        uint16_t* out_data = arena.host_ptr<uint16_t>(out_offset);
        if (a_data == nullptr || b_data == nullptr || out_data == nullptr) {
            throw std::runtime_error("Failed to map Vulkan FP16 arena buffers");
        }

        for (uint32_t i = 0; i < n; ++i) {
            const float a = -1.5f + 0.00125f * static_cast<float>(i);
            const float b = 0.75f - 0.0005f * static_cast<float>(i);
            a_data[i] = mruntime::float_to_fp16_bits(a);
            b_data[i] = mruntime::float_to_fp16_bits(b);
        }

        std::vector<float> add_expected(n);
        std::vector<float> mul_expected(n);
        for (uint32_t i = 0; i < n; ++i) {
            const float a = mruntime::fp16_bits_to_float(a_data[i]);
            const float b = mruntime::fp16_bits_to_float(b_data[i]);
            add_expected[i] = a + b;
            mul_expected[i] = a * b;
        }

        const VkDescriptorBufferInfo a_desc = arena.descriptor(a_offset, bytes);
        const VkDescriptorBufferInfo b_desc = arena.descriptor(b_offset, bytes);
        const VkDescriptorBufferInfo out_desc = arena.descriptor(out_offset, bytes);

        std::memset(out_data, 0, static_cast<size_t>(bytes));
        fp16_ops.add(a_desc, b_desc, out_desc, n);
        constexpr float kAddTolerance = 6e-3f;
        for (uint32_t i = 0; i < n; ++i) {
            const float got = mruntime::fp16_bits_to_float(out_data[i]);
            if (!std::isfinite(got)) {
                throw std::runtime_error("Non-finite result in Vulkan FP16 add");
            }
            if (std::fabs(got - add_expected[i]) > kAddTolerance) {
                throw std::runtime_error(
                    "FP16 add mismatch at n=" + std::to_string(n) +
                    ", i=" + std::to_string(i) +
                    ": got=" + std::to_string(got) +
                    ", expected=" + std::to_string(add_expected[i])
                );
            }
        }

        std::memset(out_data, 0, static_cast<size_t>(bytes));
        fp16_ops.mul(a_desc, b_desc, out_desc, n);
        constexpr float kMulTolerance = 8e-3f;
        for (uint32_t i = 0; i < n; ++i) {
            const float got = mruntime::fp16_bits_to_float(out_data[i]);
            if (!std::isfinite(got)) {
                throw std::runtime_error("Non-finite result in Vulkan FP16 mul");
            }
            if (std::fabs(got - mul_expected[i]) > kMulTolerance) {
                throw std::runtime_error(
                    "FP16 mul mismatch at n=" + std::to_string(n) +
                    ", i=" + std::to_string(i) +
                    ": got=" + std::to_string(got) +
                    ", expected=" + std::to_string(mul_expected[i])
                );
            }
        }
    }
}

}  // namespace

int main() {
    try {
        run_fp16_add_mul_test();
        std::cout << "vulkan_fp16_ops_test PASSED\n";
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_fp16_ops_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_fp16_ops_test FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_fp16_ops_test FAILED: " << error.what() << "\n";
        return 1;
    }
}
