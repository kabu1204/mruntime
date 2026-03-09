#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vk_buffer_arena.h"
#include "vk_compute_pipeline.h"
#include "vk_context.h"
#include "vk_helpers.h"
#include "vector_add_spv.h"

int main() {
    try {
        mruntime::vulkan::VkContext ctx = mruntime::vulkan::VkContext::Create();

        VkPhysicalDevice physical = ctx.physical_device();
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physical, &props);
        std::cout << "Using Vulkan device: " << props.deviceName << "\n";

        const VkDevice device = ctx.device();

        constexpr uint32_t n = 1024;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(float);

        const VkDeviceSize alignment = std::max<VkDeviceSize>(64, ctx.min_storage_buffer_offset_alignment());
        mruntime::vulkan::VkBufferArenaCreateInfo arena_ci;
        arena_ci.capacity_bytes = 3 * bytes + 3 * alignment;
        arena_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        arena_ci.memory_properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        arena_ci.default_alignment = alignment;
        mruntime::vulkan::VkBufferArena arena = mruntime::vulkan::VkBufferArena::Create(physical, device, arena_ci);

        const VkDeviceSize a_off = arena.alloc(bytes);
        const VkDeviceSize b_off = arena.alloc(bytes);
        const VkDeviceSize c_off = arena.alloc(bytes);

        float* a_f = arena.host_ptr<float>(a_off);
        float* b_f = arena.host_ptr<float>(b_off);
        float* c_f = arena.host_ptr<float>(c_off);
        if (a_f == nullptr || b_f == nullptr || c_f == nullptr) {
            throw std::runtime_error("Failed to map storage buffer arena");
        }
        for (uint32_t i = 0; i < n; ++i) {
            a_f[i] = static_cast<float>(i);
            b_f[i] = static_cast<float>(2 * i);
        }
        std::memset(c_f, 0, static_cast<size_t>(bytes));

        mruntime::vulkan::ComputePipelineCreateInfo pipeline_ci;
        pipeline_ci.spirv = mruntime::vulkan::shaders::kVectorAddSpv;
        pipeline_ci.spirv_size = mruntime::vulkan::shaders::kVectorAddSpvSize;
        pipeline_ci.storage_buffer_count = 3;
        pipeline_ci.push_constant_size = sizeof(uint32_t);
        mruntime::vulkan::VkComputePipeline pipeline = mruntime::vulkan::VkComputePipeline::Create(device, pipeline_ci);

        VkDescriptorBufferInfo buffers[3] = {
            arena.descriptor(a_off, bytes),
            arena.descriptor(b_off, bytes),
            arena.descriptor(c_off, bytes),
        };

        constexpr uint32_t local_size_x = 256;
        const uint32_t groups = (n + local_size_x - 1) / local_size_x;
        pipeline.dispatch_and_wait(
            ctx,
            buffers,
            3,
            groups,
            1,
            1,
            &n,
            sizeof(uint32_t),
            arena.buffer(),
            c_off,
            bytes
        );

        for (uint32_t i = 0; i < n; ++i) {
            const float expected = a_f[i] + b_f[i];
            if (c_f[i] != expected) {
                throw std::runtime_error(
                    "Vector add mismatch at i=" + std::to_string(i) + ": got=" + std::to_string(c_f[i]) +
                    " expected=" + std::to_string(expected)
                );
            }
        }

        std::cout << "vulkan_smoke_test PASSED\n";
        return 0;
    } catch (const mruntime::vulkan::VulkanError& e) {
        if (e.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_smoke_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_smoke_test FAILED: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "vulkan_smoke_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
