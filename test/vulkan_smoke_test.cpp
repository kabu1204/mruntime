#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vk_buffer_arena.h"
#include "vk_context.h"
#include "vk_helpers.h"
#include "vector_add_spv.h"

namespace {

template <typename Handle, void (*DestroyFn)(VkDevice, Handle, const VkAllocationCallbacks*)>
struct UniqueDeviceHandle {
    VkDevice device = VK_NULL_HANDLE;
    Handle handle = VK_NULL_HANDLE;

    UniqueDeviceHandle() = default;
    UniqueDeviceHandle(const UniqueDeviceHandle&) = delete;
    UniqueDeviceHandle& operator=(const UniqueDeviceHandle&) = delete;

    ~UniqueDeviceHandle() {
        if (handle) {
            DestroyFn(device, handle, nullptr);
        }
    }
};

}  // namespace

int main() {
    try {
        mruntime::vulkan::VkContext ctx = mruntime::vulkan::VkContext::Create();

        VkPhysicalDevice physical = ctx.physical_device();
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physical, &props);
        std::cout << "Using Vulkan device: " << props.deviceName << "\n";

        const VkDevice device = ctx.device();
        const VkQueue queue = ctx.queue();

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

        VkDescriptorSetLayoutBinding bindings[3] = {};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dsl_ci = {};
        dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = 3;
        dsl_ci.pBindings = bindings;

        UniqueDeviceHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout> dsl;
        dsl.device = device;
        mruntime::vulkan::vk_check(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &dsl.handle),
            "vkCreateDescriptorSetLayout");

        VkPushConstantRange pc_range = {};
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.offset = 0;
        pc_range.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pl_ci = {};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &dsl.handle;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &pc_range;

        UniqueDeviceHandle<VkPipelineLayout, vkDestroyPipelineLayout> pipeline_layout;
        pipeline_layout.device = device;
        mruntime::vulkan::vk_check(vkCreatePipelineLayout(device, &pl_ci, nullptr, &pipeline_layout.handle),
            "vkCreatePipelineLayout");

        static_assert((mruntime::vulkan::shaders::kVectorAddSpvSize % 4) == 0, "SPIR-V must be word-aligned");
        VkShaderModuleCreateInfo sm_ci = {};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = mruntime::vulkan::shaders::kVectorAddSpvSize;
        sm_ci.pCode = reinterpret_cast<const uint32_t*>(mruntime::vulkan::shaders::kVectorAddSpv);

        UniqueDeviceHandle<VkShaderModule, vkDestroyShaderModule> shader;
        shader.device = device;
        mruntime::vulkan::vk_check(vkCreateShaderModule(device, &sm_ci, nullptr, &shader.handle),
            "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stage = {};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader.handle;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci = {};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = pipeline_layout.handle;

        UniqueDeviceHandle<VkPipeline, vkDestroyPipeline> pipeline;
        pipeline.device = device;
        mruntime::vulkan::vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipeline.handle),
            "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_sizes[1] = {};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[0].descriptorCount = 3;

        VkDescriptorPoolCreateInfo dp_ci = {};
        dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp_ci.maxSets = 1;
        dp_ci.poolSizeCount = 1;
        dp_ci.pPoolSizes = pool_sizes;

        UniqueDeviceHandle<VkDescriptorPool, vkDestroyDescriptorPool> descriptor_pool;
        descriptor_pool.device = device;
        mruntime::vulkan::vk_check(vkCreateDescriptorPool(device, &dp_ci, nullptr, &descriptor_pool.handle),
            "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo ds_ai = {};
        ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_ai.descriptorPool = descriptor_pool.handle;
        ds_ai.descriptorSetCount = 1;
        ds_ai.pSetLayouts = &dsl.handle;

        VkDescriptorSet ds = VK_NULL_HANDLE;
        mruntime::vulkan::vk_check(vkAllocateDescriptorSets(device, &ds_ai, &ds), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo a_info = arena.descriptor(a_off, bytes);
        VkDescriptorBufferInfo b_info = arena.descriptor(b_off, bytes);
        VkDescriptorBufferInfo c_info = arena.descriptor(c_off, bytes);

        VkWriteDescriptorSet writes[3] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = ds;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &a_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = ds;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &b_info;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = ds;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &c_info;

        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cb_ai = {};
        cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_ai.commandPool = ctx.command_pool();
        cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;

        VkCommandBuffer cb = VK_NULL_HANDLE;
        mruntime::vulkan::vk_check(vkAllocateCommandBuffers(device, &cb_ai, &cb), "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        mruntime::vulkan::vk_check(vkBeginCommandBuffer(cb, &begin), "vkBeginCommandBuffer");

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        vkCmdBindDescriptorSets(
            cb,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout.handle,
            0,
            1,
            &ds,
            0,
            nullptr
        );
        vkCmdPushConstants(
            cb,
            pipeline_layout.handle,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(uint32_t),
            &n
        );

        constexpr uint32_t local_size_x = 256;
        const uint32_t groups = (n + local_size_x - 1) / local_size_x;
        vkCmdDispatch(cb, groups, 1, 1);

        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = arena.buffer();
        barrier.offset = c_off;
        barrier.size = bytes;

        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr
        );

        mruntime::vulkan::vk_check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

        VkFenceCreateInfo fence_ci = {};
        fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        UniqueDeviceHandle<VkFence, vkDestroyFence> fence;
        fence.device = device;
        mruntime::vulkan::vk_check(vkCreateFence(device, &fence_ci, nullptr, &fence.handle), "vkCreateFence");

        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cb;
        mruntime::vulkan::vk_check(vkQueueSubmit(queue, 1, &submit, fence.handle), "vkQueueSubmit");
        mruntime::vulkan::vk_check(vkWaitForFences(device, 1, &fence.handle, VK_TRUE, UINT64_MAX), "vkWaitForFences");

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
