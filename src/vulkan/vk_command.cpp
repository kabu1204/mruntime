#include "vk_command.h"

#include <cstdint>

#include "vk_helpers.h"

namespace mruntime::vulkan {

VkCommandBuffer allocate_command_buffer(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(device, &alloc, &command_buffer), "vkAllocateCommandBuffers");
    return command_buffer;
}

void free_command_buffer(VkDevice device, VkCommandPool pool, VkCommandBuffer command_buffer) {
    if (command_buffer == VK_NULL_HANDLE) {
        return;
    }
    vkFreeCommandBuffers(device, pool, 1, &command_buffer);
}

void begin_command_buffer(VkCommandBuffer command_buffer) {
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vk_check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");
}

void end_command_buffer(VkCommandBuffer command_buffer) {
    vk_check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
}

void submit_and_wait(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer) {
    VkFenceCreateInfo fence_ci = {};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    vk_check(vkCreateFence(device, &fence_ci, nullptr, &fence), "vkCreateFence");
    struct FenceGuard {
        VkDevice device = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        ~FenceGuard() {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
        }
    } fence_guard = {device, fence};

    VkCommandBufferSubmitInfo cmd_info = {};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = command_buffer;

    VkSubmitInfo2 submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;

    vk_check(vkQueueSubmit2(queue, 1, &submit, fence), "vkQueueSubmit2");
    vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

void submit_and_wait_with_fence(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer, VkFence fence) {
    vk_check(vkResetFences(device, 1, &fence), "vkResetFences");

    VkCommandBufferSubmitInfo cmd_info = {};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_info.commandBuffer = command_buffer;

    VkSubmitInfo2 submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;

    vk_check(vkQueueSubmit2(queue, 1, &submit, fence), "vkQueueSubmit2");
    vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

void cmd_buffer_barrier_to_host_read(
    VkCommandBuffer command_buffer,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
) {
    VkBufferMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;

    VkDependencyInfo dep = {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(command_buffer, &dep);
}

}  // namespace mruntime::vulkan
