#pragma once

#include <vulkan/vulkan.h>

namespace mruntime::vulkan {

VkCommandBuffer allocate_command_buffer(VkDevice device, VkCommandPool pool);

void free_command_buffer(VkDevice device, VkCommandPool pool, VkCommandBuffer command_buffer);

void begin_command_buffer(VkCommandBuffer command_buffer);

void end_command_buffer(VkCommandBuffer command_buffer);

void submit_and_wait(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer);

void submit_and_wait_with_fence(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer, VkFence fence);

void cmd_buffer_barrier_to_host_read(
    VkCommandBuffer command_buffer,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
);

}  // namespace mruntime::vulkan

