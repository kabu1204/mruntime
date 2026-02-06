#include "vk_compute_pipeline.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "vk_command.h"
#include "vk_helpers.h"

namespace mruntime::vulkan {

VkComputePipeline VkComputePipeline::Create(VkDevice device, const ComputePipelineCreateInfo& info) {
    if (device == VK_NULL_HANDLE) {
        throw std::runtime_error("VkComputePipeline::Create: device is null");
    }
    if (info.spirv == nullptr || info.spirv_size == 0) {
        throw std::runtime_error("VkComputePipeline::Create: SPIR-V is empty");
    }
    if ((info.spirv_size % 4) != 0) {
        throw std::runtime_error("VkComputePipeline::Create: SPIR-V size must be multiple of 4");
    }
    if (info.storage_buffer_count == 0) {
        throw std::runtime_error("VkComputePipeline::Create: storage_buffer_count must be > 0");
    }

    VkComputePipeline out;
    out.device_ = device;
    out.storage_buffer_count_ = info.storage_buffer_count;
    out.push_constant_size_ = info.push_constant_size;

    // Descriptor set layout: bindings [0..N) as storage buffers.
    std::vector<VkDescriptorSetLayoutBinding> bindings(info.storage_buffer_count);
    for (uint32_t i = 0; i < info.storage_buffer_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = static_cast<uint32_t>(bindings.size());
    dsl_ci.pBindings = bindings.data();
    vk_check(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &out.descriptor_set_layout_),
        "vkCreateDescriptorSetLayout");

    VkPushConstantRange pc_range = {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = info.push_constant_size;

    VkPipelineLayoutCreateInfo pl_ci = {};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &out.descriptor_set_layout_;
    if (info.push_constant_size > 0) {
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &pc_range;
    }
    vk_check(vkCreatePipelineLayout(device, &pl_ci, nullptr, &out.pipeline_layout_), "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo sm_ci = {};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = info.spirv_size;
    sm_ci.pCode = reinterpret_cast<const uint32_t*>(info.spirv);

    VkShaderModule shader = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(device, &sm_ci, nullptr, &shader), "vkCreateShaderModule");

    struct ShaderModuleGuard {
        VkDevice device = VK_NULL_HANDLE;
        VkShaderModule shader = VK_NULL_HANDLE;
        ~ShaderModuleGuard() {
            if (shader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shader, nullptr);
            }
        }
    } shader_guard = {device, shader};

    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo vk_ci = {};
    vk_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    vk_ci.stage = stage;
    vk_ci.layout = out.pipeline_layout_;

    vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &vk_ci, nullptr, &out.pipeline_),
        "vkCreateComputePipelines");

    // Descriptor pool & one reusable descriptor set.
    VkDescriptorPoolSize pool_sizes[1] = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = info.storage_buffer_count;

    VkDescriptorPoolCreateInfo dp_ci = {};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 1;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = pool_sizes;
    vk_check(vkCreateDescriptorPool(device, &dp_ci, nullptr, &out.descriptor_pool_), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ds_ai = {};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = out.descriptor_pool_;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts = &out.descriptor_set_layout_;
    vk_check(vkAllocateDescriptorSets(device, &ds_ai, &out.descriptor_set_), "vkAllocateDescriptorSets");

    return out;
}

VkComputePipeline::~VkComputePipeline() {
    destroy();
}

VkComputePipeline::VkComputePipeline(VkComputePipeline&& other) noexcept {
    *this = std::move(other);
}

VkComputePipeline& VkComputePipeline::operator=(VkComputePipeline&& other) noexcept {
    if (this == &other) return *this;
    destroy();

    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    storage_buffer_count_ = std::exchange(other.storage_buffer_count_, 0u);
    push_constant_size_ = std::exchange(other.push_constant_size_, 0u);
    descriptor_set_layout_ = std::exchange(other.descriptor_set_layout_, VK_NULL_HANDLE);
    pipeline_layout_ = std::exchange(other.pipeline_layout_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
    descriptor_pool_ = std::exchange(other.descriptor_pool_, VK_NULL_HANDLE);
    descriptor_set_ = std::exchange(other.descriptor_set_, VK_NULL_HANDLE);

    return *this;
}

void VkComputePipeline::dispatch_and_wait(
    const VkContext& ctx,
    const VkDescriptorBufferInfo* buffers,
    uint32_t buffer_count,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z,
    const void* push_constants,
    uint32_t push_constants_size,
    VkBuffer host_read_buffer,
    VkDeviceSize host_read_offset,
    VkDeviceSize host_read_size
) const {
    if (device_ == VK_NULL_HANDLE || ctx.device() != device_) {
        throw std::runtime_error("VkComputePipeline::dispatch_and_wait: invalid device/context");
    }
    if (buffers == nullptr || buffer_count != storage_buffer_count_) {
        throw std::runtime_error("VkComputePipeline::dispatch_and_wait: buffer_count mismatch");
    }
    if (push_constants_size != push_constant_size_) {
        throw std::runtime_error("VkComputePipeline::dispatch_and_wait: push_constants_size mismatch");
    }
    if (push_constant_size_ > 0 && push_constants == nullptr) {
        throw std::runtime_error("VkComputePipeline::dispatch_and_wait: push_constants is null");
    }

    std::vector<VkWriteDescriptorSet> writes(buffer_count);
    for (uint32_t i = 0; i < buffer_count; ++i) {
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptor_set_;
        w.dstBinding = i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &buffers[i];
        writes[i] = w;
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBuffer cb = allocate_command_buffer(device_, ctx.command_pool());
    struct CommandBufferGuard {
        VkDevice device = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        ~CommandBufferGuard() { free_command_buffer(device, pool, command_buffer); }
    } command_buffer_guard = {device_, ctx.command_pool(), cb};

    begin_command_buffer(cb);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(
        cb,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_,
        0,
        1,
        &descriptor_set_,
        0,
        nullptr
    );
    if (push_constant_size_ > 0) {
        vkCmdPushConstants(cb, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size_, push_constants);
    }
    vkCmdDispatch(cb, group_count_x, group_count_y, group_count_z);

    if (host_read_buffer != VK_NULL_HANDLE && host_read_size > 0) {
        cmd_buffer_barrier_to_host_read(cb, host_read_buffer, host_read_offset, host_read_size);
    }

    end_command_buffer(cb);
    submit_and_wait(device_, ctx.queue(), cb);
}

void VkComputePipeline::destroy() noexcept {
    if (device_ == VK_NULL_HANDLE) {
        return;
    }

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }
    descriptor_set_ = VK_NULL_HANDLE;

    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
    storage_buffer_count_ = 0;
    push_constant_size_ = 0;
}

}  // namespace mruntime::vulkan
