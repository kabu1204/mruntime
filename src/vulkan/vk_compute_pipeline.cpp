#include "vk_compute_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mruntime/trace.h"

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

    vk_check(vkCreateComputePipelines(device, info.pipeline_cache, 1, &vk_ci, nullptr, &out.pipeline_),
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

    std::vector<VkDescriptorUpdateTemplateEntry> update_entries(info.storage_buffer_count);
    for (uint32_t i = 0; i < info.storage_buffer_count; ++i) {
        VkDescriptorUpdateTemplateEntry entry = {};
        entry.dstBinding = i;
        entry.dstArrayElement = 0;
        entry.descriptorCount = 1;
        entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        entry.offset = static_cast<size_t>(i) * sizeof(VkDescriptorBufferInfo);
        entry.stride = sizeof(VkDescriptorBufferInfo);
        update_entries[i] = entry;
    }

    VkDescriptorUpdateTemplateCreateInfo update_template_ci = {};
    update_template_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    update_template_ci.descriptorUpdateEntryCount = static_cast<uint32_t>(update_entries.size());
    update_template_ci.pDescriptorUpdateEntries = update_entries.data();
    update_template_ci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    update_template_ci.descriptorSetLayout = out.descriptor_set_layout_;
    vk_check(vkCreateDescriptorUpdateTemplate(device, &update_template_ci, nullptr, &out.descriptor_update_template_),
        "vkCreateDescriptorUpdateTemplate");

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
    descriptor_update_template_ = std::exchange(other.descriptor_update_template_, VK_NULL_HANDLE);

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
    VkDeviceSize host_read_size,
    VkQueryPool query_pool,
    const VkDispatchTraceInfo* trace
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

    const bool enable_trace =
        trace != nullptr &&
        trace->enable_timing_trace &&
        ::mruntime::TraceCollector::instance().is_enabled();

    const int64_t dispatch_start_us =
        enable_trace ? ::mruntime::TraceCollector::instance().now_us() : 0;

    int64_t submit_end_us = dispatch_start_us;
    int64_t dispatch_end_us = dispatch_start_us;

    if (enable_trace) {
        ::mruntime::ScopedTrace dispatch_scope("vk.dispatch", "vulkan.cpu");
        const VkFence fence = ctx.fence();

        {
            ::mruntime::ScopedTrace launch_scope("vk.launch", "vulkan.cpu");

            vkUpdateDescriptorSetWithTemplate(device_, descriptor_set_, descriptor_update_template_, buffers);

            VkCommandBuffer cb = ctx.command_buffer();
            vk_check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");

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
                vkCmdPushConstants(
                    cb,
                    pipeline_layout_,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    push_constant_size_,
                    push_constants
                );
            }

            if (query_pool != VK_NULL_HANDLE) {
                vkCmdResetQueryPool(cb, query_pool, 0, 2);
                vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query_pool, 0);
            }
            vkCmdDispatch(cb, group_count_x, group_count_y, group_count_z);
            if (query_pool != VK_NULL_HANDLE) {
                vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, query_pool, 1);
            }

            if (host_read_buffer != VK_NULL_HANDLE && host_read_size > 0) {
                cmd_buffer_barrier_to_host_read(cb, host_read_buffer, host_read_offset, host_read_size);
            }

            end_command_buffer(cb);

            vk_check(vkResetFences(device_, 1, &fence), "vkResetFences");

            VkCommandBufferSubmitInfo cmd_info = {};
            cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            cmd_info.commandBuffer = cb;

            VkSubmitInfo2 submit = {};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submit.commandBufferInfoCount = 1;
            submit.pCommandBufferInfos = &cmd_info;

            vk_check(vkQueueSubmit2(ctx.queue(), 1, &submit, fence), "vkQueueSubmit2");
            submit_end_us = ::mruntime::TraceCollector::instance().now_us();
        }

        {
            ::mruntime::ScopedTrace wait_scope("vk.wait", "vulkan.cpu");
            vk_check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        }

        dispatch_end_us = ::mruntime::TraceCollector::instance().now_us();

        const float timestamp_period_ns = ctx.timestamp_period_ns();
        const uint32_t valid_bits = ctx.timestamp_valid_bits();
        const bool has_gpu_timestamps =
            query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f && valid_bits > 0;

        if (has_gpu_timestamps) {
            uint64_t timestamps[2] = {};
            VkResult result = vkGetQueryPoolResults(
                device_,
                query_pool,
                0,
                2,
                sizeof(timestamps),
                timestamps,
                sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT
            );
            if (result == VK_SUCCESS) {
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

                // Without calibrated timestamps, place the kernel at submit time (it cannot start earlier).
                int64_t kernel_start_us = submit_end_us;
                bool calibrated = false;

                if (trace->has_calibrated_timestamps && trace->calibrated_trace_base_us != 0) {
                    uint64_t dticks = 0;
                    if (valid_bits >= 64) {
                        dticks = start_ticks - trace->calibrated_device_ticks;
                    } else {
                        const uint64_t mask = (uint64_t{1} << valid_bits) - 1;
                        dticks = (start_ticks - trace->calibrated_device_ticks) & mask;
                    }

                    const double delta_us =
                        (static_cast<double>(dticks) * static_cast<double>(timestamp_period_ns)) / 1000.0;
                    kernel_start_us = trace->calibrated_trace_base_us + static_cast<int64_t>(delta_us);
                    calibrated = true;
                }

                // Sanity-check placement; fall back to dispatch start if calibration is clearly off.
                if (kernel_start_us < dispatch_start_us || kernel_start_us > dispatch_end_us) {
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
                        ::mruntime::trace_arg("max_dev_ns", static_cast<int64_t>(trace->calibrated_max_dev_ns)),
                        ::mruntime::trace_arg("queue_delay_us", queue_delay_us),
                        ::mruntime::trace_arg("gx", static_cast<int64_t>(group_count_x)),
                        ::mruntime::trace_arg("gy", static_cast<int64_t>(group_count_y)),
                        ::mruntime::trace_arg("gz", static_cast<int64_t>(group_count_z)),
                    }
                );
            } else if (result != VK_NOT_READY) {
                vk_check(result, "vkGetQueryPoolResults");
            }
        }
    } else {
        vkUpdateDescriptorSetWithTemplate(device_, descriptor_set_, descriptor_update_template_, buffers);

        VkCommandBuffer cb = ctx.command_buffer();
        vk_check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");

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

        if (query_pool != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(cb, query_pool, 0, 2);
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query_pool, 0);
        }
        vkCmdDispatch(cb, group_count_x, group_count_y, group_count_z);
        if (query_pool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        }

        if (host_read_buffer != VK_NULL_HANDLE && host_read_size > 0) {
            cmd_buffer_barrier_to_host_read(cb, host_read_buffer, host_read_offset, host_read_size);
        }

        end_command_buffer(cb);
        submit_and_wait_with_fence(device_, ctx.queue(), cb, ctx.fence());
    }
}

void VkComputePipeline::destroy() noexcept {
    if (device_ == VK_NULL_HANDLE) {
        return;
    }

    if (descriptor_update_template_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorUpdateTemplate(device_, descriptor_update_template_, nullptr);
        descriptor_update_template_ = VK_NULL_HANDLE;
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
