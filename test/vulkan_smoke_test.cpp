#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vector_add_spv.h"

namespace {

void vk_check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(what) + " failed with VkResult=" + std::to_string(result));
}

std::vector<VkExtensionProperties> enumerate_instance_extensions() {
    uint32_t count = 0;
    vk_check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
        "vkEnumerateInstanceExtensionProperties(count)");
    std::vector<VkExtensionProperties> exts(count);
    vk_check(vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data()),
        "vkEnumerateInstanceExtensionProperties(data)");
    return exts;
}

std::vector<VkExtensionProperties> enumerate_device_extensions(VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
        "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> exts(count);
    vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, exts.data()),
        "vkEnumerateDeviceExtensionProperties(data)");
    return exts;
}

bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("No Vulkan queue families found");
    }

    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }
    throw std::runtime_error("No Vulkan compute queue family found");
}

uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t memory_type_bits,
    VkMemoryPropertyFlags required
) {
    VkPhysicalDeviceMemoryProperties mem_props = {};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((memory_type_bits & (1u << i)) == 0) {
            continue;
        }
        if ((mem_props.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type found");
}

struct UniqueInstance {
    VkInstance handle = VK_NULL_HANDLE;

    UniqueInstance() = default;
    UniqueInstance(const UniqueInstance&) = delete;
    UniqueInstance& operator=(const UniqueInstance&) = delete;

    ~UniqueInstance() {
        if (handle) {
            vkDestroyInstance(handle, nullptr);
        }
    }
};

struct UniqueDevice {
    VkDevice handle = VK_NULL_HANDLE;

    UniqueDevice() = default;
    UniqueDevice(const UniqueDevice&) = delete;
    UniqueDevice& operator=(const UniqueDevice&) = delete;

    ~UniqueDevice() {
        if (handle) {
            vkDeviceWaitIdle(handle);
            vkDestroyDevice(handle, nullptr);
        }
    }
};

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

struct MappedBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;

    ~MappedBuffer() {
        if (mapped) {
            vkUnmapMemory(device, memory);
        }
        if (buffer) {
            vkDestroyBuffer(device, buffer, nullptr);
        }
        if (memory) {
            vkFreeMemory(device, memory, nullptr);
        }
    }
};

MappedBuffer create_mapped_storage_buffer(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size_bytes
) {
    MappedBuffer out;
    out.device = device;
    out.size = size_bytes;

    VkBufferCreateInfo buf_ci = {};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size_bytes;
    buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &buf_ci, nullptr, &out.buffer), "vkCreateBuffer");

    VkMemoryRequirements req = {};
    vkGetBufferMemoryRequirements(device, out.buffer, &req);

    const uint32_t memory_type = find_memory_type(
        physical_device,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type;
    vk_check(vkAllocateMemory(device, &alloc_info, nullptr, &out.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");

    vk_check(vkMapMemory(device, out.memory, 0, size_bytes, 0, &out.mapped), "vkMapMemory");
    std::memset(out.mapped, 0, static_cast<size_t>(size_bytes));
    return out;
}

void require_fp16_features(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR float16 = {};
    float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;

    VkPhysicalDevice16BitStorageFeatures storage16 = {};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    float16.pNext = &storage16;

    VkPhysicalDeviceScalarBlockLayoutFeaturesEXT scalar_layout = {};
    scalar_layout.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT;
    storage16.pNext = &scalar_layout;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &float16;

    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    if (!float16.shaderFloat16) {
        throw std::runtime_error("Required Vulkan feature missing: shaderFloat16");
    }
    if (!storage16.storageBuffer16BitAccess) {
        throw std::runtime_error("Required Vulkan feature missing: storageBuffer16BitAccess");
    }
    if (!scalar_layout.scalarBlockLayout) {
        throw std::runtime_error("Required Vulkan feature missing: scalarBlockLayout");
    }
}

}  // namespace

int main() {
    try {
        constexpr const char* kKhrPortabilitySubsetExtensionName = "VK_KHR_portability_subset";

        const auto instance_exts = enumerate_instance_extensions();

        std::vector<const char*> enabled_instance_exts;
        VkInstanceCreateFlags instance_flags = 0;

        if (has_extension(instance_exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            enabled_instance_exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }

        VkApplicationInfo app = {};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "mruntime_vulkan_smoke_test";
        app.applicationVersion = 1;
        app.pEngineName = "mruntime";
        app.engineVersion = 1;
        app.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instance_ci = {};
        instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_ci.flags = instance_flags;
        instance_ci.pApplicationInfo = &app;
        instance_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_instance_exts.size());
        instance_ci.ppEnabledExtensionNames =
            enabled_instance_exts.empty() ? nullptr : enabled_instance_exts.data();

        UniqueInstance instance;
        VkResult instance_result = vkCreateInstance(&instance_ci, nullptr, &instance.handle);
        if (instance_result == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_smoke_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        vk_check(instance_result, "vkCreateInstance");

        uint32_t device_count = 0;
        vk_check(vkEnumeratePhysicalDevices(instance.handle, &device_count, nullptr),
            "vkEnumeratePhysicalDevices(count)");
        if (device_count == 0) {
            throw std::runtime_error("No Vulkan physical devices found");
        }
        std::vector<VkPhysicalDevice> devices(device_count);
        vk_check(vkEnumeratePhysicalDevices(instance.handle, &device_count, devices.data()),
            "vkEnumeratePhysicalDevices(data)");

        VkPhysicalDevice physical = devices[0];
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physical, &props);
        std::cout << "Using Vulkan device: " << props.deviceName << "\n";

        const auto dev_exts = enumerate_device_extensions(physical);
        const char* required_exts[] = {
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
        };
        for (const char* ext : required_exts) {
            if (!has_extension(dev_exts, ext)) {
                throw std::runtime_error(std::string("Required Vulkan device extension missing: ") + ext);
            }
        }

        std::vector<const char*> enabled_dev_exts = {
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
        };
        if (has_extension(dev_exts, kKhrPortabilitySubsetExtensionName)) {
            enabled_dev_exts.push_back(kKhrPortabilitySubsetExtensionName);
        }

        require_fp16_features(physical);

        const uint32_t queue_family_index = find_compute_queue_family(physical);
        const float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo q_ci = {};
        q_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q_ci.queueFamilyIndex = queue_family_index;
        q_ci.queueCount = 1;
        q_ci.pQueuePriorities = &queue_priority;

        VkPhysicalDeviceShaderFloat16Int8FeaturesKHR float16 = {};
        float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
        float16.shaderFloat16 = VK_TRUE;

        VkPhysicalDevice16BitStorageFeatures storage16 = {};
        storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage16.storageBuffer16BitAccess = VK_TRUE;
        float16.pNext = &storage16;

        VkPhysicalDeviceScalarBlockLayoutFeaturesEXT scalar_layout = {};
        scalar_layout.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT;
        scalar_layout.scalarBlockLayout = VK_TRUE;
        storage16.pNext = &scalar_layout;

        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &float16;

        VkDeviceCreateInfo dev_ci = {};
        dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dev_ci.pNext = &features2;
        dev_ci.queueCreateInfoCount = 1;
        dev_ci.pQueueCreateInfos = &q_ci;
        dev_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_dev_exts.size());
        dev_ci.ppEnabledExtensionNames = enabled_dev_exts.data();

        UniqueDevice device;
        vk_check(vkCreateDevice(physical, &dev_ci, nullptr, &device.handle), "vkCreateDevice");

        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device.handle, queue_family_index, 0, &queue);

        VkCommandPoolCreateInfo pool_ci = {};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.queueFamilyIndex = queue_family_index;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        UniqueDeviceHandle<VkCommandPool, vkDestroyCommandPool> command_pool;
        command_pool.device = device.handle;
        vk_check(vkCreateCommandPool(device.handle, &pool_ci, nullptr, &command_pool.handle),
            "vkCreateCommandPool");

        constexpr uint32_t n = 1024;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(float);

        MappedBuffer a = create_mapped_storage_buffer(physical, device.handle, bytes);
        MappedBuffer b = create_mapped_storage_buffer(physical, device.handle, bytes);
        MappedBuffer c = create_mapped_storage_buffer(physical, device.handle, bytes);

        auto* a_f = static_cast<float*>(a.mapped);
        auto* b_f = static_cast<float*>(b.mapped);
        for (uint32_t i = 0; i < n; ++i) {
            a_f[i] = static_cast<float>(i);
            b_f[i] = static_cast<float>(2 * i);
        }

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
        dsl.device = device.handle;
        vk_check(vkCreateDescriptorSetLayout(device.handle, &dsl_ci, nullptr, &dsl.handle),
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
        pipeline_layout.device = device.handle;
        vk_check(vkCreatePipelineLayout(device.handle, &pl_ci, nullptr, &pipeline_layout.handle),
            "vkCreatePipelineLayout");

        static_assert((mruntime::vulkan::shaders::kVectorAddSpvSize % 4) == 0, "SPIR-V must be word-aligned");
        VkShaderModuleCreateInfo sm_ci = {};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = mruntime::vulkan::shaders::kVectorAddSpvSize;
        sm_ci.pCode = reinterpret_cast<const uint32_t*>(mruntime::vulkan::shaders::kVectorAddSpv);

        UniqueDeviceHandle<VkShaderModule, vkDestroyShaderModule> shader;
        shader.device = device.handle;
        vk_check(vkCreateShaderModule(device.handle, &sm_ci, nullptr, &shader.handle),
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
        pipeline.device = device.handle;
        vk_check(vkCreateComputePipelines(device.handle, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipeline.handle),
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
        descriptor_pool.device = device.handle;
        vk_check(vkCreateDescriptorPool(device.handle, &dp_ci, nullptr, &descriptor_pool.handle),
            "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo ds_ai = {};
        ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_ai.descriptorPool = descriptor_pool.handle;
        ds_ai.descriptorSetCount = 1;
        ds_ai.pSetLayouts = &dsl.handle;

        VkDescriptorSet ds = VK_NULL_HANDLE;
        vk_check(vkAllocateDescriptorSets(device.handle, &ds_ai, &ds), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo a_info = {};
        a_info.buffer = a.buffer;
        a_info.offset = 0;
        a_info.range = bytes;

        VkDescriptorBufferInfo b_info = {};
        b_info.buffer = b.buffer;
        b_info.offset = 0;
        b_info.range = bytes;

        VkDescriptorBufferInfo c_info = {};
        c_info.buffer = c.buffer;
        c_info.offset = 0;
        c_info.range = bytes;

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

        vkUpdateDescriptorSets(device.handle, 3, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cb_ai = {};
        cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_ai.commandPool = command_pool.handle;
        cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_ai.commandBufferCount = 1;

        VkCommandBuffer cb = VK_NULL_HANDLE;
        vk_check(vkAllocateCommandBuffers(device.handle, &cb_ai, &cb), "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vk_check(vkBeginCommandBuffer(cb, &begin), "vkBeginCommandBuffer");

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
        barrier.buffer = c.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr
        );

        vk_check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

        VkFenceCreateInfo fence_ci = {};
        fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        UniqueDeviceHandle<VkFence, vkDestroyFence> fence;
        fence.device = device.handle;
        vk_check(vkCreateFence(device.handle, &fence_ci, nullptr, &fence.handle), "vkCreateFence");

        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cb;
        vk_check(vkQueueSubmit(queue, 1, &submit, fence.handle), "vkQueueSubmit");
        vk_check(vkWaitForFences(device.handle, 1, &fence.handle, VK_TRUE, UINT64_MAX), "vkWaitForFences");

        auto* c_f = static_cast<const float*>(c.mapped);
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
    } catch (const std::exception& e) {
        std::cerr << "vulkan_smoke_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
