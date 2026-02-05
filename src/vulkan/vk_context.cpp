#include "vk_context.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "vk_helpers.h"

namespace mruntime::vulkan {

namespace {

constexpr const char* kKhrPortabilitySubsetExtensionName = "VK_KHR_portability_subset";

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

bool device_supports_required_extensions(VkPhysicalDevice physical_device, bool* out_has_portability_subset) {
    const auto exts = enumerate_device_extensions(physical_device);
    const std::array<const char*, 3> required = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
    };
    for (const char* ext : required) {
        if (!has_extension(exts, ext)) {
            return false;
        }
    }
    if (out_has_portability_subset) {
        *out_has_portability_subset = has_extension(exts, kKhrPortabilitySubsetExtensionName);
    }
    return true;
}

}  // namespace

VkContext VkContext::Create(const VkContextCreateInfo& info) {
    (void)info;  // Validation not wired yet.

    VkContext ctx;

    const auto instance_exts = enumerate_instance_extensions();
    std::vector<const char*> enabled_instance_exts;
    VkInstanceCreateFlags instance_flags = 0;

    if (has_extension(instance_exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabled_instance_exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "mruntime";
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

    vk_check(vkCreateInstance(&instance_ci, nullptr, &ctx.instance_), "vkCreateInstance");

    uint32_t device_count = 0;
    vk_check(vkEnumeratePhysicalDevices(ctx.instance_, &device_count, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vk_check(vkEnumeratePhysicalDevices(ctx.instance_, &device_count, devices.data()), "vkEnumeratePhysicalDevices");

    bool selected_has_portability_subset = false;
    for (VkPhysicalDevice dev : devices) {
        bool has_portability_subset = false;
        if (!device_supports_required_extensions(dev, &has_portability_subset)) {
            continue;
        }

        try {
            (void)find_compute_queue_family(dev);
            require_fp16_features(dev);
        } catch (...) {
            continue;
        }

        ctx.physical_device_ = dev;
        selected_has_portability_subset = has_portability_subset;
        break;
    }

    if (ctx.physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("No Vulkan physical device satisfies required extensions/features");
    }

    const uint32_t queue_family_index = find_compute_queue_family(ctx.physical_device_);
    ctx.queue_family_index_ = queue_family_index;

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(ctx.physical_device_, &props);
    ctx.min_storage_buffer_offset_alignment_ = props.limits.minStorageBufferOffsetAlignment;

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

    std::vector<const char*> enabled_dev_exts = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
    };
    if (selected_has_portability_subset) {
        enabled_dev_exts.push_back(kKhrPortabilitySubsetExtensionName);
    }

    VkDeviceCreateInfo dev_ci = {};
    dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.pNext = &features2;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &q_ci;
    dev_ci.enabledExtensionCount = static_cast<uint32_t>(enabled_dev_exts.size());
    dev_ci.ppEnabledExtensionNames = enabled_dev_exts.data();

    vk_check(vkCreateDevice(ctx.physical_device_, &dev_ci, nullptr, &ctx.device_), "vkCreateDevice");

    vkGetDeviceQueue(ctx.device_, queue_family_index, 0, &ctx.queue_);

    VkCommandPoolCreateInfo pool_ci = {};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = queue_family_index;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk_check(vkCreateCommandPool(ctx.device_, &pool_ci, nullptr, &ctx.command_pool_), "vkCreateCommandPool");

    return ctx;
}

VkContext::~VkContext() {
    reset();
}

VkContext::VkContext(VkContext&& other) noexcept {
    *this = std::move(other);
}

VkContext& VkContext::operator=(VkContext&& other) noexcept {
    if (this == &other) return *this;
    reset();

    instance_ = std::exchange(other.instance_, VK_NULL_HANDLE);
    physical_device_ = std::exchange(other.physical_device_, VK_NULL_HANDLE);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    queue_ = std::exchange(other.queue_, VK_NULL_HANDLE);
    queue_family_index_ = std::exchange(other.queue_family_index_, UINT32_MAX);
    command_pool_ = std::exchange(other.command_pool_, VK_NULL_HANDLE);
    min_storage_buffer_offset_alignment_ = std::exchange(other.min_storage_buffer_offset_alignment_, VkDeviceSize{0});

    return *this;
}

void VkContext::reset() noexcept {
    if (device_ != VK_NULL_HANDLE) {
        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    queue_ = VK_NULL_HANDLE;
    queue_family_index_ = UINT32_MAX;
    physical_device_ = VK_NULL_HANDLE;

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

}  // namespace mruntime::vulkan

