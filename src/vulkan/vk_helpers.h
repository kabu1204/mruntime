#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mruntime::vulkan {

class VulkanError final : public std::runtime_error {
  public:
    VulkanError(VkResult result, const char* what)
        : std::runtime_error(std::string(what) + " failed with VkResult=" + std::to_string(result)),
          result_(result) {}

    VkResult result() const noexcept { return result_; }

  private:
    VkResult result_ = VK_SUCCESS;
};

inline void vk_check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) {
        return;
    }
    throw VulkanError(result, what);
}

inline std::vector<VkExtensionProperties> enumerate_instance_extensions() {
    uint32_t count = 0;
    vk_check(
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
        "vkEnumerateInstanceExtensionProperties(count)");
    std::vector<VkExtensionProperties> exts(count);
    vk_check(
        vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data()),
        "vkEnumerateInstanceExtensionProperties(data)");
    return exts;
}

inline std::vector<VkExtensionProperties> enumerate_device_extensions(VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    vk_check(
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
        "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> exts(count);
    vk_check(
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, exts.data()),
        "vkEnumerateDeviceExtensionProperties(data)");
    return exts;
}

inline bool has_extension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

inline uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
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

inline uint32_t find_memory_type(
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

inline VkDeviceSize align_up(VkDeviceSize n, VkDeviceSize align) {
    if (align == 0) return n;
    return (n + align - 1) & ~(align - 1);
}

}  // namespace mruntime::vulkan

