#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace mruntime::vulkan {

struct VkBufferArenaCreateInfo {
    VkDeviceSize capacity_bytes = 0;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlags memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryPropertyFlags preferred_memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkDeviceSize default_alignment = 0;  // 0 -> use a reasonable default.
};

// Simple bump allocator for a single VkBuffer allocation.
// Intended to mirror `mruntime::Arena` semantics for GPU buffers.
class VkBufferArena {
  public:
    VkBufferArena() = default;
    static VkBufferArena Create(
        VkPhysicalDevice physical_device,
        VkDevice device,
        const VkBufferArenaCreateInfo& info
    );

    ~VkBufferArena();

    VkBufferArena(const VkBufferArena&) = delete;
    VkBufferArena& operator=(const VkBufferArena&) = delete;

    VkBufferArena(VkBufferArena&& other) noexcept;
    VkBufferArena& operator=(VkBufferArena&& other) noexcept;

    VkDevice device() const noexcept { return device_; }
    VkBuffer buffer() const noexcept { return buffer_; }
    VkDeviceMemory memory() const noexcept { return memory_; }
    VkMemoryPropertyFlags memory_properties() const noexcept { return memory_properties_; }

    VkDeviceSize capacity() const noexcept { return capacity_; }
    VkDeviceSize offset() const noexcept { return offset_; }
    VkDeviceSize default_alignment() const noexcept { return default_alignment_; }

    void* mapped() const noexcept { return mapped_; }

    void reset() noexcept { offset_ = 0; }

    VkDeviceSize alloc(VkDeviceSize bytes);
    VkDeviceSize alloc_aligned(VkDeviceSize bytes, VkDeviceSize alignment);

    VkDescriptorBufferInfo descriptor(VkDeviceSize offset, VkDeviceSize range) const;

    template <class T>
    T* host_ptr(VkDeviceSize offset_bytes) const noexcept {
        return mapped_ == nullptr ? nullptr : reinterpret_cast<T*>(static_cast<uint8_t*>(mapped_) + offset_bytes);
    }

  private:
    void destroy() noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkMemoryPropertyFlags memory_properties_ = 0;
    void* mapped_ = nullptr;

    VkDeviceSize capacity_ = 0;
    VkDeviceSize offset_ = 0;
    VkDeviceSize default_alignment_ = 0;
};

}  // namespace mruntime::vulkan
