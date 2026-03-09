#include "vk_buffer_arena.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "vk_helpers.h"

namespace mruntime::vulkan {

namespace {

VkDeviceSize choose_default_alignment(VkDeviceSize user_alignment) {
    constexpr VkDeviceSize kMin = 64;
    return std::max(kMin, user_alignment);
}

uint32_t popcount_u32(uint32_t value) {
    uint32_t count = 0;
    while (value != 0) {
        count += (value & 1u);
        value >>= 1u;
    }
    return count;
}

struct MemoryTypeSelection {
    uint32_t type_index = UINT32_MAX;
    VkMemoryPropertyFlags type_properties = 0;
};

MemoryTypeSelection select_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t memory_type_bits,
    VkMemoryPropertyFlags required_properties,
    VkMemoryPropertyFlags preferred_properties
) {
    VkPhysicalDeviceMemoryProperties mem_props = {};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    int32_t best_score = -1;
    MemoryTypeSelection selected = {};

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((memory_type_bits & (1u << i)) == 0) {
            continue;
        }

        const VkMemoryPropertyFlags candidate = mem_props.memoryTypes[i].propertyFlags;
        if ((candidate & required_properties) != required_properties) {
            continue;
        }

        const int32_t score = static_cast<int32_t>(popcount_u32(candidate & preferred_properties));
        if (score > best_score) {
            best_score = score;
            selected.type_index = i;
            selected.type_properties = candidate;
        }
    }

    if (best_score < 0) {
        throw std::runtime_error("No suitable Vulkan memory type found");
    }

    return selected;
}

}  // namespace

VkBufferArena VkBufferArena::Create(
    VkPhysicalDevice physical_device,
    VkDevice device,
    const VkBufferArenaCreateInfo& info
) {
    if (device == VK_NULL_HANDLE) {
        throw std::runtime_error("VkBufferArena::Create: device is null");
    }
    if (physical_device == VK_NULL_HANDLE) {
        throw std::runtime_error("VkBufferArena::Create: physical_device is null");
    }
    if (info.capacity_bytes == 0) {
        throw std::runtime_error("VkBufferArena::Create: capacity_bytes must be > 0");
    }

    VkBufferArena arena;
    arena.device_ = device;
    arena.capacity_ = info.capacity_bytes;
    arena.default_alignment_ = choose_default_alignment(info.default_alignment);

    VkBufferCreateInfo buf_ci = {};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = info.capacity_bytes;
    buf_ci.usage = info.usage;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device, &buf_ci, nullptr, &arena.buffer_), "vkCreateBuffer");

    VkMemoryRequirements req = {};
    vkGetBufferMemoryRequirements(device, arena.buffer_, &req);

    const MemoryTypeSelection memory_type = select_memory_type(
        physical_device,
        req.memoryTypeBits,
        info.memory_properties,
        info.preferred_memory_properties
    );
    arena.memory_properties_ = memory_type.type_properties;

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = memory_type.type_index;
    vk_check(vkAllocateMemory(device, &alloc_info, nullptr, &arena.memory_), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device, arena.buffer_, arena.memory_, 0), "vkBindBufferMemory");

    const bool want_map = (arena.memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (want_map) {
        vk_check(vkMapMemory(device, arena.memory_, 0, info.capacity_bytes, 0, &arena.mapped_), "vkMapMemory");
        std::memset(arena.mapped_, 0, static_cast<size_t>(info.capacity_bytes));
    }

    return arena;
}

VkBufferArena::~VkBufferArena() {
    destroy();
}

VkBufferArena::VkBufferArena(VkBufferArena&& other) noexcept {
    *this = std::move(other);
}

VkBufferArena& VkBufferArena::operator=(VkBufferArena&& other) noexcept {
    if (this == &other) return *this;
    destroy();

    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
    memory_properties_ = std::exchange(other.memory_properties_, VkMemoryPropertyFlags{0});
    mapped_ = std::exchange(other.mapped_, nullptr);
    capacity_ = std::exchange(other.capacity_, VkDeviceSize{0});
    offset_ = std::exchange(other.offset_, VkDeviceSize{0});
    default_alignment_ = std::exchange(other.default_alignment_, VkDeviceSize{0});

    return *this;
}

VkDeviceSize VkBufferArena::alloc(VkDeviceSize bytes) {
    return alloc_aligned(bytes, default_alignment_);
}

VkDeviceSize VkBufferArena::alloc_aligned(VkDeviceSize bytes, VkDeviceSize alignment) {
    if (alignment == 0) {
        throw std::runtime_error("VkBufferArena::alloc_aligned: alignment must be > 0");
    }
    const VkDeviceSize aligned_off = mruntime::vulkan::align_up(offset_, alignment);
    const VkDeviceSize end = aligned_off + bytes;
    if (end > capacity_) {
        throw std::runtime_error("VkBufferArena out of memory");
    }
    offset_ = end;
    return aligned_off;
}

VkDescriptorBufferInfo VkBufferArena::descriptor(VkDeviceSize offset, VkDeviceSize range) const {
    VkDescriptorBufferInfo info = {};
    info.buffer = buffer_;
    info.offset = offset;
    info.range = range;
    return info;
}

void VkBufferArena::destroy() noexcept {
    if (mapped_) {
        vkUnmapMemory(device_, memory_);
        mapped_ = nullptr;
    }
    if (buffer_) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (memory_) {
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    memory_properties_ = 0;
    device_ = VK_NULL_HANDLE;
    capacity_ = 0;
    offset_ = 0;
    default_alignment_ = 0;
}

}  // namespace mruntime::vulkan
