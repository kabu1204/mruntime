#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace mruntime {

// Alignment for SIMD (NEON/AVX) operations
constexpr size_t kArenaAlign = 64;

inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

// Simple bump allocator for pre-allocated memory.
// No thread safety - designed for single-threaded control flow.
struct Arena {
    uint8_t* base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;

    // Allocate bytes from the arena (aligned to kArenaAlign).
    // Returns nullptr if out of space (in release builds, asserts in debug).
    void* alloc(size_t bytes) {
        bytes = align_up(bytes, kArenaAlign);
        assert(offset + bytes <= capacity && "Arena out of memory");
        void* ptr = base + offset;
        offset += bytes;
        return ptr;
    }

    // Typed allocation helper
    template <typename T>
    T* alloc_array(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }

    // Reset to beginning (does not free memory)
    void reset() { offset = 0; }

    // Reset to a specific watermark (for scratch reuse)
    void reset_to(size_t watermark) {
        assert(watermark <= offset);
        offset = watermark;
    }

    // Get current watermark for later reset
    size_t watermark() const { return offset; }

    // Remaining capacity
    size_t remaining() const { return capacity - offset; }
};

// Create an arena with the given capacity (malloc-backed).
inline Arena create_arena(size_t capacity) {
    Arena arena;
    arena.capacity = align_up(capacity, kArenaAlign);
    arena.base = static_cast<uint8_t*>(std::aligned_alloc(kArenaAlign, arena.capacity));
    arena.offset = 0;
    return arena;
}

// Free arena memory.
inline void destroy_arena(Arena& arena) {
    if (arena.base) {
        std::free(arena.base);
        arena.base = nullptr;
        arena.capacity = 0;
        arena.offset = 0;
    }
}

// Three-arena design for Qwen2 inference:
// - weights: Model weights (immutable after load)
// - kv_cache: KV cache (grows during generation)
// - scratch: Per-forward activations (reset each forward)
struct Qwen2Arenas {
    Arena weights;
    Arena kv_cache;
    Arena scratch;
};

inline Qwen2Arenas create_qwen2_arenas(size_t weights_bytes, size_t kv_bytes, size_t scratch_bytes) {
    Qwen2Arenas arenas;
    arenas.weights = create_arena(weights_bytes);
    arenas.kv_cache = create_arena(kv_bytes);
    arenas.scratch = create_arena(scratch_bytes);
    return arenas;
}

inline void destroy_qwen2_arenas(Qwen2Arenas& arenas) {
    destroy_arena(arenas.weights);
    destroy_arena(arenas.kv_cache);
    destroy_arena(arenas.scratch);
}

}  // namespace mruntime
