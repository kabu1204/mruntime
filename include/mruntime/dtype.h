#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace mruntime {

enum class DType {
    FP16,
    BF16,
    FP32,
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::FP32: return 4;
        default: throw std::invalid_argument("Unknown dtype");
    }
}

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::FP16: return "FP16";
        case DType::BF16: return "BF16";
        case DType::FP32: return "FP32";
        default: return "UNKNOWN";
    }
}

// BF16 conversion utilities
// BF16 is the upper 16 bits of FP32 with rounding

inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even: add rounding bias based on LSB and remaining bits
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

}  // namespace mruntime
