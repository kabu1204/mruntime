#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
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

// FP16 conversion utilities
// IEEE754 binary16 stored as uint16_t.

inline uint16_t float_to_fp16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;

    // NaN / Inf
    if (exp == 0xFFu) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
        if ((payload & 0x03FFu) == 0) payload |= 0x0001u;
        return static_cast<uint16_t>(sign | 0x7C00u | payload);
    }

    // Zero / subnormal float -> zero half (flush denormals; fine for this runtime)
    if (exp == 0) {
        return static_cast<uint16_t>(sign);
    }

    const int32_t half_exp = static_cast<int32_t>(exp) - 127 + 15;

    // Overflow -> Inf
    if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Underflow -> subnormal/zero
    if (half_exp <= 0) {
        // Too small -> zero
        if (half_exp < -10) {
            return static_cast<uint16_t>(sign);
        }

        // Subnormal: shift mantissa so exponent becomes 0.
        uint32_t mant = mantissa | 0x800000u;
        const int32_t shift = 1 - half_exp;
        // We need a 10-bit mantissa; mant has 24 bits (including implicit leading 1).
        // Total shift = 13 (to drop to 11 bits) + (shift-1) (to account for exponent underflow)
        const int32_t rshift = 13 + shift;
        uint32_t half_mant = mant >> rshift;

        // Round to nearest even.
        const uint32_t round_bit = 1u << (rshift - 1);
        const uint32_t remainder = mant & (round_bit - 1);
        const uint32_t lsb = half_mant & 1u;
        if ((mant & round_bit) && (remainder || lsb)) {
            ++half_mant;
        }

        // Rounding can promote to the smallest normal.
        if (half_mant == 0x400u) {
            return static_cast<uint16_t>(sign | 0x0400u);
        }

        return static_cast<uint16_t>(sign | (half_mant & 0x03FFu));
    }

    // Normal half
    uint32_t half_mant = mantissa >> 13;
    const uint32_t round_bit = 0x1000u;
    const uint32_t remainder = mantissa & (round_bit - 1);
    const uint32_t lsb = half_mant & 1u;
    if ((mantissa & round_bit) && (remainder || lsb)) {
        ++half_mant;
        if (half_mant == 0x400u) {
            // Mantissa overflow; increment exponent.
            half_mant = 0;
            const uint32_t half_exp_inc = static_cast<uint32_t>(half_exp + 1);
            if (half_exp_inc >= 31u) {
                return static_cast<uint16_t>(sign | 0x7C00u);
            }
            return static_cast<uint16_t>(sign | (half_exp_inc << 10));
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10) | (half_mant & 0x03FFu));
}

inline float fp16_bits_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }
        // Subnormal: mantissa * 2^-24
        float val = std::ldexp(static_cast<float>(mant), -24);
        return (sign ? -val : val);
    }

    if (exp == 31) {
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Normal: (1 + mant/1024) * 2^(exp-15)
    float val = std::ldexp(1.0f + (static_cast<float>(mant) / 1024.0f), static_cast<int>(exp) - 15);
    return (sign ? -val : val);
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

inline float load_scalar_as_fp32(const void* data, DType dtype, size_t index) {
    switch (dtype) {
        case DType::FP32:
            return static_cast<const float*>(data)[index];
        case DType::BF16:
            return bf16_to_float(static_cast<const uint16_t*>(data)[index]);
        case DType::FP16:
            return fp16_bits_to_float(static_cast<const uint16_t*>(data)[index]);
    }
    throw std::invalid_argument("Unknown dtype");
}

inline void store_scalar_from_fp32(void* data, DType dtype, size_t index, float value) {
    switch (dtype) {
        case DType::FP32:
            static_cast<float*>(data)[index] = value;
            return;
        case DType::BF16:
            static_cast<uint16_t*>(data)[index] = float_to_bf16(value);
            return;
        case DType::FP16:
            static_cast<uint16_t*>(data)[index] = float_to_fp16_bits(value);
            return;
    }
    throw std::invalid_argument("Unknown dtype");
}

}  // namespace mruntime
