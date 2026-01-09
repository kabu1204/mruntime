#include "mruntime/tensor.h"

#include <cmath>
#include <stdexcept>

namespace mruntime {

Tensor Tensor::empty(Shape shape, DType dtype) {
    size_t nbytes = shape.numel() * dtype_size(dtype);
    auto storage = std::shared_ptr<std::byte[]>(new std::byte[nbytes]);
    return Tensor(std::move(shape), dtype, storage, storage.get(), true);
}

Tensor Tensor::zeros(Shape shape, DType dtype) {
    Tensor t = empty(std::move(shape), dtype);
    std::memset(t.data(), 0, t.nbytes());
    return t;
}

Tensor Tensor::from_buffer(void* data, Shape shape, DType dtype, bool owns_data) {
    std::shared_ptr<std::byte[]> storage;
    if (owns_data) {
        storage = std::shared_ptr<std::byte[]>(static_cast<std::byte*>(data));
    }
    return Tensor(std::move(shape), dtype, storage, data, owns_data);
}

Tensor Tensor::slice(size_t dim_idx, size_t start, size_t end) const {
    assert(dim_idx < ndim());
    assert(start < end && end <= dim(dim_idx));
    Shape new_shape = shape_.slice(dim_idx, start, end);
    size_t byte_offset = start * shape_.stride(dim_idx) * dtype_size(dtype_);
    void* new_data = static_cast<std::byte*>(data_) + byte_offset;
    return Tensor(std::move(new_shape), dtype_, storage_, new_data, false);
}

Tensor Tensor::to_contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    Tensor result = empty(Shape(shape_.dims()), dtype_);
    if (ndim() == 0 || numel() == 0) return result;

    const size_t elem_size = dtype_size(dtype_);
    const std::byte* src = static_cast<const std::byte*>(data_);
    std::byte* dst = static_cast<std::byte*>(result.data());

    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
        size_t src_offset = 0;
        for (size_t d = 0; d < ndim(); ++d) {
            src_offset += indices[d] * shape_.stride(d);
        }
        std::memcpy(dst + i * elem_size, src + src_offset * elem_size, elem_size);
        for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape_.dim(d)) break;
            indices[d] = 0;
        }
    }
    return result;
}

Tensor Tensor::to(DType target_dtype) const {
    if (dtype_ == target_dtype) {
        return *this;
    }
    Tensor src = to_contiguous();
    Tensor result = empty(Shape(shape_.dims()), target_dtype);
    const size_t n = numel();

    // FP32 -> FP16
    if (dtype_ == DType::FP32 && target_dtype == DType::FP16) {
        const float* src_ptr = src.data_ptr<float>();
        uint16_t* dst_ptr = result.data_ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) {
            dst_ptr[i] = float_to_fp16_bits(src_ptr[i]);
        }
    }
    // FP16 -> FP32
    else if (dtype_ == DType::FP16 && target_dtype == DType::FP32) {
        const uint16_t* src_ptr = src.data_ptr<uint16_t>();
        float* dst_ptr = result.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            dst_ptr[i] = fp16_bits_to_float(src_ptr[i]);
        }
    }
    // FP32 -> BF16
    else if (dtype_ == DType::FP32 && target_dtype == DType::BF16) {
        const float* src_ptr = src.data_ptr<float>();
        uint16_t* dst_ptr = result.data_ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) {
            dst_ptr[i] = float_to_bf16(src_ptr[i]);
        }
    }
    // BF16 -> FP32
    else if (dtype_ == DType::BF16 && target_dtype == DType::FP32) {
        const uint16_t* src_ptr = src.data_ptr<uint16_t>();
        float* dst_ptr = result.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            dst_ptr[i] = bf16_to_float(src_ptr[i]);
        }
    }
    // FP16 <-> BF16 (via FP32 intermediate)
    else if ((dtype_ == DType::FP16 && target_dtype == DType::BF16) ||
             (dtype_ == DType::BF16 && target_dtype == DType::FP16)) {
        Tensor fp32 = src.to(DType::FP32);
        return fp32.to(target_dtype);
    }
    else {
        throw std::runtime_error("Unsupported dtype conversion");
    }
    return result;
}

Tensor Tensor::permute(const std::vector<size_t>& perm) const {
    assert(perm.size() == ndim());

    // Compute new shape
    std::vector<size_t> new_dims(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        new_dims[i] = dim(perm[i]);
    }

    Tensor result = empty(Shape(new_dims), dtype_);

    // Copy data with permutation
    const size_t elem_size = dtype_size(dtype_);
    const std::byte* src = static_cast<const std::byte*>(data_);
    std::byte* dst = static_cast<std::byte*>(result.data());

    std::vector<size_t> src_indices(ndim(), 0);
    std::vector<size_t> dst_indices(ndim(), 0);

    for (size_t i = 0; i < numel(); ++i) {
        // Compute source offset
        size_t src_offset = 0;
        for (size_t d = 0; d < ndim(); ++d) {
            src_offset += src_indices[d] * shape_.stride(d);
        }

        // Copy element
        std::memcpy(dst + i * elem_size, src + src_offset * elem_size, elem_size);

        // Increment destination indices (in destination order)
        // and map to source indices via inverse permutation
        for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
            dst_indices[d]++;
            src_indices[perm[d]]++;
            if (dst_indices[d] < new_dims[d]) break;
            dst_indices[d] = 0;
            src_indices[perm[d]] = 0;
        }
    }

    return result;
}

}  // namespace mruntime
