#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "mruntime/dtype.h"

namespace mruntime {

class Shape {
public:
    Shape() = default;

    Shape(std::initializer_list<size_t> dims) : dims_(dims) {
        compute_strides();
    }

    explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {
        compute_strides();
    }

    Shape(std::vector<size_t> dims, std::vector<size_t> strides)
        : dims_(std::move(dims)), strides_(std::move(strides)) {
        assert(dims_.size() == strides_.size());
    }

    size_t ndim() const { return dims_.size(); }
    size_t dim(size_t i) const { return dims_.at(i); }
    size_t stride(size_t i) const { return strides_.at(i); }

    const std::vector<size_t>& dims() const { return dims_; }
    const std::vector<size_t>& strides() const { return strides_; }

    size_t numel() const {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), size_t(1),
                               std::multiplies<size_t>());
    }

    bool is_contiguous() const {
        if (dims_.empty()) return true;
        size_t expected_stride = 1;
        for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) return false;
            expected_stride *= dims_[i];
        }
        return true;
    }

    bool operator==(const Shape& other) const {
        return dims_ == other.dims_ && strides_ == other.strides_;
    }

    bool operator!=(const Shape& other) const { return !(*this == other); }

    Shape slice(size_t dim_idx, size_t start, size_t end) const {
        assert(dim_idx < ndim());
        assert(start < end && end <= dims_[dim_idx]);
        std::vector<size_t> new_dims = dims_;
        new_dims[dim_idx] = end - start;
        return Shape(std::move(new_dims), strides_);
    }

private:
    void compute_strides() {
        strides_.resize(dims_.size());
        if (dims_.empty()) return;
        size_t stride = 1;
        for (int i = static_cast<int>(dims_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= dims_[i];
        }
    }

    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
};

class Tensor {
public:
    Tensor() = default;

    static Tensor empty(Shape shape, DType dtype);
    static Tensor zeros(Shape shape, DType dtype);
    static Tensor from_buffer(void* data, Shape shape, DType dtype, bool owns_data = false);

    Shape shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t numel() const { return shape_.numel(); }
    size_t nbytes() const { return numel() * dtype_size(dtype_); }
    size_t ndim() const { return shape_.ndim(); }
    size_t dim(size_t i) const { return shape_.dim(i); }
    size_t stride(size_t i) const { return shape_.stride(i); }
    bool is_contiguous() const { return shape_.is_contiguous(); }
    bool owns_data() const { return owns_data_; }

    void* data() { return data_; }
    const void* data() const { return data_; }

    template <typename T>
    T* data_ptr() { return static_cast<T*>(data_); }

    template <typename T>
    const T* data_ptr() const { return static_cast<const T*>(data_); }

    Tensor slice(size_t dim_idx, size_t start, size_t end) const;

    Tensor to_contiguous() const;
    Tensor to(DType target_dtype) const;

    // Transpose dimensions (creates a new contiguous tensor)
    // e.g., permute({0, 2, 1, 3}) for [B,S,H,D] -> [B,H,S,D]
    Tensor permute(const std::vector<size_t>& dims) const;

private:
    Tensor(Shape shape, DType dtype, std::shared_ptr<std::byte[]> storage, void* data, bool owns_data)
        : shape_(std::move(shape)), dtype_(dtype), storage_(std::move(storage)),
          data_(data), owns_data_(owns_data) {}

    Shape shape_;
    DType dtype_ = DType::FP32;
    std::shared_ptr<std::byte[]> storage_;
    void* data_ = nullptr;
    bool owns_data_ = false;
};

}  // namespace mruntime
