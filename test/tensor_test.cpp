#include <cassert>
#include <cmath>
#include <iostream>

#include "mruntime/tensor.h"

using namespace mruntime;

void test_shape_creation() {
    Shape s1({2, 3, 4});
    assert(s1.ndim() == 3);
    assert(s1.dim(0) == 2);
    assert(s1.dim(1) == 3);
    assert(s1.dim(2) == 4);
    assert(s1.numel() == 24);
    assert(s1.is_contiguous());
    assert(s1.stride(0) == 12);
    assert(s1.stride(1) == 4);
    assert(s1.stride(2) == 1);
    std::cout << "test_shape_creation PASSED\n";
}

void test_tensor_empty() {
    Tensor t = Tensor::empty(Shape({2, 3}), DType::FP32);
    assert(t.ndim() == 2);
    assert(t.dim(0) == 2);
    assert(t.dim(1) == 3);
    assert(t.numel() == 6);
    assert(t.nbytes() == 24);
    assert(t.dtype() == DType::FP32);
    assert(t.owns_data());
    assert(t.is_contiguous());
    std::cout << "test_tensor_empty PASSED\n";
}

void test_tensor_zeros() {
    Tensor t = Tensor::zeros(Shape({3, 4}), DType::FP32);
    const float* data = t.data_ptr<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        assert(data[i] == 0.0f);
    }
    std::cout << "test_tensor_zeros PASSED\n";
}

void test_tensor_from_buffer() {
    float buffer[6] = {1, 2, 3, 4, 5, 6};
    Tensor t = Tensor::from_buffer(buffer, Shape({2, 3}), DType::FP32, false);
    assert(!t.owns_data());
    assert(t.data_ptr<float>() == buffer);
    std::cout << "test_tensor_from_buffer PASSED\n";
}

void test_tensor_slice() {
    Tensor t = Tensor::zeros(Shape({2, 4, 3}), DType::FP32);
    float* data = t.data_ptr<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }

    Tensor sliced = t.slice(1, 1, 3);
    assert(sliced.dim(0) == 2);
    assert(sliced.dim(1) == 2);
    assert(sliced.dim(2) == 3);
    assert(!sliced.owns_data());
    assert(sliced.data_ptr<float>() == data + 3);
    std::cout << "test_tensor_slice PASSED\n";
}

void test_tensor_to_contiguous() {
    Tensor t = Tensor::zeros(Shape({2, 4, 3}), DType::FP32);
    float* data = t.data_ptr<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }

    Tensor sliced = t.slice(1, 1, 3);
    assert(!sliced.is_contiguous());

    Tensor contig = sliced.to_contiguous();
    assert(contig.is_contiguous());
    assert(contig.owns_data());

    const float* cdata = contig.data_ptr<float>();
    assert(cdata[0] == 3.0f);
    assert(cdata[1] == 4.0f);
    assert(cdata[2] == 5.0f);
    std::cout << "test_tensor_to_contiguous PASSED\n";
}

void test_dtype_conversion() {
    Tensor t = Tensor::zeros(Shape({4}), DType::FP32);
    float* data = t.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.5f;
    data[2] = -3.0f;
    data[3] = 0.0f;

    Tensor fp16 = t.to(DType::FP16);
    assert(fp16.dtype() == DType::FP16);
    assert(fp16.nbytes() == 8);

    Tensor fp32_back = fp16.to(DType::FP32);
    assert(fp32_back.dtype() == DType::FP32);
    const float* back_data = fp32_back.data_ptr<float>();
    assert(std::abs(back_data[0] - 1.0f) < 0.01f);
    assert(std::abs(back_data[1] - 2.5f) < 0.01f);
    assert(std::abs(back_data[2] - (-3.0f)) < 0.01f);
    assert(back_data[3] == 0.0f);
    std::cout << "test_dtype_conversion PASSED\n";
}

void test_fp16_tensor() {
    Tensor t = Tensor::empty(Shape({2, 3}), DType::FP16);
    assert(t.dtype() == DType::FP16);
    assert(t.nbytes() == 12);
    std::cout << "test_fp16_tensor PASSED\n";
}

int main() {
    test_shape_creation();
    test_tensor_empty();
    test_tensor_zeros();
    test_tensor_from_buffer();
    test_tensor_slice();
    test_tensor_to_contiguous();
    test_dtype_conversion();
    test_fp16_tensor();

    std::cout << "\nAll tensor tests passed!\n";
    return 0;
}
