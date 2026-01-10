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
    Tensor t = Tensor::empty(Shape({2, 3}), DType::FP16);
    assert(t.ndim() == 2);
    assert(t.dim(0) == 2);
    assert(t.dim(1) == 3);
    assert(t.numel() == 6);
    assert(t.nbytes() == 12);
    assert(t.dtype() == DType::FP16);
    assert(t.owns_data());
    assert(t.is_contiguous());
    std::cout << "test_tensor_empty PASSED\n";
}

void test_tensor_zeros() {
    Tensor t = Tensor::zeros(Shape({3, 4}), DType::FP16);
    const uint16_t* data = t.data_ptr<uint16_t>();
    for (size_t i = 0; i < t.numel(); ++i) {
        assert(data[i] == 0);
    }
    std::cout << "test_tensor_zeros PASSED\n";
}

void test_tensor_from_buffer() {
    uint16_t buffer[6] = {
        float_to_fp16_bits(1.0f),
        float_to_fp16_bits(2.0f),
        float_to_fp16_bits(3.0f),
        float_to_fp16_bits(4.0f),
        float_to_fp16_bits(5.0f),
        float_to_fp16_bits(6.0f),
    };
    Tensor t = Tensor::from_buffer(buffer, Shape({2, 3}), DType::FP16, false);
    assert(!t.owns_data());
    assert(t.data_ptr<uint16_t>() == buffer);
    std::cout << "test_tensor_from_buffer PASSED\n";
}

void test_tensor_slice() {
    Tensor t = Tensor::zeros(Shape({2, 4, 3}), DType::FP16);
    uint16_t* data = t.data_ptr<uint16_t>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = float_to_fp16_bits(static_cast<float>(i));
    }

    Tensor sliced = t.slice(1, 1, 3);
    assert(sliced.dim(0) == 2);
    assert(sliced.dim(1) == 2);
    assert(sliced.dim(2) == 3);
    assert(!sliced.owns_data());
    assert(sliced.data_ptr<uint16_t>() == data + 3);
    std::cout << "test_tensor_slice PASSED\n";
}

void test_tensor_to_contiguous() {
    Tensor t = Tensor::zeros(Shape({2, 4, 3}), DType::FP16);
    uint16_t* data = t.data_ptr<uint16_t>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = float_to_fp16_bits(static_cast<float>(i));
    }

    Tensor sliced = t.slice(1, 1, 3);
    assert(!sliced.is_contiguous());

    Tensor contig = sliced.to_contiguous();
    assert(contig.is_contiguous());
    assert(contig.owns_data());

    const uint16_t* cdata = contig.data_ptr<uint16_t>();
    assert(fp16_bits_to_float(cdata[0]) == 3.0f);
    assert(fp16_bits_to_float(cdata[1]) == 4.0f);
    assert(fp16_bits_to_float(cdata[2]) == 5.0f);
    std::cout << "test_tensor_to_contiguous PASSED\n";
}

void test_dtype_conversion() {
    Tensor t = Tensor::zeros(Shape({4}), DType::FP16);
    uint16_t* data = t.data_ptr<uint16_t>();
    data[0] = float_to_fp16_bits(1.0f);
    data[1] = float_to_fp16_bits(2.5f);
    data[2] = float_to_fp16_bits(-3.0f);
    data[3] = float_to_fp16_bits(0.0f);

    Tensor bf16 = t.to(DType::BF16);
    assert(bf16.dtype() == DType::BF16);
    assert(bf16.nbytes() == 8);

    Tensor fp16_back = bf16.to(DType::FP16);
    assert(fp16_back.dtype() == DType::FP16);
    const uint16_t* back_data = fp16_back.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(back_data[0]) - 1.0f) < 0.01f);
    assert(std::abs(fp16_bits_to_float(back_data[1]) - 2.5f) < 0.01f);
    assert(std::abs(fp16_bits_to_float(back_data[2]) - (-3.0f)) < 0.02f);
    assert(fp16_bits_to_float(back_data[3]) == 0.0f);
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
