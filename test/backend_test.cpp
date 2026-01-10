#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mruntime/cpu_backend.h"
#include "mruntime/tensor.h"

using namespace mruntime;

void test_gemm() {
    CpuBackend backend;

    Tensor A = Tensor::zeros(Shape({2, 3}), DType::FP16);
    Tensor B = Tensor::zeros(Shape({3, 4}), DType::FP16);
    Tensor C = Tensor::zeros(Shape({2, 4}), DType::FP16);

    uint16_t* a = A.data_ptr<uint16_t>();
    uint16_t* b = B.data_ptr<uint16_t>();
    for (int i = 0; i < 6; ++i) a[i] = float_to_fp16_bits(static_cast<float>(i + 1));
    for (int i = 0; i < 12; ++i) b[i] = float_to_fp16_bits(static_cast<float>(i + 1));

    backend.gemm(A, B, C);

    const uint16_t* c = C.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(c[0]) - 38.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(c[1]) - 44.0f) < 0.05f);
    std::cout << "test_gemm PASSED\n";
}

void test_rmsnorm() {
    CpuBackend backend;

    Tensor input = Tensor::zeros(Shape({1, 2, 4}), DType::FP16);
    Tensor weight = Tensor::zeros(Shape({4}), DType::FP16);
    Tensor output = Tensor::zeros(Shape({1, 2, 4}), DType::FP16);

    uint16_t* in = input.data_ptr<uint16_t>();
    uint16_t* w = weight.data_ptr<uint16_t>();
    for (int i = 0; i < 8; ++i) in[i] = float_to_fp16_bits(static_cast<float>(i + 1));
    for (int i = 0; i < 4; ++i) w[i] = float_to_fp16_bits(1.0f);

    backend.rmsnorm(input, weight, output, 1e-6f);

    const uint16_t* out = output.data_ptr<uint16_t>();
    float expected_rms = std::sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-6f);
    assert(std::abs(fp16_bits_to_float(out[0]) - 1.0f / expected_rms) < 2e-2f);
    std::cout << "test_rmsnorm PASSED\n";
}

void test_silu() {
    CpuBackend backend;

    Tensor input = Tensor::zeros(Shape({4}), DType::FP16);
    Tensor output = Tensor::zeros(Shape({4}), DType::FP16);

    uint16_t* in = input.data_ptr<uint16_t>();
    in[0] = float_to_fp16_bits(0.0f);
    in[1] = float_to_fp16_bits(1.0f);
    in[2] = float_to_fp16_bits(-1.0f);
    in[3] = float_to_fp16_bits(2.0f);

    backend.silu(input, output);

    const uint16_t* out = output.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(out[0]) - 0.0f) < 1e-5f);

    float expected_1 = 1.0f / (1.0f + std::exp(-1.0f));
    assert(std::abs(fp16_bits_to_float(out[1]) - expected_1) < 2e-2f);

    float expected_neg1 = -1.0f / (1.0f + std::exp(1.0f));
    assert(std::abs(fp16_bits_to_float(out[2]) - expected_neg1) < 2e-2f);

    std::cout << "test_silu PASSED\n";
}

void test_elementwise_mul() {
    CpuBackend backend;

    Tensor a = Tensor::zeros(Shape({4}), DType::FP16);
    Tensor b = Tensor::zeros(Shape({4}), DType::FP16);
    Tensor output = Tensor::zeros(Shape({4}), DType::FP16);

    uint16_t* pa = a.data_ptr<uint16_t>();
    uint16_t* pb = b.data_ptr<uint16_t>();
    pa[0] = float_to_fp16_bits(1.0f); pa[1] = float_to_fp16_bits(2.0f); pa[2] = float_to_fp16_bits(3.0f);
    pa[3] = float_to_fp16_bits(4.0f);
    pb[0] = float_to_fp16_bits(2.0f); pb[1] = float_to_fp16_bits(3.0f); pb[2] = float_to_fp16_bits(4.0f);
    pb[3] = float_to_fp16_bits(5.0f);

    backend.elementwise_mul(a, b, output);

    const uint16_t* out = output.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(out[0]) - 2.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(out[1]) - 6.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(out[2]) - 12.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(out[3]) - 20.0f) < 0.05f);
    std::cout << "test_elementwise_mul PASSED\n";
}

void test_add() {
    CpuBackend backend;

    Tensor a = Tensor::zeros(Shape({4}), DType::FP16);
    Tensor b = Tensor::zeros(Shape({4}), DType::FP16);
    Tensor output = Tensor::zeros(Shape({4}), DType::FP16);

    uint16_t* pa = a.data_ptr<uint16_t>();
    uint16_t* pb = b.data_ptr<uint16_t>();
    pa[0] = float_to_fp16_bits(1.0f); pa[1] = float_to_fp16_bits(2.0f); pa[2] = float_to_fp16_bits(3.0f);
    pa[3] = float_to_fp16_bits(4.0f);
    pb[0] = float_to_fp16_bits(5.0f); pb[1] = float_to_fp16_bits(6.0f); pb[2] = float_to_fp16_bits(7.0f);
    pb[3] = float_to_fp16_bits(8.0f);

    backend.add(a, b, output);

    const uint16_t* out = output.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(out[0]) - 6.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(out[1]) - 8.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(out[2]) - 10.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(out[3]) - 12.0f) < 0.05f);
    std::cout << "test_add PASSED\n";
}

void test_softmax() {
    CpuBackend backend;

    Tensor input = Tensor::zeros(Shape({1, 4}), DType::FP16);
    Tensor output = Tensor::zeros(Shape({1, 4}), DType::FP16);

    uint16_t* in = input.data_ptr<uint16_t>();
    in[0] = float_to_fp16_bits(1.0f);
    in[1] = float_to_fp16_bits(2.0f);
    in[2] = float_to_fp16_bits(3.0f);
    in[3] = float_to_fp16_bits(4.0f);

    backend.softmax(input, output, -1);

    const uint16_t* out = output.data_ptr<uint16_t>();
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float v = fp16_bits_to_float(out[i]);
        sum += v;
        assert(v > 0.0f);
    }
    assert(std::abs(sum - 1.0f) < 2e-2f);
    assert(fp16_bits_to_float(out[3]) > fp16_bits_to_float(out[2]));
    assert(fp16_bits_to_float(out[2]) > fp16_bits_to_float(out[1]));
    assert(fp16_bits_to_float(out[1]) > fp16_bits_to_float(out[0]));
    std::cout << "test_softmax PASSED\n";
}

void test_embedding_lookup() {
    CpuBackend backend;

    Tensor weight = Tensor::zeros(Shape({10, 4}), DType::FP16);
    uint16_t* w = weight.data_ptr<uint16_t>();
    for (int i = 0; i < 40; ++i) w[i] = float_to_fp16_bits(static_cast<float>(i));

    std::vector<int> token_ids = {0, 2, 5};
    Tensor output = Tensor::zeros(Shape({1, 3, 4}), DType::FP16);

    backend.embedding_lookup(weight, token_ids, output);

    const uint16_t* out = output.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(out[0]) - 0.0f) < 1e-5f);
    assert(std::abs(fp16_bits_to_float(out[1]) - 1.0f) < 1e-3f);
    assert(std::abs(fp16_bits_to_float(out[4]) - 8.0f) < 1e-3f);
    assert(std::abs(fp16_bits_to_float(out[8]) - 20.0f) < 1e-3f);
    std::cout << "test_embedding_lookup PASSED\n";
}

int main() {
    test_gemm();
    test_rmsnorm();
    test_silu();
    test_elementwise_mul();
    test_add();
    test_softmax();
    test_embedding_lookup();

    std::cout << "\nAll backend tests passed!\n";
    return 0;
}
