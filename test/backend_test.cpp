#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mruntime/cpu_backend.h"
#include "mruntime/tensor.h"

using namespace mruntime;

void test_gemm() {
    CpuBackend backend;

    Tensor A = Tensor::zeros(Shape({2, 3}), DType::FP32);
    Tensor B = Tensor::zeros(Shape({3, 4}), DType::FP32);
    Tensor C = Tensor::zeros(Shape({2, 4}), DType::FP32);

    float* a = A.data_ptr<float>();
    float* b = B.data_ptr<float>();
    for (int i = 0; i < 6; ++i) a[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 12; ++i) b[i] = static_cast<float>(i + 1);

    backend.gemm(A, B, C);

    const float* c = C.data_ptr<float>();
    assert(std::abs(c[0] - 38.0f) < 1e-5f);
    assert(std::abs(c[1] - 44.0f) < 1e-5f);
    std::cout << "test_gemm PASSED\n";
}

void test_rmsnorm() {
    CpuBackend backend;

    Tensor input = Tensor::zeros(Shape({1, 2, 4}), DType::FP32);
    Tensor weight = Tensor::zeros(Shape({4}), DType::FP32);
    Tensor output = Tensor::zeros(Shape({1, 2, 4}), DType::FP32);

    float* in = input.data_ptr<float>();
    float* w = weight.data_ptr<float>();
    for (int i = 0; i < 8; ++i) in[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 4; ++i) w[i] = 1.0f;

    backend.rmsnorm(input, weight, output, 1e-6f);

    const float* out = output.data_ptr<float>();
    float expected_rms = std::sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-6f);
    assert(std::abs(out[0] - 1.0f / expected_rms) < 1e-4f);
    std::cout << "test_rmsnorm PASSED\n";
}

void test_silu() {
    CpuBackend backend;

    Tensor input = Tensor::zeros(Shape({4}), DType::FP32);
    Tensor output = Tensor::zeros(Shape({4}), DType::FP32);

    float* in = input.data_ptr<float>();
    in[0] = 0.0f;
    in[1] = 1.0f;
    in[2] = -1.0f;
    in[3] = 2.0f;

    backend.silu(input, output);

    const float* out = output.data_ptr<float>();
    assert(std::abs(out[0] - 0.0f) < 1e-5f);

    float expected_1 = 1.0f / (1.0f + std::exp(-1.0f));
    assert(std::abs(out[1] - expected_1) < 1e-5f);

    float expected_neg1 = -1.0f / (1.0f + std::exp(1.0f));
    assert(std::abs(out[2] - expected_neg1) < 1e-5f);

    std::cout << "test_silu PASSED\n";
}

void test_elementwise_mul() {
    CpuBackend backend;

    Tensor a = Tensor::zeros(Shape({4}), DType::FP32);
    Tensor b = Tensor::zeros(Shape({4}), DType::FP32);
    Tensor output = Tensor::zeros(Shape({4}), DType::FP32);

    float* pa = a.data_ptr<float>();
    float* pb = b.data_ptr<float>();
    pa[0] = 1.0f; pa[1] = 2.0f; pa[2] = 3.0f; pa[3] = 4.0f;
    pb[0] = 2.0f; pb[1] = 3.0f; pb[2] = 4.0f; pb[3] = 5.0f;

    backend.elementwise_mul(a, b, output);

    const float* out = output.data_ptr<float>();
    assert(out[0] == 2.0f);
    assert(out[1] == 6.0f);
    assert(out[2] == 12.0f);
    assert(out[3] == 20.0f);
    std::cout << "test_elementwise_mul PASSED\n";
}

void test_add() {
    CpuBackend backend;

    Tensor a = Tensor::zeros(Shape({4}), DType::FP32);
    Tensor b = Tensor::zeros(Shape({4}), DType::FP32);
    Tensor output = Tensor::zeros(Shape({4}), DType::FP32);

    float* pa = a.data_ptr<float>();
    float* pb = b.data_ptr<float>();
    pa[0] = 1.0f; pa[1] = 2.0f; pa[2] = 3.0f; pa[3] = 4.0f;
    pb[0] = 5.0f; pb[1] = 6.0f; pb[2] = 7.0f; pb[3] = 8.0f;

    backend.add(a, b, output);

    const float* out = output.data_ptr<float>();
    assert(out[0] == 6.0f);
    assert(out[1] == 8.0f);
    assert(out[2] == 10.0f);
    assert(out[3] == 12.0f);
    std::cout << "test_add PASSED\n";
}

void test_softmax() {
    CpuBackend backend;

    Tensor input = Tensor::zeros(Shape({1, 4}), DType::FP32);
    Tensor output = Tensor::zeros(Shape({1, 4}), DType::FP32);

    float* in = input.data_ptr<float>();
    in[0] = 1.0f; in[1] = 2.0f; in[2] = 3.0f; in[3] = 4.0f;

    backend.softmax(input, output, -1);

    const float* out = output.data_ptr<float>();
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += out[i];
        assert(out[i] > 0.0f);
    }
    assert(std::abs(sum - 1.0f) < 1e-5f);
    assert(out[3] > out[2] && out[2] > out[1] && out[1] > out[0]);
    std::cout << "test_softmax PASSED\n";
}

void test_embedding_lookup() {
    CpuBackend backend;

    Tensor weight = Tensor::zeros(Shape({10, 4}), DType::FP32);
    float* w = weight.data_ptr<float>();
    for (int i = 0; i < 40; ++i) w[i] = static_cast<float>(i);

    std::vector<int> token_ids = {0, 2, 5};
    Tensor output = Tensor::zeros(Shape({1, 3, 4}), DType::FP32);

    backend.embedding_lookup(weight, token_ids, output);

    const float* out = output.data_ptr<float>();
    assert(out[0] == 0.0f);
    assert(out[1] == 1.0f);
    assert(out[4] == 8.0f);
    assert(out[8] == 20.0f);
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
