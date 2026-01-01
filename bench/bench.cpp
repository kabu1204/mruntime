#include <chrono>
#include <iostream>

#include "mruntime/cpu_backend.h"
#include "mruntime/tensor.h"

using namespace mruntime;

double benchmark_gemm(CpuBackend& backend, size_t M, size_t N, size_t K, int iterations) {
    Tensor A = Tensor::zeros(Shape({M, K}), DType::FP32);
    Tensor B = Tensor::zeros(Shape({K, N}), DType::FP32);
    Tensor C = Tensor::zeros(Shape({M, N}), DType::FP32);

    backend.gemm(A, B, C);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        backend.gemm(A, B, C);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

double benchmark_rmsnorm(CpuBackend& backend, size_t batch, size_t seq_len, size_t hidden_size, int iterations) {
    Tensor input = Tensor::zeros(Shape({batch, seq_len, hidden_size}), DType::FP32);
    Tensor weight = Tensor::zeros(Shape({hidden_size}), DType::FP32);
    Tensor output = Tensor::zeros(Shape({batch, seq_len, hidden_size}), DType::FP32);

    backend.rmsnorm(input, weight, output, 1e-6f);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        backend.rmsnorm(input, weight, output, 1e-6f);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

int main() {
    CpuBackend backend;
    const int iterations = 10;

    std::cout << "mruntime Performance Benchmarks\n";
    std::cout << "================================\n\n";

    std::cout << "GEMM Benchmarks:\n";
    double gemm_512 = benchmark_gemm(backend, 512, 512, 512, iterations);
    std::cout << "  512x512x512: " << gemm_512 << " ms\n";

    double gemm_1024 = benchmark_gemm(backend, 1024, 1024, 1024, iterations);
    std::cout << "  1024x1024x1024: " << gemm_1024 << " ms\n";

    std::cout << "\nRMSNorm Benchmarks:\n";
    double rmsnorm_128 = benchmark_rmsnorm(backend, 1, 128, 896, iterations);
    std::cout << "  batch=1, seq=128, hidden=896: " << rmsnorm_128 << " ms\n";

    double rmsnorm_512 = benchmark_rmsnorm(backend, 1, 512, 896, iterations);
    std::cout << "  batch=1, seq=512, hidden=896: " << rmsnorm_512 << " ms\n";

    return 0;
}
