#include <chrono>
#include <cstdlib>
#include <iostream>

#include "mruntime/arena.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/pthreadpool_raii.h"

using namespace mruntime;

double benchmark_gemm(size_t M, size_t N, size_t K, int iterations, PThreadPool* pool) {
    size_t A_size = M * K;
    size_t B_size = K * N;
    size_t C_size = M * N;

    uint16_t* A = static_cast<uint16_t*>(std::aligned_alloc(64, A_size * sizeof(uint16_t)));
    uint16_t* B = static_cast<uint16_t*>(std::aligned_alloc(64, B_size * sizeof(uint16_t)));
    uint16_t* C = static_cast<uint16_t*>(std::aligned_alloc(64, C_size * sizeof(uint16_t)));

    // Initialize to zero
    std::memset(A, 0, A_size * sizeof(uint16_t));
    std::memset(B, 0, B_size * sizeof(uint16_t));
    std::memset(C, 0, C_size * sizeof(uint16_t));

    // Warmup
    qwen2_gemm_fp16(A, B, C, M, N, K, nullptr, pool);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        qwen2_gemm_fp16(A, B, C, M, N, K, nullptr, pool);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::free(A);
    std::free(B);
    std::free(C);

    return elapsed.count() / iterations;
}

double benchmark_rmsnorm(size_t num_tokens, size_t hidden_size, int iterations, PThreadPool* pool) {
    size_t input_size = num_tokens * hidden_size;

    uint16_t* input = static_cast<uint16_t*>(std::aligned_alloc(64, input_size * sizeof(uint16_t)));
    uint16_t* weight = static_cast<uint16_t*>(std::aligned_alloc(64, hidden_size * sizeof(uint16_t)));
    uint16_t* output = static_cast<uint16_t*>(std::aligned_alloc(64, input_size * sizeof(uint16_t)));

    std::memset(input, 0, input_size * sizeof(uint16_t));
    std::memset(weight, 0, hidden_size * sizeof(uint16_t));
    std::memset(output, 0, input_size * sizeof(uint16_t));

    // Warmup
    qwen2_rmsnorm_fp16(input, weight, output, num_tokens, hidden_size, 1e-6f, pool);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        qwen2_rmsnorm_fp16(input, weight, output, num_tokens, hidden_size, 1e-6f, pool);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::free(input);
    std::free(weight);
    std::free(output);

    return elapsed.count() / iterations;
}

int main() {
    PThreadPool pool = PThreadPool::Create(0);  // Use all cores
    const int iterations = 10;

    std::cout << "mruntime Performance Benchmarks (bare-metal API)\n";
    std::cout << "=================================================\n\n";

    std::cout << "GEMM Benchmarks:\n";
    double gemm_512 = benchmark_gemm(512, 512, 512, iterations, &pool);
    std::cout << "  512x512x512: " << gemm_512 << " ms\n";

    double gemm_1024 = benchmark_gemm(1024, 1024, 1024, iterations, &pool);
    std::cout << "  1024x1024x1024: " << gemm_1024 << " ms\n";

    std::cout << "\nRMSNorm Benchmarks:\n";
    double rmsnorm_128 = benchmark_rmsnorm(128, 896, iterations, &pool);
    std::cout << "  tokens=128, hidden=896: " << rmsnorm_128 << " ms\n";

    double rmsnorm_512 = benchmark_rmsnorm(512, 896, iterations, &pool);
    std::cout << "  tokens=512, hidden=896: " << rmsnorm_512 << " ms\n";

    return 0;
}
