#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "mruntime/arena.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/pthreadpool_raii.h"

using namespace mruntime;

namespace {

constexpr size_t kBenchAlign = 64;

size_t round_up(size_t n, size_t align) {
    return (n + align - 1) / align * align;
}

void* aligned_alloc_or_die(size_t bytes) {
    bytes = round_up(bytes, kBenchAlign);
    void* ptr = std::aligned_alloc(kBenchAlign, bytes);
    if (ptr == nullptr) {
        std::cerr << "aligned_alloc(" << bytes << ") failed\n";
        std::abort();
    }
    return ptr;
}

double gflops_matmul(size_t M, size_t N, size_t K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return flops / (time_ms * 1e6);
}

}  // namespace

double benchmark_gemm_packed(size_t M, size_t N, size_t K, int iterations, PThreadPool* pool) {
    size_t A_size = M * K;
    size_t C_size = M * N;
    size_t packed_B_bytes = qwen2_packed_weight_size_fp16(N, K);

    uint16_t* A = static_cast<uint16_t*>(aligned_alloc_or_die(A_size * sizeof(uint16_t)));
    uint16_t* packed_B = static_cast<uint16_t*>(aligned_alloc_or_die(packed_B_bytes));
    uint16_t* C = static_cast<uint16_t*>(aligned_alloc_or_die(C_size * sizeof(uint16_t)));

    // Initialize to zero
    std::memset(A, 0, A_size * sizeof(uint16_t));
    std::memset(packed_B, 0, packed_B_bytes);
    std::memset(C, 0, C_size * sizeof(uint16_t));

    // Warmup
    qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, pool);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, pool);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::free(A);
    std::free(packed_B);
    std::free(C);

    return elapsed.count() / iterations;
}

double benchmark_gemv_packed(size_t N, size_t K, int iterations, PThreadPool* pool) {
    constexpr size_t M = 1;
    const size_t A_size = M * K;
    const size_t C_size = M * N;
    const size_t packed_B_bytes = qwen2_packed_weight_size_fp16(N, K);

    uint16_t* A = static_cast<uint16_t*>(aligned_alloc_or_die(A_size * sizeof(uint16_t)));
    uint16_t* packed_B = static_cast<uint16_t*>(aligned_alloc_or_die(packed_B_bytes));
    uint16_t* C = static_cast<uint16_t*>(aligned_alloc_or_die(C_size * sizeof(uint16_t)));

    std::memset(A, 0, A_size * sizeof(uint16_t));
    std::memset(packed_B, 0, packed_B_bytes);
    std::memset(C, 0, C_size * sizeof(uint16_t));

    qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, pool);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, pool);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::free(A);
    std::free(packed_B);
    std::free(C);

    return elapsed.count() / iterations;
}

double benchmark_gemm_single_thread(size_t M, size_t N, size_t K, int iterations, PThreadPool* pool) {
    size_t A_size = M * K;
    size_t B_size = K * N;
    size_t C_size = M * N;

    uint16_t* A = static_cast<uint16_t*>(aligned_alloc_or_die(A_size * sizeof(uint16_t)));
    uint16_t* B = static_cast<uint16_t*>(aligned_alloc_or_die(B_size * sizeof(uint16_t)));
    uint16_t* C = static_cast<uint16_t*>(aligned_alloc_or_die(C_size * sizeof(uint16_t)));

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

    uint16_t* input = static_cast<uint16_t*>(aligned_alloc_or_die(input_size * sizeof(uint16_t)));
    uint16_t* weight = static_cast<uint16_t*>(aligned_alloc_or_die(hidden_size * sizeof(uint16_t)));
    uint16_t* output = static_cast<uint16_t*>(aligned_alloc_or_die(input_size * sizeof(uint16_t)));

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
    PThreadPool pool = PThreadPool::Create(8);
    const int iterations = 10;
    const int gemv_iterations = 10000;

    std::cout << "mruntime Performance Benchmarks (bare-metal API)\n";
    std::cout << "=================================================\n\n";

    std::cout << "GEMM Benchmarks (single-threaded):\n";
    double gemm_512 = benchmark_gemm_single_thread(512, 512, 512, iterations, &pool);
    std::cout << "  512x512x512: " << gemm_512 << " ms (" << gflops_matmul(512, 512, 512, gemm_512) << " GFLOP/s)\n";

    double gemm_1024 = benchmark_gemm_single_thread(1024, 1024, 1024, iterations, &pool);
    std::cout << "  1024x1024x1024: " << gemm_1024 << " ms (" << gflops_matmul(1024, 1024, 1024, gemm_1024) << " GFLOP/s)\n";

    double gemm_4096 = benchmark_gemm_single_thread(4096, 4096, 4096, iterations, &pool);
    std::cout << "  4096x4096x4096: " << gemm_4096 << " ms (" << gflops_matmul(4096, 4096, 4096, gemm_4096) << " GFLOP/s)\n";

    std::cout << "\nGEMM Benchmarks (single-threaded, packed B):\n";
    gemm_512 = benchmark_gemm_packed(512, 512, 512, iterations, nullptr);
    std::cout << "  512x512x512: " << gemm_512 << " ms (" << gflops_matmul(512, 512, 512, gemm_512) << " GFLOP/s)\n";

    gemm_1024 = benchmark_gemm_packed(1024, 1024, 1024, iterations, nullptr);
    std::cout << "  1024x1024x1024: " << gemm_1024 << " ms (" << gflops_matmul(1024, 1024, 1024, gemm_1024) << " GFLOP/s)\n";

    gemm_4096 = benchmark_gemm_packed(4096, 4096, 4096, iterations, nullptr);
    std::cout << "  4096x4096x4096: " << gemm_4096 << " ms (" << gflops_matmul(4096, 4096, 4096, gemm_4096) << " GFLOP/s)\n";

    std::cout << "\nGEMM Benchmarks (multi-threaded, packed B):\n";
    gemm_512 = benchmark_gemm_packed(512, 512, 512, iterations, &pool);
    std::cout << "  512x512x512: " << gemm_512 << " ms (" << gflops_matmul(512, 512, 512, gemm_512) << " GFLOP/s)\n";

    gemm_1024 = benchmark_gemm_packed(1024, 1024, 1024, iterations, &pool);
    std::cout << "  1024x1024x1024: " << gemm_1024 << " ms (" << gflops_matmul(1024, 1024, 1024, gemm_1024) << " GFLOP/s)\n";

    gemm_4096 = benchmark_gemm_packed(4096, 4096, 4096, iterations, &pool);
    std::cout << "  4096x4096x4096: " << gemm_4096 << " ms (" << gflops_matmul(4096, 4096, 4096, gemm_4096) << " GFLOP/s)\n";

    std::cout << "\nGEMV Benchmarks (single-threaded, packed B):\n";
    double gemv_512 = benchmark_gemv_packed(512, 512, gemv_iterations, nullptr);
    std::cout << "  1x512x512: " << gemv_512 << " ms (" << gflops_matmul(1, 512, 512, gemv_512) << " GFLOP/s)\n";

    double gemv_896 = benchmark_gemv_packed(896, 4864, gemv_iterations, nullptr);
    std::cout << "  1x896x4864: " << gemv_896 << " ms (" << gflops_matmul(1, 896, 4864, gemv_896) << " GFLOP/s)\n";

    double gemv_1024 = benchmark_gemv_packed(1024, 1024, gemv_iterations, nullptr);
    std::cout << "  1x1024x1024: " << gemv_1024 << " ms (" << gflops_matmul(1, 1024, 1024, gemv_1024) << " GFLOP/s)\n";

    std::cout << "\nGEMV Benchmarks (multi-threaded, packed B):\n";
    gemv_512 = benchmark_gemv_packed(512, 512, gemv_iterations, &pool);
    std::cout << "  1x512x512: " << gemv_512 << " ms (" << gflops_matmul(1, 512, 512, gemv_512) << " GFLOP/s)\n";

    gemv_896 = benchmark_gemv_packed(896, 4864, gemv_iterations, &pool);
    std::cout << "  1x896x4864: " << gemv_896 << " ms (" << gflops_matmul(1, 896, 4864, gemv_896) << " GFLOP/s)\n";

    gemv_1024 = benchmark_gemv_packed(1024, 1024, gemv_iterations, &pool);
    std::cout << "  1x1024x1024: " << gemv_1024 << " ms (" << gflops_matmul(1, 1024, 1024, gemv_1024) << " GFLOP/s)\n";

    std::cout << "\nRMSNorm Benchmarks:\n";
    double rmsnorm_128 = benchmark_rmsnorm(128, 896, iterations, &pool);
    std::cout << "  tokens=128, hidden=896: " << rmsnorm_128 << " ms\n";

    double rmsnorm_512 = benchmark_rmsnorm(512, 896, iterations, &pool);
    std::cout << "  tokens=512, hidden=896: " << rmsnorm_512 << " ms\n";

    return 0;
}
