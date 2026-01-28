#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "mruntime/arena.h"
#include "mruntime/qwen2_ops.h"
#include "mruntime/pthreadpool_raii.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace mruntime;

namespace {

constexpr size_t kBenchAlign = 64;
constexpr size_t kCacheLineBytes = 128; // depends on platforms, double-check
constexpr double kBytesPerMiB = 1024.0 * 1024.0;
constexpr double kBytesPerGB = 1e9;

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

double bytes_to_mib(size_t bytes) {
    return static_cast<double>(bytes) / kBytesPerMiB;
}

double bandwidth_gb_s(size_t bytes, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    const double seconds = time_ms / 1e3;
    return (static_cast<double>(bytes) / kBytesPerGB) / seconds;
}

class CacheFlusher {
public:
    explicit CacheFlusher(size_t size_bytes)
        : size_bytes_(round_up(size_bytes, kBenchAlign)),
          buffer_(static_cast<uint8_t*>(aligned_alloc_or_die(size_bytes_))) {
        std::memset(buffer_, 0, size_bytes_);
    }

    ~CacheFlusher() {
        std::free(buffer_);
    }

    void flush() {
        for (size_t i = 0; i < size_bytes_; i += kCacheLineBytes) {
            sink_ ^= buffer_[i];
        }
    }

private:
    size_t size_bytes_;
    uint8_t* buffer_;
    volatile uint8_t sink_ = 0;
};

class IosFormatGuard {
public:
    explicit IosFormatGuard(std::ios& stream)
        : stream_(stream), flags_(stream.flags()), precision_(stream.precision()) {}

    ~IosFormatGuard() {
        stream_.flags(flags_);
        stream_.precision(precision_);
    }

private:
    std::ios& stream_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
};

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
    const int iterations = 1;
    CacheFlusher cache_flusher(256 * 1024 * 1024);

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

    auto benchmark_gemv_ms = [&](size_t N, size_t K, int iters, bool flush_each_iter, PThreadPool* p) -> double {
        constexpr size_t M = 1;
        const size_t A_size = M * K;
        const size_t C_size = M * N;
        const size_t packed_B_bytes = qwen2_packed_weight_size_fp16(N, K);

        uint16_t* A = static_cast<uint16_t*>(aligned_alloc_or_die(A_size * sizeof(uint16_t)));
        uint16_t* packed_B = static_cast<uint16_t*>(aligned_alloc_or_die(packed_B_bytes));
        uint16_t* C = static_cast<uint16_t*>(aligned_alloc_or_die(C_size * sizeof(uint16_t)));

        std::memset(A, 0x42, A_size * sizeof(uint16_t));
        std::memset(packed_B, 0x42, packed_B_bytes);
        std::memset(C, 0, C_size * sizeof(uint16_t));

        // Warm up any one-time setup and threadpool scheduling overhead.
        qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, p);

        double total_ms = 0.0;
        if (!flush_each_iter) {
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, p);
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_ms = elapsed.count();
        } else {
            for (int i = 0; i < iters; ++i) {
                cache_flusher.flush();
                auto start = std::chrono::steady_clock::now();
                qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, p);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                total_ms += elapsed.count();
            }
        }

        std::free(A);
        std::free(packed_B);
        std::free(C);

        return total_ms / iters;
    };

    auto print_gemv_hot_cold = [&](size_t N, size_t K) {
        IosFormatGuard format_guard(std::cout);
        const size_t weight_bytes = N * K * sizeof(uint16_t);
        const size_t packed_bytes = qwen2_packed_weight_size_fp16(N, K);

        const int hot_iters = 100;
        const int cold_iters = 10;

        const double hot_ms = benchmark_gemv_ms(N, K, hot_iters, /*flush_each_iter=*/false, &pool);
        const double cold_ms = benchmark_gemv_ms(N, K, cold_iters, /*flush_each_iter=*/true, &pool);

        const double hot_bw = bandwidth_gb_s(weight_bytes, hot_ms);
        const double cold_bw = bandwidth_gb_s(weight_bytes, cold_ms);

        std::cout << "  " << N << "x" << K
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2) << bytes_to_mib(weight_bytes)
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2) << bytes_to_mib(packed_bytes)
                  << " | " << std::setw(7) << std::fixed << std::setprecision(3) << hot_ms
                  << " | " << std::setw(7) << std::fixed << std::setprecision(3) << cold_ms
                  << " | " << std::setw(7) << std::fixed << std::setprecision(1) << hot_bw
                  << " | " << std::setw(7) << std::fixed << std::setprecision(1) << cold_bw
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2) << (cold_ms / hot_ms) << "x\n";
    };

    std::cout << "\nGEMV Hot vs Cold (multi-threaded, packed B)\n";
    std::cout << "  N x K       | WeightMiB | PackedMiB | Hot(ms) | Cold(ms) | HotBW | ColdBW | Cold/Hot\n";
    std::cout << "  ------------|-----------|----------|---------|----------|-------|--------|---------\n";
    print_gemv_hot_cold(896, 4864);
    print_gemv_hot_cold(4864, 896);
    print_gemv_hot_cold(2048, 4096);
    print_gemv_hot_cold(4096, 4096);
    print_gemv_hot_cold(4096, 8192);

    std::cout << "\nRMSNorm Benchmarks:\n";
    double rmsnorm_128 = benchmark_rmsnorm(128, 896, iterations, &pool);
    std::cout << "  tokens=128, hidden=896: " << rmsnorm_128 << " ms\n";

    double rmsnorm_512 = benchmark_rmsnorm(512, 896, iterations, &pool);
    std::cout << "  tokens=512, hidden=896: " << rmsnorm_512 << " ms\n";

    // =========================================================================
    // Memory Bandwidth Benchmark
    // =========================================================================
    std::cout << "\n=== Memory Bandwidth Benchmark ===\n";

    auto benchmark_memread = [&](size_t size_mib, int num_threads) -> double {
        const size_t size_bytes = size_mib * 1024 * 1024;
        const size_t num_elements = size_bytes / sizeof(uint16_t);

        uint16_t* data = static_cast<uint16_t*>(aligned_alloc_or_die(size_bytes));

        // Touch all pages to ensure allocation
        std::memset(data, 0x42, size_bytes);
        cache_flusher.flush();

        const int warmup_iters = 2;
        const int bench_iters = 5;

        auto do_read = [&]() {
#if defined(__aarch64__)
            if (num_threads == 1) {
                // Single-threaded NEON read
                float16x8_t acc0 = vdupq_n_f16(0);
                float16x8_t acc1 = vdupq_n_f16(0);
                float16x8_t acc2 = vdupq_n_f16(0);
                float16x8_t acc3 = vdupq_n_f16(0);
                const __fp16* ptr = reinterpret_cast<const __fp16*>(data);
                size_t i = 0;
                for (; i + 32 <= num_elements; i += 32) {
                    acc0 = vaddq_f16(acc0, vld1q_f16(ptr + i));
                    acc1 = vaddq_f16(acc1, vld1q_f16(ptr + i + 8));
                    acc2 = vaddq_f16(acc2, vld1q_f16(ptr + i + 16));
                    acc3 = vaddq_f16(acc3, vld1q_f16(ptr + i + 24));
                }
                for (; i + 8 <= num_elements; i += 8) {
                    acc0 = vaddq_f16(acc0, vld1q_f16(ptr + i));
                }
                acc0 = vaddq_f16(acc0, acc1);
                acc2 = vaddq_f16(acc2, acc3);
                acc0 = vaddq_f16(acc0, acc2);
                // Prevent optimization
                volatile __fp16 s = vgetq_lane_f16(acc0, 0);
                (void)s;
            } else {
                // Multi-threaded read
                auto worker = [&](size_t tid) {
                    const size_t chunk = num_elements / num_threads;
                    const size_t start = tid * chunk;
                    const size_t end = (tid == static_cast<size_t>(num_threads - 1)) ? num_elements : start + chunk;

                    float16x8_t acc0 = vdupq_n_f16(0);
                    float16x8_t acc1 = vdupq_n_f16(0);
                    float16x8_t acc2 = vdupq_n_f16(0);
                    float16x8_t acc3 = vdupq_n_f16(0);
                    const __fp16* ptr = reinterpret_cast<const __fp16*>(data);
                    size_t i = start;
                    for (; i + 32 <= end; i += 32) {
                        acc0 = vaddq_f16(acc0, vld1q_f16(ptr + i));
                        acc1 = vaddq_f16(acc1, vld1q_f16(ptr + i + 8));
                        acc2 = vaddq_f16(acc2, vld1q_f16(ptr + i + 16));
                        acc3 = vaddq_f16(acc3, vld1q_f16(ptr + i + 24));
                    }
                    for (; i + 8 <= end; i += 8) {
                        acc0 = vaddq_f16(acc0, vld1q_f16(ptr + i));
                    }
                    acc0 = vaddq_f16(acc0, acc1);
                    acc2 = vaddq_f16(acc2, acc3);
                    acc0 = vaddq_f16(acc0, acc2);
                    volatile __fp16 s = vgetq_lane_f16(acc0, 0);
                    (void)s;
                };
                pool.parallelize_1d(num_threads, worker);
            }
#else
            // Scalar fallback
            volatile uint64_t acc = 0;
            const uint64_t* ptr = reinterpret_cast<const uint64_t*>(data);
            for (size_t i = 0; i < size_bytes / sizeof(uint64_t); ++i) {
                acc += ptr[i];
            }
            (void)acc;
#endif
        };

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            do_read();
        }

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
            do_read();
        }
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        double time_ms = elapsed.count() / bench_iters;
        double bandwidth_gbps = (size_bytes / 1e9) / (time_ms / 1e3);

        std::free(data);
        return bandwidth_gbps;
    };

    std::cout << "Sequential read (NEON vld1q_f16):\n";
    std::cout << "  Size (MiB) | 1 thread | 8 threads\n";
    std::cout << "  ----------|----------|----------\n";

    struct ReadBw {
        size_t size_mib = 0;
        double bw_1t_gb_s = 0.0;
        double bw_8t_gb_s = 0.0;
    };
    std::vector<ReadBw> read_bw_results;

    for (size_t size_mib : {32, 64, 128, 256, 512}) {
        double bw_1t = benchmark_memread(size_mib, 1);
        double bw_8t = benchmark_memread(size_mib, 8);
        read_bw_results.push_back(ReadBw{size_mib, bw_1t, bw_8t});
        std::cout << "  " << size_mib << " MiB    | " << bw_1t << " GB/s | " << bw_8t << " GB/s\n";
    }

    // =========================================================================
    // GEMV vs Pure Read Comparison (with cache flush)
    // =========================================================================
    std::cout << "\n=== GEMV vs Pure Read (cold cache) ===\n";

    auto read_bw_8t_gb_s_for_size = [&](size_t size_mib) -> double {
        for (const auto& r : read_bw_results) {
            if (r.size_mib == size_mib) return r.bw_8t_gb_s;
        }
        return 0.0;
    };

    auto benchmark_gemv_cold = [&](size_t N, size_t K, PThreadPool* p) -> double {
        constexpr size_t M = 1;
        const size_t A_size = M * K;
        const size_t C_size = M * N;
        const size_t packed_B_bytes = qwen2_packed_weight_size_fp16(N, K);

        uint16_t* A = static_cast<uint16_t*>(aligned_alloc_or_die(A_size * sizeof(uint16_t)));
        uint16_t* packed_B = static_cast<uint16_t*>(aligned_alloc_or_die(packed_B_bytes));
        uint16_t* C = static_cast<uint16_t*>(aligned_alloc_or_die(C_size * sizeof(uint16_t)));

        std::memset(A, 0x42, A_size * sizeof(uint16_t));
        std::memset(packed_B, 0x42, packed_B_bytes);
        std::memset(C, 0, C_size * sizeof(uint16_t));

        const int iters = 5;
        double total_ms = 0;

        // Warm up any one-time setup and threadpool scheduling overhead.
        qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, p);

        for (int i = 0; i < iters; ++i) {
            cache_flusher.flush();

            auto start = std::chrono::steady_clock::now();
            qwen2_gemm_fp16(A, nullptr, C, M, N, K, packed_B, p);
            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_ms += elapsed.count();
        }

        std::free(A);
        std::free(packed_B);
        std::free(C);

        return total_ms / iters;
    };

    std::cout << "  N x K       | Weights(MiB) | GEMV (ms) | Eff BW (GB/s) | Pure Read (GB/s)\n";
    std::cout << "  ------------|-------------|-----------|---------------|----------------\n";

    // 32 MiB weights
    {
        IosFormatGuard format_guard(std::cout);
        double gemv_ms = benchmark_gemv_cold(4096, 4096, &pool);
        const size_t weight_bytes = 4096ULL * 4096ULL * sizeof(uint16_t);
        const double eff_bw = bandwidth_gb_s(weight_bytes, gemv_ms);
        const double pure_read_bw = read_bw_8t_gb_s_for_size(32);
        std::cout << "  4096x4096   | " << std::setw(11) << std::fixed << std::setprecision(1) << bytes_to_mib(weight_bytes)
                  << " | " << std::setw(9) << std::fixed << std::setprecision(3) << gemv_ms
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1) << eff_bw
                  << " | " << std::setw(14) << std::fixed << std::setprecision(1) << pure_read_bw << "\n";
    }

    // 64 MiB weights
    {
        IosFormatGuard format_guard(std::cout);
        double gemv_ms = benchmark_gemv_cold(4096, 8192, &pool);
        const size_t weight_bytes = 4096ULL * 8192ULL * sizeof(uint16_t);
        const double eff_bw = bandwidth_gb_s(weight_bytes, gemv_ms);
        const double pure_read_bw = read_bw_8t_gb_s_for_size(64);
        std::cout << "  4096x8192   | " << std::setw(11) << std::fixed << std::setprecision(1) << bytes_to_mib(weight_bytes)
                  << " | " << std::setw(9) << std::fixed << std::setprecision(3) << gemv_ms
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1) << eff_bw
                  << " | " << std::setw(14) << std::fixed << std::setprecision(1) << pure_read_bw << "\n";
    }

    // 128 MiB weights
    {
        IosFormatGuard format_guard(std::cout);
        double gemv_ms = benchmark_gemv_cold(8192, 8192, &pool);
        const size_t weight_bytes = 8192ULL * 8192ULL * sizeof(uint16_t);
        const double eff_bw = bandwidth_gb_s(weight_bytes, gemv_ms);
        const double pure_read_bw = read_bw_8t_gb_s_for_size(128);
        std::cout << "  8192x8192   | " << std::setw(11) << std::fixed << std::setprecision(1) << bytes_to_mib(weight_bytes)
                  << " | " << std::setw(9) << std::fixed << std::setprecision(3) << gemv_ms
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1) << eff_bw
                  << " | " << std::setw(14) << std::fixed << std::setprecision(1) << pure_read_bw << "\n";
    }

    // =========================================================================
    // Simulated E2E Pattern: Multiple different weight matrices
    // =========================================================================
    std::cout << "\n=== Simulated E2E Pattern (no weight reuse) ===\n";

    auto benchmark_fused_mlp_pattern = [&](size_t hidden_size, size_t intermediate_size, size_t num_layers, bool flush_between, PThreadPool* p) -> double {
        constexpr size_t M = 1;

        // gate_up: [2*intermediate, hidden]
        const size_t gate_up_N = intermediate_size * 2;
        const size_t gate_up_K = hidden_size;
        const size_t gate_up_A_size = M * gate_up_K;
        const size_t gate_up_C_size = M * gate_up_N;
        const size_t gate_up_packed_B_bytes = qwen2_packed_weight_size_fp16(gate_up_N, gate_up_K);

        // down: [hidden, intermediate]
        const size_t down_N = hidden_size;
        const size_t down_K = intermediate_size;
        const size_t down_A_size = M * down_K;
        const size_t down_C_size = M * down_N;
        const size_t down_packed_B_bytes = qwen2_packed_weight_size_fp16(down_N, down_K);

        uint16_t* gate_up_A = static_cast<uint16_t*>(aligned_alloc_or_die(gate_up_A_size * sizeof(uint16_t)));
        uint16_t* gate_up_C = static_cast<uint16_t*>(aligned_alloc_or_die(gate_up_C_size * sizeof(uint16_t)));
        uint16_t* down_A = static_cast<uint16_t*>(aligned_alloc_or_die(down_A_size * sizeof(uint16_t)));
        uint16_t* down_C = static_cast<uint16_t*>(aligned_alloc_or_die(down_C_size * sizeof(uint16_t)));

        std::vector<uint16_t*> gate_up_packed_Bs(num_layers);
        std::vector<uint16_t*> down_packed_Bs(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            gate_up_packed_Bs[i] = static_cast<uint16_t*>(aligned_alloc_or_die(gate_up_packed_B_bytes));
            down_packed_Bs[i] = static_cast<uint16_t*>(aligned_alloc_or_die(down_packed_B_bytes));
            std::memset(gate_up_packed_Bs[i], static_cast<int>(0x42 + i), gate_up_packed_B_bytes);
            std::memset(down_packed_Bs[i], static_cast<int>(0xC0 + i), down_packed_B_bytes);
        }

        std::memset(gate_up_A, 0x42, gate_up_A_size * sizeof(uint16_t));
        std::memset(gate_up_C, 0, gate_up_C_size * sizeof(uint16_t));
        std::memset(down_A, 0x42, down_A_size * sizeof(uint16_t));
        std::memset(down_C, 0, down_C_size * sizeof(uint16_t));

        // Initial cache flush
        cache_flusher.flush();

        const int iters = 5;
        double total_ms = 0;

        for (int iter = 0; iter < iters; ++iter) {
            if (flush_between) {
                cache_flusher.flush();
            }

            auto start = std::chrono::steady_clock::now();
            for (size_t layer = 0; layer < num_layers; ++layer) {
                qwen2_gemm_fp16(gate_up_A, nullptr, gate_up_C, M, gate_up_N, gate_up_K, gate_up_packed_Bs[layer], p);
                qwen2_gemm_fp16(down_A, nullptr, down_C, M, down_N, down_K, down_packed_Bs[layer], p);
            }
            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_ms += elapsed.count();
        }

        const size_t total_gemvs = num_layers * 2;
        const double time_per_gemv = total_ms / (iters * total_gemvs);

        std::free(gate_up_A);
        std::free(gate_up_C);
        std::free(down_A);
        std::free(down_C);
        for (auto* b : gate_up_packed_Bs) std::free(b);
        for (auto* b : down_packed_Bs) std::free(b);

        return time_per_gemv;
    };

    // Simulate fused MLP: 24 layers × 2 ops (gate_up + down) = 48 different weight matrices.
    const size_t hidden_size = 896;
    const size_t intermediate_size = 4864;
    const size_t gate_up_weight_bytes = (intermediate_size * 2) * hidden_size * sizeof(uint16_t);
    const size_t down_weight_bytes = hidden_size * intermediate_size * sizeof(uint16_t);
    const size_t avg_weight_bytes_per_gemv = (gate_up_weight_bytes + down_weight_bytes) / 2;

    {
        double time_ms = benchmark_fused_mlp_pattern(hidden_size, intermediate_size, 24, true, &pool);
        double bw = bandwidth_gb_s(avg_weight_bytes_per_gemv, time_ms);
        std::cout << "  48 matrices (fused MLP), flush between sweeps (cold start):\n";
        std::cout << "    Per-GEMV: " << time_ms << " ms, BW: " << bw << " GB/s\n";
    }

    {
        double time_ms = benchmark_fused_mlp_pattern(hidden_size, intermediate_size, 24, false, &pool);
        double bw = bandwidth_gb_s(avg_weight_bytes_per_gemv, time_ms);
        std::cout << "  48 matrices (fused MLP), no flush between sweeps (steady-state):\n";
        std::cout << "    Per-GEMV: " << time_ms << " ms, BW: " << bw << " GB/s\n";
    }

    {
        std::cout << "  Single matrix (cold cache baseline):\n";
        {
            double time_ms = benchmark_gemv_cold(intermediate_size * 2, hidden_size, &pool);
            double bw = bandwidth_gb_s(gate_up_weight_bytes, time_ms);
            std::cout << "    gate_up (" << (intermediate_size * 2) << "x" << hidden_size << "): "
                      << time_ms << " ms, BW: " << bw << " GB/s\n";
        }
        {
            double time_ms = benchmark_gemv_cold(hidden_size, intermediate_size, &pool);
            double bw = bandwidth_gb_s(down_weight_bytes, time_ms);
            std::cout << "    down    (" << hidden_size << "x" << intermediate_size << "): "
                      << time_ms << " ms, BW: " << bw << " GB/s\n";
        }
    }

    return 0;
}
