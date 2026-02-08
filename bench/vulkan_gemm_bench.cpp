#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mruntime/dtype.h"
#include "vk_buffer_arena.h"
#include "vk_context.h"
#include "vk_fp16_ops.h"
#include "vk_helpers.h"
#include "vk_kernel_runtime.h"

using namespace mruntime::vulkan;

namespace {

struct BenchContext {
    VkContext context;
    VkKernelRuntime runtime;
    VkFp16Ops fp16_ops;
    VkDeviceSize alignment;

    static BenchContext Create() {
        BenchContext bc;
        bc.context = VkContext::Create();
        bc.alignment = std::max<VkDeviceSize>(
            64, bc.context.min_storage_buffer_offset_alignment());
        bc.runtime = VkKernelRuntime::Create(bc.context);
        bc.fp16_ops = VkFp16Ops::Create(&bc.runtime);
        return bc;
    }
};

VkBufferArena make_arena(const BenchContext& bc, VkDeviceSize capacity) {
    VkBufferArenaCreateInfo info;
    info.capacity_bytes = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    info.default_alignment = bc.alignment;
    return VkBufferArena::Create(
        bc.context.physical_device(), bc.context.device(), info);
}

double bandwidth_gb_s(size_t bytes, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / 1e9) / (time_ms / 1e3);
}

double gflops_gemm(uint32_t M, uint32_t N, uint32_t K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double flops = 2.0 * static_cast<double>(M) *
                   static_cast<double>(N) * static_cast<double>(K);
    return flops / (time_ms * 1e6);
}

double benchmark_gemm(BenchContext& bc, uint32_t M, uint32_t N, uint32_t K,
                      int warmup_iters, int bench_iters) {
    const VkDeviceSize a_bytes = M * K * sizeof(uint16_t);
    const VkDeviceSize b_bytes = N * K * sizeof(uint16_t);
    const VkDeviceSize c_bytes = M * N * sizeof(uint16_t);

    auto arena = make_arena(bc, a_bytes + b_bytes + c_bytes + 3 * bc.alignment);

    const VkDeviceSize a_offset = arena.alloc(a_bytes);
    const VkDeviceSize b_offset = arena.alloc(b_bytes);
    const VkDeviceSize c_offset = arena.alloc(c_bytes);

    // Fill A and B with fp16 1.0 (0x3C00).
    uint16_t* a_data = arena.host_ptr<uint16_t>(a_offset);
    uint16_t* b_data = arena.host_ptr<uint16_t>(b_offset);
    const uint16_t one_fp16 = 0x3C00;
    for (uint32_t i = 0; i < M * K; ++i) a_data[i] = one_fp16;
    for (uint32_t i = 0; i < N * K; ++i) b_data[i] = one_fp16;

    auto a_desc = arena.descriptor(a_offset, a_bytes);
    auto b_desc = arena.descriptor(b_offset, b_bytes);
    auto c_desc = arena.descriptor(c_offset, c_bytes);

    for (int i = 0; i < warmup_iters; ++i) {
        bc.fp16_ops.gemm(a_desc, b_desc, c_desc, M, N, K);
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        bc.fp16_ops.gemm(a_desc, b_desc, c_desc, M, N, K);
    }
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / bench_iters;
}

struct GemmConfig {
    uint32_t M, N, K;
};

void print_result(const GemmConfig& cfg, double time_ms) {
    double gflops = gflops_gemm(cfg.M, cfg.N, cfg.K, time_ms);
    std::cout << "  " << cfg.M << "x" << cfg.N << "x" << cfg.K << ":"
              << std::string(
                     std::max(1, 20 - static_cast<int>(
                         std::to_string(cfg.M).size() +
                         std::to_string(cfg.N).size() +
                         std::to_string(cfg.K).size() + 2)), ' ')
              << std::fixed << std::setprecision(3) << time_ms << " ms ("
              << std::setprecision(1) << gflops << " GFLOP/s)\n";
}

void run_benchmark() {
    BenchContext bc = BenchContext::Create();

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(bc.context.physical_device(), &properties);

    std::cout << "Vulkan FP16 GEMM Benchmark\n";
    std::cout << "===========================\n";
    std::cout << "Device: " << properties.deviceName << "\n";

    constexpr int warmup = 3;
    constexpr int iters = 10;

    std::cout << "\nSquare GEMM:\n";
    std::vector<GemmConfig> square_configs = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };
    for (const auto& cfg : square_configs) {
        double ms = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, ms);
    }

    std::cout << "\nModel shapes (Qwen2-0.5B):\n";
    std::vector<GemmConfig> model_configs = {
        {8, 1152, 896},
        {8, 9728, 896},
        {8, 896, 4864},
    };
    for (const auto& cfg : model_configs) {
        double ms = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, ms);
    }

    std::cout << "\nDecode shapes:\n";
    std::vector<GemmConfig> decode_configs = {
        {1, 896, 896},
        {1, 9728, 896},
    };
    for (const auto& cfg : decode_configs) {
        double ms = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, ms);
    }

    // GEMV: M=1, memory-bandwidth-bound regime.
    // Weight matrix B is N*K fp16 elements; report effective bandwidth.
    std::cout << "\nGEMV (M=1):\n";
    std::cout << "  N x K          | Weight(MiB) |  Time(ms) | GFLOP/s |   BW(GB/s)\n";
    std::cout << "  ---------------|-------------|-----------|---------|----------\n";

    struct GemvConfig { uint32_t N, K; };
    std::vector<GemvConfig> gemv_configs = {
        {896, 896},
        {1152, 896},
        {896, 4864},
        {4864, 896},
        {9728, 896},
        {896, 9728},
        {4096, 4096},
        {4096, 8192},
    };
    for (const auto& gc : gemv_configs) {
        double ms = benchmark_gemm(bc, 1, gc.N, gc.K, warmup, iters);
        double gflops = gflops_gemm(1, gc.N, gc.K, ms);
        size_t weight_bytes = static_cast<size_t>(gc.N) * gc.K * sizeof(uint16_t);
        double bw = bandwidth_gb_s(weight_bytes, ms);
        double weight_mib = static_cast<double>(weight_bytes) / (1024.0 * 1024.0);
        std::cout << "  " << std::setw(5) << gc.N << "x" << std::setw(5) << gc.K
                  << std::setw(5) << "" << "| "
                  << std::setw(11) << std::fixed << std::setprecision(2) << weight_mib << " | "
                  << std::setw(9) << std::setprecision(3) << ms << " | "
                  << std::setw(7) << std::setprecision(1) << gflops << " | "
                  << std::setw(9) << std::setprecision(1) << bw << "\n";
    }
}

}  // namespace

int main() {
    try {
        run_benchmark();
        return 0;
    } catch (const VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_gemm_bench SKIPPED: "
                         "Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_gemm_bench FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_gemm_bench FAILED: " << error.what() << "\n";
        return 1;
    }
}
