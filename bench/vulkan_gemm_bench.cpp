#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

    // GPU timestamp support.
    VkQueryPool query_pool = VK_NULL_HANDLE;
    float timestamp_period_ns = 0.0f;  // nanoseconds per tick

    static BenchContext Create() {
        BenchContext bc;
        bc.context = VkContext::Create();
        bc.alignment = std::max<VkDeviceSize>(
            64, bc.context.min_storage_buffer_offset_alignment());
        bc.runtime = VkKernelRuntime::Create(bc.context);
        bc.fp16_ops = VkFp16Ops::Create(&bc.runtime);

        // Check timestamp support on the compute queue family.
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            bc.context.physical_device(), &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            bc.context.physical_device(), &qf_count, qf_props.data());

        uint32_t qfi = bc.context.queue_family_index();
        if (qfi < qf_count && qf_props[qfi].timestampValidBits > 0) {
            VkPhysicalDeviceProperties props = {};
            vkGetPhysicalDeviceProperties(bc.context.physical_device(), &props);
            bc.timestamp_period_ns = props.limits.timestampPeriod;

            VkQueryPoolCreateInfo qp_ci = {};
            qp_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qp_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qp_ci.queryCount = 2;
            vk_check(vkCreateQueryPool(
                bc.context.device(), &qp_ci, nullptr, &bc.query_pool),
                "vkCreateQueryPool");
        }

        return bc;
    }

    ~BenchContext() {
        if (query_pool != VK_NULL_HANDLE && context.device() != VK_NULL_HANDLE) {
            vkDestroyQueryPool(context.device(), query_pool, nullptr);
        }
    }

    BenchContext() = default;
    BenchContext(const BenchContext&) = delete;
    BenchContext& operator=(const BenchContext&) = delete;
    BenchContext(BenchContext&& other) noexcept
        : context(std::move(other.context)),
          runtime(std::move(other.runtime)),
          fp16_ops(std::move(other.fp16_ops)),
          alignment(other.alignment),
          query_pool(std::exchange(other.query_pool, VK_NULL_HANDLE)),
          timestamp_period_ns(other.timestamp_period_ns) {}
    BenchContext& operator=(BenchContext&& other) noexcept {
        if (this != &other) {
            if (query_pool != VK_NULL_HANDLE && context.device() != VK_NULL_HANDLE) {
                vkDestroyQueryPool(context.device(), query_pool, nullptr);
            }
            context = std::move(other.context);
            runtime = std::move(other.runtime);
            fp16_ops = std::move(other.fp16_ops);
            alignment = other.alignment;
            query_pool = std::exchange(other.query_pool, VK_NULL_HANDLE);
            timestamp_period_ns = other.timestamp_period_ns;
        }
        return *this;
    }

    bool has_gpu_timestamps() const { return query_pool != VK_NULL_HANDLE; }

    // Read GPU timestamp delta in milliseconds. Call after a synchronous dispatch.
    double read_gpu_time_ms() const {
        uint64_t timestamps[2] = {};
        vkGetQueryPoolResults(
            context.device(), query_pool,
            0, 2,
            sizeof(timestamps), timestamps,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        double ticks = static_cast<double>(timestamps[1] - timestamps[0]);
        return ticks * static_cast<double>(timestamp_period_ns) / 1e6;
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

struct BenchResult {
    double cpu_ms;
    double gpu_ms;  // 0 if GPU timestamps unavailable
};

BenchResult benchmark_gemm(BenchContext& bc, uint32_t M, uint32_t N, uint32_t K,
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

    VkQueryPool qp = bc.query_pool;  // may be VK_NULL_HANDLE

    for (int i = 0; i < warmup_iters; ++i) {
        bc.fp16_ops.gemm(a_desc, b_desc, c_desc, M, N, K, qp);
    }

    double total_gpu_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        bc.fp16_ops.gemm(a_desc, b_desc, c_desc, M, N, K, qp);
        if (bc.has_gpu_timestamps()) {
            total_gpu_ms += bc.read_gpu_time_ms();
        }
    }
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    BenchResult result;
    result.cpu_ms = elapsed.count() / bench_iters;
    result.gpu_ms = bc.has_gpu_timestamps() ? total_gpu_ms / bench_iters : 0.0;
    return result;
}

BenchResult benchmark_gemv(BenchContext& bc, uint32_t N, uint32_t K, int warmup_iters, int bench_iters) {
    const VkDeviceSize x_bytes = static_cast<VkDeviceSize>(K) * sizeof(uint16_t);
    const VkDeviceSize w_bytes = static_cast<VkDeviceSize>(N) * K * sizeof(uint16_t);
    const VkDeviceSize y_bytes = static_cast<VkDeviceSize>(N) * sizeof(uint16_t);

    auto arena = make_arena(bc, x_bytes + w_bytes + y_bytes + 3 * bc.alignment);

    const VkDeviceSize x_offset = arena.alloc(x_bytes);
    const VkDeviceSize w_offset = arena.alloc(w_bytes);
    const VkDeviceSize y_offset = arena.alloc(y_bytes);

    // Fill x and W with fp16 1.0 (0x3C00).
    uint16_t* x_data = arena.host_ptr<uint16_t>(x_offset);
    uint16_t* w_data = arena.host_ptr<uint16_t>(w_offset);
    const uint16_t one_fp16 = 0x3C00;
    for (uint32_t i = 0; i < K; ++i) x_data[i] = one_fp16;
    for (uint32_t i = 0; i < N * K; ++i) w_data[i] = one_fp16;

    auto x_desc = arena.descriptor(x_offset, x_bytes);
    auto w_desc = arena.descriptor(w_offset, w_bytes);
    auto y_desc = arena.descriptor(y_offset, y_bytes);

    VkQueryPool qp = bc.query_pool;  // may be VK_NULL_HANDLE

    for (int i = 0; i < warmup_iters; ++i) {
        bc.fp16_ops.gemv(x_desc, w_desc, y_desc, N, K, qp);
    }

    double total_gpu_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        bc.fp16_ops.gemv(x_desc, w_desc, y_desc, N, K, qp);
        if (bc.has_gpu_timestamps()) {
            total_gpu_ms += bc.read_gpu_time_ms();
        }
    }
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    BenchResult result;
    result.cpu_ms = elapsed.count() / bench_iters;
    result.gpu_ms = bc.has_gpu_timestamps() ? total_gpu_ms / bench_iters : 0.0;
    return result;
}

struct GemmConfig {
    uint32_t M, N, K;
};

void print_result(const GemmConfig& cfg, const BenchResult& r) {
    auto pad = std::string(
        std::max(1, 20 - static_cast<int>(
            std::to_string(cfg.M).size() +
            std::to_string(cfg.N).size() +
            std::to_string(cfg.K).size() + 2)), ' ');
    if (r.gpu_ms > 0.0) {
        double gpu_gflops = gflops_gemm(cfg.M, cfg.N, cfg.K, r.gpu_ms);
        double cpu_gflops = gflops_gemm(cfg.M, cfg.N, cfg.K, r.cpu_ms);
        std::cout << "  " << cfg.M << "x" << cfg.N << "x" << cfg.K << ":" << pad
                  << std::fixed << std::setprecision(3)
                  << "kernel " << r.gpu_ms << " ms (" << std::setprecision(1) << gpu_gflops << " GFLOP/s)"
                  << "  submit+wait " << std::setprecision(3) << r.cpu_ms << " ms (" << std::setprecision(1) << cpu_gflops << " GFLOP/s)\n";
    } else {
        double gflops = gflops_gemm(cfg.M, cfg.N, cfg.K, r.cpu_ms);
        std::cout << "  " << cfg.M << "x" << cfg.N << "x" << cfg.K << ":" << pad
                  << std::fixed << std::setprecision(3) << "submit+wait " << r.cpu_ms << " ms ("
                  << std::setprecision(1) << gflops << " GFLOP/s)\n";
    }
}

void run_benchmark() {
    BenchContext bc = BenchContext::Create();

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(bc.context.physical_device(), &properties);

    std::cout << "Vulkan FP16 GEMM Benchmark\n";
    std::cout << "===========================\n";
    std::cout << "Device: " << properties.deviceName << "\n";
    if (bc.has_gpu_timestamps()) {
        std::cout << "Timing: GPU timestamps ("
                  << bc.timestamp_period_ns << " ns/tick)\n";
        std::cout << "  Note: kernel=GPU timestamp, submit+wait=wall time\n";
    } else {
        std::cout << "Timing: CPU wall-clock (GPU timestamps unavailable)\n";
    }

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
        auto r = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, r);
    }

    std::cout << "\nModel shapes (Qwen2-0.5B):\n";
    std::vector<GemmConfig> model_configs = {
        {8, 1152, 896},
        {8, 9728, 896},
        {8, 896, 4864},
    };
    for (const auto& cfg : model_configs) {
        auto r = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, r);
    }

    std::cout << "\nDecode shapes:\n";
    std::vector<GemmConfig> decode_configs = {
        {1, 896, 896},
        {1, 9728, 896},
    };
    for (const auto& cfg : decode_configs) {
        auto r = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, r);
    }

    std::cout << "\nSmall-M focus:\n";
    std::vector<GemmConfig> small_m_configs = {
        {1, 896, 896},
        {1, 9728, 896},
        {8, 1152, 896},
        {8, 9728, 896},
        {8, 896, 4864},
    };
    for (const auto& cfg : small_m_configs) {
        auto r = benchmark_gemm(bc, cfg.M, cfg.N, cfg.K, warmup, iters);
        print_result(cfg, r);
    }

    // GEMV: M=1, memory-bandwidth-bound regime.
    // Weight matrix B is N*K fp16 elements; report effective bandwidth.
    std::cout << "\nGEMV (M=1):\n";
    if (bc.has_gpu_timestamps()) {
        std::cout << "  N x K          | Weight(MiB) | GEMM ms | GEMM BW(GB/s) | GEMV ms | GEMV BW(GB/s) | GEMM submit ms | GEMV submit ms\n";
        std::cout << "  ---------------|-------------|--------|-------------|--------|-------------|---------------|--------------\n";
    } else {
        std::cout << "  N x K          | Weight(MiB) | GEMM submit ms | GEMM BW(GB/s) | GEMV submit ms | GEMV BW(GB/s)\n";
        std::cout << "  ---------------|-------------|--------------|-------------|--------------|-------------\n";
    }

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
        auto gemm_r = benchmark_gemm(bc, 1, gc.N, gc.K, warmup, iters);
        auto gemv_r = benchmark_gemv(bc, gc.N, gc.K, warmup, iters);
        size_t weight_bytes = static_cast<size_t>(gc.N) * gc.K * sizeof(uint16_t);
        double weight_mib = static_cast<double>(weight_bytes) / (1024.0 * 1024.0);
        if (gemm_r.gpu_ms > 0.0) {
            double gemm_bw = bandwidth_gb_s(weight_bytes, gemm_r.gpu_ms);
            double gemv_bw = bandwidth_gb_s(weight_bytes, gemv_r.gpu_ms);
            std::cout << "  " << std::setw(5) << gc.N << "x" << std::setw(5) << gc.K
                      << std::setw(5) << "" << "| "
                      << std::setw(11) << std::fixed << std::setprecision(2) << weight_mib << " | "
                      << std::setw(6) << std::setprecision(3) << gemm_r.gpu_ms << " | "
                      << std::setw(11) << std::setprecision(1) << gemm_bw << " | "
                      << std::setw(6) << std::setprecision(3) << gemv_r.gpu_ms << " | "
                      << std::setw(11) << std::setprecision(1) << gemv_bw << " | "
                      << std::setw(13) << std::setprecision(3) << gemm_r.cpu_ms << " | "
                      << std::setw(12) << std::setprecision(3) << gemv_r.cpu_ms << "\n";
        } else {
            double gemm_bw = bandwidth_gb_s(weight_bytes, gemm_r.cpu_ms);
            double gemv_bw = bandwidth_gb_s(weight_bytes, gemv_r.cpu_ms);
            std::cout << "  " << std::setw(5) << gc.N << "x" << std::setw(5) << gc.K
                      << std::setw(5) << "" << "| "
                      << std::setw(11) << std::fixed << std::setprecision(2) << weight_mib << " | "
                      << std::setw(13) << std::setprecision(3) << gemm_r.cpu_ms << " | "
                      << std::setw(11) << std::setprecision(1) << gemm_bw << " | "
                      << std::setw(13) << std::setprecision(3) << gemv_r.cpu_ms << " | "
                      << std::setw(11) << std::setprecision(1) << gemv_bw << "\n";
        }
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
