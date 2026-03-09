#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

namespace {

struct TestContext {
    mruntime::vulkan::VkContext context;
    mruntime::vulkan::VkKernelRuntime runtime;
    mruntime::vulkan::VkFp16Ops fp16_ops;
    VkDeviceSize alignment;

    static TestContext Create() {
        TestContext tc;
        tc.context = mruntime::vulkan::VkContext::Create();
        tc.alignment = std::max<VkDeviceSize>(64, tc.context.min_storage_buffer_offset_alignment());
        tc.runtime = mruntime::vulkan::VkKernelRuntime::Create(tc.context);
        tc.fp16_ops = mruntime::vulkan::VkFp16Ops::Create(&tc.runtime);
        return tc;
    }
};

mruntime::vulkan::VkBufferArena make_arena(
    const TestContext& tc, VkDeviceSize capacity) {
    mruntime::vulkan::VkBufferArenaCreateInfo info;
    info.capacity_bytes = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    info.default_alignment = tc.alignment;
    return mruntime::vulkan::VkBufferArena::Create(
        tc.context.physical_device(), tc.context.device(), info);
}

void check_close(const char* test_name, const uint16_t* got_fp16, const float* expected,
                  uint32_t n, float tolerance) {
    for (uint32_t i = 0; i < n; ++i) {
        float got = mruntime::fp16_bits_to_float(got_fp16[i]);
        float diff = std::fabs(got - expected[i]);
        if (!std::isfinite(got) || diff > tolerance) {
            throw std::runtime_error(
                std::string(test_name) + " mismatch at i=" + std::to_string(i) +
                ": got=" + std::to_string(got) +
                ", expected=" + std::to_string(expected[i]) +
                ", diff=" + std::to_string(diff));
        }
    }
}

// CPU reference: C = A @ B^T, FP32 accumulation.
void cpu_gemm(const uint16_t* a, const uint16_t* b, float* c,
              uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float av = mruntime::fp16_bits_to_float(a[m * K + k]);
                float bv = mruntime::fp16_bits_to_float(b[n * K + k]);
                acc += av * bv;
            }
            c[m * N + n] = acc;
        }
    }
}

void test_gemm(TestContext& tc, uint32_t M, uint32_t N, uint32_t K) {
    const std::string label = "gemm(" + std::to_string(M) + "," +
                              std::to_string(N) + "," + std::to_string(K) + ")";

    const uint32_t a_count = M * K;
    const uint32_t b_count = N * K;
    const uint32_t c_count = M * N;

    const VkDeviceSize a_bytes = a_count * sizeof(uint16_t);
    const VkDeviceSize b_bytes = b_count * sizeof(uint16_t);
    const VkDeviceSize c_bytes = c_count * sizeof(uint16_t);

    auto arena = make_arena(tc, a_bytes + b_bytes + c_bytes + 3 * tc.alignment);

    const VkDeviceSize a_offset = arena.alloc(a_bytes);
    const VkDeviceSize b_offset = arena.alloc(b_bytes);
    const VkDeviceSize c_offset = arena.alloc(c_bytes);

    uint16_t* a_data = arena.host_ptr<uint16_t>(a_offset);
    uint16_t* b_data = arena.host_ptr<uint16_t>(b_offset);
    uint16_t* c_data = arena.host_ptr<uint16_t>(c_offset);

    // Fill A and B with small deterministic values to avoid FP16 overflow.
    for (uint32_t i = 0; i < a_count; ++i) {
        float val = -0.5f + 0.001f * static_cast<float>(i % 1000);
        a_data[i] = mruntime::float_to_fp16_bits(val);
    }
    for (uint32_t i = 0; i < b_count; ++i) {
        float val = 0.3f - 0.001f * static_cast<float>(i % 600);
        b_data[i] = mruntime::float_to_fp16_bits(val);
    }

    // CPU reference.
    std::vector<float> expected(c_count);
    cpu_gemm(a_data, b_data, expected.data(), M, N, K);

    std::memset(c_data, 0, c_bytes);
    tc.fp16_ops.gemm(
        arena.descriptor(a_offset, a_bytes),
        arena.descriptor(b_offset, b_bytes),
        arena.descriptor(c_offset, c_bytes),
        M, N, K);

    check_close(label.c_str(), c_data, expected.data(), c_count, 5e-2f);
    std::cout << "  " << label << " PASSED\n";
}

void run_all_tests() {
    TestContext tc = TestContext::Create();

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(tc.context.physical_device(), &properties);
    std::cout << "Using Vulkan device: " << properties.deviceName << "\n";

    // Exact tile: one 8x8x32 tile.
    test_gemm(tc, 8, 8, 32);
    // Unaligned: smaller than one tile in every dimension.
    test_gemm(tc, 3, 5, 17);
    // Decode-like: single-token projection.
    test_gemm(tc, 1, 896, 896);
    // QKV projection (prefill).
    test_gemm(tc, 8, 1152, 896);
    // Gate+up projection.
    test_gemm(tc, 8, 9728, 896);
    // Down projection.
    test_gemm(tc, 8, 896, 4864);
    // Off-by-one: nothing aligns.
    test_gemm(tc, 7, 1153, 897);
}

}  // namespace

int main() {
    try {
        run_all_tests();
        std::cout << "vulkan_gemm_test PASSED\n";
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_gemm_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_gemm_test FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_gemm_test FAILED: " << error.what() << "\n";
        return 1;
    }
}
