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

mruntime::vulkan::VkBufferArena make_arena(const TestContext& tc, VkDeviceSize capacity) {
    mruntime::vulkan::VkBufferArenaCreateInfo info;
    info.capacity_bytes = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.memory_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    info.default_alignment = tc.alignment;
    return mruntime::vulkan::VkBufferArena::Create(tc.context.physical_device(), tc.context.device(), info);
}

void check_close(const char* test_name, const uint16_t* got_fp16, const float* expected, uint32_t n,
                 float tolerance) {
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

// CPU reference: y = W @ x, FP32 accumulation.
void cpu_gemv(const uint16_t* x, const uint16_t* w, float* y, uint32_t N, uint32_t K) {
    for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            float xv = mruntime::fp16_bits_to_float(x[k]);
            float wv = mruntime::fp16_bits_to_float(w[n * K + k]);
            acc += xv * wv;
        }
        y[n] = acc;
    }
}

void test_gemv(TestContext& tc, uint32_t N, uint32_t K) {
    const std::string label = "gemv(" + std::to_string(N) + "," + std::to_string(K) + ")";

    const VkDeviceSize x_bytes = static_cast<VkDeviceSize>(K) * sizeof(uint16_t);
    const VkDeviceSize w_bytes = static_cast<VkDeviceSize>(N) * K * sizeof(uint16_t);
    const VkDeviceSize y_bytes = static_cast<VkDeviceSize>(N) * sizeof(uint16_t);

    auto arena = make_arena(tc, x_bytes + w_bytes + y_bytes + 3 * tc.alignment);

    const VkDeviceSize x_offset = arena.alloc(x_bytes);
    const VkDeviceSize w_offset = arena.alloc(w_bytes);
    const VkDeviceSize y_offset = arena.alloc(y_bytes);

    uint16_t* x_data = arena.host_ptr<uint16_t>(x_offset);
    uint16_t* w_data = arena.host_ptr<uint16_t>(w_offset);
    uint16_t* y_data = arena.host_ptr<uint16_t>(y_offset);

    // Fill x and W with small deterministic values to avoid FP16 overflow.
    for (uint32_t i = 0; i < K; ++i) {
        float val = -0.5f + 0.001f * static_cast<float>(i % 1000);
        x_data[i] = mruntime::float_to_fp16_bits(val);
    }
    for (uint32_t i = 0; i < N * K; ++i) {
        float val = 0.3f - 0.001f * static_cast<float>(i % 600);
        w_data[i] = mruntime::float_to_fp16_bits(val);
    }

    // CPU reference.
    std::vector<float> expected(N);
    cpu_gemv(x_data, w_data, expected.data(), N, K);

    std::memset(y_data, 0, static_cast<size_t>(y_bytes));
    tc.fp16_ops.gemv(
        arena.descriptor(x_offset, x_bytes),
        arena.descriptor(w_offset, w_bytes),
        arena.descriptor(y_offset, y_bytes),
        N,
        K);

    check_close(label.c_str(), y_data, expected.data(), N, 5e-2f);
    std::cout << "  " << label << " PASSED\n";
}

void run_all_tests() {
    TestContext tc = TestContext::Create();

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(tc.context.physical_device(), &properties);
    std::cout << "Using Vulkan device: " << properties.deviceName << "\n";

    test_gemv(tc, 9, 16);
    test_gemv(tc, 896, 896);
    test_gemv(tc, 9728, 896);
    test_gemv(tc, 896, 4864);
}

}  // namespace

int main() {
    try {
        run_all_tests();
        std::cout << "vulkan_gemv_test PASSED\n";
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_gemv_test SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_gemv_test FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_gemv_test FAILED: " << error.what() << "\n";
        return 1;
    }
}
