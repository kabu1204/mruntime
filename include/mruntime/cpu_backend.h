#pragma once

#include "mruntime/backend.h"
#include "mruntime/pthreadpool_raii.h"

namespace mruntime {

class CpuBackend : public Backend {
public:
    CpuBackend() : pthreadpool_(PThreadPool::Create(0)) {}
    explicit CpuBackend(size_t threads_count) : pthreadpool_(PThreadPool::Create(threads_count)) {}
    ~CpuBackend() override = default;

    CpuBackend(const CpuBackend&) = delete;
    CpuBackend& operator=(const CpuBackend&) = delete;
    CpuBackend(CpuBackend&&) noexcept = default;
    CpuBackend& operator=(CpuBackend&&) noexcept = default;

    // Platform capability queries
    static bool supports_fp16_gemm();
    static bool supports_bf16_gemm();

    void gemm(
        const Tensor& A,
        const Tensor& B,
        Tensor& C,
        float alpha = 1.0f,
        float beta = 0.0f,
        bool trans_a = false,
        bool trans_b = false
    ) override;

    void flash_attention(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        Tensor& output,
        float scale,
        bool causal = true
    ) override;

    void rmsnorm(
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        float eps = 1e-6f
    ) override;

    void rope(
        Tensor& Q,
        Tensor& K,
        size_t position_offset,
        float theta = 10000.0f
    ) override;

    void silu(
        const Tensor& input,
        Tensor& output
    ) override;

    void elementwise_mul(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    ) override;

    void add(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    ) override;

    void softmax(
        const Tensor& input,
        Tensor& output,
        int dim = -1
    ) override;

    void embedding_lookup(
        const Tensor& weight,
        const std::vector<int>& token_ids,
        Tensor& output
    ) override;

private:
    struct PackedRhsCacheEntry {
        const void* data = nullptr;
        size_t n = 0;
        size_t k = 0;
        DType dtype = DType::FP32;
        std::vector<uint16_t> rhs_packed;
    };

    const std::vector<uint16_t>& get_or_create_packed_rhs_fp16(const Tensor& B, size_t n, size_t k);

    PThreadPool pthreadpool_;
    std::vector<PackedRhsCacheEntry> packed_rhs_cache_;
};

}  // namespace mruntime
