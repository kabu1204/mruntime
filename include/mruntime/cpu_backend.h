#pragma once

#include "mruntime/backend.h"

namespace mruntime {

class CpuBackend : public Backend {
public:
    CpuBackend() = default;
    ~CpuBackend() override = default;

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
};

}  // namespace mruntime
