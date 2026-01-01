#pragma once

#include <vector>

#include "mruntime/tensor.h"

namespace mruntime {

class Backend {
public:
    virtual ~Backend() = default;

    virtual void gemm(
        const Tensor& A,
        const Tensor& B,
        Tensor& C,
        float alpha = 1.0f,
        float beta = 0.0f,
        bool trans_a = false,
        bool trans_b = false
    ) = 0;

    virtual void flash_attention(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        Tensor& output,
        float scale,
        bool causal = true
    ) = 0;

    virtual void rmsnorm(
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        float eps = 1e-6f
    ) = 0;

    virtual void rope(
        Tensor& Q,
        Tensor& K,
        size_t position_offset,
        float theta = 10000.0f
    ) = 0;

    virtual void silu(
        const Tensor& input,
        Tensor& output
    ) = 0;

    virtual void elementwise_mul(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    ) = 0;

    virtual void add(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    ) = 0;

    virtual void softmax(
        const Tensor& input,
        Tensor& output,
        int dim = -1
    ) = 0;

    virtual void embedding_lookup(
        const Tensor& weight,
        const std::vector<int>& token_ids,
        Tensor& output
    ) = 0;
};

}  // namespace mruntime
