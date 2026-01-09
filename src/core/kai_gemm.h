#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mruntime/tensor.h"

namespace mruntime {

struct KaiPackedRhsFp16 {
    size_t n = 0;
    size_t k = 0;
    std::vector<uint16_t> rhs_packed;
    std::vector<uint16_t> rhs_kxn_fp16;
};

bool kai_has_fp16();

KaiPackedRhsFp16 kai_pack_rhs_fp16_kxn_with_zero_bias(const uint16_t* rhs_kxn_fp16, size_t n, size_t k);

void kai_matmul_fp16_packed_rhs(
    size_t m,
    size_t n,
    size_t k,
    const uint16_t* lhs_mxk_fp16,
    size_t lhs_stride_bytes,
    const uint16_t* rhs_packed,
    uint16_t* dst_mxn_fp16,
    size_t dst_stride_row_bytes
);

}  // namespace mruntime
