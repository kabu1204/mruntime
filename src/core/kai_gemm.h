#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mruntime {

struct KaiPackedRhsFp16 {
    size_t n = 0;
    size_t k = 0;
    std::vector<uint16_t> rhs_packed;
};

bool kai_has_fp16();

size_t kai_rhs_packed_size_fp16_kxn_with_zero_bias(size_t n, size_t k);

void kai_pack_rhs_fp16_kxn_with_zero_bias(
    const uint16_t* rhs_kxn_fp16,
    size_t n,
    size_t k,
    uint16_t* rhs_packed_out,
    size_t rhs_packed_bytes
);

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

// Tile stepping info for parallelization
size_t kai_get_m_step_fp16();
size_t kai_get_n_step_fp16();

// Compute a single tile at (m_start, n_start)
void kai_matmul_fp16_tile(
    size_t m_start,
    size_t n_start,
    size_t M, size_t N, size_t K,
    const uint16_t* lhs,
    size_t lhs_stride_bytes,
    const uint16_t* rhs_packed,
    uint16_t* dst,
    size_t dst_stride_row_bytes
);

}  // namespace mruntime
