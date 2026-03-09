#include "kai_gemm.h"

#include <cassert>
#include <limits>

#if defined(__aarch64__)

#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"

namespace mruntime {

constexpr struct kai_matmul_clamp_f16_f16_f16p_ukernel ukernel {
    .get_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_lhs_packed_offset = kai_get_lhs_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_rhs_packed_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
    .run_matmul = kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla
};

bool kai_has_fp16() {
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return true;
#else
    return false;
#endif
}

size_t kai_rhs_packed_size_fp16_kxn_with_zero_bias(size_t n, size_t k) {
    return kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon(n, k);
}

void kai_pack_rhs_fp16_kxn_with_zero_bias(
    const uint16_t* rhs_kxn_fp16,
    size_t n,
    size_t k,
    uint16_t* rhs_packed_out,
    size_t rhs_packed_bytes
) {
    const size_t rhs_stride_bytes = n * sizeof(uint16_t);
    const size_t packed_size = kai_rhs_packed_size_fp16_kxn_with_zero_bias(n, k);
    assert(rhs_packed_bytes >= packed_size && "rhs_packed_out too small for Kai packed RHS");
    (void)rhs_packed_bytes;
    (void)packed_size;

    kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon(
        1,
        n,
        k,
        kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon(),
        ukernel.get_kr(),
        ukernel.get_sr(),
        rhs_stride_bytes,
        rhs_kxn_fp16,
        nullptr,
        nullptr,
        rhs_packed_out,
        0,
        nullptr
    );
}

KaiPackedRhsFp16 kai_pack_rhs_fp16_kxn_with_zero_bias(const uint16_t* rhs_kxn_fp16, size_t n, size_t k) {
    KaiPackedRhsFp16 out;
    out.n = n;
    out.k = k;

    // KleidiAI expects RHS as KxN (row-major), with row stride in bytes.
    // The packed format includes a per-block bias; we pack it as zeros.
    const size_t packed_size = kai_rhs_packed_size_fp16_kxn_with_zero_bias(n, k);
    out.rhs_packed.resize(packed_size / sizeof(uint16_t));
    kai_pack_rhs_fp16_kxn_with_zero_bias(
        rhs_kxn_fp16, n, k, out.rhs_packed.data(), packed_size);

    return out;
}

void kai_matmul_fp16_packed_rhs(
    size_t m,
    size_t n,
    size_t k,
    const uint16_t* lhs_mxk_fp16,
    size_t lhs_stride_bytes,
    const uint16_t* rhs_packed,
    uint16_t* dst_mxn_fp16,
    size_t dst_stride_row_bytes
) {
    // dst is row-major with contiguous columns.
    kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla(
        m,
        n,
        k,
        lhs_mxk_fp16,
        lhs_stride_bytes,
        rhs_packed,
        dst_mxn_fp16,
        dst_stride_row_bytes,
        sizeof(uint16_t),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
    );
}

void kai_matmul_fp16_packed_rhs_stripe(
    size_t n_start,
    size_t n_len,
    size_t M,
    size_t N,
    size_t K,
    const uint16_t* lhs,
    size_t lhs_stride_bytes,
    const uint16_t* rhs_packed,
    uint16_t* dst,
    size_t dst_stride_row_bytes
) {
    if (M == 0 || N == 0 || n_len == 0) return;
    if (n_start >= N) return;

    const size_t n_step = ukernel.get_n_step();
    assert(n_start % n_step == 0 && "n_start must be divisible by ukernel n_step");
    if (n_start % n_step != 0) return;

    const size_t actual_n = std::min(n_len, N - n_start);

    const uint8_t* lhs_ptr = (const uint8_t*)lhs + ukernel.get_lhs_packed_offset(0, lhs_stride_bytes);
    const uint8_t* rhs_ptr = (const uint8_t*)rhs_packed + ukernel.get_rhs_packed_offset(n_start, K);
    uint8_t* dst_ptr = (uint8_t*)dst + ukernel.get_dst_offset(0, n_start, dst_stride_row_bytes);

    ukernel.run_matmul(
        M, actual_n, K,
        lhs_ptr, lhs_stride_bytes,
        rhs_ptr,
        dst_ptr, dst_stride_row_bytes, sizeof(uint16_t),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
    );
}

size_t kai_get_m_step_fp16() {
    return ukernel.get_m_step();
}

size_t kai_get_n_step_fp16() {
    return ukernel.get_n_step();
}

void kai_matmul_fp16_tile(
    size_t m_start,
    size_t n_start,
    size_t M, size_t N, size_t K,
    const uint16_t* lhs,
    size_t lhs_stride_bytes,
    const uint16_t* rhs_packed,
    uint16_t* dst,
    size_t dst_stride_row_bytes
) {
    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    const uint8_t* lhs_ptr = (const uint8_t*)lhs + ukernel.get_lhs_packed_offset(m_start, lhs_stride_bytes);
    const uint8_t* rhs_ptr = (const uint8_t*)rhs_packed + ukernel.get_rhs_packed_offset(n_start, K);
    uint8_t* dst_ptr = (uint8_t*)dst + ukernel.get_dst_offset(m_start, n_start, dst_stride_row_bytes);

    const size_t actual_m = std::min(M - m_start, m_step);
    const size_t actual_n = std::min(N - n_start, n_step);

    ukernel.run_matmul(
        actual_m, actual_n, K,
        lhs_ptr, lhs_stride_bytes,
        rhs_ptr,
        dst_ptr, dst_stride_row_bytes, sizeof(uint16_t),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
    );
}

}  // namespace mruntime

#else

namespace mruntime {

bool kai_has_fp16() {
    return false;
}

size_t kai_rhs_packed_size_fp16_kxn_with_zero_bias(size_t, size_t) {
    return 0;
}

void kai_pack_rhs_fp16_kxn_with_zero_bias(const uint16_t*, size_t, size_t, uint16_t*, size_t) {
}

KaiPackedRhsFp16 kai_pack_rhs_fp16_kxn_with_zero_bias(const uint16_t*, size_t, size_t) {
    return {};
}

void kai_matmul_fp16_packed_rhs(
    size_t,
    size_t,
    size_t,
    const uint16_t*,
    size_t,
    const uint16_t*,
    uint16_t*,
    size_t
) {
}

void kai_matmul_fp16_packed_rhs_stripe(
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    const uint16_t*,
    size_t,
    const uint16_t*,
    uint16_t*,
    size_t
) {
}

size_t kai_get_m_step_fp16() {
    return 0;
}

size_t kai_get_n_step_fp16() {
    return 0;
}

void kai_matmul_fp16_tile(
    size_t, size_t,
    size_t, size_t, size_t,
    const uint16_t*, size_t,
    const uint16_t*,
    uint16_t*, size_t
) {
}

}  // namespace mruntime

#endif
