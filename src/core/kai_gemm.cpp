#include "kai_gemm.h"

#include <limits>

#if defined(__aarch64__)

#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

namespace mruntime {

bool kai_has_fp16() {
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return true;
#else
    return false;
#endif
}

KaiPackedRhsFp16 kai_pack_rhs_fp16_kxn_with_zero_bias(const uint16_t* rhs_kxn_fp16, size_t n, size_t k) {
    KaiPackedRhsFp16 out;
    out.n = n;
    out.k = k;

    // KleidiAI expects RHS as KxN (row-major), with row stride in bytes.
    // It also requires a bias vector of length N (fp16).
    const size_t rhs_stride_bytes = n * sizeof(uint16_t);
    const size_t packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(n, k);
    out.rhs_packed.resize(packed_size / sizeof(uint16_t));

    std::vector<uint16_t> bias_fp16(n, static_cast<uint16_t>(0));

    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1,
        n,
        k,
        kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(),
        kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(),
        kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(),
        rhs_stride_bytes,
        rhs_kxn_fp16,
        bias_fp16.data(),
        nullptr,
        out.rhs_packed.data(),
        0,
        nullptr
    );

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
    kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla(
        m,
        n,
        k,
        lhs_mxk_fp16,
        lhs_stride_bytes,
        rhs_packed,
        dst_mxn_fp16,
        dst_stride_row_bytes,
        sizeof(__fp16),
        static_cast<__fp16>(-std::numeric_limits<float>::infinity()),
        static_cast<__fp16>(std::numeric_limits<float>::infinity())
    );
}

}  // namespace mruntime

#else

namespace mruntime {

bool kai_has_fp16() {
    return false;
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

}  // namespace mruntime

#endif
