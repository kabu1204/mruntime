#include "mruntime/cpu_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kai_gemm.h"

namespace mruntime {

// Platform capability queries
bool CpuBackend::supports_fp16_gemm() {
    return kai_has_fp16();
}

bool CpuBackend::supports_bf16_gemm() {
    return false;
}

const std::vector<uint16_t>& CpuBackend::get_or_create_packed_rhs_fp16(const Tensor& B, size_t n, size_t k) {
    const void* data = B.data();
    for (auto& entry : packed_rhs_cache_) {
        if (entry.data == data && entry.n == n && entry.k == k && entry.dtype == B.dtype()) {
            return entry.rhs_packed;
        }
    }

    PackedRhsCacheEntry entry;
    entry.data = data;
    entry.n = n;
    entry.k = k;
    entry.dtype = B.dtype();

    // Pack RHS as KxN FP16 (transposed from the usual NxK weight layout).
    std::vector<uint16_t> rhs_kxn_fp16(k * n);

    if (B.dtype() == DType::FP16) {
        const uint16_t* b = B.data_ptr<uint16_t>();
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = 0; col < k; ++col) {
                rhs_kxn_fp16[col * n + row] = b[row * k + col];
            }
        }
    } else if (B.dtype() == DType::BF16) {
        const uint16_t* b = B.data_ptr<uint16_t>();
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = 0; col < k; ++col) {
                float f = bf16_to_float(b[row * k + col]);
                rhs_kxn_fp16[col * n + row] = float_to_fp16_bits(f);
            }
        }
    } else {
        // Not expected today, but keep a defined behavior.
        const float* b = B.data_ptr<float>();
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = 0; col < k; ++col) {
                rhs_kxn_fp16[col * n + row] = float_to_fp16_bits(b[row * k + col]);
            }
        }
    }

    KaiPackedRhsFp16 packed = kai_pack_rhs_fp16_kxn_with_zero_bias(rhs_kxn_fp16.data(), n, k);
    entry.rhs_packed = std::move(packed.rhs_packed);
    packed_rhs_cache_.push_back(std::move(entry));
    return packed_rhs_cache_.back().rhs_packed;
}

void CpuBackend::gemm(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha,
    float beta,
    bool trans_a,
    bool trans_b
) {
    assert(A.is_contiguous() && B.is_contiguous() && C.is_contiguous());

    assert(A.ndim() == 2 || A.ndim() == 3);
    assert(B.ndim() == 2);
    assert(C.ndim() == 2 || C.ndim() == 3);

    const size_t a_rows0 = (A.ndim() == 2) ? A.dim(0) : (A.dim(0) * A.dim(1));
    const size_t a_cols0 = A.dim(A.ndim() - 1);
    const size_t b_rows0 = B.dim(0);
    const size_t b_cols0 = B.dim(1);

    const size_t M = trans_a ? a_cols0 : a_rows0;
    const size_t K = trans_a ? a_rows0 : a_cols0;
    const size_t N = trans_b ? b_rows0 : b_cols0;
    assert((trans_b ? b_cols0 : b_rows0) == K);

    assert(((C.ndim() == 2) ? C.dim(0) : (C.dim(0) * C.dim(1))) == M);
    assert(C.dim(C.ndim() - 1) == N);

    const DType a_dtype = A.dtype();
    const DType b_dtype = B.dtype();
    const DType c_dtype = C.dtype();

    // KleidiAI fast-path (Arm64): fp16 activations + (fp16/bf16) weights (transposed), output fp16.
    if (!trans_a && trans_b && alpha == 1.0f && beta == 0.0f && a_dtype == DType::FP16 && c_dtype == DType::FP16 &&
        (b_dtype == DType::FP16 || b_dtype == DType::BF16) && supports_fp16_gemm()) {
        const uint16_t* a_fp16 = A.data_ptr<uint16_t>();
        uint16_t* c_fp16 = C.data_ptr<uint16_t>();
        const std::vector<uint16_t>& rhs_packed = get_or_create_packed_rhs_fp16(B, N, K);

        kai_matmul_fp16_packed_rhs(
            M,
            N,
            K,
            a_fp16,
            K * sizeof(uint16_t),
            rhs_packed.data(),
            c_fp16,
            N * sizeof(uint16_t)
        );
        return;
    }

    if (M == 0 || N == 0) {
        return;
    }

    const void* a_data = A.data();
    const void* b_data = B.data();
    void* c_data = C.data();

    const auto a_index = [&](size_t m, size_t k) -> size_t {
        return trans_a ? (k * a_cols0 + m) : (m * a_cols0 + k);
    };
    const auto b_index = [&](size_t k, size_t n) -> size_t {
        return trans_b ? (n * b_cols0 + k) : (k * b_cols0 + n);
    };

    constexpr size_t tile_n = 128;
    const size_t n_tiles = (N + tile_n - 1) / tile_n;
    const size_t task_count = M * n_tiles;

    auto worker = [&](size_t task_id) {
        const size_t m = task_id / n_tiles;
        const size_t tile = task_id - m * n_tiles;
        const size_t n0 = tile * tile_n;
        const size_t n1 = std::min(n0 + tile_n, N);

        for (size_t n = n0; n < n1; ++n) {
            float acc = 0.0f;
            if (alpha != 0.0f) {
                for (size_t k = 0; k < K; ++k) {
                    const float a = load_scalar_as_fp32(a_data, a_dtype, a_index(m, k));
                    const float b = load_scalar_as_fp32(b_data, b_dtype, b_index(k, n));
                    acc += a * b;
                }
                acc *= alpha;
            }

            float out = acc;
            if (beta != 0.0f) {
                out += beta * load_scalar_as_fp32(c_data, c_dtype, m * N + n);
            }
            store_scalar_from_fp32(c_data, c_dtype, m * N + n, out);
        }
    };

    pthreadpool_.parallelize_1d(task_count, worker);
}

void CpuBackend::flash_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& output,
    float scale,
    bool causal
) {
    assert(Q.ndim() == 4);
    assert(K.ndim() == 4);
    assert(V.ndim() == 4);
    assert(output.ndim() == 4);

    const size_t batch_size = Q.dim(0);
    const size_t num_heads = Q.dim(1);
    const size_t q_seq_len = Q.dim(2);
    const size_t qk_head_size = Q.dim(3);
    const size_t kv_seq_len = K.dim(2);
    const size_t v_head_size = V.dim(3);

    assert(K.dim(0) == batch_size);
    assert(K.dim(1) == num_heads);
    assert(V.dim(0) == batch_size);
    assert(V.dim(1) == num_heads);
    assert(output.dim(0) == batch_size);
    assert(output.dim(1) == num_heads);
    assert(output.dim(2) == q_seq_len);
    assert(output.dim(3) == v_head_size);
    assert(K.dim(3) == qk_head_size);
    assert(V.dim(2) == kv_seq_len);

    const void* q = Q.data();
    const void* k = K.data();
    const void* v = V.data();
    void* out = output.data();

    const DType q_dtype = Q.dtype();
    const DType k_dtype = K.dtype();
    const DType v_dtype = V.dtype();
    const DType out_dtype = output.dtype();

    const size_t q_stride0 = Q.stride(0);
    const size_t q_stride1 = Q.stride(1);
    const size_t q_stride2 = Q.stride(2);
    const size_t q_stride3 = Q.stride(3);

    const size_t k_stride0 = K.stride(0);
    const size_t k_stride1 = K.stride(1);
    const size_t k_stride2 = K.stride(2);
    const size_t k_stride3 = K.stride(3);

    const size_t v_stride0 = V.stride(0);
    const size_t v_stride1 = V.stride(1);
    const size_t v_stride2 = V.stride(2);
    const size_t v_stride3 = V.stride(3);

    const size_t o_stride0 = output.stride(0);
    const size_t o_stride1 = output.stride(1);
    const size_t o_stride2 = output.stride(2);
    const size_t o_stride3 = output.stride(3);

    // For the call sites in this repo, Q corresponds to the last q_seq_len positions in K/V.
    const size_t base_position = (kv_seq_len >= q_seq_len) ? (kv_seq_len - q_seq_len) : 0;

    const size_t task_count = batch_size * num_heads * q_seq_len;
    auto worker = [&](size_t task_id) {
        size_t tmp = task_id;
        const size_t qi = tmp % q_seq_len;
        tmp /= q_seq_len;
        const size_t h = tmp % num_heads;
        const size_t b = tmp / num_heads;

        size_t max_k = kv_seq_len - 1;
        if (causal && kv_seq_len >= q_seq_len) {
            max_k = base_position + qi;
            if (max_k >= kv_seq_len) max_k = kv_seq_len - 1;
        }

        static thread_local std::vector<float> scores;
        static thread_local std::vector<float> acc;
        if (scores.size() < kv_seq_len) {
            scores.resize(kv_seq_len);
        }
        if (acc.size() < v_head_size) {
            acc.resize(v_head_size);
        }
        std::fill(acc.begin(), acc.begin() + v_head_size, 0.0f);

        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t ki = 0; ki <= max_k; ++ki) {
            float dot = 0.0f;
            for (size_t d = 0; d < qk_head_size; ++d) {
                const size_t q_idx = b * q_stride0 + h * q_stride1 + qi * q_stride2 + d * q_stride3;
                const size_t k_idx = b * k_stride0 + h * k_stride1 + ki * k_stride2 + d * k_stride3;
                dot += load_scalar_as_fp32(q, q_dtype, q_idx) * load_scalar_as_fp32(k, k_dtype, k_idx);
            }
            const float s = dot * scale;
            scores[ki] = s;
            max_score = std::max(max_score, s);
        }

        float sum_exp = 0.0f;
        for (size_t ki = 0; ki <= max_k; ++ki) {
            const float e = std::exp(scores[ki] - max_score);
            sum_exp += e;
            for (size_t d = 0; d < v_head_size; ++d) {
                const size_t v_idx = b * v_stride0 + h * v_stride1 + ki * v_stride2 + d * v_stride3;
                acc[d] += e * load_scalar_as_fp32(v, v_dtype, v_idx);
            }
        }

        const float inv_sum = 1.0f / sum_exp;
        for (size_t d = 0; d < v_head_size; ++d) {
            const size_t o_idx = b * o_stride0 + h * o_stride1 + qi * o_stride2 + d * o_stride3;
            store_scalar_from_fp32(out, out_dtype, o_idx, acc[d] * inv_sum);
        }
    };

    pthreadpool_.parallelize_1d(task_count, worker);
}

void CpuBackend::rmsnorm(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps
) {
    assert(input.is_contiguous());
    assert(weight.is_contiguous());
    assert(output.is_contiguous());

    const void* in = input.data();
    const void* w = weight.data();
    void* out = output.data();

    const DType in_dtype = input.dtype();
    const DType w_dtype = weight.dtype();
    const DType out_dtype = output.dtype();

    size_t hidden_size = input.dim(input.ndim() - 1);
    size_t num_tokens = input.numel() / hidden_size;

    auto worker = [&](size_t t) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float v = load_scalar_as_fp32(in, in_dtype, t * hidden_size + i);
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        float inv_rms = 1.0f / rms;

        for (size_t i = 0; i < hidden_size; ++i) {
            float x = load_scalar_as_fp32(in, in_dtype, t * hidden_size + i);
            float weight_v = load_scalar_as_fp32(w, w_dtype, i);
            store_scalar_from_fp32(out, out_dtype, t * hidden_size + i, x * inv_rms * weight_v);
        }
    };

    pthreadpool_.parallelize_1d(num_tokens, worker);
}

void CpuBackend::rope(
    Tensor& Q,
    Tensor& K,
    size_t position_offset,
    float theta
) {
    assert(Q.ndim() == 4);
    assert(K.ndim() == 4);
    assert(Q.is_contiguous());
    assert(K.is_contiguous());

    auto apply_rope = [this, theta, position_offset](Tensor& t) {
        void* data = t.data();
        const DType dtype = t.dtype();
        size_t batch = t.dim(0);
        size_t seq_len = t.dim(1);
        size_t num_heads = t.dim(2);
        size_t head_dim = t.dim(3);
        size_t half_dim = head_dim / 2;

        const size_t total_heads = batch * seq_len * num_heads;
        auto worker = [&](size_t idx) {
            size_t tmp = idx;
            size_t h = tmp % num_heads;
            tmp /= num_heads;
            size_t s = tmp % seq_len;
            size_t b = tmp / seq_len;

            size_t pos = position_offset + s;
            const size_t base = ((b * seq_len + s) * num_heads + h) * head_dim;
            for (size_t i = 0; i < half_dim; ++i) {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * freq;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                float x0 = load_scalar_as_fp32(data, dtype, base + i);
                float x1 = load_scalar_as_fp32(data, dtype, base + i + half_dim);
                store_scalar_from_fp32(data, dtype, base + i, x0 * cos_val - x1 * sin_val);
                store_scalar_from_fp32(data, dtype, base + i + half_dim, x0 * sin_val + x1 * cos_val);
            }
        };

        pthreadpool_.parallelize_1d(total_heads, worker);
    };

    apply_rope(Q);
    apply_rope(K);
}

void CpuBackend::silu(
    const Tensor& input,
    Tensor& output
) {
    assert(input.is_contiguous());
    assert(output.is_contiguous());
    assert(output.numel() == input.numel());

    const void* in = input.data();
    void* out = output.data();
    const DType in_dtype = input.dtype();
    const DType out_dtype = output.dtype();
    size_t n = input.numel();

    auto worker = [&](size_t i) {
        float x = load_scalar_as_fp32(in, in_dtype, i);
        store_scalar_from_fp32(out, out_dtype, i, x / (1.0f + std::exp(-x)));
    };

    pthreadpool_.parallelize_1d(n, worker);
}

void CpuBackend::elementwise_mul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    assert(a.is_contiguous());
    assert(b.is_contiguous());
    assert(output.is_contiguous());
    assert(a.numel() == b.numel() && a.numel() == output.numel());

    const void* pa = a.data();
    const void* pb = b.data();
    void* pout = output.data();
    const DType a_dtype = a.dtype();
    const DType b_dtype = b.dtype();
    const DType out_dtype = output.dtype();
    size_t n = a.numel();

    auto worker = [&](size_t i) {
        float av = load_scalar_as_fp32(pa, a_dtype, i);
        float bv = load_scalar_as_fp32(pb, b_dtype, i);
        store_scalar_from_fp32(pout, out_dtype, i, av * bv);
    };

    pthreadpool_.parallelize_1d(n, worker);
}

void CpuBackend::add(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    assert(a.is_contiguous());
    assert(b.is_contiguous());
    assert(output.is_contiguous());
    assert(a.numel() == b.numel() && a.numel() == output.numel());

    const void* pa = a.data();
    const void* pb = b.data();
    void* pout = output.data();
    const DType a_dtype = a.dtype();
    const DType b_dtype = b.dtype();
    const DType out_dtype = output.dtype();
    size_t n = a.numel();

    auto worker = [&](size_t i) {
        float av = load_scalar_as_fp32(pa, a_dtype, i);
        float bv = load_scalar_as_fp32(pb, b_dtype, i);
        store_scalar_from_fp32(pout, out_dtype, i, av + bv);
    };

    pthreadpool_.parallelize_1d(n, worker);
}

void CpuBackend::softmax(
    const Tensor& input,
    Tensor& output,
    int dim
) {
    assert(input.is_contiguous());
    assert(output.is_contiguous());
    assert(input.numel() == output.numel());

    if (dim < 0) dim = static_cast<int>(input.ndim()) + dim;
    assert(dim >= 0 && static_cast<size_t>(dim) < input.ndim());

    const void* in = input.data();
    void* out = output.data();
    const DType in_dtype = input.dtype();
    const DType out_dtype = output.dtype();

    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input.dim(i);
    }
    size_t dim_size = input.dim(dim);
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < input.ndim(); ++i) {
        inner_size *= input.dim(i);
    }

    const size_t total_rows = outer_size * inner_size;
    auto worker = [&](size_t row) {
        size_t o = row / inner_size;
        size_t inner = row % inner_size;

        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t d = 0; d < dim_size; ++d) {
            size_t idx = (o * dim_size + d) * inner_size + inner;
            max_val = std::max(max_val, load_scalar_as_fp32(in, in_dtype, idx));
        }

        static thread_local std::vector<float> tmp;
        if (tmp.size() < dim_size) {
            tmp.resize(dim_size);
        }

        float sum = 0.0f;
        for (size_t d = 0; d < dim_size; ++d) {
            size_t idx = (o * dim_size + d) * inner_size + inner;
            float v = load_scalar_as_fp32(in, in_dtype, idx);
            float e = std::exp(v - max_val);
            tmp[d] = e;
            sum += e;
        }
        for (size_t d = 0; d < dim_size; ++d) {
            size_t idx = (o * dim_size + d) * inner_size + inner;
            store_scalar_from_fp32(out, out_dtype, idx, tmp[d] / sum);
        }
    };

    pthreadpool_.parallelize_1d(total_rows, worker);
}

void CpuBackend::embedding_lookup(
    const Tensor& weight,
    const std::vector<int>& token_ids,
    Tensor& output
) {
    assert(weight.ndim() == 2);
    assert(weight.is_contiguous());
    assert(output.is_contiguous());
    assert(output.ndim() == 3);
    assert(output.dim(0) == 1);

    const size_t vocab_size = weight.dim(0);
    const size_t hidden_size = weight.dim(1);
    assert(output.dim(1) == token_ids.size());
    assert(output.dim(2) == hidden_size);

    for (int token_id : token_ids) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
            throw std::out_of_range("embedding_lookup: token_id out of range");
        }
    }

    const void* w = weight.data();
    void* out = output.data();
    const DType w_dtype = weight.dtype();
    const DType out_dtype = output.dtype();

    const size_t w_row_bytes = hidden_size * dtype_size(w_dtype);
    const size_t out_row_bytes = hidden_size * dtype_size(out_dtype);

    if (w_dtype == out_dtype) {
        auto worker = [&](size_t t) {
            int token_id = token_ids[t];
            const char* src = static_cast<const char*>(w) + static_cast<size_t>(token_id) * w_row_bytes;
            char* dst = static_cast<char*>(out) + t * out_row_bytes;
            std::memcpy(dst, src, out_row_bytes);
        };
        pthreadpool_.parallelize_1d(token_ids.size(), worker);
        return;
    }

    auto worker = [&](size_t t) {
        int token_id = token_ids[t];
        const size_t w_base = static_cast<size_t>(token_id) * hidden_size;
        const size_t out_base = t * hidden_size;
        for (size_t i = 0; i < hidden_size; ++i) {
            float v = load_scalar_as_fp32(w, w_dtype, w_base + i);
            store_scalar_from_fp32(out, out_dtype, out_base + i, v);
        }
    };

    pthreadpool_.parallelize_1d(token_ids.size(), worker);
}

}  // namespace mruntime
