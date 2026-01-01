#include "mruntime/cpu_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "MLAS/include/mlas.h"
#include "MLAS/include/mlas_float16.h"

namespace mruntime {

namespace {

// Helper to ensure tensor is FP32 for computation
Tensor ensure_fp32(const Tensor& t) {
    if (t.dtype() == DType::FP32) {
        return t;
    }
    return t.to(DType::FP32);
}

}  // namespace

// Platform capability queries
bool CpuBackend::supports_fp16_gemm() {
    return MlasFp16AccelerationSupported();
}

bool CpuBackend::supports_bf16_gemm() {
#if defined(__aarch64__) && defined(__linux__)
    return MlasBf16AccelerationSupported();
#else
    return false;
#endif
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

    // Calculate dimensions
    size_t M, N, K;
    if (!trans_a) {
        M = A.ndim() == 2 ? A.dim(0) : A.dim(0) * A.dim(1);
        K = A.dim(A.ndim() - 1);
    } else {
        K = A.ndim() == 2 ? A.dim(0) : A.dim(0) * A.dim(1);
        M = A.dim(A.ndim() - 1);
    }
    if (!trans_b) {
        assert(B.dim(0) == K);
        N = B.dim(B.ndim() - 1);
    } else {
        N = B.dim(0);
        assert(B.dim(B.ndim() - 1) == K);
    }

    size_t lda = trans_a ? M : K;
    size_t ldb = trans_b ? K : N;
    size_t ldc = N;

    DType a_dtype = A.dtype();
    DType b_dtype = B.dtype();
    DType c_dtype = C.dtype();

    // Case 1: All FP32 - use standard SGEMM
    if (a_dtype == DType::FP32 && b_dtype == DType::FP32 && c_dtype == DType::FP32) {
        MlasGemm(
            trans_a ? CblasTrans : CblasNoTrans,
            trans_b ? CblasTrans : CblasNoTrans,
            M, N, K,
            alpha,
            A.data_ptr<float>(), lda,
            B.data_ptr<float>(), ldb,
            beta,
            C.data_ptr<float>(), ldc,
            nullptr
        );
        return;
    }

    // Case 2: FP16 weights with hardware support
    if (b_dtype == DType::FP16 && supports_fp16_gemm() && !trans_b) {
        // MlasHalfGemmBatch supports FP32 activations with FP16 weights
        // Output is FP16, we'll convert to target dtype after
        Tensor C_fp16 = Tensor::empty(C.shape(), DType::FP16);

        MLAS_HALF_GEMM_DATA_PARAMS params{};
        params.A = A.data();
        params.B = B.data();
        params.C = reinterpret_cast<MLAS_FP16*>(C_fp16.data());
        params.lda = lda;
        params.ldb = ldb;
        params.ldc = ldc;
        params.AIsfp32 = (a_dtype == DType::FP32);
        params.BIsfp32 = false;
        params.Bias = nullptr;
        params.OutputProcessor = nullptr;

        MlasHalfGemmBatch(M, N, K, 1, &params, nullptr);

        // Convert output to target dtype
        if (c_dtype == DType::FP16) {
            std::memcpy(C.data(), C_fp16.data(), C.nbytes());
        } else {
            Tensor C_converted = C_fp16.to(c_dtype);
            std::memcpy(C.data(), C_converted.data(), C.nbytes());
        }
        return;
    }

#if defined(__aarch64__) && defined(__linux__)
    // Case 3: BF16 weights with hardware support (ARM64 Linux only)
    if (b_dtype == DType::BF16 && supports_bf16_gemm() && !trans_b) {
        // MlasSBGemmBatch outputs FP32 directly
        assert(c_dtype == DType::FP32 && "BF16 GEMM outputs FP32");

        MLAS_SBGEMM_DATA_PARAMS params{};
        params.A = A.data();
        params.B = B.data();
        params.C = C.data_ptr<float>();
        params.lda = lda;
        params.ldb = ldb;
        params.ldc = ldc;
        params.AIsfp32 = (a_dtype == DType::FP32);
        params.BIsfp32 = false;
        params.Bias = nullptr;
        params.OutputProcessor = nullptr;

        MlasSBGemmBatch(M, N, K, 1, &params, nullptr);
        return;
    }
#endif

    // Case 4: Fallback - convert everything to FP32
    Tensor A_fp32 = ensure_fp32(A);
    Tensor B_fp32 = ensure_fp32(B);

    // Ensure output is FP32 for computation
    bool need_convert_output = (c_dtype != DType::FP32);
    Tensor C_fp32 = need_convert_output ? Tensor::empty(C.shape(), DType::FP32) : C;

    MlasGemm(
        trans_a ? CblasTrans : CblasNoTrans,
        trans_b ? CblasTrans : CblasNoTrans,
        M, N, K,
        alpha,
        A_fp32.data_ptr<float>(), lda,
        B_fp32.data_ptr<float>(), ldb,
        beta,
        C_fp32.data_ptr<float>(), ldc,
        nullptr
    );

    // Convert output if needed
    if (need_convert_output) {
        Tensor converted = C_fp32.to(c_dtype);
        std::memcpy(C.data(), converted.data(), C.nbytes());
    }
}

void CpuBackend::flash_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& output,
    float scale,
    bool causal
) {
    // Convert inputs to FP32 if needed
    Tensor Q_fp32 = ensure_fp32(Q);
    Tensor K_fp32 = ensure_fp32(K);
    Tensor V_fp32 = ensure_fp32(V);

    assert(Q_fp32.ndim() == 4);
    assert(output.dtype() == DType::FP32);

    int batch_size = static_cast<int>(Q_fp32.dim(0));
    int num_heads = static_cast<int>(Q_fp32.dim(1));
    int q_seq_len = static_cast<int>(Q_fp32.dim(2));
    int head_dim = static_cast<int>(Q_fp32.dim(3));
    int kv_seq_len = static_cast<int>(K_fp32.dim(2));
    int v_head_dim = static_cast<int>(V_fp32.dim(3));

    int q_block_size = std::min(64, q_seq_len);
    int kv_block_size = std::min(64, kv_seq_len);
    const int thread_count = 1;

    const size_t floats_per_thread = static_cast<size_t>(q_block_size) * (2 + kv_block_size + v_head_dim);
    const size_t required_floats = floats_per_thread * static_cast<size_t>(thread_count);

    // MLAS expects a per-thread scratch buffer. Reuse a thread-local buffer to
    // avoid reallocating and zero-initializing scratch memory on every call.
    static thread_local std::vector<float> buffer;
    if (buffer.size() < required_floats) {
        buffer.resize(required_floats);
    }

    MlasFlashAttentionThreadedArgs args{};
    args.batch_size = batch_size;
    args.num_heads = num_heads;
    args.q_sequence_length = q_seq_len;
    args.kv_sequence_length = kv_seq_len;
    args.qk_head_size = head_dim;
    args.v_head_size = v_head_dim;
    args.q_block_size = q_block_size;
    args.kv_block_size = kv_block_size;
    args.scale = scale;
    args.thread_count = thread_count;
    args.buffer = buffer.data();
    args.buffer_size_per_thread = floats_per_thread * sizeof(float);
    args.query = Q_fp32.data_ptr<float>();
    args.key = K_fp32.data_ptr<float>();
    args.value = V_fp32.data_ptr<float>();
    args.output = output.data_ptr<float>();

    MlasFlashAttention(&args, nullptr);

    (void)causal;
}

void CpuBackend::rmsnorm(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps
) {
    // Convert inputs to FP32 if needed
    Tensor input_fp32 = ensure_fp32(input);
    Tensor weight_fp32 = ensure_fp32(weight);

    assert(output.dtype() == DType::FP32);
    assert(input_fp32.is_contiguous() && output.is_contiguous());

    const float* in = input_fp32.data_ptr<float>();
    const float* w = weight_fp32.data_ptr<float>();
    float* out = output.data_ptr<float>();

    size_t hidden_size = input_fp32.dim(input_fp32.ndim() - 1);
    size_t num_tokens = input_fp32.numel() / hidden_size;

    for (size_t t = 0; t < num_tokens; ++t) {
        const float* x = in + t * hidden_size;
        float* y = out + t * hidden_size;

        float sum_sq = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            sum_sq += x[i] * x[i];
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        float inv_rms = 1.0f / rms;

        for (size_t i = 0; i < hidden_size; ++i) {
            y[i] = x[i] * inv_rms * w[i];
        }
    }
}

void CpuBackend::rope(
    Tensor& Q,
    Tensor& K,
    size_t position_offset,
    float theta
) {
    // rope modifies Q and K in-place, they should be FP32
    assert(Q.dtype() == DType::FP32 && K.dtype() == DType::FP32);

    auto apply_rope = [theta, position_offset](Tensor& t) {
        float* data = t.data_ptr<float>();
        size_t batch = t.dim(0);
        size_t seq_len = t.dim(1);
        size_t num_heads = t.dim(2);
        size_t head_dim = t.dim(3);
        size_t half_dim = head_dim / 2;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t pos = position_offset + s;
                for (size_t h = 0; h < num_heads; ++h) {
                    float* head_data = data + (((b * seq_len + s) * num_heads + h) * head_dim);
                    for (size_t i = 0; i < half_dim; ++i) {
                        float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                        float angle = static_cast<float>(pos) * freq;
                        float cos_val = std::cos(angle);
                        float sin_val = std::sin(angle);
                        float x0 = head_data[i];
                        float x1 = head_data[i + half_dim];
                        head_data[i] = x0 * cos_val - x1 * sin_val;
                        head_data[i + half_dim] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }
    };

    apply_rope(Q);
    apply_rope(K);
}

void CpuBackend::silu(
    const Tensor& input,
    Tensor& output
) {
    // Convert input to FP32 if needed
    Tensor input_fp32 = ensure_fp32(input);
    assert(output.dtype() == DType::FP32);

    const float* in = input_fp32.data_ptr<float>();
    float* out = output.data_ptr<float>();
    size_t n = input_fp32.numel();

    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = x / (1.0f + std::exp(-x));
    }
}

void CpuBackend::elementwise_mul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    // Convert inputs to FP32 if needed
    Tensor a_fp32 = ensure_fp32(a);
    Tensor b_fp32 = ensure_fp32(b);

    assert(a_fp32.numel() == b_fp32.numel() && a_fp32.numel() == output.numel());
    assert(output.dtype() == DType::FP32);

    const float* pa = a_fp32.data_ptr<float>();
    const float* pb = b_fp32.data_ptr<float>();
    float* pout = output.data_ptr<float>();
    size_t n = a_fp32.numel();

    for (size_t i = 0; i < n; ++i) {
        pout[i] = pa[i] * pb[i];
    }
}

void CpuBackend::add(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    // Convert inputs to FP32 if needed
    Tensor a_fp32 = ensure_fp32(a);
    Tensor b_fp32 = ensure_fp32(b);

    assert(a_fp32.numel() == b_fp32.numel() && a_fp32.numel() == output.numel());
    assert(output.dtype() == DType::FP32);

    const float* pa = a_fp32.data_ptr<float>();
    const float* pb = b_fp32.data_ptr<float>();
    float* pout = output.data_ptr<float>();
    size_t n = a_fp32.numel();

    for (size_t i = 0; i < n; ++i) {
        pout[i] = pa[i] + pb[i];
    }
}

void CpuBackend::softmax(
    const Tensor& input,
    Tensor& output,
    int dim
) {
    // Convert input to FP32 if needed
    Tensor input_fp32 = ensure_fp32(input);

    assert(output.dtype() == DType::FP32);
    assert(input_fp32.is_contiguous() && output.is_contiguous());

    if (dim < 0) dim = static_cast<int>(input_fp32.ndim()) + dim;
    assert(dim >= 0 && static_cast<size_t>(dim) < input_fp32.ndim());

    const float* in = input_fp32.data_ptr<float>();
    float* out = output.data_ptr<float>();

    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= input_fp32.dim(i);
    }
    size_t dim_size = input_fp32.dim(dim);
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < input_fp32.ndim(); ++i) {
        inner_size *= input_fp32.dim(i);
    }

    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = (o * dim_size + d) * inner_size + inner;
                max_val = std::max(max_val, in[idx]);
            }
            float sum = 0.0f;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = (o * dim_size + d) * inner_size + inner;
                out[idx] = std::exp(in[idx] - max_val);
                sum += out[idx];
            }
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = (o * dim_size + d) * inner_size + inner;
                out[idx] /= sum;
            }
        }
    }
}

void CpuBackend::embedding_lookup(
    const Tensor& weight,
    const std::vector<int>& token_ids,
    Tensor& output
) {
    // Convert weight to FP32 if needed
    Tensor weight_fp32 = ensure_fp32(weight);

    assert(output.dtype() == DType::FP32);
    assert(weight_fp32.ndim() == 2);

    size_t hidden_size = weight_fp32.dim(1);
    const float* w = weight_fp32.data_ptr<float>();
    float* out = output.data_ptr<float>();

    for (size_t t = 0; t < token_ids.size(); ++t) {
        int token_id = token_ids[t];
        const float* emb = w + token_id * hidden_size;
        std::memcpy(out + t * hidden_size, emb, hidden_size * sizeof(float));
    }
}

}  // namespace mruntime
