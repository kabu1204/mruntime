#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mruntime {

namespace {

inline size_t idx4(const int strides[4], size_t i0, size_t i1, size_t i2, size_t i3) {
    return static_cast<size_t>(strides[0]) * i0 + static_cast<size_t>(strides[1]) * i1 +
           static_cast<size_t>(strides[2]) * i2 + static_cast<size_t>(strides[3]) * i3;
}

void FlashAttentionStridedImpl(
    const FlashAttentionArgs* args,
    size_t task_id,
    size_t task_count,
    float* buffer,
    size_t buffer_size_per_task
) {
    if (args == nullptr || buffer == nullptr || task_count == 0) {
        return;
    }

    const size_t q_block_size = static_cast<size_t>(args->q_block_size);
    const size_t kv_block_size = static_cast<size_t>(args->kv_block_size);
    const size_t batch_size = static_cast<size_t>(args->batch_size);
    const size_t num_heads = static_cast<size_t>(args->num_heads);
    const size_t q_sequence_length = static_cast<size_t>(args->q_sequence_length);
    const size_t kv_sequence_length = static_cast<size_t>(args->kv_sequence_length);
    const size_t qk_head_size = static_cast<size_t>(args->qk_head_size);
    const size_t v_head_size = static_cast<size_t>(args->v_head_size);

    const float* query = args->query;
    const float* key = args->key;
    const float* value = args->value;
    float* output = args->output;

    if (q_block_size == 0 || kv_block_size == 0 || qk_head_size == 0 || v_head_size == 0) {
        return;
    }

    const size_t q_chunk_count = (q_sequence_length + q_block_size - 1) / q_block_size;
    const size_t total_task_count = batch_size * num_heads * q_chunk_count;

    const size_t quotient = total_task_count / task_count;
    const size_t remainder = total_task_count % task_count;

    size_t task_start = 0;
    size_t task_end = 0;
    if (task_id < remainder) {
        task_start = (quotient + 1) * task_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * task_id + remainder;
        task_end = task_start + quotient;
    }

    auto* buffer_current_task = reinterpret_cast<std::byte*>(buffer) + task_id * buffer_size_per_task;
    float* l = reinterpret_cast<float*>(buffer_current_task);
    float* m = l + q_block_size;
    float* intermediate = m + q_block_size;
    float* temp_output = intermediate + q_block_size * kv_block_size;

    for (size_t i = 0; i < q_block_size; ++i) {
        l[i] = 0.0f;
        m[i] = -std::numeric_limits<float>::infinity();
    }
    std::fill(temp_output, temp_output + (q_block_size * v_head_size), 0.0f);

    for (size_t task_index = task_start; task_index < task_end; ++task_index) {
        size_t t = task_index;
        const size_t q_idx = (t % q_chunk_count) * q_block_size;
        t /= q_chunk_count;
        const size_t head_idx = t % num_heads;
        const size_t batch_idx = t / num_heads;

        const size_t row_size_q = std::min(q_block_size, q_sequence_length - q_idx);

        // Reset per-chunk stats.
        for (size_t i = 0; i < row_size_q; ++i) {
            l[i] = 0.0f;
            m[i] = -std::numeric_limits<float>::infinity();
        }
        std::fill(temp_output, temp_output + (row_size_q * v_head_size), 0.0f);

        for (size_t ir = 0; ir < kv_sequence_length; ir += kv_block_size) {
            const size_t row_size_kv = std::min(kv_block_size, kv_sequence_length - ir);

            // Compute scores S = scale * Q * K^T.
            for (size_t irow = 0; irow < row_size_q; ++irow) {
                float* p = intermediate + irow * row_size_kv;
                for (size_t j = 0; j < row_size_kv; ++j) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < qk_head_size; ++d) {
                        const size_t q_off = idx4(args->strides_q, batch_idx, head_idx, q_idx + irow, d);
                        const size_t k_off = idx4(args->strides_k, batch_idx, head_idx, ir + j, d);
                        dot += query[q_off] * key[k_off];
                    }
                    p[j] = dot * args->scale;
                }
            }

            // Online softmax update.
            for (size_t irow = 0; irow < row_size_q; ++irow) {
                float* p = intermediate + irow * row_size_kv;

                float rowmax = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < row_size_kv; ++j) {
                    rowmax = std::max(rowmax, p[j]);
                }

                const float old_m = m[irow];
                const float new_m = std::max(old_m, rowmax);
                const float exp_diff = std::exp(old_m - new_m);
                m[irow] = new_m;

                float rowsum = 0.0f;
                for (size_t j = 0; j < row_size_kv; ++j) {
                    const float e = std::exp(p[j] - new_m);
                    p[j] = e;
                    rowsum += e;
                }

                l[irow] = exp_diff * l[irow] + rowsum;

                float* out_row = temp_output + irow * v_head_size;
                for (size_t d = 0; d < v_head_size; ++d) {
                    out_row[d] *= exp_diff;
                }
            }

            // Accumulate: O += S * V
            for (size_t irow = 0; irow < row_size_q; ++irow) {
                const float* p = intermediate + irow * row_size_kv;
                float* out_row = temp_output + irow * v_head_size;
                for (size_t j = 0; j < row_size_kv; ++j) {
                    const float w = p[j];
                    for (size_t d = 0; d < v_head_size; ++d) {
                        const size_t v_off = idx4(args->strides_v, batch_idx, head_idx, ir + j, d);
                        out_row[d] += w * value[v_off];
                    }
                }
            }
        }

        // Write output: O = temp_output / l.
        for (size_t irow = 0; irow < row_size_q; ++irow) {
            const float inv_l = 1.0f / l[irow];
            for (size_t d = 0; d < v_head_size; ++d) {
                const size_t out_off = idx4(args->strides_out, batch_idx, head_idx, q_idx + irow, d);
                output[out_off] = temp_output[irow * v_head_size + d] * inv_l;
            }
        }
    }
}

}  // namespace

void FlashAttentionStrided(void* argptr, size_t task_id) {
    auto* ctx = static_cast<FlashAttentionStridedContext*>(argptr);
    if (ctx == nullptr) {
        return;
    }

    FlashAttentionStridedImpl(ctx->args, task_id, ctx->task_count, ctx->buffer, ctx->buffer_size_per_task);
}


}  // namespace mruntime
