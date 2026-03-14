// Causal FP16 GQA prefill attention.
// Q: [1, num_q_heads, q_len, head_dim] BHSD FP16
// K: [1, num_kv_heads, kv_stride, head_dim] BHSD FP16
// V: [1, num_kv_heads, kv_stride, head_dim] BHSD FP16
// O: [1, num_q_heads, q_len, head_dim] BHSD FP16

static const int WG_SIZE = 128;
static const int BLOCK_M = 128;  /* partition Q sequence length */
static const int BLOCK_N = 16;  /* partition KV sequence length */
static const int ROWS_PER_THREAD = BLOCK_M / WG_SIZE;
static const int HEAD_DIM = 64;
static const float NEG_INF = -3.402823466e38;

struct PushConstants {
    uint num_q_heads;
    uint num_kv_heads;
    uint seqlen_q;      /* chunk size of chunked prefill */
    uint seqlen_kv;     /* prompt-len */
    uint num_kv_groups;
    uint kv_stride;  /* max_seq_len */
    uint head_dim;
    float scale;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half> q_data;

[vk::binding(1, 0)]
StructuredBuffer<half> k_data;

[vk::binding(2, 0)]
StructuredBuffer<half> v_data;

[vk::binding(3, 0)]
RWStructuredBuffer<half> o_data;

groupshared half shared_q[BLOCK_M * HEAD_DIM];
groupshared half shared_k[BLOCK_N * HEAD_DIM];
groupshared half shared_v[BLOCK_N * HEAD_DIM];

uint q_index(uint row, uint dim)
{
    return row * uint(HEAD_DIM) + dim;
}

uint kv_index(uint row, uint dim)
{
    return row * uint(HEAD_DIM) + dim;
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID)
{
    const uint tid = local_id.x; /* 0 - WG_SIZE-1 */
    const uint q_head_idx = group_id.x;
    const uint q_block_idx = group_id.y;

    if (q_head_idx >= pc.num_q_heads) {
        return;
    }
    if (pc.head_dim != uint(HEAD_DIM)) {
        return;
    }

    const uint kv_head_idx = q_head_idx / pc.num_kv_groups;
    const uint q_start = q_block_idx * uint(BLOCK_M);
    if (q_start >= pc.seqlen_q) {
        return;
    }

    const uint cached_kv_len = pc.seqlen_kv - pc.seqlen_q;
    const uint offset_q_bh = q_head_idx * pc.seqlen_q;
    const uint offset_kv_bh = kv_head_idx * pc.kv_stride;

    for (uint idx = tid; idx < uint(BLOCK_M) * HEAD_DIM; idx += uint(WG_SIZE)) {
        const uint row = idx / HEAD_DIM;
        const uint col = idx % HEAD_DIM;
        const uint global_row = q_start + row;
        if (global_row < pc.seqlen_q) {
            shared_q[q_index(row, col)] = q_data[(offset_q_bh + global_row) * HEAD_DIM + col];
        } else {
            shared_q[q_index(row, col)] = half(0.0);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float m_i[ROWS_PER_THREAD];
    float l_i[ROWS_PER_THREAD];
    float acc[ROWS_PER_THREAD][HEAD_DIM];

    [unroll]
    for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        m_i[r] = NEG_INF;
        l_i[r] = 0.0;
        [unroll]
        for (uint dim = 0; dim < HEAD_DIM; ++dim) {
            acc[r][dim] = 0.0;
        }
    }

    [loop]
    for (uint kv_start = 0; kv_start < pc.seqlen_kv; kv_start += uint(BLOCK_N)) {
        for (uint idx = tid; idx < uint(BLOCK_N) * HEAD_DIM; idx += uint(WG_SIZE)) {
            const uint row = idx / HEAD_DIM;
            const uint col = idx % HEAD_DIM;
            const uint global_row = kv_start + row;
            if (global_row < pc.seqlen_kv) {
                shared_k[kv_index(row, col)] = k_data[(offset_kv_bh + global_row) * HEAD_DIM + col];
                shared_v[kv_index(row, col)] = v_data[(offset_kv_bh + global_row) * HEAD_DIM + col];
            } else {
                shared_k[kv_index(row, col)] = half(0.0);
                shared_v[kv_index(row, col)] = half(0.0);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            const uint row_local = r * uint(WG_SIZE) + tid;
            const uint row_global = q_start + row_local;

            if (row_global >= pc.seqlen_q) {
                continue;
            }

            float qk[BLOCK_N];
            float p[BLOCK_N];
            float acc_o[HEAD_DIM];
            [unroll]
            for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                acc_o[dim] = 0.0;
            }

            float block_max = NEG_INF;
            [unroll]
            for (uint kv_row = 0; kv_row < BLOCK_N; ++kv_row) {
                float score = NEG_INF;
                if (kv_start + kv_row <= cached_kv_len + row_global) {
                    score = 0.0;
                    [unroll]
                    for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                        score += float(shared_q[q_index(row_local, dim)]) * float(shared_k[kv_index(kv_row, dim)]);
                    }
                    score *= pc.scale;
                }
                qk[kv_row] = score;
                block_max = max(block_max, score);
            }

            const float m_ij = max(m_i[r], block_max);
            const float alpha = (l_i[r] > 0.0) ? exp(m_i[r] - m_ij) : 0.0;
            float l_ij = 0.0;

            [unroll]
            for (uint kv_row = 0; kv_row < BLOCK_N; ++kv_row) {
                const float weight = exp(qk[kv_row] - m_ij);
                p[kv_row] = weight;
                l_ij += weight;
            }

            [unroll]
            for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                float pv = 0.0;
                [unroll]
                for (uint kv_row = 0; kv_row < BLOCK_N; ++kv_row) {
                    pv += p[kv_row] * float(shared_v[kv_index(kv_row, dim)]);
                }
                acc_o[dim] = pv;
            }

            [unroll]
            for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                acc[r][dim] = acc[r][dim] * alpha + acc_o[dim];
            }
            l_i[r] = l_i[r] * alpha + l_ij;
            m_i[r] = m_ij;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        const uint row_local = r * uint(WG_SIZE) + tid;
        const uint row_global = q_start + row_local;
        if (row_global >= pc.seqlen_q) {
            continue;
        }

        const float inv_norm = (l_i[r] > 0.0) ? (1.0 / l_i[r]) : 0.0;
        [unroll]
        for (uint dim = 0; dim < HEAD_DIM; ++dim) {
            o_data[(offset_q_bh + row_global) * HEAD_DIM + dim] = half(acc[r][dim] * inv_norm);
        }
    }
}
