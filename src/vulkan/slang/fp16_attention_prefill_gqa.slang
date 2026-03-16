// Causal FP16 GQA prefill attention, specialized for head_dim=64.
// Q: [1, num_q_heads, q_len, 64] BHSD FP16
// K: [1, num_kv_heads, kv_stride, 64] BHSD FP16
// V: [1, num_kv_heads, kv_stride, 64] BHSD FP16
// O: [1, num_q_heads, q_len, 64] BHSD FP16
//
// Each workgroup handles BLOCK_M=128 Q rows tiled over the full KV range.
// 1 thread per Q row — no cross-thread reduction needed.
// FP16 accumulation with half4 vectorized dot products.
// Shared Q has +1 stride padding for bank-conflict avoidance;
// K/V are broadcast-read (all threads access same element) so no padding needed.

static const int WG_SIZE = 128;
static const int BLOCK_M = 128;
static const int BLOCK_N = 16;
static const int HEAD_DIM = 64;
static const int HEAD_DIM_V4 = HEAD_DIM / 4;      // 16
static const int QF_STRIDE = HEAD_DIM_V4 + 1;     // 17
static const float NEG_INF = -3.402823466e38;

struct PushConstants {
    uint num_q_heads;
    uint num_kv_heads;
    uint seqlen_q;
    uint seqlen_kv;
    uint num_kv_groups;
    uint kv_stride;
    uint head_dim;
    float scale;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)] StructuredBuffer<half> q_data;
[vk::binding(1, 0)] StructuredBuffer<half> k_data;
[vk::binding(2, 0)] StructuredBuffer<half> v_data;
[vk::binding(3, 0)] RWStructuredBuffer<half> o_data;

groupshared half4 Qf[BLOCK_M * QF_STRIDE];
groupshared half4 shared_k[BLOCK_N * HEAD_DIM_V4];
groupshared half4 shared_v[BLOCK_N * HEAD_DIM_V4];

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID)
{
    const uint tid = local_id.x;
    const uint q_head = group_id.x;
    const uint q_block = group_id.y;

    if (q_head >= pc.num_q_heads) {
        return;
    }

    const uint kv_head = q_head / pc.num_kv_groups;
    const uint q_start = q_block * uint(BLOCK_M);
    if (q_start >= pc.seqlen_q) {
        return;
    }

    const uint cached_kv_len = pc.seqlen_kv - pc.seqlen_q;
    const uint offset_q = q_head * pc.seqlen_q;
    const uint offset_kv = kv_head * pc.kv_stride;
    const uint q_end = min(q_start + uint(BLOCK_M), pc.seqlen_q);
    const uint kv_end = min(pc.seqlen_kv, cached_kv_len + q_end);
    const uint row_global = q_start + tid;

    // ---- Load Q into Qf (pre-scaled, half4, +1 padding) ----
    for (uint idx = tid; idx < uint(BLOCK_M) * uint(HEAD_DIM_V4); idx += uint(WG_SIZE)) {
        const uint r = idx / uint(HEAD_DIM_V4);
        const uint v = idx % uint(HEAD_DIM_V4);
        const uint gr = q_start + r;
        if (gr < pc.seqlen_q) {
            const uint base = (offset_q + gr) * uint(HEAD_DIM) + v * 4u;
            Qf[r * uint(QF_STRIDE) + v] = half4(
                q_data[base + 0u], q_data[base + 1u],
                q_data[base + 2u], q_data[base + 3u]) * half(pc.scale);
        } else {
            Qf[r * uint(QF_STRIDE) + v] = half4(0.0h);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Per-thread accumulators ----
    half4 acc[HEAD_DIM_V4];
    float m_i = NEG_INF;
    float l_i = 0.0;

    [unroll]
    for (uint d = 0u; d < uint(HEAD_DIM_V4); ++d) {
        acc[d] = half4(0.0h);
    }

    // ---- Main KV tile loop ----
    [loop]
    for (uint kv_tile = 0u; kv_tile < kv_end; kv_tile += uint(BLOCK_N)) {

        // Cooperative load K and V
        for (uint idx = tid; idx < uint(BLOCK_N) * uint(HEAD_DIM_V4); idx += uint(WG_SIZE)) {
            const uint col = idx / uint(HEAD_DIM_V4);
            const uint v = idx % uint(HEAD_DIM_V4);
            const uint gkv = kv_tile + col;
            if (gkv < kv_end) {
                const uint base = (offset_kv + gkv) * uint(HEAD_DIM) + v * 4u;
                shared_k[col * uint(HEAD_DIM_V4) + v] = half4(
                    k_data[base + 0u], k_data[base + 1u],
                    k_data[base + 2u], k_data[base + 3u]);
                shared_v[col * uint(HEAD_DIM_V4) + v] = half4(
                    v_data[base + 0u], v_data[base + 1u],
                    v_data[base + 2u], v_data[base + 3u]);
            } else {
                shared_k[col * uint(HEAD_DIM_V4) + v] = half4(0.0h);
                shared_v[col * uint(HEAD_DIM_V4) + v] = half4(0.0h);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (row_global < pc.seqlen_q) {
            // ---- Q·K scores via half4 dot products ----
            half qk[BLOCK_N];
            float block_max = NEG_INF;

            [unroll]
            for (uint kv_row = 0u; kv_row < uint(BLOCK_N); ++kv_row) {
                half score = half(NEG_INF);
                const uint gkv = kv_tile + kv_row;
                if (gkv < kv_end && gkv <= cached_kv_len + row_global) {
                    score = half(0.0h);
                    [unroll]
                    for (uint d = 0u; d < uint(HEAD_DIM_V4); ++d) {
                        score += dot(Qf[tid * uint(QF_STRIDE) + d],
                                     shared_k[kv_row * uint(HEAD_DIM_V4) + d]);
                    }
                }
                qk[kv_row] = score;
                block_max = max(block_max, float(score));
            }

            // ---- Online softmax ----
            const float m_ij = max(m_i, block_max);
            const float eMf = exp(m_i - m_ij);
            float l_ij = 0.0;

            [unroll]
            for (uint kv_row = 0u; kv_row < uint(BLOCK_N); ++kv_row) {
                const float w = exp(float(qk[kv_row]) - m_ij);
                qk[kv_row] = half(w);
                l_ij += w;
            }

            // ---- Rescale + V accumulation ----
            [unroll]
            for (uint d = 0u; d < uint(HEAD_DIM_V4); ++d) {
                half4 pv = half4(0.0h);
                [unroll]
                for (uint kv_row = 0u; kv_row < uint(BLOCK_N); ++kv_row) {
                    pv += qk[kv_row] * shared_v[kv_row * uint(HEAD_DIM_V4) + d];
                }
                acc[d] = half(eMf) * acc[d] + pv;
            }

            l_i = l_i * eMf + l_ij;
            m_i = m_ij;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // ---- Write output ----
    if (row_global < pc.seqlen_q) {
        const half inv_norm = half((l_i > 0.0) ? (1.0 / l_i) : 0.0);
        [unroll]
        for (uint d = 0u; d < uint(HEAD_DIM_V4); ++d) {
            const half4 out_vec = acc[d] * inv_norm;
            const uint o_base = (offset_q + row_global) * uint(HEAD_DIM) + d * 4u;
            o_data[o_base + 0u] = out_vec.x;
            o_data[o_base + 1u] = out_vec.y;
            o_data[o_base + 2u] = out_vec.z;
            o_data[o_base + 3u] = out_vec.w;
        }
    }
}
