// Decode-only FP16 GQA attention, specialized for head_dim=64.
// Q: [1, num_q_heads, 1, 64] BHSD FP16
// K: [1, num_kv_heads, kv_stride, 64] BHSD FP16
// V: [1, num_kv_heads, kv_stride, 64] BHSD FP16
// O: [1, num_q_heads, 1, 64] BHSD FP16
//
// FP16 intermediate path: accumulators, Q staging, scores all in half.
// Only running_max/running_norm kept in float for softmax stability.

static const int WG_SIZE = 128;
static const int BLOCK_N = 32;
static const int D_SPLIT = 4;
static const int HEAD_DIM = 64;
static const int HEAD_DIM_V4 = HEAD_DIM / 4;        // 16
static const int KV_STRIDE = HEAD_DIM_V4 + 1;       // 17, +1 for bank-conflict avoidance
static const int OUT_VECS = HEAD_DIM_V4 / D_SPLIT;  // 4
static const int NUM_SUBGROUPS = 4;                  // WG_SIZE / subgroup_size(32)
static const float NEG_INF = -3.402823466e38;

struct PushConstants {
    uint num_q_heads;
    uint num_kv_heads;
    uint kv_len;
    uint kv_stride;
    float scale;
    uint k_num;
    uint split_kv;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)] StructuredBuffer<half> q_data;
[vk::binding(1, 0)] StructuredBuffer<half> k_data;
[vk::binding(2, 0)] StructuredBuffer<half> v_data;
[vk::binding(3, 0)] RWStructuredBuffer<half> partial_o_data;
[vk::binding(4, 0)] RWStructuredBuffer<float> partial_stats_data;
[vk::binding(5, 0)] RWStructuredBuffer<half> o_data;

// --- Shared memory ---
// Q row pre-scaled to half4 (no padding needed — all threads broadcast-read same index)
groupshared half4 shared_q[HEAD_DIM_V4];
// K/V tile with +1 stride padding for bank-conflict avoidance
groupshared half4 shared_kv[BLOCK_N * KV_STRIDE];
// Per-column attention scores (float for exp/softmax precision)
groupshared float shared_scores[BLOCK_N];
// Cross-subgroup reduction scratch (only NUM_SUBGROUPS entries)
groupshared float subgroup_reduce[NUM_SUBGROUPS];
// Scalar broadcast
groupshared float shared_scalar;

uint q_offset(uint q_head, uint dim)
{
    return q_head * uint(HEAD_DIM) + dim;
}

uint kv_offset(uint kv_head, uint kv_seq, uint dim)
{
    return (kv_head * pc.kv_stride + kv_seq) * uint(HEAD_DIM) + dim;
}

uint partial_o_offset(uint split_idx, uint q_head, uint dim)
{
    return (split_idx * pc.num_q_heads + q_head) * uint(HEAD_DIM) + dim;
}

uint partial_stats_offset(uint split_idx, uint q_head, uint slot)
{
    return ((split_idx * pc.num_q_heads + q_head) * 2u) + slot;
}

// Workgroup-wide max using subgroup intrinsics + shared memory.
float reduce_workgroup_max(float local_value, uint tid)
{
    const float wave_max = WaveActiveMax(local_value);
    if (WaveGetLaneCount() == uint(WG_SIZE)) {
        return (tid == 0u) ? wave_max : NEG_INF;
    }

    const uint sg_id = tid / WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u) {
        subgroup_reduce[sg_id] = wave_max;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid != 0u) {
        return NEG_INF;
    }
    float total = subgroup_reduce[0u];
    [unroll]
    for (uint s = 1u; s < uint(NUM_SUBGROUPS); ++s) {
        total = max(total, subgroup_reduce[s]);
    }
    return total;
}

// Workgroup-wide sum using subgroup intrinsics + shared memory.
float reduce_workgroup_sum(float local_value, uint tid)
{
    const float wave_sum = WaveActiveSum(local_value);
    if (WaveGetLaneCount() == uint(WG_SIZE)) {
        return (tid == 0u) ? wave_sum : 0.0;
    }

    const uint sg_id = tid / WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u) {
        subgroup_reduce[sg_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid != 0u) {
        return 0.0;
    }
    float total = 0.0;
    [unroll]
    for (uint s = 0u; s < uint(NUM_SUBGROUPS); ++s) {
        total += subgroup_reduce[s];
    }
    return total;
}

float broadcast_scalar(float value, uint tid)
{
    if (tid == 0u) {
        shared_scalar = value;
    }
    GroupMemoryBarrierWithGroupSync();
    return shared_scalar;
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID)
{
    const uint tid = local_id.x;
    const uint d_tid = tid % uint(D_SPLIT);
    const uint col_tid = tid / uint(D_SPLIT);
    const uint split_count = max(pc.k_num, 1u);
    const uint dispatch_index = group_id.x;
    const uint q_head = (split_count > 1u) ? (dispatch_index / split_count) : dispatch_index;
    const uint split_idx = (split_count > 1u) ? (dispatch_index % split_count) : 0u;

    if (q_head >= pc.num_q_heads) {
        return;
    }

    const uint heads_per_kv = pc.num_q_heads / pc.num_kv_heads;
    const uint kv_head = q_head / heads_per_kv;
    const uint kv_begin = (split_count > 1u) ? min(pc.kv_len, split_idx * pc.split_kv) : 0u;
    const uint kv_end = (split_count > 1u) ? min(pc.kv_len, kv_begin + pc.split_kv) : pc.kv_len;

    // Load Q row into shared memory as pre-scaled half4.
    if (tid < uint(HEAD_DIM_V4)) {
        const uint dim = tid * 4u;
        const uint base = q_offset(q_head, dim);
        shared_q[tid] = half4(
            q_data[base + 0u],
            q_data[base + 1u],
            q_data[base + 2u],
            q_data[base + 3u]) * half(pc.scale);
    }
    GroupMemoryBarrierWithGroupSync();

    // FP16 output accumulator — 4 half4 vectors per thread.
    half4 accum[OUT_VECS];
    [unroll]
    for (uint i = 0u; i < uint(OUT_VECS); ++i) {
        accum[i] = half4(0.0h);
    }

    float running_max = NEG_INF;
    float running_norm = 0.0;

    [loop]
    for (uint tile_base = kv_begin; tile_base < kv_end; tile_base += uint(BLOCK_N)) {
        const uint tile_count = min(uint(BLOCK_N), kv_end - tile_base);

        // ---- Load K tile into shared_kv (with +1 stride padding) ----
        for (uint idx = tid; idx < tile_count * uint(HEAD_DIM_V4); idx += uint(WG_SIZE)) {
            const uint col = idx / uint(HEAD_DIM_V4);
            const uint vec_idx = idx % uint(HEAD_DIM_V4);
            const uint dim = vec_idx * 4u;
            const uint base = kv_offset(kv_head, tile_base + col, dim);
            shared_kv[col * uint(KV_STRIDE) + vec_idx] = half4(
                k_data[base + 0u],
                k_data[base + 1u],
                k_data[base + 2u],
                k_data[base + 3u]);
        }
        GroupMemoryBarrierWithGroupSync();

        // ---- Compute Q·K scores in half, reduced across D_SPLIT ----
        if (col_tid < tile_count) {
            half score = half(0.0h);
            [unroll]
            for (uint v = d_tid; v < uint(HEAD_DIM_V4); v += uint(D_SPLIT)) {
                score += dot(shared_q[v],
                             shared_kv[col_tid * uint(KV_STRIDE) + v]);
            }
            // Butterfly reduction across D_SPLIT lanes
            [unroll]
            for (uint shift = uint(D_SPLIT >> 1); shift > 0u; shift >>= 1u) {
                score += WaveReadLaneAt(score, WaveGetLaneIndex() ^ shift);
            }
            if (d_tid == 0u) {
                shared_scores[col_tid] = float(score);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ---- Online softmax: max ----
        const float tile_max = broadcast_scalar(
            reduce_workgroup_max(
                (d_tid == 0u && col_tid < tile_count)
                    ? shared_scores[col_tid]
                    : NEG_INF,
                tid),
            tid);

        // ---- Online softmax: exp & sum ----
        if (d_tid == 0u && col_tid < tile_count) {
            shared_scores[col_tid] = exp(shared_scores[col_tid] - tile_max);
        }
        GroupMemoryBarrierWithGroupSync();

        const float tile_norm = broadcast_scalar(
            reduce_workgroup_sum(
                (d_tid == 0u && col_tid < tile_count)
                    ? shared_scores[col_tid]
                    : 0.0,
                tid),
            tid);

        // ---- Fused softmax correction: pre-scale accumulator ----
        const float merged_max = max(running_max, tile_max);
        const float running_scale = (running_norm > 0.0) ? exp(running_max - merged_max) : 0.0;
        const float tile_scale = exp(tile_max - merged_max);

        [unroll]
        for (uint s = 0u; s < uint(OUT_VECS); ++s) {
            accum[s] = half(running_scale) * accum[s];
        }

        // ---- Load V tile into shared_kv (reuse buffer) ----
        for (uint idx = tid; idx < tile_count * uint(HEAD_DIM_V4); idx += uint(WG_SIZE)) {
            const uint col = idx / uint(HEAD_DIM_V4);
            const uint vec_idx = idx % uint(HEAD_DIM_V4);
            const uint dim = vec_idx * 4u;
            const uint base = kv_offset(kv_head, tile_base + col, dim);
            shared_kv[col * uint(KV_STRIDE) + vec_idx] = half4(
                v_data[base + 0u],
                v_data[base + 1u],
                v_data[base + 2u],
                v_data[base + 3u]);
        }
        GroupMemoryBarrierWithGroupSync();

        // ---- Accumulate weighted V directly into accum (fused, no tile_accum) ----
        const half h_tile_scale = half(tile_scale);
        [loop]
        for (uint col = 0u; col < tile_count; ++col) {
            const half weight = half(shared_scores[col]) * h_tile_scale;
            uint slot = 0u;
            [unroll]
            for (uint v = d_tid; v < uint(HEAD_DIM_V4); v += uint(D_SPLIT)) {
                accum[slot] += weight * shared_kv[col * uint(KV_STRIDE) + v];
                ++slot;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        running_norm = running_norm * running_scale + tile_norm * tile_scale;
        running_max = merged_max;
    }

    // ---- Write output ----
    const half inv_norm = half((running_norm > 0.0) ? (1.0 / running_norm) : 0.0);

    if (col_tid == 0u) {
        uint slot = 0u;
        [unroll]
        for (uint v = d_tid; v < uint(HEAD_DIM_V4); v += uint(D_SPLIT)) {
            const uint dim = v * 4u;
            const half4 out_vec = accum[slot] * inv_norm;
            if (split_count > 1u) {
                partial_o_data[partial_o_offset(split_idx, q_head, dim + 0u)] = out_vec.x;
                partial_o_data[partial_o_offset(split_idx, q_head, dim + 1u)] = out_vec.y;
                partial_o_data[partial_o_offset(split_idx, q_head, dim + 2u)] = out_vec.z;
                partial_o_data[partial_o_offset(split_idx, q_head, dim + 3u)] = out_vec.w;
            } else {
                o_data[q_offset(q_head, dim + 0u)] = out_vec.x;
                o_data[q_offset(q_head, dim + 1u)] = out_vec.y;
                o_data[q_offset(q_head, dim + 2u)] = out_vec.z;
                o_data[q_offset(q_head, dim + 3u)] = out_vec.w;
            }
            ++slot;
        }

        if (split_count > 1u && d_tid == 0u) {
            partial_stats_data[partial_stats_offset(split_idx, q_head, 0u)] = running_norm;
            partial_stats_data[partial_stats_offset(split_idx, q_head, 1u)] = running_max;
        }
    }
}
