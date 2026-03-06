// Causal FP16 GQA prefill attention.
// Q: [1, num_q_heads, q_len, head_dim] BHSD FP16
// K: [1, num_kv_heads, kv_stride, head_dim] BHSD FP16
// V: [1, num_kv_heads, kv_stride, head_dim] BHSD FP16
// O: [1, num_q_heads, q_len, head_dim] BHSD FP16

import layout_traits;
import scope_primitives;
import storage_traits;
import tile_views;

static const int WG_SIZE = 128;
static const int SCORE_TILE = 32;
static const int MAX_HEAD_DIM = 512;
static const float NEG_INF = -3.402823466e38;

struct PushConstants {
    uint num_q_heads;
    uint num_kv_heads;
    uint q_len;
    uint kv_len;
    uint kv_stride;
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

groupshared float shared_q[MAX_HEAD_DIM];
groupshared float score_tile[SCORE_TILE];
groupshared float subgroup_reduce[32];

public struct SharedQueryStorage : IFloatStorage1D
{
    public static void store(uint index, float value)
    {
        shared_q[index] = value;
    }

    public static float load(uint index)
    {
        return shared_q[index];
    }
}

public struct SharedScoreTileStorage : IFloatStorage1D
{
    public static void store(uint index, float value)
    {
        score_tile[index] = value;
    }

    public static float load(uint index)
    {
        return score_tile[index];
    }
}

public struct SharedScalarStorage : IFloatStorage1D
{
    public static void store(uint index, float value)
    {
        subgroup_reduce[index] = value;
    }

    public static float load(uint index)
    {
        return subgroup_reduce[index];
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID)
{
    const uint tid = local_id.x;
    const uint q_head = group_id.x;
    const uint q_seq = group_id.y;

    BHSDTensor q_tensor = {1u, pc.num_q_heads, pc.q_len, pc.head_dim};
    BHSDTensor kv_tensor = {1u, pc.num_kv_heads, pc.kv_stride, pc.head_dim};
    BHSDTensor o_tensor = q_tensor;

    if (q_head >= q_tensor.heads || q_seq >= q_tensor.seq_len) {
        return;
    }

    const uint heads_per_kv = q_tensor.heads / kv_tensor.heads;
    const uint kv_head = q_head / heads_per_kv;
    const uint q_base = BHSDLayout::offset(q_tensor, 0u, q_head, q_seq, 0u);
    const uint o_base = BHSDLayout::offset(o_tensor, 0u, q_head, q_seq, 0u);
    const uint base_position = pc.kv_len - pc.q_len;
    const uint attended_kv_len = base_position + q_seq + 1u;
    const VectorSegment4 segment = Workgroup::vector_segment4<WG_SIZE>(tid);

    Workgroup::load_strided_slice<SharedQueryStorage, WG_SIZE>(q_data, q_base, 1u, q_tensor.head_dim, tid);
    GroupMemoryBarrierWithGroupSync();

    float max_score = NEG_INF;
    for (uint seq_base = 0u; seq_base < attended_kv_len; seq_base += SCORE_TILE) {
        const uint seq_count = min(uint(SCORE_TILE), attended_kv_len - seq_base);
        if (tid < seq_count) {
            const uint k_base = BHSDLayout::offset(kv_tensor, 0u, kv_head, seq_base + tid, 0u);
            max_score = max(
                max_score,
                Thread::scaled_dot_strided_slice<SharedQueryStorage>(
                    k_data,
                    k_base,
                    1u,
                    kv_tensor.head_dim,
                    pc.scale));
        }
    }

    const float stabilized_max = Workgroup::broadcast_scalar<SharedScalarStorage>(
        Workgroup::reduce_max<SharedScalarStorage, WG_SIZE>(max_score, tid),
        tid);
    float norm_sum = 0.0;
    float4 accum = float4(0.0);

    for (uint seq_base = 0u; seq_base < attended_kv_len; seq_base += SCORE_TILE) {
        const uint seq_count = min(uint(SCORE_TILE), attended_kv_len - seq_base);
        if (tid < seq_count) {
            const uint k_base = BHSDLayout::offset(kv_tensor, 0u, kv_head, seq_base + tid, 0u);
            norm_sum += Thread::store_exp_of_scaled_dot<SharedQueryStorage, SharedScoreTileStorage>(
                k_data,
                k_base,
                1u,
                kv_tensor.head_dim,
                pc.scale,
                stabilized_max,
                tid);
        }
        GroupMemoryBarrierWithGroupSync();

        const uint v_base = BHSDLayout::offset(kv_tensor, 0u, kv_head, seq_base, 0u);
        Thread::accumulate_weighted_strided_slices4<SharedScoreTileStorage>(
            accum,
            v_data,
            v_base,
            kv_tensor.head_dim,
            1u,
            seq_count,
            kv_tensor.head_dim,
            segment);
        GroupMemoryBarrierWithGroupSync();
    }

    const float total_norm = Workgroup::broadcast_scalar<SharedScalarStorage>(
        Workgroup::reduce_sum<SharedScalarStorage, WG_SIZE>(norm_sum, tid),
        tid);

    const float inv_norm = (total_norm > 0.0) ? (1.0 / total_norm) : 0.0;
    Thread::store_strided_values4(o_data, o_base, 1u, o_tensor.head_dim, segment, accum * inv_norm);
}
