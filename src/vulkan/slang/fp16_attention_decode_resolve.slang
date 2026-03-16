// Split-K resolve kernel for decode attention, specialized for head_dim=64.

static const int WG_SIZE = 128;
static const int MAX_SPLIT_K = 4;
static const int HEAD_DIM = 64;
static const int NUM_SUBGROUPS = 4;
static const float NEG_INF = -3.402823466e38;

struct PushConstants {
    uint num_q_heads;
    uint k_num;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)] StructuredBuffer<half> partial_o_data;
[vk::binding(1, 0)] StructuredBuffer<float> partial_stats_data;
[vk::binding(2, 0)] RWStructuredBuffer<half> o_data;

groupshared float split_weights[MAX_SPLIT_K];
groupshared float subgroup_reduce[NUM_SUBGROUPS];
groupshared float shared_scalar;

uint q_offset(uint q_head, uint dim)
{
    return q_head * uint(HEAD_DIM) + dim;
}

uint partial_o_offset(uint split_idx, uint q_head, uint dim)
{
    return (split_idx * pc.num_q_heads + q_head) * uint(HEAD_DIM) + dim;
}

uint partial_stats_offset(uint split_idx, uint q_head, uint slot)
{
    return ((split_idx * pc.num_q_heads + q_head) * 2u) + slot;
}

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
    const uint q_head = group_id.x;
    if (q_head >= pc.num_q_heads) {
        return;
    }
    if (pc.k_num == 0u || pc.k_num > uint(MAX_SPLIT_K)) {
        return;
    }

    const float local_max = (tid < pc.k_num)
        ? partial_stats_data[partial_stats_offset(tid, q_head, 1u)]
        : NEG_INF;
    const float merged_max = broadcast_scalar(reduce_workgroup_max(local_max, tid), tid);

    float local_weight = 0.0;
    if (tid < pc.k_num) {
        const float norm = partial_stats_data[partial_stats_offset(tid, q_head, 0u)];
        const float part_max = partial_stats_data[partial_stats_offset(tid, q_head, 1u)];
        local_weight = norm * exp(part_max - merged_max);
        split_weights[tid] = local_weight;
    }
    GroupMemoryBarrierWithGroupSync();

    const float merged_norm = broadcast_scalar(reduce_workgroup_sum(local_weight, tid), tid);
    const float inv_norm = (merged_norm > 0.0) ? (1.0 / merged_norm) : 0.0;

    if (tid < uint(HEAD_DIM)) {
        float acc = 0.0;
        [loop]
        for (uint split_idx = 0u; split_idx < pc.k_num; ++split_idx) {
            acc += split_weights[split_idx] * float(partial_o_data[partial_o_offset(split_idx, q_head, tid)]);
        }
        o_data[q_offset(q_head, tid)] = half(acc * inv_norm);
    }
}
