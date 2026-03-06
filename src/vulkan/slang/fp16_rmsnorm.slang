// FP16 RMSNorm with FP32 reduction/accumulation.

import scope_primitives;
import storage_traits;

static const int WG_SIZE = 256;

struct PushConstants {
    uint hidden_size;
    uint num_tokens;
    float eps;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half> input_data;

[vk::binding(1, 0)]
StructuredBuffer<half> weight_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> out_data;

groupshared float shared_sum[WG_SIZE];

public struct SharedSumStorage : IFloatStorage1D
{
    public static void store(uint index, float value)
    {
        shared_sum[index] = value;
    }

    public static float load(uint index)
    {
        return shared_sum[index];
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    const uint token = group_id.x;
    if (token >= pc.num_tokens) {
        return;
    }

    const uint tid = local_id.x;
    const uint row_offset = token * pc.hidden_size;

    float partial_sum = 0.0;
    for (uint i = tid; i < pc.hidden_size; i += uint(WG_SIZE)) {
        const float value = float(input_data[row_offset + i]);
        partial_sum += value * value;
    }

    const float wave_sum = Subgroup::reduce_sum(partial_sum);
    const float block_sum = Workgroup::tree_reduce_sum<SharedSumStorage, WG_SIZE>(wave_sum, tid);

    if (tid == 0u) {
        SharedSumStorage::store(0u, block_sum);
    }
    GroupMemoryBarrierWithGroupSync();

    const float rms_inv = rsqrt(SharedSumStorage::load(0u) / float(pc.hidden_size) + pc.eps);

    for (uint i = tid; i < pc.hidden_size; i += uint(WG_SIZE)) {
        const float value = float(input_data[row_offset + i]);
        const float weight = float(weight_data[i]);
        out_data[row_offset + i] = half(value * rms_inv * weight);
    }
}
