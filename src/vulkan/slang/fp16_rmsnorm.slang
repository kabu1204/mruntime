// FP16 RMSNorm vec4 kernel for hidden sizes divisible by 4.

import scope_primitives;
import storage_traits;

static const int WG_SIZE = 64;

struct PushConstants {
    uint hidden_size;
    uint num_tokens;
    float eps;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half4> input_data;

[vk::binding(1, 0)]
StructuredBuffer<half4> weight_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half4> out_data;

groupshared float shared_sum[WG_SIZE];

public struct SharedSumStorage : IFloatStorage1D
{
    public static const uint kExtent = WG_SIZE;
    public static const StorageScope kScope = StorageScope::Workgroup;
    public static const bool kBarrierSafe = true;

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
    const uint hidden4 = pc.hidden_size >> 2u;
    const uint row_offset4 = token * hidden4;

    float partial_sum = 0.0;
    for (uint index4 = tid; index4 < hidden4; index4 += uint(WG_SIZE)) {
        const float4 value = float4(input_data[row_offset4 + index4]);
        partial_sum += dot(value, value);
    }

    const float block_sum = Workgroup::broadcast_scalar<SharedSumStorage>(
        Workgroup::leader_reduce_sum<SharedSumStorage, WG_SIZE>(partial_sum, tid),
        tid);

    const float rms_inv = rsqrt(block_sum / float(pc.hidden_size) + pc.eps);

    for (uint index4 = tid; index4 < hidden4; index4 += uint(WG_SIZE)) {
        const float4 value = float4(input_data[row_offset4 + index4]);
        const float4 weight = float4(weight_data[index4]);
        out_data[row_offset4 + index4] = half4(value * weight * rms_inv);
    }
}
