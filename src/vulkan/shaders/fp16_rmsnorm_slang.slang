// FP16 RMSNorm with FP32 reduction/accumulation.

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

groupshared float shared_sum[256];

[shader("compute")]
[numthreads(256, 1, 1)]
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
    for (uint i = tid; i < pc.hidden_size; i += 256u) {
        const float val = float(input_data[row_offset + i]);
        partial_sum += val * val;
    }

    shared_sum[tid] = partial_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float rms_inv = rsqrt(shared_sum[0] / float(pc.hidden_size) + pc.eps);

    for (uint i = tid; i < pc.hidden_size; i += 256u) {
        const float val = float(input_data[row_offset + i]);
        const float w = float(weight_data[i]);
        out_data[row_offset + i] = half(val * rms_inv * w);
    }
}
