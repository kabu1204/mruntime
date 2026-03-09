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
ByteAddressBuffer input_data;

[vk::binding(1, 0)]
ByteAddressBuffer weight_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> out_data;

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
    const uint row_offset = token * pc.hidden_size;

    float partial_sum = 0.0;
    const uint vectorized_end = pc.hidden_size & ~7u;
    const uint chunk = 8u;
    const uint chunk_stride = uint(WG_SIZE) * chunk;

    for (uint i = tid * chunk; i < vectorized_end; i += chunk_stride) {
        const FloatPack8 values = Thread::load_fp16_pack8(input_data, row_offset + i);
        partial_sum += dot(values.lo, values.lo);
        partial_sum += dot(values.hi, values.hi);
    }

    for (uint i = vectorized_end + tid; i < pc.hidden_size; i += uint(WG_SIZE)) {
        const float value = Thread::load_fp16(input_data, row_offset + i);
        partial_sum += value * value;
    }

    const float block_sum = Workgroup::broadcast_scalar<SharedSumStorage>(
        Workgroup::leader_reduce_sum<SharedSumStorage, WG_SIZE>(partial_sum, tid),
        tid);

    const float rms_inv = rsqrt(block_sum / float(pc.hidden_size) + pc.eps);

    for (uint i = tid * chunk; i < vectorized_end; i += chunk_stride) {
        const FloatPack8 values = Thread::load_fp16_pack8(input_data, row_offset + i);
        const FloatPack8 weights = Thread::load_fp16_pack8(weight_data, i);

        FloatPack8 outputs;
        outputs.lo = values.lo * weights.lo * rms_inv;
        outputs.hi = values.hi * weights.hi * rms_inv;
        Thread::scatter_contiguous_values8(out_data, row_offset + i, 8u, outputs);
    }

    for (uint i = vectorized_end + tid; i < pc.hidden_size; i += uint(WG_SIZE)) {
        const float value = Thread::load_fp16(input_data, row_offset + i);
        const float weight = Thread::load_fp16(weight_data, i);
        out_data[row_offset + i] = half(value * rms_inv * weight);
    }
}
