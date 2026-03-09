// FP16 SiLU-and-multiply over interleaved gate/up rows.
//
// gate_up layout per token:
//   [gate (intermediate_size) | up (intermediate_size)]

import layout_traits;
import pointwise_ops;
import scope_primitives;
import tile_views;

struct PushConstants {
    uint intermediate_size;
    uint num_tokens;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
ByteAddressBuffer gate_up;

[vk::binding(1, 0)]
RWStructuredBuffer<half> out_data;

public struct SiLUOp : IPointwiseOp
{
    public static float apply(float value)
    {
        return value / (1.0 + exp(-value));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    GlobalTensor2D gate_up_tensor = {pc.num_tokens, 2u * pc.intermediate_size, 2u * pc.intermediate_size};

    const uint chunk_id = Thread::linear_id(tid);
    const uint base_index = chunk_id * 8u;
    const uint total = pc.num_tokens * pc.intermediate_size;
    if (base_index >= total) {
        return;
    }

    const uint token = base_index / pc.intermediate_size;
    const uint element = base_index % pc.intermediate_size;
    if (element + 8u <= pc.intermediate_size && base_index + 8u <= total) {
        const FloatPack8 gate = Thread::load_fp16_pack8(
            gate_up,
            RowMajorTensorLayout::offset(gate_up_tensor, token, element));
        const FloatPack8 up = Thread::load_fp16_pack8(
            gate_up,
            RowMajorTensorLayout::offset(gate_up_tensor, token, pc.intermediate_size + element));

        FloatPack8 values;
        values.lo = Thread::map_values4<SiLUOp>(gate.lo) * up.lo;
        values.hi = Thread::map_values4<SiLUOp>(gate.hi) * up.hi;
        Thread::scatter_contiguous_values8(out_data, base_index, 8u, values);
        return;
    }

    for (uint lane = 0u; base_index + lane < total; ++lane) {
        const uint index = base_index + lane;
        const uint lane_token = index / pc.intermediate_size;
        const uint lane_element = index % pc.intermediate_size;
        const float gate = Thread::load_fp16(
            gate_up,
            RowMajorTensorLayout::offset(gate_up_tensor, lane_token, lane_element));
        const float up = Thread::load_fp16(
            gate_up,
            RowMajorTensorLayout::offset(gate_up_tensor, lane_token, pc.intermediate_size + lane_element));
        out_data[index] = half(SiLUOp::apply(gate) * up);
    }
}
