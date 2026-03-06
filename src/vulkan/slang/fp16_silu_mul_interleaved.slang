// FP16 SiLU-and-multiply over interleaved gate/up rows.
//
// gate_up layout per token:
//   [gate (intermediate_size) | up (intermediate_size)]

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
StructuredBuffer<half> gate_up;

[vk::binding(1, 0)]
RWStructuredBuffer<half> out_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    GateUpInterleavedTensor gate_up_tensor = {pc.num_tokens, pc.intermediate_size};
    Thread::apply_interleaved_gate_up<SiLUOp>(gate_up, out_data, gate_up_tensor, Thread::linear_id(tid));
}
