// FP16 SiLU-and-multiply over interleaved gate/up rows.
//
// gate_up layout per token:
//   [gate (intermediate_size) | up (intermediate_size)]

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
    const uint i = tid.x;
    const uint total = pc.num_tokens * pc.intermediate_size;
    if (i >= total) {
        return;
    }

    const uint token = i / pc.intermediate_size;
    const uint j = i % pc.intermediate_size;
    const uint row_offset = token * 2u * pc.intermediate_size;

    const float gate_val = float(gate_up[row_offset + j]);
    const float up_val = float(gate_up[row_offset + pc.intermediate_size + j]);
    const float silu = gate_val / (1.0 + exp(-gate_val));
    out_data[i] = half(silu * up_val);
}
