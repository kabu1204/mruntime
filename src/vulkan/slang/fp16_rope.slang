// FP16 RoPE on Q/K with FP32 cos/sin lookup table.
//
// q layout: [batch, seq_len, num_q_heads, head_dim]
// k layout: [batch, seq_len, num_kv_heads, head_dim]
// rope_cos_sin layout: [max_seq_len, head_dim/2] as float2(cos, sin)

import scope_primitives;
import tile_views;

struct PushConstants {
    uint batch;
    uint seq_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint position_offset;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
RWStructuredBuffer<half> q_data;

[vk::binding(1, 0)]
RWStructuredBuffer<half> k_data;

[vk::binding(2, 0)]
StructuredBuffer<float2> rope_cos_sin;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    BSHDTensor q_tensor = {pc.batch, pc.seq_len, pc.num_q_heads, pc.head_dim};
    BSHDTensor k_tensor = {pc.batch, pc.seq_len, pc.num_kv_heads, pc.head_dim};
    RopeLookupTensor rope = {pc.position_offset, pc.head_dim / 2u};
    Thread::apply_rope(q_data, k_data, rope_cos_sin, q_tensor, k_tensor, rope, Thread::linear_id(tid));
}
