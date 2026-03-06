// FP16 KV cache copy.
//
// k_in / v_in layout: [batch, seq_len, num_kv_heads, head_dim]
// cache layout:       [batch, num_kv_heads, max_seq_len, head_dim]

import scope_primitives;
import tile_views;

struct PushConstants {
    uint batch;
    uint seq_len;
    uint num_kv_heads;
    uint head_dim;
    uint max_seq_len;
    uint position_offset;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half> k_in_data;

[vk::binding(1, 0)]
StructuredBuffer<half> v_in_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> k_cache_data;

[vk::binding(3, 0)]
RWStructuredBuffer<half> v_cache_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    BSHDTensor src_tensor = {pc.batch, pc.seq_len, pc.num_kv_heads, pc.head_dim};
    BHSDTensor dst_tensor = {pc.batch, pc.num_kv_heads, pc.max_seq_len, pc.head_dim};
    Thread::copy_kv_cache(
        k_in_data,
        v_in_data,
        k_cache_data,
        v_cache_data,
        src_tensor,
        dst_tensor,
        pc.position_offset,
        Thread::linear_id(tid));
}
