// FP16 KV cache copy.
//
// k_in / v_in layout: [batch, seq_len, num_kv_heads, head_dim]
// cache layout:       [batch, num_kv_heads, max_seq_len, head_dim]

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
    const uint total = pc.batch * pc.seq_len * pc.num_kv_heads;
    const uint gid = tid.x;
    if (gid >= total) {
        return;
    }

    const uint h = gid % pc.num_kv_heads;
    const uint s = (gid / pc.num_kv_heads) % pc.seq_len;
    const uint b = gid / (pc.num_kv_heads * pc.seq_len);

    const uint src_base = ((b * pc.seq_len + s) * pc.num_kv_heads + h) * pc.head_dim;
    const uint cache_pos = pc.position_offset + s;
    const uint dst_base = ((b * pc.num_kv_heads + h) * pc.max_seq_len + cache_pos) * pc.head_dim;

    for (uint d = 0u; d < pc.head_dim; ++d) {
        k_cache_data[dst_base + d] = k_in_data[src_base + d];
        v_cache_data[dst_base + d] = v_in_data[src_base + d];
    }
}
