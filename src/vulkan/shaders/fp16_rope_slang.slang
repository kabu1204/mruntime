// FP16 RoPE on Q/K with FP32 cos/sin lookup table.
//
// q layout: [batch, seq_len, num_q_heads, head_dim]
// k layout: [batch, seq_len, num_kv_heads, head_dim]
// rope_cos_sin layout: [max_seq_len, head_dim/2] as float2(cos, sin)

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
    const uint half_dim = pc.head_dim / 2u;
    const uint total_heads = pc.num_q_heads + pc.num_kv_heads;
    const uint total = pc.batch * pc.seq_len * total_heads * half_dim;

    const uint gid = tid.x;
    if (gid >= total) {
        return;
    }

    const uint i = gid % half_dim;
    const uint h = (gid / half_dim) % total_heads;
    const uint s = (gid / (half_dim * total_heads)) % pc.seq_len;
    const uint b = gid / (half_dim * total_heads * pc.seq_len);

    const uint pos = pc.position_offset + s;
    const float2 cs = rope_cos_sin[pos * half_dim + i];
    const float cos_val = cs.x;
    const float sin_val = cs.y;

    if (h < pc.num_q_heads) {
        const uint base = ((b * pc.seq_len + s) * pc.num_q_heads + h) * pc.head_dim;
        const float x0 = float(q_data[base + i]);
        const float x1 = float(q_data[base + half_dim + i]);
        q_data[base + i] = half(x0 * cos_val - x1 * sin_val);
        q_data[base + half_dim + i] = half(x1 * cos_val + x0 * sin_val);
    } else {
        const uint kh = h - pc.num_q_heads;
        const uint base = ((b * pc.seq_len + s) * pc.num_kv_heads + kh) * pc.head_dim;
        const float x0 = float(k_data[base + i]);
        const float x1 = float(k_data[base + half_dim + i]);
        k_data[base + i] = half(x0 * cos_val - x1 * sin_val);
        k_data[base + half_dim + i] = half(x1 * cos_val + x0 * sin_val);
    }
}
