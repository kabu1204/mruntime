// FP16 RoPE on Q/K with FP32 cos/sin lookup table.
//
// q layout: [batch, seq_len, num_q_heads, head_dim]
// k layout: [batch, seq_len, num_kv_heads, head_dim]
// rope_cos_sin layout: [max_seq_len, head_dim/2] as float2(cos, sin)

import layout_traits;
import scope_primitives;

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
    const uint index = Thread::linear_id(tid);
    if (index >= total) {
        return;
    }

    const uint dim_half = index % half_dim;
    const uint head = (index / half_dim) % total_heads;
    const uint seq = (index / (half_dim * total_heads)) % pc.seq_len;
    const uint batch = index / (half_dim * total_heads * pc.seq_len);

    const float2 cs = rope_cos_sin[StridedLayout::offset2(0u, half_dim, 1u, pc.position_offset + seq, dim_half)];
    const float cos_value = cs.x;
    const float sin_value = cs.y;

    if (head < pc.num_q_heads) {
        const uint q_batch_stride = pc.seq_len * pc.num_q_heads * pc.head_dim;
        const uint q_seq_stride = pc.num_q_heads * pc.head_dim;
        const uint q_head_stride = pc.head_dim;
        const uint base = StridedLayout::offset4(
            0u,
            q_batch_stride,
            q_seq_stride,
            q_head_stride,
            1u,
            batch,
            seq,
            head,
            0u);
        const float x0 = float(q_data[base + dim_half]);
        const float x1 = float(q_data[base + half_dim + dim_half]);
        q_data[base + dim_half] = half(x0 * cos_value - x1 * sin_value);
        q_data[base + half_dim + dim_half] = half(x1 * cos_value + x0 * sin_value);
        return;
    }

    const uint kv_head = head - pc.num_q_heads;
    const uint k_batch_stride = pc.seq_len * pc.num_kv_heads * pc.head_dim;
    const uint k_seq_stride = pc.num_kv_heads * pc.head_dim;
    const uint k_head_stride = pc.head_dim;
    const uint base = StridedLayout::offset4(
        0u,
        k_batch_stride,
        k_seq_stride,
        k_head_stride,
        1u,
        batch,
        seq,
        kv_head,
        0u);
    const float x0 = float(k_data[base + dim_half]);
    const float x1 = float(k_data[base + half_dim + dim_half]);
    k_data[base + dim_half] = half(x0 * cos_value - x1 * sin_value);
    k_data[base + half_dim + dim_half] = half(x1 * cos_value + x0 * sin_value);
}
