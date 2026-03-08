// Decode-only fused QKV split + optional bias + RoPE + KV cache write.
//
// qkv layout:      [q_dim + kv_dim + kv_dim]
// q_out layout:    [num_q_heads, head_dim]
// k/v cache layout:[num_kv_heads, max_seq_len, head_dim]
// rope_cos_sin:    [max_seq_len, head_dim/2] as float2(cos, sin)

import scope_primitives;

struct PushConstants {
    uint q_dim;
    uint kv_dim;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint max_seq_len;
    uint position_offset;
    uint has_bias;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
ByteAddressBuffer qkv_data;

[vk::binding(1, 0)]
ByteAddressBuffer bias_data;

[vk::binding(2, 0)]
StructuredBuffer<float2> rope_cos_sin;

[vk::binding(3, 0)]
RWStructuredBuffer<half> q_out_data;

[vk::binding(4, 0)]
RWStructuredBuffer<half> k_cache_data;

[vk::binding(5, 0)]
RWStructuredBuffer<half> v_cache_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint half_dim = pc.head_dim / 2u;
    const uint q_pair_total = pc.num_q_heads * half_dim;
    const uint k_pair_total = pc.num_kv_heads * half_dim;
    const uint total = q_pair_total + k_pair_total + pc.kv_dim;
    const uint index = Thread::linear_id(tid);
    if (index >= total) {
        return;
    }

    if (index < q_pair_total) {
        const uint head = index / half_dim;
        const uint dim_half = index % half_dim;
        const uint src_base = head * pc.head_dim;
        const uint src0 = src_base + dim_half;
        const uint src1 = src_base + half_dim + dim_half;
        float x0 = Thread::load_fp16(qkv_data, src0);
        float x1 = Thread::load_fp16(qkv_data, src1);
        if (pc.has_bias != 0u) {
            x0 += Thread::load_fp16(bias_data, src0);
            x1 += Thread::load_fp16(bias_data, src1);
        }

        const float2 cs = rope_cos_sin[pc.position_offset * half_dim + dim_half];
        q_out_data[src0] = half(x0 * cs.x - x1 * cs.y);
        q_out_data[src1] = half(x1 * cs.x + x0 * cs.y);
        return;
    }

    if (index < q_pair_total + k_pair_total) {
        const uint local = index - q_pair_total;
        const uint head = local / half_dim;
        const uint dim_half = local % half_dim;
        const uint src_base = pc.q_dim + head * pc.head_dim;
        const uint src0 = src_base + dim_half;
        const uint src1 = src_base + half_dim + dim_half;
        float x0 = Thread::load_fp16(qkv_data, src0);
        float x1 = Thread::load_fp16(qkv_data, src1);
        if (pc.has_bias != 0u) {
            x0 += Thread::load_fp16(bias_data, src0);
            x1 += Thread::load_fp16(bias_data, src1);
        }

        const float2 cs = rope_cos_sin[pc.position_offset * half_dim + dim_half];
        const uint dst_base = (head * pc.max_seq_len + pc.position_offset) * pc.head_dim;
        k_cache_data[dst_base + dim_half] = half(x0 * cs.x - x1 * cs.y);
        k_cache_data[dst_base + half_dim + dim_half] = half(x1 * cs.x + x0 * cs.y);
        return;
    }

    const uint local = index - q_pair_total - k_pair_total;
    const uint head = local / pc.head_dim;
    const uint dim = local % pc.head_dim;
    const uint src = pc.q_dim + pc.kv_dim + head * pc.head_dim + dim;
    float value = Thread::load_fp16(qkv_data, src);
    if (pc.has_bias != 0u) {
        value += Thread::load_fp16(bias_data, src);
    }

    const uint dst = (head * pc.max_seq_len + pc.position_offset) * pc.head_dim + dim;
    v_cache_data[dst] = half(value);
}
