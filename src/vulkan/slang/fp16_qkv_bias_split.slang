// Split fused QKV rows into Q, K, V and optionally add bias.

import scope_primitives;

struct PushConstants {
    uint num_tokens;
    uint q_dim;
    uint kv_dim;
    uint has_bias;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
ByteAddressBuffer qkv_data;

[vk::binding(1, 0)]
ByteAddressBuffer bias_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> q_out_data;

[vk::binding(3, 0)]
RWStructuredBuffer<half> k_out_data;

[vk::binding(4, 0)]
RWStructuredBuffer<half> v_out_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint chunk_id = Thread::linear_id(tid);
    const uint chunk_base = chunk_id * 8u;
    const uint qkv_dim = pc.q_dim + 2u * pc.kv_dim;
    const uint total = pc.num_tokens * qkv_dim;
    if (chunk_base >= total) {
        return;
    }

    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint index = chunk_base + lane;
        if (index >= total) {
            return;
        }

        const uint token = index / qkv_dim;
        const uint elem = index % qkv_dim;
        float value = Thread::load_fp16(qkv_data, index);
        if (pc.has_bias != 0u) {
            value += Thread::load_fp16(bias_data, elem);
        }

        if (elem < pc.q_dim) {
            q_out_data[token * pc.q_dim + elem] = half(value);
        } else if (elem < pc.q_dim + pc.kv_dim) {
            const uint kv_index = elem - pc.q_dim;
            k_out_data[token * pc.kv_dim + kv_index] = half(value);
        } else {
            const uint kv_index = elem - pc.q_dim - pc.kv_dim;
            v_out_data[token * pc.kv_dim + kv_index] = half(value);
        }
    }
}
