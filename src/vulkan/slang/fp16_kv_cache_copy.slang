// FP16 KV cache copy.
//
// k_in / v_in layout: [batch, seq_len, num_kv_heads, head_dim]
// cache layout:       [batch, num_kv_heads, max_seq_len, head_dim]

import layout_traits;
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
ByteAddressBuffer k_in_data;

[vk::binding(1, 0)]
ByteAddressBuffer v_in_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> k_cache_data;

[vk::binding(3, 0)]
RWStructuredBuffer<half> v_cache_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint task_id = Thread::linear_id(tid);
    const uint task_count = pc.batch * pc.seq_len * pc.num_kv_heads;
    if (task_id >= task_count) {
        return;
    }

    const uint head = task_id % pc.num_kv_heads;
    const uint seq = (task_id / pc.num_kv_heads) % pc.seq_len;
    const uint batch = task_id / (pc.num_kv_heads * pc.seq_len);
    const uint cache_seq = pc.position_offset + seq;

    const uint src_batch_stride = pc.seq_len * pc.num_kv_heads * pc.head_dim;
    const uint src_seq_stride = pc.num_kv_heads * pc.head_dim;
    const uint src_head_stride = pc.head_dim;
    const uint dst_batch_stride = pc.num_kv_heads * pc.max_seq_len * pc.head_dim;
    const uint dst_head_stride = pc.max_seq_len * pc.head_dim;
    const uint dst_seq_stride = pc.head_dim;

    const uint src_base = StridedLayout::offset4(
        0u,
        src_batch_stride,
        src_seq_stride,
        src_head_stride,
        1u,
        batch,
        seq,
        head,
        0u);
    const uint dst_base = StridedLayout::offset4(
        0u,
        dst_batch_stride,
        dst_head_stride,
        dst_seq_stride,
        1u,
        batch,
        head,
        cache_seq,
        0u);

    const Fp16Span src_span = {src_base, 1u, pc.head_dim};
    const RWFp16Span dst_span = {dst_base, 1u, pc.head_dim};
    Thread::copy_span(k_cache_data, dst_span, k_in_data, src_span);
    Thread::copy_span(v_cache_data, dst_span, v_in_data, src_span);
}
