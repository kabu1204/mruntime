// FP16 transpose: BSHD -> BHSD.

import layout_traits;
import scope_primitives;
import tile_views;

struct PushConstants {
    uint B;
    uint S;
    uint H;
    uint D;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
ByteAddressBuffer input_data;

[vk::binding(1, 0)]
RWStructuredBuffer<half> out_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint task_id = Thread::linear_id(tid);
    const uint task_count = pc.B * pc.S * pc.H;
    if (task_id >= task_count) {
        return;
    }

    const uint seq = task_id % pc.S;
    const uint head = (task_id / pc.S) % pc.H;
    const uint batch = task_id / (pc.S * pc.H);

    const uint src_batch_stride = pc.S * pc.H * pc.D;
    const uint src_seq_stride = pc.H * pc.D;
    const uint src_head_stride = pc.D;
    const uint dst_batch_stride = pc.H * pc.S * pc.D;
    const uint dst_head_stride = pc.S * pc.D;
    const uint dst_seq_stride = pc.D;

    const Fp16Span src_span = {
        StridedLayout::offset4(0u, src_batch_stride, src_seq_stride, src_head_stride, 1u, batch, seq, head, 0u),
        1u,
        pc.D,
    };
    const RWFp16Span dst_span = {
        StridedLayout::offset4(0u, dst_batch_stride, dst_head_stride, dst_seq_stride, 1u, batch, head, seq, 0u),
        1u,
        pc.D,
    };
    Thread::copy_span(out_data, dst_span, input_data, src_span);
}
