// FP16 transpose: BSHD -> BHSD.

struct PushConstants {
    uint B;
    uint S;
    uint H;
    uint D;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half> input_data;

[vk::binding(1, 0)]
RWStructuredBuffer<half> out_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint total_tasks = pc.B * pc.H * pc.S;
    const uint task_id = tid.x;
    if (task_id >= total_tasks) {
        return;
    }

    const uint s = task_id % pc.S;
    const uint h = (task_id / pc.S) % pc.H;
    const uint b = task_id / (pc.S * pc.H);

    const uint src_base = ((b * pc.S + s) * pc.H + h) * pc.D;
    const uint dst_base = ((b * pc.H + h) * pc.S + s) * pc.D;

    for (uint d = 0u; d < pc.D; ++d) {
        out_data[dst_base + d] = input_data[src_base + d];
    }
}
