// FP16 transpose: BSHD -> BHSD.

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
StructuredBuffer<half> input_data;

[vk::binding(1, 0)]
RWStructuredBuffer<half> out_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    BSHDTensor src_tensor = {pc.B, pc.S, pc.H, pc.D};
    BHSDTensor dst_tensor = {pc.B, pc.H, pc.S, pc.D};
    Thread::transpose_bshd_to_bhsd(input_data, out_data, src_tensor, dst_tensor, Thread::linear_id(tid));
}
