// FP16 elementwise add: out[i] = a[i] + b[i]

import pointwise_ops;
import scope_primitives;
import tile_views;

struct PushConstants {
    uint n;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half> a;

[vk::binding(1, 0)]
StructuredBuffer<half> b;

[vk::binding(2, 0)]
RWStructuredBuffer<half> out_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    GlobalTensor1D out_tensor = {pc.n};
    Thread::apply_binary_elementwise<AddOp>(a, b, out_data, out_tensor, Thread::linear_id(tid));
}
