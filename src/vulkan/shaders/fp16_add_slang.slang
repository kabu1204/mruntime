// FP16 elementwise add: out[i] = a[i] + b[i]
//
// Bindings match mruntime's Vulkan kernel conventions:
// - descriptor set 0
// - storage buffers at bindings 0..N-1
// - one push-constant block

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
    const uint i = tid.x;
    if (i >= pc.n) {
        return;
    }
    out_data[i] = a[i] + b[i];
}
