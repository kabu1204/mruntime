// FP32 elementwise add: c[i] = a[i] + b[i]

struct PushConstants {
    uint n;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<float> a;

[vk::binding(1, 0)]
StructuredBuffer<float> b;

[vk::binding(2, 0)]
RWStructuredBuffer<float> c_data;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= pc.n) {
        return;
    }
    c_data[i] = a[i] + b[i];
}
