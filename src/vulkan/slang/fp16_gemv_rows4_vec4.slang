// FP16 GEMV (vectorized, multi-row):
//   y = W @ x
//   x: [K] FP16
//   W: [N, K] row-major FP16
//   y: [N] FP16

import scope_primitives;
import storage_traits;

static const int WG_SIZE = 128;

struct PushConstants {
    uint N;
    uint K;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half4> x4_data;

[vk::binding(1, 0)]
StructuredBuffer<half4> w4_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> y_data;

groupshared float4 subgroup_sums[WG_SIZE];

public struct SharedWaveSumsStorage : IFloat4Storage1D
{
    public static const uint kExtent = WG_SIZE;
    public static const StorageScope kScope = StorageScope::Workgroup;
    public static const bool kBarrierSafe = true;

    public static void store(uint index, float4 value)
    {
        subgroup_sums[index] = value;
    }

    public static float4 load(uint index)
    {
        return subgroup_sums[index];
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    const uint tid = local_id.x;
    const uint x_count4 = pc.K >> 2u;
    const uint weight_stride4 = x_count4;
    const uint row_base = group_id.x * 4u;

    if (row_base >= pc.N) {
        return;
    }

    float4 partial = float4(0.0);
    const uint stride = uint(WG_SIZE);
    for (uint col4 = tid; col4 < x_count4; col4 += 2u * stride) {
        const float4 x_value = float4(x4_data[col4]);

        if (row_base + 0u < pc.N) {
            partial.x += dot(x_value, float4(w4_data[(row_base + 0u) * weight_stride4 + col4]));
        }
        if (row_base + 1u < pc.N) {
            partial.y += dot(x_value, float4(w4_data[(row_base + 1u) * weight_stride4 + col4]));
        }
        if (row_base + 2u < pc.N) {
            partial.z += dot(x_value, float4(w4_data[(row_base + 2u) * weight_stride4 + col4]));
        }
        if (row_base + 3u < pc.N) {
            partial.w += dot(x_value, float4(w4_data[(row_base + 3u) * weight_stride4 + col4]));
        }

        const uint next_col4 = col4 + stride;
        if (next_col4 >= x_count4) {
            continue;
        }

        const float4 next_x_value = float4(x4_data[next_col4]);
        if (row_base + 0u < pc.N) {
            partial.x += dot(next_x_value, float4(w4_data[(row_base + 0u) * weight_stride4 + next_col4]));
        }
        if (row_base + 1u < pc.N) {
            partial.y += dot(next_x_value, float4(w4_data[(row_base + 1u) * weight_stride4 + next_col4]));
        }
        if (row_base + 2u < pc.N) {
            partial.z += dot(next_x_value, float4(w4_data[(row_base + 2u) * weight_stride4 + next_col4]));
        }
        if (row_base + 3u < pc.N) {
            partial.w += dot(next_x_value, float4(w4_data[(row_base + 3u) * weight_stride4 + next_col4]));
        }
    }

    const float4 total = Workgroup::leader_reduce_sum4<SharedWaveSumsStorage, WG_SIZE>(partial, tid);
    if (tid != 0u) {
        return;
    }

    if (row_base + 0u < pc.N) {
        y_data[row_base + 0u] = half(total.x);
    }
    if (row_base + 1u < pc.N) {
        y_data[row_base + 1u] = half(total.y);
    }
    if (row_base + 2u < pc.N) {
        y_data[row_base + 2u] = half(total.z);
    }
    if (row_base + 3u < pc.N) {
        y_data[row_base + 3u] = half(total.w);
    }
}
