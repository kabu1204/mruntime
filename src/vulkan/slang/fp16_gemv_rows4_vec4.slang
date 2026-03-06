// FP16 GEMV (vectorized, multi-row):
//   y = W @ x
//   x: [K] FP16
//   W: [N, K] row-major FP16
//   y: [N] FP16

import scope_primitives;
import storage_traits;
import tile_views;

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

groupshared float4 subgroup_sums[32];

public struct SharedWaveSumsStorage : IFloat4Storage1D
{
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

    GlobalVector4Tensor1D x_vector = {pc.K >> 2u};
    GlobalVector4Tensor2D weight_matrix = {pc.N, pc.K >> 2u, pc.K >> 2u};
    GlobalTensor1D y_vector = {pc.N};
    RowBlock4 rows = {group_id.x * 4u};

    if (rows.row_base >= y_vector.size) {
        return;
    }

    const float4 partial = Workgroup::accumulate_rows4_vec4_dot<WG_SIZE>(
        x4_data,
        w4_data,
        x_vector,
        weight_matrix,
        rows,
        tid);
    const float4 wave_sum = Subgroup::reduce_sum(partial);
    const float4 total = Workgroup::tree_reduce_sum<SharedWaveSumsStorage, WG_SIZE>(wave_sum, tid);

    Workgroup::store_rows4(y_data, y_vector, rows, total, tid);
}
