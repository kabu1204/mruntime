// FP16 GEMV (vectorized, multi-row):
//   y = W @ x
//   x: [K] FP16
//   W: [N, K] row-major FP16
//   y: [N] FP16
//
// Strategy:
// - One workgroup computes 4 output rows.
// - Use half4 vectorized loads (K must be divisible by 4).
// - FP32 accumulation.
// - Reduction uses wave-level sums + shared-memory accumulation across waves.

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

groupshared float subgroup_sums[4][32];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    const uint wg = group_id.x;
    const uint tid = local_id.x;
    const uint row_base = wg * 4u;
    if (row_base >= pc.N) {
        return;
    }

    const uint K4 = pc.K >> 2u;
    const bool row0_valid = (row_base + 0u) < pc.N;
    const bool row1_valid = (row_base + 1u) < pc.N;
    const bool row2_valid = (row_base + 2u) < pc.N;
    const bool row3_valid = (row_base + 3u) < pc.N;

    const uint row0_base = (row_base + 0u) * K4;
    const uint row1_base = (row_base + 1u) * K4;
    const uint row2_base = (row_base + 2u) * K4;
    const uint row3_base = (row_base + 3u) * K4;

    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    const uint stride = 128u;
    for (uint i = tid; i < K4; i += 2u * stride) {
        const float4 xv0 = float4(x4_data[i]);

        if (row0_valid) acc0 += dot(xv0, float4(w4_data[row0_base + i]));
        if (row1_valid) acc1 += dot(xv0, float4(w4_data[row1_base + i]));
        if (row2_valid) acc2 += dot(xv0, float4(w4_data[row2_base + i]));
        if (row3_valid) acc3 += dot(xv0, float4(w4_data[row3_base + i]));

        const uint i1 = i + stride;
        if (i1 < K4) {
            const float4 xv1 = float4(x4_data[i1]);

            if (row0_valid) acc0 += dot(xv1, float4(w4_data[row0_base + i1]));
            if (row1_valid) acc1 += dot(xv1, float4(w4_data[row1_base + i1]));
            if (row2_valid) acc2 += dot(xv1, float4(w4_data[row2_base + i1]));
            if (row3_valid) acc3 += dot(xv1, float4(w4_data[row3_base + i1]));
        }
    }

    acc0 = WaveActiveSum(acc0);
    acc1 = WaveActiveSum(acc1);
    acc2 = WaveActiveSum(acc2);
    acc3 = WaveActiveSum(acc3);

    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint wave_id = tid / wave_size;
    const uint wave_count = (128u + wave_size - 1u) / wave_size;

    if (wave_count == 1u) {
        if (lane == 0u) {
            if (row0_valid) y_data[row_base + 0u] = half(acc0);
            if (row1_valid) y_data[row_base + 1u] = half(acc1);
            if (row2_valid) y_data[row_base + 2u] = half(acc2);
            if (row3_valid) y_data[row_base + 3u] = half(acc3);
        }
        return;
    }

    if (lane == 0u) {
        subgroup_sums[0][wave_id] = acc0;
        subgroup_sums[1][wave_id] = acc1;
        subgroup_sums[2][wave_id] = acc2;
        subgroup_sums[3][wave_id] = acc3;
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        float sum0 = 0.0;
        float sum1 = 0.0;
        float sum2 = 0.0;
        float sum3 = 0.0;
        for (uint s = 0u; s < wave_count; ++s) {
            sum0 += subgroup_sums[0][s];
            sum1 += subgroup_sums[1][s];
            sum2 += subgroup_sums[2][s];
            sum3 += subgroup_sums[3][s];
        }

        if (row0_valid) y_data[row_base + 0u] = half(sum0);
        if (row1_valid) y_data[row_base + 1u] = half(sum1);
        if (row2_valid) y_data[row_base + 2u] = half(sum2);
        if (row3_valid) y_data[row_base + 3u] = half(sum3);
    }
}
