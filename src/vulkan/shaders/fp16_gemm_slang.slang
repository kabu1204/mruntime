// Dense FP16 GEMM: C = A @ B^T
// A: [M, K] row-major FP16
// B: [N, K] row-major FP16 (transposed access: B[n, :])
// C: [M, N] row-major FP16
//
// Strategy:
// - Workgroup: 16x16 threads (256 threads).
// - Each workgroup computes a 64x64 output tile.
// - Each thread computes a 4x4 sub-tile with FP32 accumulation.
// - Shared memory stores packed pairs (half2) with padding to reduce bank conflicts.
// - K is tiled by 32 for better reuse and lower barrier/loop overhead.

static const uint TILE_M = 64u;
static const uint TILE_N = 64u;
static const uint TILE_K = 32u;
static const uint TILE_K2 = TILE_K / 2u;
static const uint THREAD_TILE = 4u;
static const uint SHMEM_K2 = TILE_K2 + 1u;

struct PushConstants {
    uint M;
    uint N;
    uint K;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half> a_data;

[vk::binding(1, 0)]
StructuredBuffer<half> b_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> c_data;

groupshared half2 tile_a[TILE_M][SHMEM_K2];
groupshared half2 tile_b[TILE_N][SHMEM_K2];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    const uint tx = local_id.x;
    const uint ty = local_id.y;
    const uint tid = ty * 16u + tx;

    const uint sub_m = ty * THREAD_TILE;
    const uint sub_n = tx * THREAD_TILE;

    const uint tile_row = group_id.y * TILE_M;
    const uint tile_col = group_id.x * TILE_N;

    float acc[THREAD_TILE][THREAD_TILE];
    for (uint i = 0u; i < THREAD_TILE; ++i) {
        for (uint j = 0u; j < THREAD_TILE; ++j) {
            acc[i][j] = 0.0;
        }
    }

    const uint num_k_tiles = (pc.K + TILE_K - 1u) / TILE_K;
    for (uint kt = 0u; kt < num_k_tiles; ++kt) {
        const uint k_base = kt * TILE_K;

        [unroll]
        for (uint i = 0u; i < 4u; ++i) {
            const uint idx = i * 256u + tid;
            const uint row = idx >> 4u;
            const uint k2 = idx & 15u;

            const uint gm = tile_row + row;
            const uint gk0 = k_base + 2u * k2;
            const uint gk1 = gk0 + 1u;

            half v0 = half(0.0);
            half v1 = half(0.0);
            if (gm < pc.M && gk0 < pc.K) {
                v0 = a_data[gm * pc.K + gk0];
            }
            if (gm < pc.M && gk1 < pc.K) {
                v1 = a_data[gm * pc.K + gk1];
            }
            tile_a[row][k2] = half2(v0, v1);
        }

        [unroll]
        for (uint i = 0u; i < 4u; ++i) {
            const uint idx = i * 256u + tid;
            const uint row = idx >> 4u;
            const uint k2 = idx & 15u;

            const uint gn = tile_col + row;
            const uint gk0 = k_base + 2u * k2;
            const uint gk1 = gk0 + 1u;

            half v0 = half(0.0);
            half v1 = half(0.0);
            if (gn < pc.N && gk0 < pc.K) {
                v0 = b_data[gn * pc.K + gk0];
            }
            if (gn < pc.N && gk1 < pc.K) {
                v1 = b_data[gn * pc.K + gk1];
            }
            tile_b[row][k2] = half2(v0, v1);
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k2 = 0u; k2 < TILE_K2; ++k2) {
            half2 a_pair[THREAD_TILE] = {
                tile_a[sub_m + 0u][k2],
                tile_a[sub_m + 1u][k2],
                tile_a[sub_m + 2u][k2],
                tile_a[sub_m + 3u][k2],
            };
            half2 b_pair[THREAD_TILE] = {
                tile_b[sub_n + 0u][k2],
                tile_b[sub_n + 1u][k2],
                tile_b[sub_n + 2u][k2],
                tile_b[sub_n + 3u][k2],
            };

            float bx0[THREAD_TILE] = {
                float(b_pair[0].x), float(b_pair[1].x), float(b_pair[2].x), float(b_pair[3].x),
            };
            float bx1[THREAD_TILE] = {
                float(b_pair[0].y), float(b_pair[1].y), float(b_pair[2].y), float(b_pair[3].y),
            };

            [unroll]
            for (uint i = 0u; i < THREAD_TILE; ++i) {
                const float ax0 = float(a_pair[i].x);
                const float ax1 = float(a_pair[i].y);
                [unroll]
                for (uint j = 0u; j < THREAD_TILE; ++j) {
                    acc[i][j] = fma(ax0, bx0[j], acc[i][j]);
                    acc[i][j] = fma(ax1, bx1[j], acc[i][j]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = 0u; i < THREAD_TILE; ++i) {
        for (uint j = 0u; j < THREAD_TILE; ++j) {
            const uint gm = tile_row + sub_m + i;
            const uint gn = tile_col + sub_n + j;
            if (gm < pc.M && gn < pc.N) {
                c_data[gm * pc.N + gn] = half(acc[i][j]);
            }
        }
    }
}
