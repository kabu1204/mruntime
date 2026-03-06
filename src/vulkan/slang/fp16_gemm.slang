// Dense FP16 GEMM: C = A @ B^T
// A: [M, K] row-major FP16
// B: [N, K] row-major FP16 (transposed access: B[n, :])
// C: [M, N] row-major FP16

import compute_arch;
import register_tile;
import scope_primitives;
import storage_traits;
import tile_views;

static const int WG_X = 16;
static const int WG_Y = 16;
static const int TILE_M = 64;
static const int TILE_N = 64;
static const int TILE_K = 32;
static const int TILE_K2 = TILE_K / 2;
static const int THREAD_TILE = 4;
static const int SHMEM_K2 = TILE_K2 + 1;
static const int LOADS_PER_THREAD = 4;

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

public struct SharedTileAStorage : IHalf2TileStorage
{
    public static void store(uint row, uint col, half2 value)
    {
        tile_a[row][col] = value;
    }

    public static half2 load(uint row, uint col)
    {
        return tile_a[row][col];
    }
}

public struct SharedTileBStorage : IHalf2TileStorage
{
    public static void store(uint row, uint col, half2 value)
    {
        tile_b[row][col] = value;
    }

    public static half2 load(uint row, uint col)
    {
        return tile_b[row][col];
    }
}

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    GlobalTensor2D global_a = {pc.M, pc.K, pc.K};
    GlobalTensor2D global_b = {pc.N, pc.K, pc.K};
    GlobalTensor2D global_c = {pc.M, pc.N, pc.N};

    SharedTileRegion tile_a_region = {group_id.y * uint(TILE_M), 0u};
    SharedTileRegion tile_b_region = {group_id.x * uint(TILE_N), 0u};
    SharedTileRegion tile_c_region = {group_id.y * uint(TILE_M), group_id.x * uint(TILE_N)};

    RegisterTile<THREAD_TILE, THREAD_TILE> accum;
    accum.clear();

    const uint num_k_tiles = (pc.K + uint(TILE_K) - 1u) / uint(TILE_K);
    for (uint tile_k = 0u; tile_k < num_k_tiles; ++tile_k) {
        const uint k_base = tile_k * uint(TILE_K);

        Workgroup::load_matrix_rows_half2_tile<
            SharedTileAStorage,
            WG_X,
            WG_Y,
            TILE_M,
            TILE_K2,
            LOADS_PER_THREAD>(a_data, global_a, tile_a_region, k_base, local_id);

        Workgroup::load_matrix_rows_half2_tile<
            SharedTileBStorage,
            WG_X,
            WG_Y,
            TILE_N,
            TILE_K2,
            LOADS_PER_THREAD>(b_data, global_b, tile_b_region, k_base, local_id);

        GroupMemoryBarrierWithGroupSync();

        Subgroup::mma<
            VectorSIMTArch,
            SharedTileAStorage,
            SharedTileBStorage,
            THREAD_TILE,
            THREAD_TILE,
            TILE_K2>(accum, local_id);

        GroupMemoryBarrierWithGroupSync();
    }

    Workgroup::store_matrix_tile<THREAD_TILE, THREAD_TILE>(c_data, global_c, accum, tile_c_region, local_id);
}
