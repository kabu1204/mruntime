// Prefill FP16 GEMM fast path for wide-N projection layers.

static const int WG_X = 16;
static const int WG_Y = 16;
static const int TILE_M = 64;
static const int TILE_N = 128;
static const int TILE_K = 64;
static const int TILE_K4 = TILE_K / 4;
static const int THREAD_TILE_M = 4;
static const int THREAD_TILE_N = 8;
static const int SHMEM_K4 = TILE_K4 + 1;
static const int LOADS_A_PER_THREAD = 4;
static const int LOADS_B_PER_THREAD = 8;

struct PushConstants {
    uint M;
    uint N;
    uint K;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
StructuredBuffer<half4> a_data;

[vk::binding(1, 0)]
StructuredBuffer<half4> b_data;

[vk::binding(2, 0)]
RWStructuredBuffer<half> c_data;

groupshared half4 tile_a[TILE_M][SHMEM_K4];
groupshared half4 tile_b[TILE_N][SHMEM_K4];

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID
) {
    const uint tx = local_id.x;
    const uint ty = local_id.y;
    const uint tid = ty * uint(WG_X) + tx;
    const uint sub_m = ty * uint(THREAD_TILE_M);
    const uint sub_n = tx * uint(THREAD_TILE_N);
    const uint tile_row = group_id.y * uint(TILE_M);
    const uint tile_col = group_id.x * uint(TILE_N);
    const uint k4_total = pc.K >> 2u;
    const uint num_k_tiles = (pc.K + uint(TILE_K) - 1u) / uint(TILE_K);

    float acc[THREAD_TILE_M][THREAD_TILE_N];
    [unroll]
    for (int row = 0; row < THREAD_TILE_M; ++row) {
        [unroll]
        for (int col = 0; col < THREAD_TILE_N; ++col) {
            acc[row][col] = 0.0;
        }
    }

    for (uint tile_k = 0u; tile_k < num_k_tiles; ++tile_k) {
        const uint k_base4 = tile_k * uint(TILE_K4);

        [unroll]
        for (int iter = 0; iter < LOADS_A_PER_THREAD; ++iter) {
            const uint idx = uint(iter) * uint(WG_X * WG_Y) + tid;
            const uint row = idx / uint(TILE_K4);
            const uint k4 = idx % uint(TILE_K4);
            const uint gm = tile_row + row;
            const uint gk4 = k_base4 + k4;
            tile_a[row][k4] = (gm < pc.M && gk4 < k4_total)
                ? a_data[gm * k4_total + gk4]
                : half4(0.0);
        }

        [unroll]
        for (int iter = 0; iter < LOADS_B_PER_THREAD; ++iter) {
            const uint idx = uint(iter) * uint(WG_X * WG_Y) + tid;
            const uint row = idx / uint(TILE_K4);
            const uint k4 = idx % uint(TILE_K4);
            const uint gn = tile_col + row;
            const uint gk4 = k_base4 + k4;
            tile_b[row][k4] = (gn < pc.N && gk4 < k4_total)
                ? b_data[gn * k4_total + gk4]
                : half4(0.0);
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k4 = 0u; k4 < uint(TILE_K4); ++k4) {
            float4 b_vec[THREAD_TILE_N];
            [unroll]
            for (int col = 0; col < THREAD_TILE_N; ++col) {
                b_vec[col] = float4(tile_b[sub_n + uint(col)][k4]);
            }

            [unroll]
            for (int row = 0; row < THREAD_TILE_M; ++row) {
                const float4 a_vec = float4(tile_a[sub_m + uint(row)][k4]);
                [unroll]
                for (int col = 0; col < THREAD_TILE_N; ++col) {
                    acc[row][col] += dot(a_vec, b_vec[col]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (int row = 0; row < THREAD_TILE_M; ++row) {
        [unroll]
        for (int col = 0; col < THREAD_TILE_N; ++col) {
            const uint gm = tile_row + sub_m + uint(row);
            const uint gn = tile_col + sub_n + uint(col);
            if (gm < pc.M && gn < pc.N) {
                c_data[gm * pc.N + gn] = half(acc[row][col]);
            }
        }
    }
}
