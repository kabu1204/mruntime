// Decode-only FP16 GQA attention.
// Q: [1, num_q_heads, 1, head_dim] BHSD FP16
// K: [1, num_kv_heads, kv_stride, head_dim] BHSD FP16
// V: [1, num_kv_heads, kv_stride, head_dim] BHSD FP16
// O: [1, num_q_heads, 1, head_dim] BHSD FP16

import layout_traits;
import scope_primitives;
import storage_traits;
import tile_views;

static const int WG_SIZE = 128;
static const int SCORE_TILE = 32;
static const int MAX_HEAD_DIM = 512;
static const float NEG_INF = -3.402823466e38;

struct PushConstants {
    uint num_q_heads;
    uint num_kv_heads;
    uint kv_len;
    uint kv_stride;
    uint head_dim;
    float scale;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
ByteAddressBuffer q_data;

[vk::binding(1, 0)]
ByteAddressBuffer k_data;

[vk::binding(2, 0)]
ByteAddressBuffer v_data;

[vk::binding(3, 0)]
RWStructuredBuffer<half> o_data;

groupshared float shared_q[MAX_HEAD_DIM];
groupshared float score_tile[SCORE_TILE];
groupshared float subgroup_reduce[WG_SIZE];

public struct SharedQueryStorage : IFloatStorage1D
{
    public static const uint kExtent = MAX_HEAD_DIM;
    public static const StorageScope kScope = StorageScope::Workgroup;
    public static const bool kBarrierSafe = true;

    public static void store(uint index, float value)
    {
        shared_q[index] = value;
    }

    public static float load(uint index)
    {
        return shared_q[index];
    }
}

public struct SharedScoreTileStorage : IFloatStorage1D
{
    public static const uint kExtent = SCORE_TILE;
    public static const StorageScope kScope = StorageScope::Workgroup;
    public static const bool kBarrierSafe = true;

    public static void store(uint index, float value)
    {
        score_tile[index] = value;
    }

    public static float load(uint index)
    {
        return score_tile[index];
    }
}

public struct SharedScalarStorage : IFloatStorage1D
{
    public static const uint kExtent = WG_SIZE;
    public static const StorageScope kScope = StorageScope::Workgroup;
    public static const bool kBarrierSafe = true;

    public static void store(uint index, float value)
    {
        subgroup_reduce[index] = value;
    }

    public static float load(uint index)
    {
        return subgroup_reduce[index];
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 local_id : SV_GroupThreadID,
    uint3 group_id : SV_GroupID)
{
    const uint tid = local_id.x;
    const uint q_head = group_id.x;

    if (q_head >= pc.num_q_heads) {
        return;
    }
    if (pc.head_dim > uint(MAX_HEAD_DIM)) {
        return;
    }

    const uint heads_per_kv = pc.num_q_heads / pc.num_kv_heads;
    const uint kv_head = q_head / heads_per_kv;
    const uint q_batch_stride = pc.num_q_heads * pc.head_dim;
    const uint q_head_stride = pc.head_dim;
    const uint q_seq_stride = pc.head_dim;
    const uint kv_batch_stride = pc.num_kv_heads * pc.kv_stride * pc.head_dim;
    const uint kv_head_stride = pc.kv_stride * pc.head_dim;
    const uint kv_seq_stride = pc.head_dim;
    const uint q_base = StridedLayout::offset4(0u, q_batch_stride, q_head_stride, q_seq_stride, 1u, 0u, q_head, 0u, 0u);
    const uint kv_base = StridedLayout::offset4(0u, kv_batch_stride, kv_head_stride, kv_seq_stride, 1u, 0u, kv_head, 0u, 0u);
    const Fp16Span q_slice = {q_base, 1u, pc.head_dim};
    const Fp16SliceSet kv_slices = {kv_base, kv_seq_stride, 1u, pc.kv_len, pc.head_dim};
    const RWFp16Span o_slice = {q_base, 1u, pc.head_dim};
    const uint index0 = tid;
    const uint index1 = tid + uint(WG_SIZE);
    const uint index2 = tid + 2u * uint(WG_SIZE);
    const uint index3 = tid + 3u * uint(WG_SIZE);

    Workgroup::gather_slice<SharedQueryStorage, WG_SIZE>(q_data, q_slice, tid);
    GroupMemoryBarrierWithGroupSync();

    float max_score = NEG_INF;
    [loop]
    for (uint slice_base = 0u; slice_base < kv_slices.slice_count; slice_base += uint(SCORE_TILE)) {
        const uint tile_count = min(uint(SCORE_TILE), kv_slices.slice_count - slice_base);
        if (tid < tile_count) {
            const Fp16Span score_slice = kv_slices.slice(slice_base + tid);
            float score = 0.0;
            uint index = 0u;
            const uint vectorized_end = score_slice.vectorized_count8();

            for (; index < vectorized_end; index += 8u) {
                const FloatPack8 src_values = Thread::load_fp16_pack8(k_data, score_slice.offset(index));
                score += dot(Thread::gather_storage4<SharedQueryStorage>(index + 0u), src_values.lo);
                score += dot(Thread::gather_storage4<SharedQueryStorage>(index + 4u), src_values.hi);
            }

            for (; index < score_slice.count; ++index) {
                score += SharedQueryStorage::load(index) * Thread::load_fp16(k_data, score_slice.offset(index));
            }

            max_score = max(max_score, score * pc.scale);
        }
    }

    const float stabilized_max = Workgroup::broadcast_scalar<SharedScalarStorage>(
        Workgroup::leader_reduce_max<SharedScalarStorage, WG_SIZE>(max_score, tid),
        tid);

    float4 accum = float4(0.0);
    float norm_sum = 0.0;
    [loop]
    for (uint slice_base = 0u; slice_base < kv_slices.slice_count; slice_base += uint(SCORE_TILE)) {
        const uint tile_count = min(uint(SCORE_TILE), kv_slices.slice_count - slice_base);
        if (tid < tile_count) {
            const Fp16Span score_slice = kv_slices.slice(slice_base + tid);
            float score = 0.0;
            uint index = 0u;
            const uint vectorized_end = score_slice.vectorized_count8();

            for (; index < vectorized_end; index += 8u) {
                const FloatPack8 src_values = Thread::load_fp16_pack8(k_data, score_slice.offset(index));
                score += dot(Thread::gather_storage4<SharedQueryStorage>(index + 0u), src_values.lo);
                score += dot(Thread::gather_storage4<SharedQueryStorage>(index + 4u), src_values.hi);
            }

            for (; index < score_slice.count; ++index) {
                score += SharedQueryStorage::load(index) * Thread::load_fp16(k_data, score_slice.offset(index));
            }

            const float weight = exp(score * pc.scale - stabilized_max);
            SharedScoreTileStorage::store(tid, weight);
            norm_sum += weight;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint slice = 0u; slice < tile_count; ++slice) {
            const float weight = SharedScoreTileStorage::load(slice);
            const Fp16Span value_slice = kv_slices.slice(slice_base + slice);
            if (index0 < value_slice.count) {
                accum.x += weight * Thread::load_fp16(v_data, value_slice.offset(index0));
            }
            if (index1 < value_slice.count) {
                accum.y += weight * Thread::load_fp16(v_data, value_slice.offset(index1));
            }
            if (index2 < value_slice.count) {
                accum.z += weight * Thread::load_fp16(v_data, value_slice.offset(index2));
            }
            if (index3 < value_slice.count) {
                accum.w += weight * Thread::load_fp16(v_data, value_slice.offset(index3));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float total_norm = Workgroup::broadcast_scalar<SharedScalarStorage>(
        Workgroup::leader_reduce_sum<SharedScalarStorage, WG_SIZE>(norm_sum, tid),
        tid);

    const float inv_norm = (total_norm > 0.0) ? (1.0 / total_norm) : 0.0;
    Thread::scatter_values4(o_data, o_slice, tid, WG_SIZE, accum * inv_norm);
}
