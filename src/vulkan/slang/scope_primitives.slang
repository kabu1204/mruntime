import compute_arch;
import layout_traits;
import pointwise_ops;
import register_tile;
import storage_traits;
import tile_views;

public struct FloatPack8
{
    public float4 lo;
    public float4 hi;
}

public struct Workgroup
{
    public static uint flatten_local_id<int WorkgroupX>(uint3 local_id)
    {
        return local_id.y * uint(WorkgroupX) + local_id.x;
    }

    public static void stage_tile_half2<
        Layout : ITensor2DLayout,
        Storage : IHalf2TileStorage,
        int WorkgroupX,
        int WorkgroupY,
        int TileRows,
        int TileK2,
        int LoadsPerThread>(
            StructuredBuffer<half> src,
            GlobalTensor2D tensor,
            SharedTileRegion region,
            uint col_base,
            uint3 local_id)
    {
        static_assert(
            LoadsPerThread * WorkgroupX * WorkgroupY == TileRows * TileK2,
            "Tile loader requires exact work coverage.");
        static_assert(Storage::kRows >= uint(TileRows), "Tile storage row extent is too small.");
        static_assert(Storage::kCols >= uint(TileK2), "Tile storage column extent is too small.");
        static_assert(Storage::kScope == StorageScope::Workgroup, "Tile storage must be workgroup-visible.");
        static_assert(Storage::kBarrierSafe, "Tile storage must survive workgroup barriers.");

        const uint tid = flatten_local_id<WorkgroupX>(local_id);
        const uint workgroup_size = uint(WorkgroupX * WorkgroupY);

        [unroll]
        for (int iter = 0; iter < LoadsPerThread; ++iter) {
            const uint idx = uint(iter) * workgroup_size + tid;
            const uint row = idx / uint(TileK2);
            const uint col2 = idx % uint(TileK2);

            half v0 = half(0.0);
            half v1 = half(0.0);

            const uint global_row = region.row + row;
            const uint global_col0 = region.col + col_base + 2u * col2;
            const uint global_col1 = global_col0 + 1u;

            if (tensor.contains(global_row, global_col0)) {
                v0 = src[Layout::offset(tensor, global_row, global_col0)];
            }
            if (tensor.contains(global_row, global_col1)) {
                v1 = src[Layout::offset(tensor, global_row, global_col1)];
            }

            Storage::store(row, col2, half2(v0, v1));
        }
    }

    public static void scatter_register_tile<
        Layout : ITensor2DLayout,
        int ThreadRows,
        int ThreadCols>(
            RWStructuredBuffer<half> dst,
            GlobalTensor2D tensor,
            RegisterTile<ThreadRows, ThreadCols> accum,
            SharedTileRegion region,
            uint3 local_id)
    {
        const uint sub_m = local_id.y * uint(ThreadRows);
        const uint sub_n = local_id.x * uint(ThreadCols);

        [unroll]
        for (int row = 0; row < ThreadRows; ++row) {
            [unroll]
            for (int col = 0; col < ThreadCols; ++col) {
                const uint global_row = region.row + sub_m + uint(row);
                const uint global_col = region.col + sub_n + uint(col);
                if (tensor.contains(global_row, global_col)) {
                    dst[Layout::offset(tensor, global_row, global_col)] = half(accum.values[row][col]);
                }
            }
        }
    }

    public static void gather_slice<Storage : IFloatStorage1D, int WorkgroupSize>(
        ByteAddressBuffer src,
        Fp16Span slice,
        uint tid)
    {
        static_assert(Storage::kScope == StorageScope::Workgroup, "Tile scratch must be workgroup-visible.");
        static_assert(Storage::kBarrierSafe, "Tile scratch must survive a workgroup barrier.");

        const uint vectorized_end = slice.vectorized_count8();
        const uint chunk = 8u;
        const uint chunk_stride = uint(WorkgroupSize) * chunk;

        for (uint index = tid * chunk; index < vectorized_end; index += chunk_stride) {
            const FloatPack8 values = Thread::load_fp16_pack8(src, slice.offset(index));
            Storage::store(index + 0u, values.lo.x);
            Storage::store(index + 1u, values.lo.y);
            Storage::store(index + 2u, values.lo.z);
            Storage::store(index + 3u, values.lo.w);
            Storage::store(index + 4u, values.hi.x);
            Storage::store(index + 5u, values.hi.y);
            Storage::store(index + 6u, values.hi.z);
            Storage::store(index + 7u, values.hi.w);
        }

        for (uint index = tid; index < slice.count; index += uint(WorkgroupSize)) {
            if (index < vectorized_end) {
                continue;
            }
            Storage::store(index, Thread::load_fp16(src, slice.offset(index)));
        }
    }

    public static float leader_reduce_sum<Storage : IFloatStorage1D, int WorkgroupSize>(float value, uint tid)
    {
        static_assert(Storage::kExtent >= uint(WorkgroupSize), "Reduction scratch must cover the whole workgroup.");
        static_assert(Storage::kScope == StorageScope::Workgroup, "Reduction scratch must be workgroup-visible.");
        static_assert(Storage::kBarrierSafe, "Reduction scratch must survive workgroup barriers.");

        const float subgroup_sum = Subgroup::reduce_sum(value);
        Storage::store(tid, Subgroup::is_leader() ? subgroup_sum : 0.0);
        GroupMemoryBarrierWithGroupSync();

        if (tid != 0u) {
            return 0.0;
        }

        float total = 0.0;
        [unroll]
        for (uint index = 0u; index < uint(WorkgroupSize); ++index) {
            total += Storage::load(index);
        }
        return total;
    }

    public static float leader_reduce_max<Storage : IFloatStorage1D, int WorkgroupSize>(float value, uint tid)
    {
        static_assert(Storage::kExtent >= uint(WorkgroupSize), "Reduction scratch must cover the whole workgroup.");
        static_assert(Storage::kScope == StorageScope::Workgroup, "Reduction scratch must be workgroup-visible.");
        static_assert(Storage::kBarrierSafe, "Reduction scratch must survive workgroup barriers.");

        const float subgroup_max = Subgroup::reduce_max(value);
        Storage::store(tid, Subgroup::is_leader() ? subgroup_max : -3.402823466e38);
        GroupMemoryBarrierWithGroupSync();

        if (tid != 0u) {
            return 0.0;
        }

        float total = Storage::load(0u);
        [unroll]
        for (uint index = 1u; index < uint(WorkgroupSize); ++index) {
            total = max(total, Storage::load(index));
        }
        return total;
    }

    public static float broadcast_scalar<Storage : IFloatStorage1D>(float value, uint tid)
    {
        static_assert(Storage::kExtent > 0u, "Broadcast scratch must have at least one lane.");
        static_assert(Storage::kScope == StorageScope::Workgroup, "Broadcast scratch must be workgroup-visible.");
        static_assert(Storage::kBarrierSafe, "Broadcast scratch must survive workgroup barriers.");

        if (tid == 0u) {
            Storage::store(0u, value);
        }

        GroupMemoryBarrierWithGroupSync();
        return Storage::load(0u);
    }

    public static float4 leader_reduce_sum4<Storage : IFloat4Storage1D, int WorkgroupSize>(float4 value, uint tid)
    {
        static_assert(Storage::kExtent >= uint(WorkgroupSize), "Reduction scratch must cover the whole workgroup.");
        static_assert(Storage::kScope == StorageScope::Workgroup, "Reduction scratch must be workgroup-visible.");
        static_assert(Storage::kBarrierSafe, "Reduction scratch must survive workgroup barriers.");

        const float4 subgroup_sum = Subgroup::reduce_sum(value);
        Storage::store(tid, Subgroup::is_leader() ? subgroup_sum : float4(0.0));
        GroupMemoryBarrierWithGroupSync();

        if (tid != 0u) {
            return float4(0.0);
        }

        float4 total = float4(0.0);
        [unroll]
        for (uint index = 0u; index < uint(WorkgroupSize); ++index) {
            total += Storage::load(index);
        }
        return total;
    }
}

public struct Thread
{
    public static uint linear_id(uint3 dispatch_id)
    {
        return dispatch_id.x;
    }

    public static float load_fp16(ByteAddressBuffer src, uint elem_index)
    {
        const uint byte_offset = elem_index * 2u;
        const uint packed = src.Load(int(byte_offset & ~3u));
        const uint lane = (byte_offset & 2u) >> 1u;
        const uint bits = (lane == 0u) ? (packed & 0xffffu) : (packed >> 16);
        return f16tof32(bits);
    }

    public static float4 gather_storage4<Storage : IFloatStorage1D>(uint index)
    {
        return float4(
            Storage::load(index + 0u),
            Storage::load(index + 1u),
            Storage::load(index + 2u),
            Storage::load(index + 3u));
    }

    public static FloatPack8 load_fp16_pack8(ByteAddressBuffer src, uint base_elem)
    {
        if ((base_elem & 1u) != 0u) {
            FloatPack8 values;
            values.lo = float4(
                load_fp16(src, base_elem + 0u),
                load_fp16(src, base_elem + 1u),
                load_fp16(src, base_elem + 2u),
                load_fp16(src, base_elem + 3u));
            values.hi = float4(
                load_fp16(src, base_elem + 4u),
                load_fp16(src, base_elem + 5u),
                load_fp16(src, base_elem + 6u),
                load_fp16(src, base_elem + 7u));
            return values;
        }

        const uint byte_offset = base_elem * 2u;
        uint4 packed;
        if ((byte_offset & 15u) == 0u) {
            packed = src.Load4Aligned(int(byte_offset));
        } else {
            packed = src.Load4(int(byte_offset));
        }

        const uint4 lo_bits = uint4(
            packed.x & 0xffffu,
            packed.x >> 16,
            packed.y & 0xffffu,
            packed.y >> 16);
        const uint4 hi_bits = uint4(
            packed.z & 0xffffu,
            packed.z >> 16,
            packed.w & 0xffffu,
            packed.w >> 16);

        FloatPack8 values;
        values.lo = float4(
            f16tof32(lo_bits.x),
            f16tof32(lo_bits.y),
            f16tof32(lo_bits.z),
            f16tof32(lo_bits.w));
        values.hi = float4(
            f16tof32(hi_bits.x),
            f16tof32(hi_bits.y),
            f16tof32(hi_bits.z),
            f16tof32(hi_bits.w));
        return values;
    }

    public static float4 map_values4<Op : IPointwiseOp>(float4 values)
    {
        return float4(
            Op::apply(values.x),
            Op::apply(values.y),
            Op::apply(values.z),
            Op::apply(values.w));
    }

    public static float4 zip_values4<Op : IBinaryPointwiseOp>(float4 lhs, float4 rhs)
    {
        return float4(
            Op::apply(lhs.x, rhs.x),
            Op::apply(lhs.y, rhs.y),
            Op::apply(lhs.z, rhs.z),
            Op::apply(lhs.w, rhs.w));
    }

    public static FloatPack8 zip_values8<Op : IBinaryPointwiseOp>(FloatPack8 lhs, FloatPack8 rhs)
    {
        FloatPack8 values;
        values.lo = zip_values4<Op>(lhs.lo, rhs.lo);
        values.hi = zip_values4<Op>(lhs.hi, rhs.hi);
        return values;
    }

    public static void scatter_values4(
        RWStructuredBuffer<half> dst,
        RWFp16Span dst_span,
        uint base_index,
        uint index_stride,
        float4 values)
    {
        const uint index0 = base_index;
        const uint index1 = base_index + index_stride;
        const uint index2 = base_index + 2u * index_stride;
        const uint index3 = base_index + 3u * index_stride;

        if (dst_span.contains(index0)) {
            dst[dst_span.offset(index0)] = half(values.x);
        }
        if (dst_span.contains(index1)) {
            dst[dst_span.offset(index1)] = half(values.y);
        }
        if (dst_span.contains(index2)) {
            dst[dst_span.offset(index2)] = half(values.z);
        }
        if (dst_span.contains(index3)) {
            dst[dst_span.offset(index3)] = half(values.w);
        }
    }

    public static void scatter_contiguous_values4(
        RWStructuredBuffer<half> dst,
        uint base_elem,
        uint count,
        float4 values)
    {
        if (0u < count) {
            dst[base_elem + 0u] = half(values.x);
        }
        if (1u < count) {
            dst[base_elem + 1u] = half(values.y);
        }
        if (2u < count) {
            dst[base_elem + 2u] = half(values.z);
        }
        if (3u < count) {
            dst[base_elem + 3u] = half(values.w);
        }
    }

    public static void scatter_contiguous_values8(
        RWStructuredBuffer<half> dst,
        uint base_elem,
        uint count,
        FloatPack8 values)
    {
        scatter_contiguous_values4(dst, base_elem, min(count, 4u), values.lo);
        if (count > 4u) {
            scatter_contiguous_values4(dst, base_elem + 4u, count - 4u, values.hi);
        }
    }

    public static void copy_span(
        RWStructuredBuffer<half> dst,
        RWFp16Span dst_span,
        ByteAddressBuffer src,
        Fp16Span src_span)
    {
        const uint vectorized_end = (dst_span.is_contiguous() && src_span.supports_vectorized_load8())
            ? (dst_span.count & ~7u)
            : 0u;

        uint index = 0u;
        for (; index < vectorized_end; index += 8u) {
            scatter_contiguous_values8(
                dst,
                dst_span.offset(index),
                8u,
                load_fp16_pack8(src, src_span.offset(index)));
        }

        for (; index < dst_span.count; ++index) {
            dst[dst_span.offset(index)] = half(load_fp16(src, src_span.offset(index)));
        }
    }
}

public struct Subgroup
{
    public static uint lane_id()
    {
        return WaveGetLaneIndex();
    }

    public static uint lane_count()
    {
        return WaveGetLaneCount();
    }

    public static bool is_leader()
    {
        return lane_id() == 0u;
    }

    public static float reduce_sum(float value)
    {
        return WaveActiveSum(value);
    }

    public static float reduce_max(float value)
    {
        return WaveActiveMax(value);
    }

    public static float4 reduce_sum(float4 value)
    {
        return float4(
            WaveActiveSum(value.x),
            WaveActiveSum(value.y),
            WaveActiveSum(value.z),
            WaveActiveSum(value.w));
    }

    public static float read_lane(float value, uint lane)
    {
        return WaveReadLaneAt(value, lane);
    }

    public static void mma<
        Arch : IMmaArch,
        StorageA : IHalf2TileStorage,
        StorageB : IHalf2TileStorage,
        int ThreadRows,
        int ThreadCols,
        int TileK2>(
            inout RegisterTile<ThreadRows, ThreadCols> accum,
            uint3 local_id)
    {
        const uint sub_m = local_id.y * uint(ThreadRows);
        const uint sub_n = local_id.x * uint(ThreadCols);
        Arch::mma<StorageA, StorageB, ThreadRows, ThreadCols, TileK2>(accum, sub_m, sub_n);
    }
}
