import compute_arch;
import layout_traits;
import pointwise_ops;
import register_tile;
import storage_traits;
import tile_views;

public struct Workgroup
{
    public static uint flatten_local_id<int WorkgroupX>(uint3 local_id)
    {
        return local_id.y * uint(WorkgroupX) + local_id.x;
    }

    public static void load_matrix_rows_half2_tile<
        Storage : IHalf2TileStorage,
        int WorkgroupX,
        int WorkgroupY,
        int TileRows,
        int TileK2,
        int LoadsPerThread>(
            StructuredBuffer<half> src,
            GlobalTensor2D tensor,
            SharedTileRegion region,
            uint k_base,
            uint3 local_id)
    {
        const uint tid = flatten_local_id<WorkgroupX>(local_id);
        const uint workgroup_size = uint(WorkgroupX * WorkgroupY);

        [unroll]
        for (int iter = 0; iter < LoadsPerThread; ++iter) {
            const uint idx = uint(iter) * workgroup_size + tid;
            const uint row = idx / uint(TileK2);
            const uint k2 = idx % uint(TileK2);

            half v0 = half(0.0);
            half v1 = half(0.0);

            const uint global_row = region.row + row;
            const uint global_col0 = region.col + k_base + 2u * k2;
            const uint global_col1 = global_col0 + 1u;

            if (global_row < tensor.rows && global_col0 < tensor.cols) {
                v0 = src[RowMajorTensorLayout::offset(tensor, global_row, global_col0)];
            }
            if (global_row < tensor.rows && global_col1 < tensor.cols) {
                v1 = src[RowMajorTensorLayout::offset(tensor, global_row, global_col1)];
            }

            Storage::store(row, k2, half2(v0, v1));
        }
    }

    public static void store_matrix_tile<
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
                if (global_row < tensor.rows && global_col < tensor.cols) {
                    dst[RowMajorTensorLayout::offset(tensor, global_row, global_col)] = half(accum.values[row][col]);
                }
            }
        }
    }

    public static float4 load_rows4_dot(
        StructuredBuffer<half4> x,
        StructuredBuffer<half4> weights,
        GlobalVector4Tensor2D weight_matrix,
        RowBlock4 rows,
        uint col4)
    {
        const float4 x_value = float4(x[col4]);
        float4 partial = float4(0.0);

        [unroll]
        for (int row = 0; row < 4; ++row) {
            const uint global_row = rows.row_base + uint(row);
            if (global_row < weight_matrix.rows) {
                partial[row] = dot(x_value, float4(weights[RowMajorVector4Layout::offset(weight_matrix, global_row, col4)]));
            }
        }

        return partial;
    }

    public static float4 accumulate_rows4_vec4_dot<int WorkgroupSize>(
        StructuredBuffer<half4> x,
        StructuredBuffer<half4> weights,
        GlobalVector4Tensor1D vector,
        GlobalVector4Tensor2D weight_matrix,
        RowBlock4 rows,
        uint tid)
    {
        float4 partial = float4(0.0);
        const uint stride = uint(WorkgroupSize);

        for (uint col4 = tid; col4 < vector.size4; col4 += 2u * stride) {
            partial += load_rows4_dot(x, weights, weight_matrix, rows, col4);

            const uint next_col4 = col4 + stride;
            if (next_col4 < vector.size4) {
                partial += load_rows4_dot(x, weights, weight_matrix, rows, next_col4);
            }
        }

        return partial;
    }

    public static void store_rows4(
        RWStructuredBuffer<half> dst,
        GlobalTensor1D vector,
        RowBlock4 rows,
        float4 values,
        uint tid)
    {
        if (tid != 0u) {
            return;
        }

        [unroll]
        for (int row = 0; row < 4; ++row) {
            const uint global_row = rows.row_base + uint(row);
            if (global_row < vector.size) {
                dst[global_row] = half(values[row]);
            }
        }
    }

    public static float tree_reduce_sum<Storage : IFloatStorage1D, int WorkgroupSize>(float wave_sum, uint tid)
    {
        const uint lane = WaveGetLaneIndex();
        const uint wave_size = WaveGetLaneCount();
        const uint wave_id = tid / wave_size;
        const uint wave_count = (uint(WorkgroupSize) + wave_size - 1u) / wave_size;

        if (wave_count == 1u) {
            if (lane == 0u) {
                return wave_sum;
            }
            return 0.0;
        }

        if (lane == 0u) {
            Storage::store(wave_id, wave_sum);
        }

        GroupMemoryBarrierWithGroupSync();

        if (tid != 0u) {
            return 0.0;
        }

        float total = 0.0;
        for (uint wave = 0u; wave < wave_count; ++wave) {
            total += Storage::load(wave);
        }
        return total;
    }

    public static float4 tree_reduce_sum<Storage : IFloat4Storage1D, int WorkgroupSize>(float4 wave_sum, uint tid)
    {
        const uint lane = WaveGetLaneIndex();
        const uint wave_size = WaveGetLaneCount();
        const uint wave_id = tid / wave_size;
        const uint wave_count = (uint(WorkgroupSize) + wave_size - 1u) / wave_size;

        if (wave_count == 1u) {
            if (lane == 0u) {
                return wave_sum;
            }
            return float4(0.0);
        }

        if (lane == 0u) {
            Storage::store(wave_id, wave_sum);
        }

        GroupMemoryBarrierWithGroupSync();

        if (tid != 0u) {
            return float4(0.0);
        }

        float4 total = float4(0.0);
        for (uint wave = 0u; wave < wave_count; ++wave) {
            total += Storage::load(wave);
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

    public static void apply_binary_elementwise<Op : IBinaryPointwiseOp>(
        StructuredBuffer<half> lhs,
        StructuredBuffer<half> rhs,
        RWStructuredBuffer<half> dst,
        GlobalTensor1D tensor,
        uint index)
    {
        if (index >= tensor.size) {
            return;
        }

        dst[index] = half(Op::apply(float(lhs[index]), float(rhs[index])));
    }

    public static void apply_interleaved_gate_up<Activation : IPointwiseOp>(
        StructuredBuffer<half> src,
        RWStructuredBuffer<half> dst,
        GateUpInterleavedTensor tensor,
        uint index)
    {
        if (index >= GateUpInterleavedLayout::total_elements(tensor)) {
            return;
        }

        const uint token = index / tensor.intermediate_size;
        const uint element = index % tensor.intermediate_size;
        const float gate = float(src[GateUpInterleavedLayout::gate_offset(tensor, token, element)]);
        const float up = float(src[GateUpInterleavedLayout::up_offset(tensor, token, element)]);
        dst[index] = half(Activation::apply(gate) * up);
    }

    public static void copy_span(
        RWStructuredBuffer<half> dst,
        uint dst_base,
        StructuredBuffer<half> src,
        uint src_base,
        uint count)
    {
        for (uint i = 0u; i < count; ++i) {
            dst[dst_base + i] = src[src_base + i];
        }
    }

    public static void apply_rope(
        RWStructuredBuffer<half> q_data,
        RWStructuredBuffer<half> k_data,
        StructuredBuffer<float2> rope_cos_sin,
        BSHDTensor q_tensor,
        BSHDTensor k_tensor,
        RopeLookupTensor rope,
        uint index)
    {
        const uint total_heads = q_tensor.heads + k_tensor.heads;
        const uint total = q_tensor.batch * q_tensor.seq_len * total_heads * rope.half_dim;
        if (index >= total) {
            return;
        }

        const uint dim_half = index % rope.half_dim;
        const uint head = (index / rope.half_dim) % total_heads;
        const uint seq = (index / (rope.half_dim * total_heads)) % q_tensor.seq_len;
        const uint batch = index / (rope.half_dim * total_heads * q_tensor.seq_len);

        const float2 cs = rope_cos_sin[RopeLookupLayout::offset(rope, rope.position_offset + seq, dim_half)];
        const float cos_value = cs.x;
        const float sin_value = cs.y;

        if (head < q_tensor.heads) {
            const uint base = BSHDLayout::offset(q_tensor, batch, seq, head, 0u);
            const float x0 = float(q_data[base + dim_half]);
            const float x1 = float(q_data[base + rope.half_dim + dim_half]);
            q_data[base + dim_half] = half(x0 * cos_value - x1 * sin_value);
            q_data[base + rope.half_dim + dim_half] = half(x1 * cos_value + x0 * sin_value);
            return;
        }

        const uint kv_head = head - q_tensor.heads;
        const uint base = BSHDLayout::offset(k_tensor, batch, seq, kv_head, 0u);
        const float x0 = float(k_data[base + dim_half]);
        const float x1 = float(k_data[base + rope.half_dim + dim_half]);
        k_data[base + dim_half] = half(x0 * cos_value - x1 * sin_value);
        k_data[base + rope.half_dim + dim_half] = half(x1 * cos_value + x0 * sin_value);
    }

    public static void transpose_bshd_to_bhsd(
        StructuredBuffer<half> src,
        RWStructuredBuffer<half> dst,
        BSHDTensor src_tensor,
        BHSDTensor dst_tensor,
        uint task_id)
    {
        if (task_id >= BSHDLayout::row_task_count(src_tensor)) {
            return;
        }

        const uint seq = task_id % src_tensor.seq_len;
        const uint head = (task_id / src_tensor.seq_len) % src_tensor.heads;
        const uint batch = task_id / (src_tensor.seq_len * src_tensor.heads);

        copy_span(
            dst,
            BHSDLayout::offset(dst_tensor, batch, head, seq, 0u),
            src,
            BSHDLayout::offset(src_tensor, batch, seq, head, 0u),
            src_tensor.head_dim);
    }

    public static void copy_kv_cache(
        StructuredBuffer<half> k_in,
        StructuredBuffer<half> v_in,
        RWStructuredBuffer<half> k_cache,
        RWStructuredBuffer<half> v_cache,
        BSHDTensor src_tensor,
        BHSDTensor dst_tensor,
        uint position_offset,
        uint task_id)
    {
        if (task_id >= BSHDLayout::row_task_count(src_tensor)) {
            return;
        }

        const uint head = task_id % src_tensor.heads;
        const uint seq = (task_id / src_tensor.heads) % src_tensor.seq_len;
        const uint batch = task_id / (src_tensor.heads * src_tensor.seq_len);
        const uint cache_seq = position_offset + seq;

        const uint src_base = BSHDLayout::offset(src_tensor, batch, seq, head, 0u);
        const uint dst_base = BHSDLayout::offset(dst_tensor, batch, head, cache_seq, 0u);

        copy_span(k_cache, dst_base, k_in, src_base, src_tensor.head_dim);
        copy_span(v_cache, dst_base, v_in, src_base, src_tensor.head_dim);
    }
}

public struct Subgroup
{
    public static float reduce_sum(float value)
    {
        return WaveActiveSum(value);
    }

    public static float4 reduce_sum(float4 value)
    {
        return float4(
            WaveActiveSum(value.x),
            WaveActiveSum(value.y),
            WaveActiveSum(value.z),
            WaveActiveSum(value.w));
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
