import tile_views;

public struct RowMajorTensorLayout
{
    public static uint offset(GlobalTensor2D tensor, uint row, uint col)
    {
        return row * tensor.stride + col;
    }
}

public struct RowMajorVector4Layout
{
    public static uint offset(GlobalVector4Tensor2D tensor, uint row, uint col4)
    {
        return row * tensor.stride4 + col4;
    }
}

public struct GateUpInterleavedLayout
{
    public static uint total_elements(GateUpInterleavedTensor tensor)
    {
        return tensor.num_tokens * tensor.intermediate_size;
    }

    public static uint row_offset(GateUpInterleavedTensor tensor, uint token)
    {
        return token * 2u * tensor.intermediate_size;
    }

    public static uint gate_offset(GateUpInterleavedTensor tensor, uint token, uint index)
    {
        return row_offset(tensor, token) + index;
    }

    public static uint up_offset(GateUpInterleavedTensor tensor, uint token, uint index)
    {
        return row_offset(tensor, token) + tensor.intermediate_size + index;
    }
}

public struct BSHDLayout
{
    public static uint offset(BSHDTensor tensor, uint batch, uint seq, uint head, uint dim)
    {
        return (((batch * tensor.seq_len + seq) * tensor.heads + head) * tensor.head_dim) + dim;
    }

    public static uint row_task_count(BSHDTensor tensor)
    {
        return tensor.batch * tensor.seq_len * tensor.heads;
    }
}

public struct BHSDLayout
{
    public static uint offset(BHSDTensor tensor, uint batch, uint head, uint seq, uint dim)
    {
        return (((batch * tensor.heads + head) * tensor.seq_len + seq) * tensor.head_dim) + dim;
    }
}

public struct RopeLookupLayout
{
    public static uint offset(RopeLookupTensor tensor, uint position, uint dim_half)
    {
        return position * tensor.half_dim + dim_half;
    }
}
