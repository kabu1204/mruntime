public struct GlobalTensor1D
{
    public uint size;
}

public struct GlobalTensor2D
{
    public uint rows;
    public uint cols;
    public uint stride;
}

public struct GlobalVector4Tensor1D
{
    public uint size4;
}

public struct GlobalVector4Tensor2D
{
    public uint rows;
    public uint cols4;
    public uint stride4;
}

public struct GateUpInterleavedTensor
{
    public uint num_tokens;
    public uint intermediate_size;
}

public struct BSHDTensor
{
    public uint batch;
    public uint seq_len;
    public uint heads;
    public uint head_dim;
}

public struct BHSDTensor
{
    public uint batch;
    public uint heads;
    public uint seq_len;
    public uint head_dim;
}

public struct RopeLookupTensor
{
    public uint position_offset;
    public uint half_dim;
}

public struct SharedTileRegion
{
    public uint row;
    public uint col;
}

public struct RowBlock4
{
    public uint row_base;
}
