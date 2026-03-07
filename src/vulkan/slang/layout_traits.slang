import tile_views;

public interface ITensor2DLayout
{
    static uint offset(GlobalTensor2D tensor, uint row, uint col);
}

public struct RowMajorTensorLayout : ITensor2DLayout
{
    public static uint offset(GlobalTensor2D tensor, uint row, uint col)
    {
        return row * tensor.stride + col;
    }
}

public struct ColMajorTensorLayout : ITensor2DLayout
{
    public static uint offset(GlobalTensor2D tensor, uint row, uint col)
    {
        return col * tensor.stride + row;
    }
}

public struct StridedLayout
{
    public static uint offset(uint base, uint stride, uint index)
    {
        return base + index * stride;
    }

    public static uint offset2(uint base, uint stride0, uint stride1, uint i0, uint i1)
    {
        return base + i0 * stride0 + i1 * stride1;
    }

    public static uint offset3(
        uint base,
        uint stride0,
        uint stride1,
        uint stride2,
        uint i0,
        uint i1,
        uint i2)
    {
        return base + i0 * stride0 + i1 * stride1 + i2 * stride2;
    }

    public static uint offset4(
        uint base,
        uint stride0,
        uint stride1,
        uint stride2,
        uint stride3,
        uint i0,
        uint i1,
        uint i2,
        uint i3)
    {
        return base + i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
    }
}
