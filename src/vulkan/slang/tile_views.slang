public struct GlobalTensor1D
{
    public uint size;

    public bool contains(uint index)
    {
        return index < size;
    }
}

public struct GlobalTensor2D
{
    public uint rows;
    public uint cols;
    public uint stride;

    public bool contains(uint row, uint col)
    {
        return row < rows && col < cols;
    }
}

public struct SharedTileRegion
{
    public uint row;
    public uint col;
}

public struct Fp16Span
{
    public uint base_elem;
    public uint stride_elem;
    public uint count;

    public uint offset(uint index)
    {
        return base_elem + index * stride_elem;
    }

    public bool contains(uint index)
    {
        return index < count;
    }

    public bool is_contiguous()
    {
        return stride_elem == 1u;
    }

    public bool supports_vectorized_load8()
    {
        return is_contiguous() && ((base_elem & 1u) == 0u);
    }

    public uint vectorized_count8()
    {
        return supports_vectorized_load8() ? (count & ~7u) : 0u;
    }

    public Fp16Span subspan(uint start, uint sub_count)
    {
        Fp16Span span = {offset(start), stride_elem, sub_count};
        return span;
    }
}

public struct RWFp16Span
{
    public uint base_elem;
    public uint stride_elem;
    public uint count;

    public uint offset(uint index)
    {
        return base_elem + index * stride_elem;
    }

    public bool contains(uint index)
    {
        return index < count;
    }

    public bool is_contiguous()
    {
        return stride_elem == 1u;
    }

    public RWFp16Span subspan(uint start, uint sub_count)
    {
        RWFp16Span span = {offset(start), stride_elem, sub_count};
        return span;
    }
}

public struct Fp16SliceSet
{
    public uint base_elem;
    public uint slice_stride_elem;
    public uint value_stride_elem;
    public uint slice_count;
    public uint value_count;

    public Fp16Span slice(uint slice_index)
    {
        Fp16Span span = {base_elem + slice_index * slice_stride_elem, value_stride_elem, value_count};
        return span;
    }

    public Fp16SliceSet window(uint slice_base, uint window_slice_count)
    {
        Fp16SliceSet view = {
            base_elem + slice_base * slice_stride_elem,
            slice_stride_elem,
            value_stride_elem,
            window_slice_count,
            value_count,
        };
        return view;
    }
}
