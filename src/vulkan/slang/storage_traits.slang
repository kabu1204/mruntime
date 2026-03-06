public interface IHalf2TileStorage
{
    static void store(uint row, uint col, half2 value);
    static half2 load(uint row, uint col);
}

public interface IFloatStorage1D
{
    static void store(uint index, float value);
    static float load(uint index);
}

public interface IFloat4Storage1D
{
    static void store(uint index, float4 value);
    static float4 load(uint index);
}
