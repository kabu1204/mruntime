import register_tile;
import storage_traits;

public interface IMmaArch
{
    static void mma<
        StorageA : IHalf2TileStorage,
        StorageB : IHalf2TileStorage,
        int ThreadRows,
        int ThreadCols,
        int TileK2>(
            inout RegisterTile<ThreadRows, ThreadCols> accum,
            uint sub_m,
            uint sub_n);
}

public struct VectorSIMTArch : IMmaArch
{
    public static void mma<
        StorageA : IHalf2TileStorage,
        StorageB : IHalf2TileStorage,
        int ThreadRows,
        int ThreadCols,
        int TileK2>(
            inout RegisterTile<ThreadRows, ThreadCols> accum,
            uint sub_m,
            uint sub_n)
    {
        [unroll]
        for (uint k2 = 0u; k2 < uint(TileK2); ++k2) {
            float bx0[ThreadCols];
            float bx1[ThreadCols];

            [unroll]
            for (int col = 0; col < ThreadCols; ++col) {
                const half2 b_pair = StorageB::load(sub_n + uint(col), k2);
                bx0[col] = float(b_pair.x);
                bx1[col] = float(b_pair.y);
            }

            [unroll]
            for (int row = 0; row < ThreadRows; ++row) {
                const half2 a_pair = StorageA::load(sub_m + uint(row), k2);
                const float ax0 = float(a_pair.x);
                const float ax1 = float(a_pair.y);

                [unroll]
                for (int col = 0; col < ThreadCols; ++col) {
                    accum.values[row][col] = fma(ax0, bx0[col], accum.values[row][col]);
                    accum.values[row][col] = fma(ax1, bx1[col], accum.values[row][col]);
                }
            }
        }
    }
}
