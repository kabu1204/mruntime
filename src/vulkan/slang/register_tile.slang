public struct RegisterTile<int Rows, int Cols>
{
    public float values[Rows][Cols];

    [mutating]
    public void clear()
    {
        [unroll]
        for (int row = 0; row < Rows; ++row) {
            [unroll]
            for (int col = 0; col < Cols; ++col) {
                values[row][col] = 0.0;
            }
        }
    }
}
