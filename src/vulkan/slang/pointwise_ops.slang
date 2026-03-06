public interface IPointwiseOp
{
    static float apply(float value);
}

public interface IBinaryPointwiseOp
{
    static float apply(float lhs, float rhs);
}

public struct IdentityOp : IPointwiseOp
{
    public static float apply(float value)
    {
        return value;
    }
}

public struct SiLUOp : IPointwiseOp
{
    public static float apply(float value)
    {
        return value / (1.0 + exp(-value));
    }
}

public struct AddOp : IBinaryPointwiseOp
{
    public static float apply(float lhs, float rhs)
    {
        return lhs + rhs;
    }
}
