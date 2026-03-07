public interface IPointwiseOp
{
    static float apply(float value);
}

public interface IBinaryPointwiseOp
{
    static float apply(float lhs, float rhs);
}
