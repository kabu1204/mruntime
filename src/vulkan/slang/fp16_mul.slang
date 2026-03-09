// FP16 elementwise multiply: out[i] = a[i] * b[i]

import pointwise_ops;
import scope_primitives;
import tile_views;

struct PushConstants {
    uint n;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

[vk::binding(0, 0)]
ByteAddressBuffer a;

[vk::binding(1, 0)]
ByteAddressBuffer b;

[vk::binding(2, 0)]
RWStructuredBuffer<half> out_data;

public struct MulOp : IBinaryPointwiseOp
{
    public static float apply(float lhs, float rhs)
    {
        return lhs * rhs;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const Fp16Span lhs = {0u, 1u, pc.n};
    const Fp16Span rhs = {0u, 1u, pc.n};
    const RWFp16Span out_span = {0u, 1u, pc.n};
    const uint chunk_id = Thread::linear_id(tid);
    const uint chunk_base = chunk_id * 8u;
    if (!out_span.contains(chunk_base)) {
        return;
    }

    const uint chunk_count = min(8u, out_span.count - chunk_base);
    const Fp16Span lhs_chunk = lhs.subspan(chunk_base, chunk_count);
    const Fp16Span rhs_chunk = rhs.subspan(chunk_base, chunk_count);
    const RWFp16Span out_chunk = out_span.subspan(chunk_base, chunk_count);

    if (chunk_count == 8u && lhs_chunk.supports_vectorized_load8() && rhs_chunk.supports_vectorized_load8()) {
        Thread::scatter_contiguous_values8(
            out_data,
            out_chunk.base_elem,
            chunk_count,
            Thread::zip_values8<MulOp>(
                Thread::load_fp16_pack8(a, lhs_chunk.base_elem),
                Thread::load_fp16_pack8(b, rhs_chunk.base_elem)));
        return;
    }

    for (uint lane = 0u; lane < chunk_count; ++lane) {
        out_data[out_chunk.offset(lane)] = half(MulOp::apply(
            Thread::load_fp16(a, lhs_chunk.offset(lane)),
            Thread::load_fp16(b, rhs_chunk.offset(lane))));
    }
}
