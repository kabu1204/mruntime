#include <cassert>
#include <cmath>
#include <iostream>

#include "mruntime/cpu_backend.h"
#include "mruntime/profiling_backend.h"
#include "mruntime/tensor.h"

using namespace mruntime;

static void test_profiling_backend_gemm_records_stats_and_trace() {
    CpuBackend cpu;
    ProfilingBackend::Options opts;
    opts.enabled = true;
    opts.trace_enabled = true;
    opts.trace_path = "";
    ProfilingBackend backend(cpu, opts);

    Tensor A = Tensor::zeros(Shape({2, 3}), DType::FP16);
    Tensor B = Tensor::zeros(Shape({3, 4}), DType::FP16);
    Tensor C = Tensor::zeros(Shape({2, 4}), DType::FP16);

    uint16_t* a = A.data_ptr<uint16_t>();
    uint16_t* b = B.data_ptr<uint16_t>();
    for (int i = 0; i < 6; ++i) a[i] = float_to_fp16_bits(static_cast<float>(i + 1));
    for (int i = 0; i < 12; ++i) b[i] = float_to_fp16_bits(static_cast<float>(i + 1));

    backend.gemm(A, B, C);

    const uint16_t* c = C.data_ptr<uint16_t>();
    assert(std::abs(fp16_bits_to_float(c[0]) - 38.0f) < 0.05f);
    assert(std::abs(fp16_bits_to_float(c[1]) - 44.0f) < 0.05f);

    const auto snap = backend.profiler().snapshot(BackendOp::Gemm);
    assert(snap.calls == 1);
    assert(snap.total_ns > 0);
    assert(snap.max_ns >= snap.min_ns);

    assert(backend.trace_writer().event_count() == 1);
    const std::string json = backend.trace_writer().to_json();
    assert(json.find("backend.gemm") != std::string::npos);
}

static void test_disabled_profiling_has_no_side_effects() {
    CpuBackend cpu;
    ProfilingBackend::Options opts;
    opts.enabled = false;
    opts.trace_enabled = true;
    ProfilingBackend backend(cpu, opts);

    Tensor A = Tensor::zeros(Shape({2, 2}), DType::FP16);
    Tensor B = Tensor::zeros(Shape({2, 2}), DType::FP16);
    Tensor C = Tensor::zeros(Shape({2, 2}), DType::FP16);

    backend.gemm(A, B, C);

    const auto snap = backend.profiler().snapshot(BackendOp::Gemm);
    assert(snap.calls == 0);
    assert(backend.trace_writer().event_count() == 0);
}

int main() {
    test_profiling_backend_gemm_records_stats_and_trace();
    test_disabled_profiling_has_no_side_effects();
    std::cout << "profiling_backend_test PASSED\n";
    return 0;
}
