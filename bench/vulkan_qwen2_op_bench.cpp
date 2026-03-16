#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mruntime/qwen_config.h"
#include "mruntime/trace.h"

#include "e2e_bench_common.h"
#include "vk_buffer_arena.h"
#include "vk_context.h"
#include "vk_fp16_ops.h"
#include "vk_helpers.h"
#include "vk_kernel_runtime.h"

using namespace mruntime::vulkan;

namespace {

struct Args {
    std::string model_dir;
    size_t prompt_len = 2016;
    size_t max_new_tokens = 32;
    size_t max_seq_len = 2048;
    size_t max_batch_tokens = 64;
    int warmup_iters = 3;
    int bench_iters = 10;
};

struct WorkloadCounts {
    size_t full_prefill_chunks = 0;
    size_t last_chunk_tokens = 0;
    size_t decode_calls = 0;
    size_t logit_steps = 0;
};

struct BenchContext {
    VkContext context;
    VkKernelRuntime runtime;
    VkFp16Ops fp16_ops;
    VkDeviceSize alignment = 0;

    static auto Create() -> BenchContext {
        BenchContext bc;
        bc.context = VkContext::Create();
        bc.runtime = VkKernelRuntime::Create(bc.context);
        bc.runtime.set_timing_enabled(true);
        bc.fp16_ops = VkFp16Ops::Create(&bc.runtime);
        bc.alignment = std::max<VkDeviceSize>(64, bc.context.min_storage_buffer_offset_alignment());
        return bc;
    }
};

struct BufferAlloc {
    VkDescriptorBufferInfo desc = {};
};

struct RunMetrics {
    double kernel_ms = 0.0;
    double record_ms = 0.0;
    double submit_ms = 0.0;
    double wait_ms = 0.0;
    double queue_delay_ms = 0.0;
};

struct CaseResult {
    std::string label;
    std::string stage;
    std::string dispatch_label;
    size_t estimated_calls = 0;
    RunMetrics metrics;

    auto estimated_kernel_ms() const -> double {
        return metrics.kernel_ms * static_cast<double>(estimated_calls);
    }

    auto estimated_wait_ms() const -> double {
        return metrics.wait_ms * static_cast<double>(estimated_calls);
    }
};

using RunFn = std::function<RunMetrics(BenchContext&, const mruntime::QwenConfig&, const Args&)>;

struct CaseSpec {
    std::string label;
    std::string stage;
    std::string dispatch_label;
    size_t estimated_calls = 0;
    RunFn run;
};

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
        << "  --prompt-len N          Prompt length for contribution model (default: 2016)\n"
        << "  --max-new-tokens N      Decode steps for contribution model (default: 32)\n"
        << "  --max-seq-len N         Max sequence length for cache-shaped ops (default: 2048)\n"
        << "  --max-batch-tokens N    Prefill chunk size for contribution model (default: 64)\n"
        << "  --warmup-iters N        Warmup iterations per op (default: 3)\n"
        << "  --bench-iters N         Timed iterations per op (default: 10)\n"
        << "  -h, --help              Show this help\n";
}

auto parse_args(int argc, char** argv) -> Args {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (mruntime::bench::is_help_flag(arg)) {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--model-dir") {
            args.model_dir = require_value("--model-dir");
        } else if (arg == "--prompt-len") {
            args.prompt_len = static_cast<size_t>(std::stoull(require_value("--prompt-len")));
        } else if (arg == "--max-new-tokens") {
            args.max_new_tokens = static_cast<size_t>(std::stoull(require_value("--max-new-tokens")));
        } else if (arg == "--max-seq-len") {
            args.max_seq_len = static_cast<size_t>(std::stoull(require_value("--max-seq-len")));
        } else if (arg == "--max-batch-tokens") {
            args.max_batch_tokens = static_cast<size_t>(std::stoull(require_value("--max-batch-tokens")));
        } else if (arg == "--warmup-iters") {
            args.warmup_iters = std::stoi(require_value("--warmup-iters"));
        } else if (arg == "--bench-iters") {
            args.bench_iters = std::stoi(require_value("--bench-iters"));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.prompt_len == 0) {
        throw std::runtime_error("--prompt-len must be > 0");
    }
    if (args.max_batch_tokens == 0) {
        throw std::runtime_error("--max-batch-tokens must be > 0");
    }
    if (args.prompt_len + args.max_new_tokens > args.max_seq_len) {
        throw std::runtime_error("prompt_len + max_new_tokens exceeds max_seq_len");
    }
    if (args.warmup_iters < 0 || args.bench_iters <= 0) {
        throw std::runtime_error("--warmup-iters must be >= 0 and --bench-iters must be > 0");
    }

    return args;
}

auto compute_counts(const Args& args) -> WorkloadCounts {
    WorkloadCounts counts;
    counts.full_prefill_chunks = (args.prompt_len > 0) ? ((args.prompt_len - 1) / args.max_batch_tokens) : 0;
    counts.last_chunk_tokens = args.prompt_len - counts.full_prefill_chunks * args.max_batch_tokens;
    counts.decode_calls = (args.max_new_tokens > 0) ? (args.max_new_tokens - 1) : 0;
    counts.logit_steps = 1 + counts.decode_calls;
    return counts;
}

auto checked_u32(size_t value, const char* name) -> uint32_t {
    if (value > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t");
    }
    return static_cast<uint32_t>(value);
}

auto make_arena(const BenchContext& bc, VkDeviceSize capacity) -> VkBufferArena {
    VkBufferArenaCreateInfo info;
    info.capacity_bytes = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.memory_properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    info.default_alignment = bc.alignment;
    return VkBufferArena::Create(bc.context.physical_device(), bc.context.device(), info);
}

auto alloc_fp16(VkBufferArena& arena, size_t elements, uint16_t value = 0x3C00u) -> BufferAlloc {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elements) * sizeof(uint16_t);
    const VkDeviceSize offset = arena.alloc(bytes);
    uint16_t* ptr = arena.host_ptr<uint16_t>(offset);
    if (ptr == nullptr) {
        throw std::runtime_error("alloc_fp16: arena returned null mapped pointer");
    }
    std::fill(ptr, ptr + elements, value);
    return {.desc = arena.descriptor(offset, bytes)};
}

auto alloc_f32(VkBufferArena& arena, size_t elements, float value = 1.0f) -> BufferAlloc {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elements) * sizeof(float);
    const VkDeviceSize offset = arena.alloc(bytes);
    float* ptr = arena.host_ptr<float>(offset);
    if (ptr == nullptr) {
        throw std::runtime_error("alloc_f32: arena returned null mapped pointer");
    }
    std::fill(ptr, ptr + elements, value);
    return {.desc = arena.descriptor(offset, bytes)};
}

auto sum_event_ms(const std::vector<mruntime::TraceEvent>& events, const char* name) -> double {
    int64_t total_us = 0;
    for (const auto& event : events) {
        if (event.type == mruntime::TraceEventType::Complete && std::string_view(event.name) == name) {
            total_us += event.duration_us;
        }
    }
    return static_cast<double>(total_us) / 1000.0;
}

auto sum_queue_delay_ms(const std::vector<mruntime::TraceEvent>& events) -> double {
    int64_t total_us = 0;
    for (const auto& event : events) {
        if (event.type != mruntime::TraceEventType::Complete || std::string_view(event.name) != "vk.kernel") {
            continue;
        }
        for (uint8_t i = 0; i < event.args_count; ++i) {
            if (event.args[i].key != nullptr && std::string_view(event.args[i].key) == "queue_delay_us") {
                total_us += event.args[i].value;
            }
        }
    }
    return static_cast<double>(total_us) / 1000.0;
}

template <typename Fn>
auto benchmark_single_dispatch(BenchContext& bc, int warmup_iters, int bench_iters, Fn&& fn) -> RunMetrics {
    mruntime::TraceCollector& trace = mruntime::TraceCollector::instance();

    bc.runtime.set_timing_enabled(false);
    trace.set_enabled(false);
    for (int i = 0; i < warmup_iters; ++i) {
        auto batch = bc.runtime.begin_batch();
        fn(&batch);
        bc.runtime.finish_batch(&batch, nullptr, 0);
    }

    bc.runtime.set_timing_enabled(true);
    trace.set_enabled(true);

    RunMetrics totals;
    for (int i = 0; i < bench_iters; ++i) {
        trace.reset();
        auto batch = bc.runtime.begin_batch();
        fn(&batch);
        bc.runtime.finish_batch(&batch, nullptr, 0);

        const auto& events = trace.events();
        totals.kernel_ms += sum_event_ms(events, "vk.kernel");
        totals.record_ms += sum_event_ms(events, "vk.batch_record");
        totals.submit_ms += sum_event_ms(events, "vk.batch_submit");
        totals.wait_ms += sum_event_ms(events, "vk.batch_wait");
        totals.queue_delay_ms += sum_queue_delay_ms(events);
    }

    trace.set_enabled(false);
    bc.runtime.set_timing_enabled(false);

    const double inv_iters = 1.0 / static_cast<double>(bench_iters);
    totals.kernel_ms *= inv_iters;
    totals.record_ms *= inv_iters;
    totals.submit_ms *= inv_iters;
    totals.wait_ms *= inv_iters;
    totals.queue_delay_ms *= inv_iters;
    return totals;
}

auto run_rmsnorm_case(
    BenchContext& bc,
    const mruntime::QwenConfig& cfg,
    int warmup_iters,
    int bench_iters,
    uint32_t num_tokens
) -> RunMetrics {
    const uint32_t hidden = checked_u32(cfg.hidden_size, "hidden_size");
    const size_t token_elems = static_cast<size_t>(num_tokens) * cfg.hidden_size;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((2 * token_elems + cfg.hidden_size) * sizeof(uint16_t)) + 3 * bc.alignment
    );
    const auto input = alloc_fp16(arena, token_elems);
    const auto weight = alloc_fp16(arena, cfg.hidden_size);
    const auto output = alloc_fp16(arena, token_elems, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.rmsnorm(input.desc, weight.desc, output.desc, hidden, num_tokens, cfg.rms_norm_eps, batch);
    });
}

auto run_add_case(BenchContext& bc, int warmup_iters, int bench_iters, uint32_t elements) -> RunMetrics {
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>(3ull * elements * sizeof(uint16_t)) + 3 * bc.alignment
    );
    const auto a = alloc_fp16(arena, elements);
    const auto b = alloc_fp16(arena, elements);
    const auto out = alloc_fp16(arena, elements, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.add(a.desc, b.desc, out.desc, elements, batch);
    });
}

auto run_project_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t M,
    uint32_t N,
    uint32_t K
) -> RunMetrics {
    const bool use_gemv = (M == 1u) && ((K & 3u) == 0u);
    const size_t input_elems = static_cast<size_t>(M) * K;
    const size_t weight_elems = static_cast<size_t>(N) * K;
    const size_t output_elems = static_cast<size_t>(M) * N;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((input_elems + weight_elems + output_elems) * sizeof(uint16_t)) + 3 * bc.alignment
    );
    const auto input = alloc_fp16(arena, input_elems);
    const auto weight = alloc_fp16(arena, weight_elems);
    const auto output = alloc_fp16(arena, output_elems, 0);

    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        if (use_gemv) {
            bc.fp16_ops.gemv(input.desc, weight.desc, output.desc, N, K, VK_NULL_HANDLE, batch);
        } else {
            bc.fp16_ops.gemm(input.desc, weight.desc, output.desc, M, N, K, VK_NULL_HANDLE, batch);
        }
    });
}

auto run_silu_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t intermediate_size,
    uint32_t num_tokens
) -> RunMetrics {
    const size_t gate_up_elems = static_cast<size_t>(num_tokens) * intermediate_size * 2u;
    const size_t out_elems = static_cast<size_t>(num_tokens) * intermediate_size;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((gate_up_elems + out_elems) * sizeof(uint16_t)) + 2 * bc.alignment
    );
    const auto gate_up = alloc_fp16(arena, gate_up_elems);
    const auto out = alloc_fp16(arena, out_elems, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.silu_mul_interleaved(gate_up.desc, out.desc, intermediate_size, num_tokens, batch);
    });
}

auto run_qkv_split_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t num_tokens,
    uint32_t q_dim,
    uint32_t kv_dim
) -> RunMetrics {
    const uint32_t qkv_dim = q_dim + 2u * kv_dim;
    const size_t qkv_elems = static_cast<size_t>(num_tokens) * qkv_dim;
    const size_t q_elems = static_cast<size_t>(num_tokens) * q_dim;
    const size_t kv_elems = static_cast<size_t>(num_tokens) * kv_dim;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((qkv_elems + qkv_dim + q_elems + 2ull * kv_elems) * sizeof(uint16_t)) + 5 * bc.alignment
    );
    const auto qkv = alloc_fp16(arena, qkv_elems);
    const auto bias = alloc_fp16(arena, qkv_dim);
    const auto q_out = alloc_fp16(arena, q_elems, 0);
    const auto k_out = alloc_fp16(arena, kv_elems, 0);
    const auto v_out = alloc_fp16(arena, kv_elems, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.qkv_bias_split(qkv.desc, bias.desc, q_out.desc, k_out.desc, v_out.desc, num_tokens, q_dim, kv_dim, true, batch);
    });
}

auto run_rope_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) -> RunMetrics {
    const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_q_heads * head_dim;
    const size_t k_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
    const size_t rope_elems = static_cast<size_t>(max_seq_len) * (head_dim / 2u) * 2u;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((q_elems + k_elems) * sizeof(uint16_t) + rope_elems * sizeof(float)) + 3 * bc.alignment
    );
    const auto q = alloc_fp16(arena, q_elems);
    const auto k = alloc_fp16(arena, k_elems);
    const auto rope = alloc_f32(arena, rope_elems);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.rope(q.desc, k.desc, rope.desc, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, 0, batch);
    });
}

auto run_qkv_decode_postprocess_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t q_dim,
    uint32_t kv_dim,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len,
    uint32_t position_offset
) -> RunMetrics {
    const uint32_t qkv_dim = q_dim + 2u * kv_dim;
    const size_t qkv_elems = qkv_dim;
    const size_t rope_elems = static_cast<size_t>(max_seq_len) * (head_dim / 2u) * 2u;
    const size_t q_out_elems = q_dim;
    const size_t cache_elems = static_cast<size_t>(num_kv_heads) * max_seq_len * head_dim;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>(
            (qkv_elems + qkv_dim + q_out_elems + 2ull * cache_elems) * sizeof(uint16_t) +
            rope_elems * sizeof(float)
        ) + 6 * bc.alignment
    );
    const auto qkv = alloc_fp16(arena, qkv_elems);
    const auto bias = alloc_fp16(arena, qkv_dim);
    const auto rope = alloc_f32(arena, rope_elems);
    const auto q_out = alloc_fp16(arena, q_out_elems, 0);
    const auto k_cache = alloc_fp16(arena, cache_elems, 0);
    const auto v_cache = alloc_fp16(arena, cache_elems, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.qkv_bias_rope_cache_decode(
            qkv.desc,
            bias.desc,
            rope.desc,
            q_out.desc,
            k_cache.desc,
            v_cache.desc,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            position_offset,
            true,
            batch
        );
    });
}

auto run_transpose_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t B,
    uint32_t S,
    uint32_t H,
    uint32_t D
) -> RunMetrics {
    const size_t elements = static_cast<size_t>(B) * S * H * D;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>(2ull * elements * sizeof(uint16_t)) + 2 * bc.alignment
    );
    const auto input = alloc_fp16(arena, elements);
    const auto output = alloc_fp16(arena, elements, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.transpose_bshd_to_bhsd(input.desc, output.desc, B, S, H, D, batch);
    });
}

auto run_kv_cache_copy_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len
) -> RunMetrics {
    const size_t kv_in_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
    const size_t cache_elems = static_cast<size_t>(num_kv_heads) * max_seq_len * head_dim;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((2ull * kv_in_elems + 2ull * cache_elems) * sizeof(uint16_t)) + 4 * bc.alignment
    );
    const auto k_in = alloc_fp16(arena, kv_in_elems);
    const auto v_in = alloc_fp16(arena, kv_in_elems);
    const auto k_cache = alloc_fp16(arena, cache_elems, 0);
    const auto v_cache = alloc_fp16(arena, cache_elems, 0);
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.kv_cache_copy(
            k_in.desc,
            v_in.desc,
            k_cache.desc,
            v_cache.desc,
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim,
            max_seq_len,
            0,
            batch
        );
    });
}

auto run_attention_prefill_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t q_len,
    uint32_t kv_len,
    uint32_t kv_stride,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim
) -> RunMetrics {
    const size_t q_elems = static_cast<size_t>(num_q_heads) * q_len * head_dim;
    const size_t kv_elems = static_cast<size_t>(num_kv_heads) * kv_stride * head_dim;
    const size_t out_elems = q_elems;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((q_elems + 2ull * kv_elems + out_elems) * sizeof(uint16_t)) + 4 * bc.alignment
    );
    const auto q = alloc_fp16(arena, q_elems);
    const auto k = alloc_fp16(arena, kv_elems);
    const auto v = alloc_fp16(arena, kv_elems);
    const auto out = alloc_fp16(arena, out_elems, 0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.attention_prefill_gqa(
            q.desc,
            k.desc,
            v.desc,
            out.desc,
            num_q_heads,
            num_kv_heads,
            q_len,
            kv_len,
            kv_stride,
            head_dim,
            scale,
            batch
        );
    });
}

auto run_attention_decode_case(
    BenchContext& bc,
    int warmup_iters,
    int bench_iters,
    uint32_t kv_len,
    uint32_t kv_stride,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim
) -> RunMetrics {
    const size_t q_elems = static_cast<size_t>(num_q_heads) * head_dim;
    const size_t kv_elems = static_cast<size_t>(num_kv_heads) * kv_stride * head_dim;
    const size_t out_elems = q_elems;
    const size_t partial_out_elems =
        static_cast<size_t>(mruntime::vulkan::VkFp16Ops::kAttentionDecodeMaxSplitK) * out_elems;
    const size_t partial_stats_elems =
        static_cast<size_t>(mruntime::vulkan::VkFp16Ops::kAttentionDecodeMaxSplitK) * num_q_heads * 2u;
    auto arena = make_arena(
        bc,
        static_cast<VkDeviceSize>((q_elems + 2ull * kv_elems + out_elems + partial_out_elems) * sizeof(uint16_t)) +
            static_cast<VkDeviceSize>(partial_stats_elems * sizeof(float)) +
            6 * bc.alignment
    );
    const auto q = alloc_fp16(arena, q_elems);
    const auto k = alloc_fp16(arena, kv_elems);
    const auto v = alloc_fp16(arena, kv_elems);
    const auto partial_out = alloc_fp16(arena, partial_out_elems);
    const auto partial_stats = alloc_f32(arena, partial_stats_elems);
    const auto out = alloc_fp16(arena, out_elems, 0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    return benchmark_single_dispatch(bc, warmup_iters, bench_iters, [&](VkDispatchBatch* batch) {
        bc.fp16_ops.attention_decode_gqa(
            q.desc,
            k.desc,
            v.desc,
            partial_out.desc,
            partial_stats.desc,
            out.desc,
            num_q_heads,
            num_kv_heads,
            kv_len,
            kv_stride,
            head_dim,
            scale,
            batch
        );
    });
}

auto make_cases(const mruntime::QwenConfig& cfg, const Args& args, const WorkloadCounts& counts) -> std::vector<CaseSpec> {
    std::vector<CaseSpec> cases;
    const uint32_t hidden = checked_u32(cfg.hidden_size, "hidden_size");
    const uint32_t intermediate = checked_u32(cfg.intermediate_size, "intermediate_size");
    const uint32_t layers = checked_u32(cfg.num_layers, "num_layers");
    const uint32_t num_q_heads = checked_u32(cfg.num_attention_heads, "num_attention_heads");
    const uint32_t num_kv_heads = checked_u32(cfg.num_kv_heads, "num_kv_heads");
    const uint32_t head_dim = checked_u32(cfg.head_dim(), "head_dim");
    const uint32_t q_dim = num_q_heads * head_dim;
    const uint32_t kv_dim = num_kv_heads * head_dim;
    const uint32_t full_tokens = checked_u32(args.max_batch_tokens, "max_batch_tokens");
    const uint32_t last_tokens = checked_u32(counts.last_chunk_tokens, "last_chunk_tokens");
    const uint32_t max_seq_len = checked_u32(args.max_seq_len, "max_seq_len");
    const uint32_t prompt_len = checked_u32(args.prompt_len, "prompt_len");
    const uint32_t decode_kv_len = checked_u32(args.prompt_len + 1, "decode_kv_len");

    auto add_case = [&](CaseSpec spec) {
        if (spec.estimated_calls != 0 || spec.stage == "info") {
            cases.push_back(std::move(spec));
        }
    };

    add_case({
        "rmsnorm",
        "prefill_chunk",
        "64x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers) * 2,
        [full_tokens](BenchContext& bc, const mruntime::QwenConfig& model_cfg, const Args& bench_args) {
            return run_rmsnorm_case(bc, model_cfg, bench_args.warmup_iters, bench_args.bench_iters, full_tokens);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "rmsnorm",
            "prefill_tail",
            std::to_string(counts.last_chunk_tokens) + "x1x1",
            static_cast<size_t>(layers) * 2,
            [last_tokens](BenchContext& bc, const mruntime::QwenConfig& model_cfg, const Args& bench_args) {
                return run_rmsnorm_case(bc, model_cfg, bench_args.warmup_iters, bench_args.bench_iters, last_tokens);
            },
        });
    }
    add_case({
        "rmsnorm",
        "decode1",
        "1x1x1",
        counts.decode_calls * static_cast<size_t>(layers) * 2,
        [](BenchContext& bc, const mruntime::QwenConfig& model_cfg, const Args& bench_args) {
            return run_rmsnorm_case(bc, model_cfg, bench_args.warmup_iters, bench_args.bench_iters, 1);
        },
    });
    add_case({
        "final_norm",
        "logits",
        "1x1x1",
        counts.logit_steps,
        [](BenchContext& bc, const mruntime::QwenConfig& model_cfg, const Args& bench_args) {
            return run_rmsnorm_case(bc, model_cfg, bench_args.warmup_iters, bench_args.bench_iters, 1);
        },
    });
    add_case({
        "residual_add",
        "prefill_chunk",
        "28x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers) * 2,
        [hidden, full_tokens](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_add_case(bc, bench_args.warmup_iters, bench_args.bench_iters, hidden * full_tokens);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "residual_add",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers) * 2,
            [hidden, last_tokens](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_add_case(bc, bench_args.warmup_iters, bench_args.bench_iters, hidden * last_tokens);
            },
        });
    }
    add_case({
        "residual_add",
        "decode1",
        "1x1x1",
        counts.decode_calls * static_cast<size_t>(layers) * 2,
        [hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_add_case(bc, bench_args.warmup_iters, bench_args.bench_iters, hidden);
        },
    });
    add_case({
        "qkv_proj",
        "prefill_chunk",
        "18x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, q_dim, kv_dim, hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, full_tokens, q_dim + 2u * kv_dim, hidden);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "qkv_proj",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, q_dim, kv_dim, hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, last_tokens, q_dim + 2u * kv_dim, hidden);
            },
        });
    }
    add_case({
        "qkv_proj",
        "decode1",
        "288x1x1",
        counts.decode_calls * static_cast<size_t>(layers),
        [q_dim, kv_dim, hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, 1, q_dim + 2u * kv_dim, hidden);
        },
    });
    add_case({
        "qkv_split",
        "prefill_chunk",
        "36x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, q_dim, kv_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_qkv_split_case(bc, bench_args.warmup_iters, bench_args.bench_iters, full_tokens, q_dim, kv_dim);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "qkv_split",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, q_dim, kv_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_qkv_split_case(bc, bench_args.warmup_iters, bench_args.bench_iters, last_tokens, q_dim, kv_dim);
            },
        });
    }
    add_case({
        "qkv_decode_postprocess",
        "decode1",
        "3x1x1",
        counts.decode_calls * static_cast<size_t>(layers),
        [q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, max_seq_len, prompt_len](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_qkv_decode_postprocess_case(
                bc,
                bench_args.warmup_iters,
                bench_args.bench_iters,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                prompt_len
            );
        },
    });
    add_case({
        "rope",
        "prefill_chunk",
        "128x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, num_q_heads, num_kv_heads, head_dim, max_seq_len](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_rope_case(
                bc,
                bench_args.warmup_iters,
                bench_args.bench_iters,
                1,
                full_tokens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_seq_len
            );
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "rope",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, num_q_heads, num_kv_heads, head_dim, max_seq_len](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_rope_case(
                    bc,
                    bench_args.warmup_iters,
                    bench_args.bench_iters,
                    1,
                    last_tokens,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    max_seq_len
                );
            },
        });
    }
    add_case({
        "q_transpose",
        "prefill_chunk",
        "4x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, num_q_heads, head_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_transpose_case(bc, bench_args.warmup_iters, bench_args.bench_iters, 1, full_tokens, num_q_heads, head_dim);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "q_transpose",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, num_q_heads, head_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_transpose_case(bc, bench_args.warmup_iters, bench_args.bench_iters, 1, last_tokens, num_q_heads, head_dim);
            },
        });
    }
    add_case({
        "kv_cache_copy",
        "prefill_chunk",
        "1x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, num_kv_heads, head_dim, max_seq_len](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_kv_cache_copy_case(
                bc,
                bench_args.warmup_iters,
                bench_args.bench_iters,
                1,
                full_tokens,
                num_kv_heads,
                head_dim,
                max_seq_len
            );
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "kv_cache_copy",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, num_kv_heads, head_dim, max_seq_len](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_kv_cache_copy_case(
                    bc,
                    bench_args.warmup_iters,
                    bench_args.bench_iters,
                    1,
                    last_tokens,
                    num_kv_heads,
                    head_dim,
                    max_seq_len
                );
            },
        });
    }
    add_case({
        "attention_prefill",
        "info",
        "14x64x1",
        0,
        [full_tokens, num_q_heads, num_kv_heads, head_dim, max_seq_len, prompt_len](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_attention_prefill_case(
                bc,
                bench_args.warmup_iters,
                bench_args.bench_iters,
                full_tokens,
                prompt_len,
                max_seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim
            );
        },
    });
    add_case({
        "o_proj",
        "prefill_chunk",
        "14x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, hidden, q_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, full_tokens, hidden, q_dim);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "o_proj",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, hidden, q_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, last_tokens, hidden, q_dim);
            },
        });
    }
    add_case({
        "o_proj",
        "decode1",
        "224x1x1",
        counts.decode_calls * static_cast<size_t>(layers),
        [hidden, q_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, 1, hidden, q_dim);
        },
    });
    add_case({
        "gate_up_proj",
        "prefill_chunk",
        "152x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, intermediate, hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, full_tokens, intermediate * 2u, hidden);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "gate_up_proj",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, intermediate, hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, last_tokens, intermediate * 2u, hidden);
            },
        });
    }
    add_case({
        "gate_up_proj",
        "decode1",
        "2432x1x1",
        counts.decode_calls * static_cast<size_t>(layers),
        [intermediate, hidden](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, 1, intermediate * 2u, hidden);
        },
    });
    add_case({
        "silu_mul",
        "prefill_chunk",
        "152x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [intermediate, full_tokens](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_silu_case(bc, bench_args.warmup_iters, bench_args.bench_iters, intermediate, full_tokens);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "silu_mul",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [intermediate, last_tokens](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_silu_case(bc, bench_args.warmup_iters, bench_args.bench_iters, intermediate, last_tokens);
            },
        });
    }
    add_case({
        "silu_mul",
        "decode1",
        "3x1x1",
        counts.decode_calls * static_cast<size_t>(layers),
        [intermediate](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_silu_case(bc, bench_args.warmup_iters, bench_args.bench_iters, intermediate, 1);
        },
    });
    add_case({
        "down_proj",
        "prefill_chunk",
        "14x1x1",
        counts.full_prefill_chunks * static_cast<size_t>(layers),
        [full_tokens, hidden, intermediate](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, full_tokens, hidden, intermediate);
        },
    });
    if (counts.last_chunk_tokens != args.max_batch_tokens || counts.full_prefill_chunks == 0) {
        add_case({
            "down_proj",
            "prefill_tail",
            "tail",
            static_cast<size_t>(layers),
            [last_tokens, hidden, intermediate](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
                return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, last_tokens, hidden, intermediate);
            },
        });
    }
    add_case({
        "down_proj",
        "decode1",
        "224x1x1",
        counts.decode_calls * static_cast<size_t>(layers),
        [hidden, intermediate](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_project_case(bc, bench_args.warmup_iters, bench_args.bench_iters, 1, hidden, intermediate);
        },
    });
    add_case({
        "attention_decode",
        "info",
        "14x1x1",
        0,
        [decode_kv_len, max_seq_len, num_q_heads, num_kv_heads, head_dim](BenchContext& bc, const mruntime::QwenConfig&, const Args& bench_args) {
            return run_attention_decode_case(
                bc,
                bench_args.warmup_iters,
                bench_args.bench_iters,
                decode_kv_len,
                max_seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim
            );
        },
    });
    add_case({
        "lm_head",
        "logits",
        "37984x1x1",
        counts.logit_steps,
        [hidden](BenchContext& bc, const mruntime::QwenConfig& model_cfg, const Args& bench_args) {
            return run_project_case(
                bc,
                bench_args.warmup_iters,
                bench_args.bench_iters,
                1,
                checked_u32(model_cfg.vocab_size, "vocab_size"),
                hidden
            );
        },
    });

    return cases;
}

auto load_config(const std::string& model_dir) -> mruntime::QwenConfig {
    const std::string config_json =
        mruntime::bench::read_text_file(mruntime::bench::join_path(model_dir, "config.json"));
    return mruntime::QwenConfig::from_json(config_json);
}

auto stage_sort_key(const std::string& stage) -> int {
    if (stage == "prefill_chunk") return 0;
    if (stage == "prefill_tail") return 1;
    if (stage == "decode1") return 2;
    if (stage == "logits") return 3;
    return 4;
}

auto print_results(const std::vector<CaseResult>& results) -> void {
    constexpr int kStageWidth = 18;
    constexpr int kOpWidth = 28;
    constexpr int kDispatchWidth = 14;
    constexpr int kCallsWidth = 12;
    constexpr int kMetricWidth = 12;
    constexpr int kEstimateWidth = 14;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Vulkan Qwen2 Op Bench ===\n";
    std::cout << std::setw(kStageWidth) << "stage"
              << std::setw(kOpWidth) << "op"
              << std::setw(kDispatchWidth) << "dispatch"
              << std::setw(kCallsWidth) << "calls"
              << std::setw(kMetricWidth) << "gpu_ms"
              << std::setw(kMetricWidth) << "wait_ms"
              << std::setw(kMetricWidth) << "queue_ms"
              << std::setw(kEstimateWidth) << "est_gpu_ms"
              << std::setw(kEstimateWidth) << "est_wait_ms"
              << "\n";

    for (const auto& result : results) {
        std::cout << std::setw(kStageWidth) << result.stage
                  << std::setw(kOpWidth) << result.label
                  << std::setw(kDispatchWidth) << result.dispatch_label
                  << std::setw(kCallsWidth) << result.estimated_calls
                  << std::setw(kMetricWidth) << result.metrics.kernel_ms
                  << std::setw(kMetricWidth) << result.metrics.wait_ms
                  << std::setw(kMetricWidth) << result.metrics.queue_delay_ms
                  << std::setw(kEstimateWidth) << result.estimated_kernel_ms()
                  << std::setw(kEstimateWidth) << result.estimated_wait_ms()
                  << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const WorkloadCounts counts = compute_counts(args);
        const std::string model_dir = mruntime::bench::resolve_model_dir(args.model_dir);
        const mruntime::QwenConfig cfg = load_config(model_dir);

        std::cout << "Model dir: " << model_dir << "\n";
        std::cout << "Contribution model: prompt_len=" << args.prompt_len
                  << ", max_new_tokens=" << args.max_new_tokens
                  << ", max_batch_tokens=" << args.max_batch_tokens
                  << ", max_seq_len=" << args.max_seq_len << "\n";
        std::cout << "Workload counts: full_prefill_chunks=" << counts.full_prefill_chunks
                  << ", last_chunk_tokens=" << counts.last_chunk_tokens
                  << ", decode_calls=" << counts.decode_calls
                  << ", logit_steps=" << counts.logit_steps << "\n";

        BenchContext bc = BenchContext::Create();
        std::vector<CaseSpec> cases = make_cases(cfg, args, counts);
        std::vector<CaseResult> results;
        results.reserve(cases.size());

        for (const auto& spec : cases) {
            CaseResult result;
            result.label = spec.label;
            result.stage = spec.stage;
            result.dispatch_label = spec.dispatch_label;
            result.estimated_calls = spec.estimated_calls;
            result.metrics = spec.run(bc, cfg, args);
            results.push_back(std::move(result));
        }

        std::sort(results.begin(), results.end(), [](const CaseResult& a, const CaseResult& b) {
            if (a.estimated_kernel_ms() != b.estimated_kernel_ms()) {
                return a.estimated_kernel_ms() > b.estimated_kernel_ms();
            }
            if (stage_sort_key(a.stage) != stage_sort_key(b.stage)) {
                return stage_sort_key(a.stage) < stage_sort_key(b.stage);
            }
            return a.label < b.label;
        });

        print_results(results);
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_qwen2_op_bench SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_qwen2_op_bench FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_qwen2_op_bench FAILED: " << error.what() << "\n";
        return 1;
    }
}
