#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
    std::vector<size_t> prompt_lens = {512, 2016};
    std::vector<size_t> chunk_lens = {64, 128, 256, 512};
    size_t max_seq_len = 2048;
    int warmup_iters = 3;
    int bench_iters = 10;
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
    size_t prompt_len = 0;
    size_t chunk_len = 0;
    RunMetrics metrics = {};
};

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
        << "  --prompt-lens A,B,...   Prompt lengths to benchmark (default: 512,2016)\n"
        << "  --chunk-lens A,B,...    Prefill chunk sizes to benchmark (default: 64,128,256,512)\n"
        << "  --max-seq-len N         KV cache max sequence length (default: 2048)\n"
        << "  --warmup-iters N        Warmup iterations per case (default: 3)\n"
        << "  --bench-iters N         Timed iterations per case (default: 10)\n"
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
        } else if (arg == "--prompt-lens") {
            args.prompt_lens = mruntime::bench::parse_size_t_list(require_value("--prompt-lens"), "--prompt-lens");
        } else if (arg == "--chunk-lens") {
            args.chunk_lens = mruntime::bench::parse_size_t_list(require_value("--chunk-lens"), "--chunk-lens");
        } else if (arg == "--max-seq-len") {
            args.max_seq_len = static_cast<size_t>(std::stoull(require_value("--max-seq-len")));
        } else if (arg == "--warmup-iters") {
            args.warmup_iters = std::stoi(require_value("--warmup-iters"));
        } else if (arg == "--bench-iters") {
            args.bench_iters = std::stoi(require_value("--bench-iters"));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.prompt_lens.empty()) {
        throw std::runtime_error("--prompt-lens must not be empty");
    }
    if (args.chunk_lens.empty()) {
        throw std::runtime_error("--chunk-lens must not be empty");
    }
    if (args.max_seq_len == 0) {
        throw std::runtime_error("--max-seq-len must be > 0");
    }
    if (args.warmup_iters < 0 || args.bench_iters <= 0) {
        throw std::runtime_error("--warmup-iters must be >= 0 and --bench-iters must be > 0");
    }

    for (size_t prompt_len : args.prompt_lens) {
        if (prompt_len == 0) {
            throw std::runtime_error("--prompt-lens entries must be > 0");
        }
        if (prompt_len > args.max_seq_len) {
            throw std::runtime_error("prompt_len exceeds max_seq_len");
        }
    }
    for (size_t chunk_len : args.chunk_lens) {
        if (chunk_len == 0) {
            throw std::runtime_error("--chunk-lens entries must be > 0");
        }
    }

    return args;
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

auto load_config(const std::string& model_dir) -> mruntime::QwenConfig {
    const std::string config_json =
        mruntime::bench::read_text_file(mruntime::bench::join_path(model_dir, "config.json"));
    return mruntime::QwenConfig::from_json(config_json);
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

auto run_cases(const mruntime::QwenConfig& cfg, const Args& args) -> std::vector<CaseResult> {
    std::vector<CaseResult> results;
    const uint32_t num_q_heads = checked_u32(cfg.num_attention_heads, "num_attention_heads");
    const uint32_t num_kv_heads = checked_u32(cfg.num_kv_heads, "num_kv_heads");
    const uint32_t head_dim = checked_u32(cfg.head_dim(), "head_dim");
    const uint32_t kv_stride = checked_u32(args.max_seq_len, "max_seq_len");

    BenchContext bc = BenchContext::Create();

    for (size_t prompt_len : args.prompt_lens) {
        for (size_t chunk_len : args.chunk_lens) {
            if (chunk_len > prompt_len) {
                continue;
            }

            CaseResult result;
            result.prompt_len = prompt_len;
            result.chunk_len = chunk_len;
            result.metrics = run_attention_prefill_case(
                bc,
                args.warmup_iters,
                args.bench_iters,
                checked_u32(chunk_len, "chunk_len"),
                checked_u32(prompt_len, "prompt_len"),
                kv_stride,
                num_q_heads,
                num_kv_heads,
                head_dim
            );
            results.push_back(result);
        }
    }

    return results;
}

auto print_results(const std::vector<CaseResult>& results) -> void {
    constexpr int kPromptWidth = 10;
    constexpr int kChunkWidth = 10;
    constexpr int kMetricWidth = 12;

    std::cout << "\nPrefill attention kernel benchmark\n";
    std::cout << std::left
              << std::setw(kPromptWidth) << "prompt_len"
              << std::setw(kChunkWidth) << "chunk_len"
              << std::setw(kMetricWidth) << "gpu_ms"
              << std::setw(kMetricWidth) << "record_ms"
              << std::setw(kMetricWidth) << "submit_ms"
              << std::setw(kMetricWidth) << "wait_ms"
              << std::setw(kMetricWidth) << "queue_ms"
              << "\n";

    for (const CaseResult& result : results) {
        std::cout << std::left
                  << std::setw(kPromptWidth) << result.prompt_len
                  << std::setw(kChunkWidth) << result.chunk_len
                  << std::setw(kMetricWidth) << std::fixed << std::setprecision(3) << result.metrics.kernel_ms
                  << std::setw(kMetricWidth) << std::fixed << std::setprecision(3) << result.metrics.record_ms
                  << std::setw(kMetricWidth) << std::fixed << std::setprecision(3) << result.metrics.submit_ms
                  << std::setw(kMetricWidth) << std::fixed << std::setprecision(3) << result.metrics.wait_ms
                  << std::setw(kMetricWidth) << std::fixed << std::setprecision(3) << result.metrics.queue_delay_ms
                  << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::string model_dir = mruntime::bench::resolve_model_dir(args.model_dir);
        const mruntime::QwenConfig cfg = load_config(model_dir);

        std::cout << "Model dir: " << model_dir << "\n";
        std::cout << "Prompt lens: " << mruntime::bench::format_size_t_list(args.prompt_lens) << "\n";
        std::cout << "Chunk lens: " << mruntime::bench::format_size_t_list(args.chunk_lens) << "\n";
        std::cout << "Max seq len: " << args.max_seq_len << "\n";
        std::cout << "Warmup iters: " << args.warmup_iters << "\n";
        std::cout << "Bench iters: " << args.bench_iters << "\n";

        const std::vector<CaseResult> results = run_cases(cfg, args);
        print_results(results);
        return 0;
    } catch (const VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_prefill_attention_bench SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_prefill_attention_bench FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_prefill_attention_bench FAILED: " << error.what() << "\n";
        return 1;
    }
}
