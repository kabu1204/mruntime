#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mruntime/arena.h"
#include "mruntime/qwen2_forward_vk.h"
#include "mruntime/qwen2_generate.h"
#include "mruntime/qwen2_weights.h"
#include "mruntime/safetensors.h"

#include "e2e_bench_common.h"
#include "vk_helpers.h"

namespace {

using Clock = mruntime::bench::E2EClock;

struct Args : mruntime::bench::CommonE2EArgs {
    std::vector<size_t> prompt_lens = mruntime::bench::default_prompt_len_sweep();
};

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
        << "  --prompt-lens LIST      Comma-separated prompt lengths (default: "
        << mruntime::bench::format_size_t_list(mruntime::bench::default_prompt_len_sweep()) << ")\n"
        << "  --max-new-tokens N      Number of tokens to decode (default: 32)\n"
        << "  --max-seq-len N         KV cache max sequence length (default: 2048)\n"
        << "  --max-batch-tokens N    Max tokens per forward call (default: 64)\n"
        << "  --threads N             Number of threads (default: auto-detect)\n"
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
        } else if (arg == "--max-new-tokens") {
            args.max_new_tokens = static_cast<size_t>(std::stoull(require_value("--max-new-tokens")));
        } else if (arg == "--max-seq-len") {
            args.max_seq_len = static_cast<size_t>(std::stoull(require_value("--max-seq-len")));
        } else if (arg == "--max-batch-tokens") {
            args.max_batch_tokens = static_cast<size_t>(std::stoull(require_value("--max-batch-tokens")));
        } else if (arg == "--threads") {
            args.num_threads = static_cast<size_t>(std::stoull(require_value("--threads")));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return args;
}

auto make_prompt_tokens(size_t prompt_len) -> std::vector<int32_t> {
    std::vector<int32_t> prompt_tokens(prompt_len);
    for (size_t i = 0; i < prompt_len; ++i) {
        prompt_tokens[i] = static_cast<int32_t>((i % 1000) + 1);
    }
    return prompt_tokens;
}

auto run_warmup(
    mruntime::Qwen2VkState& state,
    const mruntime::QwenConfig& cfg,
    const std::vector<int32_t>& prompt_tokens,
    const mruntime::Qwen2GenerateConfig& gen_cfg,
    mruntime::PThreadPool& pool
) -> void {
    uint64_t rng_state = gen_cfg.seed;
    const uint16_t* logits = mruntime::qwen2_prefill_vk(state, prompt_tokens.data(), prompt_tokens.size(), &pool);
    if (gen_cfg.max_new_tokens == 0) {
        return;
    }

    int32_t token = mruntime::qwen2_sample(logits, cfg.vocab_size, gen_cfg, &rng_state);
    for (size_t i = 1; i < std::min<size_t>(gen_cfg.max_new_tokens, 4); ++i) {
        logits = mruntime::qwen2_decode_vk(state, token, &pool);
        token = mruntime::qwen2_sample(logits, cfg.vocab_size, gen_cfg, &rng_state);
    }
}

auto run_measurement(
    mruntime::Qwen2VkState& state,
    const mruntime::QwenConfig& cfg,
    const std::vector<int32_t>& prompt_tokens,
    const mruntime::Qwen2GenerateConfig& gen_cfg,
    mruntime::PThreadPool& pool
) -> mruntime::bench::E2EMetrics {
    uint64_t rng_state = gen_cfg.seed;
    size_t generated_tokens = 0;

    const auto t0 = Clock::now();
    const auto t_prefill_start = t0;
    const uint16_t* logits = mruntime::qwen2_prefill_vk(state, prompt_tokens.data(), prompt_tokens.size(), &pool);
    const auto t_prefill_end = Clock::now();

    auto t_first_token = t_prefill_end;
    auto t_decode_start = t_prefill_end;
    auto t_decode_end = t_prefill_end;

    if (gen_cfg.max_new_tokens > 0) {
        int32_t next_token = mruntime::qwen2_sample(logits, cfg.vocab_size, gen_cfg, &rng_state);
        t_first_token = Clock::now();
        generated_tokens = 1;
        t_decode_start = Clock::now();

        for (; generated_tokens < gen_cfg.max_new_tokens; ++generated_tokens) {
            if (gen_cfg.eos_token_id >= 0 && next_token == gen_cfg.eos_token_id) {
                break;
            }
            logits = mruntime::qwen2_decode_vk(state, next_token, &pool);
            next_token = mruntime::qwen2_sample(logits, cfg.vocab_size, gen_cfg, &rng_state);
        }

        t_decode_end = Clock::now();
    }

    return mruntime::bench::make_e2e_metrics(
        prompt_tokens.size(),
        gen_cfg.max_new_tokens,
        pool.threads_count(),
        generated_tokens,
        t0,
        t_first_token,
        t_prefill_start,
        t_prefill_end,
        t_decode_start,
        t_decode_end
    );
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        mruntime::bench::validate_prompt_len_sweep(
            args.prompt_lens,
            args.max_new_tokens,
            args.max_seq_len,
            args.max_batch_tokens
        );

        const std::string model_dir = mruntime::bench::resolve_model_dir(args.model_dir);
        const std::string config_json =
            mruntime::bench::read_text_file(mruntime::bench::join_path(model_dir, "config.json"));
        const mruntime::QwenConfig cfg = mruntime::QwenConfig::from_json(config_json);

        mruntime::Qwen2GenerateConfig gen_cfg;
        gen_cfg.max_new_tokens = args.max_new_tokens;
        gen_cfg.greedy = args.greedy;
        gen_cfg.temperature = args.temperature;
        gen_cfg.top_k = args.top_k;
        gen_cfg.top_p = args.top_p;
        gen_cfg.eos_token_id = args.eos_token_id;
        gen_cfg.seed = args.seed;

        auto st = mruntime::SafeTensorsFile::open(mruntime::bench::join_path(model_dir, "model.safetensors"));
        if (!st) {
            throw std::runtime_error("Failed to open model.safetensors");
        }

        const mruntime::Qwen2MemorySizes sizes = mruntime::qwen2_memory_sizes(
            cfg,
            args.max_seq_len,
            args.max_batch_tokens,
            st->has_tensor("lm_head.weight")
        );

        mruntime::Arena weights_arena = mruntime::create_arena(sizes.weights_bytes);
        mruntime::Qwen2Weights weights =
            mruntime::qwen2_load_weights(cfg, *st, weights_arena, /*pack_for_kai=*/false);
        mruntime::PThreadPool pool = mruntime::PThreadPool::Create(args.num_threads);
        auto state = mruntime::qwen2_vk_create(cfg, weights, args.max_seq_len, args.max_batch_tokens);

        mruntime::bench::print_e2e_sweep_header(
            "vulkan",
            model_dir,
            args.prompt_lens,
            args.max_new_tokens,
            args.max_seq_len,
            args.max_batch_tokens,
            pool.threads_count()
        );

        std::vector<mruntime::bench::E2EMetrics> metrics;
        metrics.reserve(args.prompt_lens.size());
        for (size_t prompt_len : args.prompt_lens) {
            const std::vector<int32_t> prompt_tokens = make_prompt_tokens(prompt_len);

            mruntime::qwen2_vk_reset_kv_cache(*state);
            run_warmup(*state, cfg, prompt_tokens, gen_cfg, pool);

            mruntime::qwen2_vk_reset_kv_cache(*state);
            metrics.push_back(run_measurement(*state, cfg, prompt_tokens, gen_cfg, pool));
        }

        mruntime::bench::print_e2e_sweep_metrics(metrics);
        mruntime::destroy_arena(weights_arena);
        return 0;
    } catch (const mruntime::vulkan::VulkanError& error) {
        if (error.result() == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::cout << "vulkan_e2e_context_sweep_bench SKIPPED: Vulkan not supported on this machine\n";
            return 77;
        }
        std::cerr << "vulkan_e2e_context_sweep_bench FAILED: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "vulkan_e2e_context_sweep_bench FAILED: " << error.what() << "\n";
        return 1;
    }
}
