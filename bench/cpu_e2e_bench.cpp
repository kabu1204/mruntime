#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mruntime/arena.h"
#include "mruntime/qwen2_forward.h"
#include "mruntime/qwen2_generate.h"
#include "mruntime/qwen2_weights.h"
#include "mruntime/safetensors.h"
#include "mruntime/trace.h"

#include "e2e_bench_common.h"

namespace {

using Clock = mruntime::bench::E2EClock;

struct Args : mruntime::bench::CommonE2EArgs {
};

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n";
    mruntime::bench::print_common_e2e_usage(std::cout);
    std::cout << "  -h, --help              Show this help\n";
}

auto parse_args(int argc, char** argv) -> Args {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (mruntime::bench::is_help_flag(a)) {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (!mruntime::bench::parse_common_e2e_arg(a, require_value, args)) {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::string model_dir = mruntime::bench::resolve_model_dir(args.model_dir);

        std::cout << "Model dir: " << model_dir << "\n";

        const std::string config_json = mruntime::bench::read_text_file(mruntime::bench::join_path(model_dir, "config.json"));
        const mruntime::QwenConfig cfg = mruntime::QwenConfig::from_json(config_json);

        mruntime::bench::validate_common_e2e_args(args);

        // Allocate a synthetic prompt.
        std::vector<int32_t> prompt_tokens(args.prompt_len);
        for (size_t i = 0; i < args.prompt_len; ++i) {
            prompt_tokens[i] = static_cast<int32_t>((i % 1000) + 1);
        }

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

        mruntime::Qwen2Arenas arenas = mruntime::create_qwen2_arenas(
            sizes.weights_bytes + sizes.packed_weights_bytes,
            sizes.kv_cache_bytes,
            sizes.scratch_bytes
        );

        mruntime::Qwen2Weights weights =
            mruntime::qwen2_load_weights(cfg, *st, arenas.weights, /*pack_for_kai=*/true);

        mruntime::Qwen2KVCache kv_cache = mruntime::qwen2_init_kv_cache(cfg, arenas.kv_cache, args.max_seq_len);
        mruntime::Qwen2Scratch scratch = mruntime::qwen2_init_scratch(cfg, arenas.scratch, args.max_batch_tokens);

        mruntime::PThreadPool pool = mruntime::PThreadPool::Create(args.num_threads);

        // Warmup: one short prefill+decode with trace disabled.
        {
            mruntime::TraceCollector::instance().set_enabled(false);
            mruntime::TraceCollector::instance().reset();
            uint64_t rng_state = gen_cfg.seed;
            const uint16_t* warm_logits =
                mruntime::qwen2_prefill(cfg, weights, kv_cache, scratch, prompt_tokens.data(), prompt_tokens.size(), &pool);
            int32_t tok = mruntime::qwen2_sample(warm_logits, cfg.vocab_size, gen_cfg, &rng_state);
            for (size_t i = 1; i < std::min<size_t>(args.max_new_tokens, 4); ++i) {
                const uint16_t* logits = mruntime::qwen2_decode(cfg, weights, kv_cache, scratch, tok, &pool);
                tok = mruntime::qwen2_sample(logits, cfg.vocab_size, gen_cfg, &rng_state);
            }
        }

        mruntime::TraceCollector::instance().set_enabled(args.trace);
        mruntime::TraceCollector::instance().reset();

        std::vector<int32_t> output_tokens(args.prompt_len + args.max_new_tokens);
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), output_tokens.begin());

        uint64_t rng_state = gen_cfg.seed;
        int32_t next_token = 0;
        size_t generated_tokens = 0;

        const auto t0 = Clock::now();
        const auto t_prefill_start = Clock::now();
        const uint16_t* last_logits = nullptr;

        last_logits = mruntime::qwen2_prefill(
            cfg,
            weights,
            kv_cache,
            scratch,
            prompt_tokens.data(),
            prompt_tokens.size(),
            &pool
        );

        const auto t_prefill_end = Clock::now();

        // Sample first new token
        next_token = mruntime::qwen2_sample(last_logits, cfg.vocab_size, gen_cfg, &rng_state);
        
        const auto t_first_token = Clock::now();

        output_tokens[args.prompt_len] = next_token;
        generated_tokens = 1;

        const auto t_decode_start = Clock::now();

        for (; generated_tokens < args.max_new_tokens; ++generated_tokens) {
            if (gen_cfg.eos_token_id >= 0 && next_token == gen_cfg.eos_token_id) {
                break;
            }
            const uint16_t* logits = mruntime::qwen2_decode(
                cfg,
                weights,
                kv_cache,
                scratch,
                next_token,
                &pool
            );
            next_token = mruntime::qwen2_sample(logits, cfg.vocab_size, gen_cfg, &rng_state);
            output_tokens[args.prompt_len + generated_tokens] = next_token;
        }

        const auto t_decode_end = Clock::now();

        const mruntime::bench::E2EMetrics metrics = mruntime::bench::make_e2e_metrics(
            args.prompt_len,
            args.max_new_tokens,
            pool.threads_count(),
            generated_tokens,
            t0,
            t_first_token,
            t_prefill_start,
            t_prefill_end,
            t_decode_start,
            t_decode_end
        );
        mruntime::bench::print_e2e_metrics(metrics);

        if (args.trace) {
            mruntime::TraceCollector::instance().print_summary();
            if (!args.trace_json.empty()) {
                if (!mruntime::TraceCollector::instance().export_chrome_json(args.trace_json.c_str())) {
                    std::cerr << "Warning: Failed to export trace JSON to: " << args.trace_json << "\n";
                } else {
                    std::cout << "\nTrace JSON: " << args.trace_json << "\n";
                }
            }
        }

        mruntime::destroy_qwen2_arenas(arenas);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
