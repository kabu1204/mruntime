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

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::string model_dir;
    size_t prompt_len = 8;
    size_t max_new_tokens = 32;
    size_t max_seq_len = 2048;
    size_t max_batch_tokens = 64;
    size_t num_threads = 0;  // 0 = auto

    bool trace = true;
    std::string trace_json = "trace.json";

    // Sampling config (default: greedy + no EOS so decode length is fixed)
    bool greedy = true;
    float temperature = 0.0f;
    size_t top_k = 0;
    float top_p = 1.0f;
    int32_t eos_token_id = -1;  // <0 disables early stop
    uint64_t seed = 42;
};

auto file_exists(const std::string& path) -> bool {
    std::ifstream file(path);
    return file.good();
}

auto join_path(const std::string& dir, const std::string& file) -> std::string {
    if (dir.empty()) return file;
    if (dir.back() == '/') return dir + file;
    return dir + "/" + file;
}

auto read_text_file(const std::string& path) -> std::string {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

auto resolve_model_dir(const std::string& user_model_dir) -> std::string {
    const std::vector<std::string> candidates = user_model_dir.empty()
        ? std::vector<std::string>{
            "models/Qwen2.5-0.5B-Instruct",
            "../models/Qwen2.5-0.5B-Instruct",
        }
        : std::vector<std::string>{user_model_dir};

    for (const auto& dir : candidates) {
        if (file_exists(join_path(dir, "config.json")) &&
            file_exists(join_path(dir, "model.safetensors"))) {
            return dir;
        }
    }

    std::ostringstream ss;
    ss << "Could not find a valid Qwen2 model directory.\nTried:\n";
    for (const auto& dir : candidates) {
        ss << "  - " << dir << "\n";
    }
    ss << "\nExpected files: config.json, model.safetensors\n";
    throw std::runtime_error(ss.str());
}

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
        << "  --prompt-len N          Prompt length in tokens (default: 8)\n"
        << "  --max-new-tokens N      Number of tokens to decode (default: 32)\n"
        << "  --max-seq-len N         KV cache max sequence length (default: 2048)\n"
        << "  --max-batch-tokens N    Max tokens per forward call (default: 64)\n"
        << "  --threads N             Number of threads (default: auto-detect)\n"
        << "  --eos-token-id ID       Stop token id (<0 disables; default: -1)\n"
        << "  --trace 0|1             Enable trace collection (default: 1)\n"
        << "  --trace-json PATH       Export Chrome trace JSON (default: trace.json)\n"
        << "  -h, --help              Show this help\n";
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

        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--model-dir") {
            args.model_dir = require_value("--model-dir");
        } else if (a == "--prompt-len") {
            args.prompt_len = static_cast<size_t>(std::stoull(require_value("--prompt-len")));
        } else if (a == "--max-new-tokens") {
            args.max_new_tokens = static_cast<size_t>(std::stoull(require_value("--max-new-tokens")));
        } else if (a == "--max-seq-len") {
            args.max_seq_len = static_cast<size_t>(std::stoull(require_value("--max-seq-len")));
        } else if (a == "--max-batch-tokens") {
            args.max_batch_tokens = static_cast<size_t>(std::stoull(require_value("--max-batch-tokens")));
        } else if (a == "--threads") {
            args.num_threads = static_cast<size_t>(std::stoull(require_value("--threads")));
        } else if (a == "--eos-token-id") {
            args.eos_token_id = static_cast<int32_t>(std::stoll(require_value("--eos-token-id")));
        } else if (a == "--trace") {
            args.trace = (std::stoi(require_value("--trace")) != 0);
        } else if (a == "--trace-json") {
            args.trace_json = require_value("--trace-json");
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    return args;
}

auto seconds_since(const Clock::time_point& start, const Clock::time_point& end) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

auto ms_since(const Clock::time_point& start, const Clock::time_point& end) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::string model_dir = resolve_model_dir(args.model_dir);

        std::cout << "Model dir: " << model_dir << "\n";

        const std::string config_json = read_text_file(join_path(model_dir, "config.json"));
        const mruntime::QwenConfig cfg = mruntime::QwenConfig::from_json(config_json);

        if (args.prompt_len == 0) {
            throw std::runtime_error("--prompt-len must be > 0");
        }
        if (args.max_batch_tokens == 0) {
            throw std::runtime_error("--max-batch-tokens must be > 0");
        }
        if (args.prompt_len + args.max_new_tokens > args.max_seq_len) {
            std::ostringstream ss;
            ss << "prompt_len + max_new_tokens exceeds max_seq_len: "
               << args.prompt_len << " + " << args.max_new_tokens << " > " << args.max_seq_len;
            throw std::runtime_error(ss.str());
        }

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

        const mruntime::Qwen2MemorySizes sizes =
            mruntime::qwen2_memory_sizes(cfg, args.max_seq_len, args.max_batch_tokens);

        mruntime::Qwen2Arenas arenas = mruntime::create_qwen2_arenas(
            sizes.weights_bytes + sizes.packed_weights_bytes,
            sizes.kv_cache_bytes,
            sizes.scratch_bytes
        );

        auto st = mruntime::SafeTensorsFile::open(join_path(model_dir, "model.safetensors"));
        if (!st) {
            throw std::runtime_error("Failed to open model.safetensors");
        }
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

        const double prefill_s = seconds_since(t_prefill_start, t_prefill_end);
        const double prefill_tok_s = (prefill_s > 0.0) ? (static_cast<double>(args.prompt_len) / prefill_s) : 0.0;

        const double ttft_ms = ms_since(t0, t_first_token);

        const size_t decode_steps = (generated_tokens > 0) ? (generated_tokens - 1) : 0;
        const double decode_s = seconds_since(t_decode_start, t_decode_end);
        const double decode_tok_s = (decode_s > 0.0) ? (static_cast<double>(decode_steps) / decode_s) : 0.0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== E2E Metrics ===\n";
        std::cout << "prompt_len:          " << args.prompt_len << " tokens\n";
        std::cout << "max_new_tokens:      " << args.max_new_tokens << " tokens\n";
        std::cout << "threads:             " << pool.threads_count() << "\n";
        std::cout << "ttft:                " << ttft_ms << " ms\n";
        std::cout << "prefill:             " << (prefill_s * 1000.0) << " ms, " << prefill_tok_s << " tok/s\n";
        std::cout << "decode:              " << decode_steps << " tokens, " << (decode_s * 1000.0) << " ms, " << decode_tok_s << " tok/s\n";

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
