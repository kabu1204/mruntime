#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mruntime::bench {

using E2EClock = std::chrono::steady_clock;

struct CommonE2EArgs {
    std::string model_dir;
    size_t prompt_len = 8;
    size_t max_new_tokens = 32;
    size_t max_seq_len = 2048;
    size_t max_batch_tokens = 64;
    size_t num_threads = 0;

    bool trace = true;
    std::string trace_json = "trace.json";

    bool greedy = true;
    float temperature = 0.0f;
    size_t top_k = 0;
    float top_p = 1.0f;
    int32_t eos_token_id = -1;
    uint64_t seed = 42;
};

struct E2EMetrics {
    size_t prompt_len = 0;
    size_t max_new_tokens = 0;
    size_t threads = 0;
    size_t decode_steps = 0;
    double ttft_ms = 0.0;
    double prefill_s = 0.0;
    double decode_s = 0.0;
};

inline auto file_exists(const std::string& path) -> bool {
    std::ifstream file(path);
    return file.good();
}

inline auto join_path(const std::string& dir, const std::string& file) -> std::string {
    if (dir.empty()) return file;
    if (dir.back() == '/') return dir + file;
    return dir + "/" + file;
}

inline auto read_text_file(const std::string& path) -> std::string {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

inline auto resolve_model_dir(const std::string& user_model_dir) -> std::string {
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

inline auto is_help_flag(const std::string& arg) -> bool {
    return arg == "-h" || arg == "--help";
}

inline auto print_common_e2e_usage(std::ostream& os) -> void {
    os << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
       << "  --prompt-len N          Prompt length in tokens (default: 8)\n"
       << "  --max-new-tokens N      Number of tokens to decode (default: 32)\n"
       << "  --max-seq-len N         KV cache max sequence length (default: 2048)\n"
       << "  --max-batch-tokens N    Max tokens per forward call (default: 64)\n"
       << "  --threads N             Number of threads (default: auto-detect)\n"
       << "  --eos-token-id ID       Stop token id (<0 disables; default: -1)\n"
       << "  --trace 0|1             Enable trace collection (default: 1)\n"
       << "  --trace-json PATH       Export Chrome trace JSON (default: trace.json)\n";
}

template <typename Args, typename RequireValue>
inline auto parse_common_e2e_arg(const std::string& arg, RequireValue&& require_value, Args& args) -> bool {
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
    } else if (arg == "--threads") {
        args.num_threads = static_cast<size_t>(std::stoull(require_value("--threads")));
    } else if (arg == "--eos-token-id") {
        args.eos_token_id = static_cast<int32_t>(std::stoll(require_value("--eos-token-id")));
    } else if (arg == "--trace") {
        args.trace = (std::stoi(require_value("--trace")) != 0);
    } else if (arg == "--trace-json") {
        args.trace_json = require_value("--trace-json");
    } else {
        return false;
    }
    return true;
}

template <typename Args>
inline auto validate_common_e2e_args(const Args& args) -> void {
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
}

inline auto seconds_since(const E2EClock::time_point& start, const E2EClock::time_point& end) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

inline auto ms_since(const E2EClock::time_point& start, const E2EClock::time_point& end) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

inline auto tokens_per_second(size_t tokens, double seconds) -> double {
    return (seconds > 0.0) ? (static_cast<double>(tokens) / seconds) : 0.0;
}

inline auto make_e2e_metrics(
    size_t prompt_len,
    size_t max_new_tokens,
    size_t threads,
    size_t generated_tokens,
    const E2EClock::time_point& t0,
    const E2EClock::time_point& t_first_token,
    const E2EClock::time_point& t_prefill_start,
    const E2EClock::time_point& t_prefill_end,
    const E2EClock::time_point& t_decode_start,
    const E2EClock::time_point& t_decode_end
) -> E2EMetrics {
    E2EMetrics metrics;
    metrics.prompt_len = prompt_len;
    metrics.max_new_tokens = max_new_tokens;
    metrics.threads = threads;
    metrics.decode_steps = (generated_tokens > 0) ? (generated_tokens - 1) : 0;
    metrics.ttft_ms = ms_since(t0, t_first_token);
    metrics.prefill_s = seconds_since(t_prefill_start, t_prefill_end);
    metrics.decode_s = seconds_since(t_decode_start, t_decode_end);
    return metrics;
}

inline auto print_e2e_metrics(const E2EMetrics& metrics, std::ostream& os = std::cout) -> void {
    os << std::fixed << std::setprecision(3);
    os << "\n=== E2E Metrics ===\n";
    os << "prompt_len:          " << metrics.prompt_len << " tokens\n";
    os << "max_new_tokens:      " << metrics.max_new_tokens << " tokens\n";
    os << "threads:             " << metrics.threads << "\n";
    os << "ttft:                " << metrics.ttft_ms << " ms\n";
    os << "prefill:             " << (metrics.prefill_s * 1000.0) << " ms, "
       << tokens_per_second(metrics.prompt_len, metrics.prefill_s) << " tok/s\n";
    os << "decode:              " << metrics.decode_steps << " tokens, " << (metrics.decode_s * 1000.0)
       << " ms, " << tokens_per_second(metrics.decode_steps, metrics.decode_s) << " tok/s\n";
}

}  // namespace mruntime::bench
