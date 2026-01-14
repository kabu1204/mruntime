#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mruntime/arena.h"
#include "mruntime/qwen_config.h"
#include "mruntime/qwen2_forward.h"
#include "mruntime/qwen2_generate.h"
#include "mruntime/qwen2_weights.h"
#include "mruntime/safetensors.h"

#include "qwen2_tokenizer.h"

namespace {

struct Message {
    std::string role;
    std::string content;
};

auto read_text_file(const std::string& path) -> std::string {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

auto file_exists(const std::string& path) -> bool {
    std::ifstream file(path);
    return file.good();
}

auto join_path(const std::string& dir, const std::string& file) -> std::string {
    if (dir.empty()) return file;
    if (dir.back() == '/') return dir + file;
    return dir + "/" + file;
}

auto apply_qwen2_chat_template(
    const std::vector<Message>& messages,
    bool add_generation_prompt
) -> std::string {
    std::string out;
    out.reserve(1024);

    for (const auto& m : messages) {
        out += "<|im_start|>";
        out += m.role;
        out += "\n";
        out += m.content;
        out += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
    }

    return out;
}

struct Args {
    std::string model_dir;
    std::string system_prompt = "You are a helpful assistant.";
    size_t max_new_tokens = 256;
    float temperature = 0.7f;
    size_t top_k = 20;
    float top_p = 0.8f;
    bool greedy = false;
    uint64_t seed = 42;
    int32_t eos_token_id = 151645;  // <|im_end|>
    size_t max_seq_len = 2048;
    size_t num_threads = 0;  // 0 = auto-detect
};

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
        << "  --system TEXT           System prompt (default: \"You are a helpful assistant.\")\n"
        << "  --max-new-tokens N      Max new tokens to generate (default: 256)\n"
        << "  --max-seq-len N         Max sequence length (default: 2048)\n"
        << "  --temperature T         Sampling temperature (default: 0.7)\n"
        << "  --top-k N               Top-k sampling (default: 20)\n"
        << "  --top-p P               Top-p sampling (default: 0.8)\n"
        << "  --greedy                Greedy decoding\n"
        << "  --seed N                RNG seed (default: 42)\n"
        << "  --eos-token-id ID       Stop token id (default: 151645 = <|im_end|>)\n"
        << "  --threads N             Number of threads (default: auto-detect)\n"
        << "  -h, --help              Show this help\n";
}

auto parse_args(int argc, char** argv) -> Args {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
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
        } else if (a == "--system") {
            args.system_prompt = require_value("--system");
        } else if (a == "--max-new-tokens") {
            args.max_new_tokens = static_cast<size_t>(std::stoull(require_value("--max-new-tokens")));
        } else if (a == "--max-seq-len") {
            args.max_seq_len = static_cast<size_t>(std::stoull(require_value("--max-seq-len")));
        } else if (a == "--temperature") {
            args.temperature = std::stof(require_value("--temperature"));
        } else if (a == "--top-k") {
            args.top_k = static_cast<size_t>(std::stoull(require_value("--top-k")));
        } else if (a == "--top-p") {
            args.top_p = std::stof(require_value("--top-p"));
        } else if (a == "--greedy") {
            args.greedy = true;
        } else if (a == "--seed") {
            args.seed = static_cast<uint64_t>(std::stoull(require_value("--seed")));
        } else if (a == "--eos-token-id") {
            args.eos_token_id = std::stoi(require_value("--eos-token-id"));
        } else if (a == "--threads") {
            args.num_threads = static_cast<size_t>(std::stoull(require_value("--threads")));
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    return args;
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
            file_exists(join_path(dir, "model.safetensors")) &&
            file_exists(join_path(dir, "vocab.json")) &&
            file_exists(join_path(dir, "merges.txt")) &&
            file_exists(join_path(dir, "tokenizer_config.json"))) {
            return dir;
        }
    }

    std::ostringstream ss;
    ss << "Could not find a valid Qwen2 model directory.\nTried:\n";
    for (const auto& dir : candidates) {
        ss << "  - " << dir << "\n";
    }
    ss << "\nExpected files: config.json, model.safetensors, vocab.json, merges.txt, tokenizer_config.json\n";
    throw std::runtime_error(ss.str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const std::string model_dir = resolve_model_dir(args.model_dir);

        std::cout << "Loading model from: " << model_dir << "\n";

        // Load config
        const std::string config_json = read_text_file(join_path(model_dir, "config.json"));
        const mruntime::QwenConfig cfg = mruntime::QwenConfig::from_json(config_json);

        std::cout << "Model config: " << cfg.num_layers << " layers, "
                  << cfg.hidden_size << " hidden, "
                  << cfg.vocab_size << " vocab\n";

        // Calculate memory requirements
        const size_t max_batch_tokens = 64;  // Max tokens per forward call
        mruntime::Qwen2MemorySizes sizes = mruntime::qwen2_memory_sizes(cfg, args.max_seq_len, max_batch_tokens);

        std::cout << "Memory: weights=" << sizes.weights_bytes / 1024 / 1024 << "MB, "
                  << "kv_cache=" << sizes.kv_cache_bytes / 1024 / 1024 << "MB, "
                  << "scratch=" << sizes.scratch_bytes / 1024 / 1024 << "MB\n";

        // Allocate arenas
        mruntime::Qwen2Arenas arenas = mruntime::create_qwen2_arenas(
            sizes.weights_bytes + sizes.packed_weights_bytes,
            sizes.kv_cache_bytes,
            sizes.scratch_bytes
        );

        // Load weights
        std::cout << "Loading weights...\n";
        auto st = mruntime::SafeTensorsFile::open(join_path(model_dir, "model.safetensors"));
        if (!st) {
            throw std::runtime_error("Failed to open model weights");
        }
        mruntime::Qwen2Weights weights = mruntime::qwen2_load_weights(cfg, *st, arenas.weights, true);

        // Initialize KV cache and scratch
        mruntime::Qwen2KVCache kv_cache = mruntime::qwen2_init_kv_cache(cfg, arenas.kv_cache, args.max_seq_len);
        mruntime::Qwen2Scratch scratch = mruntime::qwen2_init_scratch(cfg, arenas.scratch, max_batch_tokens);

        // Create thread pool
        mruntime::PThreadPool pool = mruntime::PThreadPool::Create(args.num_threads);
        std::cout << "Using " << pool.threads_count() << " threads\n";

        // Load tokenizer
        const auto tokenizer = mruntime::Qwen2Tokenizer::from_files(
            join_path(model_dir, "vocab.json"),
            join_path(model_dir, "merges.txt"),
            join_path(model_dir, "tokenizer_config.json")
        );

        // Set up generation config
        mruntime::Qwen2GenerateConfig gen_cfg;
        gen_cfg.max_new_tokens = args.max_new_tokens;
        gen_cfg.temperature = args.temperature;
        gen_cfg.top_k = args.top_k;
        gen_cfg.top_p = args.top_p;
        gen_cfg.greedy = args.greedy;
        gen_cfg.seed = args.seed;
        gen_cfg.eos_token_id = args.eos_token_id;

        // Chat history
        std::vector<Message> history;
        if (!args.system_prompt.empty()) {
            history.push_back({"system", args.system_prompt});
        }

        std::cout << "\nmruntime Qwen2 chat (bare-metal API)\n";
        std::cout << "Type /exit to quit, /reset to clear conversation.\n";

        // Allocate output buffer
        std::vector<int32_t> output_tokens(args.max_seq_len + args.max_new_tokens);

        while (true) {
            std::cout << "\n> " << std::flush;

            std::string user;
            if (!std::getline(std::cin, user)) {
                break;
            }

            if (user == "/exit" || user == "/quit") {
                break;
            }
            if (user == "/reset") {
                history.clear();
                if (!args.system_prompt.empty()) {
                    history.push_back({"system", args.system_prompt});
                }
                mruntime::qwen2_reset_kv_cache(kv_cache);
                continue;
            }
            if (user.empty()) {
                continue;
            }

            history.push_back({"user", user});

            // Apply chat template
            const std::string prompt = apply_qwen2_chat_template(history, /*add_generation_prompt=*/true);
            const std::vector<int> prompt_tokens_int = tokenizer.encode(prompt);

            // Convert to int32_t
            std::vector<int32_t> prompt_tokens(prompt_tokens_int.begin(), prompt_tokens_int.end());

            // Check sequence length
            if (prompt_tokens.size() > args.max_seq_len) {
                std::cerr << "Warning: Prompt too long (" << prompt_tokens.size() << " tokens), truncating.\n";
                prompt_tokens.resize(args.max_seq_len);
            }

            // Reset KV cache for new conversation turn
            mruntime::qwen2_reset_kv_cache(kv_cache);

            // Generate
            size_t total_len = mruntime::qwen2_generate(
                cfg,
                weights,
                kv_cache,
                scratch,
                prompt_tokens.data(),
                prompt_tokens.size(),
                output_tokens.data(),
                gen_cfg,
                &pool
            );

            // Extract generated tokens
            if (total_len <= prompt_tokens.size()) {
                std::cerr << "Warning: No tokens generated\n";
                continue;
            }

            std::vector<int> gen_tokens_int;
            for (size_t i = prompt_tokens.size(); i < total_len; ++i) {
                int32_t tok = output_tokens[i];
                if (tok == gen_cfg.eos_token_id) break;
                gen_tokens_int.push_back(static_cast<int>(tok));
            }

            // Decode and print
            const std::string reply = tokenizer.decode(gen_tokens_int);
            std::cout << reply << "\n";

            history.push_back({"assistant", reply});
        }

        // Cleanup
        mruntime::destroy_qwen2_arenas(arenas);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
