#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mruntime/cpu_backend.h"
#include "mruntime/inference_engine.h"
#include "mruntime/qwen_model.h"
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
    int eos_token_id = 151645;  // <|im_end|>
};

auto print_usage(const char* argv0) -> void {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir PATH        Path to model directory (default: auto-detect)\n"
        << "  --system TEXT           System prompt (default: \"You are a helpful assistant.\")\n"
        << "  --max-new-tokens N      Max new tokens to generate (default: 256)\n"
        << "  --temperature T         Sampling temperature (default: 0.7)\n"
        << "  --top-k N               Top-k sampling (default: 20)\n"
        << "  --top-p P               Top-p sampling (default: 0.8)\n"
        << "  --greedy                Greedy decoding\n"
        << "  --seed N                RNG seed (default: 42)\n"
        << "  --eos-token-id ID       Stop token id (default: 151645 = <|im_end|>)\n"
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

        const std::string config_json = read_text_file(join_path(model_dir, "config.json"));
        const mruntime::QwenConfig model_config = mruntime::QwenConfig::from_json(config_json);

        auto weights = mruntime::SafeTensorsFile::open(join_path(model_dir, "model.safetensors"));
        if (!weights) {
            throw std::runtime_error("Failed to open model weights");
        }

        mruntime::QwenModel model(model_config);
        model.load_weights(*weights);

        mruntime::CpuBackend backend;

        const auto tokenizer = mruntime::Qwen2Tokenizer::from_files(
            join_path(model_dir, "vocab.json"),
            join_path(model_dir, "merges.txt"),
            join_path(model_dir, "tokenizer_config.json")
        );

        mruntime::GenerationConfig gen;
        gen.max_new_tokens = args.max_new_tokens;
        gen.temperature = args.temperature;
        gen.top_k = args.top_k;
        gen.top_p = args.top_p;
        gen.greedy = args.greedy;
        gen.seed = args.seed;
        gen.eos_token_id = args.eos_token_id;

        std::vector<Message> history;
        if (!args.system_prompt.empty()) {
            history.push_back({"system", args.system_prompt});
        }

        mruntime::InferenceEngine engine(model, backend);

        std::cout << "mruntime Qwen2 chat\n";
        std::cout << "Type /exit to quit, /reset to clear conversation.\n";

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
                continue;
            }
            if (user.empty()) {
                continue;
            }

            history.push_back({"user", user});

            const std::string prompt = apply_qwen2_chat_template(history, /*add_generation_prompt=*/true);
            const std::vector<int> prompt_tokens = tokenizer.encode(prompt);

            const std::vector<int> all_tokens = engine.generate(prompt_tokens, gen);
            if (all_tokens.size() < prompt_tokens.size()) {
                throw std::runtime_error("Internal error: generated token list shorter than prompt");
            }

            std::vector<int> gen_tokens(all_tokens.begin() + static_cast<long>(prompt_tokens.size()), all_tokens.end());
            auto eos_it = std::find(gen_tokens.begin(), gen_tokens.end(), gen.eos_token_id);
            std::vector<int> reply_tokens(gen_tokens.begin(), eos_it);

            const std::string reply = tokenizer.decode(reply_tokens);
            std::cout << reply << "\n";

            history.push_back({"assistant", reply});
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}


