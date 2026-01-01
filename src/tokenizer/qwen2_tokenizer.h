#pragma once

#include <string>
#include <vector>

#include "bpe.hpp"

namespace mruntime {

// Minimal loader for the Qwen2.5 HF ByteLevel-BPE tokenizer assets shipped in
// models/Qwen2.5-0.5B-Instruct/{vocab.json,merges.txt,tokenizer_config.json}.
//
// NOTE: This is intentionally lightweight and only targets the Qwen2 tokenizer
// format currently vendored in this repo.
class Qwen2Tokenizer {
public:
    static Qwen2Tokenizer from_files(
        const std::string& vocab_json_path,
        const std::string& merges_txt_path,
        const std::string& tokenizer_config_json_path
    );

    std::vector<int> encode_ordinary(const std::string& text) const;
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

private:
    explicit Qwen2Tokenizer(tiktoken::tiktoken tok);

    tiktoken::tiktoken tok_;
};

}  // namespace mruntime


