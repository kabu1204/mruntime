#include "qwen2_tokenizer.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace mruntime {
namespace {

static constexpr std::string_view kQwen2Pat =
    // Qwen2 ships the GPT-2 regex with a `(?!\S)` negative lookahead, but RE2
    // does not support look-around. Dropping that alternative is sufficient
    // for matching the HF tokenization behavior in practice (see tests).
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";

auto read_entire_file(const std::string& path) -> std::string {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

auto parse_vocab_json(const std::string& vocab_json_path) -> std::unordered_map<std::string, int> {
    std::string content = read_entire_file(vocab_json_path);
    nlohmann::json j = nlohmann::json::parse(content);
    if (!j.is_object()) {
        throw std::runtime_error("vocab.json must be a JSON object: " + vocab_json_path);
    }

    std::unordered_map<std::string, int> vocab;
    vocab.reserve(j.size());
    for (auto it = j.begin(); it != j.end(); ++it) {
        vocab.emplace(it.key(), it.value().get<int>());
    }
    return vocab;
}

auto parse_added_tokens_decoder(const std::string& tokenizer_config_json_path)
    -> std::unordered_map<std::string, int> {
    std::unordered_map<std::string, int> added;

    std::string content = read_entire_file(tokenizer_config_json_path);
    nlohmann::json j = nlohmann::json::parse(content);
    if (!j.is_object()) {
        throw std::runtime_error("tokenizer_config.json must be a JSON object: " + tokenizer_config_json_path);
    }

    auto it = j.find("added_tokens_decoder");
    if (it == j.end() || !it->is_object()) {
        return added;
    }

    added.reserve(it->size());
    for (const auto& [id_str, tok_obj] : it->items()) {
        int id = 0;
        try {
            id = std::stoi(id_str);
        } catch (...) {
            continue;
        }
        if (!tok_obj.is_object()) continue;
        auto cit = tok_obj.find("content");
        if (cit == tok_obj.end() || !cit->is_string()) continue;
        std::string token_content = cit->get<std::string>();
        if (!token_content.empty()) {
            added.emplace(std::move(token_content), id);
        }
    }

    return added;
}

auto build_byte_decoder() -> std::unordered_map<char32_t, unsigned char> {
    std::vector<int> bs;
    bs.reserve(256);
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    std::array<bool, 256> in_bs{};
    for (int b : bs) in_bs[static_cast<size_t>(b)] = true;

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!in_bs[static_cast<size_t>(b)]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    std::unordered_map<char32_t, unsigned char> decoder;
    decoder.reserve(256);
    for (size_t idx = 0; idx < bs.size(); ++idx) {
        decoder.emplace(static_cast<char32_t>(cs[idx]), static_cast<unsigned char>(bs[idx]));
    }
    return decoder;
}

auto next_utf8_codepoint(const std::string& s, size_t& pos, char32_t& cp_out) -> bool {
    if (pos >= s.size()) return false;
    unsigned char c0 = static_cast<unsigned char>(s[pos]);
    if (c0 < 0x80) {
        cp_out = static_cast<char32_t>(c0);
        ++pos;
        return true;
    }
    if ((c0 & 0xE0) == 0xC0) {
        if (pos + 1 >= s.size()) throw std::runtime_error("Invalid UTF-8: truncated sequence");
        unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        if ((c1 & 0xC0) != 0x80) throw std::runtime_error("Invalid UTF-8: bad continuation byte");
        cp_out = static_cast<char32_t>(((c0 & 0x1F) << 6) | (c1 & 0x3F));
        pos += 2;
        return true;
    }
    if ((c0 & 0xF0) == 0xE0) {
        if (pos + 2 >= s.size()) throw std::runtime_error("Invalid UTF-8: truncated sequence");
        unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[pos + 2]);
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) {
            throw std::runtime_error("Invalid UTF-8: bad continuation byte");
        }
        cp_out = static_cast<char32_t>(((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F));
        pos += 3;
        return true;
    }
    if ((c0 & 0xF8) == 0xF0) {
        if (pos + 3 >= s.size()) throw std::runtime_error("Invalid UTF-8: truncated sequence");
        unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[pos + 2]);
        unsigned char c3 = static_cast<unsigned char>(s[pos + 3]);
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) {
            throw std::runtime_error("Invalid UTF-8: bad continuation byte");
        }
        cp_out = static_cast<char32_t>(
            ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F)
        );
        pos += 4;
        return true;
    }
    throw std::runtime_error("Invalid UTF-8: invalid lead byte");
}

auto decode_vocab_token_to_bytes(
    const std::string& token,
    const std::unordered_map<char32_t, unsigned char>& byte_decoder
) -> std::string {
    std::string out;
    out.reserve(token.size());
    size_t pos = 0;
    char32_t cp = 0;
    while (next_utf8_codepoint(token, pos, cp)) {
        auto it = byte_decoder.find(cp);
        if (it == byte_decoder.end()) {
            throw std::runtime_error("Tokenizer vocab contains unknown byte encoder codepoint");
        }
        out.push_back(static_cast<char>(it->second));
    }
    return out;
}

void validate_merges(
    const std::string& merges_txt_path,
    const std::unordered_map<std::string, int>& vocab_str_to_id
) {
    std::ifstream f(merges_txt_path);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + merges_txt_path);
    }
    std::string line;
    int expected_id = 256;
    int line_idx = 0;
    while (std::getline(f, line)) {
        // Skip an optional header, but do NOT treat '#' as a comment (it is a valid token).
        if (line_idx == 0 && line.rfind("#version:", 0) == 0) {
            ++line_idx;
            continue;
        }
        if (line.empty()) {
            ++line_idx;
            continue;
        }
        size_t sp = line.find(' ');
        if (sp == std::string::npos) {
            throw std::runtime_error("Invalid merges.txt line (missing space): " + line);
        }
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        if (a.empty() || b.empty()) {
            throw std::runtime_error("Invalid merges.txt line (empty token): " + line);
        }
        std::string merged = a + b;
        auto it = vocab_str_to_id.find(merged);
        if (it == vocab_str_to_id.end()) {
            throw std::runtime_error("merges.txt token not found in vocab: " + merged);
        }
        if (it->second != expected_id) {
            std::ostringstream ss;
            ss << "Unexpected merge token id for '" << merged << "': got " << it->second
               << ", expected " << expected_id;
            throw std::runtime_error(ss.str());
        }
        ++expected_id;
        ++line_idx;
    }
}

}  // namespace

Qwen2Tokenizer::Qwen2Tokenizer(tiktoken::tiktoken tok) : tok_(std::move(tok)) {}

Qwen2Tokenizer Qwen2Tokenizer::from_files(
    const std::string& vocab_json_path,
    const std::string& merges_txt_path,
    const std::string& tokenizer_config_json_path
) {
    // 1) Load vocab (token string -> id).
    std::unordered_map<std::string, int> vocab_str_to_id = parse_vocab_json(vocab_json_path);

    // 2) Sanity-check merges ordering matches IDs (Qwen2.5 uses GPT-2-style numbering).
    validate_merges(merges_txt_path, vocab_str_to_id);

    // 3) Decode vocab token strings into raw byte strings for tiktoken-style encoding.
    const auto byte_decoder = build_byte_decoder();

    std::unordered_map<std::string, int> encoder_bytes_to_id;
    encoder_bytes_to_id.reserve(vocab_str_to_id.size());
    for (const auto& [tok_str, id] : vocab_str_to_id) {
        std::string bytes = decode_vocab_token_to_bytes(tok_str, byte_decoder);
        encoder_bytes_to_id.emplace(std::move(bytes), id);
    }

    // 4) Added/special tokens live outside vocab.json.
    std::unordered_map<std::string, int> special = parse_added_tokens_decoder(tokenizer_config_json_path);

    // 5) Build the tokenizer.
    return Qwen2Tokenizer(tiktoken::tiktoken(std::move(encoder_bytes_to_id), std::move(special), std::string(kQwen2Pat)));
}

std::vector<int> Qwen2Tokenizer::encode_ordinary(const std::string& text) const {
    return tok_.encode_ordinary(text);
}

std::vector<int> Qwen2Tokenizer::encode(const std::string& text) const {
    return tok_.encode(text);
}

std::string Qwen2Tokenizer::decode(const std::vector<int>& tokens) const {
    return tok_.decode(tokens);
}

}  // namespace mruntime


