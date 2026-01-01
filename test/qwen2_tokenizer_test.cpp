#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "qwen2_tokenizer.h"

using mruntime::Qwen2Tokenizer;

// Assumes tests are run from the build directory (see test/CMakeLists.txt WORKING_DIRECTORY).
static constexpr const char* VOCAB_PATH = "../models/Qwen2.5-0.5B-Instruct/vocab.json";
static constexpr const char* MERGES_PATH = "../models/Qwen2.5-0.5B-Instruct/merges.txt";
static constexpr const char* TOKENIZER_CONFIG_PATH = "../models/Qwen2.5-0.5B-Instruct/tokenizer_config.json";

static void test_encode_ordinary_matches_reference() {
    const auto tok = Qwen2Tokenizer::from_files(VOCAB_PATH, MERGES_PATH, TOKENIZER_CONFIG_PATH);

    struct Case {
        const char* text;
        std::vector<int> expected;
    };

    const std::vector<Case> cases = {
        {"hello", {14990}},
        {" world", {1879}},
        {"hello world", {14990, 1879}},
        {"hello ", {14990, 220}},
        {"hello  ", {14990, 256}},
        {"I'm", {40, 2776}},
        {"\n", {198}},
        {"hello\n\n", {14990, 271}},
        {" \n", {715}},
        {"  ", {256}},
        {"hello\tworld", {14990, 76508}},
        {"😊", {144236}},
        {"", {}},
    };

    for (const auto& c : cases) {
        auto ids = tok.encode_ordinary(c.text);
        assert(ids == c.expected);
        assert(tok.decode(ids) == std::string(c.text));
    }

    std::cout << "test_encode_ordinary_matches_reference PASSED\n";
}

static void test_encode_allows_added_tokens_matches_reference() {
    const auto tok = Qwen2Tokenizer::from_files(VOCAB_PATH, MERGES_PATH, TOKENIZER_CONFIG_PATH);

    struct Case {
        const char* text;
        std::vector<int> expected;
    };

    const std::vector<Case> cases = {
        {"<|endoftext|>", {151643}},
        {"hi<|endoftext|>!", {6023, 151643, 0}},
        {"<|im_start|>user\nhi<|im_end|>", {151644, 872, 198, 6023, 151645}},
    };

    for (const auto& c : cases) {
        auto ids = tok.encode(c.text);
        assert(ids == c.expected);
        assert(tok.decode(ids) == std::string(c.text));
    }

    std::cout << "test_encode_allows_added_tokens_matches_reference PASSED\n";
}

int main() {
    test_encode_ordinary_matches_reference();
    test_encode_allows_added_tokens_matches_reference();

    std::cout << "\nAll Qwen2 tokenizer tests passed!\n";
    return 0;
}


