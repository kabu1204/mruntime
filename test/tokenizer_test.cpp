#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bpe.hpp"

namespace {

auto make_test_tokenizer() -> tiktoken::tiktoken {
    std::unordered_map<std::string, int> encoder = {
        {"h", 1},
        {"e", 2},
        {"l", 3},
        {"o", 4},
        {" ", 5},
        {"w", 6},
        {"r", 7},
        {"d", 8},
        {"i", 9},
        {"!", 10},
    };

    std::unordered_map<std::string, int> special_encoder = {
        {"<|eot|>", 1000},
    };

    return tiktoken::tiktoken(std::move(encoder), std::move(special_encoder), ".");
}

} // namespace

void test_encode_ordinary_roundtrip() {
    const auto tok = make_test_tokenizer();

    const std::string text = "hello world";
    const std::vector<int> tokens = tok.encode_ordinary(text);
    std::cout << "tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    const std::string decoded = tok.decode(tokens);

    assert(decoded == text);
    std::cout << "test_encode_ordinary_roundtrip PASSED\n";
}

void test_encode_allows_special_tokens() {
    const auto tok = make_test_tokenizer();

    const std::string text = "hi<|eot|>!";
    const std::vector<int> tokens = tok.encode(text);

    assert(tokens.size() == 4);
    assert(tokens[0] == 1);     // "h"
    assert(tokens[1] == 9);     // "i"
    assert(tokens[2] == 1000);  // "<|eot|>"
    assert(tokens[3] == 10);    // "!"

    const std::string decoded = tok.decode(tokens);
    assert(decoded == text);
    std::cout << "test_encode_allows_special_tokens PASSED\n";
}

int main() {
    test_encode_ordinary_roundtrip();
    test_encode_allows_special_tokens();

    std::cout << "\nAll tokenizer tests passed!\n";
    return 0;
}

