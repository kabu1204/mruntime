#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "mruntime/dtype.h"
#include "mruntime/safetensors.h"

using namespace mruntime;

// Path to test model - assumes running from build directory
const char* MODEL_PATH = "../models/Qwen2.5-0.5B-Instruct/model.safetensors";

void test_open_safetensors() {
    auto file = SafeTensorsFile::open(MODEL_PATH);
    assert(file != nullptr);
    std::cout << "test_open_safetensors PASSED\n";
}

void test_tensor_names() {
    auto file = SafeTensorsFile::open(MODEL_PATH);
    auto names = file->tensor_names();

    assert(!names.empty());

    // Qwen2 should have these tensors
    assert(file->has_tensor("model.embed_tokens.weight"));
    assert(file->has_tensor("model.norm.weight"));
    assert(file->has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert(file->has_tensor("model.layers.0.mlp.gate_proj.weight"));

    std::cout << "test_tensor_names PASSED (found " << names.size() << " tensors)\n";
}

void test_tensor_info() {
    auto file = SafeTensorsFile::open(MODEL_PATH);

    // Check embedding tensor info
    const auto& embed_info = file->tensor_info("model.embed_tokens.weight");
    assert(embed_info.dtype == DType::BF16);
    assert(embed_info.shape.size() == 2);
    assert(embed_info.shape[0] == 151936);  // vocab_size
    assert(embed_info.shape[1] == 896);     // hidden_size

    // Check attention projection shape
    const auto& q_proj_info = file->tensor_info("model.layers.0.self_attn.q_proj.weight");
    assert(q_proj_info.dtype == DType::BF16);
    assert(q_proj_info.shape.size() == 2);
    assert(q_proj_info.shape[0] == 896);    // hidden_size (output)
    assert(q_proj_info.shape[1] == 896);    // hidden_size (input)

    std::cout << "test_tensor_info PASSED\n";
}

void test_tensor_data() {
    auto file = SafeTensorsFile::open(MODEL_PATH);

    // Get raw pointer to norm weight
    const auto& info = file->tensor_info("model.norm.weight");
    const void* data = file->tensor_data("model.norm.weight");

    assert(data != nullptr);
    assert(info.dtype == DType::BF16);
    assert(info.shape.size() == 1);
    assert(info.shape[0] == 896);

    // Verify the data is readable and has non-zero values
    const uint16_t* bf16_data = static_cast<const uint16_t*>(data);
    bool has_nonzero = false;
    for (size_t i = 0; i < info.shape[0]; ++i) {
        float v = bf16_to_float(bf16_data[i]);
        assert(std::isfinite(v));
        if (v != 0.0f) has_nonzero = true;
    }
    assert(has_nonzero);  // Weight should have non-zero values

    std::cout << "test_tensor_data PASSED\n";
}

void test_bf16_dtype_parsing() {
    auto file = SafeTensorsFile::open(MODEL_PATH);

    // All Qwen2.5 weights should be BF16
    const auto& embed_info = file->tensor_info("model.embed_tokens.weight");
    assert(embed_info.dtype == DType::BF16);

    const auto& norm_info = file->tensor_info("model.norm.weight");
    assert(norm_info.dtype == DType::BF16);

    std::cout << "test_bf16_dtype_parsing PASSED\n";
}

void test_has_tensor() {
    auto file = SafeTensorsFile::open(MODEL_PATH);

    assert(file->has_tensor("model.embed_tokens.weight"));
    assert(file->has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert(!file->has_tensor("nonexistent.tensor"));

    // Qwen2.5 uses tied embeddings - no separate lm_head
    assert(!file->has_tensor("lm_head.weight"));

    std::cout << "test_has_tensor PASSED\n";
}

void test_layer_count() {
    auto file = SafeTensorsFile::open(MODEL_PATH);

    // Count number of layers by checking for layer tensors
    size_t num_layers = 0;
    for (size_t i = 0; i < 100; ++i) {
        std::string key = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
        if (file->has_tensor(key)) {
            num_layers = i + 1;
        } else {
            break;
        }
    }

    assert(num_layers == 24);  // Qwen2.5-0.5B has 24 layers
    std::cout << "test_layer_count PASSED (found " << num_layers << " layers)\n";
}

int main() {
    std::cout << "SafeTensors Loader Tests\n";
    std::cout << "========================\n\n";

    test_open_safetensors();
    test_tensor_names();
    test_tensor_info();
    test_tensor_data();
    test_bf16_dtype_parsing();
    test_has_tensor();
    test_layer_count();

    std::cout << "\nAll loader tests passed!\n";
    return 0;
}
