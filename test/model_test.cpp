#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mruntime/cpu_backend.h"
#include "mruntime/qwen_model.h"
#include "mruntime/safetensors.h"

using namespace mruntime;

// Paths to test model - assumes running from build directory
const char* MODEL_PATH = "../models/Qwen2.5-0.5B-Instruct/model.safetensors";
const char* CONFIG_PATH = "../models/Qwen2.5-0.5B-Instruct/config.json";

std::string read_file(const char* path) {
    std::ifstream file(path);
    assert(file.good());
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void test_config_parsing() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    assert(config.vocab_size == 151936);
    assert(config.hidden_size == 896);
    assert(config.num_layers == 24);
    assert(config.num_attention_heads == 14);
    assert(config.num_kv_heads == 2);
    assert(config.intermediate_size == 4864);
    assert(config.max_position_embeddings == 32768);
    assert(std::abs(config.rms_norm_eps - 1e-6f) < 1e-10f);
    assert(std::abs(config.rope_theta - 1000000.0f) < 1.0f);

    std::cout << "test_config_parsing PASSED\n";
}

void test_head_dim_calculation() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    size_t head_dim = config.head_dim();
    assert(head_dim == 896 / 14);  // hidden_size / num_attention_heads = 64

    std::cout << "test_head_dim_calculation PASSED\n";
}

void test_kv_cache_allocation() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    KVCache cache;
    size_t max_seq_len = 128;
    cache.allocate(config, max_seq_len, DType::FP32);

    assert(cache.key_cache.size() == config.num_layers);
    assert(cache.value_cache.size() == config.num_layers);
    assert(cache.seq_len == 0);

    // Check shapes: [1, num_kv_heads, max_seq_len, head_dim]
    for (size_t i = 0; i < config.num_layers; ++i) {
        assert(cache.key_cache[i].dim(0) == 1);
        assert(cache.key_cache[i].dim(1) == config.num_kv_heads);
        assert(cache.key_cache[i].dim(2) == max_seq_len);
        assert(cache.key_cache[i].dim(3) == config.head_dim());

        assert(cache.value_cache[i].dim(0) == 1);
        assert(cache.value_cache[i].dim(1) == config.num_kv_heads);
        assert(cache.value_cache[i].dim(2) == max_seq_len);
        assert(cache.value_cache[i].dim(3) == config.head_dim());
    }

    std::cout << "test_kv_cache_allocation PASSED\n";
}

void test_model_weight_loading() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(config);
    model.load_weights(*weights);

    // If we get here without exception, weights loaded successfully
    std::cout << "test_model_weight_loading PASSED\n";
}

void test_model_forward_shapes() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(config);
    model.load_weights(*weights);

    CpuBackend backend;
    KVCache cache;
    size_t max_seq_len = 128;
    cache.allocate(config, max_seq_len, DType::FP32);

    // Test with a small sequence
    std::vector<int> token_ids = {1, 2, 3, 4, 5};
    Tensor logits = model.forward(backend, token_ids, cache);

    // Check output shape: [batch=1, seq_len=5, vocab_size]
    assert(logits.ndim() == 3);
    assert(logits.dim(0) == 1);
    assert(logits.dim(1) == token_ids.size());
    assert(logits.dim(2) == config.vocab_size);
    assert(logits.dtype() == DType::FP32);

    // Check KV cache was updated
    assert(cache.seq_len == token_ids.size());

    std::cout << "test_model_forward_shapes PASSED\n";
}

void test_model_forward_values() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(config);
    model.load_weights(*weights);

    CpuBackend backend;
    KVCache cache;
    cache.allocate(config, 64, DType::FP32);

    std::vector<int> token_ids = {151643};  // BOS token
    Tensor logits = model.forward(backend, token_ids, cache);

    const float* logits_data = logits.data_ptr<float>();

    // Verify logits are finite and reasonable
    bool has_positive = false, has_negative = false;
    float max_val = -1e30f, min_val = 1e30f;
    for (size_t i = 0; i < config.vocab_size; ++i) {
        assert(std::isfinite(logits_data[i]));
        if (logits_data[i] > 0) has_positive = true;
        if (logits_data[i] < 0) has_negative = true;
        max_val = std::max(max_val, logits_data[i]);
        min_val = std::min(min_val, logits_data[i]);
    }

    assert(has_positive && has_negative);  // Should have varied outputs
    assert(max_val - min_val > 1.0f);      // Should have reasonable range

    std::cout << "test_model_forward_values PASSED (logit range: " << min_val << " to " << max_val << ")\n";
}

void test_incremental_decoding() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(config);
    model.load_weights(*weights);

    CpuBackend backend;
    KVCache cache;
    cache.allocate(config, 64, DType::FP32);

    // Prefill with prompt
    std::vector<int> prompt = {151643, 100, 200};
    Tensor logits1 = model.forward(backend, prompt, cache);
    assert(cache.seq_len == 3);

    // Decode one token
    std::vector<int> next_token = {300};
    Tensor logits2 = model.forward(backend, next_token, cache);
    assert(cache.seq_len == 4);

    // Output should be single token
    assert(logits2.dim(1) == 1);
    assert(logits2.dim(2) == config.vocab_size);

    // Decode another token
    std::vector<int> another_token = {400};
    Tensor logits3 = model.forward(backend, another_token, cache);
    assert(cache.seq_len == 5);

    std::cout << "test_incremental_decoding PASSED\n";
}

void test_tied_embeddings() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);

    // Qwen2.5-0.5B uses tied embeddings (no separate lm_head.weight)
    assert(!weights->has_tensor("lm_head.weight"));
    assert(weights->has_tensor("model.embed_tokens.weight"));

    // Model should still load and produce valid logits
    QwenModel model(config);
    model.load_weights(*weights);

    CpuBackend backend;
    KVCache cache;
    cache.allocate(config, 32, DType::FP32);

    std::vector<int> tokens = {151643};
    Tensor logits = model.forward(backend, tokens, cache);

    assert(logits.dim(2) == config.vocab_size);

    std::cout << "test_tied_embeddings PASSED\n";
}

int main() {
    std::cout << "Qwen Model Tests\n";
    std::cout << "================\n\n";

    test_config_parsing();
    test_head_dim_calculation();
    test_kv_cache_allocation();
    test_model_weight_loading();
    test_model_forward_shapes();
    test_model_forward_values();
    test_incremental_decoding();
    test_tied_embeddings();

    std::cout << "\nAll model tests passed!\n";
    return 0;
}
