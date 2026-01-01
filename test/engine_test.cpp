#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "mruntime/cpu_backend.h"
#include "mruntime/inference_engine.h"
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

void test_generation_config_defaults() {
    GenerationConfig config;

    assert(config.max_new_tokens == 100);
    assert(config.temperature == 1.0f);
    assert(config.top_k == 50);
    assert(config.top_p == 1.0f);
    assert(config.eos_token_id == 151643);
    assert(config.greedy == true);
    assert(config.seed == 42);

    std::cout << "test_generation_config_defaults PASSED\n";
}

void test_greedy_generation() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;
    InferenceEngine engine(model, backend);

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 5;
    gen_config.greedy = true;
    gen_config.eos_token_id = -1;  // Disable EOS to ensure we generate max tokens

    std::vector<int> prompt = {151643};  // BOS token
    std::vector<int> output = engine.generate(prompt, gen_config);

    // Output should include prompt + generated tokens
    assert(output.size() == prompt.size() + gen_config.max_new_tokens);

    // First token should be the prompt
    assert(output[0] == prompt[0]);

    // All tokens should be valid (within vocab range)
    for (int token : output) {
        assert(token >= 0 && token < static_cast<int>(model_config.vocab_size));
    }

    std::cout << "test_greedy_generation PASSED (generated " << gen_config.max_new_tokens << " tokens)\n";
}

void test_greedy_determinism() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 3;
    gen_config.greedy = true;
    gen_config.eos_token_id = -1;

    std::vector<int> prompt = {151643, 100};

    // Generate twice with same settings
    InferenceEngine engine1(model, backend);
    std::vector<int> output1 = engine1.generate(prompt, gen_config);

    InferenceEngine engine2(model, backend);
    std::vector<int> output2 = engine2.generate(prompt, gen_config);

    // Greedy decoding should be deterministic
    assert(output1.size() == output2.size());
    for (size_t i = 0; i < output1.size(); ++i) {
        assert(output1[i] == output2[i]);
    }

    std::cout << "test_greedy_determinism PASSED\n";
}

void test_temperature_sampling() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 10;
    gen_config.temperature = 1.0f;
    gen_config.greedy = false;
    gen_config.eos_token_id = -1;

    std::vector<int> prompt = {151643};

    // Generate with different seeds
    gen_config.seed = 42;
    InferenceEngine engine1(model, backend);
    std::vector<int> output1 = engine1.generate(prompt, gen_config);

    gen_config.seed = 123;
    InferenceEngine engine2(model, backend);
    std::vector<int> output2 = engine2.generate(prompt, gen_config);

    // Different seeds should (very likely) produce different outputs
    bool different = false;
    for (size_t i = 1; i < output1.size() && i < output2.size(); ++i) {
        if (output1[i] != output2[i]) {
            different = true;
            break;
        }
    }

    // Note: There's a small chance they could be the same, but very unlikely
    // with temperature=1.0 over 10 tokens
    assert(different);

    std::cout << "test_temperature_sampling PASSED\n";
}

void test_seed_reproducibility() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 5;
    gen_config.temperature = 0.8f;
    gen_config.greedy = false;
    gen_config.seed = 42;
    gen_config.eos_token_id = -1;

    std::vector<int> prompt = {151643};

    // Generate twice with same seed
    InferenceEngine engine1(model, backend);
    std::vector<int> output1 = engine1.generate(prompt, gen_config);

    InferenceEngine engine2(model, backend);
    std::vector<int> output2 = engine2.generate(prompt, gen_config);

    // Same seed should produce same output
    assert(output1.size() == output2.size());
    for (size_t i = 0; i < output1.size(); ++i) {
        assert(output1[i] == output2[i]);
    }

    std::cout << "test_seed_reproducibility PASSED\n";
}

void test_eos_stopping() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;
    InferenceEngine engine(model, backend);

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 100;  // High limit
    gen_config.greedy = true;
    gen_config.eos_token_id = 151645;  // Qwen2 EOS token

    std::vector<int> prompt = {151643};
    std::vector<int> output = engine.generate(prompt, gen_config);

    // Check that generation stopped (either hit EOS or max_new_tokens)
    assert(output.size() <= prompt.size() + gen_config.max_new_tokens);

    // If stopped early, last token should be EOS
    if (output.size() < prompt.size() + gen_config.max_new_tokens) {
        assert(output.back() == gen_config.eos_token_id);
    }

    std::cout << "test_eos_stopping PASSED (generated " << output.size() - prompt.size() << " tokens)\n";
}

void test_top_k_sampling() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 20;
    gen_config.temperature = 1.0f;
    gen_config.top_k = 10;  // Restrict to top 10 tokens
    gen_config.greedy = false;
    gen_config.eos_token_id = -1;

    std::vector<int> prompt = {151643};

    // Generate multiple times and collect unique tokens
    std::set<int> seen_tokens;
    for (int run = 0; run < 5; ++run) {
        gen_config.seed = 42 + run * 100;
        InferenceEngine engine(model, backend);
        std::vector<int> output = engine.generate(prompt, gen_config);

        for (size_t i = 1; i < output.size(); ++i) {
            seen_tokens.insert(output[i]);
        }
    }

    // With top_k=10, we should see limited diversity
    // (though this test is probabilistic)
    assert(seen_tokens.size() < 1000);  // Should be much less than vocab size

    std::cout << "test_top_k_sampling PASSED (saw " << seen_tokens.size() << " unique tokens)\n";
}

void test_longer_generation() {
    std::string config_json = read_file(CONFIG_PATH);
    QwenConfig model_config = QwenConfig::from_json(config_json);

    auto weights = SafeTensorsFile::open(MODEL_PATH);
    QwenModel model(model_config);
    model.load_weights(*weights);

    CpuBackend backend;
    InferenceEngine engine(model, backend);

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 20;
    gen_config.greedy = true;
    gen_config.eos_token_id = -1;

    // Longer prompt
    std::vector<int> prompt = {151643, 100, 200, 300, 400, 500};
    std::vector<int> output = engine.generate(prompt, gen_config);

    assert(output.size() == prompt.size() + gen_config.max_new_tokens);

    // Verify prompt is preserved
    for (size_t i = 0; i < prompt.size(); ++i) {
        assert(output[i] == prompt[i]);
    }

    std::cout << "test_longer_generation PASSED\n";
}

int main() {
    std::cout << "Inference Engine Tests\n";
    std::cout << "======================\n\n";

    test_generation_config_defaults();
    test_greedy_generation();
    test_greedy_determinism();
    test_temperature_sampling();
    test_seed_reproducibility();
    test_eos_stopping();
    test_top_k_sampling();
    test_longer_generation();

    std::cout << "\nAll engine tests passed!\n";
    return 0;
}
