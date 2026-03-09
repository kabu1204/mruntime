#include "mruntime/qwen_config.h"

#include <stdexcept>
#include <nlohmann/json.hpp>

namespace mruntime {

QwenConfig QwenConfig::from_json(const std::string& json_str) {
    QwenConfig cfg;

    try {
        auto json = nlohmann::json::parse(json_str);

        if (json.contains("vocab_size")) {
            cfg.vocab_size = json["vocab_size"].get<size_t>();
        }
        if (json.contains("hidden_size")) {
            cfg.hidden_size = json["hidden_size"].get<size_t>();
        }
        if (json.contains("num_hidden_layers")) {
            cfg.num_layers = json["num_hidden_layers"].get<size_t>();
        }
        if (json.contains("num_attention_heads")) {
            cfg.num_attention_heads = json["num_attention_heads"].get<size_t>();
        }
        if (json.contains("num_key_value_heads")) {
            cfg.num_kv_heads = json["num_key_value_heads"].get<size_t>();
        }
        if (json.contains("intermediate_size")) {
            cfg.intermediate_size = json["intermediate_size"].get<size_t>();
        }
        if (json.contains("max_position_embeddings")) {
            cfg.max_position_embeddings = json["max_position_embeddings"].get<size_t>();
        }
        if (json.contains("rms_norm_eps")) {
            cfg.rms_norm_eps = json["rms_norm_eps"].get<float>();
        }
        if (json.contains("rope_theta")) {
            cfg.rope_theta = json["rope_theta"].get<float>();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse QwenConfig JSON: ") + e.what());
    }

    return cfg;
}

}  // namespace mruntime
