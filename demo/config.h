#pragma once

#include "json.hpp"
using json = nlohmann::json;

template <typename T>
T getRequired(const json& j, const std::string& key) {
    if (j.contains(key)) return j.at(key);
    std::cerr << "Missing required setting: " << key << std::endl;
    abort();
}

template <typename T>
T getOptional(const json& j, const std::string& key, T defaultValue) {
    if (j.contains(key)) return j.at(key);
    return defaultValue;
}
