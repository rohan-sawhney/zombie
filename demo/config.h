// This file contains helper functions for reading settings from a JSON file.

#pragma once

#include "json.hpp"
using json = nlohmann::json;

template <typename T>
T getRequired(const json& j, const std::string& key) {
    if (j.contains(key)) return j.at(key);
    std::cerr << "Missing required setting: " << key << std::endl;
    exit(EXIT_FAILURE);
}

template <typename T>
T getOptional(const json& j, const std::string& key, T defaultValue) {
    if (j.contains(key)) return j.at(key);
    return defaultValue;
}
