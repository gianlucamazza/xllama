// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once
#include "nlohmann/json.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace xllama::floppy {
using Json = nlohmann::json;
using Bytes = std::vector<uint8_t>;
Json parse_json(const std::string& text);
struct Config {
    int vocab = 256, d = 96, layers = 3, heads = 6, ff = 384, ctx = 256;
    std::string mlp = "gelu", core = "ternary", emb = "4bit", policy = "row16";
    double delta = 0.5;
    bool qknorm = false;
    explicit Config(const Json& j);
    Json json() const;
};
struct Tensor {
    std::string name;
    int rows, cols;
    bool norm;
    std::vector<float> w, m, v;
};
struct Quant {
    std::vector<float> values;
    std::vector<uint16_t> symbols;
    Bytes scales;
    int levels;
};
struct Model {
    Config cfg;
    std::vector<Tensor> tensors;
    Bytes canonical;
    Json frozen_config;
    std::vector<std::vector<float>> snapshot;
    void verify_frozen() const;
    explicit Model(const Json& bundle);
    Json state() const;
    void restore(const Json& j);
};
Quant quantize(const Tensor& t, const Config& cfg, bool embedding);
Bytes pack(const Model& m);
Json artifact_sections(const Bytes& b);
Model unpack(const Bytes& b);
float half_round(float x);
struct Step {
    float loss;
    std::vector<float> logits;
    std::vector<std::vector<float>> grads;
};
class Engine {
  public:
    Engine(const Config& c, int batch, int length, int threads, bool backward);
    ~Engine();
    Step evaluate(const Model& m, const std::vector<int32_t>& x, const std::vector<int32_t>& y);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
void adam(Model& m, const std::vector<std::vector<float>>& grad, int step, float lr, float wd);
Bytes read_bytes(const std::string& path);
void write_bytes(const std::string& path, const Bytes& b);
} // namespace xllama::floppy
