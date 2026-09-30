// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "floppylm_internal.h"
#include "xllama/floppylm.h"
namespace xllama {
std::string floppylm_probe(const std::string& request) {
    using namespace floppy;
    auto j = parse_json(request);
    Model m(j);
    if (j.value("codec_only", false))
        return Json{{"artifact", pack(m)}}.dump();
    if (j.contains("optimizer_steps")) {
        Json states = Json::array();
        int step = 0;
        for (const auto& input : j.at("optimizer_steps")) {
            std::vector<std::vector<float>> gradients;
            for (const auto& t : m.tensors)
                gradients.push_back(input.at("gradients").at(t.name).get<std::vector<float>>());
            adam(m, gradients, ++step, input.at("lr"), input.at("wd"));
            states.push_back(m.state());
        }
        return Json{{"states", states}}.dump();
    }
    Engine e(m.cfg, j.at("batch"), j.at("length"), 1, true);
    auto s =
        e.evaluate(m, j.at("x").get<std::vector<int32_t>>(), j.at("y").get<std::vector<int32_t>>());
    Json o = {{"loss", s.loss}, {"logits", s.logits}, {"grads", Json::object()}};
    for (size_t i = 0; i < m.tensors.size(); ++i)
        o["grads"][m.tensors[i].name] = s.grads[i];
    o["initial_artifact"] = pack(m);
    adam(m, s.grads, 1, .003f, .1f);
    o["updated"] = m.state();
    auto artifact = pack(m);
    o["artifact"] = artifact;
    auto frozen = unpack(artifact);
    if (pack(frozen) != artifact)
        throw std::runtime_error("roundtrip failed");
    Engine inference(m.cfg, j.at("batch"), j.at("length"), 1, false);
    o["saved_logits"] = inference
                            .evaluate(frozen, j.at("x").get<std::vector<int32_t>>(),
                                      j.at("y").get<std::vector<int32_t>>())
                            .logits;
    return o.dump();
}
} // namespace xllama
