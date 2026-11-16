// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/moe_streaming.h"

#include <algorithm>

namespace xllama {

namespace {

using R = MoeRole;

// Verified against llama.cpp d812350 src/models/*.cpp: the per-expert tensors
// each architecture creates and feeds to build_moe_ffn / MUL_MAT_ID.
const MoeArch kArchs[] = {
    {"lfm2moe", {R::gate_exps, R::up_exps, R::down_exps}, false, false},
    {"bailingmoe2", {R::gate_exps, R::up_exps, R::down_exps}, true, false},
    {"ernie4_5-moe", {R::gate_exps, R::up_exps, R::down_exps}, true, false},
    {"smallthinker", {R::gate_exps, R::up_exps, R::down_exps}, false, false},
    {"gpt-oss", {R::gate_exps, R::up_exps, R::down_exps}, false, true},
    {"gemma4", {R::gate_exps, R::up_exps, R::down_exps, R::gate_up_exps}, false, false},
    {"afmoe", {R::gate_exps, R::up_exps, R::down_exps}, true, false},
    {"maple", {R::gate_exps, R::up_exps, R::down_exps}, false, false},
};

struct RoleName {
    MoeRole role;
    std::string_view base;
};
const RoleName kRoleNames[] = {
    {R::gate_exps, "ffn_gate_exps"},
    {R::up_exps, "ffn_up_exps"},
    {R::down_exps, "ffn_down_exps"},
    {R::gate_up_exps, "ffn_gate_up_exps"},
};

// "blk.<digits>.<base>.weight" -> <base>; empty when the name has another shape.
std::string_view block_weight_base(std::string_view name) {
    constexpr std::string_view prefix = "blk.";
    constexpr std::string_view suffix = ".weight";
    if (name.size() <= prefix.size() + suffix.size() || name.substr(0, prefix.size()) != prefix ||
        name.substr(name.size() - suffix.size()) != suffix)
        return {};
    std::string_view rest = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    const size_t dot = rest.find('.');
    if (dot == 0 || dot == std::string_view::npos)
        return {};
    for (size_t i = 0; i < dot; ++i)
        if (rest[i] < '0' || rest[i] > '9')
            return {};
    std::string_view base = rest.substr(dot + 1);
    if (base.empty() || base.find('.') != std::string_view::npos)
        return {};
    return base;
}

bool is_lower_hex64(std::string_view s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

bool ends_with(std::string_view s, std::string_view tail) {
    return s.size() >= tail.size() && s.substr(s.size() - tail.size()) == tail;
}

std::string mismatch(const char* field, const std::string& want, const std::string& got) {
    return std::string(field) + " mismatch: expected " + want + ", got " + got;
}

} // namespace

const MoeArch* moe_arch(std::string_view general_architecture) {
    for (const auto& a : kArchs)
        if (a.arch == general_architecture)
            return &a;
    return nullptr;
}

std::optional<MoeRole> moe_routed_role(std::string_view arch, std::string_view tensor_name) {
    const MoeArch* a = moe_arch(arch);
    if (!a)
        return std::nullopt;
    const std::string_view base = block_weight_base(tensor_name);
    for (const auto& rn : kRoleNames) {
        if (rn.base != base)
            continue;
        if (std::find(a->roles.begin(), a->roles.end(), rn.role) != a->roles.end())
            return rn.role;
        return std::nullopt;
    }
    return std::nullopt;
}

bool moe_is_unrouted_expert_tensor(std::string_view arch, std::string_view tensor_name) {
    const std::string_view base = block_weight_base(tensor_name);
    if (base.substr(0, 4) != "ffn_" || !ends_with(base, "_exps"))
        return false;
    return !moe_routed_role(arch, tensor_name).has_value();
}

MoeAdmission check_moe_checkpoint(const MoePin& pin, const MoeObserved& got) {
    MoeAdmission a;
    if (!moe_arch(pin.arch)) {
        a.reason = "arch " + pin.arch + " is not supported for expert streaming";
        return a;
    }
    if (got.filename != pin.filename)
        a.reason = mismatch("filename", pin.filename, got.filename);
    else if (got.bytes != pin.bytes)
        a.reason = mismatch("bytes", std::to_string(pin.bytes), std::to_string(got.bytes));
    else if (got.sha256 != pin.sha256)
        a.reason = mismatch("sha256", pin.sha256, got.sha256);
    else if (got.arch != pin.arch)
        a.reason = mismatch("arch", pin.arch, got.arch);
    else if (got.template_sha256 != pin.template_sha256)
        a.reason = mismatch("template_sha256", pin.template_sha256, got.template_sha256);
    else
        a.ok = true;
    return a;
}

std::optional<MoePin> moe_pin_from_catalogue(std::string_view id, std::string_view arch,
                                             std::string_view template_sha256,
                                             const std::vector<MoeCatalogueFile>& files,
                                             bool from_override) {
    if (from_override || id.empty() || arch.empty() || !is_lower_hex64(template_sha256))
        return std::nullopt;
    const MoeCatalogueFile* gguf = nullptr;
    for (const auto& f : files) {
        if (!ends_with(f.filename, ".gguf"))
            continue;
        if (gguf)
            return std::nullopt;
        gguf = &f;
    }
    if (!gguf || !is_lower_hex64(gguf->sha256) || gguf->bytes == 0)
        return std::nullopt;
    MoePin p;
    p.id = std::string(id);
    p.filename = gguf->filename;
    p.bytes = gguf->bytes;
    p.sha256 = gguf->sha256;
    p.arch = std::string(arch);
    p.template_sha256 = std::string(template_sha256);
    return p;
}

} // namespace xllama
