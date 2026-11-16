// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Optional SSD expert streaming: which architectures stream which tensors, and
// which checkpoints are admitted. One home for both decisions, shared by the
// native provider, the loader and the console activation. WinRT-free.
//
// Plan: docs/superpowers/plans/2026-11-16-moe-streaming-all-models.md.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Per-expert matrices that may be served from the expert cache. Everything else
// (shared experts, routers, expert biases and scales, norms) stays resident.
enum class MoeRole { gate_exps, up_exps, down_exps, gate_up_exps };

struct MoeArch {
    std::string_view arch; // GGUF general.architecture
    std::initializer_list<MoeRole> roles;
    bool has_shared_experts;
    bool has_expert_bias;
};

// nullptr when the architecture is not supported for streaming.
const MoeArch* moe_arch(std::string_view general_architecture);

// Role of a GGUF tensor named blk.<n>.<base>.weight, if <base> is a routed role
// of that architecture; nullopt for every other tensor or a malformed name.
std::optional<MoeRole> moe_routed_role(std::string_view arch, std::string_view tensor_name);

// True for a per-expert matrix (blk.<n>.ffn_*_exps.weight) that the table does
// not route for this architecture. The loader must refuse such a checkpoint
// rather than stream a partial set of expert tensors.
bool moe_is_unrouted_expert_tensor(std::string_view arch, std::string_view tensor_name);

// Identity of an admitted checkpoint, from the bundled catalogue.
struct MoePin {
    std::string id, filename, sha256, arch, template_sha256;
    uint64_t bytes = 0;
};

// What the loader observed on disk and in the GGUF header.
struct MoeObserved {
    std::string filename, sha256, arch, template_sha256;
    uint64_t bytes = 0;
};

struct MoeAdmission {
    bool ok = false;
    std::string reason; // names the first mismatching field and both values
};

MoeAdmission check_moe_checkpoint(const MoePin& pin, const MoeObserved& got);

struct MoeCatalogueFile {
    std::string filename, sha256;
    uint64_t bytes = 0;
};

// Build a pin from a catalogue entry's fields. Entries read from the LocalState
// override never admit (from_override = true returns nullopt).
std::optional<MoePin> moe_pin_from_catalogue(std::string_view id, std::string_view arch,
                                             std::string_view template_sha256,
                                             const std::vector<MoeCatalogueFile>& files,
                                             bool from_override);

} // namespace xllama
