// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Image-dialog credit for Stability AI Community License §IV.a.
//
// The caption is the licence credit only. It is not a claim that the app or
// project is registered with, partnered with, or endorsed by Stability AI.
#pragma once

#include <string_view>

namespace xllama {

// Exact §IV.a line. Shared by the Image dialog and host tests.
inline constexpr const char* kStabilityAttribution = "Powered by Stability AI";

// Mapping (licence first, then model-id family):
//
// 1. If `licence` is non-empty, show the credit iff it names the Stability AI
//    Community License / Licence (ASCII, case-insensitive). A different
//    licence hides the line even when the id looks like SD-Turbo, so a future
//    MIT/Apache image model can reuse an sd-* id without a false credit.
//    Today's catalogue JSON has no licence field, so callers pass empty and
//    this branch is unused until an optional field appears.
// 2. If `licence` is empty, fall back to the shipping SD-Turbo id family:
//    the id starts with "sd-turbo" (ASCII, case-insensitive) and is either
//    exactly that or continues with '-' / '_' (sd-turbo-fp16 today).
//    "sd-turbocharged" and a prefix buried mid-id do not match.
//
// TAESD is MIT and is not a catalogue model; it overlays the SD-Turbo VAE.
// The selected/used image model remains sd-turbo-fp16, so the credit still
// shows when the TAESD toggle is on.
namespace stability_attribution_detail {

inline constexpr char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline constexpr bool ascii_istarts_with(std::string_view s, std::string_view prefix) {
    if (s.size() < prefix.size())
        return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (ascii_lower(s[i]) != ascii_lower(prefix[i]))
            return false;
    }
    return true;
}

inline constexpr bool ascii_icontains(std::string_view hay, std::string_view needle) {
    if (needle.empty())
        return true;
    if (hay.size() < needle.size())
        return false;
    for (size_t i = 0; i + needle.size() <= hay.size(); ++i) {
        bool ok = true;
        for (size_t j = 0; j < needle.size(); ++j) {
            if (ascii_lower(hay[i + j]) != ascii_lower(needle[j])) {
                ok = false;
                break;
            }
        }
        if (ok)
            return true;
    }
    return false;
}

inline constexpr bool licence_is_stability_community(std::string_view licence) {
    return ascii_icontains(licence, "stability ai community license") ||
           ascii_icontains(licence, "stability ai community licence");
}

inline constexpr bool stability_model_id_family(std::string_view id) {
    constexpr std::string_view kFamily = "sd-turbo";
    if (!ascii_istarts_with(id, kFamily))
        return false;
    if (id.size() == kFamily.size())
        return true;
    const char next = id[kFamily.size()];
    return next == '-' || next == '_';
}

} // namespace stability_attribution_detail

inline constexpr bool show_stability_attribution(std::string_view model_id,
                                                 std::string_view licence = {}) {
    if (!licence.empty())
        return stability_attribution_detail::licence_is_stability_community(licence);
    return stability_attribution_detail::stability_model_id_family(model_id);
}

} // namespace xllama
