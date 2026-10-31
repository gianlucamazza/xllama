// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Per-file catalogue fetch plan.
//
// A directory is not compared to the manifest as one blob. Each files[] pin is
// checked on its own. `classify_file_pin` is the single decision: Ok (keep),
// Refetch (HTTP), or Unverified (keep the bytes, do not publish `.complete`).
// Installed-model verification uses only weight and data pins. EnsureModel and
// routing classify a size-only listing; DownloadAsync is what hashes. The
// LocalState `.complete` marker attests every pin was hashed at
// download/promotion. Adding, removing, or changing a licence or notice pin
// must not mark already-matching weight pins as stale; those sidecars fetch
// on their own.
//
// Which pins are weights is data-driven from `files[].role`. Documented
// default when `role` is omitted: a basename stem of license / licence /
// notice / readme / copying / authors / copyright (optional .txt/.md/.rst)
// is a sidecar; everything else is a weight. Set `role` to `weight` or
// `data`, or to `licence` / `license` / `notice` / `sidecar`, to override.
//
// Pure (no WinRT / no pch.h). Host doctest-testable.
#pragma once

#include "xllama/model_provision.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Hash-unavailable retries in the UWP downloader. Not a host or console pin.
inline constexpr int kHashVerifyRetries = 2;
inline constexpr int kHashVerifyRetryPauseS = 1;

enum class PinCheck {
    Ok = 0,         // size (and hash, when observed) match — do not fetch
    Refetch = 1,    // missing, truncated, or hash mismatch — HTTP-fetch
    Unverified = 2, // size matches but the hash could not be read — keep, no marker
};

// Catalogue files[].role. Unspecified applies the documented filename default.
enum class CatalogueFileRole {
    Unspecified = 0,
    Weight = 1,
    Licence = 2,
    Notice = 3,
    Sidecar = 4,
};

// One catalogue files[] pin. `approx_bytes` 0 and an empty `sha256` mean "not
// known" — those axes do not reject a present file. `role` is the manifest
// field; Unspecified means "use the documented default".
struct ManifestFilePin {
    std::wstring filename;
    uint64_t approx_bytes = 0;
    std::string sha256;
    CatalogueFileRole role = CatalogueFileRole::Unspecified;
};

// What is on disk for one relative name. An empty `sha256` is a size-only
// observation (the usual listing); a pin with a hash still matches on size.
struct PresentFile {
    std::wstring filename;
    uint64_t size = 0;
    std::string sha256;           // empty unless a hash was computed
    bool hash_unreadable = false; // size matches, hash attempted and failed
};

inline std::wstring file_basename_normalized(std::wstring path) {
    path = normalize_model_path(std::move(path));
    const auto pos = path.find_last_of(L'/');
    if (pos != std::wstring::npos)
        path = path.substr(pos + 1);
    return path;
}

inline std::wstring strip_sidecar_text_ext(std::wstring name) {
    const std::wstring exts[] = {L".txt", L".md", L".rst"};
    for (const auto& ext : exts) {
        if (name.size() > ext.size() &&
            name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
            return name.substr(0, name.size() - ext.size());
    }
    return name;
}

inline std::string ascii_lower_copy(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    return out;
}

// Parse files[].role. Empty → Unspecified (documented filename default).
// weight/data → Weight. licence/license → Licence. notice → Notice.
// sidecar → Sidecar (generic non-weight). Any other explicit value → Weight
// so a typo cannot skip verification.
inline CatalogueFileRole parse_catalogue_file_role(std::string_view raw) {
    const std::string role = ascii_lower_copy(raw);
    if (role.empty())
        return CatalogueFileRole::Unspecified;
    if (role == "weight" || role == "data")
        return CatalogueFileRole::Weight;
    if (role == "licence" || role == "license")
        return CatalogueFileRole::Licence;
    if (role == "notice")
        return CatalogueFileRole::Notice;
    if (role == "sidecar")
        return CatalogueFileRole::Sidecar;
    return CatalogueFileRole::Weight;
}

inline ManifestFilePin make_file_pin(std::wstring filename, uint64_t bytes, std::string sha256,
                                     std::string_view role = {}) {
    ManifestFilePin pin;
    pin.filename = std::move(filename);
    pin.approx_bytes = bytes;
    pin.sha256 = std::move(sha256);
    pin.role = parse_catalogue_file_role(role);
    return pin;
}

// Documented default: licence / notice family by basename stem. Not a weight,
// tokenizer, or ONNX graph. Portable — not tied to a catalogue id or host path.
inline bool is_catalogue_sidecar_name(const std::wstring& filename) {
    const std::wstring stem = strip_sidecar_text_ext(file_basename_normalized(filename));
    return stem == L"license" || stem == L"licence" || stem == L"notice" || stem == L"readme" ||
           stem == L"copying" || stem == L"authors" || stem == L"copyright";
}

inline bool is_catalogue_sidecar(const std::wstring& filename) {
    return is_catalogue_sidecar_name(filename);
}

// Manifest-driven: explicit role wins; Unspecified uses the documented default.
inline bool is_catalogue_sidecar(const ManifestFilePin& pin) {
    switch (pin.role) {
    case CatalogueFileRole::Licence:
    case CatalogueFileRole::Notice:
    case CatalogueFileRole::Sidecar:
        return true;
    case CatalogueFileRole::Weight:
        return false;
    case CatalogueFileRole::Unspecified:
        return is_catalogue_sidecar_name(pin.filename);
    }
    return is_catalogue_sidecar_name(pin.filename);
}

inline bool is_weight_pin(const ManifestFilePin& pin) {
    return !is_catalogue_sidecar(pin);
}

inline bool sha256_hex_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        auto fold = [](char c) {
            return (c >= 'A' && c <= 'F') ? static_cast<char>(c - 'A' + 'a') : c;
        };
        if (fold(a[i]) != fold(b[i]))
            return false;
    }
    return true;
}

// Classify one on-disk observation against a catalogue pin.
// `have == nullptr` or size 0 is missing → Refetch. A pin size of 0 does not
// reject. A pin hash is compared only when the observation has a hash; a
// size-only listing (empty sha256, hash_unreadable false) is Ok on size.
// hash_unreadable → Unverified (never Refetch: do not replace good-looking bytes).
inline PinCheck classify_file_pin(const PresentFile* have, const ManifestFilePin& pin) {
    if (!have || have->size == 0)
        return PinCheck::Refetch;
    if (normalize_model_path(have->filename) != normalize_model_path(pin.filename))
        return PinCheck::Refetch;
    if (pin.approx_bytes != 0 && have->size != pin.approx_bytes)
        return PinCheck::Refetch;
    if (have->hash_unreadable)
        return pin.sha256.empty() ? PinCheck::Ok : PinCheck::Unverified;
    if (!pin.sha256.empty() && !have->sha256.empty() && !sha256_hex_equal(have->sha256, pin.sha256))
        return PinCheck::Refetch;
    return PinCheck::Ok;
}

inline PinCheck classify_file_pin(const PresentFile& have, const ManifestFilePin& pin) {
    return classify_file_pin(&have, pin);
}

inline bool file_pin_satisfied(const PresentFile& have, const ManifestFilePin& pin) {
    return classify_file_pin(&have, pin) == PinCheck::Ok;
}

inline const PresentFile* find_present_file(const std::vector<PresentFile>& present,
                                            const std::wstring& filename) {
    const std::wstring want = normalize_model_path(filename);
    for (const auto& p : present) {
        if (normalize_model_path(p.filename) == want)
            return &p;
    }
    return nullptr;
}

// Filenames to HTTP-fetch, in catalogue order. Refetch only — Unverified is kept.
inline std::vector<std::wstring> files_needing_fetch(const std::vector<PresentFile>& present,
                                                     const std::vector<ManifestFilePin>& expected) {
    std::vector<std::wstring> need;
    need.reserve(expected.size());
    for (const auto& pin : expected) {
        const PresentFile* have = find_present_file(present, pin.filename);
        if (classify_file_pin(have, pin) == PinCheck::Refetch)
            need.push_back(pin.filename);
    }
    return need;
}

// Same as files_needing_fetch, excluding catalogue sidecars (role or default).
inline std::vector<std::wstring>
weight_files_needing_fetch(const std::vector<PresentFile>& present,
                           const std::vector<ManifestFilePin>& expected) {
    std::vector<std::wstring> need;
    for (const auto& pin : expected) {
        if (!is_weight_pin(pin))
            continue;
        const PresentFile* have = find_present_file(present, pin.filename);
        if (classify_file_pin(have, pin) == PinCheck::Refetch)
            need.push_back(pin.filename);
    }
    return need;
}

// Every weight/data pin classifies Ok (size-only listings count). Sidecars
// are ignored. Empty expected, or a catalogue with no weight pins, is false:
// there is nothing to load. This is the installed-model verification predicate
// (#370): licence / notice rows do not participate.
inline bool weight_pins_verified(const std::vector<PresentFile>& present,
                                 const std::vector<ManifestFilePin>& expected) {
    bool any_weight = false;
    for (const auto& pin : expected) {
        if (!is_weight_pin(pin))
            continue;
        any_weight = true;
        const PresentFile* have = find_present_file(present, pin.filename);
        if (classify_file_pin(have, pin) != PinCheck::Ok)
            return false;
    }
    return any_weight;
}

// Weights match their pins (size-only listings count); at least one sidecar
// is missing or size-mismatched. The chat model is usable; fetch sidecars
// in the background without blocking Run.
inline bool only_sidecars_missing(const std::vector<PresentFile>& present,
                                  const std::vector<ManifestFilePin>& expected) {
    if (expected.empty())
        return false;
    bool sidecar_gap = false;
    for (const auto& pin : expected) {
        const PresentFile* have = find_present_file(present, pin.filename);
        const PinCheck c = classify_file_pin(have, pin);
        if (is_catalogue_sidecar(pin)) {
            if (c == PinCheck::Refetch)
                sidecar_gap = true;
        } else if (c != PinCheck::Ok) {
            return false;
        }
    }
    return sidecar_gap;
}

// GPU (and any !set_app_ready tree) is not done when names alone match.
// A LocalState folder with every catalogue filename but no `.complete` still
// needs a write: hash Ok weights, fetch gaps, then publish the marker. That
// is how a tree recovers after begin() dropped the marker, an Unverified
// hash, or a WDP upload (#370 B4).
inline bool gpu_needs_provision(bool names_present, bool marker) {
    return !names_present || !marker;
}

// Routing-time gate: size-only weight pins AND the LocalState `.complete`
// marker. A size-only listing cannot see a same-size hash mismatch; the
// marker is published only after DownloadAsync hashed every pin, so it
// refuses Unverified / mid-write / mixed trees (#370 B1).
inline bool gpu_tree_ready_for_routing(bool weight_pins_ok, bool complete_marker) {
    return weight_pins_ok && complete_marker;
}

// EnsureModelNamedAsync Check 1 and the sidecar fast path. A background GPU
// provision (!set_app_ready) with no marker must skip both and run
// DownloadAsync(entry->files) so hashes run and complete() can fire.
inline bool skip_provision_early_exits(bool set_app_ready, bool marker) {
    return !set_app_ready && !marker;
}

// Sidecar-ready background write: if `.complete` was already published,
// fetch sidecars only. If not, pass every catalogue file so weights are
// hashed and the marker is written only when every pin is Ok (#370 B5).
inline bool sidecar_write_needs_full_hash(bool had_marker) {
    return !had_marker;
}

// Catalogue filenames a sidecar-ready background write should pass to the
// downloader. had_marker → sidecars only; otherwise the full expected set.
inline std::vector<std::wstring>
sidecar_background_write_files(const std::vector<ManifestFilePin>& expected, bool had_marker) {
    std::vector<std::wstring> out;
    out.reserve(expected.size());
    for (const auto& pin : expected) {
        if (sidecar_write_needs_full_hash(had_marker) || is_catalogue_sidecar(pin))
            out.push_back(pin.filename);
    }
    return out;
}

} // namespace xllama
