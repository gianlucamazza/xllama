// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_UWP

    #include <string>

namespace xllama {

struct ManifestTrust {
    bool trusted = false;
    std::string catalogue_version;
    std::string key_id;
    std::string error;
};

} // namespace xllama

#endif // XLLAMA_UWP
