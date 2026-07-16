// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#ifdef XLLAMA_USE_ORT

    #include "ort_genai_c.h"

    #include <memory>
    #include <stdexcept>
    #include <string>

namespace xllama {

struct OgaModelDeleter {
    void operator()(OgaModel* p) const noexcept {
        OgaDestroyModel(p);
    }
};
struct OgaTokenizerDeleter {
    void operator()(OgaTokenizer* p) const noexcept {
        OgaDestroyTokenizer(p);
    }
};
struct OgaTokenizerStreamDeleter {
    void operator()(OgaTokenizerStream* p) const noexcept {
        OgaDestroyTokenizerStream(p);
    }
};
struct OgaGeneratorParamsDeleter {
    void operator()(OgaGeneratorParams* p) const noexcept {
        OgaDestroyGeneratorParams(p);
    }
};
struct OgaGeneratorDeleter {
    void operator()(OgaGenerator* p) const noexcept {
        OgaDestroyGenerator(p);
    }
};
struct OgaSequencesDeleter {
    void operator()(OgaSequences* p) const noexcept {
        OgaDestroySequences(p);
    }
};
struct OgaResultDeleter {
    void operator()(OgaResult* p) const noexcept {
        OgaDestroyResult(p);
    }
};
struct OgaTensorDeleter {
    void operator()(OgaTensor* p) const noexcept {
        OgaDestroyTensor(p);
    }
};

using OgaModelPtr = std::unique_ptr<OgaModel, OgaModelDeleter>;
using OgaTokenizerPtr = std::unique_ptr<OgaTokenizer, OgaTokenizerDeleter>;
using OgaTokenizerStreamPtr = std::unique_ptr<OgaTokenizerStream, OgaTokenizerStreamDeleter>;
using OgaGeneratorParamsPtr = std::unique_ptr<OgaGeneratorParams, OgaGeneratorParamsDeleter>;
using OgaGeneratorPtr = std::unique_ptr<OgaGenerator, OgaGeneratorDeleter>;
using OgaSequencesPtr = std::unique_ptr<OgaSequences, OgaSequencesDeleter>;
using OgaResultPtr = std::unique_ptr<OgaResult, OgaResultDeleter>;
using OgaTensorPtr = std::unique_ptr<OgaTensor, OgaTensorDeleter>;

// Throw std::runtime_error with OgaResult error message if err != nullptr.
inline void oga_check(OgaResult* err, const char* context) {
    if (!err)
        return;
    std::string msg = context;
    msg += ": ";
    msg += OgaResultGetError(err);
    OgaDestroyResult(err);
    throw std::runtime_error(msg);
}

} // namespace xllama

#endif // XLLAMA_USE_ORT
