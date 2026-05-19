// ggml-backend-dl-stub.cpp
// UWP replacement for ggml/src/ggml-backend-dl.cpp.
// LoadLibraryW is blocked in WINAPI_FAMILY_APP; dynamic backend loading
// is disabled. All three functions return "not found" / nullptr.
// ggml-backend-reg.cpp calls these only when scanning for DL backends,
// which is unreachable with GGML_USE_CPU compile-time selection.

#ifdef XLLAMA_UWP

#include "../llama.cpp/ggml/src/ggml-backend-dl.h"

dl_handle * dl_load_library(const fs::path & /*path*/) {
    return nullptr;
}

void * dl_get_sym(dl_handle * /*handle*/, const char * /*name*/) {
    return nullptr;
}

const char * dl_error() {
    return "dynamic backend loading not supported in UWP";
}

#endif // XLLAMA_UWP
