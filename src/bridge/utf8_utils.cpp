#include "xllama/utf8_utils.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace xllama {

#ifdef _WIN32

std::wstring utf8_to_wstring(const std::string& s) {
    if (s.empty()) return {};
    int sz = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (sz <= 0) return {};
    std::wstring r(sz, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, r.data(), sz);
    if (!r.empty() && r.back() == L'\0') r.pop_back();
    return r;
}

std::string wstring_to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int sz = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (sz <= 0) return {};
    std::string r(sz, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, r.data(), sz, nullptr, nullptr);
    if (!r.empty() && r.back() == '\0') r.pop_back();
    return r;
}

#endif // _WIN32

} // namespace xllama
