// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// HTTP status for an exception that escapes a LAN API handler (#591).
// WinRT-free: the UWP server extracts the HRESULT from winrt::hresult_error and
// passes it here; std exceptions are classified directly.
//
// Only a Windows.Data.Json accessor failure (a field of the wrong type, a
// missing key, an unparseable value) is the client's fault. Out of memory is
// the server's capacity, so 503 like a busy compute slot. Anything else is an
// internal error, never a "malformed request".

#pragma once

#include <cstdint>
#include <exception>
#include <new>
#include <string>

namespace xllama {

enum class ApiFailure { MalformedRequest, OutOfMemory, Internal };

// winerror.h values, spelled out so the host build needs no Windows SDK.
inline constexpr std::int32_t kHrIllegalMethodCall = static_cast<std::int32_t>(0x8000000Eu);
inline constexpr std::int32_t kHrOutOfMemory = static_cast<std::int32_t>(0x8007000Eu);
inline constexpr std::int32_t kHrInvalidJsonString = static_cast<std::int32_t>(0x83750007u);
inline constexpr std::int32_t kHrInvalidJsonNumber = static_cast<std::int32_t>(0x83750008u);
inline constexpr std::int32_t kHrJsonValueNotFound = static_cast<std::int32_t>(0x83750009u);

// E_ILLEGAL_METHOD_CALL is what JsonValue/JsonObject getters throw on a value
// of another type; the WEB_E_* codes are the Json parser and key lookup.
inline ApiFailure classify_api_hresult(std::int32_t hr) noexcept {
    switch (hr) {
    case kHrIllegalMethodCall:
    case kHrJsonValueNotFound:
    case kHrInvalidJsonString:
    case kHrInvalidJsonNumber:
        return ApiFailure::MalformedRequest;
    case kHrOutOfMemory:
        return ApiFailure::OutOfMemory;
    default:
        return ApiFailure::Internal;
    }
}

// A non-WinRT exception. |detail| receives text for the log, not the client.
inline ApiFailure classify_std_exception(const std::exception_ptr& e, std::string& detail) {
    detail = "unknown exception";
    if (!e)
        return ApiFailure::Internal;
    try {
        std::rethrow_exception(e);
    } catch (const std::bad_alloc& ex) {
        detail = ex.what();
        return ApiFailure::OutOfMemory;
    } catch (const std::exception& ex) {
        detail = ex.what();
        return ApiFailure::Internal;
    } catch (...) {
        return ApiFailure::Internal;
    }
}

struct ApiFailureReply {
    int http_code;
    const char* status;
    const char* message;
};

// |malformed| and |internal| are the route's own client-facing messages.
inline ApiFailureReply api_failure_reply(ApiFailure kind, const char* malformed,
                                         const char* internal) noexcept {
    switch (kind) {
    case ApiFailure::MalformedRequest:
        return {400, "400 Bad Request", malformed};
    case ApiFailure::OutOfMemory:
        return {503, "503 Service Unavailable", "out of memory"};
    case ApiFailure::Internal:
    default:
        return {500, "500 Internal Server Error", internal};
    }
}

inline const char* api_failure_name(ApiFailure kind) noexcept {
    switch (kind) {
    case ApiFailure::MalformedRequest:
        return "malformed request";
    case ApiFailure::OutOfMemory:
        return "out of memory";
    case ApiFailure::Internal:
    default:
        return "internal error";
    }
}

} // namespace xllama
