// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/api_failure.h"

#include <doctest/doctest.h>
#include <new>
#include <stdexcept>
#include <string>

using xllama::api_failure_reply;
using xllama::ApiFailure;
using xllama::classify_api_hresult;
using xllama::classify_std_exception;

TEST_CASE("api failure: JSON accessor HRESULTs are a malformed request") {
    CHECK(classify_api_hresult(xllama::kHrIllegalMethodCall) == ApiFailure::MalformedRequest);
    CHECK(classify_api_hresult(xllama::kHrJsonValueNotFound) == ApiFailure::MalformedRequest);
    CHECK(classify_api_hresult(xllama::kHrInvalidJsonString) == ApiFailure::MalformedRequest);
    CHECK(classify_api_hresult(xllama::kHrInvalidJsonNumber) == ApiFailure::MalformedRequest);
}

TEST_CASE("api failure: E_OUTOFMEMORY is out of memory, any other HRESULT is internal") {
    CHECK(classify_api_hresult(xllama::kHrOutOfMemory) == ApiFailure::OutOfMemory);
    CHECK(classify_api_hresult(static_cast<std::int32_t>(0x80004005u)) == ApiFailure::Internal);
    CHECK(classify_api_hresult(static_cast<std::int32_t>(0x8000000Bu)) == ApiFailure::Internal);
}

TEST_CASE("api failure: std exceptions never become a malformed request") {
    std::string detail;
    CHECK(classify_std_exception(std::make_exception_ptr(std::bad_alloc()), detail) ==
          ApiFailure::OutOfMemory);
    CHECK(!detail.empty());

    detail.clear();
    CHECK(classify_std_exception(std::make_exception_ptr(std::runtime_error("kv full")), detail) ==
          ApiFailure::Internal);
    CHECK(detail == "kv full");

    detail.clear();
    CHECK(classify_std_exception(std::make_exception_ptr(42), detail) == ApiFailure::Internal);
    CHECK(detail == "unknown exception");

    detail = "stale";
    CHECK(classify_std_exception(nullptr, detail) == ApiFailure::Internal);
    CHECK(detail == "unknown exception");
}

TEST_CASE("api failure: reply status and message per kind") {
    const auto malformed =
        api_failure_reply(ApiFailure::MalformedRequest, "malformed request body", "x");
    CHECK(malformed.http_code == 400);
    CHECK(std::string(malformed.status) == "400 Bad Request");
    CHECK(std::string(malformed.message) == "malformed request body");

    const auto oom = api_failure_reply(ApiFailure::OutOfMemory, "m", "x");
    CHECK(oom.http_code == 503);
    CHECK(std::string(oom.status) == "503 Service Unavailable");
    CHECK(std::string(oom.message) == "out of memory");

    const auto internal = api_failure_reply(ApiFailure::Internal, "m", "image generation failed");
    CHECK(internal.http_code == 500);
    CHECK(std::string(internal.status) == "500 Internal Server Error");
    CHECK(std::string(internal.message) == "image generation failed");
}
