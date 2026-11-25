// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "../uwp/api-server.h"

#include <doctest/doctest.h>

TEST_CASE("LAN API port validation matches the Xbox bind contract") {
    CHECK(xllama::api::port_bindable(1025));
    CHECK(xllama::api::port_bindable(11434));
    CHECK(xllama::api::port_bindable(49151));
    CHECK_FALSE(xllama::api::port_bindable(1024));
    CHECK_FALSE(xllama::api::port_bindable(11443));
    CHECK_FALSE(xllama::api::port_bindable(49152));
}

TEST_CASE("configured_api_port: the Settings page shows the port the server binds") {
    using xllama::api::configured_api_port;
    using xllama::api::kDefaultApiPort;
    CHECK(configured_api_port("") == kDefaultApiPort);
    CHECK(configured_api_port("8080") == 8080);
    CHECK(configured_api_port(" 8080\r\n") == 8080);
    // Not bindable on Xbox, or not a whole number: the server falls back, so
    // the page must not show the raw text.
    CHECK(configured_api_port("11443") == kDefaultApiPort);
    CHECK(configured_api_port("60000") == kDefaultApiPort);
    CHECK(configured_api_port("80") == kDefaultApiPort);
    CHECK(configured_api_port("8080x") == kDefaultApiPort);
    CHECK(configured_api_port("port") == kDefaultApiPort);
}
