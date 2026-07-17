// Copyright (c) 2024 Venere Labs
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
