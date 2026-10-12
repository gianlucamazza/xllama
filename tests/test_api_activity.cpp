#include <doctest/doctest.h>

#include "xllama/api_activity.h"

#include <chrono>
#include <thread>
#include <vector>

using xllama::ApiActivity;
using namespace std::chrono_literals;

TEST_CASE("API activity tracks an in-flight request and its outcome") {
    ApiActivity activity;
    const auto t0 = ApiActivity::Clock::now();

    auto ticket = activity.begin("chat", t0);
    ticket.model("phi4-mini");
    auto during = activity.snapshot(t0 + 2s);
    REQUIRE(during.in_flight.has_value());
    CHECK(during.in_flight->endpoint == "chat");
    CHECK(during.in_flight->model == "phi4-mini");
    CHECK(during.in_flight->elapsed_s == doctest::Approx(2.0));
    CHECK(during.requests == 1);

    ticket.tokens(73, 6);
    ticket.done(200, "", t0 + 3s);
    auto after = activity.snapshot(t0 + 4s);
    CHECK_FALSE(after.in_flight.has_value());
    CHECK(after.ok == 1);
    CHECK(after.failed == 0);
    REQUIRE(after.last.has_value());
    CHECK(after.last->model == "phi4-mini");
    CHECK(after.last->http_status == 200);
    CHECK(after.last->elapsed_s == doctest::Approx(3.0));
    CHECK(after.last->prompt_tokens == 73);
    CHECK(after.last->completion_tokens == 6);
    CHECK_FALSE(after.last_error.has_value());
}

TEST_CASE("API activity records failures with message and age") {
    ApiActivity activity;
    const auto t0 = ApiActivity::Clock::now();
    activity.begin("chat", t0).done(400, "prompt too long", t0 + 1s);

    auto s = activity.snapshot(t0 + 61s);
    CHECK(s.failed == 1);
    REQUIRE(s.last_error.has_value());
    CHECK(s.last_error->message == "HTTP 400 chat: prompt too long");
    CHECK(s.last_error->age_s == doctest::Approx(60.0));
}

TEST_CASE("API activity treats an abandoned ticket as an aborted request") {
    ApiActivity activity;
    {
        auto ticket = activity.begin("embeddings");
    } // no done(): an exception escaped
    auto s = activity.snapshot();
    CHECK_FALSE(s.in_flight.has_value());
    CHECK(s.failed == 1);
    REQUIRE(s.last_error.has_value());
    CHECK(s.last_error->message == "HTTP 500 embeddings: request aborted");
}

TEST_CASE("API activity ignores a stale ticket after a newer request began") {
    ApiActivity activity;
    auto first = activity.begin("chat");
    auto second = activity.begin("images"); // the first was never finished
    first.done(200);
    auto s = activity.snapshot();
    REQUIRE(s.in_flight.has_value());
    CHECK(s.in_flight->endpoint == "images");
    CHECK(s.ok == 0);
    second.done(200);
    CHECK(activity.snapshot().ok == 1);
}

TEST_CASE("API activity counts busy rejections apart from served requests") {
    ApiActivity activity;
    activity.busy();
    activity.busy();
    auto s = activity.snapshot();
    CHECK(s.busy == 2);
    CHECK(s.requests == 0);
}

TEST_CASE("API activity is safe under concurrent writers and readers") {
    ApiActivity activity;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&activity] {
            for (int i = 0; i < 500; ++i) {
                activity.busy();
                (void)activity.snapshot();
            }
        });
    }
    for (auto& thread : threads)
        thread.join();
    CHECK(activity.snapshot().busy == 2000);
}
