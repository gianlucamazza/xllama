#include <doctest/doctest.h>

#include "xllama/debug_dashboard.h"

#include <string>

using xllama::ApiActivity;
using xllama::compose_debug_dashboard;
using xllama::debug_dashboard_flag_basename;
using xllama::debug_dashboard_opt_in;
using xllama::debug_dashboard_settings_key;
using xllama::DebugAdmissionState;
using xllama::DebugDashboardInputs;
using xllama::DebugLanState;
using xllama::decode_tokens_per_second;
using xllama::format_debug_dashboard;
using xllama::InferenceResult;
using xllama::kDebugUnavailable;
using xllama::prompt_tokens_per_second;

TEST_CASE("debug dashboard is off unless a setting or flag opts in") {
    CHECK_FALSE(debug_dashboard_opt_in(false, false));
    CHECK(debug_dashboard_opt_in(true, false));
    CHECK(debug_dashboard_opt_in(false, true));
    CHECK(debug_dashboard_opt_in(true, true));
}

TEST_CASE("debug dashboard flag and settings keys are portable basenames") {
    const std::string flag(debug_dashboard_flag_basename());
    const std::string key(debug_dashboard_settings_key());
    CHECK(flag == "debug-dashboard.flag");
    CHECK(key == "debug_dashboard");
    CHECK(flag.find('/') == std::string::npos);
    CHECK(flag.find('\\') == std::string::npos);
    CHECK(flag.find(':') == std::string::npos);
    CHECK(key.find('/') == std::string::npos);
    CHECK(key.find('\\') == std::string::npos);
}

TEST_CASE("debug dashboard marks unread sources unavailable") {
    const auto snap = compose_debug_dashboard({});
    CHECK_FALSE(snap.resident_model.available);
    CHECK_FALSE(snap.session_loaded.available);
    CHECK_FALSE(snap.n_ctx.available);
    CHECK_FALSE(snap.n_gpu_layers.available);
    CHECK_FALSE(snap.lora_path.available);
    CHECK(snap.preloading.available);
    CHECK_FALSE(snap.preloading.value);
    CHECK_FALSE(snap.lan_state.available);
    CHECK_FALSE(snap.lan_port.available);
    CHECK_FALSE(snap.lan_requests.available);
    CHECK_FALSE(snap.lan_connected.available);
    CHECK_FALSE(snap.lan_in_flight.available);
    CHECK_FALSE(snap.decode_tok_s.available);
    CHECK_FALSE(snap.prompt_tok_s.available);
    CHECK_FALSE(snap.t_p_eval_ms.available);
    CHECK_FALSE(snap.t_eval_ms.available);
    CHECK_FALSE(snap.queue_depth.available);
    REQUIRE(snap.admission.available);
    CHECK(snap.admission.value == DebugAdmissionState::Idle);
}

TEST_CASE("debug dashboard reports a resident hub snapshot") {
    DebugDashboardInputs in;
    in.hub_snapshot_available = true;
    in.session_loaded = true;
    in.resident_model_id = "lfm25-350m";
    in.n_ctx = 2048;
    in.n_gpu_layers = 12;
    in.lora_path = "adapters/tone.gguf";
    in.preloading = false;

    const auto snap = compose_debug_dashboard(in);
    REQUIRE(snap.resident_model.available);
    CHECK(snap.resident_model.value == "lfm25-350m");
    REQUIRE(snap.session_loaded.available);
    CHECK(snap.session_loaded.value);
    REQUIRE(snap.n_ctx.available);
    CHECK(snap.n_ctx.value == 2048);
    REQUIRE(snap.n_gpu_layers.available);
    CHECK(snap.n_gpu_layers.value == 12);
    REQUIRE(snap.lora_path.available);
    CHECK(snap.lora_path.value == "adapters/tone.gguf");
}

TEST_CASE("debug dashboard reports an empty hub as no resident model") {
    DebugDashboardInputs in;
    in.hub_snapshot_available = true;
    in.session_loaded = false;

    const auto snap = compose_debug_dashboard(in);
    REQUIRE(snap.session_loaded.available);
    CHECK_FALSE(snap.session_loaded.value);
    REQUIRE(snap.resident_model.available);
    CHECK(snap.resident_model.value.empty());
    CHECK_FALSE(snap.n_ctx.available);
    CHECK_FALSE(snap.n_gpu_layers.available);
}

TEST_CASE("debug dashboard derives tok/s only from measured inference timings") {
    InferenceResult ok;
    ok.success = true;
    ok.n_p_eval = 80;
    ok.t_p_eval_ms = 2000.0;
    ok.n_eval = 40;
    ok.t_eval_ms = 1000.0;
    ok.t_first_token_ms = 2100.0;
    ok.n_prompt = 80;

    const auto decode = decode_tokens_per_second(ok);
    const auto prompt = prompt_tokens_per_second(ok);
    REQUIRE(decode.available);
    REQUIRE(prompt.available);
    CHECK(decode.value == doctest::Approx(40.0));
    CHECK(prompt.value == doctest::Approx(40.0));

    DebugDashboardInputs in;
    in.last_inference = ok;
    const auto snap = compose_debug_dashboard(in);
    REQUIRE(snap.decode_tok_s.available);
    REQUIRE(snap.prompt_tok_s.available);
    REQUIRE(snap.t_p_eval_ms.available);
    REQUIRE(snap.t_eval_ms.available);
    REQUIRE(snap.t_first_token_ms.available);
    CHECK(snap.decode_tok_s.value == doctest::Approx(40.0));
    CHECK(snap.n_eval.value == 40);
    CHECK(snap.n_prompt.value == 80);
}

TEST_CASE("debug dashboard leaves tok/s unavailable without timings") {
    InferenceResult missing;
    missing.success = true;
    missing.n_eval = 10;
    missing.t_eval_ms = 0.0;
    missing.n_p_eval = 0;
    missing.t_p_eval_ms = 0.0;
    CHECK_FALSE(decode_tokens_per_second(missing).available);
    CHECK_FALSE(prompt_tokens_per_second(missing).available);

    DebugDashboardInputs in;
    in.last_inference = missing;
    const auto snap = compose_debug_dashboard(in);
    CHECK_FALSE(snap.decode_tok_s.available);
    CHECK_FALSE(snap.prompt_tok_s.available);
    CHECK_FALSE(snap.t_eval_ms.available);
    CHECK_FALSE(snap.t_p_eval_ms.available);
}

TEST_CASE("debug dashboard does not invent tok/s for a failed inference") {
    InferenceResult fail;
    fail.success = false;
    fail.n_eval = 8;
    fail.t_eval_ms = 400.0;
    CHECK_FALSE(decode_tokens_per_second(fail).available);

    DebugDashboardInputs in;
    in.last_inference = fail;
    CHECK_FALSE(compose_debug_dashboard(in).decode_tok_s.available);
}

TEST_CASE("debug dashboard copies LAN activity including connected clients") {
    ApiActivity activity;
    const auto t0 = ApiActivity::Clock::now();
    auto client = activity.client();
    auto ticket = activity.begin("chat", t0);
    ticket.model("phi4-mini");
    activity.busy();

    DebugDashboardInputs in;
    in.lan_status_available = true;
    in.lan_state = DebugLanState::Running;
    in.lan_port = 11434;
    in.activity_available = true;
    in.activity = activity.snapshot(t0 + std::chrono::seconds(2));

    const auto snap = compose_debug_dashboard(in);
    REQUIRE(snap.lan_state.available);
    CHECK(snap.lan_state.value == DebugLanState::Running);
    REQUIRE(snap.lan_port.available);
    CHECK(snap.lan_port.value == 11434);
    REQUIRE(snap.lan_requests.available);
    CHECK(snap.lan_requests.value == 1);
    REQUIRE(snap.lan_busy.available);
    CHECK(snap.lan_busy.value == 1);
    REQUIRE(snap.lan_connected.available);
    CHECK(snap.lan_connected.value == 1);
    REQUIRE(snap.lan_in_flight.available);
    CHECK(snap.lan_in_flight.value.find("chat") != std::string::npos);
    CHECK(snap.lan_in_flight.value.find("phi4-mini") != std::string::npos);

    ticket.done(200, "", t0 + std::chrono::seconds(3));
    client = {};
    in.activity = activity.snapshot(t0 + std::chrono::seconds(4));
    const auto after = compose_debug_dashboard(in);
    REQUIRE(after.lan_in_flight.available);
    CHECK(after.lan_in_flight.value == "idle");
    REQUIRE(after.lan_connected.available);
    CHECK(after.lan_connected.value == 0);
    REQUIRE(after.lan_ok.available);
    CHECK(after.lan_ok.value == 1);
}

TEST_CASE("debug dashboard admission uses existing busy signals only") {
    DebugDashboardInputs idle;
    CHECK(compose_debug_dashboard(idle).admission.value == DebugAdmissionState::Idle);

    DebugDashboardInputs preload;
    preload.preloading = true;
    CHECK(compose_debug_dashboard(preload).admission.value == DebugAdmissionState::Preloading);

    DebugDashboardInputs gui;
    gui.gui_generating = true;
    CHECK(compose_debug_dashboard(gui).admission.value == DebugAdmissionState::Busy);

    DebugDashboardInputs train;
    train.gui_training = true;
    CHECK(compose_debug_dashboard(train).admission.value == DebugAdmissionState::Busy);

    DebugDashboardInputs diffuse;
    diffuse.gui_diffusing = true;
    CHECK(compose_debug_dashboard(diffuse).admission.value == DebugAdmissionState::Busy);

    ApiActivity activity;
    auto ticket = activity.begin("chat");
    DebugDashboardInputs api;
    api.activity_available = true;
    api.activity = activity.snapshot();
    CHECK(compose_debug_dashboard(api).admission.value == DebugAdmissionState::Busy);
    ticket.done(200);
}

TEST_CASE("debug dashboard queue depth stays unavailable unless supplied") {
    CHECK_FALSE(compose_debug_dashboard({}).queue_depth.available);

    DebugDashboardInputs in;
    in.queue_depth = 2;
    const auto snap = compose_debug_dashboard(in);
    REQUIRE(snap.queue_depth.available);
    CHECK(snap.queue_depth.value == 2);
}

TEST_CASE("debug dashboard text uses unavailable instead of placeholder numbers") {
    const std::string text = format_debug_dashboard(compose_debug_dashboard({}));
    CHECK(text.find(kDebugUnavailable) != std::string::npos);
    CHECK(text.find("resident:") != std::string::npos);
    CHECK(text.find("connected clients:") != std::string::npos);
    CHECK(text.find("decode tok/s:") != std::string::npos);
    CHECK(text.find("Admission") != std::string::npos);
    CHECK(text.find("queue:") != std::string::npos);
    CHECK(text.find("idle") != std::string::npos);
    CHECK(text.find("999") == std::string::npos);
    CHECK(text.find("TODO") == std::string::npos);
    CHECK(text.find("lorem") == std::string::npos);
}
