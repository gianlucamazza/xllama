include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(doctest)

add_executable(xllama-tests
    test_main.cpp
    test_cli.cpp
    test_utf8.cpp
    test_path.cpp
    test_bench.cpp
    test_session.cpp
    test_kv_store.cpp
    test_chat_history.cpp
    test_diffusion.cpp
    test_stability_attribution.cpp
    test_routing_policy.cpp
    test_cancel_policy.cpp
    test_sampling.cpp
    test_decode_loop.cpp
    test_inference_progress.cpp
    test_local_knob.cpp
    test_chat_prompt.cpp
    test_prompt_budget.cpp
    test_session_spec.cpp
    test_api_policy.cpp
    test_api_sampling.cpp
    test_http_framing.cpp
    test_sse.cpp
    test_stop_filter.cpp
    test_tool_proposal.cpp
    test_tool_budget.cpp
    test_manifest_merge.cpp
    test_membw.cpp
    test_diskbw.cpp
    test_gpubw.cpp
    test_gpugemv.cpp
    test_gpustep.cpp
    test_ggml_d3d12.cpp
    test_ramceil.cpp
    test_model_provision.cpp
    test_model_file_plan.cpp
    test_moe_streaming.cpp
    test_model_write.cpp
    test_logit_parity.cpp
    test_tensor_capture.cpp
    test_training.cpp
    test_device_train.cpp
    test_json_utils.cpp
    test_app_settings.cpp
    test_model_sampling.cpp
    test_preference_capture.cpp
    test_prefix_snapshot.cpp
    test_context_shift.cpp
    test_memory_admission.cpp
    test_granite_hybrid_snapshot.cpp
    test_personalize.cpp
    test_api_config.cpp
    test_api_activity.cpp
    test_api_discovery.cpp
    test_llama_patches.cpp
    test_thinking_split.cpp
    test_debug_dashboard.cpp
    test_compute_admission.cpp
    test_api_connection.cpp
    test_api_failure.cpp
    test_api_pull.cpp
    test_embedding.cpp
    test_autopilot.cpp
    test_speculative.cpp
    test_capture_probe.cpp
)

target_link_libraries(xllama-tests PRIVATE xllama doctest::doctest)

# Diffusion tests use the vendored nlohmann/json to load golden vectors, and are
# pointed at the in-repo golden file + CLIP tokenizer assets via compile defs.
target_include_directories(xllama-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor)
target_compile_definitions(xllama-tests PRIVATE
    GOLDEN_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../diffusion/golden_vectors.json"
    CLIP_TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../diffusion/clip_tokenizer"
    PARITY_GOLDEN_BIN="${CMAKE_CURRENT_SOURCE_DIR}/golden/logits-smol-short.bin"
    LLAMA_TEMPLATES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/models/templates"
    LLAMA_PATCHES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../patches"
    MANIFEST_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../uwp/models/manifest.json")

add_test(NAME xllama-tests COMMAND xllama-tests)

# Keep replacement operator new/delete for allocation-failure tests out of the shared test runner.
add_executable(test-admission-allocation test_admission_allocation.cpp)
target_compile_features(test-admission-allocation PRIVATE cxx_std_17)
target_include_directories(test-admission-allocation PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME test-admission-allocation COMMAND test-admission-allocation)

# Invoked by run-cpu-smoke.py with a verified model and an independent oracle.
add_executable(xllama-cpu-smoke cpu_smoke.cpp)
target_include_directories(xllama-cpu-smoke PRIVATE ${PROJECT_SOURCE_DIR}/llama.cpp/vendor)
target_link_libraries(xllama-cpu-smoke PRIVATE xllama)
