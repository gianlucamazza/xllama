include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(doctest)

add_executable(xllama-tests
    test_cli.cpp
    test_utf8.cpp
    test_path.cpp
    test_bench.cpp
    test_session.cpp
    test_chat_history.cpp
    test_diffusion.cpp
    test_routing_policy.cpp
    test_chat_prompt.cpp
    test_manifest_merge.cpp
    test_membw.cpp
    test_model_provision.cpp
    test_logit_parity.cpp
)

target_link_libraries(xllama-tests PRIVATE xllama doctest::doctest)

# Diffusion tests use the vendored nlohmann/json to load golden vectors, and are
# pointed at the in-repo golden file + CLIP tokenizer assets via compile defs.
target_include_directories(xllama-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/vendor)
target_compile_definitions(xllama-tests PRIVATE
    GOLDEN_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../diffusion/golden_vectors.json"
    CLIP_TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../diffusion/clip_tokenizer"
    PARITY_GOLDEN_BIN="${CMAKE_CURRENT_SOURCE_DIR}/golden/logits-smol-short.bin")

add_test(NAME xllama-tests COMMAND xllama-tests)
