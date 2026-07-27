cmake_minimum_required(VERSION 3.22)
project(xllama LANGUAGES CXX C)

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
set(XLLAMA_TARGET "linux" CACHE STRING "Build target: linux or uwp")
set_property(CACHE XLLAMA_TARGET PROPERTY STRINGS linux uwp)

option(XLLAMA_BUILD_TESTS    "Build unit tests" OFF)
option(XLLAMA_ENABLE_ASAN    "Enable AddressSanitizer" OFF)
option(XLLAMA_ENABLE_UBSAN   "Enable UndefinedBehaviourSanitizer" OFF)

# ------------------------------------------------------------------
# Target selection
# ------------------------------------------------------------------
if(XLLAMA_TARGET STREQUAL "uwp")
    message(FATAL_ERROR
        "XLLAMA_TARGET=uwp is not supported via CMake directly.\n"
        "UWP packaging requires MSVC and the Windows SDK.\n"
        "On Windows, run: .\\scripts\\build-uwp.ps1\n"
        "See docs/uwp-constraints.md for details.")
endif()

if(NOT XLLAMA_TARGET STREQUAL "linux")
    message(FATAL_ERROR "Unknown XLLAMA_TARGET '${XLLAMA_TARGET}'. Valid values: linux, uwp.")
endif()

message(STATUS "xllama: building for target=linux")

# ------------------------------------------------------------------
# llama.cpp sub-project
# ------------------------------------------------------------------
# ISA baseline. The old LLAMA_NATIVE/LLAMA_AVX2/... spellings are inert here:
# llama.cpp maps deprecated vars only when truthy, and the AVX2/F16C/FMA ones
# have no GGML_* mapping at all — set the GGML_* vars directly.
option(XLLAMA_NATIVE_OPT "Tune ggml for the build host CPU (-march=native) instead of portable AVX2" OFF)
if(XLLAMA_NATIVE_OPT)
    set(GGML_NATIVE       ON  CACHE BOOL "" FORCE)
else()
    set(GGML_NATIVE       OFF CACHE BOOL "" FORCE)
    set(GGML_AVX2         ON  CACHE BOOL "" FORCE)
    set(GGML_F16C         ON  CACHE BOOL "" FORCE)
    set(GGML_FMA          ON  CACHE BOOL "" FORCE)
endif()
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
# Host tooling (LoRA merge, quantize, …). Subdirectory mode defaults these OFF
# (LLAMA_STANDALONE=OFF); force ON so scripts/lora-spike can build llama-export-lora.
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     ON  CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

# ------------------------------------------------------------------
# xllama library (shared between CLI and UWP)
# ------------------------------------------------------------------
add_library(xllama STATIC
    src/bridge/inference.cpp
    src/bridge/session.cpp
    src/bridge/kv_store.cpp
    src/bridge/bench.cpp
    src/bridge/membw.cpp
    src/bridge/platform.cpp
    src/bridge/path_utils.cpp
    src/bridge/utf8_utils.cpp
    src/bridge/chat_prompt.cpp
    src/bridge/cli.cpp
    src/bridge/training.cpp
    src/bridge/device_train.cpp
    src/bridge/preference_capture.cpp
    src/bridge/personalize.cpp
)

target_compile_features(xllama PUBLIC cxx_std_17)

target_include_directories(xllama PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
)

target_link_libraries(xllama PUBLIC llama)

target_compile_definitions(xllama PUBLIC
    XLLAMA_LINUX=1
    XLLAMA_USE_LLAMA=1
    XLLAMA_DEVICE_TRAIN=1
)

# Sanitizers (Linux / Clang/GCC only)
if(XLLAMA_ENABLE_ASAN)
    target_compile_options(xllama PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(xllama PUBLIC -fsanitize=address)
endif()
if(XLLAMA_ENABLE_UBSAN)
    target_compile_options(xllama PUBLIC -fsanitize=undefined)
    target_link_options(xllama PUBLIC -fsanitize=undefined)
endif()

# ------------------------------------------------------------------
# xllama-cli
# ------------------------------------------------------------------
add_executable(xllama-cli src/main.cpp)

target_link_libraries(xllama-cli PRIVATE xllama)

target_compile_options(xllama-cli PRIVATE
    -fno-omit-frame-pointer
)

set_target_properties(xllama-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install(TARGETS xllama-cli xllama
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
if(XLLAMA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "xllama: xllama-cli target configured — build with: cmake --build . -j")
