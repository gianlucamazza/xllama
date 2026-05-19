cmake_minimum_required(VERSION 3.22)
project(xllama LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Target selection: linux (default) | uwp
# UWP packaging requires MSVC + Windows SDK — use scripts/build-uwp.ps1
# ------------------------------------------------------------------
set(XLLAMA_TARGET "linux" CACHE STRING "Build target: linux or uwp")
set_property(CACHE XLLAMA_TARGET PROPERTY STRINGS linux uwp)

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

# ------------------------------------------------------------------
# Linux build — CPU-only, Zen 2 tuned
# ------------------------------------------------------------------
message(STATUS "xllama: building for target=linux")

# llama.cpp options — disable features that don't translate to UWP
# so that bridge code is exercised the same way on Linux dev host
set(LLAMA_NATIVE          OFF CACHE BOOL "" FORCE)  # use explicit arch flags
set(LLAMA_AVX2            ON  CACHE BOOL "" FORCE)
set(LLAMA_F16C            ON  CACHE BOOL "" FORCE)
set(LLAMA_FMA             ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

# ------------------------------------------------------------------
# xllama-cli — thin wrapper around the bridge for Linux dev/bench use
# ------------------------------------------------------------------
add_executable(xllama-cli
    uwp/llama-bridge.cpp
    src/main.cpp
)

target_include_directories(xllama-cli PRIVATE
    uwp/
    llama.cpp/include
    llama.cpp/common
)

target_link_libraries(xllama-cli PRIVATE
    llama
    common
)

target_compile_definitions(xllama-cli PRIVATE
    XLLAMA_LINUX=1
)

# Zen 2 = Xbox CPU microarchitecture; also valid for most Ryzen 3xxx dev hosts.
# Override with -DCMAKE_CXX_FLAGS="-march=native" if your dev host is different.
target_compile_options(xllama-cli PRIVATE
    -march=znver2 -mavx2 -O3 -fno-omit-frame-pointer
)

set_target_properties(xllama-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "xllama: xllama-cli target configured — build with: cmake --build . -j")
