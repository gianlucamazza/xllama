# Linux toolchain: Zen 2 microarchitecture (Xbox Series S CPU)
# Usage: cmake --preset linux-release
#
# On a non-Zen2 dev host, replace znver2 with native or your actual arch.

set(CMAKE_SYSTEM_NAME Linux)

set(XLLAMA_MARCH "znver2" CACHE STRING "Target CPU architecture for -march")

add_compile_options(
    -march=${XLLAMA_MARCH}
    -mavx2
    -mfma
    -mf16c
    -fno-omit-frame-pointer
)

add_link_options(
    -Wl,--as-needed
)
