# Linux toolchain: Zen 2 microarchitecture (Xbox Series S CPU)
# Usage: cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/linux-toolchain.cmake
#
# On a non-Zen2 dev host, replace znver2 with native or your actual arch.
# On the Xbox itself (cross-compiled): adjust triple to x86_64-linux-musl
# if building a standalone Linux binary via devkit.

set(CMAKE_SYSTEM_NAME Linux)

set(XLLAMA_MARCH "znver2" CACHE STRING "Target CPU architecture for -march")

add_compile_options(
    -march=${XLLAMA_MARCH}
    -mavx2
    -mfma
    -mf16c
    -O3
    -fno-omit-frame-pointer   # keeps stack frames for perf/profiling
    -fno-plt                   # reduces PLT overhead on glibc
)

add_link_options(
    -Wl,--as-needed
)
