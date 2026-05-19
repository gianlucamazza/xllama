# UWP toolchain — for reference / future CMake-based UWP build
# Currently the UWP build is done via MSBuild (scripts/build-uwp.ps1).
# This file documents the CMake variables needed for a future migration.
#
# Usage: cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/uwp-toolchain.cmake
#        (requires MSVC on Windows — will error on Linux by design)

set(CMAKE_SYSTEM_NAME WindowsStore)
set(CMAKE_SYSTEM_VERSION 10.0)

# Minimum Windows 10 SDK version targeting Xbox Dev Mode
set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "10.0.22621.0")

# UWP constraints — these mirror AppxManifest TargetDeviceFamily restrictions
add_compile_definitions(
    XLLAMA_UWP=1
    WINAPI_FAMILY=WINAPI_FAMILY_APP   # UWP app family; disables Win32 APIs
    _UNICODE
    UNICODE
)

# Disable POSIX features not available in UWP sandbox
add_compile_definitions(
    LLAMA_USE_MMAP=0   # no POSIX mmap; use CreateFileMappingFromApp instead
)

# No dynamic linking of backends
add_compile_definitions(
    GGML_USE_STATIC_BACKEND=1
)
