# The UWP MSBuild project does not run llama.cpp's configure step.
# common.cpp calls llama_build_number / llama_commit / llama_compiler /
# llama_build_target, which the upstream template emits into build-info.cpp.
if(NOT DEFINED LLAMA_SOURCE_DIR OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "LLAMA_SOURCE_DIR and OUTPUT_FILE are required")
endif()

set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "MSVC")
set(BUILD_TARGET "UWP x64")

find_program(GIT_EXE NAMES git git.exe)
if(GIT_EXE)
    execute_process(
        COMMAND "${GIT_EXE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
        OUTPUT_VARIABLE HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if(RES EQUAL 0 AND NOT HEAD STREQUAL "")
        set(BUILD_COMMIT "${HEAD}")
    endif()
    execute_process(
        COMMAND "${GIT_EXE}" rev-list --count HEAD
        WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
        OUTPUT_VARIABLE COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if(RES EQUAL 0 AND NOT COUNT STREQUAL "")
        set(BUILD_NUMBER "${COUNT}")
    endif()
endif()

set(LLAMA_BUILD_NUMBER "${BUILD_NUMBER}")
set(LLAMA_BUILD_COMMIT "${BUILD_COMMIT}")
configure_file(
    "${LLAMA_SOURCE_DIR}/common/build-info.cpp.in"
    "${OUTPUT_FILE}"
    @ONLY
)
