# The UWP MSBuild project bypasses ggml's CMake configure step. Generate its
# version header from the upstream version declarations and template.
if(NOT DEFINED GGML_SOURCE_DIR OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "GGML_SOURCE_DIR and OUTPUT_FILE are required")
endif()
file(READ "${GGML_SOURCE_DIR}/CMakeLists.txt" version_source)
foreach(component MAJOR MINOR PATCH)
    string(REGEX MATCH "set\\(GGML_VERSION_${component} ([0-9]+)\\)" match "${version_source}")
    if(NOT match)
        message(FATAL_ERROR "Cannot read upstream GGML_VERSION_${component}")
    endif()
    set(GGML_VERSION_${component} "${CMAKE_MATCH_1}")
endforeach()
set(GGML_VERSION "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")
find_program(GIT_EXE NAMES git git.exe REQUIRED)
execute_process(COMMAND "${GIT_EXE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${GGML_SOURCE_DIR}"
    OUTPUT_VARIABLE GGML_BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXE}" diff-index --quiet HEAD -- .
    WORKING_DIRECTORY "${GGML_SOURCE_DIR}" RESULT_VARIABLE dirty)
if(dirty EQUAL 1)
    string(APPEND GGML_BUILD_COMMIT "-dirty")
elseif(NOT dirty EQUAL 0)
    message(FATAL_ERROR "Cannot determine ggml working-tree identity")
endif()
configure_file("${GGML_SOURCE_DIR}/src/ggml-version.h.in" "${OUTPUT_FILE}" @ONLY)
