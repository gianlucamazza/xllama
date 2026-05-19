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
)

target_link_libraries(xllama-tests PRIVATE xllama doctest::doctest)

add_test(NAME xllama-tests COMMAND xllama-tests)
