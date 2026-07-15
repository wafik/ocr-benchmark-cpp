cmake_minimum_required(VERSION 3.20)
project(ocr_bench_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)

add_library(ocr_bench_core STATIC
    src/ocr_bench/matcher.cpp
    src/ocr_bench/metrics.cpp
    src/ocr_bench/dataset.cpp
    src/ocr_bench/paths.cpp
    src/ocr_bench/config.cpp
    src/ocr_bench/report_writer.cpp
)
target_include_directories(ocr_bench_core PUBLIC include)
target_link_libraries(ocr_bench_core PUBLIC nlohmann_json::nlohmann_json)

enable_testing()
add_executable(ocr_bench_tests
    tests/test_matcher.cpp
    tests/test_metrics.cpp
    tests/test_dataset.cpp
    tests/test_paths.cpp
    tests/test_config.cpp
    tests/test_report_writer.cpp
)
target_link_libraries(ocr_bench_tests PRIVATE ocr_bench_core doctest::doctest)
add_test(NAME ocr_bench_tests COMMAND ocr_bench_tests)
