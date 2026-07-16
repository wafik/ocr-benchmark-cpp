cmake_minimum_required(VERSION 3.20)
project(ocr_bench_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)

add_library(rapidocronnx_vendor STATIC
    vendor/rapidocronnx/clipper.cpp
)
target_include_directories(rapidocronnx_vendor PUBLIC vendor/rapidocronnx)

add_library(ocr_bench_core STATIC
    src/ocr_bench/matcher.cpp
    src/ocr_bench/metrics.cpp
    src/ocr_bench/dataset.cpp
    src/ocr_bench/paths.cpp
    src/ocr_bench/config.cpp
    src/ocr_bench/report_writer.cpp
    src/ocr_bench/ocr_utils.cpp
    src/ocr_bench/angle_net.cpp
    src/ocr_bench/crnn_net.cpp
    src/ocr_bench/db_net.cpp
    src/ocr_bench/engine.cpp
    src/ocr_bench/model_downloader.cpp
    src/ocr_bench/sysmon.cpp
    src/ocr_bench/run_status.cpp
    src/ocr_bench/runner.cpp
)
target_include_directories(ocr_bench_core PUBLIC include)
target_link_libraries(ocr_bench_core PUBLIC nlohmann_json::nlohmann_json ${OpenCV_LIBS} rapidocronnx_vendor onnxruntime::onnxruntime CURL::libcurl)
target_include_directories(ocr_bench_core PUBLIC ${OpenCV_INCLUDE_DIRS})

enable_testing()
add_executable(ocr_bench_tests
    tests/test_matcher.cpp
    tests/test_metrics.cpp
    tests/test_dataset.cpp
    tests/test_paths.cpp
    tests/test_config.cpp
    tests/test_report_writer.cpp
    tests/test_ocr_utils.cpp
    tests/test_angle_net.cpp
    tests/test_crnn_net.cpp
    tests/test_db_net.cpp
    tests/test_engine.cpp
    tests/test_engine_inference.cpp
    tests/test_model_downloader.cpp
    tests/test_sysmon.cpp
    tests/test_run_status.cpp
    tests/test_runner_evaluate_page.cpp
    tests/test_runner_serialize.cpp
    src/api/api_server.cpp
    tests/test_api_health.cpp
    tests/test_api_run.cpp
    tests/test_api_summary_results.cpp
    tests/test_api_tts_combined.cpp
    tests/test_api_auth.cpp
)
target_link_libraries(ocr_bench_tests PRIVATE ocr_bench_core doctest::doctest httplib::httplib)
add_test(NAME ocr_bench_tests COMMAND ocr_bench_tests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:ocr_bench_tests>
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(ocr_bench_tests PRIVATE stdc++fs)
endif()

# Copy test fixtures and models next to the test binary so relative paths in tests resolve.
add_custom_command(TARGET ocr_bench_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/tests/fixtures
        $<TARGET_FILE_DIR:ocr_bench_tests>/tests/fixtures
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:ocr_bench_tests>/models
)

# CLI executables
find_package(cxxopts CONFIG REQUIRED)

add_executable(ocr-bench-run cli/run_benchmark.cpp)
target_link_libraries(ocr-bench-run PRIVATE ocr_bench_core cxxopts::cxxopts)

add_executable(ocr-bench-download cli/download_models.cpp)
target_link_libraries(ocr-bench-download PRIVATE ocr_bench_core cxxopts::cxxopts)

add_executable(ocr-bench-serve src/api/main.cpp src/api/api_server.cpp)
target_link_libraries(ocr-bench-serve PRIVATE ocr_bench_core httplib::httplib cxxopts::cxxopts)
