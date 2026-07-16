cmake_minimum_required(VERSION 3.20)
project(ocr_bench_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default (vcpkg): everything resolved via vcpkg CONFIG packages, espeak-ng/onnxruntime
# built by vendor/piper/libpiper's own ExternalProject step.
# Jetson (-DOCR_BENCH_USE_SYSTEM_DEPS=ON): OpenCV/CURL from apt, onnxruntime and espeak-ng
# vendored under vendor/ (see README "Build on Jetson Nano" for why + setup commands).
option(OCR_BENCH_USE_SYSTEM_DEPS "Use apt packages + vendored onnxruntime/espeak-ng instead of vcpkg (Jetson/aarch64)" OFF)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

if(OCR_BENCH_USE_SYSTEM_DEPS)
    set(CMAKE_BUILD_TYPE Release)
    find_package(OpenCV REQUIRED)
    find_package(CURL REQUIRED)

    # onnxruntime pip wheel ships no C++ headers -> vendor the official release tarball.
    set(ORT_DIR ${CMAKE_SOURCE_DIR}/vendor/onnxruntime/dist)

    # System espeak-ng (1.51 on Ubuntu/Jetson) lacks espeak_TextToPhonemesWithTerminator,
    # which piper.cpp requires. Built from source at the same commit vendor/piper/libpiper
    # pins for its own vcpkg-based ExternalProject.
    set(ESPEAKNG_DIR ${CMAKE_SOURCE_DIR}/vendor/espeak-ng-src/install)
    set(ESPEAKNG_STATIC_LIB ${ESPEAKNG_DIR}/lib/libespeak-ng.a)
    set(UCD_STATIC_LIB ${CMAKE_SOURCE_DIR}/vendor/espeak-ng-src/build/src/ucd-tools/libucd.a)
else()
    find_package(onnxruntime CONFIG REQUIRED)
    find_package(OpenCV CONFIG REQUIRED)
    find_package(CURL CONFIG REQUIRED)
    find_package(httplib CONFIG REQUIRED)

    set(ESPEAKNG_DIR ${CMAKE_SOURCE_DIR}/vendor/piper/libpiper/build/espeak_ng-install)
    set(ESPEAKNG_STATIC_LIB espeak-ng)
    set(UCD_STATIC_LIB ucd)
endif()

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
    src/ocr_bench/tts_engine.cpp
    src/ocr_bench/tts_runner.cpp
    src/ocr_bench/combined_runner.cpp
    vendor/piper/libpiper/src/piper.cpp
)
target_include_directories(ocr_bench_core PUBLIC include)
target_include_directories(ocr_bench_core PUBLIC
    ${ESPEAKNG_DIR}/include
    vendor/piper/libpiper/include
)
target_compile_definitions(ocr_bench_core PRIVATE BUILDING_LIBPIPER)
target_link_libraries(ocr_bench_core PUBLIC
    nlohmann_json::nlohmann_json ${OpenCV_LIBS} rapidocronnx_vendor
    CURL::libcurl ${ESPEAKNG_STATIC_LIB} ${UCD_STATIC_LIB}
)
target_include_directories(ocr_bench_core PUBLIC ${OpenCV_INCLUDE_DIRS})

if(OCR_BENCH_USE_SYSTEM_DEPS)
    target_include_directories(ocr_bench_core PUBLIC ${ORT_DIR}/include)
    target_link_directories(ocr_bench_core PUBLIC ${ORT_DIR}/lib ${ESPEAKNG_DIR}/lib)
    target_link_libraries(ocr_bench_core PUBLIC onnxruntime)
    set_target_properties(ocr_bench_core PROPERTIES
        INSTALL_RPATH "${ORT_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    target_link_directories(ocr_bench_core PUBLIC
        ${ESPEAKNG_DIR}/lib
        ${CMAKE_SOURCE_DIR}/vendor/piper/libpiper/build/espeak_ng/src/espeak_ng_external-build/src/ucd-tools/Release
    )
    target_link_libraries(ocr_bench_core PUBLIC onnxruntime::onnxruntime)
endif()

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
    tests/test_tts_piper.cpp
)
target_link_libraries(ocr_bench_tests PRIVATE ocr_bench_core doctest::doctest)

if(NOT OCR_BENCH_USE_SYSTEM_DEPS)
    # api_server.cpp + its tests need httplib; on Jetson httplib is linked only into
    # ocr-bench-serve (see below) to keep the vendored-deps build minimal.
    target_sources(ocr_bench_tests PRIVATE
        src/api/api_server.cpp
        tests/test_api_health.cpp
        tests/test_api_run.cpp
        tests/test_api_summary_results.cpp
        tests/test_api_tts_combined.cpp
        tests/test_api_auth.cpp
        tests/test_ocr_e2e.cpp
    )
    target_link_libraries(ocr_bench_tests PRIVATE httplib::httplib)
    add_test(NAME ocr_bench_tests COMMAND ocr_bench_tests
        WORKING_DIRECTORY $<TARGET_FILE_DIR:ocr_bench_tests>
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(ocr_bench_tests PRIVATE stdc++fs)
    endif()
else()
    add_test(NAME ocr_bench_tests COMMAND ocr_bench_tests)
    set_target_properties(ocr_bench_tests PROPERTIES
        INSTALL_RPATH "${ORT_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Copy test fixtures and models next to the test binary so relative paths in tests resolve.
# models/ is gitignored and populated by ocr-bench-download; ensure it exists so the copy
# below doesn't fail on a fresh checkout before models have been downloaded.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
add_custom_command(TARGET ocr_bench_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/tests/fixtures
        $<TARGET_FILE_DIR:ocr_bench_tests>/tests/fixtures
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:ocr_bench_tests>/models
)

# CLI executables
add_executable(ocr-bench-run cli/run_benchmark.cpp)
target_link_libraries(ocr-bench-run PRIVATE ocr_bench_core cxxopts::cxxopts)

add_executable(ocr-bench-download cli/download_models.cpp)
target_link_libraries(ocr-bench-download PRIVATE ocr_bench_core cxxopts::cxxopts)

add_executable(ocr-bench-serve src/api/main.cpp src/api/api_server.cpp)
target_link_libraries(ocr-bench-serve PRIVATE ocr_bench_core cxxopts::cxxopts)

if(OCR_BENCH_USE_SYSTEM_DEPS)
    target_link_libraries(ocr-bench-serve PRIVATE cpp-httplib)
    foreach(tgt ocr-bench-run ocr-bench-download ocr-bench-serve)
        set_target_properties(${tgt} PROPERTIES
            INSTALL_RPATH "${ORT_DIR}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endforeach()
else()
    target_link_libraries(ocr-bench-serve PRIVATE httplib::httplib)
endif()
