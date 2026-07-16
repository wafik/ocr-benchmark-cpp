// tests/test_ocr_e2e.cpp — End-to-end OCR test with real models and real image
#include <doctest/doctest.h>
#include <iostream>
#include "ocr_bench/engine.hpp"

using namespace ocr_bench;

TEST_CASE("BenchEngine OCR on real receipt image from IND_CN dataset") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.modelType = "tiny";
    cfg.useAngleCls = false;

    BenchEngine engine(cfg);
    CHECK(engine.backend() == "cpu");

    auto result = engine.predict("tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg");

    std::cout << "=== OCR E2E Test Result ===" << std::endl;
    std::cout << "Image: " << result.image << std::endl;
    std::cout << "Lines detected: " << result.lines.size() << std::endl;
    std::cout << "Elapsed: " << result.elapsedMs << " ms" << std::endl;

    for (size_t i = 0; i < result.lines.size(); i++) {
        std::cout << "  Line " << i << ": \"" << result.lines[i].text
                  << "\" (score=" << result.lines[i].score << ")" << std::endl;
    }

    // Real receipt image should produce at least some detected lines
    CHECK(result.lines.size() > 0);
    CHECK(result.elapsedMs > 0.0f);
}
