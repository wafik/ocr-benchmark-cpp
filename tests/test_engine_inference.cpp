#include <doctest/doctest.h>
#include <iostream>
#include "ocr_bench/engine.hpp"

using namespace ocr_bench;

TEST_CASE("BenchEngine loads tiny models and runs predict on a test image") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.modelType = "tiny";
    cfg.useAngleCls = false;

    BenchEngine engine(cfg);
    CHECK(engine.backend() == "cpu");

    auto result = engine.predict("tests/fixtures/test_ocr.png");
    CHECK_FALSE(result.image.empty());
    CHECK(result.elapsedMs > 0.0f);

    // Print results for manual inspection
    std::cout << "Image: " << result.image << std::endl;
    std::cout << "Lines detected: " << result.lines.size() << std::endl;
    std::cout << "Elapsed: " << result.elapsedMs << " ms" << std::endl;
    for (size_t i = 0; i < result.lines.size(); i++) {
        std::cout << "  Line " << i << ": \"" << result.lines[i].text
                  << "\" (score=" << result.lines[i].score << ")" << std::endl;
    }
}
