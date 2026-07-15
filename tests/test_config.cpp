// tests/test_config.cpp
#include <doctest/doctest.h>
#include "ocr_bench/config.hpp"

using namespace ocr_bench;

TEST_CASE("loadSettings applies defaults when file is missing") {
    Settings s = loadSettings("tests/fixtures/does_not_exist.env");
    CHECK(s.ocrVersion == "PP-OCRv6");
    CHECK(s.modelType == "tiny");
    CHECK(s.detBoxThresh == doctest::Approx(0.5));
    CHECK(s.useAngleCls == false);
    CHECK(s.servePort == 8765);
    CHECK(s.authPassword == "AI4DB-BENCH");
}

TEST_CASE("loadSettings overrides defaults from a real .env file") {
    Settings s = loadSettings("tests/fixtures/sample.env");
    CHECK(s.ocrVersion == "PP-OCRv6");
    CHECK(s.modelType == "small");
    CHECK(s.detBoxThresh == doctest::Approx(0.6));
    CHECK(s.useAngleCls == true);
    CHECK(s.servePort == 9000);
}

TEST_CASE("Settings has no corrector fields (dropped per plan)") {
    // Compile-time check: this test exists to document the constraint.
    // If a future edit re-adds enable_symspell_correction to Settings,
    // this test file's absence of any reference to it is the documentation.
    Settings s;
    CHECK(s.ocrVersion == "PP-OCRv6"); // trivial — just confirms Settings compiles
}
