// tests/test_metrics.cpp
#include <doctest/doctest.h>
#include "ocr_bench/metrics.hpp"

using namespace ocr_bench;

TEST_CASE("normalize lowercases and collapses whitespace") {
    CHECK(normalize("HELLO   world") == "hello world");
    CHECK(normalize("  Hi  ") == "hi");
    CHECK(normalize("") == "");
}

TEST_CASE("cer of identical normalized strings is 0.0") {
    CHECK(cer("HELLO world", "hello  WORLD") == doctest::Approx(0.0f));
}

TEST_CASE("cer of one-substitution strings matches known value") {
    // "abc" vs "axc": 1 substitution / 3 chars = 1/3
    CHECK(cer("abc", "axc") == doctest::Approx(1.0f / 3.0f));
}

TEST_CASE("cer with empty ref and non-empty hyp is 1.0") {
    CHECK(cer("", "something") == doctest::Approx(1.0f));
}

TEST_CASE("cer with empty ref and empty hyp is 0.0") {
    CHECK(cer("", "") == doctest::Approx(0.0f));
}

TEST_CASE("wer of one-substitution word sequences matches known value") {
    // "the cat sat" vs "the cat sit": 1 word sub / 3 words = 1/3
    CHECK(wer("the cat sat", "the cat sit") == doctest::Approx(1.0f / 3.0f));
}

TEST_CASE("DetectionStats precision/recall/f1 match known values") {
    DetectionStats d{7, 2, 1};
    CHECK(d.precision() == doctest::Approx(7.0f / 9.0f));
    CHECK(d.recall() == doctest::Approx(7.0f / 8.0f));
    float p = d.precision();
    float r = d.recall();
    CHECK(d.f1() == doctest::Approx(2 * p * r / (p + r)));
}

TEST_CASE("DetectionStats with zero tp+fp+fn does not divide by zero") {
    DetectionStats d{0, 0, 0};
    CHECK(d.precision() == doctest::Approx(0.0f));
    CHECK(d.recall() == doctest::Approx(0.0f));
    CHECK(d.f1() == doctest::Approx(0.0f));
}
