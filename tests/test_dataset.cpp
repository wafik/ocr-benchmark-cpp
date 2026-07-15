#include <doctest/doctest.h>
#include <algorithm>
#include "ocr_bench/dataset.hpp"

using namespace ocr_bench;

static std::string normalizeSeparators(std::string s) {
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
}

TEST_CASE("loadCategory parses labelme JSON sidecar, skips empty labels") {
    auto pages = loadCategory("tests/fixtures/labelme_sample/BADGES");
    REQUIRE(pages.size() == 1);
    CHECK(pages[0].category == "BADGES");
    REQUIRE(pages[0].lines.size() == 1); // second shape has empty label, skipped
    CHECK(pages[0].lines[0].text == "PROVINSI JAWA BARAT");
    REQUIRE(pages[0].lines[0].polygon.size() == 4);
    CHECK(pages[0].lines[0].polygon[0].x == doctest::Approx(10.0f));
    CHECK(pages[0].lines[0].polygon[0].y == doctest::Approx(20.0f));
}

TEST_CASE("listCategories finds labelme category directories with images") {
    auto cats = listCategories("tests/fixtures/labelme_sample");
    REQUIRE(cats.size() == 1);
    CHECK(normalizeSeparators(cats[0]) == "tests/fixtures/labelme_sample/BADGES");
}

TEST_CASE("listCategories returns empty for a nonexistent root") {
    auto cats = listCategories("tests/fixtures/does_not_exist");
    CHECK(cats.empty());
}

TEST_CASE("iterAllImages aggregates pages across all categories") {
    auto pages = iterAllImages("tests/fixtures/labelme_sample");
    REQUIRE(pages.size() == 1);
    CHECK(pages[0].category == "BADGES");
}
