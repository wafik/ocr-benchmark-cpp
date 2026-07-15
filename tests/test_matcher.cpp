#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "ocr_bench/matcher.hpp"

TEST_CASE("aabb returns bounding box of a polygon") {
    ocr_bench::Polygon p = {{0,0}, {10,0}, {10,10}, {0,10}};
    auto box = ocr_bench::aabb(p);
    CHECK(box[0] == doctest::Approx(0.0f));
    CHECK(box[1] == doctest::Approx(0.0f));
    CHECK(box[2] == doctest::Approx(10.0f));
    CHECK(box[3] == doctest::Approx(10.0f));
}

TEST_CASE("iouPolygon of identical boxes is 1.0") {
    ocr_bench::Polygon a = {{0,0}, {10,0}, {10,10}, {0,10}};
    CHECK(ocr_bench::iouPolygon(a, a) == doctest::Approx(1.0f));
}

TEST_CASE("iouPolygon of non-overlapping boxes is 0.0") {
    ocr_bench::Polygon a = {{0,0}, {10,0}, {10,10}, {0,10}};
    ocr_bench::Polygon b = {{100,100}, {110,100}, {110,110}, {100,110}};
    CHECK(ocr_bench::iouPolygon(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("matchPolygons: 2 GT, 2 PR overlapping perfectly -> 2 matches, 0 unmatched") {
    ocr_bench::Polygon g1 = {{0,0}, {10,0}, {10,10}, {0,10}};
    ocr_bench::Polygon g2 = {{20,20}, {30,20}, {30,30}, {20,30}};
    ocr_bench::Polygon p1 = {{1,1}, {11,1}, {11,11}, {1,11}};
    ocr_bench::Polygon p2 = {{19,19}, {31,19}, {31,31}, {19,31}};

    auto result = ocr_bench::matchPolygons({g1, g2}, {p1, p2}, 0.5f);
    CHECK(result.matches.size() == 2);
    CHECK(result.unmatchedGtIdx.empty());
    CHECK(result.unmatchedPrIdx.empty());
}

TEST_CASE("matchPolygons: 1 GT, 1 PR non-overlapping -> no match") {
    ocr_bench::Polygon g1 = {{0,0}, {10,0}, {10,10}, {0,10}};
    ocr_bench::Polygon p1 = {{100,100}, {110,100}, {110,110}, {100,110}};

    auto result = ocr_bench::matchPolygons({g1}, {p1}, 0.5f);
    CHECK(result.matches.empty());
    REQUIRE(result.unmatchedGtIdx.size() == 1);
    CHECK(result.unmatchedGtIdx[0] == 0);
    REQUIRE(result.unmatchedPrIdx.size() == 1);
    CHECK(result.unmatchedPrIdx[0] == 0);
}
