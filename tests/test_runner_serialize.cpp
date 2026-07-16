#include <doctest/doctest.h>
#include "ocr_bench/runner.hpp"

using namespace ocr_bench;

namespace {
GroundTruthPage makeSerializeTestPage() {
    GroundTruthPage page;
    page.imagePath = "test.jpg";
    page.category = "BADGES";
    page.lines = {
        {{{0, 0}, {50, 0}, {50, 20}, {0, 20}}, "hello"},
        {{{0, 30}, {50, 30}, {50, 50}, {0, 50}}, "missed"},
    };
    return page;
}
}

TEST_CASE("serializePageMetrics produces matched/missed/spurious overlay entries with correct shapes") {
    auto page = makeSerializeTestPage();
    PagePrediction pred;
    pred.image = "test.jpg";
    pred.elapsedMs = 50.0f;
    pred.lines = {
        {{{0, 0}, {50, 0}, {50, 20}, {0, 20}}, "hello", 0.9f},
        {{{100, 100}, {150, 100}, {150, 120}, {100, 120}}, "extra", 0.7f},
    };
    auto pm = evaluatePage(page, pred);
    auto j = serializePageMetrics(pm, page, pred);

    CHECK(j["image"] == "test.jpg");
    CHECK(j["category"] == "BADGES");
    CHECK(j["n_gt"] == 2);
    CHECK(j["n_pred"] == 2);
    CHECK(j["detection"]["tp"] == 1);
    CHECK(j["detection"]["fp"] == 1);
    CHECK(j["detection"]["fn"] == 1);
    CHECK(j["elapsed_ms"] == doctest::Approx(50.0));
    CHECK_FALSE(j.contains("matched_cer_corrected_mean"));
    CHECK_FALSE(j.contains("correction_status"));
    CHECK_FALSE(j.contains("correction_enabled"));

    REQUIRE(j.contains("overlays"));
    auto& overlays = j["overlays"];
    REQUIRE(overlays.size() == 3);

    int matchedCount = 0, missedCount = 0, spuriousCount = 0;
    for (auto& ov : overlays) {
        if (ov["status"] == "matched") {
            matchedCount++;
            CHECK(ov["gt_text"] == "hello");
            CHECK(ov["pr_text"] == "hello");
            CHECK(ov.contains("line_cer"));
            CHECK(ov["line_cer"].get<float>() == doctest::Approx(0.0f));
        } else if (ov["status"] == "missed") {
            missedCount++;
            CHECK(ov["gt_text"] == "missed");
            CHECK(ov["pr_polygon"].is_null());
        } else if (ov["status"] == "spurious") {
            spuriousCount++;
            CHECK(ov["pr_text"] == "extra");
            CHECK(ov["gt_polygon"].is_null());
        }
    }
    CHECK(matchedCount == 1);
    CHECK(missedCount == 1);
    CHECK(spuriousCount == 1);
}

TEST_CASE("serializePageMetrics with empty prediction has all overlays as missed") {
    GroundTruthPage page;
    page.imagePath = "empty_pred.jpg";
    page.category = "TEST";
    page.lines = {GroundTruthLine{{{0, 0}, {10, 0}, {10, 10}, {0, 10}}, "text"}};

    PagePrediction pred;
    pred.image = "empty_pred.jpg";
    pred.elapsedMs = 5.0f;

    auto pm = evaluatePage(page, pred);
    auto j = serializePageMetrics(pm, page, pred);

    CHECK(j["empty_output"] == true);
    REQUIRE(j["overlays"].size() == 1);
    CHECK(j["overlays"][0]["status"] == "missed");
}
