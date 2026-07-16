#include <doctest/doctest.h>
#include "ocr_bench/runner.hpp"

using namespace ocr_bench;

namespace {

GroundTruthPage makeTestPage() {
    GroundTruthPage page;
    page.imagePath = "test.jpg";
    page.category = "BADGES";
    page.lines = {
        {{{0, 0}, {50, 0}, {50, 20}, {0, 20}}, "hello"},
        {{{0, 30}, {50, 30}, {50, 50}, {0, 50}}, "world"},
    };
    return page;
}

PagePrediction makeTestPrediction(bool perfectMatch) {
    PagePrediction pred;
    pred.image = "test.jpg";
    pred.elapsedMs = 123.0f;
    if (perfectMatch) {
        pred.lines = {
            {{{0, 0}, {50, 0}, {50, 20}, {0, 20}}, "hello", 0.95f},
            {{{0, 30}, {50, 30}, {50, 50}, {0, 50}}, "world", 0.92f},
        };
    }
    return pred;
}

} // namespace

TEST_CASE("evaluatePage with a perfect match has 2 TP, 0 FP/FN, and zero CER/WER") {
    auto page = makeTestPage();
    auto pred = makeTestPrediction(true);
    auto pm = evaluatePage(page, pred);

    CHECK(pm.image == "test.jpg");
    CHECK(pm.category == "BADGES");
    CHECK(pm.nGt == 2);
    CHECK(pm.nPred == 2);
    CHECK(pm.detection.tp == 2);
    CHECK(pm.detection.fp == 0);
    CHECK(pm.detection.fn == 0);
    REQUIRE(pm.matchedCer.size() == 2);
    CHECK(pm.matchedCer[0] == doctest::Approx(0.0f));
    CHECK(pm.matchedCer[1] == doctest::Approx(0.0f));
    CHECK(pm.joinedCer == doctest::Approx(0.0f));
    CHECK(pm.elapsedMs == doctest::Approx(123.0f));
    CHECK(pm.emptyOutput == false);
}

TEST_CASE("evaluatePage with an empty prediction has 0 TP, 0 FP, 2 FN, and flags empty_output") {
    auto page = makeTestPage();
    auto pred = makeTestPrediction(false);
    auto pm = evaluatePage(page, pred);

    CHECK(pm.nGt == 2);
    CHECK(pm.nPred == 0);
    CHECK(pm.detection.tp == 0);
    CHECK(pm.detection.fp == 0);
    CHECK(pm.detection.fn == 2);
    CHECK(pm.emptyOutput == true);
}

TEST_CASE("evaluatePage with a page that has no GT lines does not flag empty_output") {
    GroundTruthPage page;
    page.imagePath = "empty.jpg";
    page.category = "BADGES";
    page.lines = {};
    PagePrediction pred;
    pred.image = "empty.jpg";
    pred.elapsedMs = 10.0f;

    auto pm = evaluatePage(page, pred);
    CHECK(pm.nGt == 0);
    CHECK(pm.emptyOutput == false);
}

TEST_CASE("toPageMetricsInput carries the aggregation-relevant fields through") {
    auto page = makeTestPage();
    auto pred = makeTestPrediction(true);
    auto pm = evaluatePage(page, pred);
    auto input = toPageMetricsInput(pm);

    CHECK(input.nGt == pm.nGt);
    CHECK(input.detection.tp == pm.detection.tp);
    CHECK(input.matchedCer.size() == pm.matchedCer.size());
    CHECK(input.matchedWer.size() == pm.matchedWer.size());
    CHECK(input.matchedConf.size() == pm.matchedConf.size());
    CHECK(input.joinedCer == doctest::Approx(pm.joinedCer));
    CHECK(input.elapsedMs == doctest::Approx(pm.elapsedMs));
    CHECK(input.emptyOutput == pm.emptyOutput);
}
