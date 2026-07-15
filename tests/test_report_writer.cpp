#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include "ocr_bench/report_writer.hpp"

using namespace ocr_bench;
using json = nlohmann::json;

TEST_CASE("aggregateCategory with no pages returns zeroed summary") {
    auto summary = aggregateCategory("EMPTY", {});
    CHECK(summary.category == "EMPTY");
    CHECK(summary.nImages == 0);
    CHECK(summary.nLinesTotal == 0);
    CHECK(summary.matchedCerMean == doctest::Approx(0.0f));
}

TEST_CASE("aggregateCategory computes mean/median CER and detection totals") {
    PageMetricsInput page1;
    page1.nGt = 2;
    page1.detection = DetectionStats{2, 0, 0};
    page1.matchedCer = {0.1f, 0.3f};
    page1.matchedWer = {0.2f};
    page1.matchedConf = {0.9f, 0.8f};
    page1.joinedCer = 0.15f;
    page1.elapsedMs = 100.0f;
    page1.emptyOutput = false;

    PageMetricsInput page2;
    page2.nGt = 1;
    page2.detection = DetectionStats{1, 1, 0};
    page2.matchedCer = {0.2f};
    page2.matchedWer = {0.4f};
    page2.matchedConf = {0.7f};
    page2.joinedCer = 0.2f;
    page2.elapsedMs = 200.0f;
    page2.emptyOutput = false;

    std::vector<PageMetricsInput> pages = {page1, page2};
    auto summary = aggregateCategory("TESTCAT", pages);
    CHECK(summary.nImages == 2);
    CHECK(summary.nLinesTotal == 3);
    CHECK(summary.detection.tp == 3);
    CHECK(summary.detection.fp == 1);
    CHECK(summary.detection.fn == 0);
    // matched_cer = [0.1, 0.3, 0.2] -> mean = 0.2
    CHECK(summary.matchedCerMean == doctest::Approx(0.2f));
    // elapsed_ms = [100, 200] -> mean = 150
    CHECK(summary.meanMsPerImage == doctest::Approx(150.0f));
}

TEST_CASE("toOverallJson matches Python's summary.json field names") {
    std::vector<CategorySummary> cats = {
        {"BADGES", 5, 100, DetectionStats{80, 10, 20}, 0.05f, 0.04f, 0.2f, 0.06f, 0.9f, 500.0f, 0.0f},
    };
    json overallExtra = {{"n_images", 5}, {"n_lines", 100}, {"detection_f1", 0.842f}};
    json out = toOverallJson(cats, overallExtra);

    REQUIRE(out.contains("overall"));
    REQUIRE(out.contains("per_category"));
    REQUIRE(out["per_category"].size() == 1);
    auto& c = out["per_category"][0];
    CHECK(c["category"] == "BADGES");
    CHECK(c["n_images"] == 5);
    CHECK(c["n_lines"] == 100);
    CHECK(c.contains("detection"));
    CHECK(c["detection"]["tp"] == 80);
    CHECK(c["cer_mean"] == doctest::Approx(0.05));
    CHECK(c["cer_median"] == doctest::Approx(0.04));
    CHECK(c["wer_mean"] == doctest::Approx(0.2));
    CHECK(c["joined_cer_mean"] == doctest::Approx(0.06));
    CHECK(c["mean_confidence"] == doctest::Approx(0.9));
    CHECK(c["mean_ms_per_image"] == doctest::Approx(500.0));
    CHECK(c["empty_output_rate"] == doctest::Approx(0.0));
    // No corrected-variant keys anywhere.
    CHECK_FALSE(c.contains("cer_corrected_mean"));
    CHECK_FALSE(c.contains("wer_corrected_mean"));
}

TEST_CASE("writeSummaryCsv writes header + one row per category + OVERALL row") {
    std::vector<CategorySummary> cats = {
        {"BADGES", 5, 100, DetectionStats{80, 10, 20}, 0.05f, 0.04f, 0.2f, 0.06f, 0.9f, 500.0f, 0.0f},
    };
    json overall = {
        {"n_images", 5}, {"n_lines", 100},
        {"detection_precision", 0.888f}, {"detection_recall", 0.8f}, {"detection_f1", 0.842f},
        {"cer_mean", 0.05f}, {"wer_mean", 0.2f},
    };
    std::string outPath = "test_summary_output.csv";
    writeSummaryCsv(outPath, cats, overall);

    std::ifstream in(outPath);
    std::stringstream buf;
    buf << in.rdbuf();
    std::string content = buf.str();
    CHECK(content.find("category,n_images,n_lines,precision,recall,f1") != std::string::npos);
    CHECK(content.find("BADGES,5,100") != std::string::npos);
    CHECK(content.find("OVERALL,5,100") != std::string::npos);
}
