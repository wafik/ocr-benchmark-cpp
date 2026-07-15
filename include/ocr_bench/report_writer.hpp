#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ocr_bench/metrics.hpp"

namespace ocr_bench {

// Minimal per-page data needed for category aggregation. This is the C++
// analog of Python's PageMetrics, minus every corrector-related field.
struct PageMetricsInput {
    int nGt = 0;
    DetectionStats detection;
    std::vector<float> matchedCer;
    std::vector<float> matchedWer;
    std::vector<float> matchedConf;
    float joinedCer = 0.0f;
    float elapsedMs = 0.0f;
    bool emptyOutput = false;
};

// Mirrors Python's CategorySummary minus all `*_corrected` fields
// (corrector dropped entirely per this rewrite's decision).
struct CategorySummary {
    std::string category;
    int nImages = 0;
    int nLinesTotal = 0;
    DetectionStats detection;
    float matchedCerMean = 0.0f;
    float matchedCerMedian = 0.0f;
    float matchedWerMean = 0.0f;
    float joinedCerMean = 0.0f;
    float meanConfidence = 0.0f;
    float meanMsPerImage = 0.0f;
    float emptyOutputRate = 0.0f;
};

/// Aggregate per-page metrics into one category summary. Empty input
/// returns a zeroed summary with the given category name.
CategorySummary aggregateCategory(const std::string& category, const std::vector<PageMetricsInput>& pages);

/// Build the `summary.json` document shape: {"overall": overallExtra
/// (merged with resources if present), "per_category": [...]}.
/// `overallExtra` is passed straight through under the "overall" key —
/// callers (the runner, in a later plan) populate it with aggregate_overall()
/// output plus run config fields, mirroring Python's overall_dict.
nlohmann::json toOverallJson(const std::vector<CategorySummary>& perCategory, const nlohmann::json& overallExtra);

/// Write the summary.csv file: header + one row per category + one OVERALL
/// row, in the same 16-column order as Python's _write_summary_csv().
/// The four corrected-metric columns are written as reserved placeholders
/// but the header keeps only the columns that still have real data —
/// see the implementation for the exact 16-column layout.
void writeSummaryCsv(const std::string& path, const std::vector<CategorySummary>& perCategory, const nlohmann::json& overall);

} // namespace ocr_bench
