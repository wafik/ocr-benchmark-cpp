#pragma once
// Ports Python's runner.py. Covers per-page evaluation (Task 3) and
// per-image JSON serialization (Task 4). The run() orchestrator (Task 5)
// will be added in a follow-up.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ocr_bench/dataset.hpp"
#include "ocr_bench/engine.hpp"
#include "ocr_bench/metrics.hpp"
#include "ocr_bench/report_writer.hpp"

namespace ocr_bench {

// Mirrors Python's PageMetrics dataclass minus every corrector field.
struct PageMetrics {
    std::string image;
    std::string category;
    int nGt = 0;
    int nPred = 0;
    DetectionStats detection;
    std::vector<float> matchedCer;
    std::vector<float> matchedWer;
    std::vector<float> matchedConf;
    float joinedCer = 0.0f;
    float elapsedMs = 0.0f;
    bool emptyOutput = false;
};

/// Match GT lines against predicted lines via IoU, compute per-matched-pair
/// CER/WER, joined-page CER, and the empty-output flag. Mirrors
/// runner.py::_evaluate_page().
PageMetrics evaluatePage(const GroundTruthPage& page, const PagePrediction& pred, float iouThreshold = 0.5f);

/// Adapter: extracts the subset of PageMetrics that Plan 1's
/// aggregateCategory() needs.
PageMetricsInput toPageMetricsInput(const PageMetrics& pm);

/// Serialize per-page metrics into the JSON shape that goes into
/// reports/per_category/*.json under "images": [...]. Mirrors
/// runner.py::_serialize_page_metrics().
nlohmann::json serializePageMetrics(const PageMetrics& pm, const GroundTruthPage& page, const PagePrediction& pred);

} // namespace ocr_bench
