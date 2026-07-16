#pragma once
// Ports Python's runner.py. Covers per-page evaluation (Tasks 3-4) and
// the run() orchestrator (Task 5).

#include <optional>
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
/// CER/WER, joined-page CER, and the empty-output flag.
PageMetrics evaluatePage(const GroundTruthPage& page, const PagePrediction& pred, float iouThreshold = 0.5f);

/// Adapter: extracts the subset of PageMetrics that aggregateCategory() needs.
PageMetricsInput toPageMetricsInput(const PageMetrics& pm);

/// Serialize per-page metrics into the JSON shape for reports/per_category/*.json.
nlohmann::json serializePageMetrics(const PageMetrics& pm, const GroundTruthPage& page, const PagePrediction& pred);

// Mirrors Python's det_overrides dict — corrector overrides omitted.
struct RunOverrides {
    std::optional<float> detBoxThresh;
    std::optional<float> detThresh;
    std::optional<float> detUnclipRatio;
    std::optional<int> detLimitSideLen;
    std::optional<bool> useAngleCls;
    std::optional<bool> useTensorrt;
    std::optional<bool> enablePreprocessing;
    std::optional<float> iouThreshold;
};

struct RunOptions {
    std::vector<std::string> onlyCategories; // empty = all
    std::string ocrVersion;   // empty = use Settings default
    std::string modelType;    // empty = use Settings default
    RunOverrides overrides;
    std::string datasetKey;   // empty = use Settings/env default
    bool verbose = true;
};

/// Progress sidecar path, matching Python's REPORTS_ROOT / ".run_status.json".
extern const std::string kRunStatusPath;

/// Run the full benchmark: resolve dataset -> build engine -> iterate
/// categories -> evaluate + serialize each page -> write reports.
/// Returns the overall JSON summary.
nlohmann::json run(const RunOptions& options);

} // namespace ocr_bench
