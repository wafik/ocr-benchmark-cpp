#pragma once
// Orchestrates DbNet + AngleNet + CrnnNet end-to-end, mirroring both
// RapidOcrOnnx's OcrLiteImpl::detect() (box decode -> crop -> angle ->
// recognize pipeline) and this project's Python reference engine.py
// (BenchEngine class, LinePrediction/PagePrediction shapes). See
// THIRD_PARTY_NOTICES.md for the RapidOcrOnnx attribution covering the
// pipeline structure this orchestration follows.

#include <string>
#include <vector>

#include "ocr_bench/angle_net.hpp"
#include "ocr_bench/crnn_net.hpp"
#include "ocr_bench/db_net.hpp"
#include "ocr_bench/matcher.hpp" // for ocr_bench::Polygon

namespace ocr_bench {

struct LinePrediction {
    Polygon polygon;
    std::string text;
    float score = 0.0f;
};

struct PagePrediction {
    std::string image;
    std::vector<LinePrediction> lines;
    float elapsedMs = 0.0f;
};

struct EngineConfig {
    std::string ocrVersion = "PP-OCRv6";
    std::string modelType = "tiny";
    float detBoxThresh = 0.5f;
    float detThresh = 0.3f;
    float detUnclipRatio = 1.6f;
    int detLimitSideLen = 1536;
    bool useAngleCls = false;
    bool useCuda = false;
    bool useTensorrt = false;
    std::string trtCacheDir = "models/trt_engines";
    std::string modelsDir = "models";
};

/// Auto-detect CUDA execution provider availability. Mirrors
/// engine.py::_detect_cuda().
bool detectCuda();

/// Auto-detect TensorRT execution provider availability. Mirrors
/// engine.py::_detect_tensorrt().
bool detectTensorrt();

class BenchEngine {
public:
    explicit BenchEngine(const EngineConfig& config);

    /// "cuda" or "cpu" — mirrors Python's self.backend attribute.
    std::string backend() const { return backend_; }

    /// Run the full det -> crop -> angle -> recognize pipeline on one
    /// image. Never throws: image-read failures and inference exceptions
    /// both degrade to an empty-lines PagePrediction with elapsedMs set.
    PagePrediction predict(const std::string& imagePath);

private:
    DbNet dbNet_;
    AngleNet angleNet_;
    CrnnNet crnnNet_;
    EngineConfig config_;
    std::string backend_ = "cpu";
};

} // namespace ocr_bench
