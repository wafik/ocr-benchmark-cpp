#pragma once
// Adapted from RapidAI/RapidOcrOnnx's AngleNet.h/.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md.

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_bench/ocr_struct.hpp"

namespace ocr_bench {

class AngleNet {
public:
    AngleNet();
    ~AngleNet();

    AngleNet(const AngleNet&) = delete;
    AngleNet& operator=(const AngleNet&) = delete;

    void loadModel(const std::string& modelPath, bool useCuda = false);

    /// Classify each cropped text-line image's orientation (0 = upright,
    /// 1 = 180-degree rotated). If `doAngle` is false, returns index=-1 for
    /// every image without touching the model (mirrors Python's
    /// `use_angle_cls` toggle semantics — see engine.py comment on why
    /// rapidocr v3 can't actually disable this stage, which this port CAN,
    /// unlike the Python wrapper).
    /// If `mostAngle` is true, all images are forced to the majority-vote
    /// angle across the batch (reduces flicker on noisy per-line angle
    /// classification for a single scanned page).
    std::vector<RawAngle> getAngles(std::vector<cv::Mat>& partImages, bool doAngle, bool mostAngle);

private:
    RawAngle getAngle(cv::Mat& src);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "AngleNet"};
    Ort::SessionOptions sessionOptions_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;

    static constexpr int kDstWidth = 192;
    static constexpr int kDstHeight = 48;
    const float meanValues_[3] = {127.5f, 127.5f, 127.5f};
    const float normValues_[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
};

} // namespace ocr_bench
