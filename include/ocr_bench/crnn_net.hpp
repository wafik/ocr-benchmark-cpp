#pragma once
// Adapted from RapidAI/RapidOcrOnnx's CrnnNet.h/.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. scoreToTextLine/getTextLine are near-verbatim
// ports. loadKeysFromModelMetadata() is new: RapidOcrOnnx only supports
// loadKeysFromFile() (a separate keys.txt); this project's PP-OCRv6 models
// embed the dict in ONNX metadata instead (see engine.py's
// _rec_dict_path() in the Python reference for why).

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_bench/ocr_struct.hpp"

namespace ocr_bench {

class CrnnNet {
public:
    CrnnNet();
    ~CrnnNet();

    CrnnNet(const CrnnNet&) = delete;
    CrnnNet& operator=(const CrnnNet&) = delete;

    void loadModel(const std::string& modelPath, bool useCuda = false);

    /// RapidOcrOnnx's original keys-loading mechanism: one character per
    /// line in a plain text file. On success, prepends "#" (CTC blank) and
    /// appends " " (space), matching RapidOcrOnnx's exact layout. Missing
    /// file leaves keys empty (does not throw) so callers can detect
    /// failure via `keyCount() == 0` and try another source.
    void loadKeysFromFile(const std::string& keysPath);

    /// Reads the `character` field from the currently-loaded ONNX model's
    /// custom_metadata_map (this is how PP-OCRv6/rapidocr-v3 models embed
    /// their dict). Must be called after loadModel(). Applies the same
    /// "#" prefix + " " suffix convention as loadKeysFromFile() so the CTC
    /// decode logic is identical regardless of dict source. Returns false
    /// (leaving keys empty) if the model has no such metadata — callers
    /// should then fall back to loadKeysFromFile() with a bundled
    /// PP-OCRv4/v5-style keys.txt.
    bool loadKeysFromModelMetadata();

    /// Number of loaded keys (0 if neither load method has succeeded yet).
    size_t keyCount() const { return keys_.size(); }

    /// Recognize one cropped, angle-corrected text-line image per input Mat.
    std::vector<RawTextLine> getTextLines(std::vector<cv::Mat>& partImages);

    /// Test-only entry point: run the CTC decode directly on a raw output
    /// buffer without going through ONNXRuntime inference. Exposed so
    /// test_crnn_net.cpp can validate decode correctness without a real
    /// model (see Task 5 Step 2).
    RawTextLine decodeForTest(const std::vector<float>& outputData, size_t h, size_t w) const {
        return scoreToTextLine(outputData, h, w);
    }

private:
    RawTextLine scoreToTextLine(const std::vector<float>& outputData, size_t h, size_t w) const;
    RawTextLine getTextLine(const cv::Mat& src);
    void finalizeKeys(std::vector<std::string> rawKeys);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "CrnnNet"};
    Ort::SessionOptions sessionOptions_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;
    std::vector<std::string> keys_;

    static constexpr int kDstHeight = 48;
    const float meanValues_[3] = {127.5f, 127.5f, 127.5f};
    const float normValues_[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
};

} // namespace ocr_bench
