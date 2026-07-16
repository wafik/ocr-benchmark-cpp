#pragma once
// Adapted from RapidAI/RapidOcrOnnx (Apache-2.0) — see THIRD_PARTY_NOTICES.md.
// Renamed to this project's conventions; struct shapes match the original.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace ocr_bench {

struct ScaleParam {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    float ratioWidth;
    float ratioHeight;
};

// One detected text box before recognition (DbNet output).
struct RawTextBox {
    std::vector<cv::Point> boxPoint; // 4 points, clockwise from top-left-ish
    float score;
};

// AngleNet classification result for one cropped text-line image.
struct RawAngle {
    int index;  // 0 or 1 (0deg / 180deg); -1 if angle classification was skipped
    float score;
};

// CrnnNet recognition result for one cropped text-line image.
struct RawTextLine {
    std::string text;
    std::vector<float> charScores;
};

} // namespace ocr_bench
