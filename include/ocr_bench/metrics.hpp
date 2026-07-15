#pragma once

#include <string>

namespace ocr_bench {

/// Lowercase + whitespace-collapse + strip. ASCII-fold normalization
/// (sufficient for Indonesian Latin script — see Global Constraints).
std::string normalize(const std::string& text);

/// Character error rate (edit distance / ref length), in [0, inf).
/// 0.0 means perfect match. Empty ref with non-empty hyp = 1.0.
float cer(const std::string& ref, const std::string& hyp);

/// Word error rate (word-level edit distance / ref word count), in [0, inf).
float wer(const std::string& ref, const std::string& hyp);

struct DetectionStats {
    int tp = 0;
    int fp = 0;
    int fn = 0;

    float precision() const {
        return (tp + fp) > 0 ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    }
    float recall() const {
        return (tp + fn) > 0 ? static_cast<float>(tp) / (tp + fn) : 0.0f;
    }
    float f1() const {
        float p = precision();
        float r = recall();
        return (p + r) > 0.0f ? 2 * p * r / (p + r) : 0.0f;
    }
};

} // namespace ocr_bench
