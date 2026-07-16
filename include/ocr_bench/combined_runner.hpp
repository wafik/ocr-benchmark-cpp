#pragma once
// Combined OCR + TTS benchmark runner. Port of combined_runner.py.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ocr_bench/runner.hpp"
#include "ocr_bench/tts_runner.hpp"

namespace ocr_bench {

struct CombinedRunOptions {
    RunOptions ocrOptions;
    TTSRunOptions ttsOptions;
    bool verbose = true;
};

/// Run combined OCR + TTS benchmark. Returns the overall JSON summary.
nlohmann::json combinedRun(const CombinedRunOptions& options);

} // namespace ocr_bench
