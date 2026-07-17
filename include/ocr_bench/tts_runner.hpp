#pragma once
// TTS benchmark runner. Port of tts_runner.py.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ocr_bench {

struct TTSRunOptions {
    std::string datasetKey;
    std::string voicePath;    // empty = use Settings default
    bool useCuda = true;      // default: use CUDA if available, fall back to CPU
    bool verbose = true;
    int batchSize = 1;        // lines per batch (1 = no batching)
    int numWorkers = 0;       // 0 = sequential, N = parallel worker processes
};

/// Run TTS benchmark: synthesize all GT text lines, measure RTF.
/// Returns the overall JSON summary.
nlohmann::json ttsRun(const TTSRunOptions& options);

} // namespace ocr_bench
