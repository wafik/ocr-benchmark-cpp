#pragma once

#include <string>

namespace ocr_bench {

// Mirrors Python's Settings class (config.py) minus every corrector field
// (enable_symspell_correction, enable_word_segmentation,
// symspell_max_edit_distance, kbbi_top_n, kbbi_csv_path — all dropped,
// per this rewrite's decision to remove the corrector entirely).
struct Settings {
    // OCR model selection
    std::string ocrVersion = "PP-OCRv6";
    std::string modelType = "tiny";

    // Detection knobs
    float detBoxThresh = 0.5f;
    float detThresh = 0.3f;
    float detUnclipRatio = 1.6f;
    int detLimitSideLen = 1536;

    // Angle classifier
    bool useAngleCls = false;

    // Recognition knobs
    int recBatchNum = 6;
    int recImgWidth = 320;

    // Inference backend
    bool useTensorrt = false;
    std::string trtCacheDir = "models/trt_engines";

    // Runner
    float iouThreshold = 0.5f;
    bool enablePreprocessing = false;
    int preprocUpscaleMinSide = 800;

    // Server
    std::string serveHost = "127.0.0.1";
    int servePort = 8765;
    std::string authPassword = "AI4DB-BENCH";

    // Dataset
    std::string ocrBenchDataset = "ind_cn";

    // TTS
    std::string piperVoicePath = "models/piper-voices/id/id_ID-news_tts-medium.onnx";
    std::string ttsSource = "pred";
    bool useCudaTts = true;
};

/// Read a `.env`-format file (`KEY=value` per line, `#` comments allowed).
/// Missing file or unreadable file returns all-defaults — never throws.
Settings loadSettings(const std::string& envFilePath = ".env");

} // namespace ocr_bench
