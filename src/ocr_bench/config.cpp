#include "ocr_bench/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>

namespace ocr_bench {

namespace {

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool toBool(const std::string& v) {
    std::string lower = v;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "true" || lower == "1" || lower == "yes";
}

std::map<std::string, std::string> parseEnvFile(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in.is_open()) return kv; // missing file -> empty map -> all defaults
    std::string line;
    while (std::getline(in, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        auto eq = trimmed.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(trimmed.substr(0, eq));
        std::string value = trim(trimmed.substr(eq + 1));
        kv[key] = value;
    }
    return kv;
}

} // namespace

Settings loadSettings(const std::string& envFilePath) {
    Settings s; // start from defaults
    auto kv = parseEnvFile(envFilePath);

    auto getStr = [&](const char* key, std::string& target) {
        auto it = kv.find(key);
        if (it != kv.end()) target = it->second;
    };
    auto getFloat = [&](const char* key, float& target) {
        auto it = kv.find(key);
        if (it != kv.end()) { try { target = std::stof(it->second); } catch (...) {} }
    };
    auto getInt = [&](const char* key, int& target) {
        auto it = kv.find(key);
        if (it != kv.end()) { try { target = std::stoi(it->second); } catch (...) {} }
    };
    auto getBool = [&](const char* key, bool& target) {
        auto it = kv.find(key);
        if (it != kv.end()) target = toBool(it->second);
    };

    getStr("OCR_VERSION", s.ocrVersion);
    getStr("MODEL_TYPE", s.modelType);
    getFloat("DET_BOX_THRESH", s.detBoxThresh);
    getFloat("DET_THRESH", s.detThresh);
    getFloat("DET_UNCLIP_RATIO", s.detUnclipRatio);
    getInt("DET_LIMIT_SIDE_LEN", s.detLimitSideLen);
    getBool("USE_ANGLE_CLS", s.useAngleCls);
    getInt("REC_BATCH_NUM", s.recBatchNum);
    getInt("REC_IMG_WIDTH", s.recImgWidth);
    getBool("USE_TENSORRT", s.useTensorrt);
    getStr("TRT_CACHE_DIR", s.trtCacheDir);
    getFloat("IOU_THRESHOLD", s.iouThreshold);
    getBool("ENABLE_PREPROCESSING", s.enablePreprocessing);
    getInt("PREPROC_UPSCALE_MIN_SIDE", s.preprocUpscaleMinSide);
    getStr("SERVE_HOST", s.serveHost);
    getInt("SERVE_PORT", s.servePort);
    getStr("AUTH_PASSWORD", s.authPassword);
    getStr("OCR_BENCH_DATASET", s.ocrBenchDataset);
    getStr("PIPER_VOICE_PATH", s.piperVoicePath);
    getStr("TTS_SOURCE", s.ttsSource);
    getBool("USE_CUDA_TTS", s.useCudaTts);

    return s;
}

} // namespace ocr_bench
