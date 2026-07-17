#include "ocr_bench/paths.hpp"

#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace ocr_bench {

std::string packageRoot() {
    // This file lives at <repoRoot>/cpp/rebuild/src/ocr_bench/paths.cpp
    // Mirrors Python's `Path(__file__).resolve().parent.parent.parent`
    // (src/ocr_bench/paths.py -> src/ocr_bench/ -> src/ -> package root),
    // adjusted for this project's one-extra-level nesting (cpp/rebuild/src/ocr_bench/).
    fs::path here = fs::path(__FILE__).parent_path();       // src/ocr_bench
    return here.parent_path().parent_path().string();       // cpp/rebuild
}

std::string executableDir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path().string();
#else
    // Linux: /proc/self/exe
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    return fs::path(buf).parent_path().string();
#endif
}

std::string ttsWorkerBinary() {
    std::string name =
#ifdef _WIN32
        "ocr-bench-tts-worker.exe";
#else
        "ocr-bench-tts-worker";
#endif
    fs::path workerPath = fs::path(executableDir()) / name;
    if (fs::exists(workerPath)) {
        return workerPath.string();
    }
    // Fallback: look in packageRoot build dir
    fs::path buildPath = fs::path(packageRoot()) / "build" / "jetson" / name;
    if (fs::exists(buildPath)) {
        return buildPath.string();
    }
    return workerPath.string(); // let caller handle missing file
}

const std::map<std::string, std::string>& datasetRegistry() {
    static const std::map<std::string, std::string> registry = {
        {"ind_cn", (fs::path(packageRoot()) / "IMG_OCR_IND_CN").string()},
        {"new",    (fs::path(packageRoot()) / "dataset" / "dataset").string()},
    };
    return registry;
}

ResolvedDataset resolveDatasetRoot(const std::string& keyOrPath, const std::string& envOverride) {
    std::string raw = !keyOrPath.empty() ? keyOrPath
                     : !envOverride.empty() ? envOverride
                     : kDefaultDatasetKey;
    const auto& registry = datasetRegistry();
    auto it = registry.find(raw);
    if (it != registry.end()) {
        return {raw, it->second};
    }
    if (!raw.empty() && raw != kDefaultDatasetKey) {
        return {raw, raw}; // literal path override
    }
    return {kDefaultDatasetKey, registry.at(kDefaultDatasetKey)};
}

} // namespace ocr_bench
