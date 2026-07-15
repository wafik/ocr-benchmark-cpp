#include "ocr_bench/paths.hpp"

#include <filesystem>

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
