#pragma once

#include <string>
#include <vector>

namespace ocr_bench {

struct DownloadResult {
    bool ok = false;
    std::string errorMessage;
    size_t bytesWritten = 0;
};

DownloadResult downloadFile(const std::string& url, const std::string& destPath);

std::vector<DownloadResult> downloadOcrModels(
    const std::string& ocrVersion,
    const std::string& modelType,
    const std::string& modelsDir
);

} // namespace ocr_bench
