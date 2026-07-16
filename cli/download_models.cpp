// cli/download_models.cpp — thin wrapper around ocr_bench::downloadOcrModels()
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "ocr_bench/model_downloader.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options opts("ocr-bench-download-models", "Download OCR models");
    opts.add_options()
        ("ocr-version", "OCR version (e.g. PP-OCRv6)", cxxopts::value<std::string>()->default_value("PP-OCRv6"))
        ("model-type", "Model type (tiny/small/medium)", cxxopts::value<std::string>()->default_value("tiny"))
        ("models-dir", "Directory to save models", cxxopts::value<std::string>()->default_value("models"))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    std::string version = result["ocr-version"].as<std::string>();
    std::string modelType = result["model-type"].as<std::string>();
    std::string modelsDir = result["models-dir"].as<std::string>();

    std::cout << "Downloading " << version << " " << modelType << " models to " << modelsDir << "..." << std::endl;

    auto results = ocr_bench::downloadOcrModels(version, modelType, modelsDir);
    bool allOk = true;
    for (size_t i = 0; i < results.size(); i++) {
        std::string label = (i == 0) ? "det" : (i == 1) ? "cls" : "rec";
        if (results[i].ok) {
            std::cout << "  " << label << ": " << (results[i].bytesWritten > 0 ? "downloaded" : "already present")
                      << " (" << results[i].bytesWritten << " bytes)" << std::endl;
        } else {
            std::cerr << "  " << label << ": FAILED - " << results[i].errorMessage << std::endl;
            allOk = false;
        }
    }
    return allOk ? 0 : 1;
}
