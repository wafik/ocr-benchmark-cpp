// cli/run_benchmark.cpp — thin wrapper around ocr_bench::run()
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "ocr_bench/runner.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options opts("ocr-bench-run", "Run OCR benchmark");
    opts.add_options()
        ("category", "Category to evaluate (repeatable)", cxxopts::value<std::vector<std::string>>())
        ("ocr-version", "OCR model version", cxxopts::value<std::string>()->default_value(""))
        ("model-type", "Model size (tiny/small/medium)", cxxopts::value<std::string>()->default_value(""))
        ("dataset", "Dataset key or path", cxxopts::value<std::string>()->default_value(""))
        ("iou-threshold", "IoU threshold for matching", cxxopts::value<float>())
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    ocr_bench::RunOptions runOpts;
    if (result.count("category")) {
        runOpts.onlyCategories = result["category"].as<std::vector<std::string>>();
    }
    runOpts.ocrVersion = result["ocr-version"].as<std::string>();
    runOpts.modelType = result["model-type"].as<std::string>();
    runOpts.datasetKey = result["dataset"].as<std::string>();
    if (result.count("iou-threshold")) {
        runOpts.overrides.iouThreshold = result["iou-threshold"].as<float>();
    }
    runOpts.verbose = true;

    try {
        auto overall = ocr_bench::run(runOpts);
        std::cout << overall.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
