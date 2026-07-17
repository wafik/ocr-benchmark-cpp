// cli/trt_cache.cpp — Pre-build TensorRT engines for all model types.
// Run once after first build to avoid slow first-time TRT engine compilation.
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <filesystem>
#include "ocr_bench/engine.hpp"
#include "ocr_bench/config.hpp"
#include "ocr_bench/paths.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options opts("ocr-bench-trt-cache", "Pre-build TensorRT engines for all model types");
    opts.add_options()
        ("ocr-version", "OCR model version", cxxopts::value<std::string>()->default_value("PP-OCRv6"))
        ("types", "Comma-separated model types to cache", cxxopts::value<std::string>()->default_value("tiny,small,medium"))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    std::string ver = result["ocr-version"].as<std::string>();
    std::string typesStr = result["types"].as<std::string>();

    std::vector<std::string> types;
    {
        std::istringstream ss(typesStr);
        std::string t;
        while (std::getline(ss, t, ',')) {
            if (!t.empty()) types.push_back(t);
        }
    }

    namespace fs = std::filesystem;
    std::string root = ocr_bench::packageRoot();
    std::string modelsDir = (fs::path(root) / "models").string();
    std::string trtCacheDir = (fs::path(root) / "models" / "trt_engines").string();
    fs::create_directories(trtCacheDir);

    auto settings = ocr_bench::loadSettings();

    std::cout << "Pre-building TensorRT engines..." << std::endl;
    std::cout << "  models: " << modelsDir << std::endl;
    std::cout << "  cache:  " << trtCacheDir << std::endl;
    std::cout << "  types:  " << typesStr << std::endl << std::endl;

    int ok = 0, fail = 0;
    for (auto& type : types) {
        std::cout << "[" << ver << " / " << type << "] Loading models..." << std::flush;

        ocr_bench::EngineConfig cfg;
        cfg.ocrVersion = ver;
        cfg.modelType = type;
        cfg.modelsDir = modelsDir;
        cfg.trtCacheDir = trtCacheDir;
        cfg.useTensorrt = true;
        cfg.useCuda = true;
        cfg.useAngleCls = settings.useAngleCls;

        try {
            ocr_bench::BenchEngine engine(cfg);
            std::cout << " OK" << std::endl;
            ok++;
        } catch (const std::exception& e) {
            std::cout << " FAILED: " << e.what() << std::endl;
            fail++;
        }
    }

    std::cout << std::endl << "Done: " << ok << " cached, " << fail << " failed." << std::endl;
    return fail > 0 ? 1 : 0;
}
