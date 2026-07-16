// src/api/main.cpp — OCR Bench HTTP server entrypoint
#include <iostream>
#include <string>
#include <filesystem>

#include <cxxopts.hpp>

#include "ocr_bench/api_server.hpp"
#include "ocr_bench/paths.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    cxxopts::Options opts("ocr-bench-serve", "OCR Benchmark HTTP Dashboard Server");
    opts.add_options()
        ("host", "Listen host", cxxopts::value<std::string>()->default_value("127.0.0.1"))
        ("port", "Listen port", cxxopts::value<int>()->default_value("8765"))
        ("ui-root", "Path to ui/ directory for static files", cxxopts::value<std::string>()->default_value(""))
        ("reports-root", "Path to reports/ directory", cxxopts::value<std::string>()->default_value(""))
        ("auth-password", "Basic auth password (empty = no auth)", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    // Resolve paths relative to package root (cpp/rebuild/)
    std::string pkgRoot = ocr_bench::packageRoot();

    ocr_bench::ApiServerConfig cfg;
    cfg.host = result["host"].as<std::string>();
    cfg.port = result["port"].as<int>();
    cfg.authPassword = result["auth-password"].as<std::string>();

    // Resolve UI root
    std::string uiRoot = result["ui-root"].as<std::string>();
    if (uiRoot.empty()) {
        cfg.uiRoot = (fs::path(pkgRoot) / "ui").string();
    } else if (fs::path(uiRoot).is_relative()) {
        cfg.uiRoot = (fs::path(pkgRoot) / uiRoot).string();
    } else {
        cfg.uiRoot = uiRoot;
    }

    // Resolve reports root
    std::string reportsRoot = result["reports-root"].as<std::string>();
    if (reportsRoot.empty()) {
        cfg.reportsRoot = (fs::path(pkgRoot) / "reports").string();
    } else if (fs::path(reportsRoot).is_relative()) {
        cfg.reportsRoot = (fs::path(pkgRoot) / reportsRoot).string();
    } else {
        cfg.reportsRoot = reportsRoot;
    }

    std::cout << "OCR Bench serving on http://" << cfg.host << ":" << cfg.port << std::endl;
    std::cout << "  UI: " << cfg.uiRoot << std::endl;
    std::cout << "  Reports: " << cfg.reportsRoot << std::endl;
    if (!cfg.authPassword.empty()) {
        std::cout << "  Auth: enabled" << std::endl;
    }

    ocr_bench::ApiServer server(cfg);
    server.listen();
    return 0;
}
