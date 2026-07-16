// src/api/main.cpp — OCR Bench HTTP server entrypoint
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "ocr_bench/api_server.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options opts("ocr-bench-serve", "OCR Benchmark HTTP Dashboard Server");
    opts.add_options()
        ("host", "Listen host", cxxopts::value<std::string>()->default_value("127.0.0.1"))
        ("port", "Listen port", cxxopts::value<int>()->default_value("8765"))
        ("ui-root", "Path to ui/ directory for static files", cxxopts::value<std::string>()->default_value("ui"))
        ("reports-root", "Path to reports/ directory", cxxopts::value<std::string>()->default_value("reports"))
        ("auth-password", "Basic auth password (empty = no auth)", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    ocr_bench::ApiServerConfig cfg;
    cfg.host = result["host"].as<std::string>();
    cfg.port = result["port"].as<int>();
    cfg.uiRoot = result["ui-root"].as<std::string>();
    cfg.reportsRoot = result["reports-root"].as<std::string>();
    cfg.authPassword = result["auth-password"].as<std::string>();

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
