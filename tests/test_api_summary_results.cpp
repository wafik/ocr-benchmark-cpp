#include <doctest/doctest.h>
#include <httplib.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "ocr_bench/api_server.hpp"

using namespace ocr_bench;
namespace fs = std::filesystem;

namespace {
struct ResultsTestFixture {
    ApiServer* server;
    std::thread serverThread;
    int port;
    std::string reportsRoot;

    ResultsTestFixture(const std::string& root) : reportsRoot(root) {
        fs::create_directories(root + "/per_category");
        ApiServerConfig cfg;
        cfg.uiRoot = "";
        cfg.reportsRoot = root;
        cfg.authPassword = "";
        cfg.port = 0;
        server = new ApiServer(cfg);
        serverThread = std::thread([this]() { server->listen(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        port = server->boundPort();
    }
    ~ResultsTestFixture() {
        server->stop();
        serverThread.join();
        delete server;
        fs::remove_all(reportsRoot);
    }
};
}

TEST_CASE("GET /api/summary returns 404 when no report exists yet") {
    ResultsTestFixture fixture("tests/fixtures/api_tmp/summary_missing");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/summary");
    REQUIRE(res);
    CHECK(res->status == 404);
}

TEST_CASE("GET /api/summary returns the file's contents when present") {
    std::string root = "tests/fixtures/api_tmp/summary_present";
    fs::create_directories(root);
    {
        std::ofstream f(root + "/summary.json");
        f << R"({"overall": {"n_images": 5, "last_run": "2026-01-01T00:00:00Z"}, "per_category": []})";
    }
    ResultsTestFixture fixture(root);
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/summary");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("n_images") != std::string::npos);
    CHECK(res->body.find("5") != std::string::npos);
}

TEST_CASE("GET /api/results/{category} finds a matching per_category file") {
    std::string root = "tests/fixtures/api_tmp/results_found";
    fs::create_directories(root + "/per_category");
    {
        std::ofstream f(root + "/per_category/badges.json");
        f << R"({"category": "BADGES", "images": []})";
    }
    ResultsTestFixture fixture(root);
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/results/BADGES");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"category\":\"BADGES\"") != std::string::npos);
}

TEST_CASE("GET /api/results/{category} returns 404 for an unknown category") {
    ResultsTestFixture fixture("tests/fixtures/api_tmp/results_missing");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/results/NOT_A_REAL_CATEGORY");
    REQUIRE(res);
    CHECK(res->status == 404);
}

TEST_CASE("GET /api/models lists only PP-OCRv6 with tiny/small/medium types") {
    ResultsTestFixture fixture("tests/fixtures/api_tmp/models");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/models");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("PP-OCRv6") != std::string::npos);
    CHECK(res->body.find("tiny") != std::string::npos);
}

TEST_CASE("GET /api/config omits every corrector field") {
    ResultsTestFixture fixture("tests/fixtures/api_tmp/config");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/config");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("enable_symspell_correction") == std::string::npos);
    CHECK(res->body.find("kbbi") == std::string::npos);
    CHECK(res->body.find("ocr_version") != std::string::npos);
}
