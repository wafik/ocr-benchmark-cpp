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
struct TtsTestFixture {
    ApiServer* server;
    std::thread serverThread;
    int port;
    std::string reportsRoot;

    TtsTestFixture(const std::string& root) : reportsRoot(root) {
        fs::create_directories(root);
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
    ~TtsTestFixture() {
        server->stop();
        serverThread.join();
        delete server;
        fs::remove_all(reportsRoot);
    }
};
}

TEST_CASE("GET /api/tts with no voice configured returns 503") {
    TtsTestFixture fixture("tests/fixtures/api_tmp/tts_missing_voice");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/tts?text=hello");
    REQUIRE(res);
    CHECK(res->status == 503);
}

TEST_CASE("GET /api/tts/summary returns 404 when no TTS report exists yet") {
    TtsTestFixture fixture("tests/fixtures/api_tmp/tts_summary_missing");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/tts/summary");
    REQUIRE(res);
    CHECK(res->status == 404);
}

TEST_CASE("GET /api/tts/progress with no run yet returns running:false") {
    TtsTestFixture fixture("tests/fixtures/api_tmp/tts_progress_idle");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/tts/progress");
    REQUIRE(res);
    CHECK(res->body.find("\"running\":false") != std::string::npos);
}

TEST_CASE("GET /api/combined/summary returns 404 when no combined report exists yet") {
    TtsTestFixture fixture("tests/fixtures/api_tmp/combined_summary_missing");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/combined/summary");
    REQUIRE(res);
    CHECK(res->status == 404);
}

TEST_CASE("GET /api/combined/history with no history returns an empty runs array") {
    TtsTestFixture fixture("tests/fixtures/api_tmp/combined_history_empty");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/combined/history");
    REQUIRE(res);
    CHECK(res->body.find("\"runs\":[]") != std::string::npos);
}

TEST_CASE("GET /api/combined/history/{id} sanitizes path-traversal attempts") {
    TtsTestFixture fixture("tests/fixtures/api_tmp/combined_history_traversal");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/combined/history/../../../etc/passwd");
    REQUIRE(res);
    CHECK(res->status == 404);
}
