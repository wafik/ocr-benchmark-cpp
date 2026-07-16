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
struct RunTestFixture {
    ApiServer* server;
    std::thread serverThread;
    int port;
    std::string reportsRoot;

    RunTestFixture(const std::string& root) : reportsRoot(root) {
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
    ~RunTestFixture() {
        server->stop();
        serverThread.join();
        delete server;
        fs::remove_all(reportsRoot);
    }
};
}

TEST_CASE("POST /api/run returns already_running:true if a live run says running:true") {
    std::string root = "tests/fixtures/api_tmp/run_already";
    fs::create_directories(root);
    {
        std::ofstream f(root + "/.run_status.json");
        f << R"({"running": true, "updated_at": "2099-01-01T00:00:00Z"})";
    }
    RunTestFixture fixture(root);
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Post("/api/run");
    REQUIRE(res);
    CHECK(res->body.find("\"already_running\":true") != std::string::npos);
    CHECK(res->body.find("\"ok\":false") != std::string::npos);
}

TEST_CASE("POST /api/run with force=true bypasses the already_running guard") {
    std::string root = "tests/fixtures/api_tmp/run_forced";
    fs::create_directories(root);
    {
        std::ofstream f(root + "/.run_status.json");
        f << R"({"running": true, "updated_at": "2099-01-01T00:00:00Z"})";
    }
    RunTestFixture fixture(root);
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Post("/api/run?force=true&category=NONEXISTENT_CATEGORY_FOR_TEST");
    REQUIRE(res);
    CHECK(res->body.find("\"started\":true") != std::string::npos);
}

TEST_CASE("POST /api/run with no prior status file starts and reports started:true") {
    std::string root = "tests/fixtures/api_tmp/run_fresh";
    RunTestFixture fixture(root);
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Post("/api/run?category=NONEXISTENT_CATEGORY_FOR_TEST");
    REQUIRE(res);
    CHECK(res->body.find("\"started\":true") != std::string::npos);
}
