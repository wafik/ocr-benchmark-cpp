#include <doctest/doctest.h>
#include <httplib.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include "ocr_bench/api_server.hpp"

using namespace ocr_bench;
namespace fs = std::filesystem;

namespace {
struct TestServerFixture {
    ApiServer* server = nullptr;
    std::thread serverThread;
    int port;

    TestServerFixture(const std::string& reportsRoot) {
        fs::create_directories(reportsRoot);
        ApiServerConfig cfg;
        cfg.uiRoot = "";
        cfg.reportsRoot = reportsRoot;
        cfg.authPassword = "";
        cfg.host = "127.0.0.1";
        cfg.port = 0;
        server = new ApiServer(cfg);
        serverThread = std::thread([this]() { server->listen(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        port = server->boundPort();
    }
    ~TestServerFixture() {
        server->stop();
        serverThread.join();
        delete server;
    }
};
}

TEST_CASE("GET /api/health returns ok:true") {
    TestServerFixture fixture("tests/fixtures/api_tmp/health");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/health");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"ok\":true") != std::string::npos);
}

TEST_CASE("GET /api/progress with no run yet returns running:false") {
    TestServerFixture fixture("tests/fixtures/api_tmp/progress_idle");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/progress");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"running\":false") != std::string::npos);
}

TEST_CASE("GET /api/system returns a JSON object with cpu_percent") {
    TestServerFixture fixture("tests/fixtures/api_tmp/system");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/system");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("cpu_percent") != std::string::npos);
}
