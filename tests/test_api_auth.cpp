#include <doctest/doctest.h>
#include <httplib.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include "ocr_bench/api_server.hpp"

using namespace ocr_bench;
namespace fs = std::filesystem;

namespace {
struct AuthTestFixture {
    ApiServer* server;
    std::thread serverThread;
    int port;
    std::string reportsRoot;

    AuthTestFixture(const std::string& root, const std::string& password) : reportsRoot(root) {
        fs::create_directories(root);
        ApiServerConfig cfg;
        cfg.uiRoot = "";
        cfg.reportsRoot = root;
        cfg.authPassword = password;
        cfg.port = 0;
        server = new ApiServer(cfg);
        serverThread = std::thread([this]() { server->listen(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        port = server->boundPort();
    }
    ~AuthTestFixture() {
        server->stop();
        serverThread.join();
        delete server;
        fs::remove_all(reportsRoot);
    }
};
}

TEST_CASE("GET /api/health bypasses auth even when password is set") {
    AuthTestFixture fixture("tests/fixtures/api_tmp/auth_health", "secret123");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/health");
    REQUIRE(res);
    CHECK(res->status == 200);
}

TEST_CASE("GET /api/system returns 401 when auth required but no credentials sent") {
    AuthTestFixture fixture("tests/fixtures/api_tmp/auth_no_cred", "secret123");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/system");
    REQUIRE(res);
    CHECK(res->status == 401);
}

TEST_CASE("GET /api/system returns 200 when correct auth is provided") {
    AuthTestFixture fixture("tests/fixtures/api_tmp/auth_correct", "secret123");
    httplib::Client cli("127.0.0.1", fixture.port);
    httplib::Headers headers = {{"Authorization", "Basic secret123"}};
    auto res = cli.Get("/api/system", headers);
    REQUIRE(res);
    CHECK(res->status == 200);
}

TEST_CASE("GET /api/system returns 401 when wrong password is provided") {
    AuthTestFixture fixture("tests/fixtures/api_tmp/auth_wrong", "secret123");
    httplib::Client cli("127.0.0.1", fixture.port);
    httplib::Headers headers = {{"Authorization", "Basic wrongpassword"}};
    auto res = cli.Get("/api/system", headers);
    REQUIRE(res);
    CHECK(res->status == 401);
}

TEST_CASE("No auth required when authPassword is empty") {
    AuthTestFixture fixture("tests/fixtures/api_tmp/auth_none", "");
    httplib::Client cli("127.0.0.1", fixture.port);
    auto res = cli.Get("/api/system");
    REQUIRE(res);
    CHECK(res->status == 200);
}
