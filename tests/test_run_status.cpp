#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include "ocr_bench/run_status.hpp"

using namespace ocr_bench;
namespace fs = std::filesystem;

TEST_CASE("writeStatusFile then readStatusFile round-trips the same data plus updated_at") {
    std::string path = "tests/fixtures/status_tmp/status.json";
    fs::create_directories("tests/fixtures/status_tmp");
    nlohmann::json status = {{"running", true}, {"total", 5}, {"done", 2}};
    writeStatusFile(path, status);

    auto readBack = readStatusFile(path, nlohmann::json::object());
    CHECK(readBack["running"] == true);
    CHECK(readBack["total"] == 5);
    CHECK(readBack["done"] == 2);
    CHECK(readBack.contains("updated_at"));
    fs::remove_all("tests/fixtures/status_tmp");
}

TEST_CASE("readStatusFile on a missing file returns the idle default") {
    nlohmann::json idle = {{"running", false}, {"total", 0}};
    auto result = readStatusFile("tests/fixtures/status_tmp/does_not_exist.json", idle);
    CHECK(result == idle);
}

TEST_CASE("readStatusFile on malformed JSON returns the idle default, does not throw") {
    fs::create_directories("tests/fixtures/status_tmp");
    std::string path = "tests/fixtures/status_tmp/malformed.json";
    {
        std::ofstream f(path);
        f.write("{not valid json", 15);
    }
    nlohmann::json idle = {{"running", false}};
    auto result = readStatusFile(path, idle);
    CHECK(result == idle);
    fs::remove_all("tests/fixtures/status_tmp");
}

TEST_CASE("writeStatusFile does not leave a .tmp file behind after success") {
    std::string path = "tests/fixtures/status_tmp/status2.json";
    fs::create_directories("tests/fixtures/status_tmp");
    writeStatusFile(path, {{"running", false}});
    CHECK(fs::exists(path));
    CHECK_FALSE(fs::exists(path + ".tmp"));
    fs::remove_all("tests/fixtures/status_tmp");
}
