#include "ocr_bench/run_status.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

namespace ocr_bench {

namespace {

std::string nowIso8601() {
    std::time_t t = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &t);
#else
    gmtime_r(&t, &utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buf;
}

} // namespace

void writeStatusFile(const std::string& path, nlohmann::json status) {
    status["updated_at"] = nowIso8601();
    std::string payload = status.dump();

    fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out << payload;
    }

    std::error_code renameErr;
    for (int attempt = 0; attempt < 8; attempt++) {
        fs::rename(tmp, p, renameErr);
        if (!renameErr) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50 * (attempt + 1)));
    }
    // Fallback: best-effort direct (non-atomic) write if rename keeps failing
    {
        std::ofstream out(p, std::ios::binary);
        out << payload;
    }
    std::error_code removeErr;
    fs::remove(tmp, removeErr);
}

nlohmann::json readStatusFile(const std::string& path, const nlohmann::json& idleDefault) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return idleDefault;
    }
    try {
        nlohmann::json data = nlohmann::json::parse(in);
        return data;
    } catch (const nlohmann::json::parse_error&) {
        return idleDefault;
    }
}

} // namespace ocr_bench
