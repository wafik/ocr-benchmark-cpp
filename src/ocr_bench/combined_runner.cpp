#include "ocr_bench/combined_runner.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "ocr_bench/config.hpp"
#include "ocr_bench/paths.hpp"
#include "ocr_bench/run_status.hpp"

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

nlohmann::json combinedRun(const CombinedRunOptions& options) {
    fs::path reportsRoot(packageRoot() + "/reports");
    fs::create_directories(reportsRoot);
    std::string statusPath = reportsRoot.string() + "/.combined_status.json";

    writeStatusFile(statusPath, {
        {"running", true}, {"started_at", nowIso8601()},
        {"phase", "ocr"}, {"total", 2},
        {"completed", nlohmann::json::array()},
    });

    auto overallStart = std::chrono::steady_clock::now();

    // Phase 1: OCR
    nlohmann::json ocrSummary;
    try {
        ocrSummary = run(options.ocrOptions);
    } catch (const std::exception& e) {
        writeStatusFile(statusPath, {
            {"running", false}, {"error", std::string("OCR failed: ") + e.what()},
            {"finished_at", nowIso8601()},
        });
        throw;
    }

    writeStatusFile(statusPath, {
        {"running", true}, {"started_at", nowIso8601()},
        {"phase", "tts"}, {"total", 2},
        {"completed", {{"name", "OCR"}}},
    });

    // Phase 2: TTS
    // Combined mode already loads OCR engine (~2.5GB). Limit TTS workers
    // to avoid OOM — use at most 2 workers when both engines are loaded.
    TTSRunOptions ttsOpts = options.ttsOptions;
    if (ttsOpts.numWorkers <= 0) {
        ttsOpts.numWorkers = 2; // conservative: OCR + TTS engines share RAM
    }
    nlohmann::json ttsSummary;
    try {
        ttsSummary = ttsRun(ttsOpts);
    } catch (const std::exception& e) {
        writeStatusFile(statusPath, {
            {"running", false}, {"error", std::string("TTS failed: ") + e.what()},
            {"finished_at", nowIso8601()},
        });
        throw;
    }

    float totalElapsedS = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - overallStart).count();

    nlohmann::json combined = {
        {"ocr", ocrSummary},
        {"tts", ttsSummary},
        {"total_elapsed_s", totalElapsedS},
        {"last_run", nowIso8601()},
    };

    std::ofstream(reportsRoot / "combined_summary.json") << combined.dump(2);

    // History snapshot
    fs::path historyRoot = reportsRoot / "combined_history";
    fs::create_directories(historyRoot);
    std::string runId = nowIso8601();
    std::replace(runId.begin(), runId.end(), ':', '-');
    std::replace(runId.begin(), runId.end(), 'T', '_');
    runId.erase(std::remove(runId.begin(), runId.end(), 'Z'), runId.end());
    std::ofstream(historyRoot / (runId + ".json")) << combined.dump(2);

    writeStatusFile(statusPath, {
        {"running", false}, {"started_at", nowIso8601()}, {"finished_at", nowIso8601()},
        {"total", 2}, {"completed", {{"name", "OCR"}, {"name", "TTS"}}},
        {"current", nullptr},
    });

    return combined;
}

} // namespace ocr_bench
