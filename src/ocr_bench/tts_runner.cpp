#include "ocr_bench/tts_runner.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

#include "ocr_bench/config.hpp"
#include "ocr_bench/dataset.hpp"
#include "ocr_bench/paths.hpp"
#include "ocr_bench/run_status.hpp"
#include "ocr_bench/tts_engine.hpp"

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

std::string espeakDataPath() {
    return packageRoot() + "/models/espeak-ng-data";
}

} // namespace

nlohmann::json ttsRun(const TTSRunOptions& options) {
    Settings settings = loadSettings();
    std::string rawVoice = options.voicePath.empty() ? settings.piperVoicePath : options.voicePath;
    std::string voicePath = (fs::path(packageRoot()) / rawVoice).string();
    bool useCuda = options.useCuda;

    fs::path reportsRoot(packageRoot() + "/reports");
    fs::create_directories(reportsRoot);
    std::string statusPath = reportsRoot.string() + "/.tts_status.json";

    // Load TTS engine — use CUDA if available and requested, else CPU
    TTSEngine engine(voicePath, espeakDataPath(), useCuda);

    auto [datasetKey, root] = resolveDatasetRoot(options.datasetKey);
    auto categories = listCategories(root);

    writeStatusFile(statusPath, {
        {"running", true}, {"started_at", nowIso8601()},
        {"total", static_cast<int>(categories.size())},
        {"completed", nlohmann::json::array()},
        {"current", nullptr}, {"dataset", datasetKey},
    });

    nlohmann::json completed = nlohmann::json::array();
    std::vector<float> allRtfs;
    int totalChars = 0;
    float totalAudioS = 0.0f;
    float totalSynthMs = 0.0f;
    auto overallStart = std::chrono::steady_clock::now();

    for (size_t catIdx = 0; catIdx < categories.size(); catIdx++) {
        auto pages = loadCategory(categories[catIdx]);
        std::string catName = fs::path(categories[catIdx]).filename().string();

        for (size_t imgIdx = 0; imgIdx < pages.size(); imgIdx++) {
            auto& page = pages[imgIdx];
            for (auto& line : page.lines) {
                if (line.text.empty()) continue;
                auto [pcm, result] = engine.synthesize(line.text);
                (void)pcm;
                allRtfs.push_back(result.rtf());
                totalChars += result.nChars;
                totalAudioS += result.audioSeconds;
                totalSynthMs += result.synthMs;
            }
        }

        completed.push_back({{"name", catName}});
        writeStatusFile(statusPath, {
            {"running", true}, {"started_at", nowIso8601()},
            {"total", static_cast<int>(categories.size())},
            {"completed", completed},
            {"current", {{"name", catName}}},
            {"dataset", datasetKey},
        });
    }

    float totalElapsedS = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - overallStart).count();

    auto mean = [](const std::vector<float>& v) -> float {
        if (v.empty()) return 0.0f;
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };
    auto maxOf = [](const std::vector<float>& v) -> float {
        if (v.empty()) return 0.0f;
        return *std::max_element(v.begin(), v.end());
    };

    nlohmann::json summary = {
        {"n_categories", static_cast<int>(categories.size())},
        {"n_lines", static_cast<int>(allRtfs.size())},
        {"total_chars", totalChars},
        {"total_audio_seconds", totalAudioS},
        {"total_synth_ms", totalSynthMs},
        {"rtf_mean", mean(allRtfs)},
        {"rtf_max", maxOf(allRtfs)},
        {"chars_per_sec", totalChars / (totalSynthMs / 1000.0f)},
        {"total_elapsed_s", totalElapsedS},
        {"voice", voicePath},
        {"backend", useCuda ? "cuda" : "cpu"},
        {"last_run", nowIso8601()},
    };

    std::ofstream(reportsRoot / "tts_summary.json") << summary.dump(2);

    writeStatusFile(statusPath, {
        {"running", false}, {"started_at", nowIso8601()}, {"finished_at", nowIso8601()},
        {"total", static_cast<int>(categories.size())}, {"completed", completed},
        {"current", nullptr}, {"dataset", datasetKey},
    });

    return summary;
}

} // namespace ocr_bench
