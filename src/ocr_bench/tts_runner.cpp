#include "ocr_bench/tts_runner.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iostream>
#include <thread>
#include <future>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

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

// ── Worker pool for parallel TTS synthesis ──────────────────────────────

#ifndef _WIN32
struct WorkerProcess {
    pid_t pid = -1;
    FILE* stdinPipe = nullptr;
    FILE* stdoutPipe = nullptr;
    bool alive() const { return pid > 0; }
};

class TTSWorkerPool {
public:
    TTSWorkerPool(int numWorkers, const std::string& voicePath,
                  const std::string& espeakPath, bool useCuda) {
        std::string workerBin = ttsWorkerBinary();
        if (!fs::exists(workerBin)) {
            throw std::runtime_error("TTS worker binary not found: " + workerBin);
        }

        for (int i = 0; i < numWorkers; i++) {
            int stdinPipe[2], stdoutPipe[2];
            if (pipe(stdinPipe) != 0 || pipe(stdoutPipe) != 0) {
                throw std::runtime_error("pipe() failed");
            }

            pid_t pid = fork();
            if (pid == 0) {
                // Child: redirect stdin/stdout to pipes, exec worker
                close(stdinPipe[1]);
                close(stdoutPipe[0]);
                dup2(stdinPipe[0], STDIN_FILENO);
                dup2(stdoutPipe[1], STDOUT_FILENO);
                close(stdinPipe[0]);
                close(stdoutPipe[1]);

                std::string cudaFlag = useCuda ? "--cuda" : "";
                execlp(workerBin.c_str(), workerBin.c_str(),
                       "--voice", voicePath.c_str(),
                       "--espeak-data", espeakPath.c_str(),
                       cudaFlag.c_str(), nullptr);
                _exit(127); // exec failed
            }

            // Parent
            close(stdinPipe[0]);
            close(stdoutPipe[1]);

            WorkerProcess wp;
            wp.pid = pid;
            wp.stdinPipe = fdopen(stdinPipe[1], "w");
            wp.stdoutPipe = fdopen(stdoutPipe[0], "r");
            workers_.push_back(wp);
        }
    }

    ~TTSWorkerPool() {
        for (auto& w : workers_) {
            if (w.stdinPipe) { fclose(w.stdinPipe); w.stdinPipe = nullptr; }
            if (w.stdoutPipe) { fclose(w.stdoutPipe); w.stdoutPipe = nullptr; }
            if (w.pid > 0) {
                kill(w.pid, SIGTERM);
                int status;
                waitpid(w.pid, &status, 0);
            }
        }
    }

    // Send text to worker, read result back. Returns empty json on error.
    nlohmann::json submit(int workerIdx, int lineId, const std::string& text) {
        auto& w = workers_[workerIdx];
        if (!w.alive()) return {};

        nlohmann::json req = {{"id", lineId}, {"text", text}};
        std::string reqStr = req.dump() + "\n";
        if (fputs(reqStr.c_str(), w.stdinPipe) == EOF) return {};
        fflush(w.stdinPipe);

        char buf[4096];
        if (!fgets(buf, sizeof(buf), w.stdoutPipe)) return {};
        try {
            return nlohmann::json::parse(buf);
        } catch (...) {
            return {};
        }
    }

    size_t size() const { return workers_.size(); }

private:
    std::vector<WorkerProcess> workers_;
};
#endif

/// Auto-detect optimal worker count based on CPU cores and available RAM.
int autoDetectWorkers(bool useCuda) {
    int cores = static_cast<int>(std::thread::hardware_concurrency());
    if (cores <= 0) cores = 2;

    // Rough estimate: each worker uses ~300-500MB (piper model + ORT session)
    // On Jetson with 7.5GB, limit to 2 workers max
    int maxByRam = 2;
#ifdef __linux__
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            long kB = 0;
            sscanf(line.c_str(), "MemTotal: %ld kB", &kB);
            maxByRam = std::max(1, static_cast<int>(kB / (400 * 1024))); // ~400MB per worker
            break;
        }
    }
#endif
    return std::max(1, std::min(cores, maxByRam));
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

    auto [datasetKey, root] = resolveDatasetRoot(options.datasetKey);
    auto categories = listCategories(root);

    // Flatten all lines with IDs for parallel dispatch
    struct LineInfo { int id; std::string text; std::string category; };
    std::vector<LineInfo> allLines;
    int lineId = 0;
    for (auto& catDir : categories) {
        auto pages = loadCategory(catDir);
        std::string catName = fs::path(catDir).filename().string();
        for (auto& page : pages) {
            for (auto& line : page.lines) {
                if (!line.text.empty()) {
                    allLines.push_back({lineId++, line.text, catName});
                }
            }
        }
    }

    // Decide: parallel or sequential
    int numWorkers = options.numWorkers;
    if (numWorkers == 0) numWorkers = autoDetectWorkers(useCuda);
    bool parallel = numWorkers > 1;

    nlohmann::json completed = nlohmann::json::array();
    std::vector<float> allRtfs;
    int totalChars = 0;
    float totalAudioS = 0.0f;
    float totalSynthMs = 0.0f;
    auto overallStart = std::chrono::steady_clock::now();

    if (parallel) {
        // ── Parallel path: distribute lines to worker processes ──
#ifndef _WIN32
        try {
            std::cerr << "[TTS] Starting parallel pool: " << numWorkers << " workers, " << allLines.size() << " lines" << std::endl;
            TTSWorkerPool pool(numWorkers, voicePath, espeakDataPath(), useCuda);
            std::cerr << "[TTS] Pool started, dispatching lines..." << std::endl;
            std::vector<nlohmann::json> results(allLines.size());

            for (size_t i = 0; i < allLines.size(); i++) {
                int workerIdx = static_cast<int>(i % pool.size());
                results[i] = pool.submit(workerIdx, allLines[i].id, allLines[i].text);
            }

            // Aggregate results
            for (size_t i = 0; i < allLines.size(); i++) {
                auto& r = results[i];
                if (r.empty() || r.contains("error")) continue;
                float synthMs = r.value("synth_ms", 0.0f);
                float audioS = r.value("audio_seconds", 0.0f);
                int nChars = r.value("n_chars", 0);
                float rtf = audioS > 0 ? (synthMs / 1000.0f) / audioS : 0.0f;
                allRtfs.push_back(rtf);
                totalChars += nChars;
                totalAudioS += audioS;
                totalSynthMs += synthMs;
            }
        } catch (const std::exception& e) {
            std::cerr << "Worker pool failed: " << e.what() << " — falling back to sequential" << std::endl;
            parallel = false;
        }
#else
        // Windows: parallel not supported yet (fork not available)
        parallel = false;
#endif
    }

    if (!parallel) {
        // ── Sequential path (original) ──
        TTSEngine engine(voicePath, espeakDataPath(), useCuda);

        writeStatusFile(statusPath, {
            {"running", true}, {"started_at", nowIso8601()},
            {"total", static_cast<int>(categories.size())},
            {"completed", nlohmann::json::array()},
            {"current", nullptr}, {"dataset", datasetKey},
        });

        for (size_t catIdx = 0; catIdx < categories.size(); catIdx++) {
            auto pages = loadCategory(categories[catIdx]);
            std::string catName = fs::path(categories[catIdx]).filename().string();

            // Batch lines if batchSize > 1
            std::vector<std::string> batchTexts;
            for (size_t imgIdx = 0; imgIdx < pages.size(); imgIdx++) {
                auto& page = pages[imgIdx];
                for (auto& line : page.lines) {
                    if (line.text.empty()) continue;
                    batchTexts.push_back(line.text);

                    if (static_cast<int>(batchTexts.size()) >= options.batchSize) {
                        auto batch = engine.synthesizeBatch(batchTexts);
                        for (auto& r : batch.results) {
                            allRtfs.push_back(r.rtf());
                            totalChars += r.nChars;
                            totalAudioS += r.audioSeconds;
                            totalSynthMs += r.synthMs;
                        }
                        batchTexts.clear();
                    }
                }
            }
            // Flush remaining
            if (!batchTexts.empty()) {
                auto batch = engine.synthesizeBatch(batchTexts);
                for (auto& r : batch.results) {
                    allRtfs.push_back(r.rtf());
                    totalChars += r.nChars;
                    totalAudioS += r.audioSeconds;
                    totalSynthMs += r.synthMs;
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
        {"num_workers", parallel ? numWorkers : 1},
        {"batch_size", options.batchSize},
        {"parallel_mode", parallel ? "process_pool" : "sequential"},
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
