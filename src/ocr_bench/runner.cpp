// Ports Python's runner.py.
#include "ocr_bench/runner.hpp"

#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>

#include <nlohmann/json.hpp>

#include "ocr_bench/config.hpp"
#include "ocr_bench/matcher.hpp"
#include "ocr_bench/paths.hpp"
#include "ocr_bench/run_status.hpp"
#include "ocr_bench/sysmon.hpp"

namespace fs = std::filesystem;

namespace ocr_bench {

namespace {
nlohmann::json polygonToJson(const Polygon& poly) {
    nlohmann::json arr = nlohmann::json::array();
    for (auto& pt : poly) {
        arr.push_back({pt.x, pt.y});
    }
    return arr;
}
} // namespace

PageMetrics evaluatePage(const GroundTruthPage& page, const PagePrediction& pred, float iouThreshold) {
    std::vector<Polygon> gtPolys;
    for (auto& l : page.lines) gtPolys.push_back(l.polygon);
    std::vector<Polygon> prPolys;
    for (auto& l : pred.lines) prPolys.push_back(l.polygon);

    auto matchResult = matchPolygons(gtPolys, prPolys, iouThreshold);

    PageMetrics pm;
    pm.image = page.imagePath.empty() ? pred.image : page.imagePath;
    pm.category = page.category;
    pm.nGt = static_cast<int>(page.lines.size());
    pm.nPred = static_cast<int>(pred.lines.size());
    pm.detection = DetectionStats{
        static_cast<int>(matchResult.matches.size()),
        static_cast<int>(matchResult.unmatchedPrIdx.size()),
        static_cast<int>(matchResult.unmatchedGtIdx.size()),
    };

    for (auto& [gtIdx, prIdx, iouVal] : matchResult.matches) {
        (void)iouVal;
        pm.matchedCer.push_back(cer(page.lines[gtIdx].text, pred.lines[prIdx].text));
        pm.matchedWer.push_back(wer(page.lines[gtIdx].text, pred.lines[prIdx].text));
        pm.matchedConf.push_back(pred.lines[prIdx].score);
    }

    std::string gtJoined, prJoined;
    for (size_t i = 0; i < page.lines.size(); i++) {
        if (i > 0) gtJoined += "\n";
        gtJoined += page.lines[i].text;
    }
    for (size_t i = 0; i < pred.lines.size(); i++) {
        if (i > 0) prJoined += "\n";
        prJoined += pred.lines[i].text;
    }
    pm.joinedCer = cer(gtJoined, prJoined);
    pm.elapsedMs = pred.elapsedMs;
    pm.emptyOutput = (pred.lines.empty() && !page.lines.empty());

    return pm;
}

PageMetricsInput toPageMetricsInput(const PageMetrics& pm) {
    PageMetricsInput input;
    input.nGt = pm.nGt;
    input.detection = pm.detection;
    input.matchedCer = pm.matchedCer;
    input.matchedWer = pm.matchedWer;
    input.matchedConf = pm.matchedConf;
    input.joinedCer = pm.joinedCer;
    input.elapsedMs = pm.elapsedMs;
    input.emptyOutput = pm.emptyOutput;
    return input;
}

nlohmann::json serializePageMetrics(const PageMetrics& pm, const GroundTruthPage& page, const PagePrediction& pred) {
    auto vecAvg = [](const std::vector<float>& v) -> nlohmann::json {
        if (v.empty()) return nullptr;
        float sum = 0;
        for (float x : v) sum += x;
        return sum / v.size();
    };

    nlohmann::json j = {
        {"image", pm.image},
        {"category", pm.category},
        {"n_gt", pm.nGt},
        {"n_pred", pm.nPred},
        {"detection", {{"tp", pm.detection.tp}, {"fp", pm.detection.fp}, {"fn", pm.detection.fn}}},
        {"matched_cer_mean", vecAvg(pm.matchedCer)},
        {"matched_wer_mean", vecAvg(pm.matchedWer)},
        {"mean_confidence", vecAvg(pm.matchedConf)},
        {"joined_cer", pm.joinedCer},
        {"elapsed_ms", pm.elapsedMs},
        {"empty_output", pm.emptyOutput},
    };

    // Build overlays: matched, missed (GT only), spurious (PR only)
    nlohmann::json overlays = nlohmann::json::array();

    // Re-run matching to get index mapping
    std::vector<Polygon> gtPolys;
    for (auto& l : page.lines) gtPolys.push_back(l.polygon);
    std::vector<Polygon> prPolys;
    for (auto& l : pred.lines) prPolys.push_back(l.polygon);
    auto matchResult = matchPolygons(gtPolys, prPolys);

    // Matched overlays
    for (auto& [gtIdx, prIdx, iouVal] : matchResult.matches) {
        overlays.push_back({
            {"gt_polygon", polygonToJson(page.lines[gtIdx].polygon)},
            {"gt_text", page.lines[gtIdx].text},
            {"pr_polygon", polygonToJson(pred.lines[prIdx].polygon)},
            {"pr_text", pred.lines[prIdx].text},
            {"pr_score", pred.lines[prIdx].score},
            {"iou", iouVal},
            {"status", "matched"},
            {"line_cer", cer(page.lines[gtIdx].text, pred.lines[prIdx].text)},
        });
    }

    // Missed overlays (GT not matched)
    for (int gtIdx : matchResult.unmatchedGtIdx) {
        nlohmann::json ov;
        ov["gt_polygon"] = polygonToJson(page.lines[gtIdx].polygon);
        ov["gt_text"] = page.lines[gtIdx].text;
        ov["pr_polygon"] = nullptr;
        ov["pr_text"] = nullptr;
        ov["pr_score"] = nullptr;
        ov["iou"] = nullptr;
        ov["status"] = "missed";
        overlays.push_back(std::move(ov));
    }

    // Spurious overlays (PR not matched)
    for (int prIdx : matchResult.unmatchedPrIdx) {
        nlohmann::json ov;
        ov["gt_polygon"] = nullptr;
        ov["gt_text"] = nullptr;
        ov["pr_polygon"] = polygonToJson(pred.lines[prIdx].polygon);
        ov["pr_text"] = pred.lines[prIdx].text;
        ov["pr_score"] = pred.lines[prIdx].score;
        ov["iou"] = nullptr;
        ov["status"] = "spurious";
        overlays.push_back(std::move(ov));
    }

    j["overlays"] = std::move(overlays);
    return j;
}

namespace {

std::atomic<int> g_runGen{0};

struct Superseded {};

std::string nowIso8601Runner() {
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

std::string slugify(const std::string& name) {
    std::string out;
    for (char c : name) {
        out += std::isalnum(static_cast<unsigned char>(c)) ? static_cast<char>(std::tolower(c)) : '_';
    }
    size_t start = out.find_first_not_of('_');
    size_t end = out.find_last_not_of('_');
    if (start == std::string::npos) return "_";
    return out.substr(start, end - start + 1);
}

PagePrediction predictWithTimeout(BenchEngine& engine, const std::string& imagePath, int timeoutS = 300) {
    auto future = std::async(std::launch::async, [&engine, &imagePath]() {
        return engine.predict(imagePath);
    });
    if (future.wait_for(std::chrono::seconds(timeoutS)) == std::future_status::timeout) {
        PagePrediction timedOut;
        timedOut.image = fs::path(imagePath).filename().string();
        timedOut.elapsedMs = static_cast<float>(timeoutS) * 1000.0f;
        return timedOut;
    }
    return future.get();
}

} // namespace

const std::string kRunStatusPath = "reports/.run_status.json";

nlohmann::json run(const RunOptions& options) {
    auto [datasetKey, root] = resolveDatasetRoot(options.datasetKey);
    auto categories = listCategories(root);
    if (!options.onlyCategories.empty()) {
        std::vector<std::string> filtered;
        for (auto& cat : categories) {
            std::string catName = fs::path(cat).filename().string();
            for (auto& want : options.onlyCategories) {
                if (catName == want) { filtered.push_back(cat); break; }
            }
        }
        categories = filtered;
    }
    if (categories.empty()) {
        throw std::runtime_error("no categories found");
    }

    Settings settings = loadSettings();
    std::string ver = options.ocrVersion.empty() ? settings.ocrVersion : options.ocrVersion;
    std::string mtype = options.modelType.empty() ? settings.modelType : options.modelType;
    auto& ov = options.overrides;

    EngineConfig engineCfg;
    engineCfg.ocrVersion = ver;
    engineCfg.modelType = mtype;
    engineCfg.detBoxThresh = ov.detBoxThresh.value_or(settings.detBoxThresh);
    engineCfg.detThresh = ov.detThresh.value_or(settings.detThresh);
    engineCfg.detUnclipRatio = ov.detUnclipRatio.value_or(settings.detUnclipRatio);
    engineCfg.detLimitSideLen = ov.detLimitSideLen.value_or(settings.detLimitSideLen);
    engineCfg.useAngleCls = ov.useAngleCls.value_or(settings.useAngleCls);
    engineCfg.useTensorrt = ov.useTensorrt.value_or(settings.useTensorrt);
    engineCfg.useCuda = engineCfg.useTensorrt || detectCuda();
    engineCfg.trtCacheDir = (fs::path(packageRoot()) / "models" / "trt_engines").string();
    engineCfg.modelsDir = packageRoot() + "/models";

    float iouThreshold = ov.iouThreshold.value_or(settings.iouThreshold);

    fs::path reportsRoot(packageRoot() + "/reports");
    fs::path perCatDir = reportsRoot / "per_category";
    fs::create_directories(perCatDir);

    int gen = ++g_runGen;
    ResourceMonitor monitor(2.0f);
    monitor.start();

    std::string startedAt = nowIso8601Runner();
    nlohmann::json completed = nlohmann::json::array();
    std::string statusPath = reportsRoot.string() + "/.run_status.json";

    writeStatusFile(statusPath, {
        {"running", true}, {"started_at", startedAt},
        {"total", static_cast<int>(categories.size())},
        {"completed", completed}, {"current", nullptr},
        {"dataset", datasetKey},
    });

    auto checkSuperseded = [&]() { if (gen != g_runGen.load()) throw Superseded{}; };

    std::vector<CategorySummary> allSummaries;
    auto overallStart = std::chrono::steady_clock::now();

    try {
        BenchEngine engine(engineCfg);

        for (size_t catIdx = 0; catIdx < categories.size(); catIdx++) {
            checkSuperseded();
            auto& catDir = categories[catIdx];
            auto pages = loadCategory(catDir);
            std::string catName = fs::path(catDir).filename().string();
            if (pages.empty()) continue;

            std::vector<PageMetricsInput> catInputs;
            nlohmann::json perImagePayload = nlohmann::json::array();
            auto catStart = std::chrono::steady_clock::now();

            for (size_t imgIdx = 0; imgIdx < pages.size(); imgIdx++) {
                checkSuperseded();
                auto& page = pages[imgIdx];
                auto pred = predictWithTimeout(engine, page.imagePath);
                auto pm = evaluatePage(page, pred, iouThreshold);
                catInputs.push_back(toPageMetricsInput(pm));
                perImagePayload.push_back(serializePageMetrics(pm, page, pred));

                writeStatusFile(statusPath, {
                    {"running", true}, {"started_at", startedAt},
                    {"total", static_cast<int>(categories.size())},
                    {"completed", completed},
                    {"current", {{"name", catName}, {"total_images", static_cast<int>(pages.size())},
                                 {"done_images", static_cast<int>(imgIdx + 1)}}},
                    {"dataset", datasetKey},
                });
            }

            float catElapsedS = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - catStart).count();
            auto summary = aggregateCategory(catName, catInputs);
            allSummaries.push_back(summary);
            completed.push_back({{"name", catName}, {"elapsed_s", catElapsedS}});

            fs::path outFile = perCatDir / (slugify(catName) + ".json");
            nlohmann::json catJson = {
                {"category", catName}, {"dataset", datasetKey},
                {"summary", {
                    {"category", summary.category}, {"n_images", summary.nImages},
                    {"n_lines_total", summary.nLinesTotal},
                    {"detection", {{"tp", summary.detection.tp}, {"fp", summary.detection.fp}, {"fn", summary.detection.fn}}},
                    {"matched_cer_mean", summary.matchedCerMean}, {"matched_cer_median", summary.matchedCerMedian},
                    {"matched_wer_mean", summary.matchedWerMean}, {"joined_cer_mean", summary.joinedCerMean},
                    {"mean_confidence", summary.meanConfidence}, {"mean_ms_per_image", summary.meanMsPerImage},
                    {"empty_output_rate", summary.emptyOutputRate},
                }},
                {"images", perImagePayload},
            };
            std::ofstream(outFile) << catJson.dump(2);
        }

        float totalElapsedS = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - overallStart).count();

        int tp = 0, fp = 0, fn = 0, nImgs = 0, nLines = 0;
        float cerSum = 0.0f, werSum = 0.0f;
        for (auto& s : allSummaries) {
            tp += s.detection.tp; fp += s.detection.fp; fn += s.detection.fn;
            nImgs += s.nImages; nLines += s.nLinesTotal;
            cerSum += s.matchedCerMean * s.nImages;
            werSum += s.matchedWerMean * s.nImages;
        }
        float precision = (tp + fp) > 0 ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        float recall = (tp + fn) > 0 ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;

        nlohmann::json overall = {
            {"detection_precision", precision}, {"detection_recall", recall}, {"detection_f1", f1},
            {"cer_mean", nImgs > 0 ? cerSum / nImgs : 0.0f},
            {"wer_mean", nImgs > 0 ? werSum / nImgs : 0.0f},
            {"n_images", nImgs}, {"n_lines", nLines}, {"n_categories", static_cast<int>(allSummaries.size())},
            {"total_elapsed_s", totalElapsedS}, {"last_run", nowIso8601Runner()},
            {"dataset", datasetKey}, {"ocr_version", ver}, {"model_type", mtype},
            {"det_box_thresh", engineCfg.detBoxThresh}, {"det_thresh", engineCfg.detThresh},
            {"det_unclip_ratio", engineCfg.detUnclipRatio}, {"det_limit_side_len", engineCfg.detLimitSideLen},
            {"use_angle_cls", engineCfg.useAngleCls}, {"backend", engine.backend()},
            {"iou_threshold", iouThreshold},
            {"resources", monitor.summary()},
        };

        checkSuperseded();

        writeSummaryCsv((reportsRoot / "summary.csv").string(), allSummaries, overall);
        auto summaryJson = toOverallJson(allSummaries, overall);
        std::ofstream(reportsRoot / "summary.json") << summaryJson.dump(2);

        fs::path historyRoot = reportsRoot / "history";
        fs::create_directories(historyRoot);
        std::string runId = overall["last_run"].get<std::string>();
        std::replace(runId.begin(), runId.end(), ':', '-');
        std::replace(runId.begin(), runId.end(), 'T', '_');
        runId.erase(std::remove(runId.begin(), runId.end(), 'Z'), runId.end());

        nlohmann::json historyPerCat = nlohmann::json::array();
        for (auto& s : allSummaries) {
            historyPerCat.push_back({
                {"category", s.category}, {"n_images", s.nImages}, {"n_lines", s.nLinesTotal},
                {"f1", s.detection.f1()}, {"cer", s.matchedCerMean}, {"wer", s.matchedWerMean},
                {"mean_conf", s.meanConfidence}, {"ms_per_img", s.meanMsPerImage},
            });
        }
        nlohmann::json snapshot = {
            {"id", runId}, {"timestamp", overall["last_run"]}, {"dataset", datasetKey},
            {"config", {
                {"dataset", datasetKey}, {"ocr_version", ver}, {"model_type", mtype},
                {"det_box_thresh", engineCfg.detBoxThresh}, {"det_thresh", engineCfg.detThresh},
                {"det_unclip_ratio", engineCfg.detUnclipRatio}, {"det_limit_side_len", engineCfg.detLimitSideLen},
                {"use_angle_cls", engineCfg.useAngleCls}, {"backend", engine.backend()},
                {"iou_threshold", iouThreshold},
            }},
            {"total_elapsed_s", totalElapsedS}, {"resources", monitor.summary()},
            {"overall", overall}, {"per_category", historyPerCat},
        };
        std::ofstream(historyRoot / (runId + ".json")) << snapshot.dump(2);

        fs::path indexPath = historyRoot / "index.json";
        nlohmann::json index = nlohmann::json::array();
        if (fs::exists(indexPath)) {
            try {
                std::ifstream in(indexPath);
                index = nlohmann::json::parse(in);
            } catch (...) { index = nlohmann::json::array(); }
        }
        nlohmann::json filteredIndex = nlohmann::json::array();
        for (auto& entry : index) {
            if (entry.value("id", "") != runId) filteredIndex.push_back(entry);
        }
        index = filteredIndex;
        index.push_back({
            {"id", runId}, {"timestamp", overall["last_run"]}, {"dataset", datasetKey},
            {"ocr_version", ver}, {"model_type", mtype}, {"backend", engine.backend()},
            {"config", snapshot["config"]},
            {"n_images", nImgs}, {"f1", f1}, {"cer", overall["cer_mean"]}, {"wer", overall["wer_mean"]},
            {"total_elapsed_s", totalElapsedS}, {"resources", monitor.summary()},
        });
        if (index.size() > 50) {
            nlohmann::json trimmed = nlohmann::json::array();
            for (size_t i = index.size() - 50; i < index.size(); i++) trimmed.push_back(index[i]);
            index = trimmed;
        }
        std::ofstream(indexPath) << index.dump(2);

        writeStatusFile(statusPath, {
            {"running", false}, {"started_at", startedAt}, {"finished_at", nowIso8601Runner()},
            {"total", static_cast<int>(categories.size())}, {"completed", completed},
            {"current", nullptr}, {"dataset", datasetKey}, {"resources", monitor.summary()},
        });

        monitor.stop();
        return overall;

    } catch (const Superseded&) {
        monitor.stop();
        return nlohmann::json::object();
    } catch (const std::exception& e) {
        if (gen == g_runGen.load()) {
            writeStatusFile(statusPath, {
                {"running", false}, {"started_at", startedAt}, {"finished_at", nowIso8601Runner()},
                {"total", static_cast<int>(categories.size())}, {"completed", completed},
                {"current", nullptr}, {"error", std::string(e.what())},
            });
        }
        monitor.stop();
        throw;
    }
}

} // namespace ocr_bench
