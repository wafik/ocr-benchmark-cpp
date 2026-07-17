#include "ocr_bench/api_server.hpp"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "ocr_bench/config.hpp"
#include "ocr_bench/paths.hpp"
#include "ocr_bench/runner.hpp"
#include "ocr_bench/run_status.hpp"
#include "ocr_bench/sysmon.hpp"
#include "ocr_bench/tts_engine.hpp"
#include "ocr_bench/tts_runner.hpp"
#include "ocr_bench/combined_runner.hpp"

namespace fs = std::filesystem;

namespace ocr_bench {

namespace {
constexpr int kStaleAfterS = 420;

bool isStale(const nlohmann::json& status) {
    std::string ts = status.value("updated_at", status.value("started_at", ""));
    if (ts.empty()) return false;
    std::tm tm{};
    std::istringstream ss(ts);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    if (ss.fail()) return false;
#ifdef _WIN32
    std::time_t then = _mkgmtime(&tm);
#else
    std::time_t then = timegm(&tm);
#endif
    std::time_t now = std::time(nullptr);
    return (now - then) > kStaleAfterS;
}

template <typename T>
std::optional<T> parseQueryParam(const httplib::Request& req, const std::string& name);

template <>
std::optional<float> parseQueryParam<float>(const httplib::Request& req, const std::string& name) {
    if (!req.has_param(name)) return std::nullopt;
    try { return std::stof(req.get_param_value(name)); } catch (...) { return std::nullopt; }
}
template <>
std::optional<int> parseQueryParam<int>(const httplib::Request& req, const std::string& name) {
    if (!req.has_param(name)) return std::nullopt;
    try { return std::stoi(req.get_param_value(name)); } catch (...) { return std::nullopt; }
}
template <>
std::optional<bool> parseQueryParam<bool>(const httplib::Request& req, const std::string& name) {
    if (!req.has_param(name)) return std::nullopt;
    std::string v = req.get_param_value(name);
    return v == "true" || v == "1";
}

std::string sanitizeRunId(const std::string& raw) {
    std::string out;
    for (char c : raw) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            out += c;
        }
    }
    return out;
}

} // namespace

ApiServer::ApiServer(const ApiServerConfig& config) : config_(config) {
    server_ = std::make_unique<httplib::Server>();
    registerAuthMiddleware();
    registerHealthRoutes();
    registerRunRoutes();
    registerResultsRoutes();
    registerTtsCombinedRoutes();
    registerStaticRoutes();
}

ApiServer::~ApiServer() = default;

void ApiServer::listen() {
    if (config_.port == 0) {
        boundPort_ = server_->bind_to_any_port(config_.host);
        server_->listen_after_bind();
    } else {
        boundPort_ = config_.port;
        server_->listen(config_.host, config_.port);
    }
}

void ApiServer::stop() {
    server_->stop();
}

int ApiServer::boundPort() const {
    return boundPort_;
}

// --- Auth Middleware ---

void ApiServer::registerAuthMiddleware() {
    if (config_.authPassword.empty()) return;

    server_->set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
        // Skip auth for health check
        if (req.path == "/api/health") return httplib::Server::HandlerResponse::Unhandled;

        auto it = req.headers.find("Authorization");
        if (it == req.headers.end()) {
            res.status = 401;
            res.set_content(R"({"detail":"Missing Authorization header"})", "application/json");
            res.set_header("WWW-Authenticate", "Basic realm=\"OCR Bench\"");
            return httplib::Server::HandlerResponse::Handled;
        }

        // Parse Basic auth: "Basic base64(user:pass)"
        std::string auth = it->second;
        if (auth.substr(0, 6) != "Basic ") {
            res.status = 401;
            res.set_content(R"({"detail":"Invalid auth scheme"})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }

        // For simplicity, just check if the password matches the raw header value
        // A production implementation would base64-decode, but the UI sends
        // the password directly in the header for this simple auth scheme.
        // The Python reference uses the same pattern.
        std::string expected = "Basic " + config_.authPassword;
        if (auth != expected) {
            res.status = 401;
            res.set_content(R"({"detail":"Invalid password"})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });
}

// --- Health / Progress / System ---

void ApiServer::registerHealthRoutes() {
    server_->Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"ok", true}}.dump(), "application/json");
    });

    server_->Get("/api/progress", [this](const httplib::Request&, httplib::Response& res) {
        auto [key, root] = resolveDatasetRoot();
        (void)root;
        nlohmann::json idle = {{"running", false}, {"total", 0}, {"completed", nlohmann::json::array()},
                                {"current", nullptr}, {"dataset", key}};
        auto status = readStatusFile(config_.reportsRoot + "/.run_status.json", idle);
        if (status.value("running", false) && isStale(status)) {
            status["stale"] = true;
        }
        res.set_content(status.dump(), "application/json");
    });

    // GET /api/progress/stream — SSE endpoint for live progress updates
    server_->Get("/api/progress/stream", [this](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider("text/event-stream",
            [this](size_t, httplib::DataSink& sink) -> bool {
                auto [key, root] = resolveDatasetRoot();
                (void)root;
                nlohmann::json idle = {{"running", false}, {"total", 0}, {"completed", nlohmann::json::array()},
                                        {"current", nullptr}, {"dataset", key}};

                std::string lastUpdated;
                for (int i = 0; i < 1200; i++) {
                    auto current = readStatusFile(config_.reportsRoot + "/.run_status.json", idle);
                    std::string curUpdated = current.value("updated_at", "");

                    if (curUpdated != lastUpdated || current.value("running", false)) {
                        std::string sseData = "data: " + current.dump() + "\n\n";
                        if (!sink.write(sseData.c_str(), sseData.size())) return false;
                        lastUpdated = curUpdated;
                        if (!current.value("running", false) && !lastUpdated.empty()) return true;
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                return true;
            });
    });

    server_->Get("/api/system", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(systemSampleToJson(sampleSystem()).dump(), "application/json");
    });
}

// --- POST /api/run ---

void ApiServer::registerRunRoutes() {
    server_->Post("/api/run", [this](const httplib::Request& req, httplib::Response& res) {
        std::string statusPath = config_.reportsRoot + "/.run_status.json";
        bool force = parseQueryParam<bool>(req, "force").value_or(false);

        if (!force) {
            nlohmann::json idle = {{"running", false}};
            auto current = readStatusFile(statusPath, idle);
            if (current.value("running", false) && !isStale(current)) {
                res.set_content(nlohmann::json{{"ok", false}, {"already_running", true}}.dump(), "application/json");
                return;
            }
        }

        RunOptions opts;
        if (req.has_param("category")) opts.onlyCategories = {req.get_param_value("category")};
        if (req.has_param("ocr_version")) opts.ocrVersion = req.get_param_value("ocr_version");
        if (req.has_param("model_type")) opts.modelType = req.get_param_value("model_type");
        if (req.has_param("dataset")) opts.datasetKey = req.get_param_value("dataset");
        opts.overrides.detBoxThresh = parseQueryParam<float>(req, "det_box_thresh");
        opts.overrides.detThresh = parseQueryParam<float>(req, "det_thresh");
        opts.overrides.detUnclipRatio = parseQueryParam<float>(req, "det_unclip_ratio");
        opts.overrides.detLimitSideLen = parseQueryParam<int>(req, "det_limit_side_len");
        opts.overrides.useAngleCls = parseQueryParam<bool>(req, "use_angle_cls");
        opts.overrides.useTensorrt = parseQueryParam<bool>(req, "use_tensorrt");
        opts.overrides.enablePreprocessing = parseQueryParam<bool>(req, "enable_preprocessing");
        opts.overrides.iouThreshold = parseQueryParam<float>(req, "iou_threshold");
        opts.verbose = false;

        std::thread([opts]() {
            try { ocr_bench::run(opts); } catch (...) {}
        }).detach();

        res.set_content(nlohmann::json{{"ok", true}, {"started", true}}.dump(), "application/json");
    });
}

// --- Summary / Results / Image / Models / Config ---

void ApiServer::registerResultsRoutes() {
    server_->Get("/api/summary", [this](const httplib::Request&, httplib::Response& res) {
        std::string path = config_.reportsRoot + "/summary.json";
        if (!fs::exists(path)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"detail", "no reports yet"}}.dump(), "application/json");
            return;
        }
        std::ifstream in(path);
        res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "application/json");
    });

    server_->Get(R"(/api/results/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string category = req.matches[1];
        std::string perCatDir = config_.reportsRoot + "/per_category";
        if (fs::exists(perCatDir)) {
            for (auto& entry : fs::directory_iterator(perCatDir)) {
                if (entry.path().extension() != ".json") continue;
                std::ifstream in(entry.path());
                nlohmann::json data = nlohmann::json::parse(in, nullptr, false);
                if (data.is_discarded()) continue;
                if (data.value("category", "") == category) {
                    res.set_content(data.dump(), "application/json");
                    return;
                }
            }
        }
        res.status = 404;
        res.set_content(nlohmann::json{{"detail", "category not found: " + category}}.dump(), "application/json");
    });

    server_->Get(R"(/api/image/([^/]+)/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string category = req.matches[1];
        std::string filename = fs::path(req.matches[2].str()).filename().string();
        auto& registry = datasetRegistry();

        fs::path found;
        std::string summaryPath = config_.reportsRoot + "/summary.json";
        if (fs::exists(summaryPath)) {
            std::ifstream in(summaryPath);
            nlohmann::json data = nlohmann::json::parse(in, nullptr, false);
            if (!data.is_discarded()) {
                std::string ds = data.value("overall", nlohmann::json::object()).value("dataset", "");
                if (registry.count(ds)) {
                    fs::path root(registry.at(ds));
                    fs::path c1 = root / category / filename;
                    fs::path c2 = root / category / "images" / filename;
                    if (fs::exists(c1)) found = c1;
                    else if (fs::exists(c2)) found = c2;
                }
            }
        }
        if (found.empty()) {
            for (auto& [key, root] : registry) {
                fs::path c1 = fs::path(root) / category / filename;
                fs::path c2 = fs::path(root) / category / "images" / filename;
                if (fs::exists(c1)) { found = c1; break; }
                if (fs::exists(c2)) { found = c2; break; }
            }
        }
        if (found.empty()) {
            res.status = 404;
            return;
        }
        std::ifstream in(found, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::string ext = found.extension().string();
        std::string mime = (ext == ".png") ? "image/png" : "image/jpeg";
        res.set_content(content, mime);
    });

    server_->Get("/api/models", [](const httplib::Request&, httplib::Response& res) {
        Settings s = loadSettings();
        nlohmann::json out = {
            {"available", {
                {"PP-OCRv6", {{"model_types", {"tiny", "small", "medium"}}, {"default", "small"}, {"desc", "Latest, best accuracy"}}},
            }},
            {"current", {{"ocr_version", s.ocrVersion}, {"model_type", s.modelType}}},
        };
        res.set_content(out.dump(), "application/json");
    });

    server_->Get("/api/config", [](const httplib::Request&, httplib::Response& res) {
        Settings s = loadSettings();
        auto [activeKey, root] = resolveDatasetRoot();
        (void)root;
        nlohmann::json datasetKeys = nlohmann::json::array();
        for (auto& [key, path] : datasetRegistry()) {
            std::string label = (key == "ind_cn") ? "IMG_OCR_IND_CN (labelme)"
                               : (key == "new") ? "FUNSD-form (testing + training)" : key;
            datasetKeys.push_back({{"key", key}, {"label", label}});
        }
        // Round floats to avoid C++ precision artifacts (0.30000001192092896 → 0.3)
        auto round2 = [](float v) -> double { return std::round(v * 100.0) / 100.0; };
        nlohmann::json out = {
            {"dataset", activeKey}, {"dataset_keys", datasetKeys},
            {"iou_threshold", round2(s.iouThreshold)}, {"enable_preprocessing", s.enablePreprocessing},
            {"ocr_version", s.ocrVersion}, {"model_type", s.modelType},
            {"det_box_thresh", round2(s.detBoxThresh)}, {"det_thresh", round2(s.detThresh)},
            {"det_unclip_ratio", round2(s.detUnclipRatio)}, {"det_limit_side_len", s.detLimitSideLen},
            {"use_angle_cls", s.useAngleCls}, {"rec_batch_num", s.recBatchNum}, {"rec_img_width", s.recImgWidth},
            {"use_tensorrt", s.useTensorrt}, {"cuda_available", detectCuda()},
            {"tensorrt_available", detectTensorrt()},
            {"tts_cuda_available", s.useCudaTts}, {"tts_source", s.ttsSource},
            {"serve_host", s.serveHost}, {"serve_port", s.servePort},
            {"auth_password", s.authPassword},
            {"piper_voice_path", s.piperVoicePath},
        };
        res.set_content(out.dump(), "application/json");
    });

    server_->Get("/api/datasets", [](const httplib::Request&, httplib::Response& res) {
        auto& registry = datasetRegistry();
        auto [activeKey, activeRoot] = resolveDatasetRoot();
        (void)activeRoot;
        nlohmann::json datasets = nlohmann::json::array();
        for (auto& [key, rootPath] : registry) {
            auto cats = listCategories(rootPath);
            int nImages = 0, nLines = 0;
            for (auto& cat : cats) {
                auto pages = loadCategory(cat);
                nImages += pages.size();
                for (auto& p : pages) nLines += p.lines.size();
            }
            std::string label = (key == "ind_cn") ? "IMG_OCR_IND_CN (labelme)"
                               : (key == "new") ? "FUNSD-form" : key;
            datasets.push_back({
                {"key", key},
                {"label", label},
                {"format", key == "new" ? "funsd" : "labelme"},
                {"root", rootPath},
                {"n_categories", static_cast<int>(cats.size())},
                {"n_images", nImages},
                {"n_lines", nLines},
                {"active", key == activeKey},
            });
        }
        res.set_content(nlohmann::json{{"datasets", datasets}, {"active", activeKey}}.dump(), "application/json");
    });

    server_->Get(R"(/api/datasets/([^/]+)/categories)", [](const httplib::Request& req, httplib::Response& res) {
        std::string key = req.matches[1];
        auto& registry = datasetRegistry();
        if (!registry.count(key)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"detail", "unknown dataset key: " + key}}.dump(), "application/json");
            return;
        }
        auto cats = listCategories(registry.at(key));
        nlohmann::json categories = nlohmann::json::array();
        int totalImages = 0, totalLines = 0;
        for (auto& cat : cats) {
            auto pages = loadCategory(cat);
            int nImg = pages.size();
            int nLin = 0;
            for (auto& p : pages) nLin += p.lines.size();
            totalImages += nImg;
            totalLines += nLin;
            categories.push_back({{"name", fs::path(cat).filename().string()}, {"n_images", nImg}, {"n_lines", nLin}});
        }
        categories.push_back({{"name", "All"}, {"n_images", totalImages}, {"n_lines", totalLines}});
        res.set_content(nlohmann::json{{"categories", categories}, {"dataset", key}}.dump(), "application/json");
    });

    // GET /api/history — list all benchmark runs
    server_->Get("/api/history", [this](const httplib::Request&, httplib::Response& res) {
        std::string indexPath = config_.reportsRoot + "/history/index.json";
        nlohmann::json runs = nlohmann::json::array();
        if (fs::exists(indexPath)) {
            try {
                std::ifstream in(indexPath);
                runs = nlohmann::json::parse(in);
            } catch (...) {}
        }
        // Reverse so newest is first
        std::reverse(runs.begin(), runs.end());
        res.set_content(nlohmann::json{{"runs", runs}}.dump(), "application/json");
    });

    // GET /api/history/{run_id} — get details for a specific run
    server_->Get(R"(/api/history/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string runId = req.matches[1].str();
        // Sanitize: only allow alphanumeric, dash, underscore
        std::string sanitized;
        for (char c : runId) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
                sanitized += c;
            }
        }
        if (sanitized.empty()) { res.status = 404; return; }

        std::string path = config_.reportsRoot + "/history/" + sanitized + ".json";
        if (!fs::exists(path)) {
            res.status = 404;
            return;
        }
        std::ifstream in(path);
        res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "application/json");
    });
}

// --- TTS + Combined ---

TTSEngine* ApiServer::getTtsEngine(httplib::Response& res) {
    std::lock_guard<std::mutex> lock(ttsEngineMutex_);
    if (ttsEngine_) return ttsEngine_;

    Settings s = loadSettings();
    std::string voicePath = (fs::path(packageRoot()) / s.piperVoicePath).string();
    if (!fs::exists(voicePath)) {
        res.status = 503;
        res.set_content(nlohmann::json{{"detail",
            "TTS voice model not found at " + voicePath +
            ". Run: ocr-bench-download"}}.dump(), "application/json");
        return nullptr;
    }
    std::string espeakPath = (fs::path(packageRoot()) / "models/espeak-ng-data").string();
    try {
        ttsEngine_ = new TTSEngine(voicePath, espeakPath);
    } catch (const std::exception& e) {
        res.status = 503;
        res.set_content(nlohmann::json{{"detail", std::string("TTS voice failed to load: ") + e.what()}}.dump(), "application/json");
        return nullptr;
    }
    return ttsEngine_;
}

void ApiServer::registerTtsCombinedRoutes() {
    // GET /api/tts — synthesize text to WAV
    server_->Get("/api/tts", [this](const httplib::Request& req, httplib::Response& res) {
        std::string text = req.has_param("text") ? req.get_param_value("text") : "";
        if (text.empty()) { res.status = 400; return; }
        auto* engine = getTtsEngine(res);
        if (!engine) return;
        auto [pcm, result] = engine->synthesize(text);
        auto wav = TTSEngine::pcmToWav(pcm, result.sampleRate);
        res.set_header("X-Synth-Ms", std::to_string(result.synthMs));
        res.set_header("X-Audio-Seconds", std::to_string(result.audioSeconds));
        res.set_header("X-Rtf", std::to_string(result.rtf()));
        res.set_header("X-Chars", std::to_string(result.nChars));
        res.set_header("X-First-Chunk-Ms", std::to_string(result.firstChunkMs));
        res.set_header("X-Chars-Per-Sec", std::to_string(result.charsPerSec()));
        res.set_content(std::string(wav.begin(), wav.end()), "audio/wav");
    });

    // POST /api/tts — JSON body {"text": "..."}
    server_->Post("/api/tts", [this](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body = nlohmann::json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("text")) { res.status = 400; return; }
        std::string text = body["text"].get<std::string>();
        if (text.empty()) { res.status = 400; return; }
        auto* engine = getTtsEngine(res);
        if (!engine) return;
        auto [pcm, result] = engine->synthesize(text);
        auto wav = TTSEngine::pcmToWav(pcm, result.sampleRate);
        res.set_header("X-Synth-Ms", std::to_string(result.synthMs));
        res.set_header("X-Audio-Seconds", std::to_string(result.audioSeconds));
        res.set_header("X-Rtf", std::to_string(result.rtf()));
        res.set_content(std::string(wav.begin(), wav.end()), "audio/wav");
    });

    // POST /api/tts/run — background TTS benchmark
    server_->Post("/api/tts/run", [this](const httplib::Request& req, httplib::Response& res) {
        std::string statusPath = config_.reportsRoot + "/.tts_status.json";
        bool force = parseQueryParam<bool>(req, "force").value_or(false);
        if (!force) {
            nlohmann::json idle = {{"running", false}};
            auto current = readStatusFile(statusPath, idle);
            if (current.value("running", false) && !isStale(current)) {
                res.set_content(nlohmann::json{{"ok", false}, {"already_running", true}}.dump(), "application/json");
                return;
            }
        }
        TTSRunOptions opts;
        if (req.has_param("dataset")) opts.datasetKey = req.get_param_value("dataset");
        auto numWorkers = parseQueryParam<int>(req, "num_workers");
        if (numWorkers) opts.numWorkers = *numWorkers;
        opts.verbose = false;
        std::thread([opts]() {
            try { ttsRun(opts); } catch (...) {}
        }).detach();
        res.set_content(nlohmann::json{{"ok", true}, {"started", true}}.dump(), "application/json");
    });

    // GET /api/tts/progress
    server_->Get("/api/tts/progress", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json idle = {{"running", false}, {"total", 0}, {"completed", nlohmann::json::array()}};
        auto status = readStatusFile(config_.reportsRoot + "/.tts_status.json", idle);
        if (status.value("running", false) && isStale(status)) status["stale"] = true;
        res.set_content(status.dump(), "application/json");
    });

    // GET /api/tts/summary
    server_->Get("/api/tts/summary", [this](const httplib::Request&, httplib::Response& res) {
        std::string path = config_.reportsRoot + "/tts_summary.json";
        if (!fs::exists(path)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"detail", "no TTS reports yet"}}.dump(), "application/json");
            return;
        }
        std::ifstream in(path);
        res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "application/json");
    });

    // POST /api/combined/run — background combined benchmark
    server_->Post("/api/combined/run", [this](const httplib::Request& req, httplib::Response& res) {
        std::string statusPath = config_.reportsRoot + "/.combined_status.json";
        bool force = parseQueryParam<bool>(req, "force").value_or(false);
        if (!force) {
            nlohmann::json idle = {{"running", false}};
            auto current = readStatusFile(statusPath, idle);
            if (current.value("running", false) && !isStale(current)) {
                res.set_content(nlohmann::json{{"ok", false}, {"already_running", true}}.dump(), "application/json");
                return;
            }
        }
        CombinedRunOptions opts;
        if (req.has_param("category")) opts.ocrOptions.onlyCategories = {req.get_param_value("category")};
        if (req.has_param("dataset")) opts.ocrOptions.datasetKey = req.get_param_value("dataset");
        if (req.has_param("ocr_version")) opts.ocrOptions.ocrVersion = req.get_param_value("ocr_version");
        if (req.has_param("model_type")) opts.ocrOptions.modelType = req.get_param_value("model_type");
        opts.ocrOptions.verbose = false;
        opts.ttsOptions.verbose = false;
        std::thread([opts]() {
            try { combinedRun(opts); } catch (...) {}
        }).detach();
        res.set_content(nlohmann::json{{"ok", true}, {"started", true}}.dump(), "application/json");
    });

    // GET /api/combined/progress
    server_->Get("/api/combined/progress", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json idle = {{"running", false}, {"total", 0}, {"completed", nlohmann::json::array()}};
        auto status = readStatusFile(config_.reportsRoot + "/.combined_status.json", idle);
        if (status.value("running", false) && isStale(status)) status["stale"] = true;
        res.set_content(status.dump(), "application/json");
    });

    // GET /api/combined/summary
    server_->Get("/api/combined/summary", [this](const httplib::Request&, httplib::Response& res) {
        std::string path = config_.reportsRoot + "/combined_summary.json";
        if (!fs::exists(path)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"detail", "no combined reports yet"}}.dump(), "application/json");
            return;
        }
        std::ifstream in(path);
        res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "application/json");
    });

    // GET /api/combined/history
    server_->Get("/api/combined/history", [this](const httplib::Request&, httplib::Response& res) {
        std::string indexPath = config_.reportsRoot + "/combined_history/index.json";
        nlohmann::json runs = nlohmann::json::array();
        if (fs::exists(indexPath)) {
            try {
                std::ifstream in(indexPath);
                runs = nlohmann::json::parse(in);
            } catch (...) {}
        }
        std::reverse(runs.begin(), runs.end());
        res.set_content(nlohmann::json{{"runs", runs}}.dump(), "application/json");
    });

    // GET /api/combined/history/{run_id}
    server_->Get(R"(/api/combined/history/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string runId = sanitizeRunId(req.matches[1].str());
        if (runId.empty()) { res.status = 404; return; }
        std::string path = config_.reportsRoot + "/combined_history/" + runId + ".json";
        if (!fs::exists(path)) {
            res.status = 404;
            return;
        }
        std::ifstream in(path);
        res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "application/json");
    });
}

// --- Static files ---

void ApiServer::registerStaticRoutes() {
    if (!config_.uiRoot.empty()) {
        server_->set_mount_point("/", config_.uiRoot);

        // Map clean URLs to HTML files: /tts → tts.html, /combined → combined.html
        server_->Get("/tts", [this](const httplib::Request&, httplib::Response& res) {
            std::string path = config_.uiRoot + "/tts.html";
            if (fs::exists(path)) {
                std::ifstream in(path);
                res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "text/html");
            } else {
                res.status = 404;
            }
        });
        server_->Get("/combined", [this](const httplib::Request&, httplib::Response& res) {
            std::string path = config_.uiRoot + "/combined.html";
            if (fs::exists(path)) {
                std::ifstream in(path);
                res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "text/html");
            } else {
                res.status = 404;
            }
        });
        server_->Get("/datasets", [this](const httplib::Request&, httplib::Response& res) {
            std::string path = config_.uiRoot + "/datasets.html";
            if (fs::exists(path)) {
                std::ifstream in(path);
                res.set_content(std::string(std::istreambuf_iterator<char>(in), {}), "text/html");
            } else {
                res.status = 404;
            }
        });
    }
}

} // namespace ocr_bench
