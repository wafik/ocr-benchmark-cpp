#include "ocr_bench/api_server.hpp"

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

} // namespace

ApiServer::ApiServer(const ApiServerConfig& config) : config_(config) {
    server_ = std::make_unique<httplib::Server>();
    registerHealthRoutes();
    registerRunRoutes();
    registerResultsRoutes();
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
            try {
                ocr_bench::run(opts);
            } catch (...) {}
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
        nlohmann::json out = {
            {"dataset", activeKey}, {"dataset_keys", datasetKeys},
            {"iou_threshold", s.iouThreshold}, {"enable_preprocessing", s.enablePreprocessing},
            {"ocr_version", s.ocrVersion}, {"model_type", s.modelType},
            {"det_box_thresh", s.detBoxThresh}, {"det_thresh", s.detThresh},
            {"det_unclip_ratio", s.detUnclipRatio}, {"det_limit_side_len", s.detLimitSideLen},
            {"use_angle_cls", s.useAngleCls}, {"rec_batch_num", s.recBatchNum}, {"rec_img_width", s.recImgWidth},
            {"use_tensorrt", s.useTensorrt}, {"cuda_available", detectCuda()},
            {"tensorrt_available", false},
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
        nlohmann::json out = nlohmann::json::object();
        for (auto& [key, rootPath] : registry) {
            auto cats = listCategories(rootPath);
            int nImages = 0, nLines = 0;
            for (auto& cat : cats) {
                auto pages = loadCategory(cat);
                nImages += pages.size();
                for (auto& p : pages) nLines += p.lines.size();
            }
            out[key] = {
                {"n_categories", static_cast<int>(cats.size())},
                {"n_images", nImages},
                {"n_lines", nLines},
                {"active", key == activeKey},
            };
        }
        res.set_content(out.dump(), "application/json");
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
        nlohmann::json out = nlohmann::json::array();
        int totalImages = 0, totalLines = 0;
        for (auto& cat : cats) {
            auto pages = loadCategory(cat);
            int nImg = pages.size();
            int nLin = 0;
            for (auto& p : pages) nLin += p.lines.size();
            totalImages += nImg;
            totalLines += nLin;
            out.push_back({{"category", fs::path(cat).filename().string()}, {"n_images", nImg}, {"n_lines", nLin}});
        }
        out.push_back({{"category", "All"}, {"n_images", totalImages}, {"n_lines", totalLines}});
        res.set_content(out.dump(), "application/json");
    });
}

// --- Static files ---

void ApiServer::registerStaticRoutes() {
    if (!config_.uiRoot.empty()) {
        server_->set_mount_point("/", config_.uiRoot);
    }
}

} // namespace ocr_bench
