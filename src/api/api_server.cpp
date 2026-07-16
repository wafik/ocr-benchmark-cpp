#include "ocr_bench/api_server.hpp"

#include <ctime>
#include <sstream>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "ocr_bench/paths.hpp"
#include "ocr_bench/run_status.hpp"
#include "ocr_bench/sysmon.hpp"

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
} // namespace

ApiServer::ApiServer(const ApiServerConfig& config) : config_(config) {
    server_ = std::make_unique<httplib::Server>();
    registerHealthRoutes();
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

void ApiServer::registerStaticRoutes() {
    if (!config_.uiRoot.empty()) {
        server_->set_mount_point("/", config_.uiRoot);
    }
}

} // namespace ocr_bench
