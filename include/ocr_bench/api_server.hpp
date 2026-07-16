#pragma once
// HTTP dashboard server. Mirrors Python's api.py route-for-route.

#include <memory>
#include <mutex>
#include <string>

#include <httplib.h>

namespace ocr_bench {

struct ApiServerConfig {
    std::string uiRoot;
    std::string reportsRoot;
    std::string authPassword; // empty = no auth gate
    std::string host = "127.0.0.1";
    int port = 8765;
};

class TTSEngine; // forward declare

class ApiServer {
public:
    explicit ApiServer(const ApiServerConfig& config);
    ~ApiServer();

    /// Blocks, serving requests.
    void listen();

    /// Stop a running listen() call from another thread.
    void stop();

    /// Actual bound port (useful when config.port == 0).
    int boundPort() const;

private:
    void registerHealthRoutes();
    void registerRunRoutes();
    void registerResultsRoutes();
    void registerTtsCombinedRoutes();
    void registerStaticRoutes();
    void registerAuthMiddleware();

    TTSEngine* getTtsEngine(httplib::Response& res);

    ApiServerConfig config_;
    std::unique_ptr<httplib::Server> server_;
    int boundPort_ = 0;

    // Lazy-loaded TTS engine (Plan 5)
    TTSEngine* ttsEngine_ = nullptr;
    std::mutex ttsEngineMutex_;
};

} // namespace ocr_bench
