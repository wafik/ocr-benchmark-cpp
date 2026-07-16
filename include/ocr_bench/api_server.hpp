#pragma once
// HTTP dashboard server. Mirrors Python's api.py route-for-route.

#include <memory>
#include <string>

namespace httplib { class Server; }

namespace ocr_bench {

struct ApiServerConfig {
    std::string uiRoot;
    std::string reportsRoot;
    std::string authPassword; // empty = no auth gate
    std::string host = "127.0.0.1";
    int port = 8765;
};

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
    void registerStaticRoutes();

    ApiServerConfig config_;
    std::unique_ptr<httplib::Server> server_;
    int boundPort_ = 0;
};

} // namespace ocr_bench
