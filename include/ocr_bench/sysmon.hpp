#pragma once
// Ports Python's sysmon.py. Best-effort: every stat degrades to
// std::nullopt when the host doesn't expose it — this file must never
// throw, mirroring Python's contract that a monitoring feature must never
// crash a benchmark run.

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace ocr_bench {

constexpr float kHighLoadThreshold = 90.0f; // percent

struct GpuSample {
    int index = 0;
    std::string name;
    std::optional<float> utilPercent;
    std::optional<float> memUsedMb;
    std::optional<float> memTotalMb;
    std::optional<float> tempC;
};

struct SystemSample {
    float cpuPercent = 0.0f;
    int cpuCount = 0;
    float ramPercent = 0.0f;
    float ramUsedMb = 0.0f;
    float ramTotalMb = 0.0f;
    std::optional<float> diskPercent;
    std::optional<float> cpuTempC;
    std::vector<GpuSample> gpus;
    std::string timestamp;
};

/// One instantaneous reading. Mirrors sysmon.py::sample().
SystemSample sampleSystem();

/// Mirrors sysmon.py::sample_dict() — same field names.
nlohmann::json systemSampleToJson(const SystemSample& s);

/// Background sampler — records CPU/RAM/GPU/temp every `intervalS` while a
/// benchmark run is in flight. Mirrors sysmon.py::ResourceMonitor.
class ResourceMonitor {
public:
    explicit ResourceMonitor(float intervalS = 2.0f);
    ~ResourceMonitor();

    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;

    void start();
    void stop();

    /// Thread-safe read of the most recent sample. Empty if start() was
    /// never called or no sample has completed yet.
    std::optional<SystemSample> latest() const;

    /// Avg/peak rollup across every sample taken during the run. Empty
    /// object if no samples were collected. Mirrors
    /// sysmon.py::ResourceMonitor.summary()'s exact field names.
    nlohmann::json summary() const;

private:
    void loop();

    float intervalS_;
    std::atomic<bool> stopFlag_{false};
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<SystemSample> samples_;
    std::optional<SystemSample> latest_;
    int highLoadCount_ = 0;
};

} // namespace ocr_bench
