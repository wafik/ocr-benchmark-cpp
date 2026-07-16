// Ports Python's sysmon.py.
//
// Implementation note on CPU sampling: Python's psutil.cpu_percent(interval)
// supports a non-blocking mode that compares against the timestamp of the
// previous call (primed once at module load via `psutil.cpu_percent(interval=None)`).
// This port always takes a short blocking sample (100ms) instead — simpler,
// avoids needing global mutable state to track "time since last call", and
// the accuracy difference is negligible for a monitor sampled every 2s.
#include "ocr_bench/sysmon.hpp"

#include <array>
#include <cstdio>
#include <ctime>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <pdh.h>
#include <windows.h>
#pragma comment(lib, "pdh.lib")
#else
#include <fstream>
#include <unistd.h>
#endif

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

#ifdef _WIN32

float readCpuPercentWindows() {
    static PDH_HQUERY query = nullptr;
    static PDH_HCOUNTER counter = nullptr;
    if (!query) {
        PdhOpenQuery(nullptr, 0, &query);
        PdhAddEnglishCounterW(query, L"\\Processor(_Total)\\% Processor Time", 0, &counter);
        PdhCollectQueryData(query);
        Sleep(100); // PDH needs two samples to compute a rate
    }
    PdhCollectQueryData(query);
    PDH_FMT_COUNTERVALUE value;
    PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, nullptr, &value);
    return static_cast<float>(value.doubleValue);
}

void readRamWindows(float& percent, float& usedMb, float& totalMb) {
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    totalMb = static_cast<float>(mem.ullTotalPhys) / (1024.0f * 1024.0f);
    float availMb = static_cast<float>(mem.ullAvailPhys) / (1024.0f * 1024.0f);
    usedMb = totalMb - availMb;
    percent = static_cast<float>(mem.dwMemoryLoad);
}

int cpuCountWindows() {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return static_cast<int>(info.dwNumberOfProcessors);
}

#else // Linux

struct CpuTimes { long long idle, total; };

CpuTimes readProcStatOnce() {
    std::ifstream f("/proc/stat");
    std::string cpuLabel;
    long long user, nice, system, idle, iowait, irq, softirq, steal;
    f >> cpuLabel >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    long long idleAll = idle + iowait;
    long long total = user + nice + system + idle + iowait + irq + softirq + steal;
    return {idleAll, total};
}

float readCpuPercentLinux() {
    CpuTimes a = readProcStatOnce();
    usleep(100000); // 100ms window
    CpuTimes b = readProcStatOnce();
    long long totalDelta = b.total - a.total;
    long long idleDelta = b.idle - a.idle;
    if (totalDelta <= 0) return 0.0f;
    return 100.0f * static_cast<float>(totalDelta - idleDelta) / static_cast<float>(totalDelta);
}

void readRamLinux(float& percent, float& usedMb, float& totalMb) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    long totalKb = 0, availKb = 0;
    while (std::getline(f, line)) {
        if (line.rfind("MemTotal:", 0) == 0) sscanf(line.c_str(), "MemTotal: %ld kB", &totalKb);
        if (line.rfind("MemAvailable:", 0) == 0) sscanf(line.c_str(), "MemAvailable: %ld kB", &availKb);
    }
    totalMb = static_cast<float>(totalKb) / 1024.0f;
    float availMb = static_cast<float>(availKb) / 1024.0f;
    usedMb = totalMb - availMb;
    percent = totalMb > 0 ? 100.0f * usedMb / totalMb : 0.0f;
}

int cpuCountLinux() {
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
}

#endif

std::optional<float> readCpuTemp() {
#ifndef _WIN32
    std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
    if (f.is_open()) {
        long milliC = 0;
        f >> milliC;
        if (milliC > 0) return static_cast<float>(milliC) / 1000.0f;
    }
#endif
    return std::nullopt;
}

std::string runCommandCaptureStdout(const std::string& cmd) {
    std::array<char, 256> buffer;
    std::string result;
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return "";
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
        result += buffer.data();
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return result;
}

std::vector<GpuSample> readGpusNvidiaSmi() {
    std::string out = runCommandCaptureStdout(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader,nounits 2>&1"
    );
    std::vector<GpuSample> gpus;
    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        std::vector<std::string> parts;
        std::istringstream lineStream(line);
        std::string part;
        while (std::getline(lineStream, part, ',')) {
            size_t start = part.find_first_not_of(' ');
            parts.push_back(start == std::string::npos ? "" : part.substr(start));
        }
        if (parts.size() != 6) continue;
        try {
            GpuSample g;
            g.index = std::stoi(parts[0]);
            g.name = parts[1];
            g.utilPercent = std::stof(parts[2]);
            g.memUsedMb = std::stof(parts[3]);
            g.memTotalMb = std::stof(parts[4]);
            g.tempC = std::stof(parts[5]);
            gpus.push_back(g);
        } catch (...) {
            continue;
        }
    }
    return gpus;
}

std::vector<GpuSample> readGpusTegrastats() {
    // tegrastats runs continuously; capture one snapshot then kill it.
    std::string out = runCommandCaptureStdout(
        "timeout 2 tegrastats --interval 1000 2>&1 | head -1"
    );
    std::vector<GpuSample> gpus;
    if (out.empty()) return gpus;

    GpuSample g;
    g.index = 0;
    g.name = "Tegra GPU";

    // Parse GR3D_FREQ X% for utilization
    {
        auto pos = out.find("GR3D_FREQ");
        if (pos != std::string::npos) {
            auto pct = out.find('%', pos);
            if (pct != std::string::npos) {
                // walk backwards from '%' to find the number
                auto start = out.rfind(' ', pct);
                if (start != std::string::npos) {
                    g.utilPercent = std::stof(out.substr(start + 1, pct - start - 1));
                }
            }
        }
    }

    // Parse gpu@XX.XC/XX.XC for temperature
    {
        auto pos = out.find("gpu@");
        if (pos != std::string::npos) {
            auto at = out.find('@', pos) + 1;
            auto slash = out.find('/', at);
            if (slash != std::string::npos) {
                g.tempC = std::stof(out.substr(at, slash - at));
            }
        }
    }

    // Jetson shares RAM with GPU — report total system RAM as a rough proxy
    {
        auto pos = out.find("RAM ");
        if (pos != std::string::npos) {
            auto slash = out.find('/', pos + 4);
            if (slash != std::string::npos) {
                g.memTotalMb = std::stof(out.substr(pos + 4, slash - pos - 4));
            }
        }
    }

    gpus.push_back(g);
    return gpus;
}

std::vector<GpuSample> readGpus() {
    try {
        auto gpus = readGpusNvidiaSmi();
        if (!gpus.empty()) return gpus;
    } catch (...) {}
    try {
        return readGpusTegrastats();
    } catch (...) {}
    return {};
}

std::optional<float> readDiskPercent() {
#ifdef _WIN32
    ULARGE_INTEGER freeBytes, totalBytes;
    if (GetDiskFreeSpaceExA(".", &freeBytes, &totalBytes, nullptr)) {
        double total = static_cast<double>(totalBytes.QuadPart);
        double free = static_cast<double>(freeBytes.QuadPart);
        if (total > 0) return static_cast<float>(100.0 * (1.0 - free / total));
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

} // namespace

SystemSample sampleSystem() {
    SystemSample s;
#ifdef _WIN32
    s.cpuPercent = readCpuPercentWindows();
    s.cpuCount = cpuCountWindows();
    readRamWindows(s.ramPercent, s.ramUsedMb, s.ramTotalMb);
#else
    s.cpuPercent = readCpuPercentLinux();
    s.cpuCount = cpuCountLinux();
    readRamLinux(s.ramPercent, s.ramUsedMb, s.ramTotalMb);
#endif
    s.diskPercent = readDiskPercent();
    s.cpuTempC = readCpuTemp();
    s.gpus = readGpus();
    s.timestamp = nowIso8601();
    return s;
}

nlohmann::json systemSampleToJson(const SystemSample& s) {
    nlohmann::json gpusJson = nlohmann::json::array();
    for (auto& g : s.gpus) {
        gpusJson.push_back({
            {"index", g.index},
            {"name", g.name},
            {"util_percent", g.utilPercent.has_value() ? nlohmann::json(*g.utilPercent) : nlohmann::json(nullptr)},
            {"mem_used_mb", g.memUsedMb.has_value() ? nlohmann::json(*g.memUsedMb) : nlohmann::json(nullptr)},
            {"mem_total_mb", g.memTotalMb.has_value() ? nlohmann::json(*g.memTotalMb) : nlohmann::json(nullptr)},
            {"temp_c", g.tempC.has_value() ? nlohmann::json(*g.tempC) : nlohmann::json(nullptr)},
        });
    }
    return {
        {"cpu_percent", s.cpuPercent},
        {"cpu_count", s.cpuCount},
        {"ram_percent", s.ramPercent},
        {"ram_used_mb", s.ramUsedMb},
        {"ram_total_mb", s.ramTotalMb},
        {"disk_percent", s.diskPercent.has_value() ? nlohmann::json(*s.diskPercent) : nlohmann::json(nullptr)},
        {"cpu_temp_c", s.cpuTempC.has_value() ? nlohmann::json(*s.cpuTempC) : nlohmann::json(nullptr)},
        {"gpus", gpusJson},
        {"timestamp", s.timestamp},
    };
}

namespace {
bool isHighLoad(const SystemSample& s) {
    if (s.cpuPercent >= kHighLoadThreshold) return true;
    for (auto& g : s.gpus) {
        if (g.utilPercent.has_value() && *g.utilPercent >= kHighLoadThreshold) return true;
    }
    return false;
}
} // namespace

ResourceMonitor::ResourceMonitor(float intervalS) : intervalS_(intervalS) {}

ResourceMonitor::~ResourceMonitor() {
    stop();
}

void ResourceMonitor::start() {
    stopFlag_ = false;
    thread_ = std::thread(&ResourceMonitor::loop, this);
}

void ResourceMonitor::loop() {
    while (!stopFlag_.load()) {
        try {
            SystemSample s = sampleSystem();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                samples_.push_back(s);
                if (isHighLoad(s)) highLoadCount_++;
                latest_ = s;
            }
        } catch (...) {
            // monitoring must never kill the run — swallow and keep looping
        }
        auto sleepMs = static_cast<int>(intervalS_ * 1000);
        for (int waited = 0; waited < sleepMs && !stopFlag_.load(); waited += 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

void ResourceMonitor::stop() {
    stopFlag_ = true;
    if (thread_.joinable()) thread_.join();
}

std::optional<SystemSample> ResourceMonitor::latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

nlohmann::json ResourceMonitor::summary() const {
    std::vector<SystemSample> samples;
    int highLoadCount;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        samples = samples_;
        highLoadCount = highLoadCount_;
    }
    if (samples.empty()) {
        return nlohmann::json::object();
    }

    auto mean = [](const std::vector<float>& v) -> std::optional<float> {
        if (v.empty()) return std::nullopt;
        float sum = 0.0f;
        for (float x : v) sum += x;
        return sum / v.size();
    };
    auto maxOf = [](const std::vector<float>& v) -> std::optional<float> {
        if (v.empty()) return std::nullopt;
        float m = v[0];
        for (float x : v) m = std::max(m, x);
        return m;
    };
    auto toJson = [](std::optional<float> v) -> nlohmann::json {
        return v.has_value() ? nlohmann::json(*v) : nlohmann::json(nullptr);
    };

    std::vector<float> cpuVals, ramVals, ramUsed, cpuTemps, gpuUtil, gpuMem, gpuTemp;
    std::vector<SystemSample> highLoad;
    for (auto& s : samples) {
        cpuVals.push_back(s.cpuPercent);
        ramVals.push_back(s.ramPercent);
        ramUsed.push_back(s.ramUsedMb);
        if (s.cpuTempC.has_value()) cpuTemps.push_back(*s.cpuTempC);
        for (auto& g : s.gpus) {
            if (g.utilPercent.has_value()) gpuUtil.push_back(*g.utilPercent);
            if (g.memUsedMb.has_value()) gpuMem.push_back(*g.memUsedMb);
            if (g.tempC.has_value()) gpuTemp.push_back(*g.tempC);
        }
        if (isHighLoad(s)) highLoad.push_back(s);
    }

    std::vector<float> highLoadCpuTemps, highLoadGpuTemps;
    for (auto& s : highLoad) {
        if (s.cpuTempC.has_value()) highLoadCpuTemps.push_back(*s.cpuTempC);
        for (auto& g : s.gpus) {
            if (g.tempC.has_value()) highLoadGpuTemps.push_back(*g.tempC);
        }
    }

    auto& lastGpus = samples.back().gpus;
    std::string gpuName = lastGpus.empty() ? "" : lastGpus.front().name;

    return {
        {"samples", static_cast<int>(samples.size())},
        {"cpu_percent_avg", toJson(mean(cpuVals))},
        {"cpu_percent_max", toJson(maxOf(cpuVals))},
        {"ram_percent_avg", toJson(mean(ramVals))},
        {"ram_percent_max", toJson(maxOf(ramVals))},
        {"ram_used_mb_avg", toJson(mean(ramUsed))},
        {"ram_used_mb_max", toJson(maxOf(ramUsed))},
        {"ram_total_mb", samples.back().ramTotalMb},
        {"cpu_temp_c_avg", toJson(mean(cpuTemps))},
        {"cpu_temp_c_max", toJson(maxOf(cpuTemps))},
        {"gpu_name", lastGpus.empty() ? nlohmann::json(nullptr) : nlohmann::json(gpuName)},
        {"gpu_percent_avg", toJson(mean(gpuUtil))},
        {"gpu_percent_max", toJson(maxOf(gpuUtil))},
        {"gpu_mem_used_mb_avg", toJson(mean(gpuMem))},
        {"gpu_mem_used_mb_max", toJson(maxOf(gpuMem))},
        {"gpu_mem_total_mb", lastGpus.empty() ? nlohmann::json(nullptr) : nlohmann::json(lastGpus.front().memTotalMb.value_or(0.0f))},
        {"gpu_temp_c_avg", toJson(mean(gpuTemp))},
        {"gpu_temp_c_max", toJson(maxOf(gpuTemp))},
        {"high_load_threshold", kHighLoadThreshold},
        {"high_load_samples", highLoadCount},
        {"high_load_duration_s", highLoadCount * intervalS_},
        {"high_load_cpu_temp_c_avg", toJson(mean(highLoadCpuTemps))},
        {"high_load_cpu_temp_c_max", toJson(maxOf(highLoadCpuTemps))},
        {"high_load_gpu_temp_c_avg", toJson(mean(highLoadGpuTemps))},
        {"high_load_gpu_temp_c_max", toJson(maxOf(highLoadGpuTemps))},
    };
}

} // namespace ocr_bench
