#include <doctest/doctest.h>
#include <thread>
#include <chrono>
#include "ocr_bench/sysmon.hpp"

using namespace ocr_bench;

TEST_CASE("sampleSystem returns plausible CPU/RAM values") {
    SystemSample s = sampleSystem();
    CHECK(s.cpuPercent >= 0.0f);
    CHECK(s.cpuPercent <= 100.0f);
    CHECK(s.cpuCount > 0);
    CHECK(s.ramPercent >= 0.0f);
    CHECK(s.ramPercent <= 100.0f);
    CHECK(s.ramUsedMb > 0.0f);
    CHECK(s.ramTotalMb > s.ramUsedMb);
    CHECK_FALSE(s.timestamp.empty());
}

TEST_CASE("systemSampleToJson produces the same field names as Python's sample_dict()") {
    SystemSample s = sampleSystem();
    auto j = systemSampleToJson(s);
    CHECK(j.contains("cpu_percent"));
    CHECK(j.contains("cpu_count"));
    CHECK(j.contains("ram_percent"));
    CHECK(j.contains("ram_used_mb"));
    CHECK(j.contains("ram_total_mb"));
    CHECK(j.contains("disk_percent"));
    CHECK(j.contains("cpu_temp_c"));
    CHECK(j.contains("gpus"));
    CHECK(j.contains("timestamp"));
}

TEST_CASE("ResourceMonitor collects samples over time and .latest() reflects them") {
    ResourceMonitor monitor(0.1f);
    CHECK_FALSE(monitor.latest().has_value());
    monitor.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    monitor.stop();
    REQUIRE(monitor.latest().has_value());
    CHECK(monitor.latest()->cpuCount > 0);
}

TEST_CASE("ResourceMonitor.summary() with no samples returns an empty object") {
    ResourceMonitor monitor(2.0f);
    auto s = monitor.summary();
    CHECK(s.empty());
}

TEST_CASE("ResourceMonitor.summary() has the same field names as Python's summary()") {
    ResourceMonitor monitor(0.1f);
    monitor.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    monitor.stop();
    auto s = monitor.summary();
    CHECK(s.contains("samples"));
    CHECK(s.contains("cpu_percent_avg"));
    CHECK(s.contains("cpu_percent_max"));
    CHECK(s.contains("ram_percent_avg"));
    CHECK(s.contains("ram_percent_max"));
    CHECK(s.contains("ram_used_mb_avg"));
    CHECK(s.contains("ram_used_mb_max"));
    CHECK(s.contains("ram_total_mb"));
    CHECK(s.contains("high_load_threshold"));
    CHECK(s.contains("high_load_samples"));
    CHECK(s.contains("high_load_duration_s"));
    CHECK(s["samples"].get<int>() >= 2);
}
