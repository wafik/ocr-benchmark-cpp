#pragma once

#include <map>
#include <string>

namespace ocr_bench {

/// Repo root — parent directory of the `cpp/rebuild` project this binary
/// is built from. Mirrors Python's `PACKAGE_ROOT` in paths.py.
std::string packageRoot();

/// Registry key -> dataset root on disk. Add new datasets here — every
/// other layer (API, runner, UI) reads from this map.
const std::map<std::string, std::string>& datasetRegistry();

constexpr const char* kDefaultDatasetKey = "ind_cn";

struct ResolvedDataset {
    std::string key;
    std::string root;
};

/// Resolution order: `keyOrPath` argument > `envOverride` > default key.
/// If the resolved value is a known registry key, returns its root.
/// If it's a non-empty string that isn't a key, treats it as a literal
/// filesystem path (legacy back-compat, mirrors Python).
/// Empty/unknown/missing -> falls back to the default key (`ind_cn`).
ResolvedDataset resolveDatasetRoot(const std::string& keyOrPath = "", const std::string& envOverride = "");

} // namespace ocr_bench
