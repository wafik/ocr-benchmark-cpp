#pragma once
// Ports the atomic-write + graceful-read pattern from Python's
// runner.py::_write_status() / the read side inlined in api.py's
// _read_status(). Extracted here as standalone functions so runner.cpp,
// tts_runner.cpp, and combined_runner.cpp can each use their own status
// file path with the same write/read contract.

#include <string>

#include <nlohmann/json.hpp>

namespace ocr_bench {

/// Atomically write `status` (with an added `updated_at` ISO-8601 UTC
/// timestamp) to `path`. Writes to `<path>.tmp` first, then renames.
/// Retries the rename up to 8 times with backoff if the target is
/// transiently locked, falling back to a direct non-atomic write if every
/// retry fails. Never throws.
void writeStatusFile(const std::string& path, nlohmann::json status);

/// Read and parse `path`. Returns `idleDefault` if the file doesn't exist
/// or fails to parse. Never throws.
nlohmann::json readStatusFile(const std::string& path, const nlohmann::json& idleDefault);

} // namespace ocr_bench
