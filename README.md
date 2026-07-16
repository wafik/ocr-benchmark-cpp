# OCR Bench C++ Rewrite

Rewrite dari `ocr-benchmark` (Python) ke C++. Performa lebih tinggi, dependency lebih ringan.

## Arsitektur

```
cpp/rebuild/
├── include/ocr_bench/     # Headers
├── src/ocr_bench/         # Core library
├── src/api/               # HTTP server
├── cli/                   # CLI executables
├── tests/                 # Unit tests (doctest)
├── ui/                    # Dashboard UI (verbatim from Python)
├── models/                # ONNX models
└── reports/               # Generated reports
```

## Build

### Prerequisites
- Visual Studio 2022 (MSVC)
- CMake 3.20+
- vcpkg

### Build Commands

```powershell
# Set vcpkg root
$env:VCPKG_ROOT = "C:\vcpkg"

# Configure
cmake --preset windows-x64

# Build Release
cmake --build build/windows-x64 --config Release

# Build Debug
cmake --build build/windows-x64 --config Debug
```

## Running

### 1. HTTP Dashboard Server

```powershell
cd build/windows-x64/Release

# Start server
./ocr-bench-serve.exe --port 8765

# Open browser
# http://127.0.0.1:8765
```

**Options:**
- `--host` : Listen host (default: 127.0.0.1)
- `--port` : Listen port (default: 8765)
- `--ui-root` : Path to ui/ directory (default: ui)
- `--reports-root` : Path to reports/ directory (default: reports)
- `--auth-password` : Basic auth password (empty = no auth)

### 2. CLI Benchmark Runner

```powershell
# Run all categories
./ocr-bench-run.exe

# Run specific category
./ocr-bench-run.exe --category "BILLS"

# Run with custom settings
./ocr-bench-run.exe --ocr-version PP-OCRv6 --model-type tiny --dataset ind_cn
```

### 3. Model Downloader

```powershell
# Download tiny models (default)
./ocr-bench-download.exe

# Download specific model type
./ocr-bench-download.exe --model-type small --models-dir models
```

## API Endpoints

### Health & System
- `GET /api/health` → `{"ok": true}`
- `GET /api/system` → CPU/RAM/GPU stats
- `GET /api/progress` → Current run progress

### OCR Run
- `POST /api/run` → Start benchmark run
  - Params: `category`, `ocr_version`, `model_type`, `force`
- `GET /api/summary` → Overall summary
- `GET /api/results/{category}` → Per-category results
- `GET /api/image/{category}/{filename}` → Source image

### Configuration
- `GET /api/models` → Available models
- `GET /api/config` → Current settings
- `GET /api/datasets` → Dataset registry
- `GET /api/datasets/{key}/categories` → Category list

### TTS (Stub - not yet implemented)
- `GET /api/tts?text=...` → Synthesize text (503)
- `POST /api/tts/run` → Start TTS benchmark
- `GET /api/tts/summary` → TTS summary

### Combined (Stub - not yet implemented)
- `POST /api/combined/run` → Start combined benchmark
- `GET /api/combined/progress` → Progress
- `GET /api/combined/summary` → Summary
- `GET /api/combined/history` → Run history

## Testing

```powershell
# Run all tests
cd build/windows-x64/Release
./ocr_bench_tests.exe

# Run specific test
./ocr_bench_tests.exe --test-case="*OCR*"

# Run with output
./ocr_bench_tests.exe -s
```

## Models

Models located in `models/` directory:
- `PP-OCRv6_det.onnx` - Text detection (1.7MB)
- `PP-OCRv6_cls.onnx` - Angle classification (572KB)
- `PP-OCRv6_rec_tiny.onnx` - Text recognition (4.3MB)
- `PP-OCRv6_rec_tiny_dict.txt` - Character dictionary

Models copied from Jetson Nano: `/home/nvidia/ocr-benchmark/.venv/lib/python3.12/site-packages/rapidocr/models/`

## Performance

OCR processing on AMD Ryzen 5 7600:
- **Detection**: ~50ms
- **Classification**: ~10ms
- **Recognition**: ~200ms
- **Total per image**: ~250ms (tiny model, CPU)

## Test Results

```
88/88 tests passed
295/295 assertions passed
OCR E2E: 31 lines detected from real receipt image
Confidence: 0.75-0.86
```

## Project Structure

### Plan 1: Foundation
- matcher.cpp - Polygon IoU matching
- metrics.cpp - CER/WER calculation
- dataset.cpp - Labelme/FUNSD parsers
- paths.cpp - Dataset registry
- config.cpp - .env loader
- report_writer.cpp - JSON/CSV output

### Plan 2: OCR Engine
- db_net.cpp - Text detection (DbNet)
- angle_net.cpp - Angle classification
- crnn_net.cpp - Text recognition (CRNN)
- engine.cpp - End-to-end orchestrator
- model_downloader.cpp - libcurl-based downloader

### Plan 3: Runner + Sysmon
- runner.cpp - Benchmark orchestrator
- sysmon.cpp - CPU/RAM/GPU monitoring
- run_status.cpp - Atomic status writer

### Plan 4: HTTP API + CLI
- api_server.cpp - cpp-httplib server
- main.cpp - Server entrypoint
- cli/run_benchmark.cpp - CLI runner
- cli/download_models.cpp - CLI downloader

## Dependencies

- onnxruntime 1.23.2
- OpenCV 4.12.0
- nlohmann-json 3.12.0
- cpp-httplib
- cxxopts
- curl 8.21.0
- doctest 2.5.3

## License

Same as original Python project.
