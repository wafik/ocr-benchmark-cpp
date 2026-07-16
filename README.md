# OCR Bench C++ Rewrite

Rewrite dari `ocr-benchmark` (Python) ke C++. Performa lebih tinggi, dependency lebih ringan.

## Fitur

- **OCR Engine**: DbNet (detection) + AngleNet (classification) + CrnnNet (recognition)
- **TTS Engine**: Piper TTS (Bahasa Indonesia + English)
- **HTTP Dashboard**: cpp-httplib server dengan UI yang sama dengan Python
- **CLI Tools**: Benchmark runner, model downloader
- **TensorRT Support**: Auto-detection + FP16 + engine caching
- **Auth Middleware**: Basic Auth configurable

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
│   ├── piper-voices/      # TTS voice models
│   │   ├── id/            # Indonesian voice
│   │   └── en/            # English voice
│   └── trt_engines/       # TensorRT cache (auto-generated)
└── reports/               # Generated reports
```

## Build

### Prerequisites
- Visual Studio 2022 (MSVC)
- CMake 3.20+
- vcpkg
- (Optional) NVIDIA GPU + TensorRT for acceleration

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

# Start server (default port 8765)
./ocr-bench-serve.exe

# Start with custom settings
./ocr-bench-serve.exe --port 8080 --auth-password mysecret
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

# Run with TensorRT acceleration
./ocr-bench-run.exe --use-tensorrt

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
  - Params: `category`, `ocr_version`, `model_type`, `force`, `use_tensorrt`
- `GET /api/summary` → Overall summary
- `GET /api/results/{category}` → Per-category results
- `GET /api/image/{category}/{filename}` → Source image

### Configuration
- `GET /api/models` → Available models
- `GET /api/config` → Current settings
- `GET /api/datasets` → Dataset registry
- `GET /api/datasets/{key}/categories` → Category list

### TTS (Piper)
- `GET /api/tts?text=...` → Synthesize text to WAV
- `POST /api/tts` → JSON body `{"text": "..."}`
- `POST /api/tts/run` → Start TTS benchmark
- `GET /api/tts/progress` → TTS progress
- `GET /api/tts/summary` → TTS summary

### Combined (OCR + TTS)
- `POST /api/combined/run` → Start combined benchmark
- `GET /api/combined/progress` → Progress
- `GET /api/combined/summary` → Summary
- `GET /api/combined/history` → Run history

## Testing

```powershell
# Run all tests (89 tests, 298 assertions)
cd build/windows-x64/Release
./ocr_bench_tests.exe

# Run specific test
./ocr_bench_tests.exe --test-case="*OCR*"
./ocr_bench_tests.exe --test-case="*TTS*"

# Run with output
./ocr_bench_tests.exe -s
```

## Models

### OCR Models (models/)
| Model | Size | Purpose |
|-------|------|---------|
| `PP-OCRv6_det.onnx` | 1.7MB | Text detection |
| `PP-OCRv6_det_small.onnx` | 9.5MB | Text detection (better accuracy) |
| `PP-OCRv6_det_medium.onnx` | 59.2MB | Text detection (best accuracy) |
| `PP-OCRv6_cls.onnx` | 572KB | Angle classification |
| `PP-OCRv6_rec_tiny.onnx` | 4.3MB | Text recognition |
| `PP-OCRv6_rec_small.onnx` | 20.3MB | Text recognition (better accuracy) |
| `PP-OCRv6_rec_medium.onnx` | 73.1MB | Text recognition (best accuracy) |

### TTS Models (models/piper-voices/)
| Voice | Language | Size |
|-------|----------|------|
| `id_ID-news_tts-medium.onnx` | 🇮🇩 Indonesia | 60MB |
| `en_US-kristin-medium.onnx` | 🇺🇸 English | 60MB |

## Performance

### OCR (AMD Ryzen 5 7600, CPU)
- Detection: ~50ms
- Classification: ~10ms
- Recognition: ~200ms
- **Total per image: ~250ms** (tiny model)

### TTS (Piper, CPU)
- RTF: 0.089 (11x faster than real-time)
- Synthesis: 67ms for 0.75s audio
- Sample rate: 22050 Hz

### TensorRT Acceleration
- First run: ~30s model compilation (cached to `models/trt_engines/`)
- Subsequent runs: ~2-3x faster inference
- FP16 enabled for better performance

## Test Results

```
89/89 tests passed
298/298 assertions passed
OCR E2E: 31 lines detected from real receipt image
Confidence: 0.75-0.86
TTS: Indonesian voice working, 67ms synthesis, RTF 0.089
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

### Plan 5: TTS
- tts_engine.cpp - Piper TTS wrapper
- tts_runner.cpp - TTS benchmark orchestrator
- combined_runner.cpp - OCR+TTS combined runner

## Dependencies

- onnxruntime 1.23.2 (with TensorRT/CUDA support)
- OpenCV 4.12.0
- nlohmann-json 3.12.0
- cpp-httplib
- cxxopts
- curl 8.21.0
- doctest 2.5.3
- piper-tts (libpiper)
- espeak-ng (phonemization)

## License

Same as original Python project.
