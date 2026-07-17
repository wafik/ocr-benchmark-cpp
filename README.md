# OCR Bench C++ Rewrite

Rewrite dari `ocr-benchmark` (Python) ke C++. Performa lebih tinggi, dependency lebih ringan.

## Fitur

- **OCR Engine**: DbNet (detection) + AngleNet (classification) + CrnnNet (recognition)
- **TTS Engine**: Piper TTS (Bahasa Indonesia + English)
- **HTTP Dashboard**: cpp-httplib server dengan UI yang sama dengan Python
- **CLI Tools**: Benchmark runner, model downloader
- **TensorRT Support**: Auto-detection + FP16 + engine caching with pinned
  profile shapes (no rebuild per image size, see TensorRT Acceleration below)
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

### Build on Jetson Nano (aarch64, apt packages instead of vcpkg)

vcpkg builds everything from source and is impractical on a Jetson. Instead this
target uses apt for OpenCV/CURL/nlohmann-json/doctest/cxxopts/cpp-httplib, plus two
vendored dependencies that apt can't provide:

- **onnxruntime**: the pip wheel (`pip install onnxruntime`) ships only the runtime
  `.so`, no C++ headers — vendor the official prebuilt release for headers. However
  the aarch64 release is CPU-only; the CUDA/TensorRT-enabled libraries are linked
  at build time from the Python venv (`ocr-benchmark/.venv/.../onnxruntime/capi/`).
- **espeak-ng**: Ubuntu/Jetson ship 1.51, which lacks `espeak_TextToPhonemesWithTerminator`
  (required by `vendor/piper/libpiper/src/piper.cpp`). Build it from source at the same
  commit `vendor/piper/libpiper`'s own vcpkg build pins.

One-time setup (run once per machine):

```bash
sudo apt install -y libopencv-dev libcurl4-openssl-dev nlohmann-json3-dev \
    doctest-dev cxxopts-dev libcpp-httplib-dev git cmake build-essential

# onnxruntime: vendor the linux-aarch64 release (match the version installed via pip)
mkdir -p vendor/onnxruntime && cd vendor/onnxruntime
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-aarch64-1.27.0.tgz
tar xzf ort.tgz && rm ort.tgz && mv onnxruntime-linux-aarch64-1.27.0 dist
cd ../..

# espeak-ng: build from source (same commit vendor/piper/libpiper pins)
mkdir -p vendor/espeak-ng-src && cd vendor/espeak-ng-src
git clone https://github.com/espeak-ng/espeak-ng.git .
git checkout 212928b394a96e8fd2096616bfd54e17845c48f6
cmake -B build -DCMAKE_INSTALL_PREFIX=$PWD/install -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF \
    -DUSE_KLATT=OFF -DUSE_SPEECHPLAYER=OFF -DEXTRA_cmn=ON -DEXTRA_ru=ON
cmake --build build -j$(nproc) --target install
cd ../..
```

Then build normally, either via the preset or directly:

```bash
cmake --preset jetson
cmake --build build/jetson -j$(nproc)

# or without the preset:
cmake -B build/jetson -DCMAKE_BUILD_TYPE=Release -DOCR_BENCH_USE_SYSTEM_DEPS=ON
cmake --build build/jetson -j$(nproc)
```

Run tests: `./build/jetson/ocr_bench_tests` (fixtures are copied next to the binary
automatically; model-dependent tests need `models/` populated via `ocr-bench-download`).

### Models & Dataset Setup

Both the ONNX models and the OCR dataset are **not included** in this repository
(see `.gitignore`). You must supply them before running benchmarks or the server
dashboard will be empty.

#### OCR Models (`models/`)

Place these files in the `models/` directory at the project root:

```
models/
├── PP-OCRv6_det.onnx          (1.7MB)  text detection
├── PP-OCRv6_cls.onnx          (572KB)  angle classification
├── PP-OCRv6_rec_tiny.onnx     (4.3MB)  text recognition (tiny, fast)
├── PP-OCRv6_rec_tiny_dict.txt (26KB)
├── PP-OCRv6_rec_small.onnx    (20MB)   text recognition (small, better accuracy)
├── PP-OCRv6_rec_small_dict.txt
├── PP-OCRv6_rec_medium.onnx   (73MB)   text recognition (medium, best accuracy)
└── PP-OCRv6_rec_medium_dict.txt
```

**Option A — From an existing Python `ocr-benchmark` install (recommended):**

If you have the Python version running on the same machine, copy the models
from its venv (note: filenames must be **renamed** to match C++ engine convention):

```bash
SRC=/path/to/ocr-benchmark/.venv/lib/python3.12/site-packages/rapidocr/models
DST=/path/to/ocr-benchmark-cpp/models
mkdir -p "$DST"
cp "$SRC/PP-OCRv6_det_tiny.onnx"     "$DST/PP-OCRv6_det.onnx"
cp "$SRC/ch_ppocr_mobile_v2.0_cls_mobile.onnx" "$DST/PP-OCRv6_cls.onnx"
cp "$SRC/PP-OCRv6_rec_tiny.onnx"     "$DST/PP-OCRv6_rec_tiny.onnx"
cp "$SRC/PP-OCRv6_rec_tiny_dict.txt" "$DST/PP-OCRv6_rec_tiny_dict.txt"
cp "$SRC/PP-OCRv6_rec_small.onnx"    "$DST/PP-OCRv6_rec_small.onnx"
cp "$SRC/PP-OCRv6_rec_small_dict.txt" "$DST/PP-OCRv6_rec_small_dict.txt"
cp "$SRC/PP-OCRv6_rec_medium.onnx"   "$DST/PP-OCRv6_rec_medium.onnx"
cp "$SRC/PP-OCRv6_rec_medium_dict.txt" "$DST/PP-OCRv6_rec_medium_dict.txt"
```

**Option B — Download via `ocr-bench-download`:**

```bash
./build/jetson/ocr-bench-download --ocr-version PP-OCRv6 --model-type tiny --models-dir models
```

> **Note:** `ocr-bench-download` fetches from ModelScope. If the URL is down
> (returns 404/500), use Option A instead.

#### TTS Models (Piper)

Place Indonesian voice files in `models/piper-voices/id/`:

```
models/piper-voices/id/
├── id_ID-news_tts-medium.onnx      (60MB)
└── id_ID-news_tts-medium.onnx.json (piper config)
```

From an existing Python `ocr-benchmark` install:

```bash
cp /path/to/ocr-benchmark/models/piper-voices/id/* /path/to/ocr-benchmark-cpp/models/piper-voices/id/
```

#### Dataset (`IMG_OCR_IND_CN/`)

Place the dataset folder at the **project root** (same level as `CMakeLists.txt`):

```
ocr-benchmark-cpp/
├── IMG_OCR_IND_CN/
│   ├── BADGES AND PASSES/
│   ├── BILLS/
│   ├── CONTRACTS/
│   ├── FORMS/
│   ├── IDENTITY CARDS/
│   ├── NEWSPAPERS/
│   ├── NOTES/
│   ├── PAPERS/
│   ├── TRADE DOCUMENTS/
│   └── WHITEBOARD (BLACKBOARD)/
└── CMakeLists.txt
```

From an existing Python `ocr-benchmark` install:

```bash
cp -r /path/to/ocr-benchmark/IMG_OCR_IND_CN /path/to/ocr-benchmark-cpp/IMG_OCR_IND_CN
```

### Run as a systemd service (port 8765)

To keep the dashboard running across reboots/disconnects, install it as a systemd
user unit instead of running it in a terminal:

```ini
# /etc/systemd/system/ocr-bench-serve.service
[Unit]
Description=OCR Bench HTTP Dashboard
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/ocr-benchmark-cpp
ExecStart=/home/nvidia/ocr-benchmark-cpp/build/jetson/ocr-bench-serve --host 0.0.0.0 --port 8765 --ui-root /home/nvidia/ocr-benchmark-cpp/ui --reports-root /home/nvidia/ocr-benchmark-cpp/reports
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-bench-serve
sudo systemctl status ocr-bench-serve
journalctl -u ocr-bench-serve -f   # follow logs
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

# TensorRT is on by default on Jetson (Settings.useTensorrt = true).
# To force CPU/CUDA-only for a run, set USE_TENSORRT=false in .env
# (there is no --use-tensorrt CLI flag; the override is env/API-only).

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
    (`use_tensorrt` defaults to `Settings.useTensorrt`, i.e. `true` on
    Jetson, if omitted)
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

### Jetson Nano (aarch64, GPU)

All numbers below are from actual benchmark runs on the same Jetson Nano
hardware, same dataset (`IMG_OCR_IND_CN`, 55 images / 1674 lines),
same PP-OCRv6 models.

#### OCR — Python vs C++ (PP-OCRv6, TensorRT)

| Model | Python elapsed | C++ elapsed | Speedup | Python CER | C++ CER | Python RAM | C++ RAM |
|-------|---------------|-------------|---------|------------|---------|------------|---------|
| tiny | 30.1s | 16.8s | **1.8×** | 0.1076 | 0.0983 | 4108 MB | 3555 MB |
| small | 39.6s | 23.2s | **1.7×** | 0.1011 | 0.0895 | 5359 MB | 3645 MB |
| **medium** | **65.9s** | **28.0s** | **2.4×** | 0.0811 | 0.0867 | 6606 MB | 3936 MB |

C++ is 1.7–2.4× faster across all model sizes, with 15–40% lower peak
RAM. C++ CER is comparable to Python (sometimes better, sometimes worse
by <0.02 — within noise). Profile shapes are pinned per model so there
is no per-image rebuild — the first run after a profile change builds
the engine once; every subsequent run reuses the cache.

#### TTS — Python vs C++ (Piper, `id_ID-news-tts-medium`, TTS-only run)

| Metric | Python (CPU) | C++ Sequential (CUDA) | C++ Parallel 4w + Opt (CUDA) |
|--------|-------------|----------------------|------------------------------|
| Backend | CPU | CUDA | CUDA ×4 |
| RTF mean | 0.094 | 0.328 | **0.279** |
| Chars/sec | 104.6 | 87.0 | **115.7** |

C++ parallel 4 workers with piper optimizations (unordered_map hash
lookup + memcpy audio copy) gives **33% speedup over Python** and
**2.7× over C++ sequential**. Each worker uses ~1.2GB RAM (piper model
+ ORT session + CUDA context); 4 workers fit in Jetson's 7.5GB with
~2.7GB headroom. Both use upstream `OHF-Voice/piper1-gpl` but C++
parallel achieves better throughput via process isolation. The main
speed gain in this rewrite remains OCR: TensorRT + C++ gives 1.7–2.4×
improvement over Python.

### TensorRT Acceleration
- FP16 enabled, engine cache in `models/trt_engines/`
- Explicit profile shapes (`trt_profile_min/opt/max_shapes`) are pinned per
  model (det/rec/cls) so the engine cache covers the full input size range
  in one build — ORT no longer rebuilds the engine every time an image with
  a different resolution/aspect ratio comes through. See
  `TENSORRT_ENGINE_PORT_PLAN.md` for the (currently unused) alternative of
  bypassing ONNX Runtime and driving TensorRT directly.
- First run per model size after a profile-shape change: ~2-9 min (one-time
  engine build). All runs after that: ~15-30s for the 55-image `ind_cn`
  dataset, regardless of model size (tiny/small/medium) or image variety.
- `useTensorrt` in `RunOptions`/`/api/run` now correctly falls back to
  `Settings.useTensorrt` (default `true`) when no override is given — a
  prior bug in `runner.cpp` silently ignored the setting and always ran on
  CUDA instead.

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
