# Opsi 2 — Port RapidOCR TensorRT Engine Builder ke C++ (belum dikerjakan)

Status: **not started**. Opsi 1 (ganti ke `OrtTensorRTProviderOptionsV2` +
explicit profile shapes lewat ONNX Runtime) dipakai duluan karena lebih kecil
scope-nya. Dokumen ini nyimpen jalur Opsi 2 kalau suatu saat perlu reuse
langsung `.engine` file hasil build Python RapidOCR (`multi_*_sm87_fp16.engine`)
tanpa lewat ONNX Runtime TensorRT EP sama sekali.

## Kenapa ini beda dari Opsi 1

Python `rapidocr` build & jalankan TensorRT engine manual pakai `tensorrt`
Python binding, bypass ONNX Runtime buat inference. Opsi 2 = replikasi
pendekatan itu di C++ pakai TensorRT C++ API langsung (`nvinfer1::*`).
Referensi implementasi Python ada di Jetson:
`~/ocr-benchmark/.venv/lib/python3.12/site-packages/rapidocr/inference_engine/tensorrt/engine_builder.py`.

Optimization profile per task (dari `engine_builder.py`):

| Task | min shape | opt shape | max shape |
|---|---|---|---|
| det | (1,3,32,32) | (1,3,736,736) | (1,3,2048,2048) |
| rec | (1,3,48,32) | (6,3,48,320) | (6,3,48,2048) |
| cls (v5) | (1,3,80,160) | (6,3,80,160) | (6,3,80,160) |

Catatan: cls di C++ sekarang pakai `kDstWidth=192, kDstHeight=48`
(`angle_net.hpp:47-48`) — beda dari shape Python di atas (80x160). Kalau opsi
2 dikerjakan, pastikan resize target & profile shape cls disamakan dulu,
atau tetap pakai ONNX Runtime buat cls saja (modelnya kecil, gak worth
effort raw TRT).

## Scope perubahan

1. **`TrtEngineRunner` class baru** (mirror `TRTEngineBuilder` Python)
   - Load `.engine` file langsung via `nvinfer1::IRuntime::deserializeCudaEngine`
   - Kalau file belum ada / shape ga cocok → build dari ONNX pakai
     `nvinfer1::IBuilder` + `IOptimizationProfile::setDimensions(min/opt/max)`,
     serialize, simpan ke `trtCacheDir`
   - Simpan raw `.engine` — **format ini beda dengan cache ONNX Runtime**,
     jadi file `multi_PP-OCRv6_*_sm87_fp16.engine` yang sudah ada di
     `models/trt_engines/` bisa langsung dipakai tanpa build ulang.

2. **Inference path manual per model** (ganti `Ort::Session::Run()`)
   - `nvinfer1::IExecutionContext::enqueueV3` (atau `enqueueV2` tergantung
     versi TRT — Jetson sekarang TRT 10.16.2)
   - Manual `cudaMalloc`/`cudaMemcpyAsync` buat input/output buffer
   - Binding by tensor name (`setInputShape`, `setTensorAddress`)
   - Stream sync sebelum baca output balik ke host
   - Ini nyentuh `db_net.cpp`, `crnn_net.cpp`, `angle_net.cpp` — bagian
     preprocessing (`substractMeanNormalize`, resize) tetap dipakai, cuma
     bagian "kirim ke model & ambil output" yang diganti total.

3. **Dual execution path**
   - CPU/CUDA (Windows dev, Jetson tanpa TRT) → tetap ONNX Runtime, gak
     berubah
   - TensorRT (Jetson prod) → jalur baru raw TRT
   - Perlu abstraksi/interface biar `engine.cpp` gak perlu tau bedanya, atau
     runtime branch di tiap `*_net.cpp` (`useTensorrt ? rawTrtPath() : ortPath()`)

4. **Build system**
   - Link `libnvinfer.so`, `libnvinfer_plugin.so`, CUDA runtime langsung
     (bukan cuma lewat `onnxruntime_providers_tensorrt.so`)
   - Perlu conditional di CMakeLists — cuma aktifkan raw TRT path di Jetson
     build (aarch64 + JetPack), Windows dev tetap ONNX-Runtime-only

## Estimasi

- ~500-800 baris kode baru (`TrtEngineRunner` + 3x inference path rewrite)
- Dual-path maintenance permanen (ORT untuk dev/CPU, raw TRT untuk Jetson prod)
- Manual CUDA memory management → risk leak/sync bug yang sebelumnya
  otomatis dihandle ONNX Runtime

## Kapan worth dikerjain

Cuma kalau Opsi 1 (profile shape via `OrtTensorRTProviderOptionsV2`) ternyata
masih belum cukup — misal butuh startup time lebih cepat dari yang ORT bisa
kasih, atau butuh direct reuse `.engine` Python tanpa build ulang sama sekali
(termasuk build pertama). Untuk kebutuhan sekarang (skip rebuild per-shape),
Opsi 1 sudah cukup.

## Referensi

- Python source: `rapidocr/inference_engine/tensorrt/engine_builder.py`
  (di venv Jetson, path di atas)
- Engine file yang sudah ada & siap reuse kalau opsi ini dikerjakan:
  `models/trt_engines/multi_PP-OCRv6_{det,rec}_{tiny,small,medium}_sm87_fp16.engine`
- TensorRT version di Jetson: `10.16.2` (cek `libnvinfer.so.10.16.2`)
- L4T: `R39.2.0`, arch `aarch64`, GPU compute capability `sm_87`
