// Adapted from RapidAI/RapidOcrOnnx's CrnnNet.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. loadKeysFromModelMetadata() is new code, not
// present in the upstream project (see crnn_net.hpp doc comment).
#include "ocr_bench/crnn_net.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

#include "ocr_bench/ocr_utils.hpp"

namespace ocr_bench {

namespace {

std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session* session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> names;
    for (size_t i = 0; i < session->GetInputCount(); i++) {
        names.push_back(session->GetInputNameAllocated(i, allocator));
    }
    return names;
}

std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session* session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> names;
    for (size_t i = 0; i < session->GetOutputCount(); i++) {
        names.push_back(session->GetOutputNameAllocated(i, allocator));
    }
    return names;
}

template <typename ForwardIt>
size_t argmax(ForwardIt first, ForwardIt last) {
    return static_cast<size_t>(std::distance(first, std::max_element(first, last)));
}

} // namespace

CrnnNet::CrnnNet() = default;
CrnnNet::~CrnnNet() = default;

void CrnnNet::loadModel(const std::string& modelPath, bool useCuda,
                         bool useTensorrt, const std::string& trtCacheDir) {
    sessionOptions_.SetInterOpNumThreads(0);
    sessionOptions_.SetIntraOpNumThreads(0);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (useTensorrt) {
        Ort::TensorRTProviderOptions trtOpts;
        std::unordered_map<std::string, std::string> opts = {
            {"device_id", "0"},
            {"trt_max_workspace_size", "1073741824"},
            {"trt_fp16_enable", "1"},
            {"trt_engine_cache_enable", "1"},
            // Height is fixed at kDstHeight=48; width varies with crop
            // aspect ratio. Pin one profile range so ORT TRT never rebuilds
            // per-shape (see TENSORRT_ENGINE_PORT_PLAN.md Opsi 1).
            {"trt_profile_min_shapes", "x:1x3x48x32"},
            {"trt_profile_opt_shapes", "x:1x3x48x320"},
            {"trt_profile_max_shapes", "x:1x3x48x2048"},
        };
        if (!trtCacheDir.empty()) {
            opts["trt_engine_cache_path"] = trtCacheDir;
        }
        trtOpts.Update(opts);
        sessionOptions_.AppendExecutionProvider_TensorRT_V2(*trtOpts);
    }

    if (useCuda) {
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
    }
#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
#endif
    inputNamesPtr_ = getInputNames(session_.get());
    outputNamesPtr_ = getOutputNames(session_.get());
}

void CrnnNet::finalizeKeys(std::vector<std::string> rawKeys) {
    if (rawKeys.empty()) {
        keys_.clear();
        return;
    }
    keys_.clear();
    keys_.reserve(rawKeys.size() + 2);
    keys_.emplace_back("#"); // CTC blank, index 0
    for (auto& k : rawKeys) keys_.push_back(std::move(k));
    keys_.emplace_back(" "); // trailing space, matches RapidOcrOnnx's convention
}

void CrnnNet::loadKeysFromFile(const std::string& keysPath) {
    std::ifstream in(keysPath);
    if (!in.is_open()) {
        keys_.clear();
        return;
    }
    std::vector<std::string> rawKeys;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // CRLF fixtures on Windows
        rawKeys.push_back(line);
    }
    finalizeKeys(std::move(rawKeys));
}

bool CrnnNet::loadKeysFromModelMetadata() {
    if (!session_) {
        return false;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session_->GetModelMetadata();
    std::string chars;
    try {
        Ort::AllocatedStringPtr charsPtr = metadata.LookupCustomMetadataMapAllocated("character", allocator);
        if (!charsPtr) return false;
        chars = charsPtr.get();
    } catch (const Ort::Exception&) {
        return false;
    }
    if (chars.empty()) {
        return false;
    }
    std::vector<std::string> rawKeys;
    std::istringstream iss(chars);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        rawKeys.push_back(line);
    }
    if (rawKeys.empty()) {
        return false;
    }
    finalizeKeys(std::move(rawKeys));
    return true;
}

RawTextLine CrnnNet::scoreToTextLine(const std::vector<float>& outputData, size_t h, size_t w) const {
    auto keySize = keys_.size();
    auto dataSize = outputData.size();
    std::string text;
    std::vector<float> scores;
    size_t lastIndex = 0;

    for (size_t i = 0; i < h; i++) {
        size_t start = i * w;
        size_t stop = (i + 1) * w;
        if (stop > dataSize) stop = dataSize; // guard against short buffers (differs from
                                               // RapidOcrOnnx's `dataSize - 1`, which can
                                               // under-read the last timestep by one class)
        if (start >= stop) continue;
        size_t maxIndex = argmax(outputData.begin() + start, outputData.begin() + stop);
        float maxValue = *std::max_element(outputData.begin() + start, outputData.begin() + stop);

        if (maxIndex > 0 && maxIndex < keySize && !(i > 0 && maxIndex == lastIndex)) {
            scores.push_back(maxValue);
            text += keys_[maxIndex];
        }
        lastIndex = maxIndex;
    }
    return {text, scores};
}

RawTextLine CrnnNet::getTextLine(const cv::Mat& src) {
    if (!session_) {
        return {"", {}}; // guard: unloaded model
    }
    float scale = static_cast<float>(kDstHeight) / static_cast<float>(src.rows);
    int dstWidth = static_cast<int>(static_cast<float>(src.cols) * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(dstWidth, kDstHeight));

    auto inputValues = substractMeanNormalize(resized, meanValues_, normValues_);
    std::array<int64_t, 4> inputShape{1, resized.channels(), resized.rows, resized.cols};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());

    std::vector<const char*> inputNames = {inputNamesPtr_.front().get()};
    std::vector<const char*> outputNames = {outputNamesPtr_.front().get()};
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outCount = std::accumulate(outShape.begin(), outShape.end(), int64_t{1}, std::multiplies<int64_t>());
    float* raw = outputTensors.front().GetTensorMutableData<float>();
    std::vector<float> outputData(raw, raw + outCount);
    return scoreToTextLine(outputData, outShape[1], outShape[2]);
}

std::vector<RawTextLine> CrnnNet::getTextLines(std::vector<cv::Mat>& partImages) {
    std::vector<RawTextLine> lines(partImages.size());
    for (size_t i = 0; i < partImages.size(); i++) {
        lines[i] = getTextLine(partImages[i]);
    }
    return lines;
}

} // namespace ocr_bench
