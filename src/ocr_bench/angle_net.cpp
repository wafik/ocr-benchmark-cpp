// Adapted from RapidAI/RapidOcrOnnx's AngleNet.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md.
#include "ocr_bench/angle_net.hpp"

#include <numeric>
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

RawAngle scoreToAngle(const std::vector<float>& outputData) {
    int maxIndex = 0;
    float maxScore = 0.0f;
    for (size_t i = 0; i < outputData.size(); i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = static_cast<int>(i);
        }
    }
    return {maxIndex, maxScore};
}

} // namespace

AngleNet::AngleNet() = default;
AngleNet::~AngleNet() = default;

void AngleNet::loadModel(const std::string& modelPath, bool useCuda,
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
            // Input is always resized to kDstWidth x kDstHeight (192x48)
            // before inference, so min=opt=max — one shape, ever.
            {"trt_profile_min_shapes", "x:1x3x48x192"},
            {"trt_profile_opt_shapes", "x:1x3x48x192"},
            {"trt_profile_max_shapes", "x:1x3x48x192"},
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

RawAngle AngleNet::getAngle(cv::Mat& src) {
    if (!session_) {
        return {-1, 0.0f}; // guard: unloaded model
    }
    auto inputValues = substractMeanNormalize(src, meanValues_, normValues_);
    std::array<int64_t, 4> inputShape{1, src.channels(), src.rows, src.cols};
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
    return scoreToAngle(outputData);
}

std::vector<RawAngle> AngleNet::getAngles(std::vector<cv::Mat>& partImages, bool doAngle, bool mostAngle) {
    size_t n = partImages.size();
    std::vector<RawAngle> angles(n);

    if (!doAngle) {
        for (size_t i = 0; i < n; i++) {
            angles[i] = {-1, 0.0f};
        }
        return angles;
    }

    for (size_t i = 0; i < n; i++) {
        cv::Mat resized;
        cv::resize(partImages[i], resized, cv::Size(kDstWidth, kDstHeight));
        angles[i] = getAngle(resized);
    }

    if (mostAngle) {
        double sum = 0.0;
        for (auto& a : angles) sum += a.index;
        int mostAngleIndex = (sum < static_cast<double>(angles.size()) / 2.0) ? 0 : 1;
        for (auto& a : angles) a.index = mostAngleIndex;
    }

    return angles;
}

} // namespace ocr_bench
