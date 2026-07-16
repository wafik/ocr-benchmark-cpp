#include "ocr_bench/tts_engine.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>

#include "piper.h"

namespace ocr_bench {

struct TTSEngine::Impl {
    piper_synthesizer* synth = nullptr;
    int sampleRate = 22050;

    ~Impl() {
        if (synth) piper_free(synth);
    }
};

TTSEngine::TTSEngine(const std::string& modelPath, const std::string& espeakDataPath)
    : impl_(std::make_unique<Impl>())
{
    impl_->synth = piper_create(modelPath.c_str(), nullptr, espeakDataPath.c_str());
    if (!impl_->synth) {
        throw std::runtime_error("Failed to create Piper synthesizer for: " + modelPath);
    }
}

TTSEngine::~TTSEngine() = default;

std::pair<std::vector<int16_t>, TTSResult> TTSEngine::synthesize(const std::string& text) {
    auto t0 = std::chrono::steady_clock::now();

    auto opts = piper_default_synthesize_options(impl_->synth);
    piper_synthesize_start(impl_->synth, text.c_str(), &opts);

    std::vector<int16_t> allSamples;
    piper_audio_chunk chunk{};
    float firstChunkMs = 0.0f;
    bool gotFirstChunk = false;

    // Follow the CLI pattern: loop until chunk.is_last
    do {
        piper_synthesize_next(impl_->synth, &chunk);

        if (chunk.num_samples > 0 && chunk.samples) {
            if (!gotFirstChunk) {
                firstChunkMs = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                gotFirstChunk = true;
                impl_->sampleRate = chunk.sample_rate;
            }

            // Convert float samples [-1,1] to int16
            for (size_t i = 0; i < chunk.num_samples; i++) {
                float sample = chunk.samples[i];
                sample = std::max(-1.0f, std::min(1.0f, sample));
                allSamples.push_back(static_cast<int16_t>(sample * 32767.0f));
            }
        }
    } while (!chunk.is_last);

    auto elapsed = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    int bytesPerSec = 2 * impl_->sampleRate;
    float audioSeconds = allSamples.empty() ? 0.0f
        : static_cast<float>(allSamples.size()) / bytesPerSec;

    TTSResult result;
    result.text = text;
    result.synthMs = elapsed;
    result.firstChunkMs = firstChunkMs;
    result.audioSeconds = audioSeconds;
    result.sampleRate = impl_->sampleRate;
    result.nChars = static_cast<int>(text.size());

    return {allSamples, result};
}

std::vector<uint8_t> TTSEngine::pcmToWav(const std::vector<int16_t>& pcm, int sampleRate) {
    int dataSize = static_cast<int>(pcm.size() * 2);
    int fileSize = 44 + dataSize;
    std::vector<uint8_t> wav(fileSize);

    std::memcpy(wav.data(), "RIFF", 4);
    std::memcpy(wav.data() + 4, &fileSize, 4);
    std::memcpy(wav.data() + 8, "WAVE", 4);

    std::memcpy(wav.data() + 12, "fmt ", 4);
    int fmtSize = 16;
    std::memcpy(wav.data() + 16, &fmtSize, 4);
    int16_t audioFmt = 1;
    std::memcpy(wav.data() + 20, &audioFmt, 2);
    int16_t channels = 1;
    std::memcpy(wav.data() + 22, &channels, 2);
    std::memcpy(wav.data() + 24, &sampleRate, 4);
    int byteRate = sampleRate * 2;
    std::memcpy(wav.data() + 28, &byteRate, 4);
    int16_t blockAlign = 2;
    std::memcpy(wav.data() + 32, &blockAlign, 2);
    int16_t bitsPerSample = 16;
    std::memcpy(wav.data() + 34, &bitsPerSample, 2);

    std::memcpy(wav.data() + 36, "data", 4);
    std::memcpy(wav.data() + 40, &dataSize, 4);
    std::memcpy(wav.data() + 44, pcm.data(), dataSize);

    return wav;
}

} // namespace ocr_bench
