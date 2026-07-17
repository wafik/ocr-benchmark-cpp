#pragma once
// Piper TTS wrapper using libpiper C API.
// Wraps piper_create/piper_synthesize_start/piper_synthesize_next/piper_free.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ocr_bench {

struct TTSResult {
    std::string text;
    float synthMs = 0.0f;
    float firstChunkMs = 0.0f;
    float audioSeconds = 0.0f;
    int sampleRate = 22050;
    int nChars = 0;

    float rtf() const {
        return audioSeconds > 0 ? (synthMs / 1000.0f) / audioSeconds : 0.0f;
    }
    float charsPerSec() const {
        return synthMs > 0 ? nChars / (synthMs / 1000.0f) : 0.0f;
    }
};

/// Per-line result from batch synthesis.
struct BatchLineResult {
    std::string text;
    float synthMs = 0.0f;
    float audioSeconds = 0.0f;
    int nChars = 0;
    int sampleRate = 22050;

    float rtf() const {
        return audioSeconds > 0 ? (synthMs / 1000.0f) / audioSeconds : 0.0f;
    }
};

/// Result of batch synthesis (multiple lines).
struct BatchSynthesizeResult {
    std::vector<BatchLineResult> results;
    float totalSynthMs = 0.0f;
};

class TTSEngine {
public:
    /// Load a Piper voice model. Throws on failure.
    TTSEngine(const std::string& modelPath, const std::string& espeakDataPath,
              bool useCuda = false);
    ~TTSEngine();

    TTSEngine(const TTSEngine&) = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;

    /// Synthesize text to 16-bit mono PCM. Returns {pcm, result}.
    std::pair<std::vector<int16_t>, TTSResult> synthesize(const std::string& text);

    /// Synthesize multiple lines, returning per-line timing results.
    /// PCM audio is discarded (benchmark only needs timing stats).
    BatchSynthesizeResult synthesizeBatch(const std::vector<std::string>& texts);

    /// Convert raw 16-bit mono PCM to WAV bytes.
    static std::vector<uint8_t> pcmToWav(const std::vector<int16_t>& pcm, int sampleRate);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ocr_bench
