#include <doctest/doctest.h>
#include <iostream>
#include "ocr_bench/tts_engine.hpp"

using namespace ocr_bench;

TEST_CASE("TTSEngine loads piper voice and synthesizes audio") {
    std::string modelPath = "models/piper-voices/id/id_ID-news_tts-medium.onnx";
    std::string espeakPath = "vendor/piper/libpiper/build_final/install/espeak-ng-data";

    TTSEngine engine(modelPath, espeakPath);

    auto [pcm, result] = engine.synthesize("Selamat pagi dunia");

    std::cout << "=== TTS Piper Test ===" << std::endl;
    std::cout << "Text: " << result.text << std::endl;
    std::cout << "Samples: " << pcm.size() << std::endl;
    std::cout << "SynthMs: " << result.synthMs << std::endl;
    std::cout << "AudioSeconds: " << result.audioSeconds << std::endl;
    std::cout << "RTF: " << result.rtf() << std::endl;
    std::cout << "SampleRate: " << result.sampleRate << std::endl;

    CHECK(pcm.size() > 0);
    CHECK(result.audioSeconds > 0.0f);
    CHECK(result.synthMs > 0.0f);
}
