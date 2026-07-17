// cli/tts_worker.cpp — Standalone TTS worker for parallel synthesis.
// Reads JSON Lines from stdin, synthesizes each line, writes results to stdout.
// Protocol: {"id":42,"text":"..."} -> {"id":42,"synth_ms":120,"audio_seconds":0.5,...}
// Exits on EOF (stdin closed by parent).
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "ocr_bench/tts_engine.hpp"
#include "ocr_bench/paths.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options opts("ocr-bench-tts-worker", "TTS worker for parallel synthesis");
    opts.add_options()
        ("voice", "Path to Piper voice .onnx", cxxopts::value<std::string>())
        ("espeak-data", "Path to espeak-ng data dir", cxxopts::value<std::string>())
        ("cuda", "Use CUDA execution provider", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    std::string voicePath = result["voice"].as<std::string>();
    std::string espeakPath = result["espeak-data"].as<std::string>();
    bool useCuda = result["cuda"].as<bool>();

    try {
        ocr_bench::TTSEngine engine(voicePath, espeakPath, useCuda);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            nlohmann::json req;
            try {
                req = nlohmann::json::parse(line);
            } catch (const nlohmann::json::parse_error&) {
                nlohmann::json err = {{"error", "invalid JSON"}, {"raw", line}};
                std::cout << err.dump() << std::endl;
                continue;
            }

            int id = req.value("id", -1);
            std::string text = req.value("text", "");

            if (text.empty()) {
                nlohmann::json resp = {{"id", id}, {"synth_ms", 0}, {"audio_seconds", 0}, {"n_chars", 0}};
                std::cout << resp.dump() << std::endl;
                continue;
            }

            auto [pcm, result] = engine.synthesize(text);
            nlohmann::json resp = {
                {"id", id},
                {"synth_ms", result.synthMs},
                {"audio_seconds", result.audioSeconds},
                {"n_chars", result.nChars},
                {"sample_rate", result.sampleRate},
            };
            std::cout << resp.dump() << std::endl;
        }
    } catch (const std::exception& e) {
        nlohmann::json err = {{"error", e.what()}};
        std::cerr << err.dump() << std::endl;
        return 1;
    }

    return 0;
}
