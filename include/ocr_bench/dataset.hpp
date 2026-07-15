#pragma once

#include <string>
#include <vector>

#include "ocr_bench/matcher.hpp"

namespace ocr_bench {

struct GroundTruthLine {
    Polygon polygon;
    std::string text;
};

struct GroundTruthPage {
    std::string imagePath;
    std::string category;
    std::vector<GroundTruthLine> lines;
};

/// Load all pages (image + GT) in one category directory. Auto-detects
/// labelme vs FUNSD-form layout from the directory shape.
std::vector<GroundTruthPage> loadCategory(const std::string& categoryDir);

/// Return category subdirectories under `root`, sorted by name. Skips
/// hidden dirs, dirs starting with `_`, and dirs with no images.
std::vector<std::string> listCategories(const std::string& root);

/// Iterate every page across every category under `root`.
std::vector<GroundTruthPage> iterAllImages(const std::string& root);

} // namespace ocr_bench
