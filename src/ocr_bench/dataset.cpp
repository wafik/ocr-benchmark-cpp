#include "ocr_bench/dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace ocr_bench {

namespace {

bool isFunsdSplitDir(const fs::path& dir) {
    return fs::is_directory(dir / "images") && fs::is_directory(dir / "annotations");
}

bool hasImages(const fs::path& dir) {
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".png") return true;
    }
    return false;
}

bool isLabelmeCategory(const fs::path& dir) {
    if (!fs::is_directory(dir)) return false;
    if (fs::is_directory(dir / "images")) return false; // FUNSD, not labelme
    return hasImages(dir);
}

std::vector<GroundTruthLine> parseLabelme(const fs::path& jsonPath) {
    std::ifstream in(jsonPath);
    json data = json::parse(in, nullptr, /*allow_exceptions=*/false);
    std::vector<GroundTruthLine> out;
    if (data.is_discarded() || !data.contains("shapes")) return out;
    for (const auto& shape : data["shapes"]) {
        std::string text = shape.value("label", "");
        if (text.empty()) continue;
        auto points = shape.value("points", json::array());
        if (points.size() < 2) continue;
        Polygon poly;
        for (const auto& pt : points) {
            poly.push_back({pt[0].get<float>(), pt[1].get<float>()});
        }
        out.push_back({poly, text});
    }
    return out;
}

std::vector<GroundTruthPage> loadLabelmeCategory(const fs::path& categoryDir) {
    std::vector<GroundTruthPage> pages;
    for (const std::string& ext : {".jpg", ".png"}) {
        std::vector<fs::path> matches;
        for (const auto& entry : fs::directory_iterator(categoryDir)) {
            if (!entry.is_regular_file()) continue;
            auto entryExt = entry.path().extension().string();
            std::transform(entryExt.begin(), entryExt.end(), entryExt.begin(), ::tolower);
            if (entryExt == ext) matches.push_back(entry.path());
        }
        std::sort(matches.begin(), matches.end());
        for (const auto& imagePath : matches) {
            fs::path jsonPath = imagePath;
            jsonPath.replace_extension(".json");
            if (!fs::exists(jsonPath)) continue;
            auto lines = parseLabelme(jsonPath);
            pages.push_back({imagePath.string(), categoryDir.filename().string(), lines});
        }
    }
    return pages;
}

Polygon boxToPolygon(const std::array<float, 4>& box) {
    return {
        {box[0], box[1]}, {box[2], box[1]},
        {box[2], box[3]}, {box[0], box[3]},
    };
}

std::vector<GroundTruthLine> parseFunsd(const fs::path& jsonPath) {
    std::ifstream in(jsonPath);
    json data = json::parse(in, nullptr, false);
    std::vector<GroundTruthLine> out;
    if (data.is_discarded() || !data.contains("form")) return out;
    for (const auto& entry : data["form"]) {
        std::string text = entry.value("text", "");
        auto box = entry.value("box", json::array());
        if (text.empty() || box.size() < 4) continue;
        std::array<float, 4> b = {
            box[0].get<float>(), box[1].get<float>(),
            box[2].get<float>(), box[3].get<float>(),
        };
        out.push_back({boxToPolygon(b), text});
    }
    return out;
}

std::vector<GroundTruthPage> loadFunsdCategory(const fs::path& splitDir) {
    std::vector<GroundTruthPage> pages;
    fs::path imagesDir = splitDir / "images";
    fs::path annosDir = splitDir / "annotations";
    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            matches.push_back(entry.path());
        }
    }
    std::sort(matches.begin(), matches.end());
    for (const auto& imagePath : matches) {
        fs::path jsonPath = annosDir / (imagePath.stem().string() + ".json");
        if (!fs::exists(jsonPath)) continue;
        auto lines = parseFunsd(jsonPath);
        pages.push_back({imagePath.string(), splitDir.filename().string(), lines});
    }
    return pages;
}

} // namespace

std::vector<GroundTruthPage> loadCategory(const std::string& categoryDir) {
    fs::path dir(categoryDir);
    if (isFunsdSplitDir(dir)) {
        return loadFunsdCategory(dir);
    }
    return loadLabelmeCategory(dir); // also the fallback for empty/unknown dirs
}

std::vector<std::string> listCategories(const std::string& root) {
    fs::path rootPath(root);
    std::vector<std::string> cats;
    if (!fs::exists(rootPath)) return cats;
    std::vector<fs::path> entries;
    for (const auto& entry : fs::directory_iterator(rootPath)) {
        entries.push_back(entry.path());
    }
    std::sort(entries.begin(), entries.end());
    for (const auto& p : entries) {
        if (!fs::is_directory(p)) continue;
        std::string name = p.filename().string();
        if (!name.empty() && (name[0] == '_' || name[0] == '.')) continue;
        if (isFunsdSplitDir(p)) {
            cats.push_back(p.string());
            continue;
        }
        if (hasImages(p)) {
            cats.push_back(p.string());
        }
    }
    return cats;
}

std::vector<GroundTruthPage> iterAllImages(const std::string& root) {
    std::vector<GroundTruthPage> pages;
    for (const auto& cat : listCategories(root)) {
        auto catPages = loadCategory(cat);
        pages.insert(pages.end(), catPages.begin(), catPages.end());
    }
    return pages;
}

} // namespace ocr_bench
