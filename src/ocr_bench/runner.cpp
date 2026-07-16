// Ports Python's runner.py.
#include "ocr_bench/runner.hpp"

#include "ocr_bench/matcher.hpp"

namespace ocr_bench {

namespace {
nlohmann::json polygonToJson(const Polygon& poly) {
    nlohmann::json arr = nlohmann::json::array();
    for (auto& pt : poly) {
        arr.push_back({pt.x, pt.y});
    }
    return arr;
}
} // namespace

PageMetrics evaluatePage(const GroundTruthPage& page, const PagePrediction& pred, float iouThreshold) {
    std::vector<Polygon> gtPolys;
    for (auto& l : page.lines) gtPolys.push_back(l.polygon);
    std::vector<Polygon> prPolys;
    for (auto& l : pred.lines) prPolys.push_back(l.polygon);

    auto matchResult = matchPolygons(gtPolys, prPolys, iouThreshold);

    PageMetrics pm;
    pm.image = page.imagePath.empty() ? pred.image : page.imagePath;
    pm.category = page.category;
    pm.nGt = static_cast<int>(page.lines.size());
    pm.nPred = static_cast<int>(pred.lines.size());
    pm.detection = DetectionStats{
        static_cast<int>(matchResult.matches.size()),
        static_cast<int>(matchResult.unmatchedPrIdx.size()),
        static_cast<int>(matchResult.unmatchedGtIdx.size()),
    };

    for (auto& [gtIdx, prIdx, iouVal] : matchResult.matches) {
        (void)iouVal;
        pm.matchedCer.push_back(cer(page.lines[gtIdx].text, pred.lines[prIdx].text));
        pm.matchedWer.push_back(wer(page.lines[gtIdx].text, pred.lines[prIdx].text));
        pm.matchedConf.push_back(pred.lines[prIdx].score);
    }

    std::string gtJoined, prJoined;
    for (size_t i = 0; i < page.lines.size(); i++) {
        if (i > 0) gtJoined += "\n";
        gtJoined += page.lines[i].text;
    }
    for (size_t i = 0; i < pred.lines.size(); i++) {
        if (i > 0) prJoined += "\n";
        prJoined += pred.lines[i].text;
    }
    pm.joinedCer = cer(gtJoined, prJoined);
    pm.elapsedMs = pred.elapsedMs;
    pm.emptyOutput = (pred.lines.empty() && !page.lines.empty());

    return pm;
}

PageMetricsInput toPageMetricsInput(const PageMetrics& pm) {
    PageMetricsInput input;
    input.nGt = pm.nGt;
    input.detection = pm.detection;
    input.matchedCer = pm.matchedCer;
    input.matchedWer = pm.matchedWer;
    input.matchedConf = pm.matchedConf;
    input.joinedCer = pm.joinedCer;
    input.elapsedMs = pm.elapsedMs;
    input.emptyOutput = pm.emptyOutput;
    return input;
}

nlohmann::json serializePageMetrics(const PageMetrics& pm, const GroundTruthPage& page, const PagePrediction& pred) {
    auto vecAvg = [](const std::vector<float>& v) -> nlohmann::json {
        if (v.empty()) return nullptr;
        float sum = 0;
        for (float x : v) sum += x;
        return sum / v.size();
    };

    nlohmann::json j = {
        {"image", pm.image},
        {"category", pm.category},
        {"n_gt", pm.nGt},
        {"n_pred", pm.nPred},
        {"detection", {{"tp", pm.detection.tp}, {"fp", pm.detection.fp}, {"fn", pm.detection.fn}}},
        {"matched_cer_mean", vecAvg(pm.matchedCer)},
        {"matched_wer_mean", vecAvg(pm.matchedWer)},
        {"mean_confidence", vecAvg(pm.matchedConf)},
        {"joined_cer", pm.joinedCer},
        {"elapsed_ms", pm.elapsedMs},
        {"empty_output", pm.emptyOutput},
    };

    // Build overlays: matched, missed (GT only), spurious (PR only)
    nlohmann::json overlays = nlohmann::json::array();

    // Re-run matching to get index mapping
    std::vector<Polygon> gtPolys;
    for (auto& l : page.lines) gtPolys.push_back(l.polygon);
    std::vector<Polygon> prPolys;
    for (auto& l : pred.lines) prPolys.push_back(l.polygon);
    auto matchResult = matchPolygons(gtPolys, prPolys);

    // Matched overlays
    for (auto& [gtIdx, prIdx, iouVal] : matchResult.matches) {
        overlays.push_back({
            {"gt_polygon", polygonToJson(page.lines[gtIdx].polygon)},
            {"gt_text", page.lines[gtIdx].text},
            {"pr_polygon", polygonToJson(pred.lines[prIdx].polygon)},
            {"pr_text", pred.lines[prIdx].text},
            {"pr_score", pred.lines[prIdx].score},
            {"iou", iouVal},
            {"status", "matched"},
            {"line_cer", cer(page.lines[gtIdx].text, pred.lines[prIdx].text)},
        });
    }

    // Missed overlays (GT not matched)
    for (int gtIdx : matchResult.unmatchedGtIdx) {
        nlohmann::json ov;
        ov["gt_polygon"] = polygonToJson(page.lines[gtIdx].polygon);
        ov["gt_text"] = page.lines[gtIdx].text;
        ov["pr_polygon"] = nullptr;
        ov["pr_text"] = nullptr;
        ov["pr_score"] = nullptr;
        ov["iou"] = nullptr;
        ov["status"] = "missed";
        overlays.push_back(std::move(ov));
    }

    // Spurious overlays (PR not matched)
    for (int prIdx : matchResult.unmatchedPrIdx) {
        nlohmann::json ov;
        ov["gt_polygon"] = nullptr;
        ov["gt_text"] = nullptr;
        ov["pr_polygon"] = polygonToJson(pred.lines[prIdx].polygon);
        ov["pr_text"] = pred.lines[prIdx].text;
        ov["pr_score"] = pred.lines[prIdx].score;
        ov["iou"] = nullptr;
        ov["status"] = "spurious";
        overlays.push_back(std::move(ov));
    }

    j["overlays"] = std::move(overlays);
    return j;
}

} // namespace ocr_bench
