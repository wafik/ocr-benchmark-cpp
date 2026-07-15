#include "ocr_bench/matcher.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace ocr_bench {

std::array<float, 4> aabb(const Polygon& polygon) {
    float xmin = std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymax = std::numeric_limits<float>::lowest();
    for (const auto& p : polygon) {
        xmin = std::min(xmin, p.x);
        ymin = std::min(ymin, p.y);
        xmax = std::max(xmax, p.x);
        ymax = std::max(ymax, p.y);
    }
    return {xmin, ymin, xmax, ymax};
}

float iouPolygon(const Polygon& a, const Polygon& b) {
    auto boxA = aabb(a);
    auto boxB = aabb(b);
    float ix1 = std::max(boxA[0], boxB[0]);
    float iy1 = std::max(boxA[1], boxB[1]);
    float ix2 = std::min(boxA[2], boxB[2]);
    float iy2 = std::min(boxA[3], boxB[3]);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    if (inter == 0.0f) {
        return 0.0f;
    }
    float areaA = std::max(0.0f, boxA[2] - boxA[0]) * std::max(0.0f, boxA[3] - boxA[1]);
    float areaB = std::max(0.0f, boxB[2] - boxB[0]) * std::max(0.0f, boxB[3] - boxB[1]);
    float unionArea = areaA + areaB - inter;
    return unionArea > 0.0f ? inter / unionArea : 0.0f;
}

MatchResult matchPolygons(
    const std::vector<Polygon>& gtPolys,
    const std::vector<Polygon>& prPolys,
    float iouThreshold
) {
    std::vector<std::tuple<float, size_t, size_t>> pairs; // (iou, gtIdx, prIdx)
    for (size_t g = 0; g < gtPolys.size(); ++g) {
        for (size_t p = 0; p < prPolys.size(); ++p) {
            float iouVal = iouPolygon(gtPolys[g], prPolys[p]);
            if (iouVal >= iouThreshold) {
                pairs.emplace_back(iouVal, g, p);
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });

    std::unordered_set<size_t> usedG, usedP;
    MatchResult result;
    for (const auto& [iouVal, g, p] : pairs) {
        if (usedG.count(g) || usedP.count(p)) {
            continue;
        }
        result.matches.emplace_back(g, p, iouVal);
        usedG.insert(g);
        usedP.insert(p);
    }
    for (size_t g = 0; g < gtPolys.size(); ++g) {
        if (!usedG.count(g)) result.unmatchedGtIdx.push_back(g);
    }
    for (size_t p = 0; p < prPolys.size(); ++p) {
        if (!usedP.count(p)) result.unmatchedPrIdx.push_back(p);
    }
    return result;
}

} // namespace ocr_bench
