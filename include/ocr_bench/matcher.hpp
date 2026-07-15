#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <vector>

namespace ocr_bench {

struct Point2f {
    float x;
    float y;
};

using Polygon = std::vector<Point2f>;

/// Returns {xmin, ymin, xmax, ymax} of a polygon's axis-aligned bounding box.
std::array<float, 4> aabb(const Polygon& polygon);

/// IoU of two polygons via their AABBs. Returns 0.0f on zero-area boxes.
/// Polygons are treated as tight rectangles (line text); rotated IoU is
/// intentionally not implemented — see docs/plan.md note on this tradeoff.
float iouPolygon(const Polygon& a, const Polygon& b);

struct MatchResult {
    // Each entry: (gtIndex, prIndex, iouValue)
    std::vector<std::tuple<size_t, size_t, float>> matches;
    std::vector<size_t> unmatchedGtIdx;
    std::vector<size_t> unmatchedPrIdx;
};

/// Greedy IoU matcher. Pairs sorted by IoU descending; first valid match
/// wins. O(n*m) — fine at this scale (n lines per image < 50).
MatchResult matchPolygons(
    const std::vector<Polygon>& gtPolys,
    const std::vector<Polygon>& prPolys,
    float iouThreshold = 0.5f
);

} // namespace ocr_bench
