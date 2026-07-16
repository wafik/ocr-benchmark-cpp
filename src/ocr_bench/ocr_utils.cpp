// Adapted from RapidAI/RapidOcrOnnx's OcrUtils.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. getScaleParam/getMinBoxes/boxScoreFast/
// unClipBox/substractMeanNormalize/getRotateCropImage/matRotateClockWise180
// are near-verbatim ports of the corresponding upstream functions.
#include "ocr_bench/ocr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "clipper.hpp" // vendored, see vendor/rapidocronnx/

namespace ocr_bench {

ScaleParam getScaleParam(const cv::Mat& src, int targetSize) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;
    int dstWidth = srcWidth;
    int dstHeight = srcHeight;

    float ratio = (srcWidth > srcHeight)
        ? static_cast<float>(targetSize) / static_cast<float>(srcWidth)
        : static_cast<float>(targetSize) / static_cast<float>(srcHeight);

    dstWidth = static_cast<int>(static_cast<float>(srcWidth) * ratio);
    dstHeight = static_cast<int>(static_cast<float>(srcHeight) * ratio);
    if (dstWidth % 32 != 0) {
        dstWidth = std::max((dstWidth / 32) * 32, 32);
    }
    if (dstHeight % 32 != 0) {
        dstHeight = std::max((dstHeight / 32) * 32, 32);
    }
    float ratioWidth = static_cast<float>(dstWidth) / static_cast<float>(srcWidth);
    float ratioHeight = static_cast<float>(dstHeight) / static_cast<float>(srcHeight);
    return {srcWidth, srcHeight, dstWidth, dstHeight, ratioWidth, ratioHeight};
}

namespace {
bool pointCompareX(const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; }
}

std::vector<cv::Point2f> getMinBoxes(const cv::RotatedRect& boxRect, float& maxSideLen) {
    maxSideLen = std::max(boxRect.size.width, boxRect.size.height);
    cv::Point2f vertices[4];
    boxRect.points(vertices);
    std::vector<cv::Point2f> boxPoint(vertices, vertices + 4);
    std::sort(boxPoint.begin(), boxPoint.end(), pointCompareX);

    int index1, index2, index3, index4;
    if (boxPoint[1].y > boxPoint[0].y) { index1 = 0; index4 = 1; } else { index1 = 1; index4 = 0; }
    if (boxPoint[3].y > boxPoint[2].y) { index2 = 2; index3 = 3; } else { index2 = 3; index3 = 2; }

    std::vector<cv::Point2f> minBox(4);
    minBox[0] = boxPoint[index1];
    minBox[1] = boxPoint[index2];
    minBox[2] = boxPoint[index3];
    minBox[3] = boxPoint[index4];
    return minBox;
}

float boxScoreFast(const std::vector<cv::Point2f>& boxes, const cv::Mat& pred) {
    int width = pred.cols;
    int height = pred.rows;

    float arrayX[4] = {boxes[0].x, boxes[1].x, boxes[2].x, boxes[3].x};
    float arrayY[4] = {boxes[0].y, boxes[1].y, boxes[2].y, boxes[3].y};

    int minX = clampValue(static_cast<int>(std::floor(*std::min_element(arrayX, arrayX + 4))), 0, width - 1);
    int maxX = clampValue(static_cast<int>(std::ceil(*std::max_element(arrayX, arrayX + 4))), 0, width - 1);
    int minY = clampValue(static_cast<int>(std::floor(*std::min_element(arrayY, arrayY + 4))), 0, height - 1);
    int maxY = clampValue(static_cast<int>(std::ceil(*std::max_element(arrayY, arrayY + 4))), 0, height - 1);

    cv::Mat mask = cv::Mat::zeros(maxY - minY + 1, maxX - minX + 1, CV_8UC1);
    cv::Point box[4];
    for (int i = 0; i < 4; ++i) {
        box[i] = cv::Point(static_cast<int>(boxes[i].x) - minX, static_cast<int>(boxes[i].y) - minY);
    }
    const cv::Point* pts[1] = {box};
    int npts[] = {4};
    cv::fillPoly(mask, pts, npts, 1, cv::Scalar(1));

    cv::Mat cropped;
    pred(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(cropped);
    return static_cast<float>(cv::mean(cropped, mask)[0]);
}

namespace {
float getContourArea(const std::vector<cv::Point2f>& box, float unClipRatio) {
    size_t size = box.size();
    float area = 0.0f, dist = 0.0f;
    for (size_t i = 0; i < size; i++) {
        area += box[i].x * box[(i + 1) % size].y - box[i].y * box[(i + 1) % size].x;
        dist += std::sqrt(
            (box[i].x - box[(i + 1) % size].x) * (box[i].x - box[(i + 1) % size].x) +
            (box[i].y - box[(i + 1) % size].y) * (box[i].y - box[(i + 1) % size].y));
    }
    area = std::fabs(area / 2.0f);
    return area * unClipRatio / dist;
}
}

cv::RotatedRect unClipBox(std::vector<cv::Point2f> box, float unClipRatio) {
    float distance = getContourArea(box, unClipRatio);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(static_cast<int>(box[0].x), static_cast<int>(box[0].y))
      << ClipperLib::IntPoint(static_cast<int>(box[1].x), static_cast<int>(box[1].y))
      << ClipperLib::IntPoint(static_cast<int>(box[2].x), static_cast<int>(box[2].y))
      << ClipperLib::IntPoint(static_cast<int>(box[3].x), static_cast<int>(box[3].y));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point2f> points;
    for (auto& path : soln) {
        for (auto& pt : path) {
            points.emplace_back(static_cast<float>(pt.X), static_cast<float>(pt.Y));
        }
    }
    if (points.empty()) {
        return cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    }
    return cv::minAreaRect(points);
}

std::vector<float> substractMeanNormalize(const cv::Mat& src, const float* meanVals, const float* normVals) {
    size_t numChannels = src.channels();
    size_t imageSize = static_cast<size_t>(src.cols) * src.rows;
    std::vector<float> out(imageSize * numChannels);
    for (size_t pid = 0; pid < imageSize; pid++) {
        for (size_t ch = 0; ch < numChannels; ++ch) {
            float data = static_cast<float>(src.data[pid * numChannels + ch]) * normVals[ch] - meanVals[ch] * normVals[ch];
            out[ch * imageSize + pid] = data;
        }
    }
    return out;
}

cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box) {
    cv::Mat image;
    src.copyTo(image);
    std::vector<cv::Point> points = box;

    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    int left = *std::min_element(collectX, collectX + 4);
    int right = *std::max_element(collectX, collectX + 4);
    int top = *std::min_element(collectY, collectY + 4);
    int bottom = *std::max_element(collectY, collectY + 4);

    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    for (auto& point : points) {
        point.x -= left;
        point.y -= top;
    }

    int imgCropWidth = static_cast<int>(std::sqrt(
        std::pow(points[0].x - points[1].x, 2) + std::pow(points[0].y - points[1].y, 2)));
    int imgCropHeight = static_cast<int>(std::sqrt(
        std::pow(points[0].x - points[3].x, 2) + std::pow(points[0].y - points[3].y, 2)));

    cv::Point2f ptsDst[4] = {
        {0.f, 0.f}, {static_cast<float>(imgCropWidth), 0.f},
        {static_cast<float>(imgCropWidth), static_cast<float>(imgCropHeight)},
        {0.f, static_cast<float>(imgCropHeight)},
    };
    cv::Point2f ptsSrc[4] = {
        {static_cast<float>(points[0].x), static_cast<float>(points[0].y)},
        {static_cast<float>(points[1].x), static_cast<float>(points[1].y)},
        {static_cast<float>(points[2].x), static_cast<float>(points[2].y)},
        {static_cast<float>(points[3].x), static_cast<float>(points[3].y)},
    };
    cv::Mat m = cv::getPerspectiveTransform(ptsSrc, ptsDst);

    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, m, cv::Size(imgCropWidth, imgCropHeight), cv::BORDER_REPLICATE);

    if (static_cast<float>(partImg.rows) >= static_cast<float>(partImg.cols) * 1.5f) {
        cv::Mat rotated(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, rotated);
        cv::flip(rotated, rotated, 0);
        return rotated;
    }
    return partImg;
}

cv::Mat matRotateClockWise180(cv::Mat src) {
    cv::flip(src, src, 0);
    cv::flip(src, src, 1);
    return src;
}

} // namespace ocr_bench
