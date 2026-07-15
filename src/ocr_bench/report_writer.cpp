#include "ocr_bench/report_writer.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>

namespace ocr_bench {

CategorySummary aggregateCategory(const std::string& category, const std::vector<PageMetricsInput>& pages) {
    CategorySummary summary;
    summary.category = category;
    if (pages.empty()) {
        return summary;
    }

    int tp = 0, fp = 0, fn = 0;
    std::vector<float> allCer, allWer, allConf, joined;
    int emptyCount = 0;
    int nLinesTotal = 0;
    float msTotal = 0.0f;

    for (const auto& p : pages) {
        tp += p.detection.tp;
        fp += p.detection.fp;
        fn += p.detection.fn;
        allCer.insert(allCer.end(), p.matchedCer.begin(), p.matchedCer.end());
        allWer.insert(allWer.end(), p.matchedWer.begin(), p.matchedWer.end());
        allConf.insert(allConf.end(), p.matchedConf.begin(), p.matchedConf.end());
        if (p.nGt > 0) joined.push_back(p.joinedCer);
        if (p.emptyOutput) emptyCount++;
        nLinesTotal += p.nGt;
        msTotal += p.elapsedMs;
    }

    auto mean = [](const std::vector<float>& v) -> float {
        return v.empty() ? 0.0f : std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };
    auto median = [](std::vector<float> v) -> float {
        if (v.empty()) return 0.0f;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    summary.nImages = static_cast<int>(pages.size());
    summary.nLinesTotal = nLinesTotal;
    summary.detection = DetectionStats{tp, fp, fn};
    summary.matchedCerMean = mean(allCer);
    summary.matchedCerMedian = median(allCer);
    summary.matchedWerMean = mean(allWer);
    summary.joinedCerMean = mean(joined);
    summary.meanConfidence = mean(allConf);
    summary.meanMsPerImage = pages.empty() ? 0.0f : msTotal / pages.size();
    summary.emptyOutputRate = static_cast<float>(emptyCount) / pages.size();
    return summary;
}

nlohmann::json toOverallJson(const std::vector<CategorySummary>& perCategory, const nlohmann::json& overallExtra) {
    nlohmann::json perCatJson = nlohmann::json::array();
    for (const auto& c : perCategory) {
        perCatJson.push_back({
            {"category", c.category},
            {"n_images", c.nImages},
            {"n_lines", c.nLinesTotal},
            {"detection", {{"tp", c.detection.tp}, {"fp", c.detection.fp}, {"fn", c.detection.fn}}},
            {"cer_mean", c.matchedCerMean},
            {"cer_median", c.matchedCerMedian},
            {"wer_mean", c.matchedWerMean},
            {"joined_cer_mean", c.joinedCerMean},
            {"mean_confidence", c.meanConfidence},
            {"mean_ms_per_image", c.meanMsPerImage},
            {"empty_output_rate", c.emptyOutputRate},
        });
    }
    nlohmann::json out;
    out["overall"] = overallExtra;
    out["per_category"] = perCatJson;
    return out;
}

void writeSummaryCsv(const std::string& path, const std::vector<CategorySummary>& perCategory, const nlohmann::json& overall) {
    std::ofstream f(path);
    f << "category,n_images,n_lines,precision,recall,f1,"
      << "cer_mean,cer_median,wer_mean,joined_cer_mean,"
      << "cer_corrected_mean,wer_corrected_mean,joined_cer_corrected_mean,"
      << "mean_confidence,mean_ms_per_image,empty_output_rate\n";
    for (const auto& c : perCategory) {
        f << c.category << "," << c.nImages << "," << c.nLinesTotal << ","
          << c.detection.precision() << "," << c.detection.recall() << "," << c.detection.f1() << ","
          << c.matchedCerMean << "," << c.matchedCerMedian << "," << c.matchedWerMean << "," << c.joinedCerMean << ","
          << "-,-,-,"
          << c.meanConfidence << "," << c.meanMsPerImage << "," << c.emptyOutputRate << "\n";
    }
    f << "OVERALL," << overall.value("n_images", 0) << "," << overall.value("n_lines", 0) << ","
      << overall.value("detection_precision", 0.0) << "," << overall.value("detection_recall", 0.0) << ","
      << overall.value("detection_f1", 0.0) << ","
      << overall.value("cer_mean", 0.0) << ",-," << overall.value("wer_mean", 0.0) << ",-,"
      << "-,-,-,-,-,-\n";
}

} // namespace ocr_bench
