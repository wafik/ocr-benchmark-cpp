#include "ocr_bench/metrics.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace ocr_bench {

std::string normalize(const std::string& text) {
    if (text.empty()) {
        return "";
    }
    // Lowercase (ASCII-fold; NFKC full-width normalization is not needed
    // for this project's Latin-script dataset — see Global Constraints).
    std::string lowered;
    lowered.reserve(text.size());
    for (unsigned char c : text) {
        lowered.push_back(static_cast<char>(std::tolower(c)));
    }
    // Collapse all whitespace runs to single spaces, then strip.
    std::istringstream iss(lowered);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += ' ';
        result += tokens[i];
    }
    return result;
}

namespace {

// Generic Levenshtein edit distance over a sequence of tokens (chars or words).
template <typename T>
int levenshtein(const std::vector<T>& a, const std::vector<T>& b) {
    const size_t n = a.size();
    const size_t m = b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (size_t i = 0; i <= n; ++i) dp[i][0] = static_cast<int>(i);
    for (size_t j = 0; j <= m; ++j) dp[0][j] = static_cast<int>(j);
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }
    return dp[n][m];
}

std::vector<std::string> splitWords(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

} // namespace

float cer(const std::string& ref, const std::string& hyp) {
    if (ref.empty()) {
        return hyp.empty() ? 0.0f : 1.0f;
    }
    std::string r = normalize(ref);
    std::string h = normalize(hyp);
    if (r.empty()) {
        return h.empty() ? 0.0f : 1.0f;
    }
    std::vector<char> rChars(r.begin(), r.end());
    std::vector<char> hChars(h.begin(), h.end());
    int dist = levenshtein(rChars, hChars);
    return static_cast<float>(dist) / static_cast<float>(rChars.size());
}

float wer(const std::string& ref, const std::string& hyp) {
    if (ref.empty()) {
        return hyp.empty() ? 0.0f : 1.0f;
    }
    std::string r = normalize(ref);
    std::string h = normalize(hyp);
    if (r.empty()) {
        return h.empty() ? 0.0f : 1.0f;
    }
    auto rWords = splitWords(r);
    auto hWords = splitWords(h);
    int dist = levenshtein(rWords, hWords);
    return static_cast<float>(dist) / static_cast<float>(rWords.size());
}

} // namespace ocr_bench
