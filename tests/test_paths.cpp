#include <doctest/doctest.h>
#include "ocr_bench/paths.hpp"

using namespace ocr_bench;

TEST_CASE("datasetRegistry has exactly the ind_cn and new keys") {
    auto& reg = datasetRegistry();
    REQUIRE(reg.count("ind_cn") == 1);
    REQUIRE(reg.count("new") == 1);
    CHECK(reg.size() == 2);
}

TEST_CASE("resolveDatasetRoot with explicit known key returns that key's root") {
    auto resolved = resolveDatasetRoot("ind_cn", "");
    CHECK(resolved.key == "ind_cn");
    CHECK(resolved.root == datasetRegistry().at("ind_cn"));
}

TEST_CASE("resolveDatasetRoot with env override (known key) wins over default") {
    auto resolved = resolveDatasetRoot("", "new");
    CHECK(resolved.key == "new");
    CHECK(resolved.root == datasetRegistry().at("new"));
}

TEST_CASE("resolveDatasetRoot with unknown non-empty value treats it as a literal path") {
    auto resolved = resolveDatasetRoot("/some/custom/path", "");
    CHECK(resolved.key == "/some/custom/path");
    CHECK(resolved.root == "/some/custom/path");
}

TEST_CASE("resolveDatasetRoot with nothing provided falls back to default key ind_cn") {
    auto resolved = resolveDatasetRoot("", "");
    CHECK(resolved.key == "ind_cn");
    CHECK(resolved.root == datasetRegistry().at("ind_cn"));
}

TEST_CASE("argument wins over env override") {
    auto resolved = resolveDatasetRoot("ind_cn", "new");
    CHECK(resolved.key == "ind_cn");
}
