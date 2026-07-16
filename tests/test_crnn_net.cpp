// tests/test_crnn_net.cpp
#include <doctest/doctest.h>
#include "ocr_bench/crnn_net.hpp"

using namespace ocr_bench;

TEST_CASE("loadKeysFromFile loads keys and prepends blank + appends space") {
    CrnnNet net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    // Internal keys layout mirrors RapidOcrOnnx: {"#", "a", "b", "c", "d", "e", " "}
    // Exposed indirectly via decode behavior in the next test, but we can also
    // check the count via a public accessor added for testability.
    CHECK(net.keyCount() == 7); // 5 chars + blank prefix + space suffix
}

TEST_CASE("loadKeysFromFile on a missing file leaves keys empty, does not throw") {
    CrnnNet net;
    net.loadKeysFromFile("tests/fixtures/does_not_exist.txt");
    CHECK(net.keyCount() == 0);
}

TEST_CASE("loadKeysFromModelMetadata without a loaded model returns false") {
    CrnnNet net; // never loadModel()
    CHECK(net.loadKeysFromModelMetadata() == false);
    CHECK(net.keyCount() == 0);
}

TEST_CASE("CTC greedy decode collapses repeats and skips blank index 0") {
    CrnnNet net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    // keys = ["#", "a", "b", "c", "d", "e", " "]  (indices 0..6)
    // Simulate a CRNN output over h=6 timesteps, w=7 classes (one-hot-ish):
    // timestep sequence of argmax indices: [1, 1, 2, 0, 3, 3] -> "a" (collapsed) + "b" + (blank skipped) + "d" (collapsed)
    // Expected decoded text: "abd"
    std::vector<float> output(6 * 7, 0.0f);
    auto setArgmax = [&](int t, int idx) { output[t * 7 + idx] = 1.0f; };
    setArgmax(0, 1); // a
    setArgmax(1, 1); // a (repeat, collapsed)
    setArgmax(2, 2); // b
    setArgmax(3, 0); // blank
    setArgmax(4, 3); // c... wait index 3 is 'c' per keys layout above (0=#,1=a,2=b,3=c,4=d,5=e,6=space)
    setArgmax(5, 3); // c (repeat, collapsed)
    auto line = net.decodeForTest(output, 6, 7);
    CHECK(line.text == "abc");
}
