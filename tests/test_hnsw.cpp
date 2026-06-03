#define private public
#include "hnsw/hnsw_index.hpp"
#undef private

#include "hnsw/distance.hpp"

#include <functional>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hnsw::HNSWIndex;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expectThrowsInvalidArgument(const std::function<void()>& fn, const std::string& expected_message) {
    try {
        fn();
    } catch (const std::invalid_argument& ex) {
        expect(ex.what() == expected_message, "unexpected exception message: " + std::string(ex.what()));
        return;
    }

    throw std::runtime_error("expected std::invalid_argument");
}

void testConstructorRejectsInvalidM() {
    expectThrowsInvalidArgument(
        []() { HNSWIndex index(2, hnsw::euclidean_distance, 0); },
        "M must be greater than 1.");

    expectThrowsInvalidArgument(
        []() { HNSWIndex index(2, hnsw::euclidean_distance, 1); },
        "M must be greater than 1.");
}

void testShouldStopSearchRespectsEfConstruction() {
    std::priority_queue<HNSWIndex::Candidate> candidates;
    candidates.emplace(4.0f, 4);
    candidates.emplace(2.0f, 2);

    expect(!HNSWIndex::shouldStopSearch(5.0f, candidates, 3), "search should continue until ef_construction is reached");
    expect(HNSWIndex::shouldStopSearch(5.0f, candidates, 2), "search should stop once ef_construction is full and candidate is worse");
    expect(!HNSWIndex::shouldStopSearch(1.0f, candidates, 2), "search should continue for a better candidate");
}

void testSelectClosestNeighborsKeepsNearestPoints() {
    std::priority_queue<HNSWIndex::Candidate> candidates;
    candidates.emplace(9.0f, 9);
    candidates.emplace(1.0f, 1);
    candidates.emplace(4.0f, 4);
    candidates.emplace(2.0f, 2);

    const std::vector<size_t> selected = HNSWIndex::selectClosestNeighbors(candidates, 2);
    expect(selected.size() == 2, "expected exactly 2 selected neighbors");
    expect(selected[0] == 1 && selected[1] == 2, "expected the nearest neighbors to be preserved");
}

void testAddPointSmoke() {
    HNSWIndex index(2, hnsw::euclidean_distance, 4, 3, 2);

    const std::vector<float> v0 {0.0f, 0.0f};
    const std::vector<float> v1 {1.0f, 1.0f};
    const std::vector<float> v2 {0.5f, 0.5f};

    index.addPoint(v0, 100);
    index.addPoint(v1, 101);
    index.addPoint(v2, 102);

    expect(index.nodes.size() == 3, "expected three nodes after insertions");
    expect(index.enter_node_id != HNSWIndex::INVALID_ID, "expected a valid enter point");
    expect(index.nodes[2].neighbors.size() >= 1, "expected inserted node to have at least level 0");
    expect(!index.nodes[2].neighbors[0].empty(), "expected inserted node to connect to at least one level-0 neighbor");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests {
        {"constructor rejects invalid M", testConstructorRejectsInvalidM},
        {"stop condition respects ef_construction", testShouldStopSearchRespectsEfConstruction},
        {"neighbor selection keeps nearest points", testSelectClosestNeighborsKeepsNearestPoints},
        {"addPoint smoke test", testAddPointSmoke},
    };

    for (const auto& [name, test] : tests) {
        try {
            test();
        } catch (const std::exception& ex) {
            std::cerr << "FAILED: " << name << ": " << ex.what() << '\n';
            return 1;
        }
    }

    std::cout << "All tests passed\n";
    return 0;
}
