#include "hnsw/hnsw_index.hpp"
#include "hnsw/distance.hpp"
#include <queue>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <unordered_set>

namespace {
    HNSWIndex::HNSWIndex(size_t dimension, size_t m, size_t ef_construction, size_t ef_search)
        :dim(dimension), M(m), M0(m * 2), ef_construction(ef_construction), ef_search(ef_search),
         max_level(-1), enter_node_id(HNSWIndex::INVALID_ID), rng(42) {
        
        if (dim == 0) {
            throw std::invalid_argument("Dimension must be greater than 0.");
        }
        
        // Hệ số level_factor quy định xác suất một node được nhảy lên tầng cao hơn
        level_factor = 1.0 / std::log(static_cast<double>(M));
    }

    int HNSWIndex::getRandomLevel() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
    }
}

