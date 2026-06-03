#include "hnsw/hnsw_index.hpp"
#include "hnsw/distance.hpp"
#include <queue>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <unordered_set>

namespace {
    HNSWIndex::HNSWIndex(
        size_t dimension, DistanceFunction distance_metric, size_t m, size_t ef_construction, size_t ef_search)
        :dim(dimension), distance_func(distance_metric), M(m), M0(m * 2), ef_construction(ef_construction), ef_search(ef_search),
         max_level(-1), enter_node_id(HNSWIndex::INVALID_ID), rng(42) {
        
        if (dim == 0) {
            throw std::invalid_argument("Dimension must be greater than 0.");
        }

        if (distance_func == nullptr) {
            throw std::invalid_argument("Distance metric function cannot be null.");
        }
        
        // Hệ số level_factor quy định xác suất một node được nhảy lên tầng cao hơn
        level_factor = 1.0 / std::log(static_cast<double>(M));
    }

    // Quyết định xem node mới sẽ được đặt ở tầng nào, sử dụng phân phối ngẫu nhiên với xác suất giảm dần theo tầng
    int HNSWIndex::getRandomLevel() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        if (r == 0.0) {
            r = 0.0000001; // Tránh log(0)
        }
        return static_cast<int>(-std::log(r) * level_factor);
    }

    std::span<const float> HNSWIndex::getNodeVector(size_t internal_id) const {
        size_t offset = nodes[internal_id].vector_index * dim;
        return std::span<const float>(&all_vectors_data[offset], dim);
    }

    size_t HNSWIndex::searchAtLayer(std::span<const float> query, size_t enter_node, size_t level) const {
        size_t current_node = enter_node;
        float current_dist = distance_func(query, getNodeVector(current_node));

        // flag để kiểm tra xem có node nào gần hơn được tìm thấy trong quá trình duyệt neighbors không
        bool changed = true;

        while(changed) {
            changed = false;
            for (size_t neighbor_id : nodes[current_node].neighbors[level]) {
                float dist = distance_func(query, getNodeVector(neighbor_id));
                if (dist < current_dist) {
                    current_dist = dist;
                    current_node = neighbor_id;
                    changed = true; // Nếu tìm thấy node gần hơn, tiếp tục duyệt neighbors của node mới này
                }
            }
        }


    }
}

