#include "hnsw/hnsw_index.hpp"
#include "hnsw/distance.hpp"
#include <queue>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace hnsw {
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

        if (M <= 1) {
            throw std::invalid_argument("M must be greater than 1.");
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

        return current_node; // Trả về node gần nhất tìm được ở tầng này
    }

    bool HNSWIndex::shouldStopSearch(
        float candidate_distance,
        const std::priority_queue<Candidate>& candidates,
        size_t ef) {
        return candidates.size() >= ef && candidate_distance > candidates.top().first;
    }

    std::vector<size_t> HNSWIndex::selectClosestNeighbors(
        std::priority_queue<Candidate> candidates,
        size_t limit) {
        while (candidates.size() > limit) {
            candidates.pop();
        }

        std::vector<size_t> selected_neighbors;
        selected_neighbors.reserve(candidates.size());

        while (!candidates.empty()) {
            selected_neighbors.push_back(candidates.top().second);
            candidates.pop();
        }

        std::reverse(selected_neighbors.begin(), selected_neighbors.end());
        return selected_neighbors;
    }

    void HNSWIndex::addPoint(std::span<const float> vector_view, size_t external_label) {

        // 1. Kiểm tra đầu vào và khởi tạo Node
        if (vector_view.size() != dim) {
            throw std::invalid_argument("Input vector dimension does not match index dimension.");
        }

        size_t new_node_id = nodes.size();
        size_t new_vector_index = new_node_id; // Mỗi node mới sẽ có một vector_index tương ứng (bắt đầu từ 0)

        all_vectors_data.insert(all_vectors_data.end(), vector_view.begin(), vector_view.end());

        int insert_level = getRandomLevel();

        if(new_node_id ==0) {
            nodes.emplace_back(new_node_id, external_label, new_vector_index, insert_level);
            enter_node_id = new_node_id;
            max_level = insert_level;
            return;
        }

        nodes.emplace_back(new_node_id, external_label, new_vector_index, insert_level);

        // 2. Fast Routing
        // Nếu đồ thị đã có dữ liệu, cần tìm 1 điểm đáp xuống hợp lý tại tầng insert_level 
        // Tìm đến node gần với vector mới nhất ở các tầng trên
        size_t current_objective = enter_node_id;
        for (int level = max_level; level > insert_level; --level) {
            current_objective = searchAtLayer(vector_view, current_objective, level);
        }

        // 3. Kết nối node mới vào đồ thị
        for (int level = std::min(insert_level, max_level); level >= 0; --level) {
            // Tìm các neighbors tốt nhất ở tầng này để kết nối
            // 3.1. Thay vì tìm 1 điểm gần nhất như searchAtLayer, cần tìm một tập candidates gồm ef_construction node gần nhất
            std::unordered_set<size_t> visited; // Để tránh duyệt lại nodes đã thăm

            // Min heap để lưu các ứng viên chuẩn bị đi thám hiểm hàng xóm
            std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::greater<std::pair<float, size_t>>> W; 

            // Max heap để lưu các ứng viên tốt nhất đã tìm được
            std::priority_queue<std::pair<float, size_t>> candidates;

            // Tính khoảng cách từ vector_view đến current_objective và thêm current_objective vào W, visited và candidates
            float dist_to_objective = distance_func(vector_view, getNodeVector(current_objective));
            W.emplace(dist_to_objective, current_objective);
            visited.insert(current_objective);
            candidates.emplace(dist_to_objective, current_objective);

            while (!W.empty()) {
                // Lấy ứng đầu tiên từ W
                auto [dist, candidate_id] = W.top();
                W.pop();

                // Nếu khoảng cách của candidate này lớn hơn khoảng cách của ứng viên tệ nhất trong candidates, dừng duyệt
                if (shouldStopSearch(dist, candidates, ef_construction)) {
                    break;
                }

                // Duyệt qua neighbors của candidate này ở tầng level
                for (size_t neighbor_id : nodes[candidate_id].neighbors[level]) {
                    if (visited.count(neighbor_id) > 0) {
                        continue; // Bỏ qua nếu đã thăm neighbor này    
                    }
                    // Nếu neighbor chưa được thăm, bỏ nó vào visited và tính khoảng cách từ vector_view đến neighbor này
                    visited.insert(neighbor_id);
                    float dist_to_neighbor = distance_func(vector_view, getNodeVector(neighbor_id));

                    // Nếu dist_to_neighbor nhỏ hơn khoảng cách của ứng viên tệ nhất trong candidates hoặc kích thước candidates chưa đạt ef_construction, thêm neighbor này vào W và candidates
                    if (candidates.size() < ef_construction || dist_to_neighbor < candidates.top().first) {
                        W.emplace(dist_to_neighbor, neighbor_id);
                        candidates.emplace(dist_to_neighbor, neighbor_id);

                        // Nếu candidates vượt quá kích thước ef_construction, loại bỏ ứng viên tệ nhất
                        if (candidates.size() > ef_construction) {
                            candidates.pop();
                        }
                    }
                }
            }

            // 3.2. Lọc ra M hàng xóm tốt nhất
            // Sau khi vòng lặp while dừng thì max-heap candidates chứa những điểm tốt nhất

            // Xác định số lượng liên kết tối đa cho tầng này (M0 cho tầng 0, M cho các tầng trên)
            size_t current_M = (level == 0) ? M0 : M;
            std::vector<size_t> selected_neighbors = selectClosestNeighbors(std::move(candidates), current_M);

            // 3.3. Thiết lập liên kết 2 chiều và Heuristic Shrink
            // Khi một node mới được chèn vào một tầng, hành động kết nối có thể làm số lượng neighbors của một node nào đó vượt quá M hoặc M0
            // Đồ thị bắt buộc phải thực hiện heuristic shrink để giữ số lượng cạnh kết nối với mỗi node nằm trong tầm kiểm soát

            for (size_t neighbor_id : selected_neighbors) {
                // Kết nối chiều đi: đẩy neighbor_id vào danh sách nodes[new_node_id].neighbors[level] 
                nodes[new_node_id].neighbors[level].push_back(neighbor_id);

                // Kết nối chiều về: đẩy new_node_id vào danh sách nodes[neighbor_id].neighbors[level]
                nodes[neighbor_id].neighbors[level].push_back(new_node_id);

                // Kiểm tra nếu neighbors của neighbor_id vượt quá giới hạn current_M, nếu có thì thực hiện heuristic shrink
                if (nodes[neighbor_id].neighbors[level].size() > current_M) {
                    // Tính lại khoảng cách từ neighbor_id đến tất cả hàng xóm hiện tại của nó ở tầng này
                    // Sắp xếp lại danh sách theo khoảng cách tăng dần
                    // Xóa sạch danh sách cũ và nạp đúng current_M neighbors gần nhất vào danh sách neighbors của neighbor_id ở tầng này

                    std::vector<std::pair<float, size_t>> neighbor_distances;
                    for (size_t n_id : nodes[neighbor_id].neighbors[level]) {
                        float dist = distance_func(getNodeVector(neighbor_id), getNodeVector(n_id));
                        neighbor_distances.emplace_back(dist, n_id);
                    }
                    std::sort(neighbor_distances.begin(), neighbor_distances.end());
                    nodes[neighbor_id].neighbors[level].clear();
                    for (size_t i = 0; i < current_M && i < neighbor_distances.size(); ++i) {
                        nodes[neighbor_id].neighbors[level].push_back(neighbor_distances[i].second);
                    }
                }
            }
            // Điểm xuất phát cho tầng dưới chính là phần tử gần nhất vừa tìm được ở tầng này
            if (!selected_neighbors.empty()) {
                current_objective = selected_neighbors.front();
            }
        }
        // Cập nhật Enter Point nếu node mới có tầng cao hơn max_level hiện tại
        if (insert_level > max_level) {
            enter_node_id = new_node_id;
            max_level = insert_level;
        }

    }

}
