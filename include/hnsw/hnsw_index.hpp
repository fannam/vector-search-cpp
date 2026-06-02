#pragma once
#include "hnsw_node.hpp"
#include "distance.hpp"
#include <vector>
#include <random>
#include <span>
#include <cstddef>

namespace hnsw {
    class HNSWIndex {
        private:
            // 1. Quản lý bộ nhớ phẳng để cache line hiệu quả
            size_t dim; // Số chiều của vector (cố định)
            std::vector<float> all_vectors_data; // Lưu trừ liên tục toàn bộ các giá trị float của tất cả vector (vector_index * dim là vị trí bắt đầu của vector đó)
            std::vector<Node> nodes; // Danh sách các node, mỗi node chứa thôngtin về vector và neighbors
            
            // 2. Tham số của HNSW
            size_t M; // Số lượng liên kết tối đa của một node ở các tầng trên
            size_t M0; // Số lượng liên kết tối đa của một node ở tầng 0
            size_t ef_construction; // Kích thước danh sách động khi build index
            size_t ef_search; // Kích thước danh sách động khi tìm kiếm

            // 3. Trạng thái đồ thị (sử dụng sentinel value để đánh dấu node chưa được khởi tạo)
            static constexpr size_t INVALID_ID = static_cast<size_t>(-1);
            int max_level; // Tầng cao nhất hiện tại (bắt đầu từ -1)
            size_t enter_node_id; // internal_id của node dùng làm entry point

            // 4. Công cụ phân tầng ngẫu nhiên
            std::mt19937 rng;
            double level_factor; // Xác suất giảm dần theo tầng (thường là 1 / M)

            // 5. Hàm hỗ trợ nội bộ
            // Sinh tầng ngẫu nhiên cho node mới, sử dụng phân phối geometric với xác suất giảm dần theo tầng
            int getRandomLevel();
            // Trích xuất vector của node dựa trên internal_id, trả về std::span để truy cập hiệu quả mà không cần copy
            std::span<const float> getNodeVector(size_t internal_id) const; 

            size_t searchAtLayer(std::span<const float> query, size_t enter_node, size_t level) const;

        
        public:
            HNSWIndex(size_t dimension, size_t m = 16, size_t ef_construction = 200, size_t ef_search = 50);

            void addPoint(std::span<const float> vector_view, size_t external_label);

            std::vector<size_t> searchKNN(std::span<const float> query_vector, size_t k) const;

            std::vector<size_t> searchFlat(std::span<const float> query_vector, size_t k) const;
    };
}