#pragma once
#include <vector>
#include <cstddef>

namespace hnsw {
    struct Node {
        // ID nội bộ của Node (khớp với chỉ số mảng trong HNSWIndex::nodes)
        size_t internal_id;
        // Nhãn định danh bên ngoài do người dùng quản lý
        size_t external_label;
        // Vị trí bắt đầu của vector trong all_vectors_data
        // Thực tế: Vị trí float đầu tiên = vector_index * dimension
        size_t vector_index; 

        // Dach sách các neighbors ở mỗi level, lưu internal_id của các node khác
        // neighbors[level] trả về mảng các internal_id lân cận ở tầng tương ứng
        std::vector<std::vector<size_t>> neighbors; 

        Node(size_t int_id, size_t ext_label, size_t vec_index, size_t level)
            : internal_id(int_id), external_label(ext_label), vector_index(vec_index) {
                neighbors.resize(level + 1); // Khởi tạo danh sách neighbors cho tất cả các level
            }
        
        void printDebugInfo() const;
    };
}