#include "hnsw/hnsw_node.hpp"
#include "hnsw/distance.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace hnsw {
    float euclidean_distance(std::span<const float> a, std::span<const float> b) {
        if(a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension for Euclidean distance.");
        }
        float sum = 0.0f;
        size_t size = a.size();

        for(size_t i = 0; i < size; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        // Trả về bình phương khoảng cách để tránh tính toán căn bậc hai không cần thiết trong nhiều trường hợp (ví dụ: khi so sánh khoảng cách)
        return sum;
    }

    float cosine_distance(std::span<const float> a, std::span<const float> b) {
        if(a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension for Cosine distance.");
        }
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        size_t size = a.size();

        for(size_t i = 0; i < size; ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        // Kiểm tra vector zero
        if(norm_a == 0.0f || norm_b == 0.0f) {
            throw std::invalid_argument("Cannot compute cosine distance for zero-vector.");
        }
        // Cosine similarity = dot_product / (norm_a * norm_b)
        // Cosine distance = 1 - cosine similarity
        float similarity = dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
        return 1.0f - similarity;
    }

    float inner_product_distance(std::span<const float> a, std::span<const float> b) {
        if(a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension for Inner Product distance.");
        }
        float dot_product = 0.0f;
        size_t size = a.size();

        for(size_t i = 0; i < size; ++i) {
            dot_product += a[i] * b[i];
        }
        // Trả về âm của tích vô hướng để biến nó thành khoảng cách (khoảng cách nhỏ hơn có nghĩa là gần hơn)
        return -dot_product;
    }
}

