#pragma once
#include <cmath>
#include <span>

namespace hnsw {
    float euclidean_distance(std::span<const float> a, std::span<const float> b);
    float cosine_distance(std::span<const float> a, std::span<const float> b);
    float inner_product_distance(std::span<const float> a, std::span<const float> b);
}