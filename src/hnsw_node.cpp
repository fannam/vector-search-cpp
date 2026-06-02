#include "hnsw/hnsw_node.hpp"

namespace hnsw {

    void Node::printDebugInfo() const {
        std::cout << "Node ID: " << internal_id 
                  << ", External Label: " << external_label 
                  << ", Vector Index: " << vector_index 
                  << ", Levels: " << neighbors.size() - 1 << std::endl;
        for (size_t level = 0; level < neighbors.size(); ++level) {
            std::cout << "  Level " << level << ": Neighbors = [";
            for (size_t neighbor_id : neighbors[level]) {
                std::cout << neighbor_id << " ";
            }
            std::cout << "]" << std::endl;
        }
    }
}