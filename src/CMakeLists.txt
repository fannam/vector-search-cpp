add_library(hnsw
    distance.cpp
    hnsw_index.cpp
    hnsw_node.cpp
)

target_include_directories(hnsw
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(hnsw PUBLIC cxx_std_20)
