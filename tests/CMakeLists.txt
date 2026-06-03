add_executable(hnsw_tests
    test_hnsw.cpp
)

target_link_libraries(hnsw_tests PRIVATE hnsw)
target_compile_features(hnsw_tests PRIVATE cxx_std_20)

add_test(NAME hnsw_tests COMMAND hnsw_tests)
