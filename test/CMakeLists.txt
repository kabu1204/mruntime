enable_testing()

add_executable(loader_test loader_test.cpp)
target_link_libraries(loader_test PRIVATE mruntime)
target_include_directories(loader_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME loader_test COMMAND loader_test)

add_executable(pthreadpool_raii_test pthreadpool_raii_test.cpp)
target_link_libraries(pthreadpool_raii_test PRIVATE mruntime)
target_include_directories(pthreadpool_raii_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/core
)
add_test(NAME pthreadpool_raii_test COMMAND pthreadpool_raii_test)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mruntime)
target_include_directories(tokenizer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/tokenizer
)
add_test(NAME tokenizer_test COMMAND tokenizer_test)

add_executable(qwen2_tokenizer_test qwen2_tokenizer_test.cpp)
target_link_libraries(qwen2_tokenizer_test PRIVATE mruntime)
target_include_directories(qwen2_tokenizer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/tokenizer
)
add_test(NAME qwen2_tokenizer_test COMMAND qwen2_tokenizer_test)

# Bare-metal API tests
add_executable(qwen2_ops_test qwen2_ops_test.cpp)
target_link_libraries(qwen2_ops_test PRIVATE mruntime GTest::gtest_main)
target_include_directories(qwen2_ops_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME qwen2_ops_test COMMAND qwen2_ops_test)

# Some tests assume working-directory == ${CMAKE_BINARY_DIR} (out-of-source build root).
set_tests_properties(loader_test qwen2_tokenizer_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
