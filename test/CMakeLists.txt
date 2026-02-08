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

if(MRUNTIME_ENABLE_VULKAN)
    add_executable(vulkan_smoke_test vulkan_smoke_test.cpp)
    target_link_libraries(vulkan_smoke_test PRIVATE mruntime Vulkan::Vulkan)
    target_include_directories(vulkan_smoke_test PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    add_dependencies(vulkan_smoke_test mruntime_vulkan_shaders)
    add_test(NAME vulkan_smoke_test COMMAND vulkan_smoke_test)
    set_tests_properties(vulkan_smoke_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(vulkan_fp16_ops_test vulkan_fp16_ops_test.cpp)
    target_link_libraries(vulkan_fp16_ops_test PRIVATE mruntime Vulkan::Vulkan)
    target_include_directories(vulkan_fp16_ops_test PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    add_dependencies(vulkan_fp16_ops_test mruntime_vulkan_shaders)
    add_test(NAME vulkan_fp16_ops_test COMMAND vulkan_fp16_ops_test)
    set_tests_properties(vulkan_fp16_ops_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(vulkan_fp16_kernels_test vulkan_fp16_kernels_test.cpp)
    target_link_libraries(vulkan_fp16_kernels_test PRIVATE mruntime Vulkan::Vulkan)
    target_include_directories(vulkan_fp16_kernels_test PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    add_dependencies(vulkan_fp16_kernels_test mruntime_vulkan_shaders)
    add_test(NAME vulkan_fp16_kernels_test COMMAND vulkan_fp16_kernels_test)
    set_tests_properties(vulkan_fp16_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(vulkan_gemm_test vulkan_gemm_test.cpp)
    target_link_libraries(vulkan_gemm_test PRIVATE mruntime Vulkan::Vulkan)
    target_include_directories(vulkan_gemm_test PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    add_dependencies(vulkan_gemm_test mruntime_vulkan_shaders)
    add_test(NAME vulkan_gemm_test COMMAND vulkan_gemm_test)
    set_tests_properties(vulkan_gemm_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Some tests assume working-directory == ${CMAKE_BINARY_DIR} (out-of-source build root).
set_tests_properties(loader_test qwen2_tokenizer_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
