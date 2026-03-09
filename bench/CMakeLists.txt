add_executable(cpu_micro_bench cpu_micro_bench.cpp)
target_link_libraries(cpu_micro_bench PRIVATE mruntime)
target_include_directories(cpu_micro_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(cpu_e2e_bench cpu_e2e_bench.cpp)
target_link_libraries(cpu_e2e_bench PRIVATE mruntime)
target_include_directories(cpu_e2e_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MRUNTIME_ENABLE_VULKAN)
    add_executable(vulkan_gemm_bench vulkan_gemm_bench.cpp)
    target_link_libraries(vulkan_gemm_bench PRIVATE mruntime Vulkan::Vulkan)
    target_include_directories(vulkan_gemm_bench PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    add_dependencies(vulkan_gemm_bench mruntime_vulkan_shaders)

    add_executable(vulkan_e2e_bench vulkan_e2e_bench.cpp)
    target_link_libraries(vulkan_e2e_bench PRIVATE mruntime Vulkan::Vulkan)
    target_include_directories(vulkan_e2e_bench PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
endif()
