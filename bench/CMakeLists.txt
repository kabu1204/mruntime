add_executable(cpu_micro_bench cpu_micro_bench.cpp)
target_link_libraries(cpu_micro_bench PRIVATE mruntime)
target_include_directories(cpu_micro_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(cpu_e2e_bench cpu_e2e_bench.cpp)
target_link_libraries(cpu_e2e_bench PRIVATE mruntime)
target_include_directories(cpu_e2e_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(cpu_e2e_context_sweep_bench cpu_e2e_context_sweep_bench.cpp)
target_link_libraries(cpu_e2e_context_sweep_bench PRIVATE mruntime)
target_include_directories(cpu_e2e_context_sweep_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MRUNTIME_ENABLE_VULKAN)
    function(mruntime_add_vulkan_bench target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE mruntime Vulkan::Vulkan)
        target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src/vulkan
        )
    endfunction()

    mruntime_add_vulkan_bench(vulkan_gemm_bench vulkan_gemm_bench.cpp)
    target_include_directories(vulkan_gemm_bench PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    add_dependencies(vulkan_gemm_bench mruntime_vulkan_shaders)

    mruntime_add_vulkan_bench(vulkan_e2e_bench vulkan_e2e_bench.cpp)

    mruntime_add_vulkan_bench(vulkan_e2e_context_sweep_bench vulkan_e2e_context_sweep_bench.cpp)

    mruntime_add_vulkan_bench(vulkan_qwen2_op_bench vulkan_qwen2_op_bench.cpp)
    add_dependencies(vulkan_qwen2_op_bench mruntime_vulkan_shaders)

    mruntime_add_vulkan_bench(vulkan_prefill_attention_bench vulkan_prefill_attention_bench.cpp)
    add_dependencies(vulkan_prefill_attention_bench mruntime_vulkan_shaders)
endif()
