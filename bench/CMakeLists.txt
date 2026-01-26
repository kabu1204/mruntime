add_executable(mruntime_bench mruntime_bench.cpp)
target_link_libraries(mruntime_bench PRIVATE mruntime)
target_include_directories(mruntime_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(mruntime_e2e_bench mruntime_e2e_bench.cpp)
target_link_libraries(mruntime_e2e_bench PRIVATE mruntime)
target_include_directories(mruntime_e2e_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
