add_executable(mruntime_bench bench.cpp)
target_link_libraries(mruntime_bench PRIVATE mruntime)
target_include_directories(mruntime_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
