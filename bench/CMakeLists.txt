set(MLAS_LIB_DIR "${CMAKE_SOURCE_DIR}/MLAS/install/lib")

add_executable(mruntime_bench bench.cpp)
target_link_libraries(mruntime_bench PRIVATE mruntime)
target_include_directories(mruntime_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(mruntime_bench PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(mruntime_bench PRIVATE onnxruntime_mlas onnxruntime_common)
