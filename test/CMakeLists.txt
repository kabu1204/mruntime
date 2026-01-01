enable_testing()

set(MLAS_LIB_DIR "${CMAKE_SOURCE_DIR}/MLAS/install/lib")

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE mruntime)
target_include_directories(tensor_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(tensor_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(tensor_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(backend_test backend_test.cpp)
target_link_libraries(backend_test PRIVATE mruntime)
target_include_directories(backend_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(backend_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(backend_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME backend_test COMMAND backend_test)

add_executable(loader_test loader_test.cpp)
target_link_libraries(loader_test PRIVATE mruntime)
target_include_directories(loader_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(loader_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(loader_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME loader_test COMMAND loader_test)

add_executable(model_test model_test.cpp)
target_link_libraries(model_test PRIVATE mruntime)
target_include_directories(model_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(model_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(model_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME model_test COMMAND model_test)

add_executable(engine_test engine_test.cpp)
target_link_libraries(engine_test PRIVATE mruntime)
target_include_directories(engine_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(engine_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(engine_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME engine_test COMMAND engine_test)

add_executable(thread_pool_test thread_pool_test.cpp)
target_link_libraries(thread_pool_test PRIVATE mruntime)
target_include_directories(thread_pool_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/core
)
target_link_directories(thread_pool_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(thread_pool_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME thread_pool_test COMMAND thread_pool_test)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE mruntime)
target_include_directories(tokenizer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/tokenizer
)
target_link_directories(tokenizer_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(tokenizer_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME tokenizer_test COMMAND tokenizer_test)

add_executable(qwen2_tokenizer_test qwen2_tokenizer_test.cpp)
target_link_libraries(qwen2_tokenizer_test PRIVATE mruntime)
target_include_directories(qwen2_tokenizer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/tokenizer
)
target_link_directories(qwen2_tokenizer_test PRIVATE ${MLAS_LIB_DIR})
target_link_libraries(qwen2_tokenizer_test PRIVATE onnxruntime_mlas onnxruntime_common)
add_test(NAME qwen2_tokenizer_test COMMAND qwen2_tokenizer_test)

# Some tests assume working-directory == ${CMAKE_BINARY_DIR} (out-of-source build root).
set_tests_properties(loader_test model_test engine_test qwen2_tokenizer_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
