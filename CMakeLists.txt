cmake_minimum_required(VERSION 3.18)
project(mruntime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(MRUNTIME_BUILD_TESTS "Build unit tests" ON)
option(MRUNTIME_BUILD_BENCH "Build benchmarks" ON)
option(MRUNTIME_BUILD_CLI "Build CLI tools" ON)
option(MRUNTIME_ENABLE_VULKAN "Enable Vulkan backend (experimental)" OFF)
option(MRUNTIME_ENABLE_VULKAN_SLANG "Use Slang instead of GLSL for Vulkan shaders (experimental)" OFF)

include(FetchContent)

# Dependencies
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240116.2
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    re2
    GIT_REPOSITORY https://github.com/google/re2.git
    GIT_TAG 2024-07-02
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    pthreadpool
    URL https://github.com/google/pthreadpool/archive/0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e.zip
    URL_HASH SHA256=f602ab141bdc5d5872a79d6551e9063b5bfa7ad6ad60cceaa641de5c45c86d70
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_Declare(
    kleidiai
    GIT_REPOSITORY https://github.com/ARM-software/kleidiai.git
    GIT_TAG v1.20.0
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "Disable Abseil tests")
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Let Abseil match consumer C++ standard")
set(RE2_BUILD_TESTING OFF CACHE BOOL "Disable RE2 tests")
set(RE2_BUILD_BENCHMARK OFF CACHE BOOL "Disable RE2 benchmark binary")
set(JSON_BuildTests OFF CACHE BOOL "Disable nlohmann/json tests")
set(JSON_Install OFF CACHE BOOL "Disable nlohmann/json install rules")
set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "Disable pthreadpool tests")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "Disable pthreadpool benchmarks")
set(PTHREADPOOL_BUILD_SHARED_LIBRARY OFF CACHE BOOL "Build pthreadpool as a shared library")
set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "Disable KleidiAI tests")
set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "Disable KleidiAI benchmark tool")
set(KLEIDIAI_ENABLE_CLANG_TIDY OFF CACHE BOOL "Disable KleidiAI clang-tidy checks")
set(BUILD_GMOCK OFF CACHE BOOL "Disable gMock")
set(INSTALL_GTEST OFF CACHE BOOL "Disable googletest install rules")
set(_MRUNTIME_PREV_SKIP_INSTALL_RULES "${CMAKE_SKIP_INSTALL_RULES}")
set(CMAKE_SKIP_INSTALL_RULES TRUE)
FetchContent_MakeAvailable(spdlog absl re2 nlohmann_json pthreadpool kleidiai)
set(CMAKE_SKIP_INSTALL_RULES "${_MRUNTIME_PREV_SKIP_INSTALL_RULES}")

# Vulkan (optional)
set(MRUNTIME_VULKAN_GENERATED_DIR "")
set(MRUNTIME_VULKAN_SHADER_HEADERS "")
if(MRUNTIME_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)

    set(MRUNTIME_VULKAN_SLANG_MODULE_DIR "${CMAKE_SOURCE_DIR}/src/vulkan/slang")
    set(MRUNTIME_VULKAN_SLANG_MODULE_SOURCES
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/pointwise_ops.slang"
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/register_tile.slang"
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/tile_views.slang"
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/layout_traits.slang"
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/storage_traits.slang"
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/compute_arch.slang"
        "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}/scope_primitives.slang"
    )

    if(MRUNTIME_ENABLE_VULKAN_SLANG)
        # Used to compile Slang compute shaders into SPIR-V during the build.
        # Vulkan SDK may provide this tool.
        if(DEFINED MRUNTIME_SLANGC_EXECUTABLE AND MRUNTIME_SLANGC_EXECUTABLE STREQUAL "")
            unset(MRUNTIME_SLANGC_EXECUTABLE CACHE)
        endif()

        find_program(
            MRUNTIME_SLANGC_EXECUTABLE
            NAMES slangc
            DOC "Path to slangc"
        )
        if(NOT MRUNTIME_SLANGC_EXECUTABLE OR NOT EXISTS "${MRUNTIME_SLANGC_EXECUTABLE}")
            message(
                FATAL_ERROR
                "slangc not found. Install Vulkan SDK or set MRUNTIME_SLANGC_EXECUTABLE."
            )
        endif()
    else()
        # Used to compile GLSL compute shaders into SPIR-V during the build.
        # Vulkan SDK provides this tool.
        find_program(MRUNTIME_GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator)
        if(NOT MRUNTIME_GLSLANG_VALIDATOR_EXECUTABLE)
            message(
                FATAL_ERROR
                "glslangValidator not found. Install Vulkan SDK or set MRUNTIME_GLSLANG_VALIDATOR_EXECUTABLE."
            )
        endif()
    endif()

    set(MRUNTIME_VULKAN_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/vulkan")
    file(MAKE_DIRECTORY "${MRUNTIME_VULKAN_GENERATED_DIR}")

    function(mruntime_add_vulkan_glsl_shader shader_stem shader_source var_name)
        set(shader_spv "${MRUNTIME_VULKAN_GENERATED_DIR}/${shader_stem}.spv")
        set(shader_hdr "${MRUNTIME_VULKAN_GENERATED_DIR}/${shader_stem}_spv.h")

        add_custom_command(
            OUTPUT "${shader_spv}"
            COMMAND "${MRUNTIME_GLSLANG_VALIDATOR_EXECUTABLE}"
                --target-env vulkan1.3
                -V "${shader_source}"
                -o "${shader_spv}"
            DEPENDS "${shader_source}"
            COMMENT "Compiling GLSL -> SPIR-V: ${shader_stem}.comp"
            VERBATIM
        )

        add_custom_command(
            OUTPUT "${shader_hdr}"
            COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${shader_spv}"
                -DOUTPUT="${shader_hdr}"
                -DVAR_NAME="${var_name}"
                -DNAMESPACE="mruntime::vulkan::shaders"
                -P "${CMAKE_SOURCE_DIR}/cmake/spirv_to_header.cmake"
            DEPENDS
                "${shader_spv}"
                "${CMAKE_SOURCE_DIR}/cmake/spirv_to_header.cmake"
            COMMENT "Embedding SPIR-V into header: ${shader_stem}_spv.h"
            VERBATIM
        )

        set(MRUNTIME_VULKAN_SHADER_HEADERS
            ${MRUNTIME_VULKAN_SHADER_HEADERS}
            "${shader_hdr}"
            PARENT_SCOPE
        )
    endfunction()

    function(mruntime_add_vulkan_slang_shader shader_stem shader_source var_name)
        set(shader_spv "${MRUNTIME_VULKAN_GENERATED_DIR}/${shader_stem}.spv")
        set(shader_hdr "${MRUNTIME_VULKAN_GENERATED_DIR}/${shader_stem}_spv.h")
        set(shader_dep "${MRUNTIME_VULKAN_GENERATED_DIR}/${shader_stem}.spv.d")

        add_custom_command(
            OUTPUT "${shader_spv}"
            COMMAND "${MRUNTIME_SLANGC_EXECUTABLE}"
                -target spirv
                -profile glsl_460+spirv_1_6+spvGroupNonUniformBallot+spvGroupNonUniformArithmetic
                -entry main
                -stage compute
                -fvk-use-scalar-layout
                -I "${MRUNTIME_VULKAN_SLANG_MODULE_DIR}"
                -depfile "${shader_dep}"
                "${shader_source}"
                -o "${shader_spv}"
            DEPENDS
                "${shader_source}"
                ${MRUNTIME_VULKAN_SLANG_MODULE_SOURCES}
            DEPFILE "${shader_dep}"
            COMMENT "Compiling Slang -> SPIR-V: ${shader_stem}.slang"
            VERBATIM
        )

        add_custom_command(
            OUTPUT "${shader_hdr}"
            COMMAND "${CMAKE_COMMAND}"
                -DINPUT="${shader_spv}"
                -DOUTPUT="${shader_hdr}"
                -DVAR_NAME="${var_name}"
                -DNAMESPACE="mruntime::vulkan::shaders"
                -P "${CMAKE_SOURCE_DIR}/cmake/spirv_to_header.cmake"
            DEPENDS
                "${shader_spv}"
                "${CMAKE_SOURCE_DIR}/cmake/spirv_to_header.cmake"
            COMMENT "Embedding SPIR-V into header: ${shader_stem}_spv.h"
            VERBATIM
        )

        set(MRUNTIME_VULKAN_SHADER_HEADERS
            ${MRUNTIME_VULKAN_SHADER_HEADERS}
            "${shader_hdr}"
            PARENT_SCOPE
        )
    endfunction()

    macro(mruntime_add_vulkan_shader_variant shader_stem glsl_source slang_source var_name)
        if(MRUNTIME_ENABLE_VULKAN_SLANG)
            mruntime_add_vulkan_slang_shader(${shader_stem} "${slang_source}" ${var_name})
        else()
            mruntime_add_vulkan_glsl_shader(${shader_stem} "${glsl_source}" ${var_name})
        endif()
    endmacro()

    mruntime_add_vulkan_shader_variant(
        vector_add
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/vector_add.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/vector_add.slang"
        kVectorAddSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_add
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_add.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_add.slang"
        kFp16AddSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_mul
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_mul.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_mul.slang"
        kFp16MulSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_silu_mul_interleaved
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_silu_mul_interleaved.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_silu_mul_interleaved.slang"
        kFp16SiluMulInterleavedSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_rmsnorm
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_rmsnorm.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_rmsnorm.slang"
        kFp16RmsnormSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_qkv_bias_split
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_qkv_bias_split.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_qkv_bias_split.slang"
        kFp16QkvBiasSplitSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_rope
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_rope.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_rope.slang"
        kFp16RopeSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_transpose
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_transpose.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_transpose.slang"
        kFp16TransposeSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_kv_cache_copy
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_kv_cache_copy.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_kv_cache_copy.slang"
        kFp16KvCacheCopySpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_attention_decode_gqa
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_attention_decode_gqa.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_attention_decode_gqa.slang"
        kFp16AttentionDecodeGqaSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_attention_prefill_gqa
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_attention_prefill_gqa.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_attention_prefill_gqa.slang"
        kFp16AttentionPrefillGqaSpv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_gemv_rows4_vec4
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_gemv_rows4_vec4.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_gemv_rows4_vec4.slang"
        kFp16GemvRows4Vec4Spv
    )
    mruntime_add_vulkan_shader_variant(
        fp16_gemm
        "${CMAKE_SOURCE_DIR}/src/vulkan/glsl/fp16_gemm.comp"
        "${CMAKE_SOURCE_DIR}/src/vulkan/slang/fp16_gemm.slang"
        kFp16GemmSpv
    )

    add_custom_target(mruntime_vulkan_shaders DEPENDS "${MRUNTIME_VULKAN_SHADER_HEADERS}")
endif()

# Core library sources (bare-metal API)
set(MRUNTIME_CORE_SOURCES
    src/core/kai_gemm.cpp
    src/core/trace.cpp
    src/qwen_config.cpp
    src/qwen2_ops.cpp
    src/qwen2_weights.cpp
    src/qwen2_forward.cpp
    src/qwen2_generate.cpp
)

if(MRUNTIME_ENABLE_VULKAN)
    list(APPEND MRUNTIME_CORE_SOURCES
        src/qwen2_forward_vk.cpp
        src/qwen2_generate_vk.cpp
        src/vulkan/vk_buffer_arena.cpp
        src/vulkan/vk_command.cpp
        src/vulkan/vk_compute_pipeline.cpp
        src/vulkan/vk_fp16_ops.cpp
        src/vulkan/vk_kernel_runtime.cpp
        src/vulkan/vk_context.cpp
    )
endif()

set(MRUNTIME_LOADER_SOURCES
    src/loader/safetensors.cpp
)

set(MRUNTIME_TOKENIZER_SOURCES
    src/tokenizer/bpe.cpp
    src/tokenizer/qwen2_tokenizer.cpp
)

# Main library target
add_library(mruntime STATIC
    ${MRUNTIME_CORE_SOURCES}
    ${MRUNTIME_LOADER_SOURCES}
    ${MRUNTIME_TOKENIZER_SOURCES}
)

target_include_directories(mruntime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(mruntime PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/core
)

target_link_libraries(mruntime
    PUBLIC
        re2::re2
        spdlog::spdlog_header_only
        pthreadpool
        kleidiai
    PRIVATE
        nlohmann_json::nlohmann_json
)

if(MRUNTIME_ENABLE_VULKAN)
    target_link_libraries(mruntime PRIVATE Vulkan::Vulkan)
    target_include_directories(mruntime PRIVATE
        ${MRUNTIME_VULKAN_GENERATED_DIR}
        ${CMAKE_SOURCE_DIR}/src/vulkan
    )
    target_compile_definitions(mruntime PUBLIC MRUNTIME_ENABLE_VULKAN=1)
    add_dependencies(mruntime mruntime_vulkan_shaders)
endif()

if((CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") AND NOT MSVC)
    set_source_files_properties(src/core/kai_gemm.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
    set_source_files_properties(src/qwen2_ops.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
endif()

# Platform-specific settings
if(APPLE)
    target_compile_options(mruntime PRIVATE -Wall -Wextra -Wpedantic)
elseif(UNIX)
    target_compile_options(mruntime PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(mruntime PRIVATE pthread)
endif()

# Tests
if(MRUNTIME_BUILD_TESTS)
    enable_testing()
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(test)
endif()

# Benchmarks
if(MRUNTIME_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# CLI tools
if(MRUNTIME_BUILD_CLI)
    add_executable(mruntime_chat cli/mruntime_chat_lite.cpp)
    target_link_libraries(mruntime_chat PRIVATE mruntime)
    target_include_directories(mruntime_chat PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/tokenizer
    )
endif()
