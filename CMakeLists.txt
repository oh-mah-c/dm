cmake_minimum_required(VERSION 3.18)
project(dm LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Paths ─────────────────────────────────────────────────────────────────────
set(DM_ROOT     ${CMAKE_SOURCE_DIR})
set(DM_INCLUDE  ${DM_ROOT}/include)
set(DM_SRC      ${DM_ROOT}/src)

set(LIBTORCH_DIR  ${DM_ROOT}/src/core/pytorch/dist)
set(VULKANHPP_DIR ${DM_ROOT}/src/core/gpu/Vulkan-Hpp)

# ── LibTorch ──────────────────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# ── Vulkan (C API — for dm_gpu.c) ────────────────────────────────────────────
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan: ${Vulkan_LIBRARIES}")

# ── Vulkan-Hpp (C++ header-only) ──────────────────────────────────────────────
# Headers live in Vulkan-Hpp/dist/include/ (installed) and
# Vulkan-Hpp/vulkan/ (source) — both work; use installed path.
set(VULKANHPP_INCLUDE ${VULKANHPP_DIR}/dist/include)

# ── Collect sources ───────────────────────────────────────────────────────────

# Core C files (dm_arena, dm_dataset, etc.)
file(GLOB DM_CORE_SRCS ${DM_SRC}/core/*.c)

# GPU files (pure C Vulkan backend)
file(GLOB DM_GPU_SRCS  ${DM_SRC}/core/gpu/*.c)

# Tokenizer
file(GLOB DM_TOK_SRCS  ${DM_SRC}/tokenizer/*.c)
# tokenizer/main.c is a standalone binary — exclude from library
list(FILTER DM_TOK_SRCS EXCLUDE REGEX ".*/main\\.c$")

# Algorithms
file(GLOB DM_ALG_SRCS  ${DM_SRC}/algorithms/*.c)

# Tools — each is its own binary, collect them separately
file(GLOB DM_TOOL_SRCS ${DM_SRC}/tools/*.c)

# ── dm_core library ───────────────────────────────────────────────────────────
add_library(dm_core STATIC
    ${DM_CORE_SRCS}
    ${DM_GPU_SRCS}
    ${DM_TOK_SRCS}
    ${DM_ALG_SRCS}
)

target_include_directories(dm_core PUBLIC
    ${DM_INCLUDE}
    ${DM_INCLUDE}/core
    ${DM_INCLUDE}/algorithms
    ${DM_INCLUDE}/tokenizer
    ${DM_SRC}
    ${VULKANHPP_INCLUDE}
    ${Vulkan_INCLUDE_DIRS}
    # LibTorch headers (C++ — used by any .cpp that links dm_core)
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(dm_core PUBLIC
    Vulkan::Vulkan
    ${TORCH_LIBRARIES}
    m       # math
    dl      # dlopen (Vulkan loader)
    pthread
    icuuc   # ICU Unicode (sentencepiece_lite)
    icui18n
    icudata
)

target_compile_definitions(dm_core PUBLIC
    DM_VULKAN_ENABLED=1
    DM_TORCH_ENABLED=1
)

# ── dm main binary ────────────────────────────────────────────────────────────
add_executable(dm ${DM_SRC}/main.c)
target_link_libraries(dm PRIVATE dm_core)

# ── dm_tokenizer binary ───────────────────────────────────────────────────────
add_executable(dm_tokenizer ${DM_SRC}/tokenizer/main.c)
target_link_libraries(dm_tokenizer PRIVATE dm_core)

# ── Tool binaries (one per .c in src/tools/) ─────────────────────────────────
foreach(tool_src ${DM_TOOL_SRCS})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    add_executable(${tool_name} ${tool_src})
    target_link_libraries(${tool_name} PRIVATE dm_core)
endforeach()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS dm dm_core dm_tokenizer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${DM_INCLUDE}/ DESTINATION include)
