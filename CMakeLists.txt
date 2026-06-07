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

set(LIBTORCH_DIR  ${DM_ROOT}/src/core/dm_torch/dist)
set(VULKANHPP_DIR ${DM_ROOT}/src/core/gpu/dm_vulkan)
set(SDL2_DIR ${DM_ROOT}/src/core/dm_sdl)
set(MUJOCO_DIR ${DM_ROOT}/src/core/physics/mujoco)

# ── SDL2 (Native Core) ────────────────────────────────────────────────────────
set(SDL_SHARED OFF CACHE BOOL "Build a shared version of the library" FORCE)
set(SDL_STATIC ON CACHE BOOL "Build a static version of the library" FORCE)
set(SDL_TEST OFF CACHE BOOL "Build the test directory" FORCE)
add_subdirectory(${SDL2_DIR})

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

# ── MuJoCo (Physics Engine) ───────────────────────────────────────────────────
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "Disable MuJoCo examples" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "Disable MuJoCo simulate binary" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "Disable MuJoCo tests" FORCE)
add_subdirectory(${MUJOCO_DIR} EXCLUDE_FROM_ALL)

# ── Collect sources ───────────────────────────────────────────────────────────

# Core C files (dm_arena, dm_dataset, etc.)
file(GLOB DM_CORE_SRCS ${DM_SRC}/core/*.c)

# GPU files (pure C Vulkan backend)
file(GLOB DM_GPU_SRCS  ${DM_SRC}/core/gpu/*.c)

# Tokenizer
file(GLOB DM_TOK_SRCS  ${DM_SRC}/tokenizer/*.c)
# Standalone binaries with their own main() — exclude from library
list(FILTER DM_TOK_SRCS EXCLUDE REGEX ".*/main\\.c$")
list(FILTER DM_TOK_SRCS EXCLUDE REGEX ".*/hust_tokenize\\.c$")

# Algorithms
file(GLOB DM_ALG_SRCS  ${DM_SRC}/algorithms/*.c ${DM_SRC}/algorithms/*.cpp)

# Models — C++ files only (train.cpp are standalone binaries, excluded here)
file(GLOB_RECURSE DM_MODEL_SRCS ${DM_SRC}/models/**/*.cpp)
set(DM_PYTORCH_CUSTOM_SRCS ${DM_SRC}/core/dm_torch/torch/csrc/api/src/nn/modules/ohm_qpa.cpp)
list(APPEND DM_MODEL_SRCS ${DM_PYTORCH_CUSTOM_SRCS})
list(FILTER DM_MODEL_SRCS EXCLUDE REGEX ".*/train\\.cpp$")
list(FILTER DM_MODEL_SRCS EXCLUDE REGEX ".*/infer\\.cpp$")
list(FILTER DM_MODEL_SRCS EXCLUDE REGEX ".*/test_.*\\.cpp$")

# Tools — each is its own binary, collect them separately
file(GLOB DM_TOOL_SRCS ${DM_SRC}/tools/*.c)

# ── dm_core library ───────────────────────────────────────────────────────────
add_library(dm_core STATIC
    ${DM_CORE_SRCS}
    ${DM_GPU_SRCS}
    ${DM_TOK_SRCS}
    ${DM_ALG_SRCS}
    ${DM_MODEL_SRCS}
)

target_include_directories(dm_core PUBLIC
    ${DM_INCLUDE}
    ${DM_INCLUDE}/core
    ${DM_INCLUDE}/algorithms
    ${DM_INCLUDE}/tokenizer
    ${DM_INCLUDE}/models
    ${DM_INCLUDE}/modules
    ${DM_SRC}
    ${VULKANHPP_INCLUDE}
    ${Vulkan_INCLUDE_DIRS}
    ${SDL2_DIR}/include
    # LibTorch headers (C++ — used by any .cpp that links dm_core)
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(dm_core PUBLIC
    SDL2::SDL2-static
    Vulkan::Vulkan
    mujoco::mujoco
    ${TORCH_LIBRARIES}
    m       # math
    dl      # dlopen (Vulkan loader)
    pthread
    icuuc   # ICU Unicode (sentencepiece_lite)
    icui18n
    icudata
    /usr/local/cuda-12.4/lib64/libcupti.so
)

target_compile_definitions(dm_core PUBLIC
    DM_VULKAN_ENABLED=1
    DM_TORCH_ENABLED=1
)

target_compile_options(dm_core PUBLIC
    -mavx2 -mbmi -mbmi2 -mpopcnt
)

# ── dm main binary ────────────────────────────────────────────────────────────
add_executable(dm ${DM_SRC}/main.c)
target_link_libraries(dm PRIVATE dm_core)

# ── dm_tokenizer binary ───────────────────────────────────────────────────────
add_executable(dm_tokenizer ${DM_SRC}/tokenizer/main.c)
target_link_libraries(dm_tokenizer PRIVATE dm_core)

# ── hust_tokenize binary ──────────────────────────────────────────────────────
add_executable(hust_tokenize ${DM_SRC}/tokenizer/hust_tokenize.c)
target_include_directories(hust_tokenize PRIVATE ${DM_INCLUDE}/tokenizer)
target_link_libraries(hust_tokenize PRIVATE m)

# ── llama3_train_hust binary ──────────────────────────────────────────────────
add_executable(llama3_train_hust ${DM_SRC}/models/nlp/llama3/train_hust.cpp)
target_link_libraries(llama3_train_hust PRIVATE dm_core)
target_include_directories(llama3_train_hust PRIVATE
    ${DM_INCLUDE}/models
    ${TORCH_INCLUDE_DIRS})

# ── Tool binaries (one per .c in src/tools/) ─────────────────────────────────
foreach(tool_src ${DM_TOOL_SRCS})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    add_executable(${tool_name} ${tool_src})
    target_link_libraries(${tool_name} PRIVATE dm_core)
endforeach()

# ── Model training binaries (one per train.cpp under src/models/) ─────────────
file(GLOB_RECURSE DM_MODEL_TRAIN_SRCS ${DM_SRC}/models/**/train.cpp)
foreach(train_src ${DM_MODEL_TRAIN_SRCS})
    # derive name from parent dir, e.g. src/models/vision/resnet/train.cpp → resnet_train
    get_filename_component(train_dir ${train_src} DIRECTORY)
    get_filename_component(model_name ${train_dir} NAME)
    set(train_target "${model_name}_train")
    add_executable(${train_target} ${train_src})
    target_link_libraries(${train_target} PRIVATE dm_core)
    target_include_directories(${train_target} PRIVATE
        ${DM_INCLUDE}/models
        ${TORCH_INCLUDE_DIRS})
endforeach()

# ── Model inference binaries (one per infer.cpp under src/models/) ─────────────
file(GLOB_RECURSE DM_MODEL_INFER_SRCS ${DM_SRC}/models/**/infer.cpp)
foreach(infer_src ${DM_MODEL_INFER_SRCS})
    get_filename_component(infer_dir ${infer_src} DIRECTORY)
    get_filename_component(model_name ${infer_dir} NAME)
    set(infer_target "${model_name}_infer")
    add_executable(${infer_target} ${infer_src})
    target_link_libraries(${infer_target} PRIVATE dm_core)
    target_include_directories(${infer_target} PRIVATE
        ${DM_INCLUDE}/models
        ${TORCH_INCLUDE_DIRS})
endforeach()

# ── Model test binaries (one per test_*.cpp under src/models/) ───────────────
file(GLOB_RECURSE DM_MODEL_TEST_SRCS ${DM_SRC}/models/**/test_*.cpp)
foreach(test_src ${DM_MODEL_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)  # e.g. test_resnet
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE dm_core)
    target_include_directories(${test_name} PRIVATE
        ${DM_INCLUDE}/models
        ${TORCH_INCLUDE_DIRS})
endforeach()

# ── General test binaries (e.g. C/C++ files under tests/) ────────────────────
file(GLOB DM_GEN_TEST_SRCS ${DM_ROOT}/tests/*.c ${DM_ROOT}/tests/*.cpp)
foreach(test_src ${DM_GEN_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE dm_core)
    target_include_directories(${test_name} PRIVATE
        ${DM_INCLUDE}
        ${TORCH_INCLUDE_DIRS})
endforeach()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS dm dm_core dm_tokenizer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${DM_INCLUDE}/ DESTINATION include)
