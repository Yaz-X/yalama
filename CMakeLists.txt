cmake_minimum_required(VERSION 3.20)

# -------------------------
# Toolchain
# -------------------------

set(CMAKE_C_COMPILER /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

if(NOT DEFINED ENV{LIBTORCH_PATH})
    message(FATAL_ERROR "LIBTORCH_PATH not set. Please export LIBTORCH_PATH to your libtorch root.")
endif()

set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# -------------------------
# project
# -------------------------

project(yalama LANGUAGES CXX CUDA)

set(CMAKE_LIBRARY_PATH "/usr/local/cuda/lib64")


# -------------------------
# Build type
# -------------------------

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# -------------------------
# Packages
# -------------------------
set(CUDA_NVRTC_SHORTHASH "skip")
set(Torch_USE_CUDA OFF)

find_package(Torch REQUIRED)

# -------------------------
# Sources
# -------------------------

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
add_executable(yalama)

target_sources(yalama PRIVATE ${SRC})


target_include_directories(yalama PRIVATE include)
target_include_directories(yalama PRIVATE src/services/openAI)
target_include_directories(yalama PRIVATE extern)
set_target_properties(yalama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# -------------------------
# Linking
# -------------------------

target_link_libraries(yalama
    PRIVATE
    ${TORCH_LIBRARIES}    
)

# Needed by libtorch
set_property(TARGET yalama PROPERTY CXX_STANDARD 20)
