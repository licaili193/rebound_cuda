cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_sample LANGUAGES CXX CUDA)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add executable
add_executable(vector_add src/vector_add.cu)

# Set CUDA specific properties
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA architecture
set_target_properties(vector_add PROPERTIES
    CUDA_ARCHITECTURES 75
)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif() 