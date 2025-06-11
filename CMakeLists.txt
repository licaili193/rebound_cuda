cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(rebound_cuda LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create REBOUND CUDA library with all source files
add_library(rebound_cuda_lib 
    src/rebound_simulation.cu
    src/rebound_gravity.cu
    src/rebound_tree.cu
    src/rebound_integration.cu
    src/rebound_utils.cu
)

set_target_properties(rebound_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Add main examples executable
add_executable(rebound_examples 
    examples/main.cu
    examples/planetary_system.cu
    examples/gravity_comparison.cu
)

target_link_libraries(rebound_examples rebound_cuda_lib)
set_target_properties(rebound_examples PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Set build type to Release by default for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags for optimization
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable CUDA device debugging for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif() 