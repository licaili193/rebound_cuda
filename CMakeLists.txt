cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(rebound_cuda LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create REBOUND CUDA library with all source files
add_library(rebound_cuda_lib 
    src/rebound_simulation.cu
    src/rebound_gravity.cu
    src/rebound_tree.cu
    src/rebound_integration.cu
    src/rebound_integrator.cu
    src/rebound_utils.cu
    src/rebound_collision.cu
    src/rebound_streaming.cu
)

set_target_properties(rebound_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
    POSITION_INDEPENDENT_CODE ON
)

# Add main examples executable
add_executable(rebound_examples 
    examples/main.cu
    examples/planetary_system.cpp
    examples/gravity_comparison.cu
)

target_link_libraries(rebound_examples rebound_cuda_lib)
set_target_properties(rebound_examples PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
)

# Add collision example executable
add_executable(collision_example 
    examples/collision_example.cpp
)

target_link_libraries(collision_example rebound_cuda_lib)
set_target_properties(collision_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
)

# Add streaming example executable
add_executable(streaming_example 
    examples/streaming_example.cpp
)

target_link_libraries(streaming_example rebound_cuda_lib)
set_target_properties(streaming_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75
)

# Add observer pattern example executable
add_executable(observer_example 
    examples/observer_pattern_example.cpp
)

target_link_libraries(observer_example rebound_cuda_lib)
set_target_properties(observer_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
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

# Add tests subdirectory
add_subdirectory(tests) 