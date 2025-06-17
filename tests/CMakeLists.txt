# Unit Tests for REBOUND CUDA
cmake_minimum_required(VERSION 3.18)

# Find CUDA
find_package(CUDA REQUIRED)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Test executable
add_executable(rebound_tests
    run_tests.cpp
)

# Link with the main library
target_link_libraries(rebound_tests rebound_cuda_lib)

# Set CUDA properties
set_target_properties(rebound_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(rebound_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

# Suppress relative include path warnings on Windows
if(MSVC)
    target_compile_options(rebound_tests PRIVATE
        /wd4464  # Relative include path contains '..'
        /wd4820  # Struct padding warnings
        /wd4100  # Unreferenced formal parameter
        /wd4625  # Copy constructor implicitly deleted
        /wd4626  # Assignment operator implicitly deleted
        /wd5219  # Implicit conversion warnings
    )
endif()

# Compiler flags
target_compile_options(rebound_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-code=arch=compute_75,code=[compute_75,sm_75]
        -Xcompiler=-Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

# Test commands
add_custom_target(test_physics
    COMMAND rebound_tests Physics
    DEPENDS rebound_tests
    COMMENT "Running Physics tests"
)

add_custom_target(test_collisions
    COMMAND rebound_tests Collisions
    DEPENDS rebound_tests
    COMMENT "Running Collision tests"
)

add_custom_target(test_streaming
    COMMAND rebound_tests Streaming
    DEPENDS rebound_tests
    COMMENT "Running Streaming tests"
)

add_custom_target(test_all
    COMMAND rebound_tests
    DEPENDS rebound_tests
    COMMENT "Running all tests"
) 