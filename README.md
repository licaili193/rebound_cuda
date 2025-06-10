# CUDA Vector Addition Sample

This is a simple CUDA sample project that demonstrates vector addition using GPU acceleration. The project uses CMake as its build system.

## Prerequisites

- CUDA Toolkit (version 11.0 or later recommended)
- CMake (version 3.8 or later)
- A CUDA-capable GPU
- C++ compiler (Visual Studio on Windows, GCC on Linux)

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure the project with CMake:
```bash
cmake ..
```

3. Build the project:
```bash
cmake --build .
```

## Running the Sample

After building, you can run the sample from the build directory:
```bash
./vector_add
```

The program will:
1. Initialize two vectors with random values
2. Perform vector addition on the GPU
3. Verify the results
4. Print the device information and test results

## Project Structure

- `CMakeLists.txt` - Main CMake configuration file
- `include/vector_add.h` - Header file with function declarations
- `src/vector_add.cu` - CUDA source file with implementation
- `README.md` - This file

## Notes

- The sample uses CUDA architecture 7.5 by default. You may need to adjust this in CMakeLists.txt based on your GPU.
- The vector size is set to 50,000 elements by default.
- Error checking is implemented for all CUDA operations. 