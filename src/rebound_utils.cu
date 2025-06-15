#include "rebound_utils.h"
#include <iostream>
#include <cstdlib>

// CUDA error checking function
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Print device information
void printDeviceInfo() {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);
    checkCudaError(err, "Failed to get device properties");
    
    std::cout << "=== CUDA Device Information ===" << std::endl;
    std::cout << "Device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "================================" << std::endl;
} 