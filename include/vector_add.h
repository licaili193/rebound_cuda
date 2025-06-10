#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <cuda_runtime.h>

// Function to perform vector addition on GPU
void vectorAdd(const float* A, const float* B, float* C, int numElements);

// Function to check for CUDA errors
void checkCudaError(cudaError_t error, const char* msg);

#endif // VECTOR_ADD_H 