#ifndef REBOUND_UTILS_H
#define REBOUND_UTILS_H

#include <cuda_runtime.h>

// Utility functions
void checkCudaError(cudaError_t error, const char* msg);
void printDeviceInfo();

#endif // REBOUND_UTILS_H 