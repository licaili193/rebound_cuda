#include <stdio.h>
#include <cuda_runtime.h>
#include "vector_add.h"

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t error, const char* msg)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Function to perform vector addition on GPU
void vectorAdd(const float* A, const float* B, float* C, int numElements)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the device input vectors A and B
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, numElements * sizeof(float));
    checkCudaError(err, "Failed to allocate device vector A");

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, numElements * sizeof(float));
    checkCudaError(err, "Failed to allocate device vector B");

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, numElements * sizeof(float));
    checkCudaError(err, "Failed to allocate device vector C");

    // Copy the host input vectors A and B in host memory to the device input vectors in device memory
    err = cudaMemcpy(d_A, A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vector A from host to device");

    err = cudaMemcpy(d_B, B, numElements * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy vector B from host to device");

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    checkCudaError(err, "Failed to launch vectorAddKernel");

    // Copy the device result vector in device memory to the host result vector in host memory
    err = cudaMemcpy(C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy vector C from device to host");

    // Free device global memory
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the device properties
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, 0);
    checkCudaError(err, "Failed to get device properties");
    printf("Using device: %s\n", deviceProp.name);

    // Vector size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vectors A and B
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Call the vector addition function
    vectorAdd(h_A, h_B, h_C, numElements);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
} 