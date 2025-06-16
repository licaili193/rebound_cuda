#ifndef REBOUND_INTEGRATION_H
#define REBOUND_INTEGRATION_H

#include "rebound_types.h"
#include <cuda_runtime.h>

// CUDA kernels for integration
__global__ void zeroAccelerationsKernel(Particle* particles, int n);
__global__ void updatePositionsKernel(Particle* particles, int n, double dt);
__global__ void updateVelocitiesKernel(Particle* particles, int n, double dt);

#endif // REBOUND_INTEGRATION_H 