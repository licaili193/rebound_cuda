#ifndef REBOUND_GRAVITY_H
#define REBOUND_GRAVITY_H

#include "rebound_types.h"
#include <cuda_runtime.h>

// CUDA kernels for different gravity modes
__global__ void computeForcesBasicKernel(Particle* particles, int n, double G, double softening);
__global__ void computeForcesCompensatedKernel(Particle* particles, int n, double G, double softening);
__global__ void computeForcesTreeKernel(Particle* particles, TreeNode* tree_nodes, int n, 
                                       double G, double opening_angle, double softening);

// Device utility functions for gravity calculations
__device__ double distance_squared(const Particle& p, const TreeNode& node);
__device__ double node_size(const TreeNode& node);

#endif // REBOUND_GRAVITY_H 