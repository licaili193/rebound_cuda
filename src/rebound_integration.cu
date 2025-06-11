#include "rebound_cuda.h"

// CUDA kernel to update particle positions (leapfrog integration - drift step)
__global__ void updatePositionsKernel(Particle* particles, int n, double dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

// CUDA kernel to update particle velocities (leapfrog integration - kick step)
__global__ void updateVelocitiesKernel(Particle* particles, int n, double dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n) {
        particles[i].vx += particles[i].ax * dt;
        particles[i].vy += particles[i].ay * dt;
        particles[i].vz += particles[i].az * dt;
    }
} 