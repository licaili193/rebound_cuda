#include "rebound_integrator.h"
#include "rebound_integration.h"   // kernel declarations
#include "rebound_utils.h"          // CUDA error helper

// Helper to launch 1-D grid
static inline dim3 launchGrid(int n, int threadsPerBlock = 256) {
    return dim3((n + threadsPerBlock - 1) / threadsPerBlock);
}

void LeapfrogIntegrator::drift(Particle* d_particles, int n_particles, double dt) {
    if (n_particles <= 0 || !d_particles) return;
    const int TPB = 256;
    dim3 blocks = launchGrid(n_particles, TPB);
    updatePositionsKernel<<<blocks, TPB>>>(d_particles, n_particles, dt);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel execution failed in LeapfrogIntegrator::drift");
}

void LeapfrogIntegrator::kick(Particle* d_particles, int n_particles, double dt) {
    if (n_particles <= 0 || !d_particles) return;
    const int TPB = 256;
    dim3 blocks = launchGrid(n_particles, TPB);
    updateVelocitiesKernel<<<blocks, TPB>>>(d_particles, n_particles, dt);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel execution failed in LeapfrogIntegrator::kick");
} 