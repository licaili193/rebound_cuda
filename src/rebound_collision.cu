#include "../include/rebound_collision.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdio.h>

// CUDA kernel for direct collision detection (O(N²))
__global__ void detectCollisionsDirectKernel(Particle* particles, int n_particles,
                                            Collision* collisions, int* collision_count,
                                            int max_collisions, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= n_particles || j >= n_particles || i >= j) return;
    
    const Particle& p1 = particles[i];
    const Particle& p2 = particles[j];
    
    // Skip if either particle has zero radius
    if (p1.r <= 0.0f || p2.r <= 0.0f) return;
    
    // Calculate distance between particles
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    // Check if collision occurred (distance <= sum of radii)
    double collision_distance = p1.r + p2.r;
    if (distance <= collision_distance) {
        // Check relative velocity (particles must be approaching)
        double dvx = p1.vx - p2.vx;
        double dvy = p1.vy - p2.vy;
        double dvz = p1.vz - p2.vz;
        double relative_velocity = dx*dvx + dy*dvy + dz*dvz;
        
        if (relative_velocity < 0.0) { // Approaching
            int collision_idx = atomicAdd(collision_count, 1);
            if (collision_idx < max_collisions) {
                collisions[collision_idx].p1 = i;
                collisions[collision_idx].p2 = j;
                collisions[collision_idx].distance = distance;
                collisions[collision_idx].time = 0.0f; // Will be set by host
            }
        }
    }
}

// CUDA kernel for resolving collisions
__global__ void resolveCollisionsKernel(Particle* particles, Collision* collisions,
                                       int n_collisions, float coefficient_of_restitution,
                                       CollisionResolution method) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_collisions) return;
    
    const Collision& collision = collisions[idx];
    Particle& p1 = particles[collision.p1];
    Particle& p2 = particles[collision.p2];
    
    if (method == COLLISION_RESOLVE_HARDSPHERE) {
        // Hard sphere collision resolution
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance > 0.0) {
            // Normalize collision vector
            double nx = dx / distance;
            double ny = dy / distance;
            double nz = dz / distance;
            
            // Relative velocity
            double dvx = p1.vx - p2.vx;
            double dvy = p1.vy - p2.vy;
            double dvz = p1.vz - p2.vz;
            
            // Velocity component along collision normal
            double dvn = dvx*nx + dvy*ny + dvz*nz;
            
            if (dvn < 0.0) { // Only resolve if approaching
                // Calculate impulse
                double total_mass = p1.m + p2.m;
                double impulse = -(1.0 + coefficient_of_restitution) * dvn / total_mass;
                
                // Apply impulse to both particles
                double impulse_x = impulse * nx;
                double impulse_y = impulse * ny;
                double impulse_z = impulse * nz;
                
                p1.vx += impulse_x * p2.m;
                p1.vy += impulse_y * p2.m;
                p1.vz += impulse_z * p2.m;
                
                p2.vx -= impulse_x * p1.m;
                p2.vy -= impulse_y * p1.m;
                p2.vz -= impulse_z * p1.m;
            }
        }
    }
    else if (method == COLLISION_RESOLVE_MERGE) {
        // Merge particles - combine mass, momentum, conserve volume
        double total_mass = p1.m + p2.m;
        double total_momentum_x = p1.m * p1.vx + p2.m * p2.vx;
        double total_momentum_y = p1.m * p1.vy + p2.m * p2.vy;
        double total_momentum_z = p1.m * p1.vz + p2.m * p2.vz;
        
        // Combined position (center of mass)
        p1.x = (p1.m * p1.x + p2.m * p2.x) / total_mass;
        p1.y = (p1.m * p1.y + p2.m * p2.y) / total_mass;
        p1.z = (p1.m * p1.z + p2.m * p2.z) / total_mass;
        
        // Combined velocity (conserve momentum)
        p1.vx = total_momentum_x / total_mass;
        p1.vy = total_momentum_y / total_mass;
        p1.vz = total_momentum_z / total_mass;
        
        // Combined mass
        p1.m = total_mass;
        
        // Combined radius (conserve volume: V = 4/3 * π * r³)
        double volume1 = p1.r * p1.r * p1.r;
        double volume2 = p2.r * p2.r * p2.r;
        p1.r = pow(volume1 + volume2, 1.0/3.0);
        
        // Mark second particle for removal
        p2.m = 0.0f;
        p2.r = 0.0f;
    }
}

// CollisionDetector implementation
CollisionDetector::CollisionDetector() 
    : detection_method_(COLLISION_NONE)
    , resolution_method_(COLLISION_RESOLVE_HALT)
    , coefficient_of_restitution_(0.5f) {
}

CollisionDetector::~CollisionDetector() {
}

void CollisionDetector::setDetectionMethod(CollisionDetection method) {
    detection_method_ = method;
}

void CollisionDetector::setResolutionMethod(CollisionResolution method) {
    resolution_method_ = method;
}

void CollisionDetector::setCoefficientOfRestitution(float epsilon) {
    coefficient_of_restitution_ = epsilon;
}

int CollisionDetector::detectCollisions(Particle* particles, int n_particles, float dt,
                                       Collision* collisions, int max_collisions) {
    if (detection_method_ == COLLISION_NONE) {
        return 0;
    }
    
    switch (detection_method_) {
        case COLLISION_DIRECT:
            return detectDirect(particles, n_particles, dt, collisions, max_collisions);
        case COLLISION_TREE:
            return detectTree(particles, n_particles, dt, collisions, max_collisions);
        default:
            return 0;
    }
}

int CollisionDetector::detectDirect(Particle* particles, int n_particles, float dt,
                                   Collision* collisions, int max_collisions) {
    // Allocate device memory for collision count
    int* d_collision_count;
    cudaMalloc(&d_collision_count, sizeof(int));
    cudaMemset(d_collision_count, 0, sizeof(int));
    
    // Launch collision detection kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((n_particles + blockSize.x - 1) / blockSize.x,
                  (n_particles + blockSize.y - 1) / blockSize.y);
    
    detectCollisionsDirectKernel<<<gridSize, blockSize>>>(
        particles, n_particles, collisions, d_collision_count, max_collisions, dt);
    
    // Get collision count
    int collision_count;
    cudaMemcpy(&collision_count, d_collision_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_collision_count);
    
    // Limit to max_collisions
    return std::min(collision_count, max_collisions);
}

int CollisionDetector::detectTree(Particle* particles, int n_particles, float dt,
                                 Collision* collisions, int max_collisions) {
    // For now, fall back to direct detection
    // Tree-based detection would require octree implementation
    return detectDirect(particles, n_particles, dt, collisions, max_collisions);
}

int CollisionDetector::resolveCollisions(Particle* particles, Collision* collisions,
                                        int n_collisions, float current_time) {
    if (n_collisions == 0 || resolution_method_ == COLLISION_RESOLVE_HALT) {
        return (n_collisions > 0) ? 1 : 0; // Return 1 if should halt, 0 otherwise
    }
    
    // Launch collision resolution kernel
    dim3 blockSize(256);
    dim3 gridSize((n_collisions + blockSize.x - 1) / blockSize.x);
    
    resolveCollisionsKernel<<<gridSize, blockSize>>>(
        particles, collisions, n_collisions, coefficient_of_restitution_, resolution_method_);
    
    cudaDeviceSynchronize();
    return 0; // Continue simulation
}

bool CollisionDetector::checkCollision(const Particle& p1, const Particle& p2, float dt) {
    if (p1.r <= 0.0 || p2.r <= 0.0) return false;
    
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    return distance < (p1.r + p2.r);
}

float CollisionDetector::getDistance(const Particle& p1, const Particle& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
} 