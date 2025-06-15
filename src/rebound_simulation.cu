#include "rebound_simulation.h"
#include "rebound_gravity.h"
#include "rebound_integration.h"
#include "rebound_utils.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>

// ReboundCudaSimulation class implementation
ReboundCudaSimulation::ReboundCudaSimulation() {
    h_particles = nullptr;
    d_particles = nullptr;
    particles_allocated = false;
    device_particles_current = false;  // Initially device particles are not current
    particle_count = 0;
    
    // Initialize configuration with default values
    config.n_particles = 0;
    config.t = 0.0;
    config.dt = 0.01;
    config.G = 1.0;
    config.gravity_mode = GRAVITY_BASIC;
    config.softening = 0.0;
    config.opening_angle = 0.5;
    config.max_iterations = 1000000;
    config.max_tree_depth = 20;
    config.collision_detection = false;
}

ReboundCudaSimulation::~ReboundCudaSimulation() {
    // Free host memory
    if (h_particles) {
        free(h_particles);
        h_particles = nullptr;
    }
    
    // Free device memory
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
    
    particles_allocated = false;
}

void ReboundCudaSimulation::initializeSimulation(int n_particles, double dt, double G) {
    // Clean up previous allocation if any
    if (h_particles) {
        free(h_particles);
        h_particles = nullptr;
    }
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
    
    config.n_particles = n_particles;
    config.dt = dt;
    config.G = G;
    config.t = 0.0;
    
    // Reset particle counter for new simulation
    particle_count = 0;
    
    // Allocate host memory for particles
    h_particles = (Particle*)malloc(n_particles * sizeof(Particle));
    if (!h_particles) {
        std::cerr << "Failed to allocate host memory for particles" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    memset(h_particles, 0, n_particles * sizeof(Particle));
    
    // Allocate device memory for particles
    cudaError_t err = cudaMalloc((void**)&d_particles, n_particles * sizeof(Particle));
    checkCudaError(err, "Failed to allocate device memory for particles");
    
    particles_allocated = true;
}

void ReboundCudaSimulation::addParticle(double m, double x, double y, double z, 
                                       double vx, double vy, double vz, double r) {
    std::cout << "DEBUG: Adding particle " << particle_count << " with mass=" << m << ", pos=(" << x << ", " << y << ", " << z << ")" << std::endl;
    
    if (particle_count >= config.n_particles) {
        std::cerr << "Error: Cannot add more particles than allocated (" << config.n_particles << ")" << std::endl;
        return;
    }
    
    if (!h_particles) {
        std::cerr << "Error: Particles not allocated. Call initializeSimulation first." << std::endl;
        return;
    }
    
    Particle& p = h_particles[particle_count];
    p.m = m;
    p.x = x; p.y = y; p.z = z;
    p.vx = vx; p.vy = vy; p.vz = vz;
    p.ax = 0.0; p.ay = 0.0; p.az = 0.0;
    p.r = r;
    
    std::cout << "DEBUG: After setting values: mass=" << p.m << ", pos=(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    
    particle_count++;
    device_particles_current = false;  // Device is no longer current after adding particles
    std::cout << "DEBUG: particle_count now = " << particle_count << std::endl;
}

void ReboundCudaSimulation::setGravityMode(GravityMode mode) {
    config.gravity_mode = mode;
}

void ReboundCudaSimulation::setTreeParameters(double opening_angle, double softening) {
    config.opening_angle = opening_angle;
    config.softening = softening;
}

void ReboundCudaSimulation::copyParticlesToDevice() {
    if (!particles_allocated) return;
    
    std::cout << "DEBUG: Copying " << particle_count << " particles to device (config.n_particles=" << config.n_particles << ")" << std::endl;
    
    cudaError_t err = cudaMemcpy(d_particles, h_particles, 
                                particle_count * sizeof(Particle), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy particles to device");
    
    device_particles_current = true;  // Device is now current
}

void ReboundCudaSimulation::copyParticlesFromDevice() {
    if (!particles_allocated || !device_particles_current) return;
    
    std::cout << "DEBUG: Copying " << particle_count << " particles from device (config.n_particles=" << config.n_particles << ")" << std::endl;
    
    cudaError_t err = cudaMemcpy(h_particles, d_particles, 
                                particle_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy particles from device");
}

void ReboundCudaSimulation::computeForces() {
    if (particle_count == 0) return;
    
    // Set up kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (particle_count + threadsPerBlock - 1) / threadsPerBlock;
    
    switch (config.gravity_mode) {
        case GRAVITY_NONE:
            // No gravity calculation - just zero out accelerations
            {
                int threadsPerBlock = 256;
                int blocksPerGrid = (particle_count + threadsPerBlock - 1) / threadsPerBlock;
                
                // Zero out only acceleration components, not the entire particle
                dim3 block(threadsPerBlock);
                dim3 grid(blocksPerGrid);
                
                // Simple kernel to zero accelerations (we'll need to add this)
                // For now, just skip gravity calculation
            }
            break;
            
        case GRAVITY_BASIC:
            computeForcesBasicKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_particles, particle_count, config.G, config.softening);
            break;
            
        case GRAVITY_COMPENSATED:
            computeForcesCompensatedKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_particles, particle_count, config.G, config.softening);
            break;
            
        case GRAVITY_TREE:
            // Build tree before computing forces
            buildTree();
            computeForcesTreeKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_particles, oct_tree.getDeviceNodes(), particle_count, 
                config.G, config.opening_angle, config.softening);
            break;
    }
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel execution failed in computeForces");
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed in computeForces");
}

void ReboundCudaSimulation::updatePositions() {
    if (particle_count == 0) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (particle_count + threadsPerBlock - 1) / threadsPerBlock;
    
    updatePositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, particle_count, config.dt);
    
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel execution failed in updatePositions");
    
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed in updatePositions");
}

void ReboundCudaSimulation::step() {
    // Leapfrog integration steps:
    
    // 1. Update velocities by half step (kick)
    int threadsPerBlock = 256;
    int blocksPerGrid = (particle_count + threadsPerBlock - 1) / threadsPerBlock;
    
    updateVelocitiesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, particle_count, config.dt * 0.5);
    
    // 2. Update positions by full step (drift)
    updatePositions();
    
    // 3. Compute new forces
    computeForces();
    
    // 4. Update velocities by half step (kick)
    updateVelocitiesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, particle_count, config.dt * 0.5);
    
    // Update simulation time
    config.t += config.dt;
    
    // Synchronize to ensure all operations are complete
    cudaError_t err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed in step");
}

void ReboundCudaSimulation::integrate(double t_end) {
    std::cout << "DEBUG: Starting integration..." << std::endl;
    
    // Debug: Check particles before copying to device
    std::cout << "DEBUG: Particles before copying to device:" << std::endl;
    for (int i = 0; i < particle_count; i++) {
        Particle& p = h_particles[i];
        std::cout << "  Particle " << i << ": mass=" << p.m << ", pos=(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    }
    
    // Copy particles to device
    copyParticlesToDevice();
    
    // Debug: Copy back immediately to check if copy worked
    copyParticlesFromDevice();
    std::cout << "DEBUG: Particles after round-trip copy:" << std::endl;
    for (int i = 0; i < particle_count; i++) {
        Particle& p = h_particles[i];
        std::cout << "  Particle " << i << ": mass=" << p.m << ", pos=(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    }
    
    // Copy to device again for simulation
    copyParticlesToDevice();
    
    // Initial force calculation
    computeForces();
    
    // Debug: Check after force calculation
    copyParticlesFromDevice();
    std::cout << "DEBUG: Particles after force calculation:" << std::endl;
    for (int i = 0; i < particle_count; i++) {
        Particle& p = h_particles[i];
        std::cout << "  Particle " << i << ": mass=" << p.m << ", pos=(" << p.x << ", " << p.y << ", " << p.z << "), acc=(" << p.ax << ", " << p.ay << ", " << p.az << ")" << std::endl;
    }
    
    // Copy back to device for integration
    copyParticlesToDevice();
    
    int steps = 0;
    while (config.t < t_end && steps < config.max_iterations) {
        step();
        steps++;
        
        // Optional: print progress every 1000 steps
        if (steps % 1000 == 0) {
            std::cout << "Step " << steps << ", t = " << config.t << std::endl;
        }
        
        // Early exit for debugging
        if (steps >= 2) break;
    }
    
    // Copy final results back to host
    copyParticlesFromDevice();
}

void ReboundCudaSimulation::printParticles() {
    // If device particles are current (simulation has run), copy from device
    // Otherwise use host particles directly
    if (device_particles_current) {
        copyParticlesFromDevice();
    }
    
    std::cout << "\n=== Particle States ===" << std::endl;
    for (int i = 0; i < particle_count; i++) {
        Particle& p = h_particles[i];
        std::cout << "Particle " << i << ": ";
        std::cout << "pos=(" << p.x << ", " << p.y << ", " << p.z << ") ";
        std::cout << "vel=(" << p.vx << ", " << p.vy << ", " << p.vz << ") ";
        std::cout << "mass=" << p.m << std::endl;
    }
}

double ReboundCudaSimulation::getTotalEnergy() {
    // If device particles are current (simulation has run), copy from device
    // Otherwise use host particles directly
    if (device_particles_current) {
        copyParticlesFromDevice();
    }
    
    double kinetic = 0.0;
    double potential = 0.0;
    
    // Calculate kinetic energy
    for (int i = 0; i < particle_count; i++) {
        Particle& p = h_particles[i];
        double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        kinetic += 0.5 * p.m * v2;
    }
    
    // Calculate potential energy
    for (int i = 0; i < particle_count; i++) {
        for (int j = i + 1; j < particle_count; j++) {
            Particle& pi = h_particles[i];
            Particle& pj = h_particles[j];
            
            double dx = pj.x - pi.x;
            double dy = pj.y - pi.y;
            double dz = pj.z - pi.z;
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r > 1e-15) {
                potential -= config.G * pi.m * pj.m / r;
            }
        }
    }
    
    return kinetic + potential;
}

void ReboundCudaSimulation::buildTree() {
    // Copy particles from device to host for tree construction
    copyParticlesFromDevice();
    
    // Build tree using the OctTree class
    oct_tree.buildTree(h_particles, particle_count);
    
    // Copy tree to device for GPU kernels
    oct_tree.copyToDevice();
} 