#include "rebound_simulation.h"
#include "rebound_gravity.h"
#include "rebound_integration.h"
#include "rebound_utils.h"
#include "rebound_integrator.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ReboundCudaSimulation class implementation
ReboundCudaSimulation::ReboundCudaSimulation() 
    : h_particles_(nullptr), d_particles_(nullptr), particles_allocated_(false), 
      device_particles_current_(false), particle_count_(0), max_collisions_(1000),
      collision_enabled_(false), d_collisions_(nullptr), integrator_(nullptr) {
    
    // Initialize configuration with default values
    config_.n_particles = 0;
    config_.t = 0.0;
    config_.dt = 0.01;
    config_.G = 1.0;
    config_.gravity_mode = GRAVITY_BASIC;
    config_.softening = 0.0;
    config_.opening_angle = 0.5;
    config_.max_iterations = 1000000;
    config_.max_tree_depth = 20;
    config_.collision_detection = false;
    
    // Initialize collision system
    collision_detector_.setDetectionMethod(COLLISION_NONE);
    collision_detector_.setResolutionMethod(COLLISION_RESOLVE_HALT);
    collision_detector_.setCoefficientOfRestitution(0.5f);
    
    // Default to leapfrog integrator
    static LeapfrogIntegrator default_leapfrog;
    integrator_ = &default_leapfrog;
}

ReboundCudaSimulation::~ReboundCudaSimulation() {
    // Free host memory
    if (h_particles_) {
        free(h_particles_);
        h_particles_ = nullptr;
    }
    
    // Free device memory
    if (d_particles_) {
        cudaFree(d_particles_);
        d_particles_ = nullptr;
    }
    
    particles_allocated_ = false;
    freeCollisionMemory();
}

void ReboundCudaSimulation::initializeSimulation(int n_particles, double dt, double G) {
    // Clean up previous allocation if any
    if (h_particles_) {
        free(h_particles_);
        h_particles_ = nullptr;
    }
    if (d_particles_) {
        cudaFree(d_particles_);
        d_particles_ = nullptr;
    }
    
    config_.n_particles = n_particles;
    config_.dt = dt;
    config_.G = G;
    config_.t = 0.0;
    
    // Reset particle counter for new simulation
    particle_count_ = 0;
    
    // Allocate host memory for particles
    h_particles_ = (Particle*)malloc(n_particles * sizeof(Particle));
    if (!h_particles_) {
        std::cerr << "Failed to allocate host memory for particles" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    memset(h_particles_, 0, n_particles * sizeof(Particle));
    
    // Allocate device memory for particles
    cudaError_t err = cudaMalloc((void**)&d_particles_, n_particles * sizeof(Particle));
    checkCudaError(err, "Failed to allocate device memory for particles");
    
    particles_allocated_ = true;
}

void ReboundCudaSimulation::addParticle(double m, double x, double y, double z, 
                                       double vx, double vy, double vz, double r) {
    if (particle_count_ >= config_.n_particles) {
        std::cerr << "Error: Cannot add more particles than allocated (" << config_.n_particles << ")" << std::endl;
        return;
    }
    
    if (!h_particles_) {
        std::cerr << "Error: Particles not allocated. Call initializeSimulation first." << std::endl;
        return;
    }
    
    Particle& p = h_particles_[particle_count_];
    p.m = m;
    p.x = x; p.y = y; p.z = z;
    p.vx = vx; p.vy = vy; p.vz = vz;
    p.ax = 0.0; p.ay = 0.0; p.az = 0.0;
    p.r = r;
    
    particle_count_++;
    device_particles_current_ = false;  // Device is no longer current after adding particles
}

void ReboundCudaSimulation::setGravityMode(GravityMode mode) {
    config_.gravity_mode = mode;
}

void ReboundCudaSimulation::setTreeParameters(double opening_angle, double softening) {
    config_.opening_angle = opening_angle;
    config_.softening = softening;
}

void ReboundCudaSimulation::copyParticlesToDevice() {
    if (!particles_allocated_) return;
    
    cudaError_t err = cudaMemcpy(d_particles_, h_particles_, 
                                particle_count_ * sizeof(Particle), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy particles to device");
    
    device_particles_current_ = true;  // Device is now current
}

void ReboundCudaSimulation::copyParticlesFromDevice() {
    if (!particles_allocated_ || !device_particles_current_) return;
    
    cudaError_t err = cudaMemcpy(h_particles_, d_particles_, 
                                particle_count_ * sizeof(Particle), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy particles from device");
}

void ReboundCudaSimulation::computeForces() {
    if (particle_count_ == 0) return;
    
    // Skip all force calculations if gravity is disabled
    if (config_.gravity_mode == GRAVITY_NONE) {
        // Ensure accelerations are zero only once at the beginning
        if (config_.t == 0.0) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (particle_count_ + threadsPerBlock - 1) / threadsPerBlock;
            zeroAccelerationsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles_, particle_count_);
            cudaError_t err = cudaGetLastError();
            checkCudaError(err, "Kernel execution failed in computeForces (GRAVITY_NONE)");
            err = cudaDeviceSynchronize();
            checkCudaError(err, "Device synchronization failed in computeForces (GRAVITY_NONE)");
        }
        return; // No gravitational forces
    }

    // Set up kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (particle_count_ + threadsPerBlock - 1) / threadsPerBlock;

    switch (config_.gravity_mode) {
        case GRAVITY_BASIC:
            {
                size_t sharedBytes = threadsPerBlock * sizeof(Particle);
                computeForcesBasicKernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(
                    d_particles_, particle_count_, config_.G, config_.softening);
            }
            break;
        case GRAVITY_COMPENSATED:
            computeForcesCompensatedKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_particles_, particle_count_, config_.G, config_.softening);
            break;
        case GRAVITY_TREE:
            buildTree();
            computeForcesTreeKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_particles_, oct_tree_.getDeviceNodes(), particle_count_,
                config_.G, config_.opening_angle, config_.softening);
            break;
        default:
            break;
    }
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel execution failed in computeForces");
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed in computeForces");
}

void ReboundCudaSimulation::updatePositions() {
    if (particle_count_ == 0) return;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (particle_count_ + threadsPerBlock - 1) / threadsPerBlock;
    
    updatePositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles_, particle_count_, config_.dt);
    
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel execution failed in updatePositions");
    
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed in updatePositions");
}

bool ReboundCudaSimulation::step() {
    // Proper DKD integration pattern following rebound's approach:
    
    // Detect and handle collisions before integration
    if (collision_enabled_ && detectAndResolveCollisions(config_.dt, config_.t) == 1) {
        return false; // signal halt
    }
    
    // 1. Drift half-step
    if (integrator_) integrator_->drift(d_particles_, particle_count_, config_.dt * 0.5);
    
    // 2. Tree updates (if needed for tree-based gravity)
    // This happens inside computeForces for GRAVITY_TREE mode
    
    // 3. Calculate accelerations (gravity and other forces)
    computeForces();
    
    // 4. Kick full-step then drift half-step
    if (integrator_) integrator_->kick(d_particles_, particle_count_, config_.dt);
    if (integrator_) integrator_->drift(d_particles_, particle_count_, config_.dt * 0.5);
    
    // 5. Update simulation time
    config_.t += config_.dt;
    
    // Detect and handle collisions after integration
    if (collision_enabled_ && detectAndResolveCollisions(config_.dt, config_.t) == 1) {
        return false;
    }
    
    // 6. Synchronize to ensure all operations are complete
    cudaError_t err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed in step");

    return true; // continue
}

// =============================================================================
// Observer Pattern Implementation
// =============================================================================

void ReboundCudaSimulation::addObserver(SimulationObserver* observer) {
    if (observer && std::find(observers_.begin(), observers_.end(), observer) == observers_.end()) {
        observers_.push_back(observer);
    }
}

void ReboundCudaSimulation::removeObserver(SimulationObserver* observer) {
    auto it = std::find(observers_.begin(), observers_.end(), observer);
    if (it != observers_.end()) {
        observers_.erase(it);
    }
}

void ReboundCudaSimulation::notifySimulationStart() {
    for (auto* observer : observers_) {
        observer->onSimulationStart(particle_count_);
    }
}

void ReboundCudaSimulation::notifySimulationStep(int step) {
    for (auto* observer : observers_) {
        observer->onSimulationStep(config_.t, step, particle_count_, 0.0, 0);
    }
}

void ReboundCudaSimulation::notifySimulationEnd(int total_steps) {
    for (auto* observer : observers_) {
        observer->onSimulationEnd(config_.t, total_steps);
    }
}

void ReboundCudaSimulation::notifyCollisionDetected(int particle1, int particle2) {
    for (auto* observer : observers_) {
        observer->onCollisionDetected(particle1, particle2, config_.t);
    }
}

// Public method to explicitly synchronize host particle array
void ReboundCudaSimulation::copyParticlesToHost() {
    copyParticlesFromDevice();
    device_particles_current_ = false;
}

// =============================================================================
// Efficient Integration Method (completely independent of streaming)
// =============================================================================

void ReboundCudaSimulation::integrate(double t_end) {
    std::cout << "Starting integration to time " << t_end << std::endl;
    
    if (particle_count_ == 0) {
        std::cout << "No particles to integrate" << std::endl;
        return;
    }
    
    // Notify observers that simulation is starting
    notifySimulationStart();
    
    // One-time setup: copy particles to device only once
    copyParticlesToDevice();
    
    // Initial force calculation
    computeForces();
    
    int steps = 0;
    while (config_.t < t_end && steps < config_.max_iterations) {
        // Adjust timestep for final step to not overshoot
        double original_dt = config_.dt;
        if (config_.t + config_.dt > t_end) {
            config_.dt = t_end - config_.t;
        }
        
        // Execute simulation step (all on GPU)
        if(!step()) {
            break; // collision halted
        }
        steps++;
        
        // Restore original timestep
        config_.dt = original_dt;
        
        // Notify observers of simulation step (optional - no overhead if no observers)
        if (!observers_.empty()) {
            notifySimulationStep(steps);
        }
        
        // Optional progress reporting
        if (steps % 1000 == 0) {
            std::cout << "Step " << steps << ", t = " << config_.t << " (GPU-resident)" << std::endl;
        }
    }
    
    notifySimulationEnd(steps);
    
    std::cout << "Integration completed. " << steps << " steps executed." << std::endl;
    std::cout << "Final time: " << config_.t << std::endl;
    std::cout << "Data remains on GPU. Use observer pattern or legacy methods to access." << std::endl;
}

void ReboundCudaSimulation::printParticles() {
    // If device particles are current (simulation has run), copy from device
    // Otherwise use host particles directly
    if (device_particles_current_) {
        copyParticlesFromDevice();
    }
    
    std::cout << "\n=== Particle States ===" << std::endl;
    for (int i = 0; i < particle_count_; i++) {
        Particle& p = h_particles_[i];
        std::cout << "Particle " << i << ": ";
        std::cout << "pos=(" << p.x << ", " << p.y << ", " << p.z << ") ";
        std::cout << "vel=(" << p.vx << ", " << p.vy << ", " << p.vz << ") ";
        std::cout << "mass=" << p.m << std::endl;
    }
}

double ReboundCudaSimulation::getTotalEnergy() {
    // If device particles are current (simulation has run), copy from device
    // Otherwise use host particles directly
    if (device_particles_current_) {
        copyParticlesFromDevice();
    }
    
    double kinetic = 0.0;
    double potential = 0.0;
    
    // Calculate kinetic energy
    for (int i = 0; i < particle_count_; i++) {
        Particle& p = h_particles_[i];
        double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        kinetic += 0.5 * p.m * v2;
    }
    
    // Calculate potential energy
    for (int i = 0; i < particle_count_; i++) {
        for (int j = i + 1; j < particle_count_; j++) {
            Particle& pi = h_particles_[i];
            Particle& pj = h_particles_[j];
            
            double dx = pj.x - pi.x;
            double dy = pj.y - pi.y;
            double dz = pj.z - pi.z;
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r > 1e-15) {
                potential -= config_.G * pi.m * pj.m / r;
            }
        }
    }
    
    return kinetic + potential;
}

void ReboundCudaSimulation::buildTree() {
    // Copy particles from device to host for tree construction
    copyParticlesFromDevice();
    
    // Build tree using the OctTree class
    oct_tree_.buildTree(h_particles_, particle_count_);
    
    // Copy tree to device for GPU kernels
    oct_tree_.copyToDevice();
}

void ReboundCudaSimulation::freeCollisionMemory() {
    if (d_collisions_) {
        cudaFree(d_collisions_);
        d_collisions_ = nullptr;
    }
}

void ReboundCudaSimulation::setCollisionDetection(CollisionDetection method) {
    collision_detector_.setDetectionMethod(method);
    collision_enabled_ = (method != COLLISION_NONE);
    
    // Allocate collision memory if needed
    if (collision_enabled_ && !d_collisions_) {
        cudaError_t err = cudaMalloc(&d_collisions_, max_collisions_ * sizeof(Collision));
        checkCudaError(err, "Failed to allocate collision memory");
    }
    
    // Free collision memory if disabled
    if (!collision_enabled_ && d_collisions_) {
        freeCollisionMemory();
    }
}

void ReboundCudaSimulation::setCollisionResolution(CollisionResolution method) {
    collision_detector_.setResolutionMethod(method);
}

void ReboundCudaSimulation::setCoefficientOfRestitution(float epsilon) {
    collision_detector_.setCoefficientOfRestitution(epsilon);
}

int ReboundCudaSimulation::detectAndResolveCollisions(float dt, float current_time) {
    if (!collision_enabled_ || !d_collisions_) {
        return 0;
    }
    
    // Detect collisions
    int n_collisions = collision_detector_.detectCollisions(
        d_particles_, particle_count_, dt, d_collisions_, max_collisions_);
    
    if (n_collisions > 0) {
        printf("Detected %d collisions at time %f\n", n_collisions, current_time);
        
        // Resolve collisions
        int halt_status = collision_detector_.resolveCollisions(
            d_particles_, d_collisions_, n_collisions, current_time);
        
        if (halt_status == 1) {
            printf("Simulation halted due to collision\n");
            return 1; // Signal to halt simulation
        }
    }
    
    return 0; // Continue simulation
}

// The integratorPart1/Part2 member functions are now obsolete – superseded by
// the standalone Integrator hierarchy. They are kept disabled to avoid
// symbol collisions while preserving history for reference.
#if 0
// ... old integratorPart1 and integratorPart2 implementations ...
#endif 