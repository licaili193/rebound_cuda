#ifndef REBOUND_SIMULATION_H
#define REBOUND_SIMULATION_H

#include "rebound_types.h"
#include "rebound_collision.h"
#include "rebound_tree.h"
#include <cuda_runtime.h>
#include <vector>

// REBOUND CUDA simulation class
class ReboundCudaSimulation {
private:
    // Configuration
    SimulationConfig config_;
    
    // Particle data
    Particle* h_particles_;
    Particle* d_particles_;
    bool particles_allocated_;
    bool device_particles_current_;
    int particle_count_;
    
    // Tree structure for hierarchical force calculation
    OctTree oct_tree_;
    
    // Collision detection and resolution
    CollisionDetector collision_detector_;
    bool collision_enabled_;
    Collision* d_collisions_;
    int max_collisions_;
    
    // Observer pattern for optional streaming/monitoring
    std::vector<SimulationObserver*> observers_;
    
    // Helper functions
    void copyParticlesToDevice();
    void copyParticlesFromDevice();
    void buildTree();
    void freeCollisionMemory();
    
    // Observer notification methods
    void notifySimulationStart();
    void notifySimulationStep(int step);
    void notifySimulationEnd(int total_steps);
    void notifyCollisionDetected(int particle1, int particle2);
    
public:
    ReboundCudaSimulation();
    ~ReboundCudaSimulation();
    
    // Initialization
    void initializeSimulation(int n_particles, double dt, double G);
    
    // Particle management
    void addParticle(double m, double x, double y, double z, double vx, double vy, double vz, double r = 0.0);
    
    // Configuration
    void setGravityMode(GravityMode mode);
    void setTreeParameters(double opening_angle, double softening);
    
    // Collision system
    void setCollisionDetection(CollisionDetection method);
    void setCollisionResolution(CollisionResolution method);
    void setCoefficientOfRestitution(float epsilon);
    
    // Observer pattern for optional streaming/monitoring
    void addObserver(SimulationObserver* observer);
    void removeObserver(SimulationObserver* observer);
    
    // Simulation execution (completely independent of streaming)
    bool step();  // returns true if simulation should continue, false if halted
    void integrate(double t_end);
    
    // Legacy methods (for compatibility - efficient access without copying)
    void printParticles();                             // Triggers device copy only when called
    double getTotalEnergy();                           // Triggers device copy only when called
    
    // Explicit synchronization utility
    void copyParticlesToHost();                       // Manually copy device → host
    
    // Getters for direct access (avoid copying)
    const Particle* getParticles() const { return h_particles_; }
    Particle* getDeviceParticles() const { return d_particles_; }  // For streaming systems
    int getNumParticles() const { return particle_count_; }
    double getCurrentTime() const { return config_.t; }
    double getTimeStep() const { return config_.dt; }
    
private:
    // Integration methods
    void integratorPart1();                          // First drift (D)
    void integratorPart2();                          // Kick + second drift (KD)
    void computeForces();
    void updatePositions();
    int detectAndResolveCollisions(float dt, float current_time);
};

#endif // REBOUND_SIMULATION_H 