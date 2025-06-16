#ifndef REBOUND_SIMULATION_H
#define REBOUND_SIMULATION_H

#include "rebound_types.h"
#include "rebound_tree.h"
#include "rebound_collision.h"

// REBOUND CUDA simulation class
class ReboundCudaSimulation {
private:
    SimulationConfig config_;
    Particle* h_particles_;      // Host particles
    Particle* d_particles_;      // Device particles
    OctTree oct_tree_;           // Tree for Barnes-Hut algorithm
    bool particles_allocated_;
    bool device_particles_current_; // Flag to track if device particles are up to date
    int particle_count_;         // Current number of particles added
    
    // Collision system
    CollisionDetector collision_detector_;
    Collision* d_collisions_;
    int max_collisions_;
    bool collision_enabled_;
    
    // Helper methods
    void freeCollisionMemory();
    int detectAndResolveCollisions(float dt, float current_time);
    
public:
    ReboundCudaSimulation();
    ~ReboundCudaSimulation();
    
    // Initialization functions
    void initializeSimulation(int n_particles, double dt = 0.01, double G = 1.0);
    void addParticle(double m, double x, double y, double z, 
                    double vx = 0.0, double vy = 0.0, double vz = 0.0, double r = 0.0);
    
    // Gravity mode configuration
    void setGravityMode(GravityMode mode);
    void setTreeParameters(double opening_angle = 0.5, double softening = 0.0);
    
    // Simulation functions
    void integrate(double t_end);
    void step();
    void computeForces();
    void updatePositions();
    
    // Integrator functions (DKD pattern)
    void integratorPart1();  // First drift part
    void integratorPart2();  // Kick + second drift part
    
    // Tree-specific functions
    void buildTree();
    
    // Utility functions
    void copyParticlesToDevice();
    void copyParticlesFromDevice();
    void printParticles();
    double getTotalEnergy();
    
    // Getter functions
    int getNumParticles() const { return config_.n_particles; }
    double getCurrentTime() const { return config_.t; }
    const Particle* getParticles() const { return h_particles_; }
    GravityMode getGravityMode() const { return config_.gravity_mode; }
    
    // Collision system methods
    void setCollisionDetection(CollisionDetection method);
    void setCollisionResolution(CollisionResolution method);
    void setCoefficientOfRestitution(float epsilon);
};

#endif // REBOUND_SIMULATION_H 