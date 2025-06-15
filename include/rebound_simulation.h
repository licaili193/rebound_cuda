#ifndef REBOUND_SIMULATION_H
#define REBOUND_SIMULATION_H

#include "rebound_types.h"
#include "rebound_tree.h"

// REBOUND CUDA simulation class
class ReboundCudaSimulation {
private:
    SimulationConfig config;
    Particle* h_particles;      // Host particles
    Particle* d_particles;      // Device particles
    OctTree oct_tree;           // Tree for Barnes-Hut algorithm
    bool particles_allocated;
    bool device_particles_current; // Flag to track if device particles are up to date
    int particle_count;         // Current number of particles added
    
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
    
    // Tree-specific functions
    void buildTree();
    
    // Utility functions
    void copyParticlesToDevice();
    void copyParticlesFromDevice();
    void printParticles();
    double getTotalEnergy();
    
    // Getter functions
    int getNumParticles() const { return config.n_particles; }
    double getCurrentTime() const { return config.t; }
    const Particle* getParticles() const { return h_particles; }
    GravityMode getGravityMode() const { return config.gravity_mode; }
};

#endif // REBOUND_SIMULATION_H 