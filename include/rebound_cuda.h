#ifndef REBOUND_CUDA_H
#define REBOUND_CUDA_H

#include "rebound_types.h"
#include <vector>

// All types are now defined in rebound_types.h

// REBOUND CUDA simulation class
class ReboundCudaSimulation {
private:
    SimulationConfig config;
    Particle* h_particles;      // Host particles
    Particle* d_particles;      // Device particles
    TreeNode* h_tree_nodes;     // Host tree nodes
    TreeNode* d_tree_nodes;     // Device tree nodes
    bool particles_allocated;
    bool tree_allocated;
    bool device_particles_current; // Flag to track if device particles are up to date
    int max_tree_nodes;
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
    void allocateTreeMemory();
    void findBoundingBox(double& x_min, double& x_max, double& y_min, double& y_max, 
                        double& z_min, double& z_max);
    
    // Utility functions
    void copyParticlesToDevice();
    void copyParticlesFromDevice();
    void copyTreeToDevice();
    void printParticles();
    double getTotalEnergy();
    
    // Getter functions
    int getNumParticles() const { return config.n_particles; }
    double getCurrentTime() const { return config.t; }
    const Particle* getParticles() const { return h_particles; }
    GravityMode getGravityMode() const { return config.gravity_mode; }
};

// CUDA kernels for different gravity modes
__global__ void computeForcesBasicKernel(Particle* particles, int n, double G, double softening);
__global__ void computeForcesCompensatedKernel(Particle* particles, int n, double G, double softening);
__global__ void computeForcesTreeKernel(Particle* particles, TreeNode* tree_nodes, int n, 
                                       double G, double opening_angle, double softening);

// Tree construction kernels
__global__ void buildTreeKernel(Particle* particles, TreeNode* tree_nodes, int n,
                               double x_min, double x_max, double y_min, double y_max,
                               double z_min, double z_max);
__global__ void computeCenterOfMassKernel(TreeNode* tree_nodes, int n_nodes);

// Integration kernels
__global__ void updatePositionsKernel(Particle* particles, int n, double dt);
__global__ void updateVelocitiesKernel(Particle* particles, int n, double dt);

// Utility functions (defined in rebound_utils.cu)
void checkCudaError(cudaError_t error, const char* msg);
void printDeviceInfo();

#endif // REBOUND_CUDA_H 