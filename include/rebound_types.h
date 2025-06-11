#ifndef REBOUND_TYPES_H
#define REBOUND_TYPES_H

#include <cuda_runtime.h>

// Gravity calculation modes (similar to REBOUND)
enum GravityMode {
    GRAVITY_NONE = 0,           // No gravity calculation
    GRAVITY_BASIC = 1,          // Direct summation O(N²) - our current implementation
    GRAVITY_COMPENSATED = 2,    // Direct summation with compensated arithmetic
    GRAVITY_TREE = 3            // Barnes-Hut tree algorithm O(N log N)
};

// Structure to represent a particle in the N-body simulation
struct Particle {
    double x, y, z;     // Position
    double vx, vy, vz;  // Velocity
    double ax, ay, az;  // Acceleration
    double m;           // Mass
    double r;           // Radius (for collision detection)
};

// Tree node structure for Barnes-Hut algorithm
struct TreeNode {
    // Spatial bounds
    double x_min, x_max, y_min, y_max, z_min, z_max;
    
    // Center of mass and total mass
    double com_x, com_y, com_z;  // Center of mass
    double total_mass;           // Total mass in this node
    
    // Tree structure
    int children[8];             // Indices of 8 children (oct-tree), -1 if no child
    int particle_index;          // Index of particle if leaf node, -1 if internal node
    int parent;                  // Index of parent node, -1 if root
    
    // Tree parameters
    bool is_leaf;                // True if this is a leaf node
    int depth;                   // Depth in the tree
};

// Structure to hold simulation parameters
struct SimulationConfig {
    int n_particles;
    double dt;                  // Time step
    double t;                   // Current time
    double G;                   // Gravitational constant
    int max_iterations;
    bool collision_detection;
    
    // Tree parameters
    GravityMode gravity_mode;   // Gravity calculation mode
    double opening_angle;       // θ parameter for Barnes-Hut (typically 0.5-1.0)
    double softening;           // Gravitational softening parameter
    int max_tree_depth;         // Maximum tree depth
};

#endif // REBOUND_TYPES_H 