#include "rebound_gravity.h"
#include <cmath>

// CUDA kernel for basic gravity calculation (direct summation)
__global__ void computeForcesBasicKernel(Particle* particles, int n, double G, double softening) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        
        // Compute force from all other particles
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                
                double r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                double r = sqrt(r2);
                
                // Avoid division by zero
                if (r > 1e-15) {
                    double force_magnitude = G * particles[i].m * particles[j].m / r2;
                    double force_unit = force_magnitude / r;
                    
                    fx += force_unit * dx;
                    fy += force_unit * dy;
                    fz += force_unit * dz;
                }
            }
        }
        
        // Update accelerations (F = ma, so a = F/m)
        particles[i].ax = fx / particles[i].m;
        particles[i].ay = fy / particles[i].m;
        particles[i].az = fz / particles[i].m;
    }
}

// CUDA kernel for compensated gravity calculation (Kahan summation for better precision)
__global__ void computeForcesCompensatedKernel(Particle* particles, int n, double G, double softening) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double cfx = 0.0, cfy = 0.0, cfz = 0.0;  // Compensation terms for Kahan summation
        
        // Compute force from all other particles using compensated summation
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                
                double r2 = dx*dx + dy*dy + dz*dz + softening*softening;
                double r = sqrt(r2);
                
                if (r > 1e-15) {
                    double force_magnitude = G * particles[i].m * particles[j].m / r2;
                    double force_unit = force_magnitude / r;
                    
                    // Kahan summation for better numerical precision
                    double tfx = force_unit * dx - cfx;
                    double tfy = force_unit * dy - cfy;
                    double tfz = force_unit * dz - cfz;
                    
                    double yfx = fx + tfx;
                    double yfy = fy + tfy;
                    double yfz = fz + tfz;
                    
                    cfx = (yfx - fx) - tfx;
                    cfy = (yfy - fy) - tfy;
                    cfz = (yfz - fz) - tfz;
                    
                    fx = yfx;
                    fy = yfy;
                    fz = yfz;
                }
            }
        }
        
        // Update accelerations
        particles[i].ax = fx / particles[i].m;
        particles[i].ay = fy / particles[i].m;
        particles[i].az = fz / particles[i].m;
    }
}

// Device function to calculate distance squared between particle and node center of mass
__device__ double distance_squared(const Particle& p, const TreeNode& node) {
    double dx = node.com_x - p.x;
    double dy = node.com_y - p.y;
    double dz = node.com_z - p.z;
    return dx*dx + dy*dy + dz*dz;
}

// Device function to calculate node size (max dimension)
__device__ double node_size(const TreeNode& node) {
    double dx = node.x_max - node.x_min;
    double dy = node.y_max - node.y_min;
    double dz = node.z_max - node.z_min;
    return fmax(dx, fmax(dy, dz));
}

// CUDA kernel for tree-based gravity calculation (Barnes-Hut)
__global__ void computeForcesTreeKernel(Particle* particles, TreeNode* tree_nodes, int n,
                                       double G, double opening_angle, double softening) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        
        // Stack for tree traversal (using local array, limit depth)
        int stack[64];  // Maximum tree depth of 64
        int stack_size = 0;
        
        // Start traversal from root node (index 0)
        if (tree_nodes[0].total_mass > 0) {
            stack[stack_size++] = 0;
        }
        
        while (stack_size > 0) {
            // Pop node from stack
            int node_idx = stack[--stack_size];
            TreeNode& node = tree_nodes[node_idx];
            
            if (node.total_mass <= 0) continue;
            
            // Skip if this node contains the current particle
            if (node.is_leaf && node.particle_index == i) continue;
            
            double r2 = distance_squared(particles[i], node);
            double r = sqrt(r2 + softening*softening);
            
            if (r < 1e-15) continue;
            
            // Barnes-Hut opening criterion: s/d < θ
            double s = node_size(node);
            bool use_node = (node.is_leaf) || (s / r < opening_angle);
            
            if (use_node) {
                // Use this node's center of mass
                double dx = node.com_x - particles[i].x;
                double dy = node.com_y - particles[i].y;
                double dz = node.com_z - particles[i].z;
                
                double force_magnitude = G * particles[i].m * node.total_mass / (r2 + softening*softening);
                double force_unit = force_magnitude / r;
                
                fx += force_unit * dx;
                fy += force_unit * dy;
                fz += force_unit * dz;
            } else {
                // Add children to stack for further traversal
                for (int c = 0; c < 8; c++) {
                    if (node.children[c] != -1 && stack_size < 63) {
                        stack[stack_size++] = node.children[c];
                    }
                }
            }
        }
        
        // Update accelerations
        particles[i].ax = fx / particles[i].m;
        particles[i].ay = fy / particles[i].m;
        particles[i].az = fz / particles[i].m;
    }
} 