#include "rebound_tree.h"
#include "rebound_utils.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// OctTree class implementation

OctTree::OctTree(int max_depth) {
    h_tree_nodes = nullptr;
    d_tree_nodes = nullptr;
    tree_allocated = false;
    max_tree_nodes = 0;
    max_tree_depth = max_depth;
}

OctTree::~OctTree() {
    cleanup();
}

void OctTree::cleanup() {
    // Free host memory
    if (h_tree_nodes) {
        free(h_tree_nodes);
        h_tree_nodes = nullptr;
    }
    
    // Free device memory
    if (d_tree_nodes) {
        cudaFree(d_tree_nodes);
        d_tree_nodes = nullptr;
    }
    
    tree_allocated = false;
    max_tree_nodes = 0;
}

void OctTree::findBoundingBox(Particle* particles, int n_particles,
                              double& x_min, double& x_max, double& y_min, double& y_max, 
                              double& z_min, double& z_max) {
    if (n_particles == 0) return;
    
    x_min = x_max = particles[0].x;
    y_min = y_max = particles[0].y;
    z_min = z_max = particles[0].z;
    
    for (int i = 1; i < n_particles; i++) {
        x_min = std::min(x_min, particles[i].x);
        x_max = std::max(x_max, particles[i].x);
        y_min = std::min(y_min, particles[i].y);
        y_max = std::max(y_max, particles[i].y);
        z_min = std::min(z_min, particles[i].z);
        z_max = std::max(z_max, particles[i].z);
    }
    
    // Add small margin to avoid edge cases
    double margin = 1e-10;
    x_min -= margin; x_max += margin;
    y_min -= margin; y_max += margin;
    z_min -= margin; z_max += margin;
}

void OctTree::allocateMemory(int n_particles) {
    if (tree_allocated) {
        cleanup();  // Clean up previous allocation
    }
    
    // Estimate maximum number of tree nodes (conservative upper bound)
    max_tree_nodes = 8 * n_particles;  // Oct-tree can have at most 8*N nodes
    
    // Allocate host memory for tree nodes
    h_tree_nodes = (TreeNode*)malloc(max_tree_nodes * sizeof(TreeNode));
    if (!h_tree_nodes) {
        std::cerr << "Failed to allocate host memory for tree nodes" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    memset(h_tree_nodes, 0, max_tree_nodes * sizeof(TreeNode));
    
    // Initialize all nodes
    for (int i = 0; i < max_tree_nodes; i++) {
        for (int j = 0; j < 8; j++) {
            h_tree_nodes[i].children[j] = -1;
        }
        h_tree_nodes[i].particle_index = -1;
        h_tree_nodes[i].parent = -1;
        h_tree_nodes[i].is_leaf = true;
        h_tree_nodes[i].total_mass = 0.0;
        h_tree_nodes[i].depth = 0;
    }
    
    // Allocate device memory for tree nodes
    cudaError_t err = cudaMalloc((void**)&d_tree_nodes, max_tree_nodes * sizeof(TreeNode));
    checkCudaError(err, "Failed to allocate device memory for tree nodes");
    
    tree_allocated = true;
}

int OctTree::getChildIndex(double px, double py, double pz, 
                          double cx, double cy, double cz) {
    int index = 0;
    if (px >= cx) index |= 1;  // x bit
    if (py >= cy) index |= 2;  // y bit
    if (pz >= cz) index |= 4;  // z bit
    return index;
}

int OctTree::buildTreeRecursive(TreeNode* nodes, Particle* particles, int* particle_indices, 
                               int n_particles, int node_index, int& next_free_node,
                               double x_min, double x_max, double y_min, double y_max,
                               double z_min, double z_max, int depth, int max_depth) {
    
    if (depth > max_depth || next_free_node >= max_tree_nodes) {
        return -1;  // Tree too deep or out of memory
    }
    
    TreeNode& node = nodes[node_index];
    node.x_min = x_min; node.x_max = x_max;
    node.y_min = y_min; node.y_max = y_max;
    node.z_min = z_min; node.z_max = z_max;
    node.depth = depth;
    
    // Calculate center of mass and total mass
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    
    for (int i = 0; i < n_particles; i++) {
        int p_idx = particle_indices[i];
        total_mass += particles[p_idx].m;
        com_x += particles[p_idx].m * particles[p_idx].x;
        com_y += particles[p_idx].m * particles[p_idx].y;
        com_z += particles[p_idx].m * particles[p_idx].z;
    }
    
    if (total_mass > 0) {
        com_x /= total_mass;
        com_y /= total_mass;
        com_z /= total_mass;
    }
    
    node.total_mass = total_mass;
    node.com_x = com_x;
    node.com_y = com_y;
    node.com_z = com_z;
    
    // If only one particle, this is a leaf
    if (n_particles <= 1) {
        node.is_leaf = true;
        node.particle_index = (n_particles == 1) ? particle_indices[0] : -1;
        return node_index;
    }
    
    // Split particles into 8 octants
    node.is_leaf = false;
    
    double cx = (x_min + x_max) * 0.5;
    double cy = (y_min + y_max) * 0.5;
    double cz = (z_min + z_max) * 0.5;
    
    // Count particles in each octant
    int octant_counts[8] = {0};
    int* octant_particles[8];
    
    // Allocate arrays for each octant
    for (int oct = 0; oct < 8; oct++) {
        octant_particles[oct] = (int*)malloc(n_particles * sizeof(int));
    }
    
    // Distribute particles to octants
    for (int i = 0; i < n_particles; i++) {
        int p_idx = particle_indices[i];
        int oct = getChildIndex(particles[p_idx].x, particles[p_idx].y, particles[p_idx].z,
                               cx, cy, cz);
        octant_particles[oct][octant_counts[oct]++] = p_idx;
    }
    
    // Create child nodes for non-empty octants
    for (int oct = 0; oct < 8; oct++) {
        if (octant_counts[oct] > 0) {
            if (next_free_node >= max_tree_nodes) break;
            
            int child_index = next_free_node++;
            node.children[oct] = child_index;
            nodes[child_index].parent = node_index;
            
            // Calculate child bounds
            double child_x_min = (oct & 1) ? cx : x_min;
            double child_x_max = (oct & 1) ? x_max : cx;
            double child_y_min = (oct & 2) ? cy : y_min;
            double child_y_max = (oct & 2) ? y_max : cy;
            double child_z_min = (oct & 4) ? cz : z_min;
            double child_z_max = (oct & 4) ? z_max : cz;
            
            buildTreeRecursive(nodes, particles, octant_particles[oct], octant_counts[oct],
                             child_index, next_free_node,
                             child_x_min, child_x_max,
                             child_y_min, child_y_max,
                             child_z_min, child_z_max,
                             depth + 1, max_depth);
        } else {
            node.children[oct] = -1;
        }
    }
    
    // Clean up octant arrays
    for (int oct = 0; oct < 8; oct++) {
        free(octant_particles[oct]);
    }
    
    return node_index;
}

void OctTree::buildTree(Particle* particles, int n_particles) {
    if (!tree_allocated) {
        allocateMemory(n_particles);
    }
    
    if (n_particles == 0) return;
    
    // Find bounding box
    double x_min, x_max, y_min, y_max, z_min, z_max;
    findBoundingBox(particles, n_particles, x_min, x_max, y_min, y_max, z_min, z_max);
    
    // Create array of particle indices
    int* particle_indices = (int*)malloc(n_particles * sizeof(int));
    for (int i = 0; i < n_particles; i++) {
        particle_indices[i] = i;
    }
    
    // Reset tree nodes
    memset(h_tree_nodes, 0, max_tree_nodes * sizeof(TreeNode));
    for (int i = 0; i < max_tree_nodes; i++) {
        for (int j = 0; j < 8; j++) {
            h_tree_nodes[i].children[j] = -1;
        }
        h_tree_nodes[i].particle_index = -1;
        h_tree_nodes[i].parent = -1;
        h_tree_nodes[i].is_leaf = true;
        h_tree_nodes[i].total_mass = 0.0;
    }
    
    // Build tree recursively starting from root (index 0)
    int next_free_node = 1;
    buildTreeRecursive(h_tree_nodes, particles, particle_indices, n_particles,
                      0, next_free_node, x_min, x_max, y_min, y_max, z_min, z_max,
                      0, max_tree_depth);
    
    free(particle_indices);
}

void OctTree::copyToDevice() {
    if (!tree_allocated) return;
    
    cudaError_t err = cudaMemcpy(d_tree_nodes, h_tree_nodes, 
                                max_tree_nodes * sizeof(TreeNode), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy tree to device");
} 