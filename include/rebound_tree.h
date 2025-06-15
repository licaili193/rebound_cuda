#ifndef REBOUND_TREE_H
#define REBOUND_TREE_H

#include "rebound_types.h"
#include <cuda_runtime.h>

// OctTree class for Barnes-Hut algorithm
class OctTree {
private:
    TreeNode* h_tree_nodes;     // Host tree nodes
    TreeNode* d_tree_nodes;     // Device tree nodes
    bool tree_allocated;
    int max_tree_nodes;
    int max_tree_depth;

    // Private utility methods
    int getChildIndex(double px, double py, double pz, double cx, double cy, double cz);
    int buildTreeRecursive(TreeNode* nodes, Particle* particles, int* particle_indices, 
                          int n_particles, int node_index, int& next_free_node,
                          double x_min, double x_max, double y_min, double y_max,
                          double z_min, double z_max, int depth, int max_depth);
    void findBoundingBox(Particle* particles, int n_particles,
                        double& x_min, double& x_max, double& y_min, double& y_max, 
                        double& z_min, double& z_max);

public:
    OctTree(int max_depth = 20);
    ~OctTree();

    // Tree management
    void allocateMemory(int n_particles);
    void buildTree(Particle* particles, int n_particles);
    void copyToDevice();
    void cleanup();

    // Getters
    TreeNode* getDeviceNodes() const { return d_tree_nodes; }
    TreeNode* getHostNodes() const { return h_tree_nodes; }
    bool isAllocated() const { return tree_allocated; }
    int getMaxNodes() const { return max_tree_nodes; }
};

#endif // REBOUND_TREE_H 