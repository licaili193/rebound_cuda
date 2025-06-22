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

// Collision detection methods
enum CollisionDetection {
    COLLISION_NONE = 0,
    COLLISION_DIRECT = 1,
    COLLISION_TREE = 2
};

// Collision resolution methods  
enum CollisionResolution {
    COLLISION_RESOLVE_HALT = 0,
    COLLISION_RESOLVE_HARDSPHERE = 1,
    COLLISION_RESOLVE_MERGE = 2
};

// Collision information structure
struct Collision {
    int p1;          // Index of first particle
    int p2;          // Index of second particle
    float time;      // Time when collision occurred
    float distance;  // Distance between particles at collision
};

// Data streaming modes for different use cases
enum StreamingMode {
    STREAM_NONE = 0,        // No streaming, data stays on GPU (for end-of-sim or GPU visualization)
    STREAM_PERIODIC = 1,    // Stream at fixed intervals (configurable)
    STREAM_CONTINUOUS = 2,  // Stream every step (high bandwidth, real-time)
    STREAM_ON_DEMAND = 3    // Stream only when explicitly requested
};

// GPU logging levels
enum LogLevel {
    LOG_NONE = 0,
    LOG_ERROR = 1,
    LOG_WARNING = 2,
    LOG_INFO = 3,
    LOG_DEBUG = 4
};

// Data buffer configuration
struct BufferConfig {
    int max_frames = 1000;         // Maximum number of simulation frames to buffer (sensible default)
    int stream_interval = 1;       // For STREAM_PERIODIC mode: stream every N steps (default every step)
    bool enable_gpu_logging = false; // Disable GPU-side logging by default
    LogLevel log_level = LOG_INFO;   // Default minimum log level
    bool async_streaming = false;  // Disable asynchronous host-side processing by default
    int worker_threads = 2;        // Default number of worker threads if async_streaming is enabled
};

// Simulation frame data (what gets streamed/buffered)
struct SimulationFrame {
    double time;            // Simulation time
    int step;              // Step number
    int n_particles;       // Number of active particles
    Particle* particles;   // Particle data (device or host pointer)
    double total_energy;   // Energy at this frame
    int n_collisions;      // Number of collisions in this step
};

// GPU log entry
struct GPULogEntry {
    double time;           // Simulation time when logged
    int step;             // Simulation step
    LogLevel level;       // Log level
    int thread_id;        // GPU thread ID
    char message[256];    // Log message
};

// Observer pattern for simulation events (decoupled streaming)
class SimulationObserver {
public:
    virtual ~SimulationObserver() = default;
    virtual void onSimulationStep(double time, int step, int n_particles, double energy = 0.0, int n_collisions = 0) = 0;
    virtual void onSimulationStart(int n_particles) {}
    virtual void onSimulationEnd(double final_time, int total_steps) {}
    virtual void onCollisionDetected(int particle1, int particle2, double time) {}
};

// Callback function types for async processing
typedef void (*FrameCallback)(const SimulationFrame* frame, void* user_data);
typedef void (*LogCallback)(const GPULogEntry* log_entry, void* user_data);

#endif // REBOUND_TYPES_H 