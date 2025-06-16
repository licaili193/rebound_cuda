#ifndef REBOUND_COLLISION_H
#define REBOUND_COLLISION_H

#include "rebound_types.h"

class CollisionDetector {
public:
    CollisionDetector();
    ~CollisionDetector();
    
    // Set collision detection method
    void setDetectionMethod(CollisionDetection method);
    void setResolutionMethod(CollisionResolution method);
    void setCoefficientOfRestitution(float epsilon);
    
    // Main collision detection function
    int detectCollisions(Particle* particles, int n_particles, float dt, 
                        Collision* collisions, int max_collisions);
    
    // Collision resolution functions
    int resolveCollisions(Particle* particles, Collision* collisions, 
                         int n_collisions, float current_time);
    
    // Individual resolution methods
    bool resolveHalt(Particle* particles, const Collision& collision);
    bool resolveHardSphere(Particle* particles, const Collision& collision);
    bool resolveMerge(Particle* particles, const Collision& collision);
    
private:
    CollisionDetection detection_method_;
    CollisionResolution resolution_method_;
    float coefficient_of_restitution_;
    
    // Detection methods
    int detectDirect(Particle* particles, int n_particles, float dt,
                    Collision* collisions, int max_collisions);
    int detectTree(Particle* particles, int n_particles, float dt,
                  Collision* collisions, int max_collisions);
    
    // Helper functions
    bool checkCollision(const Particle& p1, const Particle& p2, float dt);
    float getDistance(const Particle& p1, const Particle& p2);
};

// CUDA kernel declarations
__global__ void detectCollisionsDirectKernel(Particle* particles, int n_particles,
                                            Collision* collisions, int* collision_count,
                                            int max_collisions, float dt);

__global__ void resolveCollisionsKernel(Particle* particles, Collision* collisions,
                                       int n_collisions, float coefficient_of_restitution,
                                       CollisionResolution method);

#endif // REBOUND_COLLISION_H 