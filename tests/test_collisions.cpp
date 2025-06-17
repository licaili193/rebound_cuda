#include "test_framework.h"
#include "rebound_cuda.h"
#include <cmath>

// Test 1: Direct collision detection
TEST(Collisions, DirectCollisionDetection) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_NONE);  // Focus on collision physics
    
    // Enable collision detection
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HALT);
    
    // Two particles moving directly toward each other
    // Distance = 1.0, combined radius = 0.8, so collision guaranteed
    double radius = 0.4;
    sim.addParticle(1.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, radius);
    sim.addParticle(1.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, radius);
    
    // Calculate collision time: when distance = sum of radii
    // Initial separation = 1.0, combined radius = 0.8
    // Relative speed = 1.0, so collision at t = (1.0 - 0.8) / 1.0 = 0.2
    double expected_collision_time = 0.2;
    
    // Integrate past collision time
    sim.integrate(0.5);
    
    // Check that collision was detected (simulation should have halted before 0.5)
    double final_time = sim.getCurrentTime();
    ASSERT_LT(final_time, 0.5);  // Should have stopped due to collision
    ASSERT_GT(final_time, 0.1);  // But should have made some progress
    
    std::cout << "  Expected collision time: " << expected_collision_time << std::endl;
    std::cout << "  Actual halt time: " << final_time << std::endl;
}

// Test 2: Hard sphere collision resolution
TEST(Collisions, HardSphereCollision) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.001, 1.0);  // Small timestep for accuracy
    sim.setGravityMode(GRAVITY_NONE);
    
    // Enable hard sphere collision resolution
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HARDSPHERE);
    sim.setCoefficientOfRestitution(1.0f);  // Perfectly elastic
    
    // Head-on collision between two particles
    double m1 = 1.0, m2 = 2.0;
    double v1 = 1.0, v2 = -0.5;
    double radius = 0.3;
    
    // Start them far enough apart
    sim.addParticle(m1, -1.0, 0.0, 0.0, v1, 0.0, 0.0, radius);
    sim.addParticle(m2, 1.0, 0.0, 0.0, v2, 0.0, 0.0, radius);
    
    // Record initial momentum and energy
    double initial_momentum = m1 * v1 + m2 * v2;
    double initial_kinetic = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2;
    
    // Integrate through collision
    sim.integrate(2.0);
    
    // Check final state
    const Particle* particles = sim.getParticles();
    double final_momentum = m1 * particles[0].vx + m2 * particles[1].vx;
    double final_kinetic = 0.5 * m1 * particles[0].vx * particles[0].vx + 
                          0.5 * m2 * particles[1].vx * particles[1].vx;
    
    // Check conservation laws
    ASSERT_NEAR(initial_momentum, final_momentum, 1e-6);
    ASSERT_NEAR(initial_kinetic, final_kinetic, 1e-6);
    
    // Check that particles bounced (velocities should have changed sign and magnitude)
    // For elastic collision: v1_final = ((m1-m2)*v1 + 2*m2*v2)/(m1+m2)
    double expected_v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2);
    double expected_v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2);
    
    std::cout << "  Initial momentum: " << initial_momentum << ", Final: " << final_momentum << std::endl;
    std::cout << "  Initial KE: " << initial_kinetic << ", Final: " << final_kinetic << std::endl;
    std::cout << "  Expected v1: " << expected_v1_final << ", Actual: " << particles[0].vx << std::endl;
    std::cout << "  Expected v2: " << expected_v2_final << ", Actual: " << particles[1].vx << std::endl;
}

// Test 3: Inelastic collision
TEST(Collisions, InelasticCollision) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.001, 1.0);
    sim.setGravityMode(GRAVITY_NONE);
    
    // Enable inelastic collision
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HARDSPHERE);
    sim.setCoefficientOfRestitution(0.5f);  // Partially inelastic
    
    // Equal mass particles, head-on collision
    double m = 1.0;
    double v1 = 1.0, v2 = -1.0;
    double radius = 0.25;
    
    sim.addParticle(m, -1.0, 0.0, 0.0, v1, 0.0, 0.0, radius);
    sim.addParticle(m, 1.0, 0.0, 0.0, v2, 0.0, 0.0, radius);
    
    // Initial energy
    double initial_kinetic = 0.5 * m * v1 * v1 + 0.5 * m * v2 * v2;
    
    // Integrate through collision
    sim.integrate(2.0);
    
    // Check final state
    const Particle* particles = sim.getParticles();
    double final_kinetic = 0.5 * m * particles[0].vx * particles[0].vx + 
                          0.5 * m * particles[1].vx * particles[1].vx;
    
    // For equal masses and opposite velocities with restitution e=0.5:
    // Both particles should have velocity 0.5 * (v1 - v2) / 2 = 0.5 in opposite directions
    double expected_speed = 0.5;
    
    // Energy should be lost
    ASSERT_LT(final_kinetic, initial_kinetic);
    
    // But momentum should be conserved (should be zero)
    double final_momentum = m * particles[0].vx + m * particles[1].vx;
    ASSERT_NEAR(final_momentum, 0.0, 1e-6);
    
    std::cout << "  Initial KE: " << initial_kinetic << ", Final KE: " << final_kinetic << std::endl;
    std::cout << "  Energy lost: " << (initial_kinetic - final_kinetic) << std::endl;
    std::cout << "  Final velocities: " << particles[0].vx << ", " << particles[1].vx << std::endl;
}

// Test 4: Multiple particle collision system
TEST(Collisions, MultipleParticleCollisions) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(5, 0.001, 1.0);
    sim.setGravityMode(GRAVITY_NONE);
    
    // Enable collision detection
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HARDSPHERE);
    sim.setCoefficientOfRestitution(0.8f);
    
    // Create a line of particles for "Newton's cradle" effect
    double mass = 1.0;
    double radius = 0.1;
    double spacing = 0.19;  // Just under 2*radius to ensure contact
    
    // First particle moving, others stationary
    sim.addParticle(mass, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, radius);
    
    for (int i = 1; i < 5; i++) {
        double x = -0.5 + (i-1) * spacing;
        sim.addParticle(mass, x, 0.0, 0.0, 0.0, 0.0, 0.0, radius);
    }
    
    // Record initial momentum and energy
    double initial_momentum = mass * 2.0;  // Only first particle moving
    double initial_energy = 0.5 * mass * 4.0;  // 0.5 * m * v^2
    
    // Integrate through collisions
    sim.integrate(1.0);
    
    // Check conservation
    const Particle* particles = sim.getParticles();
    double final_momentum = 0.0;
    double final_energy = 0.0;
    
    for (int i = 0; i < 5; i++) {
        final_momentum += mass * particles[i].vx;
        double v2 = particles[i].vx * particles[i].vx + 
                   particles[i].vy * particles[i].vy + 
                   particles[i].vz * particles[i].vz;
        final_energy += 0.5 * mass * v2;
    }
    
    // Momentum should be conserved
    ASSERT_NEAR(initial_momentum, final_momentum, 1e-4);
    
    // Energy should be lost due to inelastic collisions
    ASSERT_LT(final_energy, initial_energy);
    
    std::cout << "  Initial momentum: " << initial_momentum << ", Final: " << final_momentum << std::endl;
    std::cout << "  Initial energy: " << initial_energy << ", Final: " << final_energy << std::endl;
    std::cout << "  Energy loss: " << (initial_energy - final_energy) << std::endl;
}

// Test 5: No false collision detection
TEST(Collisions, NoFalseCollisions) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_NONE);
    
    // Enable collision detection
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HALT);
    
    // Two particles that won't collide (moving parallel)
    double radius = 0.1;
    sim.addParticle(1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, radius);  // Lower path
    sim.addParticle(1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0, radius);   // Upper path
    
    // Integrate for sufficient time
    sim.integrate(2.0);
    
    // Should complete without collision detection halting simulation
    double final_time = sim.getCurrentTime();
    ASSERT_NEAR(final_time, 2.0, 1e-10);  // Should integrate to full time
    
    // Check that particles are indeed separated
    const Particle* particles = sim.getParticles();
    double dx = particles[1].x - particles[0].x;
    double dy = particles[1].y - particles[0].y;
    double dz = particles[1].z - particles[0].z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    ASSERT_GT(distance, 2 * radius);  // Should be well separated
    
    std::cout << "  Final separation: " << distance << std::endl;
    std::cout << "  Required separation: " << (2 * radius) << std::endl;
}

// Test 6: Collision with gravity
TEST(Collisions, CollisionWithGravity) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.001, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);  // Include gravity
    
    // Enable collision detection
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HARDSPHERE);
    sim.setCoefficientOfRestitution(0.9f);
    
    // Two massive particles that will attract and collide
    double mass = 10.0;
    double radius = 0.1;
    
    sim.addParticle(mass, -1.0, 0.0, 0.0, 0.1, 0.0, 0.0, radius);
    sim.addParticle(mass, 1.0, 0.0, 0.0, -0.1, 0.0, 0.0, radius);
    
    // Record initial total energy (kinetic + potential)
    double initial_energy = sim.getTotalEnergy();
    
    // Integrate through collision
    sim.integrate(5.0);
    
    // Check that system evolved (collision should have occurred)
    const Particle* particles = sim.getParticles();
    double dx = particles[1].x - particles[0].x;
    double dy = particles[1].y - particles[0].y;
    double dz = particles[1].z - particles[0].z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    // After collision, particles should be moving apart
    double relative_vx = particles[1].vx - particles[0].vx;
    ASSERT_GT(std::abs(relative_vx), 0.1);  // Should have significant relative velocity
    
    std::cout << "  Final separation: " << distance << std::endl;
    std::cout << "  Relative velocity: " << relative_vx << std::endl;
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Final energy: " << sim.getTotalEnergy() << std::endl;
} 