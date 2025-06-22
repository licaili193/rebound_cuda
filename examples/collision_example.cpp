#include <iostream>
#include <cmath>
#include "../include/rebound_cuda.h"

int main() {
    std::cout << "REBOUND CUDA Collision Example" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Create simulation
    ReboundCudaSimulation sim;
    
    // Initialize simulation
    sim.initializeSimulation(3, 0.01, 1.0); // 3 particles, dt=0.01, G=1.0
    
    // Configure simulation
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Enable collision detection and resolution
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HARDSPHERE);
    sim.setCoefficientOfRestitution(0.8f); // Bouncy collisions
    
    // Scenario: Guaranteed collision - particles start very close with high speed
    std::cout << "\n=== GUARANTEED COLLISION SCENARIO ===" << std::endl;
    
    // Turn off gravity to focus on pure collision physics
    sim.setGravityMode(GRAVITY_NONE);
    
    // Two particles moving directly toward each other - much closer start
    // Distance = 1.0, combined radius = 0.8, so collision guaranteed
    std::cout << "Initial separation: 1.0 units" << std::endl;
    std::cout << "Combined radius: 0.8 units" << std::endl;
    std::cout << "Collision will occur when distance ≤ 0.8" << std::endl;
    
    // Particle 1: moving right with large radius
    sim.addParticle(1.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.4); // mass, x, y, z, vx, vy, vz, radius
    
    // Particle 2: moving left with large radius  
    sim.addParticle(1.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.4);
    
    // Particle 3: stationary observer (far away)
    sim.addParticle(0.1, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.1);
    
    std::cout << "Initial system:" << std::endl;
    sim.printParticles();
    
    double initial_energy = sim.getTotalEnergy();
    std::cout << "Initial energy: " << initial_energy << std::endl;
    
    // Integrate for a short time to see collision
    std::cout << "\nIntegrating system..." << std::endl;
    std::cout << "Expected collision time: ~0.2 time units" << std::endl;
    std::cout << "Calculation: (1.0 - 0.8) / (0.5 + 0.5) = 0.2 / 1.0 = 0.2" << std::endl;
    
    // Integrate step by step to see the collision happen
    for (int step = 0; step < 10; step++) {
        double target_time = (step + 1) * 0.02;  // Cumulative time: 0.02, 0.04, 0.06, etc.
        std::cout << "\n--- Time: " << target_time << " ---" << std::endl;
        sim.integrate(target_time);  // Integrate to cumulative time
        sim.printParticles();
        
        // Calculate distance between particles 0 and 1
        const Particle* particles = sim.getParticles();
        double dx = particles[1].x - particles[0].x;
        double dy = particles[1].y - particles[0].y;
        double dz = particles[1].z - particles[0].z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        std::cout << "Distance between particles 0 and 1: " << distance << std::endl;
        
        // Check if collision occurred by looking at particle count
        if (sim.getNumParticles() < 3) {
            std::cout << "*** COLLISION DETECTED! Particles merged! ***" << std::endl;
            break;
        }
        
        // Stop if particles have passed each other (collision missed)
        if (step > 3 && distance > 1.0) {
            std::cout << "*** Particles passed each other - collision system may not be working ***" << std::endl;
            break;
        }
    }
    
    std::cout << "\nFinal system:" << std::endl;
    sim.printParticles();
    
    double final_energy = sim.getTotalEnergy();
    std::cout << "Final energy: " << final_energy << std::endl;
    std::cout << "Energy change: " << (final_energy - initial_energy) << std::endl;
    
    return 0;
} 