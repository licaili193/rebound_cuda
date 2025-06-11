#include "../include/rebound_cuda.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void runGravityModeComparison() {
    std::cout << "\n=== GRAVITY MODE COMPARISON ===" << std::endl;
    
    // Test system: 100 particles in a cluster
    int n_particles = 100;
    double dt = 0.001;
    double G = 1.0;
    double t_end = 0.1;  // Short integration for comparison
    
    GravityMode modes[] = {GRAVITY_BASIC, GRAVITY_COMPENSATED, GRAVITY_TREE};
    const char* mode_names[] = {"BASIC", "COMPENSATED", "TREE"};
    
    for (int mode_idx = 0; mode_idx < 3; mode_idx++) {
        std::cout << "\n--- Testing " << mode_names[mode_idx] << " gravity mode ---" << std::endl;
        
        ReboundCudaSimulation sim;
        sim.initializeSimulation(n_particles, dt, G);
        sim.setGravityMode(modes[mode_idx]);
        
        if (modes[mode_idx] == GRAVITY_TREE) {
            sim.setTreeParameters(0.5, 1e-4);  // Opening angle = 0.5, softening = 1e-4
        }
        
        // Create random particle cluster
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<> pos_dist(-1.0, 1.0);
        std::uniform_real_distribution<> vel_dist(-0.1, 0.1);
        std::uniform_real_distribution<> mass_dist(0.001, 0.01);
        
        // Add central massive particle
        sim.addParticle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1);
        
        // Add cluster particles
        for (int i = 1; i < n_particles; i++) {
            double x = pos_dist(gen);
            double y = pos_dist(gen);
            double z = pos_dist(gen);
            double vx = vel_dist(gen);
            double vy = vel_dist(gen);
            double vz = vel_dist(gen);
            double m = mass_dist(gen);
            
            sim.addParticle(m, x, y, z, vx, vy, vz, 0.01);
        }
        
        double initial_energy = sim.getTotalEnergy();
        std::cout << "Initial energy: " << initial_energy << std::endl;
        
        // Time the integration
        auto start = std::chrono::high_resolution_clock::now();
        
        sim.integrate(t_end);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double final_energy = sim.getTotalEnergy();
        double energy_error = std::abs((final_energy - initial_energy) / initial_energy) * 100.0;
        
        std::cout << "Final energy: " << final_energy << std::endl;
        std::cout << "Energy error: " << energy_error << "%" << std::endl;
        std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final time: " << sim.getCurrentTime() << std::endl;
    }
} 