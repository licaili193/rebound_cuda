#include "../include/rebound_cuda.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void runPlanetarySystemExample() {
    std::cout << "\n=== PLANETARY SYSTEM EXAMPLE ===" << std::endl;
    
    // Create a REBOUND CUDA simulation
    ReboundCudaSimulation sim;
    
    // Initialize simulation with 3 particles (Sun-Earth-Moon system)
    sim.initializeSimulation(3, 0.001, 1.0);  // 3 particles, dt=0.001, G=1.0
    
    // Add the Sun (at origin, stationary)
    sim.addParticle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1);
    
    // Add Earth (at distance 1, with circular velocity)
    double earth_dist = 1.0;
    double earth_vel = sqrt(1.0 / earth_dist);  // Circular velocity for G*M=1, r=1
    sim.addParticle(3e-6, earth_dist, 0.0, 0.0, 0.0, earth_vel, 0.0, 0.01);
    
    // Add Moon (relative to Earth)
    double moon_dist = 0.05;
    double moon_vel = sqrt(1.0 / moon_dist) * 0.1;  // Slower orbit around Earth-Sun system
    sim.addParticle(4e-8, earth_dist + moon_dist, 0.0, 0.0, 0.0, earth_vel + moon_vel, 0.0, 0.001);
    
    std::cout << "Initial system:" << std::endl;
    sim.printParticles();
    
    double initial_energy = sim.getTotalEnergy();
    std::cout << "Initial total energy: " << initial_energy << std::endl;
    
    // Integrate for one orbital period (2π for unit system)
    double t_end = 2.0 * M_PI;
    std::cout << "\nIntegrating for time = " << t_end << " (one orbital period)..." << std::endl;
    
    sim.integrate(t_end);
    
    std::cout << "\nFinal system:" << std::endl;
    sim.printParticles();
    
    double final_energy = sim.getTotalEnergy();
    std::cout << "Final total energy: " << final_energy << std::endl;
    
    double energy_error = std::abs((final_energy - initial_energy) / initial_energy) * 100.0;
    std::cout << "Energy conservation error: " << energy_error << "%" << std::endl;
    
    std::cout << "Integration completed successfully!" << std::endl;
} 