#define _USE_MATH_DEFINES  // For M_PI on Windows
#include "test_framework.h"
#include "rebound_cuda.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test 1: Two particle system with known analytical solution
TEST(Physics, TwoBodyCircularOrbit) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.001, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Set up circular orbit: two equal mass particles
    double m = 0.5;
    double r = 1.0;  // separation distance
    double v = sqrt(2 * m / r);  // circular orbital velocity for binary system
    
    // Particle 1: at (-r/2, 0, 0) with velocity (0, -v/2, 0)
    sim.addParticle(m, -r/2, 0.0, 0.0, 0.0, -v/2, 0.0, 0.01);
    
    // Particle 2: at (r/2, 0, 0) with velocity (0, v/2, 0)
    sim.addParticle(m, r/2, 0.0, 0.0, 0.0, v/2, 0.0, 0.01);
    
    // Record initial energy
    double initial_energy = sim.getTotalEnergy();
    
    // Integrate for one orbital period (approximate)
    double period = 2.0 * M_PI * sqrt(r * r * r / (2 * m));
    sim.integrate(period);
    
    // Check energy conservation (should be within 1% for this simple case)
    double final_energy = sim.getTotalEnergy();
    double energy_error = std::abs((final_energy - initial_energy) / initial_energy);
    
    ASSERT_LT(energy_error, 0.01);  // Energy should be conserved within 1%
    
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Final energy: " << final_energy << std::endl;
    std::cout << "  Energy error: " << (energy_error * 100) << "%" << std::endl;
}

// Test 2: Two particle head-on collision (no forces, just kinematics)
TEST(Physics, TwoBodyKinematics) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(2, 0.001, 1.0);
    sim.setGravityMode(GRAVITY_NONE);  // No gravity for pure kinematics
    
    // Two particles moving toward each other
    double m1 = 1.0, m2 = 2.0;
    double v1 = 1.0, v2 = -0.5;
    double x1_initial = -1.0, x2_initial = 1.0;
    
    sim.addParticle(m1, x1_initial, 0.0, 0.0, v1, 0.0, 0.0, 0.01);
    sim.addParticle(m2, x2_initial, 0.0, 0.0, v2, 0.0, 0.0, 0.01);
    
    // Integrate for 1 time unit
    double t = 1.0;
    sim.integrate(t);
    
    // Check positions analytically
    double expected_x1 = x1_initial + v1 * t;
    double expected_x2 = x2_initial + v2 * t;
    
    // Get final positions
    const Particle* particles = sim.getParticles();
    
    ASSERT_NEAR(expected_x1, particles[0].x, 1e-10);
    ASSERT_NEAR(expected_x2, particles[1].x, 1e-10);
    
    // Check momentum conservation
    double initial_momentum = m1 * v1 + m2 * v2;
    double final_momentum = m1 * particles[0].vx + m2 * particles[1].vx;
    
    ASSERT_NEAR(initial_momentum, final_momentum, 1e-10);
    
    std::cout << "  Expected x1: " << expected_x1 << ", Actual: " << particles[0].x << std::endl;
    std::cout << "  Expected x2: " << expected_x2 << ", Actual: " << particles[1].x << std::endl;
}

// Test 3: Energy conservation in 100-particle system
TEST(Physics, EnergyConservation100Particles) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(100, 0.001, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add particles in a roughly spherical distribution
    srand(12345);  // Fixed seed for reproducibility
    for (int i = 0; i < 100; i++) {
        double x = ((double)rand() / RAND_MAX - 0.5) * 4.0;
        double y = ((double)rand() / RAND_MAX - 0.5) * 4.0;
        double z = ((double)rand() / RAND_MAX - 0.5) * 4.0;
        
        // Small random velocities
        double vx = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        double vy = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        double vz = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        
        double mass = 0.01;
        sim.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
    }
    
    // Record initial energy
    double initial_energy = sim.getTotalEnergy();
    
    // Integrate for a short time
    sim.integrate(0.1);
    
    // Check energy conservation
    double final_energy = sim.getTotalEnergy();
    double energy_error = std::abs((final_energy - initial_energy) / initial_energy);
    
    // For a 100-body system, we expect some numerical drift but should be < 5%
    ASSERT_LT(energy_error, 0.05);
    
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Final energy: " << final_energy << std::endl;
    std::cout << "  Energy drift: " << (energy_error * 100) << "%" << std::endl;
}

// Test 4: Gravity modes comparison
TEST(Physics, GravityModesConsistency) {
    // Test that different gravity modes give similar results for small systems
    ReboundCudaSimulation sim_basic, sim_compensated;
    
    // Initialize both simulations identically
    sim_basic.initializeSimulation(3, 0.01, 1.0);
    sim_compensated.initializeSimulation(3, 0.01, 1.0);
    
    sim_basic.setGravityMode(GRAVITY_BASIC);
    sim_compensated.setGravityMode(GRAVITY_COMPENSATED);
    
    // Add same particles to both
    srand(54321);
    for (int i = 0; i < 3; i++) {
        double x = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double y = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double z = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double vx = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        double vy = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        double vz = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        double mass = 0.1;
        
        sim_basic.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
        sim_compensated.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
    }
    
    // Integrate both
    sim_basic.integrate(0.1);
    sim_compensated.integrate(0.1);
    
    // Compare final energies (should be very close)
    double energy_basic = sim_basic.getTotalEnergy();
    double energy_compensated = sim_compensated.getTotalEnergy();
    double energy_diff = std::abs(energy_basic - energy_compensated) / std::abs(energy_basic);
    
    // Should be within 0.1% for this simple case
    ASSERT_LT(energy_diff, 0.001);
    
    std::cout << "  Basic gravity energy: " << energy_basic << std::endl;
    std::cout << "  Compensated gravity energy: " << energy_compensated << std::endl;
    std::cout << "  Relative difference: " << (energy_diff * 100) << "%" << std::endl;
}

// Test 5: Time stepping accuracy
TEST(Physics, TimeSteppingAccuracy) {
    // Test that smaller timesteps give more accurate results
    ReboundCudaSimulation sim_large_dt, sim_small_dt;
    
    // Same system, different timesteps
    sim_large_dt.initializeSimulation(2, 0.01, 1.0);   // Large timestep
    sim_small_dt.initializeSimulation(2, 0.001, 1.0);  // Small timestep
    
    sim_large_dt.setGravityMode(GRAVITY_BASIC);
    sim_small_dt.setGravityMode(GRAVITY_BASIC);
    
    // Simple two-body problem
    // For two particles of mass m each, separated by distance r,
    // circular orbital velocity v = sqrt(G * (m1 + m2) / r) = sqrt(G * 2m / r)
    // With G = 1.0, m = 0.5: v = sqrt(1.0 * 2 * 0.5 / 1.0) = sqrt(1.0) = 1.0
    double m = 0.5, r = 1.0, v = sqrt(2 * m / r);
    
    // Add same particles to both simulations
    sim_large_dt.addParticle(m, -r/2, 0.0, 0.0, 0.0, -v/2, 0.0, 0.01);
    sim_large_dt.addParticle(m, r/2, 0.0, 0.0, 0.0, v/2, 0.0, 0.01);
    
    sim_small_dt.addParticle(m, -r/2, 0.0, 0.0, 0.0, -v/2, 0.0, 0.01);
    sim_small_dt.addParticle(m, r/2, 0.0, 0.0, 0.0, v/2, 0.0, 0.01);
    
    // Get initial energy from both simulations (should be the same)
    double initial_energy = sim_large_dt.getTotalEnergy();
    
    // Integrate to same final time
    double final_time = 0.1;
    sim_large_dt.integrate(final_time);
    sim_small_dt.integrate(final_time);
    
    // Get final energies
    double energy_large = sim_large_dt.getTotalEnergy();
    double energy_small = sim_small_dt.getTotalEnergy();
    
    double error_large = std::abs((energy_large - initial_energy) / initial_energy);
    double error_small = std::abs((energy_small - initial_energy) / initial_energy);
    
    // Smaller timestep should have smaller error
    ASSERT_LT(error_small, error_large);
    
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Large dt final energy: " << energy_large << std::endl;
    std::cout << "  Small dt final energy: " << energy_small << std::endl;
    std::cout << "  Large dt error: " << (error_large * 100) << "%" << std::endl;
    std::cout << "  Small dt error: " << (error_small * 100) << "%" << std::endl;
} 