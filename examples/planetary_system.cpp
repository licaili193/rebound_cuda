#include "../include/rebound_cuda.h"
#include <iostream>
#include <cmath>

// Physical constants (SI units)
constexpr double G_SI           = 6.67430e-11;        // m^3 kg^-1 s^-2
constexpr double M_SUN          = 1.98847e30;         // kg
constexpr double M_EARTH        = 5.97219e24;         // kg
constexpr double M_MOON         = 7.342e22;           // kg
constexpr double AU_IN_METERS   = 1.495978707e11;     // m
constexpr double R_SUN          = 6.9634e8;           // m
constexpr double R_EARTH        = 6.371e6;            // m
constexpr double R_MOON         = 1.737e6;            // m

// Derived quantities (constants that don't require runtime sqrt)
constexpr double EARTH_ORBIT_RADIUS = AU_IN_METERS;           // 1 AU
constexpr double MOON_ORBIT_RADIUS  = 3.844e8;                // m (avg)

void runPlanetarySystemExample() {
    std::cout << "\n=== PLANETARY SYSTEM EXAMPLE (REALISTIC UNITS) ===" << std::endl;

    // ------------------------------
    // Create and initialise simulation
    // ------------------------------
    ReboundCudaSimulation sim;

    const double dt_seconds = 3600.0;                 // 1-hour timestep
    const int    n_particles = 3;                     // Sun, Earth, Moon

    sim.initializeSimulation(n_particles, dt_seconds, G_SI);

    // ------------------------------
    // Compute orbital velocities
    // ------------------------------
    const double earth_orbit_vel    = std::sqrt(G_SI * M_SUN   / EARTH_ORBIT_RADIUS);  // ~29.8 km/s
    const double moon_orbit_vel_rel = std::sqrt(G_SI * M_EARTH / MOON_ORBIT_RADIUS);   // ~1.02 km/s

    // ------------------------------
    // Add bodies (positions on x-axis, velocities along +y)
    // ------------------------------

    // Sun (barycentre)
    sim.addParticle(M_SUN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R_SUN);

    // Earth (1 AU from Sun)
    sim.addParticle(M_EARTH, EARTH_ORBIT_RADIUS, 0.0, 0.0,
                    0.0, earth_orbit_vel, 0.0, R_EARTH);

    // Moon (offset from Earth, share velocity + relative orbital velocity)
    sim.addParticle(M_MOON, EARTH_ORBIT_RADIUS + MOON_ORBIT_RADIUS, 0.0, 0.0,
                    0.0, earth_orbit_vel + moon_orbit_vel_rel, 0.0, R_MOON);

    std::cout << "Initial system:" << std::endl;
    sim.printParticles();

    double initial_energy = sim.getTotalEnergy();
    std::cout << "Initial total energy: " << initial_energy << " J" << std::endl;

    // Integrate for one sidereal year (365.25 days)
    const double seconds_per_year = 365.25 * 24.0 * 3600.0; // ~3.15576e7 s
    std::cout << "\nIntegrating for " << seconds_per_year << " seconds (~1 year)..." << std::endl;

    sim.integrate(seconds_per_year);

    std::cout << "\nFinal system:" << std::endl;
    sim.printParticles();

    double final_energy = sim.getTotalEnergy();
    std::cout << "Final total energy: " << final_energy << " J" << std::endl;

    double energy_error = std::abs((final_energy - initial_energy) / initial_energy) * 100.0;
    std::cout << "Energy conservation error: " << energy_error << "%" << std::endl;

    std::cout << "Integration completed successfully!" << std::endl;
} 