#include "../include/rebound_cuda.h"
#include "../include/rebound_streaming.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <mutex>
#include <vector>

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

// Global mutex for safe file writing from possible worker threads
static std::mutex g_file_mutex;

// Frame callback used by DataStreamingManager
static void frameWriterCallback(const SimulationFrame* frame, void* user_data) {
    if (!frame || !frame->particles) return;

    // Copy particle data from device to host (small n, so this is cheap)
    int n = frame->n_particles;
    std::vector<Particle> host_particles(n);
    cudaMemcpy(host_particles.data(), frame->particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Write to CSV
    std::ofstream* out = static_cast<std::ofstream*>(user_data);
    if (!out || !out->is_open()) return;

    std::lock_guard<std::mutex> lock(g_file_mutex);
    // CSV format: time, step, x0,y0,z0,x1,y1,z1,...
    *out << frame->time << "," << frame->step;
    for (const auto& p : host_particles) {
        *out << "," << p.x << "," << p.y << "," << p.z;
    }
    *out << "\n";
}

void runPlanetarySystemExample() {
    std::cout << "\n=== PLANETARY SYSTEM EXAMPLE (REALISTIC UNITS) ===" << std::endl;

    // ------------------------------
    // Create and initialise simulation
    // ------------------------------
    ReboundCudaSimulation sim;

    const double dt_seconds = 3600.0;                 // 1-hour timestep
    const int    n_particles = 3;                     // Sun, Earth, Moon

    sim.initializeSimulation(n_particles, dt_seconds, G_SI);

    // ----------------------------------------
    // Set up data streaming (observer pattern)
    // ----------------------------------------
    DataStreamingManager streamer;
    BufferConfig buf_cfg;
    buf_cfg.max_frames       = 10000;   // plenty for ~9000 steps
    buf_cfg.stream_interval  = 1;       // every step
    buf_cfg.enable_gpu_logging = false;
    buf_cfg.async_streaming  = false;   // simpler offline fetch

    streamer.initialize(buf_cfg);
    streamer.setStreamingMode(STREAM_PERIODIC); // every step

    // Attach streamer
    sim.addObserver(&streamer);
    streamer.setDeviceParticlePointer(sim.getDeviceParticles());

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

    // ----------------------------------------
    // Dump buffered frames to CSV
    // ----------------------------------------
    std::ofstream snapshot_file("snapshots.csv");
    snapshot_file << "time,step";
    for (int i = 0; i < n_particles; ++i) snapshot_file << ",x" << i << ",y" << i << ",z" << i;
    snapshot_file << "\n";

    auto frames = streamer.getAllFrames();
    for (const auto& frm : frames) {
        snapshot_file << frm.time << "," << frm.step;
        const Particle* host_particles = frm.particles; // Already on host
        for (int i = 0; i < n_particles; ++i) {
            const Particle& p = host_particles[i];
            snapshot_file << "," << p.x << "," << p.y << "," << p.z;
        }
        snapshot_file << "\n";
    }

    snapshot_file.close();

    sim.removeObserver(&streamer);
    streamer.shutdown();
} 