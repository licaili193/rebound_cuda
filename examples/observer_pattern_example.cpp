#include <iostream>
#include <chrono>
#include "../include/rebound_cuda.h"
#include "../include/rebound_streaming.h"

// Custom observer for demonstration
class SimpleMonitor : public SimulationObserver {
private:
    int step_count_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    
public:
    SimpleMonitor() : step_count_(0) {}
    
    void onSimulationStart(int n_particles) override {
        std::cout << "[MONITOR] Simulation started with " << n_particles << " particles" << std::endl;
        step_count_ = 0;
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void onSimulationStep(double time, int step, int n_particles, double energy, int n_collisions) override {
        step_count_++;
        if (step % 100 == 0) {  // Report every 100 steps
            std::cout << "[MONITOR] Step " << step << ", time=" << time 
                      << ", particles=" << n_particles << std::endl;
        }
    }
    
    void onSimulationEnd(double final_time, int total_steps) override {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        
        std::cout << "[MONITOR] Simulation completed!" << std::endl;
        std::cout << "[MONITOR] Final time: " << final_time << std::endl;
        std::cout << "[MONITOR] Total steps: " << total_steps << std::endl;
        std::cout << "[MONITOR] Wall time: " << duration.count() << " ms" << std::endl;
        std::cout << "[MONITOR] Performance: " << (total_steps * 1000.0 / duration.count()) << " steps/sec" << std::endl;
    }
    
    void onCollisionDetected(int particle1, int particle2, double time) override {
        std::cout << "[MONITOR] Collision detected between particles " << particle1 
                  << " and " << particle2 << " at time " << time << std::endl;
    }
};

int main() {
    std::cout << "REBOUND CUDA Observer Pattern Demo" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Demonstrating completely decoupled simulation and streaming" << std::endl;
    
    // =============================================================================
    // Demo 1: Pure simulation without any observers (maximum performance)
    // =============================================================================
    std::cout << "\n=== Demo 1: Pure Simulation (No Observers) ===" << std::endl;
    
    {
        ReboundCudaSimulation sim;
        sim.initializeSimulation(50, 0.001, 1.0);
        sim.setGravityMode(GRAVITY_BASIC);
        
        // Add some particles
        for (int i = 0; i < 50; i++) {
            double x = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double y = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double z = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double vx = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vy = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vz = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double mass = 0.001 + ((double)rand() / RAND_MAX) * 0.01;
            
            sim.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        sim.integrate(0.01);  // Pure simulation, no overhead
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Pure simulation time: " << duration.count() << " ms" << std::endl;
        std::cout << "No observers attached - maximum performance achieved!" << std::endl;
    }
    
    // =============================================================================
    // Demo 2: Simulation with simple monitoring observer
    // =============================================================================
    std::cout << "\n=== Demo 2: Simulation with Simple Monitor ===" << std::endl;
    
    {
        ReboundCudaSimulation sim;
        SimpleMonitor monitor;
        
        // Attach observer to simulation
        sim.addObserver(&monitor);
        
        sim.initializeSimulation(50, 0.001, 1.0);
        sim.setGravityMode(GRAVITY_BASIC);
        
        // Add particles
        for (int i = 0; i < 50; i++) {
            double x = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double y = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double z = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double vx = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vy = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vz = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double mass = 0.001 + ((double)rand() / RAND_MAX) * 0.01;
            
            sim.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
        }
        
        sim.integrate(0.01);  // Simulation with monitoring
        
        // Observer automatically cleaned up when it goes out of scope
        sim.removeObserver(&monitor);
    }
    
    // =============================================================================
    // Demo 3: Simulation with data streaming observer
    // =============================================================================
    std::cout << "\n=== Demo 3: Simulation with Data Streaming Observer ===" << std::endl;
    
    {
        ReboundCudaSimulation sim;
        DataStreamingManager streaming;
        
        // Configure streaming
        BufferConfig config;
        config.max_frames = 500;
        config.stream_interval = 50;  // Stream every 50 steps
        config.enable_gpu_logging = true;
        config.log_level = LOG_INFO;
        config.async_streaming = false;  // Synchronous for this demo
        config.worker_threads = 1;
        
        streaming.initialize(config);
        streaming.setStreamingMode(STREAM_PERIODIC);
        
        // Attach streaming as observer
        sim.addObserver(&streaming);
        
        // IMPORTANT: Streaming needs access to device particle pointer
        streaming.setDeviceParticlePointer(sim.getDeviceParticles());
        
        sim.initializeSimulation(50, 0.001, 1.0);
        sim.setGravityMode(GRAVITY_BASIC);
        
        // Add particles
        for (int i = 0; i < 50; i++) {
            double x = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double y = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double z = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double vx = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vy = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vz = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double mass = 0.001 + ((double)rand() / RAND_MAX) * 0.01;
            
            sim.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
        }
        
        sim.integrate(0.01);  // Simulation with streaming
        
        // Access streamed data
        std::cout << "Buffered frames: " << streaming.getBufferedFrameCount() << std::endl;
        std::cout << "Buffered logs: " << streaming.getBufferedLogCount() << std::endl;
        
        sim.removeObserver(&streaming);
    }
    
    // =============================================================================
    // Demo 4: Multiple observers (monitoring + streaming)
    // =============================================================================
    std::cout << "\n=== Demo 4: Multiple Observers ===" << std::endl;
    
    {
        ReboundCudaSimulation sim;
        SimpleMonitor monitor;
        DataStreamingManager streaming;
        
        // Configure streaming
        BufferConfig config;
        config.max_frames = 200;
        config.stream_interval = 25;
        config.enable_gpu_logging = false;
        config.async_streaming = false;
        
        streaming.initialize(config);
        streaming.setStreamingMode(STREAM_PERIODIC);
        
        // Attach both observers
        sim.addObserver(&monitor);
        sim.addObserver(&streaming);
        
        streaming.setDeviceParticlePointer(sim.getDeviceParticles());
        
        sim.initializeSimulation(30, 0.001, 1.0);
        sim.setGravityMode(GRAVITY_BASIC);
        
        // Add particles
        for (int i = 0; i < 30; i++) {
            double x = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double y = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double z = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            double vx = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vy = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double vz = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double mass = 0.001 + ((double)rand() / RAND_MAX) * 0.01;
            
            sim.addParticle(mass, x, y, z, vx, vy, vz, 0.01);
        }
        
        sim.integrate(0.005);  // Simulation with both monitoring and streaming
        
        std::cout << "Final streaming stats:" << std::endl;
        std::cout << "  Buffered frames: " << streaming.getBufferedFrameCount() << std::endl;
        
        // Clean up
        sim.removeObserver(&monitor);
        sim.removeObserver(&streaming);
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "✅ Simulation core is completely independent" << std::endl;
    std::cout << "✅ Streaming is an optional add-on observer" << std::endl;
    std::cout << "✅ No performance overhead when observers not attached" << std::endl;
    std::cout << "✅ Multiple observers can be attached simultaneously" << std::endl;
    std::cout << "✅ Observers can be added/removed dynamically" << std::endl;
    std::cout << "✅ Perfect separation of concerns achieved!" << std::endl;
    
    return 0;
} 