#include "test_framework.h"
#include "rebound_cuda.h"
#include "rebound_streaming.h"
#include <vector>

// Test 1: Basic Observer Pattern functionality
TEST(Streaming, ObserverBasicFunctionality) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(10, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add some particles
    for (int i = 0; i < 10; i++) {
        double x = ((double)i / 10.0 - 0.5) * 2.0;
        sim.addParticle(0.1, x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
    }
    
    // Create streaming observer
    DataStreamingManager observer;
    BufferConfig config;
    config.max_frames = 100;
    config.enable_gpu_logging = false;
    config.async_streaming = false;
    
    ASSERT_TRUE(observer.initialize(config));
    observer.setStreamingMode(STREAM_NONE);
    
    // Test observer attachment/detachment
    sim.addObserver(&observer);
    
    // Run simulation
    sim.integrate(0.1);
    
    // Remove observer
    sim.removeObserver(&observer);
    
    // Should complete without errors
    ASSERT_GT(sim.getCurrentTime(), 0.0);
    
    // Cleanup
    observer.shutdown();
    
    std::cout << "  Observer attached and detached successfully" << std::endl;
    std::cout << "  Final simulation time: " << sim.getCurrentTime() << std::endl;
}

// Test 2: Multiple observers
TEST(Streaming, MultipleObservers) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(5, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add particles
    for (int i = 0; i < 5; i++) {
        sim.addParticle(0.1, i * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
    }
    
    // Create two different observers
    DataStreamingManager observer1, observer2;
    
    BufferConfig config1;
    config1.max_frames = 50;
    config1.stream_interval = 1;
    config1.enable_gpu_logging = false;
    
    BufferConfig config2;
    config2.max_frames = 100;
    config2.stream_interval = 2;
    config2.enable_gpu_logging = false;
    
    ASSERT_TRUE(observer1.initialize(config1));
    ASSERT_TRUE(observer2.initialize(config2));
    
    observer1.setStreamingMode(STREAM_PERIODIC);
    observer2.setStreamingMode(STREAM_PERIODIC);
    
    // Attach both observers
    sim.addObserver(&observer1);
    sim.addObserver(&observer2);
    
    // Run simulation
    sim.integrate(0.1);
    
    // Both observers should have captured data
    int frames1 = observer1.getBufferedFrameCount();
    int frames2 = observer2.getBufferedFrameCount();
    
    ASSERT_GT(frames1, 0);
    ASSERT_GT(frames2, 0);
    
    // Observer1 should have more frames (lower interval)
    ASSERT_GE(frames1, frames2);
    
    // Cleanup
    sim.removeObserver(&observer1);
    sim.removeObserver(&observer2);
    observer1.shutdown();
    observer2.shutdown();
    
    std::cout << "  Observer 1 frames: " << frames1 << std::endl;
    std::cout << "  Observer 2 frames: " << frames2 << std::endl;
}

// Test 3: Streaming modes
TEST(Streaming, StreamingModes) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(5, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add particles
    for (int i = 0; i < 5; i++) {
        sim.addParticle(0.1, i * 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
    }
    
    // Test STREAM_NONE mode
    {
        DataStreamingManager observer;
        BufferConfig config;
        config.max_frames = 100;
        config.enable_gpu_logging = false;
        
        observer.initialize(config);
        observer.setStreamingMode(STREAM_NONE);
        
        sim.addObserver(&observer);
        sim.integrate(0.05);
        sim.removeObserver(&observer);
        
        // Should have 0 frames in STREAM_NONE mode
        int frames = observer.getBufferedFrameCount();
        ASSERT_EQ(frames, 0);
        
        observer.shutdown();
        std::cout << "  STREAM_NONE frames: " << frames << std::endl;
    }
    
    // Test STREAM_PERIODIC mode
    {
        DataStreamingManager observer;
        BufferConfig config;
        config.max_frames = 100;
        config.stream_interval = 3;  // Every 3 steps
        config.enable_gpu_logging = false;
        
        observer.initialize(config);
        observer.setStreamingMode(STREAM_PERIODIC);
        
        sim.addObserver(&observer);
        sim.integrate(0.1);  // Additional integration
        sim.removeObserver(&observer);
        
        // Should have some frames
        int frames = observer.getBufferedFrameCount();
        ASSERT_GT(frames, 0);
        
        observer.shutdown();
        std::cout << "  STREAM_PERIODIC frames: " << frames << std::endl;
    }
}

// Test 4: Buffer management
TEST(Streaming, BufferManagement) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(3, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add particles
    for (int i = 0; i < 3; i++) {
        sim.addParticle(0.1, i * 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
    }
    
    DataStreamingManager observer;
    BufferConfig config;
    config.max_frames = 10;  // Small buffer
    config.stream_interval = 1;
    config.enable_gpu_logging = false;
    
    observer.initialize(config);
    observer.setStreamingMode(STREAM_PERIODIC);
    
    sim.addObserver(&observer);
    
    // Run enough steps to potentially overflow buffer
    sim.integrate(0.2);
    
    // Buffer should not exceed max_frames
    int frames = observer.getBufferedFrameCount();
    ASSERT_LE(frames, config.max_frames);
    
    // Should still have some data
    ASSERT_GT(frames, 0);
    
    sim.removeObserver(&observer);
    observer.shutdown();
    
    std::cout << "  Buffered frames: " << frames << " (max: " << config.max_frames << ")" << std::endl;
}

// Test 5: Data retrieval
TEST(Streaming, DataRetrieval) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(3, 0.01, 1.0);
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add particles
    sim.addParticle(1.0, -1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.01);
    sim.addParticle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
    sim.addParticle(1.0, 1.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.01);
    
    DataStreamingManager observer;
    BufferConfig config;
    config.max_frames = 50;
    config.stream_interval = 2;
    config.enable_gpu_logging = false;
    
    observer.initialize(config);
    observer.setStreamingMode(STREAM_PERIODIC);
    
    sim.addObserver(&observer);
    sim.integrate(0.1);
    sim.removeObserver(&observer);
    
    // Retrieve data
    auto frames = observer.getAllFrames();
    auto logs = observer.getAllLogs();
    
    ASSERT_GT(frames.size(), 0);
    
    // Check frame data validity
    for (const auto& frame : frames) {
        ASSERT_GT(frame.time, 0.0);
        ASSERT_GT(frame.step, 0);
        ASSERT_EQ(frame.n_particles, 3);
        // Note: We can't easily check particles data since it's a device pointer
    }
    
    observer.shutdown();
    
    std::cout << "  Retrieved " << frames.size() << " frames" << std::endl;
    std::cout << "  Retrieved " << logs.size() << " log entries" << std::endl;
}

// Test 6: Simulation independence
TEST(Streaming, SimulationIndependence) {
    // Test that simulation runs independently without observers
    ReboundCudaSimulation sim1, sim2;
    
    // Setup identical simulations
    sim1.initializeSimulation(5, 0.01, 1.0);
    sim2.initializeSimulation(5, 0.01, 1.0);
    
    sim1.setGravityMode(GRAVITY_BASIC);
    sim2.setGravityMode(GRAVITY_BASIC);
    
    // Add same particles
    srand(98765);
    for (int i = 0; i < 5; i++) {
        double x = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double y = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double z = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double mass = 0.1;
        
        sim1.addParticle(mass, x, y, z, 0.0, 0.0, 0.0, 0.01);
        sim2.addParticle(mass, x, y, z, 0.0, 0.0, 0.0, 0.01);
    }
    
    // Add observer only to sim2
    DataStreamingManager observer;
    BufferConfig config;
    config.max_frames = 50;
    config.stream_interval = 1;
    config.enable_gpu_logging = false;
    
    observer.initialize(config);
    observer.setStreamingMode(STREAM_PERIODIC);
    sim2.addObserver(&observer);
    
    // Run both simulations
    sim1.integrate(0.1);  // No observer
    sim2.integrate(0.1);  // With observer
    
    // Both should reach the same final time
    ASSERT_NEAR(sim1.getCurrentTime(), sim2.getCurrentTime(), 1e-10);
    
    // Both should have the same energy (approximately)
    double energy1 = sim1.getTotalEnergy();
    double energy2 = sim2.getTotalEnergy();
    double energy_diff = std::abs(energy1 - energy2) / std::abs(energy1);
    
    ASSERT_LT(energy_diff, 1e-10);  // Should be nearly identical
    
    sim2.removeObserver(&observer);
    observer.shutdown();
    
    std::cout << "  Sim1 (no observer) energy: " << energy1 << std::endl;
    std::cout << "  Sim2 (with observer) energy: " << energy2 << std::endl;
    std::cout << "  Energy difference: " << (energy_diff * 100) << "%" << std::endl;
}

// Test 7: Performance with observers
TEST(Streaming, PerformanceWithObservers) {
    ReboundCudaSimulation sim;
    sim.initializeSimulation(50, 0.001, 1.0);  // Larger system
    sim.setGravityMode(GRAVITY_BASIC);
    
    // Add particles
    srand(11111);
    for (int i = 0; i < 50; i++) {
        double x = ((double)rand() / RAND_MAX - 0.5) * 4.0;
        double y = ((double)rand() / RAND_MAX - 0.5) * 4.0;
        double z = ((double)rand() / RAND_MAX - 0.5) * 4.0;
        sim.addParticle(0.01, x, y, z, 0.0, 0.0, 0.0, 0.01);
    }
    
    // Time without observer
    auto start1 = std::chrono::high_resolution_clock::now();
    sim.integrate(0.05);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    
    // Add observer and continue
    DataStreamingManager observer;
    BufferConfig config;
    config.max_frames = 100;
    config.stream_interval = 5;
    config.enable_gpu_logging = false;
    config.async_streaming = false;
    
    observer.initialize(config);
    observer.setStreamingMode(STREAM_PERIODIC);
    sim.addObserver(&observer);
    
    // Time with observer
    auto start2 = std::chrono::high_resolution_clock::now();
    sim.integrate(0.1);  // Continue to 0.1
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    
    // Observer overhead should be minimal (< 50% increase)
    double overhead = (double)duration2.count() / duration1.count();
    ASSERT_LT(overhead, 1.5);  // Less than 50% overhead
    
    sim.removeObserver(&observer);
    observer.shutdown();
    
    std::cout << "  Time without observer: " << duration1.count() << " μs" << std::endl;
    std::cout << "  Time with observer: " << duration2.count() << " μs" << std::endl;
    std::cout << "  Observer overhead: " << ((overhead - 1.0) * 100) << "%" << std::endl;
} 