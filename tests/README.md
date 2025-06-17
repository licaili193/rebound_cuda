# REBOUND CUDA Unit Tests

This directory contains a comprehensive unit test suite for the REBOUND CUDA N-body simulation library.

## Test Framework

The tests use a custom lightweight testing framework (`test_framework.h`) with the following features:
- Automatic test registration using macros
- Category-based test organization  
- Assertion macros for common checks
- Performance timing
- Detailed failure reporting

## Test Categories

### Physics Tests (`test_physics.cpp`)
- **TwoBodyCircularOrbit**: Tests energy conservation in a two-body circular orbit
- **TwoBodyKinematics**: Validates particle motion without gravity (pure kinematics)
- **EnergyConservation100Particles**: Tests energy conservation in a 100-particle system
- **GravityModesConsistency**: Compares different gravity calculation modes
- **TimeSteppingAccuracy**: Verifies that smaller timesteps give more accurate results

### Collision Tests (`test_collisions.cpp`)
- **DirectCollisionDetection**: Tests basic collision detection functionality
- **HardSphereCollision**: Tests elastic collision resolution
- **InelasticCollision**: Tests inelastic collision with coefficient of restitution
- **MultipleParticleCollisions**: Tests complex multi-particle collision scenarios
- **NoFalseCollisions**: Ensures no false positive collision detection
- **CollisionWithGravity**: Tests collisions in the presence of gravitational forces

### Streaming Tests (`test_streaming.cpp`)
- **ObserverBasicFunctionality**: Tests basic Observer Pattern implementation
- **MultipleObservers**: Tests multiple observers attached to one simulation
- **StreamingModes**: Tests different streaming modes (NONE, PERIODIC, etc.)
- **BufferManagement**: Tests GPU buffer overflow handling
- **DataRetrieval**: Tests data extraction from GPU buffers
- **SimulationIndependence**: Ensures simulation runs identically with/without observers
- **PerformanceWithObservers**: Measures observer overhead

## Building and Running Tests

### Build Tests
```bash
cd build
cmake --build . --target rebound_tests
```

### Run All Tests
```bash
./tests/rebound_tests
```

### Run Specific Category
```bash
./tests/rebound_tests Physics     # Run only physics tests
./tests/rebound_tests Collisions # Run only collision tests
./tests/rebound_tests Streaming  # Run only streaming tests
```

### Alternative Build Targets
```bash
# Build and run specific test categories
cmake --build . --target test_physics
cmake --build . --target test_collisions
cmake --build . --target test_streaming
cmake --build . --target test_all
```

## Test Output

Example output:
```
Running REBOUND CUDA Unit Tests
===============================
[Physics] Running: TwoBodyCircularOrbit... PASSED
  Initial energy: -0.25
  Final energy: -0.249876
  Energy error: 0.0496%
[Physics] Running: EnergyConservation100Particles... PASSED
  Initial energy: -2.34567
  Final energy: -2.34234
  Energy drift: 0.142%
...
===============================
Test Results:
  Passed: 15
  Failed: 0
  Total:  15
  Time:   2841 ms
All tests PASSED!
```

## Adding New Tests

To add a new test:

1. Choose appropriate test file (`test_physics.cpp`, `test_collisions.cpp`, or `test_streaming.cpp`)
2. Add test using the `TEST` macro:
```cpp
TEST(Category, TestName) {
    // Test implementation
    ReboundCudaSimulation sim;
    // ... setup ...
    
    ASSERT_TRUE(condition);
    ASSERT_NEAR(expected, actual, tolerance);
    // ... other assertions ...
}
```

3. Rebuild and run tests

## Available Assertion Macros

- `ASSERT_TRUE(condition)` - Assert condition is true
- `ASSERT_FALSE(condition)` - Assert condition is false  
- `ASSERT_EQ(expected, actual)` - Assert values are equal
- `ASSERT_NEAR(expected, actual, tolerance)` - Assert values are within tolerance
- `ASSERT_GT(value, threshold)` - Assert value is greater than threshold
- `ASSERT_LT(value, threshold)` - Assert value is less than threshold

## Test Design Principles

1. **Deterministic**: All tests use fixed random seeds for reproducibility
2. **Fast**: Tests should complete in seconds, not minutes
3. **Independent**: Tests don't depend on each other's state
4. **Focused**: Each test checks one specific aspect of functionality
5. **Self-contained**: Tests include their own setup and cleanup
6. **Analytical**: Where possible, compare against known analytical solutions

## Performance Benchmarks

The tests also serve as performance benchmarks:
- Energy conservation accuracy over time
- Observer pattern overhead measurement
- Collision detection performance
- Memory transfer efficiency

## Continuous Integration

These tests are designed to be run in CI/CD pipelines to catch regressions early. The test runner returns appropriate exit codes for automation:
- Exit code 0: All tests passed
- Exit code > 0: Number of failed tests 