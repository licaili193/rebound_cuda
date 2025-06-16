# REBOUND CUDA Collision System

This document describes the collision detection and resolution system implemented in REBOUND CUDA, based on the collision handling in the original REBOUND library.

## Overview

The collision system consists of two main components:
1. **Collision Detection**: Finding when particles collide
2. **Collision Resolution**: Deciding what happens when particles collide

## Collision Detection Methods

### COLLISION_NONE
- **Default**: No collision detection
- **Performance**: Fastest (no collision checks)
- **Use case**: When collisions are not important for your simulation

### COLLISION_DIRECT
- **Algorithm**: Brute force O(N²) collision search
- **Checks**: Instantaneous overlaps between every particle pair
- **Performance**: Suitable for small to medium systems (N < 1000)
- **Use case**: General purpose collision detection

### COLLISION_TREE
- **Algorithm**: Octree-based collision detection O(N log N)
- **Performance**: Better for large systems (N > 1000)
- **Requirements**: Requires simulation box to be configured
- **Use case**: Large particle systems, granular dynamics

## Collision Resolution Methods

### COLLISION_RESOLVE_HALT
- **Action**: Stops simulation when collision is detected
- **Use case**: Studying the onset of instabilities
- **Conservation**: N/A (simulation stops)

### COLLISION_RESOLVE_HARDSPHERE
- **Action**: Elastic collision with configurable coefficient of restitution
- **Physics**: Conserves momentum and mass
- **Energy**: Conserves energy if coefficient of restitution = 1.0
- **Use case**: Bouncing ball simulations, granular dynamics

### COLLISION_RESOLVE_MERGE
- **Action**: Combines two particles into one
- **Physics**: Conserves mass, momentum, and volume
- **Energy**: Does not conserve energy (inelastic collision)
- **Particle removal**: Higher index particle is removed
- **Use case**: Accretion disks, planetary formation

## Usage Example

```cpp
#include "rebound_cuda.h"

int main() {
    ReboundCudaSimulation sim;
    
    // Enable collision detection
    sim.setCollisionDetection(COLLISION_DIRECT);
    sim.setCollisionResolution(COLLISION_RESOLVE_HARDSPHERE);
    sim.setCoefficientOfRestitution(0.8f);  // 80% energy retention
    
    // Add particles with finite radii
    sim.addParticle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1);  // radius = 0.1
    sim.addParticle(1e-4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05); // radius = 0.05
    
    sim.integrate(100.0);
    
    return 0;
}
```

## Implementation Details

### Collision Detection
- Particles must have finite radii (radius > 0) to participate in collisions
- Collision occurs when distance between particle centers < sum of radii
- Only approaching particles (relative velocity < 0) are considered for collision
- CUDA kernels use atomic operations to safely record collisions

### Collision Resolution
- Hard sphere collisions use impulse-based resolution
- Collision normal is computed from particle positions
- Impulse is applied along collision normal
- Particle properties are updated in-place on GPU

### Performance Considerations
- Direct collision detection: O(N²) - suitable for N < 1000
- Tree collision detection: O(N log N) - better for large N
- Resolution kernels are parallelized across detected collisions

## Physical Accuracy

The collision system follows the same physics as the original REBOUND:
- Momentum conservation in all collision types
- Energy conservation (hard sphere with ε = 1.0)
- Volume conservation (merge collisions)
- Mass conservation in all collision types

## Limitations

1. **Particle Tunneling**: Fast-moving small particles may tunnel through each other if timestep is too large
2. **Tree Detection**: Currently falls back to direct detection (tree implementation in progress)
3. **Custom Collision Functions**: Not yet implemented (planned for future release)

## Future Enhancements

- Full octree-based collision detection
- Custom collision resolution functions
- Line-based collision detection (trajectory checking)
- Collision statistics and logging
- Periodic boundary condition support for collisions 