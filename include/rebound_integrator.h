#ifndef REBOUND_INTEGRATOR_BASE_H
#define REBOUND_INTEGRATOR_BASE_H

#include "rebound_types.h"
#include <cuda_runtime.h>

// Abstract interface for GPU-based N-body integrators.
class Integrator {
public:
    virtual ~Integrator() = default;

    // Human-readable identifier
    virtual const char* name() const = 0;

    // Drift: advance positions by dt using current velocities.
    virtual void drift(Particle* d_particles, int n_particles, double dt) = 0;

    // Kick: advance velocities by dt using current accelerations.
    virtual void kick(Particle* d_particles, int n_particles, double dt) = 0;
};

// -----------------------------------------------------------------------------
// Leapfrog (DKD) integrator implementation
// -----------------------------------------------------------------------------
class LeapfrogIntegrator : public Integrator {
public:
    const char* name() const override { return "Leapfrog DKD"; }

    // Implementations are defined in rebound_integrator.cu
    void drift(Particle* d_particles, int n_particles, double dt) override;
    void kick(Particle* d_particles, int n_particles, double dt) override;
};

#endif // REBOUND_INTEGRATOR_BASE_H 