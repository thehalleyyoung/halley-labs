"""
Symplectic Euler with Nose-Hoover Thermostat
=============================================

A harmonic oscillator chain integrated with symplectic Euler, coupled to a
Nose-Hoover thermostat for canonical (NVT) ensemble sampling. Energy is
NOT conserved by design: the thermostat exchanges energy with the system
to maintain constant temperature.

Expected ConservationLint output:
  ✗ Energy conservation: VIOLATION DETECTED
    → Source: lines 48-56 (thermostat coupling)
    → Magnitude: O(1) per step (not a discretization artifact)
    → Cause: Nose-Hoover thermostat intentionally breaks time-translation
             symmetry by coupling the system to a thermal reservoir.
    → Obstruction: ARCHITECTURAL (by design)
      This is an INTENTIONAL violation. The Nose-Hoover thermostat is
      designed to break energy conservation for NVT ensemble sampling.
      The extended Hamiltonian H_NH = H_sys + p_xi^2/(2*Q) + N*kT*xi
      IS conserved, but the physical subsystem energy H_sys is not.
      Recommendation: No action needed. If microcanonical (NVE) ensemble
      is desired, remove the thermostat coupling.
  ✓ Linear momentum: PRESERVED
  ✓ Angular momentum: N/A (1D system)
"""

import numpy as np


def nose_hoover_symplectic_euler(x0, v0, k_spring, masses, dt, n_steps,
                                  target_temp, Q_thermostat):
    """
    Symplectic Euler integrator with Nose-Hoover thermostat for a 1D
    harmonic oscillator chain.

    Parameters
    ----------
    x0 : np.ndarray
        Initial positions of chain particles.
    v0 : np.ndarray
        Initial velocities.
    k_spring : float
        Spring constant between adjacent particles.
    masses : np.ndarray
        Particle masses.
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.
    target_temp : float
        Target temperature (k_B * T).
    Q_thermostat : float
        Thermostat mass parameter (controls coupling strength).
    """
    n = len(x0)
    x = x0.copy()
    v = v0.copy()
    xi = 0.0       # thermostat variable
    p_xi = 0.0     # thermostat momentum

    energies_sys = np.zeros(n_steps + 1)
    energies_ext = np.zeros(n_steps + 1)
    temperatures = np.zeros(n_steps + 1)

    def compute_forces(positions):
        """Nearest-neighbor spring forces with fixed endpoints."""
        f = np.zeros(n)
        for i in range(n):
            if i > 0:
                f[i] += -k_spring * (positions[i] - positions[i - 1])
            if i < n - 1:
                f[i] += -k_spring * (positions[i] - positions[i + 1])
        return f

    def kinetic_energy(velocities):
        return 0.5 * np.sum(masses * velocities**2)

    def potential_energy(positions):
        pe = 0.0
        for i in range(n - 1):
            pe += 0.5 * k_spring * (positions[i + 1] - positions[i])**2
        return pe

    def instantaneous_temp(velocities):
        return 2.0 * kinetic_energy(velocities) / n

    energies_sys[0] = kinetic_energy(v) + potential_energy(x)
    energies_ext[0] = energies_sys[0] + 0.5 * p_xi**2 / Q_thermostat
    temperatures[0] = instantaneous_temp(v)

    for step in range(n_steps):
        # Compute forces
        f = compute_forces(x)

        # Thermostat friction coefficient
        # This coupling intentionally breaks energy conservation
        friction = p_xi / Q_thermostat

        # Symplectic Euler: update velocities first (implicit in p)
        v_new = v + dt * (f / masses - friction * v)

        # Symplectic Euler: update positions (explicit in q)
        x_new = x + dt * v_new

        # Update thermostat momentum
        inst_temp = instantaneous_temp(v_new)
        p_xi_new = p_xi + dt * (inst_temp - target_temp)

        # Update thermostat position
        xi_new = xi + dt * p_xi_new / Q_thermostat

        x = x_new
        v = v_new
        xi = xi_new
        p_xi = p_xi_new

        ke = kinetic_energy(v)
        pe = potential_energy(x)
        energies_sys[step + 1] = ke + pe
        energies_ext[step + 1] = (ke + pe +
                                   0.5 * p_xi**2 / Q_thermostat +
                                   n * target_temp * xi)
        temperatures[step + 1] = instantaneous_temp(v)

    return x, v, energies_sys, energies_ext, temperatures


if __name__ == "__main__":
    n_particles = 16
    x0 = np.linspace(0, n_particles - 1, n_particles, dtype=float)
    np.random.seed(123)
    v0 = np.random.randn(n_particles) * 0.5
    masses = np.ones(n_particles)

    k_spring = 10.0
    dt = 0.005
    n_steps = 20000
    target_temp = 1.0
    Q_thermostat = 10.0

    x, v, e_sys, e_ext, temps = nose_hoover_symplectic_euler(
        x0, v0, k_spring, masses, dt, n_steps, target_temp, Q_thermostat
    )

    print(f"System energy drift:   {np.abs(e_sys[-1] - e_sys[0]):.4e}")
    print(f"Extended energy drift: {np.abs(e_ext[-1] - e_ext[0]):.4e}")
    print(f"Target temperature:    {target_temp:.2f}")
    print(f"Final temperature:     {temps[-1]:.4f}")
    print(f"Mean temperature:      {np.mean(temps[n_steps//2:]):.4f}")
    # Expected: system energy fluctuates (thermostat working)
    # Extended Hamiltonian approximately conserved (symplectic structure)
