"""
Verlet Integrator Energy Conservation Check
============================================

A simple harmonic oscillator integrated with the velocity Verlet method.
ConservationLint should detect that energy is preserved (symplectic integrator)
and that momentum is preserved (translation-invariant force).

Expected ConservationLint output:
  ✓ Energy conservation: PRESERVED
    Symplectic integrator detected (velocity Verlet, order 2).
    Modified Hamiltonian: H_mod = H + O(dt^2) [bounded oscillation].
  ✓ Linear momentum: PRESERVED
    Translation-invariant force field detected.
  ✓ Angular momentum: PRESERVED
    Central force (radial potential) detected.
"""

import numpy as np


def harmonic_oscillator_verlet(x0, v0, k, m, dt, n_steps):
    """
    Velocity Verlet integration of a 2D harmonic oscillator.

    Parameters
    ----------
    x0 : np.ndarray, shape (2,)
        Initial position.
    v0 : np.ndarray, shape (2,)
        Initial velocity.
    k : float
        Spring constant.
    m : float
        Particle mass.
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.

    Returns
    -------
    positions : np.ndarray, shape (n_steps+1, 2)
    velocities : np.ndarray, shape (n_steps+1, 2)
    energies : np.ndarray, shape (n_steps+1,)
    """
    positions = np.zeros((n_steps + 1, 2))
    velocities = np.zeros((n_steps + 1, 2))
    energies = np.zeros(n_steps + 1)

    positions[0] = x0
    velocities[0] = v0

    def force(x):
        return -k * x

    def kinetic_energy(v):
        return 0.5 * m * np.dot(v, v)

    def potential_energy(x):
        return 0.5 * k * np.dot(x, x)

    energies[0] = kinetic_energy(v0) + potential_energy(x0)

    x = x0.copy()
    v = v0.copy()

    for i in range(n_steps):
        # Velocity Verlet: symplectic, order 2
        a = force(x) / m
        x_new = x + v * dt + 0.5 * a * dt**2      # position update
        a_new = force(x_new) / m
        v_new = v + 0.5 * (a + a_new) * dt         # velocity update

        x = x_new
        v = v_new

        positions[i + 1] = x
        velocities[i + 1] = v
        energies[i + 1] = kinetic_energy(v) + potential_energy(x)

    return positions, velocities, energies


if __name__ == "__main__":
    x0 = np.array([1.0, 0.0])
    v0 = np.array([0.0, 1.0])
    k, m, dt = 1.0, 1.0, 0.01
    n_steps = 10000

    positions, velocities, energies = harmonic_oscillator_verlet(
        x0, v0, k, m, dt, n_steps
    )

    energy_drift = np.max(np.abs(energies - energies[0]))
    print(f"Energy drift over {n_steps} steps: {energy_drift:.2e}")
    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Final energy:   {energies[-1]:.6f}")
    print(f"Relative drift: {energy_drift / energies[0]:.2e}")
    # Expected: energy drift ~ O(dt^2) bounded oscillation, not secular growth
