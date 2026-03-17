"""
Leapfrog Integrator with Force Splitting
=========================================

An N-body simulation with a leapfrog integrator that splits forces into
short-range (Lennard-Jones) and long-range (Coulomb via Ewald-like cutoff).
The asymmetric force splitting breaks rotational equivariance, leading to
angular momentum violation.

Expected ConservationLint output:
  ✓ Energy conservation: PRESERVED (within O(dt^2) symplectic bound)
  ✓ Linear momentum: PRESERVED (Newton's third law satisfied in both splits)
  ✗ Angular momentum: VIOLATION DETECTED
    → Source: lines 55-72 (force splitting region)
    → Magnitude: O(dt^2) per step, secular growth
    → Cause: Short-range/long-range force split with different cutoff radii
             breaks rotational equivariance. The short-range cutoff sphere
             is not rotationally invariant w.r.t. the long-range contribution.
    → Obstruction: ARCHITECTURAL
      The force splitting strategy inherently breaks SO(3) symmetry.
      No local code modification can restore angular momentum conservation.
      Recommendation: Use a rotationally symmetric splitting (e.g., Ewald
      summation with proper real/reciprocal space decomposition).
"""

import numpy as np


def lennard_jones_force(r_vec, epsilon=1.0, sigma=1.0):
    """Short-range Lennard-Jones force between two particles."""
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return np.zeros(3)
    r_hat = r_vec / r
    magnitude = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
    return magnitude * r_hat


def coulomb_force(r_vec, q1=1.0, q2=1.0, k_e=1.0):
    """Long-range Coulomb force between two charges."""
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return np.zeros(3)
    r_hat = r_vec / r
    magnitude = k_e * q1 * q2 / r**2
    return magnitude * r_hat


def compute_short_range_forces(positions, cutoff_short=2.5):
    """Compute short-range (LJ) forces with a spherical cutoff."""
    n = len(positions)
    forces = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r < cutoff_short:
                f = lennard_jones_force(r_vec)
                forces[i] += f
                forces[j] -= f  # Newton's third law
    return forces


def compute_long_range_forces(positions, charges, cutoff_long=5.0):
    """
    Compute long-range (Coulomb) forces with a DIFFERENT cutoff.

    BUG: Using a different cutoff radius than short-range forces breaks
    rotational equivariance of the total force decomposition. Particles
    near the cutoff boundary experience asymmetric force contributions
    that do not respect rotational symmetry.
    """
    n = len(positions)
    forces = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r < cutoff_long:
                f = coulomb_force(r_vec, charges[i], charges[j])
                forces[i] += f
                forces[j] -= f
    return forces


def leapfrog_split(positions, velocities, masses, charges, dt, n_steps,
                   cutoff_short=2.5, cutoff_long=5.0):
    """
    Leapfrog integrator with split short-range / long-range forces.

    The splitting F_total = F_short + F_long with different cutoffs
    breaks rotational equivariance → angular momentum violation.
    """
    n = len(positions)
    pos = positions.copy()
    vel = velocities.copy()

    angular_momenta = np.zeros(n_steps + 1)
    energies = np.zeros(n_steps + 1)

    def total_angular_momentum(p, v, m):
        L = np.zeros(3)
        for i in range(len(p)):
            L += m[i] * np.cross(p[i], v[i])
        return np.linalg.norm(L)

    angular_momenta[0] = total_angular_momentum(pos, vel, masses)

    for step in range(n_steps):
        # Compute split forces
        f_short = compute_short_range_forces(pos, cutoff_short)
        f_long = compute_long_range_forces(pos, charges, cutoff_long)
        f_total = f_short + f_long

        # Leapfrog: half-kick
        for i in range(n):
            vel[i] += 0.5 * dt * f_total[i] / masses[i]

        # Leapfrog: drift
        pos += vel * dt

        # Recompute forces at new positions
        f_short = compute_short_range_forces(pos, cutoff_short)
        f_long = compute_long_range_forces(pos, charges, cutoff_long)
        f_total = f_short + f_long

        # Leapfrog: half-kick
        for i in range(n):
            vel[i] += 0.5 * dt * f_total[i] / masses[i]

        angular_momenta[step + 1] = total_angular_momentum(pos, vel, masses)

    return pos, vel, angular_momenta


if __name__ == "__main__":
    np.random.seed(42)
    n_particles = 8
    positions = np.random.randn(n_particles, 3) * 2.0
    velocities = np.random.randn(n_particles, 3) * 0.5
    masses = np.ones(n_particles)
    charges = np.array([1.0, -1.0] * (n_particles // 2))

    dt = 0.001
    n_steps = 5000

    pos, vel, angular_momenta = leapfrog_split(
        positions, velocities, masses, charges, dt, n_steps,
        cutoff_short=2.5, cutoff_long=5.0  # Different cutoffs → violation
    )

    L_drift = np.abs(angular_momenta[-1] - angular_momenta[0])
    print(f"Angular momentum drift: {L_drift:.6e}")
    print(f"Initial |L|: {angular_momenta[0]:.6f}")
    print(f"Final   |L|: {angular_momenta[-1]:.6f}")
    print(f"Relative drift: {L_drift / (angular_momenta[0] + 1e-15):.2e}")
    # Expected: secular angular momentum drift due to asymmetric force splitting
