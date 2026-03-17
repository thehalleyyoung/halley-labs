//! N-body gravitational simulation with conservation auditing.
//!
//! Simulates a 3-body gravitational system using Velocity Verlet and monitors
//! conservation of energy, linear momentum, and angular momentum at every step.
//!
//! Run: `cargo run --example nbody_audit`

use sim_types::Vec3;

fn main() {
    println!("=== N-Body Conservation Audit ===\n");

    // Set up a 3-body system (figure-eight-like initial conditions)
    let mut positions = vec![
        Vec3::new(0.97000436, -0.24308753, 0.0),
        Vec3::new(-0.97000436, 0.24308753, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
    ];
    let mut velocities = vec![
        Vec3::new(0.4662036850, 0.4323657300, 0.0),
        Vec3::new(0.4662036850, 0.4323657300, 0.0),
        Vec3::new(-0.9324073700, -0.8647314600, 0.0),
    ];
    let masses = vec![1.0, 1.0, 1.0];

    let n = positions.len();

    let dt = 0.001;
    let steps = 50_000;
    let g = 1.0; // gravitational constant in natural units

    // Compute initial conserved quantities
    let initial_energy = compute_total_energy(&positions, &velocities, &masses, g);
    let initial_momentum = compute_total_momentum(&velocities, &masses);
    let initial_angular_momentum = compute_total_angular_momentum(&positions, &velocities, &masses);

    println!("Initial conditions:");
    println!("  Total energy:           {:.12e}", initial_energy);
    println!("  Total momentum:         ({:.12e}, {:.12e}, {:.12e})",
             initial_momentum.x, initial_momentum.y, initial_momentum.z);
    println!("  Total angular momentum: ({:.12e}, {:.12e}, {:.12e})",
             initial_angular_momentum.x, initial_angular_momentum.y, initial_angular_momentum.z);
    println!();

    // Velocity Verlet integration loop
    let mut max_energy_violation = 0.0f64;
    let mut max_momentum_violation = 0.0f64;
    let mut max_angmom_violation = 0.0f64;
    let mut violation_count = 0usize;
    let tolerance = 1e-6;

    let mut accelerations = compute_accelerations(&positions, &masses, g);

    for step in 0..steps {
        // Velocity Verlet: half-kick
        for i in 0..n {
            velocities[i] = velocities[i] + accelerations[i] * (0.5 * dt);
        }
        // Drift
        for i in 0..n {
            positions[i] = positions[i] + velocities[i] * dt;
        }
        // Recompute accelerations
        accelerations = compute_accelerations(&positions, &masses, g);
        // Half-kick
        for i in 0..n {
            velocities[i] = velocities[i] + accelerations[i] * (0.5 * dt);
        }

        // Check conservation every 100 steps
        if step % 100 == 0 {
            let energy = compute_total_energy(&positions, &velocities, &masses, g);
            let momentum = compute_total_momentum(&velocities, &masses);
            let angular_momentum = compute_total_angular_momentum(&positions, &velocities, &masses);

            let de = (energy - initial_energy).abs() / initial_energy.abs().max(1e-30);
            let dp = (momentum - initial_momentum).magnitude();
            let dl = (angular_momentum - initial_angular_momentum).magnitude();

            max_energy_violation = max_energy_violation.max(de);
            max_momentum_violation = max_momentum_violation.max(dp);
            max_angmom_violation = max_angmom_violation.max(dl);

            if de > tolerance || dp > tolerance || dl > tolerance {
                violation_count += 1;
                if step % 10000 == 0 {
                    println!("  Step {step}: ΔE/E = {de:.4e}, |Δp| = {dp:.4e}, |ΔL| = {dl:.4e}");
                }
            }
        }
    }

    // Final report
    let final_energy = compute_total_energy(&positions, &velocities, &masses, g);
    let final_momentum = compute_total_momentum(&velocities, &masses);
    let final_angular_momentum = compute_total_angular_momentum(&positions, &velocities, &masses);

    println!("\n=== Conservation Audit Report ===");
    println!("Integrator:       Velocity Verlet (symplectic, order 2)");
    println!("Steps:            {steps}");
    println!("Time step:        {dt}");
    println!("Total time:       {:.1}", steps as f64 * dt);
    println!();
    println!("Energy:");
    println!("  Initial:        {:.12e}", initial_energy);
    println!("  Final:          {:.12e}", final_energy);
    println!("  Max |ΔE/E|:     {:.4e}", max_energy_violation);
    println!("  Status:         {}", if max_energy_violation < tolerance { "PASS ✓" } else { "WARN ⚠" });
    println!();
    println!("Linear Momentum:");
    println!("  Initial:        ({:.6e}, {:.6e}, {:.6e})", initial_momentum.x, initial_momentum.y, initial_momentum.z);
    println!("  Final:          ({:.6e}, {:.6e}, {:.6e})", final_momentum.x, final_momentum.y, final_momentum.z);
    println!("  Max |Δp|:       {:.4e}", max_momentum_violation);
    println!("  Status:         {}", if max_momentum_violation < tolerance { "PASS ✓" } else { "WARN ⚠" });
    println!();
    println!("Angular Momentum:");
    println!("  Initial:        ({:.6e}, {:.6e}, {:.6e})", initial_angular_momentum.x, initial_angular_momentum.y, initial_angular_momentum.z);
    println!("  Final:          ({:.6e}, {:.6e}, {:.6e})", final_angular_momentum.x, final_angular_momentum.y, final_angular_momentum.z);
    println!("  Max |ΔL|:       {:.4e}", max_angmom_violation);
    println!("  Status:         {}", if max_angmom_violation < tolerance { "PASS ✓" } else { "WARN ⚠" });
    println!();
    println!("Violations:       {} checks exceeded tolerance {:.0e}", violation_count, tolerance);
}

fn compute_gravitational_force(pos_i: Vec3, pos_j: Vec3, m_i: f64, m_j: f64, g: f64) -> Vec3 {
    let r = pos_j - pos_i;
    let dist_sq = r.x * r.x + r.y * r.y + r.z * r.z;
    let dist = dist_sq.sqrt();
    let softening = 1e-10;
    let force_mag = g * m_i * m_j / (dist_sq + softening);
    r * (force_mag / (dist + 1e-30))
}

fn compute_accelerations(positions: &[Vec3], masses: &[f64], g: f64) -> Vec<Vec3> {
    let n = positions.len();
    let mut acc = vec![Vec3::ZERO; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let f = compute_gravitational_force(positions[i], positions[j], masses[i], masses[j], g);
            acc[i] = acc[i] + f * (1.0 / masses[i]);
            acc[j] = acc[j] - f * (1.0 / masses[j]);
        }
    }
    acc
}

fn compute_total_energy(positions: &[Vec3], velocities: &[Vec3], masses: &[f64], g: f64) -> f64 {
    let n = positions.len();
    let mut ke = 0.0;
    for i in 0..n {
        ke += 0.5 * masses[i] * velocities[i].magnitude_squared();
    }
    let mut pe = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (positions[j] - positions[i]).magnitude();
            pe -= g * masses[i] * masses[j] / r;
        }
    }
    ke + pe
}

fn compute_total_momentum(velocities: &[Vec3], masses: &[f64]) -> Vec3 {
    let mut p = Vec3::ZERO;
    for i in 0..velocities.len() {
        p = p + velocities[i] * masses[i];
    }
    p
}

fn compute_total_angular_momentum(positions: &[Vec3], velocities: &[Vec3], masses: &[f64]) -> Vec3 {
    let mut l = Vec3::ZERO;
    for i in 0..positions.len() {
        l = l + positions[i].cross(velocities[i] * masses[i]);
    }
    l
}
