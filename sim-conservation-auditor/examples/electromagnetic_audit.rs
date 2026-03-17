//! Electromagnetic simulation conservation audit.
//!
//! Simulates charged particles in electromagnetic fields and monitors
//! conservation of energy, canonical momentum, and magnetic moment
//! (adiabatic invariant).

use sim_types::Vec3;

fn main() {
    println!("=== Electromagnetic Conservation Audit ===\n");

    // --- Part 1: Cyclotron Motion (uniform B-field) ---
    println!("--- Part 1: Cyclotron Motion ---");
    let q = 1.0; // charge
    let m = 1.0; // mass
    let b_field = Vec3::new(0.0, 0.0, 1.0); // uniform B along z
    let dt = 0.01;
    let steps = 100_000;

    // Initial conditions: circular orbit in x-y plane
    let omega_c = q * b_field.z / m; // cyclotron frequency
    let v_perp = 1.0; // perpendicular velocity
    let r_l = m * v_perp / (q * b_field.z); // Larmor radius

    let mut pos = Vec3::new(r_l, 0.0, 0.0);
    let mut vel = Vec3::new(0.0, v_perp, 0.1); // small v_z for helical motion

    let initial_energy = 0.5 * m * vel.magnitude_squared();
    let initial_magnetic_moment = 0.5 * m * v_perp * v_perp / b_field.z.abs();
    let initial_vz = vel.z;

    println!("  Charge: {q}, Mass: {m}");
    println!("  B = (0, 0, {:.1}), ωc = {omega_c:.4}", b_field.z);
    println!("  Larmor radius: {r_l:.4}");
    println!("  Initial energy: {initial_energy:.12e}");
    println!("  Initial μ (mag moment): {initial_magnetic_moment:.12e}");
    println!("  dt = {dt}, steps = {steps}");

    let mut max_energy_err = 0.0f64;
    let mut max_mu_err = 0.0f64;

    // Boris integrator
    for step in 0..steps {
        // Boris push algorithm
        let (new_pos, new_vel) = boris_push(pos, vel, q, m, &Vec3::ZERO, &b_field, dt);
        pos = new_pos;
        vel = new_vel;

        if step % 1000 == 0 {
            let energy = 0.5 * m * vel.magnitude_squared();
            let v_perp_sq = vel.x * vel.x + vel.y * vel.y;
            let mu = 0.5 * m * v_perp_sq / b_field.z.abs();

            let de = (energy - initial_energy).abs() / initial_energy;
            let dmu = (mu - initial_magnetic_moment).abs() / initial_magnetic_moment;

            max_energy_err = max_energy_err.max(de);
            max_mu_err = max_mu_err.max(dmu);
        }
    }

    let final_energy = 0.5 * m * vel.magnitude_squared();
    println!("\n  Boris integrator results:");
    println!("  Final energy:      {final_energy:.12e}");
    println!("  Max |ΔE/E|:        {max_energy_err:.4e}  {}", if max_energy_err < 1e-10 { "PASS ✓" } else { "WARN ⚠" });
    println!("  Max |Δμ/μ|:        {max_mu_err:.4e}  {}", if max_mu_err < 1e-4 { "PASS ✓" } else { "WARN ⚠" });
    println!("  v_z preserved:     {:.12e} → {:.12e}", initial_vz, vel.z);

    // --- Part 2: E×B Drift ---
    println!("\n--- Part 2: E×B Drift ---");
    let e_field = Vec3::new(0.1, 0.0, 0.0); // E along x
    let b_field2 = Vec3::new(0.0, 0.0, 1.0); // B along z
    // Expected drift: v_d = E×B / B² = (0, -0.1, 0)
    let expected_drift = Vec3::new(0.0, -e_field.x / b_field2.z, 0.0);
    let dt2 = 0.005;
    let steps2 = 50_000;

    let mut pos2 = Vec3::new(0.0, 0.0, 0.0);
    let mut vel2 = Vec3::new(0.0, 1.0, 0.0);

    let initial_total_energy = 0.5 * m * vel2.magnitude_squared() + q * e_field.x * pos2.x;

    for _step in 0..steps2 {
        let (np, nv) = boris_push(pos2, vel2, q, m, &e_field, &b_field2, dt2);
        pos2 = np;
        vel2 = nv;
    }

    let total_time = steps2 as f64 * dt2;
    let measured_drift_vy = pos2.y / total_time;
    let drift_error = (measured_drift_vy - expected_drift.y).abs() / expected_drift.y.abs();

    println!("  E = ({:.1}, 0, 0), B = (0, 0, {:.1})", e_field.x, b_field2.z);
    println!("  Expected drift: v_d = (0, {:.4}, 0)", expected_drift.y);
    println!("  Measured drift: v_y_avg = {measured_drift_vy:.6}");
    println!("  Drift error:    {drift_error:.4e}  {}", if drift_error < 0.01 { "PASS ✓" } else { "WARN ⚠" });

    // --- Part 3: Coulomb Scattering (energy conservation) ---
    println!("\n--- Part 3: Coulomb Scattering ---");
    let k_coulomb = 1.0;
    let q1 = 1.0;
    let q2 = 1.0;
    let m1 = 1.0;
    let m2 = 1000.0; // heavy target (approx fixed)

    let mut pos_proj = Vec3::new(-10.0, 1.0, 0.0); // impact parameter b=1
    let mut vel_proj = Vec3::new(1.0, 0.0, 0.0);
    let pos_target = Vec3::new(0.0, 0.0, 0.0);

    let dt3 = 0.001;
    let steps3 = 40_000;

    let initial_ke = 0.5 * m1 * vel_proj.magnitude_squared();
    let initial_r = (pos_proj - pos_target).magnitude();
    let initial_pe = k_coulomb * q1 * q2 / initial_r;
    let initial_total = initial_ke + initial_pe;

    println!("  Projectile: q={q1}, m={m1}, v=(1,0,0), b=1.0");
    println!("  Target:     q={q2}, m={m2} (fixed)");
    println!("  Initial E:  {initial_total:.12e}");

    let mut max_coulomb_energy_err = 0.0f64;

    for step in 0..steps3 {
        // Velocity Verlet for Coulomb force
        let r_vec = pos_proj - pos_target;
        let r = r_vec.magnitude();
        let force = r_vec * (-k_coulomb * q1 * q2 / (r * r * r));
        let acc = force * (1.0 / m1);

        vel_proj = vel_proj + acc * (0.5 * dt3);
        pos_proj = pos_proj + vel_proj * dt3;

        let r_vec2 = pos_proj - pos_target;
        let r2 = r_vec2.magnitude();
        let force2 = r_vec2 * (-k_coulomb * q1 * q2 / (r2 * r2 * r2));
        let acc2 = force2 * (1.0 / m1);
        vel_proj = vel_proj + acc2 * (0.5 * dt3);

        if step % 1000 == 0 {
            let ke = 0.5 * m1 * vel_proj.magnitude_squared();
            let pe = k_coulomb * q1 * q2 / (pos_proj - pos_target).magnitude();
            let total = ke + pe;
            let err = (total - initial_total).abs() / initial_total.abs().max(1e-30);
            max_coulomb_energy_err = max_coulomb_energy_err.max(err);
        }
    }

    let final_ke = 0.5 * m1 * vel_proj.magnitude_squared();
    let final_r = (pos_proj - pos_target).magnitude();
    let final_pe = k_coulomb * q1 * q2 / final_r;
    let final_total = final_ke + final_pe;

    let scattering_angle = vel_proj.y.atan2(vel_proj.x).to_degrees();
    // Rutherford: tan(θ/2) = k*q1*q2 / (m*v²*b) = 1/(1*1*1) = 1 → θ ≈ 90°
    println!("\n  Coulomb scattering results:");
    println!("  Final energy:      {final_total:.12e}");
    println!("  Max |ΔE/E|:        {max_coulomb_energy_err:.4e}  {}", if max_coulomb_energy_err < 1e-6 { "PASS ✓" } else { "WARN ⚠" });
    println!("  Scattering angle:  {scattering_angle:.2}°");

    println!("\n=== Electromagnetic Audit Summary ===");
    println!("  Cyclotron energy:    {}", if max_energy_err < 1e-10 { "PASS ✓ (Boris preserves energy)" } else { "WARN ⚠" });
    println!("  Magnetic moment:     {}", if max_mu_err < 1e-4 { "PASS ✓ (adiabatic invariant)" } else { "WARN ⚠" });
    println!("  E×B drift:           {}", if drift_error < 0.01 { "PASS ✓" } else { "WARN ⚠" });
    println!("  Coulomb energy:      {}", if max_coulomb_energy_err < 1e-6 { "PASS ✓ (Verlet symplectic)" } else { "WARN ⚠" });
}

/// Boris push algorithm for charged particle motion in electromagnetic fields.
/// This is a volume-preserving (though not strictly symplectic) integrator
/// that exactly conserves energy in a pure magnetic field.
fn boris_push(pos: Vec3, vel: Vec3, q: f64, m: f64, e: &Vec3, b: &Vec3, dt: f64) -> (Vec3, Vec3) {
    let qdt2m = q * dt / (2.0 * m);

    // Half electric acceleration
    let v_minus = vel + *e * qdt2m;

    // Magnetic rotation
    let t_vec = *b * qdt2m;
    let t_mag_sq = t_vec.magnitude_squared();
    let s_vec = t_vec * (2.0 / (1.0 + t_mag_sq));

    let v_prime = v_minus + v_minus.cross(t_vec);
    let v_plus = v_minus + v_prime.cross(s_vec);

    // Half electric acceleration
    let new_vel = v_plus + *e * qdt2m;
    let new_pos = pos + new_vel * dt;

    (new_pos, new_vel)
}
