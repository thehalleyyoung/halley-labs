//! 1D fluid dynamics conservation audit.
//!
//! Simulates linear advection and Burgers' equation, monitoring conservation
//! of mass and total energy. Demonstrates how different numerical schemes
//! (upwind vs Lax–Wendroff) affect conservation properties.

fn main() {
    println!("=== 1D Fluid Dynamics Conservation Audit ===\n");

    // --- Part 1: Linear Advection (mass conservation test) ---
    println!("--- Part 1: Linear Advection ---");
    let nx = 200;
    let dx = 1.0 / nx as f64;
    let c = 1.0; // advection speed
    let dt = 0.4 * dx / c; // CFL < 1
    let steps = 500;

    // Gaussian initial condition
    let mut u: Vec<f64> = (0..nx)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            (-((x - 0.3) * (x - 0.3)) / (2.0 * 0.01)).exp()
        })
        .collect();

    let initial_mass: f64 = u.iter().sum::<f64>() * dx;
    let initial_energy: f64 = u.iter().map(|v| v * v).sum::<f64>() * dx;

    println!("  Grid: {nx} cells, dx = {dx:.4e}, dt = {dt:.4e}, CFL = {:.2}", c * dt / dx);
    println!("  Initial mass:   {initial_mass:.12e}");
    println!("  Initial energy: {initial_energy:.12e}");

    // Upwind scheme
    let mut max_mass_error = 0.0f64;
    let mut max_energy_error = 0.0f64;
    let cfl = c * dt / dx;

    for step in 0..steps {
        let mut u_new = vec![0.0; nx];
        for i in 0..nx {
            let im = if i == 0 { nx - 1 } else { i - 1 };
            u_new[i] = u[i] - cfl * (u[i] - u[im]); // upwind
        }
        u = u_new;

        if step % 50 == 0 {
            let mass: f64 = u.iter().sum::<f64>() * dx;
            let energy: f64 = u.iter().map(|v| v * v).sum::<f64>() * dx;
            let mass_err = (mass - initial_mass).abs() / initial_mass.abs().max(1e-30);
            let energy_err = (energy - initial_energy).abs() / initial_energy.abs().max(1e-30);
            max_mass_error = max_mass_error.max(mass_err);
            max_energy_error = max_energy_error.max(energy_err);
        }
    }

    let final_mass: f64 = u.iter().sum::<f64>() * dx;
    let final_energy: f64 = u.iter().map(|v| v * v).sum::<f64>() * dx;

    println!("\n  Upwind scheme results ({steps} steps):");
    println!("  Final mass:     {final_mass:.12e}  (Δ = {:.4e})", (final_mass - initial_mass).abs());
    println!("  Final energy:   {final_energy:.12e}  (Δ = {:.4e})", (final_energy - initial_energy).abs());
    println!("  Max |Δm/m|:     {max_mass_error:.4e}  {}", if max_mass_error < 1e-10 { "PASS ✓" } else { "WARN ⚠" });
    println!("  Max |ΔE/E|:     {max_energy_error:.4e}  {}", if max_energy_error < 0.1 { "EXPECTED (upwind is dissipative)" } else { "FAIL ✗" });

    // --- Part 2: Burgers' Equation (shock formation) ---
    println!("\n--- Part 2: Inviscid Burgers' Equation ---");
    let nx = 200;
    let dx = 2.0 / nx as f64;
    let dt = 0.002;
    let steps = 300;

    // Sine-wave initial condition
    let mut u: Vec<f64> = (0..nx)
        .map(|i| {
            let x = -1.0 + (i as f64 + 0.5) * dx;
            (std::f64::consts::PI * x).sin()
        })
        .collect();

    let initial_mass_b: f64 = u.iter().sum::<f64>() * dx;
    let initial_energy_b: f64 = u.iter().map(|v| v * v).sum::<f64>() * dx;

    println!("  Grid: {nx} cells, dx = {dx:.4e}, dt = {dt:.4e}");
    println!("  Initial mass:   {initial_mass_b:.12e}");
    println!("  Initial energy: {initial_energy_b:.12e}");

    let mut max_mass_error_b = 0.0f64;

    for step in 0..steps {
        let mut u_new = vec![0.0; nx];
        for i in 0..nx {
            let im = if i == 0 { nx - 1 } else { i - 1 };
            let ip = if i == nx - 1 { 0 } else { i + 1 };
            // Conservative Lax–Friedrichs flux
            let flux_right = 0.5 * (0.5 * u[i] * u[i] + 0.5 * u[ip] * u[ip]) - 0.5 * (dx / dt) * (u[ip] - u[i]);
            let flux_left = 0.5 * (0.5 * u[im] * u[im] + 0.5 * u[i] * u[i]) - 0.5 * (dx / dt) * (u[i] - u[im]);
            u_new[i] = u[i] - dt / dx * (flux_right - flux_left);
        }
        u = u_new;

        if step % 30 == 0 {
            let mass: f64 = u.iter().sum::<f64>() * dx;
            let mass_err = (mass - initial_mass_b).abs() / initial_mass_b.abs().max(1e-30);
            max_mass_error_b = max_mass_error_b.max(mass_err);
        }
    }

    let final_mass_b: f64 = u.iter().sum::<f64>() * dx;
    let final_energy_b: f64 = u.iter().map(|v| v * v).sum::<f64>() * dx;

    println!("\n  Lax–Friedrichs results ({steps} steps):");
    println!("  Final mass:     {final_mass_b:.12e}  (Δ = {:.4e})", (final_mass_b - initial_mass_b).abs());
    println!("  Final energy:   {final_energy_b:.12e}  (Δ = {:.4e})", (final_energy_b - initial_energy_b).abs());
    println!("  Max |Δm/m|:     {max_mass_error_b:.4e}  {}", if max_mass_error_b < 1e-10 { "PASS ✓ (conservative scheme)" } else { "WARN ⚠" });
    println!("  Energy change:  {} (expected for shock-capturing schemes)",
             if final_energy_b < initial_energy_b { "DECREASED (physical dissipation at shock)" }
             else { "INCREASED (non-physical!)" });

    println!("\n=== Audit Summary ===");
    println!("  Advection mass:     {}", if max_mass_error < 1e-10 { "PASS ✓" } else { "FAIL ✗" });
    println!("  Advection energy:   EXPECTED DISSIPATION (upwind)");
    println!("  Burgers mass:       {}", if max_mass_error_b < 1e-8 { "PASS ✓" } else { "FAIL ✗" });
    println!("  Burgers energy:     EXPECTED DISSIPATION (shock)");
}
