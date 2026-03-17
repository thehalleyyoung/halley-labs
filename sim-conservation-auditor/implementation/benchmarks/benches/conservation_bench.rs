//! Criterion benchmarks for conservation law computation performance.
//!
//! Measures the throughput and latency of:
//! - Conservation quantity computation (energy, momentum, angular momentum)
//! - Violation detection (CUSUM, Page-Hinkley, statistical tests)
//! - Integrator steps (Verlet, Yoshida4, RK4)
//! - Repair operations (projection, velocity scaling)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

/// Benchmark conservation law computation on N-body systems of varying size.
fn bench_conservation_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("conservation_computation");

    for n in [10, 100, 1000, 5000].iter() {
        // Generate random N-body state
        let (positions, velocities, masses) = generate_nbody_state(*n);

        group.bench_with_input(BenchmarkId::new("energy", n), n, |b, _| {
            b.iter(|| {
                black_box(compute_total_energy(&positions, &velocities, &masses));
            });
        });

        group.bench_with_input(BenchmarkId::new("momentum", n), n, |b, _| {
            b.iter(|| {
                black_box(compute_total_momentum(&velocities, &masses));
            });
        });

        group.bench_with_input(BenchmarkId::new("angular_momentum", n), n, |b, _| {
            b.iter(|| {
                black_box(compute_total_angular_momentum(&positions, &velocities, &masses));
            });
        });
    }

    group.finish();
}

/// Benchmark integrator step performance on 2-body Kepler problem.
fn bench_integrator_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrator_step");
    let dt = 0.001;

    // Initial Kepler orbit
    let q0 = [0.4, 0.0];
    let p0 = [0.0, 2.0];

    group.bench_function("forward_euler", |b| {
        b.iter(|| {
            let mut q = q0;
            let mut p = p0;
            for _ in 0..100 {
                let f = kepler_force_2d(q);
                let qn = [q[0] + p[0] * dt, q[1] + p[1] * dt];
                let pn = [p[0] + f[0] * dt, p[1] + f[1] * dt];
                q = qn;
                p = pn;
            }
            black_box((q, p));
        });
    });

    group.bench_function("velocity_verlet", |b| {
        b.iter(|| {
            let mut q = q0;
            let mut p = p0;
            let mut f = kepler_force_2d(q);
            for _ in 0..100 {
                p = [p[0] + f[0] * 0.5 * dt, p[1] + f[1] * 0.5 * dt];
                q = [q[0] + p[0] * dt, q[1] + p[1] * dt];
                f = kepler_force_2d(q);
                p = [p[0] + f[0] * 0.5 * dt, p[1] + f[1] * 0.5 * dt];
            }
            black_box((q, p));
        });
    });

    group.bench_function("yoshida4", |b| {
        b.iter(|| {
            let mut q = q0;
            let mut p = p0;
            let cbrt2: f64 = 2.0f64.cbrt();
            let w1 = 1.0 / (2.0 - cbrt2);
            let w0 = -cbrt2 / (2.0 - cbrt2);
            let c_coeff = [w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0];
            let d_coeff = [w1, w0, w1];
            for _ in 0..100 {
                q = [q[0] + p[0] * c_coeff[0] * dt, q[1] + p[1] * c_coeff[0] * dt];
                for i in 0..3 {
                    let f = kepler_force_2d(q);
                    p = [p[0] + f[0] * d_coeff[i] * dt, p[1] + f[1] * d_coeff[i] * dt];
                    q = [q[0] + p[0] * c_coeff[i + 1] * dt, q[1] + p[1] * c_coeff[i + 1] * dt];
                }
            }
            black_box((q, p));
        });
    });

    group.bench_function("rk4", |b| {
        b.iter(|| {
            let mut q = q0;
            let mut p = p0;
            for _ in 0..100 {
                let k1q = p;
                let k1p = kepler_force_2d(q);
                let k2q = [p[0] + k1p[0]*0.5*dt, p[1] + k1p[1]*0.5*dt];
                let k2p = kepler_force_2d([q[0]+k1q[0]*0.5*dt, q[1]+k1q[1]*0.5*dt]);
                let k3q = [p[0] + k2p[0]*0.5*dt, p[1] + k2p[1]*0.5*dt];
                let k3p = kepler_force_2d([q[0]+k2q[0]*0.5*dt, q[1]+k2q[1]*0.5*dt]);
                let k4q = [p[0] + k3p[0]*dt, p[1] + k3p[1]*dt];
                let k4p = kepler_force_2d([q[0]+k3q[0]*dt, q[1]+k3q[1]*dt]);
                q = [
                    q[0] + (k1q[0]+2.0*k2q[0]+2.0*k3q[0]+k4q[0])*dt/6.0,
                    q[1] + (k1q[1]+2.0*k2q[1]+2.0*k3q[1]+k4q[1])*dt/6.0,
                ];
                p = [
                    p[0] + (k1p[0]+2.0*k2p[0]+2.0*k3p[0]+k4p[0])*dt/6.0,
                    p[1] + (k1p[1]+2.0*k2p[1]+2.0*k3p[1]+k4p[1])*dt/6.0,
                ];
            }
            black_box((q, p));
        });
    });

    group.finish();
}

/// Benchmark violation detection algorithms.
fn bench_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("violation_detection");

    // Generate synthetic energy time series with drift
    for n in [1000, 10000, 100000].iter() {
        let series = generate_drifting_series(*n, 1e-8);

        group.bench_with_input(BenchmarkId::new("cusum", n), n, |b, _| {
            b.iter(|| {
                black_box(cusum_detect(&series, 5.0, 0.5));
            });
        });

        group.bench_with_input(BenchmarkId::new("page_hinkley", n), n, |b, _| {
            b.iter(|| {
                black_box(page_hinkley_detect(&series, 50.0, 0.01));
            });
        });

        group.bench_with_input(BenchmarkId::new("moving_zscore", n), n, |b, _| {
            b.iter(|| {
                black_box(moving_zscore(&series, 100, 3.0));
            });
        });
    }

    group.finish();
}

/// Benchmark monitoring overhead (conservation check cost relative to integration cost).
fn bench_monitoring_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("monitoring_overhead");
    let n = 100; // 100-body system
    let dt = 0.001;
    let (positions, velocities, masses) = generate_nbody_state(n);

    group.bench_function("integration_only", |b| {
        b.iter(|| {
            let mut pos = positions.clone();
            let mut vel = velocities.clone();
            let mut acc = compute_nbody_accelerations(&pos, &masses);
            for _ in 0..100 {
                for i in 0..n {
                    vel[i] = [vel[i][0]+acc[i][0]*0.5*dt, vel[i][1]+acc[i][1]*0.5*dt, vel[i][2]+acc[i][2]*0.5*dt];
                }
                for i in 0..n {
                    pos[i] = [pos[i][0]+vel[i][0]*dt, pos[i][1]+vel[i][1]*dt, pos[i][2]+vel[i][2]*dt];
                }
                acc = compute_nbody_accelerations(&pos, &masses);
                for i in 0..n {
                    vel[i] = [vel[i][0]+acc[i][0]*0.5*dt, vel[i][1]+acc[i][1]*0.5*dt, vel[i][2]+acc[i][2]*0.5*dt];
                }
            }
            black_box((&pos, &vel));
        });
    });

    group.bench_function("integration_plus_monitoring", |b| {
        b.iter(|| {
            let mut pos = positions.clone();
            let mut vel = velocities.clone();
            let mut acc = compute_nbody_accelerations(&pos, &masses);
            for step in 0..100 {
                for i in 0..n {
                    vel[i] = [vel[i][0]+acc[i][0]*0.5*dt, vel[i][1]+acc[i][1]*0.5*dt, vel[i][2]+acc[i][2]*0.5*dt];
                }
                for i in 0..n {
                    pos[i] = [pos[i][0]+vel[i][0]*dt, pos[i][1]+vel[i][1]*dt, pos[i][2]+vel[i][2]*dt];
                }
                acc = compute_nbody_accelerations(&pos, &masses);
                for i in 0..n {
                    vel[i] = [vel[i][0]+acc[i][0]*0.5*dt, vel[i][1]+acc[i][1]*0.5*dt, vel[i][2]+acc[i][2]*0.5*dt];
                }
                // Conservation check every 10 steps
                if step % 10 == 0 {
                    let _e = compute_total_energy(&pos, &vel, &masses);
                    let _p = compute_total_momentum_arr(&vel, &masses);
                    let _l = compute_total_angular_momentum_arr(&pos, &vel, &masses);
                }
            }
            black_box((&pos, &vel));
        });
    });

    group.finish();
}

// --- Helper functions ---

fn generate_nbody_state(n: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>) {
    let mut positions = Vec::with_capacity(n);
    let mut velocities = Vec::with_capacity(n);
    let mut masses = Vec::with_capacity(n);
    // Deterministic pseudo-random (LCG)
    let mut seed: u64 = 12345;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    };
    for _ in 0..n {
        positions.push([lcg(&mut seed) * 10.0, lcg(&mut seed) * 10.0, lcg(&mut seed) * 10.0]);
        velocities.push([lcg(&mut seed) * 0.1, lcg(&mut seed) * 0.1, lcg(&mut seed) * 0.1]);
        masses.push(1.0 + lcg(&mut seed).abs());
    }
    (positions, velocities, masses)
}

fn compute_total_energy(pos: &[[f64; 3]], vel: &[[f64; 3]], masses: &[f64]) -> f64 {
    let n = pos.len();
    let mut ke = 0.0;
    for i in 0..n {
        ke += 0.5 * masses[i] * (vel[i][0]*vel[i][0] + vel[i][1]*vel[i][1] + vel[i][2]*vel[i][2]);
    }
    let mut pe = 0.0;
    for i in 0..n {
        for j in (i+1)..n {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let r = (dx*dx + dy*dy + dz*dz).sqrt();
            pe -= masses[i] * masses[j] / r;
        }
    }
    ke + pe
}

fn compute_total_momentum(vel: &[[f64; 3]], masses: &[f64]) -> [f64; 3] {
    let mut p = [0.0, 0.0, 0.0];
    for i in 0..vel.len() {
        p[0] += masses[i] * vel[i][0];
        p[1] += masses[i] * vel[i][1];
        p[2] += masses[i] * vel[i][2];
    }
    p
}

fn compute_total_momentum_arr(vel: &[[f64; 3]], masses: &[f64]) -> [f64; 3] {
    compute_total_momentum(vel, masses)
}

fn compute_total_angular_momentum(pos: &[[f64; 3]], vel: &[[f64; 3]], masses: &[f64]) -> [f64; 3] {
    let mut l = [0.0, 0.0, 0.0];
    for i in 0..pos.len() {
        let px = masses[i] * vel[i][0];
        let py = masses[i] * vel[i][1];
        let pz = masses[i] * vel[i][2];
        l[0] += pos[i][1] * pz - pos[i][2] * py;
        l[1] += pos[i][2] * px - pos[i][0] * pz;
        l[2] += pos[i][0] * py - pos[i][1] * px;
    }
    l
}

fn compute_total_angular_momentum_arr(pos: &[[f64; 3]], vel: &[[f64; 3]], masses: &[f64]) -> [f64; 3] {
    compute_total_angular_momentum(pos, vel, masses)
}

fn compute_nbody_accelerations(pos: &[[f64; 3]], masses: &[f64]) -> Vec<[f64; 3]> {
    let n = pos.len();
    let mut acc = vec![[0.0, 0.0, 0.0]; n];
    let softening = 1e-4;
    for i in 0..n {
        for j in (i+1)..n {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let r2 = dx*dx + dy*dy + dz*dz + softening*softening;
            let r = r2.sqrt();
            let r3 = r2 * r;
            let fx = dx / r3;
            let fy = dy / r3;
            let fz = dz / r3;
            acc[i][0] += masses[j] * fx;
            acc[i][1] += masses[j] * fy;
            acc[i][2] += masses[j] * fz;
            acc[j][0] -= masses[i] * fx;
            acc[j][1] -= masses[i] * fy;
            acc[j][2] -= masses[i] * fz;
        }
    }
    acc
}

fn kepler_force_2d(q: [f64; 2]) -> [f64; 2] {
    let r2 = q[0]*q[0] + q[1]*q[1];
    let r = r2.sqrt();
    let r3 = r2 * r;
    [-q[0]/r3, -q[1]/r3]
}

fn generate_drifting_series(n: usize, drift_rate: f64) -> Vec<f64> {
    let mut series = Vec::with_capacity(n);
    let mut seed: u64 = 42;
    for i in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((seed >> 33) as f64 / u32::MAX as f64 - 0.5) * 1e-10;
        series.push(-0.5 + i as f64 * drift_rate + noise);
    }
    series
}

fn cusum_detect(series: &[f64], threshold: f64, drift: f64) -> Option<usize> {
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let mut s_pos = 0.0;
    let mut s_neg = 0.0;
    for (i, &x) in series.iter().enumerate() {
        s_pos = (s_pos + (x - mean) - drift).max(0.0);
        s_neg = (s_neg - (x - mean) - drift).max(0.0);
        if s_pos > threshold || s_neg > threshold {
            return Some(i);
        }
    }
    None
}

fn page_hinkley_detect(series: &[f64], threshold: f64, delta: f64) -> Option<usize> {
    let mut sum = 0.0;
    let mut min_sum = 0.0;
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    for (i, &x) in series.iter().enumerate() {
        sum += x - mean - delta;
        if sum < min_sum { min_sum = sum; }
        if sum - min_sum > threshold {
            return Some(i);
        }
    }
    None
}

fn moving_zscore(series: &[f64], window: usize, threshold: f64) -> Vec<usize> {
    let mut anomalies = Vec::new();
    if series.len() < window { return anomalies; }
    for i in window..series.len() {
        let win = &series[i-window..i];
        let mean = win.iter().sum::<f64>() / window as f64;
        let var = win.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / window as f64;
        let std = var.sqrt();
        if std > 0.0 && ((series[i] - mean) / std).abs() > threshold {
            anomalies.push(i);
        }
    }
    anomalies
}

criterion_group!(
    benches,
    bench_conservation_computation,
    bench_integrator_steps,
    bench_detection,
    bench_monitoring_overhead,
);
criterion_main!(benches);
