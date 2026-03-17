//! Automatic conservation law detector: discover conserved quantities from trace data.
//!
//! Given a simulation trace with no a priori knowledge of which quantities are conserved,
//! this module uses SVD on the matrix of time-differenced observables to find approximate
//! invariants. Columns with near-zero singular values correspond to conserved linear
//! combinations.

use serde::{Deserialize, Serialize};
use sim_types::{SimulationState, TimeSeries};

// ─── Result Types ───────────────────────────────────────────────────────────

/// A detected approximate invariant from trace data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedInvariant {
    /// Human-readable label for the invariant.
    pub label: String,
    /// Coefficients in the observable basis (kinetic energy, momentum components, etc.).
    pub coefficients: Vec<f64>,
    /// Names of the observable basis functions.
    pub observable_names: Vec<String>,
    /// Maximum drift of this invariant over the trace.
    pub max_drift: f64,
    /// Singular value associated with this invariant (smaller = better conserved).
    pub singular_value: f64,
    /// Time series of the invariant.
    pub series: TimeSeries,
}

// ─── Observable Computation ─────────────────────────────────────────────────

/// Standard observable functions evaluated on a simulation state.
///
/// Returns a vector of (name, value) pairs for each observable.
fn compute_observables(state: &SimulationState) -> Vec<(&'static str, f64)> {
    let mut obs = Vec::new();

    // Kinetic energy
    let ke: f64 = state
        .particles
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.magnitude_squared())
        .sum();
    obs.push(("KineticEnergy", ke));

    // Momentum components
    let (px, py, pz) = state.particles.iter().fold((0.0, 0.0, 0.0), |acc, p| {
        (
            acc.0 + p.mass * p.velocity.x,
            acc.1 + p.mass * p.velocity.y,
            acc.2 + p.mass * p.velocity.z,
        )
    });
    obs.push(("MomentumX", px));
    obs.push(("MomentumY", py));
    obs.push(("MomentumZ", pz));

    // Angular momentum components
    let (lx, ly, lz) = state.particles.iter().fold((0.0, 0.0, 0.0), |acc, p| {
        let r = p.position;
        let mv = sim_types::Vec3::new(
            p.mass * p.velocity.x,
            p.mass * p.velocity.y,
            p.mass * p.velocity.z,
        );
        (
            acc.0 + (r.y * mv.z - r.z * mv.y),
            acc.1 + (r.z * mv.x - r.x * mv.z),
            acc.2 + (r.x * mv.y - r.y * mv.x),
        )
    });
    obs.push(("AngMomentumX", lx));
    obs.push(("AngMomentumY", ly));
    obs.push(("AngMomentumZ", lz));

    // Total mass
    let mass: f64 = state.particles.iter().map(|p| p.mass).sum();
    obs.push(("TotalMass", mass));

    // Total charge
    let charge: f64 = state.particles.iter().map(|p| p.charge).sum();
    obs.push(("TotalCharge", charge));

    obs
}

// ─── SVD via Jacobi rotations (self-contained, no external dependency) ──────

/// Compute the thin SVD of an m×n matrix A (m >= n) using one-sided Jacobi rotations.
/// Returns (singular_values, V) where A ≈ U · diag(σ) · V^T.
/// We only need σ and V for invariant detection.
fn jacobi_svd(a: &[Vec<f64>], max_iter: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 {
        return (vec![], vec![]);
    }
    let n = a[0].len();

    // Work on A^T A to get eigenvalues (singular values squared) and eigenvectors V
    // For small n this is efficient.
    let mut ata = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..m {
                s += a[k][i] * a[k][j];
            }
            ata[i][j] = s;
        }
    }

    // Jacobi eigenvalue algorithm on the symmetric matrix A^T A
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..max_iter {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let off_diag = ata[p][q].abs();
                if off_diag < 1e-15 {
                    continue;
                }
                converged = false;

                let tau = (ata[q][q] - ata[p][p]) / (2.0 * ata[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply Givens rotation to ata
                let app = ata[p][p];
                let aqq = ata[q][q];
                let apq = ata[p][q];

                ata[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                ata[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                ata[p][q] = 0.0;
                ata[q][p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = ata[r][p];
                    let arq = ata[r][q];
                    ata[r][p] = c * arp - s * arq;
                    ata[p][r] = ata[r][p];
                    ata[r][q] = s * arp + c * arq;
                    ata[q][r] = ata[r][q];
                }

                // Update V
                for r in 0..n {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Extract singular values (sqrt of eigenvalues of A^T A)
    let singular_values: Vec<f64> = (0..n).map(|i| ata[i][i].abs().sqrt()).collect();

    (singular_values, v)
}

// ─── Detector ───────────────────────────────────────────────────────────────

/// Automatically detects conserved quantities from simulation trace data.
///
/// Constructs a matrix of time-differenced observables, applies SVD, and
/// identifies directions with near-zero singular values as approximate invariants.
pub struct AutoLawDetector {
    /// Singular-value threshold below which a direction is considered conserved.
    threshold: f64,
}

impl AutoLawDetector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn default_detector() -> Self {
        Self { threshold: 1e-6 }
    }

    /// Detect conserved quantities from a simulation trace.
    pub fn detect(&self, states: &[SimulationState]) -> Vec<DetectedInvariant> {
        if states.len() < 3 {
            return Vec::new();
        }

        // Compute observables at each timestep
        let obs_at_steps: Vec<Vec<(&str, f64)>> =
            states.iter().map(|s| compute_observables(s)).collect();

        let n_obs = obs_at_steps[0].len();
        let n_steps = obs_at_steps.len();
        let obs_names: Vec<String> = obs_at_steps[0].iter().map(|(n, _)| n.to_string()).collect();

        // Build the time-difference matrix: ΔO[i][j] = O_j(t_{i+1}) − O_j(t_i)
        let n_diffs = n_steps - 1;
        let mut diff_matrix: Vec<Vec<f64>> = Vec::with_capacity(n_diffs);
        for i in 0..n_diffs {
            let row: Vec<f64> = (0..n_obs)
                .map(|j| obs_at_steps[i + 1][j].1 - obs_at_steps[i][j].1)
                .collect();
            diff_matrix.push(row);
        }

        // Normalize columns for numerical stability
        let mut col_norms = vec![0.0_f64; n_obs];
        for row in &diff_matrix {
            for (j, &v) in row.iter().enumerate() {
                col_norms[j] += v * v;
            }
        }
        for norm in &mut col_norms {
            *norm = norm.sqrt().max(1e-30);
        }
        let mut normalized = diff_matrix.clone();
        for row in &mut normalized {
            for (j, v) in row.iter_mut().enumerate() {
                *v /= col_norms[j];
            }
        }

        // SVD
        let (singular_values, v_matrix) = jacobi_svd(&normalized, 200);

        // Find invariants: columns of V with near-zero singular values
        let mut invariants = Vec::new();
        for (k, &sv) in singular_values.iter().enumerate() {
            if sv < self.threshold {
                // The k-th right singular vector defines the invariant direction
                let coeffs: Vec<f64> = (0..n_obs).map(|j| v_matrix[j][k] * col_norms[j]).collect();

                // Evaluate the invariant across the trace
                let series_values: Vec<f64> = obs_at_steps
                    .iter()
                    .map(|obs| {
                        coeffs
                            .iter()
                            .zip(obs.iter())
                            .map(|(c, (_, v))| c * v)
                            .sum()
                    })
                    .collect();
                let times: Vec<f64> = states.iter().map(|s| s.time).collect();
                let series = TimeSeries::new(times, series_values.clone());

                let initial = series_values[0];
                let max_drift = series_values
                    .iter()
                    .map(|v| (v - initial).abs())
                    .fold(0.0_f64, f64::max);

                // Identify the dominant observable
                let max_coeff_idx = coeffs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let label = format!("Invariant_{} (dominant: {})", k, obs_names[max_coeff_idx]);

                invariants.push(DetectedInvariant {
                    label,
                    coefficients: coeffs,
                    observable_names: obs_names.clone(),
                    max_drift,
                    singular_value: sv,
                    series,
                });
            }
        }

        // Sort by singular value (best conserved first)
        invariants.sort_by(|a, b| {
            a.singular_value
                .partial_cmp(&b.singular_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        invariants
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{Particle, Vec3};

    /// Free particles — KE, mass, charge, and all momentum/angular-momentum
    /// components should be detected as conserved.
    fn free_particle_trace(n: usize) -> Vec<SimulationState> {
        let dt = 0.01;
        (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                let p = Particle::new(
                    2.0,
                    Vec3::new(1.0 + 3.0 * t, 2.0 - 1.0 * t, 0.5 * t),
                    Vec3::new(3.0, -1.0, 0.5),
                );
                SimulationState::new(vec![p], t)
            })
            .collect()
    }

    #[test]
    fn test_detect_free_particle_invariants() {
        let states = free_particle_trace(200);
        let detector = AutoLawDetector::new(1e-6);
        let invariants = detector.detect(&states);

        // Should find several invariants (KE, px, py, pz, mass, charge are all constant)
        assert!(
            !invariants.is_empty(),
            "should detect at least one invariant for free particle"
        );

        // All detected invariants should have tiny drift
        for inv in &invariants {
            assert!(
                inv.max_drift < 1e-6,
                "invariant {} has drift {}, expected < 1e-6",
                inv.label,
                inv.max_drift
            );
        }
    }

    #[test]
    fn test_detect_with_drift() {
        // A system where energy drifts but mass is conserved
        let dt = 0.01;
        let states: Vec<SimulationState> = (0..100)
            .map(|i| {
                let t = i as f64 * dt;
                let speed = 1.0 + 0.1 * t; // accelerating → KE not conserved
                let p = Particle::new(1.0, Vec3::new(t, 0.0, 0.0), Vec3::new(speed, 0.0, 0.0));
                SimulationState::new(vec![p], t)
            })
            .collect();

        let detector = AutoLawDetector::new(1e-4);
        let invariants = detector.detect(&states);

        // Mass and charge should still be detected
        // Not all observables should be invariant (KE is drifting)
        let n_obs = compute_observables(&states[0]).len();
        assert!(
            invariants.len() < n_obs,
            "should not detect all observables as invariant when KE drifts"
        );
    }

    #[test]
    fn test_empty_trace() {
        let detector = AutoLawDetector::default_detector();
        let invariants = detector.detect(&[]);
        assert!(invariants.is_empty());
    }

    #[test]
    fn test_short_trace() {
        let states = vec![SimulationState::new(
            vec![Particle::new(1.0, Vec3::ZERO, Vec3::X)],
            0.0,
        )];
        let detector = AutoLawDetector::default_detector();
        let invariants = detector.detect(&states);
        assert!(invariants.is_empty());
    }

    #[test]
    fn test_jacobi_svd_identity() {
        // SVD of identity should give all singular values = 1
        let id = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (sv, _) = jacobi_svd(&id, 100);
        for s in &sv {
            assert!(
                (s - 1.0).abs() < 1e-10,
                "singular value of identity should be 1, got {}",
                s
            );
        }
    }
}
