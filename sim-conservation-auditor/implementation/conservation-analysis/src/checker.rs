//! Conservation law checker: given a simulation trace, verify conservation laws hold.
//!
//! For each registered law and each timestep, computes the conserved quantity,
//! measures drift from the initial value, and reports maximum drift, drift rate,
//! and the timestep of maximum violation.

use serde::{Deserialize, Serialize};
use sim_laws::ConservationChecker;
use sim_types::{
    ConservationKind, SimulationState, TimeSeries, Tolerance, Violation,
};

// ─── Per-Law Result ─────────────────────────────────────────────────────────

/// Result of checking a single conservation law across a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawCheckResult {
    pub kind: ConservationKind,
    pub name: String,
    pub initial_value: f64,
    pub final_value: f64,
    pub max_drift: f64,
    pub max_relative_drift: f64,
    pub max_drift_timestep: usize,
    pub max_drift_time: f64,
    pub mean_drift_rate: f64,
    pub conserved: bool,
    pub series: TimeSeries,
    pub violations: Vec<Violation>,
}

/// Aggregate report for a full trace audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceAuditReport {
    pub results: Vec<LawCheckResult>,
    pub total_timesteps: usize,
    pub duration: f64,
    pub all_conserved: bool,
}

impl TraceAuditReport {
    pub fn summary(&self) -> String {
        let n_violated = self.results.iter().filter(|r| !r.conserved).count();
        if n_violated == 0 {
            format!(
                "All {} laws conserved over {} timesteps (T={:.4})",
                self.results.len(),
                self.total_timesteps,
                self.duration
            )
        } else {
            let names: Vec<&str> = self
                .results
                .iter()
                .filter(|r| !r.conserved)
                .map(|r| r.name.as_str())
                .collect();
            format!(
                "{} of {} laws violated: [{}]",
                n_violated,
                self.results.len(),
                names.join(", ")
            )
        }
    }
}

// ─── Checker ────────────────────────────────────────────────────────────────

/// Verifies conservation laws across a simulation trace.
///
/// Given a sequence of [`SimulationState`]s and a set of [`ConservationChecker`]
/// implementations, the checker computes each conserved quantity at every
/// timestep, measures drift from the initial value, and reports diagnostics.
pub struct ConservationLawChecker {
    laws: Vec<Box<dyn ConservationChecker>>,
    tolerance: Tolerance,
}

impl ConservationLawChecker {
    pub fn new(tolerance: Tolerance) -> Self {
        Self {
            laws: Vec::new(),
            tolerance,
        }
    }

    /// Register a conservation law to be checked.
    pub fn add_law(&mut self, law: Box<dyn ConservationChecker>) {
        self.laws.push(law);
    }

    /// Register multiple conservation laws at once.
    pub fn add_laws(&mut self, laws: Vec<Box<dyn ConservationChecker>>) {
        self.laws.extend(laws);
    }

    /// Check a single law across the trace, returning a detailed result.
    fn check_law(
        &self,
        law: &dyn ConservationChecker,
        states: &[SimulationState],
    ) -> LawCheckResult {
        assert!(!states.is_empty(), "trace must contain at least one state");

        let values: Vec<f64> = states.iter().map(|s| law.compute_scalar(s)).collect();
        let times: Vec<f64> = states.iter().map(|s| s.time).collect();
        let series = TimeSeries::new(times.clone(), values.clone());

        let initial = values[0];
        let final_val = *values.last().unwrap();

        // Find timestep of maximum drift
        let mut max_drift = 0.0_f64;
        let mut max_drift_idx = 0;
        for (i, &v) in values.iter().enumerate() {
            let drift = (v - initial).abs();
            if drift > max_drift {
                max_drift = drift;
                max_drift_idx = i;
            }
        }

        let max_relative_drift = if initial.abs() > 1e-30 {
            max_drift / initial.abs()
        } else {
            max_drift
        };

        let total_time = times.last().unwrap_or(&0.0) - times.first().unwrap_or(&0.0);
        let mean_drift_rate = if total_time.abs() > 1e-30 {
            (final_val - initial) / total_time
        } else {
            0.0
        };

        // Collect violations
        let violations: Vec<Violation> = values
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(i, &v)| {
                if !self.tolerance.check(initial, v) {
                    let severity = sim_laws::classify_violation_severity(initial, v);
                    Some(Violation::new(law.kind(), severity, times[i], initial, v))
                } else {
                    None
                }
            })
            .collect();

        let conserved = violations.is_empty();

        LawCheckResult {
            kind: law.kind(),
            name: law.name().to_string(),
            initial_value: initial,
            final_value: final_val,
            max_drift,
            max_relative_drift,
            max_drift_timestep: max_drift_idx,
            max_drift_time: times[max_drift_idx],
            mean_drift_rate,
            conserved,
            series,
            violations,
        }
    }

    /// Audit an entire simulation trace against all registered laws.
    pub fn audit(&self, states: &[SimulationState]) -> TraceAuditReport {
        if states.is_empty() {
            return TraceAuditReport {
                results: Vec::new(),
                total_timesteps: 0,
                duration: 0.0,
                all_conserved: true,
            };
        }

        let results: Vec<LawCheckResult> = self
            .laws
            .iter()
            .map(|law| self.check_law(law.as_ref(), states))
            .collect();

        let all_conserved = results.iter().all(|r| r.conserved);
        let duration = states.last().unwrap().time - states.first().unwrap().time;

        TraceAuditReport {
            results,
            total_timesteps: states.len(),
            duration,
            all_conserved,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{Particle, Vec3};

    /// A trivial system where energy is exactly conserved (no potential).
    fn free_particle_trace(n: usize) -> Vec<SimulationState> {
        let dt = 0.01;
        let mut p = Particle::new(1.0, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let mut states = Vec::with_capacity(n);
        for i in 0..n {
            states.push(SimulationState::new(vec![p.clone()], i as f64 * dt));
            p.position += p.velocity * dt;
        }
        states
    }

    /// A system with intentional energy drift.
    fn drifting_trace(n: usize) -> Vec<SimulationState> {
        let dt = 0.01;
        let mut states = Vec::with_capacity(n);
        for i in 0..n {
            let speed = 1.0 + 0.01 * i as f64; // velocity increases
            let p = Particle::new(1.0, Vec3::new(i as f64 * dt, 0.0, 0.0), Vec3::new(speed, 0.0, 0.0));
            states.push(SimulationState::new(vec![p], i as f64 * dt));
        }
        states
    }

    #[test]
    fn test_free_particle_energy_conserved() {
        let states = free_particle_trace(100);
        let mut checker = ConservationLawChecker::new(Tolerance::absolute(1e-10));
        checker.add_law(Box::new(sim_laws::KineticEnergy));

        let report = checker.audit(&states);
        assert!(report.all_conserved);
        assert_eq!(report.results.len(), 1);
        assert!(report.results[0].max_drift < 1e-12);
    }

    #[test]
    fn test_drifting_energy_detected() {
        let states = drifting_trace(200);
        let mut checker = ConservationLawChecker::new(Tolerance::absolute(1e-6));
        checker.add_law(Box::new(sim_laws::KineticEnergy));

        let report = checker.audit(&states);
        assert!(!report.all_conserved);
        assert!(!report.results[0].violations.is_empty());
        assert!(report.results[0].max_drift > 0.0);
    }

    #[test]
    fn test_momentum_conserved_free_particles() {
        let dt = 0.01;
        let states: Vec<SimulationState> = (0..50)
            .map(|i| {
                let p1 = Particle::new(
                    1.0,
                    Vec3::new(i as f64 * dt, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                );
                let p2 = Particle::new(
                    2.0,
                    Vec3::new(-(i as f64 * dt * 0.5), 0.0, 0.0),
                    Vec3::new(-0.5, 0.0, 0.0),
                );
                SimulationState::new(vec![p1, p2], i as f64 * dt)
            })
            .collect();

        let mut checker = ConservationLawChecker::new(Tolerance::absolute(1e-10));
        checker.add_law(Box::new(sim_laws::TotalLinearMomentum));

        let report = checker.audit(&states);
        assert!(report.all_conserved);
    }

    #[test]
    fn test_empty_trace() {
        let checker = ConservationLawChecker::new(Tolerance::default());
        let report = checker.audit(&[]);
        assert!(report.all_conserved);
        assert_eq!(report.total_timesteps, 0);
    }

    #[test]
    fn test_multi_law_audit() {
        let states = free_particle_trace(50);
        let mut checker = ConservationLawChecker::new(Tolerance::absolute(1e-10));
        checker.add_law(Box::new(sim_laws::KineticEnergy));
        checker.add_law(Box::new(sim_laws::TotalLinearMomentum));
        checker.add_law(Box::new(sim_laws::TotalMass));

        let report = checker.audit(&states);
        assert_eq!(report.results.len(), 3);
        assert!(report.all_conserved);
    }

    #[test]
    fn test_report_summary() {
        let states = drifting_trace(200);
        let mut checker = ConservationLawChecker::new(Tolerance::absolute(1e-6));
        checker.add_law(Box::new(sim_laws::KineticEnergy));

        let report = checker.audit(&states);
        let summary = report.summary();
        assert!(summary.contains("violated"));
    }
}
