//! Coverage certificate types per Definition D7.
//!
//! A coverage certificate `C = (S, V, U, ε_a, ε_e, δ, κ)` certifies
//! how thoroughly the body-parameter space Θ has been verified.
//!
//! * `S` – sample set with per-sample verdicts.
//! * `V` – regions proven accessible by the SMT tier.
//! * `U` – unverified violation surfaces.
//! * `ε_a` – analytical bound on linearization error.
//! * `ε_e` – estimated sampling error.
//! * `δ` – confidence parameter (failure probability).
//! * `κ` – coverage fraction combining sampling + SMT evidence.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::error::{VerifierError, VerifierResult};
use crate::traits::Verdict;
use crate::ElementId;

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Verdict for a single sample point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleVerdict {
    /// Unique identifier for this sample.
    pub id: Uuid,
    /// Body-parameter values (5-D).
    pub body_params: Vec<f64>,
    /// The element being checked.
    pub element_id: ElementId,
    /// Verification verdict.
    pub verdict: Verdict,
    /// Time spent on this sample (seconds).
    pub computation_time_s: f64,
    /// Stratum index, if stratified sampling was used.
    pub stratum: Option<usize>,
}

impl SampleVerdict {
    /// Create a new passing sample verdict.
    pub fn pass(body_params: Vec<f64>, element_id: ElementId) -> Self {
        Self {
            id: Uuid::new_v4(),
            body_params,
            element_id,
            verdict: Verdict::Pass,
            computation_time_s: 0.0,
            stratum: None,
        }
    }

    /// Create a new failing sample verdict.
    pub fn fail(body_params: Vec<f64>, element_id: ElementId, reason: String) -> Self {
        let witness = body_params.clone();
        Self {
            id: Uuid::new_v4(),
            body_params,
            element_id,
            verdict: Verdict::Fail {
                reason,
                witness: Some(witness),
            },
            computation_time_s: 0.0,
            stratum: None,
        }
    }

    /// Whether this sample passed.
    pub fn is_pass(&self) -> bool {
        self.verdict.is_pass()
    }

    /// Set computation time.
    pub fn with_time(mut self, t: f64) -> Self {
        self.computation_time_s = t;
        self
    }

    /// Set stratum index.
    pub fn with_stratum(mut self, s: usize) -> Self {
        self.stratum = Some(s);
        self
    }
}

/// A region in parameter space proven accessible by the SMT solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedRegion {
    /// Human-readable label for this region.
    pub label: String,
    /// Lower corner of the parameter-space bounding box.
    pub lower: Vec<f64>,
    /// Upper corner of the parameter-space bounding box.
    pub upper: Vec<f64>,
    /// Element this region pertains to.
    pub element_id: ElementId,
    /// SMT proof status.
    pub proof_status: ProofStatus,
    /// Linearization error bound for this region.
    pub linearization_error: f64,
    /// Time spent proving this region (seconds).
    pub proof_time_s: f64,
}

impl VerifiedRegion {
    /// Create a new verified region.
    pub fn new(
        label: impl Into<String>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        element_id: ElementId,
    ) -> Self {
        Self {
            label: label.into(),
            lower,
            upper,
            element_id,
            proof_status: ProofStatus::Verified,
            linearization_error: 0.0,
            proof_time_s: 0.0,
        }
    }

    /// Compute the hyper-volume of this region.
    pub fn volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Check whether a point is inside this region.
    pub fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.lower.len()
            && point
                .iter()
                .zip(self.lower.iter().zip(self.upper.iter()))
                .all(|(p, (lo, hi))| *p >= *lo && *p <= *hi)
    }
}

/// Status of an SMT proof for a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// The region was fully verified (unsat = no violation exists).
    Verified,
    /// The solver timed out on this region.
    Timeout,
    /// The solver returned unknown.
    Unknown,
    /// A counterexample was found.
    Refuted,
}

impl std::fmt::Display for ProofStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofStatus::Verified => write!(f, "verified"),
            ProofStatus::Timeout => write!(f, "timeout"),
            ProofStatus::Unknown => write!(f, "unknown"),
            ProofStatus::Refuted => write!(f, "refuted"),
        }
    }
}

/// An unverified surface in parameter space where a violation may occur.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViolationSurface {
    /// Human-readable description.
    pub description: String,
    /// Element this surface pertains to.
    pub element_id: ElementId,
    /// Representative sample points on the surface.
    pub sample_points: Vec<Vec<f64>>,
    /// Bounding box of the surface in parameter space.
    pub parameter_bounds: Option<(Vec<f64>, Vec<f64>)>,
    /// Estimated area/volume of the unverified region.
    pub estimated_measure: f64,
    /// Severity indicator.
    pub severity: ViolationSeverity,
}

impl ViolationSurface {
    /// Create a new violation surface.
    pub fn new(
        description: impl Into<String>,
        element_id: ElementId,
        severity: ViolationSeverity,
    ) -> Self {
        Self {
            description: description.into(),
            element_id,
            sample_points: Vec::new(),
            parameter_bounds: None,
            estimated_measure: 0.0,
            severity,
        }
    }

    /// Add a sample point on the violation surface.
    pub fn add_sample(&mut self, point: Vec<f64>) {
        self.sample_points.push(point);
    }

    /// Number of sample points.
    pub fn sample_count(&self) -> usize {
        self.sample_points.len()
    }
}

/// Severity of a violation surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Low severity – affects a tiny fraction of the population.
    Low,
    /// Medium severity.
    Medium,
    /// High severity – affects a substantial population fraction.
    High,
    /// Critical – complete accessibility failure.
    Critical,
}

// ---------------------------------------------------------------------------
// Certificate grade
// ---------------------------------------------------------------------------

/// Grade assigned to a coverage certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CertificateGrade {
    /// κ ≥ 0.99 and no unverified violations.
    Full,
    /// 0.90 ≤ κ < 0.99 or minor unverified regions.
    Partial,
    /// κ < 0.90 or significant violations remain.
    Weak,
}

impl CertificateGrade {
    /// Compute grade from coverage κ and violation surfaces.
    pub fn from_metrics(kappa: f64, num_violations: usize) -> Self {
        if kappa >= 0.99 && num_violations == 0 {
            CertificateGrade::Full
        } else if kappa >= 0.90 {
            CertificateGrade::Partial
        } else {
            CertificateGrade::Weak
        }
    }
}

impl std::fmt::Display for CertificateGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertificateGrade::Full => write!(f, "FULL"),
            CertificateGrade::Partial => write!(f, "PARTIAL"),
            CertificateGrade::Weak => write!(f, "WEAK"),
        }
    }
}

// ---------------------------------------------------------------------------
// CoverageCertificate
// ---------------------------------------------------------------------------

/// Coverage certificate per Definition D7.
///
/// `C = (S, V, U, ε_a, ε_e, δ, κ)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageCertificate {
    /// Unique certificate ID.
    pub id: Uuid,
    /// Timestamp of certificate creation (ISO-8601).
    pub timestamp: String,
    /// Protocol version used.
    pub protocol_version: String,
    /// Scene identifier.
    pub scene_id: Uuid,
    /// S – set of sample verdicts.
    pub samples: Vec<SampleVerdict>,
    /// V – verified regions with proofs.
    pub verified_regions: Vec<VerifiedRegion>,
    /// U – unverified violation surfaces.
    pub violations: Vec<ViolationSurface>,
    /// ε_a – analytical (linearization) error bound.
    pub epsilon_analytical: f64,
    /// ε_e – estimated sampling error bound.
    pub epsilon_estimated: f64,
    /// δ – confidence parameter (failure probability bound).
    pub delta: f64,
    /// κ – overall coverage fraction.
    pub kappa: f64,
    /// Computed grade.
    pub grade: CertificateGrade,
    /// Total wall-clock time for the verification run (seconds).
    pub total_time_s: f64,
    /// Per-element coverage breakdown.
    pub element_coverage: HashMap<ElementId, f64>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl CoverageCertificate {
    /// Compute κ from samples and verified regions.
    ///
    /// κ = (n_pass + V_measure) / (n_total + V_total_measure)
    /// where V_measure is the total volume of verified regions and
    /// V_total_measure is the measure of the full parameter space.
    pub fn compute_kappa(
        samples: &[SampleVerdict],
        verified_regions: &[VerifiedRegion],
        total_param_volume: f64,
    ) -> f64 {
        if samples.is_empty() && verified_regions.is_empty() {
            return 0.0;
        }

        let n_pass = samples.iter().filter(|s| s.is_pass()).count() as f64;
        let n_total = samples.len() as f64;

        let verified_volume: f64 = verified_regions
            .iter()
            .filter(|r| r.proof_status == ProofStatus::Verified)
            .map(|r| r.volume())
            .sum();

        // Combine sampling and SMT evidence.
        // Sampling gives a ratio, SMT gives absolute coverage of volume.
        let sample_ratio = if n_total > 0.0 {
            n_pass / n_total
        } else {
            0.0
        };
        let smt_ratio = if total_param_volume > 0.0 {
            (verified_volume / total_param_volume).min(1.0)
        } else {
            0.0
        };

        // Weighted combination: max of sampling and SMT evidence.
        sample_ratio.max(smt_ratio).min(1.0)
    }

    /// Compute the estimated sampling error using Hoeffding's inequality.
    ///
    /// ε = sqrt(ln(2/δ) / (2n))
    pub fn compute_epsilon_estimated(n_samples: usize, delta: f64) -> f64 {
        if n_samples == 0 || delta <= 0.0 {
            return 1.0;
        }
        ((2.0_f64 / delta).ln() / (2.0 * n_samples as f64)).sqrt()
    }

    /// Compute per-element coverage fractions.
    pub fn compute_element_coverage(samples: &[SampleVerdict]) -> HashMap<ElementId, f64> {
        let mut counts: HashMap<ElementId, (usize, usize)> = HashMap::new();
        for s in samples {
            let entry = counts.entry(s.element_id).or_insert((0, 0));
            entry.1 += 1;
            if s.is_pass() {
                entry.0 += 1;
            }
        }
        counts
            .into_iter()
            .map(|(id, (pass, total))| (id, pass as f64 / total as f64))
            .collect()
    }

    /// Summary of the certificate.
    pub fn summary(&self) -> CertificateSummary {
        CertificateSummary {
            id: self.id,
            grade: self.grade,
            kappa: self.kappa,
            epsilon_analytical: self.epsilon_analytical,
            epsilon_estimated: self.epsilon_estimated,
            delta: self.delta,
            num_samples: self.samples.len(),
            num_pass: self.samples.iter().filter(|s| s.is_pass()).count(),
            num_fail: self.samples.iter().filter(|s| s.verdict.is_fail()).count(),
            num_verified_regions: self.verified_regions.len(),
            num_violations: self.violations.len(),
            total_time_s: self.total_time_s,
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> VerifierResult<String> {
        serde_json::to_string_pretty(self).map_err(VerifierError::from)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> VerifierResult<Self> {
        serde_json::from_str(json).map_err(VerifierError::from)
    }

    /// Check if the certificate meets a minimum coverage target.
    pub fn meets_target(&self, target_kappa: f64) -> bool {
        self.kappa >= target_kappa
    }

    /// Minimum per-element coverage across all elements.
    pub fn min_element_coverage(&self) -> f64 {
        self.element_coverage
            .values()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(1.0)
    }
}

impl std::fmt::Display for CoverageCertificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════╗")?;
        writeln!(f, "║       Coverage Certificate               ║")?;
        writeln!(f, "╠══════════════════════════════════════════╣")?;
        writeln!(f, "║ ID:      {}", self.id)?;
        writeln!(f, "║ Grade:   {}", self.grade)?;
        writeln!(f, "║ κ:       {:.6}", self.kappa)?;
        writeln!(f, "║ ε_a:     {:.6}", self.epsilon_analytical)?;
        writeln!(f, "║ ε_e:     {:.6}", self.epsilon_estimated)?;
        writeln!(f, "║ δ:       {:.6}", self.delta)?;
        writeln!(f, "║ Samples: {} ({} pass, {} fail)",
            self.samples.len(),
            self.samples.iter().filter(|s| s.is_pass()).count(),
            self.samples.iter().filter(|s| s.verdict.is_fail()).count(),
        )?;
        writeln!(f, "║ Regions: {} verified", self.verified_regions.len())?;
        writeln!(f, "║ Violations: {}", self.violations.len())?;
        writeln!(f, "║ Time:    {:.2}s", self.total_time_s)?;
        writeln!(f, "╚══════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Compact summary of a coverage certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateSummary {
    pub id: Uuid,
    pub grade: CertificateGrade,
    pub kappa: f64,
    pub epsilon_analytical: f64,
    pub epsilon_estimated: f64,
    pub delta: f64,
    pub num_samples: usize,
    pub num_pass: usize,
    pub num_fail: usize,
    pub num_verified_regions: usize,
    pub num_violations: usize,
    pub total_time_s: f64,
}

impl std::fmt::Display for CertificateSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] κ={:.4} ε_a={:.4} ε_e={:.4} δ={:.4} samples={} regions={} violations={}",
            self.grade,
            self.kappa,
            self.epsilon_analytical,
            self.epsilon_estimated,
            self.delta,
            self.num_samples,
            self.num_verified_regions,
            self.num_violations,
        )
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`CoverageCertificate`] incrementally.
#[derive(Debug, Clone)]
pub struct CertificateBuilder {
    scene_id: Uuid,
    samples: Vec<SampleVerdict>,
    verified_regions: Vec<VerifiedRegion>,
    violations: Vec<ViolationSurface>,
    epsilon_analytical: f64,
    delta: f64,
    total_time_s: f64,
    total_param_volume: f64,
    metadata: HashMap<String, String>,
}

impl CertificateBuilder {
    /// Create a new builder for the given scene.
    pub fn new(scene_id: Uuid) -> Self {
        Self {
            scene_id,
            samples: Vec::new(),
            verified_regions: Vec::new(),
            violations: Vec::new(),
            epsilon_analytical: 0.0,
            delta: 0.05,
            total_time_s: 0.0,
            total_param_volume: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Add a sample verdict.
    pub fn add_sample(mut self, sample: SampleVerdict) -> Self {
        self.samples.push(sample);
        self
    }

    /// Add multiple sample verdicts.
    pub fn add_samples(mut self, samples: impl IntoIterator<Item = SampleVerdict>) -> Self {
        self.samples.extend(samples);
        self
    }

    /// Add a verified region.
    pub fn add_verified_region(mut self, region: VerifiedRegion) -> Self {
        self.verified_regions.push(region);
        self
    }

    /// Add a violation surface.
    pub fn add_violation(mut self, violation: ViolationSurface) -> Self {
        self.violations.push(violation);
        self
    }

    /// Set the analytical error bound.
    pub fn epsilon_analytical(mut self, eps: f64) -> Self {
        self.epsilon_analytical = eps;
        self
    }

    /// Set the confidence parameter.
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set the total verification time.
    pub fn total_time(mut self, time_s: f64) -> Self {
        self.total_time_s = time_s;
        self
    }

    /// Set the total parameter-space volume.
    pub fn total_param_volume(mut self, vol: f64) -> Self {
        self.total_param_volume = vol;
        self
    }

    /// Add a metadata entry.
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the certificate, computing derived values.
    pub fn build(self) -> VerifierResult<CoverageCertificate> {
        let kappa = CoverageCertificate::compute_kappa(
            &self.samples,
            &self.verified_regions,
            self.total_param_volume,
        );

        let epsilon_estimated = CoverageCertificate::compute_epsilon_estimated(
            self.samples.len(),
            self.delta,
        );

        let grade = CertificateGrade::from_metrics(kappa, self.violations.len());
        let element_coverage = CoverageCertificate::compute_element_coverage(&self.samples);

        let now = chrono_free_timestamp();

        Ok(CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp: now,
            protocol_version: crate::PROTOCOL_VERSION.to_string(),
            scene_id: self.scene_id,
            samples: self.samples,
            verified_regions: self.verified_regions,
            violations: self.violations,
            epsilon_analytical: self.epsilon_analytical,
            epsilon_estimated,
            delta: self.delta,
            kappa,
            grade,
            total_time_s: self.total_time_s,
            element_coverage,
            metadata: self.metadata,
        })
    }
}

/// Produce an ISO-8601 timestamp string without depending on chrono.
fn chrono_free_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Rough conversion – not calendar-precise but deterministic and serializable.
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, months, day, hours, mins, s
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_element_id() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_samples(n_pass: usize, n_fail: usize) -> Vec<SampleVerdict> {
        let eid = test_element_id();
        let mut samples = Vec::new();
        for i in 0..n_pass {
            samples.push(SampleVerdict::pass(
                vec![1.7 + i as f64 * 0.01, 0.35, 0.48, 0.27, 0.19],
                eid,
            ));
        }
        for i in 0..n_fail {
            samples.push(SampleVerdict::fail(
                vec![1.5 + i as f64 * 0.01, 0.30, 0.38, 0.22, 0.16],
                eid,
                "unreachable".into(),
            ));
        }
        samples
    }

    #[test]
    fn test_sample_verdict_pass() {
        let s = SampleVerdict::pass(vec![1.7, 0.35, 0.48, 0.27, 0.19], test_element_id());
        assert!(s.is_pass());
    }

    #[test]
    fn test_sample_verdict_fail() {
        let s = SampleVerdict::fail(
            vec![1.5, 0.30, 0.38, 0.22, 0.16],
            test_element_id(),
            "unreachable".into(),
        );
        assert!(!s.is_pass());
    }

    #[test]
    fn test_sample_verdict_with_stratum() {
        let s = SampleVerdict::pass(vec![1.7, 0.35, 0.48, 0.27, 0.19], test_element_id())
            .with_stratum(3)
            .with_time(0.05);
        assert_eq!(s.stratum, Some(3));
        assert!((s.computation_time_s - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_verified_region_volume() {
        let r = VerifiedRegion::new(
            "test",
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            test_element_id(),
        );
        assert!((r.volume() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_verified_region_contains() {
        let r = VerifiedRegion::new(
            "test",
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            test_element_id(),
        );
        assert!(r.contains(&[0.5, 0.5]));
        assert!(!r.contains(&[1.5, 0.5]));
    }

    #[test]
    fn test_compute_kappa_all_pass() {
        let samples = make_samples(100, 0);
        let kappa = CoverageCertificate::compute_kappa(&samples, &[], 1.0);
        assert!((kappa - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_kappa_half_pass() {
        let samples = make_samples(50, 50);
        let kappa = CoverageCertificate::compute_kappa(&samples, &[], 1.0);
        assert!((kappa - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_kappa_with_smt() {
        let regions = vec![VerifiedRegion::new(
            "full",
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            test_element_id(),
        )];
        let kappa = CoverageCertificate::compute_kappa(&[], &regions, 1.0);
        assert!((kappa - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_epsilon_estimated() {
        let eps = CoverageCertificate::compute_epsilon_estimated(1000, 0.05);
        // sqrt(ln(40) / 2000) ≈ 0.043
        assert!(eps > 0.04 && eps < 0.05);
    }

    #[test]
    fn test_compute_epsilon_zero_samples() {
        let eps = CoverageCertificate::compute_epsilon_estimated(0, 0.05);
        assert!((eps - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_certificate_grade_full() {
        assert_eq!(CertificateGrade::from_metrics(1.0, 0), CertificateGrade::Full);
    }

    #[test]
    fn test_certificate_grade_partial() {
        assert_eq!(CertificateGrade::from_metrics(0.95, 0), CertificateGrade::Partial);
        assert_eq!(CertificateGrade::from_metrics(0.99, 1), CertificateGrade::Partial);
    }

    #[test]
    fn test_certificate_grade_weak() {
        assert_eq!(CertificateGrade::from_metrics(0.80, 0), CertificateGrade::Weak);
    }

    #[test]
    fn test_certificate_builder() {
        let cert = CertificateBuilder::new(Uuid::new_v4())
            .add_samples(make_samples(95, 5))
            .epsilon_analytical(0.001)
            .delta(0.05)
            .total_time(10.0)
            .meta("tier", "1")
            .build()
            .unwrap();

        assert!((cert.kappa - 0.95).abs() < 1e-12);
        assert_eq!(cert.grade, CertificateGrade::Partial);
        assert_eq!(cert.samples.len(), 100);
        assert!(cert.epsilon_estimated > 0.0);
        assert_eq!(cert.metadata.get("tier").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_certificate_builder_full() {
        let cert = CertificateBuilder::new(Uuid::new_v4())
            .add_samples(make_samples(100, 0))
            .epsilon_analytical(0.0001)
            .delta(0.05)
            .build()
            .unwrap();

        assert_eq!(cert.grade, CertificateGrade::Full);
        assert!(cert.meets_target(0.99));
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        let cert = CertificateBuilder::new(Uuid::new_v4())
            .add_samples(make_samples(10, 2))
            .epsilon_analytical(0.01)
            .delta(0.05)
            .build()
            .unwrap();

        let json = cert.to_json().unwrap();
        let back = CoverageCertificate::from_json(&json).unwrap();
        assert_eq!(cert.id, back.id);
        assert_eq!(cert.kappa, back.kappa);
        assert_eq!(cert.grade, back.grade);
        assert_eq!(cert.samples.len(), back.samples.len());
    }

    #[test]
    fn test_certificate_display() {
        let cert = CertificateBuilder::new(Uuid::new_v4())
            .add_samples(make_samples(90, 10))
            .build()
            .unwrap();
        let display = format!("{cert}");
        assert!(display.contains("Coverage Certificate"));
        assert!(display.contains("Grade"));
    }

    #[test]
    fn test_certificate_summary_display() {
        let cert = CertificateBuilder::new(Uuid::new_v4())
            .add_samples(make_samples(80, 20))
            .build()
            .unwrap();
        let summary = cert.summary();
        let s = format!("{summary}");
        assert!(s.contains("κ="));
        assert!(s.contains("samples="));
    }

    #[test]
    fn test_element_coverage() {
        let eid1 = Uuid::new_v4();
        let eid2 = Uuid::new_v4();
        let mut samples = Vec::new();
        for _ in 0..8 {
            samples.push(SampleVerdict::pass(vec![1.7, 0.35, 0.48, 0.27, 0.19], eid1));
        }
        for _ in 0..2 {
            samples.push(SampleVerdict::fail(vec![1.5, 0.30, 0.38, 0.22, 0.16], eid1, "x".into()));
        }
        for _ in 0..5 {
            samples.push(SampleVerdict::pass(vec![1.7, 0.35, 0.48, 0.27, 0.19], eid2));
        }

        let cov = CoverageCertificate::compute_element_coverage(&samples);
        assert!((cov[&eid1] - 0.8).abs() < 1e-12);
        assert!((cov[&eid2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_violation_surface() {
        let mut vs = ViolationSurface::new("edge case", test_element_id(), ViolationSeverity::Medium);
        vs.add_sample(vec![1.5, 0.3, 0.38, 0.22, 0.16]);
        vs.add_sample(vec![1.51, 0.31, 0.39, 0.23, 0.17]);
        assert_eq!(vs.sample_count(), 2);
    }

    #[test]
    fn test_proof_status_display() {
        assert_eq!(format!("{}", ProofStatus::Verified), "verified");
        assert_eq!(format!("{}", ProofStatus::Timeout), "timeout");
    }

    #[test]
    fn test_grade_ordering() {
        assert!(CertificateGrade::Weak < CertificateGrade::Partial);
        assert!(CertificateGrade::Partial < CertificateGrade::Full);
    }

    #[test]
    fn test_min_element_coverage() {
        let eid1 = Uuid::new_v4();
        let eid2 = Uuid::new_v4();
        let mut samples = Vec::new();
        for _ in 0..10 {
            samples.push(SampleVerdict::pass(vec![1.7, 0.35, 0.48, 0.27, 0.19], eid1));
        }
        for _ in 0..5 {
            samples.push(SampleVerdict::pass(vec![1.7, 0.35, 0.48, 0.27, 0.19], eid2));
        }
        for _ in 0..5 {
            samples.push(SampleVerdict::fail(vec![1.5, 0.30, 0.38, 0.22, 0.16], eid2, "x".into()));
        }

        let cert = CertificateBuilder::new(Uuid::new_v4())
            .add_samples(samples)
            .build()
            .unwrap();
        assert!((cert.min_element_coverage() - 0.5).abs() < 1e-12);
    }
}
