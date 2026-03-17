//! Certificate construction: assembles all evidence into a `CoverageCertificate`.
//!
//! The `CertificateAssembler` builds the certificate
//! C = ⟨S, V, U, ε\_analytical, ε\_estimated, δ, κ⟩
//! from sample verdicts, verified regions, violation surfaces, and statistical
//! bound computations.

use std::collections::HashMap;
use uuid::Uuid;
use xr_types::certificate::{
    CertificateGrade, CoverageCertificate, ProofStatus, SampleVerdict,
    VerifiedRegion, ViolationSeverity, ViolationSurface,
};
use xr_types::{ElementId, VerifierError, VerifierResult};

use crate::hoeffding::{HoeffdingBound, VolumeSubtractionModel};

// ────────────────────── Certificate Assembler ─────────────────────────────

/// Assembles all verification evidence into a `CoverageCertificate`.
///
/// Collects samples, verified regions, violation surfaces, and computes
/// the derived quantities (κ, ε, grade).
#[derive(Debug, Clone)]
pub struct CertificateAssembler {
    /// Scene identifier.
    scene_id: Uuid,
    /// Collected sample verdicts.
    samples: Vec<SampleVerdict>,
    /// Verified regions (from Tier 1 + SMT).
    verified_regions: Vec<VerifiedRegion>,
    /// Violation surfaces (unverified boundary regions).
    violations: Vec<ViolationSurface>,
    /// Confidence parameter δ.
    delta: f64,
    /// Maximum Lipschitz constant observed.
    l_max: f64,
    /// Estimated Lipschitz constant (from sampling).
    l_hat: f64,
    /// Volume-subtraction model state.
    volume_model: VolumeSubtractionModel,
    /// Total parameter-space volume.
    total_param_volume: f64,
    /// Number of strata.
    num_strata: usize,
    /// Number of elements.
    num_elements: usize,
    /// Number of devices.
    num_devices: usize,
    /// Metadata key-value pairs.
    metadata: HashMap<String, String>,
    /// Total wall-clock time so far.
    total_time_s: f64,
    /// Per-stratum sample counts for stratified bound.
    stratum_sample_counts: HashMap<usize, (usize, usize, usize, f64)>,
}

impl CertificateAssembler {
    /// Create a new assembler for the given scene.
    pub fn new(scene_id: Uuid) -> Self {
        Self {
            scene_id,
            samples: Vec::new(),
            verified_regions: Vec::new(),
            violations: Vec::new(),
            delta: 0.05,
            l_max: 0.0,
            l_hat: 0.0,
            volume_model: VolumeSubtractionModel::new(1.0),
            total_param_volume: 1.0,
            num_strata: 1,
            num_elements: 1,
            num_devices: 1,
            metadata: HashMap::new(),
            total_time_s: 0.0,
            stratum_sample_counts: HashMap::new(),
        }
    }

    /// Set the confidence parameter δ.
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta.clamp(1e-12, 1.0 - 1e-12);
        self
    }

    /// Set the total parameter-space volume.
    pub fn with_param_volume(mut self, volume: f64) -> Self {
        self.total_param_volume = volume.max(1e-15);
        self.volume_model = VolumeSubtractionModel::new(volume);
        self
    }

    /// Set the number of strata, elements, and devices for bound computation.
    pub fn with_dimensions(
        mut self,
        num_strata: usize,
        num_elements: usize,
        num_devices: usize,
    ) -> Self {
        self.num_strata = num_strata.max(1);
        self.num_elements = num_elements.max(1);
        self.num_devices = num_devices.max(1);
        self
    }

    /// Set the maximum Lipschitz constant.
    pub fn with_lipschitz_max(mut self, l_max: f64) -> Self {
        self.l_max = l_max;
        self
    }

    /// Set the estimated Lipschitz constant.
    pub fn with_lipschitz_hat(mut self, l_hat: f64) -> Self {
        self.l_hat = l_hat;
        self
    }

    /// Set total wall-clock time.
    pub fn with_total_time(mut self, time_s: f64) -> Self {
        self.total_time_s = time_s;
        self
    }

    /// Add metadata.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a single sample verdict.
    pub fn add_sample(&mut self, params: &[f64], element_id: ElementId, passed: bool) {
        let body_params = params.to_vec();
        let verdict = if passed {
            SampleVerdict::pass(body_params, element_id)
        } else {
            SampleVerdict::fail(
                body_params,
                element_id,
                "Accessibility check failed".to_string(),
            )
        };
        self.samples.push(verdict);
    }

    /// Add a sample verdict with stratum information.
    pub fn add_sample_with_stratum(
        &mut self,
        params: &[f64],
        element_id: ElementId,
        passed: bool,
        stratum_idx: usize,
        stratum_volume_weight: f64,
    ) {
        let body_params = params.to_vec();
        let mut verdict = if passed {
            SampleVerdict::pass(body_params, element_id)
        } else {
            SampleVerdict::fail(
                body_params,
                element_id,
                "Accessibility check failed".to_string(),
            )
        };
        verdict = verdict.with_stratum(stratum_idx);
        self.samples.push(verdict);

        // Track per-stratum counts
        let entry = self
            .stratum_sample_counts
            .entry(stratum_idx)
            .or_insert((0, 0, 0, stratum_volume_weight));
        entry.0 += 1; // total
        if passed {
            entry.1 += 1; // pass
        } else {
            entry.2 += 1; // fail
        }
    }

    /// Add a pre-constructed sample verdict.
    pub fn add_sample_verdict(&mut self, verdict: SampleVerdict) {
        self.samples.push(verdict);
    }

    /// Add a verified region (Tier 1 green or SMT-proved).
    pub fn add_verified_region(&mut self, region: VerifiedRegion) {
        let vol = region.volume();
        if region.proof_status == ProofStatus::Verified {
            self.volume_model.add_green_volume(vol);
        }
        self.verified_regions.push(region);
    }

    /// Add a verified region from bounds.
    pub fn add_verified_region_from_bounds(
        &mut self,
        label: impl Into<String>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        element_id: ElementId,
        proof_status: ProofStatus,
        linearization_error: f64,
    ) {
        let region = VerifiedRegion::new(label, lower, upper, element_id);
        let mut region = region;
        region.proof_status = proof_status;
        region.linearization_error = linearization_error;
        self.add_verified_region(region);
    }

    /// Add a violation surface.
    pub fn add_violation_surface(&mut self, surface: ViolationSurface) {
        self.violations.push(surface);
    }

    /// Add a violation from description.
    pub fn add_violation(
        &mut self,
        description: impl Into<String>,
        element_id: ElementId,
        severity: ViolationSeverity,
        estimated_measure: f64,
    ) {
        let mut v = ViolationSurface::new(description, element_id, severity);
        v.estimated_measure = estimated_measure;
        self.violations.push(v);
    }

    /// Mark Tier 1 red volume (provably inaccessible).
    pub fn add_red_volume(&mut self, volume: f64) {
        self.volume_model.add_red_volume(volume);
    }

    /// Mark SMT-verified volume.
    pub fn add_smt_verified_volume(&mut self, volume: f64) {
        self.volume_model.add_smt_volume(volume);
    }

    /// Compute κ (coverage fraction).
    ///
    /// κ is the fraction of the accessible parameter space that is verified:
    /// - Verified regions contribute their volume fraction directly.
    /// - Sampling contributes via the estimated pass rate over the
    ///   remaining (yellow) space.
    pub fn compute_kappa(&self) -> f64 {
        let n_total = self.samples.len();
        if n_total == 0 && self.verified_regions.is_empty() {
            return 0.0;
        }

        let n_pass = self.samples.iter().filter(|s| s.is_pass()).count();
        let sampling_kappa = if n_total > 0 {
            n_pass as f64 / n_total as f64
        } else {
            0.0
        };

        self.volume_model.effective_kappa(sampling_kappa)
    }

    /// Compute ε\_analytical: linearization error bound.
    ///
    /// Uses the maximum Lipschitz constant L\_max and the maximum
    /// region diameter across all verified regions.
    ///
    /// ε\_analytical = L\_max · max_diameter / 2
    pub fn compute_epsilon_analytical(&self, l_max: f64) -> f64 {
        if self.verified_regions.is_empty() {
            return 0.0;
        }
        let max_diameter = self
            .verified_regions
            .iter()
            .map(|r| region_diameter(&r.lower, &r.upper))
            .fold(0.0_f64, f64::max);
        l_max * max_diameter / 2.0
    }

    /// Compute ε\_estimated: sampling error bound using Hoeffding's inequality.
    ///
    /// Uses the estimated Lipschitz constant L\_hat for a tighter bound.
    pub fn compute_epsilon_estimated(&self, l_hat: f64) -> f64 {
        let bound = HoeffdingBound::new(
            self.delta,
            self.num_strata.max(1),
            self.num_elements.max(1),
            self.num_devices.max(1),
        );
        let sampling_eps = bound.epsilon(self.samples.len());

        // Use volume subtraction to reduce the effective epsilon
        let effective_eps = self.volume_model.effective_epsilon(sampling_eps);

        // Add linearization contribution from L_hat
        let lin_contribution = if l_hat > 0.0 && !self.verified_regions.is_empty() {
            let max_diam = self
                .verified_regions
                .iter()
                .map(|r| region_diameter(&r.lower, &r.upper))
                .fold(0.0_f64, f64::max);
            l_hat * max_diam / 2.0
        } else {
            0.0
        };

        effective_eps + lin_contribution
    }

    /// Compute per-element coverage.
    pub fn compute_element_coverage(&self) -> HashMap<ElementId, f64> {
        let mut per_element: HashMap<ElementId, (usize, usize)> = HashMap::new();
        for s in &self.samples {
            let entry = per_element.entry(s.element_id).or_insert((0, 0));
            entry.0 += 1;
            if s.is_pass() {
                entry.1 += 1;
            }
        }
        per_element
            .into_iter()
            .map(|(id, (total, pass))| (id, pass as f64 / total as f64))
            .collect()
    }

    /// Compute the certificate grade based on κ and violation count.
    pub fn compute_grade(&self, kappa: f64) -> CertificateGrade {
        CertificateGrade::from_metrics(kappa, self.violations.len())
    }

    /// Validate the assembled data for consistency before finalizing.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.samples.is_empty() && self.verified_regions.is_empty() {
            issues.push("No evidence: neither samples nor verified regions present".into());
        }

        if self.delta <= 0.0 || self.delta >= 1.0 {
            issues.push(format!("Invalid δ = {}; must be in (0, 1)", self.delta));
        }

        if self.total_param_volume <= 0.0 {
            issues.push(format!(
                "Invalid total parameter volume: {}",
                self.total_param_volume
            ));
        }

        // Check for overlapping verified regions
        for i in 0..self.verified_regions.len() {
            for j in (i + 1)..self.verified_regions.len() {
                if regions_overlap(&self.verified_regions[i], &self.verified_regions[j]) {
                    issues.push(format!(
                        "Verified regions '{}' and '{}' overlap",
                        self.verified_regions[i].label, self.verified_regions[j].label,
                    ));
                }
            }
        }

        // Check verified region volumes don't exceed total
        let total_verified: f64 = self.verified_regions.iter().map(|r| r.volume()).sum();
        if total_verified > self.total_param_volume * 1.01 {
            issues.push(format!(
                "Verified volume ({:.6}) exceeds total ({:.6})",
                total_verified, self.total_param_volume,
            ));
        }

        issues
    }

    /// Finalize and build the `CoverageCertificate`.
    pub fn finalize(self) -> VerifierResult<CoverageCertificate> {
        let issues = self.validate();
        if issues.iter().any(|i| i.contains("Invalid δ") || i.contains("Invalid total")) {
            return Err(VerifierError::CertificateGeneration(
                issues.join("; "),
            ));
        }

        let kappa = self.compute_kappa();
        let eps_analytical = self.compute_epsilon_analytical(self.l_max);
        let eps_estimated = self.compute_epsilon_estimated(self.l_hat);
        let grade = self.compute_grade(kappa);
        let element_coverage = self.compute_element_coverage();

        let timestamp = format_timestamp();

        let mut metadata = self.metadata;
        metadata.insert("num_strata".into(), self.num_strata.to_string());
        metadata.insert("l_max".into(), format!("{:.6}", self.l_max));
        metadata.insert("l_hat".into(), format!("{:.6}", self.l_hat));
        metadata.insert(
            "volume_verified_fraction".into(),
            format!("{:.6}", self.volume_model.verified_fraction()),
        );
        if !issues.is_empty() {
            metadata.insert("validation_warnings".into(), issues.join("; "));
        }

        Ok(CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp,
            protocol_version: xr_types::PROTOCOL_VERSION.to_string(),
            scene_id: self.scene_id,
            samples: self.samples,
            verified_regions: self.verified_regions,
            violations: self.violations,
            epsilon_analytical: eps_analytical,
            epsilon_estimated: eps_estimated,
            delta: self.delta,
            kappa,
            grade,
            total_time_s: self.total_time_s,
            element_coverage,
            metadata,
        })
    }
}

/// Compute the diameter of a hyper-rectangular region.
fn region_diameter(lower: &[f64], upper: &[f64]) -> f64 {
    lower
        .iter()
        .zip(upper.iter())
        .map(|(lo, hi)| (hi - lo) * (hi - lo))
        .sum::<f64>()
        .sqrt()
}

/// Check if two verified regions overlap in parameter space.
fn regions_overlap(a: &VerifiedRegion, b: &VerifiedRegion) -> bool {
    if a.lower.len() != b.lower.len() {
        return false;
    }
    for i in 0..a.lower.len() {
        if a.upper[i] <= b.lower[i] || b.upper[i] <= a.lower[i] {
            return false;
        }
    }
    true
}

/// Generate an ISO-8601 timestamp.
fn format_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
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

// ────────────────────── Builder Pattern Variant ───────────────────────────

/// Fluent builder for constructing certificates step-by-step.
pub struct CertificateFluentBuilder {
    assembler: CertificateAssembler,
}

impl CertificateFluentBuilder {
    pub fn new(scene_id: Uuid) -> Self {
        Self {
            assembler: CertificateAssembler::new(scene_id),
        }
    }

    pub fn delta(mut self, delta: f64) -> Self {
        self.assembler = self.assembler.with_delta(delta);
        self
    }

    pub fn param_volume(mut self, volume: f64) -> Self {
        self.assembler = self.assembler.with_param_volume(volume);
        self
    }

    pub fn dimensions(
        mut self,
        strata: usize,
        elements: usize,
        devices: usize,
    ) -> Self {
        self.assembler = self.assembler.with_dimensions(strata, elements, devices);
        self
    }

    pub fn lipschitz(mut self, l_max: f64, l_hat: f64) -> Self {
        self.assembler = self.assembler.with_lipschitz_max(l_max).with_lipschitz_hat(l_hat);
        self
    }

    pub fn total_time(mut self, time_s: f64) -> Self {
        self.assembler = self.assembler.with_total_time(time_s);
        self
    }

    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.assembler = self.assembler.with_meta(key, value);
        self
    }

    pub fn sample(
        mut self,
        params: &[f64],
        element_id: ElementId,
        passed: bool,
    ) -> Self {
        self.assembler.add_sample(params, element_id, passed);
        self
    }

    pub fn verified_region(mut self, region: VerifiedRegion) -> Self {
        self.assembler.add_verified_region(region);
        self
    }

    pub fn violation(mut self, surface: ViolationSurface) -> Self {
        self.assembler.add_violation_surface(surface);
        self
    }

    pub fn build(self) -> VerifierResult<CoverageCertificate> {
        self.assembler.finalize()
    }
}

// ──────────────────── Certificate Merge ───────────────────────────────────

/// Merge multiple certificates into a single combined certificate.
///
/// Used when parallel verification pipelines produce separate certificates
/// that need to be consolidated.
pub fn merge_certificates(
    certs: &[CoverageCertificate],
    scene_id: Uuid,
    delta: f64,
) -> VerifierResult<CoverageCertificate> {
    if certs.is_empty() {
        return Err(VerifierError::CertificateGeneration(
            "No certificates to merge".into(),
        ));
    }

    let mut assembler = CertificateAssembler::new(scene_id);
    assembler = assembler.with_delta(delta);

    let mut total_time = 0.0_f64;
    let num_elements = certs
        .iter()
        .map(|c| c.element_coverage.len())
        .max()
        .unwrap_or(1);
    let num_devices = 1;

    for cert in certs {
        for sample in &cert.samples {
            assembler.add_sample_verdict(sample.clone());
        }
        for region in &cert.verified_regions {
            assembler.add_verified_region(region.clone());
        }
        for violation in &cert.violations {
            assembler.add_violation_surface(violation.clone());
        }
        total_time = total_time.max(cert.total_time_s);
    }

    assembler = assembler
        .with_total_time(total_time)
        .with_dimensions(1, num_elements, num_devices)
        .with_meta("merged_from", format!("{}", certs.len()));

    assembler.finalize()
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_element_id() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn test_scene_id() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000010").unwrap()
    }

    #[test]
    fn test_assembler_basic() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        asm = asm.with_delta(0.05).with_dimensions(4, 2, 1);

        let eid = test_element_id();
        for _ in 0..50 {
            asm.add_sample(&[1.7, 0.34, 0.42, 0.26, 0.19], eid, true);
        }
        for _ in 0..5 {
            asm.add_sample(&[1.5, 0.25, 0.35, 0.22, 0.16], eid, false);
        }

        let kappa = asm.compute_kappa();
        assert!(kappa > 0.0);

        let cert = asm.finalize().unwrap();
        assert_eq!(cert.samples.len(), 55);
        assert!(cert.kappa > 0.0);
        assert!(cert.delta > 0.0);
    }

    #[test]
    fn test_assembler_with_verified_regions() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        asm = asm.with_delta(0.05).with_param_volume(1.0);

        let eid = test_element_id();
        let region = VerifiedRegion::new(
            "test_region",
            vec![1.5, 0.25, 0.35, 0.22, 0.16],
            vec![1.7, 0.32, 0.42, 0.27, 0.19],
            eid,
        );
        asm.add_verified_region(region);

        let cert = asm.finalize().unwrap();
        assert_eq!(cert.verified_regions.len(), 1);
    }

    #[test]
    fn test_assembler_validation() {
        let asm = CertificateAssembler::new(test_scene_id());
        let issues = asm.validate();
        assert!(!issues.is_empty()); // no evidence
    }

    #[test]
    fn test_region_diameter() {
        let lo = vec![0.0, 0.0, 0.0];
        let hi = vec![1.0, 1.0, 1.0];
        let d = region_diameter(&lo, &hi);
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_regions_overlap() {
        let a = VerifiedRegion::new("a", vec![0.0, 0.0], vec![1.0, 1.0], test_element_id());
        let b = VerifiedRegion::new("b", vec![0.5, 0.5], vec![1.5, 1.5], test_element_id());
        let c = VerifiedRegion::new("c", vec![2.0, 2.0], vec![3.0, 3.0], test_element_id());

        assert!(regions_overlap(&a, &b));
        assert!(!regions_overlap(&a, &c));
    }

    #[test]
    fn test_epsilon_analytical() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        let eid = test_element_id();
        asm.add_verified_region(VerifiedRegion::new(
            "r1",
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            eid,
        ));

        let eps = asm.compute_epsilon_analytical(2.0);
        assert!(eps > 0.0);
    }

    #[test]
    fn test_fluent_builder() {
        let eid = test_element_id();
        let cert = CertificateFluentBuilder::new(test_scene_id())
            .delta(0.05)
            .param_volume(1.0)
            .dimensions(4, 2, 1)
            .total_time(1.5)
            .sample(&[1.7, 0.34, 0.42, 0.26, 0.19], eid, true)
            .sample(&[1.6, 0.32, 0.40, 0.24, 0.18], eid, true)
            .build()
            .unwrap();

        assert_eq!(cert.samples.len(), 2);
        assert!(cert.kappa > 0.0);
    }

    #[test]
    fn test_element_coverage() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        let e1 = Uuid::new_v4();
        let e2 = Uuid::new_v4();

        asm.add_sample(&[1.7, 0.34, 0.42, 0.26, 0.19], e1, true);
        asm.add_sample(&[1.6, 0.32, 0.40, 0.24, 0.18], e1, false);
        asm.add_sample(&[1.7, 0.34, 0.42, 0.26, 0.19], e2, true);
        asm.add_sample(&[1.6, 0.32, 0.40, 0.24, 0.18], e2, true);

        let cov = asm.compute_element_coverage();
        assert!((cov[&e1] - 0.5).abs() < 1e-10);
        assert!((cov[&e2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grade_computation() {
        let asm = CertificateAssembler::new(test_scene_id());
        assert!(matches!(
            asm.compute_grade(0.99),
            CertificateGrade::Full
        ));
        assert!(matches!(
            asm.compute_grade(0.95),
            CertificateGrade::Partial
        ));
        assert!(matches!(
            asm.compute_grade(0.80),
            CertificateGrade::Weak
        ));
    }

    #[test]
    fn test_merge_certificates() {
        let sid = test_scene_id();
        let eid = test_element_id();

        let cert1 = CertificateFluentBuilder::new(sid)
            .delta(0.05)
            .sample(&[1.7, 0.34, 0.42, 0.26, 0.19], eid, true)
            .build()
            .unwrap();

        let cert2 = CertificateFluentBuilder::new(sid)
            .delta(0.05)
            .sample(&[1.6, 0.32, 0.40, 0.24, 0.18], eid, true)
            .build()
            .unwrap();

        let merged = merge_certificates(&[cert1, cert2], sid, 0.05).unwrap();
        assert_eq!(merged.samples.len(), 2);
    }

    #[test]
    fn test_stratum_tracking() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        asm = asm.with_dimensions(4, 1, 1);
        let eid = test_element_id();

        asm.add_sample_with_stratum(&[1.7, 0.34, 0.42, 0.26, 0.19], eid, true, 0, 0.25);
        asm.add_sample_with_stratum(&[1.6, 0.32, 0.40, 0.24, 0.18], eid, false, 0, 0.25);
        asm.add_sample_with_stratum(&[1.8, 0.36, 0.44, 0.28, 0.20], eid, true, 1, 0.25);

        assert_eq!(asm.stratum_sample_counts.len(), 2);
        let s0 = asm.stratum_sample_counts[&0];
        assert_eq!(s0.0, 2); // total
        assert_eq!(s0.1, 1); // pass
        assert_eq!(s0.2, 1); // fail
    }

    #[test]
    fn test_volume_contribution() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        asm = asm.with_param_volume(1.0).with_delta(0.05);

        let eid = test_element_id();
        let mut region = VerifiedRegion::new(
            "green",
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.5, 0.5, 0.5],
            eid,
        );
        region.proof_status = ProofStatus::Verified;
        asm.add_verified_region(region);

        assert!(asm.volume_model.verified_fraction() > 0.0);
    }

    #[test]
    fn test_empty_finalize_succeeds() {
        let mut asm = CertificateAssembler::new(test_scene_id());
        asm.add_sample(&[1.7, 0.34, 0.42, 0.26, 0.19], test_element_id(), true);
        let cert = asm.finalize().unwrap();
        assert!(cert.kappa > 0.0);
    }

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp();
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
    }
}
