//! Certificate composition: combines Tier 1, Tier 2 sampling, and SMT
//! verification results into a unified coverage certificate.
//!
//! Implements the composition theorem:
//! - Tier 1 green volume + SMT proofs form V (verified regions)
//! - Sampling covers the remaining (yellow) space
//! - The final ε accounts for volume subtraction and linearization error

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::certificate::{
    CertificateGrade, CoverageCertificate, ProofStatus, SampleVerdict,
    VerifiedRegion, ViolationSeverity, ViolationSurface,
};
use xr_types::{ElementId, VerifierResult, NUM_BODY_PARAMS};

use crate::hoeffding::{HoeffdingBound, StratifiedBound, VolumeSubtractionModel};

// ────────────────────── Tier 1 Results ─────────────────────────────────────

/// Summary of Tier 1 verification results for composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1CompositionData {
    /// Green (provably reachable) regions per element.
    pub green_regions: Vec<CompositionRegion>,
    /// Red (provably unreachable) regions per element.
    pub red_regions: Vec<CompositionRegion>,
    /// Yellow (uncertain) regions needing further analysis.
    pub yellow_regions: Vec<CompositionRegion>,
    /// Total parameter-space volume classified as green.
    pub green_volume: f64,
    /// Total parameter-space volume classified as red.
    pub red_volume: f64,
    /// Total parameter-space volume classified as yellow.
    pub yellow_volume: f64,
}

impl Tier1CompositionData {
    pub fn new() -> Self {
        Self {
            green_regions: Vec::new(),
            red_regions: Vec::new(),
            yellow_regions: Vec::new(),
            green_volume: 0.0,
            red_volume: 0.0,
            yellow_volume: 0.0,
        }
    }

    /// Fraction of total volume that is green (verified accessible).
    pub fn green_fraction(&self) -> f64 {
        let total = self.green_volume + self.red_volume + self.yellow_volume;
        if total <= 0.0 {
            return 0.0;
        }
        self.green_volume / total
    }

    /// Fraction requiring sampling (yellow).
    pub fn yellow_fraction(&self) -> f64 {
        let total = self.green_volume + self.red_volume + self.yellow_volume;
        if total <= 0.0 {
            return 1.0;
        }
        self.yellow_volume / total
    }

    /// Add a classified region.
    pub fn add_green(&mut self, region: CompositionRegion) {
        self.green_volume += region.volume();
        self.green_regions.push(region);
    }

    pub fn add_red(&mut self, region: CompositionRegion) {
        self.red_volume += region.volume();
        self.red_regions.push(region);
    }

    pub fn add_yellow(&mut self, region: CompositionRegion) {
        self.yellow_volume += region.volume();
        self.yellow_regions.push(region);
    }
}

impl Default for Tier1CompositionData {
    fn default() -> Self {
        Self::new()
    }
}

/// A classified region in the parameter space for composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRegion {
    pub element_id: ElementId,
    pub lower: [f64; NUM_BODY_PARAMS],
    pub upper: [f64; NUM_BODY_PARAMS],
    pub linearization_error: f64,
    pub confidence: f64,
}

impl CompositionRegion {
    pub fn new(
        element_id: ElementId,
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
    ) -> Self {
        Self {
            element_id,
            lower,
            upper,
            linearization_error: 0.0,
            confidence: 1.0,
        }
    }

    /// Hyper-volume of this region.
    pub fn volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Convert to a VerifiedRegion for the certificate.
    pub fn to_verified_region(&self, label: impl Into<String>) -> VerifiedRegion {
        let mut vr = VerifiedRegion::new(
            label,
            self.lower.to_vec(),
            self.upper.to_vec(),
            self.element_id,
        );
        vr.proof_status = ProofStatus::Verified;
        vr.linearization_error = self.linearization_error;
        vr
    }
}

// ────────────────────── Sampling Results ───────────────────────────────────

/// Summary of Tier 2 sampling results for composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCompositionData {
    /// All sample verdicts collected.
    pub samples: Vec<SampleVerdict>,
    /// Per-stratum data: (stratum_idx, volume_weight, n_samples, n_pass, n_fail).
    pub strata_data: Vec<(usize, f64, usize, usize, usize)>,
    /// Confidence parameter δ used.
    pub delta: f64,
    /// Total number of strata.
    pub num_strata: usize,
}

impl SamplingCompositionData {
    pub fn new(delta: f64, num_strata: usize) -> Self {
        Self {
            samples: Vec::new(),
            strata_data: Vec::new(),
            delta,
            num_strata,
        }
    }

    /// Total number of samples.
    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }

    /// Overall pass rate.
    pub fn overall_pass_rate(&self) -> f64 {
        let n = self.samples.len();
        if n == 0 {
            return 0.0;
        }
        let pass = self.samples.iter().filter(|s| s.is_pass()).count();
        pass as f64 / n as f64
    }

    /// Add a sample verdict with stratum info.
    pub fn add_sample(
        &mut self,
        verdict: SampleVerdict,
        stratum_idx: usize,
        volume_weight: f64,
    ) {
        let passed = verdict.is_pass();
        self.samples.push(verdict);

        // Update or insert stratum data
        if let Some(sd) = self.strata_data.iter_mut().find(|s| s.0 == stratum_idx) {
            sd.2 += 1;
            if passed {
                sd.3 += 1;
            } else {
                sd.4 += 1;
            }
        } else {
            let (p, f) = if passed { (1, 0) } else { (0, 1) };
            self.strata_data
                .push((stratum_idx, volume_weight, 1, p, f));
        }
    }
}

// ────────────────────── SMT Results ───────────────────────────────────────

/// Summary of SMT verification results for composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtCompositionData {
    /// Regions verified by SMT (proved accessible).
    pub verified_regions: Vec<CompositionRegion>,
    /// Regions refuted by SMT (counterexamples found).
    pub refuted_regions: Vec<CompositionRegion>,
    /// Regions where SMT timed out.
    pub timeout_regions: Vec<CompositionRegion>,
    /// Total volume verified by SMT.
    pub verified_volume: f64,
    /// Total volume refuted by SMT.
    pub refuted_volume: f64,
}

impl SmtCompositionData {
    pub fn new() -> Self {
        Self {
            verified_regions: Vec::new(),
            refuted_regions: Vec::new(),
            timeout_regions: Vec::new(),
            verified_volume: 0.0,
            refuted_volume: 0.0,
        }
    }

    pub fn add_verified(&mut self, region: CompositionRegion) {
        self.verified_volume += region.volume();
        self.verified_regions.push(region);
    }

    pub fn add_refuted(&mut self, region: CompositionRegion) {
        self.refuted_volume += region.volume();
        self.refuted_regions.push(region);
    }

    pub fn add_timeout(&mut self, region: CompositionRegion) {
        self.timeout_regions.push(region);
    }
}

impl Default for SmtCompositionData {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────── Certificate Composer ──────────────────────────────

/// Composes Tier 1, Tier 2 sampling, and SMT results into a unified
/// coverage certificate.
///
/// Implements the composition theorem:
///
/// κ_final = (V_green + V_smt) / V_accessible + ρ_s · κ_sampling
///
/// where ρ_s = 1 - (V_green + V_smt) / V_accessible is the unverified fraction
/// and V_accessible = V_total - V_red.
pub struct CertificateComposer {
    /// Scene identifier.
    scene_id: Uuid,
    /// Total parameter-space volume.
    total_volume: f64,
    /// Number of elements in the scene.
    num_elements: usize,
    /// Number of device configurations.
    num_devices: usize,
    /// Maximum Lipschitz constant (for ε_analytical).
    l_max: f64,
    /// Estimated Lipschitz constant (for ε_estimated).
    l_hat: f64,
    /// Total wall-clock time.
    total_time_s: f64,
}

impl CertificateComposer {
    /// Create a new composer.
    pub fn new(scene_id: Uuid) -> Self {
        Self {
            scene_id,
            total_volume: 1.0,
            num_elements: 1,
            num_devices: 1,
            l_max: 0.0,
            l_hat: 0.0,
            total_time_s: 0.0,
        }
    }

    /// Set total parameter-space volume.
    pub fn with_volume(mut self, volume: f64) -> Self {
        self.total_volume = volume.max(1e-15);
        self
    }

    /// Set scene dimensions.
    pub fn with_dimensions(mut self, num_elements: usize, num_devices: usize) -> Self {
        self.num_elements = num_elements.max(1);
        self.num_devices = num_devices.max(1);
        self
    }

    /// Set Lipschitz constants.
    pub fn with_lipschitz(mut self, l_max: f64, l_hat: f64) -> Self {
        self.l_max = l_max;
        self.l_hat = l_hat;
        self
    }

    /// Set total time.
    pub fn with_time(mut self, time_s: f64) -> Self {
        self.total_time_s = time_s;
        self
    }

    /// Compute the verified volume fraction.
    ///
    /// (V_green + V_smt) / V_total
    pub fn compute_verified_volume_fraction(
        &self,
        tier1_green_volume: f64,
        smt_verified_volume: f64,
    ) -> f64 {
        if self.total_volume <= 0.0 {
            return 0.0;
        }
        ((tier1_green_volume + smt_verified_volume) / self.total_volume).min(1.0)
    }

    /// Compute the effective epsilon for the combined certificate.
    ///
    /// ε_effective = ρ_s · ε_sampling + ε_linearization
    pub fn compute_effective_epsilon(
        &self,
        verified_fraction: f64,
        sampling_epsilon: f64,
    ) -> f64 {
        let rho = (1.0 - verified_fraction).max(0.0);
        rho * sampling_epsilon
    }

    /// Compose all results into a final certificate.
    pub fn compose(
        &self,
        tier1: &Tier1CompositionData,
        sampling: &SamplingCompositionData,
        smt: &SmtCompositionData,
    ) -> VerifierResult<CoverageCertificate> {
        // Build the volume subtraction model
        let mut vol_model = VolumeSubtractionModel::new(self.total_volume);
        vol_model.add_green_volume(tier1.green_volume);
        vol_model.add_red_volume(tier1.red_volume);
        vol_model.add_smt_volume(smt.verified_volume);

        let verified_fraction = vol_model.verified_fraction();
        let rho = vol_model.effective_unverified_fraction();

        // Compute sampling epsilon using stratified bound
        let num_strata = sampling.num_strata.max(1);
        let mut strat_bound =
            StratifiedBound::new(sampling.delta, self.num_elements, self.num_devices);
        for &(_, vw, ns, np, nf) in &sampling.strata_data {
            strat_bound.add_stratum(vw, ns, np, nf);
        }

        let sampling_epsilon = if strat_bound.num_strata() > 0 {
            strat_bound.overall_epsilon()
        } else {
            let hb = HoeffdingBound::new(
                sampling.delta,
                num_strata,
                self.num_elements,
                self.num_devices,
            );
            hb.epsilon(sampling.total_samples())
        };

        let sampling_kappa = sampling.overall_pass_rate();

        // Compose final metrics
        let kappa_final = vol_model.effective_kappa(sampling_kappa);
        let eps_estimated = vol_model.effective_epsilon(sampling_epsilon);

        // Compute analytical epsilon from Lipschitz and region diameters
        let max_diameter = tier1
            .green_regions
            .iter()
            .chain(smt.verified_regions.iter())
            .map(|r| region_diameter(&r.lower, &r.upper))
            .fold(0.0_f64, f64::max);
        let eps_analytical = self.l_max * max_diameter / 2.0;

        let grade = CertificateGrade::from_metrics(kappa_final, smt.refuted_regions.len());

        // Assemble verified regions
        let mut verified_regions: Vec<VerifiedRegion> = Vec::new();
        for (i, region) in tier1.green_regions.iter().enumerate() {
            verified_regions.push(region.to_verified_region(format!("tier1_green_{}", i)));
        }
        for (i, region) in smt.verified_regions.iter().enumerate() {
            let mut vr = region.to_verified_region(format!("smt_verified_{}", i));
            vr.linearization_error = region.linearization_error;
            verified_regions.push(vr);
        }

        // Assemble violation surfaces from refuted SMT regions
        let mut violations: Vec<ViolationSurface> = Vec::new();
        for region in &smt.refuted_regions {
            let severity = if region.volume() > self.total_volume * 0.05 {
                ViolationSeverity::High
            } else if region.volume() > self.total_volume * 0.01 {
                ViolationSeverity::Medium
            } else {
                ViolationSeverity::Low
            };
            let mut v = ViolationSurface::new(
                format!("SMT-refuted region for element {:?}", region.element_id),
                region.element_id,
                severity,
            );
            v.parameter_bounds = Some((region.lower.to_vec(), region.upper.to_vec()));
            v.estimated_measure = region.volume();
            violations.push(v);
        }

        // Build element coverage map
        let mut element_coverage: HashMap<ElementId, f64> = HashMap::new();
        for sample in &sampling.samples {
            let entry = element_coverage.entry(sample.element_id).or_insert(0.0);
            *entry += if sample.is_pass() { 1.0 } else { 0.0 };
        }
        let sample_counts: HashMap<ElementId, usize> = {
            let mut counts = HashMap::new();
            for sample in &sampling.samples {
                *counts.entry(sample.element_id).or_insert(0) += 1;
            }
            counts
        };
        for (eid, total) in &sample_counts {
            if *total > 0 {
                if let Some(pass) = element_coverage.get_mut(eid) {
                    *pass /= *total as f64;
                }
            }
        }

        // Construct final certificate
        let mut metadata = HashMap::new();
        metadata.insert("composer".into(), "CertificateComposer".into());
        metadata.insert("tier1_green_volume".into(), format!("{:.6}", tier1.green_volume));
        metadata.insert("tier1_red_volume".into(), format!("{:.6}", tier1.red_volume));
        metadata.insert("smt_verified_volume".into(), format!("{:.6}", smt.verified_volume));
        metadata.insert("verified_fraction".into(), format!("{:.6}", verified_fraction));
        metadata.insert("rho_s".into(), format!("{:.6}", rho));
        metadata.insert("sampling_epsilon".into(), format!("{:.6}", sampling_epsilon));
        metadata.insert("sampling_kappa".into(), format!("{:.6}", sampling_kappa));
        metadata.insert("num_samples".into(), sampling.total_samples().to_string());
        metadata.insert("l_max".into(), format!("{:.6}", self.l_max));

        let timestamp = {
            use std::time::{SystemTime, UNIX_EPOCH};
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let d = secs / 86400;
            let y = 1970 + d / 365;
            let rd = d % 365;
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                y,
                rd / 30 + 1,
                rd % 30 + 1,
                (secs % 86400) / 3600,
                (secs % 3600) / 60,
                secs % 60
            )
        };

        Ok(CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp,
            protocol_version: xr_types::PROTOCOL_VERSION.to_string(),
            scene_id: self.scene_id,
            samples: sampling.samples.clone(),
            verified_regions,
            violations,
            epsilon_analytical: eps_analytical,
            epsilon_estimated: eps_estimated,
            delta: sampling.delta,
            kappa: kappa_final,
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
        .map(|(lo, hi)| {
            let d = hi - lo;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

// ──────────────────── Composition Utilities ───────────────────────────────

/// Compute how much volume is covered by Tier 1 green for each element.
pub fn per_element_green_volume(
    tier1: &Tier1CompositionData,
) -> HashMap<ElementId, f64> {
    let mut volumes: HashMap<ElementId, f64> = HashMap::new();
    for region in &tier1.green_regions {
        *volumes.entry(region.element_id).or_insert(0.0) += region.volume();
    }
    volumes
}

/// Compute the minimum per-element coverage across all elements.
pub fn min_element_coverage(
    tier1: &Tier1CompositionData,
    sampling: &SamplingCompositionData,
    total_volume: f64,
) -> f64 {
    let green_per_elem = per_element_green_volume(tier1);

    let mut per_elem_counts: HashMap<ElementId, (usize, usize)> = HashMap::new();
    for s in &sampling.samples {
        let entry = per_elem_counts.entry(s.element_id).or_insert((0, 0));
        entry.0 += 1;
        if s.is_pass() {
            entry.1 += 1;
        }
    }

    if per_elem_counts.is_empty() && green_per_elem.is_empty() {
        return 0.0;
    }

    let all_elements: std::collections::HashSet<ElementId> = green_per_elem
        .keys()
        .chain(per_elem_counts.keys())
        .cloned()
        .collect();

    let mut min_cov = f64::INFINITY;
    for eid in all_elements {
        let green_vol = green_per_elem.get(&eid).copied().unwrap_or(0.0);
        let (total, pass) = per_elem_counts.get(&eid).copied().unwrap_or((0, 0));
        let sampling_rate = if total > 0 {
            pass as f64 / total as f64
        } else {
            0.0
        };
        let accessible = (total_volume - tier1.red_volume).max(1e-15);
        let verified_frac = green_vol / accessible;
        let unverified_frac = 1.0 - verified_frac;
        let elem_cov = verified_frac + unverified_frac * sampling_rate;
        min_cov = min_cov.min(elem_cov);
    }

    if min_cov == f64::INFINITY {
        0.0
    } else {
        min_cov
    }
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scene_id() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000010").unwrap()
    }

    fn test_element_id() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_region(lo: f64, hi: f64) -> CompositionRegion {
        CompositionRegion::new(
            test_element_id(),
            [lo; NUM_BODY_PARAMS],
            [hi; NUM_BODY_PARAMS],
        )
    }

    #[test]
    fn test_tier1_composition_data() {
        let mut t1 = Tier1CompositionData::new();
        t1.add_green(make_region(0.0, 0.5));
        t1.add_red(make_region(0.8, 1.0));
        t1.add_yellow(make_region(0.5, 0.8));

        assert!(t1.green_volume > 0.0);
        assert!(t1.red_volume > 0.0);
        assert!(t1.yellow_volume > 0.0);
        assert!(t1.green_fraction() > 0.0);
    }

    #[test]
    fn test_sampling_composition_data() {
        let mut sd = SamplingCompositionData::new(0.05, 4);
        let eid = test_element_id();
        sd.add_sample(
            SampleVerdict::pass(vec![0.5; 5], eid),
            0,
            0.25,
        );
        sd.add_sample(
            SampleVerdict::fail(vec![0.3; 5], eid, "fail".into()),
            0,
            0.25,
        );
        assert_eq!(sd.total_samples(), 2);
        assert!((sd.overall_pass_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_smt_composition_data() {
        let mut smt = SmtCompositionData::new();
        smt.add_verified(make_region(0.0, 0.3));
        smt.add_refuted(make_region(0.7, 0.9));
        assert!(smt.verified_volume > 0.0);
        assert!(smt.refuted_volume > 0.0);
    }

    #[test]
    fn test_composer_basic() {
        let composer = CertificateComposer::new(test_scene_id())
            .with_volume(1.0)
            .with_dimensions(2, 1)
            .with_lipschitz(1.0, 0.5)
            .with_time(2.0);

        let mut tier1 = Tier1CompositionData::new();
        tier1.add_green(make_region(0.0, 0.3));
        tier1.add_red(make_region(0.9, 1.0));
        tier1.add_yellow(make_region(0.3, 0.9));

        let eid = test_element_id();
        let mut sampling = SamplingCompositionData::new(0.05, 4);
        for i in 0..50 {
            sampling.add_sample(
                SampleVerdict::pass(vec![0.3 + i as f64 * 0.01; 5], eid),
                i % 4,
                0.25,
            );
        }

        let smt = SmtCompositionData::new();

        let cert = composer.compose(&tier1, &sampling, &smt).unwrap();
        assert!(cert.kappa > 0.0);
        assert!(cert.epsilon_estimated >= 0.0);
        assert_eq!(cert.samples.len(), 50);
        assert!(!cert.verified_regions.is_empty());
    }

    #[test]
    fn test_verified_volume_fraction() {
        let composer = CertificateComposer::new(test_scene_id()).with_volume(1.0);
        let frac = composer.compute_verified_volume_fraction(0.3, 0.2);
        assert!((frac - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_effective_epsilon() {
        let composer = CertificateComposer::new(test_scene_id());
        let eps = composer.compute_effective_epsilon(0.5, 0.1);
        assert!((eps - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_fully_verified() {
        let composer = CertificateComposer::new(test_scene_id())
            .with_volume(1.0)
            .with_dimensions(1, 1);

        let mut tier1 = Tier1CompositionData::new();
        tier1.add_green(make_region(0.0, 1.0));

        let sampling = SamplingCompositionData::new(0.05, 1);
        let smt = SmtCompositionData::new();

        let cert = composer.compose(&tier1, &sampling, &smt).unwrap();
        assert!(cert.kappa >= 0.99);
    }

    #[test]
    fn test_with_smt_refutations() {
        let composer = CertificateComposer::new(test_scene_id())
            .with_volume(1.0)
            .with_dimensions(1, 1);

        let tier1 = Tier1CompositionData::new();

        let eid = test_element_id();
        let mut sampling = SamplingCompositionData::new(0.05, 1);
        for _ in 0..100 {
            sampling.add_sample(
                SampleVerdict::pass(vec![0.5; 5], eid),
                0,
                1.0,
            );
        }

        let mut smt = SmtCompositionData::new();
        smt.add_refuted(make_region(0.8, 0.9));

        let cert = composer.compose(&tier1, &sampling, &smt).unwrap();
        assert!(!cert.violations.is_empty());
    }

    #[test]
    fn test_per_element_green_volume() {
        let mut t1 = Tier1CompositionData::new();
        let e1 = Uuid::new_v4();
        let e2 = Uuid::new_v4();
        t1.add_green(CompositionRegion::new(e1, [0.0; 5], [0.5; 5]));
        t1.add_green(CompositionRegion::new(e2, [0.0; 5], [0.3; 5]));

        let vols = per_element_green_volume(&t1);
        assert!(vols[&e1] > vols[&e2]);
    }

    #[test]
    fn test_min_element_coverage() {
        let mut t1 = Tier1CompositionData::new();
        let e1 = Uuid::new_v4();
        t1.add_green(CompositionRegion::new(e1, [0.0; 5], [0.5; 5]));

        let mut sampling = SamplingCompositionData::new(0.05, 1);
        for _ in 0..10 {
            sampling.add_sample(
                SampleVerdict::pass(vec![0.5; 5], e1),
                0,
                1.0,
            );
        }

        let min_cov = min_element_coverage(&t1, &sampling, 1.0);
        assert!(min_cov > 0.0);
    }

    #[test]
    fn test_region_diameter() {
        let d = region_diameter(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_composition_region_volume() {
        let r = make_region(0.0, 1.0);
        assert!((r.volume() - 1.0).abs() < 1e-10);

        let r2 = make_region(0.0, 0.5);
        assert!((r2.volume() - 0.5_f64.powi(5)).abs() < 1e-10);
    }

    #[test]
    fn test_empty_composition() {
        let composer = CertificateComposer::new(test_scene_id())
            .with_volume(1.0)
            .with_dimensions(1, 1);

        let tier1 = Tier1CompositionData::new();
        let sampling = SamplingCompositionData::new(0.05, 1);
        let smt = SmtCompositionData::new();

        let cert = composer.compose(&tier1, &sampling, &smt).unwrap();
        assert_eq!(cert.kappa, 0.0);
    }
}
