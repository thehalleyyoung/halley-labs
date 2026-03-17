//! Population PK modeling with covariate effects.
//!
//! Provides allometric scaling, renal/hepatic adjustments,
//! Monte Carlo sampling, and worst-case parameter computation.

use serde::{Deserialize, Serialize};
use guardpharma_types::drug::{PatientInfo, Sex, ChildPughClass, AscitesGrade};
use guardpharma_types::concentration::ConcentrationInterval;

// ---------------------------------------------------------------------------
// PopulationModel
// ---------------------------------------------------------------------------

/// Population PK model with parameter variability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationModel {
    pub clearance_mean: f64,
    pub clearance_cv: f64,
    pub volume_mean: f64,
    pub volume_cv: f64,
    pub bioavailability_mean: f64,
    pub bioavailability_cv: f64,
    pub absorption_rate_mean: Option<f64>,
    pub absorption_rate_cv: Option<f64>,
    pub covariate_model: Option<CovariateModel>,
}

impl PopulationModel {
    pub fn new(cl: f64, cl_cv: f64, v: f64, v_cv: f64, f_mean: f64, f_cv: f64) -> Self {
        Self {
            clearance_mean: cl, clearance_cv: cl_cv,
            volume_mean: v, volume_cv: v_cv,
            bioavailability_mean: f_mean, bioavailability_cv: f_cv,
            absorption_rate_mean: None, absorption_rate_cv: None,
            covariate_model: None,
        }
    }

    pub fn with_covariate_model(mut self, cm: CovariateModel) -> Self {
        self.covariate_model = Some(cm);
        self
    }

    pub fn typical_clearance(&self) -> f64 { self.clearance_mean }
    pub fn typical_volume(&self) -> f64 { self.volume_mean }

    pub fn clearance_bounds(&self, n_sigma: f64) -> (f64, f64) {
        let sd = self.clearance_mean * self.clearance_cv;
        ((self.clearance_mean - n_sigma * sd).max(0.01),
         self.clearance_mean + n_sigma * sd)
    }

    pub fn volume_bounds(&self, n_sigma: f64) -> (f64, f64) {
        let sd = self.volume_mean * self.volume_cv;
        ((self.volume_mean - n_sigma * sd).max(0.01),
         self.volume_mean + n_sigma * sd)
    }

    pub fn adjusted_clearance(&self, patient: &PatientInfo) -> f64 {
        if let Some(ref cm) = self.covariate_model {
            cm.apply_to_clearance(self.clearance_mean, patient)
        } else {
            self.clearance_mean
        }
    }

    pub fn adjusted_volume(&self, patient: &PatientInfo) -> f64 {
        if let Some(ref cm) = self.covariate_model {
            cm.apply_to_volume(self.volume_mean, patient)
        } else {
            self.volume_mean
        }
    }
}

// ---------------------------------------------------------------------------
// CovariateModel
// ---------------------------------------------------------------------------

/// Covariate effects on PK parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateModel {
    pub weight_effect_cl: f64,
    pub weight_effect_v: f64,
    pub age_effect_cl: f64,
    pub sex_effect_cl: f64,
    pub renal_effect_cl: f64,
    pub hepatic_effect_cl: f64,
    pub reference_weight: f64,
}

impl CovariateModel {
    pub fn default_model() -> Self {
        Self {
            weight_effect_cl: 0.75,
            weight_effect_v: 1.0,
            age_effect_cl: -0.02,
            sex_effect_cl: 0.85,
            renal_effect_cl: 0.3,
            hepatic_effect_cl: 0.5,
            reference_weight: 70.0,
        }
    }

    pub fn apply_to_clearance(&self, base_cl: f64, patient: &PatientInfo) -> f64 {
        let mut cl = base_cl;
        cl *= self.weight_adjustment_cl(patient.weight_kg);
        cl *= self.age_adjustment(patient.age_years);
        cl *= self.sex_adjustment(patient.sex);

        // Renal adjustment
        let crcl = cockcroft_gault(
            patient.age_years, patient.weight_kg,
            patient.serum_creatinine, patient.sex,
        );
        let renal_factor = renal_dose_adjustment(1.0, self.renal_effect_cl, crcl, 120.0);
        cl *= renal_factor;

        cl
    }

    pub fn apply_to_volume(&self, base_v: f64, patient: &PatientInfo) -> f64 {
        base_v * self.weight_adjustment_v(patient.weight_kg)
    }

    pub fn weight_adjustment_cl(&self, weight: f64) -> f64 {
        (weight / self.reference_weight).powf(self.weight_effect_cl)
    }

    pub fn weight_adjustment_v(&self, weight: f64) -> f64 {
        (weight / self.reference_weight).powf(self.weight_effect_v)
    }

    pub fn age_adjustment(&self, age: f64) -> f64 {
        let decades_above_40 = ((age - 40.0) / 10.0).max(0.0);
        (1.0 + self.age_effect_cl * decades_above_40).max(0.3)
    }

    pub fn sex_adjustment(&self, sex: Sex) -> f64 {
        match sex {
            Sex::Female => self.sex_effect_cl,
            Sex::Male => 1.0,
        }
    }

    pub fn renal_adjustment(&self, patient: &PatientInfo) -> f64 {
        let crcl = cockcroft_gault(
            patient.age_years, patient.weight_kg,
            patient.serum_creatinine, patient.sex,
        );
        renal_dose_adjustment(1.0, self.renal_effect_cl, crcl, 120.0)
    }
}

// ---------------------------------------------------------------------------
// Allometric scaling functions
// ---------------------------------------------------------------------------

/// Allometric clearance scaling: CL = CL_ref * (W/W_ref)^exp.
pub fn allometric_clearance(base_cl: f64, weight_kg: f64, exponent: f64) -> f64 {
    base_cl * (weight_kg / 70.0).powf(exponent)
}

/// Allometric volume scaling: V = V_ref * (W/W_ref)^exp.
pub fn allometric_volume(base_v: f64, weight_kg: f64, exponent: f64) -> f64 {
    base_v * (weight_kg / 70.0).powf(exponent)
}

/// Body surface area (DuBois formula, m²).
pub fn body_surface_area(weight_kg: f64, height_cm: f64) -> f64 {
    0.007184 * weight_kg.powf(0.425) * height_cm.powf(0.725)
}

/// Ideal body weight (Devine formula, kg).
pub fn ideal_body_weight(height_cm: f64, sex: Sex) -> f64 {
    let height_in = height_cm / 2.54;
    match sex {
        Sex::Male => 50.0 + 2.3 * (height_in - 60.0).max(0.0),
        Sex::Female => 45.5 + 2.3 * (height_in - 60.0).max(0.0),
    }
}

/// Adjusted body weight.
pub fn adjusted_body_weight(actual: f64, ideal: f64, factor: f64) -> f64 {
    ideal + factor * (actual - ideal)
}

/// Lean body mass (Boer formula, kg).
pub fn lean_body_mass(weight_kg: f64, height_cm: f64, sex: Sex) -> f64 {
    match sex {
        Sex::Male => 0.407 * weight_kg + 0.267 * height_cm - 19.2,
        Sex::Female => 0.252 * weight_kg + 0.473 * height_cm - 48.3,
    }
}

// ---------------------------------------------------------------------------
// Renal adjustment functions
// ---------------------------------------------------------------------------

/// Cockcroft-Gault creatinine clearance (mL/min).
pub fn cockcroft_gault(age: f64, weight_kg: f64, scr: f64, sex: Sex) -> f64 {
    let factor = match sex {
        Sex::Female => 0.85,
        Sex::Male => 1.0,
    };
    ((140.0 - age) * weight_kg * factor) / (72.0 * scr)
}

/// MDRD eGFR (mL/min/1.73m²).
pub fn mdrd(scr: f64, age: f64, sex: Sex) -> f64 {
    let sex_factor = match sex {
        Sex::Female => 0.742,
        Sex::Male => 1.0,
    };
    175.0 * scr.powf(-1.154) * age.powf(-0.203) * sex_factor
}

/// CKD-EPI eGFR (mL/min/1.73m²) — 2009 equation.
pub fn ckd_epi(scr: f64, age: f64, sex: Sex) -> f64 {
    let (kappa, alpha, sex_factor) = match sex {
        Sex::Female => (0.7, -0.329, 1.018),
        Sex::Male => (0.9, -0.411, 1.0),
    };
    let ratio = scr / kappa;
    let term = if ratio < 1.0 {
        ratio.powf(alpha)
    } else {
        ratio.powf(-1.209)
    };
    141.0 * term * 0.993_f64.powf(age) * sex_factor
}

/// Renal dose adjustment.
/// CL_adj = CL * (non_renal_fraction + renal_fraction * patient_CrCl / normal_CrCl)
pub fn renal_dose_adjustment(
    base_clearance: f64,
    renal_fraction: f64,
    patient_crcl: f64,
    normal_crcl: f64,
) -> f64 {
    let non_renal = 1.0 - renal_fraction;
    base_clearance * (non_renal + renal_fraction * patient_crcl / normal_crcl)
}

// ---------------------------------------------------------------------------
// Hepatic adjustment functions
// ---------------------------------------------------------------------------

/// Child-Pugh score from patient info.
pub fn child_pugh_score(patient: &PatientInfo) -> u8 {
    let mut score: u8 = 0;

    // Bilirubin
    if let Some(bil) = patient.bilirubin {
        score += if bil < 2.0 { 1 } else if bil <= 3.0 { 2 } else { 3 };
    } else {
        score += 1;
    }

    // Albumin
    if let Some(alb) = patient.albumin {
        score += if alb > 3.5 { 1 } else if alb >= 2.8 { 2 } else { 3 };
    } else {
        score += 1;
    }

    // INR
    if let Some(inr) = patient.inr {
        score += if inr < 1.7 { 1 } else if inr <= 2.3 { 2 } else { 3 };
    } else {
        score += 1;
    }

    // Encephalopathy
    if let Some(enc) = patient.encephalopathy_grade {
        score += if enc == 0 { 1 } else if enc <= 2 { 2 } else { 3 };
    } else {
        score += 1;
    }

    // Ascites
    if let Some(asc) = &patient.ascites {
        score += match asc {
            AscitesGrade::None => 1,
            AscitesGrade::Mild => 2,
            AscitesGrade::ModerateToSevere => 3,
        };
    } else {
        score += 1;
    }

    score
}

/// Child-Pugh class from score.
pub fn child_pugh_class(score: u8) -> ChildPughClass {
    if score <= 6 { ChildPughClass::A }
    else if score <= 9 { ChildPughClass::B }
    else { ChildPughClass::C }
}

/// Hepatic dose adjustment based on Child-Pugh class.
pub fn hepatic_dose_adjustment(
    base_clearance: f64,
    hepatic_fraction: f64,
    class: ChildPughClass,
) -> f64 {
    let reduction = match class {
        ChildPughClass::A => 1.0,
        ChildPughClass::B => 0.6,
        ChildPughClass::C => 0.35,
    };
    let non_hepatic = 1.0 - hepatic_fraction;
    base_clearance * (non_hepatic + hepatic_fraction * reduction)
}

// ---------------------------------------------------------------------------
// ParameterSampler
// ---------------------------------------------------------------------------

/// Monte Carlo parameter sampler for population PK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSampler {
    pub population: PopulationModel,
    pub n_samples: usize,
    pub seed: Option<u64>,
}

impl ParameterSampler {
    pub fn new(pop: PopulationModel, n: usize) -> Self {
        Self { population: pop, n_samples: n, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn next_uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn next_normal(state: &mut u64) -> f64 {
        // Box-Muller transform
        let u1 = Self::next_uniform(state).max(1e-15);
        let u2 = Self::next_uniform(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn sample_lognormal(state: &mut u64, mean: f64, cv: f64) -> f64 {
        if cv <= 0.0 { return mean; }
        let omega2 = (1.0 + cv * cv).ln();
        let mu = mean.ln() - omega2 / 2.0;
        (mu + omega2.sqrt() * Self::next_normal(state)).exp()
    }

    pub fn sample_clearances(&self) -> Vec<f64> {
        let mut state = self.seed.unwrap_or(42);
        (0..self.n_samples)
            .map(|_| Self::sample_lognormal(&mut state, self.population.clearance_mean, self.population.clearance_cv))
            .collect()
    }

    pub fn sample_volumes(&self) -> Vec<f64> {
        let mut state = self.seed.unwrap_or(42).wrapping_add(12345);
        (0..self.n_samples)
            .map(|_| Self::sample_lognormal(&mut state, self.population.volume_mean, self.population.volume_cv))
            .collect()
    }

    pub fn sample_steady_states(&self, dose: f64, interval: f64) -> Vec<f64> {
        let cls = self.sample_clearances();
        let f_mean = self.population.bioavailability_mean;
        cls.iter().map(|&cl| f_mean * dose / (cl * interval)).collect()
    }

    pub fn percentile(values: &[f64], p: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn confidence_interval(values: &[f64], level: f64) -> (f64, f64) {
        let alpha = (1.0 - level) / 2.0;
        (Self::percentile(values, alpha), Self::percentile(values, 1.0 - alpha))
    }
}

// ---------------------------------------------------------------------------
// WorstCaseParameters
// ---------------------------------------------------------------------------

/// Worst-case PK parameters from population bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstCaseParameters {
    pub min_clearance: f64,
    pub max_clearance: f64,
    pub min_volume: f64,
    pub max_volume: f64,
    pub min_bioavailability: f64,
    pub max_bioavailability: f64,
}

impl WorstCaseParameters {
    pub fn from_population(pop: &PopulationModel, n_sigma: f64) -> Self {
        let (cl_lo, cl_hi) = pop.clearance_bounds(n_sigma);
        let (v_lo, v_hi) = pop.volume_bounds(n_sigma);
        let f_sd = pop.bioavailability_mean * pop.bioavailability_cv;
        Self {
            min_clearance: cl_lo,
            max_clearance: cl_hi,
            min_volume: v_lo,
            max_volume: v_hi,
            min_bioavailability: (pop.bioavailability_mean - n_sigma * f_sd).max(0.01).min(1.0),
            max_bioavailability: (pop.bioavailability_mean + n_sigma * f_sd).min(1.0),
        }
    }

    pub fn worst_case_high_concentration(&self, dose: f64, interval: f64) -> f64 {
        self.max_bioavailability * dose / (self.min_clearance * interval)
    }

    pub fn worst_case_low_concentration(&self, dose: f64, interval: f64) -> f64 {
        self.min_bioavailability * dose / (self.max_clearance * interval)
    }

    pub fn concentration_interval(&self, dose: f64, interval: f64) -> ConcentrationInterval {
        ConcentrationInterval::new(
            self.worst_case_low_concentration(dose, interval).max(0.0),
            self.worst_case_high_concentration(dose, interval),
        )
    }
}

// ---------------------------------------------------------------------------
// BoundedParameterSpace
// ---------------------------------------------------------------------------

/// Bounded parameter space for worst-case analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedParameterSpace {
    pub clearance_range: (f64, f64),
    pub volume_range: (f64, f64),
    pub bioavailability_range: (f64, f64),
    pub absorption_rate_range: Option<(f64, f64)>,
}

impl BoundedParameterSpace {
    pub fn from_worst_case(wc: &WorstCaseParameters) -> Self {
        Self {
            clearance_range: (wc.min_clearance, wc.max_clearance),
            volume_range: (wc.min_volume, wc.max_volume),
            bioavailability_range: (wc.min_bioavailability, wc.max_bioavailability),
            absorption_rate_range: None,
        }
    }

    pub fn contains_parameters(&self, cl: f64, v: f64, f_bio: f64) -> bool {
        cl >= self.clearance_range.0 && cl <= self.clearance_range.1
            && v >= self.volume_range.0 && v <= self.volume_range.1
            && f_bio >= self.bioavailability_range.0 && f_bio <= self.bioavailability_range.1
    }

    pub fn steady_state_bounds(&self, dose: f64, interval: f64) -> ConcentrationInterval {
        let lo = self.bioavailability_range.0 * dose / (self.clearance_range.1 * interval);
        let hi = self.bioavailability_range.1 * dose / (self.clearance_range.0 * interval);
        ConcentrationInterval::new(lo.max(0.0), hi)
    }

    /// All 8 corners of the 3D parameter box.
    pub fn corner_cases(&self) -> Vec<(f64, f64, f64)> {
        let cl = [self.clearance_range.0, self.clearance_range.1];
        let v = [self.volume_range.0, self.volume_range.1];
        let f = [self.bioavailability_range.0, self.bioavailability_range.1];
        let mut corners = Vec::with_capacity(8);
        for &c in &cl {
            for &vi in &v {
                for &fi in &f {
                    corners.push((c, vi, fi));
                }
            }
        }
        corners
    }

    /// Evaluate Css = F*dose/(CL*tau) at all corners.
    pub fn worst_case_steady_states(&self, dose: f64, interval: f64) -> Vec<f64> {
        self.corner_cases()
            .iter()
            .map(|&(cl, _v, f_bio)| f_bio * dose / (cl * interval))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cockcroft_gault_male() {
        let crcl = cockcroft_gault(70.0, 80.0, 1.0, Sex::Male);
        // (140-70)*80/(72*1) = 77.78
        assert!((crcl - 77.78).abs() < 0.1);
    }

    #[test]
    fn test_cockcroft_gault_female() {
        let male = cockcroft_gault(70.0, 80.0, 1.0, Sex::Male);
        let female = cockcroft_gault(70.0, 80.0, 1.0, Sex::Female);
        assert!((female - male * 0.85).abs() < 0.1);
    }

    #[test]
    fn test_mdrd() {
        let egfr = mdrd(1.0, 50.0, Sex::Male);
        assert!(egfr > 50.0 && egfr < 200.0);
    }

    #[test]
    fn test_ckd_epi() {
        let egfr = ckd_epi(1.0, 50.0, Sex::Male);
        assert!(egfr > 50.0 && egfr < 200.0);
    }

    #[test]
    fn test_allometric_scaling() {
        let cl_80 = allometric_clearance(10.0, 80.0, 0.75);
        let cl_70 = allometric_clearance(10.0, 70.0, 0.75);
        assert!(cl_80 > cl_70);
    }

    #[test]
    fn test_body_surface_area() {
        let bsa = body_surface_area(70.0, 170.0);
        assert!((bsa - 1.8).abs() < 0.2);
    }

    #[test]
    fn test_child_pugh_scoring() {
        let p = PatientInfo::default();
        let score = child_pugh_score(&p);
        assert!(score >= 5 && score <= 6); // Normal patient = class A
        assert_eq!(child_pugh_class(score), ChildPughClass::A);
    }

    #[test]
    fn test_hepatic_adjustment() {
        let cl_a = hepatic_dose_adjustment(10.0, 0.7, ChildPughClass::A);
        let cl_c = hepatic_dose_adjustment(10.0, 0.7, ChildPughClass::C);
        assert!(cl_a > cl_c);
        assert!((cl_a - 10.0).abs() < 0.01); // Class A: no reduction
    }

    #[test]
    fn test_worst_case_parameters() {
        let pop = PopulationModel::new(10.0, 0.3, 50.0, 0.2, 0.8, 0.1);
        let wc = WorstCaseParameters::from_population(&pop, 2.0);
        assert!(wc.min_clearance < 10.0);
        assert!(wc.max_clearance > 10.0);
        let ci = wc.concentration_interval(100.0, 12.0);
        assert!(ci.lo > 0.0 && ci.hi > ci.lo);
    }

    #[test]
    fn test_bounded_parameter_space() {
        let wc = WorstCaseParameters {
            min_clearance: 5.0, max_clearance: 15.0,
            min_volume: 30.0, max_volume: 70.0,
            min_bioavailability: 0.6, max_bioavailability: 1.0,
        };
        let bps = BoundedParameterSpace::from_worst_case(&wc);
        assert!(bps.contains_parameters(10.0, 50.0, 0.8));
        assert!(!bps.contains_parameters(2.0, 50.0, 0.8));
        let corners = bps.corner_cases();
        assert_eq!(corners.len(), 8);
    }

    #[test]
    fn test_covariate_model() {
        let cm = CovariateModel::default_model();
        let patient = PatientInfo::default();
        let cl = cm.apply_to_clearance(10.0, &patient);
        assert!(cl > 0.0);
        let v = cm.apply_to_volume(50.0, &patient);
        assert!(v > 0.0);
    }

    #[test]
    fn test_renal_dose_adjustment() {
        let adj = renal_dose_adjustment(10.0, 0.5, 60.0, 120.0);
        // non_renal=0.5 + 0.5*60/120 = 0.5 + 0.25 = 0.75
        assert!((adj - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_ideal_body_weight() {
        let ibw = ideal_body_weight(170.0, Sex::Male);
        assert!(ibw > 60.0 && ibw < 80.0);
    }

    #[test]
    fn test_population_model_adjusted() {
        let pop = PopulationModel::new(10.0, 0.3, 50.0, 0.2, 0.8, 0.1)
            .with_covariate_model(CovariateModel::default_model());
        let patient = PatientInfo::default();
        let cl = pop.adjusted_clearance(&patient);
        assert!(cl > 0.0);
    }
}
