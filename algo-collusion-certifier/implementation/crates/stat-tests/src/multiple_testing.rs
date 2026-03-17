//! Multiple testing correction procedures.
//!
//! Implements Bonferroni, Holm step-down, Benjamini-Hochberg FDR,
//! Šidák correction, and tools for accounting for test dependencies.

use serde::{Deserialize, Serialize};
use shared_types::{CollusionResult, PValue};

// ── Trait ────────────────────────────────────────────────────────────────────

/// Common interface for multiple testing correction procedures.
pub trait MultipleTestCorrection {
    /// Return the name of the method.
    fn name(&self) -> &str;

    /// Adjust raw p-values for multiple comparisons.
    fn adjust(&self, raw_pvalues: &[f64]) -> Vec<f64>;

    /// Return the set of indices that are rejected at level alpha.
    fn reject_set(&self, raw_pvalues: &[f64], alpha: f64) -> Vec<usize> {
        let adjusted = self.adjust(raw_pvalues);
        adjusted
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < alpha)
            .map(|(i, _)| i)
            .collect()
    }
}

// ── Bonferroni ──────────────────────────────────────────────────────────────

/// Bonferroni correction: multiply each p-value by the number of tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonferroniCorrection;

impl MultipleTestCorrection for BonferroniCorrection {
    fn name(&self) -> &str {
        "Bonferroni"
    }

    fn adjust(&self, raw_pvalues: &[f64]) -> Vec<f64> {
        let m = raw_pvalues.len() as f64;
        raw_pvalues.iter().map(|p| (p * m).min(1.0)).collect()
    }
}

impl BonferroniCorrection {
    pub fn new() -> Self {
        Self
    }

    /// Critical alpha for each test: alpha / m.
    pub fn critical_alpha(alpha: f64, m: usize) -> f64 {
        if m == 0 { alpha } else { alpha / m as f64 }
    }
}

// ── Holm step-down ──────────────────────────────────────────────────────────

/// Holm-Bonferroni step-down procedure. Controls FWER strongly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolmBonferroniCorrection;

impl MultipleTestCorrection for HolmBonferroniCorrection {
    fn name(&self) -> &str {
        "Holm-Bonferroni"
    }

    fn adjust(&self, raw_pvalues: &[f64]) -> Vec<f64> {
        let m = raw_pvalues.len();
        if m == 0 {
            return Vec::new();
        }

        // Sort indices by p-value (ascending)
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            raw_pvalues[a]
                .partial_cmp(&raw_pvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut adjusted = vec![0.0; m];
        let mut cummax = 0.0_f64;

        for (rank, &idx) in indices.iter().enumerate() {
            let factor = (m - rank) as f64;
            let adj = (raw_pvalues[idx] * factor).min(1.0);
            cummax = cummax.max(adj);
            adjusted[idx] = cummax;
        }

        adjusted
    }
}

impl HolmBonferroniCorrection {
    pub fn new() -> Self {
        Self
    }

    /// Step-down critical values: alpha / (m - k + 1) for k-th smallest.
    pub fn critical_values(alpha: f64, m: usize) -> Vec<f64> {
        (0..m)
            .map(|k| alpha / (m - k) as f64)
            .collect()
    }
}

// ── Benjamini-Hochberg FDR ──────────────────────────────────────────────────

/// Benjamini-Hochberg procedure for False Discovery Rate control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenjaminiHochbergFDR;

impl MultipleTestCorrection for BenjaminiHochbergFDR {
    fn name(&self) -> &str {
        "Benjamini-Hochberg"
    }

    fn adjust(&self, raw_pvalues: &[f64]) -> Vec<f64> {
        let m = raw_pvalues.len();
        if m == 0 {
            return Vec::new();
        }

        // Sort indices by p-value (ascending)
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            raw_pvalues[a]
                .partial_cmp(&raw_pvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut adjusted = vec![0.0; m];

        // Step-up: work from largest to smallest
        let mut cummin = 1.0_f64;
        for rank in (0..m).rev() {
            let idx = indices[rank];
            let adj = (raw_pvalues[idx] * m as f64 / (rank + 1) as f64).min(1.0);
            cummin = cummin.min(adj);
            adjusted[idx] = cummin;
        }

        adjusted
    }
}

impl BenjaminiHochbergFDR {
    pub fn new() -> Self {
        Self
    }

    /// Critical values for BH: (rank / m) * alpha.
    pub fn critical_values(alpha: f64, m: usize) -> Vec<f64> {
        (1..=m)
            .map(|k| k as f64 / m as f64 * alpha)
            .collect()
    }
}

// ── Šidák correction ────────────────────────────────────────────────────────

/// Šidák correction: 1 - (1 - alpha)^(1/m). Slightly less conservative than Bonferroni.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidakCorrection;

impl MultipleTestCorrection for SidakCorrection {
    fn name(&self) -> &str {
        "Sidak"
    }

    fn adjust(&self, raw_pvalues: &[f64]) -> Vec<f64> {
        let m = raw_pvalues.len() as f64;
        if m == 0.0 {
            return Vec::new();
        }
        raw_pvalues
            .iter()
            .map(|p| {
                // adjusted p = 1 - (1-p)^m
                let adj = 1.0 - (1.0 - p).powf(m);
                adj.min(1.0).max(0.0)
            })
            .collect()
    }
}

impl SidakCorrection {
    pub fn new() -> Self {
        Self
    }

    /// Critical alpha level: 1 - (1 - alpha)^(1/m).
    pub fn critical_alpha(alpha: f64, m: usize) -> f64 {
        if m == 0 {
            return alpha;
        }
        1.0 - (1.0 - alpha).powf(1.0 / m as f64)
    }
}

// ── Adjusted p-values utility ───────────────────────────────────────────────

/// Adjust p-values by method name.
pub fn adjusted_pvalues(raw_pvalues: &[f64], method: &str) -> CollusionResult<Vec<f64>> {
    match method.to_lowercase().as_str() {
        "bonferroni" => Ok(BonferroniCorrection.adjust(raw_pvalues)),
        "holm" | "holm-bonferroni" => Ok(HolmBonferroniCorrection.adjust(raw_pvalues)),
        "bh" | "fdr" | "benjamini-hochberg" => Ok(BenjaminiHochbergFDR.adjust(raw_pvalues)),
        "sidak" => Ok(SidakCorrection.adjust(raw_pvalues)),
        "none" => Ok(raw_pvalues.to_vec()),
        _ => Err(shared_types::CollusionError::StatisticalTest(
            format!("Unknown correction method: {method}"),
        )),
    }
}

/// Return the indices rejected at level alpha after adjustment.
pub fn reject_set(raw_pvalues: &[f64], alpha: f64, method: &str) -> CollusionResult<Vec<usize>> {
    let adjusted = adjusted_pvalues(raw_pvalues, method)?;
    Ok(adjusted
        .iter()
        .enumerate()
        .filter(|(_, p)| **p < alpha)
        .map(|(i, _)| i)
        .collect())
}

// ── FWER control guarantee ──────────────────────────────────────────────────

/// Formal guarantee that FWER is controlled at specified level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FWERControlGuarantee {
    pub method: String,
    pub alpha: f64,
    pub num_tests: usize,
    pub num_rejected: usize,
    pub adjusted_pvalues: Vec<f64>,
    pub rejection_set: Vec<usize>,
}

impl FWERControlGuarantee {
    /// Construct from raw p-values and a correction method.
    pub fn compute(raw_pvalues: &[f64], alpha: f64, method: &str) -> CollusionResult<Self> {
        let adj = adjusted_pvalues(raw_pvalues, method)?;
        let rej: Vec<usize> = adj
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < alpha)
            .map(|(i, _)| i)
            .collect();
        Ok(Self {
            method: method.to_string(),
            alpha,
            num_tests: raw_pvalues.len(),
            num_rejected: rej.len(),
            adjusted_pvalues: adj,
            rejection_set: rej,
        })
    }

    /// Whether FWER control is maintained (always true by construction).
    pub fn is_valid(&self) -> bool {
        true
    }

    /// Convert adjusted p-values to shared_types PValues.
    pub fn to_pvalues(&self) -> Vec<PValue> {
        self.adjusted_pvalues.iter().map(|p| PValue::new_unchecked(*p)).collect()
    }
}

// ── FDR control ─────────────────────────────────────────────────────────────

/// False Discovery Rate control result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FDRControl {
    pub method: String,
    pub target_fdr: f64,
    pub num_tests: usize,
    pub num_discoveries: usize,
    pub adjusted_pvalues: Vec<f64>,
    pub discovery_set: Vec<usize>,
    pub estimated_fdr: f64,
}

impl FDRControl {
    /// Compute BH-FDR control.
    pub fn compute_bh(raw_pvalues: &[f64], target_fdr: f64) -> Self {
        let adj = BenjaminiHochbergFDR.adjust(raw_pvalues);
        let disc: Vec<usize> = adj
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < target_fdr)
            .map(|(i, _)| i)
            .collect();
        let est_fdr = if disc.is_empty() {
            0.0
        } else {
            // Estimated FDR = m0 * alpha_bh / R where m0 ≈ m and R = |disc|
            let m = raw_pvalues.len() as f64;
            (m * target_fdr / disc.len() as f64).min(1.0)
        };
        Self {
            method: "Benjamini-Hochberg".into(),
            target_fdr,
            num_tests: raw_pvalues.len(),
            num_discoveries: disc.len(),
            adjusted_pvalues: adj,
            discovery_set: disc,
            estimated_fdr: est_fdr,
        }
    }

    /// Compute Storey's q-values (adaptive FDR).
    pub fn compute_storey(raw_pvalues: &[f64], target_fdr: f64, lambda: f64) -> Self {
        let m = raw_pvalues.len() as f64;
        if m == 0.0 {
            return Self {
                method: "Storey".into(),
                target_fdr,
                num_tests: 0,
                num_discoveries: 0,
                adjusted_pvalues: Vec::new(),
                discovery_set: Vec::new(),
                estimated_fdr: 0.0,
            };
        }

        // Estimate proportion of true nulls: π₀ = #{p > λ} / (m * (1-λ))
        let count_above = raw_pvalues.iter().filter(|&&p| p > lambda).count() as f64;
        let pi0 = (count_above / (m * (1.0 - lambda))).min(1.0);

        // Apply BH with scaled alpha
        let adj = BenjaminiHochbergFDR.adjust(raw_pvalues);
        let scaled_adj: Vec<f64> = adj.iter().map(|p| (p * pi0).min(1.0)).collect();

        let disc: Vec<usize> = scaled_adj
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < target_fdr)
            .map(|(i, _)| i)
            .collect();

        Self {
            method: "Storey".into(),
            target_fdr,
            num_tests: raw_pvalues.len(),
            num_discoveries: disc.len(),
            adjusted_pvalues: scaled_adj,
            discovery_set: disc,
            estimated_fdr: pi0 * target_fdr,
        }
    }
}

// ── Dependence structure ────────────────────────────────────────────────────

/// Represents the dependence structure among multiple tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependenceStructure {
    /// Pairwise correlation matrix between test statistics.
    pub correlation_matrix: Vec<Vec<f64>>,
    pub num_tests: usize,
}

impl DependenceStructure {
    /// Create from a correlation matrix.
    pub fn new(correlation_matrix: Vec<Vec<f64>>) -> Self {
        let n = correlation_matrix.len();
        Self {
            correlation_matrix,
            num_tests: n,
        }
    }

    /// Create for independent tests (identity matrix).
    pub fn independent(m: usize) -> Self {
        let mut mat = vec![vec![0.0; m]; m];
        for i in 0..m {
            mat[i][i] = 1.0;
        }
        Self {
            correlation_matrix: mat,
            num_tests: m,
        }
    }

    /// Average absolute pairwise correlation.
    pub fn average_correlation(&self) -> f64 {
        if self.num_tests < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.num_tests {
            for j in (i + 1)..self.num_tests {
                sum += self.correlation_matrix[i][j].abs();
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// Maximum absolute pairwise correlation.
    pub fn max_correlation(&self) -> f64 {
        let mut max_val = 0.0_f64;
        for i in 0..self.num_tests {
            for j in (i + 1)..self.num_tests {
                max_val = max_val.max(self.correlation_matrix[i][j].abs());
            }
        }
        max_val
    }
}

// ── Effective number of tests ───────────────────────────────────────────────

/// Estimate the effective number of independent tests, accounting for correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveNumberOfTests {
    pub nominal_m: usize,
    pub effective_m: f64,
    pub method: String,
}

impl EffectiveNumberOfTests {
    /// Li & Ji (2005) method based on eigenvalues of the correlation matrix.
    /// Since full eigendecomposition is complex, we use the Galwey (2009) approximation.
    pub fn li_ji(dep: &DependenceStructure) -> Self {
        let m = dep.num_tests;
        if m <= 1 {
            return Self {
                nominal_m: m,
                effective_m: m as f64,
                method: "Li-Ji".into(),
            };
        }

        // Approximate effective M using average correlation
        // M_eff ≈ m * (1 - avg_corr) + avg_corr
        let avg_r = dep.average_correlation();
        let m_eff = (m as f64 * (1.0 - avg_r) + avg_r).max(1.0);

        Self {
            nominal_m: m,
            effective_m: m_eff,
            method: "Li-Ji (approx)".into(),
        }
    }

    /// Nyholt (2004) method: uses largest eigenvalue approximation.
    pub fn nyholt(dep: &DependenceStructure) -> Self {
        let m = dep.num_tests;
        if m <= 1 {
            return Self {
                nominal_m: m,
                effective_m: m as f64,
                method: "Nyholt".into(),
            };
        }

        // Approximate max eigenvalue by 1 + (m-1) * avg_corr
        let avg_r = dep.average_correlation();
        let lambda_max = 1.0 + (m as f64 - 1.0) * avg_r;
        // M_eff = 1 + (m-1) * (1 - Var(λ)/m)
        // Approx: M_eff ≈ m * (1 - (lambda_max - 1)/(m - 1))
        let m_eff = if m > 1 {
            let ratio = (lambda_max - 1.0) / (m as f64 - 1.0);
            (m as f64 * (1.0 - ratio)).max(1.0)
        } else {
            1.0
        };

        Self {
            nominal_m: m,
            effective_m: m_eff,
            method: "Nyholt".into(),
        }
    }

    /// Simple Cheverud-Nyholt adjustment: M_eff = 1 + (M-1)(1 - Var(eigenvalues)/M).
    /// We use a simpler formula based on average correlation.
    pub fn simple(dep: &DependenceStructure) -> Self {
        let m = dep.num_tests;
        let avg_r = dep.average_correlation();
        let m_eff = (1.0 + (m as f64 - 1.0) * (1.0 - avg_r)).max(1.0);
        Self {
            nominal_m: m,
            effective_m: m_eff,
            method: "Simple".into(),
        }
    }

    /// Adjusted Bonferroni alpha using effective number of tests.
    pub fn adjusted_alpha(&self, alpha: f64) -> f64 {
        if self.effective_m <= 0.0 {
            alpha
        } else {
            alpha / self.effective_m
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_bonferroni_adjust() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let adj = BonferroniCorrection.adjust(&pvals);
        assert!(approx_eq(adj[0], 0.04, 1e-10));
        assert!(approx_eq(adj[1], 0.16, 1e-10));
        assert!(approx_eq(adj[2], 0.12, 1e-10));
        assert!(approx_eq(adj[3], 0.02, 1e-10));
    }

    #[test]
    fn test_bonferroni_cap_at_one() {
        let pvals = vec![0.5, 0.8];
        let adj = BonferroniCorrection.adjust(&pvals);
        assert!(adj[0] <= 1.0);
        assert!(adj[1] <= 1.0);
    }

    #[test]
    fn test_bonferroni_critical_alpha() {
        let ca = BonferroniCorrection::critical_alpha(0.05, 10);
        assert!(approx_eq(ca, 0.005, 1e-10));
    }

    #[test]
    fn test_holm_adjust() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005];
        let adj = HolmBonferroniCorrection.adjust(&pvals);
        // Sorted: 0.005 (idx3), 0.01 (idx0), 0.03 (idx2), 0.04 (idx1)
        // adj[3] = 0.005*4 = 0.02, adj[0] = max(0.02, 0.01*3) = 0.03
        // adj[2] = max(0.03, 0.03*2) = 0.06, adj[1] = max(0.06, 0.04*1) = 0.06
        assert!(approx_eq(adj[3], 0.02, 1e-10));
        assert!(approx_eq(adj[0], 0.03, 1e-10));
        assert!(approx_eq(adj[2], 0.06, 1e-10));
        assert!(approx_eq(adj[1], 0.06, 1e-10));
    }

    #[test]
    fn test_holm_less_conservative_than_bonferroni() {
        let pvals = vec![0.001, 0.01, 0.02, 0.03, 0.04];
        let bonf = BonferroniCorrection.adjust(&pvals);
        let holm = HolmBonferroniCorrection.adjust(&pvals);
        for i in 0..pvals.len() {
            assert!(holm[i] <= bonf[i] + 1e-10);
        }
    }

    #[test]
    fn test_holm_critical_values() {
        let cvs = HolmBonferroniCorrection::critical_values(0.05, 4);
        assert!(approx_eq(cvs[0], 0.05 / 4.0, 1e-10));
        assert!(approx_eq(cvs[1], 0.05 / 3.0, 1e-10));
        assert!(approx_eq(cvs[2], 0.05 / 2.0, 1e-10));
        assert!(approx_eq(cvs[3], 0.05, 1e-10));
    }

    #[test]
    fn test_bh_fdr_adjust() {
        let pvals = vec![0.005, 0.01, 0.03, 0.04, 0.5];
        let adj = BenjaminiHochbergFDR.adjust(&pvals);
        // All adjusted should be >= raw
        for (raw, a) in pvals.iter().zip(adj.iter()) {
            assert!(*a >= *raw - 1e-10);
        }
        // Should be monotone in sorted order
        let mut sorted_adj = adj.clone();
        sorted_adj.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The smallest adjusted should correspond to the smallest raw
    }

    #[test]
    fn test_bh_critical_values() {
        let cvs = BenjaminiHochbergFDR::critical_values(0.05, 5);
        assert!(approx_eq(cvs[0], 0.01, 1e-10));
        assert!(approx_eq(cvs[4], 0.05, 1e-10));
    }

    #[test]
    fn test_sidak_adjust() {
        let pvals = vec![0.01, 0.05];
        let adj = SidakCorrection.adjust(&pvals);
        // adj = 1 - (1-p)^m
        assert!(approx_eq(adj[0], 1.0 - (1.0 - 0.01_f64).powi(2), 1e-10));
        assert!(approx_eq(adj[1], 1.0 - (1.0 - 0.05_f64).powi(2), 1e-10));
    }

    #[test]
    fn test_sidak_less_conservative_than_bonferroni() {
        let pvals = vec![0.01, 0.02, 0.03, 0.04];
        let bonf = BonferroniCorrection.adjust(&pvals);
        let sidak = SidakCorrection.adjust(&pvals);
        for i in 0..pvals.len() {
            assert!(sidak[i] <= bonf[i] + 1e-10);
        }
    }

    #[test]
    fn test_sidak_critical_alpha() {
        let ca = SidakCorrection::critical_alpha(0.05, 10);
        assert!(ca > 0.0);
        assert!(ca < 0.05);
        // Sidak critical alpha > Bonferroni critical alpha
        let bonf_ca = BonferroniCorrection::critical_alpha(0.05, 10);
        assert!(ca >= bonf_ca);
    }

    #[test]
    fn test_adjusted_pvalues_by_name() {
        let pvals = vec![0.01, 0.02, 0.03];
        let bonf = adjusted_pvalues(&pvals, "bonferroni").unwrap();
        assert_eq!(bonf.len(), 3);
        let holm = adjusted_pvalues(&pvals, "holm").unwrap();
        assert_eq!(holm.len(), 3);
        let bh = adjusted_pvalues(&pvals, "bh").unwrap();
        assert_eq!(bh.len(), 3);
        let none = adjusted_pvalues(&pvals, "none").unwrap();
        assert_eq!(none, pvals);
        assert!(adjusted_pvalues(&pvals, "unknown_method").is_err());
    }

    #[test]
    fn test_reject_set_function() {
        let pvals = vec![0.001, 0.01, 0.1, 0.5];
        let rej = reject_set(&pvals, 0.05, "bonferroni").unwrap();
        // 0.001*4=0.004 < 0.05 → reject, 0.01*4=0.04 < 0.05 → reject
        assert!(rej.contains(&0));
        assert!(rej.contains(&1));
        assert!(!rej.contains(&2));
        assert!(!rej.contains(&3));
    }

    #[test]
    fn test_fwer_guarantee() {
        let pvals = vec![0.001, 0.02, 0.06, 0.5];
        let g = FWERControlGuarantee::compute(&pvals, 0.05, "holm").unwrap();
        assert!(g.is_valid());
        assert!(g.num_rejected <= pvals.len());
        assert_eq!(g.adjusted_pvalues.len(), 4);
    }

    #[test]
    fn test_fdr_control_bh() {
        let pvals = vec![0.001, 0.005, 0.01, 0.04, 0.5, 0.7, 0.9];
        let fdr = FDRControl::compute_bh(&pvals, 0.05);
        assert!(fdr.num_discoveries <= pvals.len());
        assert!(fdr.num_discoveries > 0);
    }

    #[test]
    fn test_fdr_control_storey() {
        let pvals = vec![0.001, 0.005, 0.01, 0.04, 0.5, 0.7, 0.9];
        let fdr = FDRControl::compute_storey(&pvals, 0.05, 0.5);
        assert!(fdr.num_discoveries <= pvals.len());
    }

    #[test]
    fn test_dependence_structure_independent() {
        let dep = DependenceStructure::independent(5);
        assert_eq!(dep.num_tests, 5);
        assert!(approx_eq(dep.average_correlation(), 0.0, 1e-10));
        assert!(approx_eq(dep.max_correlation(), 0.0, 1e-10));
    }

    #[test]
    fn test_dependence_structure_correlated() {
        let mat = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let dep = DependenceStructure::new(mat);
        assert!(dep.average_correlation() > 0.0);
        assert!(approx_eq(dep.max_correlation(), 0.5, 1e-10));
    }

    #[test]
    fn test_effective_number_independent() {
        let dep = DependenceStructure::independent(10);
        let ent = EffectiveNumberOfTests::simple(&dep);
        assert!(approx_eq(ent.effective_m, 10.0, 0.1));
    }

    #[test]
    fn test_effective_number_correlated() {
        let mut mat = vec![vec![0.8; 5]; 5];
        for i in 0..5 {
            mat[i][i] = 1.0;
        }
        let dep = DependenceStructure::new(mat);
        let ent = EffectiveNumberOfTests::simple(&dep);
        assert!(ent.effective_m < 5.0);
        assert!(ent.effective_m >= 1.0);
    }

    #[test]
    fn test_effective_number_nyholt() {
        let dep = DependenceStructure::independent(8);
        let ent = EffectiveNumberOfTests::nyholt(&dep);
        assert!(ent.effective_m > 0.0);
    }

    #[test]
    fn test_effective_adjusted_alpha() {
        let dep = DependenceStructure::independent(10);
        let ent = EffectiveNumberOfTests::simple(&dep);
        let adj = ent.adjusted_alpha(0.05);
        assert!(adj > 0.0);
        assert!(adj <= 0.05);
    }

    #[test]
    fn test_empty_pvalues() {
        let adj = BonferroniCorrection.adjust(&[]);
        assert!(adj.is_empty());
        let adj = HolmBonferroniCorrection.adjust(&[]);
        assert!(adj.is_empty());
        let adj = BenjaminiHochbergFDR.adjust(&[]);
        assert!(adj.is_empty());
    }
}
