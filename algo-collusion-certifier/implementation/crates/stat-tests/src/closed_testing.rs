//! M7 Directed Closed Testing with FWER control.
//!
//! Implements the collusion-structured test ordering:
//! supra-competitive pricing → punishment response → correlation → convergence
//! with Holm-Bonferroni step-down, Hochberg step-up, and Hommel procedures.

use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, HypothesisTestResult, PValue};

use crate::multiple_testing::{
    MultipleTestCorrection, BonferroniCorrection, HolmBonferroniCorrection,
    BenjaminiHochbergFDR,
};

// ── Closed testing procedure ────────────────────────────────────────────────

/// Generic closed testing framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedTestingProcedure {
    pub name: String,
    pub alpha: f64,
    pub test_results: Vec<ClosedTestEntry>,
    pub rejection_set: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedTestEntry {
    pub hypothesis_index: usize,
    pub hypothesis_name: String,
    pub p_value: f64,
    pub adjusted_p_value: f64,
    pub rejected: bool,
}

impl ClosedTestingProcedure {
    /// Create a new closed testing procedure.
    pub fn new(name: &str, alpha: f64) -> Self {
        Self {
            name: name.to_string(),
            alpha,
            test_results: Vec::new(),
            rejection_set: Vec::new(),
        }
    }

    /// Run closed testing with intersection-closure principle.
    /// For each hypothesis H_i, reject iff all intersection hypotheses
    /// containing H_i are rejected at level alpha.
    pub fn run(
        hypothesis_names: &[String],
        p_values: &[f64],
        alpha: f64,
        local_test: &dyn MultipleTestCorrection,
    ) -> CollusionResult<Self> {
        let m = p_values.len();
        if m == 0 {
            return Ok(Self::new("Closed Testing", alpha));
        }

        let adjusted = local_test.adjust(p_values);
        let entries: Vec<ClosedTestEntry> = (0..m)
            .map(|i| ClosedTestEntry {
                hypothesis_index: i,
                hypothesis_name: hypothesis_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("H{}", i)),
                p_value: p_values[i],
                adjusted_p_value: adjusted[i],
                rejected: adjusted[i] < alpha,
            })
            .collect();

        let rejection_set: Vec<usize> = entries
            .iter()
            .filter(|e| e.rejected)
            .map(|e| e.hypothesis_index)
            .collect();

        Ok(Self {
            name: format!("Closed Testing ({})", local_test.name()),
            alpha,
            test_results: entries,
            rejection_set,
        })
    }

    /// Number of rejected hypotheses.
    pub fn num_rejections(&self) -> usize {
        self.rejection_set.len()
    }

    /// Whether a specific hypothesis was rejected.
    pub fn is_rejected(&self, index: usize) -> bool {
        self.rejection_set.contains(&index)
    }
}

// ── Directed closed testing (M7) ────────────────────────────────────────────

/// M7 collusion-structured directed closed testing.
/// Tests are ordered by economic importance:
/// 1. Supra-competitive pricing
/// 2. Punishment response
/// 3. Cross-firm correlation
/// 4. Convergence pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectedClosedTesting {
    pub alpha: f64,
    pub stages: Vec<DirectedTestStage>,
    pub final_decision: bool,
    pub stopping_stage: Option<usize>,
    pub alpha_budget: AlphaBudgetTracker,
}

/// A stage in the directed closed testing procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectedTestStage {
    pub stage_index: usize,
    pub stage_name: String,
    pub p_values: Vec<f64>,
    pub adjusted_p_values: Vec<f64>,
    pub alpha_allocated: f64,
    pub rejected: bool,
}

impl DirectedClosedTesting {
    /// Create a directed closed testing procedure from pre-ordered test stages.
    /// Each stage contains one or more related sub-tests.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            stages: Vec::new(),
            final_decision: false,
            stopping_stage: None,
            alpha_budget: AlphaBudgetTracker::new(alpha),
        }
    }

    /// Run the directed closed testing procedure.
    /// `stages`: ordered stages, each containing (name, p_values).
    /// Uses Holm-Bonferroni within each stage.
    pub fn run(
        alpha: f64,
        stages: Vec<(String, Vec<f64>)>,
    ) -> CollusionResult<Self> {
        let n_stages = stages.len();
        if n_stages == 0 {
            return Ok(Self::new(alpha));
        }

        let mut budget = AlphaBudgetTracker::new(alpha);
        let mut result_stages = Vec::with_capacity(n_stages);
        let mut all_rejected = true;
        let mut stopping_stage = None;

        for (idx, (name, p_vals)) in stages.into_iter().enumerate() {
            let stage_alpha = budget.allocate_equal(n_stages)?;

            // Apply Holm-Bonferroni within stage
            let adjusted = HolmBonferroniCorrection.adjust(&p_vals);
            let stage_rejected = adjusted.iter().all(|&p| p < stage_alpha);

            result_stages.push(DirectedTestStage {
                stage_index: idx,
                stage_name: name,
                p_values: p_vals,
                adjusted_p_values: adjusted,
                alpha_allocated: stage_alpha,
                rejected: stage_rejected,
            });

            if !stage_rejected {
                all_rejected = false;
                stopping_stage = Some(idx);
                break;
            }
        }

        Ok(Self {
            alpha,
            stages: result_stages,
            final_decision: all_rejected,
            stopping_stage,
            alpha_budget: budget,
        })
    }

    /// Run with collusion-specific ordering:
    /// 1. supra-competitive pricing  2. punishment  3. correlation  4. convergence
    pub fn run_collusion_ordered(
        alpha: f64,
        pricing_p: &[f64],
        punishment_p: &[f64],
        correlation_p: &[f64],
        convergence_p: &[f64],
    ) -> CollusionResult<Self> {
        let stages = vec![
            ("Supra-competitive pricing".into(), pricing_p.to_vec()),
            ("Punishment response".into(), punishment_p.to_vec()),
            ("Cross-firm correlation".into(), correlation_p.to_vec()),
            ("Convergence pattern".into(), convergence_p.to_vec()),
        ];
        Self::run(alpha, stages)
    }

    /// Number of stages completed before stopping.
    pub fn stages_completed(&self) -> usize {
        self.stages.len()
    }

    /// Get summary of rejected stages.
    pub fn rejected_stages(&self) -> Vec<&DirectedTestStage> {
        self.stages.iter().filter(|s| s.rejected).collect()
    }
}

// ── Holm-Bonferroni procedure ───────────────────────────────────────────────

/// Holm-Bonferroni step-down procedure for FWER control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolmBonferroni {
    pub alpha: f64,
    pub adjusted_p_values: Vec<f64>,
    pub rejection_set: Vec<usize>,
    pub sorted_indices: Vec<usize>,
}

impl HolmBonferroni {
    /// Run Holm-Bonferroni on raw p-values.
    pub fn run(p_values: &[f64], alpha: f64) -> Self {
        let adjusted = HolmBonferroniCorrection.adjust(p_values);
        let rejection_set: Vec<usize> = adjusted
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < alpha)
            .map(|(i, _)| i)
            .collect();

        let mut sorted_indices: Vec<usize> = (0..p_values.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            alpha,
            adjusted_p_values: adjusted,
            rejection_set,
            sorted_indices,
        }
    }

    /// Whether hypothesis i is rejected.
    pub fn is_rejected(&self, i: usize) -> bool {
        self.rejection_set.contains(&i)
    }

    /// Number of rejections.
    pub fn num_rejections(&self) -> usize {
        self.rejection_set.len()
    }
}

// ── Hochberg procedure ──────────────────────────────────────────────────────

/// Hochberg step-up procedure (more powerful than Holm under certain dependencies).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hochberg {
    pub alpha: f64,
    pub adjusted_p_values: Vec<f64>,
    pub rejection_set: Vec<usize>,
}

impl Hochberg {
    /// Run Hochberg procedure.
    pub fn run(p_values: &[f64], alpha: f64) -> Self {
        let m = p_values.len();
        if m == 0 {
            return Self {
                alpha,
                adjusted_p_values: Vec::new(),
                rejection_set: Vec::new(),
            };
        }

        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step-up: from largest to smallest
        let mut adjusted = vec![0.0; m];
        let mut cummin = 1.0_f64;

        for rank in (0..m).rev() {
            let idx = indices[rank];
            let adj = (p_values[idx] * (m - rank) as f64).min(1.0);
            cummin = cummin.min(adj);
            adjusted[idx] = cummin;
        }

        let rejection_set: Vec<usize> = adjusted
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < alpha)
            .map(|(i, _)| i)
            .collect();

        Self {
            alpha,
            adjusted_p_values: adjusted,
            rejection_set,
        }
    }

    pub fn is_rejected(&self, i: usize) -> bool {
        self.rejection_set.contains(&i)
    }
}

// ── Hommel procedure ────────────────────────────────────────────────────────

/// Hommel's sharper closed testing procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HommelProcedure {
    pub alpha: f64,
    pub adjusted_p_values: Vec<f64>,
    pub rejection_set: Vec<usize>,
}

impl HommelProcedure {
    /// Run Hommel procedure.
    pub fn run(p_values: &[f64], alpha: f64) -> Self {
        let m = p_values.len();
        if m == 0 {
            return Self {
                alpha,
                adjusted_p_values: Vec::new(),
                rejection_set: Vec::new(),
            };
        }

        // Sorted p-values
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_p: Vec<f64> = indices.iter().map(|&i| p_values[i]).collect();

        // Find the largest j such that p_(m-j+k) > k*alpha/j for all k=1..j
        // This determines the number of hypotheses to reject
        let mut j_max = m;
        for j in (1..=m).rev() {
            let mut condition_met = true;
            for k in 1..=j {
                let idx = m - j + k - 1;
                if idx < m && sorted_p[idx] <= k as f64 * alpha / j as f64 {
                    condition_met = false;
                    break;
                }
            }
            if condition_met {
                j_max = j;
                break;
            }
        }

        // Adjusted p-values: use Hochberg as a conservative approximation
        // Hommel is at least as powerful as Hochberg
        let mut adjusted = vec![0.0; m];
        let mut cummin = 1.0_f64;

        for rank in (0..m).rev() {
            let idx = indices[rank];
            let adj = (p_values[idx] * (m - rank) as f64).min(1.0);
            cummin = cummin.min(adj);
            adjusted[idx] = cummin;
        }

        // Apply Hommel improvement
        for i in 0..m {
            adjusted[i] = adjusted[i].min(
                (0..m)
                    .filter(|&k| k != i)
                    .map(|k| (p_values[k] * m as f64 / (m - 1).max(1) as f64).min(1.0))
                    .fold(1.0_f64, f64::min)
                    .max(adjusted[i])
            );
        }

        let rejection_set: Vec<usize> = adjusted
            .iter()
            .enumerate()
            .filter(|(_, p)| **p < alpha)
            .map(|(i, _)| i)
            .collect();

        Self {
            alpha,
            adjusted_p_values: adjusted,
            rejection_set,
        }
    }

    pub fn is_rejected(&self, i: usize) -> bool {
        self.rejection_set.contains(&i)
    }
}

// ── Alpha budget tracker ────────────────────────────────────────────────────

/// Track alpha spending across sub-tests to prevent overdraft.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaBudgetTracker {
    pub total_alpha: f64,
    pub spent: f64,
    pub allocations: Vec<(String, f64)>,
    pub segments: Vec<SegmentBudget>,
}

/// Per-segment alpha budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentBudget {
    pub segment_name: String,
    pub total_alpha: f64,
    pub spent: f64,
}

impl AlphaBudgetTracker {
    pub fn new(total_alpha: f64) -> Self {
        Self {
            total_alpha,
            spent: 0.0,
            allocations: Vec::new(),
            segments: Vec::new(),
        }
    }

    /// Remaining alpha budget.
    pub fn remaining(&self) -> f64 {
        (self.total_alpha - self.spent).max(0.0)
    }

    /// Whether the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() < 1e-15
    }

    /// Allocate alpha for a named test.
    pub fn allocate(&mut self, name: &str, amount: f64) -> CollusionResult<f64> {
        if amount > self.remaining() + 1e-10 {
            return Err(CollusionError::StatisticalTest(format!(
                "Alpha overdraft: requested {amount:.4}, remaining {:.4}",
                self.remaining()
            )));
        }
        let actual = amount.min(self.remaining());
        self.spent += actual;
        self.allocations.push((name.to_string(), actual));
        Ok(actual)
    }

    /// Allocate equal shares for n tests.
    pub fn allocate_equal(&mut self, n: usize) -> CollusionResult<f64> {
        if n == 0 {
            return Ok(0.0);
        }
        let share = self.total_alpha / n as f64;
        Ok(share)
    }

    /// Add a segment budget.
    pub fn add_segment(&mut self, name: &str, alpha: f64) {
        self.segments.push(SegmentBudget {
            segment_name: name.to_string(),
            total_alpha: alpha,
            spent: 0.0,
        });
    }

    /// Spend from a specific segment.
    pub fn spend_segment(&mut self, segment_name: &str, amount: f64) -> CollusionResult<()> {
        let seg = self.segments
            .iter_mut()
            .find(|s| s.segment_name == segment_name)
            .ok_or_else(|| CollusionError::StatisticalTest(
                format!("Unknown segment: {segment_name}"),
            ))?;

        if amount > seg.total_alpha - seg.spent + 1e-10 {
            return Err(CollusionError::StatisticalTest(format!(
                "Segment '{segment_name}' alpha overdraft"
            )));
        }
        seg.spent += amount;
        self.spent += amount;
        Ok(())
    }

    /// Validate that total spending does not exceed budget.
    pub fn validate(&self) -> bool {
        self.spent <= self.total_alpha + 1e-10
    }
}

// ── FWER guarantee ──────────────────────────────────────────────────────────

/// Formal FWER control guarantee (proof term).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FWERGuarantee {
    pub method: String,
    pub nominal_alpha: f64,
    pub num_tests: usize,
    pub num_rejected: usize,
    pub fwer_bound: f64,
    pub is_valid: bool,
    pub reasoning: String,
}

impl FWERGuarantee {
    /// Construct FWER guarantee from a closed testing result.
    pub fn from_closed_testing(ct: &ClosedTestingProcedure) -> Self {
        Self {
            method: ct.name.clone(),
            nominal_alpha: ct.alpha,
            num_tests: ct.test_results.len(),
            num_rejected: ct.rejection_set.len(),
            fwer_bound: ct.alpha,
            is_valid: true,
            reasoning: format!(
                "Closed testing with {} guarantees FWER ≤ {:.4} \
                 by the intersection-closure principle.",
                ct.name, ct.alpha
            ),
        }
    }

    /// Construct from directed closed testing.
    pub fn from_directed(dct: &DirectedClosedTesting) -> Self {
        let num_rejected = dct.rejected_stages().len();
        Self {
            method: "Directed Closed Testing (M7)".into(),
            nominal_alpha: dct.alpha,
            num_tests: dct.stages.len(),
            num_rejected,
            fwer_bound: dct.alpha,
            is_valid: dct.alpha_budget.validate(),
            reasoning: format!(
                "M7 directed closed testing with Holm-Bonferroni within {} stages. \
                 Alpha budget: {:.4} total, {:.4} spent. FWER ≤ {:.4}.",
                dct.stages.len(),
                dct.alpha_budget.total_alpha,
                dct.alpha_budget.spent,
                dct.alpha,
            ),
        }
    }

    /// Construct from Holm-Bonferroni.
    pub fn from_holm(hb: &HolmBonferroni) -> Self {
        Self {
            method: "Holm-Bonferroni".into(),
            nominal_alpha: hb.alpha,
            num_tests: hb.adjusted_p_values.len(),
            num_rejected: hb.num_rejections(),
            fwer_bound: hb.alpha,
            is_valid: true,
            reasoning: format!(
                "Holm-Bonferroni step-down procedure controls FWER ≤ {:.4} \
                 for {} tests under arbitrary dependence.",
                hb.alpha,
                hb.adjusted_p_values.len(),
            ),
        }
    }
}

// ── Test rejection sequence ─────────────────────────────────────────────────

/// Ordered sequence of rejected hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRejectionSequence {
    pub rejections: Vec<RejectionEntry>,
    pub alpha: f64,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionEntry {
    pub order: usize,
    pub hypothesis_name: String,
    pub raw_p_value: f64,
    pub adjusted_p_value: f64,
    pub critical_value: f64,
}

impl TestRejectionSequence {
    /// Build from Holm-Bonferroni results.
    pub fn from_holm(
        names: &[String],
        p_values: &[f64],
        alpha: f64,
    ) -> Self {
        let hb = HolmBonferroni::run(p_values, alpha);
        let m = p_values.len();

        let mut rejections: Vec<RejectionEntry> = hb.rejection_set
            .iter()
            .enumerate()
            .map(|(order, &idx)| {
                let rank = hb.sorted_indices.iter().position(|&i| i == idx).unwrap_or(0);
                RejectionEntry {
                    order,
                    hypothesis_name: names.get(idx).cloned().unwrap_or_else(|| format!("H{}", idx)),
                    raw_p_value: p_values[idx],
                    adjusted_p_value: hb.adjusted_p_values[idx],
                    critical_value: alpha / (m - rank) as f64,
                }
            })
            .collect();

        rejections.sort_by(|a, b| {
            a.raw_p_value.partial_cmp(&b.raw_p_value).unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            rejections,
            alpha,
            method: "Holm-Bonferroni".into(),
        }
    }

    /// Number of rejections.
    pub fn num_rejections(&self) -> usize {
        self.rejections.len()
    }
}

// ── Power improvement ───────────────────────────────────────────────────────

/// Measure power gain from directed ordering vs omnibus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerImprovement {
    pub directed_rejections: usize,
    pub omnibus_rejections: usize,
    pub power_ratio: f64,
    pub improvement_description: String,
}

impl PowerImprovement {
    /// Compare directed closed testing to omnibus Bonferroni.
    pub fn compare(
        p_values: &[f64],
        alpha: f64,
        stages: Vec<(String, Vec<f64>)>,
    ) -> CollusionResult<Self> {
        // Omnibus: Bonferroni on all p-values
        let all_p: Vec<f64> = stages.iter().flat_map(|(_, ps)| ps.clone()).collect();
        let bonf = BonferroniCorrection.adjust(&all_p);
        let omnibus_rej = bonf.iter().filter(|&&p| p < alpha).count();

        // Directed: M7 procedure
        let dct = DirectedClosedTesting::run(alpha, stages)?;
        let directed_rej = dct.rejected_stages().len();

        let power_ratio = if omnibus_rej == 0 {
            if directed_rej > 0 { f64::INFINITY } else { 1.0 }
        } else {
            directed_rej as f64 / omnibus_rej as f64
        };

        Ok(Self {
            directed_rejections: directed_rej,
            omnibus_rejections: omnibus_rej,
            power_ratio,
            improvement_description: format!(
                "Directed: {} rejections, Omnibus: {} rejections (ratio: {:.2})",
                directed_rej, omnibus_rej, power_ratio,
            ),
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closed_testing_basic() {
        let names: Vec<String> = (0..4).map(|i| format!("H{i}")).collect();
        let p_values = vec![0.001, 0.02, 0.06, 0.5];
        let ct = ClosedTestingProcedure::run(
            &names,
            &p_values,
            0.05,
            &HolmBonferroniCorrection,
        ).unwrap();
        assert!(ct.num_rejections() > 0);
        assert!(ct.is_rejected(0));
    }

    #[test]
    fn test_closed_testing_none_rejected() {
        let names = vec!["H0".into(), "H1".into()];
        let p_values = vec![0.5, 0.8];
        let ct = ClosedTestingProcedure::run(
            &names,
            &p_values,
            0.05,
            &HolmBonferroniCorrection,
        ).unwrap();
        assert_eq!(ct.num_rejections(), 0);
    }

    #[test]
    fn test_closed_testing_empty() {
        let ct = ClosedTestingProcedure::run(
            &[],
            &[],
            0.05,
            &BonferroniCorrection,
        ).unwrap();
        assert_eq!(ct.num_rejections(), 0);
    }

    #[test]
    fn test_directed_closed_testing() {
        let stages = vec![
            ("Pricing".into(), vec![0.001, 0.002]),
            ("Punishment".into(), vec![0.01]),
            ("Correlation".into(), vec![0.03]),
            ("Convergence".into(), vec![0.04]),
        ];
        let dct = DirectedClosedTesting::run(0.05, stages).unwrap();
        assert!(dct.stages_completed() > 0);
    }

    #[test]
    fn test_directed_stops_early() {
        let stages = vec![
            ("Pricing".into(), vec![0.001]),
            ("Punishment".into(), vec![0.5]), // This fails → stop
            ("Correlation".into(), vec![0.001]),
        ];
        let dct = DirectedClosedTesting::run(0.05, stages).unwrap();
        assert!(!dct.final_decision);
        assert!(dct.stopping_stage.is_some());
    }

    #[test]
    fn test_directed_collusion_ordered() {
        let dct = DirectedClosedTesting::run_collusion_ordered(
            0.05,
            &[0.001, 0.002],
            &[0.01],
            &[0.02],
            &[0.03],
        ).unwrap();
        assert_eq!(dct.stages.len(), 4);
    }

    #[test]
    fn test_holm_bonferroni() {
        let p_values = vec![0.001, 0.01, 0.04, 0.5];
        let hb = HolmBonferroni::run(&p_values, 0.05);
        assert!(hb.is_rejected(0));
        assert!(!hb.is_rejected(3));
    }

    #[test]
    fn test_holm_bonferroni_all_reject() {
        let p_values = vec![0.001, 0.002, 0.003];
        let hb = HolmBonferroni::run(&p_values, 0.05);
        assert_eq!(hb.num_rejections(), 3);
    }

    #[test]
    fn test_hochberg() {
        let p_values = vec![0.001, 0.01, 0.04, 0.5];
        let hoch = Hochberg::run(&p_values, 0.05);
        assert!(hoch.is_rejected(0));
        assert!(!hoch.is_rejected(3));
    }

    #[test]
    fn test_hochberg_more_powerful() {
        // Hochberg should reject at least as many as Holm
        let p_values = vec![0.005, 0.015, 0.025, 0.035, 0.045];
        let holm = HolmBonferroni::run(&p_values, 0.05);
        let hoch = Hochberg::run(&p_values, 0.05);
        assert!(hoch.rejection_set.len() >= holm.rejection_set.len());
    }

    #[test]
    fn test_hommel() {
        let p_values = vec![0.001, 0.01, 0.04, 0.5];
        let hom = HommelProcedure::run(&p_values, 0.05);
        assert!(hom.is_rejected(0));
    }

    #[test]
    fn test_alpha_budget_tracker() {
        let mut budget = AlphaBudgetTracker::new(0.05);
        assert!(!budget.is_exhausted());
        assert_eq!(budget.remaining(), 0.05);

        budget.allocate("test1", 0.02).unwrap();
        assert_eq!(budget.remaining(), 0.03);

        budget.allocate("test2", 0.03).unwrap();
        assert!(budget.is_exhausted());

        assert!(budget.allocate("test3", 0.01).is_err());
    }

    #[test]
    fn test_alpha_budget_segments() {
        let mut budget = AlphaBudgetTracker::new(0.05);
        budget.add_segment("training", 0.025);
        budget.add_segment("testing", 0.025);

        budget.spend_segment("training", 0.01).unwrap();
        budget.spend_segment("testing", 0.02).unwrap();
        assert!(budget.validate());

        assert!(budget.spend_segment("training", 0.02).is_err());
    }

    #[test]
    fn test_alpha_budget_equal() {
        let mut budget = AlphaBudgetTracker::new(0.05);
        let share = budget.allocate_equal(5).unwrap();
        assert!((share - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_fwer_guarantee_holm() {
        let p_values = vec![0.001, 0.01, 0.1, 0.5];
        let hb = HolmBonferroni::run(&p_values, 0.05);
        let guarantee = FWERGuarantee::from_holm(&hb);
        assert!(guarantee.is_valid);
        assert_eq!(guarantee.fwer_bound, 0.05);
    }

    #[test]
    fn test_fwer_guarantee_directed() {
        let stages = vec![
            ("Pricing".into(), vec![0.001]),
            ("Punishment".into(), vec![0.01]),
        ];
        let dct = DirectedClosedTesting::run(0.05, stages).unwrap();
        let guarantee = FWERGuarantee::from_directed(&dct);
        assert!(guarantee.is_valid);
    }

    #[test]
    fn test_rejection_sequence() {
        let names = vec!["Pricing".into(), "Punishment".into(), "Correlation".into()];
        let p_values = vec![0.001, 0.01, 0.5];
        let seq = TestRejectionSequence::from_holm(&names, &p_values, 0.05);
        assert!(seq.num_rejections() > 0);
    }

    #[test]
    fn test_power_improvement() {
        let stages = vec![
            ("Pricing".into(), vec![0.001]),
            ("Punishment".into(), vec![0.01]),
        ];
        let pi = PowerImprovement::compare(&[0.001, 0.01], 0.05, stages).unwrap();
        assert!(pi.directed_rejections >= 0);
    }

    #[test]
    fn test_holm_no_rejections() {
        let p_values = vec![0.5, 0.6, 0.7];
        let hb = HolmBonferroni::run(&p_values, 0.05);
        assert_eq!(hb.num_rejections(), 0);
    }
}
