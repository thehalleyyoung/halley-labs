//! Barinel (Bayesian) fault localization adapted for continuous differentials.
//!
//! Barinel computes posterior probabilities of fault candidates (single or
//! multi-stage) using a noisy-OR model and an EM algorithm for estimating
//! per-stage fault probabilities.

use crate::dstar::{rank_by_score, separation_ratio};
use crate::{DifferentialMatrix, ViolationVector};
use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};

// ── Public types ────────────────────────────────────────────────────────────

/// Configuration for the Barinel metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarinelMetric {
    pub prior_fault_probability: f64,
    pub max_candidates: usize,
    pub em_max_iterations: usize,
    pub em_epsilon: f64,
}

impl Default for BarinelMetric {
    fn default() -> Self {
        Self {
            prior_fault_probability: 0.1,
            max_candidates: 50,
            em_max_iterations: 200,
            em_epsilon: 1e-8,
        }
    }
}

/// A fault candidate: one or more stages suspected of being faulty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultCandidate {
    pub stage_subset: Vec<usize>,
    pub posterior_probability: f64,
    pub goodness_of_fit: f64,
}

/// Full Barinel result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarinelResult {
    pub candidate_rankings: Vec<FaultCandidate>,
    pub most_probable_fault: Vec<usize>,
    pub bayes_factor: f64,
    pub stage_fault_probs: Vec<f64>,
    pub em_iterations: usize,
    pub converged: bool,
}

/// Internal per-stage fault probability estimate used in the EM loop.
#[derive(Debug, Clone)]
struct StageFaultProb {
    prob: f64,
}

// ── Implementation ──────────────────────────────────────────────────────────

impl BarinelMetric {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_prior(mut self, prior: f64) -> Self {
        self.prior_fault_probability = prior;
        self
    }

    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max;
        self
    }

    /// Run Barinel fault localization.
    pub fn compute_posteriors(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<BarinelResult> {
        self.validate(matrix, violations)?;

        // Phase 1: EM to estimate per-stage fault probabilities
        let (stage_probs, iters, converged) = self.run_em(matrix, violations);

        // Phase 2: Generate single-fault candidates and rank by posterior
        let candidates = self.generate_candidates(matrix, violations, &stage_probs);

        let most_probable = if candidates.is_empty() {
            vec![]
        } else {
            candidates[0].stage_subset.clone()
        };

        let bf = self.compute_bayes_factor(&candidates);

        Ok(BarinelResult {
            candidate_rankings: candidates,
            most_probable_fault: most_probable,
            bayes_factor: bf,
            stage_fault_probs: stage_probs,
            em_iterations: iters,
            converged,
        })
    }

    /// Human-readable explanation of the localization result.
    pub fn explain_result(&self, result: &BarinelResult, stage_names: &[String]) -> String {
        if result.most_probable_fault.is_empty() {
            return "No fault candidates identified.".to_string();
        }

        let top_names: Vec<&str> = result
            .most_probable_fault
            .iter()
            .filter_map(|&i| stage_names.get(i).map(|s| s.as_str()))
            .collect();

        let top_prob = result
            .candidate_rankings
            .first()
            .map(|c| c.posterior_probability)
            .unwrap_or(0.0);

        let mut explanation = format!(
            "Most probable fault location: {} (posterior = {:.4})\n",
            top_names.join(", "),
            top_prob
        );

        if result.bayes_factor > 10.0 {
            explanation.push_str(&format!(
                "Strong evidence (Bayes factor = {:.2}) favouring this diagnosis.\n",
                result.bayes_factor
            ));
        } else if result.bayes_factor > 3.0 {
            explanation.push_str(&format!(
                "Moderate evidence (Bayes factor = {:.2}).\n",
                result.bayes_factor
            ));
        } else {
            explanation.push_str(&format!(
                "Weak evidence (Bayes factor = {:.2}); consider more tests.\n",
                result.bayes_factor
            ));
        }

        explanation.push_str("\nPer-stage fault probabilities (EM estimates):\n");
        for (i, &p) in result.stage_fault_probs.iter().enumerate() {
            let name = stage_names.get(i).map(|s| s.as_str()).unwrap_or("?");
            explanation.push_str(&format!("  {}: {:.4}\n", name, p));
        }

        explanation
    }

    // ── EM algorithm ────────────────────────────────────────────────────

    fn run_em(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> (Vec<f64>, usize, bool) {
        let n_stages = matrix.n_stages;
        let n_tests = matrix.n_tests;

        let mut probs: Vec<StageFaultProb> = (0..n_stages)
            .map(|_| StageFaultProb {
                prob: self.prior_fault_probability,
            })
            .collect();

        let mut prev_ll = f64::NEG_INFINITY;
        let mut converged = false;
        let mut iter = 0;

        // Pre-compute per-stage involvement weights based on differentials.
        // High differential ⟹ stage more "involved" in this test.
        let involvement = self.compute_involvement(matrix);

        for it in 0..self.em_max_iterations {
            iter = it + 1;

            // ── E-step: compute responsibilities ────────────────────────
            // For each test i, for each stage k:
            //   r_{i,k} = P(stage k is the fault | test i outcome, current probs)
            let mut responsibilities = vec![vec![0.0; n_stages]; n_tests];

            for i in 0..n_tests {
                let is_fail = violations.violations[i];
                let mut total_weight = 0.0;

                for k in 0..n_stages {
                    let w = involvement[i][k];
                    let p_fault = probs[k].prob;
                    let r = if is_fail {
                        // More likely fault if high involvement and high fault prob
                        w * p_fault
                    } else {
                        // Passing test: low involvement OR low fault prob
                        w * (1.0 - p_fault)
                    };
                    responsibilities[i][k] = r;
                    total_weight += r;
                }

                if total_weight > f64::EPSILON {
                    for k in 0..n_stages {
                        responsibilities[i][k] /= total_weight;
                    }
                }
            }

            // ── M-step: update fault probabilities ──────────────────────
            for k in 0..n_stages {
                let fail_resp: f64 = (0..n_tests)
                    .filter(|&i| violations.violations[i])
                    .map(|i| responsibilities[i][k])
                    .sum();

                let total_resp: f64 = (0..n_tests).map(|i| responsibilities[i][k]).sum();

                probs[k].prob = if total_resp > f64::EPSILON {
                    (fail_resp / total_resp).clamp(1e-10, 1.0 - 1e-10)
                } else {
                    self.prior_fault_probability
                };
            }

            // ── Convergence check (log-likelihood) ──────────────────────
            let ll = self.log_likelihood(matrix, violations, &probs, &involvement);
            if (ll - prev_ll).abs() < self.em_epsilon {
                converged = true;
                break;
            }
            prev_ll = ll;
        }

        let final_probs: Vec<f64> = probs.iter().map(|p| p.prob).collect();
        (final_probs, iter, converged)
    }

    fn compute_involvement(&self, matrix: &DifferentialMatrix) -> Vec<Vec<f64>> {
        let n_tests = matrix.n_tests;
        let n_stages = matrix.n_stages;
        let mut inv = vec![vec![0.0; n_stages]; n_tests];

        for i in 0..n_tests {
            let row_sum: f64 = matrix.data[i].iter().sum();
            for k in 0..n_stages {
                inv[i][k] = if row_sum > f64::EPSILON {
                    matrix.data[i][k] / row_sum
                } else {
                    1.0 / n_stages as f64
                };
            }
        }
        inv
    }

    fn log_likelihood(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
        probs: &[StageFaultProb],
        involvement: &[Vec<f64>],
    ) -> f64 {
        let mut ll = 0.0;
        for i in 0..matrix.n_tests {
            // P(outcome_i) under noisy-OR model
            let p_fail = self.noisy_or_fail_prob(i, probs, involvement);
            if violations.violations[i] {
                ll += p_fail.max(1e-300).ln();
            } else {
                ll += (1.0 - p_fail).max(1e-300).ln();
            }
        }
        ll
    }

    /// Noisy-OR: P(fail | all stage probs) = 1 - Π_k (1 - p_k * involvement_k)
    fn noisy_or_fail_prob(
        &self,
        test_idx: usize,
        probs: &[StageFaultProb],
        involvement: &[Vec<f64>],
    ) -> f64 {
        let mut prod = 1.0;
        for k in 0..probs.len() {
            prod *= 1.0 - probs[k].prob * involvement[test_idx][k];
        }
        1.0 - prod
    }

    // ── Candidate generation ────────────────────────────────────────────

    fn generate_candidates(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
        stage_probs: &[f64],
    ) -> Vec<FaultCandidate> {
        let n_stages = matrix.n_stages;
        let involvement = self.compute_involvement(matrix);

        // Single-fault candidates
        let mut candidates: Vec<FaultCandidate> = (0..n_stages)
            .map(|k| {
                let posterior = self.single_fault_posterior(k, stage_probs, violations, &involvement);
                let gof = self.goodness_of_fit(k, matrix, violations);
                FaultCandidate {
                    stage_subset: vec![k],
                    posterior_probability: posterior,
                    goodness_of_fit: gof,
                }
            })
            .collect();

        // Two-fault candidates (top pairs by prior)
        if n_stages >= 2 && self.max_candidates > n_stages {
            let mut ranked = rank_by_score(stage_probs);
            ranked.truncate(n_stages.min(6)); // top-6 stages
            for i in 0..ranked.len() {
                for j in (i + 1)..ranked.len() {
                    let subset = vec![ranked[i], ranked[j]];
                    let posterior =
                        self.multi_fault_posterior(&subset, stage_probs, violations, &involvement);
                    let gof = (self.goodness_of_fit(ranked[i], matrix, violations)
                        + self.goodness_of_fit(ranked[j], matrix, violations))
                        / 2.0;
                    candidates.push(FaultCandidate {
                        stage_subset: subset,
                        posterior_probability: posterior,
                        goodness_of_fit: gof,
                    });
                }
            }
        }

        // Sort by posterior probability (descending)
        candidates.sort_by(|a, b| {
            b.posterior_probability
                .partial_cmp(&a.posterior_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.max_candidates);
        candidates
    }

    fn single_fault_posterior(
        &self,
        stage: usize,
        stage_probs: &[f64],
        violations: &ViolationVector,
        involvement: &[Vec<f64>],
    ) -> f64 {
        // P(stage k faulty | data) ∝ P(data | stage k faulty) * P(stage k faulty)
        let prior = stage_probs[stage];
        let mut ll = 0.0;
        for i in 0..violations.len() {
            let p_fail = (stage_probs[stage] * involvement[i][stage]).min(0.999);
            if violations.violations[i] {
                ll += p_fail.max(1e-300).ln();
            } else {
                ll += (1.0 - p_fail).max(1e-300).ln();
            }
        }
        (prior * ll.exp()).max(0.0)
    }

    fn multi_fault_posterior(
        &self,
        subset: &[usize],
        stage_probs: &[f64],
        violations: &ViolationVector,
        involvement: &[Vec<f64>],
    ) -> f64 {
        // noisy-OR over the subset
        let prior: f64 = subset.iter().map(|&k| stage_probs[k]).product();
        let mut ll = 0.0;
        for i in 0..violations.len() {
            let mut prod = 1.0;
            for &k in subset {
                prod *= 1.0 - stage_probs[k] * involvement[i][k];
            }
            let p_fail = (1.0 - prod).min(0.999);
            if violations.violations[i] {
                ll += p_fail.max(1e-300).ln();
            } else {
                ll += (1.0 - p_fail).max(1e-300).ln();
            }
        }
        (prior * ll.exp()).max(0.0)
    }

    fn goodness_of_fit(
        &self,
        stage: usize,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> f64 {
        // Proportion of failing tests where this stage has above-median differential
        let col = matrix.column(stage);
        let mut sorted = col.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        };

        let n_fail = violations.n_violations();
        if n_fail == 0 {
            return 0.0;
        }

        let above_median_in_fail: usize = col
            .iter()
            .zip(violations.violations.iter())
            .filter(|(&d, &v)| v && d >= median)
            .count();

        above_median_in_fail as f64 / n_fail as f64
    }

    /// Bayes factor between top two hypotheses.
    fn compute_bayes_factor(&self, candidates: &[FaultCandidate]) -> f64 {
        if candidates.len() < 2 {
            return f64::INFINITY;
        }
        let p1 = candidates[0].posterior_probability;
        let p2 = candidates[1].posterior_probability;
        if p2.abs() < f64::EPSILON {
            if p1.abs() < f64::EPSILON {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            p1 / p2
        }
    }

    fn validate(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<()> {
        if matrix.n_tests != violations.len() {
            return Err(LocalizerError::matrix(
                "row count / violation vector mismatch",
                matrix.n_tests,
                matrix.n_stages,
            ));
        }
        if matrix.n_stages == 0 {
            return Err(LocalizerError::matrix("zero stages", matrix.n_tests, 0));
        }
        if !(0.0..=1.0).contains(&self.prior_fault_probability) {
            return Err(LocalizerError::validation(
                "barinel",
                "prior must be in [0, 1]",
            ));
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> DifferentialMatrix {
        let data = vec![
            vec![0.9, 0.05, 0.1],
            vec![0.85, 0.1, 0.05],
            vec![0.8, 0.0, 0.15],
            vec![0.05, 0.3, 0.05],
            vec![0.0, 0.25, 0.0],
            vec![0.1, 0.35, 0.1],
        ];
        let names = vec!["tokenizer".into(), "pos_tagger".into(), "parser".into()];
        DifferentialMatrix::new(data, names).unwrap()
    }

    fn sample_violations() -> ViolationVector {
        ViolationVector::new(vec![true, true, true, false, false, false])
    }

    #[test]
    fn test_basic_barinel() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        assert!(!result.candidate_rankings.is_empty());
        assert!(!result.most_probable_fault.is_empty());
    }

    #[test]
    fn test_top_candidate_is_faulty_stage() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        // Stage 0 (tokenizer) has highest differential on failing tests
        assert!(
            result.most_probable_fault.contains(&0),
            "expected stage 0 in top candidate, got {:?}",
            result.most_probable_fault
        );
    }

    #[test]
    fn test_em_convergence() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        assert!(result.converged, "EM should converge on clean data");
        assert!(result.em_iterations < 200);
    }

    #[test]
    fn test_bayes_factor() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        assert!(result.bayes_factor >= 1.0);
    }

    #[test]
    fn test_explain_result() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        let names = vec!["tokenizer".into(), "pos_tagger".into(), "parser".into()];
        let explanation = m.explain_result(&result, &names);
        assert!(explanation.contains("tokenizer") || explanation.contains("pos_tagger"));
        assert!(explanation.contains("posterior"));
    }

    #[test]
    fn test_no_violations() {
        let m = BarinelMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![false; 6]);
        let result = m.compute_posteriors(&matrix, &v).unwrap();
        // With no violations, priors dominate
        assert!(!result.candidate_rankings.is_empty());
    }

    #[test]
    fn test_all_violations() {
        let m = BarinelMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![true; 6]);
        let result = m.compute_posteriors(&matrix, &v).unwrap();
        assert!(!result.candidate_rankings.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let m = BarinelMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![true]);
        assert!(m.compute_posteriors(&matrix, &v).is_err());
    }

    #[test]
    fn test_multi_fault_candidates() {
        let m = BarinelMetric::new().with_max_candidates(100);
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        // Should include both single and two-fault candidates
        let has_multi = result
            .candidate_rankings
            .iter()
            .any(|c| c.stage_subset.len() > 1);
        assert!(has_multi, "should generate multi-fault candidates");
    }

    #[test]
    fn test_goodness_of_fit_bounds() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        for c in &result.candidate_rankings {
            assert!(
                (0.0..=1.0).contains(&c.goodness_of_fit),
                "GoF out of [0,1]: {}",
                c.goodness_of_fit
            );
        }
    }

    #[test]
    fn test_stage_probs_in_range() {
        let m = BarinelMetric::new();
        let result = m
            .compute_posteriors(&sample_matrix(), &sample_violations())
            .unwrap();
        for &p in &result.stage_fault_probs {
            assert!((0.0..=1.0).contains(&p), "prob out of [0,1]: {}", p);
        }
    }
}
