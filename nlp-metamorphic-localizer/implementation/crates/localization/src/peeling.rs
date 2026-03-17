//! Multi-fault peeling (Phase 4).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeelingFaultType { Introduction, Amplification, Both, None }

impl std::fmt::Display for PeelingFaultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Introduction => write!(f, "introduction"),
            Self::Amplification => write!(f, "amplification"),
            Self::Both => write!(f, "both"),
            Self::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedFault {
    pub stage_index: usize,
    pub stage_name: String,
    pub fault_type: PeelingFaultType,
    pub confidence: f64,
    pub discovery_round: usize,
    pub supporting_violations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeelingRoundInfo {
    pub round: usize,
    pub identified_stage: usize,
    pub identified_stage_name: String,
    pub residual_violations: usize,
    pub max_residual_suspiciousness: f64,
    pub suspiciousness_scores: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeelingResult {
    pub rounds: Vec<PeelingRoundInfo>,
    pub identified_faults: Vec<IdentifiedFault>,
    pub converged: bool,
    pub residual_violation_rate: f64,
    pub total_rounds: usize,
}

impl PeelingResult {
    pub fn fault_count(&self) -> usize { self.identified_faults.len() }
    pub fn is_multi_fault(&self) -> bool { self.identified_faults.len() > 1 }
    pub fn primary_fault(&self) -> Option<&IdentifiedFault> { self.identified_faults.first() }
}

#[derive(Debug, Clone)]
pub struct PeelingState {
    pub current_matrix: Vec<Vec<f64>>,
    pub violation_vector: Vec<bool>,
    pub identified_faults: Vec<IdentifiedFault>,
    pub residual_violations: usize,
    pub round: usize,
}

impl PeelingState {
    pub fn new(matrix: Vec<Vec<f64>>, violations: Vec<bool>) -> Self {
        let rv = violations.iter().filter(|&&v| v).count();
        Self { current_matrix: matrix, violation_vector: violations, identified_faults: Vec::new(), residual_violations: rv, round: 0 }
    }
}

pub struct FaultPeeler {
    pub max_rounds: usize,
    pub significance_threshold: f64,
    pub min_violations: usize,
    pub stage_names: Vec<String>,
}

impl FaultPeeler {
    pub fn new(stage_names: Vec<String>, max_rounds: usize, threshold: f64) -> Self {
        Self { max_rounds, significance_threshold: threshold, min_violations: 2, stage_names }
    }

    pub fn peel_faults(&self, matrix: &[Vec<f64>], violations: &[bool]) -> PeelingResult {
        let mut state = PeelingState::new(matrix.to_vec(), violations.to_vec());
        let mut rounds = Vec::new();
        let n_stages = self.stage_names.len();

        while state.round < self.max_rounds {
            let viol_count = state.violation_vector.iter().filter(|&&v| v).count();
            if viol_count < self.min_violations { break; }

            let scores = self.compute_ochiai(&state.current_matrix, &state.violation_vector, n_stages);
            if scores.is_empty() { break; }

            let (top_stage, top_score) = scores[0];
            if top_score < self.significance_threshold { break; }

            // Check if already identified
            if state.identified_faults.iter().any(|f| f.stage_index == top_stage) { break; }

            state.identified_faults.push(IdentifiedFault {
                stage_index: top_stage,
                stage_name: self.stage_names.get(top_stage).cloned().unwrap_or_default(),
                fault_type: if state.round == 0 { PeelingFaultType::Introduction } else { PeelingFaultType::Amplification },
                confidence: top_score,
                discovery_round: state.round,
                supporting_violations: viol_count,
            });

            // Zero out the identified stage
            for row in &mut state.current_matrix {
                if top_stage < row.len() { row[top_stage] = 0.0; }
            }

            // Recheck violations
            let mut new_viol_count = 0;
            for (i, v) in state.violation_vector.iter_mut().enumerate() {
                if *v {
                    let remaining: f64 = state.current_matrix[i].iter().sum();
                    if remaining < self.significance_threshold { *v = false; }
                    else { new_viol_count += 1; }
                }
            }

            let residual_susp = if scores.len() > 1 { scores[1].1 } else { 0.0 };

            rounds.push(PeelingRoundInfo {
                round: state.round,
                identified_stage: top_stage,
                identified_stage_name: self.stage_names.get(top_stage).cloned().unwrap_or_default(),
                residual_violations: new_viol_count,
                max_residual_suspiciousness: residual_susp,
                suspiciousness_scores: scores.clone(),
            });

            state.residual_violations = new_viol_count;
            state.round += 1;
        }

        let converged = state.residual_violations < self.min_violations
            || state.round >= self.max_rounds;
        let n_tests = violations.len().max(1) as f64;
        let residual_rate = state.residual_violations as f64 / n_tests;

        PeelingResult {
            rounds, identified_faults: state.identified_faults,
            converged, residual_violation_rate: residual_rate, total_rounds: state.round,
        }
    }

    fn compute_ochiai(&self, matrix: &[Vec<f64>], violations: &[bool], n_stages: usize) -> Vec<(usize, f64)> {
        let viol_count = violations.iter().filter(|&&v| v).count() as f64;
        if viol_count == 0.0 { return Vec::new(); }
        let n = matrix.len();

        let mut scores: Vec<(usize, f64)> = (0..n_stages).map(|k| {
            let mut sum_fail = 0.0;
            let mut sum_total = 0.0;
            for i in 0..n {
                let d = matrix[i].get(k).copied().unwrap_or(0.0);
                sum_total += d.abs();
                if violations[i] { sum_fail += d.abs(); }
            }
            let denom = (sum_total * viol_count).sqrt();
            let score = if denom > 0.0 { sum_fail / denom } else { 0.0 };
            (k, score)
        }).collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    pub fn convergence_check(&self, state: &PeelingState) -> bool {
        state.residual_violations < self.min_violations || state.round >= self.max_rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peeler() -> FaultPeeler {
        FaultPeeler::new(vec!["tok".into(), "tag".into(), "parse".into(), "ner".into()], 5, 0.05)
    }

    #[test]
    fn test_peel_single_fault() {
        let peeler = make_peeler();
        let matrix = vec![
            vec![0.1, 0.8, 0.1, 0.1],
            vec![0.0, 0.9, 0.0, 0.0],
            vec![0.1, 0.7, 0.2, 0.1],
            vec![0.0, 0.1, 0.0, 0.0],
        ];
        let violations = vec![true, true, true, false];
        let result = peeler.peel_faults(&matrix, &violations);
        assert!(result.fault_count() >= 1);
    }

    #[test]
    fn test_peel_no_violations() {
        let peeler = make_peeler();
        let matrix = vec![vec![0.1, 0.1, 0.1, 0.1]];
        let violations = vec![false];
        let result = peeler.peel_faults(&matrix, &violations);
        assert_eq!(result.fault_count(), 0);
        assert!(result.converged);
    }

    #[test]
    fn test_peel_converges() {
        let peeler = make_peeler();
        let matrix = vec![
            vec![0.5, 0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let violations = vec![true, true, true];
        let result = peeler.peel_faults(&matrix, &violations);
        assert!(result.total_rounds <= 5);
    }

    #[test]
    fn test_peeling_result_methods() {
        let result = PeelingResult {
            rounds: Vec::new(),
            identified_faults: vec![
                IdentifiedFault { stage_index: 1, stage_name: "tag".into(), fault_type: PeelingFaultType::Introduction, confidence: 0.9, discovery_round: 0, supporting_violations: 3 },
                IdentifiedFault { stage_index: 2, stage_name: "parse".into(), fault_type: PeelingFaultType::Amplification, confidence: 0.5, discovery_round: 1, supporting_violations: 2 },
            ],
            converged: true, residual_violation_rate: 0.1, total_rounds: 2,
        };
        assert!(result.is_multi_fault());
        assert_eq!(result.primary_fault().unwrap().stage_index, 1);
    }

    #[test]
    fn test_convergence_check() {
        let peeler = make_peeler();
        let state = PeelingState {
            current_matrix: Vec::new(), violation_vector: Vec::new(),
            identified_faults: Vec::new(), residual_violations: 0, round: 0,
        };
        assert!(peeler.convergence_check(&state));
    }

    #[test]
    fn test_fault_type_display() {
        assert_eq!(format!("{}", PeelingFaultType::Introduction), "introduction");
        assert_eq!(format!("{}", PeelingFaultType::Both), "both");
    }

    #[test]
    fn test_peeling_state() {
        let state = PeelingState::new(
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![true, false],
        );
        assert_eq!(state.residual_violations, 1);
        assert_eq!(state.round, 0);
    }
}
