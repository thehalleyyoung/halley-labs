//! Causal refinement (Phase 3) for CAUSAL_LOCALIZE.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementFaultType {
    Introduction,
    Amplification,
    Both,
    None,
}

impl std::fmt::Display for RefinementFaultType {
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
pub struct ViolationRef {
    pub test_id: String,
    pub violation_type: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEffect {
    pub stage_index: usize,
    pub stage_name: String,
    pub dce: f64,
    pub ie: f64,
    pub fault_type: RefinementFaultType,
    pub confidence: f64,
    pub changed_output: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementPlan {
    pub suspect_stages: Vec<usize>,
    pub violations_to_refine: Vec<ViolationRef>,
    pub budget: usize,
}

impl RefinementPlan {
    pub fn new(suspects: Vec<usize>, budget: usize) -> Self {
        Self {
            suspect_stages: suspects,
            violations_to_refine: Vec::new(),
            budget,
        }
    }

    pub fn add_violation(&mut self, vref: ViolationRef) {
        self.violations_to_refine.push(vref);
    }

    pub fn is_within_budget(&self) -> bool {
        self.violations_to_refine.len() * self.suspect_stages.len() <= self.budget
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementResult {
    pub violation_id: String,
    pub per_stage_effects: Vec<StageEffect>,
}

impl RefinementResult {
    pub fn primary_fault_stage(&self) -> Option<&StageEffect> {
        self.per_stage_effects.iter()
            .filter(|e| e.fault_type != RefinementFaultType::None)
            .max_by(|a, b| a.dce.partial_cmp(&b.dce).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedStageResult {
    pub stage_index: usize,
    pub stage_name: String,
    pub introduction_count: usize,
    pub amplification_count: usize,
    pub both_count: usize,
    pub none_count: usize,
    pub mean_dce: f64,
    pub mean_ie: f64,
    pub overall_classification: RefinementFaultType,
}

pub struct CausalRefiner {
    pub significance_threshold: f64,
    pub stage_names: Vec<String>,
}

impl CausalRefiner {
    pub fn new(stage_names: Vec<String>, threshold: f64) -> Self {
        Self { significance_threshold: threshold, stage_names }
    }

    pub fn plan_refinement(
        &self,
        suspiciousness_scores: &[(usize, f64)],
        violations: &[ViolationRef],
        budget: usize,
    ) -> RefinementPlan {
        let mut suspects: Vec<usize> = suspiciousness_scores.iter()
            .filter(|(_, s)| *s > self.significance_threshold)
            .map(|(idx, _)| *idx)
            .take(3)
            .collect();
        if suspects.is_empty() && !suspiciousness_scores.is_empty() {
            suspects.push(suspiciousness_scores[0].0);
        }

        let mut plan = RefinementPlan::new(suspects, budget);
        let mut sorted_violations: Vec<ViolationRef> = violations.to_vec();
        sorted_violations.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));

        let max_violations = budget / plan.suspect_stages.len().max(1);
        for v in sorted_violations.into_iter().take(max_violations) {
            plan.add_violation(v);
        }
        plan
    }

    pub fn refine_violation(
        &self,
        matrix_row: &[f64],
        suspect_stages: &[usize],
    ) -> RefinementResult {
        let total_delta: f64 = matrix_row.iter().sum::<f64>() / matrix_row.len().max(1) as f64;

        let effects: Vec<StageEffect> = suspect_stages.iter().map(|&k| {
            let delta_k = matrix_row.get(k).copied().unwrap_or(0.0);

            let dce = if total_delta > 0.0 { delta_k / total_delta } else { 0.0 };
            let downstream: f64 = matrix_row.iter().skip(k + 1)
                .sum::<f64>() / matrix_row.len().max(1) as f64;
            let ie = if total_delta > 0.0 { downstream / total_delta } else { 0.0 };

            let fault_type = classify_effect(dce, ie, self.significance_threshold);
            let changed = dce > self.significance_threshold;

            StageEffect {
                stage_index: k,
                stage_name: self.stage_names.get(k).cloned().unwrap_or_default(),
                dce, ie, fault_type, confidence: dce.min(1.0), changed_output: changed,
            }
        }).collect();

        RefinementResult { violation_id: String::new(), per_stage_effects: effects }
    }

    pub fn aggregate_effects(&self, results: &[RefinementResult]) -> Vec<AggregatedStageResult> {
        let mut stage_map: HashMap<usize, Vec<&StageEffect>> = HashMap::new();
        for r in results {
            for e in &r.per_stage_effects {
                stage_map.entry(e.stage_index).or_default().push(e);
            }
        }

        stage_map.into_iter().map(|(idx, effects)| {
            let n = effects.len();
            let introduction_count = effects.iter().filter(|e| e.fault_type == RefinementFaultType::Introduction).count();
            let amplification_count = effects.iter().filter(|e| e.fault_type == RefinementFaultType::Amplification).count();
            let both_count = effects.iter().filter(|e| e.fault_type == RefinementFaultType::Both).count();
            let none_count = effects.iter().filter(|e| e.fault_type == RefinementFaultType::None).count();
            let mean_dce = effects.iter().map(|e| e.dce).sum::<f64>() / n.max(1) as f64;
            let mean_ie = effects.iter().map(|e| e.ie).sum::<f64>() / n.max(1) as f64;

            let overall = if introduction_count >= amplification_count && introduction_count >= both_count {
                if introduction_count > 0 { RefinementFaultType::Introduction } else { RefinementFaultType::None }
            } else if both_count >= amplification_count {
                RefinementFaultType::Both
            } else {
                RefinementFaultType::Amplification
            };

            AggregatedStageResult {
                stage_index: idx,
                stage_name: self.stage_names.get(idx).cloned().unwrap_or_default(),
                introduction_count, amplification_count, both_count, none_count,
                mean_dce, mean_ie, overall_classification: overall,
            }
        }).collect()
    }

    pub fn prioritize_refinement(&self, violations: &mut [ViolationRef]) {
        violations.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    }
}

pub fn classify_effect(dce: f64, ie: f64, threshold: f64) -> RefinementFaultType {
    let dce_sig = dce > threshold;
    let ie_sig = ie > threshold;
    match (dce_sig, ie_sig) {
        (true, true) => RefinementFaultType::Both,
        (true, false) => RefinementFaultType::Introduction,
        (false, true) => RefinementFaultType::Amplification,
        (false, false) => RefinementFaultType::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_effect() {
        assert_eq!(classify_effect(0.5, 0.0, 0.05), RefinementFaultType::Introduction);
        assert_eq!(classify_effect(0.0, 0.5, 0.05), RefinementFaultType::Amplification);
        assert_eq!(classify_effect(0.5, 0.5, 0.05), RefinementFaultType::Both);
        assert_eq!(classify_effect(0.01, 0.01, 0.05), RefinementFaultType::None);
    }

    #[test]
    fn test_refinement_plan() {
        let refiner = CausalRefiner::new(vec!["a".into(), "b".into()], 0.1);
        let scores = vec![(0, 0.8), (1, 0.3)];
        let violations = vec![ViolationRef { test_id: "t1".into(), violation_type: "entity".into(), severity: 0.9 }];
        let plan = refiner.plan_refinement(&scores, &violations, 10);
        assert!(!plan.suspect_stages.is_empty());
    }

    #[test]
    fn test_refine_violation() {
        let refiner = CausalRefiner::new(vec!["a".into(), "b".into(), "c".into()], 0.05);
        let row = vec![0.1, 0.8, 0.2];
        let result = refiner.refine_violation(&row, &[1]);
        assert_eq!(result.per_stage_effects.len(), 1);
    }

    #[test]
    fn test_aggregate_effects() {
        let refiner = CausalRefiner::new(vec!["a".into(), "b".into()], 0.05);
        let results = vec![
            RefinementResult {
                violation_id: "v1".into(),
                per_stage_effects: vec![StageEffect {
                    stage_index: 0, stage_name: "a".into(), dce: 0.5, ie: 0.1,
                    fault_type: RefinementFaultType::Introduction, confidence: 0.8, changed_output: true,
                }],
            },
            RefinementResult {
                violation_id: "v2".into(),
                per_stage_effects: vec![StageEffect {
                    stage_index: 0, stage_name: "a".into(), dce: 0.6, ie: 0.05,
                    fault_type: RefinementFaultType::Introduction, confidence: 0.9, changed_output: true,
                }],
            },
        ];
        let aggregated = refiner.aggregate_effects(&results);
        assert!(!aggregated.is_empty());
        assert_eq!(aggregated[0].introduction_count, 2);
    }

    #[test]
    fn test_fault_type_display() {
        assert_eq!(format!("{}", RefinementFaultType::Introduction), "introduction");
    }

    #[test]
    fn test_refinement_result_primary() {
        let result = RefinementResult {
            violation_id: "v1".into(),
            per_stage_effects: vec![
                StageEffect { stage_index: 0, stage_name: "a".into(), dce: 0.1, ie: 0.0, fault_type: RefinementFaultType::None, confidence: 0.1, changed_output: false },
                StageEffect { stage_index: 1, stage_name: "b".into(), dce: 0.8, ie: 0.1, fault_type: RefinementFaultType::Introduction, confidence: 0.9, changed_output: true },
            ],
        };
        let primary = result.primary_fault_stage().unwrap();
        assert_eq!(primary.stage_index, 1);
    }

    #[test]
    fn test_prioritize() {
        let refiner = CausalRefiner::new(vec!["a".into()], 0.05);
        let mut violations = vec![
            ViolationRef { test_id: "t1".into(), violation_type: "a".into(), severity: 0.3 },
            ViolationRef { test_id: "t2".into(), violation_type: "b".into(), severity: 0.9 },
        ];
        refiner.prioritize_refinement(&mut violations);
        assert!(violations[0].severity > violations[1].severity);
    }

    #[test]
    fn test_plan_budget() {
        let plan = RefinementPlan::new(vec![0, 1], 10);
        assert!(plan.is_within_budget());
    }
}
