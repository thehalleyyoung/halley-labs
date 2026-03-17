//! Interventional replay for causal refinement.
//!
//! Performs do-calculus-style interventions: replace stage k's input with the
//! original IR, re-execute stages k..n, and observe whether the violation persists.

use serde::{Deserialize, Serialize};
use shared_types::{
    IntermediateRepresentation, IRType, LocalizerError, PipelineStage, Result, Sentence,
    StageId,
};
use std::collections::HashMap;
use std::sync::Arc;

// ── Intervention ────────────────────────────────────────────────────────────

/// Describes a single intervention: replace input to stage k with an alternative IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intervention {
    pub target_stage_index: usize,
    pub target_stage_id: StageId,
    pub replacement_ir: IntermediateRepresentation,
    pub description: String,
}

impl Intervention {
    pub fn new(
        stage_index: usize,
        stage_id: StageId,
        replacement: IntermediateRepresentation,
        description: impl Into<String>,
    ) -> Self {
        Self {
            target_stage_index: stage_index,
            target_stage_id: stage_id,
            replacement_ir: replacement,
            description: description.into(),
        }
    }
}

// ── InterventionResult ──────────────────────────────────────────────────────

/// Result of performing a single intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionResult {
    pub intervention: Intervention,
    pub pre_intervention_output: IntermediateRepresentation,
    pub post_intervention_output: IntermediateRepresentation,
    pub violation_before: bool,
    pub violation_after: bool,
    pub attenuation_factor: f64,
}

impl InterventionResult {
    /// Whether the intervention resolved the violation.
    pub fn resolved_violation(&self) -> bool {
        self.violation_before && !self.violation_after
    }

    /// Whether the intervention had no effect.
    pub fn no_effect(&self) -> bool {
        self.violation_before == self.violation_after
            && self.attenuation_factor.abs() < 1e-6
    }
}

// ── InterventionPlan ────────────────────────────────────────────────────────

/// An ordered list of interventions to try, prioritized by suspicion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionPlan {
    pub interventions: Vec<Intervention>,
    pub strategy: String,
}

impl InterventionPlan {
    pub fn new(strategy: impl Into<String>) -> Self {
        Self {
            interventions: Vec::new(),
            strategy: strategy.into(),
        }
    }

    pub fn add(&mut self, intervention: Intervention) {
        self.interventions.push(intervention);
    }

    pub fn len(&self) -> usize {
        self.interventions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interventions.is_empty()
    }
}

// ── InterventionCache ───────────────────────────────────────────────────────

/// Memoizes sub-pipeline executions to avoid redundant computation.
#[derive(Debug, Clone)]
pub struct InterventionCache {
    entries: HashMap<String, IntermediateRepresentation>,
}

impl InterventionCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn cache_key(stage_index: usize, ir: &IntermediateRepresentation) -> String {
        // Use stage index + sentence text + IR type as a rough cache key
        format!(
            "{}:{}:{:?}",
            stage_index, ir.sentence.raw_text, ir.ir_type
        )
    }

    pub fn get(
        &self,
        stage_index: usize,
        input: &IntermediateRepresentation,
    ) -> Option<&IntermediateRepresentation> {
        let key = Self::cache_key(stage_index, input);
        self.entries.get(&key)
    }

    pub fn insert(
        &mut self,
        stage_index: usize,
        input: &IntermediateRepresentation,
        output: IntermediateRepresentation,
    ) {
        let key = Self::cache_key(stage_index, input);
        self.entries.insert(key, output);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for InterventionCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── InterventionEngine ──────────────────────────────────────────────────────

/// Engine for performing interventional replay on an NLP pipeline.
pub struct InterventionEngine {
    stages: Vec<Arc<dyn PipelineStage>>,
    pub cache: InterventionCache,
}

impl InterventionEngine {
    pub fn new(stages: Vec<Arc<dyn PipelineStage>>) -> Self {
        Self {
            stages,
            cache: InterventionCache::new(),
        }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Execute stages from `start_index` to the end, feeding `input` to stage[start_index].
    pub fn execute_suffix(
        &mut self,
        start_index: usize,
        input: &IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation> {
        if start_index >= self.stages.len() {
            return Ok(input.clone());
        }
        let mut current = input.clone();
        for idx in start_index..self.stages.len() {
            if let Some(cached) = self.cache.get(idx, &current) {
                current = cached.clone();
                continue;
            }
            let output = self.stages[idx].process(&current)?;
            self.cache.insert(idx, &current, output.clone());
            current = output;
        }
        Ok(current)
    }

    /// Execute the full pipeline from input.
    pub fn execute_full(
        &mut self,
        input: &IntermediateRepresentation,
    ) -> Result<IntermediateRepresentation> {
        self.execute_suffix(0, input)
    }

    /// Perform a single intervention: replace stage k's input with `replacement`,
    /// execute stages k..n, and compare with the non-intervention output.
    pub fn perform_intervention(
        &mut self,
        intervention: &Intervention,
        original_final_output: &IntermediateRepresentation,
        transformed_final_output: &IntermediateRepresentation,
        violation_threshold: f64,
    ) -> Result<InterventionResult> {
        let post_output =
            self.execute_suffix(intervention.target_stage_index, &intervention.replacement_ir)?;

        let violation_before =
            compute_output_divergence(original_final_output, transformed_final_output)
                > violation_threshold;
        let violation_after =
            compute_output_divergence(original_final_output, &post_output) > violation_threshold;

        let pre_div =
            compute_output_divergence(original_final_output, transformed_final_output);
        let post_div = compute_output_divergence(original_final_output, &post_output);
        let attenuation = if pre_div.abs() < 1e-12 {
            0.0
        } else {
            (pre_div - post_div) / pre_div
        };

        Ok(InterventionResult {
            intervention: intervention.clone(),
            pre_intervention_output: transformed_final_output.clone(),
            post_intervention_output: post_output,
            violation_before,
            violation_after,
            attenuation_factor: attenuation,
        })
    }

    /// Detect whether a violation changed after intervention.
    pub fn detect_violation_change(result: &InterventionResult) -> ViolationChange {
        match (result.violation_before, result.violation_after) {
            (true, false) => ViolationChange::Resolved,
            (true, true) => {
                if result.attenuation_factor > 0.1 {
                    ViolationChange::Attenuated
                } else {
                    ViolationChange::Persisted
                }
            }
            (false, true) => ViolationChange::Introduced,
            (false, false) => ViolationChange::NonePresent,
        }
    }

    /// Compute how much the violation decreased.
    pub fn compute_attenuation(result: &InterventionResult) -> f64 {
        result.attenuation_factor
    }

    /// Given suspect stages (ordered by suspicion, most suspect first),
    /// create an intervention plan using original IRs at each stage.
    pub fn plan_interventions(
        &self,
        suspect_stage_indices: &[usize],
        original_trace: &[IntermediateRepresentation],
    ) -> Result<InterventionPlan> {
        let mut plan = InterventionPlan::new("suspect-first");
        for &idx in suspect_stage_indices {
            if idx >= self.stages.len() {
                continue;
            }
            let replacement = if idx < original_trace.len() {
                original_trace[idx].clone()
            } else {
                continue;
            };
            let stage_id = self.stages[idx].id().clone();
            plan.add(Intervention::new(
                idx,
                stage_id.clone(),
                replacement,
                format!("Replace input to stage {} with original IR", stage_id),
            ));
        }
        Ok(plan)
    }

    /// Perform all interventions in a plan, returning results.
    pub fn batch_intervene(
        &mut self,
        plan: &InterventionPlan,
        original_final_output: &IntermediateRepresentation,
        transformed_final_output: &IntermediateRepresentation,
        violation_threshold: f64,
    ) -> Result<Vec<InterventionResult>> {
        let mut results = Vec::with_capacity(plan.interventions.len());
        for intervention in &plan.interventions {
            let result = self.perform_intervention(
                intervention,
                original_final_output,
                transformed_final_output,
                violation_threshold,
            )?;
            results.push(result);
        }
        Ok(results)
    }
}

// ── ViolationChange ─────────────────────────────────────────────────────────

/// Describes how a violation changed after intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationChange {
    Resolved,
    Attenuated,
    Persisted,
    Introduced,
    NonePresent,
}

// ── Helper: output divergence ───────────────────────────────────────────────

/// Simple divergence measure between two final outputs.
fn compute_output_divergence(a: &IntermediateRepresentation, b: &IntermediateRepresentation) -> f64 {
    let tokens_a: Vec<&str> = a.sentence.tokens.iter().map(|t| t.text.as_str()).collect();
    let tokens_b: Vec<&str> = b.sentence.tokens.iter().map(|t| t.text.as_str()).collect();

    if tokens_a.is_empty() && tokens_b.is_empty() {
        let conf_a = a.confidence.unwrap_or(0.5);
        let conf_b = b.confidence.unwrap_or(0.5);
        return (conf_a - conf_b).abs();
    }

    // Jaccard distance on token sets
    let set_a: std::collections::HashSet<&str> = tokens_a.iter().copied().collect();
    let set_b: std::collections::HashSet<&str> = tokens_b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        1.0 - (intersection as f64 / union as f64)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::Token;
    use std::sync::Arc;

    struct MockStage {
        id: StageId,
        name: String,
        suffix: String,
    }

    impl MockStage {
        fn new(name: &str, suffix: &str) -> Self {
            Self {
                id: StageId::new(name),
                name: name.to_string(),
                suffix: suffix.to_string(),
            }
        }
    }

    impl PipelineStage for MockStage {
        fn id(&self) -> &StageId {
            &self.id
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
            let mut out = input.clone();
            // Append suffix to each token
            for tok in &mut out.sentence.tokens {
                tok.text = format!("{}_{}", tok.text, self.suffix);
            }
            out.sentence.text = out
                .sentence
                .tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            Ok(out)
        }
        fn input_type(&self) -> IRType {
            IRType::Tokenized
        }
        fn output_type(&self) -> IRType {
            IRType::Tokenized
        }
    }

    fn make_ir(words: &[&str]) -> IntermediateRepresentation {
        let tokens: Vec<Token> = words
            .iter()
            .enumerate()
            .map(|(i, w)| Token::new(*w, i, i * 5, i * 5 + w.len()))
            .collect();
        let text = words.join(" ");
        IntermediateRepresentation::new(
            IRType::Tokenized,
            Sentence {
                text,
                tokens,
                entities: Vec::new(),
                dependencies: Vec::new(),
                parse_tree: None,
                features: None,
            },
        )
    }

    fn make_engine() -> InterventionEngine {
        let stages: Vec<Arc<dyn PipelineStage>> = vec![
            Arc::new(MockStage::new("tok", "T")),
            Arc::new(MockStage::new("pos", "P")),
            Arc::new(MockStage::new("ner", "N")),
        ];
        InterventionEngine::new(stages)
    }

    #[test]
    fn test_execute_full() {
        let mut engine = make_engine();
        let input = make_ir(&["hello", "world"]);
        let output = engine.execute_full(&input).unwrap();
        assert!(output.sentence.text.contains("hello_T_P_N"));
    }

    #[test]
    fn test_execute_suffix() {
        let mut engine = make_engine();
        let input = make_ir(&["hello"]);
        let output = engine.execute_suffix(1, &input).unwrap();
        // Should only apply stages 1 and 2 (pos, ner)
        assert!(output.sentence.text.contains("hello_P_N"));
        assert!(!output.sentence.text.contains("hello_T"));
    }

    #[test]
    fn test_perform_intervention() {
        let mut engine = make_engine();
        let original_input = make_ir(&["hello"]);
        let transformed_input = make_ir(&["goodbye"]);

        let orig_out = engine.execute_full(&original_input).unwrap();
        engine.cache.clear();
        let trans_out = engine.execute_full(&transformed_input).unwrap();
        engine.cache.clear();

        // Intervene at stage 1: feed original IR after stage 0
        let replacement = {
            let mut e = InterventionEngine::new(vec![Arc::new(MockStage::new("tok", "T"))]);
            e.execute_full(&original_input).unwrap()
        };

        let intervention = Intervention::new(
            1,
            StageId::new("pos"),
            replacement,
            "Replace pos input with original",
        );

        let result = engine
            .perform_intervention(&intervention, &orig_out, &trans_out, 0.1)
            .unwrap();

        // After intervention, the output should be closer to original
        assert!(result.attenuation_factor > 0.0 || !result.violation_before);
    }

    #[test]
    fn test_violation_change_resolved() {
        let result = InterventionResult {
            intervention: Intervention::new(0, StageId::new("s"), make_ir(&[]), "test"),
            pre_intervention_output: make_ir(&["a"]),
            post_intervention_output: make_ir(&["b"]),
            violation_before: true,
            violation_after: false,
            attenuation_factor: 1.0,
        };
        assert_eq!(
            InterventionEngine::detect_violation_change(&result),
            ViolationChange::Resolved
        );
    }

    #[test]
    fn test_violation_change_persisted() {
        let result = InterventionResult {
            intervention: Intervention::new(0, StageId::new("s"), make_ir(&[]), "test"),
            pre_intervention_output: make_ir(&["a"]),
            post_intervention_output: make_ir(&["b"]),
            violation_before: true,
            violation_after: true,
            attenuation_factor: 0.01,
        };
        assert_eq!(
            InterventionEngine::detect_violation_change(&result),
            ViolationChange::Persisted
        );
    }

    #[test]
    fn test_plan_interventions() {
        let engine = make_engine();
        let original_trace = vec![
            make_ir(&["hello_T"]),
            make_ir(&["hello_T_P"]),
            make_ir(&["hello_T_P_N"]),
        ];
        let plan = engine
            .plan_interventions(&[2, 0], &original_trace)
            .unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.interventions[0].target_stage_index, 2);
        assert_eq!(plan.interventions[1].target_stage_index, 0);
    }

    #[test]
    fn test_intervention_cache() {
        let mut cache = InterventionCache::new();
        let ir = make_ir(&["hello"]);
        let output = make_ir(&["hello_processed"]);
        cache.insert(0, &ir, output.clone());
        assert_eq!(cache.len(), 1);
        let cached = cache.get(0, &ir).unwrap();
        assert_eq!(cached.sentence.text, "hello_processed");
    }

    #[test]
    fn test_cache_miss() {
        let cache = InterventionCache::new();
        let ir = make_ir(&["hello"]);
        assert!(cache.get(0, &ir).is_none());
    }

    #[test]
    fn test_batch_intervene() {
        let mut engine = make_engine();
        let original_input = make_ir(&["hello"]);
        let transformed_input = make_ir(&["goodbye"]);

        let orig_out = engine.execute_full(&original_input).unwrap();
        engine.cache.clear();
        let trans_out = engine.execute_full(&transformed_input).unwrap();
        engine.cache.clear();

        let original_trace = vec![
            make_ir(&["hello_T"]),
            make_ir(&["hello_T_P"]),
            make_ir(&["hello_T_P_N"]),
        ];
        let plan = engine
            .plan_interventions(&[0, 1, 2], &original_trace)
            .unwrap();
        let results = engine
            .batch_intervene(&plan, &orig_out, &trans_out, 0.1)
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_compute_output_divergence_identical() {
        let a = make_ir(&["hello", "world"]);
        let b = make_ir(&["hello", "world"]);
        let div = compute_output_divergence(&a, &b);
        assert!(div < 1e-9);
    }

    #[test]
    fn test_compute_output_divergence_different() {
        let a = make_ir(&["hello"]);
        let b = make_ir(&["goodbye"]);
        let div = compute_output_divergence(&a, &b);
        assert_eq!(div, 1.0);
    }

    #[test]
    fn test_intervention_plan_empty() {
        let plan = InterventionPlan::new("empty");
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_resolved_violation_helper() {
        let result = InterventionResult {
            intervention: Intervention::new(0, StageId::new("s"), make_ir(&[]), "test"),
            pre_intervention_output: make_ir(&[]),
            post_intervention_output: make_ir(&[]),
            violation_before: true,
            violation_after: false,
            attenuation_factor: 1.0,
        };
        assert!(result.resolved_violation());
    }
}
