//! Direct Causal Effect (DCE) and Indirect Effect (IE) computation
//! for causal fault localization in NLP pipelines.

use serde::{Deserialize, Serialize};
use shared_types::{
    ConfidenceInterval, IntermediateRepresentation, IRType, LocalizerError, PipelineStage,
    Result, Sentence, StageId, Token,
};
use std::collections::HashMap;
use std::sync::Arc;

use crate::intervention::{InterventionEngine, Intervention, InterventionResult, ViolationChange};
use crate::stage_differential::{DifferentialComputer, StageDifferential};

// ── FaultType ───────────────────────────────────────────────────────────────

/// Classification of the causal role of a stage in a violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaultType {
    /// Stage introduces the fault directly (DCE > 0, IE ≈ 0).
    Introduction,
    /// Stage amplifies a fault from upstream (DCE ≈ 0, IE > 0).
    Amplification,
    /// Stage both introduces and amplifies (DCE > 0, IE > 0).
    Both,
    /// Stage is not causally involved.
    None,
}

impl FaultType {
    pub fn is_faulty(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Introduction => "directly introduces fault",
            Self::Amplification => "amplifies upstream fault",
            Self::Both => "introduces and amplifies",
            Self::None => "not involved",
        }
    }
}

impl std::fmt::Display for FaultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.description())
    }
}

// ── CausalEffect ────────────────────────────────────────────────────────────

/// Causal effect decomposition for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEffect {
    pub stage_index: usize,
    pub stage_name: StageId,
    pub dce_value: f64,
    pub ie_value: f64,
    pub total_effect: f64,
    pub effect_type: FaultType,
}

impl CausalEffect {
    pub fn new(
        stage_index: usize,
        stage_name: StageId,
        dce: f64,
        ie: f64,
    ) -> Self {
        let total = dce + ie;
        let effect_type = classify_fault_type_internal(dce, ie, 0.05);
        Self {
            stage_index,
            stage_name,
            dce_value: dce,
            ie_value: ie,
            total_effect: total,
            effect_type,
        }
    }

    /// Fraction of total effect due to direct cause.
    pub fn dce_fraction(&self) -> f64 {
        if self.total_effect.abs() < 1e-12 {
            0.0
        } else {
            self.dce_value / self.total_effect
        }
    }

    /// Fraction of total effect due to indirect cause.
    pub fn ie_fraction(&self) -> f64 {
        if self.total_effect.abs() < 1e-12 {
            0.0
        } else {
            self.ie_value / self.total_effect
        }
    }
}

// ── CausalDecomposition ─────────────────────────────────────────────────────

/// Complete causal decomposition for all stages in a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDecomposition {
    pub effects: Vec<CausalEffect>,
    pub primary_fault_stage: Option<usize>,
    pub confidence: Option<ConfidenceInterval>,
}

impl CausalDecomposition {
    pub fn new(effects: Vec<CausalEffect>) -> Self {
        let primary = effects
            .iter()
            .filter(|e| e.effect_type.is_faulty())
            .max_by(|a, b| {
                a.total_effect
                    .partial_cmp(&b.total_effect)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.stage_index);

        Self {
            effects,
            primary_fault_stage: primary,
            confidence: None,
        }
    }

    pub fn with_confidence(mut self, ci: ConfidenceInterval) -> Self {
        self.confidence = Some(ci);
        self
    }

    /// Get all stages classified as fault-introducing.
    pub fn fault_stages(&self) -> Vec<&CausalEffect> {
        self.effects.iter().filter(|e| e.effect_type.is_faulty()).collect()
    }

    /// Get the stage with the largest total effect.
    pub fn max_effect_stage(&self) -> Option<&CausalEffect> {
        self.effects.iter().max_by(|a, b| {
            a.total_effect
                .partial_cmp(&b.total_effect)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ── CausalChain ─────────────────────────────────────────────────────────────

/// Ordered list of causal contributions through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    pub links: Vec<CausalChainLink>,
}

/// A single link in the causal chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChainLink {
    pub stage_index: usize,
    pub stage_name: StageId,
    pub fault_type: FaultType,
    pub effect_magnitude: f64,
    pub cumulative_effect: f64,
}

impl CausalChain {
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    pub fn push(&mut self, link: CausalChainLink) {
        self.links.push(link);
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// The first stage in the chain that introduces the fault.
    pub fn root_cause(&self) -> Option<&CausalChainLink> {
        self.links
            .iter()
            .find(|l| l.fault_type == FaultType::Introduction || l.fault_type == FaultType::Both)
    }

    /// Total cumulative effect.
    pub fn total_effect(&self) -> f64 {
        self.links.last().map(|l| l.cumulative_effect).unwrap_or(0.0)
    }
}

impl Default for CausalChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── CausalAnalyzer ──────────────────────────────────────────────────────────

/// Main analyzer for computing DCE/IE decompositions.
pub struct CausalAnalyzer {
    pub intervention_engine: InterventionEngine,
    pub differential_computer: DifferentialComputer,
    pub significance_threshold: f64,
}

impl CausalAnalyzer {
    pub fn new(
        stages: Vec<Arc<dyn PipelineStage>>,
        significance_threshold: f64,
    ) -> Self {
        Self {
            intervention_engine: InterventionEngine::new(stages),
            differential_computer: DifferentialComputer::new(),
            significance_threshold,
        }
    }

    /// Compute the Direct Causal Effect for stage k.
    ///
    /// DCE_k = Δ_k(transformed) - Δ_k(intervened)
    /// where "intervened" means we replace stage k's input with the original IR.
    pub fn compute_dce(
        &mut self,
        stage_index: usize,
        original_trace: &[IntermediateRepresentation],
        transformed_trace: &[IntermediateRepresentation],
    ) -> Result<f64> {
        if stage_index >= original_trace.len() || stage_index >= transformed_trace.len() {
            return Err(LocalizerError::validation("validation", format!(
                "Stage index {} out of range (orig={}, trans={})",
                stage_index,
                original_trace.len(),
                transformed_trace.len()
            )));
        }

        // Compute pre-intervention differential at stage k
        let stage_id = StageId::new(&format!("stage_{}", stage_index));

        let pre_diff = self.differential_computer.compute_stage_differential(
            &stage_id,
            stage_index,
            &original_trace[stage_index],
            &transformed_trace[stage_index],
        )?;

        // Intervene: replace stage k's input with original, re-execute from k
        let replacement = if stage_index > 0 {
            original_trace[stage_index - 1].clone()
        } else {
            original_trace[0].clone()
        };

        let intervention = Intervention::new(
            stage_index,
            stage_id.clone(),
            replacement,
            "DCE intervention",
        );

        let post_ir = self
            .intervention_engine
            .execute_suffix(stage_index, &intervention.replacement_ir)?;

        // Compute post-intervention differential
        let post_diff = self.differential_computer.compute_stage_differential(
            &stage_id,
            stage_index,
            &original_trace[stage_index],
            &post_ir,
        )?;

        let dce = pre_diff.delta_value - post_diff.delta_value;
        Ok(dce.max(0.0)) // DCE is non-negative
    }

    /// Compute the Indirect Effect for stage k.
    /// IE_k = total_effect_k - DCE_k
    pub fn compute_ie(
        &mut self,
        stage_index: usize,
        original_trace: &[IntermediateRepresentation],
        transformed_trace: &[IntermediateRepresentation],
    ) -> Result<f64> {
        let stage_id = StageId::new(&format!("stage_{}", stage_index));

        let total_effect = self.differential_computer.compute_stage_differential(
            &stage_id,
            stage_index,
            &original_trace[stage_index],
            &transformed_trace[stage_index],
        )?;

        let dce = self.compute_dce(stage_index, original_trace, transformed_trace)?;
        let ie = total_effect.delta_value - dce;
        Ok(ie.max(0.0))
    }

    /// Classify the fault type based on DCE and IE magnitudes.
    pub fn classify_fault_type(&self, dce: f64, ie: f64) -> FaultType {
        classify_fault_type_internal(dce, ie, self.significance_threshold)
    }

    /// For a single violation, compute DCE/IE for all suspect stages.
    pub fn analyze_violation(
        &mut self,
        suspect_stages: &[usize],
        original_trace: &[IntermediateRepresentation],
        transformed_trace: &[IntermediateRepresentation],
    ) -> Result<CausalDecomposition> {
        let mut effects = Vec::with_capacity(suspect_stages.len());

        for &idx in suspect_stages {
            if idx >= original_trace.len() || idx >= transformed_trace.len() {
                continue;
            }

            let dce = self.compute_dce(idx, original_trace, transformed_trace)?;
            self.intervention_engine.cache.clear();
            let ie = self.compute_ie(idx, original_trace, transformed_trace)?;
            self.intervention_engine.cache.clear();

            let stage_name = StageId::new(&format!("stage_{}", idx));

            effects.push(CausalEffect::new(idx, stage_name, dce, ie));
        }

        Ok(CausalDecomposition::new(effects))
    }

    /// Compute bootstrap confidence intervals for DCE/IE estimates.
    pub fn compute_confidence(
        &mut self,
        stage_index: usize,
        original_traces: &[Vec<IntermediateRepresentation>],
        transformed_traces: &[Vec<IntermediateRepresentation>],
        confidence_level: f64,
    ) -> Result<(ConfidenceInterval, ConfidenceInterval)> {
        let n = original_traces.len().min(transformed_traces.len());
        if n == 0 {
            return Err(LocalizerError::validation(
                "validation",
                "Need at least one trace pair for bootstrap",
            ));
        }

        let mut dce_samples = Vec::with_capacity(n);
        let mut ie_samples = Vec::with_capacity(n);

        for i in 0..n {
            let dce = self.compute_dce(stage_index, &original_traces[i], &transformed_traces[i])?;
            self.intervention_engine.cache.clear();
            let ie = self.compute_ie(stage_index, &original_traces[i], &transformed_traces[i])?;
            self.intervention_engine.cache.clear();
            dce_samples.push(dce);
            ie_samples.push(ie);
        }

        let dce_ci = bootstrap_ci(&dce_samples, confidence_level);
        let ie_ci = bootstrap_ci(&ie_samples, confidence_level);

        Ok((dce_ci, ie_ci))
    }

    /// Trace the full causal chain through all pipeline stages.
    pub fn causal_chain_analysis(
        &mut self,
        original_trace: &[IntermediateRepresentation],
        transformed_trace: &[IntermediateRepresentation],
    ) -> Result<CausalChain> {
        let n = original_trace.len().min(transformed_trace.len());
        let mut chain = CausalChain::new();
        let mut cumulative = 0.0;

        for idx in 0..n {
            let dce = self.compute_dce(idx, original_trace, transformed_trace)?;
            self.intervention_engine.cache.clear();
            let ie = self.compute_ie(idx, original_trace, transformed_trace)?;
            self.intervention_engine.cache.clear();

            let total = dce + ie;
            cumulative += total;
            let fault_type = self.classify_fault_type(dce, ie);

            let stage_name = StageId::new(&format!("stage_{}", idx));

            chain.push(CausalChainLink {
                stage_index: idx,
                stage_name,
                fault_type,
                effect_magnitude: total,
                cumulative_effect: cumulative,
            });
        }

        Ok(chain)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn classify_fault_type_internal(dce: f64, ie: f64, threshold: f64) -> FaultType {
    let dce_significant = dce > threshold;
    let ie_significant = ie > threshold;
    match (dce_significant, ie_significant) {
        (true, false) => FaultType::Introduction,
        (false, true) => FaultType::Amplification,
        (true, true) => FaultType::Both,
        (false, false) => FaultType::None,
    }
}

fn bootstrap_ci(samples: &[f64], confidence_level: f64) -> ConfidenceInterval {
    if samples.is_empty() {
        return ConfidenceInterval::new(0.0, 0.0, confidence_level);
    }
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;

    if n == 1 {
        return ConfidenceInterval::new(mean, mean, confidence_level);
    }

    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();

    // Use z-score for the confidence level (normal approximation)
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };

    ConfidenceInterval::new(mean - z * se, mean + z * se, confidence_level)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token(text: &str, idx: usize) -> Token {
        Token::new(text, idx, idx * 5, idx * 5 + text.len())
    }

    fn make_ir_with_stage(words: &[&str], stage: &str) -> IntermediateRepresentation {
        let tokens: Vec<Token> = words
            .iter()
            .enumerate()
            .map(|(i, w)| make_token(w, i))
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
        .with_stage(StageId::new(stage))
    }

    struct IdentityStage {
        id: StageId,
        name: String,
    }

    impl IdentityStage {
        fn new(name: &str) -> Self {
            Self {
                id: StageId::new(name),
                name: name.to_string(),
            }
        }
    }

    impl PipelineStage for IdentityStage {
        fn id(&self) -> &StageId {
            &self.id
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn process(&self, input: &IntermediateRepresentation) -> Result<IntermediateRepresentation> {
            Ok(input.clone())
        }
        fn input_type(&self) -> IRType {
            IRType::Tokenized
        }
        fn output_type(&self) -> IRType {
            IRType::Tokenized
        }
    }

    fn make_analyzer() -> CausalAnalyzer {
        let stages: Vec<Arc<dyn PipelineStage>> = vec![
            Arc::new(IdentityStage::new("tok")),
            Arc::new(IdentityStage::new("pos")),
            Arc::new(IdentityStage::new("ner")),
        ];
        CausalAnalyzer::new(stages, 0.05)
    }

    #[test]
    fn test_fault_type_classification() {
        assert_eq!(classify_fault_type_internal(0.5, 0.01, 0.05), FaultType::Introduction);
        assert_eq!(classify_fault_type_internal(0.01, 0.5, 0.05), FaultType::Amplification);
        assert_eq!(classify_fault_type_internal(0.5, 0.5, 0.05), FaultType::Both);
        assert_eq!(classify_fault_type_internal(0.01, 0.01, 0.05), FaultType::None);
    }

    #[test]
    fn test_fault_type_is_faulty() {
        assert!(FaultType::Introduction.is_faulty());
        assert!(FaultType::Amplification.is_faulty());
        assert!(FaultType::Both.is_faulty());
        assert!(!FaultType::None.is_faulty());
    }

    #[test]
    fn test_causal_effect_fractions() {
        let effect = CausalEffect::new(0, StageId::new("tok"), 0.3, 0.7);
        assert!((effect.dce_fraction() - 0.3).abs() < 1e-9);
        assert!((effect.ie_fraction() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_causal_decomposition_primary() {
        let effects = vec![
            CausalEffect::new(0, StageId::new("tok"), 0.1, 0.0),
            CausalEffect::new(1, StageId::new("pos"), 0.5, 0.3),
            CausalEffect::new(2, StageId::new("ner"), 0.0, 0.02),
        ];
        let decomp = CausalDecomposition::new(effects);
        assert_eq!(decomp.primary_fault_stage, Some(1));
        assert_eq!(decomp.fault_stages().len(), 2);
    }

    #[test]
    fn test_causal_chain_root_cause() {
        let mut chain = CausalChain::new();
        chain.push(CausalChainLink {
            stage_index: 0,
            stage_name: StageId::new("tok"),
            fault_type: FaultType::None,
            effect_magnitude: 0.01,
            cumulative_effect: 0.01,
        });
        chain.push(CausalChainLink {
            stage_index: 1,
            stage_name: StageId::new("pos"),
            fault_type: FaultType::Introduction,
            effect_magnitude: 0.5,
            cumulative_effect: 0.51,
        });
        chain.push(CausalChainLink {
            stage_index: 2,
            stage_name: StageId::new("ner"),
            fault_type: FaultType::Amplification,
            effect_magnitude: 0.3,
            cumulative_effect: 0.81,
        });
        let root = chain.root_cause().unwrap();
        assert_eq!(root.stage_index, 1);
        assert!((chain.total_effect() - 0.81).abs() < 1e-9);
    }

    #[test]
    fn test_bootstrap_ci() {
        let samples = vec![0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.17, 0.19];
        let ci = bootstrap_ci(&samples, 0.95);
        assert!(ci.lower < ci.upper);
        assert!(ci.contains(ci.point_estimate));
        assert!((ci.confidence_level - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_bootstrap_ci_single_sample() {
        let samples = vec![0.5];
        let ci = bootstrap_ci(&samples, 0.95);
        assert_eq!(ci.lower, 0.5);
        assert_eq!(ci.upper, 0.5);
    }

    #[test]
    fn test_compute_dce_identity_pipeline() {
        let mut analyzer = make_analyzer();
        let orig_trace = vec![
            make_ir_with_stage(&["hello", "world"], "tok"),
            make_ir_with_stage(&["hello", "world"], "pos"),
            make_ir_with_stage(&["hello", "world"], "ner"),
        ];
        let trans_trace = vec![
            make_ir_with_stage(&["goodbye", "world"], "tok"),
            make_ir_with_stage(&["goodbye", "world"], "pos"),
            make_ir_with_stage(&["goodbye", "world"], "ner"),
        ];

        let dce = analyzer.compute_dce(0, &orig_trace, &trans_trace).unwrap();
        // With identity stages, intervention at 0 gives original → 0 differential
        assert!(dce >= 0.0);
    }

    #[test]
    fn test_compute_ie_identity_pipeline() {
        let mut analyzer = make_analyzer();
        let orig_trace = vec![
            make_ir_with_stage(&["hello"], "tok"),
            make_ir_with_stage(&["hello"], "pos"),
        ];
        let trans_trace = vec![
            make_ir_with_stage(&["goodbye"], "tok"),
            make_ir_with_stage(&["goodbye"], "pos"),
        ];
        let ie = analyzer.compute_ie(0, &orig_trace, &trans_trace).unwrap();
        assert!(ie >= 0.0);
    }

    #[test]
    fn test_analyze_violation() {
        let mut analyzer = make_analyzer();
        let orig_trace = vec![
            make_ir_with_stage(&["the", "cat", "sat"], "tok"),
            make_ir_with_stage(&["the", "cat", "sat"], "pos"),
            make_ir_with_stage(&["the", "cat", "sat"], "ner"),
        ];
        let trans_trace = vec![
            make_ir_with_stage(&["the", "dog", "sat"], "tok"),
            make_ir_with_stage(&["the", "dog", "sat"], "pos"),
            make_ir_with_stage(&["the", "dog", "sat"], "ner"),
        ];
        let decomp = analyzer
            .analyze_violation(&[0, 1, 2], &orig_trace, &trans_trace)
            .unwrap();
        assert_eq!(decomp.effects.len(), 3);
    }

    #[test]
    fn test_causal_chain_analysis() {
        let mut analyzer = make_analyzer();
        let orig_trace = vec![
            make_ir_with_stage(&["hello"], "tok"),
            make_ir_with_stage(&["hello"], "pos"),
        ];
        let trans_trace = vec![
            make_ir_with_stage(&["goodbye"], "tok"),
            make_ir_with_stage(&["goodbye"], "pos"),
        ];
        let chain = analyzer
            .causal_chain_analysis(&orig_trace, &trans_trace)
            .unwrap();
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_causal_chain_empty() {
        let chain = CausalChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.total_effect(), 0.0);
        assert!(chain.root_cause().is_none());
    }

    #[test]
    fn test_fault_type_display() {
        assert_eq!(
            format!("{}", FaultType::Introduction),
            "directly introduces fault"
        );
        assert_eq!(
            format!("{}", FaultType::Amplification),
            "amplifies upstream fault"
        );
    }

    #[test]
    fn test_causal_effect_new() {
        let effect = CausalEffect::new(0, StageId::new("tok"), 0.0, 0.0);
        assert_eq!(effect.effect_type, FaultType::None);
        assert_eq!(effect.total_effect, 0.0);
    }

    #[test]
    fn test_decomposition_max_effect() {
        let effects = vec![
            CausalEffect::new(0, StageId::new("s0"), 0.1, 0.1),
            CausalEffect::new(1, StageId::new("s1"), 0.5, 0.4),
        ];
        let decomp = CausalDecomposition::new(effects);
        let max_e = decomp.max_effect_stage().unwrap();
        assert_eq!(max_e.stage_index, 1);
    }
}
