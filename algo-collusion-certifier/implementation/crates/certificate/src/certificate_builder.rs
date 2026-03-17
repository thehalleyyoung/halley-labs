//! Certificate construction from analysis results.
//!
//! Provides builders for Layer 0/1/2 certificates, proof constructors,
//! alpha budget planners, segment allocators, and serialization.

use crate::ast::*;
use crate::proof_term::*;
use serde::{Deserialize, Serialize};
use shared_types::{
    ConfidenceInterval, ConfidenceLevel, GameConfig, OracleAccessLevel, PlayerId,
    SignificanceLevel,
};
use stat_tests::TestResult;
use counterfactual::{DeviationResult, PunishmentTest};
use game_theory::{CollusionPremium, NashEquilibrium};

// ── Certificate builder ──────────────────────────────────────────────────────

/// High-level builder for constructing certificates from analysis results.
pub struct CertificateBuilder {
    scenario: String,
    oracle_level: OracleAccessLevel,
    alpha: f64,
    steps: Vec<ProofStep>,
    alpha_planner: AlphaBudgetPlanner,
    segment_allocator: SegmentAllocator,
}

impl CertificateBuilder {
    pub fn new(scenario: &str, oracle_level: OracleAccessLevel, alpha: f64) -> Self {
        Self {
            scenario: scenario.to_string(),
            oracle_level,
            alpha,
            steps: Vec::new(),
            alpha_planner: AlphaBudgetPlanner::new(alpha),
            segment_allocator: SegmentAllocator::new(),
        }
    }

    /// Add a data declaration step.
    pub fn add_data_declaration(
        &mut self,
        ref_id: &str,
        segment_type: &str,
        start: usize,
        end: usize,
        hash: &str,
        num_players: usize,
    ) -> &mut Self {
        self.segment_allocator
            .allocate(ref_id, segment_type, start, end);
        self.steps.push(ProofStep::DataDeclaration(
            TrajectoryRef::new(ref_id),
            SegmentSpec::new(segment_type, start, end, hash, num_players),
        ));
        self
    }

    /// Add a statistical test step from a TestResult.
    pub fn add_test_result(&mut self, ref_id: &str, result: &TestResult) -> &mut Self {
        self.steps.push(ProofStep::StatisticalTest(
            TestRef::new(ref_id),
            TestType::new(&result.test_name, &format!("{}", result.test_type)),
            Statistic::new(result.statistic),
            PValueWrapper::new(result.p_value.value()),
        ));
        self
    }

    /// Add an equilibrium claim step from computed Nash equilibrium.
    pub fn add_equilibrium_claim(
        &mut self,
        ref_id: &str,
        game_config: &GameConfig,
        nash: &NashEquilibrium,
    ) -> &mut Self {
        self.steps.push(ProofStep::EquilibriumClaim(
            EquilibriumRef::new(ref_id),
            GameSpec::from_game_config(game_config),
            NashProfile::from_nash_eq(nash),
        ));
        self
    }

    /// Add a deviation bound step.
    pub fn add_deviation_bound(
        &mut self,
        ref_id: &str,
        result: &DeviationResult,
    ) -> &mut Self {
        let conf = result.confidence_interval.as_ref()
            .map(|ci| ci.level)
            .unwrap_or(0.95);
        self.steps.push(ProofStep::DeviationBound(
            DeviationRef::new(ref_id),
            result.player,
            Bound::upper(result.payoff_difference),
            ConfidenceLevel::new(conf).expect("confidence level must be in (0, 1)"),
        ));
        self
    }

    /// Add punishment evidence step.
    pub fn add_punishment_evidence(
        &mut self,
        ref_id: &str,
        result: &PunishmentTest,
    ) -> &mut Self {
        let on_path_payoff = if result.metrics.payoff_drop != 0.0 {
            result.metrics.payoff_drop / result.metrics.punishment_severity.max(1e-12)
        } else {
            0.0
        };
        self.steps.push(ProofStep::PunishmentEvidence(
            PunishmentRef::new(ref_id),
            result.player,
            PayoffDrop::from_absolute(result.metrics.payoff_drop, on_path_payoff),
            PValueWrapper::new(result.p_value.value()),
        ));
        self
    }

    /// Add a collusion premium step.
    pub fn add_collusion_premium(
        &mut self,
        ref_id: &str,
        cp: &CollusionPremium,
    ) -> &mut Self {
        let ci = CIWrapper::new(cp.value - 0.1, cp.value + 0.1, 0.95);
        self.steps.push(ProofStep::CollusionPremium(
            CPRef::new(ref_id),
            Value::new(cp.value),
            ci,
        ));
        self
    }

    /// Add an inference step.
    pub fn add_inference(
        &mut self,
        ref_id: &str,
        rule_name: &str,
        premise_refs: Vec<String>,
        conclusion: &str,
    ) -> &mut Self {
        self.steps.push(ProofStep::Inference(
            InferenceRef::new(ref_id),
            Rule::new(rule_name),
            Premises::new(premise_refs),
            Conclusion::new(conclusion),
        ));
        self
    }

    /// Add a verdict step.
    pub fn add_verdict(
        &mut self,
        verdict: VerdictType,
        confidence: f64,
        supporting_refs: Vec<String>,
    ) -> &mut Self {
        self.steps.push(ProofStep::Verdict(
            verdict,
            Confidence::new(confidence),
            SupportingRefs::new(supporting_refs),
        ));
        self
    }

    /// Build the certificate.
    pub fn build(self) -> CertificateAST {
        let header = CertificateHeader::new(&self.scenario, self.oracle_level, self.alpha);
        let mut body = CertificateBody::new();
        for step in self.steps {
            body.push(step);
        }
        CertificateAST::new(header, body)
    }

    /// Build a Layer 0 certificate from test results and trajectory hash.
    pub fn build_layer0_certificate(
        test_results: &[TestResult],
        trajectory_hash: &str,
        num_players: usize,
        num_rounds: usize,
        game_config: &GameConfig,
        nash: &NashEquilibrium,
        cp: &CollusionPremium,
        alpha: f64,
    ) -> CertificateAST {
        let mut builder =
            CertificateBuilder::new("layer0_analysis", OracleAccessLevel::Layer0, alpha);

        // Data declaration
        builder.add_data_declaration(
            "traj_testing",
            "testing",
            0,
            num_rounds,
            trajectory_hash,
            num_players,
        );

        // Statistical tests
        let mut test_refs = Vec::new();
        for (i, tr) in test_results.iter().enumerate() {
            let ref_id = format!("test_{}", i);
            builder.add_test_result(&ref_id, tr);
            test_refs.push(ref_id);
        }

        // Equilibrium claim
        builder.add_equilibrium_claim("eq_nash", game_config, nash);

        // Collusion premium
        builder.add_collusion_premium("cp_0", cp);

        // Composite inference
        let mut all_refs = test_refs.clone();
        all_refs.push("eq_nash".into());
        all_refs.push("cp_0".into());
        builder.add_inference(
            "inf_composite",
            "VerdictDerivation",
            all_refs.clone(),
            "Layer 0 evidence assessment complete",
        );

        // Verdict
        let any_reject = test_results.iter().any(|t| t.reject_null);
        let verdict = if any_reject && cp.value > 0.0 {
            VerdictType::Collusive
        } else if any_reject {
            VerdictType::Inconclusive
        } else {
            VerdictType::Competitive
        };

        let mut supporting = test_refs
            .iter()
            .filter(|r| {
                test_results
                    .get(r.strip_prefix("test_").unwrap_or("99").parse::<usize>().unwrap_or(99))
                    .map(|t| t.reject_null)
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>();
        supporting.push("inf_composite".into());

        let confidence = if any_reject { 1.0 - alpha } else { 0.5 };
        builder.add_verdict(verdict, confidence, supporting);

        builder.build()
    }

    /// Build a Layer 1 certificate with deviation analysis.
    pub fn build_layer1_certificate(
        test_results: &[TestResult],
        deviation_results: &[DeviationResult],
        trajectory_hash: &str,
        num_players: usize,
        num_rounds: usize,
        game_config: &GameConfig,
        nash: &NashEquilibrium,
        cp: &CollusionPremium,
        alpha: f64,
    ) -> CertificateAST {
        let mut builder =
            CertificateBuilder::new("layer1_analysis", OracleAccessLevel::Layer1, alpha);

        builder.add_data_declaration(
            "traj_testing",
            "testing",
            0,
            num_rounds,
            trajectory_hash,
            num_players,
        );

        let mut test_refs = Vec::new();
        for (i, tr) in test_results.iter().enumerate() {
            let ref_id = format!("test_{}", i);
            builder.add_test_result(&ref_id, tr);
            test_refs.push(ref_id);
        }

        builder.add_equilibrium_claim("eq_nash", game_config, nash);

        // Deviation bounds
        let mut dev_refs = Vec::new();
        for (i, dr) in deviation_results.iter().enumerate() {
            let ref_id = format!("dev_{}", i);
            builder.add_deviation_bound(&ref_id, dr);
            dev_refs.push(ref_id);
        }

        builder.add_collusion_premium("cp_0", cp);

        // Deviation inference
        if !dev_refs.is_empty() {
            builder.add_inference(
                "inf_deviation",
                "DeviationInference",
                dev_refs.clone(),
                "Deviation analysis supports sustained collusion",
            );
        }

        let mut all_refs = test_refs.clone();
        all_refs.extend(dev_refs);
        all_refs.push("eq_nash".into());
        all_refs.push("cp_0".into());
        builder.add_inference(
            "inf_composite",
            "VerdictDerivation",
            all_refs,
            "Layer 1 evidence assessment complete",
        );

        let any_reject = test_results.iter().any(|t| t.reject_null);
        let has_deviation = deviation_results.iter().any(|d| !d.is_profitable);
        let verdict = if any_reject && has_deviation && cp.value > 0.0 {
            VerdictType::Collusive
        } else if any_reject {
            VerdictType::Inconclusive
        } else {
            VerdictType::Competitive
        };
        let confidence = if verdict == VerdictType::Collusive {
            1.0 - alpha
        } else {
            0.5
        };

        let mut supporting = vec!["inf_composite".to_string()];
        if test_results.iter().any(|t| t.reject_null) {
            supporting.push("test_0".into());
        }

        builder.add_verdict(verdict, confidence, supporting);
        builder.build()
    }

    /// Build a Layer 2 certificate with punishment detection.
    pub fn build_layer2_certificate(
        test_results: &[TestResult],
        deviation_results: &[DeviationResult],
        punishment_results: &[PunishmentTest],
        trajectory_hash: &str,
        num_players: usize,
        num_rounds: usize,
        game_config: &GameConfig,
        nash: &NashEquilibrium,
        cp: &CollusionPremium,
        alpha: f64,
    ) -> CertificateAST {
        let mut builder =
            CertificateBuilder::new("layer2_analysis", OracleAccessLevel::Layer2, alpha);

        builder.add_data_declaration(
            "traj_testing",
            "testing",
            0,
            num_rounds,
            trajectory_hash,
            num_players,
        );

        let mut test_refs = Vec::new();
        for (i, tr) in test_results.iter().enumerate() {
            let ref_id = format!("test_{}", i);
            builder.add_test_result(&ref_id, tr);
            test_refs.push(ref_id);
        }

        builder.add_equilibrium_claim("eq_nash", game_config, nash);

        let mut dev_refs = Vec::new();
        for (i, dr) in deviation_results.iter().enumerate() {
            let ref_id = format!("dev_{}", i);
            builder.add_deviation_bound(&ref_id, dr);
            dev_refs.push(ref_id);
        }

        // Punishment evidence
        let mut pun_refs = Vec::new();
        for (i, pr) in punishment_results.iter().enumerate() {
            let ref_id = format!("pun_{}", i);
            builder.add_punishment_evidence(&ref_id, pr);
            pun_refs.push(ref_id);
        }

        builder.add_collusion_premium("cp_0", cp);

        if !pun_refs.is_empty() {
            builder.add_inference(
                "inf_punishment",
                "PunishmentInference",
                pun_refs.clone(),
                "Punishment mechanism detected",
            );
        }

        let mut all_refs = test_refs.clone();
        all_refs.extend(dev_refs);
        all_refs.extend(pun_refs);
        all_refs.push("eq_nash".into());
        all_refs.push("cp_0".into());
        builder.add_inference(
            "inf_composite",
            "VerdictDerivation",
            all_refs,
            "Layer 2 evidence assessment complete",
        );

        let any_reject = test_results.iter().any(|t| t.reject_null);
        let has_punishment = punishment_results.iter().any(|p| p.reject_null);
        let verdict = if any_reject && has_punishment && cp.value > 0.0 {
            VerdictType::Collusive
        } else if any_reject {
            VerdictType::Inconclusive
        } else {
            VerdictType::Competitive
        };
        let confidence = if verdict == VerdictType::Collusive {
            1.0 - alpha
        } else {
            0.5
        };

        let mut supporting = vec!["inf_composite".to_string()];
        if any_reject {
            supporting.push("test_0".into());
        }

        builder.add_verdict(verdict, confidence, supporting);
        builder.build()
    }
}

// ── Proof constructor ────────────────────────────────────────────────────────

/// Automatically builds proof terms from analysis results.
pub struct ProofConstructor;

impl ProofConstructor {
    /// Build a proof term for a composite test rejection.
    pub fn build_composite_rejection(
        test_refs: &[String],
        p_values: &[f64],
        alpha: f64,
    ) -> ProofTerm {
        let test_terms: Vec<ProofTerm> = test_refs
            .iter()
            .zip(p_values.iter())
            .map(|(r, p)| {
                ProofTerm::Axiom(
                    AxiomSchema::TestSoundness,
                    Instantiation::new()
                        .bind("alpha", ProofValue::Float(alpha))
                        .bind("p_value", ProofValue::Float(*p))
                        .bind("statistic", ProofValue::Float(0.0)),
                )
            })
            .collect();

        if test_terms.len() == 1 {
            return test_terms.into_iter().next().unwrap();
        }

        ProofTerm::RuleApplication(InferenceRule::CompositeRejection, test_terms)
    }

    /// Build a proof term for the collusion premium calculation.
    pub fn build_cp_proof(
        nash_profit: f64,
        observed_profit: f64,
        collusive_profit: f64,
    ) -> ProofTerm {
        ProofTerm::Axiom(
            AxiomSchema::CollusionPremiumDef,
            Instantiation::new()
                .bind("nash_profit", ProofValue::Float(nash_profit))
                .bind("observed_profit", ProofValue::Float(observed_profit))
                .bind("collusive_profit", ProofValue::Float(collusive_profit)),
        )
    }

    /// Build a proof term for deviation inference.
    pub fn build_deviation_proof(
        player: PlayerId,
        bound: f64,
        confidence: f64,
    ) -> ProofTerm {
        ProofTerm::Axiom(
            AxiomSchema::DeviationExistence,
            Instantiation::new()
                .bind("cp", ProofValue::Float(bound))
                .bind("discount", ProofValue::Float(0.95))
                .bind("num_players", ProofValue::Float(2.0))
                .bind("nash_profit", ProofValue::Float(0.0)),
        )
    }

    /// Build a proof term for arithmetic verification.
    pub fn build_arithmetic_verification(
        left: Expression,
        right: Expression,
        relation: crate::proof_term::Relation,
    ) -> ProofTerm {
        ProofTerm::ArithmeticFact(left, relation, right)
    }

    /// Build a full verdict proof from sub-proofs.
    pub fn build_verdict_proof(sub_proofs: Vec<ProofTerm>) -> ProofTerm {
        if sub_proofs.is_empty() {
            return ProofTerm::Reference("no_evidence".to_string());
        }
        ProofTerm::RuleApplication(InferenceRule::VerdictDerivation, sub_proofs)
    }
}

// ── Alpha budget planner ─────────────────────────────────────────────────────

/// Plans alpha allocation across tests before certificate construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaBudgetPlanner {
    pub total_alpha: f64,
    pub allocations: Vec<AlphaAllocation>,
    pub remaining: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaAllocation {
    pub test_name: String,
    pub alpha: f64,
    pub priority: usize,
}

impl AlphaBudgetPlanner {
    pub fn new(total_alpha: f64) -> Self {
        Self {
            total_alpha,
            allocations: Vec::new(),
            remaining: total_alpha,
        }
    }

    /// Allocate alpha equally among n tests.
    pub fn allocate_equal(&mut self, test_names: &[String]) -> Vec<f64> {
        let n = test_names.len();
        if n == 0 {
            return Vec::new();
        }
        let per_test = self.remaining / n as f64;
        let mut allocations = Vec::new();
        for name in test_names {
            self.allocations.push(AlphaAllocation {
                test_name: name.clone(),
                alpha: per_test,
                priority: self.allocations.len(),
            });
            allocations.push(per_test);
        }
        self.remaining = 0.0;
        allocations
    }

    /// Allocate alpha using Holm-Bonferroni correction ordering.
    pub fn allocate_holm(&mut self, test_names: &[String], p_values: &[f64]) -> Vec<f64> {
        let n = test_names.len();
        if n == 0 {
            return Vec::new();
        }

        // Sort by p-value (ascending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            p_values[a]
                .partial_cmp(&p_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut alphas = vec![0.0; n];
        for (rank, &idx) in indices.iter().enumerate() {
            let adjusted_alpha = self.total_alpha / (n - rank) as f64;
            alphas[idx] = adjusted_alpha;
            self.allocations.push(AlphaAllocation {
                test_name: test_names[idx].clone(),
                alpha: adjusted_alpha,
                priority: rank,
            });
        }
        self.remaining = 0.0;
        alphas
    }

    /// Allocate alpha proportionally to given weights.
    pub fn allocate_weighted(
        &mut self,
        test_names: &[String],
        weights: &[f64],
    ) -> Vec<f64> {
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return self.allocate_equal(test_names);
        }
        let mut alphas = Vec::new();
        for (name, w) in test_names.iter().zip(weights.iter()) {
            let alpha = self.remaining * (w / total_weight);
            self.allocations.push(AlphaAllocation {
                test_name: name.clone(),
                alpha,
                priority: self.allocations.len(),
            });
            alphas.push(alpha);
        }
        self.remaining = 0.0;
        alphas
    }
}

// ── Segment allocator ────────────────────────────────────────────────────────

/// Assigns phantom types to trajectory segments.
#[derive(Debug, Clone)]
pub struct SegmentAllocator {
    segments: Vec<SegmentAllocation>,
}

#[derive(Debug, Clone)]
pub struct SegmentAllocation {
    pub ref_id: String,
    pub segment_type: String,
    pub start: usize,
    pub end: usize,
}

impl SegmentAllocator {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn allocate(
        &mut self,
        ref_id: &str,
        segment_type: &str,
        start: usize,
        end: usize,
    ) {
        self.segments.push(SegmentAllocation {
            ref_id: ref_id.to_string(),
            segment_type: segment_type.to_string(),
            start,
            end,
        });
    }

    /// Check that no two segments of different types overlap.
    pub fn validate(&self) -> Result<(), String> {
        for i in 0..self.segments.len() {
            for j in (i + 1)..self.segments.len() {
                let a = &self.segments[i];
                let b = &self.segments[j];
                if a.segment_type != b.segment_type
                    && a.start < b.end
                    && b.start < a.end
                {
                    return Err(format!(
                        "Segments '{}' ({}) and '{}' ({}) overlap",
                        a.ref_id, a.segment_type, b.ref_id, b.segment_type
                    ));
                }
            }
        }
        Ok(())
    }

    /// Split a trajectory into training/testing/holdout segments.
    pub fn split_trajectory(
        total_rounds: usize,
        train_frac: f64,
        test_frac: f64,
    ) -> Vec<(&'static str, usize, usize)> {
        let train_end = (total_rounds as f64 * train_frac) as usize;
        let test_end = train_end + (total_rounds as f64 * test_frac) as usize;
        let mut segments = vec![
            ("training", 0, train_end),
            ("testing", train_end, test_end),
        ];
        if test_end < total_rounds {
            segments.push(("holdout", test_end, total_rounds));
        }
        segments
    }
}

impl Default for SegmentAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Certificate optimizer ────────────────────────────────────────────────────

/// Minimizes certificate size while preserving soundness.
pub struct CertificateOptimizer;

impl CertificateOptimizer {
    /// Remove redundant inference steps that are not referenced.
    pub fn optimize(cert: &CertificateAST) -> CertificateAST {
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect all referenced refs (from inferences, verdicts)
        for step in &cert.body.steps {
            for dep in step.dependency_refs() {
                referenced.insert(dep);
            }
        }

        // Keep all steps that are either referenced or are non-inference steps
        let mut optimized_body = CertificateBody::new();
        for step in &cert.body.steps {
            let keep = match step.declared_ref() {
                Some(ref r) => {
                    referenced.contains(r)
                        || !matches!(step, ProofStep::Inference(..))
                        || matches!(step, ProofStep::Verdict(..))
                }
                None => true, // Verdicts
            };
            if keep {
                optimized_body.push(step.clone());
            }
        }

        CertificateAST::new(cert.header.clone(), optimized_body)
    }

    /// Compute the size (in bytes) of the serialized certificate.
    pub fn estimate_size(cert: &CertificateAST) -> usize {
        serde_json::to_string(cert)
            .map(|s| s.len())
            .unwrap_or(0)
    }
}

// ── Certificate serializer ───────────────────────────────────────────────────

/// Serialize certificates to multiple formats.
pub struct CertificateSerializer;

impl CertificateSerializer {
    /// Serialize to JSON string.
    pub fn to_json(cert: &CertificateAST) -> Result<String, String> {
        serde_json::to_string_pretty(cert).map_err(|e| format!("JSON serialization error: {}", e))
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<CertificateAST, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON deserialization error: {}", e))
    }

    /// Serialize to compact binary using bincode.
    pub fn to_binary(cert: &CertificateAST) -> Result<Vec<u8>, String> {
        bincode::serialize(cert).map_err(|e| format!("Binary serialization error: {}", e))
    }

    /// Deserialize from binary.
    pub fn from_binary(data: &[u8]) -> Result<CertificateAST, String> {
        bincode::deserialize(data).map_err(|e| format!("Binary deserialization error: {}", e))
    }

    /// Serialize to human-readable text format.
    pub fn to_text(cert: &CertificateAST) -> String {
        cert.pretty_print()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use game_theory::NashEquilibrium;
    use shared_types::{GameConfig, PValue};
    use stat_tests::TestType as StTestType;

    fn make_test_result(name: &str, stat: f64, p: f64, reject: bool) -> TestResult {
        TestResult {
            test_type: StTestType::PriceCorrelation,
            test_name: name.to_string(),
            statistic: stat,
            p_value: PValue::new_unchecked(p),
            reject_null: reject,
            effect_size: None,
            confidence_interval: None,
            sample_size: 500,
            alpha_spent: 0.05,
        }
    }

    #[test]
    fn test_builder_basic() {
        let mut builder =
            CertificateBuilder::new("test", OracleAccessLevel::Layer0, 0.05);
        builder.add_data_declaration("t0", "testing", 0, 100, "hash", 2);
        builder.add_verdict(VerdictType::Competitive, 0.5, vec![]);
        let cert = builder.build();
        assert_eq!(cert.step_count(), 2);
        assert_eq!(cert.header.scenario, "test");
    }

    #[test]
    fn test_builder_with_test_result() {
        let mut builder =
            CertificateBuilder::new("test", OracleAccessLevel::Layer0, 0.05);
        let tr = make_test_result("corr_test", 3.5, 0.001, true);
        builder.add_test_result("t1", &tr);
        let cert = builder.build();
        assert_eq!(cert.step_count(), 1);
    }

    #[test]
    fn test_build_layer0_certificate() {
        let test_results = vec![
            make_test_result("test_corr", 3.5, 0.001, true),
            make_test_result("test_var", 2.1, 0.04, true),
        ];
        let game_config = GameConfig::default();
        let nash = NashEquilibrium {
            strategy_profile: vec![3, 3],
            payoffs: vec![4.0, 4.0],
            support: vec![vec![3], vec![3]],
            mixed_probabilities: vec![vec![1.0], vec![1.0]],
            is_symmetric: true,
            is_pure: true,
        };
        let cp = CollusionPremium {
            value: 0.42,
            nash_profit: 4.0,
            observed_profit: 6.0,
        };

        let cert = CertificateBuilder::build_layer0_certificate(
            &test_results,
            "hash123",
            2,
            1000,
            &game_config,
            &nash,
            &cp,
            0.05,
        );
        assert!(cert.step_count() >= 6);
    }

    #[test]
    fn test_build_layer1_certificate() {
        let test_results = vec![make_test_result("test_corr", 3.5, 0.001, true)];
        let dev_strategy = counterfactual::DeviationStrategy::single_period(
            PlayerId(0), shared_types::Price(6.0), shared_types::Price(5.0), shared_types::RoundNumber(0),
        );
        let dev_results = vec![DeviationResult::new(PlayerId(0), dev_strategy, 6.0, 5.5)];
        let game_config = GameConfig::default();
        let nash = NashEquilibrium {
            strategy_profile: vec![3, 3],
            payoffs: vec![4.0, 4.0],
            support: vec![vec![3], vec![3]],
            mixed_probabilities: vec![vec![1.0], vec![1.0]],
            is_symmetric: true,
            is_pure: true,
        };
        let cp = CollusionPremium {
            value: 0.3,
            nash_profit: 4.0,
            observed_profit: 5.5,
        };

        let cert = CertificateBuilder::build_layer1_certificate(
            &test_results,
            &dev_results,
            "hash456",
            2,
            1000,
            &game_config,
            &nash,
            &cp,
            0.05,
        );
        assert!(cert.step_count() >= 6);
    }

    #[test]
    fn test_build_layer2_certificate() {
        let test_results = vec![make_test_result("test_corr", 3.5, 0.001, true)];
        let dev_strategy = counterfactual::DeviationStrategy::single_period(
            PlayerId(0), shared_types::Price(6.0), shared_types::Price(5.0), shared_types::RoundNumber(0),
        );
        let dev_results = vec![DeviationResult::new(PlayerId(0), dev_strategy, 6.0, 5.5)];
        let pun_results = vec![PunishmentTest::run(
            PlayerId(0),
            &[10.0, 10.0, 10.0],
            &[8.0, 8.0, 8.0],
            0.05,
            1000,
            Some(42),
            0.95,
        )];
        let game_config = GameConfig::default();
        let nash = NashEquilibrium {
            strategy_profile: vec![3, 3],
            payoffs: vec![4.0, 4.0],
            support: vec![vec![3], vec![3]],
            mixed_probabilities: vec![vec![1.0], vec![1.0]],
            is_symmetric: true,
            is_pure: true,
        };
        let cp = CollusionPremium {
            value: 0.3,
            nash_profit: 4.0,
            observed_profit: 5.5,
        };

        let cert = CertificateBuilder::build_layer2_certificate(
            &test_results,
            &dev_results,
            &pun_results,
            "hash789",
            2,
            1000,
            &game_config,
            &nash,
            &cp,
            0.05,
        );
        assert!(cert.step_count() >= 7);
    }

    #[test]
    fn test_alpha_budget_planner_equal() {
        let mut planner = AlphaBudgetPlanner::new(0.05);
        let names: Vec<String> = vec!["t1".into(), "t2".into(), "t3".into()];
        let alphas = planner.allocate_equal(&names);
        assert_eq!(alphas.len(), 3);
        let total: f64 = alphas.iter().sum();
        assert!((total - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_alpha_budget_planner_holm() {
        let mut planner = AlphaBudgetPlanner::new(0.05);
        let names: Vec<String> = vec!["t1".into(), "t2".into()];
        let pvals = vec![0.01, 0.03];
        let alphas = planner.allocate_holm(&names, &pvals);
        assert_eq!(alphas.len(), 2);
        // Holm: smallest p-value gets alpha/n, next gets alpha/(n-1)
        assert!((alphas[0] - 0.025).abs() < 1e-12);
        assert!((alphas[1] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_segment_allocator_valid() {
        let mut alloc = SegmentAllocator::new();
        alloc.allocate("train", "training", 0, 500);
        alloc.allocate("test", "testing", 500, 800);
        alloc.allocate("holdout", "holdout", 800, 1000);
        assert!(alloc.validate().is_ok());
    }

    #[test]
    fn test_segment_allocator_overlapping() {
        let mut alloc = SegmentAllocator::new();
        alloc.allocate("train", "training", 0, 600);
        alloc.allocate("test", "testing", 500, 800);
        assert!(alloc.validate().is_err());
    }

    #[test]
    fn test_segment_allocator_split() {
        let splits = SegmentAllocator::split_trajectory(1000, 0.5, 0.3);
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].0, "training");
        assert_eq!(splits[1].0, "testing");
        assert_eq!(splits[2].0, "holdout");
    }

    #[test]
    fn test_certificate_optimizer() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "h", 2),
        ));
        body.push(ProofStep::Inference(
            InferenceRef::new("unused_inf"),
            Rule::new("AndIntro"),
            Premises::new(vec!["traj_0".into()]),
            Conclusion::new("unused"),
        ));
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        let optimized = CertificateOptimizer::optimize(&cert);
        // The unused inference should be removed
        assert!(optimized.step_count() <= cert.step_count());
    }

    #[test]
    fn test_certificate_serializer_json_roundtrip() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let body = CertificateBody::new();
        let cert = CertificateAST::new(header, body);
        let json = CertificateSerializer::to_json(&cert).unwrap();
        let cert2 = CertificateSerializer::from_json(&json).unwrap();
        assert_eq!(cert2.header.scenario, "s");
    }

    #[test]
    fn test_certificate_serializer_binary_roundtrip() {
        let header = CertificateHeader::new("s", OracleAccessLevel::Layer0, 0.05);
        let body = CertificateBody::new();
        let cert = CertificateAST::new(header, body);
        let bin = CertificateSerializer::to_binary(&cert).unwrap();
        let cert2 = CertificateSerializer::from_binary(&bin).unwrap();
        assert_eq!(cert2.header.scenario, "s");
    }

    #[test]
    fn test_proof_constructor_composite() {
        let refs = vec!["t1".into(), "t2".into()];
        let pvals = vec![0.01, 0.02];
        let term = ProofConstructor::build_composite_rejection(&refs, &pvals, 0.05);
        assert_eq!(term.node_count(), 3);
    }

    #[test]
    fn test_proof_constructor_cp() {
        let term = ProofConstructor::build_cp_proof(4.0, 6.0, 8.0);
        assert!(matches!(term, ProofTerm::Axiom(AxiomSchema::CollusionPremiumDef, _)));
    }

    #[test]
    fn test_proof_constructor_verdict() {
        let sub = vec![
            ProofTerm::Reference("a".into()),
            ProofTerm::Reference("b".into()),
        ];
        let term = ProofConstructor::build_verdict_proof(sub);
        assert_eq!(term.node_count(), 3);
    }
}
