//! Equivalent mutant detection.
//!
//! Equivalent mutants are semantically identical to the original program and
//! thus cannot be detected by any test. Detecting them is important because
//! they inflate the denominator of the mutation score.
//!
//! Three strategies:
//! 1. **Trivial Compiler Equivalence (TCE)** - syntactic patterns known to
//!    produce equivalent mutants.
//! 2. **Bounded symbolic equivalence** - SMT-based checking.
//! 3. **Heuristic** - mutants with identical output on all test inputs.

use crate::{
    CoverageError, Formula, KillMatrix, MutantDescriptor, MutantId, MutationOperator, Result,
    SmtSolver, SolverResult,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// EquivalenceResult
// ---------------------------------------------------------------------------

/// Result of an equivalence check for a single mutant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EquivalenceResult {
    /// Mutant is provably equivalent to the original.
    Equivalent,
    /// Mutant is not equivalent; witness input distinguishes them.
    NonEquivalent(DistinguishingInput),
    /// Could not determine equivalence.
    Unknown(String),
}

impl EquivalenceResult {
    pub fn is_equivalent(&self) -> bool {
        matches!(self, Self::Equivalent)
    }
    pub fn is_non_equivalent(&self) -> bool {
        matches!(self, Self::NonEquivalent(_))
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Get the distinguishing input, if any.
    pub fn witness(&self) -> Option<&DistinguishingInput> {
        match self {
            Self::NonEquivalent(w) => Some(w),
            _ => None,
        }
    }
}

impl fmt::Display for EquivalenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equivalent => write!(f, "EQUIVALENT"),
            Self::NonEquivalent(w) => write!(f, "NON-EQUIVALENT (witness: {})", w),
            Self::Unknown(reason) => write!(f, "UNKNOWN: {}", reason),
        }
    }
}

/// An input that distinguishes a mutant from the original program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistinguishingInput {
    /// Variable name to value mapping.
    pub values: BTreeMap<String, i64>,
    /// Original program output on this input.
    pub original_output: Option<String>,
    /// Mutant program output on this input.
    pub mutant_output: Option<String>,
}

impl DistinguishingInput {
    pub fn new(values: BTreeMap<String, i64>) -> Self {
        Self {
            values,
            original_output: None,
            mutant_output: None,
        }
    }

    pub fn with_outputs(mut self, original: impl Into<String>, mutant: impl Into<String>) -> Self {
        self.original_output = Some(original.into());
        self.mutant_output = Some(mutant.into());
        self
    }
}

impl fmt::Display for DistinguishingInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pairs: Vec<String> = self
            .values
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        write!(f, "{{{}}}", pairs.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for equivalence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceConfig {
    /// Enable Trivial Compiler Equivalence detection.
    pub use_tce: bool,
    /// Enable SMT-based equivalence checking.
    pub use_smt: bool,
    /// Enable heuristic detection from test outputs.
    pub use_heuristic: bool,
    /// Maximum SMT queries for batch checking.
    pub max_smt_queries: usize,
    /// SMT timeout in milliseconds.
    pub smt_timeout_ms: u64,
    /// Minimum number of test cases for heuristic confidence.
    pub heuristic_min_tests: usize,
}

impl Default for EquivalenceConfig {
    fn default() -> Self {
        Self {
            use_tce: true,
            use_smt: true,
            use_heuristic: true,
            max_smt_queries: 5_000,
            smt_timeout_ms: 10_000,
            heuristic_min_tests: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// TCE patterns
// ---------------------------------------------------------------------------

/// Known syntactic patterns that produce equivalent mutants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcePattern {
    pub name: String,
    pub description: String,
    pub operator: MutationOperator,
    pub original_pattern: String,
    pub replacement_pattern: String,
}

/// Built-in TCE patterns.
fn builtin_tce_patterns() -> Vec<TcePattern> {
    vec![
        TcePattern {
            name: "aor_identity_add_zero".into(),
            description: "Adding zero is identity".into(),
            operator: MutationOperator::AOR,
            original_pattern: "x + 0".into(),
            replacement_pattern: "x - 0".into(),
        },
        TcePattern {
            name: "aor_identity_mul_one".into(),
            description: "Multiplying by one is identity".into(),
            operator: MutationOperator::AOR,
            original_pattern: "x * 1".into(),
            replacement_pattern: "x / 1".into(),
        },
        TcePattern {
            name: "ror_tautology_eq_self".into(),
            description: "Variable equals itself is always true".into(),
            operator: MutationOperator::ROR,
            original_pattern: "x == x".into(),
            replacement_pattern: "x <= x".into(),
        },
        TcePattern {
            name: "ror_tautology_le_self".into(),
            description: "Variable <= itself is always true".into(),
            operator: MutationOperator::ROR,
            original_pattern: "x <= x".into(),
            replacement_pattern: "x >= x".into(),
        },
        TcePattern {
            name: "cor_idempotent_and".into(),
            description: "p && p is equivalent to p || p".into(),
            operator: MutationOperator::COR,
            original_pattern: "p && p".into(),
            replacement_pattern: "p || p".into(),
        },
        TcePattern {
            name: "uoi_double_neg".into(),
            description: "Double negation is identity".into(),
            operator: MutationOperator::UOI,
            original_pattern: "-(-x)".into(),
            replacement_pattern: "x".into(),
        },
        TcePattern {
            name: "cr_zero_mul".into(),
            description: "0 * x = 0 regardless of constant change".into(),
            operator: MutationOperator::CR,
            original_pattern: "0 * x".into(),
            replacement_pattern: "0 * x".into(),
        },
        TcePattern {
            name: "abs_positive_literal".into(),
            description: "abs of positive literal is identity".into(),
            operator: MutationOperator::ABS,
            original_pattern: "abs(positive)".into(),
            replacement_pattern: "positive".into(),
        },
        TcePattern {
            name: "sdl_unreachable".into(),
            description: "Deleting unreachable code is equivalent".into(),
            operator: MutationOperator::SDL,
            original_pattern: "if(false) { stmt }".into(),
            replacement_pattern: "/* deleted */".into(),
        },
        TcePattern {
            name: "ror_bool_literal_true".into(),
            description: "Replacing true == true with true >= true".into(),
            operator: MutationOperator::ROR,
            original_pattern: "true == true".into(),
            replacement_pattern: "true >= true".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics about equivalence detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EquivalenceStats {
    pub total_checked: usize,
    pub equivalent_count: usize,
    pub non_equivalent_count: usize,
    pub unknown_count: usize,
    pub tce_detections: usize,
    pub smt_detections: usize,
    pub heuristic_detections: usize,
    pub smt_queries: usize,
    pub smt_timeouts: usize,
    pub per_operator: BTreeMap<String, OperatorEquivalenceStats>,
}

/// Per-operator equivalence stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperatorEquivalenceStats {
    pub checked: usize,
    pub equivalent: usize,
    pub equivalence_rate: f64,
}

impl EquivalenceStats {
    pub fn equivalence_rate(&self) -> f64 {
        if self.total_checked == 0 {
            0.0
        } else {
            self.equivalent_count as f64 / self.total_checked as f64
        }
    }

    pub fn determination_rate(&self) -> f64 {
        if self.total_checked == 0 {
            0.0
        } else {
            (self.equivalent_count + self.non_equivalent_count) as f64 / self.total_checked as f64
        }
    }
}

// ---------------------------------------------------------------------------
// EquivalenceDetector
// ---------------------------------------------------------------------------

/// Detects equivalent mutants using TCE, SMT, and heuristic methods.
pub struct EquivalenceDetector {
    config: EquivalenceConfig,
    solver: Option<Box<dyn SmtSolver>>,
    /// Original program semantics formula.
    original_formula: Option<Formula>,
    /// Mutant semantics formulas.
    mutant_formulas: HashMap<MutantId, Formula>,
    /// Mutant descriptors.
    descriptors: HashMap<MutantId, MutantDescriptor>,
    /// TCE patterns.
    tce_patterns: Vec<TcePattern>,
}

impl EquivalenceDetector {
    pub fn new() -> Self {
        Self {
            config: EquivalenceConfig::default(),
            solver: None,
            original_formula: None,
            mutant_formulas: HashMap::new(),
            descriptors: HashMap::new(),
            tce_patterns: builtin_tce_patterns(),
        }
    }

    pub fn with_config(config: EquivalenceConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    pub fn set_solver(&mut self, solver: Box<dyn SmtSolver>) {
        self.solver = Some(solver);
    }

    pub fn set_original_formula(&mut self, formula: Formula) {
        self.original_formula = Some(formula);
    }

    pub fn register_mutant_formula(&mut self, id: MutantId, formula: Formula) {
        self.mutant_formulas.insert(id, formula);
    }

    pub fn register_descriptor(&mut self, desc: MutantDescriptor) {
        self.descriptors.insert(desc.id.clone(), desc);
    }

    pub fn register_descriptors(&mut self, descs: Vec<MutantDescriptor>) {
        for d in descs {
            self.register_descriptor(d);
        }
    }

    pub fn add_tce_pattern(&mut self, pattern: TcePattern) {
        self.tce_patterns.push(pattern);
    }

    // -- Single mutant check -----------------------------------------------

    /// Check a single mutant for equivalence using all configured strategies.
    pub fn check(&self, id: &MutantId) -> EquivalenceResult {
        // 1. TCE
        if self.config.use_tce {
            if let Some(result) = self.check_tce(id) {
                return result;
            }
        }
        // 2. SMT
        if self.config.use_smt {
            if let Some(result) = self.check_smt(id) {
                return result;
            }
        }
        EquivalenceResult::Unknown("no strategy could determine equivalence".to_string())
    }

    /// Check using TCE patterns.
    fn check_tce(&self, id: &MutantId) -> Option<EquivalenceResult> {
        let desc = self.descriptors.get(id)?;
        for pattern in &self.tce_patterns {
            if pattern.operator == desc.operator
                && Self::matches_pattern(&desc.original, &pattern.original_pattern)
                && Self::matches_pattern(&desc.replacement, &pattern.replacement_pattern)
            {
                return Some(EquivalenceResult::Equivalent);
            }
        }
        None
    }

    fn matches_pattern(text: &str, pattern: &str) -> bool {
        // Simplified pattern matching: check if text contains the pattern structure.
        // A real implementation would use AST matching.
        let t = text.trim().to_lowercase();
        let p = pattern.trim().to_lowercase();
        t == p || t.contains(&p)
    }

    /// Check using SMT solver.
    fn check_smt(&self, id: &MutantId) -> Option<EquivalenceResult> {
        let solver = self.solver.as_ref()?;
        let orig = self.original_formula.as_ref()?;
        let mutant = self.mutant_formulas.get(id)?;

        // Equivalent iff (orig XOR mutant) is UNSAT.
        let result = solver.check_equivalent(orig, mutant);
        match result {
            SolverResult::Unsat => Some(EquivalenceResult::Equivalent),
            SolverResult::Sat(model) => {
                let witness = DistinguishingInput::new(model);
                Some(EquivalenceResult::NonEquivalent(witness))
            }
            SolverResult::Unknown(reason) => Some(EquivalenceResult::Unknown(reason)),
        }
    }

    // -- Heuristic detection -----------------------------------------------

    /// Heuristic: mutants that survive all tests are candidate equivalents.
    pub fn detect_heuristic(&self, km: &KillMatrix) -> Vec<MutantId> {
        if km.num_tests() < self.config.heuristic_min_tests {
            return Vec::new();
        }
        km.surviving_set()
            .iter()
            .map(|&i| km.mutants[i].clone())
            .collect()
    }

    // -- Batch checking ----------------------------------------------------

    /// Check multiple mutants for equivalence.
    pub fn check_batch(&self, ids: &[MutantId]) -> Vec<(MutantId, EquivalenceResult)> {
        ids.iter().map(|id| (id.clone(), self.check(id))).collect()
    }

    /// Check all surviving mutants from a kill matrix.
    pub fn check_survivors(&self, km: &KillMatrix) -> Vec<(MutantId, EquivalenceResult)> {
        let survivors: Vec<MutantId> = km
            .surviving_set()
            .iter()
            .map(|&i| km.mutants[i].clone())
            .collect();
        self.check_batch(&survivors)
    }

    // -- Full analysis -----------------------------------------------------

    /// Run full equivalence analysis and return statistics.
    pub fn analyze(
        &self,
        km: &KillMatrix,
    ) -> (Vec<(MutantId, EquivalenceResult)>, EquivalenceStats) {
        let results = self.check_survivors(km);
        let mut stats = EquivalenceStats::default();
        stats.total_checked = results.len();

        for (id, result) in &results {
            match result {
                EquivalenceResult::Equivalent => {
                    stats.equivalent_count += 1;
                    // Attribute to detection method (simplified).
                    if self.config.use_tce && self.check_tce(id).is_some() {
                        stats.tce_detections += 1;
                    } else if self.config.use_smt {
                        stats.smt_detections += 1;
                    }
                }
                EquivalenceResult::NonEquivalent(_) => {
                    stats.non_equivalent_count += 1;
                }
                EquivalenceResult::Unknown(_) => {
                    stats.unknown_count += 1;
                }
            }

            // Per-operator stats.
            if let Some(desc) = self.descriptors.get(id) {
                let op_name = desc.operator.short_name().to_string();
                let entry = stats.per_operator.entry(op_name).or_default();
                entry.checked += 1;
                if result.is_equivalent() {
                    entry.equivalent += 1;
                }
            }
        }

        // Compute per-operator rates.
        for s in stats.per_operator.values_mut() {
            s.equivalence_rate = if s.checked == 0 {
                0.0
            } else {
                s.equivalent as f64 / s.checked as f64
            };
        }

        (results, stats)
    }

    // -- Distinguishing input generation -----------------------------------

    /// Generate a distinguishing input for a non-equivalent mutant.
    pub fn generate_distinguishing_input(&self, id: &MutantId) -> Option<DistinguishingInput> {
        let solver = self.solver.as_ref()?;
        let orig = self.original_formula.as_ref()?;
        let mutant = self.mutant_formulas.get(id)?;

        let xor = Formula::or(vec![
            Formula::and(vec![orig.clone(), Formula::not(mutant.clone())]),
            Formula::and(vec![Formula::not(orig.clone()), mutant.clone()]),
        ]);

        match solver.check_sat(&xor) {
            SolverResult::Sat(model) => Some(DistinguishingInput::new(model)),
            _ => None,
        }
    }

    /// Batch-generate distinguishing inputs.
    pub fn generate_distinguishing_inputs(
        &self,
        ids: &[MutantId],
    ) -> Vec<(MutantId, Option<DistinguishingInput>)> {
        ids.iter()
            .map(|id| (id.clone(), self.generate_distinguishing_input(id)))
            .collect()
    }

    // -- Equivalence class merging -----------------------------------------

    /// Given equivalence results, return the set of equivalent mutant IDs.
    pub fn equivalent_set(results: &[(MutantId, EquivalenceResult)]) -> BTreeSet<MutantId> {
        results
            .iter()
            .filter(|(_, r)| r.is_equivalent())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Given equivalence results, return the set of non-equivalent mutant IDs.
    pub fn non_equivalent_set(results: &[(MutantId, EquivalenceResult)]) -> BTreeSet<MutantId> {
        results
            .iter()
            .filter(|(_, r)| r.is_non_equivalent())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Return mutant indices that are equivalent.
    pub fn equivalent_indices(
        results: &[(MutantId, EquivalenceResult)],
        km: &KillMatrix,
    ) -> BTreeSet<usize> {
        let eq_ids = Self::equivalent_set(results);
        km.mutants
            .iter()
            .enumerate()
            .filter(|(_, id)| eq_ids.contains(id))
            .map(|(i, _)| i)
            .collect()
    }
}

impl Default for EquivalenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        make_test_kill_matrix, make_test_mutant, MutantId, MutationOperator, TrivialSolver,
        TrivialSolverMode,
    };

    fn setup_detector() -> EquivalenceDetector {
        let mut det = EquivalenceDetector::new();
        // Register some descriptors.
        for i in 0..5 {
            let id = MutantId::new(format!("m{}", i));
            let op = match i % 3 {
                0 => MutationOperator::AOR,
                1 => MutationOperator::ROR,
                _ => MutationOperator::COR,
            };
            det.register_descriptor(make_test_mutant(id.as_str(), op));
        }
        det
    }

    #[test]
    fn test_equivalence_result_display() {
        let eq = EquivalenceResult::Equivalent;
        assert_eq!(format!("{}", eq), "EQUIVALENT");

        let neq = EquivalenceResult::NonEquivalent(DistinguishingInput::new(BTreeMap::from([(
            "x".into(),
            42,
        )])));
        let s = format!("{}", neq);
        assert!(s.contains("NON-EQUIVALENT"));
        assert!(s.contains("x=42"));
    }

    #[test]
    fn test_equivalence_result_methods() {
        assert!(EquivalenceResult::Equivalent.is_equivalent());
        let neq = EquivalenceResult::NonEquivalent(DistinguishingInput::new(BTreeMap::new()));
        assert!(neq.is_non_equivalent());
        assert!(neq.witness().is_some());
        assert!(EquivalenceResult::Unknown("test".into()).is_unknown());
    }

    #[test]
    fn test_tce_basic() {
        let mut det = EquivalenceDetector::new();
        let desc = MutantDescriptor::new(
            MutantId::new("m_tce"),
            MutationOperator::AOR,
            crate::MutationSite::new("test.c", "f", 1, 1),
            "x + 0",
            "x - 0",
        );
        det.register_descriptor(desc);
        let result = det.check(&MutantId::new("m_tce"));
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_tce_no_match() {
        let mut det = EquivalenceDetector::new();
        let desc = MutantDescriptor::new(
            MutantId::new("m_no"),
            MutationOperator::AOR,
            crate::MutationSite::new("test.c", "f", 1, 1),
            "x + y",
            "x - y",
        );
        det.register_descriptor(desc);
        let result = det.check(&MutantId::new("m_no"));
        // Should be Unknown since no TCE match and no SMT solver.
        assert!(result.is_unknown());
    }

    #[test]
    fn test_smt_equivalent() {
        let mut det = EquivalenceDetector::new();
        det.set_solver(Box::new(TrivialSolver {
            default_result: TrivialSolverMode::AlwaysUnsat,
        }));
        det.set_original_formula(Formula::True);
        det.register_mutant_formula(MutantId::new("m0"), Formula::True);
        det.register_descriptor(make_test_mutant("m0", MutationOperator::AOR));

        let result = det.check_smt(&MutantId::new("m0")).unwrap();
        assert!(result.is_equivalent());
    }

    #[test]
    fn test_smt_non_equivalent() {
        let mut det = EquivalenceDetector::new();
        det.set_solver(Box::new(TrivialSolver {
            default_result: TrivialSolverMode::AlwaysSat,
        }));
        det.set_original_formula(Formula::True);
        det.register_mutant_formula(MutantId::new("m0"), Formula::False);
        det.register_descriptor(make_test_mutant("m0", MutationOperator::AOR));

        let result = det.check_smt(&MutantId::new("m0")).unwrap();
        assert!(result.is_non_equivalent());
    }

    #[test]
    fn test_heuristic_detection() {
        let km = make_test_kill_matrix(15, 3, &[(0, 0), (1, 0)]);
        let det = EquivalenceDetector::new();
        let candidates = det.detect_heuristic(&km);
        // m1 and m2 survive all tests.
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_heuristic_min_tests() {
        let km = make_test_kill_matrix(2, 3, &[(0, 0)]);
        let config = EquivalenceConfig {
            heuristic_min_tests: 10,
            ..Default::default()
        };
        let det = EquivalenceDetector::with_config(config);
        let candidates = det.detect_heuristic(&km);
        assert!(candidates.is_empty()); // Too few tests.
    }

    #[test]
    fn test_batch_check() {
        let det = setup_detector();
        let ids = vec![MutantId::new("m0"), MutantId::new("m1")];
        let results = det.check_batch(&ids);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_check_survivors() {
        let km = make_test_kill_matrix(3, 4, &[(0, 0), (1, 1)]);
        let det = setup_detector();
        let results = det.check_survivors(&km);
        assert_eq!(results.len(), 2); // m2, m3 survive.
    }

    #[test]
    fn test_full_analyze() {
        let km = make_test_kill_matrix(3, 5, &[(0, 0), (1, 1), (2, 2)]);
        let det = setup_detector();
        let (results, stats) = det.analyze(&km);
        assert_eq!(stats.total_checked, 2); // m3, m4 survive.
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_equivalent_set() {
        let results = vec![
            (MutantId::new("m0"), EquivalenceResult::Equivalent),
            (
                MutantId::new("m1"),
                EquivalenceResult::NonEquivalent(DistinguishingInput::new(BTreeMap::new())),
            ),
            (MutantId::new("m2"), EquivalenceResult::Equivalent),
        ];
        let eq = EquivalenceDetector::equivalent_set(&results);
        assert_eq!(eq.len(), 2);
        assert!(eq.contains(&MutantId::new("m0")));
        assert!(eq.contains(&MutantId::new("m2")));
    }

    #[test]
    fn test_non_equivalent_set() {
        let results = vec![
            (MutantId::new("m0"), EquivalenceResult::Equivalent),
            (
                MutantId::new("m1"),
                EquivalenceResult::NonEquivalent(DistinguishingInput::new(BTreeMap::new())),
            ),
        ];
        let neq = EquivalenceDetector::non_equivalent_set(&results);
        assert_eq!(neq.len(), 1);
        assert!(neq.contains(&MutantId::new("m1")));
    }

    #[test]
    fn test_equivalent_indices() {
        let km = make_test_kill_matrix(2, 3, &[(0, 0)]);
        let results = vec![
            (MutantId::new("m1"), EquivalenceResult::Equivalent),
            (
                MutantId::new("m2"),
                EquivalenceResult::Unknown("test".into()),
            ),
        ];
        let indices = EquivalenceDetector::equivalent_indices(&results, &km);
        assert_eq!(indices, BTreeSet::from([1]));
    }

    #[test]
    fn test_distinguishing_input_display() {
        let di = DistinguishingInput::new(BTreeMap::from([("x".into(), 1), ("y".into(), 2)]));
        let s = format!("{}", di);
        assert!(s.contains("x=1"));
        assert!(s.contains("y=2"));
    }

    #[test]
    fn test_generate_distinguishing_input() {
        let mut det = EquivalenceDetector::new();
        det.set_solver(Box::new(TrivialSolver {
            default_result: TrivialSolverMode::AlwaysSat,
        }));
        det.set_original_formula(Formula::True);
        det.register_mutant_formula(MutantId::new("m0"), Formula::False);
        let input = det.generate_distinguishing_input(&MutantId::new("m0"));
        assert!(input.is_some());
    }

    #[test]
    fn test_stats_rates() {
        let stats = EquivalenceStats {
            total_checked: 10,
            equivalent_count: 3,
            non_equivalent_count: 5,
            unknown_count: 2,
            ..Default::default()
        };
        assert!((stats.equivalence_rate() - 0.3).abs() < 1e-9);
        assert!((stats.determination_rate() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_config_defaults() {
        let config = EquivalenceConfig::default();
        assert!(config.use_tce);
        assert!(config.use_smt);
        assert!(config.use_heuristic);
    }

    #[test]
    fn test_custom_tce_pattern() {
        let mut det = EquivalenceDetector::new();
        det.add_tce_pattern(TcePattern {
            name: "custom".into(),
            description: "Custom pattern".into(),
            operator: MutationOperator::AOR,
            original_pattern: "custom_orig".into(),
            replacement_pattern: "custom_repl".into(),
        });
        let desc = MutantDescriptor::new(
            MutantId::new("mc"),
            MutationOperator::AOR,
            crate::MutationSite::new("t.c", "f", 1, 1),
            "custom_orig",
            "custom_repl",
        );
        det.register_descriptor(desc);
        assert!(det.check(&MutantId::new("mc")).is_equivalent());
    }

    #[test]
    fn test_builtin_patterns() {
        let patterns = builtin_tce_patterns();
        assert!(patterns.len() >= 5);
        for p in &patterns {
            assert!(!p.name.is_empty());
        }
    }

    #[test]
    fn test_with_outputs() {
        let di =
            DistinguishingInput::new(BTreeMap::from([("x".into(), 5)])).with_outputs("10", "15");
        assert_eq!(di.original_output.unwrap(), "10");
        assert_eq!(di.mutant_output.unwrap(), "15");
    }

    #[test]
    fn test_batch_distinguishing() {
        let mut det = EquivalenceDetector::new();
        det.set_solver(Box::new(TrivialSolver {
            default_result: TrivialSolverMode::AlwaysSat,
        }));
        det.set_original_formula(Formula::True);
        det.register_mutant_formula(MutantId::new("m0"), Formula::False);
        det.register_mutant_formula(MutantId::new("m1"), Formula::False);

        let results = det.generate_distinguishing_inputs(&[
            MutantId::new("m0"),
            MutantId::new("m1"),
            MutantId::new("m_unknown"),
        ]);
        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_some());
        assert!(results[1].1.is_some());
        assert!(results[2].1.is_none());
    }
}
