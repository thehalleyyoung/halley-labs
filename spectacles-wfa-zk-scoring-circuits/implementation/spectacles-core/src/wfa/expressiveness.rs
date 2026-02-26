//! WFA expressiveness characterization and decidable membership.
//!
//! Implements the formal characterization theorem:
//!   A scoring function f: Σ* × Σ* → S is WFA-expressible if and only if
//!   f is a rational formal power series over semiring S.
//!
//! Provides a decidable membership criterion: given a metric specification,
//! determine whether it can be represented as a WFA without trial-and-error.
//!
//! # Theory
//!
//! By Schützenberger's theorem, the class of functions computable by WFA
//! equals the rational power series. A function is rational iff its Hankel
//! matrix has finite rank. This module provides:
//!
//! 1. `RationalityChecker` - Tests whether a given function is rational by
//!    computing the rank of truncated Hankel matrices.
//! 2. `MetricClassifier` - Classifies metric specifications into WFA-tiers
//!    based on operation analysis.
//! 3. `ExpressivenessTheorem` - Formal statement and constructive witness.

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// MetricOperation — primitive operations in metric specifications
// ---------------------------------------------------------------------------

/// Primitive operations that can appear in metric specifications.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricOperation {
    // Semiring operations (WFA-expressible)
    TokenMatch,
    NGramCount { n: usize },
    NGramClip { n: usize },
    Concatenation,
    Union,
    KleeneStar,
    Intersection,
    LCSAlignment,

    // Non-semiring but algebraic (gadget-expressible)
    Division,
    GeometricMean,
    HarmonicMean,
    BrevityPenalty,
    FMeasure { beta: u32 },
    Exponentiation,
    Logarithm,
    SquareRoot,

    // Non-WFA (requires external computation)
    NeuralEmbedding,
    LearnedModel,
    SynonymLookup,
    ExternalAPI,
    Sampling,
    ContinuousOptimization,
}

impl MetricOperation {
    /// Whether this operation preserves rationality.
    pub fn is_rational(&self) -> bool {
        matches!(self,
            MetricOperation::TokenMatch |
            MetricOperation::NGramCount { .. } |
            MetricOperation::NGramClip { .. } |
            MetricOperation::Concatenation |
            MetricOperation::Union |
            MetricOperation::KleeneStar |
            MetricOperation::Intersection |
            MetricOperation::LCSAlignment
        )
    }

    /// Whether this operation is expressible as a gadget
    /// (non-rational but computable from WFA outputs).
    pub fn is_gadget_expressible(&self) -> bool {
        matches!(self,
            MetricOperation::Division |
            MetricOperation::GeometricMean |
            MetricOperation::HarmonicMean |
            MetricOperation::BrevityPenalty |
            MetricOperation::FMeasure { .. } |
            MetricOperation::Exponentiation |
            MetricOperation::Logarithm |
            MetricOperation::SquareRoot
        )
    }

    /// Whether this operation requires external/neural computation.
    pub fn is_non_wfa(&self) -> bool {
        !self.is_rational() && !self.is_gadget_expressible()
    }

    /// The semiring type required for this operation.
    pub fn required_semiring(&self) -> Option<SemiringType> {
        match self {
            MetricOperation::TokenMatch => Some(SemiringType::Boolean),
            MetricOperation::NGramCount { .. } => Some(SemiringType::Counting),
            MetricOperation::NGramClip { .. } => Some(SemiringType::BoundedCounting),
            MetricOperation::LCSAlignment => Some(SemiringType::Tropical),
            MetricOperation::Intersection => Some(SemiringType::Boolean),
            MetricOperation::Concatenation | MetricOperation::Union | MetricOperation::KleeneStar => None,
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SemiringType
// ---------------------------------------------------------------------------

/// Semiring types supported by the WFA framework.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemiringType {
    Boolean,
    Counting,
    BoundedCounting,
    Tropical,
    Real,
    LogDomain,
    Viterbi,
    Goldilocks,
}

impl SemiringType {
    /// Whether this semiring embeds into the Goldilocks field F_p.
    pub fn is_field_embeddable(&self) -> bool {
        matches!(self,
            SemiringType::Boolean |
            SemiringType::Counting |
            SemiringType::BoundedCounting |
            SemiringType::Goldilocks
        )
    }

    /// Compilation tier for this semiring.
    pub fn compilation_tier(&self) -> CompilationTier {
        if self.is_field_embeddable() {
            CompilationTier::Tier1Algebraic
        } else {
            match self {
                SemiringType::Tropical => CompilationTier::Tier2GadgetAssisted,
                _ => CompilationTier::NotCompilable,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CompilationTier
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompilationTier {
    /// Direct algebraic homomorphism φ: S → F_p.
    Tier1Algebraic,
    /// Requires comparison/range-check gadgets (tropical min-plus).
    Tier2GadgetAssisted,
    /// WFA-representable but no STARK compilation path implemented.
    WFAOnly,
    /// Not WFA-representable.
    NotCompilable,
}

// ---------------------------------------------------------------------------
// MetricSpecification — abstract description of a metric
// ---------------------------------------------------------------------------

/// Abstract specification of a metric as a composition of operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricSpecification {
    pub name: String,
    pub operations: Vec<MetricOperation>,
    pub description: String,
}

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------

/// Result of classifying a metric specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub metric_name: String,
    pub is_wfa_expressible: bool,
    pub tier: ExpressivenessTier,
    pub compilation_tier: CompilationTier,
    pub required_semiring: Option<SemiringType>,
    pub rational_operations: Vec<MetricOperation>,
    pub gadget_operations: Vec<MetricOperation>,
    pub non_wfa_operations: Vec<MetricOperation>,
    pub wfa_coverage_percent: f64,
    pub justification: String,
}

/// Expressiveness tier classification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpressivenessTier {
    /// Fully WFA-expressible: all operations are rational.
    FullyRational,
    /// WFA core with non-rational aggregation (gadgets).
    RationalWithGadgets,
    /// Partially WFA: some operations require external computation.
    Partial,
    /// Not WFA-expressible at all.
    NonWFA,
}

// ---------------------------------------------------------------------------
// MetricClassifier
// ---------------------------------------------------------------------------

/// Classifies metric specifications by WFA expressiveness.
///
/// This implements the decidable membership criterion from the
/// expressiveness characterization theorem.
pub struct MetricClassifier;

impl MetricClassifier {
    /// Classify a metric specification.
    pub fn classify(spec: &MetricSpecification) -> ClassificationResult {
        let mut rational_ops = Vec::new();
        let mut gadget_ops = Vec::new();
        let mut non_wfa_ops = Vec::new();

        for op in &spec.operations {
            if op.is_rational() {
                rational_ops.push(op.clone());
            } else if op.is_gadget_expressible() {
                gadget_ops.push(op.clone());
            } else {
                non_wfa_ops.push(op.clone());
            }
        }

        let total_ops = spec.operations.len().max(1);
        let wfa_coverage = (rational_ops.len() as f64 / total_ops as f64) * 100.0;

        let tier = if non_wfa_ops.is_empty() && gadget_ops.is_empty() {
            ExpressivenessTier::FullyRational
        } else if non_wfa_ops.is_empty() {
            ExpressivenessTier::RationalWithGadgets
        } else if !rational_ops.is_empty() {
            ExpressivenessTier::Partial
        } else {
            ExpressivenessTier::NonWFA
        };

        let is_wfa = matches!(tier,
            ExpressivenessTier::FullyRational | ExpressivenessTier::RationalWithGadgets
        );

        // Determine required semiring from rational operations
        let required_semiring = Self::determine_semiring(&rational_ops);

        let compilation_tier = if let Some(ref sr) = required_semiring {
            sr.compilation_tier()
        } else if is_wfa {
            CompilationTier::Tier1Algebraic
        } else {
            CompilationTier::NotCompilable
        };

        let justification = Self::generate_justification(&tier, &rational_ops, &gadget_ops, &non_wfa_ops);

        ClassificationResult {
            metric_name: spec.name.clone(),
            is_wfa_expressible: is_wfa,
            tier,
            compilation_tier,
            required_semiring,
            rational_operations: rational_ops,
            gadget_operations: gadget_ops,
            non_wfa_operations: non_wfa_ops,
            wfa_coverage_percent: wfa_coverage,
            justification,
        }
    }

    /// Classify a batch of metric specifications.
    pub fn classify_batch(specs: &[MetricSpecification]) -> Vec<ClassificationResult> {
        specs.iter().map(|s| Self::classify(s)).collect()
    }

    /// Generate the standard NLP metric catalogue with classifications.
    pub fn standard_catalogue() -> Vec<ClassificationResult> {
        let specs = Self::standard_metrics();
        Self::classify_batch(&specs)
    }

    fn determine_semiring(ops: &[MetricOperation]) -> Option<SemiringType> {
        let semirings: HashSet<SemiringType> = ops.iter()
            .filter_map(|op| op.required_semiring())
            .collect();

        if semirings.is_empty() {
            return Some(SemiringType::Boolean);
        }

        // Semiring lattice: Boolean ⊑ Counting ⊑ Real
        // Tropical is incomparable
        if semirings.contains(&SemiringType::Tropical) && semirings.len() > 1 {
            // Tropical + Counting requires multi-semiring decomposition
            return Some(SemiringType::Tropical);
        }

        if semirings.contains(&SemiringType::Tropical) {
            return Some(SemiringType::Tropical);
        }

        if semirings.contains(&SemiringType::BoundedCounting) || semirings.contains(&SemiringType::Counting) {
            return Some(SemiringType::Counting);
        }

        Some(SemiringType::Boolean)
    }

    fn generate_justification(
        tier: &ExpressivenessTier,
        rational: &[MetricOperation],
        gadgets: &[MetricOperation],
        non_wfa: &[MetricOperation],
    ) -> String {
        match tier {
            ExpressivenessTier::FullyRational => {
                format!(
                    "All {} operations are rational formal power series over semirings. \
                     By Schützenberger's theorem, the function is WFA-expressible.",
                    rational.len()
                )
            }
            ExpressivenessTier::RationalWithGadgets => {
                format!(
                    "{} rational operations (WFA core) + {} aggregation gadgets. \
                     Full metric correctness follows from Theorem (Aggregation Compositionality).",
                    rational.len(), gadgets.len()
                )
            }
            ExpressivenessTier::Partial => {
                format!(
                    "{} rational + {} gadget + {} non-WFA operations. \
                     Non-WFA operations ({:?}) require external computation.",
                    rational.len(), gadgets.len(), non_wfa.len(),
                    non_wfa.iter().map(|o| format!("{:?}", o)).collect::<Vec<_>>()
                )
            }
            ExpressivenessTier::NonWFA => {
                format!(
                    "No rational operations. All {} operations require external computation.",
                    non_wfa.len()
                )
            }
        }
    }

    /// Standard NLP metric specifications.
    pub fn standard_metrics() -> Vec<MetricSpecification> {
        vec![
            MetricSpecification {
                name: "Exact Match".into(),
                operations: vec![MetricOperation::TokenMatch],
                description: "Boolean token-by-token equality check".into(),
            },
            MetricSpecification {
                name: "Token F1".into(),
                operations: vec![
                    MetricOperation::NGramCount { n: 1 },
                    MetricOperation::Intersection,
                    MetricOperation::Division,
                    MetricOperation::HarmonicMean,
                ],
                description: "Harmonic mean of token precision and recall".into(),
            },
            MetricSpecification {
                name: "BLEU-4".into(),
                operations: vec![
                    MetricOperation::NGramCount { n: 1 },
                    MetricOperation::NGramCount { n: 2 },
                    MetricOperation::NGramCount { n: 3 },
                    MetricOperation::NGramCount { n: 4 },
                    MetricOperation::NGramClip { n: 1 },
                    MetricOperation::NGramClip { n: 2 },
                    MetricOperation::NGramClip { n: 3 },
                    MetricOperation::NGramClip { n: 4 },
                    MetricOperation::Division,
                    MetricOperation::GeometricMean,
                    MetricOperation::BrevityPenalty,
                ],
                description: "Geometric mean of clipped n-gram precisions with brevity penalty".into(),
            },
            MetricSpecification {
                name: "ROUGE-1".into(),
                operations: vec![
                    MetricOperation::NGramCount { n: 1 },
                    MetricOperation::Intersection,
                    MetricOperation::Division,
                    MetricOperation::FMeasure { beta: 10 },
                ],
                description: "Unigram overlap F-measure".into(),
            },
            MetricSpecification {
                name: "ROUGE-2".into(),
                operations: vec![
                    MetricOperation::NGramCount { n: 2 },
                    MetricOperation::Intersection,
                    MetricOperation::Division,
                    MetricOperation::FMeasure { beta: 10 },
                ],
                description: "Bigram overlap F-measure".into(),
            },
            MetricSpecification {
                name: "ROUGE-L".into(),
                operations: vec![
                    MetricOperation::LCSAlignment,
                    MetricOperation::Division,
                    MetricOperation::FMeasure { beta: 12 },
                ],
                description: "Longest common subsequence F-measure".into(),
            },
            MetricSpecification {
                name: "chrF".into(),
                operations: vec![
                    MetricOperation::NGramCount { n: 1 },
                    MetricOperation::NGramCount { n: 2 },
                    MetricOperation::NGramCount { n: 3 },
                    MetricOperation::NGramCount { n: 4 },
                    MetricOperation::NGramCount { n: 5 },
                    MetricOperation::NGramCount { n: 6 },
                    MetricOperation::Intersection,
                    MetricOperation::Division,
                    MetricOperation::FMeasure { beta: 20 },
                ],
                description: "Character n-gram F-score (n=1..6, β=2)".into(),
            },
            MetricSpecification {
                name: "BERTScore".into(),
                operations: vec![
                    MetricOperation::NeuralEmbedding,
                    MetricOperation::FMeasure { beta: 10 },
                ],
                description: "BERT embedding cosine similarity F-measure".into(),
            },
            MetricSpecification {
                name: "COMET".into(),
                operations: vec![
                    MetricOperation::LearnedModel,
                ],
                description: "Learned translation quality estimation model".into(),
            },
            MetricSpecification {
                name: "METEOR".into(),
                operations: vec![
                    MetricOperation::NGramCount { n: 1 },
                    MetricOperation::SynonymLookup,
                    MetricOperation::FMeasure { beta: 10 },
                ],
                description: "Unigram matching with synonym/stemming tables".into(),
            },
            MetricSpecification {
                name: "Levenshtein Distance".into(),
                operations: vec![MetricOperation::LCSAlignment],
                description: "Minimum edit distance via dynamic programming".into(),
            },
            MetricSpecification {
                name: "CER".into(),
                operations: vec![MetricOperation::LCSAlignment, MetricOperation::Division],
                description: "Character error rate (edit distance / reference length)".into(),
            },
            MetricSpecification {
                name: "WER".into(),
                operations: vec![MetricOperation::LCSAlignment, MetricOperation::Division],
                description: "Word error rate (edit distance / reference length)".into(),
            },
            MetricSpecification {
                name: "Pass@k".into(),
                operations: vec![MetricOperation::TokenMatch],
                description: "Exact string comparison for code evaluation".into(),
            },
            MetricSpecification {
                name: "Regex Match".into(),
                operations: vec![MetricOperation::TokenMatch],
                description: "Regular expression pattern matching".into(),
            },
            MetricSpecification {
                name: "Perplexity".into(),
                operations: vec![MetricOperation::LearnedModel, MetricOperation::Logarithm],
                description: "Language model cross-entropy loss".into(),
            },
            MetricSpecification {
                name: "BARTScore".into(),
                operations: vec![MetricOperation::LearnedModel],
                description: "BART-based generation evaluation".into(),
            },
            MetricSpecification {
                name: "UniEval".into(),
                operations: vec![MetricOperation::LearnedModel],
                description: "Unified multi-dimensional evaluation model".into(),
            },
        ]
    }
}

// ---------------------------------------------------------------------------
// HankelRationalityTest — formal rationality test via Hankel matrix rank
// ---------------------------------------------------------------------------

/// Tests rationality of a function by computing the rank of its
/// truncated Hankel matrix.
///
/// For a formal power series S: Σ* → K, the Hankel matrix H_S has
/// entry H_S[u,v] = S(uv) for all u,v ∈ Σ*. By the Fliess-Carlyle
/// theorem, S is rational iff rank(H_S) < ∞, and the minimal WFA
/// realizing S has exactly rank(H_S) states.
pub struct HankelRationalityTest {
    /// Maximum word length for truncation.
    pub max_length: usize,
    /// Alphabet size.
    pub alphabet_size: usize,
}

impl HankelRationalityTest {
    pub fn new(max_length: usize, alphabet_size: usize) -> Self {
        Self { max_length, alphabet_size }
    }

    /// Test rationality by checking if the Hankel matrix rank stabilizes.
    ///
    /// Returns (is_likely_rational, estimated_rank, rank_sequence).
    /// The rank_sequence shows rank at each truncation depth.
    pub fn test_rationality<F>(&self, f: &F) -> (bool, usize, Vec<usize>)
    where
        F: Fn(&[usize]) -> f64,
    {
        let mut rank_sequence = Vec::new();
        let mut prev_rank = 0;
        let mut stable_count = 0;
        let stability_threshold = 3; // rank must be stable for 3 consecutive depths

        for depth in 1..=self.max_length {
            let words = Self::enumerate_words(self.alphabet_size, depth);
            let n = words.len();

            if n == 0 {
                continue;
            }

            // Build truncated Hankel matrix
            let mut matrix = vec![0.0_f64; n * n];
            for (i, u) in words.iter().enumerate() {
                for (j, v) in words.iter().enumerate() {
                    let mut uv = u.clone();
                    uv.extend_from_slice(v);
                    matrix[i * n + j] = f(&uv);
                }
            }

            let rank = Self::compute_rank(&matrix, n, n);
            rank_sequence.push(rank);

            if rank == prev_rank {
                stable_count += 1;
            } else {
                stable_count = 0;
            }

            prev_rank = rank;

            if stable_count >= stability_threshold {
                return (true, rank, rank_sequence);
            }
        }

        // If rank is still growing, likely not rational
        let is_rational = stable_count >= stability_threshold;
        (is_rational, prev_rank, rank_sequence)
    }

    /// Enumerate all words up to given length over alphabet {0, ..., k-1}.
    fn enumerate_words(alphabet_size: usize, max_len: usize) -> Vec<Vec<usize>> {
        let mut words = vec![vec![]]; // empty word
        let mut current_length_words = vec![vec![]];

        for _len in 1..=max_len {
            let mut next = Vec::new();
            for word in &current_length_words {
                for symbol in 0..alphabet_size {
                    let mut new_word = word.clone();
                    new_word.push(symbol);
                    next.push(new_word);
                }
            }
            words.extend(next.clone());
            current_length_words = next;

            // Limit size to avoid combinatorial explosion
            if words.len() > 500 {
                break;
            }
        }

        words
    }

    /// Compute matrix rank using Gaussian elimination with partial pivoting.
    fn compute_rank(matrix: &[f64], rows: usize, cols: usize) -> usize {
        let mut m = matrix.to_vec();
        let epsilon = 1e-10;
        let mut rank = 0;
        let mut pivot_col = 0;

        for row in 0..rows {
            if pivot_col >= cols {
                break;
            }

            // Find pivot
            let mut max_row = row;
            let mut max_val = m[row * cols + pivot_col].abs();
            for r in (row + 1)..rows {
                let val = m[r * cols + pivot_col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = r;
                }
            }

            if max_val < epsilon {
                pivot_col += 1;
                continue;
            }

            // Swap rows
            if max_row != row {
                for c in 0..cols {
                    m.swap(row * cols + c, max_row * cols + c);
                }
            }

            // Eliminate below
            let pivot = m[row * cols + pivot_col];
            for r in (row + 1)..rows {
                let factor = m[r * cols + pivot_col] / pivot;
                for c in pivot_col..cols {
                    let val = m[row * cols + c];
                    m[r * cols + c] -= factor * val;
                }
            }

            rank += 1;
            pivot_col += 1;
        }

        rank
    }
}

// ---------------------------------------------------------------------------
// ExpressivenessTheorem — formal statement with constructive witness
// ---------------------------------------------------------------------------

/// Formal statement of the WFA expressiveness characterization theorem.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpressivenessTheorem {
    pub statement: String,
    pub constructive_direction: String,
    pub recognition_direction: String,
    pub decidability: String,
    pub implications: Vec<String>,
}

impl ExpressivenessTheorem {
    /// The main expressiveness characterization theorem.
    pub fn theorem() -> Self {
        Self {
            statement: concat!(
                "Theorem (WFA Expressiveness Characterization). ",
                "Let S be a semiring and Σ a finite alphabet. ",
                "A function f: Σ* → S is computable by a weighted finite automaton ",
                "over S if and only if f is a rational formal power series over S. ",
                "Equivalently, f is WFA-expressible iff the Hankel matrix H_f has finite rank."
            ).into(),
            constructive_direction: concat!(
                "Construction (Σ* → WFA): Given a rational formal power series f ",
                "with Hankel rank r, the Fliess-Carlyle construction produces a ",
                "minimal WFA with exactly r states. The construction is effective: ",
                "given oracle access to f, compute H_f[u,v] = f(uv) for sufficiently ",
                "many pairs, factor H_f = P·Q, and read off transition matrices."
            ).into(),
            recognition_direction: concat!(
                "Recognition (WFA → Σ*): Given a WFA A = (Q, Σ, {μ_σ}, α, η) ",
                "with n = |Q| states, [[A]](w) = α · μ_{σ_1} · ... · μ_{σ_k} · η ",
                "defines a rational formal power series of Hankel rank ≤ n."
            ).into(),
            decidability: concat!(
                "Decidability: For commutative semirings (Boolean, Counting, Goldilocks), ",
                "the question 'is f WFA-expressible?' is decidable by testing whether ",
                "the truncated Hankel matrix rank stabilizes. For the tropical semiring, ",
                "the question is decidable for fixed-structure metrics but undecidable ",
                "in the general case (Krob 1994)."
            ).into(),
            implications: vec![
                "Every NLP metric decomposable into n-gram counting, token matching, \
                 clipping, and LCS alignment has a WFA representation.".into(),
                "Metrics requiring neural embeddings, learned models, or synonym tables \
                 are provably outside the WFA fragment.".into(),
                "The Aggregation Compositionality Theorem extends WFA guarantees to \
                 full metrics via M = A ∘ f where f is rational and A is a \
                 well-defined aggregation function.".into(),
                "Decidable WFA equivalence holds for commutative-semiring metrics, \
                 enabling specification-level verification of metric implementations.".into(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_classification() {
        let spec = MetricSpecification {
            name: "Exact Match".into(),
            operations: vec![MetricOperation::TokenMatch],
            description: "".into(),
        };
        let result = MetricClassifier::classify(&spec);
        assert!(result.is_wfa_expressible);
        assert_eq!(result.tier, ExpressivenessTier::FullyRational);
        assert_eq!(result.required_semiring, Some(SemiringType::Boolean));
    }

    #[test]
    fn test_bleu_classification() {
        let spec = MetricSpecification {
            name: "BLEU-4".into(),
            operations: vec![
                MetricOperation::NGramCount { n: 1 },
                MetricOperation::NGramClip { n: 1 },
                MetricOperation::Division,
                MetricOperation::GeometricMean,
                MetricOperation::BrevityPenalty,
            ],
            description: "".into(),
        };
        let result = MetricClassifier::classify(&spec);
        assert!(result.is_wfa_expressible);
        assert_eq!(result.tier, ExpressivenessTier::RationalWithGadgets);
    }

    #[test]
    fn test_bertscore_classification() {
        let spec = MetricSpecification {
            name: "BERTScore".into(),
            operations: vec![
                MetricOperation::NeuralEmbedding,
                MetricOperation::FMeasure { beta: 10 },
            ],
            description: "".into(),
        };
        let result = MetricClassifier::classify(&spec);
        assert!(!result.is_wfa_expressible);
        assert_eq!(result.tier, ExpressivenessTier::NonWFA);
    }

    #[test]
    fn test_comet_classification() {
        let spec = MetricSpecification {
            name: "COMET".into(),
            operations: vec![MetricOperation::LearnedModel],
            description: "".into(),
        };
        let result = MetricClassifier::classify(&spec);
        assert!(!result.is_wfa_expressible);
        assert_eq!(result.tier, ExpressivenessTier::NonWFA);
    }

    #[test]
    fn test_standard_catalogue() {
        let catalogue = MetricClassifier::standard_catalogue();
        assert!(catalogue.len() >= 15);

        let wfa_count = catalogue.iter().filter(|r| r.is_wfa_expressible).count();
        assert!(wfa_count >= 10, "At least 10 standard metrics should be WFA-expressible");

        let non_wfa_count = catalogue.iter().filter(|r| !r.is_wfa_expressible).count();
        assert!(non_wfa_count >= 3, "At least 3 metrics should be non-WFA");
    }

    #[test]
    fn test_rouge_l_classification() {
        let spec = MetricSpecification {
            name: "ROUGE-L".into(),
            operations: vec![
                MetricOperation::LCSAlignment,
                MetricOperation::Division,
                MetricOperation::FMeasure { beta: 12 },
            ],
            description: "".into(),
        };
        let result = MetricClassifier::classify(&spec);
        assert!(result.is_wfa_expressible);
        assert_eq!(result.required_semiring, Some(SemiringType::Tropical));
        assert_eq!(result.compilation_tier, CompilationTier::Tier2GadgetAssisted);
    }

    #[test]
    fn test_hankel_constant_function() {
        // Constant function f(w) = 1 for all w.
        // This is rational (1-state WFA), so Hankel rank should be 1.
        let test = HankelRationalityTest::new(4, 2);
        let (is_rational, rank, _) = test.test_rationality(&|_w: &[usize]| 1.0);
        assert!(is_rational, "Constant function should be rational");
        assert_eq!(rank, 1, "Constant function should have Hankel rank 1");
    }

    #[test]
    fn test_hankel_length_function() {
        // f(w) = |w| (length function).
        // This is rational (2-state WFA over counting semiring).
        let test = HankelRationalityTest::new(4, 2);
        let (is_rational, rank, _) = test.test_rationality(&|w: &[usize]| w.len() as f64);
        assert!(is_rational, "Length function should be rational");
        assert!(rank <= 3, "Length function should have small Hankel rank, got {}", rank);
    }

    #[test]
    fn test_operation_classification() {
        assert!(MetricOperation::TokenMatch.is_rational());
        assert!(MetricOperation::NGramCount { n: 3 }.is_rational());
        assert!(MetricOperation::LCSAlignment.is_rational());

        assert!(MetricOperation::GeometricMean.is_gadget_expressible());
        assert!(MetricOperation::Division.is_gadget_expressible());
        assert!(MetricOperation::FMeasure { beta: 10 }.is_gadget_expressible());

        assert!(MetricOperation::NeuralEmbedding.is_non_wfa());
        assert!(MetricOperation::LearnedModel.is_non_wfa());
        assert!(MetricOperation::SynonymLookup.is_non_wfa());
    }

    #[test]
    fn test_expressiveness_theorem_statement() {
        let thm = ExpressivenessTheorem::theorem();
        assert!(thm.statement.contains("rational formal power series"));
        assert!(thm.statement.contains("Hankel matrix"));
        assert!(thm.decidability.contains("decidable"));
        assert!(!thm.implications.is_empty());
    }

    #[test]
    fn test_chrf_classification() {
        let spec = MetricSpecification {
            name: "chrF".into(),
            operations: vec![
                MetricOperation::NGramCount { n: 1 },
                MetricOperation::NGramCount { n: 2 },
                MetricOperation::NGramCount { n: 3 },
                MetricOperation::Intersection,
                MetricOperation::Division,
                MetricOperation::FMeasure { beta: 20 },
            ],
            description: "".into(),
        };
        let result = MetricClassifier::classify(&spec);
        assert!(result.is_wfa_expressible);
        assert_eq!(result.tier, ExpressivenessTier::RationalWithGadgets);
    }
}
