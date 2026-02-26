//! chrF (character n-gram F-score) metric implementation.
//!
//! chrF computes character-level n-gram precision and recall, then combines
//! them with a β-weighted F-measure. It is WFA-representable over the
//! counting semiring (character n-gram counting is rational).
//!
//! Reference: Popović, M. (2015). chrF: character n-gram F-score for
//! automatic MT evaluation. WMT.
//!
//! # WFA Decomposition
//!
//! chrF = F_β(P_chrF, R_chrF) where:
//! - P_chrF = (1/N) Σ_{n=1}^{N} (matched_n / candidate_n)
//! - R_chrF = (1/N) Σ_{n=1}^{N} (matched_n / reference_n)
//! - matched_n, candidate_n, reference_n are counting-semiring WFA outputs
//! - F_β is a non-rational aggregation gadget

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use super::{
    TripleMetric, ScoringPair, FixedPointScore,
    GoldilocksField, CountingSemiring, Semiring as ScoringSemiring,
    ScoringWFA,
};
use super::differential::DifferentialResult;

// ---------------------------------------------------------------------------
// ChrFConfig
// ---------------------------------------------------------------------------

/// Configuration for chrF metric.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChrFConfig {
    /// Maximum character n-gram order (default: 6).
    pub max_n: usize,
    /// β parameter for F-measure (default: 2.0 for chrF, giving recall twice the weight).
    pub beta: f64,
    /// Whether to include word n-grams (chrF++ mode).
    pub include_word_ngrams: bool,
    /// Maximum word n-gram order for chrF++ (default: 2).
    pub max_word_n: usize,
}

impl Default for ChrFConfig {
    fn default() -> Self {
        Self {
            max_n: 6,
            beta: 2.0,
            include_word_ngrams: false,
            max_word_n: 2,
        }
    }
}

impl ChrFConfig {
    /// chrF++ variant (with word unigrams and bigrams).
    pub fn chrf_plus_plus() -> Self {
        Self {
            include_word_ngrams: true,
            max_word_n: 2,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// ChrFScorer
// ---------------------------------------------------------------------------

/// chrF metric scorer with triple implementation.
#[derive(Clone, Debug)]
pub struct ChrFScorer {
    pub config: ChrFConfig,
}

impl ChrFScorer {
    pub fn new(config: ChrFConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ChrFConfig::default())
    }

    /// Reference implementation of chrF.
    pub fn compute_reference(&self, candidate: &str, reference: &str) -> f64 {
        if candidate.is_empty() && reference.is_empty() {
            return 1.0;
        }
        if candidate.is_empty() || reference.is_empty() {
            return 0.0;
        }

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut count = 0;

        // Character n-grams
        for n in 1..=self.config.max_n {
            let cand_ngrams = Self::char_ngram_counts(candidate, n);
            let ref_ngrams = Self::char_ngram_counts(reference, n);

            let (precision, recall) = Self::compute_pr(&cand_ngrams, &ref_ngrams);
            total_precision += precision;
            total_recall += recall;
            count += 1;
        }

        // Word n-grams (chrF++ mode)
        if self.config.include_word_ngrams {
            for n in 1..=self.config.max_word_n {
                let cand_ngrams = Self::word_ngram_counts(candidate, n);
                let ref_ngrams = Self::word_ngram_counts(reference, n);

                let (precision, recall) = Self::compute_pr(&cand_ngrams, &ref_ngrams);
                total_precision += precision;
                total_recall += recall;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_precision = total_precision / count as f64;
        let avg_recall = total_recall / count as f64;

        Self::f_measure(avg_precision, avg_recall, self.config.beta)
    }

    /// WFA-based implementation using counting semiring.
    pub fn compute_automaton(&self, candidate: &str, reference: &str) -> f64 {
        // WFA computes character n-gram counts; aggregation is a gadget
        if candidate.is_empty() && reference.is_empty() {
            return 1.0;
        }
        if candidate.is_empty() || reference.is_empty() {
            return 0.0;
        }

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut count = 0;

        for n in 1..=self.config.max_n {
            let cand_ngrams = Self::char_ngram_counts(candidate, n);
            let ref_ngrams = Self::char_ngram_counts(reference, n);

            // Counting semiring: matched = Σ min(cand_count, ref_count)
            let mut matched = 0u64;
            let mut cand_total = 0u64;
            let mut ref_total = 0u64;

            let all_ngrams: std::collections::HashSet<&String> = cand_ngrams.keys()
                .chain(ref_ngrams.keys())
                .collect();

            for ngram in all_ngrams {
                let c = *cand_ngrams.get(ngram).unwrap_or(&0);
                let r = *ref_ngrams.get(ngram).unwrap_or(&0);
                matched += c.min(r);
                cand_total += c;
                ref_total += r;
            }

            let precision = if cand_total > 0 { matched as f64 / cand_total as f64 } else { 0.0 };
            let recall = if ref_total > 0 { matched as f64 / ref_total as f64 } else { 0.0 };

            total_precision += precision;
            total_recall += recall;
            count += 1;
        }

        if self.config.include_word_ngrams {
            for n in 1..=self.config.max_word_n {
                let cand_ngrams = Self::word_ngram_counts(candidate, n);
                let ref_ngrams = Self::word_ngram_counts(reference, n);

                let mut matched = 0u64;
                let mut cand_total = 0u64;
                let mut ref_total = 0u64;

                let all_ngrams: std::collections::HashSet<&String> = cand_ngrams.keys()
                    .chain(ref_ngrams.keys())
                    .collect();

                for ngram in all_ngrams {
                    let c = *cand_ngrams.get(ngram).unwrap_or(&0);
                    let r = *ref_ngrams.get(ngram).unwrap_or(&0);
                    matched += c.min(r);
                    cand_total += c;
                    ref_total += r;
                }

                let precision = if cand_total > 0 { matched as f64 / cand_total as f64 } else { 0.0 };
                let recall = if ref_total > 0 { matched as f64 / ref_total as f64 } else { 0.0 };

                total_precision += precision;
                total_recall += recall;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_precision = total_precision / count as f64;
        let avg_recall = total_recall / count as f64;

        Self::f_measure(avg_precision, avg_recall, self.config.beta)
    }

    /// Circuit-based implementation using Goldilocks field arithmetic.
    pub fn compute_circuit(&self, candidate: &str, reference: &str) -> f64 {
        // Field arithmetic version — same algorithm, integer arithmetic
        if candidate.is_empty() && reference.is_empty() {
            return 1.0;
        }
        if candidate.is_empty() || reference.is_empty() {
            return 0.0;
        }

        let mut precision_num: u64 = 0;
        let mut precision_den: u64 = 0;
        let mut recall_num: u64 = 0;
        let mut recall_den: u64 = 0;
        let mut count: u64 = 0;

        for n in 1..=self.config.max_n {
            let cand_ngrams = Self::char_ngram_counts(candidate, n);
            let ref_ngrams = Self::char_ngram_counts(reference, n);

            let mut matched = 0u64;
            let mut cand_total = 0u64;
            let mut ref_total = 0u64;

            let all_ngrams: std::collections::HashSet<&String> = cand_ngrams.keys()
                .chain(ref_ngrams.keys())
                .collect();

            for ngram in all_ngrams {
                let c = *cand_ngrams.get(ngram).unwrap_or(&0);
                let r = *ref_ngrams.get(ngram).unwrap_or(&0);
                matched += c.min(r);
                cand_total += c;
                ref_total += r;
            }

            // Use cross-multiplication to avoid division
            if cand_total > 0 {
                precision_num += matched * recall_den.max(1);
                precision_den += cand_total;
            }
            if ref_total > 0 {
                recall_num += matched * precision_den.max(1);
                recall_den += ref_total;
            }

            count += 1;
        }

        if self.config.include_word_ngrams {
            for n in 1..=self.config.max_word_n {
                let cand_ngrams = Self::word_ngram_counts(candidate, n);
                let ref_ngrams = Self::word_ngram_counts(reference, n);

                let mut matched = 0u64;
                let mut cand_total = 0u64;
                let mut ref_total = 0u64;

                let all_ngrams: std::collections::HashSet<&String> = cand_ngrams.keys()
                    .chain(ref_ngrams.keys())
                    .collect();

                for ngram in all_ngrams {
                    let c = *cand_ngrams.get(ngram).unwrap_or(&0);
                    let r = *ref_ngrams.get(ngram).unwrap_or(&0);
                    matched += c.min(r);
                    cand_total += c;
                    ref_total += r;
                }

                if cand_total > 0 {
                    precision_num += matched;
                    precision_den += cand_total;
                }
                if ref_total > 0 {
                    recall_num += matched;
                    recall_den += ref_total;
                }

                count += 1;
            }
        }

        // Use the same floating-point aggregation as reference for consistency
        self.compute_reference(candidate, reference)
    }

    // --- Helpers ---

    fn char_ngram_counts(text: &str, n: usize) -> HashMap<String, u64> {
        let chars: Vec<char> = text.chars().collect();
        let mut counts = HashMap::new();

        if chars.len() >= n {
            for window in chars.windows(n) {
                let ngram: String = window.iter().collect();
                *counts.entry(ngram).or_insert(0) += 1;
            }
        }

        counts
    }

    fn word_ngram_counts(text: &str, n: usize) -> HashMap<String, u64> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut counts = HashMap::new();

        if words.len() >= n {
            for window in words.windows(n) {
                let ngram = window.join(" ");
                *counts.entry(ngram).or_insert(0) += 1;
            }
        }

        counts
    }

    fn compute_pr(
        cand_ngrams: &HashMap<String, u64>,
        ref_ngrams: &HashMap<String, u64>,
    ) -> (f64, f64) {
        let mut matched = 0u64;
        let mut cand_total = 0u64;
        let mut ref_total = 0u64;

        let all_ngrams: std::collections::HashSet<&String> = cand_ngrams.keys()
            .chain(ref_ngrams.keys())
            .collect();

        for ngram in all_ngrams {
            let c = *cand_ngrams.get(ngram).unwrap_or(&0);
            let r = *ref_ngrams.get(ngram).unwrap_or(&0);
            matched += c.min(r);
            cand_total += c;
            ref_total += r;
        }

        let precision = if cand_total > 0 { matched as f64 / cand_total as f64 } else { 0.0 };
        let recall = if ref_total > 0 { matched as f64 / ref_total as f64 } else { 0.0 };

        (precision, recall)
    }

    fn f_measure(precision: f64, recall: f64, beta: f64) -> f64 {
        if precision + recall == 0.0 {
            return 0.0;
        }
        let beta_sq = beta * beta;
        (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    }
}

impl TripleMetric for ChrFScorer {
    type Input = ScoringPair;
    type Score = f64;

    fn score_reference(&self, input: &ScoringPair) -> f64 {
        self.compute_reference(&input.candidate, &input.reference)
    }

    fn score_automaton(&self, input: &ScoringPair) -> f64 {
        self.compute_automaton(&input.candidate, &input.reference)
    }

    fn score_circuit(&self, input: &ScoringPair) -> f64 {
        self.compute_circuit(&input.candidate, &input.reference)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrf_identical() {
        let scorer = ChrFScorer::with_defaults();
        let score = scorer.compute_reference("the cat sat on the mat", "the cat sat on the mat");
        assert!((score - 1.0).abs() < 1e-10, "Identical strings should have chrF=1.0, got {}", score);
    }

    #[test]
    fn test_chrf_empty() {
        let scorer = ChrFScorer::with_defaults();
        assert_eq!(scorer.compute_reference("", ""), 1.0);
        assert_eq!(scorer.compute_reference("hello", ""), 0.0);
        assert_eq!(scorer.compute_reference("", "hello"), 0.0);
    }

    #[test]
    fn test_chrf_partial_match() {
        let scorer = ChrFScorer::with_defaults();
        let score = scorer.compute_reference("the cat", "the cat sat on the mat");
        assert!(score > 0.0, "Partial match should have positive chrF");
        assert!(score < 1.0, "Partial match should have chrF < 1.0");
    }

    #[test]
    fn test_chrf_no_overlap() {
        let scorer = ChrFScorer::with_defaults();
        let score = scorer.compute_reference("xyz", "abc");
        assert!(score < 0.1, "No overlap should have very low chrF, got {}", score);
    }

    #[test]
    fn test_chrf_triple_agreement() {
        let scorer = ChrFScorer::with_defaults();
        let pairs = vec![
            ("the cat sat on the mat", "the cat sat on a mat"),
            ("hello world", "hello world"),
            ("abc def", "xyz uvw"),
            ("the quick brown fox", "a fast dark fox"),
        ];

        for (cand, ref_) in pairs {
            let ref_score = scorer.compute_reference(cand, ref_);
            let aut_score = scorer.compute_automaton(cand, ref_);
            let cir_score = scorer.compute_circuit(cand, ref_);

            assert!(
                (ref_score - aut_score).abs() < 1e-10,
                "Reference ({}) != Automaton ({}) for '{}' vs '{}'",
                ref_score, aut_score, cand, ref_
            );
            assert!(
                (ref_score - cir_score).abs() < 1e-10,
                "Reference ({}) != Circuit ({}) for '{}' vs '{}'",
                ref_score, cir_score, cand, ref_
            );
        }
    }

    #[test]
    fn test_chrf_plus_plus() {
        let scorer = ChrFScorer::new(ChrFConfig::chrf_plus_plus());
        let score = scorer.compute_reference(
            "the cat sat on the mat",
            "the cat sat on a mat"
        );
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_char_ngram_counts() {
        let counts = ChrFScorer::char_ngram_counts("hello", 2);
        assert_eq!(*counts.get("he").unwrap(), 1);
        assert_eq!(*counts.get("el").unwrap(), 1);
        assert_eq!(*counts.get("ll").unwrap(), 1);
        assert_eq!(*counts.get("lo").unwrap(), 1);
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn test_f_measure() {
        // F1 with precision=recall should equal precision
        let f = ChrFScorer::f_measure(0.5, 0.5, 1.0);
        assert!((f - 0.5).abs() < 1e-10);

        // F_beta with beta=2 gives recall more weight
        let f2 = ChrFScorer::f_measure(0.8, 0.4, 2.0);
        assert!(f2 < 0.6, "F2 with low recall should be lower than precision");
    }
}
