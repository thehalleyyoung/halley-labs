//! # ranking
//!
//! Bug report ranking for gap witnesses.
//!
//! After the gap analysis engine identifies specification gaps, the ranking
//! module prioritises them by *severity* and *confidence* so that developers
//! can focus on the most impactful findings first.
//!
//! ## Ranking criteria
//!
//! Each gap witness is scored along multiple axes:
//!
//! 1. **Severity** – How dangerous is the gap?  Gaps in postconditions are
//!    more severe than gaps in invariants; arithmetic operator replacement
//!    in security-critical functions is more severe than in display helpers.
//! 2. **Confidence** – How certain are we that the gap is real?  SMT-proven
//!    gaps rank higher than heuristic-based ones.
//! 3. **Impact** – How large is the behavioural divergence?  A mutant that
//!    can produce an off-by-one error is less impactful than one that can
//!    flip a sign.
//! 4. **Novelty** – Does this gap cover a new clause or duplicate an existing
//!    finding?

use std::cmp::Ordering;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use shared_types::operators::MutationOperator;

use crate::analyzer::{GapClassification, GapReport, MutantAnalysisResult};
use crate::witness::{GapWitness, InputGenerationMethod};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level of a gap witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Informational – the gap is minor and unlikely to cause real bugs.
    Info,
    /// Low severity – the gap is worth noting but not urgent.
    Low,
    /// Medium severity – the gap could lead to bugs under some inputs.
    Medium,
    /// High severity – the gap is likely exploitable and should be fixed.
    High,
    /// Critical – the gap represents a serious correctness or security issue.
    Critical,
}

impl Severity {
    /// Numeric weight for scoring (higher is more severe).
    pub fn weight(&self) -> f64 {
        match self {
            Self::Info => 1.0,
            Self::Low => 2.0,
            Self::Medium => 5.0,
            Self::High => 10.0,
            Self::Critical => 20.0,
        }
    }

    /// Parse from a string.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" | "med" => Self::Medium,
            "low" => Self::Low,
            _ => Self::Info,
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

/// Confidence level in a gap finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Confidence {
    /// Speculative – based on heuristics with limited evidence.
    Speculative,
    /// Likely – supported by partial evidence.
    Likely,
    /// Confirmed – fully supported by formal or concrete evidence.
    Confirmed,
}

impl Confidence {
    /// Numeric weight for scoring (higher is more confident).
    pub fn weight(&self) -> f64 {
        match self {
            Self::Speculative => 0.3,
            Self::Likely => 0.7,
            Self::Confirmed => 1.0,
        }
    }

    /// Derive confidence from a numeric score in `[0.0, 1.0]`.
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Confirmed
        } else if score >= 0.4 {
            Self::Likely
        } else {
            Self::Speculative
        }
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Speculative => write!(f, "speculative"),
            Self::Likely => write!(f, "likely"),
            Self::Confirmed => write!(f, "confirmed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Ranked witness
// ---------------------------------------------------------------------------

/// A gap witness with attached ranking metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedWitness {
    /// The underlying gap witness.
    pub witness: GapWitness,

    /// Assigned severity.
    pub severity: Severity,

    /// Assigned confidence.
    pub confidence: Confidence,

    /// Composite score (higher = more important).
    pub score: f64,

    /// Rank position (1-based, assigned after sorting).
    pub rank: usize,

    /// Individual scoring components for transparency.
    pub score_breakdown: ScoreBreakdown,
}

impl RankedWitness {
    /// Returns the mutant ID.
    pub fn mutant_id(&self) -> &shared_types::operators::MutantId {
        &self.witness.mutant_id
    }

    /// Returns the function name.
    pub fn function_name(&self) -> &str {
        &self.witness.function_name
    }

    /// Returns the mutation operator.
    pub fn operator(&self) -> &MutationOperator {
        &self.witness.operator
    }
}

impl fmt::Display for RankedWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{rank}: [{severity}] {func}::{op} (score={score:.2}, confidence={conf})",
            rank = self.rank,
            severity = self.severity,
            func = self.witness.function_name,
            op = self.witness.operator,
            score = self.score,
            conf = self.confidence,
        )
    }
}

/// Breakdown of the composite score into individual components.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Score from severity assessment.
    pub severity_score: f64,
    /// Score from confidence assessment.
    pub confidence_score: f64,
    /// Score from impact assessment.
    pub impact_score: f64,
    /// Score from novelty assessment.
    pub novelty_score: f64,
    /// Score from input quality.
    pub input_quality_score: f64,
}

impl ScoreBreakdown {
    /// Compute the weighted composite score.
    pub fn composite(&self) -> f64 {
        const W_SEVERITY: f64 = 0.35;
        const W_CONFIDENCE: f64 = 0.25;
        const W_IMPACT: f64 = 0.20;
        const W_NOVELTY: f64 = 0.10;
        const W_INPUT: f64 = 0.10;

        self.severity_score * W_SEVERITY
            + self.confidence_score * W_CONFIDENCE
            + self.impact_score * W_IMPACT
            + self.novelty_score * W_NOVELTY
            + self.input_quality_score * W_INPUT
    }
}

// ---------------------------------------------------------------------------
// Ranking engine
// ---------------------------------------------------------------------------

/// Configuration for the ranking engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Operators considered high-severity (domain-specific).
    pub high_severity_operators: Vec<MutationOperator>,

    /// Functions considered security-critical.
    pub critical_functions: Vec<String>,

    /// Minimum score to include in the ranked output.
    pub minimum_score: f64,

    /// Maximum number of ranked witnesses to return.
    pub max_results: usize,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            high_severity_operators: vec![MutationOperator::Ror, MutationOperator::Bcn],
            critical_functions: Vec::new(),
            minimum_score: 0.0,
            max_results: 500,
        }
    }
}

/// The ranking engine scores and sorts gap witnesses.
pub struct RankingEngine {
    config: RankingConfig,
}

impl RankingEngine {
    /// Create a new ranking engine.
    pub fn new(config: RankingConfig) -> Self {
        Self { config }
    }

    /// Create a ranking engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RankingConfig::default())
    }

    /// Rank all gap witnesses from a gap report.
    pub fn rank(&self, report: &GapReport) -> Vec<RankedWitness> {
        let witnesses: Vec<&GapWitness> = report.all_witnesses();

        // Collect all function names for novelty computation.
        let mut function_counts: IndexMap<String, usize> = IndexMap::new();
        for w in &witnesses {
            *function_counts.entry(w.function_name.clone()).or_insert(0) += 1;
        }

        // Collect all operator counts for novelty computation.
        let mut operator_counts: IndexMap<MutationOperator, usize> = IndexMap::new();
        for w in &witnesses {
            *operator_counts.entry(w.operator.clone()).or_insert(0) += 1;
        }

        let total_witnesses = witnesses.len().max(1);

        let mut ranked: Vec<RankedWitness> = witnesses
            .into_iter()
            .map(|w| self.score_witness(w, &function_counts, &operator_counts, total_witnesses))
            .filter(|rw| rw.score >= self.config.minimum_score)
            .collect();

        // Sort by score descending.
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        // Truncate and assign ranks.
        ranked.truncate(self.config.max_results);
        for (i, rw) in ranked.iter_mut().enumerate() {
            rw.rank = i + 1;
        }

        ranked
    }

    /// Rank witnesses grouped by function.
    pub fn rank_by_function(&self, report: &GapReport) -> IndexMap<String, Vec<RankedWitness>> {
        let ranked = self.rank(report);
        let mut by_function: IndexMap<String, Vec<RankedWitness>> = IndexMap::new();
        for rw in ranked {
            by_function
                .entry(rw.witness.function_name.clone())
                .or_default()
                .push(rw);
        }
        by_function
    }

    /// Get the top-N ranked witnesses.
    pub fn top_n(&self, report: &GapReport, n: usize) -> Vec<RankedWitness> {
        let mut ranked = self.rank(report);
        ranked.truncate(n);
        ranked
    }

    /// Compute a summary of the ranking distribution.
    pub fn distribution_summary(&self, ranked: &[RankedWitness]) -> RankingDistribution {
        let mut dist = RankingDistribution::default();
        for rw in ranked {
            match rw.severity {
                Severity::Critical => dist.critical += 1,
                Severity::High => dist.high += 1,
                Severity::Medium => dist.medium += 1,
                Severity::Low => dist.low += 1,
                Severity::Info => dist.info += 1,
            }
            dist.total_score += rw.score;
        }
        dist.count = ranked.len();
        if dist.count > 0 {
            dist.average_score = dist.total_score / dist.count as f64;
        }
        dist
    }

    // -- internal scoring ---------------------------------------------------

    /// Score a single witness across all dimensions.
    fn score_witness(
        &self,
        witness: &GapWitness,
        function_counts: &IndexMap<String, usize>,
        operator_counts: &IndexMap<MutationOperator, usize>,
        total_witnesses: usize,
    ) -> RankedWitness {
        let severity = self.assess_severity(witness);
        let confidence = self.assess_confidence(witness);

        let severity_score = severity.weight() / Severity::Critical.weight();
        let confidence_score = confidence.weight();
        let impact_score = self.assess_impact(witness);
        let novelty_score =
            self.assess_novelty(witness, function_counts, operator_counts, total_witnesses);
        let input_quality_score = self.assess_input_quality(witness);

        let breakdown = ScoreBreakdown {
            severity_score,
            confidence_score,
            impact_score,
            novelty_score,
            input_quality_score,
        };

        let score = breakdown.composite();

        RankedWitness {
            witness: witness.clone(),
            severity,
            confidence,
            score,
            rank: 0, // assigned after sorting
            score_breakdown: breakdown,
        }
    }

    /// Assess the severity of a gap witness.
    fn assess_severity(&self, witness: &GapWitness) -> Severity {
        // Critical if in a critical function.
        if self
            .config
            .critical_functions
            .iter()
            .any(|f| f == &witness.function_name)
        {
            return Severity::Critical;
        }

        // High if the operator is in the high-severity set.
        if self
            .config
            .high_severity_operators
            .contains(&witness.operator)
        {
            return Severity::High;
        }

        // Score based on operator type.
        match witness.operator {
            MutationOperator::Ror | MutationOperator::Bcn => Severity::High,
            MutationOperator::Aor | MutationOperator::Cor => Severity::Medium,
            MutationOperator::Lcr | MutationOperator::Uoi => Severity::Medium,
            MutationOperator::Sdl => Severity::Low,
            MutationOperator::Abs | MutationOperator::Rvr => Severity::Medium,
            MutationOperator::Crc | MutationOperator::Air => Severity::Medium,
            MutationOperator::Osw => Severity::Low,
        }
    }

    /// Assess the confidence in a gap finding.
    fn assess_confidence(&self, witness: &GapWitness) -> Confidence {
        let avg = witness.average_confidence();

        // Boost confidence if we have concrete divergence.
        if witness.has_concrete_divergence() {
            return Confidence::Confirmed;
        }

        // Boost if we have SMT-derived inputs.
        if witness
            .inputs
            .iter()
            .any(|i| i.generation_method == InputGenerationMethod::SmtModel)
        {
            return if avg >= 0.6 {
                Confidence::Confirmed
            } else {
                Confidence::Likely
            };
        }

        Confidence::from_score(avg)
    }

    /// Assess the impact of the behavioural divergence.
    ///
    /// Returns a score in `[0.0, 1.0]`.
    fn assess_impact(&self, witness: &GapWitness) -> f64 {
        // Based on maximum output delta.
        if let Some(delta) = witness.max_output_delta() {
            // Log-scale: delta of 1 → 0.3, delta of 10 → 0.6, delta of 100 → 0.9
            let log_delta = (delta.max(1) as f64).log10();
            return (log_delta / 3.0).clamp(0.1, 1.0);
        }

        // If no concrete delta, score based on operator severity.
        match witness.operator {
            MutationOperator::Ror | MutationOperator::Bcn => 0.7,
            MutationOperator::Aor => 0.5,
            MutationOperator::Sdl => 0.3,
            _ => 0.4,
        }
    }

    /// Assess the novelty of a witness.
    ///
    /// Witnesses in rarely-seen functions or from rarely-seen operators are
    /// more novel.  Returns a score in `[0.0, 1.0]`.
    fn assess_novelty(
        &self,
        witness: &GapWitness,
        function_counts: &IndexMap<String, usize>,
        operator_counts: &IndexMap<MutationOperator, usize>,
        total_witnesses: usize,
    ) -> f64 {
        let func_freq = *function_counts.get(&witness.function_name).unwrap_or(&1) as f64
            / total_witnesses as f64;

        let op_freq =
            *operator_counts.get(&witness.operator).unwrap_or(&1) as f64 / total_witnesses as f64;

        // Novelty is inverse of frequency (rare = novel).
        let func_novelty = 1.0 - func_freq;
        let op_novelty = 1.0 - op_freq;

        (func_novelty * 0.6 + op_novelty * 0.4).clamp(0.0, 1.0)
    }

    /// Assess the quality of the distinguishing inputs.
    ///
    /// Higher quality means more diverse and more confidently generated inputs.
    fn assess_input_quality(&self, witness: &GapWitness) -> f64 {
        if witness.inputs.is_empty() {
            return 0.0;
        }

        // Diversity: how many distinct generation methods were used?
        let methods: std::collections::HashSet<InputGenerationMethod> =
            witness.inputs.iter().map(|i| i.generation_method).collect();
        let method_diversity = (methods.len() as f64 / 4.0).clamp(0.0, 1.0);

        // Average confidence of inputs.
        let avg_confidence = witness.average_confidence();

        // Number of inputs (more is better, up to a point).
        let count_score = (witness.inputs.len() as f64 / 5.0).clamp(0.0, 1.0);

        // Has concrete outputs?
        let concrete_bonus = if witness.has_concrete_divergence() {
            0.2
        } else {
            0.0
        };

        ((method_diversity * 0.3 + avg_confidence * 0.4 + count_score * 0.3) + concrete_bonus)
            .clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Ranking distribution
// ---------------------------------------------------------------------------

/// Summary of the severity distribution of ranked witnesses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankingDistribution {
    pub count: usize,
    pub critical: usize,
    pub high: usize,
    pub medium: usize,
    pub low: usize,
    pub info: usize,
    pub total_score: f64,
    pub average_score: f64,
}

impl fmt::Display for RankingDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Ranking Distribution ({} witnesses):", self.count)?;
        writeln!(f, "  Critical: {}", self.critical)?;
        writeln!(f, "  High:     {}", self.high)?;
        writeln!(f, "  Medium:   {}", self.medium)?;
        writeln!(f, "  Low:      {}", self.low)?;
        writeln!(f, "  Info:     {}", self.info)?;
        writeln!(f, "  Average score: {:.3}", self.average_score)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::operators::MutantId;

    fn make_witness(op: MutationOperator, func: &str) -> GapWitness {
        GapWitness::new(
            MutantId::new(),
            func.to_string(),
            op,
            "x + y".into(),
            "x - y".into(),
        )
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Low);
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn severity_weights() {
        assert!(Severity::Critical.weight() > Severity::High.weight());
        assert!(Severity::Info.weight() > 0.0);
    }

    #[test]
    fn confidence_ordering() {
        assert!(Confidence::Speculative < Confidence::Likely);
        assert!(Confidence::Likely < Confidence::Confirmed);
    }

    #[test]
    fn confidence_from_score() {
        assert_eq!(Confidence::from_score(0.9), Confidence::Confirmed);
        assert_eq!(Confidence::from_score(0.5), Confidence::Likely);
        assert_eq!(Confidence::from_score(0.1), Confidence::Speculative);
    }

    #[test]
    fn score_breakdown_composite() {
        let bd = ScoreBreakdown {
            severity_score: 1.0,
            confidence_score: 1.0,
            impact_score: 1.0,
            novelty_score: 1.0,
            input_quality_score: 1.0,
        };
        let c = bd.composite();
        // Sum of all weights should be 1.0.
        assert!((c - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ranking_engine_empty_report() {
        let engine = RankingEngine::with_defaults();
        let report = GapReport::new(crate::analyzer::GapAnalysisConfig::default());
        let ranked = engine.rank(&report);
        assert!(ranked.is_empty());
    }

    #[test]
    fn severity_display() {
        assert_eq!(format!("{}", Severity::Critical), "critical");
        assert_eq!(format!("{}", Severity::Info), "info");
    }

    #[test]
    fn confidence_display() {
        assert_eq!(format!("{}", Confidence::Confirmed), "confirmed");
        assert_eq!(format!("{}", Confidence::Speculative), "speculative");
    }

    #[test]
    fn ranking_distribution_display() {
        let dist = RankingDistribution {
            count: 10,
            critical: 1,
            high: 3,
            medium: 4,
            low: 1,
            info: 1,
            total_score: 5.5,
            average_score: 0.55,
        };
        let s = format!("{dist}");
        assert!(s.contains("10 witnesses"));
        assert!(s.contains("Critical: 1"));
    }

    #[test]
    fn severity_from_str_loose() {
        assert_eq!(Severity::from_str_loose("critical"), Severity::Critical);
        assert_eq!(Severity::from_str_loose("HIGH"), Severity::High);
        assert_eq!(Severity::from_str_loose("unknown"), Severity::Info);
    }
}
