//! # statistics
//!
//! Aggregate statistics and metrics for gap analysis results.
//!
//! Provides [`GapStatistics`] for computing and presenting summary metrics
//! over a [`GapReport`] and its ranked witnesses, including per-operator
//! breakdowns, severity distributions, and synthesis tier distributions.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use shared_types::contracts::{ContractProvenance, SynthesisTier};
use shared_types::operators::MutationOperator;

use crate::analyzer::GapReport;
use crate::ranking::{RankedWitness, RankingEngine, Severity};

// ---------------------------------------------------------------------------
// GapMetrics
// ---------------------------------------------------------------------------

/// Core numeric metrics extracted from a gap analysis run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GapMetrics {
    /// Total number of surviving mutants analysed.
    pub total_survivors: usize,

    /// Number classified as semantically equivalent.
    pub equivalents: usize,

    /// Number classified as non-equivalent (covered + gaps + inconclusive).
    pub non_equivalents: usize,

    /// Number of non-equivalent survivors covered by the contract.
    pub covered: usize,

    /// Number of specification gaps found.
    pub gaps: usize,

    /// Number of inconclusive results.
    pub inconclusive: usize,

    /// Coverage rate: `covered / (covered + gaps)`, or 1.0 if no
    /// non-equivalent survivors exist.
    pub coverage_rate: f64,

    /// Gap rate: `gaps / total_survivors`, or 0.0 when empty.
    pub gap_rate: f64,

    /// Equivalence rate: `equivalents / total_survivors`, or 0.0 when empty.
    pub equivalence_rate: f64,

    /// Total number of gap witnesses (some gaps may have multiple witnesses).
    pub total_witnesses: usize,
}

impl fmt::Display for GapMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gap Analysis Metrics")?;
        writeln!(f, "  Total survivors:  {}", self.total_survivors)?;
        writeln!(f, "  Equivalents:      {}", self.equivalents)?;
        writeln!(f, "  Non-equivalents:  {}", self.non_equivalents)?;
        writeln!(f, "    Covered:        {}", self.covered)?;
        writeln!(f, "    Gaps:           {}", self.gaps)?;
        writeln!(f, "    Inconclusive:   {}", self.inconclusive)?;
        writeln!(f, "  Coverage rate:    {:.1}%", self.coverage_rate * 100.0)?;
        writeln!(f, "  Gap rate:         {:.1}%", self.gap_rate * 100.0)?;
        writeln!(
            f,
            "  Equivalence rate: {:.1}%",
            self.equivalence_rate * 100.0
        )?;
        writeln!(f, "  Total witnesses:  {}", self.total_witnesses)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-operator breakdown
// ---------------------------------------------------------------------------

/// Gap rate breakdown for a single mutation operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorBreakdown {
    /// The mutation operator.
    pub operator: MutationOperator,

    /// Total survivors produced by this operator.
    pub total: usize,

    /// Number of gaps from this operator.
    pub gaps: usize,

    /// Number of equivalents from this operator.
    pub equivalents: usize,

    /// Number covered by the contract from this operator.
    pub covered: usize,

    /// Gap rate for this operator.
    pub gap_rate: f64,
}

impl OperatorBreakdown {
    /// Create a zeroed breakdown for the given operator.
    pub fn new(operator: MutationOperator) -> Self {
        Self {
            operator,
            total: 0,
            gaps: 0,
            equivalents: 0,
            covered: 0,
            gap_rate: 0.0,
        }
    }
}

impl fmt::Display for OperatorBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:>5}  total={:<4} gaps={:<4} equiv={:<4} covered={:<4} gap_rate={:.1}%",
            self.operator.mnemonic(),
            self.total,
            self.gaps,
            self.equivalents,
            self.covered,
            self.gap_rate * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Severity distribution
// ---------------------------------------------------------------------------

/// Distribution of findings across severity levels.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeverityDistribution {
    pub critical: usize,
    pub high: usize,
    pub medium: usize,
    pub low: usize,
    pub info: usize,
}

impl SeverityDistribution {
    /// Total count across all severity levels.
    pub fn total(&self) -> usize {
        self.critical + self.high + self.medium + self.low + self.info
    }

    /// Record a severity observation.
    pub fn record(&mut self, severity: &Severity) {
        match severity {
            Severity::Critical => self.critical += 1,
            Severity::High => self.high += 1,
            Severity::Medium => self.medium += 1,
            Severity::Low => self.low += 1,
            Severity::Info => self.info += 1,
        }
    }
}

impl fmt::Display for SeverityDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Severity Distribution ({} total):", self.total())?;
        writeln!(f, "  Critical: {}", self.critical)?;
        writeln!(f, "  High:     {}", self.high)?;
        writeln!(f, "  Medium:   {}", self.medium)?;
        writeln!(f, "  Low:      {}", self.low)?;
        writeln!(f, "  Info:     {}", self.info)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Synthesis tier distribution
// ---------------------------------------------------------------------------

/// Distribution of gap-covering contract clauses across synthesis tiers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierDistribution {
    pub tier1_lattice_walk: usize,
    pub tier2_template: usize,
    pub tier3_fallback: usize,
}

impl TierDistribution {
    /// Total count.
    pub fn total(&self) -> usize {
        self.tier1_lattice_walk + self.tier2_template + self.tier3_fallback
    }

    /// Record a synthesis tier observation.
    pub fn record(&mut self, tier: &SynthesisTier) {
        match tier {
            SynthesisTier::Tier1LatticeWalk => self.tier1_lattice_walk += 1,
            SynthesisTier::Tier2Template => self.tier2_template += 1,
            SynthesisTier::Tier3Fallback => self.tier3_fallback += 1,
        }
    }
}

impl fmt::Display for TierDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Synthesis Tier Distribution ({} total):", self.total())?;
        writeln!(f, "  Tier 1 (lattice walk): {}", self.tier1_lattice_walk)?;
        writeln!(f, "  Tier 2 (template):     {}", self.tier2_template)?;
        writeln!(f, "  Tier 3 (fallback):     {}", self.tier3_fallback)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GapStatistics
// ---------------------------------------------------------------------------

/// Complete statistics over a gap analysis result set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapStatistics {
    /// Core aggregate metrics.
    pub metrics: GapMetrics,

    /// Per-operator breakdown, keyed by operator mnemonic.
    pub by_operator: IndexMap<String, OperatorBreakdown>,

    /// Severity distribution of ranked witnesses.
    pub severity_distribution: SeverityDistribution,

    /// Synthesis tier distribution from contract provenance.
    pub tier_distribution: TierDistribution,
}

impl GapStatistics {
    /// Compute statistics from a [`GapReport`].
    pub fn from_report(report: &GapReport) -> Self {
        let metrics = Self::compute_metrics(report);
        let by_operator = Self::compute_operator_breakdown(report);
        // Without ranked witnesses, severity distribution will be empty.
        let severity_distribution = SeverityDistribution::default();
        let tier_distribution = TierDistribution::default();

        Self {
            metrics,
            by_operator,
            severity_distribution,
            tier_distribution,
        }
    }

    /// Compute statistics from a [`GapReport`] together with ranked witnesses
    /// and optional contract provenance entries.
    pub fn from_report_and_witnesses(
        report: &GapReport,
        ranked: &[RankedWitness],
        provenance: &[ContractProvenance],
    ) -> Self {
        let metrics = Self::compute_metrics(report);
        let by_operator = Self::compute_operator_breakdown(report);
        let severity_distribution = Self::compute_severity_distribution(ranked);
        let tier_distribution = Self::compute_tier_distribution(provenance);

        Self {
            metrics,
            by_operator,
            severity_distribution,
            tier_distribution,
        }
    }

    /// Convenience: compute everything using a default ranking engine and no
    /// provenance data.
    pub fn full(report: &GapReport) -> Self {
        let engine = RankingEngine::with_defaults();
        let ranked = engine.rank(report);
        let metrics = Self::compute_metrics(report);
        let by_operator = Self::compute_operator_breakdown(report);
        let severity_distribution = Self::compute_severity_distribution(&ranked);
        let tier_distribution = TierDistribution::default();

        Self {
            metrics,
            by_operator,
            severity_distribution,
            tier_distribution,
        }
    }

    // -- internal computations ----------------------------------------------

    fn compute_metrics(report: &GapReport) -> GapMetrics {
        let total = report.total_analysed();
        let equivalents = report.equivalent_count;
        let covered = report.covered_count;
        let gaps = report.gap_count;
        let inconclusive = report.inconclusive_count;
        let non_equivalents = covered + gaps + inconclusive;
        let total_witnesses = report.all_witnesses().len();

        let gap_rate = if total > 0 {
            gaps as f64 / total as f64
        } else {
            0.0
        };

        let equivalence_rate = if total > 0 {
            equivalents as f64 / total as f64
        } else {
            0.0
        };

        let coverage_rate = {
            let denom = covered + gaps;
            if denom > 0 {
                covered as f64 / denom as f64
            } else {
                1.0
            }
        };

        GapMetrics {
            total_survivors: total,
            equivalents,
            non_equivalents,
            covered,
            gaps,
            inconclusive,
            coverage_rate,
            gap_rate,
            equivalence_rate,
            total_witnesses,
        }
    }

    fn compute_operator_breakdown(report: &GapReport) -> IndexMap<String, OperatorBreakdown> {
        let mut map: IndexMap<String, OperatorBreakdown> = IndexMap::new();

        for result in report.results.values() {
            let key = result.operator.mnemonic().to_string();
            let entry = map
                .entry(key)
                .or_insert_with(|| OperatorBreakdown::new(result.operator.clone()));

            entry.total += 1;
            if result.classification.is_gap() {
                entry.gaps += 1;
            } else if result.classification.is_equivalent() {
                entry.equivalents += 1;
            } else if result.classification.is_covered() {
                entry.covered += 1;
            }
        }

        // Compute per-operator gap rates.
        for entry in map.values_mut() {
            entry.gap_rate = if entry.total > 0 {
                entry.gaps as f64 / entry.total as f64
            } else {
                0.0
            };
        }

        map
    }

    fn compute_severity_distribution(ranked: &[RankedWitness]) -> SeverityDistribution {
        let mut dist = SeverityDistribution::default();
        for rw in ranked {
            dist.record(&rw.severity);
        }
        dist
    }

    fn compute_tier_distribution(provenance: &[ContractProvenance]) -> TierDistribution {
        let mut dist = TierDistribution::default();
        for prov in provenance {
            dist.record(&prov.tier);
        }
        dist
    }
}

impl fmt::Display for GapStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.metrics)?;
        if !self.by_operator.is_empty() {
            writeln!(f, "Per-Operator Breakdown:")?;
            for entry in self.by_operator.values() {
                writeln!(f, "  {entry}")?;
            }
            writeln!(f)?;
        }
        write!(f, "{}", self.severity_distribution)?;
        write!(f, "{}", self.tier_distribution)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{GapAnalysisConfig, GapClassification, GapReport, MutantAnalysisResult};
    use crate::witness::GapWitness;
    use shared_types::contracts::SynthesisTier;
    use shared_types::operators::{MutantId, MutationOperator};
    use std::time::Duration;

    fn make_gap_result(op: MutationOperator, func: &str) -> MutantAnalysisResult {
        let id = MutantId::new();
        let witness = GapWitness::new(
            id.clone(),
            func.to_string(),
            op.clone(),
            "x + y".into(),
            "x - y".into(),
        );
        MutantAnalysisResult {
            mutant_id: id,
            function_name: func.to_string(),
            operator: op,
            classification: GapClassification::SpecificationGap {
                witnesses: vec![witness],
                checked_clauses: vec![],
            },
            analysis_duration: Duration::from_millis(10),
            equivalence_class: None,
        }
    }

    fn make_equivalent_result(op: MutationOperator, func: &str) -> MutantAnalysisResult {
        MutantAnalysisResult {
            mutant_id: MutantId::new(),
            function_name: func.to_string(),
            operator: op,
            classification: GapClassification::Equivalent,
            analysis_duration: Duration::from_millis(5),
            equivalence_class: None,
        }
    }

    fn make_covered_result(op: MutationOperator, func: &str) -> MutantAnalysisResult {
        MutantAnalysisResult {
            mutant_id: MutantId::new(),
            function_name: func.to_string(),
            operator: op,
            classification: GapClassification::CoveredByContract {
                covering_clauses: vec![],
            },
            analysis_duration: Duration::from_millis(8),
            equivalence_class: None,
        }
    }

    fn sample_report() -> GapReport {
        let mut report = GapReport::new(GapAnalysisConfig::default());
        report.insert(make_gap_result(MutationOperator::Aor, "add"));
        report.insert(make_gap_result(MutationOperator::Aor, "sub"));
        report.insert(make_gap_result(MutationOperator::Ror, "compare"));
        report.insert(make_equivalent_result(MutationOperator::Sdl, "helper"));
        report.insert(make_covered_result(MutationOperator::Aor, "add"));
        report
    }

    #[test]
    fn metrics_empty_report() {
        let report = GapReport::new(GapAnalysisConfig::default());
        let stats = GapStatistics::from_report(&report);

        assert_eq!(stats.metrics.total_survivors, 0);
        assert_eq!(stats.metrics.gaps, 0);
        assert_eq!(stats.metrics.gap_rate, 0.0);
        assert_eq!(stats.metrics.coverage_rate, 1.0);
    }

    #[test]
    fn metrics_sample_report() {
        let report = sample_report();
        let stats = GapStatistics::from_report(&report);

        assert_eq!(stats.metrics.total_survivors, 5);
        assert_eq!(stats.metrics.equivalents, 1);
        assert_eq!(stats.metrics.covered, 1);
        assert_eq!(stats.metrics.gaps, 3);
        assert_eq!(stats.metrics.inconclusive, 0);
        assert_eq!(stats.metrics.non_equivalents, 4); // 1 covered + 3 gaps
        assert_eq!(stats.metrics.total_witnesses, 3);

        // gap_rate = 3/5 = 0.6
        assert!((stats.metrics.gap_rate - 0.6).abs() < 1e-9);
        // coverage_rate = 1/(1+3) = 0.25
        assert!((stats.metrics.coverage_rate - 0.25).abs() < 1e-9);
        // equivalence_rate = 1/5 = 0.2
        assert!((stats.metrics.equivalence_rate - 0.2).abs() < 1e-9);
    }

    #[test]
    fn operator_breakdown() {
        let report = sample_report();
        let stats = GapStatistics::from_report(&report);

        let aor = &stats.by_operator["AOR"];
        assert_eq!(aor.total, 3);
        assert_eq!(aor.gaps, 2);
        assert_eq!(aor.covered, 1);

        let ror = &stats.by_operator["ROR"];
        assert_eq!(ror.total, 1);
        assert_eq!(ror.gaps, 1);
        assert!((ror.gap_rate - 1.0).abs() < 1e-9);

        let sdl = &stats.by_operator["SDL"];
        assert_eq!(sdl.total, 1);
        assert_eq!(sdl.equivalents, 1);
        assert_eq!(sdl.gaps, 0);
        assert!((sdl.gap_rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn severity_distribution_record() {
        let mut dist = SeverityDistribution::default();
        dist.record(&Severity::Critical);
        dist.record(&Severity::High);
        dist.record(&Severity::High);
        dist.record(&Severity::Medium);
        dist.record(&Severity::Low);
        dist.record(&Severity::Info);

        assert_eq!(dist.critical, 1);
        assert_eq!(dist.high, 2);
        assert_eq!(dist.medium, 1);
        assert_eq!(dist.low, 1);
        assert_eq!(dist.info, 1);
        assert_eq!(dist.total(), 6);
    }

    #[test]
    fn tier_distribution_record() {
        let mut dist = TierDistribution::default();
        dist.record(&SynthesisTier::Tier1LatticeWalk);
        dist.record(&SynthesisTier::Tier1LatticeWalk);
        dist.record(&SynthesisTier::Tier2Template);
        dist.record(&SynthesisTier::Tier3Fallback);

        assert_eq!(dist.tier1_lattice_walk, 2);
        assert_eq!(dist.tier2_template, 1);
        assert_eq!(dist.tier3_fallback, 1);
        assert_eq!(dist.total(), 4);
    }

    #[test]
    fn full_statistics() {
        let report = sample_report();
        let stats = GapStatistics::full(&report);

        // Should have metrics and severity distribution populated.
        assert_eq!(stats.metrics.total_survivors, 5);
        assert!(stats.severity_distribution.total() > 0);
    }

    #[test]
    fn display_formatting() {
        let report = sample_report();
        let stats = GapStatistics::from_report(&report);
        let display = format!("{stats}");

        assert!(display.contains("Gap Analysis Metrics"));
        assert!(display.contains("Total survivors:  5"));
        assert!(display.contains("Gaps:           3"));
        assert!(display.contains("Per-Operator Breakdown:"));
    }

    #[test]
    fn metrics_display() {
        let metrics = GapMetrics {
            total_survivors: 10,
            equivalents: 2,
            non_equivalents: 8,
            covered: 5,
            gaps: 3,
            inconclusive: 0,
            coverage_rate: 0.625,
            gap_rate: 0.3,
            equivalence_rate: 0.2,
            total_witnesses: 4,
        };
        let display = format!("{metrics}");
        assert!(display.contains("Total survivors:  10"));
        assert!(display.contains("62.5%"));
    }

    #[test]
    fn operator_breakdown_display() {
        let entry = OperatorBreakdown {
            operator: MutationOperator::Aor,
            total: 10,
            gaps: 3,
            equivalents: 2,
            covered: 5,
            gap_rate: 0.3,
        };
        let display = format!("{entry}");
        assert!(display.contains("AOR"));
        assert!(display.contains("30.0%"));
    }

    #[test]
    fn from_report_and_witnesses() {
        let report = sample_report();
        let engine = RankingEngine::with_defaults();
        let ranked = engine.rank(&report);
        let provenance = vec![
            ContractProvenance {
                targeted_mutants: vec![MutantId::new()],
                tier: SynthesisTier::Tier1LatticeWalk,
                solver_queries: 5,
                synthesis_time_ms: 100.0,
            },
            ContractProvenance {
                targeted_mutants: vec![MutantId::new()],
                tier: SynthesisTier::Tier2Template,
                solver_queries: 10,
                synthesis_time_ms: 200.0,
            },
        ];

        let stats = GapStatistics::from_report_and_witnesses(&report, &ranked, &provenance);

        assert_eq!(stats.metrics.total_survivors, 5);
        assert_eq!(stats.tier_distribution.tier1_lattice_walk, 1);
        assert_eq!(stats.tier_distribution.tier2_template, 1);
    }
}
