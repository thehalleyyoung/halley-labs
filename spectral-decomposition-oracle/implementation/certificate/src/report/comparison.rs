//! Cross-instance comparison reports.
//!
//! Compare certificates across multiple instances, aggregate statistics,
//! and identify patterns.

use crate::report::generator::CertificateReport;
use crate::report::DecomposabilityTier;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Summary of one instance for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceSummary {
    pub name: String,
    pub structure_type: String,
    pub tier: DecomposabilityTier,
    pub l3_bound: Option<f64>,
    pub t2_bound: Option<f64>,
    pub t2_vacuous: bool,
    pub spectral_ratio: Option<f64>,
    pub futility_score: Option<f64>,
    pub actual_gap: Option<f64>,
    pub num_blocks: Option<usize>,
    pub crossing_edges: Option<usize>,
    pub verified: Option<bool>,
}

impl From<&CertificateReport> for InstanceSummary {
    fn from(report: &CertificateReport) -> Self {
        Self {
            name: report.instance_name.clone(),
            structure_type: report.structure_type.clone(),
            tier: report.tier,
            l3_bound: report.l3_partition_cert.as_ref().map(|c| c.total_bound),
            t2_bound: report.t2_scaling_cert.as_ref().map(|c| c.bound_value),
            t2_vacuous: report
                .t2_scaling_cert
                .as_ref()
                .map_or(true, |c| c.is_vacuous),
            spectral_ratio: report.t2_scaling_cert.as_ref().map(|c| c.spectral_ratio()),
            futility_score: report.futility_cert.as_ref().map(|c| c.futility_score),
            actual_gap: report.actual_gap,
            num_blocks: report.num_blocks,
            crossing_edges: report
                .l3_partition_cert
                .as_ref()
                .map(|c| c.crossing_edges.len()),
            verified: report.verification.as_ref().map(|v| v.all_verified),
        }
    }
}

/// Aggregate statistics for a group of instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
}

impl AggregateStats {
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                q25: 0.0,
                q75: 0.0,
            };
        }
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let q25 = sorted[n / 4];
        let q75 = sorted[(3 * n) / 4];

        Self {
            count: n,
            mean,
            std_dev,
            min,
            max,
            median,
            q25,
            q75,
        }
    }
}

/// Pattern identified across instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub affected_instances: Vec<String>,
    pub strength: f64,
}

/// Cross-instance comparison report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub num_instances: usize,
    pub instances: Vec<InstanceSummary>,
    pub tier_distribution: IndexMap<String, usize>,
    pub structure_stats: IndexMap<String, AggregateStats>,
    pub l3_bound_stats: Option<AggregateStats>,
    pub t2_bound_stats: Option<AggregateStats>,
    pub spectral_ratio_stats: Option<AggregateStats>,
    pub patterns: Vec<Pattern>,
    pub rankings: Vec<(String, f64)>,
}

impl ComparisonReport {
    /// Build a comparison report from multiple certificate reports.
    pub fn from_reports(reports: &[CertificateReport]) -> Self {
        let instances: Vec<InstanceSummary> = reports.iter().map(InstanceSummary::from).collect();

        let tier_distribution = Self::compute_tier_distribution(&instances);
        let structure_stats = Self::compute_structure_stats(&instances);
        let l3_bounds: Vec<f64> = instances.iter().filter_map(|i| i.l3_bound).collect();
        let t2_bounds: Vec<f64> = instances
            .iter()
            .filter(|i| !i.t2_vacuous)
            .filter_map(|i| i.t2_bound)
            .collect();
        let spectral_ratios: Vec<f64> = instances.iter().filter_map(|i| i.spectral_ratio).collect();

        let l3_stats = if !l3_bounds.is_empty() {
            Some(AggregateStats::compute(&l3_bounds))
        } else {
            None
        };
        let t2_stats = if !t2_bounds.is_empty() {
            Some(AggregateStats::compute(&t2_bounds))
        } else {
            None
        };
        let sr_stats = if !spectral_ratios.is_empty() {
            Some(AggregateStats::compute(&spectral_ratios))
        } else {
            None
        };

        let patterns = Self::identify_patterns(&instances);
        let rankings = Self::rank_by_decomposability(&instances);

        Self {
            num_instances: instances.len(),
            instances,
            tier_distribution,
            structure_stats,
            l3_bound_stats: l3_stats,
            t2_bound_stats: t2_stats,
            spectral_ratio_stats: sr_stats,
            patterns,
            rankings,
        }
    }

    fn compute_tier_distribution(instances: &[InstanceSummary]) -> IndexMap<String, usize> {
        let mut dist = IndexMap::new();
        for inst in instances {
            *dist.entry(inst.tier.to_string()).or_insert(0) += 1;
        }
        dist
    }

    fn compute_structure_stats(
        instances: &[InstanceSummary],
    ) -> IndexMap<String, AggregateStats> {
        let mut by_structure: IndexMap<String, Vec<f64>> = IndexMap::new();
        for inst in instances {
            if let Some(l3) = inst.l3_bound {
                by_structure
                    .entry(inst.structure_type.clone())
                    .or_default()
                    .push(l3);
            }
        }
        by_structure
            .into_iter()
            .map(|(k, v)| (k, AggregateStats::compute(&v)))
            .collect()
    }

    fn identify_patterns(instances: &[InstanceSummary]) -> Vec<Pattern> {
        let mut patterns = Vec::new();

        // Pattern: which structure types have tightest L3 bounds?
        let mut structure_bounds: IndexMap<String, Vec<f64>> = IndexMap::new();
        for inst in instances {
            if let Some(l3) = inst.l3_bound {
                structure_bounds
                    .entry(inst.structure_type.clone())
                    .or_default()
                    .push(l3);
            }
        }

        if let Some((best_structure, bounds)) = structure_bounds
            .iter()
            .min_by(|(_, a), (_, b)| {
                let mean_a = a.iter().sum::<f64>() / a.len() as f64;
                let mean_b = b.iter().sum::<f64>() / b.len() as f64;
                mean_a
                    .partial_cmp(&mean_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            let mean = bounds.iter().sum::<f64>() / bounds.len() as f64;
            let affected: Vec<String> = instances
                .iter()
                .filter(|i| i.structure_type == *best_structure)
                .map(|i| i.name.clone())
                .collect();
            patterns.push(Pattern {
                name: "tightest_l3_structure".to_string(),
                description: format!(
                    "'{}' has tightest L3 bounds (mean={:.4})",
                    best_structure, mean
                ),
                affected_instances: affected,
                strength: 1.0 / (1.0 + mean),
            });
        }

        // Pattern: where is T2 vacuous?
        let vacuous_instances: Vec<String> = instances
            .iter()
            .filter(|i| i.t2_vacuous)
            .map(|i| i.name.clone())
            .collect();
        if !vacuous_instances.is_empty() {
            let fraction = vacuous_instances.len() as f64 / instances.len().max(1) as f64;
            patterns.push(Pattern {
                name: "t2_vacuousness".to_string(),
                description: format!(
                    "T2 bound vacuous for {:.0}% of instances",
                    fraction * 100.0
                ),
                affected_instances: vacuous_instances,
                strength: fraction,
            });
        }

        // Pattern: futility prediction correlation with tier
        let futile_hard: Vec<String> = instances
            .iter()
            .filter(|i| {
                i.futility_score.map_or(false, |s| s > 0.7)
                    && (i.tier == DecomposabilityTier::Hard
                        || i.tier == DecomposabilityTier::Intractable)
            })
            .map(|i| i.name.clone())
            .collect();
        if !futile_hard.is_empty() {
            patterns.push(Pattern {
                name: "futility_tier_correlation".to_string(),
                description: format!(
                    "{} instances predicted futile and classified hard/intractable",
                    futile_hard.len()
                ),
                affected_instances: futile_hard,
                strength: 0.8,
            });
        }

        patterns
    }

    fn rank_by_decomposability(instances: &[InstanceSummary]) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = instances
            .iter()
            .map(|inst| {
                let mut score = 0.0;
                // Lower L3 bound = more decomposable
                if let Some(l3) = inst.l3_bound {
                    score += 1.0 / (1.0 + l3);
                }
                // Non-vacuous T2 = more informative
                if !inst.t2_vacuous {
                    score += 0.3;
                }
                // Lower spectral ratio = better structure
                if let Some(sr) = inst.spectral_ratio {
                    score += 1.0 / (1.0 + sr);
                }
                // Not futile = better
                if let Some(fs) = inst.futility_score {
                    score += 1.0 - fs;
                }
                (inst.name.clone(), score)
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Get instances in a specific tier.
    pub fn instances_in_tier(&self, tier: DecomposabilityTier) -> Vec<&InstanceSummary> {
        self.instances.iter().filter(|i| i.tier == tier).collect()
    }

    /// Summary text.
    pub fn summary_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Comparison Report: {} instances", self.num_instances));
        lines.push(String::new());
        lines.push("Tier Distribution:".to_string());
        for (tier, count) in &self.tier_distribution {
            lines.push(format!("  {}: {}", tier, count));
        }
        if let Some(ref stats) = self.l3_bound_stats {
            lines.push(format!(
                "\nL3 Bounds: mean={:.4}, median={:.4}, range=[{:.4}, {:.4}]",
                stats.mean, stats.median, stats.min, stats.max
            ));
        }
        if !self.patterns.is_empty() {
            lines.push("\nPatterns:".to_string());
            for p in &self.patterns {
                lines.push(format!("  • {}: {} (strength={:.2})", p.name, p.description, p.strength));
            }
        }
        if !self.rankings.is_empty() {
            lines.push("\nTop 5 Most Decomposable:".to_string());
            for (name, score) in self.rankings.iter().take(5) {
                lines.push(format!("  {}: {:.4}", name, score));
            }
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instance(name: &str, tier: DecomposabilityTier, l3: f64) -> InstanceSummary {
        InstanceSummary {
            name: name.to_string(),
            structure_type: "BlockAngular".to_string(),
            tier,
            l3_bound: Some(l3),
            t2_bound: Some(l3 * 10.0),
            t2_vacuous: l3 > 50.0,
            spectral_ratio: Some(l3 / 100.0),
            futility_score: Some(if l3 > 50.0 { 0.8 } else { 0.2 }),
            actual_gap: Some(l3 * 0.5),
            num_blocks: Some(3),
            crossing_edges: Some(10),
            verified: Some(true),
        }
    }

    #[test]
    fn test_aggregate_stats() {
        let stats = AggregateStats::compute(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_stats_empty() {
        let stats = AggregateStats::compute(&[]);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_comparison_report_empty() {
        let report = ComparisonReport::from_reports(&[]);
        assert_eq!(report.num_instances, 0);
    }

    #[test]
    fn test_comparison_from_instances() {
        let instances = vec![
            make_instance("inst1", DecomposabilityTier::Easy, 1.0),
            make_instance("inst2", DecomposabilityTier::Medium, 5.0),
            make_instance("inst3", DecomposabilityTier::Hard, 100.0),
        ];
        let report = ComparisonReport {
            num_instances: 3,
            instances: instances.clone(),
            tier_distribution: {
                let mut d = IndexMap::new();
                d.insert("EASY".to_string(), 1);
                d.insert("MEDIUM".to_string(), 1);
                d.insert("HARD".to_string(), 1);
                d
            },
            structure_stats: IndexMap::new(),
            l3_bound_stats: Some(AggregateStats::compute(&[1.0, 5.0, 100.0])),
            t2_bound_stats: None,
            spectral_ratio_stats: None,
            patterns: Vec::new(),
            rankings: vec![
                ("inst1".to_string(), 2.0),
                ("inst2".to_string(), 1.0),
                ("inst3".to_string(), 0.5),
            ],
        };
        assert_eq!(report.num_instances, 3);
        assert_eq!(report.tier_distribution.len(), 3);
    }

    #[test]
    fn test_instances_in_tier() {
        let instances = vec![
            make_instance("inst1", DecomposabilityTier::Easy, 1.0),
            make_instance("inst2", DecomposabilityTier::Easy, 2.0),
            make_instance("inst3", DecomposabilityTier::Hard, 100.0),
        ];
        let report = ComparisonReport {
            num_instances: 3,
            instances,
            tier_distribution: IndexMap::new(),
            structure_stats: IndexMap::new(),
            l3_bound_stats: None,
            t2_bound_stats: None,
            spectral_ratio_stats: None,
            patterns: Vec::new(),
            rankings: Vec::new(),
        };
        let easy = report.instances_in_tier(DecomposabilityTier::Easy);
        assert_eq!(easy.len(), 2);
    }

    #[test]
    fn test_summary_text() {
        let report = ComparisonReport {
            num_instances: 1,
            instances: vec![make_instance("test", DecomposabilityTier::Medium, 5.0)],
            tier_distribution: {
                let mut d = IndexMap::new();
                d.insert("MEDIUM".to_string(), 1);
                d
            },
            structure_stats: IndexMap::new(),
            l3_bound_stats: Some(AggregateStats::compute(&[5.0])),
            t2_bound_stats: None,
            spectral_ratio_stats: None,
            patterns: Vec::new(),
            rankings: vec![("test".to_string(), 1.0)],
        };
        let text = report.summary_text();
        assert!(text.contains("1 instances"));
        assert!(text.contains("MEDIUM"));
    }

    #[test]
    fn test_identify_patterns() {
        let instances = vec![
            make_instance("a", DecomposabilityTier::Easy, 1.0),
            make_instance("b", DecomposabilityTier::Hard, 100.0),
            make_instance("c", DecomposabilityTier::Intractable, 200.0),
        ];
        let patterns = ComparisonReport::identify_patterns(&instances);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_rank_by_decomposability() {
        let instances = vec![
            make_instance("easy", DecomposabilityTier::Easy, 1.0),
            make_instance("hard", DecomposabilityTier::Hard, 100.0),
        ];
        let ranked = ComparisonReport::rank_by_decomposability(&instances);
        assert_eq!(ranked[0].0, "easy");
        assert!(ranked[0].1 > ranked[1].1);
    }
}
