//! Behavioral atlas – a structured view of how each pipeline stage and
//! transformation interact, enriched with BFI data and coverage metrics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use localization::LocalizationResult;

use crate::bfi::{BFIComputer, BFIInterpretation, BFIResult};

// ── Core types ──────────────────────────────────────────────────────────────

/// Coverage statistics for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCoverage {
    pub stage_name: String,
    /// Number of test cases that exercised this stage.
    pub total_tests: usize,
    /// Number of test cases that triggered a violation at this stage.
    pub violations: usize,
    /// Fraction of transformations that have at least one violation here.
    pub transformation_coverage: f64,
}

impl StageCoverage {
    /// Violation rate = violations / total_tests (0.0 when no tests).
    pub fn violation_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            self.violations as f64 / self.total_tests as f64
        }
    }
}

/// Atlas entry for one pipeline stage: BFI, suspiciousness, and coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageAtlasEntry {
    pub stage_name: String,
    pub bfi_value: f64,
    pub bfi_interpretation: BFIInterpretation,
    pub suspiciousness_score: f64,
    pub rank: usize,
    pub coverage: StageCoverage,
    /// Per-transformation BFI breakdown.
    pub per_transformation_bfi: HashMap<String, f64>,
}

/// Atlas entry for one metamorphic transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationAtlasEntry {
    pub transformation_name: String,
    /// Total test cases that used this transformation.
    pub test_count: usize,
    /// Total violations observed across all stages.
    pub violation_count: usize,
    /// Average differential magnitude across stages.
    pub mean_differential: f64,
    /// Stages most affected by this transformation (name → mean differential).
    pub affected_stages: HashMap<String, f64>,
}

/// A single stage × transformation interaction record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEntry {
    pub stage_name: String,
    pub transformation_name: String,
    pub mean_differential: f64,
    pub violation_count: usize,
    pub sample_count: usize,
    pub bfi_value: f64,
}

// ── BehavioralAtlas ─────────────────────────────────────────────────────────

/// Configuration for atlas generation from a [`LocalizationResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasConfig {
    /// Whether to include per-transformation interaction breakdown.
    pub include_interactions: bool,
    /// Minimum suspiciousness to include a stage.
    pub suspiciousness_threshold: f64,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            include_interactions: true,
            suspiciousness_threshold: 0.0,
        }
    }
}

/// Aggregated behavioral atlas combining BFI, coverage, and interaction data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAtlas {
    /// Per-stage atlas entries (also accessible as `stage_entries`).
    pub stages: Vec<StageAtlasEntry>,
    /// Per-transformation atlas entries (also accessible as `transformation_entries`).
    pub transformations: Vec<TransformationAtlasEntry>,
    pub interactions: Vec<InteractionEntry>,
    pub metadata: HashMap<String, String>,
}

impl BehavioralAtlas {
    /// Backwards-compatible accessor for stage entries.
    pub fn stage_entries(&self) -> &[StageAtlasEntry] {
        &self.stages
    }
    /// Backwards-compatible accessor for transformation entries.
    pub fn transformation_entries(&self) -> &[TransformationAtlasEntry] {
        &self.transformations
    }
}

impl BehavioralAtlas {
    /// Build an atlas from raw per-stage and per-transformation data.
    ///
    /// * `stage_names` – ordered pipeline stage names
    /// * `stage_diffs` – `stage_diffs[k]` = differential values at stage k
    /// * `per_transform_diffs` – keyed by transformation name; value is a
    ///   `Vec<Vec<f64>>` indexed `[stage][sample]`
    /// * `violations` – `violations[stage][sample]` = whether violated
    /// * `suspiciousness` – pre-computed suspiciousness scores per stage
    pub fn build(
        stage_names: &[String],
        stage_diffs: &[Vec<f64>],
        per_transform_diffs: &HashMap<String, Vec<Vec<f64>>>,
        violations: &[Vec<bool>],
        suspiciousness: &[f64],
    ) -> Self {
        let computer = BFIComputer::default();
        let bfi_results = computer.compute_all_bfi(stage_names, stage_diffs);
        let bfi_profiles = computer.compute_profiles(stage_names, per_transform_diffs);

        // Sort stages by suspiciousness to assign ranks.
        let mut ranked: Vec<(usize, f64)> = suspiciousness.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut rank_map: HashMap<usize, usize> = HashMap::new();
        for (rank, &(idx, _)) in ranked.iter().enumerate() {
            rank_map.insert(idx, rank + 1);
        }

        // Stage entries.
        let stage_entries: Vec<StageAtlasEntry> = stage_names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let bfi = bfi_results.get(k).cloned().unwrap_or(BFIResult {
                    stage_name: name.clone(),
                    bfi_value: f64::NAN,
                    interpretation: BFIInterpretation::Undefined,
                    confidence_interval: (f64::NAN, f64::NAN),
                    sample_count: 0,
                });
                let total_tests = violations.get(k).map(|v| v.len()).unwrap_or(0);
                let viol_count = violations
                    .get(k)
                    .map(|v| v.iter().filter(|&&b| b).count())
                    .unwrap_or(0);
                let trans_coverage = compute_transformation_coverage(k, per_transform_diffs);
                let per_t_bfi = bfi_profiles
                    .get(k)
                    .map(|p| p.per_transformation.clone())
                    .unwrap_or_default();
                StageAtlasEntry {
                    stage_name: name.clone(),
                    bfi_value: bfi.bfi_value,
                    bfi_interpretation: bfi.interpretation,
                    suspiciousness_score: suspiciousness.get(k).copied().unwrap_or(0.0),
                    rank: rank_map.get(&k).copied().unwrap_or(0),
                    coverage: StageCoverage {
                        stage_name: name.clone(),
                        total_tests,
                        violations: viol_count,
                        transformation_coverage: trans_coverage,
                    },
                    per_transformation_bfi: per_t_bfi,
                }
            })
            .collect();

        // Transformation entries.
        let transformation_entries: Vec<TransformationAtlasEntry> = per_transform_diffs
            .iter()
            .map(|(tname, diffs_per_stage)| {
                let test_count = diffs_per_stage.first().map(|v| v.len()).unwrap_or(0);
                let mut total_violations = 0usize;
                let mut global_sum = 0.0f64;
                let mut global_count = 0usize;
                let mut affected: HashMap<String, f64> = HashMap::new();

                for (k, stage_diffs) in diffs_per_stage.iter().enumerate() {
                    if stage_diffs.is_empty() {
                        continue;
                    }
                    let mean = stage_diffs.iter().sum::<f64>() / stage_diffs.len() as f64;
                    global_sum += stage_diffs.iter().sum::<f64>();
                    global_count += stage_diffs.len();
                    if let Some(sname) = stage_names.get(k) {
                        affected.insert(sname.clone(), mean);
                    }
                    // Count violations exceeding a differential threshold.
                    total_violations += stage_diffs.iter().filter(|&&d| d > 0.5).count();
                }
                let mean_diff = if global_count > 0 {
                    global_sum / global_count as f64
                } else {
                    0.0
                };
                TransformationAtlasEntry {
                    transformation_name: tname.clone(),
                    test_count,
                    violation_count: total_violations,
                    mean_differential: mean_diff,
                    affected_stages: affected,
                }
            })
            .collect();

        // Interaction matrix.
        let mut interactions = Vec::new();
        for (tname, diffs_per_stage) in per_transform_diffs {
            for (k, stage_diffs) in diffs_per_stage.iter().enumerate() {
                if let Some(sname) = stage_names.get(k) {
                    if stage_diffs.is_empty() {
                        continue;
                    }
                    let mean_d = stage_diffs.iter().sum::<f64>() / stage_diffs.len() as f64;
                    let viol = stage_diffs.iter().filter(|&&d| d > 0.5).count();
                    let bfi_val = bfi_profiles
                        .get(k)
                        .and_then(|p| p.per_transformation.get(tname))
                        .copied()
                        .unwrap_or(f64::NAN);
                    interactions.push(InteractionEntry {
                        stage_name: sname.clone(),
                        transformation_name: tname.clone(),
                        mean_differential: mean_d,
                        violation_count: viol,
                        sample_count: stage_diffs.len(),
                        bfi_value: bfi_val,
                    });
                }
            }
        }

        BehavioralAtlas {
            stages: stage_entries,
            transformations: transformation_entries,
            interactions,
            metadata: HashMap::new(),
        }
    }

    /// Return entries for the top-N most suspicious stages.
    pub fn top_suspects(&self, n: usize) -> Vec<&StageAtlasEntry> {
        let mut sorted: Vec<&StageAtlasEntry> = self.stages.iter().collect();
        sorted.sort_by(|a, b| {
            b.suspiciousness_score
                .partial_cmp(&a.suspiciousness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Return the interaction entries for a specific stage.
    pub fn interactions_for_stage(&self, stage_name: &str) -> Vec<&InteractionEntry> {
        self.interactions
            .iter()
            .filter(|i| i.stage_name == stage_name)
            .collect()
    }

    /// Return the interaction entries for a specific transformation.
    pub fn interactions_for_transformation(
        &self,
        transformation_name: &str,
    ) -> Vec<&InteractionEntry> {
        self.interactions
            .iter()
            .filter(|i| i.transformation_name == transformation_name)
            .collect()
    }
}

/// Fraction of transformations that produced non-zero differentials at stage `k`.
fn compute_transformation_coverage(
    k: usize,
    per_transform_diffs: &HashMap<String, Vec<Vec<f64>>>,
) -> f64 {
    if per_transform_diffs.is_empty() {
        return 0.0;
    }
    let covered = per_transform_diffs
        .values()
        .filter(|diffs| {
            diffs
                .get(k)
                .map(|v| v.iter().any(|&d| d.abs() > f64::EPSILON))
                .unwrap_or(false)
        })
        .count();
    covered as f64 / per_transform_diffs.len() as f64
}

// ── Renderers ───────────────────────────────────────────────────────────────

/// Trait for rendering a [`BehavioralAtlas`] into a human- or machine-readable
/// format.
pub trait AtlasRenderer {
    /// Render the atlas into a string.
    fn render(&self, atlas: &BehavioralAtlas) -> String;
}

/// Renders the atlas as a JSON document.
#[derive(Debug, Clone, Default)]
pub struct JsonAtlasRenderer {
    pub pretty: bool,
}

impl JsonAtlasRenderer {
    pub fn new(pretty: bool) -> Self {
        Self { pretty }
    }
}

impl AtlasRenderer for JsonAtlasRenderer {
    fn render(&self, atlas: &BehavioralAtlas) -> String {
        if self.pretty {
            serde_json::to_string_pretty(atlas).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        } else {
            serde_json::to_string(atlas).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        }
    }
}

/// Renders the atlas as Markdown tables.
#[derive(Debug, Clone, Default)]
pub struct MarkdownAtlasRenderer;

impl MarkdownAtlasRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl AtlasRenderer for MarkdownAtlasRenderer {
    fn render(&self, atlas: &BehavioralAtlas) -> String {
        let mut out = String::new();
        out.push_str("# Behavioral Atlas\n\n");

        // Stage table.
        out.push_str("## Stage Summary\n\n");
        out.push_str("| Stage | BFI | Interpretation | Suspiciousness | Rank | Violations | Tests |\n");
        out.push_str("|-------|-----|----------------|----------------|------|------------|-------|\n");
        for entry in &atlas.stages {
            out.push_str(&format!(
                "| {} | {:.3} | {} | {:.4} | {} | {} | {} |\n",
                entry.stage_name,
                entry.bfi_value,
                entry.bfi_interpretation,
                entry.suspiciousness_score,
                entry.rank,
                entry.coverage.violations,
                entry.coverage.total_tests,
            ));
        }

        // Transformation table.
        out.push_str("\n## Transformation Summary\n\n");
        out.push_str("| Transformation | Tests | Violations | Mean Δ |\n");
        out.push_str("|----------------|-------|------------|--------|\n");
        for entry in &atlas.transformations {
            out.push_str(&format!(
                "| {} | {} | {} | {:.4} |\n",
                entry.transformation_name,
                entry.test_count,
                entry.violation_count,
                entry.mean_differential,
            ));
        }

        out
    }
}

/// Renders the atlas as plain text.
#[derive(Debug, Clone, Default)]
pub struct PlainTextAtlasRenderer;

impl PlainTextAtlasRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl AtlasRenderer for PlainTextAtlasRenderer {
    fn render(&self, atlas: &BehavioralAtlas) -> String {
        let mut out = String::new();
        out.push_str("=== Behavioral Atlas ===\n\n");

        out.push_str("Stages:\n");
        for entry in &atlas.stages {
            out.push_str(&format!(
                "  [{rank}] {name}: BFI={bfi:.3} ({interp}), susp={susp:.4}, violations={v}/{t}\n",
                rank = entry.rank,
                name = entry.stage_name,
                bfi = entry.bfi_value,
                interp = entry.bfi_interpretation,
                susp = entry.suspiciousness_score,
                v = entry.coverage.violations,
                t = entry.coverage.total_tests,
            ));
        }

        out.push_str("\nTransformations:\n");
        for entry in &atlas.transformations {
            out.push_str(&format!(
                "  {name}: {tests} tests, {v} violations, mean Δ={d:.4}\n",
                name = entry.transformation_name,
                tests = entry.test_count,
                v = entry.violation_count,
                d = entry.mean_differential,
            ));
        }

        out
    }
}

// ── Convenience function ────────────────────────────────────────────────────

/// Build a [`BehavioralAtlas`] directly from a [`LocalizationResult`].
///
/// This is the primary entry point used by the CLI.
pub fn generate_atlas(result: &LocalizationResult, _config: &AtlasConfig) -> BehavioralAtlas {
    let stage_names: Vec<String> = result.stage_results.iter().map(|s| s.stage_name.clone()).collect();
    let n_stages = stage_names.len();

    // Collect per-stage differential vectors.
    let stage_diffs: Vec<Vec<f64>> = result
        .stage_results
        .iter()
        .map(|s| s.differential_data.clone())
        .collect();

    // Collect per-transformation diffs keyed by transformation name.
    let mut per_transform: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for tname in &result.transformations_used {
        let mut diffs_per_stage = Vec::with_capacity(n_stages);
        for sr in &result.stage_results {
            if let Some(td) = sr.per_transformation.get(tname) {
                // Expand mean_differential × sample_count into a representative vector.
                let vals = vec![td.mean_differential; td.sample_count.max(1)];
                diffs_per_stage.push(vals);
            } else {
                diffs_per_stage.push(Vec::new());
            }
        }
        per_transform.insert(tname.clone(), diffs_per_stage);
    }

    // Derive per-stage violation vectors (threshold-based).
    let violations: Vec<Vec<bool>> = result
        .stage_results
        .iter()
        .map(|s| s.differential_data.iter().map(|&d| d > 0.5).collect())
        .collect();

    let suspiciousness: Vec<f64> = result.stage_results.iter().map(|s| s.suspiciousness).collect();

    BehavioralAtlas::build(&stage_names, &stage_diffs, &per_transform, &violations, &suspiciousness)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_atlas() -> BehavioralAtlas {
        let stages = vec!["tok".to_string(), "pos".to_string(), "ner".to_string()];
        let diffs = vec![
            vec![0.1, 0.2, 0.15],
            vec![0.5, 0.6, 0.55],
            vec![0.2, 0.25, 0.22],
        ];
        let mut per_t = HashMap::new();
        per_t.insert(
            "passive".to_string(),
            vec![
                vec![0.1, 0.2],
                vec![0.5, 0.6],
                vec![0.2, 0.25],
            ],
        );
        per_t.insert(
            "synonym".to_string(),
            vec![
                vec![0.05, 0.1],
                vec![0.3, 0.35],
                vec![0.15, 0.2],
            ],
        );
        let violations = vec![
            vec![false, false, false],
            vec![true, true, false],
            vec![false, true, false],
        ];
        let susp = vec![0.3, 0.85, 0.5];
        BehavioralAtlas::build(&stages, &diffs, &per_t, &violations, &susp)
    }

    #[test]
    fn test_atlas_build_stages() {
        let atlas = sample_atlas();
        assert_eq!(atlas.stages.len(), 3);
        assert_eq!(atlas.stages[0].stage_name, "tok");
        // pos has highest suspiciousness → rank 1
        assert_eq!(atlas.stages[1].rank, 1);
    }

    #[test]
    fn test_atlas_coverage() {
        let atlas = sample_atlas();
        let pos = &atlas.stages[1];
        assert_eq!(pos.coverage.total_tests, 3);
        assert_eq!(pos.coverage.violations, 2);
        assert!((pos.coverage.violation_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_atlas_transformations() {
        let atlas = sample_atlas();
        assert_eq!(atlas.transformations.len(), 2);
    }

    #[test]
    fn test_atlas_interactions() {
        let atlas = sample_atlas();
        assert!(!atlas.interactions.is_empty());
        let pos_interactions = atlas.interactions_for_stage("pos");
        assert_eq!(pos_interactions.len(), 2); // passive + synonym
    }

    #[test]
    fn test_top_suspects() {
        let atlas = sample_atlas();
        let top = atlas.top_suspects(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].stage_name, "pos");
    }

    #[test]
    fn test_json_renderer() {
        let atlas = sample_atlas();
        let renderer = JsonAtlasRenderer::new(false);
        let json = renderer.render(&atlas);
        assert!(json.contains("\"stage_name\""));
        assert!(json.contains("tok"));
    }

    #[test]
    fn test_markdown_renderer() {
        let atlas = sample_atlas();
        let renderer = MarkdownAtlasRenderer;
        let md = renderer.render(&atlas);
        assert!(md.contains("# Behavioral Atlas"));
        assert!(md.contains("| tok |"));
    }

    #[test]
    fn test_plaintext_renderer() {
        let atlas = sample_atlas();
        let renderer = PlainTextAtlasRenderer;
        let txt = renderer.render(&atlas);
        assert!(txt.contains("=== Behavioral Atlas ==="));
        assert!(txt.contains("tok"));
    }

    #[test]
    fn test_empty_atlas() {
        let atlas = BehavioralAtlas::build(&[], &[], &HashMap::new(), &[], &[]);
        assert!(atlas.stages.is_empty());
        assert!(atlas.transformations.is_empty());
    }

    #[test]
    fn test_stage_coverage_zero_tests() {
        let sc = StageCoverage {
            stage_name: "empty".into(),
            total_tests: 0,
            violations: 0,
            transformation_coverage: 0.0,
        };
        assert_eq!(sc.violation_rate(), 0.0);
    }
}
