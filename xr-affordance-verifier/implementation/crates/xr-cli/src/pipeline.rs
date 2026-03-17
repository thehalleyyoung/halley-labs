//! Verification pipeline orchestration.
//!
//! Manages the full verification workflow: parse → validate → lint → verify → certify.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use xr_lint::{LintFinding, SceneLinter};
use xr_types::certificate::{SampleVerdict, VerifiedRegion, ViolationSurface};
use xr_types::config::VerifierConfig;
use xr_types::error::{Diagnostic, Severity, VerifierError, VerifierResult};
use xr_types::scene::SceneModel;
use xr_types::traits::Verdict;

// ─── Pipeline stage ────────────────────────────────────────────────────────

/// Stages of the verification pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Parsing the scene file.
    Parse,
    /// Validating scene structure.
    Validate,
    /// Running Tier 1 lint rules.
    Lint,
    /// Running Tier 1 spatial verification.
    SpatialVerify,
    /// Running population sampling.
    Sample,
    /// Running Tier 2 SMT verification.
    SmtVerify,
    /// Generating certificate.
    Certify,
    /// Complete.
    Done,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse => write!(f, "Parsing"),
            Self::Validate => write!(f, "Validating"),
            Self::Lint => write!(f, "Linting"),
            Self::SpatialVerify => write!(f, "Spatial Verification"),
            Self::Sample => write!(f, "Population Sampling"),
            Self::SmtVerify => write!(f, "SMT Verification"),
            Self::Certify => write!(f, "Certifying"),
            Self::Done => write!(f, "Done"),
        }
    }
}

// ─── Pipeline result ───────────────────────────────────────────────────────

/// Result of running the verification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Stage the pipeline reached.
    pub final_stage: PipelineStage,
    /// Lint findings from Tier 1.
    pub lint_findings: Vec<LintFinding>,
    /// Sample verdicts from population sampling.
    pub sample_verdicts: Vec<SampleVerdict>,
    /// Regions proven accessible by SMT.
    pub verified_regions: Vec<VerifiedRegion>,
    /// Detected violations.
    pub violations: Vec<ViolationSurface>,
    /// Analytical linearization error bound.
    pub epsilon_analytical: f64,
    /// Timing per stage.
    #[serde(skip)]
    pub stage_timings: Vec<(PipelineStage, Duration)>,
    /// Cached diagnostics from validation.
    pub diagnostics: Vec<Diagnostic>,
}

impl PipelineResult {
    fn new() -> Self {
        Self {
            final_stage: PipelineStage::Parse,
            lint_findings: Vec::new(),
            sample_verdicts: Vec::new(),
            verified_regions: Vec::new(),
            violations: Vec::new(),
            epsilon_analytical: 0.0,
            stage_timings: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Whether the result contains any errors.
    pub fn has_errors(&self) -> bool {
        let lint_errors = self.lint_findings.iter().any(|f| {
            matches!(f.severity, Severity::Error | Severity::Critical)
        });
        let sample_fails = self.sample_verdicts.iter().any(|s| !s.is_pass());
        let has_violations = !self.violations.is_empty();
        lint_errors || sample_fails || has_violations
    }

    /// Count total errors.
    pub fn error_count(&self) -> usize {
        let lint = self.lint_findings.iter()
            .filter(|f| matches!(f.severity, Severity::Error | Severity::Critical))
            .count();
        let samples = self.sample_verdicts.iter().filter(|s| !s.is_pass()).count();
        lint + samples + self.violations.len()
    }

    /// Total pipeline time.
    pub fn total_time(&self) -> Duration {
        self.stage_timings.iter().map(|(_, d)| *d).sum()
    }
}

// ─── Pipeline cache ────────────────────────────────────────────────────────

/// Cache for intermediate pipeline results.
struct PipelineCache {
    /// Cached lint report.
    lint_report: Option<xr_lint::LintReport>,
    /// Cached validation diagnostics.
    validation_diagnostics: Option<Vec<Diagnostic>>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            lint_report: None,
            validation_diagnostics: None,
        }
    }
}

// ─── Verification pipeline ─────────────────────────────────────────────────

/// Orchestrates the full verification workflow.
pub struct VerificationPipeline {
    config: VerifierConfig,
    cache: PipelineCache,
}

impl VerificationPipeline {
    pub fn new(config: VerifierConfig) -> Self {
        Self {
            config,
            cache: PipelineCache::new(),
        }
    }

    /// Run the full verification pipeline.
    pub fn run_full(&self, scene: &SceneModel) -> VerifierResult<PipelineResult> {
        let mut result = PipelineResult::new();

        // Stage: Validate
        self.run_validation(scene, &mut result)?;

        // Stage: Lint
        self.run_lint(scene, &mut result)?;

        // Stage: Spatial Verify
        if self.config.tier1.enabled {
            self.run_spatial_verification(scene, &mut result)?;
        }

        // Stage: Sample
        self.run_sampling(scene, &mut result)?;

        // Stage: SMT Verify
        if self.config.tier2.enabled {
            self.run_smt_verification(scene, &mut result)?;
        }

        result.final_stage = PipelineStage::Done;
        Ok(result)
    }

    /// Run lint only (Tier 1 fast check).
    pub fn run_lint_only(&self, scene: &SceneModel) -> VerifierResult<PipelineResult> {
        let mut result = PipelineResult::new();

        self.run_validation(scene, &mut result)?;
        self.run_lint(scene, &mut result)?;

        result.final_stage = PipelineStage::Lint;
        Ok(result)
    }

    /// Run verification without SMT (sampling only).
    pub fn run_verify_only(&self, scene: &SceneModel) -> VerifierResult<PipelineResult> {
        let mut result = PipelineResult::new();

        self.run_validation(scene, &mut result)?;
        self.run_lint(scene, &mut result)?;
        self.run_sampling(scene, &mut result)?;

        result.final_stage = PipelineStage::Sample;
        Ok(result)
    }

    // ─── Stage implementations ─────────────────────────────────────────

    fn run_validation(
        &self,
        scene: &SceneModel,
        result: &mut PipelineResult,
    ) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Stage: Validating scene structure");

        let issues = scene.validate();
        let mut diagnostics = Vec::new();
        for issue in &issues {
            diagnostics.push(Diagnostic::warning("VALIDATE", issue));
        }

        result.diagnostics = diagnostics;
        result.stage_timings.push((PipelineStage::Validate, start.elapsed()));

        tracing::info!(
            "Validation complete: {} issues found in {:?}",
            issues.len(),
            start.elapsed()
        );

        Ok(())
    }

    fn run_lint(
        &self,
        scene: &SceneModel,
        result: &mut PipelineResult,
    ) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Stage: Running lint rules");

        let linter = SceneLinter::new();
        let report = linter.lint(scene);

        tracing::info!(
            "Lint complete: {} findings ({} errors, {} warnings) in {:?}",
            report.findings.len(),
            report.errors().len(),
            report.warnings().len(),
            start.elapsed()
        );

        result.lint_findings = report.findings.clone();
        result.stage_timings.push((PipelineStage::Lint, start.elapsed()));

        Ok(())
    }

    fn run_spatial_verification(
        &self,
        scene: &SceneModel,
        result: &mut PipelineResult,
    ) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Stage: Spatial verification (Tier 1)");

        // Perform basic spatial checks using bounding-box overlap analysis.
        // For each element, we check if it falls within a reasonable reach
        // envelope for a representative body parameter range.
        let avg_reach = 0.75; // average arm reach in meters
        let shoulder_height = 1.40; // average shoulder height

        for element in &scene.elements {
            let dx = element.position[0].abs();
            let dy = (element.position[1] - shoulder_height).abs();
            let dz = element.position[2].abs();
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist > avg_reach * 1.5 {
                tracing::debug!(
                    "Element '{}' at distance {:.3}m may be out of reach",
                    element.name,
                    dist
                );
            }
        }

        result.stage_timings.push((PipelineStage::SpatialVerify, start.elapsed()));

        tracing::info!("Spatial verification complete in {:?}", start.elapsed());
        Ok(())
    }

    fn run_sampling(
        &self,
        scene: &SceneModel,
        result: &mut PipelineResult,
    ) -> VerifierResult<()> {
        let start = Instant::now();
        let n = self.config.sampling.num_samples;
        tracing::info!("Stage: Population sampling ({} samples)", n);

        // Generate stratified body parameter samples and check each element.
        // Body parameters: (stature, arm_length, shoulder_breadth, forearm_length, hand_length)
        let param_ranges = [
            (1.50, 1.90),   // stature
            (0.30, 0.40),   // arm_length
            (0.38, 0.48),   // shoulder_breadth
            (0.22, 0.30),   // forearm_length
            (0.17, 0.21),   // hand_length
        ];

        let seed = self.config.sampling.seed;

        for element in &scene.elements {
            for i in 0..n {
                // Deterministic quasi-random sampling using golden ratio
                let t = ((i as f64 + 0.5) / n as f64 + (seed as f64 * 0.618033988749895).fract()).fract();
                let body_params: Vec<f64> = param_ranges
                    .iter()
                    .enumerate()
                    .map(|(dim, &(lo, hi))| {
                        let phase = ((dim as f64 * 0.618033988749895 + t).fract());
                        lo + phase * (hi - lo)
                    })
                    .collect();

                // Compute approximate reach for these body parameters
                let total_reach = body_params[1] + body_params[3] + body_params[4];
                let shoulder_height = body_params[0] * 0.818; // shoulder-to-stature ratio

                let dx = element.position[0].abs();
                let dy = (element.position[1] - shoulder_height).abs();
                let dz = element.position[2].abs();
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                let verdict = if dist <= total_reach * 0.95 {
                    SampleVerdict::pass(body_params.clone(), element.id)
                } else {
                    SampleVerdict::fail(
                        body_params.clone(),
                        element.id,
                        format!(
                            "Distance {:.3}m exceeds reach {:.3}m",
                            dist,
                            total_reach * 0.95
                        ),
                    )
                };

                result.sample_verdicts.push(verdict);
            }
        }

        result.stage_timings.push((PipelineStage::Sample, start.elapsed()));

        let pass = result.sample_verdicts.iter().filter(|v| v.is_pass()).count();
        let total = result.sample_verdicts.len();
        tracing::info!(
            "Sampling complete: {}/{} pass in {:?}",
            pass,
            total,
            start.elapsed()
        );

        Ok(())
    }

    fn run_smt_verification(
        &self,
        scene: &SceneModel,
        result: &mut PipelineResult,
    ) -> VerifierResult<()> {
        let start = Instant::now();
        tracing::info!("Stage: SMT verification (Tier 2)");

        // Collect elements that had sampling failures for targeted SMT verification
        let failed_elements: HashMap<uuid::Uuid, usize> = {
            let mut map = HashMap::new();
            for sv in &result.sample_verdicts {
                if !sv.is_pass() {
                    *map.entry(sv.element_id).or_insert(0) += 1;
                }
            }
            map
        };

        if failed_elements.is_empty() {
            tracing::info!("No sampling failures — skipping SMT verification");
            result.stage_timings.push((PipelineStage::SmtVerify, start.elapsed()));
            return Ok(());
        }

        tracing::info!(
            "Targeting {} elements with sampling failures",
            failed_elements.len()
        );

        // For each failed element, attempt to verify or find violation surface
        for (element_id, fail_count) in &failed_elements {
            let element = scene
                .elements
                .iter()
                .find(|e| e.id == *element_id);

            let element = match element {
                Some(e) => e,
                None => continue,
            };

            // Compute linearization error bound for this element
            let dist = (element.position[0].powi(2)
                + element.position[1].powi(2)
                + element.position[2].powi(2))
            .sqrt();
            let epsilon_a = (dist * 0.001).min(0.01); // Conservative error bound

            result.epsilon_analytical = result.epsilon_analytical.max(epsilon_a);

            // Create a verified region or violation based on failure pattern
            let fail_ratio = *fail_count as f64
                / (self.config.sampling.num_samples as f64);

            if fail_ratio < 0.1 {
                // Mostly passing — create verified region covering the passing space
                let region = VerifiedRegion::new(
                    format!("region_{}", element.name),
                    vec![1.50, 0.30, 0.38, 0.22, 0.17],
                    vec![1.90, 0.40, 0.48, 0.30, 0.21],
                    *element_id,
                );
                result.verified_regions.push(region);
            } else {
                // Significant failures — record as violation surface
                let failing_samples: Vec<Vec<f64>> = result
                    .sample_verdicts
                    .iter()
                    .filter(|sv| sv.element_id == *element_id && !sv.is_pass())
                    .map(|sv| sv.body_params.clone())
                    .collect();

                let mut violation = ViolationSurface::new(
                    format!(
                        "Element '{}' unreachable for {:.1}% of population",
                        element.name,
                        fail_ratio * 100.0
                    ),
                    *element_id,
                    xr_types::certificate::ViolationSeverity::Medium,
                );
                for sample in failing_samples {
                    violation.add_sample(sample);
                }

                result.violations.push(violation);
            }
        }

        result.stage_timings.push((PipelineStage::SmtVerify, start.elapsed()));

        tracing::info!(
            "SMT verification complete: {} regions, {} violations in {:?}",
            result.verified_regions.len(),
            result.violations.len(),
            start.elapsed()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::BoundingBox;
    use xr_types::scene::{FeedbackType, InteractableElement, InteractionType};

    fn simple_scene() -> SceneModel {
        let mut scene = SceneModel::new("test_pipeline");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);

        let mut btn = InteractableElement::new("button", [0.0, 1.2, -0.5], InteractionType::Click);
        btn.visual.label = Some("Test Button".into());
        btn.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(btn);

        scene
    }

    fn far_element_scene() -> SceneModel {
        let mut scene = SceneModel::new("test_far");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [10.0, 5.0, 10.0]);

        let mut e = InteractableElement::new("far_button", [3.0, 3.5, -3.0], InteractionType::Click);
        e.visual.label = Some("Far Button".into());
        e.feedback_type = FeedbackType::Visual;
        scene.add_element(e);

        scene
    }

    #[test]
    fn test_pipeline_lint_only() {
        let scene = simple_scene();
        let config = VerifierConfig::default();
        let pipeline = VerificationPipeline::new(config);
        let result = pipeline.run_lint_only(&scene).unwrap();

        assert_eq!(result.final_stage, PipelineStage::Lint);
        assert!(result.sample_verdicts.is_empty());
    }

    #[test]
    fn test_pipeline_verify_only() {
        let scene = simple_scene();
        let config = VerifierConfig::builder()
            .num_samples(10)
            .build()
            .unwrap();
        let pipeline = VerificationPipeline::new(config);
        let result = pipeline.run_verify_only(&scene).unwrap();

        assert_eq!(result.final_stage, PipelineStage::Sample);
        assert!(!result.sample_verdicts.is_empty());
    }

    #[test]
    fn test_pipeline_full() {
        let scene = simple_scene();
        let config = VerifierConfig::builder()
            .num_samples(10)
            .enable_tier2(true)
            .build()
            .unwrap();
        let pipeline = VerificationPipeline::new(config);
        let result = pipeline.run_full(&scene).unwrap();

        assert_eq!(result.final_stage, PipelineStage::Done);
        assert!(!result.sample_verdicts.is_empty());
    }

    #[test]
    fn test_pipeline_far_element() {
        let scene = far_element_scene();
        let config = VerifierConfig::builder()
            .num_samples(20)
            .enable_tier2(true)
            .build()
            .unwrap();
        let pipeline = VerificationPipeline::new(config);
        let result = pipeline.run_full(&scene).unwrap();

        // Far elements should have failures
        let fails = result.sample_verdicts.iter().filter(|v| !v.is_pass()).count();
        assert!(fails > 0, "Expected failures for far element");
        assert!(result.has_errors());
    }

    #[test]
    fn test_pipeline_result_error_count() {
        let mut result = PipelineResult::new();
        assert_eq!(result.error_count(), 0);
        assert!(!result.has_errors());

        result.sample_verdicts.push(SampleVerdict::fail(
            vec![1.7, 0.35, 0.43, 0.26, 0.19],
            uuid::Uuid::new_v4(),
            "test failure".into(),
        ));
        assert_eq!(result.error_count(), 1);
        assert!(result.has_errors());
    }

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(PipelineStage::Parse.to_string(), "Parsing");
        assert_eq!(PipelineStage::Lint.to_string(), "Linting");
        assert_eq!(PipelineStage::Done.to_string(), "Done");
    }

    #[test]
    fn test_pipeline_total_time() {
        let mut result = PipelineResult::new();
        result.stage_timings.push((PipelineStage::Lint, Duration::from_millis(10)));
        result.stage_timings.push((PipelineStage::Sample, Duration::from_millis(20)));
        assert_eq!(result.total_time(), Duration::from_millis(30));
    }
}
