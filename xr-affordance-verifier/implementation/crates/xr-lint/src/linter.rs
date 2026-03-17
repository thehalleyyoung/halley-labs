//! Accessibility linter engine: context construction, rule orchestration,
//! parallel evaluation, and result aggregation.
//!
//! [`AccessibilityLinter`] is the top-level entry point.  It builds a
//! [`LintContext`] from a scene model and configuration, executes every
//! enabled [`LintRule`](crate::rules::LintRule), collects
//! [`LintDiagnostic`](crate::diagnostics::LintDiagnostic)s, and produces a
//! structured [`LintResult`].

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::error::Severity;
use xr_types::geometry::BoundingBox;
use xr_types::kinematic::{BodyParameterRange, BodyParameters};
use xr_types::device::DeviceConfig;
use xr_types::scene::{
    InteractableElement, InteractionType, SceneModel,
};

use crate::diagnostics::{severity_counts, LintDiagnostic, SeverityCounts};
use crate::rules::LintRule;

// ── Lint context ────────────────────────────────────────────────────────────

/// Shared context passed to every lint rule.
///
/// Contains the scene, body-parameter range, device list, and a simple
/// spatial index so rules can perform fast spatial queries.
pub struct LintContext<'a> {
    /// The scene under inspection.
    pub scene: &'a SceneModel,
    /// Body parameter range for the target population.
    pub body_range: BodyParameterRange,
    /// Computed min/max total reach for the population.
    pub reach_range: (f64, f64),
    /// Computed min/max shoulder height for the population.
    pub shoulder_height_range: (f64, f64),
    /// Device configurations to check against.
    pub devices: &'a [DeviceConfig],
    /// Simple bounding-box spatial index over scene elements.
    pub spatial_index: SpatialIndex,
    /// Lint thresholds.
    pub config: &'a AccessibilityLintConfig,
}

impl<'a> LintContext<'a> {
    /// Build a context from a scene and configuration.
    pub fn build(
        scene: &'a SceneModel,
        config: &'a AccessibilityLintConfig,
    ) -> Self {
        let body_range = config
            .body_range
            .clone()
            .unwrap_or_default();

        let min_reach = body_range.min.total_reach();
        let max_reach = body_range.max.total_reach();
        let min_shoulder = body_range.min.shoulder_height();
        let max_shoulder = body_range.max.shoulder_height();

        let spatial_index = SpatialIndex::from_elements(&scene.elements);

        Self {
            scene,
            body_range,
            reach_range: (min_reach, max_reach),
            shoulder_height_range: (min_shoulder, max_shoulder),
            devices: &scene.devices,
            spatial_index,
            config,
        }
    }

    /// Iterate over scene elements.
    pub fn elements(&self) -> &[InteractableElement] {
        &self.scene.elements
    }

    /// Look up an element by uuid.
    pub fn element_by_id(&self, id: Uuid) -> Option<&InteractableElement> {
        self.scene.elements.iter().find(|e| e.id == id)
    }

    /// Get the Euclidean distance between two element positions.
    pub fn distance_between(&self, a: &InteractableElement, b: &InteractableElement) -> f64 {
        dist3(&a.position, &b.position)
    }

    /// Min population reach radius.
    pub fn min_reach(&self) -> f64 {
        self.reach_range.0
    }

    /// Max population reach radius.
    pub fn max_reach(&self) -> f64 {
        self.reach_range.1
    }

    /// Compute the distance from an element to the nearest shoulder center.
    /// Shoulder center is at (±shoulder_breadth/2, shoulder_height, 0) in
    /// the user-origin frame; we return the minimum over left/right shoulder
    /// for the closest population member.
    pub fn element_distance_to_shoulder(&self, elem: &InteractableElement) -> f64 {
        let sb_half = self.body_range.min.shoulder_breadth / 2.0;
        let sh = self.shoulder_height_range.0; // conservative: smallest person
        let left = [sb_half, sh, 0.0];
        let right = [-sb_half, sh, 0.0];
        dist3(&elem.position, &left).min(dist3(&elem.position, &right))
    }

    /// Check whether any configured device supports a given interaction type.
    pub fn any_device_supports(&self, itype: InteractionType) -> bool {
        self.devices.iter().any(|d| d.supports_interaction(itype))
    }
}

// ── Simple bounding-box spatial index ───────────────────────────────────────

/// A lightweight spatial index storing element AABBs for fast queries.
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    entries: Vec<SpatialEntry>,
}

#[derive(Debug, Clone)]
struct SpatialEntry {
    element_index: usize,
    element_id: Uuid,
    aabb: BoundingBox,
    center: [f64; 3],
}

impl SpatialIndex {
    /// Build from a slice of elements.
    pub fn from_elements(elements: &[InteractableElement]) -> Self {
        let entries = elements
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let aabb = e.activation_volume.bounding_box();
                SpatialEntry {
                    element_index: i,
                    element_id: e.id,
                    aabb,
                    center: aabb.center(),
                }
            })
            .collect();
        Self { entries }
    }

    /// Get all element indices whose AABBs intersect `query`.
    pub fn query_intersecting(&self, query: &BoundingBox) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.aabb.intersects(query))
            .map(|e| e.element_index)
            .collect()
    }

    /// Get the k nearest element indices to `point` (by center distance).
    pub fn nearest_k(&self, point: &[f64; 3], k: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f64)> = self
            .entries
            .iter()
            .map(|e| (e.element_index, dist3_sq(point, &e.center)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().take(k).map(|(idx, _)| *idx).collect()
    }

    /// Get the AABB for element at given index.
    pub fn aabb_of(&self, element_index: usize) -> Option<&BoundingBox> {
        self.entries
            .iter()
            .find(|e| e.element_index == element_index)
            .map(|e| &e.aabb)
    }

    /// Get all pairwise distances (index_a, index_b, distance).
    pub fn pairwise_distances(&self) -> Vec<(usize, usize, f64)> {
        let n = self.entries.len();
        let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist3(&self.entries[i].center, &self.entries[j].center);
                pairs.push((self.entries[i].element_index, self.entries[j].element_index, d));
            }
        }
        pairs
    }
}

// ── Lint configuration ──────────────────────────────────────────────────────

/// Configuration for the accessibility linter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityLintConfig {
    /// Per-rule enable flags (rule_id → enabled).  Rules not listed are enabled.
    pub rule_enables: HashMap<String, bool>,
    /// Per-rule severity overrides.
    pub severity_overrides: HashMap<String, Severity>,
    /// Population body parameter range.
    pub body_range: Option<BodyParameterRange>,
    /// Minimum button/target size in meters (default 0.02).
    pub min_button_size_m: f64,
    /// Minimum spacing between adjacent elements in meters (default 0.01).
    pub min_element_spacing_m: f64,
    /// Maximum interaction dependency depth (default 3).
    pub max_interaction_depth: usize,
    /// Comfort zone height tolerance around shoulder (default ±0.40 m).
    pub comfort_height_tolerance_m: f64,
    /// Maximum gaze angle from forward in radians (default 60° = π/3).
    pub max_gaze_angle_rad: f64,
    /// Whether to run rules in parallel (default true).
    pub parallel: bool,
}

impl Default for AccessibilityLintConfig {
    fn default() -> Self {
        Self {
            rule_enables: HashMap::new(),
            severity_overrides: HashMap::new(),
            body_range: None,
            min_button_size_m: 0.02,
            min_element_spacing_m: 0.01,
            max_interaction_depth: 3,
            comfort_height_tolerance_m: 0.40,
            max_gaze_angle_rad: std::f64::consts::FRAC_PI_3,
            parallel: true,
        }
    }
}

impl AccessibilityLintConfig {
    /// Check if a rule is enabled (defaults to true if not listed).
    pub fn is_rule_enabled(&self, rule_id: &str) -> bool {
        self.rule_enables.get(rule_id).copied().unwrap_or(true)
    }

    /// Get severity override for a rule, if any.
    pub fn severity_for(&self, rule_id: &str) -> Option<Severity> {
        self.severity_overrides.get(rule_id).copied()
    }
}

// ── Lint result ─────────────────────────────────────────────────────────────

/// Result of running the accessibility linter on a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintResult {
    /// Name of the scene that was linted.
    pub scene_name: String,
    /// All diagnostics produced.
    pub diagnostics: Vec<LintDiagnostic>,
    /// Number of elements checked.
    pub elements_checked: usize,
    /// Number of rules evaluated.
    pub rules_evaluated: usize,
    /// Aggregated counts.
    pub counts: SeverityCounts,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: f64,
    /// Per-rule timing in milliseconds.
    pub rule_timings: HashMap<String, f64>,
}

impl LintResult {
    pub fn has_errors(&self) -> bool {
        self.counts.has_errors()
    }

    pub fn score(&self) -> u32 {
        let raw = 100i32
            - (self.counts.critical as i32 * 15)
            - (self.counts.error as i32 * 10)
            - (self.counts.warning as i32 * 2);
        raw.max(0) as u32
    }
}

// ── Accessibility linter ────────────────────────────────────────────────────

/// The main accessibility linter engine.
///
/// Call [`AccessibilityLinter::lint`] to run all enabled rules against a
/// scene and collect diagnostics.
pub struct AccessibilityLinter {
    rules: Vec<Box<dyn LintRule>>,
    config: AccessibilityLintConfig,
}

impl AccessibilityLinter {
    /// Create a linter with the default rule set.
    pub fn new() -> Self {
        Self {
            rules: crate::rules::default_rules(),
            config: AccessibilityLintConfig::default(),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: AccessibilityLintConfig) -> Self {
        Self {
            rules: crate::rules::default_rules(),
            config,
        }
    }

    /// Replace the rule set entirely.
    pub fn with_rules(mut self, rules: Vec<Box<dyn LintRule>>) -> Self {
        self.rules = rules;
        self
    }

    /// Add a single rule.
    pub fn add_rule(&mut self, rule: Box<dyn LintRule>) {
        self.rules.push(rule);
    }

    /// Run all enabled rules against the scene.
    pub fn lint(&self, scene: &SceneModel) -> LintResult {
        let start = Instant::now();
        let ctx = LintContext::build(scene, &self.config);

        let enabled_rules: Vec<&dyn LintRule> = self
            .rules
            .iter()
            .filter(|r| self.config.is_rule_enabled(r.id()))
            .map(|r| r.as_ref())
            .collect();

        let (diagnostics, rule_timings) = if self.config.parallel && enabled_rules.len() > 1 {
            self.run_parallel(&ctx, &enabled_rules)
        } else {
            self.run_sequential(&ctx, &enabled_rules)
        };

        let counts = severity_counts(&diagnostics);

        LintResult {
            scene_name: scene.name.clone(),
            diagnostics,
            elements_checked: scene.elements.len(),
            rules_evaluated: enabled_rules.len(),
            counts,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            rule_timings,
        }
    }

    fn run_sequential(
        &self,
        ctx: &LintContext<'_>,
        rules: &[&dyn LintRule],
    ) -> (Vec<LintDiagnostic>, HashMap<String, f64>) {
        let mut all_diags = Vec::new();
        let mut timings = HashMap::new();
        for rule in rules {
            let t0 = Instant::now();
            let mut diags = rule.check(ctx);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            // Apply severity overrides
            for d in &mut diags {
                if let Some(sev) = self.config.severity_for(rule.id()) {
                    d.severity = sev;
                }
            }
            timings.insert(rule.id().to_string(), elapsed);
            all_diags.extend(diags);
        }
        (all_diags, timings)
    }

    fn run_parallel(
        &self,
        ctx: &LintContext<'_>,
        rules: &[&dyn LintRule],
    ) -> (Vec<LintDiagnostic>, HashMap<String, f64>) {
        let results: Vec<(String, Vec<LintDiagnostic>, f64)> = rules
            .par_iter()
            .map(|rule| {
                let t0 = Instant::now();
                let diags = rule.check(ctx);
                let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
                (rule.id().to_string(), diags, elapsed)
            })
            .collect();

        let mut all_diags = Vec::new();
        let mut timings = HashMap::new();
        for (id, mut diags, elapsed) in results {
            if let Some(sev) = self.config.severity_for(&id) {
                for d in &mut diags {
                    d.severity = sev;
                }
            }
            timings.insert(id, elapsed);
            all_diags.extend(diags);
        }
        (all_diags, timings)
    }
}

impl Default for AccessibilityLinter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Geometry helpers ────────────────────────────────────────────────────────

#[inline]
pub(crate) fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    dist3_sq(a, b).sqrt()
}

#[inline]
pub(crate) fn dist3_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
pub(crate) fn vec3_len(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
pub(crate) fn vec3_dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn vec3_sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn vec3_normalize(v: &[f64; 3]) -> [f64; 3] {
    let l = vec3_len(v);
    if l < 1e-15 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::BoundingBox;
    use xr_types::scene::*;

    fn test_scene() -> SceneModel {
        let mut scene = SceneModel::new("lint_test");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);

        let mut e = InteractableElement::new("good_btn", [0.0, 1.2, -0.5]);
        e.visual.label = Some("OK".into());
        e.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(e);

        let mut e2 = InteractableElement::new("low_btn", [0.3, 0.1, -0.5]);
        e2.visual.label = Some("Low".into());
        e2.feedback_type = FeedbackType::Visual;
        scene.add_element(e2);

        scene
    }

    #[test]
    fn test_lint_context_build() {
        let scene = test_scene();
        let config = AccessibilityLintConfig::default();
        let ctx = LintContext::build(&scene, &config);
        assert_eq!(ctx.elements().len(), 2);
        assert!(ctx.min_reach() > 0.0);
        assert!(ctx.max_reach() > ctx.min_reach());
    }

    #[test]
    fn test_spatial_index_pairwise() {
        let scene = test_scene();
        let idx = SpatialIndex::from_elements(&scene.elements);
        let pairs = idx.pairwise_distances();
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].2 > 0.0);
    }

    #[test]
    fn test_spatial_index_nearest() {
        let scene = test_scene();
        let idx = SpatialIndex::from_elements(&scene.elements);
        let nearest = idx.nearest_k(&[0.0, 1.2, -0.5], 1);
        assert_eq!(nearest.len(), 1);
        assert_eq!(nearest[0], 0);
    }

    #[test]
    fn test_accessibility_linter_runs() {
        let scene = test_scene();
        let linter = AccessibilityLinter::new();
        let result = linter.lint(&scene);
        assert_eq!(result.elements_checked, 2);
        assert!(result.rules_evaluated > 0);
        assert!(result.elapsed_ms >= 0.0);
    }

    #[test]
    fn test_lint_result_score() {
        let result = LintResult {
            scene_name: "test".into(),
            diagnostics: Vec::new(),
            elements_checked: 1,
            rules_evaluated: 1,
            counts: SeverityCounts { critical: 0, error: 1, warning: 2, info: 0 },
            elapsed_ms: 1.0,
            rule_timings: HashMap::new(),
        };
        // 100 - 10*0 - 10*1 - 2*2 = 86
        assert_eq!(result.score(), 86);
    }

    #[test]
    fn test_config_rule_enable() {
        let mut config = AccessibilityLintConfig::default();
        config.rule_enables.insert("R001".into(), false);
        assert!(!config.is_rule_enabled("R001"));
        assert!(config.is_rule_enabled("R002")); // not listed → enabled
    }

    #[test]
    fn test_config_severity_override() {
        let mut config = AccessibilityLintConfig::default();
        config.severity_overrides.insert("R001".into(), Severity::Warning);
        assert_eq!(config.severity_for("R001"), Some(Severity::Warning));
        assert_eq!(config.severity_for("R002"), None);
    }

    #[test]
    fn test_dist3() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert!((dist3(&a, &b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = [3.0, 0.0, 4.0];
        let n = vec3_normalize(&v);
        assert!((vec3_len(&n) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parallel_vs_sequential_same_results() {
        let scene = test_scene();
        let mut config_seq = AccessibilityLintConfig::default();
        config_seq.parallel = false;
        let mut config_par = AccessibilityLintConfig::default();
        config_par.parallel = true;

        let result_seq = AccessibilityLinter::with_config(config_seq).lint(&scene);
        let result_par = AccessibilityLinter::with_config(config_par).lint(&scene);

        assert_eq!(result_seq.diagnostics.len(), result_par.diagnostics.len());
    }
}
