//! `LintRule` trait and concrete rule implementations.
//!
//! Each rule examines a [`LintContext`](crate::linter::LintContext) and returns
//! zero or more [`LintDiagnostic`](crate::diagnostics::LintDiagnostic)s.
//! [`default_rules`] returns the standard set of 12 built-in rules.
//!
//! # Adding a custom rule
//!
//! Implement [`LintRule`] on your type, then register it via
//! [`AccessibilityLinter::add_rule`](crate::linter::AccessibilityLinter::add_rule).

use xr_types::error::Severity;
use xr_types::scene::{ActuatorType, FeedbackType, InteractionType};

use crate::diagnostics::LintDiagnostic;
use crate::linter::LintContext;

// ── Trait ───────────────────────────────────────────────────────────────────

/// A single lint rule that can inspect a scene and produce diagnostics.
///
/// Implementations must be `Send + Sync` so that the linter engine can
/// evaluate rules in parallel with Rayon.
pub trait LintRule: Send + Sync {
    /// Machine-readable rule identifier (e.g. `"R001"`).
    fn id(&self) -> &str;

    /// Human-readable rule name.
    fn name(&self) -> &str;

    /// Default severity for diagnostics produced by this rule.
    fn default_severity(&self) -> Severity;

    /// Inspect the scene via `ctx` and return any findings.
    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic>;
}

// ── Rule set ────────────────────────────────────────────────────────────────

/// A named collection of lint rules.
pub struct RuleSet {
    /// Display name for this set.
    pub name: String,
    /// The rules in evaluation order.
    pub rules: Vec<Box<dyn LintRule>>,
}

impl RuleSet {
    /// Create an empty rule set.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rules: Vec::new(),
        }
    }

    /// Add a rule to the set.
    pub fn add(&mut self, rule: Box<dyn LintRule>) {
        self.rules.push(rule);
    }

    /// Number of rules in the set.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// Return the default set of 12 built-in lint rules.
pub fn default_rules() -> Vec<Box<dyn LintRule>> {
    vec![
        Box::new(HeightTooLowRule),
        Box::new(HeightTooHighRule),
        Box::new(ElementSpacingRule),
        Box::new(VolumeTooSmallRule),
        Box::new(VolumeTooLargeRule),
        Box::new(OutOfBoundsRule),
        Box::new(DependencyCycleRule),
        Box::new(DepthExceededRule),
        Box::new(UnsupportedInteractionRule),
        Box::new(MissingFeedbackRule),
        Box::new(MissingLabelRule),
        Box::new(TwoHandedIncompleteRule),
    ]
}

// ── Concrete rules ──────────────────────────────────────────────────────────

// ── R001: HeightTooLow ─────────────────────────────────────────────────────

/// Flags elements whose vertical position is below the minimum comfortable
/// reach for the smallest member of the target population.
pub struct HeightTooLowRule;

impl LintRule for HeightTooLowRule {
    fn id(&self) -> &str { "R001" }
    fn name(&self) -> &str { "height-too-low" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        let min_h = ctx.shoulder_height_range.0 - ctx.config.comfort_height_tolerance_m;
        ctx.elements()
            .iter()
            .filter(|e| e.position[1] < min_h)
            .map(|e| {
                LintDiagnostic::error("R001", format!(
                    "Element '{}' at height {:.3}m is below comfortable minimum {:.2}m",
                    e.name, e.position[1], min_h,
                ))
                .with_element(e.id, &e.name)
                .with_suggestion(format!("Move element to at least {:.2}m height", min_h))
            })
            .collect()
    }
}

// ── R002: HeightTooHigh ────────────────────────────────────────────────────

/// Flags elements whose vertical position exceeds the maximum comfortable
/// reach for the tallest member of the target population.
pub struct HeightTooHighRule;

impl LintRule for HeightTooHighRule {
    fn id(&self) -> &str { "R002" }
    fn name(&self) -> &str { "height-too-high" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        let max_h = ctx.shoulder_height_range.1 + ctx.config.comfort_height_tolerance_m;
        ctx.elements()
            .iter()
            .filter(|e| e.position[1] > max_h)
            .map(|e| {
                LintDiagnostic::error("R002", format!(
                    "Element '{}' at height {:.3}m exceeds comfortable maximum {:.2}m",
                    e.name, e.position[1], max_h,
                ))
                .with_element(e.id, &e.name)
                .with_suggestion(format!("Move element to at most {:.2}m height", max_h))
            })
            .collect()
    }
}

// ── R003: ElementSpacingTooSmall ───────────────────────────────────────────

/// Flags pairs of elements that are closer than the minimum spacing threshold,
/// which can make precise selection difficult.
pub struct ElementSpacingRule;

impl LintRule for ElementSpacingRule {
    fn id(&self) -> &str { "R003" }
    fn name(&self) -> &str { "element-spacing-too-small" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        let min_spacing = ctx.config.min_element_spacing_m;
        let elements = ctx.elements();
        let mut diags = Vec::new();
        for i in 0..elements.len() {
            for j in (i + 1)..elements.len() {
                let d = ctx.distance_between(&elements[i], &elements[j]);
                if d < min_spacing {
                    diags.push(
                        LintDiagnostic::error("R003", format!(
                            "Elements '{}' and '{}' are only {:.4}m apart (min {:.3}m)",
                            elements[i].name, elements[j].name, d, min_spacing,
                        ))
                        .with_element(elements[i].id, &elements[i].name)
                        .with_suggestion(format!(
                            "Increase spacing to at least {:.3}m", min_spacing,
                        )),
                    );
                }
            }
        }
        diags
    }
}

// ── R004: VolumeTooSmall ───────────────────────────────────────────────────

/// Flags elements whose activation volume is smaller than the minimum
/// button/target size, making them hard to activate.
pub struct VolumeTooSmallRule;

impl LintRule for VolumeTooSmallRule {
    fn id(&self) -> &str { "R004" }
    fn name(&self) -> &str { "volume-too-small" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        let min_size = ctx.config.min_button_size_m;
        ctx.elements()
            .iter()
            .filter_map(|e| {
                let vol = e.activation_volume.approximate_volume();
                if vol < min_size {
                    Some(
                        LintDiagnostic::error("R004", format!(
                            "Element '{}' activation volume {:.6} is below minimum {:.6}",
                            e.name, vol, min_size,
                        ))
                        .with_element(e.id, &e.name)
                        .with_suggestion("Increase the activation volume size"),
                    )
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── R005: VolumeTooLarge ───────────────────────────────────────────────────

/// Flags elements whose activation volume is unreasonably large, which may
/// cause unintended activations.
pub struct VolumeTooLargeRule;

impl LintRule for VolumeTooLargeRule {
    fn id(&self) -> &str { "R005" }
    fn name(&self) -> &str { "volume-too-large" }
    fn default_severity(&self) -> Severity { Severity::Warning }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        // Use 100× min_button_size as a heuristic upper bound.
        let max_vol = ctx.config.min_button_size_m * 100.0;
        ctx.elements()
            .iter()
            .filter_map(|e| {
                let vol = e.activation_volume.approximate_volume();
                if vol > max_vol {
                    Some(
                        LintDiagnostic::warning("R005", format!(
                            "Element '{}' activation volume {:.2} exceeds recommended maximum {:.2}",
                            e.name, vol, max_vol,
                        ))
                        .with_element(e.id, &e.name),
                    )
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── R006: OutOfBounds ──────────────────────────────────────────────────────

/// Flags elements whose activation volume extends outside the scene bounds.
pub struct OutOfBoundsRule;

impl LintRule for OutOfBoundsRule {
    fn id(&self) -> &str { "R006" }
    fn name(&self) -> &str { "out-of-bounds" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        ctx.elements()
            .iter()
            .filter_map(|e| {
                let ebb = e.activation_volume.bounding_box();
                if !ctx.scene.bounds.contains_box(&ebb) {
                    Some(
                        LintDiagnostic::error("R006", format!(
                            "Element '{}' activation volume extends outside scene bounds",
                            e.name,
                        ))
                        .with_element(e.id, &e.name)
                        .with_suggestion("Move element within scene bounds or expand scene bounds"),
                    )
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── R007: DependencyCycle ──────────────────────────────────────────────────

/// Flags when the scene dependency graph contains cycles, which would make
/// some interactions impossible to complete.
pub struct DependencyCycleRule;

impl LintRule for DependencyCycleRule {
    fn id(&self) -> &str { "R007" }
    fn name(&self) -> &str { "dependency-cycle" }
    fn default_severity(&self) -> Severity { Severity::Critical }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        if !ctx.scene.is_dag() {
            vec![
                LintDiagnostic::critical("R007", "Scene dependency graph contains cycles")
                    .with_suggestion("Remove circular dependencies between elements"),
            ]
        } else {
            Vec::new()
        }
    }
}

// ── R008: DepthExceeded ────────────────────────────────────────────────────

/// Flags when the maximum interaction dependency depth exceeds the configured
/// limit, indicating an overly complex interaction chain.
pub struct DepthExceededRule;

impl LintRule for DepthExceededRule {
    fn id(&self) -> &str { "R008" }
    fn name(&self) -> &str { "depth-exceeded" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        let depth = ctx.scene.max_interaction_depth();
        if depth > ctx.config.max_interaction_depth {
            vec![
                LintDiagnostic::error("R008", format!(
                    "Interaction depth {} exceeds maximum {}",
                    depth, ctx.config.max_interaction_depth,
                ))
                .with_suggestion("Reduce the number of sequential dependency steps"),
            ]
        } else {
            Vec::new()
        }
    }
}

// ── R009: UnsupportedInteraction ───────────────────────────────────────────

/// Flags elements that require an interaction type not supported by any
/// configured device in the scene.
pub struct UnsupportedInteractionRule;

impl LintRule for UnsupportedInteractionRule {
    fn id(&self) -> &str { "R009" }
    fn name(&self) -> &str { "unsupported-interaction" }
    fn default_severity(&self) -> Severity { Severity::Error }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        if ctx.devices.is_empty() {
            return Vec::new();
        }
        ctx.elements()
            .iter()
            .filter(|e| !ctx.any_device_supports(e.interaction_type))
            .map(|e| {
                LintDiagnostic::error("R009", format!(
                    "Element '{}' requires {:?} interaction not supported by any configured device",
                    e.name, e.interaction_type,
                ))
                .with_element(e.id, &e.name)
            })
            .collect()
    }
}

// ── R010: MissingFeedback ──────────────────────────────────────────────────

/// Flags elements that have no feedback configured, making it impossible for
/// users to know when an interaction has been registered.
pub struct MissingFeedbackRule;

impl LintRule for MissingFeedbackRule {
    fn id(&self) -> &str { "R010" }
    fn name(&self) -> &str { "missing-feedback" }
    fn default_severity(&self) -> Severity { Severity::Warning }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        ctx.elements()
            .iter()
            .filter(|e| matches!(e.feedback_type, FeedbackType::None))
            .map(|e| {
                LintDiagnostic::warning("R010", format!(
                    "Element '{}' has no feedback configured", e.name,
                ))
                .with_element(e.id, &e.name)
                .with_suggestion("Add visual, haptic, or audio feedback")
            })
            .collect()
    }
}

// ── R011: MissingLabel ─────────────────────────────────────────────────────

/// Flags elements that have no accessibility label, which prevents screen
/// readers and assistive technologies from describing them.
pub struct MissingLabelRule;

impl LintRule for MissingLabelRule {
    fn id(&self) -> &str { "R011" }
    fn name(&self) -> &str { "missing-label" }
    fn default_severity(&self) -> Severity { Severity::Warning }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        ctx.elements()
            .iter()
            .filter(|e| e.visual.label.as_ref().map_or(true, |l| l.is_empty()))
            .map(|e| {
                LintDiagnostic::warning("R011", format!(
                    "Element '{}' has no accessibility label", e.name,
                ))
                .with_element(e.id, &e.name)
                .with_suggestion("Add a descriptive label for screen readers")
            })
            .collect()
    }
}

// ── R012: TwoHandedIncomplete ──────────────────────────────────────────────

/// Flags elements that require two-handed interaction but whose actuator is
/// not set to `BothHands`, indicating an incomplete affordance definition.
pub struct TwoHandedIncompleteRule;

impl LintRule for TwoHandedIncompleteRule {
    fn id(&self) -> &str { "R012" }
    fn name(&self) -> &str { "two-handed-incomplete" }
    fn default_severity(&self) -> Severity { Severity::Warning }

    fn check(&self, ctx: &LintContext<'_>) -> Vec<LintDiagnostic> {
        ctx.elements()
            .iter()
            .filter(|e| {
                e.interaction_type == InteractionType::TwoHanded
                    && !matches!(e.actuator, ActuatorType::BothHands)
            })
            .map(|e| {
                LintDiagnostic::warning("R012", format!(
                    "Element '{}' is TwoHanded but actuator is not BothHands",
                    e.name,
                ))
                .with_element(e.id, &e.name)
                .with_suggestion("Set actuator to BothHands for two-handed interactions")
            })
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linter::{AccessibilityLintConfig, LintContext};
    use xr_types::geometry::{BoundingBox, Sphere, Volume};
    use xr_types::scene::*;

    fn test_scene() -> SceneModel {
        let mut scene = SceneModel::new("rule_test");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);

        let mut e = InteractableElement::new("good_btn", [0.0, 1.2, -0.5], InteractionType::Click);
        e.visual.label = Some("OK".into());
        e.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(e);

        scene
    }

    #[test]
    fn default_rules_returns_12() {
        assert_eq!(default_rules().len(), 12);
    }

    #[test]
    fn rule_set_basic() {
        let mut rs = RuleSet::new("test");
        assert!(rs.is_empty());
        rs.add(Box::new(HeightTooLowRule));
        assert_eq!(rs.len(), 1);
    }

    #[test]
    fn height_too_low_triggers() {
        let mut scene = test_scene();
        let mut low = InteractableElement::new("low_btn", [0.0, 0.01, -0.5], InteractionType::Click);
        low.visual.label = Some("Low".into());
        low.feedback_type = FeedbackType::Visual;
        scene.add_element(low);

        let config = AccessibilityLintConfig::default();
        let ctx = LintContext::build(&scene, &config);
        let diags = HeightTooLowRule.check(&ctx);
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("low_btn"));
    }

    #[test]
    fn no_false_positive_on_good_scene() {
        let scene = test_scene();
        let config = AccessibilityLintConfig::default();
        let ctx = LintContext::build(&scene, &config);
        let diags = HeightTooHighRule.check(&ctx);
        assert!(diags.is_empty());
    }

    #[test]
    fn missing_label_triggers() {
        let mut scene = test_scene();
        let mut unlabeled = InteractableElement::new("no_label", [0.5, 1.2, -0.5], InteractionType::Click);
        unlabeled.feedback_type = FeedbackType::Visual;
        scene.add_element(unlabeled);

        let config = AccessibilityLintConfig::default();
        let ctx = LintContext::build(&scene, &config);
        let diags = MissingLabelRule.check(&ctx);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("no_label"));
    }

    #[test]
    fn missing_feedback_triggers() {
        let mut scene = test_scene();
        let mut nofb = InteractableElement::new("no_fb", [0.5, 1.2, -0.5], InteractionType::Click);
        nofb.visual.label = Some("NF".into());
        nofb.feedback_type = FeedbackType::None;
        scene.add_element(nofb);

        let config = AccessibilityLintConfig::default();
        let ctx = LintContext::build(&scene, &config);
        let diags = MissingFeedbackRule.check(&ctx);
        assert_eq!(diags.len(), 1);
    }

    #[test]
    fn dependency_cycle_clean() {
        let scene = test_scene();
        let config = AccessibilityLintConfig::default();
        let ctx = LintContext::build(&scene, &config);
        let diags = DependencyCycleRule.check(&ctx);
        assert!(diags.is_empty());
    }
}
