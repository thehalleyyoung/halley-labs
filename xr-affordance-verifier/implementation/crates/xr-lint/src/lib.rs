//! # xr-lint
//!
//! Accessibility linting system for XR scenes. Implements lint rules,
//! diagnostics, and the Tier 1 fast verification engine.
//!
//! # Modules
//!
//! - [`diagnostics`] – Rich diagnostic types with formatting and grouping.
//! - [`linter`] – Accessibility linter engine: context, parallel rule eval, results.
//! - [`rules`] – `LintRule` trait and 12+ concrete rule implementations.
//! - [`reachability`] – Reach-sphere reachability analysis over a population range.
//! - [`report`] – Scored reports with text/JSON/HTML output.
//! - [`tier1_engine`] – Fast interval-arithmetic Green/Yellow/Red classifier.
//! - [`fix_suggestions`] – Automatic fix suggestion engine.

pub mod diagnostics;
pub mod linter;
pub mod rules;
pub mod reachability;
pub mod report;
pub mod tier1_engine;
pub mod fix_suggestions;

use std::collections::HashMap;
use uuid::Uuid;
use xr_types::error::{Diagnostic, DiagnosticCollection, Severity};
use xr_types::scene::{InteractableElement, InteractionType, SceneModel};
use serde::{Deserialize, Serialize};

/// Identifies a specific lint rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LintRuleId {
    /// Element is below minimum height for standing users.
    HeightTooLow,
    /// Element is above maximum comfortable reach.
    HeightTooHigh,
    /// Two elements are too close together.
    ElementSpacingTooSmall,
    /// Activation volume is unreasonably small.
    VolumeTooSmall,
    /// Activation volume is unreasonably large.
    VolumeTooLarge,
    /// Element is outside the scene bounds.
    OutOfBounds,
    /// Dependency creates a cycle.
    DependencyCycle,
    /// Interaction depth exceeds max.
    DepthExceeded,
    /// No device supports the required interaction type.
    UnsupportedInteraction,
    /// Element has no feedback configured.
    MissingFeedback,
    /// Element label is missing for accessibility.
    MissingLabel,
    /// Two-handed interaction has no second-hand affordance.
    TwoHandedIncomplete,
    /// Custom rule.
    Custom(String),
}

impl std::fmt::Display for LintRuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HeightTooLow => write!(f, "E001"),
            Self::HeightTooHigh => write!(f, "E002"),
            Self::ElementSpacingTooSmall => write!(f, "E003"),
            Self::VolumeTooSmall => write!(f, "E004"),
            Self::VolumeTooLarge => write!(f, "E005"),
            Self::OutOfBounds => write!(f, "E006"),
            Self::DependencyCycle => write!(f, "E007"),
            Self::DepthExceeded => write!(f, "E008"),
            Self::UnsupportedInteraction => write!(f, "E009"),
            Self::MissingFeedback => write!(f, "W001"),
            Self::MissingLabel => write!(f, "W002"),
            Self::TwoHandedIncomplete => write!(f, "W003"),
            Self::Custom(s) => write!(f, "C-{s}"),
        }
    }
}

/// Result of a single lint rule applied to a single element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintFinding {
    pub rule: LintRuleId,
    pub element_id: Option<Uuid>,
    pub element_name: Option<String>,
    pub severity: Severity,
    pub message: String,
    pub suggestion: Option<String>,
}

impl LintFinding {
    pub fn error(rule: LintRuleId, message: impl Into<String>) -> Self {
        Self {
            rule,
            element_id: None,
            element_name: None,
            severity: Severity::Error,
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn warning(rule: LintRuleId, message: impl Into<String>) -> Self {
        Self {
            rule,
            element_id: None,
            element_name: None,
            severity: Severity::Warning,
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn with_element(mut self, id: Uuid, name: impl Into<String>) -> Self {
        self.element_id = Some(id);
        self.element_name = Some(name.into());
        self
    }

    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    pub fn to_diagnostic(&self) -> Diagnostic {
        let mut d = match self.severity {
            Severity::Error => Diagnostic::error(self.rule.to_string(), &self.message),
            Severity::Warning => Diagnostic::warning(self.rule.to_string(), &self.message),
            Severity::Critical => Diagnostic::critical(self.rule.to_string(), &self.message),
            Severity::Info => Diagnostic::info(self.rule.to_string(), &self.message),
        };
        if let Some(id) = self.element_id {
            d = d.with_element(id);
        }
        if let Some(ref s) = self.suggestion {
            d = d.with_suggestion(s);
        }
        d
    }
}

/// Configuration for lint thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintConfig {
    pub min_element_height: f64,
    pub max_element_height: f64,
    pub min_element_spacing: f64,
    pub min_activation_volume: f64,
    pub max_activation_volume: f64,
    pub max_interaction_depth: usize,
    pub require_labels: bool,
    pub require_feedback: bool,
    pub disabled_rules: Vec<LintRuleId>,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            min_element_height: 0.4,
            max_element_height: 2.2,
            min_element_spacing: 0.03,
            min_activation_volume: 1e-6,
            max_activation_volume: 100.0,
            max_interaction_depth: 3,
            require_labels: true,
            require_feedback: true,
            disabled_rules: Vec::new(),
        }
    }
}

/// Overall result of running all lint rules on a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    pub scene_name: String,
    pub findings: Vec<LintFinding>,
    pub elements_checked: usize,
    pub rules_applied: usize,
    pub elapsed_ms: f64,
}

impl LintReport {
    pub fn new(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            findings: Vec::new(),
            elements_checked: 0,
            rules_applied: 0,
            elapsed_ms: 0.0,
        }
    }

    pub fn errors(&self) -> Vec<&LintFinding> {
        self.findings
            .iter()
            .filter(|f| matches!(f.severity, Severity::Error | Severity::Critical))
            .collect()
    }

    pub fn warnings(&self) -> Vec<&LintFinding> {
        self.findings
            .iter()
            .filter(|f| matches!(f.severity, Severity::Warning))
            .collect()
    }

    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|f| matches!(f.severity, Severity::Error | Severity::Critical))
    }

    pub fn to_diagnostics(&self) -> DiagnosticCollection {
        let mut coll = DiagnosticCollection::new();
        for f in &self.findings {
            coll.push(f.to_diagnostic());
        }
        coll
    }
}

/// The scene linter: runs all lint rules against a scene.
pub struct SceneLinter {
    config: LintConfig,
}

impl SceneLinter {
    pub fn new() -> Self {
        Self {
            config: LintConfig::default(),
        }
    }

    pub fn with_config(config: LintConfig) -> Self {
        Self { config }
    }

    fn is_enabled(&self, rule: &LintRuleId) -> bool {
        !self.config.disabled_rules.contains(rule)
    }

    /// Run all lint rules against a scene and return a report.
    pub fn lint(&self, scene: &SceneModel) -> LintReport {
        let start = std::time::Instant::now();
        let mut report = LintReport::new(&scene.name);
        let mut rules_applied = 0;

        for (_idx, element) in scene.elements.iter().enumerate() {
            if self.is_enabled(&LintRuleId::HeightTooLow) {
                rules_applied += 1;
                if element.position[1] < self.config.min_element_height {
                    report.findings.push(
                        LintFinding::error(
                            LintRuleId::HeightTooLow,
                            format!(
                                "Element '{}' at height {:.3}m is below minimum {:.2}m",
                                element.name, element.position[1], self.config.min_element_height
                            ),
                        )
                        .with_element(element.id, &element.name)
                        .with_suggestion(format!(
                            "Move element to at least {:.2}m height",
                            self.config.min_element_height
                        )),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::HeightTooHigh) {
                rules_applied += 1;
                if element.position[1] > self.config.max_element_height {
                    report.findings.push(
                        LintFinding::error(
                            LintRuleId::HeightTooHigh,
                            format!(
                                "Element '{}' at height {:.3}m exceeds maximum {:.2}m",
                                element.name, element.position[1], self.config.max_element_height
                            ),
                        )
                        .with_element(element.id, &element.name)
                        .with_suggestion(format!(
                            "Move element to at most {:.2}m height",
                            self.config.max_element_height
                        )),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::VolumeTooSmall) {
                rules_applied += 1;
                let vol = element.activation_volume.approximate_volume();
                if vol < self.config.min_activation_volume {
                    report.findings.push(
                        LintFinding::error(
                            LintRuleId::VolumeTooSmall,
                            format!(
                                "Element '{}' activation volume {:.6} is below minimum {:.6}",
                                element.name, vol, self.config.min_activation_volume
                            ),
                        )
                        .with_element(element.id, &element.name)
                        .with_suggestion("Increase the activation volume size"),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::VolumeTooLarge) {
                rules_applied += 1;
                let vol = element.activation_volume.approximate_volume();
                if vol > self.config.max_activation_volume {
                    report.findings.push(
                        LintFinding::warning(
                            LintRuleId::VolumeTooLarge,
                            format!(
                                "Element '{}' activation volume {:.2} exceeds maximum {:.2}",
                                element.name, vol, self.config.max_activation_volume
                            ),
                        )
                        .with_element(element.id, &element.name),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::OutOfBounds) {
                rules_applied += 1;
                let ebb = element.activation_volume.bounding_box();
                if !scene.bounds.contains_box(&ebb) {
                    report.findings.push(
                        LintFinding::error(
                            LintRuleId::OutOfBounds,
                            format!(
                                "Element '{}' activation volume extends outside scene bounds",
                                element.name
                            ),
                        )
                        .with_element(element.id, &element.name)
                        .with_suggestion("Move element within scene bounds or expand scene bounds"),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::MissingLabel) && self.config.require_labels {
                rules_applied += 1;
                let has_label = element.visual.label.as_ref().map_or(false, |l| !l.is_empty());
                if !has_label {
                    report.findings.push(
                        LintFinding::warning(
                            LintRuleId::MissingLabel,
                            format!("Element '{}' has no accessibility label", element.name),
                        )
                        .with_element(element.id, &element.name)
                        .with_suggestion("Add a descriptive label for screen readers"),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::MissingFeedback) && self.config.require_feedback {
                rules_applied += 1;
                if matches!(element.feedback_type, xr_types::scene::FeedbackType::None) {
                    report.findings.push(
                        LintFinding::warning(
                            LintRuleId::MissingFeedback,
                            format!("Element '{}' has no feedback configured", element.name),
                        )
                        .with_element(element.id, &element.name)
                        .with_suggestion("Add visual, haptic, or audio feedback"),
                    );
                }
            }

            if self.is_enabled(&LintRuleId::TwoHandedIncomplete) {
                rules_applied += 1;
                if element.interaction_type == InteractionType::TwoHanded {
                    if !matches!(
                        element.actuator,
                        xr_types::scene::ActuatorType::BothHands
                    ) {
                        report.findings.push(
                            LintFinding::warning(
                                LintRuleId::TwoHandedIncomplete,
                                format!(
                                    "Element '{}' is TwoHanded but actuator is not BothHands",
                                    element.name
                                ),
                            )
                            .with_element(element.id, &element.name),
                        );
                    }
                }
            }
        }

        // Element spacing check
        if self.is_enabled(&LintRuleId::ElementSpacingTooSmall) {
            rules_applied += 1;
            for i in 0..scene.elements.len() {
                for j in (i + 1)..scene.elements.len() {
                    let a = &scene.elements[i];
                    let b = &scene.elements[j];
                    let dx = a.position[0] - b.position[0];
                    let dy = a.position[1] - b.position[1];
                    let dz = a.position[2] - b.position[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < self.config.min_element_spacing {
                        report.findings.push(
                            LintFinding::error(
                                LintRuleId::ElementSpacingTooSmall,
                                format!(
                                    "Elements '{}' and '{}' are only {:.4}m apart (min {:.3}m)",
                                    a.name, b.name, dist, self.config.min_element_spacing
                                ),
                            )
                            .with_element(a.id, &a.name)
                            .with_suggestion(format!(
                                "Increase spacing to at least {:.3}m",
                                self.config.min_element_spacing
                            )),
                        );
                    }
                }
            }
        }

        // Dependency cycle check
        if self.is_enabled(&LintRuleId::DependencyCycle) {
            rules_applied += 1;
            if !scene.is_dag() {
                report.findings.push(
                    LintFinding::error(
                        LintRuleId::DependencyCycle,
                        "Scene dependency graph contains cycles".to_string(),
                    )
                    .with_suggestion("Remove circular dependencies between elements"),
                );
            }
        }

        // Depth check
        if self.is_enabled(&LintRuleId::DepthExceeded) {
            rules_applied += 1;
            let depth = scene.max_interaction_depth();
            if depth > self.config.max_interaction_depth {
                report.findings.push(
                    LintFinding::error(
                        LintRuleId::DepthExceeded,
                        format!(
                            "Interaction depth {} exceeds maximum {}",
                            depth, self.config.max_interaction_depth
                        ),
                    )
                    .with_suggestion("Reduce the number of sequential dependency steps"),
                );
            }
        }

        // Device support check
        if self.is_enabled(&LintRuleId::UnsupportedInteraction) {
            rules_applied += 1;
            for element in &scene.elements {
                let supported = scene
                    .devices
                    .iter()
                    .any(|d| d.supports_interaction(element.interaction_type));
                if !supported && !scene.devices.is_empty() {
                    report.findings.push(
                        LintFinding::error(
                            LintRuleId::UnsupportedInteraction,
                            format!(
                                "Element '{}' requires {:?} interaction not supported by any configured device",
                                element.name, element.interaction_type
                            ),
                        )
                        .with_element(element.id, &element.name),
                    );
                }
            }
        }

        report.elements_checked = scene.elements.len();
        report.rules_applied = rules_applied;
        report.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        report
    }
}

impl Default for SceneLinter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::{BoundingBox, Volume};
    use xr_types::scene::*;

    fn test_scene() -> SceneModel {
        let mut scene = SceneModel::new("test_scene");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);

        let mut e1 = InteractableElement::new("button_ok", [0.0, 1.2, -0.5]);
        e1.visual.label = Some("OK Button".into());
        e1.feedback_type = FeedbackType::Visual;
        scene.add_element(e1);

        let mut e2 = InteractableElement::new("button_low", [0.3, 0.1, -0.5]);
        e2.visual.label = Some("Low Button".into());
        e2.feedback_type = FeedbackType::Visual;
        scene.add_element(e2);

        scene
    }

    #[test]
    fn test_lint_detects_low_height() {
        let scene = test_scene();
        let linter = SceneLinter::new();
        let report = linter.lint(&scene);
        assert!(report.has_errors());
        let low_findings: Vec<_> = report
            .findings
            .iter()
            .filter(|f| f.rule == LintRuleId::HeightTooLow)
            .collect();
        assert_eq!(low_findings.len(), 1);
        assert!(low_findings[0].message.contains("button_low"));
    }

    #[test]
    fn test_lint_ok_scene() {
        let mut scene = SceneModel::new("good_scene");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);

        let mut e = InteractableElement::new("btn", [0.0, 1.2, -0.5]);
        e.visual.label = Some("Button".into());
        e.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(e);

        let linter = SceneLinter::new();
        let report = linter.lint(&scene);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_lint_report_to_diagnostics() {
        let scene = test_scene();
        let linter = SceneLinter::new();
        let report = linter.lint(&scene);
        let diagnostics = report.to_diagnostics();
        assert!(!diagnostics.is_empty());
    }
}
