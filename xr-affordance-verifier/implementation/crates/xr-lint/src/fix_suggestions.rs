//! Fix suggestion engine: analyse lint failures and propose concrete repairs.
//!
//! [`FixSuggestionEngine`] takes a list of diagnostics (and optionally a scene)
//! and generates [`SuggestedFix`]es with descriptions, confidence scores, and
//! deltas for each fixable issue.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::geometry::Volume;
use xr_types::kinematic::BodyParameterRange;
use xr_types::scene::{InteractableElement, SceneModel};

use crate::diagnostics::LintDiagnostic;
use crate::linter::dist3;

// ── Suggested fix ───────────────────────────────────────────────────────────

/// A concrete fix suggestion for a lint diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedFix {
    /// Which diagnostic this fix addresses (code + element id).
    pub diagnostic_code: String,
    /// Element id, if element-level.
    pub element_id: Option<Uuid>,
    /// Human-readable description of the fix.
    pub description: String,
    /// Confidence that this fix resolves the issue (0.0–1.0).
    pub confidence: f64,
    /// The property being changed.
    pub property: String,
    /// Value delta (signed, in the property's unit).
    pub delta: FixDelta,
    /// Category of fix.
    pub category: FixCategory,
}

/// The numeric or semantic delta of a fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixDelta {
    /// Move a position by [dx, dy, dz] metres.
    PositionShift([f64; 3]),
    /// Scale an activation volume by this factor.
    VolumeScale(f64),
    /// Change a discrete property to a new value.
    PropertyChange { from: String, to: String },
    /// Add a new property or component.
    AddComponent(String),
    /// Compound: multiple sub-fixes.
    Compound(Vec<FixDelta>),
}

/// Category of a fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FixCategory {
    /// Move element closer / higher / lower.
    Reposition,
    /// Enlarge or reshape activation volume.
    ResizeVolume,
    /// Change interaction type to one supported by devices.
    ChangeInteraction,
    /// Add feedback modality.
    AddFeedback,
    /// Adjust spacing between elements.
    AdjustSpacing,
    /// Other / compound.
    Other,
}

// ── Engine ──────────────────────────────────────────────────────────────────

/// Analyse diagnostics and produce fix suggestions.
pub struct FixSuggestionEngine {
    body_range: BodyParameterRange,
}

impl FixSuggestionEngine {
    pub fn new() -> Self {
        Self {
            body_range: BodyParameterRange::default(),
        }
    }

    pub fn with_body_range(range: BodyParameterRange) -> Self {
        Self { body_range: range }
    }

    /// Generate fixes for a list of diagnostics, optionally referencing the
    /// original scene for positional context.
    pub fn suggest(
        &self,
        diagnostics: &[LintDiagnostic],
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let mut fixes = Vec::new();
        for d in diagnostics {
            fixes.extend(self.suggest_one(d, scene));
        }
        // De-dup: keep highest-confidence fix per (code, element_id)
        fixes.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = std::collections::HashSet::new();
        fixes.retain(|f| seen.insert((f.diagnostic_code.clone(), f.element_id)));
        fixes
    }

    fn suggest_one(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        match d.code.as_str() {
            "R001" | "R002" => self.suggest_reachability_fix(d, scene),
            "R003" => self.suggest_button_size_fix(d, scene),
            "R004" => self.suggest_spacing_fix(d, scene),
            "R005" | "R005-cycle" => self.suggest_depth_fix(d),
            "R006" => self.suggest_interaction_fix(d),
            "R007" | "R008" => self.suggest_comfort_fix(d, scene),
            "R009" => self.suggest_horizontal_move(d, scene),
            "R010" | "R011" => self.suggest_seated_fix(d, scene),
            "R015" | "R016" => self.suggest_feedback_fix(d),
            "R017" | "R018" => self.suggest_volume_fix(d, scene),
            _ => Vec::new(),
        }
    }

    // ── Per-rule fix generators ─────────────────────────────────────────

    fn suggest_reachability_fix(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();

        if let Some(elem) = elem {
            let max_reach = self.body_range.max.total_reach();
            let sh = self.body_range.max.shoulder_height();
            let half_sb = self.body_range.max.shoulder_breadth / 2.0;

            // Compute ideal position: move towards nearest shoulder
            let left_sh = [half_sb, sh, 0.0];
            let right_sh = [-half_sb, sh, 0.0];
            let dl = dist3(&elem.position, &left_sh);
            let dr = dist3(&elem.position, &right_sh);
            let target_sh = if dl <= dr { left_sh } else { right_sh };

            // Direction from element toward shoulder
            let delta = [
                target_sh[0] - elem.position[0],
                target_sh[1] - elem.position[1],
                target_sh[2] - elem.position[2],
            ];
            let d_len = dist3(&elem.position, &target_sh);
            if d_len > max_reach && d_len > 1e-9 {
                let move_dist = d_len - max_reach * 0.9;
                let scale = move_dist / d_len;
                let shift = [delta[0] * scale, delta[1] * scale, delta[2] * scale];
                fixes.push(SuggestedFix {
                    diagnostic_code: d.code.clone(),
                    element_id: d.element_id,
                    description: format!(
                        "Move '{}' {:.3}m toward user center",
                        elem.name, move_dist
                    ),
                    confidence: 0.8,
                    property: "position".into(),
                    delta: FixDelta::PositionShift(shift),
                    category: FixCategory::Reposition,
                });
            }

            // Alternative: enlarge activation volume
            let vol = elem.activation_volume.approximate_volume();
            if vol > 0.0 {
                let needed_radius_increase = (d_len - max_reach).max(0.0);
                let scale_factor = 1.0 + (needed_radius_increase / 0.1).min(3.0);
                fixes.push(SuggestedFix {
                    diagnostic_code: d.code.clone(),
                    element_id: d.element_id,
                    description: format!(
                        "Enlarge activation volume by {:.1}× to extend reachable boundary",
                        scale_factor
                    ),
                    confidence: 0.4,
                    property: "activation_volume".into(),
                    delta: FixDelta::VolumeScale(scale_factor),
                    category: FixCategory::ResizeVolume,
                });
            }
        }

        fixes
    }

    fn suggest_button_size_fix(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();

        if let Some(elem) = elem {
            let bb = elem.activation_volume.bounding_box();
            let extents = bb.extents();
            let min_ext = extents[0].min(extents[1]).min(extents[2]);
            let target_size = 0.02; // 2cm
            if min_ext > 0.0 {
                let scale = target_size / min_ext;
                fixes.push(SuggestedFix {
                    diagnostic_code: d.code.clone(),
                    element_id: d.element_id,
                    description: format!(
                        "Scale activation volume by {:.2}× to meet minimum size",
                        scale.max(1.0)
                    ),
                    confidence: 0.9,
                    property: "activation_volume".into(),
                    delta: FixDelta::VolumeScale(scale.max(1.0)),
                    category: FixCategory::ResizeVolume,
                });
            }
        }

        fixes
    }

    fn suggest_spacing_fix(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();

        if let Some(elem) = elem {
            let other_name = d.context.get("other_element").cloned().unwrap_or_default();
            let other = scene.and_then(|s| s.elements.iter().find(|e| e.name == other_name));

            if let Some(other) = other {
                let dist = dist3(&elem.position, &other.position);
                let min_spacing = 0.01;
                let needed = min_spacing - dist;
                if needed > 0.0 {
                    let dir = [
                        elem.position[0] - other.position[0],
                        elem.position[1] - other.position[1],
                        elem.position[2] - other.position[2],
                    ];
                    let len = dist.max(1e-9);
                    let shift = [
                        dir[0] / len * (needed + 0.005),
                        dir[1] / len * (needed + 0.005),
                        dir[2] / len * (needed + 0.005),
                    ];
                    fixes.push(SuggestedFix {
                        diagnostic_code: d.code.clone(),
                        element_id: d.element_id,
                        description: format!(
                            "Move '{}' {:.4}m away from '{}'",
                            elem.name,
                            needed + 0.005,
                            other_name
                        ),
                        confidence: 0.85,
                        property: "position".into(),
                        delta: FixDelta::PositionShift(shift),
                        category: FixCategory::AdjustSpacing,
                    });
                }
            } else {
                fixes.push(SuggestedFix {
                    diagnostic_code: d.code.clone(),
                    element_id: d.element_id,
                    description: "Increase spacing between overlapping elements".into(),
                    confidence: 0.5,
                    property: "position".into(),
                    delta: FixDelta::PositionShift([0.01, 0.0, 0.0]),
                    category: FixCategory::AdjustSpacing,
                });
            }
        }

        fixes
    }

    fn suggest_depth_fix(&self, d: &LintDiagnostic) -> Vec<SuggestedFix> {
        vec![SuggestedFix {
            diagnostic_code: d.code.clone(),
            element_id: None,
            description: "Flatten the dependency hierarchy by adding shortcut interactions".into(),
            confidence: 0.3,
            property: "dependencies".into(),
            delta: FixDelta::PropertyChange {
                from: "deep chain".into(),
                to: "flattened".into(),
            },
            category: FixCategory::Other,
        }]
    }

    fn suggest_interaction_fix(&self, d: &LintDiagnostic) -> Vec<SuggestedFix> {
        vec![SuggestedFix {
            diagnostic_code: d.code.clone(),
            element_id: d.element_id,
            description: "Change to a device-compatible interaction type (e.g. Click or Grab)".into(),
            confidence: 0.6,
            property: "interaction_type".into(),
            delta: FixDelta::PropertyChange {
                from: "unsupported".into(),
                to: "Click".into(),
            },
            category: FixCategory::ChangeInteraction,
        }]
    }

    fn suggest_comfort_fix(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();
        if let Some(elem) = elem {
            let sh_min = self.body_range.min.shoulder_height();
            let sh_max = self.body_range.max.shoulder_height();
            let target_y = (sh_min + sh_max) / 2.0; // middle of comfort zone
            let dy = target_y - elem.position[1];
            fixes.push(SuggestedFix {
                diagnostic_code: d.code.clone(),
                element_id: d.element_id,
                description: format!(
                    "Move '{}' to shoulder height ({:.3}m)",
                    elem.name, target_y
                ),
                confidence: 0.75,
                property: "position.y".into(),
                delta: FixDelta::PositionShift([0.0, dy, 0.0]),
                category: FixCategory::Reposition,
            });
        }
        fixes
    }

    fn suggest_horizontal_move(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();
        if let Some(elem) = elem {
            let comfort_dist = self.body_range.max.total_reach() * 0.6;
            let horiz = (elem.position[0].powi(2) + elem.position[2].powi(2)).sqrt();
            if horiz > 1e-9 {
                let scale = comfort_dist / horiz;
                let dx = elem.position[0] * (scale - 1.0);
                let dz = elem.position[2] * (scale - 1.0);
                fixes.push(SuggestedFix {
                    diagnostic_code: d.code.clone(),
                    element_id: d.element_id,
                    description: format!(
                        "Move '{}' closer horizontally to {:.3}m",
                        elem.name, comfort_dist
                    ),
                    confidence: 0.65,
                    property: "position".into(),
                    delta: FixDelta::PositionShift([dx, 0.0, dz]),
                    category: FixCategory::Reposition,
                });
            }
        }
        fixes
    }

    fn suggest_seated_fix(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();
        if let Some(elem) = elem {
            let seated_sh = self.body_range.min.stature * 0.52;
            let dy = seated_sh - elem.position[1];
            fixes.push(SuggestedFix {
                diagnostic_code: d.code.clone(),
                element_id: d.element_id,
                description: format!(
                    "Move '{}' to seated shoulder height ({:.3}m)",
                    elem.name, seated_sh
                ),
                confidence: 0.7,
                property: "position.y".into(),
                delta: FixDelta::PositionShift([0.0, dy, 0.0]),
                category: FixCategory::Reposition,
            });
        }
        fixes
    }

    fn suggest_feedback_fix(&self, d: &LintDiagnostic) -> Vec<SuggestedFix> {
        let desc = if d.code == "R015" {
            "Add at least visual and haptic feedback"
        } else {
            "Add haptic or audio feedback alongside visual"
        };
        vec![SuggestedFix {
            diagnostic_code: d.code.clone(),
            element_id: d.element_id,
            description: desc.into(),
            confidence: 0.9,
            property: "feedback_type".into(),
            delta: FixDelta::PropertyChange {
                from: "Visual".into(),
                to: "VisualHaptic".into(),
            },
            category: FixCategory::AddFeedback,
        }]
    }

    fn suggest_volume_fix(
        &self,
        d: &LintDiagnostic,
        scene: Option<&SceneModel>,
    ) -> Vec<SuggestedFix> {
        let elem = self.find_element(d, scene);
        let mut fixes = Vec::new();
        if let Some(elem) = elem {
            let vol = elem.activation_volume.approximate_volume();
            let min_vol = 1e-7;
            if vol < min_vol {
                let target_size: f64 = 0.02; // 2cm cube
                fixes.push(SuggestedFix {
                    diagnostic_code: d.code.clone(),
                    element_id: d.element_id,
                    description: format!(
                        "Set activation volume to at least {:.3}m cube",
                        target_size
                    ),
                    confidence: 0.85,
                    property: "activation_volume".into(),
                    delta: FixDelta::VolumeScale(if vol > 1e-15 {
                        target_size.powi(3) / vol
                    } else {
                        1.0
                    }),
                    category: FixCategory::ResizeVolume,
                });
            }
        }
        fixes
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn find_element<'a>(
        &self,
        d: &LintDiagnostic,
        scene: Option<&'a SceneModel>,
    ) -> Option<&'a InteractableElement> {
        let scene = scene?;
        let id = d.element_id?;
        scene.elements.iter().find(|e| e.id == id)
    }
}

impl Default for FixSuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::BoundingBox;
    use xr_types::scene::*;

    fn make_scene() -> SceneModel {
        let mut s = SceneModel::new("fix_test");
        s.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);

        let mut e = InteractableElement::new("far_btn", [0.0, 1.2, -3.0]);
        e.feedback_type = FeedbackType::Visual;
        s.add_element(e);

        let mut e2 = InteractableElement::new("tiny_btn", [0.3, 1.2, -0.5]);
        e2.activation_volume = Volume::Box(BoundingBox::from_center_extents(
            [0.3, 1.2, -0.5],
            [0.005, 0.005, 0.005],
        ));
        e2.feedback_type = FeedbackType::VisualHaptic;
        s.add_element(e2);

        s
    }

    #[test]
    fn test_reachability_fix() {
        let scene = make_scene();
        let d = LintDiagnostic::error("R001", "unreachable")
            .with_element(scene.elements[0].id, "far_btn");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], Some(&scene));
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|f| f.category == FixCategory::Reposition));
    }

    #[test]
    fn test_button_size_fix() {
        let scene = make_scene();
        let d = LintDiagnostic::error("R003", "too small")
            .with_element(scene.elements[1].id, "tiny_btn");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], Some(&scene));
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|f| f.category == FixCategory::ResizeVolume));
    }

    #[test]
    fn test_feedback_fix() {
        let d = LintDiagnostic::warning("R016", "visual only");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], None);
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|f| f.category == FixCategory::AddFeedback));
    }

    #[test]
    fn test_depth_fix() {
        let d = LintDiagnostic::warning("R005", "depth exceeded");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], None);
        assert!(!fixes.is_empty());
    }

    #[test]
    fn test_interaction_fix() {
        let d = LintDiagnostic::error("R006", "unsupported interaction");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], None);
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|f| f.category == FixCategory::ChangeInteraction));
    }

    #[test]
    fn test_comfort_fix() {
        let scene = make_scene();
        let d = LintDiagnostic::warning("R007", "too low")
            .with_element(scene.elements[0].id, "far_btn");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], Some(&scene));
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|f| f.category == FixCategory::Reposition));
    }

    #[test]
    fn test_no_fixes_for_unknown_code() {
        let d = LintDiagnostic::info("I999", "informational");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], None);
        assert!(fixes.is_empty());
    }

    #[test]
    fn test_dedup_fixes() {
        let scene = make_scene();
        let d1 = LintDiagnostic::error("R001", "unreachable")
            .with_element(scene.elements[0].id, "far_btn");
        let d2 = LintDiagnostic::error("R001", "unreachable (duplicate)")
            .with_element(scene.elements[0].id, "far_btn");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d1, d2], Some(&scene));
        // Should be de-duplicated: only one fix per (code, element_id)
        let r001_fixes: Vec<_> = fixes
            .iter()
            .filter(|f| f.diagnostic_code == "R001" && f.element_id == Some(scene.elements[0].id))
            .collect();
        assert_eq!(r001_fixes.len(), 1);
    }

    #[test]
    fn test_fix_confidence_range() {
        let scene = make_scene();
        let d = LintDiagnostic::error("R001", "unreachable")
            .with_element(scene.elements[0].id, "far_btn");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], Some(&scene));
        for fix in &fixes {
            assert!(
                fix.confidence >= 0.0 && fix.confidence <= 1.0,
                "Confidence {} out of range for {}",
                fix.confidence,
                fix.description
            );
        }
    }

    #[test]
    fn test_spacing_fix() {
        let mut scene = SceneModel::new("spacing_test");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);
        let mut e1 = InteractableElement::new("btn_a", [0.0, 1.2, -0.5]);
        e1.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(e1);
        let mut e2 = InteractableElement::new("btn_b", [0.005, 1.2, -0.5]);
        e2.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(e2);

        let d = LintDiagnostic::error("R004", "too close")
            .with_element(scene.elements[0].id, "btn_a")
            .with_context("other_element", "btn_b");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], Some(&scene));
        assert!(fixes.iter().any(|f| f.category == FixCategory::AdjustSpacing));
    }

    #[test]
    fn test_volume_fix() {
        let mut scene = SceneModel::new("vol_test");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);
        let mut e = InteractableElement::new("tiny", [0.0, 1.2, -0.5]);
        e.activation_volume = Volume::Box(BoundingBox::new(
            [0.0, 1.2, -0.5],
            [0.0001, 1.2001, -0.4999],
        ));
        e.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(e);

        let d = LintDiagnostic::error("R018", "volume too small")
            .with_element(scene.elements[0].id, "tiny");
        let engine = FixSuggestionEngine::new();
        let fixes = engine.suggest(&[d], Some(&scene));
        assert!(fixes.iter().any(|f| f.category == FixCategory::ResizeVolume));
    }
}
