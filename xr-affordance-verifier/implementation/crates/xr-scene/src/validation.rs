//! Scene validation for structural and semantic correctness.

use std::collections::HashSet;
use xr_types::error::Diagnostic;
use xr_types::geometry::{BoundingBox, point_distance};
use xr_types::scene::{DependencyType, InteractionType, SceneModel};
use xr_types::device::DeviceConfig;
use crate::graph::SceneGraph;
use crate::spatial_index::{RTreeEntry, SpatialIndex};
use crate::transform::TransformHierarchy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity { Info, Warning, Error, Critical }

#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: ValidationSeverity,
    pub code: String,
    pub message: String,
    pub element_indices: Vec<usize>,
}

impl ValidationFinding {
    pub fn info(code: impl Into<String>, msg: impl Into<String>) -> Self { Self { severity: ValidationSeverity::Info, code: code.into(), message: msg.into(), element_indices: vec![] } }
    pub fn warning(code: impl Into<String>, msg: impl Into<String>) -> Self { Self { severity: ValidationSeverity::Warning, code: code.into(), message: msg.into(), element_indices: vec![] } }
    pub fn error(code: impl Into<String>, msg: impl Into<String>) -> Self { Self { severity: ValidationSeverity::Error, code: code.into(), message: msg.into(), element_indices: vec![] } }
    pub fn critical(code: impl Into<String>, msg: impl Into<String>) -> Self { Self { severity: ValidationSeverity::Critical, code: code.into(), message: msg.into(), element_indices: vec![] } }
    pub fn with_elements(mut self, indices: Vec<usize>) -> Self { self.element_indices = indices; self }
}

#[derive(Debug, Clone)]
pub struct ValidationReport { pub findings: Vec<ValidationFinding> }

impl ValidationReport {
    pub fn new() -> Self { Self { findings: Vec::new() } }
    pub fn push(&mut self, finding: ValidationFinding) { self.findings.push(finding); }
    pub fn is_valid(&self) -> bool { !self.findings.iter().any(|f| f.severity >= ValidationSeverity::Error) }
    pub fn error_count(&self) -> usize { self.findings.iter().filter(|f| f.severity >= ValidationSeverity::Error).count() }
    pub fn warning_count(&self) -> usize { self.findings.iter().filter(|f| f.severity == ValidationSeverity::Warning).count() }
    pub fn by_severity(&self, severity: ValidationSeverity) -> Vec<&ValidationFinding> { self.findings.iter().filter(|f| f.severity == severity).collect() }
    pub fn for_element(&self, index: usize) -> Vec<&ValidationFinding> { self.findings.iter().filter(|f| f.element_indices.contains(&index)).collect() }

    pub fn to_diagnostics(&self) -> Vec<Diagnostic> {
        self.findings.iter().map(|f| {
            let diag = match f.severity {
                ValidationSeverity::Info => Diagnostic::info(&f.code, &f.message),
                ValidationSeverity::Warning => Diagnostic::warning(&f.code, &f.message),
                ValidationSeverity::Error => Diagnostic::error(&f.code, &f.message),
                ValidationSeverity::Critical => Diagnostic::critical(&f.code, &f.message),
            };
            if let Some(&idx) = f.element_indices.first() { diag.with_context(format!("element index {}", idx)) } else { diag }
        }).collect()
    }
}

impl Default for ValidationReport { fn default() -> Self { Self::new() } }

pub struct SceneValidator {
    pub check_bounds: bool,
    pub check_volumes: bool,
    pub check_dependencies: bool,
    pub check_device_compatibility: bool,
    pub check_overlapping: bool,
    pub check_transforms: bool,
    pub max_scene_extent: f64,
    pub min_volume_size: f64,
    pub min_element_spacing: f64,
}

impl SceneValidator {
    pub fn new() -> Self {
        Self { check_bounds: true, check_volumes: true, check_dependencies: true,
               check_device_compatibility: true, check_overlapping: true, check_transforms: true,
               max_scene_extent: 100.0, min_volume_size: 1e-6, min_element_spacing: 0.005 }
    }

    pub fn with_max_extent(mut self, max: f64) -> Self { self.max_scene_extent = max; self }
    pub fn with_min_volume(mut self, min: f64) -> Self { self.min_volume_size = min; self }
    pub fn with_min_spacing(mut self, min: f64) -> Self { self.min_element_spacing = min; self }

    pub fn validate(&self, scene: &SceneModel) -> ValidationReport {
        let mut report = ValidationReport::new();
        if scene.elements.is_empty() { report.push(ValidationFinding::warning("V001", "Scene has no interactable elements")); }
        if scene.devices.is_empty() { report.push(ValidationFinding::warning("V002", "Scene has no target devices configured")); }
        if self.check_bounds { self.validate_bounds(scene, &mut report); }
        if self.check_volumes { self.validate_volumes(scene, &mut report); }
        if self.check_dependencies { self.validate_dependencies(scene, &mut report); }
        if self.check_device_compatibility { self.validate_device_compatibility(scene, &mut report); }
        if self.check_overlapping { self.validate_overlapping(scene, &mut report); }
        if self.check_transforms { self.validate_transforms(scene, &mut report); }
        report
    }

    fn validate_bounds(&self, scene: &SceneModel, report: &mut ValidationReport) {
        let scene_bounds = BoundingBox::from_center_extents([0.0; 3], [self.max_scene_extent; 3]);
        for (i, elem) in scene.elements.iter().enumerate() {
            if !scene_bounds.contains_point(&elem.position) {
                report.push(ValidationFinding::error("V100", format!("Element '{}' position {:?} is outside scene bounds", elem.name, elem.position)).with_elements(vec![i]));
            }
            for (axis, &val) in elem.position.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
                    report.push(ValidationFinding::critical("V101", format!("Element '{}' has non-finite position on axis {}", elem.name, axis)).with_elements(vec![i]));
                }
            }
        }
    }

    fn validate_volumes(&self, scene: &SceneModel, report: &mut ValidationReport) {
        for (i, elem) in scene.elements.iter().enumerate() {
            let vol = elem.activation_volume.approximate_volume();
            if vol < self.min_volume_size {
                report.push(ValidationFinding::warning("V200", format!("Element '{}' has very small activation volume ({:.2e})", elem.name, vol)).with_elements(vec![i]));
            }
            for (axis, &s) in elem.scale.iter().enumerate() {
                if s.abs() < 1e-8 {
                    report.push(ValidationFinding::error("V201", format!("Element '{}' has near-zero scale on axis {}", elem.name, axis)).with_elements(vec![i]));
                }
                if s < 0.0 {
                    report.push(ValidationFinding::warning("V202", format!("Element '{}' has negative scale on axis {}", elem.name, axis)).with_elements(vec![i]));
                }
            }
            let q = elem.orientation;
            let q_len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if (q_len - 1.0).abs() > 0.01 {
                report.push(ValidationFinding::warning("V203", format!("Element '{}' has non-unit quaternion (length {:.4})", elem.name, q_len)).with_elements(vec![i]));
            }
        }
    }

    fn validate_dependencies(&self, scene: &SceneModel, report: &mut ValidationReport) {
        let n = scene.elements.len();
        for (i, dep) in scene.dependencies.iter().enumerate() {
            if dep.source_index >= n { report.push(ValidationFinding::error("V300", format!("Dependency {} has invalid source index {}", i, dep.source_index))); }
            if dep.target_index >= n { report.push(ValidationFinding::error("V301", format!("Dependency {} has invalid target index {}", i, dep.target_index))); }
            if dep.source_index == dep.target_index {
                report.push(ValidationFinding::error("V302", format!("Dependency {} is a self-loop on element {}", i, dep.source_index)).with_elements(vec![dep.source_index]));
            }
        }
        let graph = SceneGraph::from_scene(scene);
        if !graph.is_dag() {
            if let Some(cycle) = graph.find_cycle() {
                report.push(ValidationFinding::error("V310", format!("Dependency graph contains a cycle through elements: {:?}", cycle)).with_elements(cycle));
            } else {
                report.push(ValidationFinding::error("V310", "Dependency graph contains cycles"));
            }
        }
        let depth = scene.max_interaction_depth();
        if depth > xr_types::MAX_INTERACTION_DEPTH {
            report.push(ValidationFinding::warning("V311", format!("Interaction depth {} exceeds recommended maximum {}", depth, xr_types::MAX_INTERACTION_DEPTH)));
        }
    }

    fn validate_device_compatibility(&self, scene: &SceneModel, report: &mut ValidationReport) {
        for device in &scene.devices {
            for (i, elem) in scene.elements.iter().enumerate() {
                if !device.supports_interaction(elem.interaction_type) {
                    report.push(ValidationFinding::warning("V400", format!("Element '{}' uses {:?} not supported by '{}'", elem.name, elem.interaction_type, device.name)).with_elements(vec![i]));
                }
                if !device.is_tracked(&elem.position) {
                    report.push(ValidationFinding::warning("V401", format!("Element '{}' at {:?} is outside tracking volume of '{}'", elem.name, elem.position, device.name)).with_elements(vec![i]));
                }
            }
        }
    }

    fn validate_overlapping(&self, scene: &SceneModel, report: &mut ValidationReport) {
        if scene.elements.len() < 2 { return; }
        let mut index = SpatialIndex::new();
        for (i, elem) in scene.elements.iter().enumerate() {
            index.insert(RTreeEntry::new(i, elem.activation_volume.bounding_box()));
        }
        let pairs = index.find_overlapping_pairs();
        for (a, b) in pairs {
            let dist = point_distance(&scene.elements[a].position, &scene.elements[b].position);
            if dist < self.min_element_spacing {
                report.push(ValidationFinding::warning("V500", format!("Elements '{}' and '{}' are very close (distance: {:.4}m)", scene.elements[a].name, scene.elements[b].name, dist)).with_elements(vec![a, b]));
            }
        }
    }

    fn validate_transforms(&self, scene: &SceneModel, report: &mut ValidationReport) {
        if scene.transform_nodes.is_empty() { return; }
        let hierarchy = TransformHierarchy::from_scene(scene);
        for err in hierarchy.validate() { report.push(ValidationFinding::error("V600", err)); }
        for (i, elem) in scene.elements.iter().enumerate() {
            if let Some(tn_idx) = elem.transform_node {
                if tn_idx >= scene.transform_nodes.len() {
                    report.push(ValidationFinding::error("V601", format!("Element '{}' references invalid transform node {}", elem.name, tn_idx)).with_elements(vec![i]));
                }
            }
        }
    }
}

impl Default for SceneValidator { fn default() -> Self { Self::new() } }

pub fn is_scene_valid(scene: &SceneModel) -> bool { SceneValidator::new().validate(scene).is_valid() }

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::{Sphere, Volume};
    use xr_types::scene::{InteractableElement, DependencyType};

    fn make_valid_scene() -> SceneModel {
        let mut scene = SceneModel::new("ValidScene");
        scene.add_element(InteractableElement::new("Button", [0.0, 1.0, -0.5], InteractionType::Click));
        scene.add_element(InteractableElement::new("Handle", [0.3, 1.2, -0.5], InteractionType::Grab));
        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene.devices.push(DeviceConfig::quest_3());
        scene
    }

    #[test] fn test_valid_scene() { assert!(SceneValidator::new().validate(&make_valid_scene()).is_valid()); }

    #[test] fn test_empty_scene() {
        let report = SceneValidator::new().validate(&SceneModel::new("Empty"));
        assert!(report.warning_count() > 0);
    }

    #[test] fn test_out_of_bounds() {
        let mut scene = make_valid_scene();
        scene.add_element(InteractableElement::new("FarAway", [200.0, 0.0, 0.0], InteractionType::Click));
        assert!(SceneValidator::new().validate(&scene).error_count() > 0);
    }

    #[test] fn test_cycle_detection() {
        let mut scene = SceneModel::new("Cyclic");
        scene.add_element(InteractableElement::new("A", [0.0; 3], InteractionType::Click));
        scene.add_element(InteractableElement::new("B", [1.0, 0.0, 0.0], InteractionType::Click));
        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene.add_dependency(1, 0, DependencyType::Sequential);
        scene.devices.push(DeviceConfig::quest_3());
        assert!(!SceneValidator::new().validate(&scene).is_valid());
    }

    #[test] fn test_self_loop() {
        let mut scene = make_valid_scene();
        scene.add_dependency(0, 0, DependencyType::Sequential);
        assert!(SceneValidator::new().validate(&scene).error_count() > 0);
    }

    #[test] fn test_device_incompatibility() {
        let mut scene = SceneModel::new("Incompat");
        scene.add_element(InteractableElement::new("VoiceCmd", [0.0, 1.0, 0.0], InteractionType::Voice));
        scene.devices.push(DeviceConfig::quest_3());
        assert!(SceneValidator::new().validate(&scene).warning_count() > 0);
    }

    #[test] fn test_overlapping() {
        let mut scene = SceneModel::new("Overlap");
        scene.add_element(InteractableElement::new("A", [0.0; 3], InteractionType::Click).with_volume(Volume::Sphere(Sphere::new([0.0; 3], 0.1))));
        scene.add_element(InteractableElement::new("B", [0.001, 0.0, 0.0], InteractionType::Click).with_volume(Volume::Sphere(Sphere::new([0.001, 0.0, 0.0], 0.1))));
        scene.devices.push(DeviceConfig::quest_3());
        assert!(SceneValidator::new().validate(&scene).warning_count() > 0);
    }

    #[test] fn test_invalid_dep_index() {
        let mut scene = make_valid_scene();
        scene.add_dependency(0, 99, DependencyType::Sequential);
        assert!(SceneValidator::new().validate(&scene).error_count() > 0);
    }

    #[test] fn test_near_zero_scale() {
        let mut scene = SceneModel::new("ZeroScale");
        let mut elem = InteractableElement::new("Bad", [0.0, 1.0, 0.0], InteractionType::Click);
        elem.scale = [0.0, 1.0, 1.0];
        scene.add_element(elem);
        scene.devices.push(DeviceConfig::quest_3());
        assert!(SceneValidator::new().validate(&scene).error_count() > 0);
    }

    #[test] fn test_to_diagnostics() {
        let mut report = ValidationReport::new();
        report.push(ValidationFinding::error("V100", "Test error").with_elements(vec![0]));
        report.push(ValidationFinding::warning("V200", "Test warning"));
        assert_eq!(report.to_diagnostics().len(), 2);
    }

    #[test] fn test_for_element() {
        let mut report = ValidationReport::new();
        report.push(ValidationFinding::error("V100", "E1").with_elements(vec![0]));
        report.push(ValidationFinding::error("V101", "E2").with_elements(vec![1]));
        report.push(ValidationFinding::warning("V200", "W").with_elements(vec![0, 1]));
        assert_eq!(report.for_element(0).len(), 2);
        assert_eq!(report.for_element(1).len(), 2);
    }

    #[test] fn test_is_scene_valid_helper() { assert!(is_scene_valid(&make_valid_scene())); }

    #[test] fn test_non_unit_quaternion() {
        let mut scene = SceneModel::new("BadQuat");
        let mut elem = InteractableElement::new("BadQ", [0.0, 1.0, 0.0], InteractionType::Click);
        elem.orientation = [2.0, 0.0, 0.0, 0.0];
        scene.add_element(elem);
        scene.devices.push(DeviceConfig::quest_3());
        assert!(SceneValidator::new().validate(&scene).warning_count() > 0);
    }

    #[test] fn test_custom_thresholds() {
        let validator = SceneValidator::new().with_max_extent(10.0).with_min_volume(0.001).with_min_spacing(0.1);
        let mut scene = SceneModel::new("Custom");
        scene.add_element(InteractableElement::new("Near", [5.0, 0.0, 0.0], InteractionType::Click));
        scene.devices.push(DeviceConfig::quest_3());
        assert!(validator.validate(&scene).is_valid());
    }
}
