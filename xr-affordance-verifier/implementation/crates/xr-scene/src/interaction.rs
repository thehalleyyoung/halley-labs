//! Interaction pattern extraction and classification.
//!
//! Provides `InteractionExtractor` for analyzing scene elements to determine
//! interaction patterns, and `InteractionSequenceBuilder` for constructing
//! multi-step interaction sequences.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use xr_types::geometry::{BoundingBox, point_distance};
use xr_types::scene::{
    DependencyType, InteractableElement, InteractionType, SceneModel,
};

/// Classification of common XR interaction patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionPattern {
    ButtonPress,
    GrabAndHold,
    GrabAndDrag,
    LinearSlider,
    RotaryDial,
    HoverActivate,
    GazeDwell,
    TwoHandedScale,
    MultiStep,
    ToggleSwitch,
    ProximityTrigger,
    VoiceCommand,
    GestureRecognition,
    Custom,
}

impl InteractionPattern {
    pub fn description(&self) -> &str {
        match self {
            Self::ButtonPress => "Single click/tap on a button",
            Self::GrabAndHold => "Grab and hold an object",
            Self::GrabAndDrag => "Grab and drag to a destination",
            Self::LinearSlider => "Slide along a linear track",
            Self::RotaryDial => "Rotate a dial or knob",
            Self::HoverActivate => "Hover over to activate",
            Self::GazeDwell => "Look at for a duration to activate",
            Self::TwoHandedScale => "Use two hands to resize",
            Self::MultiStep => "Multi-step sequential interaction",
            Self::ToggleSwitch => "Toggle between on and off states",
            Self::ProximityTrigger => "Enter proximity zone to trigger",
            Self::VoiceCommand => "Use voice to command",
            Self::GestureRecognition => "Perform a gesture",
            Self::Custom => "Custom or unrecognized pattern",
        }
    }

    pub fn requires_precision(&self) -> bool {
        matches!(self, Self::LinearSlider | Self::RotaryDial | Self::GrabAndDrag | Self::TwoHandedScale)
    }

    pub fn min_time(&self) -> f64 {
        match self {
            Self::ButtonPress => 0.1, Self::GrabAndHold => 0.5,
            Self::GrabAndDrag => 1.0, Self::LinearSlider => 0.5,
            Self::RotaryDial => 0.5, Self::HoverActivate => 0.3,
            Self::GazeDwell => 1.0, Self::TwoHandedScale => 1.5,
            Self::MultiStep => 2.0, Self::ToggleSwitch => 0.2,
            Self::ProximityTrigger => 0.0, Self::VoiceCommand => 1.0,
            Self::GestureRecognition => 0.5, Self::Custom => 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedInteraction {
    pub element_index: usize,
    pub element_name: String,
    pub pattern: InteractionPattern,
    pub position: [f64; 3],
    pub bounds: BoundingBox,
    pub confidence: f64,
    pub notes: Vec<String>,
}

pub struct InteractionExtractor {
    spatial_threshold: f64,
    slider_aspect_ratio: f64,
    dial_roundness_threshold: f64,
}

impl InteractionExtractor {
    pub fn new() -> Self {
        Self { spatial_threshold: 0.1, slider_aspect_ratio: 3.0, dial_roundness_threshold: 0.8 }
    }

    pub fn with_spatial_threshold(mut self, threshold: f64) -> Self { self.spatial_threshold = threshold; self }
    pub fn with_slider_aspect_ratio(mut self, ratio: f64) -> Self { self.slider_aspect_ratio = ratio; self }

    pub fn extract_all(&self, scene: &SceneModel) -> Vec<ExtractedInteraction> {
        scene.elements.iter().enumerate()
            .map(|(i, elem)| self.classify_element(elem, i, scene)).collect()
    }

    pub fn classify_element(&self, element: &InteractableElement, index: usize, scene: &SceneModel) -> ExtractedInteraction {
        let (pattern, confidence, notes) = self.determine_pattern(element, index, scene);
        ExtractedInteraction {
            element_index: index, element_name: element.name.clone(), pattern,
            position: element.position, bounds: element.activation_volume.bounding_box(),
            confidence, notes,
        }
    }

    fn determine_pattern(&self, element: &InteractableElement, index: usize, scene: &SceneModel) -> (InteractionPattern, f64, Vec<String>) {
        let mut notes = Vec::new();
        let bb = element.activation_volume.bounding_box();
        let extents = bb.extents();
        let has_deps = scene.dependencies.iter().any(|d| d.target_index == index);
        let has_dependents = scene.dependencies.iter().any(|d| d.source_index == index);

        let pattern = match element.interaction_type {
            InteractionType::Click => {
                if has_deps && has_dependents {
                    notes.push("Part of a multi-step sequence".to_string());
                    InteractionPattern::MultiStep
                } else { InteractionPattern::ButtonPress }
            }
            InteractionType::Grab => {
                let has_drag_dep = scene.dependencies.iter().any(|d|
                    d.source_index == index && matches!(d.dependency_type, DependencyType::Sequential | DependencyType::Enable));
                if has_drag_dep {
                    notes.push("Grab leads to further interaction".to_string());
                    InteractionPattern::GrabAndDrag
                } else { InteractionPattern::GrabAndHold }
            }
            InteractionType::Drag => InteractionPattern::GrabAndDrag,
            InteractionType::Slider => {
                let max_ext = extents[0].max(extents[1]).max(extents[2]);
                let min_ext = extents[0].min(extents[1]).min(extents[2]);
                if min_ext > 0.0 && max_ext / min_ext >= self.slider_aspect_ratio {
                    notes.push(format!("Aspect ratio: {:.1}", max_ext / min_ext));
                }
                InteractionPattern::LinearSlider
            }
            InteractionType::Dial => {
                let aspect = if extents[1] > 0.0 { extents[0].min(extents[2]) / extents[1] } else { 0.0 };
                if aspect > self.dial_roundness_threshold { notes.push("Cylindrical volume detected".to_string()); }
                InteractionPattern::RotaryDial
            }
            InteractionType::Proximity => InteractionPattern::ProximityTrigger,
            InteractionType::Gaze => InteractionPattern::GazeDwell,
            InteractionType::Voice => InteractionPattern::VoiceCommand,
            InteractionType::TwoHanded => InteractionPattern::TwoHandedScale,
            InteractionType::Gesture => InteractionPattern::GestureRecognition,
            InteractionType::Hover => InteractionPattern::HoverActivate,
            InteractionType::Toggle => InteractionPattern::ToggleSwitch,
            InteractionType::Custom => InteractionPattern::Custom,
        };

        let volume_size = bb.volume();
        let confidence = if volume_size > 1e-6 && volume_size < 1.0 { 0.9 }
            else if volume_size >= 1.0 { notes.push("Large activation volume".to_string()); 0.7 }
            else { notes.push("Very small activation volume".to_string()); 0.6 };
        (pattern, confidence, notes)
    }

    pub fn find_interaction_pairs(&self, scene: &SceneModel) -> Vec<(usize, usize, f64)> {
        let mut pairs = Vec::new();
        for i in 0..scene.elements.len() {
            for j in (i + 1)..scene.elements.len() {
                let dist = point_distance(&scene.elements[i].position, &scene.elements[j].position);
                if dist < self.spatial_threshold { pairs.push((i, j, dist)); }
            }
        }
        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    pub fn group_by_type(&self, scene: &SceneModel) -> HashMap<InteractionType, Vec<usize>> {
        let mut groups: HashMap<InteractionType, Vec<usize>> = HashMap::new();
        for (i, elem) in scene.elements.iter().enumerate() { groups.entry(elem.interaction_type).or_default().push(i); }
        groups
    }

    pub fn scene_complexity(&self, scene: &SceneModel) -> f64 {
        let n = scene.elements.len() as f64;
        let d = scene.dependencies.len() as f64;
        let depth = scene.max_interaction_depth() as f64;
        let type_diversity = self.group_by_type(scene).len() as f64;
        n * (1.0 + d / n.max(1.0)) * (1.0 + depth * 0.3) * (1.0 + type_diversity * 0.1)
    }
}

impl Default for InteractionExtractor { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionStep {
    pub id: Uuid, pub name: String, pub pattern: InteractionPattern,
    pub element_index: Option<usize>, pub position: [f64; 3],
    pub duration: f64, pub preconditions: Vec<Uuid>,
}

pub struct InteractionSequenceBuilder { name: String, steps: Vec<InteractionStep> }

impl InteractionSequenceBuilder {
    pub fn new(name: impl Into<String>) -> Self { Self { name: name.into(), steps: Vec::new() } }

    pub fn step(mut self, name: impl Into<String>, pattern: InteractionPattern, position: [f64; 3]) -> Self {
        let id = Uuid::new_v4();
        let preconditions = if let Some(last) = self.steps.last() { vec![last.id] } else { vec![] };
        self.steps.push(InteractionStep { id, name: name.into(), pattern, element_index: None, position, duration: pattern.min_time(), preconditions });
        self
    }

    pub fn step_after(mut self, name: impl Into<String>, pattern: InteractionPattern, position: [f64; 3], after_indices: &[usize]) -> Self {
        let preconditions: Vec<Uuid> = after_indices.iter().filter_map(|&i| self.steps.get(i).map(|s| s.id)).collect();
        self.steps.push(InteractionStep { id: Uuid::new_v4(), name: name.into(), pattern, element_index: None, position, duration: pattern.min_time(), preconditions });
        self
    }

    pub fn with_duration(mut self, duration: f64) -> Self { if let Some(last) = self.steps.last_mut() { last.duration = duration; } self }
    pub fn with_element(mut self, element_index: usize) -> Self { if let Some(last) = self.steps.last_mut() { last.element_index = Some(element_index); } self }

    pub fn build(self) -> InteractionSequence {
        let total_duration = self.steps.iter().map(|s| s.duration).sum();
        let mut total_distance = 0.0;
        for i in 1..self.steps.len() { total_distance += point_distance(&self.steps[i - 1].position, &self.steps[i].position); }
        InteractionSequence { name: self.name, steps: self.steps, total_duration, total_distance }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionSequence {
    pub name: String, pub steps: Vec<InteractionStep>,
    pub total_duration: f64, pub total_distance: f64,
}

impl InteractionSequence {
    pub fn step_count(&self) -> usize { self.steps.len() }
    pub fn patterns(&self) -> Vec<InteractionPattern> { self.steps.iter().map(|s| s.pattern).collect() }
    pub fn contains_pattern(&self, pattern: InteractionPattern) -> bool { self.steps.iter().any(|s| s.pattern == pattern) }
    pub fn step(&self, index: usize) -> Option<&InteractionStep> { self.steps.get(index) }
    pub fn avg_step_distance(&self) -> f64 {
        if self.steps.len() <= 1 { return 0.0; }
        self.total_distance / (self.steps.len() - 1) as f64
    }
}

pub fn extract_sequence_from_scene(scene: &SceneModel) -> InteractionSequence {
    let extractor = InteractionExtractor::new();
    let interactions = extractor.extract_all(scene);
    let order = scene.topological_order();
    let mut builder = InteractionSequenceBuilder::new(&scene.name);
    for &idx in &order {
        if idx < interactions.len() {
            let inter = &interactions[idx];
            builder = builder.step(&inter.element_name, inter.pattern, inter.position).with_element(idx);
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::{Sphere, Volume};
    use xr_types::scene::DependencyType;

    fn make_scene() -> SceneModel {
        let mut scene = SceneModel::new("InteractionTest");
        scene.add_element(InteractableElement::new("Button1", [0.0, 1.0, -0.5], InteractionType::Click));
        scene.add_element(InteractableElement::new("Handle1", [0.3, 1.2, -0.5], InteractionType::Grab)
            .with_volume(Volume::Sphere(Sphere::new([0.3, 1.2, -0.5], 0.04))));
        scene.add_element(InteractableElement::new("Slider1", [0.0, 0.8, -0.5], InteractionType::Slider));
        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene
    }

    #[test] fn test_extract_all() {
        let scene = make_scene();
        let extractor = InteractionExtractor::new();
        let interactions = extractor.extract_all(&scene);
        assert_eq!(interactions.len(), 3);
        assert_eq!(interactions[2].pattern, InteractionPattern::LinearSlider);
    }

    #[test] fn test_classify_click() {
        let scene = SceneModel::new("Empty");
        let elem = InteractableElement::new("Btn", [0.0; 3], InteractionType::Click);
        let extractor = InteractionExtractor::new();
        let inter = extractor.classify_element(&elem, 0, &scene);
        assert_eq!(inter.pattern, InteractionPattern::ButtonPress);
    }

    #[test] fn test_interaction_patterns() {
        assert!(InteractionPattern::LinearSlider.requires_precision());
        assert!(!InteractionPattern::ButtonPress.requires_precision());
    }

    #[test] fn test_group_by_type() {
        let scene = make_scene();
        let extractor = InteractionExtractor::new();
        let groups = extractor.group_by_type(&scene);
        assert!(groups.contains_key(&InteractionType::Click));
    }

    #[test] fn test_scene_complexity() {
        let scene = make_scene();
        let extractor = InteractionExtractor::new();
        assert!(extractor.scene_complexity(&scene) > 0.0);
    }

    #[test] fn test_sequence_builder() {
        let seq = InteractionSequenceBuilder::new("TestSequence")
            .step("Press", InteractionPattern::ButtonPress, [0.0, 1.0, 0.0])
            .step("Grab", InteractionPattern::GrabAndHold, [0.3, 1.2, 0.0])
            .with_duration(2.0)
            .step("Slide", InteractionPattern::LinearSlider, [0.0, 0.8, 0.0])
            .build();
        assert_eq!(seq.step_count(), 3);
        assert!(seq.total_duration > 0.0);
    }

    #[test] fn test_sequence_step_after() {
        let seq = InteractionSequenceBuilder::new("Branching")
            .step("A", InteractionPattern::ButtonPress, [0.0; 3])
            .step("B", InteractionPattern::ButtonPress, [1.0, 0.0, 0.0])
            .step_after("C", InteractionPattern::GrabAndHold, [2.0, 0.0, 0.0], &[0, 1])
            .build();
        assert_eq!(seq.steps[2].preconditions.len(), 2);
    }

    #[test] fn test_extract_sequence_from_scene() {
        let scene = make_scene();
        let seq = extract_sequence_from_scene(&scene);
        assert_eq!(seq.step_count(), 3);
    }

    #[test] fn test_find_interaction_pairs() {
        let mut scene = SceneModel::new("Pairs");
        scene.add_element(InteractableElement::new("A", [0.0; 3], InteractionType::Click));
        scene.add_element(InteractableElement::new("B", [0.05, 0.0, 0.0], InteractionType::Click));
        scene.add_element(InteractableElement::new("C", [10.0, 0.0, 0.0], InteractionType::Click));
        let extractor = InteractionExtractor::new();
        assert_eq!(extractor.find_interaction_pairs(&scene).len(), 1);
    }

    #[test] fn test_sequence_avg_distance() {
        let seq = InteractionSequenceBuilder::new("Dist")
            .step("A", InteractionPattern::ButtonPress, [0.0; 3])
            .step("B", InteractionPattern::ButtonPress, [1.0, 0.0, 0.0])
            .step("C", InteractionPattern::ButtonPress, [2.0, 0.0, 0.0])
            .build();
        assert!((seq.avg_step_distance() - 1.0).abs() < 1e-10);
    }
}
