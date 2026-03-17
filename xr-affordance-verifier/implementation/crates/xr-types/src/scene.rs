//! Scene model types for XR accessibility verification.
//!
//! Defines the core scene representation per Definition D1:
//! An XR accessibility scene Ω = (E, G, D) where E is the set of
//! interactable elements, G is the interaction dependency graph,
//! and D is the set of target device configurations.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::geometry::{BoundingBox, Volume};
use crate::device::DeviceConfig;

/// An XR accessibility scene model (Definition D1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneModel {
    /// Unique scene identifier.
    pub id: Uuid,
    /// Human-readable scene name.
    pub name: String,
    /// Scene description.
    pub description: String,
    /// Version string for the scene format.
    pub version: String,
    /// World-space bounding box of the entire scene.
    pub bounds: BoundingBox,
    /// Set of interactable elements E = {e_1, ..., e_m}.
    pub elements: Vec<InteractableElement>,
    /// Interaction dependency graph edges (source_idx -> target_idx).
    pub dependencies: Vec<DependencyEdge>,
    /// Target device configurations D = {d_1, ..., d_p}.
    pub devices: Vec<DeviceConfig>,
    /// Scene-level metadata.
    pub metadata: SceneMetadata,
    /// Transform hierarchy nodes.
    pub transform_nodes: Vec<TransformNode>,
    /// Named coordinate frames.
    pub coordinate_frames: HashMap<String, CoordinateFrame>,
}

impl SceneModel {
    /// Create a new empty scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: String::new(),
            version: crate::PROTOCOL_VERSION.to_string(),
            bounds: BoundingBox::default(),
            elements: Vec::new(),
            dependencies: Vec::new(),
            devices: Vec::new(),
            metadata: SceneMetadata::default(),
            transform_nodes: Vec::new(),
            coordinate_frames: HashMap::new(),
        }
    }

    /// Add an interactable element to the scene.
    pub fn add_element(&mut self, element: InteractableElement) -> usize {
        let idx = self.elements.len();
        self.elements.push(element);
        self.recompute_bounds();
        idx
    }

    /// Add a dependency edge.
    pub fn add_dependency(&mut self, source: usize, target: usize, dep_type: DependencyType) {
        self.dependencies.push(DependencyEdge {
            source_index: source,
            target_index: target,
            dependency_type: dep_type,
        });
    }

    /// Get element by index.
    pub fn element(&self, idx: usize) -> Option<&InteractableElement> {
        self.elements.get(idx)
    }

    /// Get element by ID.
    pub fn element_by_id(&self, id: Uuid) -> Option<&InteractableElement> {
        self.elements.iter().find(|e| e.id == id)
    }

    /// Find elements within a bounding box.
    pub fn elements_in_bounds(&self, bounds: &BoundingBox) -> Vec<usize> {
        self.elements
            .iter()
            .enumerate()
            .filter(|(_, e)| e.activation_volume.bounding_box().intersects(bounds))
            .map(|(i, _)| i)
            .collect()
    }

    /// Get dependencies for a given element.
    pub fn dependencies_of(&self, element_idx: usize) -> Vec<&DependencyEdge> {
        self.dependencies
            .iter()
            .filter(|d| d.target_index == element_idx)
            .collect()
    }

    /// Get elements that depend on a given element.
    pub fn dependents_of(&self, element_idx: usize) -> Vec<&DependencyEdge> {
        self.dependencies
            .iter()
            .filter(|d| d.source_index == element_idx)
            .collect()
    }

    /// Get the topological order of elements (respecting dependencies).
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.elements.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for dep in &self.dependencies {
            if dep.source_index < n && dep.target_index < n {
                adj[dep.source_index].push(dep.target_index);
                in_degree[dep.target_index] += 1;
            }
        }

        let mut queue: std::collections::VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        order
    }

    /// Check if the dependency graph is acyclic.
    pub fn is_dag(&self) -> bool {
        self.topological_order().len() == self.elements.len()
    }

    /// Get the interaction depth (longest dependency chain).
    pub fn max_interaction_depth(&self) -> usize {
        let n = self.elements.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for dep in &self.dependencies {
            if dep.source_index < n && dep.target_index < n {
                adj[dep.source_index].push(dep.target_index);
            }
        }

        let mut depth = vec![0usize; n];
        let order = self.topological_order();
        for &node in &order {
            for &next in &adj[node] {
                depth[next] = depth[next].max(depth[node] + 1);
            }
        }
        depth.into_iter().max().unwrap_or(0)
    }

    /// Recompute scene bounds from all elements.
    pub fn recompute_bounds(&mut self) {
        if self.elements.is_empty() {
            self.bounds = BoundingBox::default();
            return;
        }
        let mut bounds = self.elements[0].activation_volume.bounding_box();
        for elem in &self.elements[1..] {
            bounds = bounds.union(&elem.activation_volume.bounding_box());
        }
        self.bounds = bounds;
    }

    /// Validate the scene structure.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.elements.is_empty() {
            errors.push("Scene has no interactable elements".to_string());
        }
        if self.devices.is_empty() {
            errors.push("Scene has no target devices".to_string());
        }
        if !self.is_dag() {
            errors.push("Dependency graph contains cycles".to_string());
        }
        let n = self.elements.len();
        for dep in &self.dependencies {
            if dep.source_index >= n {
                errors.push(format!(
                    "Dependency source index {} out of range (max {})",
                    dep.source_index,
                    n - 1
                ));
            }
            if dep.target_index >= n {
                errors.push(format!(
                    "Dependency target index {} out of range (max {})",
                    dep.target_index,
                    n - 1
                ));
            }
        }
        if self.max_interaction_depth() > crate::MAX_INTERACTION_DEPTH {
            errors.push(format!(
                "Interaction depth {} exceeds maximum {}",
                self.max_interaction_depth(),
                crate::MAX_INTERACTION_DEPTH
            ));
        }
        errors
    }

    /// Count elements by interaction type.
    pub fn count_by_type(&self) -> HashMap<InteractionType, usize> {
        let mut counts = HashMap::new();
        for elem in &self.elements {
            *counts.entry(elem.interaction_type).or_insert(0) += 1;
        }
        counts
    }

    /// Get the total number of verification tasks (elements × devices).
    pub fn verification_task_count(&self) -> usize {
        self.elements.len() * self.devices.len().max(1)
    }
}

/// An interactable element in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractableElement {
    /// Unique element identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Position in world space.
    pub position: [f64; 3],
    /// Orientation as quaternion [w, x, y, z].
    pub orientation: [f64; 4],
    /// Scale factors.
    pub scale: [f64; 3],
    /// Activation volume V_i ⊂ R^3 (compact semialgebraic).
    pub activation_volume: Volume,
    /// Interaction type τ_i.
    pub interaction_type: InteractionType,
    /// Required actuator.
    pub actuator: ActuatorType,
    /// Required pose region r_i ⊂ SE(3).
    pub required_pose: Option<PoseConstraint>,
    /// Visual properties.
    pub visual: VisualProperties,
    /// Tags for filtering.
    pub tags: Vec<String>,
    /// Custom properties.
    pub properties: HashMap<String, String>,
    /// Transform node index in the hierarchy.
    pub transform_node: Option<usize>,
    /// Minimum interaction duration in seconds.
    pub min_duration: f64,
    /// Whether the element requires sustained contact.
    pub sustained_contact: bool,
    /// Feedback type provided when interacted with.
    pub feedback_type: FeedbackType,
    /// Priority for accessibility (higher = more important).
    pub priority: u32,
}

impl InteractableElement {
    /// Create a new element with minimal properties.
    pub fn new(name: impl Into<String>, position: [f64; 3], interaction_type: InteractionType) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            position,
            orientation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            activation_volume: Volume::Sphere(crate::geometry::Sphere::new(position, 0.05)),
            interaction_type,
            actuator: ActuatorType::Hand,
            required_pose: None,
            visual: VisualProperties::default(),
            tags: Vec::new(),
            properties: HashMap::new(),
            transform_node: None,
            min_duration: 0.0,
            sustained_contact: false,
            feedback_type: FeedbackType::Visual,
            priority: 1,
        }
    }

    /// Set the activation volume.
    pub fn with_volume(mut self, volume: Volume) -> Self {
        self.activation_volume = volume;
        self
    }

    /// Set the actuator type.
    pub fn with_actuator(mut self, actuator: ActuatorType) -> Self {
        self.actuator = actuator;
        self
    }

    /// Set the pose constraint.
    pub fn with_pose_constraint(mut self, constraint: PoseConstraint) -> Self {
        self.required_pose = Some(constraint);
        self
    }

    /// Get the world-space bounding box.
    pub fn world_bounds(&self) -> BoundingBox {
        self.activation_volume.bounding_box()
    }

    /// Check if a point is in the activation volume.
    pub fn is_reachable_from(&self, point: &[f64; 3]) -> bool {
        self.activation_volume.contains_point(point)
    }

    /// Compute the distance from a point to the closest point in the activation volume.
    pub fn distance_to(&self, point: &[f64; 3]) -> f64 {
        let bb = self.activation_volume.bounding_box();
        bb.signed_distance(point).max(0.0)
    }
}

/// Types of interaction supported by elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    /// Simple click/press.
    Click,
    /// Grab and hold.
    Grab,
    /// Grab and move/drag.
    Drag,
    /// Slider interaction.
    Slider,
    /// Dial/rotation interaction.
    Dial,
    /// Proximity trigger (no direct contact needed).
    Proximity,
    /// Gaze-based interaction.
    Gaze,
    /// Voice-activated interaction.
    Voice,
    /// Two-handed interaction.
    TwoHanded,
    /// Gesture-based interaction.
    Gesture,
    /// Hover interaction.
    Hover,
    /// Toggle switch.
    Toggle,
    /// Custom interaction type.
    Custom,
}

impl InteractionType {
    /// Whether this interaction type requires hand tracking.
    pub fn requires_hand_tracking(&self) -> bool {
        matches!(
            self,
            InteractionType::Grab
                | InteractionType::Drag
                | InteractionType::Gesture
                | InteractionType::TwoHanded
        )
    }

    /// Whether this interaction type requires line-of-sight.
    pub fn requires_line_of_sight(&self) -> bool {
        matches!(
            self,
            InteractionType::Gaze | InteractionType::Click
        )
    }

    /// Minimum number of hands required.
    pub fn min_hands(&self) -> usize {
        match self {
            InteractionType::TwoHanded => 2,
            InteractionType::Voice | InteractionType::Gaze | InteractionType::Proximity => 0,
            _ => 1,
        }
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &str {
        match self {
            InteractionType::Click => "Click/press interaction",
            InteractionType::Grab => "Grab and hold interaction",
            InteractionType::Drag => "Grab and drag interaction",
            InteractionType::Slider => "Slider interaction",
            InteractionType::Dial => "Dial/rotation interaction",
            InteractionType::Proximity => "Proximity-triggered interaction",
            InteractionType::Gaze => "Gaze-based interaction",
            InteractionType::Voice => "Voice-activated interaction",
            InteractionType::TwoHanded => "Two-handed interaction",
            InteractionType::Gesture => "Gesture-based interaction",
            InteractionType::Hover => "Hover interaction",
            InteractionType::Toggle => "Toggle switch",
            InteractionType::Custom => "Custom interaction type",
        }
    }
}

/// Actuator types for element interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActuatorType {
    /// Either hand.
    Hand,
    /// Specifically the left hand.
    LeftHand,
    /// Specifically the right hand.
    RightHand,
    /// Both hands simultaneously.
    BothHands,
    /// Controller (wand/gamepad).
    Controller,
    /// Head/gaze.
    Head,
    /// Eye tracking.
    Eye,
    /// Foot/body.
    Body,
}

/// Pose constraint for element interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseConstraint {
    /// Required position region (bounding box).
    pub position_bounds: BoundingBox,
    /// Required orientation (target rotation as quaternion [w,x,y,z]).
    pub target_orientation: Option<[f64; 4]>,
    /// Orientation tolerance in radians.
    pub orientation_tolerance: f64,
    /// Minimum approach direction (normalized vector).
    pub approach_direction: Option<[f64; 3]>,
    /// Approach cone half-angle in radians.
    pub approach_cone_angle: f64,
}

impl Default for PoseConstraint {
    fn default() -> Self {
        Self {
            position_bounds: BoundingBox::default(),
            target_orientation: None,
            orientation_tolerance: std::f64::consts::PI,
            approach_direction: None,
            approach_cone_angle: std::f64::consts::PI,
        }
    }
}

impl PoseConstraint {
    /// Check if a position satisfies this constraint.
    pub fn satisfies_position(&self, pos: &[f64; 3]) -> bool {
        self.position_bounds.contains_point(pos)
    }

    /// Check if an orientation satisfies this constraint.
    pub fn satisfies_orientation(&self, quat: &[f64; 4]) -> bool {
        match self.target_orientation {
            Some(target) => {
                let dot = (quat[0] * target[0]
                    + quat[1] * target[1]
                    + quat[2] * target[2]
                    + quat[3] * target[3])
                    .abs();
                let angle = 2.0 * (1.0f64.min(dot)).acos();
                angle <= self.orientation_tolerance
            }
            None => true,
        }
    }
}

/// Visual properties of an element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualProperties {
    /// Color as [R, G, B, A] in 0..1.
    pub color: [f32; 4],
    /// Whether the element is currently visible.
    pub visible: bool,
    /// Opacity (0..1).
    pub opacity: f32,
    /// Whether the element provides hover highlight.
    pub hover_highlight: bool,
    /// Label text.
    pub label: Option<String>,
}

impl Default for VisualProperties {
    fn default() -> Self {
        Self {
            color: [0.5, 0.5, 0.5, 1.0],
            visible: true,
            opacity: 1.0,
            hover_highlight: true,
            label: None,
        }
    }
}

/// Feedback type when element is interacted with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeedbackType {
    Visual,
    Haptic,
    Audio,
    VisualHaptic,
    VisualAudio,
    HapticAudio,
    All,
    None,
}

/// Dependency edge in the interaction graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Source element index (must be completed first).
    pub source_index: usize,
    /// Target element index (depends on source).
    pub target_index: usize,
    /// Type of dependency.
    pub dependency_type: DependencyType,
}

/// Types of interaction dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    /// Target requires source to be activated first.
    Sequential,
    /// Target becomes visible after source is activated.
    Visibility,
    /// Target becomes enabled after source is activated.
    Enable,
    /// Both must be active simultaneously.
    Concurrent,
    /// Source unlocks target.
    Unlock,
}

/// Scene-level metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneMetadata {
    /// Author of the scene.
    pub author: String,
    /// Creation timestamp.
    pub created_at: String,
    /// Last modification timestamp.
    pub modified_at: String,
    /// Unity version.
    pub unity_version: Option<String>,
    /// Target platform.
    pub target_platform: Option<String>,
    /// Custom metadata fields.
    pub custom: HashMap<String, String>,
    /// Scene complexity estimate.
    pub estimated_treewidth: Option<usize>,
}

/// Transform hierarchy node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformNode {
    /// Node name.
    pub name: String,
    /// Parent node index (None for root).
    pub parent: Option<usize>,
    /// Local position.
    pub local_position: [f64; 3],
    /// Local rotation as quaternion [w,x,y,z].
    pub local_rotation: [f64; 4],
    /// Local scale.
    pub local_scale: [f64; 3],
    /// Children indices.
    pub children: Vec<usize>,
}

impl TransformNode {
    /// Create a root transform node.
    pub fn root(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            local_position: [0.0; 3],
            local_rotation: [1.0, 0.0, 0.0, 0.0],
            local_scale: [1.0, 1.0, 1.0],
            children: Vec::new(),
        }
    }

    /// Compute the local 4x4 transform matrix.
    pub fn local_matrix(&self) -> nalgebra::Matrix4<f64> {
        let [w, x, y, z] = self.local_rotation;
        let rot = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z));
        let translation = nalgebra::Translation3::new(
            self.local_position[0],
            self.local_position[1],
            self.local_position[2],
        );
        let scale = nalgebra::Matrix4::new_nonuniform_scaling(&nalgebra::Vector3::new(
            self.local_scale[0],
            self.local_scale[1],
            self.local_scale[2],
        ));
        translation.to_homogeneous() * rot.to_homogeneous() * scale
    }
}

/// Named coordinate frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateFrame {
    /// Frame name.
    pub name: String,
    /// Origin in world space.
    pub origin: [f64; 3],
    /// Rotation as quaternion.
    pub rotation: [f64; 4],
    /// Description of the frame.
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    #[test]
    fn test_scene_creation() {
        let scene = SceneModel::new("Test Scene");
        assert_eq!(scene.name, "Test Scene");
        assert!(scene.elements.is_empty());
    }

    #[test]
    fn test_add_element() {
        let mut scene = SceneModel::new("Test");
        let elem = InteractableElement::new("Button", [0.0, 1.0, 0.5], InteractionType::Click);
        let idx = scene.add_element(elem);
        assert_eq!(idx, 0);
        assert_eq!(scene.elements.len(), 1);
    }

    #[test]
    fn test_topological_order() {
        let mut scene = SceneModel::new("Test");
        let e0 = InteractableElement::new("A", [0.0, 0.0, 0.0], InteractionType::Click);
        let e1 = InteractableElement::new("B", [1.0, 0.0, 0.0], InteractionType::Click);
        let e2 = InteractableElement::new("C", [2.0, 0.0, 0.0], InteractionType::Click);
        scene.add_element(e0);
        scene.add_element(e1);
        scene.add_element(e2);
        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene.add_dependency(1, 2, DependencyType::Sequential);

        let order = scene.topological_order();
        assert_eq!(order, vec![0, 1, 2]);
        assert!(scene.is_dag());
        assert_eq!(scene.max_interaction_depth(), 2);
    }

    #[test]
    fn test_interaction_type_properties() {
        assert!(InteractionType::Grab.requires_hand_tracking());
        assert!(!InteractionType::Click.requires_hand_tracking());
        assert_eq!(InteractionType::TwoHanded.min_hands(), 2);
        assert_eq!(InteractionType::Voice.min_hands(), 0);
    }

    #[test]
    fn test_pose_constraint() {
        let constraint = PoseConstraint {
            position_bounds: BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            target_orientation: Some([1.0, 0.0, 0.0, 0.0]),
            orientation_tolerance: 0.5,
            approach_direction: None,
            approach_cone_angle: std::f64::consts::PI,
        };
        assert!(constraint.satisfies_position(&[0.5, 0.5, 0.5]));
        assert!(!constraint.satisfies_position(&[1.5, 0.5, 0.5]));
        assert!(constraint.satisfies_orientation(&[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_count_by_type() {
        let mut scene = SceneModel::new("Test");
        scene.add_element(InteractableElement::new("A", [0.0; 3], InteractionType::Click));
        scene.add_element(InteractableElement::new("B", [1.0, 0.0, 0.0], InteractionType::Click));
        scene.add_element(InteractableElement::new("C", [2.0, 0.0, 0.0], InteractionType::Grab));
        let counts = scene.count_by_type();
        assert_eq!(counts[&InteractionType::Click], 2);
        assert_eq!(counts[&InteractionType::Grab], 1);
    }

    #[test]
    fn test_transform_node() {
        let node = TransformNode::root("Root");
        let mat = node.local_matrix();
        let id: nalgebra::Matrix4<f64> = nalgebra::Matrix4::identity();
        for i in 0..4 {
            for j in 0..4 {
                assert!((mat[(i, j)] - id[(i, j)]).abs() < 1e-10);
            }
        }
    }
}
