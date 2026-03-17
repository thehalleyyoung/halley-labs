//! Scene file parser for XR scene descriptions.
//!
//! Provides `SceneParser` for reading JSON scene files and `SceneBuilder`
//! for programmatic scene construction with a fluent API.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use xr_types::error::{VerifierError, VerifierResult};
use xr_types::geometry::{BoundingBox, Capsule, Cylinder, Sphere, Volume};
use xr_types::scene::{
    CoordinateFrame, DependencyEdge, DependencyType, FeedbackType, InteractableElement,
    InteractionType, SceneMetadata, SceneModel, TransformNode,
};
use xr_types::device::DeviceConfig;

/// Detected scene file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneFormat {
    /// Native JSON format used by the verifier.
    NativeJson,
    /// Simplified Unity YAML export.
    UnityYaml,
    /// glTF binary or JSON.
    Gltf,
    /// USD (Universal Scene Description) text or binary.
    Usd,
    /// Unknown format.
    Unknown,
}

/// Intermediate parsed representation before full SceneModel construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedScene {
    pub name: String,
    pub format: SceneFormat,
    pub elements: Vec<ParsedElement>,
    pub dependencies: Vec<ParsedDependency>,
    pub transform_nodes: Vec<ParsedTransformNode>,
    pub metadata: HashMap<String, String>,
    pub warnings: Vec<String>,
}

impl ParsedScene {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            format: SceneFormat::NativeJson,
            elements: Vec::new(),
            dependencies: Vec::new(),
            transform_nodes: Vec::new(),
            metadata: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    pub fn dependency_count(&self) -> usize {
        self.dependencies.len()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Validate the parsed scene for basic structural correctness.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let n = self.elements.len();
        for dep in &self.dependencies {
            if dep.source_index >= n {
                errors.push(format!(
                    "Dependency source index {} out of range (max {})",
                    dep.source_index,
                    n.saturating_sub(1)
                ));
            }
            if dep.target_index >= n {
                errors.push(format!(
                    "Dependency target index {} out of range (max {})",
                    dep.target_index,
                    n.saturating_sub(1)
                ));
            }
        }
        for (i, elem) in self.elements.iter().enumerate() {
            if elem.name.is_empty() {
                errors.push(format!("Element at index {} has empty name", i));
            }
        }
        errors
    }
}

/// An element as parsed from the scene file, before full validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedElement {
    pub id: Option<Uuid>,
    pub name: String,
    pub position: [f64; 3],
    pub orientation: [f64; 4],
    pub scale: [f64; 3],
    pub volume: ParsedVolume,
    pub interaction_type: String,
    pub tags: Vec<String>,
    pub properties: HashMap<String, String>,
    pub transform_node: Option<usize>,
    pub priority: u32,
}

/// Parsed volume specification before conversion to xr_types::Volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParsedVolume {
    Box {
        min: [f64; 3],
        max: [f64; 3],
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    Capsule {
        start: [f64; 3],
        end: [f64; 3],
        radius: f64,
    },
    Cylinder {
        center: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        half_height: f64,
    },
}

impl ParsedVolume {
    pub fn to_volume(&self) -> Volume {
        match self {
            ParsedVolume::Box { min, max } => Volume::Box(BoundingBox::new(*min, *max)),
            ParsedVolume::Sphere { center, radius } => {
                Volume::Sphere(Sphere::new(*center, *radius))
            }
            ParsedVolume::Capsule {
                start,
                end,
                radius,
            } => Volume::Capsule(Capsule::new(*start, *end, *radius)),
            ParsedVolume::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => Volume::Cylinder(Cylinder::new(*center, *axis, *radius, *half_height)),
        }
    }
}

/// Parsed dependency edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDependency {
    pub source_index: usize,
    pub target_index: usize,
    pub dependency_type: String,
}

/// Parsed transform node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedTransformNode {
    pub name: String,
    pub parent: Option<usize>,
    pub local_position: [f64; 3],
    pub local_rotation: [f64; 4],
    pub local_scale: [f64; 3],
    pub children: Vec<usize>,
}

/// Raw JSON structures for deserialization.
#[derive(Debug, Deserialize)]
struct RawScene {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    elements: Vec<RawElement>,
    #[serde(default)]
    dependencies: Vec<RawDependency>,
    #[serde(default)]
    transform_nodes: Vec<RawTransformNode>,
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    devices: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawElement {
    #[serde(default)]
    id: Option<String>,
    name: String,
    #[serde(default = "default_position")]
    position: [f64; 3],
    #[serde(default = "default_orientation")]
    orientation: [f64; 4],
    #[serde(default = "default_scale")]
    scale: [f64; 3],
    #[serde(default)]
    volume: Option<RawVolume>,
    #[serde(default = "default_interaction")]
    interaction_type: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    properties: HashMap<String, String>,
    #[serde(default)]
    transform_node: Option<usize>,
    #[serde(default = "default_priority")]
    priority: u32,
}

fn default_position() -> [f64; 3] {
    [0.0; 3]
}
fn default_orientation() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}
fn default_scale() -> [f64; 3] {
    [1.0; 3]
}
fn default_interaction() -> String {
    "Click".to_string()
}
fn default_priority() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct RawVolume {
    #[serde(rename = "type")]
    vol_type: String,
    #[serde(default)]
    min: Option<[f64; 3]>,
    #[serde(default)]
    max: Option<[f64; 3]>,
    #[serde(default)]
    center: Option<[f64; 3]>,
    #[serde(default)]
    radius: Option<f64>,
    #[serde(default)]
    start: Option<[f64; 3]>,
    #[serde(default)]
    end: Option<[f64; 3]>,
    #[serde(default)]
    axis: Option<[f64; 3]>,
    #[serde(default)]
    half_height: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawDependency {
    #[serde(alias = "source_index")]
    source: usize,
    #[serde(alias = "target_index")]
    target: usize,
    #[serde(default = "default_dep_type")]
    dependency_type: String,
}

fn default_dep_type() -> String {
    "Sequential".to_string()
}

#[derive(Debug, Deserialize)]
struct RawTransformNode {
    name: String,
    #[serde(default)]
    parent: Option<usize>,
    #[serde(default = "default_position")]
    local_position: [f64; 3],
    #[serde(default = "default_orientation")]
    local_rotation: [f64; 4],
    #[serde(default = "default_scale")]
    local_scale: [f64; 3],
    #[serde(default)]
    children: Vec<usize>,
}

/// Scene file parser.
pub struct SceneParser {
    strict_mode: bool,
    max_elements: usize,
    max_depth: usize,
}

impl SceneParser {
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            max_elements: 10_000,
            max_depth: 10,
        }
    }

    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    pub fn with_max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }

    pub fn with_max_depth(mut self, max: usize) -> Self {
        self.max_depth = max;
        self
    }

    /// Detect the format of a scene file from its content.
    pub fn detect_format(content: &str) -> SceneFormat {
        let trimmed = content.trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            if trimmed.contains("\"m_Component\"") || trimmed.contains("\"m_GameObject\"") {
                return SceneFormat::UnityYaml;
            }
            if trimmed.contains("\"accessors\"") && trimmed.contains("\"bufferViews\"") {
                return SceneFormat::Gltf;
            }
            SceneFormat::NativeJson
        } else if trimmed.starts_with('%') || trimmed.contains("--- !u!") {
            SceneFormat::UnityYaml
        } else {
            SceneFormat::Unknown
        }
    }

    /// Parse a JSON scene string into a ParsedScene.
    pub fn parse_json(&self, json_str: &str) -> VerifierResult<ParsedScene> {
        let raw: RawScene = serde_json::from_str(json_str)
            .map_err(|e| VerifierError::SceneParsing(format!("JSON parse error: {}", e)))?;

        if raw.elements.len() > self.max_elements {
            return Err(VerifierError::SceneParsing(format!(
                "Scene has {} elements, exceeding maximum of {}",
                raw.elements.len(),
                self.max_elements
            )));
        }

        let mut parsed = ParsedScene::new(raw.name.unwrap_or_else(|| "Unnamed".to_string()));
        parsed.format = SceneFormat::NativeJson;

        // Parse metadata
        for (k, v) in &raw.metadata {
            parsed.metadata.insert(k.clone(), v.to_string());
        }
        if let Some(desc) = &raw.description {
            parsed.metadata.insert("description".to_string(), desc.clone());
        }
        if let Some(ver) = &raw.version {
            parsed.metadata.insert("version".to_string(), ver.clone());
        }

        // Parse elements
        for (i, raw_elem) in raw.elements.iter().enumerate() {
            match self.parse_element(raw_elem, i) {
                Ok(elem) => parsed.elements.push(elem),
                Err(e) => {
                    if self.strict_mode {
                        return Err(e);
                    }
                    parsed
                        .warnings
                        .push(format!("Element {} skipped: {}", i, e));
                }
            }
        }

        // Parse dependencies
        for raw_dep in &raw.dependencies {
            match self.parse_dependency(raw_dep, parsed.elements.len()) {
                Ok(dep) => parsed.dependencies.push(dep),
                Err(e) => {
                    if self.strict_mode {
                        return Err(e);
                    }
                    parsed.warnings.push(format!("Dependency skipped: {}", e));
                }
            }
        }

        // Parse transform nodes
        for raw_node in &raw.transform_nodes {
            parsed.transform_nodes.push(ParsedTransformNode {
                name: raw_node.name.clone(),
                parent: raw_node.parent,
                local_position: raw_node.local_position,
                local_rotation: raw_node.local_rotation,
                local_scale: raw_node.local_scale,
                children: raw_node.children.clone(),
            });
        }

        Ok(parsed)
    }

    fn parse_element(&self, raw: &RawElement, index: usize) -> VerifierResult<ParsedElement> {
        let id = match &raw.id {
            Some(id_str) => Some(
                Uuid::parse_str(id_str).map_err(|e| {
                    VerifierError::SceneParsing(format!(
                        "Invalid UUID for element {}: {}",
                        index, e
                    ))
                })?,
            ),
            None => None,
        };

        let volume = self.parse_volume(&raw.volume, &raw.position)?;

        Ok(ParsedElement {
            id,
            name: raw.name.clone(),
            position: raw.position,
            orientation: raw.orientation,
            scale: raw.scale,
            volume,
            interaction_type: raw.interaction_type.clone(),
            tags: raw.tags.clone(),
            properties: raw.properties.clone(),
            transform_node: raw.transform_node,
            priority: raw.priority,
        })
    }

    fn parse_volume(
        &self,
        raw: &Option<RawVolume>,
        default_center: &[f64; 3],
    ) -> VerifierResult<ParsedVolume> {
        match raw {
            Some(vol) => match vol.vol_type.to_lowercase().as_str() {
                "box" | "aabb" => {
                    let min = vol
                        .min
                        .unwrap_or([default_center[0] - 0.05, default_center[1] - 0.05, default_center[2] - 0.05]);
                    let max = vol
                        .max
                        .unwrap_or([default_center[0] + 0.05, default_center[1] + 0.05, default_center[2] + 0.05]);
                    Ok(ParsedVolume::Box { min, max })
                }
                "sphere" => {
                    let center = vol.center.unwrap_or(*default_center);
                    let radius = vol.radius.unwrap_or(0.05);
                    if radius <= 0.0 {
                        return Err(VerifierError::SceneParsing(
                            "Sphere radius must be positive".to_string(),
                        ));
                    }
                    Ok(ParsedVolume::Sphere { center, radius })
                }
                "capsule" => {
                    let start = vol.start.unwrap_or(*default_center);
                    let end = vol.end.unwrap_or([
                        default_center[0],
                        default_center[1] + 0.1,
                        default_center[2],
                    ]);
                    let radius = vol.radius.unwrap_or(0.02);
                    Ok(ParsedVolume::Capsule {
                        start,
                        end,
                        radius,
                    })
                }
                "cylinder" => {
                    let center = vol.center.unwrap_or(*default_center);
                    let axis = vol.axis.unwrap_or([0.0, 1.0, 0.0]);
                    let radius = vol.radius.unwrap_or(0.05);
                    let half_height = vol.half_height.unwrap_or(0.1);
                    Ok(ParsedVolume::Cylinder {
                        center,
                        axis,
                        radius,
                        half_height,
                    })
                }
                other => Err(VerifierError::SceneParsing(format!(
                    "Unknown volume type: {}",
                    other
                ))),
            },
            None => Ok(ParsedVolume::Sphere {
                center: *default_center,
                radius: 0.05,
            }),
        }
    }

    fn parse_dependency(
        &self,
        raw: &RawDependency,
        element_count: usize,
    ) -> VerifierResult<ParsedDependency> {
        if raw.source >= element_count {
            return Err(VerifierError::SceneParsing(format!(
                "Dependency source index {} out of range",
                raw.source
            )));
        }
        if raw.target >= element_count {
            return Err(VerifierError::SceneParsing(format!(
                "Dependency target index {} out of range",
                raw.target
            )));
        }
        Ok(ParsedDependency {
            source_index: raw.source,
            target_index: raw.target,
            dependency_type: raw.dependency_type.clone(),
        })
    }

    /// Convert a ParsedScene into a fully-typed SceneModel.
    pub fn build_scene_model(&self, parsed: &ParsedScene) -> VerifierResult<SceneModel> {
        let mut scene = SceneModel::new(&parsed.name);

        if let Some(desc) = parsed.metadata.get("description") {
            scene.description = desc.clone();
        }
        if let Some(ver) = parsed.metadata.get("version") {
            scene.version = ver.clone();
        }

        // Build metadata
        scene.metadata = SceneMetadata {
            author: parsed
                .metadata
                .get("author")
                .cloned()
                .unwrap_or_default(),
            created_at: parsed
                .metadata
                .get("created_at")
                .cloned()
                .unwrap_or_default(),
            modified_at: parsed
                .metadata
                .get("modified_at")
                .cloned()
                .unwrap_or_default(),
            unity_version: parsed.metadata.get("unity_version").cloned(),
            target_platform: parsed.metadata.get("target_platform").cloned(),
            custom: parsed.metadata.clone(),
            estimated_treewidth: None,
        };

        // Convert elements
        for parsed_elem in &parsed.elements {
            let interaction_type = parse_interaction_type(&parsed_elem.interaction_type);
            let mut elem =
                InteractableElement::new(&parsed_elem.name, parsed_elem.position, interaction_type);

            if let Some(id) = parsed_elem.id {
                elem.id = id;
            }
            elem.orientation = parsed_elem.orientation;
            elem.scale = parsed_elem.scale;
            elem.activation_volume = parsed_elem.volume.to_volume();
            elem.tags = parsed_elem.tags.clone();
            elem.properties = parsed_elem.properties.clone();
            elem.transform_node = parsed_elem.transform_node;
            elem.priority = parsed_elem.priority;
            scene.add_element(elem);
        }

        // Convert dependencies
        for parsed_dep in &parsed.dependencies {
            let dep_type = parse_dependency_type(&parsed_dep.dependency_type);
            scene.add_dependency(parsed_dep.source_index, parsed_dep.target_index, dep_type);
        }

        // Convert transform nodes
        for parsed_node in &parsed.transform_nodes {
            scene.transform_nodes.push(TransformNode {
                name: parsed_node.name.clone(),
                parent: parsed_node.parent,
                local_position: parsed_node.local_position,
                local_rotation: parsed_node.local_rotation,
                local_scale: parsed_node.local_scale,
                children: parsed_node.children.clone(),
            });
        }

        scene.recompute_bounds();
        Ok(scene)
    }

    /// Parse and build in one step.
    pub fn parse_and_build(&self, json_str: &str) -> VerifierResult<SceneModel> {
        let parsed = self.parse_json(json_str)?;
        self.build_scene_model(&parsed)
    }
}

impl Default for SceneParser {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_interaction_type(s: &str) -> InteractionType {
    match s.to_lowercase().as_str() {
        "click" | "press" | "tap" => InteractionType::Click,
        "grab" | "grasp" => InteractionType::Grab,
        "drag" | "move" => InteractionType::Drag,
        "slider" | "slide" => InteractionType::Slider,
        "dial" | "rotate" | "rotation" => InteractionType::Dial,
        "proximity" | "zone" => InteractionType::Proximity,
        "gaze" | "look" => InteractionType::Gaze,
        "voice" | "speech" => InteractionType::Voice,
        "twohanded" | "two_handed" | "bimanual" => InteractionType::TwoHanded,
        "gesture" => InteractionType::Gesture,
        "hover" => InteractionType::Hover,
        "toggle" | "switch" => InteractionType::Toggle,
        _ => InteractionType::Custom,
    }
}

fn parse_dependency_type(s: &str) -> DependencyType {
    match s.to_lowercase().as_str() {
        "sequential" | "sequence" => DependencyType::Sequential,
        "visibility" | "visible" => DependencyType::Visibility,
        "enable" | "enabled" => DependencyType::Enable,
        "concurrent" | "simultaneous" => DependencyType::Concurrent,
        "unlock" => DependencyType::Unlock,
        _ => DependencyType::Sequential,
    }
}

// ---------------------------------------------------------------------------
// SceneBuilder — fluent API for programmatic scene construction
// ---------------------------------------------------------------------------

/// Fluent builder for constructing scenes programmatically.
pub struct SceneBuilder {
    name: String,
    description: String,
    elements: Vec<InteractableElement>,
    dependencies: Vec<(usize, usize, DependencyType)>,
    devices: Vec<DeviceConfig>,
    transform_nodes: Vec<TransformNode>,
    coordinate_frames: HashMap<String, CoordinateFrame>,
    metadata: SceneMetadata,
}

impl SceneBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            elements: Vec::new(),
            dependencies: Vec::new(),
            devices: Vec::new(),
            transform_nodes: Vec::new(),
            coordinate_frames: HashMap::new(),
            metadata: SceneMetadata::default(),
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.metadata.author = author.into();
        self
    }

    /// Add an element and return (builder, element_index).
    pub fn add_element(mut self, element: InteractableElement) -> (Self, usize) {
        let idx = self.elements.len();
        self.elements.push(element);
        (self, idx)
    }

    /// Add a click button at a position.
    pub fn add_button(self, name: impl Into<String>, position: [f64; 3], size: f64) -> (Self, usize) {
        let half = size / 2.0;
        let elem = InteractableElement::new(name, position, InteractionType::Click).with_volume(
            Volume::Box(BoundingBox::from_center_extents(position, [half, half, half])),
        );
        self.add_element(elem)
    }

    /// Add a grab handle at a position.
    pub fn add_grab_handle(
        self,
        name: impl Into<String>,
        position: [f64; 3],
        radius: f64,
    ) -> (Self, usize) {
        let elem = InteractableElement::new(name, position, InteractionType::Grab)
            .with_volume(Volume::Sphere(Sphere::new(position, radius)));
        self.add_element(elem)
    }

    /// Add a slider element.
    pub fn add_slider(
        self,
        name: impl Into<String>,
        start: [f64; 3],
        end: [f64; 3],
        radius: f64,
    ) -> (Self, usize) {
        let center = [
            (start[0] + end[0]) * 0.5,
            (start[1] + end[1]) * 0.5,
            (start[2] + end[2]) * 0.5,
        ];
        let elem = InteractableElement::new(name, center, InteractionType::Slider)
            .with_volume(Volume::Capsule(Capsule::new(start, end, radius)));
        self.add_element(elem)
    }

    /// Add a dial element.
    pub fn add_dial(
        self,
        name: impl Into<String>,
        center: [f64; 3],
        radius: f64,
        half_height: f64,
    ) -> (Self, usize) {
        let elem = InteractableElement::new(name, center, InteractionType::Dial).with_volume(
            Volume::Cylinder(Cylinder::new(center, [0.0, 1.0, 0.0], radius, half_height)),
        );
        self.add_element(elem)
    }

    pub fn add_dependency(
        mut self,
        source: usize,
        target: usize,
        dep_type: DependencyType,
    ) -> Self {
        self.dependencies.push((source, target, dep_type));
        self
    }

    pub fn sequential(self, source: usize, target: usize) -> Self {
        self.add_dependency(source, target, DependencyType::Sequential)
    }

    pub fn visibility(self, source: usize, target: usize) -> Self {
        self.add_dependency(source, target, DependencyType::Visibility)
    }

    pub fn enable(self, source: usize, target: usize) -> Self {
        self.add_dependency(source, target, DependencyType::Enable)
    }

    pub fn unlock(self, source: usize, target: usize) -> Self {
        self.add_dependency(source, target, DependencyType::Unlock)
    }

    pub fn add_device(mut self, device: DeviceConfig) -> Self {
        self.devices.push(device);
        self
    }

    pub fn quest_3(self) -> Self {
        self.add_device(DeviceConfig::quest_3())
    }

    pub fn vision_pro(self) -> Self {
        self.add_device(DeviceConfig::vision_pro())
    }

    pub fn add_transform_node(mut self, node: TransformNode) -> (Self, usize) {
        let idx = self.transform_nodes.len();
        self.transform_nodes.push(node);
        (self, idx)
    }

    pub fn add_coordinate_frame(mut self, frame: CoordinateFrame) -> Self {
        self.coordinate_frames.insert(frame.name.clone(), frame);
        self
    }

    /// Build the final SceneModel.
    pub fn build(self) -> SceneModel {
        let mut scene = SceneModel::new(&self.name);
        scene.description = self.description;
        scene.metadata = self.metadata;

        for elem in self.elements {
            scene.add_element(elem);
        }
        for (src, tgt, dep) in &self.dependencies {
            scene.add_dependency(*src, *tgt, *dep);
        }
        scene.devices = self.devices;
        scene.transform_nodes = self.transform_nodes;
        scene.coordinate_frames = self.coordinate_frames;
        scene.recompute_bounds();
        scene
    }
}

/// Serialize a SceneModel to JSON.
pub fn scene_to_json(scene: &SceneModel) -> VerifierResult<String> {
    serde_json::to_string_pretty(scene).map_err(|e| VerifierError::SceneParsing(e.to_string()))
}

/// Deserialize a SceneModel from JSON.
pub fn scene_from_json(json: &str) -> VerifierResult<SceneModel> {
    serde_json::from_str(json).map_err(|e| VerifierError::SceneParsing(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
            "name": "TestScene",
            "description": "A test scene",
            "version": "0.1.0",
            "elements": [
                {
                    "name": "Button1",
                    "position": [0.0, 1.0, -0.5],
                    "interaction_type": "Click",
                    "volume": {
                        "type": "box",
                        "min": [-0.05, 0.95, -0.55],
                        "max": [0.05, 1.05, -0.45]
                    }
                },
                {
                    "name": "Handle1",
                    "position": [0.3, 1.2, -0.5],
                    "interaction_type": "Grab",
                    "volume": {
                        "type": "sphere",
                        "center": [0.3, 1.2, -0.5],
                        "radius": 0.04
                    }
                },
                {
                    "name": "Slider1",
                    "position": [0.0, 0.8, -0.5],
                    "interaction_type": "Slider",
                    "volume": {
                        "type": "capsule",
                        "start": [-0.1, 0.8, -0.5],
                        "end": [0.1, 0.8, -0.5],
                        "radius": 0.02
                    }
                }
            ],
            "dependencies": [
                { "source": 0, "target": 1, "dependency_type": "Sequential" },
                { "source": 0, "target": 2, "dependency_type": "Enable" }
            ],
            "metadata": {}
        }"#
    }

    #[test]
    fn test_parse_json_basic() {
        let parser = SceneParser::new();
        let parsed = parser.parse_json(sample_json()).unwrap();
        assert_eq!(parsed.name, "TestScene");
        assert_eq!(parsed.elements.len(), 3);
        assert_eq!(parsed.dependencies.len(), 2);
    }

    #[test]
    fn test_parse_and_build() {
        let parser = SceneParser::new();
        let scene = parser.parse_and_build(sample_json()).unwrap();
        assert_eq!(scene.name, "TestScene");
        assert_eq!(scene.elements.len(), 3);
        assert_eq!(scene.dependencies.len(), 2);
        assert!(scene.is_dag());
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            SceneParser::detect_format(r#"{"name": "test"}"#),
            SceneFormat::NativeJson
        );
        assert_eq!(
            SceneParser::detect_format("--- !u!29 &1"),
            SceneFormat::UnityYaml
        );
        assert_eq!(
            SceneParser::detect_format("random text"),
            SceneFormat::Unknown
        );
    }

    #[test]
    fn test_volume_parsing() {
        let json = r#"{
            "elements": [
                {
                    "name": "CylinderElem",
                    "position": [0.0, 1.0, 0.0],
                    "interaction_type": "Dial",
                    "volume": {
                        "type": "cylinder",
                        "center": [0.0, 1.0, 0.0],
                        "axis": [0.0, 1.0, 0.0],
                        "radius": 0.05,
                        "half_height": 0.1
                    }
                }
            ]
        }"#;
        let parser = SceneParser::new();
        let parsed = parser.parse_json(json).unwrap();
        assert_eq!(parsed.elements.len(), 1);
        match &parsed.elements[0].volume {
            ParsedVolume::Cylinder { radius, .. } => assert!((radius - 0.05).abs() < 1e-10),
            _ => panic!("Expected cylinder volume"),
        }
    }

    #[test]
    fn test_error_recovery_lenient() {
        let json = r#"{
            "elements": [
                {
                    "name": "Good",
                    "position": [0.0, 1.0, 0.0],
                    "interaction_type": "Click"
                },
                {
                    "name": "Bad",
                    "position": [0.0, 1.0, 0.0],
                    "interaction_type": "Click",
                    "volume": { "type": "unknown_shape" }
                }
            ]
        }"#;
        let parser = SceneParser::new();
        let parsed = parser.parse_json(json).unwrap();
        assert_eq!(parsed.elements.len(), 1);
        assert!(parsed.has_warnings());
    }

    #[test]
    fn test_strict_mode() {
        let json = r#"{
            "elements": [
                {
                    "name": "Bad",
                    "position": [0.0, 0.0, 0.0],
                    "volume": { "type": "unknown_shape" }
                }
            ]
        }"#;
        let parser = SceneParser::new().strict();
        assert!(parser.parse_json(json).is_err());
    }

    #[test]
    fn test_scene_builder() {
        let (builder, btn_idx) = SceneBuilder::new("BuiltScene")
            .description("A programmatically built scene")
            .author("test")
            .add_button("Button1", [0.0, 1.0, -0.5], 0.1);
        let (builder, handle_idx) = builder.add_grab_handle("Handle1", [0.3, 1.2, -0.5], 0.04);
        let scene = builder
            .sequential(btn_idx, handle_idx)
            .quest_3()
            .build();

        assert_eq!(scene.name, "BuiltScene");
        assert_eq!(scene.elements.len(), 2);
        assert_eq!(scene.dependencies.len(), 1);
        assert_eq!(scene.devices.len(), 1);
    }

    #[test]
    fn test_scene_serialization_roundtrip() {
        let (builder, _) = SceneBuilder::new("RoundTrip")
            .add_button("Btn", [0.0, 1.0, 0.0], 0.1);
        let scene = builder.quest_3().build();
        let json = scene_to_json(&scene).unwrap();
        let restored = scene_from_json(&json).unwrap();
        assert_eq!(restored.name, "RoundTrip");
        assert_eq!(restored.elements.len(), 1);
    }

    #[test]
    fn test_parsed_scene_validate() {
        let mut parsed = ParsedScene::new("Test");
        parsed.elements.push(ParsedElement {
            id: None,
            name: String::new(),
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0; 3],
            volume: ParsedVolume::Sphere {
                center: [0.0; 3],
                radius: 0.05,
            },
            interaction_type: "Click".to_string(),
            tags: vec![],
            properties: HashMap::new(),
            transform_node: None,
            priority: 1,
        });
        parsed.dependencies.push(ParsedDependency {
            source_index: 0,
            target_index: 99,
            dependency_type: "Sequential".to_string(),
        });
        let errors = parsed.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_interaction_type_parsing() {
        assert_eq!(parse_interaction_type("click"), InteractionType::Click);
        assert_eq!(parse_interaction_type("Grab"), InteractionType::Grab);
        assert_eq!(parse_interaction_type("DRAG"), InteractionType::Drag);
        assert_eq!(parse_interaction_type("slider"), InteractionType::Slider);
        assert_eq!(parse_interaction_type("dial"), InteractionType::Dial);
        assert_eq!(parse_interaction_type("proximity"), InteractionType::Proximity);
        assert_eq!(parse_interaction_type("gaze"), InteractionType::Gaze);
        assert_eq!(parse_interaction_type("voice"), InteractionType::Voice);
        assert_eq!(parse_interaction_type("twohanded"), InteractionType::TwoHanded);
        assert_eq!(parse_interaction_type("gesture"), InteractionType::Gesture);
        assert_eq!(parse_interaction_type("hover"), InteractionType::Hover);
        assert_eq!(parse_interaction_type("toggle"), InteractionType::Toggle);
        assert_eq!(parse_interaction_type("unknown"), InteractionType::Custom);
    }

    #[test]
    fn test_dependency_type_parsing() {
        assert_eq!(parse_dependency_type("sequential"), DependencyType::Sequential);
        assert_eq!(parse_dependency_type("Visibility"), DependencyType::Visibility);
        assert_eq!(parse_dependency_type("Enable"), DependencyType::Enable);
        assert_eq!(parse_dependency_type("concurrent"), DependencyType::Concurrent);
        assert_eq!(parse_dependency_type("unlock"), DependencyType::Unlock);
        assert_eq!(parse_dependency_type("other"), DependencyType::Sequential);
    }

    #[test]
    fn test_builder_slider_and_dial() {
        let (builder, _) = SceneBuilder::new("S")
            .add_slider("S1", [-0.1, 1.0, 0.0], [0.1, 1.0, 0.0], 0.02);
        let (builder, _) = builder.add_dial("D1", [0.0, 1.2, 0.0], 0.05, 0.02);
        let scene = builder.build();
        assert_eq!(scene.elements.len(), 2);
        assert_eq!(scene.elements[0].interaction_type, InteractionType::Slider);
        assert_eq!(scene.elements[1].interaction_type, InteractionType::Dial);
    }

    #[test]
    fn test_max_elements_limit() {
        let mut elements = String::from(r#"{"elements": ["#);
        for i in 0..5 {
            if i > 0 {
                elements.push(',');
            }
            elements.push_str(&format!(
                r#"{{"name":"E{}","position":[0,0,0],"interaction_type":"Click"}}"#,
                i
            ));
        }
        elements.push_str("]}");
        let parser = SceneParser::new().with_max_elements(3);
        assert!(parser.parse_json(&elements).is_err());
    }
}
