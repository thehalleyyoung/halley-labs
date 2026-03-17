//! USD (Universal Scene Description) scene format adapter.
//!
//! Provides `UsdSceneAdapter` for converting USD stage representations into the
//! internal `SceneModel` format used by the XR affordance verifier.
//!
//! This module is self-contained and does not depend on any external USD crate.
//! It defines its own USD type representations suitable for ingestion from
//! pre-parsed USD data (e.g. exported from a Python `pxr` pipeline or a
//! lightweight USDA text parser).

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::error::{VerifierError, VerifierResult};
use xr_types::geometry::{BoundingBox, Capsule, Cylinder, Sphere, Volume};
use xr_types::scene::{
    DependencyType, InteractableElement, InteractionType, SceneMetadata, SceneModel, TransformNode,
};

// ---------------------------------------------------------------------------
// USD format detection
// ---------------------------------------------------------------------------

/// Binary vs text USD encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsdFormat {
    /// USDA – human-readable ASCII representation.
    Usda,
    /// USDC – compact binary (crate) representation.
    Usdc,
    /// USDZ – zip archive containing USDC + assets.
    Usdz,
}

impl UsdFormat {
    /// Detect format from the first bytes of a buffer.
    ///
    /// USDC files begin with the 8-byte magic `PXR-USDC`.
    /// USDZ files begin with the standard ZIP local-file header `PK\x03\x04`.
    /// Everything else is assumed to be USDA.
    pub fn detect(bytes: &[u8]) -> Self {
        if bytes.len() >= 8 && &bytes[..8] == b"PXR-USDC" {
            UsdFormat::Usdc
        } else if bytes.len() >= 4 && &bytes[..4] == b"PK\x03\x04" {
            UsdFormat::Usdz
        } else {
            UsdFormat::Usda
        }
    }
}

// ---------------------------------------------------------------------------
// Composition arc types
// ---------------------------------------------------------------------------

/// The six USD composition arc types in strength order (LIVRPS).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CompositionArcKind {
    LocalOpinion,
    Inherits,
    VariantSets,
    References,
    Payloads,
    Specializes,
}

/// A single composition arc attached to a prim.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdCompositionArc {
    pub kind: CompositionArcKind,
    /// Asset path or prim path targeted by this arc.
    pub target_path: String,
    /// Optional layer offset (time-offset, time-scale).
    pub layer_offset: Option<[f64; 2]>,
}

// ---------------------------------------------------------------------------
// Variant sets
// ---------------------------------------------------------------------------

/// A named variant set with its available selections.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdVariantSet {
    pub name: String,
    pub variants: Vec<String>,
    pub selected: Option<String>,
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// Represents a USD layer (sublayer or root).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdLayer {
    pub identifier: String,
    pub sublayers: Vec<String>,
    pub default_prim: Option<String>,
    pub up_axis: Option<String>,
    pub meters_per_unit: Option<f64>,
    pub custom_layer_data: HashMap<String, String>,
}

impl UsdLayer {
    pub fn new(identifier: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            sublayers: Vec::new(),
            default_prim: None,
            up_axis: None,
            meters_per_unit: None,
            custom_layer_data: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Prim types
// ---------------------------------------------------------------------------

/// Well-known USD prim type names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsdPrimType {
    Xform,
    Mesh,
    Cube,
    Sphere,
    Capsule,
    Cylinder,
    Cone,
    Camera,
    DistantLight,
    DomeLight,
    SphereLight,
    Scope,
    Material,
    Shader,
    Other(String),
}

impl UsdPrimType {
    /// Parse a USD schema type string (e.g. `"UsdGeomMesh"`, `"Xform"`, `"Mesh"`).
    pub fn from_type_name(name: &str) -> Self {
        let normalized = name
            .trim_start_matches("UsdGeom")
            .trim_start_matches("UsdLux")
            .trim_start_matches("UsdShade");
        match normalized {
            "Xform" => Self::Xform,
            "Mesh" => Self::Mesh,
            "Cube" => Self::Cube,
            "Sphere" => Self::Sphere,
            "Capsule" => Self::Capsule,
            "Cylinder" => Self::Cylinder,
            "Cone" => Self::Cone,
            "Camera" => Self::Camera,
            "DistantLight" => Self::DistantLight,
            "DomeLight" => Self::DomeLight,
            "SphereLight" => Self::SphereLight,
            "Scope" => Self::Scope,
            "Material" => Self::Material,
            "Shader" => Self::Shader,
            other => Self::Other(other.to_string()),
        }
    }

    /// Returns `true` for geometry types that produce a visible shape.
    pub fn is_geometry(&self) -> bool {
        matches!(
            self,
            Self::Mesh | Self::Cube | Self::Sphere | Self::Capsule | Self::Cylinder | Self::Cone
        )
    }
}

// ---------------------------------------------------------------------------
// Attributes & relationships
// ---------------------------------------------------------------------------

/// A typed USD attribute value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UsdAttributeValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Double(f64),
    String(String),
    Token(String),
    Float3([f64; 3]),
    Double3([f64; 3]),
    Quatf([f64; 4]),
    Quatd([f64; 4]),
    Matrix4d([[f64; 4]; 4]),
    Float3Array(Vec<[f64; 3]>),
    IntArray(Vec<i64>),
    Asset(String),
}

/// A single attribute on a USD prim.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdAttribute {
    pub name: String,
    pub type_name: String,
    pub value: UsdAttributeValue,
    pub is_custom: bool,
    pub interpolation: Option<String>,
    pub namespace: Option<String>,
}

impl UsdAttribute {
    pub fn new(name: impl Into<String>, value: UsdAttributeValue) -> Self {
        let name = name.into();
        let type_name = match &value {
            UsdAttributeValue::Bool(_) => "bool",
            UsdAttributeValue::Int(_) => "int",
            UsdAttributeValue::Float(_) => "float",
            UsdAttributeValue::Double(_) => "double",
            UsdAttributeValue::String(_) => "string",
            UsdAttributeValue::Token(_) => "token",
            UsdAttributeValue::Float3(_) => "float3",
            UsdAttributeValue::Double3(_) => "double3",
            UsdAttributeValue::Quatf(_) => "quatf",
            UsdAttributeValue::Quatd(_) => "quatd",
            UsdAttributeValue::Matrix4d(_) => "matrix4d",
            UsdAttributeValue::Float3Array(_) => "float3[]",
            UsdAttributeValue::IntArray(_) => "int[]",
            UsdAttributeValue::Asset(_) => "asset",
        }
        .to_string();
        Self {
            name,
            type_name,
            value,
            is_custom: false,
            interpolation: None,
            namespace: None,
        }
    }
}

/// A USD relationship (target prim path list).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdRelationship {
    pub name: String,
    pub target_paths: Vec<String>,
    pub is_custom: bool,
}

// ---------------------------------------------------------------------------
// Geometry subset & material binding
// ---------------------------------------------------------------------------

/// A face-set partition on a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdGeomSubset {
    pub name: String,
    pub element_type: String,
    pub indices: Vec<i64>,
    pub family_name: Option<String>,
    pub material_binding: Option<String>,
}

/// A minimal USD shading material representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdShadeMaterial {
    pub path: String,
    pub display_name: Option<String>,
    pub surface_shader: Option<String>,
    pub inputs: HashMap<String, UsdAttributeValue>,
}

// ---------------------------------------------------------------------------
// Xform (transform) representation
// ---------------------------------------------------------------------------

/// Decomposed Xform ops for a prim.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdXform {
    pub translate: [f64; 3],
    /// Rotation stored as `[w, x, y, z]` quaternion.
    pub rotate_quaternion: [f64; 4],
    pub scale: [f64; 3],
}

impl Default for UsdXform {
    fn default() -> Self {
        Self {
            translate: [0.0; 3],
            rotate_quaternion: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0; 3],
        }
    }
}

impl UsdXform {
    /// Build from Euler angles (degrees) using USD's default `XYZ` rotation
    /// order.  Converts to quaternion in `[w, x, y, z]` form.
    pub fn from_euler_degrees(translate: [f64; 3], euler_xyz: [f64; 3], scale: [f64; 3]) -> Self {
        let to_rad = std::f64::consts::PI / 180.0;
        let (sx, cx) = (euler_xyz[0] * to_rad * 0.5).sin_cos();
        let (sy, cy) = (euler_xyz[1] * to_rad * 0.5).sin_cos();
        let (sz, cz) = (euler_xyz[2] * to_rad * 0.5).sin_cos();
        let w = cx * cy * cz + sx * sy * sz;
        let x = sx * cy * cz - cx * sy * sz;
        let y = cx * sy * cz + sx * cy * sz;
        let z = cx * cy * sz - sx * sy * cz;
        Self {
            translate,
            rotate_quaternion: [w, x, y, z],
            scale,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh representation
// ---------------------------------------------------------------------------

/// Minimal mesh geometry used for volume estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdMesh {
    pub points: Vec<[f64; 3]>,
    pub face_vertex_counts: Vec<i64>,
    pub face_vertex_indices: Vec<i64>,
    pub normals: Option<Vec<[f64; 3]>>,
    pub extent: Option<[[f64; 3]; 2]>,
    pub subsets: Vec<UsdGeomSubset>,
}

impl UsdMesh {
    /// Compute an axis-aligned bounding box from the point cloud.
    pub fn compute_extent(&self) -> Option<[[f64; 3]; 2]> {
        if self.points.is_empty() {
            return None;
        }
        let mut min = self.points[0];
        let mut max = self.points[0];
        for p in &self.points[1..] {
            for i in 0..3 {
                if p[i] < min[i] { min[i] = p[i]; }
                if p[i] > max[i] { max[i] = p[i]; }
            }
        }
        Some([min, max])
    }
}

// ---------------------------------------------------------------------------
// Prim
// ---------------------------------------------------------------------------

/// A single prim in a USD stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdPrim {
    /// Absolute prim path (e.g. `/World/Cube`).
    pub path: String,
    pub name: String,
    pub prim_type: UsdPrimType,
    pub active: bool,
    pub xform: UsdXform,
    pub attributes: Vec<UsdAttribute>,
    pub relationships: Vec<UsdRelationship>,
    pub composition_arcs: Vec<UsdCompositionArc>,
    pub variant_sets: Vec<UsdVariantSet>,
    pub children_paths: Vec<String>,
    pub parent_path: Option<String>,
    /// If this prim is a Mesh, holds mesh-specific data.
    pub mesh: Option<UsdMesh>,
    /// Material binding path (resolved from `material:binding` relationship).
    pub material_binding: Option<String>,
    /// Optional explicit `purpose` attribute (render | proxy | guide).
    pub purpose: Option<String>,
    /// Extent authored on gprim schema.
    pub extent: Option<[[f64; 3]; 2]>,
    /// Size for Cube prims (default 2.0 in USD).
    pub size: Option<f64>,
    /// Radius for Sphere / Capsule / Cylinder / Cone prims.
    pub radius: Option<f64>,
    /// Height for Capsule / Cylinder / Cone prims.
    pub height: Option<f64>,
    /// Axis for Capsule / Cylinder / Cone prims ("X", "Y", or "Z").
    pub axis: Option<String>,
}

impl UsdPrim {
    pub fn new(path: impl Into<String>, prim_type: UsdPrimType) -> Self {
        let path = path.into();
        let name = path.rsplit('/').next().unwrap_or("").to_string();
        Self {
            path,
            name,
            prim_type,
            active: true,
            xform: UsdXform::default(),
            attributes: Vec::new(),
            relationships: Vec::new(),
            composition_arcs: Vec::new(),
            variant_sets: Vec::new(),
            children_paths: Vec::new(),
            parent_path: None,
            mesh: None,
            material_binding: None,
            purpose: None,
            extent: None,
            size: None,
            radius: None,
            height: None,
            axis: None,
        }
    }

    /// Look up a custom attribute by name.
    pub fn get_custom_attr(&self, name: &str) -> Option<&UsdAttribute> {
        self.attributes.iter().find(|a| a.is_custom && a.name == name)
    }

    /// Look up any attribute by name.
    pub fn get_attr(&self, name: &str) -> Option<&UsdAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Look up a relationship by name.
    pub fn get_relationship(&self, name: &str) -> Option<&UsdRelationship> {
        self.relationships.iter().find(|r| r.name == name)
    }

    /// Convenience: extract `xr:interactionType` custom attribute as a string.
    pub fn xr_interaction_type(&self) -> Option<&str> {
        self.get_custom_attr("xr:interactionType").and_then(|a| match &a.value {
            UsdAttributeValue::String(s) | UsdAttributeValue::Token(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Convenience: check whether the `xr:accessible` flag is set to true.
    pub fn is_xr_accessible(&self) -> bool {
        self.get_custom_attr("xr:accessible")
            .map(|a| matches!(&a.value, UsdAttributeValue::Bool(true)))
            .unwrap_or(false)
    }

    /// Returns `true` when this prim should be treated as a visible geometry.
    pub fn is_renderable(&self) -> bool {
        self.active
            && self.prim_type.is_geometry()
            && self.purpose.as_deref() != Some("guide")
    }
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// A full USD stage – the top-level container.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsdStage {
    pub name: String,
    pub root_layer: UsdLayer,
    pub session_layer: Option<UsdLayer>,
    pub prims: Vec<UsdPrim>,
    pub materials: Vec<UsdShadeMaterial>,
    pub default_prim: Option<String>,
    pub up_axis: String,
    pub meters_per_unit: f64,
    pub time_codes_per_second: Option<f64>,
    pub start_time_code: Option<f64>,
    pub end_time_code: Option<f64>,
    /// Custom metadata on the stage pseudoroot.
    pub custom_data: HashMap<String, String>,
}

impl UsdStage {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            root_layer: UsdLayer::new(&name),
            session_layer: None,
            prims: Vec::new(),
            materials: Vec::new(),
            default_prim: None,
            up_axis: "Y".to_string(),
            meters_per_unit: 0.01,
            time_codes_per_second: None,
            start_time_code: None,
            end_time_code: None,
            custom_data: HashMap::new(),
            name,
        }
    }

    /// Find a prim by its absolute path.
    pub fn get_prim(&self, path: &str) -> Option<&UsdPrim> {
        self.prims.iter().find(|p| p.path == path)
    }

    /// Collect all prims of a given type.
    pub fn prims_of_type(&self, prim_type: &UsdPrimType) -> Vec<&UsdPrim> {
        self.prims.iter().filter(|p| &p.prim_type == prim_type).collect()
    }

    /// Iterate prims that carry `xr:accessible = true`.
    pub fn accessible_prims(&self) -> Vec<&UsdPrim> {
        self.prims.iter().filter(|p| p.is_xr_accessible()).collect()
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// Converts a `UsdStage` into the verifier's internal `SceneModel`.
pub struct UsdSceneAdapter {
    /// When true, prims without an explicit `xr:interactionType` but carrying
    /// geometry are still converted using a best-effort heuristic.
    infer_interactions: bool,
    /// Default activation radius used when a prim has no authored extent.
    default_activation_radius: f64,
    /// When true, apply the stage's `metersPerUnit` scaling to positions.
    apply_meters_per_unit: bool,
}

impl UsdSceneAdapter {
    pub fn new() -> Self {
        Self {
            infer_interactions: true,
            default_activation_radius: 0.05,
            apply_meters_per_unit: true,
        }
    }

    pub fn with_infer_interactions(mut self, enabled: bool) -> Self {
        self.infer_interactions = enabled;
        self
    }

    pub fn with_activation_radius(mut self, radius: f64) -> Self {
        self.default_activation_radius = radius;
        self
    }

    pub fn with_meters_per_unit(mut self, enabled: bool) -> Self {
        self.apply_meters_per_unit = enabled;
        self
    }

    // ------------------------------------------------------------------
    // Main entry point
    // ------------------------------------------------------------------

    /// Adapt a USD stage into a `SceneModel`.
    pub fn adapt(&self, stage: &UsdStage) -> VerifierResult<SceneModel> {
        let scale = if self.apply_meters_per_unit {
            stage.meters_per_unit
        } else {
            1.0
        };

        let mut scene = SceneModel::new(&stage.name);
        scene.metadata = self.build_metadata(stage);

        // Build path → index map for transform hierarchy.
        let mut path_to_idx: HashMap<&str, usize> = HashMap::new();
        for prim in &stage.prims {
            let parent_idx = prim
                .parent_path
                .as_deref()
                .and_then(|pp| path_to_idx.get(pp).copied());
            let node = TransformNode {
                name: prim.name.clone(),
                parent: parent_idx,
                local_position: self.scaled_position(&prim.xform.translate, scale),
                local_rotation: prim.xform.rotate_quaternion,
                local_scale: prim.xform.scale,
                children: Vec::new(),
            };
            let idx = scene.transform_nodes.len();
            scene.transform_nodes.push(node);
            path_to_idx.insert(&prim.path, idx);
        }

        // Wire children.
        for prim in &stage.prims {
            if let Some(&idx) = path_to_idx.get(prim.path.as_str()) {
                let children: Vec<usize> = prim
                    .children_paths
                    .iter()
                    .filter_map(|cp| path_to_idx.get(cp.as_str()).copied())
                    .collect();
                scene.transform_nodes[idx].children = children;
            }
        }

        // Convert interactable prims.
        let mut path_to_element: HashMap<&str, usize> = HashMap::new();
        for prim in &stage.prims {
            if !prim.active {
                continue;
            }
            if let Some(element) = self.convert_prim(prim, &path_to_idx, scale)? {
                let idx = scene.add_element(element);
                path_to_element.insert(&prim.path, idx);
            }
        }

        // Add dependencies from `xr:depends` custom relationships.
        for prim in &stage.prims {
            if let Some(rel) = prim.get_relationship("xr:depends") {
                if let Some(&src) = path_to_element.get(prim.path.as_str()) {
                    for target_path in &rel.target_paths {
                        if let Some(&tgt) = path_to_element.get(target_path.as_str()) {
                            let dep_type = self.resolve_dependency_type(prim);
                            scene.add_dependency(src, tgt, dep_type);
                        }
                    }
                }
            }
        }

        scene.recompute_bounds();
        Ok(scene)
    }

    /// Parse a JSON string representing a `UsdStage` and convert it.
    pub fn parse_json(&self, json_str: &str) -> VerifierResult<UsdStage> {
        serde_json::from_str(json_str)
            .map_err(|e| VerifierError::SceneParsing(format!("USD JSON parse error: {}", e)))
    }

    /// Convenience: parse + adapt in one call.
    pub fn parse_and_adapt(&self, json_str: &str) -> VerifierResult<SceneModel> {
        let stage = self.parse_json(json_str)?;
        self.adapt(&stage)
    }

    // ------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------

    fn build_metadata(&self, stage: &UsdStage) -> SceneMetadata {
        let mut custom = stage.custom_data.clone();
        custom.insert("upAxis".to_string(), stage.up_axis.clone());
        custom.insert(
            "metersPerUnit".to_string(),
            stage.meters_per_unit.to_string(),
        );
        if let Some(fps) = stage.time_codes_per_second {
            custom.insert("timeCodesPerSecond".to_string(), fps.to_string());
        }
        SceneMetadata {
            author: stage
                .custom_data
                .get("author")
                .cloned()
                .unwrap_or_default(),
            target_platform: stage.custom_data.get("targetPlatform").cloned(),
            custom,
            ..SceneMetadata::default()
        }
    }

    // ------------------------------------------------------------------
    // Prim conversion
    // ------------------------------------------------------------------

    fn convert_prim(
        &self,
        prim: &UsdPrim,
        path_to_idx: &HashMap<&str, usize>,
        scale: f64,
    ) -> VerifierResult<Option<InteractableElement>> {
        // Explicit XR interactable.
        if let Some(itype_str) = prim.xr_interaction_type() {
            let itype = Self::map_interaction_type_str(itype_str);
            let pos = self.scaled_position(&prim.xform.translate, scale);
            let mut elem = InteractableElement::new(&prim.name, pos, itype);
            elem.orientation = prim.xform.rotate_quaternion;
            elem.scale = prim.xform.scale;
            elem.activation_volume = self.build_volume(prim, scale);
            elem.transform_node = path_to_idx.get(prim.path.as_str()).copied();
            self.apply_custom_properties(prim, &mut elem);
            return Ok(Some(elem));
        }

        // Explicit `xr:accessible` flag – treat as Click by default.
        if prim.is_xr_accessible() {
            let pos = self.scaled_position(&prim.xform.translate, scale);
            let mut elem = InteractableElement::new(&prim.name, pos, InteractionType::Click);
            elem.orientation = prim.xform.rotate_quaternion;
            elem.scale = prim.xform.scale;
            elem.activation_volume = self.build_volume(prim, scale);
            elem.transform_node = path_to_idx.get(prim.path.as_str()).copied();
            self.apply_custom_properties(prim, &mut elem);
            return Ok(Some(elem));
        }

        // Heuristic: infer from geometry + naming conventions.
        if self.infer_interactions && prim.is_renderable() {
            if let Some(itype) = self.infer_interaction_from_name(&prim.name) {
                let pos = self.scaled_position(&prim.xform.translate, scale);
                let mut elem = InteractableElement::new(&prim.name, pos, itype);
                elem.orientation = prim.xform.rotate_quaternion;
                elem.scale = prim.xform.scale;
                elem.activation_volume = self.build_volume(prim, scale);
                elem.transform_node = path_to_idx.get(prim.path.as_str()).copied();
                elem.tags.push("inferred".to_string());
                return Ok(Some(elem));
            }
        }

        Ok(None)
    }

    // ------------------------------------------------------------------
    // Interaction type mapping
    // ------------------------------------------------------------------

    fn map_interaction_type_str(s: &str) -> InteractionType {
        match s.to_lowercase().as_str() {
            "click" | "press" | "button" | "poke" => InteractionType::Click,
            "grab" | "pickup" => InteractionType::Grab,
            "drag" | "move" => InteractionType::Drag,
            "slider" | "slide" => InteractionType::Slider,
            "dial" | "rotate" | "knob" => InteractionType::Dial,
            "proximity" | "zone" | "trigger" => InteractionType::Proximity,
            "gaze" | "look" => InteractionType::Gaze,
            "voice" | "speech" => InteractionType::Voice,
            "twohanded" | "two_handed" | "bimanual" => InteractionType::TwoHanded,
            "gesture" | "wave" | "swipe" => InteractionType::Gesture,
            "hover" | "float" => InteractionType::Hover,
            "toggle" | "switch" => InteractionType::Toggle,
            _ => InteractionType::Custom,
        }
    }

    /// Best-effort interaction type from prim name substrings.
    fn infer_interaction_from_name(&self, name: &str) -> Option<InteractionType> {
        let lower = name.to_lowercase();
        if lower.contains("button") || lower.contains("btn") {
            Some(InteractionType::Click)
        } else if lower.contains("slider") {
            Some(InteractionType::Slider)
        } else if lower.contains("dial") || lower.contains("knob") {
            Some(InteractionType::Dial)
        } else if lower.contains("grab") || lower.contains("handle") {
            Some(InteractionType::Grab)
        } else if lower.contains("toggle") || lower.contains("switch") {
            Some(InteractionType::Toggle)
        } else if lower.contains("lever") || lower.contains("drag") {
            Some(InteractionType::Drag)
        } else if lower.contains("hover") {
            Some(InteractionType::Hover)
        } else if lower.contains("gaze") {
            Some(InteractionType::Gaze)
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // Volume construction
    // ------------------------------------------------------------------

    fn build_volume(&self, prim: &UsdPrim, scale: f64) -> Volume {
        let pos = self.scaled_position(&prim.xform.translate, scale);
        let prim_scale = prim.xform.scale;

        match &prim.prim_type {
            UsdPrimType::Cube => {
                let size = prim.size.unwrap_or(2.0) * scale;
                let half = [
                    size * prim_scale[0] * 0.5,
                    size * prim_scale[1] * 0.5,
                    size * prim_scale[2] * 0.5,
                ];
                Volume::Box(BoundingBox::from_center_extents(pos, half))
            }
            UsdPrimType::Sphere => {
                let r = prim.radius.unwrap_or(1.0) * scale;
                let max_s = prim_scale[0].max(prim_scale[1]).max(prim_scale[2]);
                Volume::Sphere(Sphere::new(pos, r * max_s))
            }
            UsdPrimType::Capsule => {
                let r = prim.radius.unwrap_or(0.5) * scale;
                let h = prim.height.unwrap_or(1.0) * scale;
                let axis_vec = self.axis_vector(prim.axis.as_deref().unwrap_or("Y"));
                let half_h = h * 0.5;
                let start = [
                    pos[0] - axis_vec[0] * half_h,
                    pos[1] - axis_vec[1] * half_h,
                    pos[2] - axis_vec[2] * half_h,
                ];
                let end = [
                    pos[0] + axis_vec[0] * half_h,
                    pos[1] + axis_vec[1] * half_h,
                    pos[2] + axis_vec[2] * half_h,
                ];
                let max_s = match prim.axis.as_deref().unwrap_or("Y") {
                    "X" => prim_scale[1].max(prim_scale[2]),
                    "Z" => prim_scale[0].max(prim_scale[1]),
                    _ => prim_scale[0].max(prim_scale[2]),
                };
                Volume::Capsule(Capsule::new(start, end, r * max_s))
            }
            UsdPrimType::Cylinder => {
                let r = prim.radius.unwrap_or(1.0) * scale;
                let h = prim.height.unwrap_or(2.0) * scale;
                let axis_vec = self.axis_vector(prim.axis.as_deref().unwrap_or("Y"));
                let max_s = match prim.axis.as_deref().unwrap_or("Y") {
                    "X" => prim_scale[1].max(prim_scale[2]),
                    "Z" => prim_scale[0].max(prim_scale[1]),
                    _ => prim_scale[0].max(prim_scale[2]),
                };
                Volume::Cylinder(Cylinder::new(pos, axis_vec, r * max_s, h * 0.5))
            }
            UsdPrimType::Cone => {
                // Approximate cone with a cylinder of half the radius.
                let r = prim.radius.unwrap_or(1.0) * scale * 0.5;
                let h = prim.height.unwrap_or(2.0) * scale;
                let axis_vec = self.axis_vector(prim.axis.as_deref().unwrap_or("Y"));
                Volume::Cylinder(Cylinder::new(pos, axis_vec, r, h * 0.5))
            }
            UsdPrimType::Mesh => self.mesh_volume(prim, scale),
            _ => {
                // Fall back to authored extent or a small default sphere.
                if let Some(ext) = prim.extent {
                    let min = self.scaled_position(&ext[0], scale);
                    let max = self.scaled_position(&ext[1], scale);
                    Volume::Box(BoundingBox::new(min, max))
                } else {
                    Volume::Sphere(Sphere::new(pos, self.default_activation_radius))
                }
            }
        }
    }

    /// Build a bounding-box volume from mesh extent or point cloud.
    fn mesh_volume(&self, prim: &UsdPrim, scale: f64) -> Volume {
        // Prefer authored extent, then compute from points.
        let ext = prim
            .extent
            .or_else(|| prim.mesh.as_ref().and_then(|m| m.compute_extent()));
        if let Some(ext) = ext {
            let min = self.scaled_position(&ext[0], scale);
            let max = self.scaled_position(&ext[1], scale);
            Volume::Box(BoundingBox::new(min, max))
        } else {
            let pos = self.scaled_position(&prim.xform.translate, scale);
            Volume::Sphere(Sphere::new(pos, self.default_activation_radius))
        }
    }

    // ------------------------------------------------------------------
    // Custom properties
    // ------------------------------------------------------------------

    fn apply_custom_properties(&self, prim: &UsdPrim, elem: &mut InteractableElement) {
        for attr in &prim.attributes {
            if !attr.is_custom {
                continue;
            }
            match attr.name.as_str() {
                "xr:priority" => {
                    if let UsdAttributeValue::Int(v) = &attr.value {
                        elem.priority = (*v).max(0) as u32;
                    }
                }
                "xr:minDuration" => {
                    if let UsdAttributeValue::Double(v) | UsdAttributeValue::Float(v) = &attr.value
                    {
                        elem.min_duration = *v;
                    }
                }
                "xr:sustainedContact" => {
                    if let UsdAttributeValue::Bool(v) = &attr.value {
                        elem.sustained_contact = *v;
                    }
                }
                "xr:tags" => {
                    if let UsdAttributeValue::String(s) = &attr.value {
                        elem.tags.extend(s.split(',').map(|t| t.trim().to_string()));
                    }
                }
                _ => {
                    if let Some(key) = attr.name.strip_prefix("xr:") {
                        let val = match &attr.value {
                            UsdAttributeValue::String(s) | UsdAttributeValue::Token(s) => {
                                s.clone()
                            }
                            UsdAttributeValue::Int(v) => v.to_string(),
                            UsdAttributeValue::Float(v) | UsdAttributeValue::Double(v) => {
                                v.to_string()
                            }
                            UsdAttributeValue::Bool(v) => v.to_string(),
                            _ => continue,
                        };
                        elem.properties.insert(key.to_string(), val);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Dependency type resolution
    // ------------------------------------------------------------------

    fn resolve_dependency_type(&self, prim: &UsdPrim) -> DependencyType {
        if let Some(attr) = prim.get_custom_attr("xr:dependencyType") {
            if let UsdAttributeValue::String(s) | UsdAttributeValue::Token(s) = &attr.value {
                return match s.to_lowercase().as_str() {
                    "sequential" | "sequence" => DependencyType::Sequential,
                    "visibility" | "visible" => DependencyType::Visibility,
                    "enable" | "enabled" => DependencyType::Enable,
                    "concurrent" | "simultaneous" => DependencyType::Concurrent,
                    "unlock" => DependencyType::Unlock,
                    _ => DependencyType::Sequential,
                };
            }
        }
        DependencyType::Sequential
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn scaled_position(&self, pos: &[f64; 3], scale: f64) -> [f64; 3] {
        [pos[0] * scale, pos[1] * scale, pos[2] * scale]
    }

    fn axis_vector(&self, axis: &str) -> [f64; 3] {
        match axis {
            "X" => [1.0, 0.0, 0.0],
            "Z" => [0.0, 0.0, 1.0],
            _ => [0.0, 1.0, 0.0],
        }
    }
}

impl Default for UsdSceneAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn make_xr_attr(name: &str, value: UsdAttributeValue) -> UsdAttribute {
        UsdAttribute {
            name: name.to_string(),
            type_name: String::new(),
            value,
            is_custom: true,
            interpolation: None,
            namespace: None,
        }
    }

    fn make_accessible_prim(path: &str, prim_type: UsdPrimType) -> UsdPrim {
        let mut prim = UsdPrim::new(path, prim_type);
        prim.attributes.push(make_xr_attr(
            "xr:accessible",
            UsdAttributeValue::Bool(true),
        ));
        prim
    }

    fn make_interactable_prim(
        path: &str,
        prim_type: UsdPrimType,
        interaction: &str,
    ) -> UsdPrim {
        let mut prim = UsdPrim::new(path, prim_type);
        prim.attributes.push(make_xr_attr(
            "xr:accessible",
            UsdAttributeValue::Bool(true),
        ));
        prim.attributes.push(make_xr_attr(
            "xr:interactionType",
            UsdAttributeValue::Token(interaction.to_string()),
        ));
        prim
    }

    fn make_test_stage() -> UsdStage {
        let mut stage = UsdStage::new("TestStage");

        let root = UsdPrim::new("/World", UsdPrimType::Xform);

        let mut cube = make_interactable_prim("/World/Button", UsdPrimType::Cube, "click");
        cube.parent_path = Some("/World".to_string());
        cube.size = Some(0.1);
        cube.xform.translate = [1.0, 0.5, 0.0];

        let mut sphere = make_interactable_prim("/World/Grabball", UsdPrimType::Sphere, "grab");
        sphere.parent_path = Some("/World".to_string());
        sphere.radius = Some(0.2);
        sphere.xform.translate = [0.0, 1.0, 0.0];

        let mut capsule =
            make_interactable_prim("/World/Slider", UsdPrimType::Capsule, "slider");
        capsule.parent_path = Some("/World".to_string());
        capsule.radius = Some(0.05);
        capsule.height = Some(0.5);
        capsule.xform.translate = [-1.0, 0.5, 0.0];

        let mesh_prim = make_accessible_prim("/World/Panel", UsdPrimType::Mesh);

        stage.prims = vec![root, cube, sphere, capsule, mesh_prim];
        stage.prims[0].children_paths = vec![
            "/World/Button".to_string(),
            "/World/Grabball".to_string(),
            "/World/Slider".to_string(),
            "/World/Panel".to_string(),
        ];
        stage
    }

    // -- tests ------------------------------------------------------------

    #[test]
    fn test_format_detection() {
        assert_eq!(UsdFormat::detect(b"PXR-USDC-extra"), UsdFormat::Usdc);
        assert_eq!(UsdFormat::detect(b"PK\x03\x04rest"), UsdFormat::Usdz);
        assert_eq!(UsdFormat::detect(b"#usda 1.0\n"), UsdFormat::Usda);
        assert_eq!(UsdFormat::detect(b""), UsdFormat::Usda);
    }

    #[test]
    fn test_prim_type_parsing() {
        assert_eq!(UsdPrimType::from_type_name("UsdGeomMesh"), UsdPrimType::Mesh);
        assert_eq!(UsdPrimType::from_type_name("Xform"), UsdPrimType::Xform);
        assert_eq!(UsdPrimType::from_type_name("UsdLuxDomeLight"), UsdPrimType::DomeLight);
        assert_eq!(UsdPrimType::from_type_name("UsdShadeMaterial"), UsdPrimType::Material);
        assert!(matches!(UsdPrimType::from_type_name("CustomThing"), UsdPrimType::Other(_)));
    }

    #[test]
    fn test_prim_type_is_geometry() {
        assert!(UsdPrimType::Mesh.is_geometry());
        assert!(UsdPrimType::Cube.is_geometry());
        assert!(!UsdPrimType::Xform.is_geometry());
        assert!(!UsdPrimType::Camera.is_geometry());
    }

    #[test]
    fn test_xform_from_euler() {
        let xf = UsdXform::from_euler_degrees([0.0; 3], [0.0, 90.0, 0.0], [1.0; 3]);
        // 90° around Y: w ≈ cos(45°) ≈ 0.707, y ≈ sin(45°) ≈ 0.707
        assert!((xf.rotate_quaternion[0] - 0.7071).abs() < 0.01);
        assert!((xf.rotate_quaternion[2] - 0.7071).abs() < 0.01);
    }

    #[test]
    fn test_mesh_compute_extent() {
        let mesh = UsdMesh {
            points: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -0.5, 0.5]],
            face_vertex_counts: Vec::new(),
            face_vertex_indices: Vec::new(),
            normals: None,
            extent: None,
            subsets: Vec::new(),
        };
        let ext = mesh.compute_extent().unwrap();
        assert_eq!(ext[0], [-1.0, -0.5, 0.0]);
        assert_eq!(ext[1], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mesh_compute_extent_empty() {
        let mesh = UsdMesh {
            points: Vec::new(),
            face_vertex_counts: Vec::new(),
            face_vertex_indices: Vec::new(),
            normals: None,
            extent: None,
            subsets: Vec::new(),
        };
        assert!(mesh.compute_extent().is_none());
    }

    #[test]
    fn test_prim_xr_helpers() {
        let prim = make_interactable_prim("/X", UsdPrimType::Cube, "grab");
        assert_eq!(prim.xr_interaction_type(), Some("grab"));
        assert!(prim.is_xr_accessible());

        let plain = UsdPrim::new("/Y", UsdPrimType::Mesh);
        assert_eq!(plain.xr_interaction_type(), None);
        assert!(!plain.is_xr_accessible());
    }

    #[test]
    fn test_adapt_basic_stage() {
        let stage = make_test_stage();
        let adapter = UsdSceneAdapter::new();
        let scene = adapter.adapt(&stage).unwrap();

        assert_eq!(scene.name, "TestStage");
        // Button, Grabball, Slider, Panel (accessible)
        assert_eq!(scene.elements.len(), 4);
        assert_eq!(scene.transform_nodes.len(), 5);
    }

    #[test]
    fn test_adapt_interaction_types() {
        let stage = make_test_stage();
        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();

        let types: Vec<InteractionType> =
            scene.elements.iter().map(|e| e.interaction_type).collect();
        assert!(types.contains(&InteractionType::Click));
        assert!(types.contains(&InteractionType::Grab));
        assert!(types.contains(&InteractionType::Slider));
    }

    #[test]
    fn test_adapt_inactive_prim_skipped() {
        let mut stage = UsdStage::new("InactiveTest");
        let mut prim = make_interactable_prim("/Thing", UsdPrimType::Cube, "click");
        prim.active = false;
        stage.prims = vec![prim];
        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        assert_eq!(scene.elements.len(), 0);
    }

    #[test]
    fn test_adapt_infer_from_name() {
        let mut stage = UsdStage::new("InferTest");
        let mut prim = UsdPrim::new("/ButtonThing", UsdPrimType::Cube);
        prim.active = true;
        prim.size = Some(0.1);
        stage.prims = vec![prim];

        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        assert_eq!(scene.elements.len(), 1);
        assert_eq!(scene.elements[0].interaction_type, InteractionType::Click);
        assert!(scene.elements[0].tags.contains(&"inferred".to_string()));
    }

    #[test]
    fn test_adapt_no_infer_when_disabled() {
        let mut stage = UsdStage::new("NoInfer");
        let prim = UsdPrim::new("/ButtonThing", UsdPrimType::Cube);
        stage.prims = vec![prim];

        let scene = UsdSceneAdapter::new()
            .with_infer_interactions(false)
            .adapt(&stage)
            .unwrap();
        assert_eq!(scene.elements.len(), 0);
    }

    #[test]
    fn test_adapt_dependencies() {
        let mut stage = UsdStage::new("DepTest");
        let mut p1 = make_interactable_prim("/A", UsdPrimType::Cube, "click");
        p1.relationships.push(UsdRelationship {
            name: "xr:depends".to_string(),
            target_paths: vec!["/B".to_string()],
            is_custom: true,
        });
        p1.attributes.push(make_xr_attr(
            "xr:dependencyType",
            UsdAttributeValue::Token("enable".to_string()),
        ));
        let p2 = make_interactable_prim("/B", UsdPrimType::Cube, "click");
        stage.prims = vec![p1, p2];

        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        assert_eq!(scene.dependencies.len(), 1);
        assert_eq!(scene.dependencies[0].dependency_type, DependencyType::Enable);
    }

    #[test]
    fn test_adapt_custom_properties() {
        let mut stage = UsdStage::new("PropTest");
        let mut prim = make_interactable_prim("/Widget", UsdPrimType::Cube, "click");
        prim.attributes.push(make_xr_attr(
            "xr:priority",
            UsdAttributeValue::Int(5),
        ));
        prim.attributes.push(make_xr_attr(
            "xr:minDuration",
            UsdAttributeValue::Double(0.3),
        ));
        prim.attributes.push(make_xr_attr(
            "xr:sustainedContact",
            UsdAttributeValue::Bool(true),
        ));
        prim.attributes.push(make_xr_attr(
            "xr:tags",
            UsdAttributeValue::String("ui,primary".to_string()),
        ));
        prim.attributes.push(make_xr_attr(
            "xr:tooltip",
            UsdAttributeValue::String("Press me".to_string()),
        ));
        stage.prims = vec![prim];

        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        let elem = &scene.elements[0];
        assert_eq!(elem.priority, 5);
        assert!((elem.min_duration - 0.3).abs() < 1e-10);
        assert!(elem.sustained_contact);
        assert!(elem.tags.contains(&"ui".to_string()));
        assert!(elem.tags.contains(&"primary".to_string()));
        assert_eq!(elem.properties.get("tooltip").unwrap(), "Press me");
    }

    #[test]
    fn test_adapt_meters_per_unit_scaling() {
        let mut stage = UsdStage::new("ScaleTest");
        stage.meters_per_unit = 0.01; // centimeters
        let mut prim = make_interactable_prim("/Btn", UsdPrimType::Cube, "click");
        prim.xform.translate = [100.0, 50.0, 0.0];
        stage.prims = vec![prim];

        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        let pos = scene.elements[0].position;
        assert!((pos[0] - 1.0).abs() < 1e-10);
        assert!((pos[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adapt_no_scaling_when_disabled() {
        let mut stage = UsdStage::new("NoScaleTest");
        stage.meters_per_unit = 0.01;
        let mut prim = make_interactable_prim("/Btn", UsdPrimType::Cube, "click");
        prim.xform.translate = [100.0, 50.0, 0.0];
        stage.prims = vec![prim];

        let scene = UsdSceneAdapter::new()
            .with_meters_per_unit(false)
            .adapt(&stage)
            .unwrap();
        let pos = scene.elements[0].position;
        assert!((pos[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_cube() {
        let mut prim = UsdPrim::new("/C", UsdPrimType::Cube);
        prim.size = Some(2.0);
        let adapter = UsdSceneAdapter::new().with_meters_per_unit(false);
        let vol = adapter.build_volume(&prim, 1.0);
        match vol {
            Volume::Box(bb) => {
                assert!((bb.min[0] - (-1.0)).abs() < 1e-10);
                assert!((bb.max[0] - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected Box volume for Cube prim"),
        }
    }

    #[test]
    fn test_volume_sphere() {
        let mut prim = UsdPrim::new("/S", UsdPrimType::Sphere);
        prim.radius = Some(0.5);
        let adapter = UsdSceneAdapter::new().with_meters_per_unit(false);
        let vol = adapter.build_volume(&prim, 1.0);
        match vol {
            Volume::Sphere(s) => assert!((s.radius - 0.5).abs() < 1e-10),
            _ => panic!("expected Sphere volume"),
        }
    }

    #[test]
    fn test_volume_capsule() {
        let mut prim = UsdPrim::new("/Cap", UsdPrimType::Capsule);
        prim.radius = Some(0.1);
        prim.height = Some(1.0);
        prim.axis = Some("Y".to_string());
        let adapter = UsdSceneAdapter::new().with_meters_per_unit(false);
        let vol = adapter.build_volume(&prim, 1.0);
        match vol {
            Volume::Capsule(c) => {
                assert!((c.radius - 0.1).abs() < 1e-10);
                assert!((c.start[1] - (-0.5)).abs() < 1e-10);
                assert!((c.end[1] - 0.5).abs() < 1e-10);
            }
            _ => panic!("expected Capsule volume"),
        }
    }

    #[test]
    fn test_volume_cylinder() {
        let mut prim = UsdPrim::new("/Cyl", UsdPrimType::Cylinder);
        prim.radius = Some(0.3);
        prim.height = Some(2.0);
        let adapter = UsdSceneAdapter::new().with_meters_per_unit(false);
        let vol = adapter.build_volume(&prim, 1.0);
        assert!(matches!(vol, Volume::Cylinder(_)));
    }

    #[test]
    fn test_volume_mesh_with_extent() {
        let mut prim = UsdPrim::new("/M", UsdPrimType::Mesh);
        prim.extent = Some([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]);
        let adapter = UsdSceneAdapter::new().with_meters_per_unit(false);
        let vol = adapter.build_volume(&prim, 1.0);
        match vol {
            Volume::Box(bb) => {
                assert_eq!(bb.min, [-1.0, -2.0, -3.0]);
                assert_eq!(bb.max, [1.0, 2.0, 3.0]);
            }
            _ => panic!("expected Box volume from mesh extent"),
        }
    }

    #[test]
    fn test_stage_helpers() {
        let stage = make_test_stage();
        assert!(stage.get_prim("/World/Button").is_some());
        assert!(stage.get_prim("/Nonexistent").is_none());
        assert_eq!(stage.prims_of_type(&UsdPrimType::Sphere).len(), 1);
        // Button, Grabball, Slider, Panel are all accessible
        assert_eq!(stage.accessible_prims().len(), 4);
    }

    #[test]
    fn test_composition_arc_serde() {
        let arc = UsdCompositionArc {
            kind: CompositionArcKind::References,
            target_path: "./props.usd".to_string(),
            layer_offset: Some([0.0, 1.0]),
        };
        let json = serde_json::to_string(&arc).unwrap();
        let roundtrip: UsdCompositionArc = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.kind, CompositionArcKind::References);
        assert_eq!(roundtrip.target_path, "./props.usd");
    }

    #[test]
    fn test_variant_set_serde() {
        let vs = UsdVariantSet {
            name: "color".to_string(),
            variants: vec!["red".to_string(), "blue".to_string()],
            selected: Some("red".to_string()),
        };
        let json = serde_json::to_string(&vs).unwrap();
        let roundtrip: UsdVariantSet = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.variants.len(), 2);
    }

    #[test]
    fn test_layer_serde() {
        let layer = UsdLayer::new("root.usda");
        let json = serde_json::to_string(&layer).unwrap();
        let roundtrip: UsdLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.identifier, "root.usda");
    }

    #[test]
    fn test_stage_serde_roundtrip() {
        let stage = make_test_stage();
        let json = serde_json::to_string(&stage).unwrap();
        let roundtrip: UsdStage = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.name, "TestStage");
        assert_eq!(roundtrip.prims.len(), stage.prims.len());
    }

    #[test]
    fn test_parse_json_convenience() {
        let stage = make_test_stage();
        let json = serde_json::to_string(&stage).unwrap();
        let adapter = UsdSceneAdapter::new();
        let parsed = adapter.parse_json(&json).unwrap();
        assert_eq!(parsed.name, "TestStage");
    }

    #[test]
    fn test_parse_and_adapt() {
        let stage = make_test_stage();
        let json = serde_json::to_string(&stage).unwrap();
        let scene = UsdSceneAdapter::new().parse_and_adapt(&json).unwrap();
        assert_eq!(scene.name, "TestStage");
        assert_eq!(scene.elements.len(), 4);
    }

    #[test]
    fn test_interaction_type_mapping() {
        assert_eq!(
            UsdSceneAdapter::map_interaction_type_str("click"),
            InteractionType::Click
        );
        assert_eq!(
            UsdSceneAdapter::map_interaction_type_str("Grab"),
            InteractionType::Grab
        );
        assert_eq!(
            UsdSceneAdapter::map_interaction_type_str("SLIDER"),
            InteractionType::Slider
        );
        assert_eq!(
            UsdSceneAdapter::map_interaction_type_str("dial"),
            InteractionType::Dial
        );
        assert_eq!(
            UsdSceneAdapter::map_interaction_type_str("unknownThing"),
            InteractionType::Custom
        );
    }

    #[test]
    fn test_transform_hierarchy() {
        let stage = make_test_stage();
        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        // /World is idx 0, its children should reference 1..4
        assert_eq!(scene.transform_nodes[0].children.len(), 4);
        assert!(scene.transform_nodes[1].parent == Some(0));
    }

    #[test]
    fn test_metadata() {
        let mut stage = make_test_stage();
        stage.custom_data.insert("author".to_string(), "test_author".to_string());
        stage.time_codes_per_second = Some(24.0);
        let scene = UsdSceneAdapter::new().adapt(&stage).unwrap();
        assert_eq!(scene.metadata.author, "test_author");
        assert_eq!(scene.metadata.custom.get("upAxis").unwrap(), "Y");
        assert!(scene.metadata.custom.contains_key("timeCodesPerSecond"));
    }

    #[test]
    fn test_geom_subset_serde() {
        let subset = UsdGeomSubset {
            name: "front_faces".to_string(),
            element_type: "face".to_string(),
            indices: vec![0, 1, 2],
            family_name: Some("materialBind".to_string()),
            material_binding: Some("/Materials/Red".to_string()),
        };
        let json = serde_json::to_string(&subset).unwrap();
        let rt: UsdGeomSubset = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.indices.len(), 3);
    }

    #[test]
    fn test_material_serde() {
        let mat = UsdShadeMaterial {
            path: "/Materials/Default".to_string(),
            display_name: Some("Default".to_string()),
            surface_shader: Some("/Materials/Default/Surface".to_string()),
            inputs: HashMap::new(),
        };
        let json = serde_json::to_string(&mat).unwrap();
        let rt: UsdShadeMaterial = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.path, "/Materials/Default");
    }
}
