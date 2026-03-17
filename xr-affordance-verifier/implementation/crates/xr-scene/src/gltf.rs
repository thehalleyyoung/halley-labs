//! glTF 2.0 scene format adapter.
//!
//! Provides `GltfSceneAdapter` for converting glTF 2.0 scene descriptions
//! (both `.gltf` JSON and `.glb` binary detection) into the internal
//! `SceneModel` format.  Interaction annotations are read from the
//! `XR_accessibility` extras convention.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use xr_types::error::{VerifierError, VerifierResult};
use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::scene::{
    DependencyType, InteractableElement, InteractionType, SceneMetadata, SceneModel, TransformNode,
};

// ---------------------------------------------------------------------------
// glTF 2.0 JSON types
// ---------------------------------------------------------------------------

/// Top-level glTF 2.0 document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfScene {
    pub asset: GltfAsset,
    #[serde(default)]
    pub scenes: Vec<GltfSceneEntry>,
    #[serde(default)]
    pub nodes: Vec<GltfNode>,
    #[serde(default)]
    pub meshes: Vec<GltfMesh>,
    #[serde(default)]
    pub accessors: Vec<GltfAccessor>,
    #[serde(default)]
    pub buffer_views: Vec<GltfBufferView>,
    #[serde(default)]
    pub buffers: Vec<GltfBuffer>,
    #[serde(default)]
    pub materials: Vec<GltfMaterial>,
    #[serde(default)]
    pub animations: Vec<GltfAnimation>,
    #[serde(default)]
    pub skins: Vec<GltfSkin>,
    #[serde(default)]
    pub cameras: Vec<GltfCamera>,
    #[serde(default)]
    pub scene: Option<usize>,
    #[serde(default)]
    pub extensions_used: Vec<String>,
    #[serde(default)]
    pub extensions_required: Vec<String>,
    #[serde(default)]
    pub extras: Option<GltfExtras>,
}

/// glTF asset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfAsset {
    pub version: String,
    #[serde(default)]
    pub generator: Option<String>,
    #[serde(default)]
    pub copyright: Option<String>,
    #[serde(default)]
    pub min_version: Option<String>,
}

/// A named scene that references root node indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfSceneEntry {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub nodes: Vec<usize>,
    #[serde(default)]
    pub extras: Option<GltfExtras>,
}

/// A node in the scene graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfNode {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub children: Vec<usize>,
    /// Translation [x, y, z].
    #[serde(default = "default_translation")]
    pub translation: [f64; 3],
    /// Rotation as quaternion [x, y, z, w] (glTF convention).
    #[serde(default = "default_rotation")]
    pub rotation: [f64; 4],
    /// Scale [x, y, z].
    #[serde(default = "default_scale")]
    pub scale: [f64; 3],
    /// Optional 4×4 column-major matrix (overrides TRS if present).
    #[serde(default)]
    pub matrix: Option<[f64; 16]>,
    /// Index into the meshes array.
    #[serde(default)]
    pub mesh: Option<usize>,
    /// Index into the cameras array.
    #[serde(default)]
    pub camera: Option<usize>,
    /// Index into the skins array.
    #[serde(default)]
    pub skin: Option<usize>,
    #[serde(default)]
    pub weights: Vec<f64>,
    #[serde(default)]
    pub extras: Option<GltfExtras>,
    #[serde(default)]
    pub extensions: Option<HashMap<String, serde_json::Value>>,
}

fn default_translation() -> [f64; 3] {
    [0.0; 3]
}
fn default_rotation() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}
fn default_scale() -> [f64; 3] {
    [1.0; 3]
}

/// A mesh consisting of one or more primitives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfMesh {
    #[serde(default)]
    pub name: Option<String>,
    pub primitives: Vec<GltfPrimitive>,
    #[serde(default)]
    pub weights: Vec<f64>,
    #[serde(default)]
    pub extras: Option<GltfExtras>,
}

/// A single mesh primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfPrimitive {
    pub attributes: HashMap<String, usize>,
    #[serde(default)]
    pub indices: Option<usize>,
    #[serde(default)]
    pub material: Option<usize>,
    #[serde(default = "default_primitive_mode")]
    pub mode: u32,
    #[serde(default)]
    pub targets: Vec<HashMap<String, usize>>,
    #[serde(default)]
    pub extras: Option<GltfExtras>,
}

fn default_primitive_mode() -> u32 {
    4 // TRIANGLES
}

/// An accessor into a buffer view.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfAccessor {
    #[serde(default)]
    pub buffer_view: Option<usize>,
    #[serde(default)]
    pub byte_offset: usize,
    pub component_type: u32,
    #[serde(default)]
    pub normalized: bool,
    pub count: usize,
    #[serde(rename = "type")]
    pub accessor_type: String,
    #[serde(default)]
    pub max: Option<Vec<f64>>,
    #[serde(default)]
    pub min: Option<Vec<f64>>,
    #[serde(default)]
    pub name: Option<String>,
}

/// A buffer view (a slice of a buffer).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfBufferView {
    pub buffer: usize,
    #[serde(default)]
    pub byte_offset: usize,
    pub byte_length: usize,
    #[serde(default)]
    pub byte_stride: Option<usize>,
    #[serde(default)]
    pub target: Option<u32>,
    #[serde(default)]
    pub name: Option<String>,
}

/// A data buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfBuffer {
    pub byte_length: usize,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

/// A PBR material.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfMaterial {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub double_sided: bool,
    #[serde(default = "default_alpha_mode")]
    pub alpha_mode: String,
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f64,
    #[serde(default)]
    pub extras: Option<GltfExtras>,
}

fn default_alpha_mode() -> String {
    "OPAQUE".to_string()
}
fn default_alpha_cutoff() -> f64 {
    0.5
}

/// Animation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfAnimation {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub channels: Vec<GltfAnimationChannel>,
    #[serde(default)]
    pub samplers: Vec<GltfAnimationSampler>,
}

/// Animation channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfAnimationChannel {
    pub sampler: usize,
    pub target: GltfAnimationTarget,
}

/// Animation target.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfAnimationTarget {
    #[serde(default)]
    pub node: Option<usize>,
    pub path: String,
}

/// Animation sampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfAnimationSampler {
    pub input: usize,
    pub output: usize,
    #[serde(default = "default_interpolation")]
    pub interpolation: String,
}

fn default_interpolation() -> String {
    "LINEAR".to_string()
}

/// Skin for skeletal animation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfSkin {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub inverse_bind_matrices: Option<usize>,
    pub joints: Vec<usize>,
    #[serde(default)]
    pub skeleton: Option<usize>,
}

/// Camera definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfCamera {
    #[serde(rename = "type")]
    pub camera_type: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub perspective: Option<GltfPerspective>,
    #[serde(default)]
    pub orthographic: Option<GltfOrthographic>,
}

/// Perspective camera parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfPerspective {
    pub yfov: f64,
    #[serde(default)]
    pub aspect_ratio: Option<f64>,
    pub znear: f64,
    #[serde(default)]
    pub zfar: Option<f64>,
}

/// Orthographic camera parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GltfOrthographic {
    pub xmag: f64,
    pub ymag: f64,
    pub znear: f64,
    pub zfar: f64,
}

/// Arbitrary extras attached to any glTF object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfExtras {
    #[serde(flatten)]
    pub values: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// XR_accessibility extension convention
// ---------------------------------------------------------------------------

/// Annotation placed in a node's `extensions["XR_accessibility"]` object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct XrAccessibilityAnnotation {
    #[serde(default)]
    pub interaction_type: Option<String>,
    #[serde(default)]
    pub actuator: Option<String>,
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub priority: Option<u32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(default)]
    pub activation_radius: Option<f64>,
    #[serde(default)]
    pub sustained_contact: Option<bool>,
    #[serde(default)]
    pub min_duration: Option<f64>,
    #[serde(default)]
    pub dependencies: Vec<XrDependencyAnnotation>,
}

/// Dependency annotation inside `XR_accessibility`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct XrDependencyAnnotation {
    pub target_node: usize,
    #[serde(default = "default_dep_type")]
    pub dependency_type: String,
}

fn default_dep_type() -> String {
    "sequential".to_string()
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Detected file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GltfFormat {
    /// JSON-based `.gltf`.
    Gltf,
    /// Binary `.glb` container.
    Glb,
}

impl GltfFormat {
    /// Detect format from a file path extension.
    pub fn from_path(path: &str) -> Option<Self> {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".glb") {
            Some(GltfFormat::Glb)
        } else if lower.ends_with(".gltf") {
            Some(GltfFormat::Gltf)
        } else {
            None
        }
    }

    /// Detect format from the first bytes of a file.
    /// GLB files start with the magic bytes `0x46546C67` ("glTF" little-endian).
    pub fn from_magic(bytes: &[u8]) -> Option<Self> {
        if bytes.len() >= 4 && bytes[0..4] == [0x67, 0x6C, 0x54, 0x46] {
            Some(GltfFormat::Glb)
        } else if bytes.first().copied() == Some(b'{') {
            Some(GltfFormat::Gltf)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// Adapter that converts a `GltfScene` document into a `SceneModel`.
pub struct GltfSceneAdapter {
    default_activation_radius: f64,
    require_annotation: bool,
}

impl GltfSceneAdapter {
    pub fn new() -> Self {
        Self {
            default_activation_radius: 0.05,
            require_annotation: false,
        }
    }

    /// Set the default activation sphere radius for nodes without mesh bounds.
    pub fn with_activation_radius(mut self, radius: f64) -> Self {
        self.default_activation_radius = radius;
        self
    }

    /// When `true`, only nodes carrying an `XR_accessibility` annotation are
    /// converted to interactable elements; unannotated mesh nodes are ignored.
    pub fn with_require_annotation(mut self, require: bool) -> Self {
        self.require_annotation = require;
        self
    }

    // -- public entry points ------------------------------------------------

    /// Parse a glTF JSON string into a `GltfScene`.
    pub fn parse_json(&self, json_str: &str) -> VerifierResult<GltfScene> {
        serde_json::from_str(json_str)
            .map_err(|e| VerifierError::SceneParsing(format!("glTF JSON parse error: {e}")))
    }

    /// Convenience: parse + adapt in one call.
    pub fn parse_and_adapt(&self, json_str: &str) -> VerifierResult<SceneModel> {
        let gltf = self.parse_json(json_str)?;
        self.adapt(&gltf)
    }

    /// Convert a parsed `GltfScene` into a `SceneModel`.
    pub fn adapt(&self, gltf: &GltfScene) -> VerifierResult<SceneModel> {
        self.validate_asset(gltf)?;

        let scene_entry = self.active_scene(gltf);
        let scene_name = scene_entry
            .and_then(|s| s.name.clone())
            .unwrap_or_else(|| "glTF Scene".to_string());

        let mut model = SceneModel::new(&scene_name);

        // Metadata
        model.metadata = SceneMetadata {
            author: gltf
                .asset
                .copyright
                .clone()
                .unwrap_or_default(),
            custom: self.build_custom_metadata(gltf),
            ..SceneMetadata::default()
        };

        // Build transform hierarchy
        let root_nodes: Vec<usize> = scene_entry
            .map(|s| s.nodes.clone())
            .unwrap_or_else(|| (0..gltf.nodes.len()).collect());

        let mut node_to_transform: HashMap<usize, usize> = HashMap::new();
        self.build_transform_hierarchy(gltf, &root_nodes, None, &mut model, &mut node_to_transform);

        // Wire up children in TransformNode entries
        for (node_idx, &tf_idx) in &node_to_transform {
            let children_tf: Vec<usize> = gltf.nodes[*node_idx]
                .children
                .iter()
                .filter_map(|c| node_to_transform.get(c).copied())
                .collect();
            model.transform_nodes[tf_idx].children = children_tf;
        }

        // Convert annotated / mesh-bearing nodes into InteractableElements
        let mut node_to_element: HashMap<usize, usize> = HashMap::new();
        for (node_idx, node) in gltf.nodes.iter().enumerate() {
            if let Some(element) = self.convert_node(gltf, node_idx, node, &node_to_transform)? {
                let elem_idx = model.add_element(element);
                node_to_element.insert(node_idx, elem_idx);
            }
        }

        // Extract dependency edges from XR_accessibility annotations
        for (node_idx, node) in gltf.nodes.iter().enumerate() {
            if let Some(annotation) = self.parse_xr_annotation(node) {
                if let Some(&src_elem) = node_to_element.get(&node_idx) {
                    for dep in &annotation.dependencies {
                        if let Some(&tgt_elem) = node_to_element.get(&dep.target_node) {
                            let dep_type = Self::map_dependency_type(&dep.dependency_type);
                            model.add_dependency(src_elem, tgt_elem, dep_type);
                        }
                    }
                }
            }
        }

        // Extract animation-implied sequential dependencies
        self.extract_animation_dependencies(gltf, &node_to_element, &mut model);

        model.recompute_bounds();
        Ok(model)
    }

    // -- private helpers ----------------------------------------------------

    fn validate_asset(&self, gltf: &GltfScene) -> VerifierResult<()> {
        if !gltf.asset.version.starts_with("2.") {
            return Err(VerifierError::SceneParsing(format!(
                "Unsupported glTF version: {} (expected 2.x)",
                gltf.asset.version
            )));
        }
        Ok(())
    }

    fn active_scene<'a>(&self, gltf: &'a GltfScene) -> Option<&'a GltfSceneEntry> {
        gltf.scene
            .and_then(|idx| gltf.scenes.get(idx))
            .or_else(|| gltf.scenes.first())
    }

    fn build_custom_metadata(&self, gltf: &GltfScene) -> HashMap<String, String> {
        let mut custom = HashMap::new();
        custom.insert("gltf_version".to_string(), gltf.asset.version.clone());
        if let Some(gen) = &gltf.asset.generator {
            custom.insert("gltf_generator".to_string(), gen.clone());
        }
        if !gltf.extensions_used.is_empty() {
            custom.insert(
                "gltf_extensions_used".to_string(),
                gltf.extensions_used.join(","),
            );
        }
        custom
    }

    /// Recursively build `TransformNode` entries for every node reachable from
    /// the given root list.
    fn build_transform_hierarchy(
        &self,
        gltf: &GltfScene,
        node_indices: &[usize],
        parent_tf: Option<usize>,
        model: &mut SceneModel,
        mapping: &mut HashMap<usize, usize>,
    ) {
        for &ni in node_indices {
            if mapping.contains_key(&ni) {
                continue; // avoid cycles
            }
            let node = match gltf.nodes.get(ni) {
                Some(n) => n,
                None => continue,
            };

            let (pos, rot, scl) = self.extract_trs(node);

            let tf = TransformNode {
                name: node.name.clone().unwrap_or_else(|| format!("node_{ni}")),
                parent: parent_tf,
                local_position: pos,
                local_rotation: rot,
                local_scale: scl,
                children: Vec::new(), // filled in later
            };
            let tf_idx = model.transform_nodes.len();
            model.transform_nodes.push(tf);
            mapping.insert(ni, tf_idx);

            self.build_transform_hierarchy(gltf, &node.children, Some(tf_idx), model, mapping);
        }
    }

    /// Extract translation / rotation / scale from a node.
    /// If a matrix is present it takes priority (decomposed into TRS).
    fn extract_trs(&self, node: &GltfNode) -> ([f64; 3], [f64; 4], [f64; 3]) {
        if let Some(m) = &node.matrix {
            self.decompose_matrix(m)
        } else {
            let pos = node.translation;
            // glTF stores quaternion as [x,y,z,w] → internal [w,x,y,z]
            let rot = [
                node.rotation[3],
                node.rotation[0],
                node.rotation[1],
                node.rotation[2],
            ];
            let scl = node.scale;
            (pos, rot, scl)
        }
    }

    /// Decompose a column-major 4×4 matrix into (translation, rotation[w,x,y,z], scale).
    fn decompose_matrix(&self, m: &[f64; 16]) -> ([f64; 3], [f64; 4], [f64; 3]) {
        let tx = m[12];
        let ty = m[13];
        let tz = m[14];

        let sx = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        let sy = (m[4] * m[4] + m[5] * m[5] + m[6] * m[6]).sqrt();
        let sz = (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]).sqrt();

        let (sx, sy, sz) = if sx < 1e-12 || sy < 1e-12 || sz < 1e-12 {
            (1.0, 1.0, 1.0)
        } else {
            (sx, sy, sz)
        };

        // Normalised rotation columns
        let r00 = m[0] / sx;
        let r01 = m[4] / sy;
        let r02 = m[8] / sz;
        let r10 = m[1] / sx;
        let r11 = m[5] / sy;
        let r12 = m[9] / sz;
        let r20 = m[2] / sx;
        let r21 = m[6] / sy;
        let r22 = m[10] / sz;

        let trace = r00 + r11 + r22;
        let (w, x, y, z) = if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            (
                0.25 / s,
                (r21 - r12) * s,
                (r02 - r20) * s,
                (r10 - r01) * s,
            )
        } else if r00 > r11 && r00 > r22 {
            let s = 2.0 * (1.0 + r00 - r11 - r22).sqrt();
            (
                (r21 - r12) / s,
                0.25 * s,
                (r01 + r10) / s,
                (r02 + r20) / s,
            )
        } else if r11 > r22 {
            let s = 2.0 * (1.0 + r11 - r00 - r22).sqrt();
            (
                (r02 - r20) / s,
                (r01 + r10) / s,
                0.25 * s,
                (r12 + r21) / s,
            )
        } else {
            let s = 2.0 * (1.0 + r22 - r00 - r11).sqrt();
            (
                (r10 - r01) / s,
                (r02 + r20) / s,
                (r12 + r21) / s,
                0.25 * s,
            )
        };

        ([tx, ty, tz], [w, x, y, z], [sx, sy, sz])
    }

    /// Try to convert a glTF node into an `InteractableElement`.
    fn convert_node(
        &self,
        gltf: &GltfScene,
        node_idx: usize,
        node: &GltfNode,
        transform_map: &HashMap<usize, usize>,
    ) -> VerifierResult<Option<InteractableElement>> {
        let annotation = self.parse_xr_annotation(node);

        // If we require annotations and this node has none, skip it.
        if self.require_annotation && annotation.is_none() {
            return Ok(None);
        }

        // If there is no annotation and no mesh, skip.
        if annotation.is_none() && node.mesh.is_none() {
            return Ok(None);
        }

        let (pos, rot, scl) = self.extract_trs(node);

        let interaction_type = annotation
            .as_ref()
            .and_then(|a| a.interaction_type.as_deref())
            .map(Self::map_interaction_type)
            .unwrap_or(InteractionType::Click);

        let name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("node_{node_idx}"));

        let volume = self.build_activation_volume(gltf, node, &pos, &scl, &annotation);

        let mut element = InteractableElement::new(&name, pos, interaction_type);
        element.orientation = rot;
        element.scale = scl;
        element.activation_volume = volume;
        element.transform_node = transform_map.get(&node_idx).copied();

        if let Some(ann) = &annotation {
            if let Some(p) = ann.priority {
                element.priority = p;
            }
            if let Some(d) = ann.min_duration {
                element.min_duration = d;
            }
            if let Some(sc) = ann.sustained_contact {
                element.sustained_contact = sc;
            }
            element.tags.extend(ann.tags.iter().cloned());
            for (k, v) in &ann.properties {
                element.properties.insert(k.clone(), v.clone());
            }
        }

        element
            .properties
            .insert("gltf_node_index".to_string(), node_idx.to_string());

        Ok(Some(element))
    }

    /// Parse the `XR_accessibility` extension from a node.
    fn parse_xr_annotation(&self, node: &GltfNode) -> Option<XrAccessibilityAnnotation> {
        let ext_map = node.extensions.as_ref()?;
        let value = ext_map.get("XR_accessibility")?;
        serde_json::from_value(value.clone()).ok()
    }

    /// Build an activation volume for a node.
    /// Prefers accessor min/max bounds when available, falls back to a default sphere.
    fn build_activation_volume(
        &self,
        gltf: &GltfScene,
        node: &GltfNode,
        position: &[f64; 3],
        scale: &[f64; 3],
        annotation: &Option<XrAccessibilityAnnotation>,
    ) -> Volume {
        // If the annotation specifies a radius, use a sphere.
        if let Some(ann) = annotation {
            if let Some(r) = ann.activation_radius {
                return Volume::Sphere(Sphere::new(*position, r));
            }
        }

        // Try to derive bounds from the mesh's POSITION accessor.
        if let Some(mesh_idx) = node.mesh {
            if let Some(vol) = self.mesh_bounds_volume(gltf, mesh_idx, position, scale) {
                return vol;
            }
        }

        // Fallback: small sphere at the node position.
        Volume::Sphere(Sphere::new(*position, self.default_activation_radius))
    }

    /// Compute a `Volume::Box` from the POSITION accessor's min/max on a mesh.
    fn mesh_bounds_volume(
        &self,
        gltf: &GltfScene,
        mesh_idx: usize,
        position: &[f64; 3],
        scale: &[f64; 3],
    ) -> Option<Volume> {
        let mesh = gltf.meshes.get(mesh_idx)?;

        let mut global_min = [f64::MAX; 3];
        let mut global_max = [f64::MIN; 3];
        let mut found = false;

        for prim in &mesh.primitives {
            let acc_idx = prim.attributes.get("POSITION")?;
            let accessor = gltf.accessors.get(*acc_idx)?;

            if let (Some(acc_min), Some(acc_max)) = (&accessor.min, &accessor.max) {
                if acc_min.len() >= 3 && acc_max.len() >= 3 {
                    for i in 0..3 {
                        global_min[i] = global_min[i].min(acc_min[i]);
                        global_max[i] = global_max[i].max(acc_max[i]);
                    }
                    found = true;
                }
            }
        }

        if !found {
            return None;
        }

        // Apply node scale and translate to world position
        let center = [
            position[0] + (global_min[0] + global_max[0]) * 0.5 * scale[0],
            position[1] + (global_min[1] + global_max[1]) * 0.5 * scale[1],
            position[2] + (global_min[2] + global_max[2]) * 0.5 * scale[2],
        ];
        let half_extents = [
            (global_max[0] - global_min[0]) * 0.5 * scale[0].abs(),
            (global_max[1] - global_min[1]) * 0.5 * scale[1].abs(),
            (global_max[2] - global_min[2]) * 0.5 * scale[2].abs(),
        ];

        Some(Volume::Box(BoundingBox::from_center_extents(
            center,
            half_extents,
        )))
    }

    // -- mapping helpers ----------------------------------------------------

    /// Map an interaction type string from the annotation to `InteractionType`.
    fn map_interaction_type(s: &str) -> InteractionType {
        match s.to_ascii_lowercase().as_str() {
            "click" | "press" | "poke" => InteractionType::Click,
            "grab" | "grasp" => InteractionType::Grab,
            "drag" | "move" => InteractionType::Drag,
            "slider" | "slide" => InteractionType::Slider,
            "dial" | "rotate" | "knob" => InteractionType::Dial,
            "proximity" | "zone" => InteractionType::Proximity,
            "gaze" | "look" => InteractionType::Gaze,
            "voice" | "speech" => InteractionType::Voice,
            "two_handed" | "twohanded" | "bimanual" => InteractionType::TwoHanded,
            "gesture" => InteractionType::Gesture,
            "hover" => InteractionType::Hover,
            "toggle" | "switch" => InteractionType::Toggle,
            _ => InteractionType::Custom,
        }
    }

    fn map_dependency_type(s: &str) -> DependencyType {
        match s.to_ascii_lowercase().as_str() {
            "sequential" | "sequence" => DependencyType::Sequential,
            "visibility" | "visible" => DependencyType::Visibility,
            "enable" | "enabled" => DependencyType::Enable,
            "concurrent" | "simultaneous" => DependencyType::Concurrent,
            "unlock" => DependencyType::Unlock,
            _ => DependencyType::Sequential,
        }
    }

    /// Infer sequential dependencies from animations that target different
    /// nodes on non-overlapping time channels.
    fn extract_animation_dependencies(
        &self,
        gltf: &GltfScene,
        node_to_element: &HashMap<usize, usize>,
        model: &mut SceneModel,
    ) {
        for anim in &gltf.animations {
            // Collect (node_idx, sampler_input_accessor) pairs.
            let mut node_samplers: Vec<(usize, usize)> = Vec::new();
            for ch in &anim.channels {
                if let Some(ni) = ch.target.node {
                    if node_to_element.contains_key(&ni) {
                        node_samplers.push((ni, ch.sampler));
                    }
                }
            }

            // For each consecutive pair of channels that reference *different*
            // elements, add a Sequential dependency.
            let mut prev: Option<usize> = None;
            for &(ni, _) in &node_samplers {
                if let Some(prev_ni) = prev {
                    if prev_ni != ni {
                        if let (Some(&src), Some(&tgt)) =
                            (node_to_element.get(&prev_ni), node_to_element.get(&ni))
                        {
                            model.add_dependency(src, tgt, DependencyType::Sequential);
                        }
                    }
                }
                prev = Some(ni);
            }
        }
    }

    /// Detect file format from path or magic bytes.
    pub fn detect_format(path: Option<&str>, bytes: Option<&[u8]>) -> Option<GltfFormat> {
        if let Some(p) = path {
            if let Some(fmt) = GltfFormat::from_path(p) {
                return Some(fmt);
            }
        }
        if let Some(b) = bytes {
            return GltfFormat::from_magic(b);
        }
        None
    }
}

impl Default for GltfSceneAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Create a minimal glTF scene with the given nodes and meshes (for testing).
pub fn make_test_gltf(
    name: &str,
    nodes: Vec<GltfNode>,
    meshes: Vec<GltfMesh>,
    accessors: Vec<GltfAccessor>,
) -> GltfScene {
    GltfScene {
        asset: GltfAsset {
            version: "2.0".to_string(),
            generator: Some("xr-scene-test".to_string()),
            copyright: None,
            min_version: None,
        },
        scenes: vec![GltfSceneEntry {
            name: Some(name.to_string()),
            nodes: (0..nodes.len()).collect(),
            extras: None,
        }],
        nodes,
        meshes,
        accessors,
        buffer_views: Vec::new(),
        buffers: Vec::new(),
        materials: Vec::new(),
        animations: Vec::new(),
        skins: Vec::new(),
        cameras: Vec::new(),
        scene: Some(0),
        extensions_used: Vec::new(),
        extensions_required: Vec::new(),
        extras: None,
    }
}

/// Helper: create a `GltfNode` carrying an `XR_accessibility` annotation.
pub fn make_annotated_node(
    name: &str,
    position: [f64; 3],
    interaction: &str,
    mesh: Option<usize>,
) -> GltfNode {
    let annotation = serde_json::json!({
        "interactionType": interaction,
    });
    let mut extensions = HashMap::new();
    extensions.insert("XR_accessibility".to_string(), annotation);
    GltfNode {
        name: Some(name.to_string()),
        children: Vec::new(),
        translation: position,
        rotation: default_rotation(),
        scale: default_scale(),
        matrix: None,
        mesh,
        camera: None,
        skin: None,
        weights: Vec::new(),
        extras: None,
        extensions: Some(extensions),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_accessor(min: [f64; 3], max: [f64; 3]) -> GltfAccessor {
        GltfAccessor {
            buffer_view: Some(0),
            byte_offset: 0,
            component_type: 5126, // FLOAT
            normalized: false,
            count: 36,
            accessor_type: "VEC3".to_string(),
            min: Some(min.to_vec()),
            max: Some(max.to_vec()),
            name: None,
        }
    }

    fn simple_mesh(pos_accessor: usize) -> GltfMesh {
        let mut attrs = HashMap::new();
        attrs.insert("POSITION".to_string(), pos_accessor);
        GltfMesh {
            name: Some("TestMesh".to_string()),
            primitives: vec![GltfPrimitive {
                attributes: attrs,
                indices: None,
                material: None,
                mode: 4,
                targets: Vec::new(),
                extras: None,
            }],
            weights: Vec::new(),
            extras: None,
        }
    }

    fn plain_node(name: &str, position: [f64; 3], mesh: Option<usize>) -> GltfNode {
        GltfNode {
            name: Some(name.to_string()),
            children: Vec::new(),
            translation: position,
            rotation: default_rotation(),
            scale: default_scale(),
            matrix: None,
            mesh,
            camera: None,
            skin: None,
            weights: Vec::new(),
            extras: None,
            extensions: None,
        }
    }

    // -- basic conversion ---------------------------------------------------

    #[test]
    fn test_adapt_empty_scene() {
        let gltf = make_test_gltf("Empty", Vec::new(), Vec::new(), Vec::new());
        let adapter = GltfSceneAdapter::new();
        let model = adapter.adapt(&gltf).unwrap();
        assert_eq!(model.name, "Empty");
        assert!(model.elements.is_empty());
    }

    #[test]
    fn test_adapt_node_with_mesh_bounds() {
        let acc = simple_accessor([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        let mesh = simple_mesh(0);
        let node = plain_node("Cube", [1.0, 2.0, 3.0], Some(0));
        let gltf = make_test_gltf("MeshScene", vec![node], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();

        assert_eq!(model.elements.len(), 1);
        assert_eq!(model.elements[0].name, "Cube");
        assert!((model.elements[0].position[0] - 1.0).abs() < 1e-10);
        match &model.elements[0].activation_volume {
            Volume::Box(bb) => {
                let ext = bb.extents();
                assert!((ext[0] - 1.0).abs() < 1e-10);
            }
            other => panic!("Expected Volume::Box, got {:?}", other),
        }
    }

    #[test]
    fn test_adapt_annotated_node() {
        let node = make_annotated_node("Lever", [0.0, 1.5, -1.0], "grab", None);
        let gltf = make_test_gltf("Annotated", vec![node], Vec::new(), Vec::new());
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();

        assert_eq!(model.elements.len(), 1);
        assert_eq!(model.elements[0].interaction_type, InteractionType::Grab);
    }

    #[test]
    fn test_skip_unannotated_meshless_nodes() {
        let node = plain_node("EmptyNode", [0.0; 3], None);
        let gltf = make_test_gltf("NoMesh", vec![node], Vec::new(), Vec::new());
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();
        assert!(model.elements.is_empty());
    }

    #[test]
    fn test_require_annotation_mode() {
        let node = plain_node("MeshOnly", [0.0; 3], Some(0));
        let acc = simple_accessor([-1.0; 3], [1.0; 3]);
        let mesh = simple_mesh(0);
        let gltf = make_test_gltf("FilterTest", vec![node], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new()
            .with_require_annotation(true)
            .adapt(&gltf)
            .unwrap();
        assert!(model.elements.is_empty());
    }

    // -- transform ----------------------------------------------------------

    #[test]
    fn test_rotation_conversion() {
        // glTF quaternion [x,y,z,w] = [0.1, 0.2, 0.3, 0.9]
        let mut node = plain_node("Rotated", [0.0; 3], Some(0));
        node.rotation = [0.1, 0.2, 0.3, 0.9];
        let acc = simple_accessor([-0.5; 3], [0.5; 3]);
        let mesh = simple_mesh(0);
        let gltf = make_test_gltf("RotScene", vec![node], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();

        // Internal format is [w,x,y,z]
        let orient = model.elements[0].orientation;
        assert!((orient[0] - 0.9).abs() < 1e-10); // w
        assert!((orient[1] - 0.1).abs() < 1e-10); // x
    }

    #[test]
    fn test_matrix_decomposition() {
        // Identity matrix
        #[rustfmt::skip]
        let identity: [f64; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            5.0, 6.0, 7.0, 1.0,
        ];
        let mut node = plain_node("MatrixNode", [0.0; 3], Some(0));
        node.matrix = Some(identity);
        let acc = simple_accessor([-0.5; 3], [0.5; 3]);
        let mesh = simple_mesh(0);
        let gltf = make_test_gltf("MatrixScene", vec![node], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();

        assert!((model.elements[0].position[0] - 5.0).abs() < 1e-10);
        assert!((model.elements[0].position[1] - 6.0).abs() < 1e-10);
        assert!((model.elements[0].position[2] - 7.0).abs() < 1e-10);
    }

    // -- hierarchy ----------------------------------------------------------

    #[test]
    fn test_transform_hierarchy() {
        let mut parent = plain_node("Parent", [1.0, 0.0, 0.0], Some(0));
        parent.children = vec![1];
        let child = plain_node("Child", [0.0, 2.0, 0.0], Some(0));
        let acc = simple_accessor([-0.1; 3], [0.1; 3]);
        let mesh = simple_mesh(0);
        let gltf = make_test_gltf("Hierarchy", vec![parent, child], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();

        assert_eq!(model.transform_nodes.len(), 2);
        assert!(model.transform_nodes[0].parent.is_none());
        assert_eq!(model.transform_nodes[1].parent, Some(0));
        assert_eq!(model.transform_nodes[0].children, vec![1]);
    }

    // -- interaction type mapping -------------------------------------------

    #[test]
    fn test_interaction_type_mapping() {
        let cases = vec![
            ("click", InteractionType::Click),
            ("grab", InteractionType::Grab),
            ("drag", InteractionType::Drag),
            ("slider", InteractionType::Slider),
            ("dial", InteractionType::Dial),
            ("proximity", InteractionType::Proximity),
            ("gaze", InteractionType::Gaze),
            ("voice", InteractionType::Voice),
            ("two_handed", InteractionType::TwoHanded),
            ("gesture", InteractionType::Gesture),
            ("hover", InteractionType::Hover),
            ("toggle", InteractionType::Toggle),
            ("unknown_type", InteractionType::Custom),
        ];
        for (input, expected) in cases {
            assert_eq!(
                GltfSceneAdapter::map_interaction_type(input),
                expected,
                "failed for {input}"
            );
        }
    }

    // -- dependency extraction ----------------------------------------------

    #[test]
    fn test_annotation_dependencies() {
        let dep_ann = serde_json::json!({
            "interactionType": "click",
            "dependencies": [
                { "targetNode": 1, "dependencyType": "sequential" }
            ]
        });
        let mut ext0 = HashMap::new();
        ext0.insert("XR_accessibility".to_string(), dep_ann);
        let node0 = GltfNode {
            name: Some("Button".to_string()),
            extensions: Some(ext0),
            mesh: Some(0),
            ..plain_node("", [0.0; 3], None)
        };
        let node1 = make_annotated_node("Door", [2.0, 0.0, 0.0], "grab", Some(0));
        let acc = simple_accessor([-0.1; 3], [0.1; 3]);
        let mesh = simple_mesh(0);
        let gltf = make_test_gltf("DepScene", vec![node0, node1], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();

        assert_eq!(model.dependencies.len(), 1);
        assert_eq!(model.dependencies[0].dependency_type, DependencyType::Sequential);
    }

    // -- format detection ---------------------------------------------------

    #[test]
    fn test_format_from_path() {
        assert_eq!(GltfFormat::from_path("model.gltf"), Some(GltfFormat::Gltf));
        assert_eq!(GltfFormat::from_path("model.GLB"), Some(GltfFormat::Glb));
        assert_eq!(GltfFormat::from_path("model.obj"), None);
    }

    #[test]
    fn test_format_from_magic() {
        let glb_magic: [u8; 4] = [0x67, 0x6C, 0x54, 0x46];
        assert_eq!(GltfFormat::from_magic(&glb_magic), Some(GltfFormat::Glb));
        assert_eq!(GltfFormat::from_magic(b"{\"asset\""), Some(GltfFormat::Gltf));
        assert_eq!(GltfFormat::from_magic(b"PNG"), None);
    }

    #[test]
    fn test_detect_format_combined() {
        assert_eq!(
            GltfSceneAdapter::detect_format(Some("foo.glb"), None),
            Some(GltfFormat::Glb)
        );
        let json_bytes = b"{\"asset\":{}}";
        assert_eq!(
            GltfSceneAdapter::detect_format(None, Some(json_bytes)),
            Some(GltfFormat::Gltf)
        );
    }

    // -- JSON round-trip ----------------------------------------------------

    #[test]
    fn test_parse_json_roundtrip() {
        let node = make_annotated_node("TestBtn", [0.0, 1.0, 0.0], "click", None);
        let gltf = make_test_gltf("RT", vec![node], Vec::new(), Vec::new());
        let json = serde_json::to_string(&gltf).unwrap();
        let adapter = GltfSceneAdapter::new();
        let parsed = adapter.parse_json(&json).unwrap();
        assert_eq!(parsed.asset.version, "2.0");
        assert_eq!(parsed.nodes.len(), 1);
    }

    #[test]
    fn test_parse_and_adapt() {
        let node = make_annotated_node("Btn", [0.0; 3], "click", None);
        let gltf = make_test_gltf("PA", vec![node], Vec::new(), Vec::new());
        let json = serde_json::to_string(&gltf).unwrap();
        let model = GltfSceneAdapter::new().parse_and_adapt(&json).unwrap();
        assert_eq!(model.elements.len(), 1);
    }

    // -- metadata -----------------------------------------------------------

    #[test]
    fn test_metadata_propagation() {
        let gltf = make_test_gltf("MetaTest", Vec::new(), Vec::new(), Vec::new());
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();
        assert_eq!(
            model.metadata.custom.get("gltf_version").map(|s| s.as_str()),
            Some("2.0")
        );
        assert_eq!(
            model.metadata.custom.get("gltf_generator").map(|s| s.as_str()),
            Some("xr-scene-test")
        );
    }

    // -- version validation -------------------------------------------------

    #[test]
    fn test_reject_unsupported_version() {
        let mut gltf = make_test_gltf("Old", Vec::new(), Vec::new(), Vec::new());
        gltf.asset.version = "1.0".to_string();
        let result = GltfSceneAdapter::new().adapt(&gltf);
        assert!(result.is_err());
    }

    // -- activation volume fallback -----------------------------------------

    #[test]
    fn test_activation_volume_fallback_sphere() {
        let node = make_annotated_node("NoMesh", [1.0, 2.0, 3.0], "click", None);
        let gltf = make_test_gltf("Fallback", vec![node], Vec::new(), Vec::new());
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();
        match &model.elements[0].activation_volume {
            Volume::Sphere(s) => {
                assert!((s.radius - 0.05).abs() < 1e-10);
                assert!((s.center[0] - 1.0).abs() < 1e-10);
            }
            other => panic!("Expected fallback sphere, got {:?}", other),
        }
    }

    #[test]
    fn test_activation_radius_override() {
        let ann = serde_json::json!({
            "interactionType": "proximity",
            "activationRadius": 2.5,
        });
        let mut ext = HashMap::new();
        ext.insert("XR_accessibility".to_string(), ann);
        let node = GltfNode {
            name: Some("Zone".to_string()),
            extensions: Some(ext),
            mesh: Some(0),
            ..plain_node("", [0.0; 3], None)
        };
        let acc = simple_accessor([-1.0; 3], [1.0; 3]);
        let mesh = simple_mesh(0);
        let gltf = make_test_gltf("RadiusOverride", vec![node], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();
        match &model.elements[0].activation_volume {
            Volume::Sphere(s) => assert!((s.radius - 2.5).abs() < 1e-10),
            other => panic!("Expected sphere with custom radius, got {:?}", other),
        }
    }

    // -- scale applied to bounds --------------------------------------------

    #[test]
    fn test_scale_applied_to_mesh_bounds() {
        let acc = simple_accessor([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let mesh = simple_mesh(0);
        let mut node = plain_node("Scaled", [0.0; 3], Some(0));
        node.scale = [2.0, 3.0, 4.0];
        let gltf = make_test_gltf("ScaleTest", vec![node], vec![mesh], vec![acc]);
        let model = GltfSceneAdapter::new().adapt(&gltf).unwrap();
        match &model.elements[0].activation_volume {
            Volume::Box(bb) => {
                let ext = bb.extents();
                assert!((ext[0] - 4.0).abs() < 1e-10); // 2.0 * 2.0
                assert!((ext[1] - 6.0).abs() < 1e-10); // 2.0 * 3.0
                assert!((ext[2] - 8.0).abs() < 1e-10); // 2.0 * 4.0
            }
            other => panic!("Expected scaled box, got {:?}", other),
        }
    }
}
