//! glTF 2.0 scene importer for Choreo.
//!
//! Parses the JSON representation of glTF files (`.gltf`, not binary `.glb`)
//! and converts the scene graph into a Choreo [`SceneConfiguration`].

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use choreo_types::{
    EntityId, RegionId, SceneConfiguration, SceneEntity, SpatialRegion, Transform3D, AABB,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum GltfError {
    #[error("failed to parse glTF JSON: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("invalid glTF: {0}")]
    ValidationError(String),

    #[error("unsupported glTF feature: {0}")]
    UnsupportedFeature(String),

    #[error("node index {0} out of range (document has {1} nodes)")]
    NodeIndexOutOfRange(usize, usize),

    #[error("mesh index {0} out of range (document has {1} meshes)")]
    MeshIndexOutOfRange(usize, usize),

    #[error("accessor index {0} out of range (document has {1} accessors)")]
    AccessorIndexOutOfRange(usize, usize),
}

pub type GltfResult<T> = Result<T, GltfError>;

// ---------------------------------------------------------------------------
// glTF JSON schema structs (subset)
// ---------------------------------------------------------------------------

/// Top-level glTF 2.0 document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfDocument {
    pub asset: GltfAsset,
    #[serde(default)]
    pub scenes: Vec<GltfScene>,
    #[serde(default)]
    pub nodes: Vec<GltfNode>,
    #[serde(default)]
    pub meshes: Vec<GltfMesh>,
    #[serde(default)]
    pub accessors: Vec<GltfAccessor>,
    #[serde(rename = "bufferViews", default)]
    pub buffer_views: Vec<GltfBufferView>,
    #[serde(default)]
    pub buffers: Vec<GltfBuffer>,
    #[serde(default)]
    pub cameras: Vec<GltfCamera>,
    #[serde(default)]
    pub scene: Option<usize>,
}

/// Asset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfAsset {
    pub version: String,
    #[serde(default)]
    pub generator: Option<String>,
    #[serde(rename = "minVersion", default)]
    pub min_version: Option<String>,
}

/// A glTF scene – a set of root node indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfScene {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub nodes: Vec<usize>,
}

/// A node in the scene graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfNode {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub children: Vec<usize>,
    #[serde(default)]
    pub mesh: Option<usize>,
    #[serde(default)]
    pub camera: Option<usize>,
    #[serde(default)]
    pub translation: Option<[f64; 3]>,
    #[serde(default)]
    pub rotation: Option<[f64; 4]>,
    #[serde(default)]
    pub scale: Option<[f64; 3]>,
    #[serde(default)]
    pub matrix: Option<[f64; 16]>,
}

/// A mesh with primitives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfMesh {
    #[serde(default)]
    pub name: Option<String>,
    pub primitives: Vec<GltfPrimitive>,
}

/// A mesh primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfPrimitive {
    #[serde(default)]
    pub attributes: HashMap<String, usize>,
    #[serde(default)]
    pub indices: Option<usize>,
    #[serde(default)]
    pub mode: Option<u32>,
}

/// An accessor into buffer data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfAccessor {
    #[serde(rename = "bufferView", default)]
    pub buffer_view: Option<usize>,
    #[serde(rename = "byteOffset", default)]
    pub byte_offset: usize,
    #[serde(rename = "componentType")]
    pub component_type: u32,
    pub count: usize,
    #[serde(rename = "type")]
    pub accessor_type: String,
    #[serde(default)]
    pub min: Option<Vec<f64>>,
    #[serde(default)]
    pub max: Option<Vec<f64>>,
    #[serde(default)]
    pub normalized: bool,
}

/// A buffer view into a buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfBufferView {
    pub buffer: usize,
    #[serde(rename = "byteOffset", default)]
    pub byte_offset: usize,
    #[serde(rename = "byteLength")]
    pub byte_length: usize,
    #[serde(rename = "byteStride", default)]
    pub byte_stride: Option<usize>,
    #[serde(default)]
    pub target: Option<u32>,
}

/// A buffer (we only store metadata; actual binary data is not loaded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfBuffer {
    #[serde(rename = "byteLength")]
    pub byte_length: usize,
    #[serde(default)]
    pub uri: Option<String>,
}

/// A camera definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GltfCamera {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub camera_type: String,
}

// ---------------------------------------------------------------------------
// Node transform helpers
// ---------------------------------------------------------------------------

/// Resolved world-space transform for a node.
#[derive(Debug, Clone)]
struct ResolvedNode {
    name: String,
    world_transform: Transform3D,
    mesh_index: Option<usize>,
    camera_index: Option<usize>,
}

fn node_local_transform(node: &GltfNode) -> Transform3D {
    if let Some(m) = &node.matrix {
        decompose_matrix(m)
    } else {
        Transform3D {
            position: node.translation.unwrap_or([0.0, 0.0, 0.0]),
            rotation: node.rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            scale: node.scale.unwrap_or([1.0, 1.0, 1.0]),
        }
    }
}

/// Decompose a column-major 4×4 matrix into TRS.
fn decompose_matrix(m: &[f64; 16]) -> Transform3D {
    let translation = [m[12], m[13], m[14]];

    // Extract scale as column lengths.
    let sx = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
    let sy = (m[4] * m[4] + m[5] * m[5] + m[6] * m[6]).sqrt();
    let sz = (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]).sqrt();
    let scale = [sx, sy, sz];

    // Normalised rotation matrix columns.
    let (r00, r10, r20) = if sx > 1e-12 {
        (m[0] / sx, m[1] / sx, m[2] / sx)
    } else {
        (1.0, 0.0, 0.0)
    };
    let (r01, r11, r21) = if sy > 1e-12 {
        (m[4] / sy, m[5] / sy, m[6] / sy)
    } else {
        (0.0, 1.0, 0.0)
    };
    let (r02, r12, r22) = if sz > 1e-12 {
        (m[8] / sz, m[9] / sz, m[10] / sz)
    } else {
        (0.0, 0.0, 1.0)
    };

    // Convert 3×3 rotation to quaternion (Shepperd's method).
    let trace = r00 + r11 + r22;
    let rotation = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        [
            (r21 - r12) * s, // x
            (r02 - r20) * s, // y
            (r10 - r01) * s, // z
            0.25 / s,        // w
        ]
    } else if r00 > r11 && r00 > r22 {
        let s = 2.0 * (1.0 + r00 - r11 - r22).sqrt();
        [
            0.25 * s,
            (r01 + r10) / s,
            (r02 + r20) / s,
            (r21 - r12) / s,
        ]
    } else if r11 > r22 {
        let s = 2.0 * (1.0 + r11 - r00 - r22).sqrt();
        [
            (r01 + r10) / s,
            0.25 * s,
            (r12 + r21) / s,
            (r02 - r20) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + r22 - r00 - r11).sqrt();
        [
            (r02 + r20) / s,
            (r12 + r21) / s,
            0.25 * s,
            (r10 - r01) / s,
        ]
    };

    Transform3D {
        position: translation,
        rotation,
        scale,
    }
}

/// Combine parent and child transforms (parent applied first).
fn compose_transforms(parent: &Transform3D, child: &Transform3D) -> Transform3D {
    use nalgebra as na;
    let pq = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
        parent.rotation[3],
        parent.rotation[0],
        parent.rotation[1],
        parent.rotation[2],
    ));
    let cq = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
        child.rotation[3],
        child.rotation[0],
        child.rotation[1],
        child.rotation[2],
    ));

    let ps = na::Vector3::new(parent.scale[0], parent.scale[1], parent.scale[2]);
    let cp = na::Vector3::new(child.position[0], child.position[1], child.position[2]);

    // Scale the child position, rotate, then translate.
    let scaled = na::Vector3::new(cp.x * ps.x, cp.y * ps.y, cp.z * ps.z);
    let rotated = pq * scaled;
    let position = [
        parent.position[0] + rotated.x,
        parent.position[1] + rotated.y,
        parent.position[2] + rotated.z,
    ];

    let combined_rotation = pq * cq;
    let q = combined_rotation.quaternion();
    let rotation = [q.i, q.j, q.k, q.w];

    let scale = [
        parent.scale[0] * child.scale[0],
        parent.scale[1] * child.scale[1],
        parent.scale[2] * child.scale[2],
    ];

    Transform3D {
        position,
        rotation,
        scale,
    }
}

// ---------------------------------------------------------------------------
// Importer
// ---------------------------------------------------------------------------

/// Imports glTF 2.0 JSON scenes into Choreo [`SceneConfiguration`].
pub struct GltfImporter {
    /// Prefix for generated entity IDs (default: `"gltf"`).
    pub entity_prefix: String,
    /// Default bounding box half-extent for nodes without mesh data.
    pub default_extent: f64,
}

impl Default for GltfImporter {
    fn default() -> Self {
        Self {
            entity_prefix: "gltf".to_string(),
            default_extent: 0.5,
        }
    }
}

impl GltfImporter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Import a glTF scene from a JSON string.
    pub fn import_from_str(&self, json: &str) -> GltfResult<SceneConfiguration> {
        let doc: GltfDocument = serde_json::from_str(json)?;
        self.import_document(&doc)
    }

    /// Import a glTF scene from a file path.
    pub fn import_from_file(&self, path: &Path) -> GltfResult<SceneConfiguration> {
        let contents = std::fs::read_to_string(path)?;
        self.import_from_str(&contents)
    }

    /// Import the default scene (or scene 0) from a parsed document.
    pub fn import_document(&self, doc: &GltfDocument) -> GltfResult<SceneConfiguration> {
        self.validate_document(doc)?;

        let scene_idx = doc.scene.unwrap_or(0);
        if scene_idx >= doc.scenes.len() {
            return Err(GltfError::ValidationError(format!(
                "default scene index {} out of range (document has {} scenes)",
                scene_idx,
                doc.scenes.len()
            )));
        }

        let scene = &doc.scenes[scene_idx];
        let resolved = self.resolve_scene_nodes(doc, scene)?;
        self.build_scene_config(doc, &resolved)
    }

    // -- validation --

    fn validate_document(&self, doc: &GltfDocument) -> GltfResult<()> {
        if doc.asset.version != "2.0" {
            return Err(GltfError::UnsupportedFeature(format!(
                "glTF version {} (only 2.0 supported)",
                doc.asset.version
            )));
        }
        if doc.scenes.is_empty() {
            return Err(GltfError::ValidationError(
                "document has no scenes".to_string(),
            ));
        }
        Ok(())
    }

    // -- node traversal --

    fn resolve_scene_nodes(
        &self,
        doc: &GltfDocument,
        scene: &GltfScene,
    ) -> GltfResult<Vec<ResolvedNode>> {
        let mut resolved = Vec::new();
        let mut counter: usize = 0;
        for &root_idx in &scene.nodes {
            self.resolve_node_recursive(
                doc,
                root_idx,
                &Transform3D::identity(),
                &mut resolved,
                &mut counter,
            )?;
        }
        Ok(resolved)
    }

    fn resolve_node_recursive(
        &self,
        doc: &GltfDocument,
        node_idx: usize,
        parent_transform: &Transform3D,
        out: &mut Vec<ResolvedNode>,
        counter: &mut usize,
    ) -> GltfResult<()> {
        if node_idx >= doc.nodes.len() {
            return Err(GltfError::NodeIndexOutOfRange(node_idx, doc.nodes.len()));
        }

        let node = &doc.nodes[node_idx];
        let local = node_local_transform(node);
        let world = compose_transforms(parent_transform, &local);

        let name = node
            .name
            .clone()
            .unwrap_or_else(|| format!("node_{}", counter));
        *counter += 1;

        out.push(ResolvedNode {
            name,
            world_transform: world.clone(),
            mesh_index: node.mesh,
            camera_index: node.camera,
        });

        for &child_idx in &node.children {
            self.resolve_node_recursive(doc, child_idx, &world, out, counter)?;
        }

        Ok(())
    }

    // -- conversion to Choreo types --

    fn build_scene_config(
        &self,
        doc: &GltfDocument,
        resolved: &[ResolvedNode],
    ) -> GltfResult<SceneConfiguration> {
        let mut config = SceneConfiguration::new();

        for rn in resolved {
            let suffix = if rn.camera_index.is_some() {
                // Camera nodes keep their name as-is (useful for identifying
                // potential User/Head entities during later pipeline stages).
                &rn.name
            } else {
                &rn.name
            };
            let entity_id = EntityId(format!("{}_{}", self.entity_prefix, suffix));
            let bounds = self.compute_node_bounds(doc, rn)?;
            let region = Some(SpatialRegion::Aabb(bounds));

            let region_id = RegionId(format!("{}_{}_region", self.entity_prefix, rn.name));
            config
                .regions
                .insert(region_id, SpatialRegion::Aabb(bounds));

            let entity = SceneEntity {
                id: entity_id.clone(),
                transform: rn.world_transform.clone(),
                bounds,
                region,
            };
            config.entities.insert(entity_id, entity);
        }

        Ok(config)
    }

    fn compute_node_bounds(&self, doc: &GltfDocument, rn: &ResolvedNode) -> GltfResult<AABB> {
        if let Some(mesh_idx) = rn.mesh_index {
            if mesh_idx >= doc.meshes.len() {
                return Err(GltfError::MeshIndexOutOfRange(mesh_idx, doc.meshes.len()));
            }
            let mesh = &doc.meshes[mesh_idx];
            self.compute_mesh_bounds(doc, mesh, rn)
        } else {
            // Nodes without meshes get a default-sized AABB.
            let ext = self.default_extent;
            let p = &rn.world_transform.position;
            Ok(AABB::new(
                [p[0] - ext, p[1] - ext, p[2] - ext],
                [p[0] + ext, p[1] + ext, p[2] + ext],
            ))
        }
    }

    fn compute_mesh_bounds(
        &self,
        doc: &GltfDocument,
        mesh: &GltfMesh,
        rn: &ResolvedNode,
    ) -> GltfResult<AABB> {
        let mut combined = AABB::empty();

        for prim in &mesh.primitives {
            if let Some(&pos_accessor_idx) = prim.attributes.get("POSITION") {
                let aabb = self.accessor_bounds(doc, pos_accessor_idx)?;
                combined = combined.merge(&aabb);
            }
        }

        if combined.is_empty() {
            let ext = self.default_extent;
            combined = AABB::new([-ext, -ext, -ext], [ext, ext, ext]);
        }

        // Apply world transform (position + scale; rotation of AABB is approximated).
        let s = &rn.world_transform.scale;
        let p = &rn.world_transform.position;
        Ok(AABB::new(
            [
                combined.min[0] * s[0] + p[0],
                combined.min[1] * s[1] + p[1],
                combined.min[2] * s[2] + p[2],
            ],
            [
                combined.max[0] * s[0] + p[0],
                combined.max[1] * s[1] + p[1],
                combined.max[2] * s[2] + p[2],
            ],
        ))
    }

    fn accessor_bounds(&self, doc: &GltfDocument, idx: usize) -> GltfResult<AABB> {
        if idx >= doc.accessors.len() {
            return Err(GltfError::AccessorIndexOutOfRange(
                idx,
                doc.accessors.len(),
            ));
        }

        let accessor = &doc.accessors[idx];
        if accessor.accessor_type != "VEC3" {
            return Err(GltfError::ValidationError(format!(
                "POSITION accessor must be VEC3, got {}",
                accessor.accessor_type
            )));
        }

        match (&accessor.min, &accessor.max) {
            (Some(min_vals), Some(max_vals)) if min_vals.len() >= 3 && max_vals.len() >= 3 => {
                Ok(AABB::new(
                    [min_vals[0], min_vals[1], min_vals[2]],
                    [max_vals[0], max_vals[1], max_vals[2]],
                ))
            }
            _ => {
                log::warn!(
                    "accessor {} lacks min/max; using default extent",
                    idx
                );
                let ext = self.default_extent;
                Ok(AABB::new([-ext, -ext, -ext], [ext, ext, ext]))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_gltf() -> &'static str {
        r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "name": "Root" }],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "scene": 0
        }"#
    }

    fn scene_with_mesh() -> &'static str {
        r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{
                "name": "Cube",
                "mesh": 0,
                "translation": [1.0, 2.0, 3.0]
            }],
            "meshes": [{
                "name": "CubeMesh",
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            }],
            "bufferViews": [{
                "buffer": 0,
                "byteLength": 96
            }],
            "buffers": [{
                "byteLength": 96
            }],
            "scene": 0
        }"#
    }

    fn scene_with_hierarchy() -> &'static str {
        r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [
                {
                    "name": "Parent",
                    "children": [1],
                    "translation": [10.0, 0.0, 0.0]
                },
                {
                    "name": "Child",
                    "mesh": 0,
                    "translation": [0.0, 5.0, 0.0]
                }
            ],
            "meshes": [{
                "name": "ChildMesh",
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-0.5, -0.5, -0.5],
                "max": [0.5, 0.5, 0.5]
            }],
            "bufferViews": [{ "buffer": 0, "byteLength": 48 }],
            "buffers": [{ "byteLength": 48 }],
            "scene": 0
        }"#
    }

    fn scene_with_camera() -> &'static str {
        r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0, 1] }],
            "nodes": [
                {
                    "name": "MeshNode",
                    "mesh": 0
                },
                {
                    "name": "CameraNode",
                    "camera": 0,
                    "translation": [0.0, 1.7, 5.0]
                }
            ],
            "cameras": [{ "name": "MainCamera", "type": "perspective" }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            }],
            "bufferViews": [{ "buffer": 0, "byteLength": 48 }],
            "buffers": [{ "byteLength": 48 }],
            "scene": 0
        }"#
    }

    fn scene_with_matrix() -> &'static str {
        // Identity matrix with translation (5, 6, 7).
        r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{
                "name": "MatrixNode",
                "matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    5.0, 6.0, 7.0, 1.0
                ]
            }],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "scene": 0
        }"#
    }

    fn scene_with_scale() -> &'static str {
        r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{
                "name": "ScaledCube",
                "mesh": 0,
                "scale": [2.0, 3.0, 4.0]
            }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            }],
            "bufferViews": [{ "buffer": 0, "byteLength": 96 }],
            "buffers": [{ "byteLength": 96 }],
            "scene": 0
        }"#
    }

    #[test]
    fn test_parse_minimal_gltf() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(minimal_gltf()).unwrap();
        assert_eq!(config.entities.len(), 1);
        assert!(config.entities.contains_key(&EntityId("gltf_Root".to_string())));
    }

    #[test]
    fn test_mesh_bounds_from_accessor() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(scene_with_mesh()).unwrap();
        let entity = config
            .entities
            .get(&EntityId("gltf_Cube".to_string()))
            .unwrap();
        // Mesh AABB [-1,1]^3 translated by (1,2,3).
        assert!((entity.bounds.min[0] - 0.0).abs() < 1e-6);
        assert!((entity.bounds.min[1] - 1.0).abs() < 1e-6);
        assert!((entity.bounds.min[2] - 2.0).abs() < 1e-6);
        assert!((entity.bounds.max[0] - 2.0).abs() < 1e-6);
        assert!((entity.bounds.max[1] - 3.0).abs() < 1e-6);
        assert!((entity.bounds.max[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_hierarchy_transform() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(scene_with_hierarchy()).unwrap();
        // Child is at parent(10,0,0) + child(0,5,0) = (10,5,0).
        let child = config
            .entities
            .get(&EntityId("gltf_Child".to_string()))
            .unwrap();
        assert!((child.transform.position[0] - 10.0).abs() < 1e-6);
        assert!((child.transform.position[1] - 5.0).abs() < 1e-6);
        assert!((child.transform.position[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_camera_node_gets_entity() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(scene_with_camera()).unwrap();
        // Camera node should still become an entity.
        assert!(config
            .entities
            .contains_key(&EntityId("gltf_CameraNode".to_string())));
        let cam = config
            .entities
            .get(&EntityId("gltf_CameraNode".to_string()))
            .unwrap();
        assert!((cam.transform.position[1] - 1.7).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_decomposition() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(scene_with_matrix()).unwrap();
        let entity = config
            .entities
            .get(&EntityId("gltf_MatrixNode".to_string()))
            .unwrap();
        assert!((entity.transform.position[0] - 5.0).abs() < 1e-6);
        assert!((entity.transform.position[1] - 6.0).abs() < 1e-6);
        assert!((entity.transform.position[2] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_applied_to_bounds() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(scene_with_scale()).unwrap();
        let entity = config
            .entities
            .get(&EntityId("gltf_ScaledCube".to_string()))
            .unwrap();
        // Mesh AABB [-1,1]^3 scaled by (2,3,4) → [-2,2]×[-3,3]×[-4,4].
        assert!((entity.bounds.min[0] - (-2.0)).abs() < 1e-6);
        assert!((entity.bounds.max[0] - 2.0).abs() < 1e-6);
        assert!((entity.bounds.min[1] - (-3.0)).abs() < 1e-6);
        assert!((entity.bounds.max[1] - 3.0).abs() < 1e-6);
        assert!((entity.bounds.min[2] - (-4.0)).abs() < 1e-6);
        assert!((entity.bounds.max[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_region_created_for_each_entity() {
        let importer = GltfImporter::new();
        let config = importer.import_from_str(scene_with_mesh()).unwrap();
        assert!(config
            .regions
            .contains_key(&RegionId("gltf_Cube_region".to_string())));
    }

    #[test]
    fn test_invalid_version_rejected() {
        let json = r#"{
            "asset": { "version": "1.0" },
            "scenes": [{ "nodes": [] }]
        }"#;
        let importer = GltfImporter::new();
        let result = importer.import_from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("version"));
    }

    #[test]
    fn test_no_scenes_rejected() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scenes": []
        }"#;
        let importer = GltfImporter::new();
        let result = importer.import_from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_prefix() {
        let importer = GltfImporter {
            entity_prefix: "scene".to_string(),
            ..Default::default()
        };
        let config = importer.import_from_str(minimal_gltf()).unwrap();
        assert!(config
            .entities
            .contains_key(&EntityId("scene_Root".to_string())));
    }

    #[test]
    fn test_accessor_without_min_max_uses_default() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "name": "Obj", "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            }],
            "accessors": [{
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3"
            }],
            "bufferViews": [{ "buffer": 0, "byteLength": 48 }],
            "buffers": [{ "byteLength": 48 }],
            "scene": 0
        }"#;
        let importer = GltfImporter::new();
        let config = importer.import_from_str(json).unwrap();
        assert_eq!(config.entities.len(), 1);
    }

    #[test]
    fn test_multi_primitive_mesh_merges_bounds() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "name": "Multi", "mesh": 0 }],
            "meshes": [{
                "primitives": [
                    { "attributes": { "POSITION": 0 } },
                    { "attributes": { "POSITION": 1 } }
                ]
            }],
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126, "count": 4,
                    "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]
                },
                {
                    "bufferView": 0, "componentType": 5126, "count": 4,
                    "type": "VEC3", "min": [-2.0, -2.0, -2.0], "max": [0.0, 0.0, 0.0]
                }
            ],
            "bufferViews": [{ "buffer": 0, "byteLength": 96 }],
            "buffers": [{ "byteLength": 96 }],
            "scene": 0
        }"#;
        let importer = GltfImporter::new();
        let config = importer.import_from_str(json).unwrap();
        let entity = config
            .entities
            .get(&EntityId("gltf_Multi".to_string()))
            .unwrap();
        assert!((entity.bounds.min[0] - (-2.0)).abs() < 1e-6);
        assert!((entity.bounds.max[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_unnamed_node() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{}],
            "scene": 0
        }"#;
        let importer = GltfImporter::new();
        let config = importer.import_from_str(json).unwrap();
        assert!(config
            .entities
            .contains_key(&EntityId("gltf_node_0".to_string())));
    }

    #[test]
    fn test_roundtrip_gltf_document_serde() {
        let doc: GltfDocument = serde_json::from_str(scene_with_mesh()).unwrap();
        let json = serde_json::to_string(&doc).unwrap();
        let doc2: GltfDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(doc.nodes.len(), doc2.nodes.len());
        assert_eq!(doc.meshes.len(), doc2.meshes.len());
    }
}
