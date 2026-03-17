//! Transform hierarchy for computing world-space transforms.
//!
//! Provides `TransformHierarchy` for managing and computing world-space transforms
//! from a local transform hierarchy, including reparenting, interpolation, and
//! decomposition.

use nalgebra::{Matrix4, Quaternion, UnitQuaternion, Vector3, Translation3};
use xr_types::scene::TransformNode;
use xr_types::geometry::BoundingBox;

/// Transform components decomposed from a 4x4 matrix.
#[derive(Debug, Clone, Copy)]
pub struct TransformComponents {
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub scale: [f64; 3],
}

impl TransformComponents {
    pub fn identity() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0; 3],
        }
    }

    pub fn to_matrix(&self) -> Matrix4<f64> {
        let [w, x, y, z] = self.rotation;
        let rot = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
        let trans = Translation3::new(self.position[0], self.position[1], self.position[2]);
        let scale = Matrix4::new_nonuniform_scaling(&Vector3::new(
            self.scale[0],
            self.scale[1],
            self.scale[2],
        ));
        trans.to_homogeneous() * rot.to_homogeneous() * scale
    }

    /// Linearly interpolate between two transforms.
    pub fn lerp(&self, other: &TransformComponents, t: f64) -> TransformComponents {
        let pos = [
            self.position[0] + (other.position[0] - self.position[0]) * t,
            self.position[1] + (other.position[1] - self.position[1]) * t,
            self.position[2] + (other.position[2] - self.position[2]) * t,
        ];

        let q1 = UnitQuaternion::from_quaternion(Quaternion::new(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        ));
        let q2 = UnitQuaternion::from_quaternion(Quaternion::new(
            other.rotation[0],
            other.rotation[1],
            other.rotation[2],
            other.rotation[3],
        ));
        let qi = q1.slerp(&q2, t);

        let scale = [
            self.scale[0] + (other.scale[0] - self.scale[0]) * t,
            self.scale[1] + (other.scale[1] - self.scale[1]) * t,
            self.scale[2] + (other.scale[2] - self.scale[2]) * t,
        ];

        TransformComponents {
            position: pos,
            rotation: [qi.w, qi.i, qi.j, qi.k],
            scale,
        }
    }
}

/// A hierarchy of transforms for computing world-space positions.
pub struct TransformHierarchy {
    nodes: Vec<TransformNode>,
    world_cache: Vec<Option<Matrix4<f64>>>,
}

impl TransformHierarchy {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            world_cache: Vec::new(),
        }
    }

    /// Build from a list of TransformNodes.
    pub fn from_nodes(nodes: Vec<TransformNode>) -> Self {
        let len = nodes.len();
        Self {
            nodes,
            world_cache: vec![None; len],
        }
    }

    /// Build from a SceneModel's transform nodes.
    pub fn from_scene(scene: &xr_types::scene::SceneModel) -> Self {
        Self::from_nodes(scene.transform_nodes.clone())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, node: TransformNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.world_cache.push(None);
        idx
    }

    /// Add a child node under a parent.
    pub fn add_child(
        &mut self,
        parent: usize,
        name: impl Into<String>,
        position: [f64; 3],
        rotation: [f64; 4],
        scale: [f64; 3],
    ) -> usize {
        let idx = self.nodes.len();
        let node = TransformNode {
            name: name.into(),
            parent: Some(parent),
            local_position: position,
            local_rotation: rotation,
            local_scale: scale,
            children: Vec::new(),
        };
        self.nodes.push(node);
        self.world_cache.push(None);
        if parent < self.nodes.len() {
            self.nodes[parent].children.push(idx);
        }
        self.invalidate_subtree(idx);
        idx
    }

    /// Get a reference to a node.
    pub fn node(&self, index: usize) -> Option<&TransformNode> {
        self.nodes.get(index)
    }

    /// Get the local 4x4 matrix for a node.
    pub fn local_matrix(&self, index: usize) -> Option<Matrix4<f64>> {
        self.nodes.get(index).map(|n| n.local_matrix())
    }

    /// Compute the world transform for a node (caches the result).
    pub fn world_transform(&mut self, index: usize) -> Option<Matrix4<f64>> {
        if index >= self.nodes.len() {
            return None;
        }
        if let Some(cached) = self.world_cache[index] {
            return Some(cached);
        }

        let chain = self.ancestor_chain(index);
        let mut world = Matrix4::identity();
        for &ancestor_idx in &chain {
            world *= self.nodes[ancestor_idx].local_matrix();
        }

        self.world_cache[index] = Some(world);
        Some(world)
    }

    /// Get the inverse of the world transform.
    pub fn inverse_world_transform(&mut self, index: usize) -> Option<Matrix4<f64>> {
        self.world_transform(index)
            .and_then(|m| m.try_inverse())
    }

    /// Get the chain of ancestor indices from root to this node (inclusive).
    pub fn ancestor_chain(&self, index: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut current = Some(index);
        while let Some(idx) = current {
            if idx >= self.nodes.len() {
                break;
            }
            chain.push(idx);
            current = self.nodes[idx].parent;
        }
        chain.reverse();
        chain
    }

    /// Get the world position of a node.
    pub fn world_position(&mut self, index: usize) -> Option<[f64; 3]> {
        self.world_transform(index).map(|m| [m[(0, 3)], m[(1, 3)], m[(2, 3)]])
    }

    /// Set local position and invalidate caches.
    pub fn set_local_position(&mut self, index: usize, position: [f64; 3]) {
        if index < self.nodes.len() {
            self.nodes[index].local_position = position;
            self.invalidate_subtree(index);
        }
    }

    /// Set local rotation and invalidate caches.
    pub fn set_local_rotation(&mut self, index: usize, rotation: [f64; 4]) {
        if index < self.nodes.len() {
            self.nodes[index].local_rotation = rotation;
            self.invalidate_subtree(index);
        }
    }

    /// Set local scale and invalidate caches.
    pub fn set_local_scale(&mut self, index: usize, scale: [f64; 3]) {
        if index < self.nodes.len() {
            self.nodes[index].local_scale = scale;
            self.invalidate_subtree(index);
        }
    }

    /// Reparent a node to a new parent, preserving world transform.
    pub fn reparent(&mut self, node_index: usize, new_parent: Option<usize>) {
        if node_index >= self.nodes.len() {
            return;
        }

        // Compute current world transform
        let world = match self.world_transform(node_index) {
            Some(m) => m,
            None => return,
        };

        // Remove from old parent's children
        if let Some(old_parent) = self.nodes[node_index].parent {
            if old_parent < self.nodes.len() {
                self.nodes[old_parent].children.retain(|&c| c != node_index);
            }
        }

        // Set new parent
        self.nodes[node_index].parent = new_parent;
        if let Some(np) = new_parent {
            if np < self.nodes.len() {
                self.nodes[np].children.push(node_index);
            }
        }

        // Compute new local transform to preserve world transform
        let parent_world = match new_parent {
            Some(pi) => self.world_transform(pi).unwrap_or_else(Matrix4::identity),
            None => Matrix4::identity(),
        };

        if let Some(parent_inv) = parent_world.try_inverse() {
            let new_local = parent_inv * world;
            let decomposed = decompose_matrix(&new_local);
            self.nodes[node_index].local_position = decomposed.position;
            self.nodes[node_index].local_rotation = decomposed.rotation;
            self.nodes[node_index].local_scale = decomposed.scale;
        }

        self.invalidate_subtree(node_index);
    }

    /// Invalidate the world cache for a node and all its descendants.
    fn invalidate_subtree(&mut self, index: usize) {
        if index >= self.nodes.len() {
            return;
        }
        self.world_cache[index] = None;
        let children: Vec<usize> = self.nodes[index].children.clone();
        for child in children {
            self.invalidate_subtree(child);
        }
    }

    /// Invalidate all caches.
    pub fn invalidate_all(&mut self) {
        for c in &mut self.world_cache {
            *c = None;
        }
    }

    /// Get all root node indices.
    pub fn roots(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get all leaf node indices.
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.children.is_empty())
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute depth of each node.
    pub fn depths(&self) -> Vec<usize> {
        let mut result = vec![0; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let chain = self.ancestor_chain(i);
            result[i] = chain.len().saturating_sub(1);
        }
        result
    }

    /// Transform a bounding box from local space to world space.
    pub fn transform_bounds(&mut self, node_index: usize, local_bounds: &BoundingBox) -> Option<BoundingBox> {
        self.world_transform(node_index).map(|m| local_bounds.transform(&m))
    }

    /// Transform a point from local space to world space.
    pub fn local_to_world(&mut self, node_index: usize, local_point: &[f64; 3]) -> Option<[f64; 3]> {
        self.world_transform(node_index).map(|m| {
            let p = nalgebra::Vector4::new(local_point[0], local_point[1], local_point[2], 1.0);
            let wp = m * p;
            [wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]]
        })
    }

    /// Transform a point from world space to local space.
    pub fn world_to_local(&mut self, node_index: usize, world_point: &[f64; 3]) -> Option<[f64; 3]> {
        self.inverse_world_transform(node_index).map(|inv| {
            let p = nalgebra::Vector4::new(world_point[0], world_point[1], world_point[2], 1.0);
            let lp = inv * p;
            [lp[0] / lp[3], lp[1] / lp[3], lp[2] / lp[3]]
        })
    }

    /// Check if the hierarchy is consistent (no cycles, valid parent refs).
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(parent) = node.parent {
                if parent >= self.nodes.len() {
                    errors.push(format!(
                        "Node {} '{}' has invalid parent index {}",
                        i, node.name, parent
                    ));
                } else if parent == i {
                    errors.push(format!("Node {} '{}' is its own parent", i, node.name));
                }
            }
            for &child in &node.children {
                if child >= self.nodes.len() {
                    errors.push(format!(
                        "Node {} '{}' has invalid child index {}",
                        i, node.name, child
                    ));
                }
            }
            // Check for non-positive scale
            for dim in 0..3 {
                if node.local_scale[dim].abs() < 1e-12 {
                    errors.push(format!(
                        "Node {} '{}' has near-zero scale on axis {}",
                        i, node.name, dim
                    ));
                }
            }
        }

        // Cycle detection
        for i in 0..self.nodes.len() {
            let mut visited = std::collections::HashSet::new();
            let mut current = Some(i);
            while let Some(idx) = current {
                if !visited.insert(idx) {
                    errors.push(format!("Cycle detected involving node {}", i));
                    break;
                }
                current = self.nodes.get(idx).and_then(|n| n.parent);
            }
        }

        errors
    }
}

impl Default for TransformHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompose a 4x4 homogeneous matrix into position, rotation (quaternion), and scale.
pub fn decompose_matrix(m: &Matrix4<f64>) -> TransformComponents {
    let position = [m[(0, 3)], m[(1, 3)], m[(2, 3)]];

    // Extract scale from column lengths
    let sx = Vector3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]).norm();
    let sy = Vector3::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]).norm();
    let sz = Vector3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]).norm();

    // Handle degenerate scale
    let sx = if sx < 1e-12 { 1.0 } else { sx };
    let sy = if sy < 1e-12 { 1.0 } else { sy };
    let sz = if sz < 1e-12 { 1.0 } else { sz };

    let scale = [sx, sy, sz];

    // Extract rotation by normalizing columns
    let rot_matrix = nalgebra::Matrix3::new(
        m[(0, 0)] / sx, m[(0, 1)] / sy, m[(0, 2)] / sz,
        m[(1, 0)] / sx, m[(1, 1)] / sy, m[(1, 2)] / sz,
        m[(2, 0)] / sx, m[(2, 1)] / sy, m[(2, 2)] / sz,
    );

    let rotation = nalgebra::Rotation3::from_matrix_unchecked(rot_matrix);
    let quat = UnitQuaternion::from_rotation_matrix(&rotation);

    TransformComponents {
        position,
        rotation: [quat.w, quat.i, quat.j, quat.k],
        scale,
    }
}

/// Compose a 4x4 matrix from components.
pub fn compose_matrix(components: &TransformComponents) -> Matrix4<f64> {
    components.to_matrix()
}

/// Interpolate between two 4x4 matrices.
pub fn interpolate_transforms(a: &Matrix4<f64>, b: &Matrix4<f64>, t: f64) -> Matrix4<f64> {
    let ca = decompose_matrix(a);
    let cb = decompose_matrix(b);
    ca.lerp(&cb, t).to_matrix()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::scene::TransformNode;

    fn make_hierarchy() -> TransformHierarchy {
        let nodes = vec![
            TransformNode {
                name: "Root".to_string(),
                parent: None,
                local_position: [0.0, 0.0, 0.0],
                local_rotation: [1.0, 0.0, 0.0, 0.0],
                local_scale: [1.0, 1.0, 1.0],
                children: vec![1, 2],
            },
            TransformNode {
                name: "Child1".to_string(),
                parent: Some(0),
                local_position: [1.0, 0.0, 0.0],
                local_rotation: [1.0, 0.0, 0.0, 0.0],
                local_scale: [1.0, 1.0, 1.0],
                children: vec![3],
            },
            TransformNode {
                name: "Child2".to_string(),
                parent: Some(0),
                local_position: [0.0, 1.0, 0.0],
                local_rotation: [1.0, 0.0, 0.0, 0.0],
                local_scale: [2.0, 2.0, 2.0],
                children: vec![],
            },
            TransformNode {
                name: "Grandchild".to_string(),
                parent: Some(1),
                local_position: [0.0, 0.0, 1.0],
                local_rotation: [1.0, 0.0, 0.0, 0.0],
                local_scale: [1.0, 1.0, 1.0],
                children: vec![],
            },
        ];
        TransformHierarchy::from_nodes(nodes)
    }

    #[test]
    fn test_world_position() {
        let mut h = make_hierarchy();
        let pos = h.world_position(0).unwrap();
        assert!((pos[0]).abs() < 1e-10);

        let pos1 = h.world_position(1).unwrap();
        assert!((pos1[0] - 1.0).abs() < 1e-10);

        let pos3 = h.world_position(3).unwrap();
        // Root(0,0,0) → Child1(1,0,0) → Grandchild(0,0,1)
        assert!((pos3[0] - 1.0).abs() < 1e-10);
        assert!((pos3[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_propagation() {
        let mut h = make_hierarchy();
        // Child2 has scale [2,2,2], so a local point (1,0,0) should become world (2,1,0)
        let world_pt = h.local_to_world(2, &[1.0, 0.0, 0.0]).unwrap();
        assert!((world_pt[0] - 2.0).abs() < 1e-10);
        assert!((world_pt[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_transform() {
        let mut h = make_hierarchy();
        let local = [1.0, 2.0, 3.0];
        let world = h.local_to_world(1, &local).unwrap();
        let back = h.world_to_local(1, &world).unwrap();
        for i in 0..3 {
            assert!((back[i] - local[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_ancestor_chain() {
        let h = make_hierarchy();
        let chain = h.ancestor_chain(3);
        assert_eq!(chain, vec![0, 1, 3]);
    }

    #[test]
    fn test_roots_and_leaves() {
        let h = make_hierarchy();
        assert_eq!(h.roots(), vec![0]);
        let mut leaves = h.leaves();
        leaves.sort();
        assert_eq!(leaves, vec![2, 3]);
    }

    #[test]
    fn test_depths() {
        let h = make_hierarchy();
        let depths = h.depths();
        assert_eq!(depths[0], 0);
        assert_eq!(depths[1], 1);
        assert_eq!(depths[2], 1);
        assert_eq!(depths[3], 2);
    }

    #[test]
    fn test_set_position_invalidates_cache() {
        let mut h = make_hierarchy();
        let _ = h.world_position(3); // cache
        h.set_local_position(1, [5.0, 0.0, 0.0]);
        let pos3 = h.world_position(3).unwrap();
        assert!((pos3[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_child() {
        let mut h = make_hierarchy();
        let new_idx = h.add_child(0, "NewChild", [0.0, 0.0, 5.0], [1.0, 0.0, 0.0, 0.0], [1.0; 3]);
        assert_eq!(h.node_count(), 5);
        let pos = h.world_position(new_idx).unwrap();
        assert!((pos[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate() {
        let h = make_hierarchy();
        let errors = h.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_catches_issues() {
        let nodes = vec![TransformNode {
            name: "Self-parent".to_string(),
            parent: Some(0),
            local_position: [0.0; 3],
            local_rotation: [1.0, 0.0, 0.0, 0.0],
            local_scale: [0.0, 1.0, 1.0],
            children: vec![],
        }];
        let h = TransformHierarchy::from_nodes(nodes);
        let errors = h.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_decompose_identity() {
        let m = Matrix4::identity();
        let c = decompose_matrix(&m);
        assert!((c.position[0]).abs() < 1e-10);
        assert!((c.scale[0] - 1.0).abs() < 1e-10);
        assert!((c.rotation[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_roundtrip() {
        let components = TransformComponents {
            position: [1.0, 2.0, 3.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            scale: [2.0, 3.0, 4.0],
        };
        let m = components.to_matrix();
        let dc = decompose_matrix(&m);
        for i in 0..3 {
            assert!((dc.position[i] - components.position[i]).abs() < 1e-8);
            assert!((dc.scale[i] - components.scale[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_interpolate_transforms() {
        let a = Matrix4::identity();
        let mut b = Matrix4::identity();
        b[(0, 3)] = 10.0;

        let mid = interpolate_transforms(&a, &b, 0.5);
        assert!((mid[(0, 3)] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_reparent() {
        let mut h = make_hierarchy();
        let old_world = h.world_position(3).unwrap();
        h.reparent(3, Some(2)); // Move grandchild from child1 to child2
        let new_world = h.world_position(3).unwrap();
        for i in 0..3 {
            assert!(
                (old_world[i] - new_world[i]).abs() < 1e-6,
                "World position changed on axis {}: {} vs {}",
                i,
                old_world[i],
                new_world[i]
            );
        }
    }

    #[test]
    fn test_transform_bounds() {
        let mut h = make_hierarchy();
        let local_bb = BoundingBox::from_center_extents([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let world_bb = h.transform_bounds(2, &local_bb).unwrap();
        // Child2 has scale [2,2,2] and offset [0,1,0], so bounds grow
        assert!(world_bb.extents()[0] >= 2.0 - 1e-8);
    }

    #[test]
    fn test_components_lerp() {
        let a = TransformComponents {
            position: [0.0, 0.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let b = TransformComponents {
            position: [10.0, 0.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            scale: [2.0, 2.0, 2.0],
        };
        let mid = a.lerp(&b, 0.5);
        assert!((mid.position[0] - 5.0).abs() < 1e-8);
        assert!((mid.scale[0] - 1.5).abs() < 1e-8);
    }
}
