//! # choreo-spatial
//!
//! R-tree spatial indexing, GJK/EPA collision detection, bounding volume
//! hierarchies, spatial predicate evaluation, containment reasoning, and
//! geometric separability analysis for the Choreo XR interaction compiler.
//!
//! ## Modules
//!
//! - [`rtree`] – Generic R-tree spatial index with range, KNN, and ray queries.
//! - [`gjk`] – Gilbert-Johnson-Keerthi distance / intersection algorithm.
//! - [`epa`] – Expanding Polytope Algorithm for penetration depth.
//! - [`predicates`] – Spatial predicate evaluator for scene configurations.
//! - [`containment`] – Containment DAG and spatial subtyping.
//! - [`separability`] – Interference graphs, tree decomposition, separators.
//! - [`bvh`] – Bounding Volume Hierarchy with SAH construction.
//! - [`transforms`] – Spatial transform composition and interpolation.

pub mod rtree;
pub mod gjk;
pub mod epa;
pub mod predicates;
pub mod containment;
pub mod separability;
pub mod bvh;
pub mod transforms;

pub use rtree::{RTree, RTreeConfig, RTreeEntry, RTreeNode, VersionedRTree};
pub use gjk::{gjk_intersection, gjk_distance, gjk_closest_points, GjkResult, CollisionShape};
pub use epa::{epa_penetration, EpaResult};
pub use predicates::SpatialPredicateEvaluatorImpl;
pub use containment::{ContainmentDAG, SpatialSubtypeChecker};
pub use separability::{SpatialInterferenceGraph, TreeDecomposition};
pub use bvh::BVH;
pub use transforms::{
    compose_transforms, invert_transform, transform_point, transform_aabb,
    interpolate_transforms, look_at,
};
