//! XR Scene parsing, graph construction, and spatial indexing.
//!
//! This crate provides the scene processing pipeline for the XR Affordance Verifier:
//! - **parser**: Reads JSON scene descriptions into an intermediate representation.
//! - **graph**: Builds a petgraph-based scene graph with dependency edges.
//! - **spatial_index**: R-tree spatial index for efficient spatial queries.
//! - **transform**: Computes world-space transforms from local transform hierarchies.
//! - **interaction**: Extracts and classifies interaction patterns.
//! - **unity**: Adapts Unity-style scene data to our internal model.
//! - **usd**: Adapts USD (Universal Scene Description) data to our internal model.
//! - **validation**: Structural and semantic scene validation.
//! - **query**: Complex scene queries (spatial, graph, frustum).
//! - **optimizer**: Scene preprocessing and optimization passes.

pub mod parser;
pub mod graph;
pub mod spatial_index;
pub mod transform;
pub mod interaction;
pub mod unity;
pub mod usd;
pub mod gltf;
pub mod validation;
pub mod query;
pub mod optimizer;

pub use parser::{SceneParser, ParsedScene, SceneBuilder};
pub use graph::{SceneGraph, InteractionCluster};
pub use spatial_index::SpatialIndex;
pub use transform::TransformHierarchy;
pub use interaction::{InteractionPattern, InteractionExtractor, InteractionSequenceBuilder};
pub use unity::UnitySceneAdapter;
pub use usd::UsdSceneAdapter;
pub use gltf::GltfSceneAdapter;
pub use validation::SceneValidator;
pub use query::SceneQueryEngine;
pub use optimizer::SceneOptimizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_reexports() {
        let _parser = SceneParser::new();
        let _builder = SceneBuilder::new("test");
        let _index = SpatialIndex::new();
        let _hierarchy = TransformHierarchy::new();
        let _extractor = InteractionExtractor::new();
        let _adapter = UnitySceneAdapter::new();
        let _validator = SceneValidator::new();
    }
}
