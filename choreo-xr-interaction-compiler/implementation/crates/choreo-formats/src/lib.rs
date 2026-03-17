//! File format support for the Choreo XR interaction compiler.
//!
//! This crate provides importers and exporters for external file formats:
//! - **glTF**: Import 3D scenes from glTF 2.0 JSON files into Choreo scene configurations.
//! - **OpenXR**: Generate OpenXR action manifests from Choreo interaction declarations.

pub mod gltf;
pub mod openxr;

pub use gltf::GltfImporter;
pub use openxr::OpenXrManifestGenerator;
