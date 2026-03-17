#!/usr/bin/env python3
"""Generate the complete xr-affordance-verifier Rust workspace."""
import os, textwrap

BASE = "/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/xr-affordance-verifier/implementation"

def write(path, content):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(textwrap.dedent(content).lstrip('\n'))

# ============================================================
# ROOT Cargo.toml
# ============================================================
write("Cargo.toml", """
[workspace]
resolver = "2"
members = [
    "crates/xr-types",
    "crates/xr-scene",
    "crates/xr-affordance",
    "crates/xr-spatial",
    "crates/xr-smt",
    "crates/xr-lint",
    "crates/xr-certificate",
    "crates/xr-cli",
]

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
uuid = { version = "1", features = ["v4", "serde"] }
nalgebra = { version = "0.32", features = ["serde-serialize"] }
rand = "0.8"
rand_distr = "0.4"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
ordered-float = { version = "4", features = ["serde"] }
petgraph = "0.6"
indexmap = { version = "2", features = ["serde"] }
rayon = "1.8"
""")

# ============================================================
# xr-types Cargo.toml
# ============================================================
write("crates/xr-types/Cargo.toml", """
[package]
name = "xr-types"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
uuid = { workspace = true }
nalgebra = { workspace = true }
ordered-float = { workspace = true }
indexmap = { workspace = true }
""")

# ============================================================
# xr-types/src/lib.rs
# ============================================================
write("crates/xr-types/src/lib.rs", '''
//! Core types for the XR Affordance Verifier system.
//!
//! This crate defines the fundamental data structures used throughout
//! the verification pipeline: scene models, kinematic chains, device
//! configurations, geometric primitives, certificates, and error types.

pub mod scene;
pub mod kinematic;
pub mod device;
pub mod geometry;
pub mod error;
pub mod certificate;
pub mod config;
pub mod traits;
pub mod anthropometric;
pub mod interaction;
pub mod accessibility;
pub mod dsl;
pub mod report;

pub use error::{VerifierError, VerifierResult};
pub use scene::{SceneModel, InteractableElement, InteractionType};
pub use kinematic::{KinematicChain, JointType, JointLimits, BodyParameters};
pub use device::{DeviceConfig, DeviceType, TrackingVolume};
pub use geometry::{BoundingBox, Sphere, Capsule, ConvexHull, Volume};
pub use certificate::{CoverageCertificate, CertificateGrade};
pub use config::VerifierConfig;
pub use anthropometric::AnthropometricDatabase;

/// Semantic version of the verifier protocol.
pub const PROTOCOL_VERSION: &str = "0.1.0";

/// Maximum supported interaction depth for multi-step sequences.
pub const MAX_INTERACTION_DEPTH: usize = 3;

/// Maximum number of joints in a kinematic chain.
pub const MAX_JOINTS: usize = 7;

/// Default number of ANSUR-II body parameters.
pub const NUM_BODY_PARAMS: usize = 5;

/// Default population coverage target (5th to 95th percentile).
pub const DEFAULT_COVERAGE_LOW: f64 = 0.05;
pub const DEFAULT_COVERAGE_HIGH: f64 = 0.95;

/// Type alias for 3D points.
pub type Point3 = nalgebra::Point3<f64>;

/// Type alias for 3D vectors.
pub type Vector3 = nalgebra::Vector3<f64>;

/// Type alias for 4x4 homogeneous transformation matrices.
pub type Transform = nalgebra::Matrix4<f64>;

/// Type alias for 3x3 rotation matrices.
pub type Rotation3 = nalgebra::Rotation3<f64>;

/// Type alias for unit quaternions.
pub type UnitQuat = nalgebra::UnitQuaternion<f64>;

/// Unique identifier for scene elements.
pub type ElementId = uuid::Uuid;

/// Unique identifier for devices.
pub type DeviceId = uuid::Uuid;

/// Parameter space dimensionality.
pub type ParamVec = nalgebra::SVector<f64, 5>;

/// Joint angle vector (7-DOF).
pub type JointVec = nalgebra::SVector<f64, 7>;

/// A stratum index in the stratified sampling scheme.
pub type StratumIndex = usize;

/// Convert degrees to radians.
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees.
#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / std::f64::consts::PI
}

/// Clamp a value to a range.
#[inline]
pub fn clamp(val: f64, min: f64, max: f64) -> f64 {
    if val < min { min } else if val > max { max } else { val }
}

/// Linear interpolation.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Smooth step interpolation.
#[inline]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deg_to_rad() {
        let rad = deg_to_rad(180.0);
        assert!((rad - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_rad_to_deg() {
        let deg = rad_to_deg(std::f64::consts::PI);
        assert!((deg - 180.0).abs() < 1e-12);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-1.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-12);
        assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
    }
}
''')

# ============================================================
# xr-types/src/error.rs
# ============================================================
write("crates/xr-types/src/error.rs", '''
//! Error types for the XR Affordance Verifier.

use thiserror::Error;

/// Main error type for the verification system.
#[derive(Error, Debug)]
pub enum VerifierError {
    #[error("Scene parsing error: {0}")]
    SceneParsing(String),

    #[error("Invalid scene structure: {0}")]
    InvalidScene(String),

    #[error("Kinematic model error: {0}")]
    KinematicModel(String),

    #[error("Joint limit violation: joint {joint_index}, angle {angle} outside [{min}, {max}]")]
    JointLimitViolation {
        joint_index: usize,
        angle: f64,
        min: f64,
        max: f64,
    },

    #[error("Forward kinematics computation failed: {0}")]
    ForwardKinematics(String),

    #[error("Inverse kinematics failed to converge after {iterations} iterations")]
    InverseKinematicsConvergence { iterations: usize },

    #[error("Device configuration error: {0}")]
    DeviceConfig(String),

    #[error("Tracking volume exceeded for device {device_name}")]
    TrackingVolumeExceeded { device_name: String },

    #[error("SMT encoding error: {0}")]
    SmtEncoding(String),

    #[error("SMT solver error: {0}")]
    SmtSolver(String),

    #[error("SMT solver timeout after {seconds}s")]
    SmtTimeout { seconds: f64 },

    #[error("Linearization error exceeds bound: {actual} > {bound}")]
    LinearizationError { actual: f64, bound: f64 },

    #[error("Interval arithmetic overflow in {operation}")]
    IntervalOverflow { operation: String },

    #[error("Affine arithmetic error: {0}")]
    AffineArithmetic(String),

    #[error("Certificate generation failed: {0}")]
    CertificateGeneration(String),

    #[error("Insufficient samples: need {needed}, have {have}")]
    InsufficientSamples { needed: usize, have: usize },

    #[error("Coverage target not met: {achieved:.4} < {target:.4}")]
    CoverageNotMet { achieved: f64, target: f64 },

    #[error("Accessibility violation: element {element_name} unreachable for body params {body_params:?}")]
    AccessibilityViolation {
        element_name: String,
        body_params: Vec<f64>,
    },

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Numeric error: {0}")]
    Numeric(String),

    #[error("Topology error: {0}")]
    Topology(String),

    #[error("DSL parse error at line {line}, column {column}: {message}")]
    DslParse {
        line: usize,
        column: usize,
        message: String,
    },

    #[error("DSL type error: {0}")]
    DslType(String),

    #[error("Decomposition error: {0}")]
    Decomposition(String),

    #[error("Zone abstraction error: {0}")]
    ZoneAbstraction(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenience result type for the verifier.
pub type VerifierResult<T> = Result<T, VerifierError>;

/// Severity level for verification diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// Informational message.
    Info,
    /// Warning that may indicate an issue.
    Warning,
    /// Error that must be addressed.
    Error,
    /// Critical error that blocks certification.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A diagnostic message from the verification pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    /// Severity of the diagnostic.
    pub severity: Severity,
    /// Short code identifying the diagnostic type.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Optional element ID this diagnostic pertains to.
    pub element_id: Option<uuid::Uuid>,
    /// Optional source location or context.
    pub context: Option<String>,
    /// Suggested fix, if any.
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// Create a new info diagnostic.
    pub fn info(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Create a new error diagnostic.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Create a new critical diagnostic.
    pub fn critical(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Critical,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Set the element ID.
    pub fn with_element(mut self, id: uuid::Uuid) -> Self {
        self.element_id = Some(id);
        self
    }

    /// Set the context.
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    /// Set the suggestion.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.code, self.message)?;
        if let Some(ref ctx) = self.context {
            write!(f, " ({})", ctx)?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " -> {}", suggestion)?;
        }
        Ok(())
    }
}

/// Collection of diagnostics accumulated during verification.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticCollection {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticCollection {
    /// Create a new empty collection.
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Add a diagnostic.
    pub fn push(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    /// Get all diagnostics.
    pub fn all(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Get diagnostics filtered by severity.
    pub fn by_severity(&self, severity: Severity) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == severity)
            .collect()
    }

    /// Check if there are any errors or critical diagnostics.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity >= Severity::Error)
    }

    /// Count diagnostics of a given severity.
    pub fn count(&self, severity: Severity) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == severity)
            .count()
    }

    /// Total number of diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Merge another collection into this one.
    pub fn merge(&mut self, other: DiagnosticCollection) {
        self.diagnostics.extend(other.diagnostics);
    }

    /// Get the highest severity in the collection.
    pub fn max_severity(&self) -> Option<Severity> {
        self.diagnostics.iter().map(|d| d.severity).max()
    }

    /// Filter diagnostics for a specific element.
    pub fn for_element(&self, id: uuid::Uuid) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.element_id == Some(id))
            .collect()
    }

    /// Remove all diagnostics below a given severity.
    pub fn filter_min_severity(&mut self, min: Severity) {
        self.diagnostics.retain(|d| d.severity >= min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let d = Diagnostic::info("XR001", "Test message")
            .with_context("test context")
            .with_suggestion("fix it");
        assert_eq!(d.severity, Severity::Info);
        assert_eq!(d.code, "XR001");
        assert_eq!(d.message, "Test message");
        assert_eq!(d.context.as_deref(), Some("test context"));
        assert_eq!(d.suggestion.as_deref(), Some("fix it"));
    }

    #[test]
    fn test_diagnostic_collection() {
        let mut coll = DiagnosticCollection::new();
        coll.push(Diagnostic::info("XR001", "info"));
        coll.push(Diagnostic::warning("XR002", "warning"));
        coll.push(Diagnostic::error("XR003", "error"));

        assert_eq!(coll.len(), 3);
        assert!(coll.has_errors());
        assert_eq!(coll.count(Severity::Info), 1);
        assert_eq!(coll.count(Severity::Warning), 1);
        assert_eq!(coll.count(Severity::Error), 1);
        assert_eq!(coll.max_severity(), Some(Severity::Error));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_filter_min_severity() {
        let mut coll = DiagnosticCollection::new();
        coll.push(Diagnostic::info("XR001", "info"));
        coll.push(Diagnostic::warning("XR002", "warning"));
        coll.push(Diagnostic::error("XR003", "error"));
        coll.filter_min_severity(Severity::Warning);
        assert_eq!(coll.len(), 2);
    }
}
''')

print("Generated error.rs and lib.rs for xr-types")

# ============================================================
# xr-types/src/geometry.rs
# ============================================================
write("crates/xr-types/src/geometry.rs", '''
//! Geometric primitives for spatial reasoning in XR scenes.

use serde::{Deserialize, Serialize};
use nalgebra::{Point3, Vector3, Matrix3, Matrix4};

/// Axis-aligned bounding box in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl BoundingBox {
    /// Create a new bounding box from min and max corners.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            min: [min[0].min(max[0]), min[1].min(max[1]), min[2].min(max[2])],
            max: [min[0].max(max[0]), min[1].max(max[1]), min[2].max(max[2])],
        }
    }

    /// Create a bounding box from center and half-extents.
    pub fn from_center_extents(center: [f64; 3], half_extents: [f64; 3]) -> Self {
        Self {
            min: [
                center[0] - half_extents[0],
                center[1] - half_extents[1],
                center[2] - half_extents[2],
            ],
            max: [
                center[0] + half_extents[0],
                center[1] + half_extents[1],
                center[2] + half_extents[2],
            ],
        }
    }

    /// Create a unit bounding box centered at the origin.
    pub fn unit() -> Self {
        Self::from_center_extents([0.0; 3], [0.5; 3])
    }

    /// Create a bounding box from a set of points.
    pub fn from_points(points: &[[f64; 3]]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let mut min = points[0];
        let mut max = points[0];
        for p in &points[1..] {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        Some(Self { min, max })
    }

    /// Get the center of the bounding box.
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get the half-extents of the bounding box.
    pub fn half_extents(&self) -> [f64; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Get the full extents (dimensions) of the bounding box.
    pub fn extents(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Compute the volume of the bounding box.
    pub fn volume(&self) -> f64 {
        let e = self.extents();
        e[0] * e[1] * e[2]
    }

    /// Compute the surface area of the bounding box.
    pub fn surface_area(&self) -> f64 {
        let e = self.extents();
        2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])
    }

    /// Compute the diagonal length of the bounding box.
    pub fn diagonal(&self) -> f64 {
        let e = self.extents();
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }

    /// Check if a point is inside the bounding box.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }

    /// Check if this box contains another box entirely.
    pub fn contains_box(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.min[0]
            && self.max[0] >= other.max[0]
            && self.min[1] <= other.min[1]
            && self.max[1] >= other.max[1]
            && self.min[2] <= other.min[2]
            && self.max[2] >= other.max[2]
    }

    /// Check if this box intersects another box.
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Compute the intersection of two bounding boxes.
    pub fn intersection(&self, other: &BoundingBox) -> Option<BoundingBox> {
        if !self.intersects(other) {
            return None;
        }
        Some(BoundingBox {
            min: [
                self.min[0].max(other.min[0]),
                self.min[1].max(other.min[1]),
                self.min[2].max(other.min[2]),
            ],
            max: [
                self.max[0].min(other.max[0]),
                self.max[1].min(other.max[1]),
                self.max[2].min(other.max[2]),
            ],
        })
    }

    /// Compute the union of two bounding boxes.
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Expand the bounding box by a uniform margin.
    pub fn expand(&self, margin: f64) -> BoundingBox {
        BoundingBox {
            min: [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            max: [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        }
    }

    /// Compute the signed distance from a point to the bounding box.
    /// Negative values indicate the point is inside.
    pub fn signed_distance(&self, p: &[f64; 3]) -> f64 {
        let mut d = [0.0f64; 3];
        for i in 0..3 {
            if p[i] < self.min[i] {
                d[i] = self.min[i] - p[i];
            } else if p[i] > self.max[i] {
                d[i] = p[i] - self.max[i];
            }
        }
        let outside = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if outside > 0.0 {
            return outside;
        }
        let mut max_inside = f64::NEG_INFINITY;
        for i in 0..3 {
            let dist_to_min = p[i] - self.min[i];
            let dist_to_max = self.max[i] - p[i];
            max_inside = max_inside.max(-dist_to_min.min(dist_to_max));
        }
        max_inside
    }

    /// Transform the bounding box by a 4x4 matrix (conservative over-approximation).
    pub fn transform(&self, mat: &Matrix4<f64>) -> BoundingBox {
        let corners = [
            [self.min[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.max[1], self.max[2]],
        ];
        let transformed: Vec<[f64; 3]> = corners
            .iter()
            .map(|c| {
                let p = nalgebra::Vector4::new(c[0], c[1], c[2], 1.0);
                let tp = mat * p;
                [tp[0] / tp[3], tp[1] / tp[3], tp[2] / tp[3]]
            })
            .collect();
        BoundingBox::from_points(&transformed).unwrap()
    }

    /// Subdivide the bounding box along the longest axis.
    pub fn subdivide_longest(&self) -> (BoundingBox, BoundingBox) {
        let e = self.extents();
        let axis = if e[0] >= e[1] && e[0] >= e[2] {
            0
        } else if e[1] >= e[2] {
            1
        } else {
            2
        };
        self.subdivide_axis(axis)
    }

    /// Subdivide the bounding box along a specific axis.
    pub fn subdivide_axis(&self, axis: usize) -> (BoundingBox, BoundingBox) {
        let mid = (self.min[axis] + self.max[axis]) * 0.5;
        let mut left_max = self.max;
        left_max[axis] = mid;
        let mut right_min = self.min;
        right_min[axis] = mid;
        (
            BoundingBox { min: self.min, max: left_max },
            BoundingBox { min: right_min, max: self.max },
        )
    }

    /// Octree subdivision: split into 8 sub-boxes.
    pub fn octree_split(&self) -> [BoundingBox; 8] {
        let c = self.center();
        [
            BoundingBox::new(self.min, c),
            BoundingBox::new([c[0], self.min[1], self.min[2]], [self.max[0], c[1], c[2]]),
            BoundingBox::new([self.min[0], c[1], self.min[2]], [c[0], self.max[1], c[2]]),
            BoundingBox::new([c[0], c[1], self.min[2]], [self.max[0], self.max[1], c[2]]),
            BoundingBox::new([self.min[0], self.min[1], c[2]], [c[0], c[1], self.max[2]]),
            BoundingBox::new([c[0], self.min[1], c[2]], [self.max[0], c[1], self.max[2]]),
            BoundingBox::new([self.min[0], c[1], c[2]], [c[0], self.max[1], self.max[2]]),
            BoundingBox::new(c, self.max),
        ]
    }
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self::unit()
    }
}

/// A sphere in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: [f64; 3],
    pub radius: f64,
}

impl Sphere {
    /// Create a new sphere.
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self {
            center,
            radius: radius.abs(),
        }
    }

    /// Create a unit sphere at the origin.
    pub fn unit() -> Self {
        Self::new([0.0; 3], 1.0)
    }

    /// Compute the volume.
    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius.powi(3)
    }

    /// Compute the surface area.
    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius.powi(2)
    }

    /// Check if a point is inside the sphere.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }

    /// Check if this sphere intersects another.
    pub fn intersects_sphere(&self, other: &Sphere) -> bool {
        let dx = self.center[0] - other.center[0];
        let dy = self.center[1] - other.center[1];
        let dz = self.center[2] - other.center[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r_sum = self.radius + other.radius;
        dist_sq <= r_sum * r_sum
    }

    /// Check if this sphere intersects a bounding box.
    pub fn intersects_box(&self, bbox: &BoundingBox) -> bool {
        let mut dist_sq = 0.0f64;
        for i in 0..3 {
            let c = self.center[i];
            if c < bbox.min[i] {
                let d = bbox.min[i] - c;
                dist_sq += d * d;
            } else if c > bbox.max[i] {
                let d = c - bbox.max[i];
                dist_sq += d * d;
            }
        }
        dist_sq <= self.radius * self.radius
    }

    /// Get the bounding box of this sphere.
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox::from_center_extents(self.center, [self.radius; 3])
    }

    /// Compute the distance from the sphere surface to a point.
    pub fn distance_to_point(&self, p: &[f64; 3]) -> f64 {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt() - self.radius
    }
}

/// A capsule (swept sphere) in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Capsule {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub radius: f64,
}

impl Capsule {
    /// Create a new capsule.
    pub fn new(start: [f64; 3], end: [f64; 3], radius: f64) -> Self {
        Self {
            start,
            end,
            radius: radius.abs(),
        }
    }

    /// Get the length of the capsule's axis.
    pub fn axis_length(&self) -> f64 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let dz = self.end[2] - self.start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get the direction vector of the capsule's axis.
    pub fn axis_direction(&self) -> [f64; 3] {
        let len = self.axis_length();
        if len < 1e-12 {
            return [0.0, 1.0, 0.0];
        }
        [
            (self.end[0] - self.start[0]) / len,
            (self.end[1] - self.start[1]) / len,
            (self.end[2] - self.start[2]) / len,
        ]
    }

    /// Compute the volume.
    pub fn volume(&self) -> f64 {
        let r = self.radius;
        let h = self.axis_length();
        std::f64::consts::PI * r * r * h + (4.0 / 3.0) * std::f64::consts::PI * r * r * r
    }

    /// Check if a point is inside the capsule.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        let closest = self.closest_point_on_axis(p);
        let dx = p[0] - closest[0];
        let dy = p[1] - closest[1];
        let dz = p[2] - closest[2];
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }

    /// Find the closest point on the axis to a given point.
    pub fn closest_point_on_axis(&self, p: &[f64; 3]) -> [f64; 3] {
        let ab = [
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
            self.end[2] - self.start[2],
        ];
        let ap = [
            p[0] - self.start[0],
            p[1] - self.start[1],
            p[2] - self.start[2],
        ];
        let ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        if ab_sq < 1e-24 {
            return self.start;
        }
        let t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_sq;
        let t = t.clamp(0.0, 1.0);
        [
            self.start[0] + t * ab[0],
            self.start[1] + t * ab[1],
            self.start[2] + t * ab[2],
        ]
    }

    /// Get the bounding box of this capsule.
    pub fn bounding_box(&self) -> BoundingBox {
        let min = [
            self.start[0].min(self.end[0]) - self.radius,
            self.start[1].min(self.end[1]) - self.radius,
            self.start[2].min(self.end[2]) - self.radius,
        ];
        let max = [
            self.start[0].max(self.end[0]) + self.radius,
            self.start[1].max(self.end[1]) + self.radius,
            self.start[2].max(self.end[2]) + self.radius,
        ];
        BoundingBox::new(min, max)
    }
}

/// A convex hull defined by a set of vertices and face indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvexHull {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    pub normals: Vec<[f64; 3]>,
}

impl ConvexHull {
    /// Create a new convex hull from vertices and faces.
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<Vec<usize>>) -> Self {
        let normals = faces
            .iter()
            .map(|face| {
                if face.len() < 3 {
                    return [0.0, 1.0, 0.0];
                }
                let v0 = vertices[face[0]];
                let v1 = vertices[face[1]];
                let v2 = vertices[face[2]];
                let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len < 1e-12 {
                    [0.0, 1.0, 0.0]
                } else {
                    [n[0] / len, n[1] / len, n[2] / len]
                }
            })
            .collect();
        Self {
            vertices,
            faces,
            normals,
        }
    }

    /// Create a box convex hull from a bounding box.
    pub fn from_bbox(bbox: &BoundingBox) -> Self {
        let [x0, y0, z0] = bbox.min;
        let [x1, y1, z1] = bbox.max;
        let vertices = vec![
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ];
        let faces = vec![
            vec![0, 3, 2, 1], // -Z
            vec![4, 5, 6, 7], // +Z
            vec![0, 1, 5, 4], // -Y
            vec![2, 3, 7, 6], // +Y
            vec![0, 4, 7, 3], // -X
            vec![1, 2, 6, 5], // +X
        ];
        Self::new(vertices, faces)
    }

    /// Compute the centroid of the convex hull.
    pub fn centroid(&self) -> [f64; 3] {
        if self.vertices.is_empty() {
            return [0.0; 3];
        }
        let n = self.vertices.len() as f64;
        let sum = self.vertices.iter().fold([0.0; 3], |acc, v| {
            [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// Get the bounding box of this convex hull.
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox::from_points(&self.vertices).unwrap_or_default()
    }

    /// Check if a point is inside the convex hull using face normal tests.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        for (face, normal) in self.faces.iter().zip(self.normals.iter()) {
            if face.is_empty() {
                continue;
            }
            let v0 = self.vertices[face[0]];
            let d = (p[0] - v0[0]) * normal[0]
                + (p[1] - v0[1]) * normal[1]
                + (p[2] - v0[2]) * normal[2];
            if d > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Support function for GJK: find the furthest point in a direction.
    pub fn support(&self, direction: &[f64; 3]) -> [f64; 3] {
        let mut best_dot = f64::NEG_INFINITY;
        let mut best_vertex = self.vertices[0];
        for v in &self.vertices {
            let dot = v[0] * direction[0] + v[1] * direction[1] + v[2] * direction[2];
            if dot > best_dot {
                best_dot = dot;
                best_vertex = *v;
            }
        }
        best_vertex
    }
}

/// Generic volume type for interaction activation regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Volume {
    Box(BoundingBox),
    Sphere(Sphere),
    Capsule(Capsule),
    ConvexHull(ConvexHull),
    Cylinder(Cylinder),
    Composite(Vec<Volume>),
}

impl Volume {
    /// Check if a point is inside this volume.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        match self {
            Volume::Box(b) => b.contains_point(p),
            Volume::Sphere(s) => s.contains_point(p),
            Volume::Capsule(c) => c.contains_point(p),
            Volume::ConvexHull(h) => h.contains_point(p),
            Volume::Cylinder(c) => c.contains_point(p),
            Volume::Composite(volumes) => volumes.iter().any(|v| v.contains_point(p)),
        }
    }

    /// Get the bounding box of this volume.
    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Volume::Box(b) => *b,
            Volume::Sphere(s) => s.bounding_box(),
            Volume::Capsule(c) => c.bounding_box(),
            Volume::ConvexHull(h) => h.bounding_box(),
            Volume::Cylinder(c) => c.bounding_box(),
            Volume::Composite(volumes) => {
                if volumes.is_empty() {
                    return BoundingBox::default();
                }
                let mut bb = volumes[0].bounding_box();
                for v in &volumes[1..] {
                    bb = bb.union(&v.bounding_box());
                }
                bb
            }
        }
    }

    /// Approximate volume computation.
    pub fn approximate_volume(&self) -> f64 {
        match self {
            Volume::Box(b) => b.volume(),
            Volume::Sphere(s) => s.volume(),
            Volume::Capsule(c) => c.volume(),
            Volume::ConvexHull(_) => {
                // Approximate with bounding box
                self.bounding_box().volume() * 0.6
            }
            Volume::Cylinder(c) => c.volume(),
            Volume::Composite(volumes) => {
                // Approximate as sum (overcounts overlaps)
                volumes.iter().map(|v| v.approximate_volume()).sum()
            }
        }
    }

    /// Check if this volume intersects a bounding box.
    pub fn intersects_box(&self, bbox: &BoundingBox) -> bool {
        match self {
            Volume::Box(b) => b.intersects(bbox),
            Volume::Sphere(s) => s.intersects_box(bbox),
            Volume::Capsule(c) => c.bounding_box().intersects(bbox),
            Volume::ConvexHull(h) => h.bounding_box().intersects(bbox),
            Volume::Cylinder(c) => c.bounding_box().intersects(bbox),
            Volume::Composite(volumes) => volumes.iter().any(|v| v.intersects_box(bbox)),
        }
    }
}

/// A cylinder defined by center, axis, radius, and half-height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cylinder {
    pub center: [f64; 3],
    pub axis: [f64; 3],
    pub radius: f64,
    pub half_height: f64,
}

impl Cylinder {
    /// Create a new cylinder.
    pub fn new(center: [f64; 3], axis: [f64; 3], radius: f64, half_height: f64) -> Self {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let normalized_axis = if len > 1e-12 {
            [axis[0] / len, axis[1] / len, axis[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };
        Self {
            center,
            axis: normalized_axis,
            radius: radius.abs(),
            half_height: half_height.abs(),
        }
    }

    /// Compute the volume.
    pub fn volume(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius * 2.0 * self.half_height
    }

    /// Check if a point is inside the cylinder.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        let dp = [
            p[0] - self.center[0],
            p[1] - self.center[1],
            p[2] - self.center[2],
        ];
        let proj = dp[0] * self.axis[0] + dp[1] * self.axis[1] + dp[2] * self.axis[2];
        if proj.abs() > self.half_height {
            return false;
        }
        let perp = [
            dp[0] - proj * self.axis[0],
            dp[1] - proj * self.axis[1],
            dp[2] - proj * self.axis[2],
        ];
        let perp_sq = perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2];
        perp_sq <= self.radius * self.radius
    }

    /// Get the bounding box.
    pub fn bounding_box(&self) -> BoundingBox {
        let top = [
            self.center[0] + self.axis[0] * self.half_height,
            self.center[1] + self.axis[1] * self.half_height,
            self.center[2] + self.axis[2] * self.half_height,
        ];
        let bot = [
            self.center[0] - self.axis[0] * self.half_height,
            self.center[1] - self.axis[1] * self.half_height,
            self.center[2] - self.axis[2] * self.half_height,
        ];
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for corner in [top, bot] {
            for i in 0..3 {
                min[i] = min[i].min(corner[i] - self.radius);
                max[i] = max[i].max(corner[i] + self.radius);
            }
        }
        BoundingBox::new(min, max)
    }
}

/// A ray in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

impl Ray {
    /// Create a new ray.
    pub fn new(origin: [f64; 3], direction: [f64; 3]) -> Self {
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        let dir = if len > 1e-12 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };
        Self {
            origin,
            direction: dir,
        }
    }

    /// Get a point along the ray.
    pub fn at(&self, t: f64) -> [f64; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }

    /// Ray-AABB intersection test.
    pub fn intersects_bbox(&self, bbox: &BoundingBox) -> Option<f64> {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;
        for i in 0..3 {
            if self.direction[i].abs() < 1e-12 {
                if self.origin[i] < bbox.min[i] || self.origin[i] > bbox.max[i] {
                    return None;
                }
            } else {
                let inv_d = 1.0 / self.direction[i];
                let mut t1 = (bbox.min[i] - self.origin[i]) * inv_d;
                let mut t2 = (bbox.max[i] - self.origin[i]) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                tmin = tmin.max(t1);
                tmax = tmax.min(t2);
                if tmin > tmax {
                    return None;
                }
            }
        }
        if tmax < 0.0 {
            None
        } else {
            Some(tmin.max(0.0))
        }
    }

    /// Ray-sphere intersection test.
    pub fn intersects_sphere(&self, sphere: &Sphere) -> Option<f64> {
        let oc = [
            self.origin[0] - sphere.center[0],
            self.origin[1] - sphere.center[1],
            self.origin[2] - sphere.center[2],
        ];
        let b = oc[0] * self.direction[0]
            + oc[1] * self.direction[1]
            + oc[2] * self.direction[2];
        let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]
            - sphere.radius * sphere.radius;
        let discriminant = b * b - c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();
        let t1 = -b - sqrt_d;
        let t2 = -b + sqrt_d;
        if t1 >= 0.0 {
            Some(t1)
        } else if t2 >= 0.0 {
            Some(t2)
        } else {
            None
        }
    }
}

/// Compute the Minkowski difference support point.
pub fn minkowski_support(
    hull_a: &ConvexHull,
    hull_b: &ConvexHull,
    direction: &[f64; 3],
) -> [f64; 3] {
    let a = hull_a.support(direction);
    let neg_dir = [-direction[0], -direction[1], -direction[2]];
    let b = hull_b.support(&neg_dir);
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Compute distance between two points.
pub fn point_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute squared distance between two points.
pub fn point_distance_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Dot product of 3D vectors.
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of 3D vectors.
pub fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3D vector.
pub fn normalize3(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Homogeneous transformation from position and rotation matrix.
pub fn transform_from_pos_rot(pos: &[f64; 3], rot: &Matrix3<f64>) -> Matrix4<f64> {
    Matrix4::new(
        rot[(0, 0)], rot[(0, 1)], rot[(0, 2)], pos[0],
        rot[(1, 0)], rot[(1, 1)], rot[(1, 2)], pos[1],
        rot[(2, 0)], rot[(2, 1)], rot[(2, 2)], pos[2],
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Extract position from a 4x4 homogeneous transformation matrix.
pub fn transform_position(t: &Matrix4<f64>) -> [f64; 3] {
    [t[(0, 3)], t[(1, 3)], t[(2, 3)]]
}

/// Extract rotation matrix from a 4x4 homogeneous transformation.
pub fn transform_rotation(t: &Matrix4<f64>) -> Matrix3<f64> {
    Matrix3::new(
        t[(0, 0)], t[(0, 1)], t[(0, 2)],
        t[(1, 0)], t[(1, 1)], t[(1, 2)],
        t[(2, 0)], t[(2, 1)], t[(2, 2)],
    )
}

/// Identity 4x4 transformation.
pub fn identity_transform() -> Matrix4<f64> {
    Matrix4::identity()
}

/// Translation transformation.
pub fn translation_transform(x: f64, y: f64, z: f64) -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    m[(0, 3)] = x;
    m[(1, 3)] = y;
    m[(2, 3)] = z;
    m
}

/// Rotation about X axis.
pub fn rotation_x(angle: f64) -> Matrix4<f64> {
    let c = angle.cos();
    let s = angle.sin();
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Rotation about Y axis.
pub fn rotation_y(angle: f64) -> Matrix4<f64> {
    let c = angle.cos();
    let s = angle.sin();
    Matrix4::new(
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Rotation about Z axis.
pub fn rotation_z(angle: f64) -> Matrix4<f64> {
    let c = angle.cos();
    let s = angle.sin();
    Matrix4::new(
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Rotation about an arbitrary axis (Rodrigues).
pub fn rotation_axis_angle(axis: &[f64; 3], angle: f64) -> Matrix4<f64> {
    let n = normalize3(axis);
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    Matrix4::new(
        t * n[0] * n[0] + c,        t * n[0] * n[1] - s * n[2], t * n[0] * n[2] + s * n[1], 0.0,
        t * n[0] * n[1] + s * n[2], t * n[1] * n[1] + c,        t * n[1] * n[2] - s * n[0], 0.0,
        t * n[0] * n[2] - s * n[1], t * n[1] * n[2] + s * n[0], t * n[2] * n[2] + c,        0.0,
        0.0,                         0.0,                         0.0,                         1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_creation() {
        let bb = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(bb.center(), [0.5, 0.5, 0.5]);
        assert_eq!(bb.extents(), [1.0, 1.0, 1.0]);
        assert!((bb.volume() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bbox_contains() {
        let bb = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(bb.contains_point(&[0.5, 0.5, 0.5]));
        assert!(!bb.contains_point(&[1.5, 0.5, 0.5]));
    }

    #[test]
    fn test_bbox_intersection() {
        let a = BoundingBox::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = BoundingBox::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.min, [1.0, 1.0, 1.0]);
        assert_eq!(inter.max, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_sphere_contains() {
        let s = Sphere::new([0.0, 0.0, 0.0], 1.0);
        assert!(s.contains_point(&[0.0, 0.0, 0.0]));
        assert!(s.contains_point(&[0.5, 0.5, 0.5]));
        assert!(!s.contains_point(&[1.0, 1.0, 0.0]));
    }

    #[test]
    fn test_capsule_contains() {
        let c = Capsule::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5);
        assert!(c.contains_point(&[0.0, 0.5, 0.0]));
        assert!(c.contains_point(&[0.4, 0.5, 0.0]));
        assert!(!c.contains_point(&[1.0, 0.5, 0.0]));
    }

    #[test]
    fn test_ray_bbox_intersection() {
        let ray = Ray::new([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        let bb = BoundingBox::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let t = ray.intersects_bbox(&bb);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_identity() {
        let r = rotation_x(0.0);
        let id = Matrix4::identity();
        for i in 0..4 {
            for j in 0..4 {
                assert!((r[(i, j)] - id[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_subdivision() {
        let bb = BoundingBox::new([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let (left, right) = bb.subdivide_longest();
        assert!((left.max[0] - 1.0).abs() < 1e-12);
        assert!((right.min[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_octree_split() {
        let bb = BoundingBox::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let children = bb.octree_split();
        assert_eq!(children.len(), 8);
        for child in &children {
            assert!((child.volume() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cylinder_contains() {
        let c = Cylinder::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0, 2.0);
        assert!(c.contains_point(&[0.0, 0.0, 0.0]));
        assert!(c.contains_point(&[0.5, 1.0, 0.0]));
        assert!(!c.contains_point(&[0.0, 3.0, 0.0]));
    }

    #[test]
    fn test_convex_hull_bbox() {
        let hull = ConvexHull::from_bbox(&BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        assert_eq!(hull.vertices.len(), 8);
        assert_eq!(hull.faces.len(), 6);
    }
}
''')

print("Generated geometry.rs")
