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
pub mod openxr;
pub mod report;
pub mod webxr;

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
