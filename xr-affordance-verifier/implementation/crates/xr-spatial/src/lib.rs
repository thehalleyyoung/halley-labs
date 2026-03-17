//! XR Spatial – Interval arithmetic, affine arithmetic, zone abstraction,
//! and Tier 1 spatial verification for the XR Affordance Verifier.
//!
//! This crate implements the foundational spatial-reasoning layer
//! (Tier 1) of the verification pipeline:
//!
//! * **Interval arithmetic** – sound floating-point enclosures with
//!   outward rounding for every primitive operation and transcendental.
//! * **Affine arithmetic** – correlation-tracking enclosures that
//!   reduce the dependency problem of plain interval arithmetic.
//! * **Zone abstraction** – partitioning of the body-parameter space
//!   into zones classified as accessible / inaccessible / uncertain.
//! * **Spatial regions** – hyper-rectangle regions in parameter, joint,
//!   and pose spaces with containment / intersection / subdivision.
//! * **Intersection tests** – conservative geometric overlap tests
//!   between reach envelopes and activation volumes.
//! * **Subdivision strategies** – adaptive, uniform, and octree-based
//!   refinement of the parameter space.
//! * **Bound computation** – wrapping-factor analysis and affine-
//!   arithmetic reach-envelope bounding (Theorem B1).
//! * **Tier 1 verifier** – the fast accessibility linter that classifies
//!   every interactable element as Green / Yellow / Red.
//! * **Lipschitz estimation** – analytical and empirical Lipschitz
//!   constants for the accessibility frontier.

pub mod interval;
pub mod affine;
pub mod zone;
pub mod region;
pub mod intersection;
pub mod subdivision;
pub mod bounds;
pub mod tier1;
pub mod lipschitz;

// ── Re-exports ──────────────────────────────────────────────────────────
pub use interval::{Interval, IntervalVector, IntervalMatrix};
pub use affine::{AffineForm, AffineVector3};
pub use zone::{Zone, ZonePartition, ZoneTree, ZoneClassification};
pub use region::{ParameterRegion, PoseRegion, JointRegion, RegionClassification};
pub use intersection::IntersectionTest;
pub use subdivision::{SubdivisionStrategy, SubdivisionTree};
pub use bounds::BoundComputer;
pub use tier1::{Tier1Verifier, Tier1Result, ElementClassification};
pub use lipschitz::LipschitzEstimator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test_re_exports() {
        let iv = Interval::new(0.0, 1.0);
        assert!(iv.contains(0.5));

        let af = AffineForm::constant(3.14);
        assert!((af.central_value() - 3.14).abs() < 1e-12);
    }
}
