// LITMUS∞ Algebraic Engine
// Symmetry group computation, orbit enumeration, and algebraic compression
// for GPU memory model verification.

pub mod types;
pub mod group;
pub mod symmetry;
pub mod orbit;
pub mod compress;
pub mod wreath;

pub use types::*;
pub use group::PermutationGroup;
pub use symmetry::FullSymmetryGroup;
pub use orbit::OrbitEnumerator;
pub use compress::StateSpaceCompressor;
pub use wreath::WreathProduct;

pub mod spectral;
pub mod galois;
pub mod representation;
pub mod abstract_algebra;
pub mod polynomial;
pub mod homological;
pub mod permutation;
pub mod lattice;
pub mod quotient;
pub mod matrix_repr;
pub mod action;
pub mod automorphism;
pub mod invariant_theory;
