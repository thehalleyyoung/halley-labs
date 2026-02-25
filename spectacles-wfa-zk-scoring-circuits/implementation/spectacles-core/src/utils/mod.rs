//! Shared utilities for the Spectacles framework.

pub mod hash;
pub mod serialization;
pub mod math;

pub use hash::{SpectaclesHasher, MerkleTree, HashChain, DomainSeparatedHasher, Commitment};
pub use serialization::{ProofSerializer, ProofFormat, CompactProof};
pub use math::{extended_gcd, mod_pow, mod_inv, polynomial_eval, lagrange_interpolate};
