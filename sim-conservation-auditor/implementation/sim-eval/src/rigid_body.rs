//! Rigid body benchmarks.

/// Free rigid body (Euler equations).
#[derive(Debug, Clone)]
pub struct FreeRigidBody { pub inertia: [f64; 3] }
impl Default for FreeRigidBody { fn default() -> Self { Self { inertia: [1.0, 2.0, 3.0] } } }

/// Symmetric top.
#[derive(Debug, Clone)]
pub struct SymmetricTop { pub i_parallel: f64, pub i_perp: f64 }
impl Default for SymmetricTop { fn default() -> Self { Self { i_parallel: 1.0, i_perp: 2.0 } } }

/// Asymmetric top (chaotic).
#[derive(Debug, Clone)]
pub struct AsymmetricTop { pub i1: f64, pub i2: f64, pub i3: f64 }
impl Default for AsymmetricTop { fn default() -> Self { Self { i1: 1.0, i2: 2.0, i3: 3.0 } } }
