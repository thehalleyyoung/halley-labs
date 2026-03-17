//! Fluid dynamics benchmarks.

/// Linear advection equation.
#[derive(Debug, Clone)]
pub struct AdvectionEquation { pub velocity: f64, pub nx: usize }
impl Default for AdvectionEquation { fn default() -> Self { Self { velocity: 1.0, nx: 100 } } }

/// Burgers equation.
#[derive(Debug, Clone)]
pub struct BurgersEquation { pub viscosity: f64, pub nx: usize }
impl Default for BurgersEquation { fn default() -> Self { Self { viscosity: 0.01, nx: 100 } } }

/// 1D shallow water equations.
#[derive(Debug, Clone)]
pub struct ShallowWater1D { pub g: f64, pub nx: usize }
impl Default for ShallowWater1D { fn default() -> Self { Self { g: 9.81, nx: 200 } } }

/// Sod shock tube problem.
#[derive(Debug, Clone)]
pub struct SodShockTube { pub nx: usize }
impl Default for SodShockTube { fn default() -> Self { Self { nx: 200 } } }
