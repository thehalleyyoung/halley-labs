//! Wave equation benchmarks.

/// 1D wave equation.
#[derive(Debug, Clone)]
pub struct WaveEquation1D { pub c: f64, pub nx: usize }
impl Default for WaveEquation1D { fn default() -> Self { Self { c: 1.0, nx: 100 } } }

/// Standing wave solution.
#[derive(Debug, Clone)]
pub struct StandingWave { pub amplitude: f64, pub wavenumber: f64 }
impl Default for StandingWave { fn default() -> Self { Self { amplitude: 1.0, wavenumber: std::f64::consts::PI } } }

/// Traveling wave solution.
#[derive(Debug, Clone)]
pub struct TravelingWave { pub amplitude: f64, pub velocity: f64 }
impl Default for TravelingWave { fn default() -> Self { Self { amplitude: 1.0, velocity: 1.0 } } }
