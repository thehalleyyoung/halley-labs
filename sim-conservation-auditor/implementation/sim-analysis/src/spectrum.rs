//! Spectral analysis (DFT, power spectrum).
use serde::{Serialize, Deserialize};

/// Result of a discrete Fourier transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DftResult { pub frequencies: Vec<f64>, pub amplitudes: Vec<f64>, pub phases: Vec<f64> }

/// Power spectral density estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSpectralDensity { pub frequencies: Vec<f64>, pub power: Vec<f64> }

/// A peak in the power spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralPeak { pub frequency: f64, pub power: f64, pub width: f64 }

/// Window functions for spectral analysis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WindowFunction { Rectangular, Hann, Hamming, Blackman, Kaiser(f64) }

/// Spectral analysis engine.
#[derive(Debug, Clone)]
pub struct SpectralAnalyzer { pub window: WindowFunction }
impl Default for SpectralAnalyzer { fn default() -> Self { Self { window: WindowFunction::Hann } } }
impl SpectralAnalyzer {
    /// Compute the DFT of a real-valued signal.
    pub fn dft(&self, signal: &[f64], dt: f64) -> DftResult {
        let n = signal.len();
        let freqs: Vec<f64> = (0..n/2).map(|k| k as f64 / (n as f64 * dt)).collect();
        let amps = vec![0.0; freqs.len()];
        let phases = vec![0.0; freqs.len()];
        DftResult { frequencies: freqs, amplitudes: amps, phases }
    }
    /// Compute the power spectral density.
    pub fn psd(&self, signal: &[f64], dt: f64) -> PowerSpectralDensity {
        let dft = self.dft(signal, dt);
        let power: Vec<f64> = dft.amplitudes.iter().map(|a| a * a).collect();
        PowerSpectralDensity { frequencies: dft.frequencies, power }
    }
}
