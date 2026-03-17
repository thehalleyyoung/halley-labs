//! Trace compression algorithms.
use serde::{Serialize, Deserialize};

/// Compression statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats { pub original_size: usize, pub compressed_size: usize, pub ratio: f64 }

/// Delta encoding compression.
#[derive(Debug, Clone, Default)] pub struct DeltaCompression;

/// Quantization compression.
#[derive(Debug, Clone)]
pub struct QuantizationCompression { pub bits: u8 }
impl Default for QuantizationCompression { fn default() -> Self { Self { bits: 16 } } }

/// Keyframe compression with interpolation.
#[derive(Debug, Clone)]
pub struct KeyframeCompression { pub interval: usize }
impl Default for KeyframeCompression { fn default() -> Self { Self { interval: 10 } } }

/// Run-length encoding.
#[derive(Debug, Clone, Default)] pub struct RunLengthEncoding;

/// Lossy compression with configurable error bound.
#[derive(Debug, Clone)]
pub struct LossyCompression { pub max_error: f64 }
impl Default for LossyCompression { fn default() -> Self { Self { max_error: 1e-6 } } }
