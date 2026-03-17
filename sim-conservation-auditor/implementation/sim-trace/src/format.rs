//! Trace file format definitions.
use serde::{Serialize, Deserialize};

/// Trace file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFileHeader { pub magic: [u8; 4], pub version: FormatVersion, pub num_particles: u32, pub num_frames: u64 }

/// Index for fast seeking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceFileIndex { pub offsets: Vec<u64> }

/// Trace format identifier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceFormat { Json, Binary, Csv, Hdf5 }

/// Format version.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FormatVersion { pub major: u16, pub minor: u16 }
