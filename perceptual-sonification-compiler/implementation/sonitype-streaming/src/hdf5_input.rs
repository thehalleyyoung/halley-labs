//! HDF5 scientific data input for SoniType.
//!
//! Provides a shim layer for reading HDF5 (Hierarchical Data Format) files,
//! commonly used in scientific computing (climate data, astronomy, bioinformatics).
//!
//! # Design Note
//!
//! HDF5 C library bindings are platform-specific and require `libhdf5` to be
//! installed. This module provides a pure-Rust fallback that reads HDF5-like
//! metadata from JSON sidecar files, avoiding the native dependency.
//!
//! For full HDF5 support, compile with `--features hdf5-native` and ensure
//! `libhdf5-dev` is available on the system.
//!
//! # Sidecar Format
//!
//! SoniType expects an HDF5 sidecar JSON file (`.h5meta.json`) containing:
//! ```json
//! {
//!   "datasets": {
//!     "/temperature": { "shape": [365, 180, 360], "dtype": "float64" },
//!     "/pressure": { "shape": [365, 180, 360], "dtype": "float64" }
//!   },
//!   "attributes": {
//!     "source": "ERA5 reanalysis",
//!     "units": { "/temperature": "K", "/pressure": "Pa" }
//!   }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Metadata for an HDF5 dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdf5DatasetMeta {
    /// Shape of the dataset (e.g., [365, 180, 360] for a 3D grid).
    pub shape: Vec<usize>,
    /// Data type string (e.g., "float32", "float64", "int32").
    pub dtype: String,
    /// Optional unit string (e.g., "K", "Pa", "m/s").
    pub unit: Option<String>,
    /// Optional long name / description.
    pub long_name: Option<String>,
}

/// Top-level HDF5 file metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdf5FileMeta {
    /// Map from dataset path (e.g., "/temperature") to metadata.
    pub datasets: HashMap<String, Hdf5DatasetMeta>,
    /// Top-level file attributes.
    pub attributes: HashMap<String, serde_json::Value>,
}

/// An HDF5 data source for sonification.
#[derive(Debug, Clone)]
pub struct Hdf5DataSource {
    pub file_path: String,
    pub metadata: Hdf5FileMeta,
    /// Loaded numeric data, keyed by dataset path.
    pub data: HashMap<String, Vec<f64>>,
}

impl Hdf5DataSource {
    /// Open an HDF5 file via its JSON sidecar metadata.
    ///
    /// Looks for `<path>.h5meta.json` alongside the HDF5 file. If a binary
    /// data file `<path>.h5data.bin` exists, loads flattened f64 arrays from it.
    pub fn from_sidecar(path: impl AsRef<Path>) -> Result<Self, Hdf5InputError> {
        let path = path.as_ref();
        let meta_path = path.with_extension("h5meta.json");

        let meta_str = std::fs::read_to_string(&meta_path)
            .map_err(|e| Hdf5InputError::MetadataReadError(format!(
                "Cannot read {}: {e}", meta_path.display()
            )))?;

        let metadata: Hdf5FileMeta = serde_json::from_str(&meta_str)
            .map_err(|e| Hdf5InputError::MetadataParseError(e.to_string()))?;

        // Try to load binary data if available.
        let bin_path = path.with_extension("h5data.bin");
        let mut data = HashMap::new();

        if bin_path.exists() {
            let bytes = std::fs::read(&bin_path)
                .map_err(|e| Hdf5InputError::DataReadError(e.to_string()))?;

            // Simple format: for each dataset in alphabetical order,
            // N f64 values where N = product of shape dimensions.
            let mut offset = 0;
            let mut sorted_keys: Vec<&String> = metadata.datasets.keys().collect();
            sorted_keys.sort();

            for key in sorted_keys {
                if let Some(meta) = metadata.datasets.get(key.as_str()) {
                    let n_elements: usize = meta.shape.iter().product();
                    let n_bytes = n_elements * 8; // f64 = 8 bytes

                    if offset + n_bytes <= bytes.len() {
                        let values: Vec<f64> = bytes[offset..offset + n_bytes]
                            .chunks_exact(8)
                            .map(|chunk| {
                                let arr: [u8; 8] = chunk.try_into().unwrap();
                                f64::from_le_bytes(arr)
                            })
                            .collect();
                        data.insert(key.to_string(), values);
                        offset += n_bytes;
                    }
                }
            }
        }

        Ok(Hdf5DataSource {
            file_path: path.to_string_lossy().to_string(),
            metadata,
            data,
        })
    }

    /// List available dataset paths.
    pub fn dataset_names(&self) -> Vec<&str> {
        self.metadata.datasets.keys().map(|k| k.as_str()).collect()
    }

    /// Get shape of a dataset.
    pub fn dataset_shape(&self, name: &str) -> Option<&[usize]> {
        self.metadata.datasets.get(name).map(|d| d.shape.as_slice())
    }

    /// Get loaded data for a dataset (returns None if binary data was not loaded).
    pub fn dataset_values(&self, name: &str) -> Option<&[f64]> {
        self.data.get(name).map(|v| v.as_slice())
    }

    /// Extract a 1D slice from a dataset for sonification.
    /// For multi-dimensional data, selects along the first axis (time-like).
    pub fn extract_time_series(
        &self,
        dataset: &str,
        spatial_index: &[usize],
    ) -> Result<Vec<f64>, Hdf5InputError> {
        let meta = self.metadata.datasets.get(dataset)
            .ok_or_else(|| Hdf5InputError::DatasetNotFound(dataset.to_string()))?;

        let values = self.data.get(dataset)
            .ok_or_else(|| Hdf5InputError::DataNotLoaded(dataset.to_string()))?;

        if meta.shape.is_empty() {
            return Err(Hdf5InputError::InvalidShape(dataset.to_string()));
        }

        let time_steps = meta.shape[0];
        let spatial_dims = &meta.shape[1..];

        if spatial_index.len() != spatial_dims.len() {
            return Err(Hdf5InputError::InvalidShape(format!(
                "Expected {} spatial indices, got {}",
                spatial_dims.len(),
                spatial_index.len()
            )));
        }

        // Compute flat offset for spatial position.
        let mut spatial_offset = 0;
        let mut stride = 1;
        for (i, &dim) in spatial_dims.iter().enumerate().rev() {
            spatial_offset += spatial_index[i] * stride;
            stride *= dim;
        }
        let spatial_stride = stride;

        let mut series = Vec::with_capacity(time_steps);
        for t in 0..time_steps {
            let idx = t * spatial_stride + spatial_offset;
            if idx < values.len() {
                series.push(values[idx]);
            }
        }

        Ok(series)
    }
}

/// Errors specific to HDF5 data input.
#[derive(Debug, Clone)]
pub enum Hdf5InputError {
    MetadataReadError(String),
    MetadataParseError(String),
    DataReadError(String),
    DatasetNotFound(String),
    DataNotLoaded(String),
    InvalidShape(String),
}

impl std::fmt::Display for Hdf5InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MetadataReadError(msg) => write!(f, "HDF5 metadata read error: {msg}"),
            Self::MetadataParseError(msg) => write!(f, "HDF5 metadata parse error: {msg}"),
            Self::DataReadError(msg) => write!(f, "HDF5 data read error: {msg}"),
            Self::DatasetNotFound(name) => write!(f, "dataset '{name}' not found"),
            Self::DataNotLoaded(name) => write!(f, "data for '{name}' not loaded"),
            Self::InvalidShape(msg) => write!(f, "invalid shape: {msg}"),
        }
    }
}

impl std::error::Error for Hdf5InputError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_meta_deserialize() {
        let json = r#"{
            "datasets": {
                "/temperature": {"shape": [365, 10], "dtype": "float64", "unit": "K", "long_name": "Surface temperature"}
            },
            "attributes": {"source": "test"}
        }"#;
        let meta: Hdf5FileMeta = serde_json::from_str(json).unwrap();
        assert_eq!(meta.datasets.len(), 1);
        assert_eq!(meta.datasets["/temperature"].shape, vec![365, 10]);
    }
}
