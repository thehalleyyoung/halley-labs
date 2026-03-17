//! Spectral clustering implementation.

use spectral_types::partition::Partition;
use crate::error::Result;

/// Perform spectral clustering on eigenvectors.
pub fn spectral_clustering(_eigenvectors: &[Vec<f64>], _k: usize) -> Result<Partition> {
    todo!("will be implemented in a future PR")
}
