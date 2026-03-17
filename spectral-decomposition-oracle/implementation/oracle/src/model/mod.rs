// Trained model management: persistence, versioning, and selection.

pub mod persistence;
pub mod selection;

pub use persistence::{ModelStore, ModelMetadata, ModelCheckpoint};
pub use selection::{ModelSelector, ModelComparison, AutoMLConfig};
