// Structure detection submodule: identify decomposable structures in MIP instances.

pub mod detector;
pub mod benders_detector;
pub mod dw_detector;

pub use detector::{StructureDetector, StructureType, StructureScore, DetectionResult};
pub use benders_detector::{BendersDetector, BendersScore, ComplicatingVariable};
pub use dw_detector::{DWDetector, DWScore, LinkingConstraint};
