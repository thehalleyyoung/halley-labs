// Futility predictor submodule: predict when decomposition won't help.

pub mod predictor;
pub mod threshold;

pub use predictor::{FutilityPredictor, FutilityPrediction, FutilityFeatures};
pub use threshold::{ThresholdCalibrator, ThresholdStrategy, ROCPoint, PRPoint};
