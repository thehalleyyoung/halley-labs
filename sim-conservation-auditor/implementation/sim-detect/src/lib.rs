//! # sim-detect
//!
//! Statistical and symbolic detection of conservation law violations.
//!
//! Provides a comprehensive suite of algorithms for detecting, classifying,
//! and localizing violations of conservation laws in physics simulations.

pub mod detector;
pub mod statistical;
pub mod symbolic;
pub mod drift_detect;
pub mod cusum;
pub mod classifier;
pub mod localization;
pub mod ensemble;
pub mod noise_model;
pub mod anomaly;

pub use detector::{Detector, ViolationDetector, DetectionResult, DetectionConfig};
pub use statistical::{
    ChiSquaredTest, KolmogorovSmirnovTest, GrubbsTest, FTest, WaldTest, TTest,
    StatisticalTest, TestResult,
};
pub use symbolic::{SymbolicExpression, SymbolicChecker};
pub use drift_detect::{PageHinkley, Adwin, Ddm, Eddm, DriftDetector, DriftStatus};
pub use cusum::{Cusum, TwoSidedCusum, VmaskCusum, CusumResult};
pub use classifier::{ViolationType, ViolationClassifier, ClassificationResult};
pub use localization::{
    ChangePointLocalization, BinarySearchLocalization, Pelt, CostFunction, LocalizationResult,
};
pub use ensemble::{EnsembleDetector, VotingStrategy, EnsembleResult};
pub use noise_model::{
    WhiteNoiseModel, PinkNoiseModel, BrownNoiseModel, NoiseEstimation, NoiseModel,
};
pub use anomaly::{
    ZScoreAnomaly, IqrAnomaly, IsolationScore, LocalOutlierFactor, MovingMedianAnomaly,
    AnomalyDetector, AnomalyResult,
};
