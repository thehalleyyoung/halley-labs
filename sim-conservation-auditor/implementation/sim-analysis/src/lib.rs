pub mod analyzer;
pub mod spectrum;
pub mod statistics;
pub mod correlation;
pub mod regression;
pub mod phase_portrait;
pub mod lyapunov;
pub mod poincare;
pub mod convergence;
pub mod backward_error;
pub mod sensitivity;

pub use analyzer::{TrajectoryAnalyzer, AnalysisReport, ViolationInterval, ErrorGrowthRate};
pub use spectrum::{DftResult, PowerSpectralDensity, SpectralPeak, WindowFunction, SpectralAnalyzer};
pub use statistics::{
    DescriptiveStats, Histogram, HistogramBin, BootstrapResult, KsTestResult,
    AndersonDarlingResult, StatisticalAnalyzer,
};
pub use correlation::{
    CorrelationResult, AutocorrelationResult, CrossCorrelationResult, RankCorrelation,
    CorrelationAnalyzer,
};
pub use regression::{
    LinearRegression, LinearFitResult, PolynomialRegression, PolynomialFitResult,
    ExponentialFit, ExponentialFitResult, PowerLawFit, PowerLawFitResult,
    RegressionDiagnostics, ResidualsAnalysis,
};
pub use phase_portrait::{
    PhasePortrait, PhasePoint2D, FixedPoint, StabilityKind, PhaseAreaResult,
    KamTorusResult,
};
pub use lyapunov::{
    LyapunovResult, LyapunovSpectrum, FtleField, LyapunovAnalyzer,
};
pub use poincare::{
    PoincareSection, SectionCrossing, ReturnMap, ReturnMapIsland, WindingNumberResult,
    PoincareAnalyzer,
};
pub use convergence::{
    RichardsonResult, ConvergenceOrderResult, StabilityRegion, ConvergenceAnalyzer,
};
pub use backward_error::{
    ModifiedEquation, ModifiedHamiltonian, ShadowOrbit, BackwardErrorAnalyzer,
};
pub use sensitivity::{
    ParameterSensitivity, MorrisResult, SobolIndices, SensitivityAnalyzer,
};
