//! sonitype-psychoacoustic — Psychoacoustic cost models for the SoniType
//! perceptual sonification compiler.
//!
//! This crate provides:
//! - **Critical-band masking** ([`masking`]): Schroeder spreading, Zwicker model,
//!   global masking threshold, excitation patterns, specific loudness.
//! - **JND models** ([`jnd`]): pitch, loudness, temporal, timbre, and spatial
//!   just-noticeable differences; composite d′ discriminability.
//! - **Bregman stream segregation** ([`segregation`]): onset synchrony,
//!   harmonicity, spectral proximity, common fate, spatial separation predicates;
//!   pairwise segregation matrix.
//! - **Cognitive load** ([`cognitive_load`]): resource algebra (L, ⊕, ≤ B),
//!   working memory model, attention allocation, load optimization.
//! - **Loudness** ([`loudness`]): Zwicker & Stevens models, ISO 226 equal-loudness
//!   contours, A/C weighting.
//! - **Pitch** ([`pitch`]): place/temporal/virtual pitch, pitch scale converters,
//!   pitch contour modelling.
//! - **Timbre** ([`timbre`]): spectral moments, ADSR, Plomp-Levelt roughness,
//!   perceptual timbre distance.
//! - **Integrated model** ([`models`]): unified analysis pipeline, constraint
//!   checker, cross-model comparison utilities.

pub mod masking;
pub mod jnd;
pub mod segregation;
pub mod cognitive_load;
pub mod loudness;
pub mod pitch;
pub mod timbre;
pub mod models;

// Re-exports of the most commonly used types and models.
pub use masking::{
    SchroederSpreadingFunction, ZwickerMaskingModel, MaskingAnalyzer,
    SpectralMaskingMatrix, MaskedRegion, ReallocationSuggestion, MaskingReport,
    BarkBand, NUM_BARK_BANDS, hz_to_bark, bark_to_hz, bark_band_edges,
};
pub use jnd::{
    PitchJnd, LoudnessJnd, TemporalJnd, TimbreJnd, SpatialJnd,
    JndValidator, JndDimension, PerceptualParams, ValidationReport,
    DimensionResult, JndMatrix,
    d_prime_from_margins, p_correct_from_d_prime, multi_dimensional_d_prime,
};
pub use segregation::{
    SegregationResult, OnsetSynchronyPredicate, HarmonicityPredicate,
    SpectralProximityPredicate, CommonFatePredicate, SpatialSeparationPredicate,
    StreamSegregationAnalyzer, SegregationMatrix,
};
pub use cognitive_load::{
    CognitiveLoadModel, CognitiveLoadBudget, WorkingMemoryModel,
    CognitiveLoadOptimizer, LoadComposition,
};
pub use loudness::{
    ZwickerLoudnessModel, StevensLoudnessModel, LoudnessNormalization,
    EqualLoudnessContour, WeightingType, LoudnessEqualizer,
};
pub use pitch::{
    PitchModel, PitchScaleConverter, PitchContour, PitchDirection,
    PitchAnalyzer,
};
pub use timbre::{
    TimbreSpace, TimbreDescriptor, TimbreDistance, TimbreWeights,
    TimbreFeatureExtractor, AdsrParams,
};
pub use models::{
    IntegratedPsychoacousticModel, PerceptualAnalysisResult,
    PerceptualConstraintChecker, ConstraintReport, ConstraintViolation,
    ConstraintSeverity,
};
