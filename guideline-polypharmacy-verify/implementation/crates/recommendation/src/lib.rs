//! guardpharma-recommendation: Polypharmacy conflict resolution recommendations.
//!
//! Provides dose adjustment, alternative medication finding, temporal scheduling,
//! and synthesis of actionable clinical recommendations from detected conflicts.

pub mod types;
pub mod schedule;
pub mod dose_adjustment;
pub mod alternative;
pub mod temporal;
pub mod synthesis;
pub mod verification_bridge;

pub use types::*;
pub use dose_adjustment::DoseAdjuster;
pub use alternative::AlternativeFinder;
pub use temporal::TemporalRecommender;
pub use synthesis::RecommendationSynthesizer;
pub use verification_bridge::VerificationBridge;
