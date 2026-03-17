//! Test oracle implementations for metamorphic relation validation.
//!
//! Provides various oracle strategies for determining whether a metamorphic
//! relation holds between original and transformed pipeline outputs.

pub mod composite;
pub mod distance_oracle;
pub mod entity_oracle;
pub mod pos_oracle;
pub mod structural_oracle;
pub mod threshold;

pub use composite::{CompositeOracle, OracleVote, VotingStrategy};
pub use distance_oracle::{DistanceOracle, DistanceMetric};
pub use entity_oracle::EntityPreservationOracle;
pub use pos_oracle::POSConsistencyOracle;
pub use structural_oracle::StructuralOracle;
pub use threshold::{ThresholdOracle, AdaptiveThreshold, ThresholdConfig};
