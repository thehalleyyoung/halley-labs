//! Trace query engine.
use serde::{Serialize, Deserialize};

/// Base trace query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceQuery { TimeRange(TimeRangeQuery), Particle(ParticleQuery), Conservation(ConservationQuery), MaxViolation(MaxViolationQuery), Aggregate(AggregateQuery) }

/// Query for a time range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRangeQuery { pub start: f64, pub end: f64 }

/// Query for a specific particle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleQuery { pub particle_index: usize }

/// Query for conservation law values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationQuery { pub law_name: String }

/// Query for maximum violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxViolationQuery { pub law_name: String }

/// Aggregate query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateQuery { pub field: String, pub operation: String }
