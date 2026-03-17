//! Fixpoint iteration engine result types for abstract interpretation over PTA.
//!
//! Provides [`FixpointResult`] (the per-location abstract state map produced by
//! a fixpoint computation) and [`FixpointStatistics`] (iteration metrics).

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::domain::ProductDomain;
use crate::pta_types::LocationId;

// ---------------------------------------------------------------------------
// FixpointStatistics
// ---------------------------------------------------------------------------

/// Metrics collected during a fixpoint computation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixpointStatistics {
    /// Total number of Kleene iterations (ascending + descending).
    pub total_iterations: usize,
    /// Number of widening operator applications.
    pub widening_applications: usize,
    /// Number of narrowing (descending) iterations performed.
    pub narrowing_iterations: usize,
    /// Wall-clock computation time in milliseconds.
    pub computation_time_ms: u64,
    /// Number of distinct PTA locations processed.
    pub locations_processed: usize,
}

impl FixpointStatistics {
    pub fn new() -> Self {
        FixpointStatistics {
            total_iterations: 0,
            widening_applications: 0,
            narrowing_iterations: 0,
            computation_time_ms: 0,
            locations_processed: 0,
        }
    }
}

impl Default for FixpointStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FixpointStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "iterations={}, widen={}, narrow={}, time={}ms, locs={}",
            self.total_iterations,
            self.widening_applications,
            self.narrowing_iterations,
            self.computation_time_ms,
            self.locations_processed,
        )
    }
}

// ---------------------------------------------------------------------------
// FixpointResult
// ---------------------------------------------------------------------------

/// Result of a fixpoint computation over a pharmacological timed automaton.
///
/// Maps each PTA location to the abstract state (product domain) computed at
/// that location after the ascending chain (with widening) and optional
/// descending refinement (narrowing) have terminated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixpointResult {
    /// Abstract state at each PTA location.
    pub domain_at_location: HashMap<LocationId, ProductDomain>,
    /// Whether the iteration converged within the budget.
    pub converged: bool,
    /// Total number of Kleene iterations executed.
    pub iterations: usize,
    /// Detailed computation statistics.
    pub statistics: FixpointStatistics,
}

impl FixpointResult {
    /// Create an empty, unconverged result.
    pub fn new() -> Self {
        FixpointResult {
            domain_at_location: HashMap::new(),
            converged: false,
            iterations: 0,
            statistics: FixpointStatistics::new(),
        }
    }

    /// Create a converged result from a location map.
    pub fn converged(domain_at_location: HashMap<LocationId, ProductDomain>, iterations: usize) -> Self {
        let locations_processed = domain_at_location.len();
        FixpointResult {
            domain_at_location,
            converged: true,
            iterations,
            statistics: FixpointStatistics {
                total_iterations: iterations,
                locations_processed,
                ..FixpointStatistics::new()
            },
        }
    }

    /// Iterate over all `(location, state)` pairs.
    pub fn all_states(&self) -> Vec<(&LocationId, &ProductDomain)> {
        self.domain_at_location.iter().collect()
    }

    /// Look up the abstract state at a specific location.
    pub fn state_at(&self, location: &LocationId) -> Option<&ProductDomain> {
        self.domain_at_location.get(location)
    }

    /// Number of locations with computed states.
    pub fn location_count(&self) -> usize {
        self.domain_at_location.len()
    }
}

impl Default for FixpointResult {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FixpointResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "FixpointResult {{ converged={}, iterations={}, locations={} }}",
            self.converged,
            self.iterations,
            self.domain_at_location.len(),
        )?;
        for (loc, state) in &self.domain_at_location {
            writeln!(f, "  {}: {}", loc, state)?;
        }
        Ok(())
    }
}
