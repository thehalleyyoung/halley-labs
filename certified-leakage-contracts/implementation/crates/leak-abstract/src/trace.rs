//! Abstract trace semantics for observable side-channel events.
//!
//! Tracks an over-approximation of the sequence of observations (cache
//! accesses, branch directions, speculative execution markers) that an
//! attacker may witness.  This forms the basis for computing leakage
//! contracts.

use serde::{Serialize, Deserialize};
use smallvec::SmallVec;

use shared_types::SecurityLevel;

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// A single observable event in a trace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Observation {
    /// The kind of observation.
    pub kind: ObservationKind,
    /// Program address at which the observation occurs.
    pub address: u64,
    /// Optional cache line involved (for memory observations).
    pub cache_line: Option<u64>,
    /// Security level of the data that influences this observation.
    pub security_level: SecurityLevel,
    /// Whether this observation occurs on a speculative path.
    pub speculative: bool,
}

/// Classification of observable events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationKind {
    /// A cache load or store.
    CacheAccess,
    /// A branch direction (taken / not-taken).
    BranchDirection,
    /// A speculative execution boundary (begin / commit / rollback).
    SpeculationBoundary,
    /// An observable timing variation.
    Timing,
}

impl Observation {
    /// Create a new cache-access observation.
    pub fn cache_access(address: u64, cache_line: u64, level: SecurityLevel) -> Self {
        Self {
            kind: ObservationKind::CacheAccess,
            address,
            cache_line: Some(cache_line),
            security_level: level,
            speculative: false,
        }
    }

    /// Create a new branch-direction observation.
    pub fn branch(address: u64, level: SecurityLevel) -> Self {
        Self {
            kind: ObservationKind::BranchDirection,
            address,
            cache_line: None,
            security_level: level,
            speculative: false,
        }
    }

    /// Mark this observation as occurring on a speculative path.
    pub fn as_speculative(mut self) -> Self {
        self.speculative = true;
        self
    }
}

// ---------------------------------------------------------------------------
// TraceElement
// ---------------------------------------------------------------------------

/// An element of an abstract trace – either a concrete observation or a
/// structural marker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceElement {
    /// A single observable event.
    Observe(Observation),
    /// Entry into a basic block.
    BlockEntry(u64),
    /// Exit from a basic block.
    BlockExit(u64),
    /// Start of a speculative window.
    SpeculationStart {
        /// Address of the mis-speculated branch.
        branch_address: u64,
        /// Maximum speculation depth (in μops).
        depth: usize,
    },
    /// End of a speculative window (commit or rollback).
    SpeculationEnd {
        /// Whether the speculative path was committed.
        committed: bool,
    },
    /// An opaque marker carrying an arbitrary tag (for extensibility).
    Marker(String),
}

// ---------------------------------------------------------------------------
// AbstractTrace
// ---------------------------------------------------------------------------

/// An over-approximation of the set of possible observation traces.
///
/// Internally stored as an ordered sequence of [`TraceElement`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbstractTrace {
    /// Ordered sequence of trace elements.
    pub elements: SmallVec<[TraceElement; 16]>,
    /// Whether the trace is known to be complete (covers all paths).
    pub complete: bool,
}

impl AbstractTrace {
    /// Create an empty trace.
    pub fn empty() -> Self {
        Self {
            elements: SmallVec::new(),
            complete: true,
        }
    }

    /// Append an observation to the trace.
    pub fn observe(&mut self, obs: Observation) {
        self.elements.push(TraceElement::Observe(obs));
    }

    /// Append a block-entry marker.
    pub fn enter_block(&mut self, block_id: u64) {
        self.elements.push(TraceElement::BlockEntry(block_id));
    }

    /// Append a block-exit marker.
    pub fn exit_block(&mut self, block_id: u64) {
        self.elements.push(TraceElement::BlockExit(block_id));
    }

    /// Append a speculation-start marker.
    pub fn begin_speculation(&mut self, branch_address: u64, depth: usize) {
        self.elements.push(TraceElement::SpeculationStart {
            branch_address,
            depth,
        });
    }

    /// Append a speculation-end marker.
    pub fn end_speculation(&mut self, committed: bool) {
        self.elements.push(TraceElement::SpeculationEnd { committed });
    }

    /// Concatenate another trace onto this one.
    pub fn extend(&mut self, other: &AbstractTrace) {
        self.elements.extend(other.elements.iter().cloned());
        self.complete = self.complete && other.complete;
    }

    /// Return the number of elements in the trace.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` if the trace contains no elements.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Iterate over all [`Observation`]s in the trace.
    pub fn observations(&self) -> impl Iterator<Item = &Observation> {
        self.elements.iter().filter_map(|e| match e {
            TraceElement::Observe(obs) => Some(obs),
            _ => None,
        })
    }

    /// Count the number of speculative observations.
    pub fn speculative_observation_count(&self) -> usize {
        self.observations().filter(|o| o.speculative).count()
    }
}

impl Default for AbstractTrace {
    fn default() -> Self {
        Self::empty()
    }
}
