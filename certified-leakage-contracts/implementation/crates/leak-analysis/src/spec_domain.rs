//! # Speculative Reachability Domain (D\_spec)
//!
//! Tracks which program points are reachable under transient execution,
//! including branch misprediction, store-to-load forwarding, and return
//! stack buffer misspeculation. The domain maintains a speculative window
//! bounding how far ahead of the architectural state the processor may
//! execute speculatively.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Serialize, Deserialize};
use thiserror::Error;

use shared_types::{BlockId, VirtualAddress};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors originating in the speculative reachability domain.
#[derive(Debug, Error)]
pub enum SpecDomainError {
    #[error("speculation window overflow: depth {depth} exceeds bound {bound}")]
    WindowOverflow { depth: u32, bound: u32 },

    #[error("invalid misspeculation kind for block {0}")]
    InvalidMisspecKind(BlockId),
}

// ---------------------------------------------------------------------------
// MisspecKind
// ---------------------------------------------------------------------------

/// Classification of the micro-architectural misspeculation that causes a
/// transient execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MisspecKind {
    /// Conditional or indirect branch misprediction (Spectre-v1 / v2).
    BranchMispredict,
    /// Store-to-load forwarding bypass (Spectre-v1.1 / v4).
    StoreToLoadForwarding,
    /// Return stack buffer mis-prediction (Spectre-RSB / ret2spec).
    ReturnStackBuffer,
    /// Speculative store bypass (Spectre-v4 / SSBD).
    SpeculativeStoreBypass,
    /// Microcode-assist–triggered speculation.
    MicrocodeAssist,
    /// No misspeculation — architecturally reachable path.
    Architectural,
}

impl fmt::Display for MisspecKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BranchMispredict => write!(f, "branch-mispredict"),
            Self::StoreToLoadForwarding => write!(f, "store-to-load"),
            Self::ReturnStackBuffer => write!(f, "rsb"),
            Self::SpeculativeStoreBypass => write!(f, "ssb"),
            Self::MicrocodeAssist => write!(f, "ucode-assist"),
            Self::Architectural => write!(f, "architectural"),
        }
    }
}

// ---------------------------------------------------------------------------
// SpecTag
// ---------------------------------------------------------------------------

/// A tag that uniquely labels a speculative execution path so that distinct
/// transient windows can be tracked independently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SpecTag {
    /// The block at which the misspeculation originates.
    pub origin: BlockId,
    /// The kind of misspeculation that spawned this transient path.
    pub kind: MisspecKind,
    /// Nesting depth (0 for the outermost speculative window).
    pub depth: u32,
}

impl SpecTag {
    /// Create a new speculative tag.
    pub fn new(origin: BlockId, kind: MisspecKind, depth: u32) -> Self {
        Self { origin, kind, depth }
    }

    /// Returns `true` when this tag represents an architectural (non-speculative)
    /// execution path.
    pub fn is_architectural(&self) -> bool {
        self.kind == MisspecKind::Architectural
    }
}

// ---------------------------------------------------------------------------
// SpecWindow
// ---------------------------------------------------------------------------

/// Represents the bounds of a speculative execution window — the maximum
/// number of instructions (or µ-ops) the processor may execute ahead of
/// the retirement point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecWindow {
    /// Maximum speculation depth in instructions.
    pub max_depth: u32,
    /// Current depth consumed so far along this path.
    pub current_depth: u32,
}

impl SpecWindow {
    /// Create a new speculation window with the given maximum depth.
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth, current_depth: 0 }
    }

    /// The number of speculative instructions remaining before the window
    /// is exhausted.
    pub fn remaining(&self) -> u32 {
        self.max_depth.saturating_sub(self.current_depth)
    }

    /// Returns `true` when the window has been fully consumed.
    pub fn is_exhausted(&self) -> bool {
        self.current_depth >= self.max_depth
    }

    /// Advance the window by `n` instructions, clamping at the maximum.
    pub fn advance(&mut self, n: u32) {
        self.current_depth = (self.current_depth + n).min(self.max_depth);
    }
}

impl Default for SpecWindow {
    fn default() -> Self {
        // Intel Skylake-class ROB size.
        Self::new(224)
    }
}

// ---------------------------------------------------------------------------
// SpecState
// ---------------------------------------------------------------------------

/// Per-program-point speculative state, recording all active speculative
/// execution contexts that may reach a given block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecState {
    /// Set of speculative tags active at this program point.
    pub active_tags: BTreeSet<SpecTag>,
    /// The speculation window governing transient execution depth.
    pub window: SpecWindow,
    /// Whether this state has been visited during the current fixpoint
    /// iteration (used by the worklist solver).
    pub visited: bool,
}

impl SpecState {
    /// Create a new state with no active speculative tags.
    pub fn empty(window: SpecWindow) -> Self {
        Self {
            active_tags: BTreeSet::new(),
            window,
            visited: false,
        }
    }

    /// Returns `true` when no speculative paths reach this point.
    pub fn is_architectural_only(&self) -> bool {
        self.active_tags.iter().all(|t| t.is_architectural())
    }

    /// Join (⊔) two speculative states by taking the union of active tags
    /// and the wider window.
    pub fn join(&self, other: &Self) -> Self {
        let mut tags = self.active_tags.clone();
        tags.extend(other.active_tags.iter().cloned());
        Self {
            active_tags: tags,
            window: SpecWindow {
                max_depth: self.window.max_depth.max(other.window.max_depth),
                current_depth: self.window.current_depth.min(other.window.current_depth),
            },
            visited: self.visited || other.visited,
        }
    }

    /// Widen the state to accelerate fixpoint convergence.
    pub fn widen(&self, _previous: &Self) -> Self {
        // Default widening: keep current state (overridden by concrete
        // implementations for termination guarantees).
        self.clone()
    }
}

impl Default for SpecState {
    fn default() -> Self {
        Self::empty(SpecWindow::default())
    }
}

// ---------------------------------------------------------------------------
// SpecDomain
// ---------------------------------------------------------------------------

/// The speculative reachability abstract domain (D\_spec).
///
/// Maintains a map from basic-block identifiers to their [`SpecState`],
/// providing lattice operations (join, meet, widen) and transfer functions
/// for speculative execution modelling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecDomain {
    /// Per-block speculative states.
    pub states: indexmap::IndexMap<BlockId, SpecState>,
    /// Global speculation window configuration.
    pub window: SpecWindow,
}

impl SpecDomain {
    /// Create a new, empty speculative reachability domain.
    pub fn new(window: SpecWindow) -> Self {
        Self {
            states: indexmap::IndexMap::new(),
            window,
        }
    }

    /// Retrieve the speculative state for a given block, or `None` if the
    /// block has not been visited.
    pub fn state_for(&self, block: BlockId) -> Option<&SpecState> {
        self.states.get(&block)
    }

    /// Update the speculative state for a given block, returning the
    /// previous state (if any).
    pub fn update(&mut self, block: BlockId, state: SpecState) -> Option<SpecState> {
        self.states.insert(block, state)
    }

    /// Mark a speculative path originating at `origin` with misspeculation
    /// `kind`, adding a fresh [`SpecTag`] to the target block's state.
    pub fn add_speculative_path(
        &mut self,
        origin: BlockId,
        target: BlockId,
        kind: MisspecKind,
        depth: u32,
    ) {
        let tag = SpecTag::new(origin, kind, depth);
        let state = self
            .states
            .entry(target)
            .or_insert_with(|| SpecState::empty(self.window));
        state.active_tags.insert(tag);
    }

    /// Perform a point-wise join across all blocks.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (block, other_state) in &other.states {
            let entry = result
                .states
                .entry(*block)
                .or_insert_with(|| SpecState::empty(self.window));
            *entry = entry.join(other_state);
        }
        result
    }

    /// Check whether the domain state has stabilised (no change from the
    /// previous iteration).
    pub fn is_stable(&self, previous: &Self) -> bool {
        self.states == previous.states
    }

    /// Retire all speculation for the given origin block, removing the
    /// corresponding tags across all program points.
    pub fn retire_speculation(&mut self, origin: BlockId) {
        for state in self.states.values_mut() {
            state.active_tags.retain(|t| t.origin != origin);
        }
    }

    /// Count the total number of active speculative tags across all blocks.
    pub fn total_active_tags(&self) -> usize {
        self.states.values().map(|s| s.active_tags.len()).sum()
    }
}

impl Default for SpecDomain {
    fn default() -> Self {
        Self::new(SpecWindow::default())
    }
}
