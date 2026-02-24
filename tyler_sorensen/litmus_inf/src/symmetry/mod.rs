//! Fixed symmetry framework + partial-order reduction.
//!
//! Addresses critique #2 (negative speedup) by:
//!   1. Canonical labeling instead of full automorphism
//!   2. Separate thread/memory symmetry (cheaper)
//!   3. Symmetry-aware BFS that skips symmetric states
//!   4. Persistent/sleep set partial-order reduction

pub mod canonical_labeling;
pub mod partial_order_reduction;
pub mod symmetry_aware_bfs;
pub mod separate_symmetry;

pub use canonical_labeling::{
    CanonicalLabeler, CanonicalForm, LabelingConfig,
    ColorRefinement, CellSelector,
};
pub use partial_order_reduction::{
    PersistentSetComputer, SleepSetTracker, PorExplorer,
    PorConfig, PorStatistics, AmpleSetComputer,
};
pub use symmetry_aware_bfs::{
    SymmetryAwareBfs, BfsConfig, BfsStatistics, BfsState,
};
pub use separate_symmetry::{
    ThreadSymmetryComputer, MemorySymmetryComputer,
    SeparateSymmetryConfig, SeparateSymmetryResult,
};
