//! Runtime execution engine for Choreo spatial-event automata.
//!
//! Provides NFA token-passing execution, timer management, event scheduling,
//! execution trace recording, and scene state management.

pub mod executor;
pub mod timer;
pub mod scheduler;
pub mod trace_recorder;
pub mod scene_manager;
