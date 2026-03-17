//! Conflict detection and analysis for Choreo interaction protocols.
//!
//! Provides deadlock detection, race condition analysis, unreachable state
//! identification, interference analysis, and consolidated conflict reporting.

pub mod deadlock;
pub mod race;
pub mod unreachable;
pub mod interference;
pub mod report;
