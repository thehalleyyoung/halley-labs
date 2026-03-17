//! Headless scene simulator with CPU-only collision for Choreo.
//!
//! This crate provides a deterministic simulation loop that advances
//! entities along scripted trajectories, monitors spatial predicates,
//! generates events, and steps a finite-state automaton.

pub mod simulator;
pub mod scenario;
pub mod collision;
pub mod trajectory;
pub mod benchmark;
