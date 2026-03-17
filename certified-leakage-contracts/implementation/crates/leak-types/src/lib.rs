//! Core type definitions for the Certified Leakage Contracts framework.
//!
//! This crate provides the abstract domain types and leakage measurement types
//! shared across the analysis pipeline.

pub mod domain;
pub mod leakage;

pub use domain::{AbstractDomain, FiniteDomain, ProductDomain, PowersetDomain, FlatLattice, BoolDomain};
pub use leakage::{LeakageBound, LeakageVector, LeakageMetric, ChannelCapacity, LeakageClassification, Observation, ObservationTrace};
