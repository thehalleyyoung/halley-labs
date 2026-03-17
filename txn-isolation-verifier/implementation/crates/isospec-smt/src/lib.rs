//! # isospec-smt
//!
//! SMT-based verification engine for transaction isolation anomaly detection.
//! Uses bounded model checking to encode schedule constraints and check for
//! isolation level violations via SMT solving.

pub mod solver;
pub mod encoding;
pub mod model;
pub mod optimizer;
pub mod incremental;
pub mod formula;
