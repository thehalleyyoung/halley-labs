//! # isospec-witness
//!
//! Witness synthesis and minimization for transaction isolation anomalies.
//! Converts SAT SMT models into minimal, runnable SQL witness scripts that
//! demonstrate anomalous behavior on real database engines.

pub mod synthesizer;
pub mod minimizer;
pub mod sql_gen;
pub mod timing;
pub mod validator;
