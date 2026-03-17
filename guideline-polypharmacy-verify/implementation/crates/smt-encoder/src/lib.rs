//! # GuardPharma SMT Encoder
//!
//! Minimal SMT encoding module. Provides types used by the model-checker.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A simplified SMT variable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SmtVariable {
    pub name: String,
    pub sort: SmtSort,
}

/// SMT sort (type).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SmtSort {
    Bool,
    Real,
    Int,
}

/// An encoded BMC problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedProblem {
    pub variables: Vec<SmtVariable>,
    pub assertions: Vec<String>,
    pub bound: usize,
    pub num_clauses: usize,
    pub num_variables: usize,
}

impl EncodedProblem {
    pub fn empty(bound: usize) -> Self {
        Self {
            variables: Vec::new(),
            assertions: Vec::new(),
            bound,
            num_clauses: 0,
            num_variables: 0,
        }
    }
}

/// SMT value in a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Real(f64),
    Int(i64),
}

/// An SMT model (assignment from variables to values).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmtModel {
    pub assignments: HashMap<String, SmtValue>,
}

/// Solver result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverResult {
    Sat,
    Unsat,
    Unknown,
    Timeout,
}
