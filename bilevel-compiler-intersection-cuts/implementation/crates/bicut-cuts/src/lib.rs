//! # bicut-cuts
//!
//! Bilevel intersection cuts and cut management.

#[cfg(feature = "extended")]
pub mod balas;
#[cfg(feature = "extended")]
pub mod cut_pool;
#[cfg(feature = "extended")]
pub mod disjunctive;
#[cfg(feature = "extended")]
pub mod gomory;
#[cfg(feature = "extended")]
pub mod intersection;
#[cfg(feature = "extended")]
pub mod manager;
#[cfg(feature = "extended")]
pub mod ray_tracing;
#[cfg(feature = "extended")]
pub mod separation;
#[cfg(feature = "extended")]
pub mod strengthening;

pub mod adaptive_cache;

use bicut_types::*;
use serde::{Deserialize, Serialize};

/// A linear cut: coefficients^T x {sense} rhs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cut {
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub sense: ConstraintSense,
    pub name: String,
    pub efficacy: f64,
}

impl Cut {
    pub fn new(coefficients: Vec<f64>, rhs: f64, sense: ConstraintSense) -> Self {
        Self {
            coefficients,
            rhs,
            sense,
            name: String::new(),
            efficacy: 0.0,
        }
    }
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
    pub fn is_violated(&self, point: &[f64], tol: f64) -> bool {
        let lhs: f64 = self
            .coefficients
            .iter()
            .zip(point)
            .map(|(a, x)| a * x)
            .sum();
        match self.sense {
            ConstraintSense::Ge => lhs < self.rhs - tol,
            ConstraintSense::Le => lhs > self.rhs + tol,
            ConstraintSense::Eq => (lhs - self.rhs).abs() > tol,
        }
    }
}

/// A pool of cuts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CutPool {
    pub cuts: Vec<Cut>,
    pub max_size: usize,
}

impl CutPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            cuts: Vec::new(),
            max_size,
        }
    }
    pub fn add_cut(&mut self, cut: Cut) -> bool {
        if self.cuts.len() < self.max_size {
            self.cuts.push(cut);
            true
        } else {
            false
        }
    }
    pub fn len(&self) -> usize {
        self.cuts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }
    pub fn clear(&mut self) {
        self.cuts.clear();
    }
}

/// Cut generation statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CutStats {
    pub total_generated: usize,
    pub total_added: usize,
    pub total_violated: usize,
    pub avg_efficacy: f64,
}
