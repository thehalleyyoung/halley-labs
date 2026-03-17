//! Change point localization algorithms.
use serde::{Serialize, Deserialize};

/// Localization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationResult { pub change_points: Vec<usize>, pub cost: f64 }

/// Change point localization using binary search.
#[derive(Debug, Clone, Default)]
pub struct ChangePointLocalization;

/// Binary search localization.
#[derive(Debug, Clone, Default)]
pub struct BinarySearchLocalization;
impl BinarySearchLocalization {
    /// Find change points in a data series.
    pub fn detect(&self, data: &[f64], threshold: f64) -> LocalizationResult {
        let mut points = Vec::new();
        for i in 1..data.len() {
            if (data[i] - data[i-1]).abs() > threshold { points.push(i); }
        }
        LocalizationResult { change_points: points, cost: 0.0 }
    }
}

/// PELT (Pruned Exact Linear Time) change point detection.
#[derive(Debug, Clone)]
pub struct Pelt { pub penalty: f64 }
impl Pelt {
    pub fn new(penalty: f64) -> Self { Self { penalty } }
}

/// Cost functions for change point detection.
#[derive(Debug, Clone, Copy)]
pub enum CostFunction { L2, L1, Normal, Poisson }
