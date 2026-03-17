//! Interval arithmetic operations for Penumbra.
//!
//! Provides validated interval computation with directed rounding.

use serde::{Deserialize, Serialize};

/// A closed interval [lo, hi] bounding a real value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        0.5 * (self.lo + self.hi)
    }

    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }
}

impl Default for Interval {
    fn default() -> Self {
        Self::point(0.0)
    }
}
