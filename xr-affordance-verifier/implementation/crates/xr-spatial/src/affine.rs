//! Affine arithmetic forms for sound over-approximation.

use crate::interval::Interval;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Affine form: x₀ + Σ xᵢεᵢ where εᵢ ∈ [-1, 1].
/// Tracks correlations between variables for tighter bounds than interval arithmetic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineForm {
    /// Central value.
    pub center: f64,
    /// Noise symbol coefficients (correlated noise terms).
    pub terms: Vec<f64>,
    /// Accumulated rounding error (uncorrelated).
    pub radius: f64,
}

static NEXT_SYMBOL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn fresh_symbol() -> usize {
    NEXT_SYMBOL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl AffineForm {
    /// Create affine form from a constant.
    pub fn constant(val: f64) -> Self {
        Self {
            center: val,
            terms: Vec::new(),
            radius: 0.0,
        }
    }

    /// Create affine form from an interval, introducing a fresh noise symbol.
    pub fn from_interval(iv: &Interval) -> Self {
        let center = iv.midpoint();
        let rad = iv.radius();
        let idx = fresh_symbol();
        let mut terms = vec![0.0; idx + 1];
        terms[idx] = rad;
        Self {
            center,
            terms,
            radius: 0.0,
        }
    }

    /// Convert back to an interval.
    pub fn to_interval(&self) -> Interval {
        let total_rad: f64 = self.terms.iter().map(|t| t.abs()).sum::<f64>() + self.radius;
        Interval::new(self.center - total_rad, self.center + total_rad)
    }

    /// Total radius including all noise.
    pub fn total_radius(&self) -> f64 {
        self.terms.iter().map(|t| t.abs()).sum::<f64>() + self.radius
    }

    /// Multiply by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            center: self.center * s,
            terms: self.terms.iter().map(|t| t * s).collect(),
            radius: self.radius * s.abs(),
        }
    }

    /// Add a constant.
    pub fn offset(&self, c: f64) -> Self {
        Self {
            center: self.center + c,
            terms: self.terms.clone(),
            radius: self.radius,
        }
    }

    /// Affine addition.
    pub fn add(&self, other: &AffineForm) -> Self {
        let len = self.terms.len().max(other.terms.len());
        let mut terms = vec![0.0; len];
        for (i, t) in self.terms.iter().enumerate() {
            terms[i] += t;
        }
        for (i, t) in other.terms.iter().enumerate() {
            terms[i] += t;
        }
        Self {
            center: self.center + other.center,
            terms,
            radius: self.radius + other.radius,
        }
    }

    /// Affine subtraction.
    pub fn sub(&self, other: &AffineForm) -> Self {
        self.add(&other.scale(-1.0))
    }

    /// Affine multiplication (introduces a new noise symbol for nonlinear term).
    pub fn mul(&self, other: &AffineForm) -> Self {
        let len = self.terms.len().max(other.terms.len());
        let mut terms = vec![0.0; len];
        for (i, t) in self.terms.iter().enumerate() {
            terms[i] += t * other.center;
        }
        for (i, t) in other.terms.iter().enumerate() {
            terms[i] += t * self.center;
        }
        let self_rad = self.total_radius();
        let other_rad = other.total_radius();
        let nonlinear_err = self_rad * other_rad + self.radius * other.radius;
        Self {
            center: self.center * other.center,
            terms,
            radius: self.radius * other.center.abs()
                + other.radius * self.center.abs()
                + nonlinear_err,
        }
    }

    /// Conservative cosine over an affine form using Chebyshev linearization.
    pub fn cos(&self) -> Self {
        let iv = self.to_interval();
        let cos_iv = iv.cos();
        let mid = cos_iv.midpoint();
        let rad = cos_iv.radius();
        let idx = fresh_symbol();
        let mut terms = vec![0.0; idx + 1];
        terms[idx] = rad;
        Self {
            center: mid,
            terms,
            radius: 0.0,
        }
    }

    /// Conservative sine.
    pub fn sin(&self) -> Self {
        let iv = self.to_interval();
        let sin_iv = iv.sin();
        let mid = sin_iv.midpoint();
        let rad = sin_iv.radius();
        let idx = fresh_symbol();
        let mut terms = vec![0.0; idx + 1];
        terms[idx] = rad;
        Self {
            center: mid,
            terms,
            radius: 0.0,
        }
    }

    /// Number of noise symbols.
    pub fn num_symbols(&self) -> usize {
        self.terms.len()
    }

    /// Get the central value.
    pub fn central_value(&self) -> f64 {
        self.center
    }

    /// Check if this affine form's interval is a subset of another.
    pub fn is_subset_of(&self, other: &AffineForm) -> bool {
        self.to_interval().is_subset_of(&other.to_interval())
    }
}

impl fmt::Display for AffineForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.center)?;
        for (i, t) in self.terms.iter().enumerate() {
            if t.abs() > 1e-15 {
                write!(f, " + {:.4}·ε{}", t, i)?;
            }
        }
        if self.radius > 1e-15 {
            write!(f, " ± {:.4}", self.radius)?;
        }
        Ok(())
    }
}

/// 3D vector of affine forms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineVector3 {
    pub x: AffineForm,
    pub y: AffineForm,
    pub z: AffineForm,
}

impl AffineVector3 {
    pub fn new(x: AffineForm, y: AffineForm, z: AffineForm) -> Self { Self { x, y, z } }
    pub fn constant(x: f64, y: f64, z: f64) -> Self {
        Self { x: AffineForm::constant(x), y: AffineForm::constant(y), z: AffineForm::constant(z) }
    }
    pub fn to_intervals(&self) -> [Interval; 3] {
        [self.x.to_interval(), self.y.to_interval(), self.z.to_interval()]
    }
    pub fn add(&self, other: &AffineVector3) -> AffineVector3 {
        AffineVector3 { x: self.x.add(&other.x), y: self.y.add(&other.y), z: self.z.add(&other.z) }
    }
    pub fn sub(&self, other: &AffineVector3) -> AffineVector3 {
        AffineVector3 { x: self.x.sub(&other.x), y: self.y.sub(&other.y), z: self.z.sub(&other.z) }
    }
    pub fn norm_squared(&self) -> AffineForm {
        self.x.mul(&self.x).add(&self.y.mul(&self.y)).add(&self.z.mul(&self.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let a = AffineForm::constant(5.0);
        let iv = a.to_interval();
        assert!((iv.lo - 5.0).abs() < 1e-10);
        assert!((iv.hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_interval() {
        let iv = Interval::new(1.0, 3.0);
        let a = AffineForm::from_interval(&iv);
        let back = a.to_interval();
        assert!(back.lo <= 1.0 + 1e-10);
        assert!(back.hi >= 3.0 - 1e-10);
    }

    #[test]
    fn test_add_sub() {
        let a = AffineForm::from_interval(&Interval::new(1.0, 2.0));
        let b = AffineForm::from_interval(&Interval::new(3.0, 4.0));
        let sum = a.add(&b);
        let iv = sum.to_interval();
        assert!(iv.lo <= 4.0 + 1e-10);
        assert!(iv.hi >= 6.0 - 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = AffineForm::from_interval(&Interval::new(-1.0, 1.0));
        let diff = x.sub(&x);
        let iv = diff.to_interval();
        assert!((iv.lo).abs() < 1e-10);
        assert!((iv.hi).abs() < 1e-10);
    }
}
