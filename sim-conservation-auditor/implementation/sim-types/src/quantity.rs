use crate::units::{Dimension, Unit, UnitSystem};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Neg, Sub};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantityError {
    #[error("Incompatible dimensions: {0} vs {1}")]
    IncompatibleDimensions(Dimension, Dimension),
}

/// A physical quantity: a scalar value with an associated unit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Quantity {
    pub value: f64,
    pub unit: Unit,
}

impl Quantity {
    pub fn new(value: f64, unit: Unit) -> Self {
        Self { value, unit }
    }

    pub fn dimensionless(value: f64) -> Self {
        Self {
            value,
            unit: Unit::dimensionless(),
        }
    }

    pub fn meters(value: f64) -> Self {
        Self::new(value, Unit::si_meter())
    }

    pub fn kilograms(value: f64) -> Self {
        Self::new(value, Unit::si_kilogram())
    }

    pub fn seconds(value: f64) -> Self {
        Self::new(value, Unit::si_second())
    }

    pub fn joules(value: f64) -> Self {
        Self::new(value, Unit::si_joule())
    }

    pub fn newtons(value: f64) -> Self {
        Self::new(value, Unit::si_newton())
    }

    pub fn dimension(&self) -> Dimension {
        self.unit.dimension
    }

    pub fn is_compatible(&self, other: &Quantity) -> bool {
        self.unit.is_compatible(&other.unit)
    }

    /// Convert this quantity to SI base units.
    pub fn to_si(&self) -> f64 {
        self.unit.to_si(self.value)
    }

    /// Convert this quantity to another unit system.
    pub fn convert_to(&self, target_system: UnitSystem) -> Self {
        let si = self.to_si();
        let target_unit = Unit::new(self.unit.dimension, target_system);
        let value = target_unit.from_si(si);
        Self::new(value, target_unit)
    }

    pub fn abs(&self) -> Self {
        Self::new(self.value.abs(), self.unit)
    }

    pub fn sqrt(&self) -> Self {
        Self {
            value: self.value.sqrt(),
            unit: Unit {
                dimension: Dimension {
                    length: self.unit.dimension.length / 2,
                    mass: self.unit.dimension.mass / 2,
                    time: self.unit.dimension.time / 2,
                    current: self.unit.dimension.current / 2,
                    temperature: self.unit.dimension.temperature / 2,
                    amount: self.unit.dimension.amount / 2,
                    luminosity: self.unit.dimension.luminosity / 2,
                },
                system: self.unit.system,
                scale: self.unit.scale.sqrt(),
                offset: 0.0,
            },
        }
    }

    pub fn powi(&self, n: i32) -> Self {
        Self {
            value: self.value.powi(n),
            unit: Unit {
                dimension: self.unit.dimension.pow(n as i8),
                system: self.unit.system,
                scale: self.unit.scale.powi(n),
                offset: 0.0,
            },
        }
    }

    /// Check that two quantities have the same dimension, return error otherwise.
    pub fn check_compatible(&self, other: &Quantity) -> Result<(), QuantityError> {
        if self.unit.dimension == other.unit.dimension {
            Ok(())
        } else {
            Err(QuantityError::IncompatibleDimensions(
                self.unit.dimension,
                other.unit.dimension,
            ))
        }
    }

    /// Try to add two quantities. Returns error if dimensions don't match.
    pub fn try_add(&self, other: &Quantity) -> Result<Quantity, QuantityError> {
        self.check_compatible(other)?;
        Ok(Quantity::new(self.value + other.value, self.unit))
    }

    /// Try to subtract two quantities.
    pub fn try_sub(&self, other: &Quantity) -> Result<Quantity, QuantityError> {
        self.check_compatible(other)?;
        Ok(Quantity::new(self.value - other.value, self.unit))
    }

    /// Relative difference between two quantities of the same dimension.
    pub fn relative_difference(&self, other: &Quantity) -> Option<f64> {
        if !self.is_compatible(other) {
            return None;
        }
        let avg = (self.value.abs() + other.value.abs()) / 2.0;
        if avg < 1e-30 {
            Some(0.0)
        } else {
            Some((self.value - other.value).abs() / avg)
        }
    }
}

impl Add for Quantity {
    type Output = Self;
    /// Panics if dimensions are incompatible.
    fn add(self, rhs: Self) -> Self {
        assert_eq!(
            self.unit.dimension, rhs.unit.dimension,
            "Cannot add quantities with different dimensions"
        );
        Self::new(self.value + rhs.value, self.unit)
    }
}

impl Sub for Quantity {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_eq!(
            self.unit.dimension, rhs.unit.dimension,
            "Cannot subtract quantities with different dimensions"
        );
        Self::new(self.value - rhs.value, self.unit)
    }
}

impl Mul for Quantity {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            unit: self.unit.multiply_units(&rhs.unit),
        }
    }
}

impl Mul<f64> for Quantity {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs, self.unit)
    }
}

impl Mul<Quantity> for f64 {
    type Output = Quantity;
    fn mul(self, rhs: Quantity) -> Quantity {
        Quantity::new(self * rhs.value, rhs.unit)
    }
}

impl Div for Quantity {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            value: self.value / rhs.value,
            unit: self.unit.divide_units(&rhs.unit),
        }
    }
}

impl Div<f64> for Quantity {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.value / rhs, self.unit)
    }
}

impl Neg for Quantity {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value, self.unit)
    }
}

impl PartialEq for Quantity {
    fn eq(&self, other: &Self) -> bool {
        self.unit.dimension == other.unit.dimension
            && (self.value - other.value).abs() < 1e-15
    }
}

impl std::fmt::Display for Quantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.6} [{}]", self.value, self.unit.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_add_same_dimension() {
        let a = Quantity::meters(3.0);
        let b = Quantity::meters(4.0);
        let c = a + b;
        assert!((c.value - 7.0).abs() < EPS);
        assert_eq!(c.unit.dimension, Dimension::LENGTH);
    }

    #[test]
    #[should_panic]
    fn test_add_different_dimension_panics() {
        let a = Quantity::meters(3.0);
        let b = Quantity::seconds(4.0);
        let _ = a + b;
    }

    #[test]
    fn test_multiply_dimensions() {
        let force = Quantity::newtons(10.0);
        let dist = Quantity::meters(5.0);
        let work = force * dist;
        assert!((work.value - 50.0).abs() < EPS);
        assert_eq!(work.unit.dimension, Dimension::ENERGY);
    }

    #[test]
    fn test_divide_dimensions() {
        let dist = Quantity::meters(100.0);
        let time = Quantity::seconds(10.0);
        let vel = dist / time;
        assert!((vel.value - 10.0).abs() < EPS);
        assert_eq!(vel.unit.dimension, Dimension::VELOCITY);
    }

    #[test]
    fn test_powi() {
        let length = Quantity::meters(3.0);
        let area = length.powi(2);
        assert!((area.value - 9.0).abs() < EPS);
        assert_eq!(area.unit.dimension.length, 2);
    }

    #[test]
    fn test_scalar_mul() {
        let q = Quantity::meters(5.0) * 3.0;
        assert!((q.value - 15.0).abs() < EPS);
    }

    #[test]
    fn test_neg() {
        let q = -Quantity::meters(5.0);
        assert!((q.value - (-5.0)).abs() < EPS);
    }

    #[test]
    fn test_try_add_compatible() {
        let a = Quantity::meters(3.0);
        let b = Quantity::meters(4.0);
        let c = a.try_add(&b).unwrap();
        assert!((c.value - 7.0).abs() < EPS);
    }

    #[test]
    fn test_try_add_incompatible() {
        let a = Quantity::meters(3.0);
        let b = Quantity::seconds(4.0);
        assert!(a.try_add(&b).is_err());
    }

    #[test]
    fn test_relative_difference() {
        let a = Quantity::joules(100.0);
        let b = Quantity::joules(101.0);
        let rd = a.relative_difference(&b).unwrap();
        assert!((rd - 1.0 / 100.5).abs() < 1e-10);
    }

    #[test]
    fn test_convert_cgs_to_si() {
        let q = Quantity::new(
            100.0,
            Unit::new(Dimension::LENGTH, UnitSystem::CGS),
        );
        let si = q.convert_to(UnitSystem::SI);
        assert!((si.value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dimensionless() {
        let q = Quantity::dimensionless(42.0);
        assert!(q.dimension().is_dimensionless());
    }

    #[test]
    fn test_abs() {
        let q = Quantity::meters(-5.0);
        assert!((q.abs().value - 5.0).abs() < EPS);
    }
}
