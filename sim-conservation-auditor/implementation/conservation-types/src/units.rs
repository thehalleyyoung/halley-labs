//! Physical unit system for dimensional analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Physical dimensions in SI base units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Dimension {
    pub length: i32,
    pub mass: i32,
    pub time: i32,
    pub current: i32,
    pub temperature: i32,
    pub amount: i32,
    pub luminosity: i32,
}

impl Dimension {
    pub fn dimensionless() -> Self {
        Self { length: 0, mass: 0, time: 0, current: 0, temperature: 0, amount: 0, luminosity: 0 }
    }

    pub fn length() -> Self {
        Self { length: 1, ..Self::dimensionless() }
    }

    pub fn mass() -> Self {
        Self { mass: 1, ..Self::dimensionless() }
    }

    pub fn time() -> Self {
        Self { time: 1, ..Self::dimensionless() }
    }

    pub fn velocity() -> Self {
        Self { length: 1, time: -1, ..Self::dimensionless() }
    }

    pub fn acceleration() -> Self {
        Self { length: 1, time: -2, ..Self::dimensionless() }
    }

    pub fn force() -> Self {
        Self { mass: 1, length: 1, time: -2, ..Self::dimensionless() }
    }

    pub fn energy() -> Self {
        Self { mass: 1, length: 2, time: -2, ..Self::dimensionless() }
    }

    pub fn momentum() -> Self {
        Self { mass: 1, length: 1, time: -1, ..Self::dimensionless() }
    }

    pub fn angular_momentum() -> Self {
        Self { mass: 1, length: 2, time: -1, ..Self::dimensionless() }
    }

    pub fn charge() -> Self {
        Self { current: 1, time: 1, ..Self::dimensionless() }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.length == 0 && self.mass == 0 && self.time == 0
            && self.current == 0 && self.temperature == 0
            && self.amount == 0 && self.luminosity == 0
    }

    pub fn multiply(&self, other: &Dimension) -> Dimension {
        Dimension {
            length: self.length + other.length,
            mass: self.mass + other.mass,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
        }
    }

    pub fn divide(&self, other: &Dimension) -> Dimension {
        Dimension {
            length: self.length - other.length,
            mass: self.mass - other.mass,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
        }
    }

    pub fn power(&self, n: i32) -> Dimension {
        Dimension {
            length: self.length * n,
            mass: self.mass * n,
            time: self.time * n,
            current: self.current * n,
            temperature: self.temperature * n,
            amount: self.amount * n,
            luminosity: self.luminosity * n,
        }
    }

    pub fn is_compatible(&self, other: &Dimension) -> bool {
        *self == *other
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }
        let mut parts = Vec::new();
        if self.length != 0 { parts.push(format!("L^{}", self.length)); }
        if self.mass != 0 { parts.push(format!("M^{}", self.mass)); }
        if self.time != 0 { parts.push(format!("T^{}", self.time)); }
        if self.current != 0 { parts.push(format!("I^{}", self.current)); }
        if self.temperature != 0 { parts.push(format!("Θ^{}", self.temperature)); }
        write!(f, "{}", parts.join(" "))
    }
}

/// A unit system for consistent dimensional analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitSystem {
    pub name: String,
    pub units: HashMap<String, UnitDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitDef {
    pub name: String,
    pub symbol: String,
    pub dimension: Dimension,
    pub si_factor: f64,
}

impl UnitSystem {
    pub fn si() -> Self {
        let mut units = HashMap::new();
        units.insert("meter".into(), UnitDef { name: "meter".into(), symbol: "m".into(), dimension: Dimension::length(), si_factor: 1.0 });
        units.insert("kilogram".into(), UnitDef { name: "kilogram".into(), symbol: "kg".into(), dimension: Dimension::mass(), si_factor: 1.0 });
        units.insert("second".into(), UnitDef { name: "second".into(), symbol: "s".into(), dimension: Dimension::time(), si_factor: 1.0 });
        units.insert("joule".into(), UnitDef { name: "joule".into(), symbol: "J".into(), dimension: Dimension::energy(), si_factor: 1.0 });
        units.insert("newton".into(), UnitDef { name: "newton".into(), symbol: "N".into(), dimension: Dimension::force(), si_factor: 1.0 });
        Self { name: "SI".into(), units }
    }

    pub fn natural() -> Self {
        let mut units = HashMap::new();
        units.insert("length".into(), UnitDef { name: "natural length".into(), symbol: "l".into(), dimension: Dimension::length(), si_factor: 1.0 });
        units.insert("mass".into(), UnitDef { name: "natural mass".into(), symbol: "m".into(), dimension: Dimension::mass(), si_factor: 1.0 });
        units.insert("time".into(), UnitDef { name: "natural time".into(), symbol: "t".into(), dimension: Dimension::time(), si_factor: 1.0 });
        Self { name: "Natural".into(), units }
    }

    pub fn check_dimensional_consistency(&self, dim1: &Dimension, dim2: &Dimension) -> bool {
        dim1.is_compatible(dim2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_energy() {
        let e = Dimension::energy();
        assert_eq!(e.mass, 1);
        assert_eq!(e.length, 2);
        assert_eq!(e.time, -2);
    }

    #[test]
    fn test_dimension_multiply() {
        let m = Dimension::mass();
        let v = Dimension::velocity();
        let p = m.multiply(&v);
        assert_eq!(p, Dimension::momentum());
    }

    #[test]
    fn test_dimension_divide() {
        let e = Dimension::energy();
        let t = Dimension::time();
        let power = e.divide(&t);
        assert_eq!(power.mass, 1);
        assert_eq!(power.length, 2);
        assert_eq!(power.time, -3);
    }

    #[test]
    fn test_si_system() {
        let si = UnitSystem::si();
        assert!(si.units.contains_key("joule"));
    }

    #[test]
    fn test_dimensional_consistency() {
        let si = UnitSystem::si();
        assert!(si.check_dimensional_consistency(&Dimension::energy(), &Dimension::energy()));
        assert!(!si.check_dimensional_consistency(&Dimension::energy(), &Dimension::momentum()));
    }
}
