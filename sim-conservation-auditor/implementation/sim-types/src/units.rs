use serde::{Deserialize, Serialize};
use std::fmt;

/// Physical dimension represented as exponents of base SI quantities.
/// [Length^a * Mass^b * Time^c * Current^d * Temperature^e * Amount^f * Luminosity^g]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dimension {
    pub length: i8,
    pub mass: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminosity: i8,
}

impl Dimension {
    pub const DIMENSIONLESS: Dimension = Dimension {
        length: 0, mass: 0, time: 0, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const LENGTH: Dimension = Dimension {
        length: 1, mass: 0, time: 0, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const MASS: Dimension = Dimension {
        length: 0, mass: 1, time: 0, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const TIME: Dimension = Dimension {
        length: 0, mass: 0, time: 1, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const VELOCITY: Dimension = Dimension {
        length: 1, mass: 0, time: -1, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const ACCELERATION: Dimension = Dimension {
        length: 1, mass: 0, time: -2, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const FORCE: Dimension = Dimension {
        length: 1, mass: 1, time: -2, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const ENERGY: Dimension = Dimension {
        length: 2, mass: 1, time: -2, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const MOMENTUM: Dimension = Dimension {
        length: 1, mass: 1, time: -1, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const ANGULAR_MOMENTUM: Dimension = Dimension {
        length: 2, mass: 1, time: -1, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const CHARGE: Dimension = Dimension {
        length: 0, mass: 0, time: 1, current: 1,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub const POWER: Dimension = Dimension {
        length: 2, mass: 1, time: -3, current: 0,
        temperature: 0, amount: 0, luminosity: 0,
    };

    pub fn mul(self, other: Self) -> Self {
        Self {
            length: self.length + other.length,
            mass: self.mass + other.mass,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
        }
    }

    pub fn div(self, other: Self) -> Self {
        Self {
            length: self.length - other.length,
            mass: self.mass - other.mass,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
        }
    }

    pub fn pow(self, n: i8) -> Self {
        Self {
            length: self.length * n,
            mass: self.mass * n,
            time: self.time * n,
            current: self.current * n,
            temperature: self.temperature * n,
            amount: self.amount * n,
            luminosity: self.luminosity * n,
        }
    }

    pub fn is_dimensionless(self) -> bool {
        self == Self::DIMENSIONLESS
    }

    pub fn is_compatible(self, other: Self) -> bool {
        self == other
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        let labels = [
            ("L", self.length),
            ("M", self.mass),
            ("T", self.time),
            ("I", self.current),
            ("Θ", self.temperature),
            ("N", self.amount),
            ("J", self.luminosity),
        ];
        for (label, exp) in &labels {
            if *exp != 0 {
                if *exp == 1 {
                    parts.push(label.to_string());
                } else {
                    parts.push(format!("{}^{}", label, exp));
                }
            }
        }
        if parts.is_empty() {
            write!(f, "1")
        } else {
            write!(f, "{}", parts.join("·"))
        }
    }
}

/// System of units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnitSystem {
    SI,
    CGS,
    Natural,
    Geometrized,
    Planck,
}

impl UnitSystem {
    /// Conversion factor from `self` to SI for the given dimension.
    pub fn to_si_factor(self, dim: Dimension) -> f64 {
        match self {
            UnitSystem::SI => 1.0,
            UnitSystem::CGS => {
                // CGS: length in cm, mass in g, time in s
                let length_factor = 0.01_f64.powi(dim.length as i32);
                let mass_factor = 0.001_f64.powi(dim.mass as i32);
                length_factor * mass_factor
            }
            UnitSystem::Natural => {
                // Natural units: c = ℏ = 1, energy in GeV
                // Simplified: everything is 1 (dimensionless in natural units)
                1.0
            }
            UnitSystem::Geometrized => 1.0,
            UnitSystem::Planck => 1.0,
        }
    }

    /// Convert a value from one unit system to another.
    pub fn convert(value: f64, dim: Dimension, from: UnitSystem, to: UnitSystem) -> f64 {
        let si_value = value * from.to_si_factor(dim);
        si_value / to.to_si_factor(dim)
    }
}

/// A concrete unit (dimension + unit system + optional scale factor).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub dimension: Dimension,
    pub system: UnitSystem,
    pub scale: f64,
    pub offset: f64,
}

impl Unit {
    pub fn new(dimension: Dimension, system: UnitSystem) -> Self {
        Self {
            dimension,
            system,
            scale: 1.0,
            offset: 0.0,
        }
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_offset(mut self, offset: f64) -> Self {
        self.offset = offset;
        self
    }

    pub fn si_meter() -> Self {
        Self::new(Dimension::LENGTH, UnitSystem::SI)
    }

    pub fn si_kilogram() -> Self {
        Self::new(Dimension::MASS, UnitSystem::SI)
    }

    pub fn si_second() -> Self {
        Self::new(Dimension::TIME, UnitSystem::SI)
    }

    pub fn si_joule() -> Self {
        Self::new(Dimension::ENERGY, UnitSystem::SI)
    }

    pub fn si_newton() -> Self {
        Self::new(Dimension::FORCE, UnitSystem::SI)
    }

    pub fn si_coulomb() -> Self {
        Self::new(Dimension::CHARGE, UnitSystem::SI)
    }

    pub fn dimensionless() -> Self {
        Self::new(Dimension::DIMENSIONLESS, UnitSystem::SI)
    }

    /// Convert a value in this unit to SI base units.
    pub fn to_si(&self, value: f64) -> f64 {
        (value * self.scale + self.offset) * self.system.to_si_factor(self.dimension)
    }

    /// Convert a value from SI base units to this unit.
    pub fn from_si(&self, value: f64) -> f64 {
        let base = value / self.system.to_si_factor(self.dimension);
        (base - self.offset) / self.scale
    }

    pub fn is_compatible(&self, other: &Unit) -> bool {
        self.dimension.is_compatible(other.dimension)
    }

    pub fn multiply_units(&self, other: &Unit) -> Unit {
        Unit {
            dimension: self.dimension.mul(other.dimension),
            system: self.system,
            scale: self.scale * other.scale,
            offset: 0.0,
        }
    }

    pub fn divide_units(&self, other: &Unit) -> Unit {
        Unit {
            dimension: self.dimension.div(other.dimension),
            system: self.system,
            scale: self.scale / other.scale,
            offset: 0.0,
        }
    }
}

impl Default for Unit {
    fn default() -> Self {
        Self::dimensionless()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_multiply() {
        let vel = Dimension::LENGTH.mul(Dimension::TIME.pow(-1));
        assert_eq!(vel, Dimension::VELOCITY);
    }

    #[test]
    fn test_dimension_divide() {
        let accel = Dimension::VELOCITY.div(Dimension::TIME);
        assert_eq!(accel, Dimension::ACCELERATION);
    }

    #[test]
    fn test_dimension_energy() {
        // Energy = Force * Length
        let energy = Dimension::FORCE.mul(Dimension::LENGTH);
        assert_eq!(energy, Dimension::ENERGY);
    }

    #[test]
    fn test_dimension_momentum() {
        let momentum = Dimension::MASS.mul(Dimension::VELOCITY);
        assert_eq!(momentum, Dimension::MOMENTUM);
    }

    #[test]
    fn test_dimension_dimensionless() {
        let d = Dimension::LENGTH.div(Dimension::LENGTH);
        assert!(d.is_dimensionless());
    }

    #[test]
    fn test_dimension_display() {
        let d = Dimension::FORCE;
        let s = format!("{}", d);
        assert!(s.contains("L"));
        assert!(s.contains("M"));
        assert!(s.contains("T"));
    }

    #[test]
    fn test_cgs_to_si_length() {
        let factor = UnitSystem::CGS.to_si_factor(Dimension::LENGTH);
        assert!((factor - 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_cgs_to_si_mass() {
        let factor = UnitSystem::CGS.to_si_factor(Dimension::MASS);
        assert!((factor - 0.001).abs() < 1e-15);
    }

    #[test]
    fn test_cgs_to_si_force() {
        // 1 dyne = 1 g·cm/s² = 10⁻⁵ N
        let factor = UnitSystem::CGS.to_si_factor(Dimension::FORCE);
        assert!((factor - 1e-5).abs() < 1e-15);
    }

    #[test]
    fn test_unit_conversion_roundtrip() {
        let value = 100.0; // 100 cm
        let si_value = UnitSystem::convert(value, Dimension::LENGTH, UnitSystem::CGS, UnitSystem::SI);
        assert!((si_value - 1.0).abs() < 1e-12);
        let back = UnitSystem::convert(si_value, Dimension::LENGTH, UnitSystem::SI, UnitSystem::CGS);
        assert!((back - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_unit_to_si() {
        let meter = Unit::si_meter();
        assert!((meter.to_si(5.0) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_unit_compatible() {
        let meter = Unit::si_meter();
        let second = Unit::si_second();
        assert!(!meter.is_compatible(&second));
        assert!(meter.is_compatible(&meter));
    }

    #[test]
    fn test_unit_multiply() {
        let meter = Unit::si_meter();
        let second_inv = Unit::new(Dimension::TIME.pow(-1), UnitSystem::SI);
        let mps = meter.multiply_units(&second_inv);
        assert_eq!(mps.dimension, Dimension::VELOCITY);
    }
}
