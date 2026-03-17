//! Conservation law types and representations.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A conservation law detected or declared for a system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationLaw {
    pub id: String,
    pub name: String,
    pub kind: ConservationKind,
    pub quantity: ConservedQuantity,
    pub tolerance: f64,
    pub status: ConservationStatus,
    pub source: ConservationSource,
    pub associated_symmetry: Option<String>,
}

impl ConservationLaw {
    pub fn new(
        name: impl Into<String>,
        kind: ConservationKind,
        quantity: ConservedQuantity,
    ) -> Self {
        let name = name.into();
        let id = format!("cl_{}", name.to_lowercase().replace(' ', "_"));
        Self {
            id,
            name,
            kind,
            quantity,
            tolerance: 1e-10,
            status: ConservationStatus::Unknown,
            source: ConservationSource::Declared,
            associated_symmetry: None,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_symmetry(mut self, sym: impl Into<String>) -> Self {
        self.associated_symmetry = Some(sym.into());
        self
    }

    pub fn energy(expression: impl Into<String>) -> Self {
        Self::new(
            "Total Energy",
            ConservationKind::Energy,
            ConservedQuantity::scalar(expression),
        )
        .with_symmetry("time_translation")
    }

    pub fn linear_momentum(component: usize) -> Self {
        let names = ["x", "y", "z"];
        let comp_name = names.get(component).unwrap_or(&"?");
        Self::new(
            format!("Linear Momentum ({})", comp_name),
            ConservationKind::Momentum,
            ConservedQuantity::vector_component(
                format!("sum(p_{})", comp_name),
                component,
            ),
        )
        .with_symmetry(format!("translation_{}", comp_name))
    }

    pub fn angular_momentum(axis: usize) -> Self {
        let names = ["x", "y", "z"];
        let axis_name = names.get(axis).unwrap_or(&"?");
        Self::new(
            format!("Angular Momentum ({})", axis_name),
            ConservationKind::AngularMomentum,
            ConservedQuantity::vector_component(
                format!("sum(r x p)_{}", axis_name),
                axis,
            ),
        )
        .with_symmetry(format!("rotation_{}", axis_name))
    }

    pub fn mass() -> Self {
        Self::new(
            "Total Mass",
            ConservationKind::Mass,
            ConservedQuantity::scalar("sum(m_i)"),
        )
    }

    pub fn charge() -> Self {
        Self::new(
            "Total Charge",
            ConservationKind::Charge,
            ConservedQuantity::scalar("sum(q_i)"),
        )
    }

    pub fn is_violated(&self) -> bool {
        matches!(self.status, ConservationStatus::Violated { .. })
    }

    pub fn is_preserved(&self) -> bool {
        matches!(self.status, ConservationStatus::Preserved { .. })
    }

    pub fn violation_magnitude(&self) -> Option<f64> {
        match &self.status {
            ConservationStatus::Violated { magnitude, .. } => Some(*magnitude),
            _ => None,
        }
    }
}

impl fmt::Display for ConservationLaw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.name, self.kind)?;
        match &self.status {
            ConservationStatus::Preserved { drift_rate } => {
                write!(f, " ✓ (drift: {:.2e})", drift_rate)?;
            }
            ConservationStatus::Violated { magnitude, order, .. } => {
                write!(f, " ✗ (violation: {:.2e}, O(h^{}))", magnitude, order)?;
            }
            ConservationStatus::Unknown => {
                write!(f, " ?")?;
            }
            ConservationStatus::NotApplicable(reason) => {
                write!(f, " N/A ({})", reason)?;
            }
        }
        Ok(())
    }
}

/// Classification of conservation laws.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConservationKind {
    Energy,
    Momentum,
    AngularMomentum,
    Mass,
    Charge,
    Vorticity,
    Enstrophy,
    Helicity,
    Casimir,
    Symplectic,
    Custom,
}

impl fmt::Display for ConservationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConservationKind::Energy => write!(f, "energy"),
            ConservationKind::Momentum => write!(f, "momentum"),
            ConservationKind::AngularMomentum => write!(f, "angular_momentum"),
            ConservationKind::Mass => write!(f, "mass"),
            ConservationKind::Charge => write!(f, "charge"),
            ConservationKind::Vorticity => write!(f, "vorticity"),
            ConservationKind::Enstrophy => write!(f, "enstrophy"),
            ConservationKind::Helicity => write!(f, "helicity"),
            ConservationKind::Casimir => write!(f, "casimir"),
            ConservationKind::Symplectic => write!(f, "symplectic"),
            ConservationKind::Custom => write!(f, "custom"),
        }
    }
}

/// The status of a conservation law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConservationStatus {
    /// Conservation law is preserved within tolerance.
    Preserved { drift_rate: f64 },
    /// Conservation law is violated.
    Violated {
        magnitude: f64,
        order: usize,
        source_lines: Vec<String>,
    },
    /// Status has not been determined.
    Unknown,
    /// Conservation law is not applicable to this system.
    NotApplicable(String),
}

/// How the conservation law was identified.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConservationSource {
    /// User-declared annotation.
    Declared,
    /// Detected via Noether's theorem from symmetry.
    NoetherDetected { symmetry_id: String },
    /// Detected numerically from trajectory data.
    NumericallyDetected { confidence: f64 },
    /// Known from the physical domain.
    DomainKnowledge { domain: String },
}

/// Representation of a conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservedQuantity {
    pub expression: String,
    pub quantity_type: QuantityType,
    pub components: Vec<QuantityComponent>,
}

impl ConservedQuantity {
    pub fn scalar(expr: impl Into<String>) -> Self {
        Self {
            expression: expr.into(),
            quantity_type: QuantityType::Scalar,
            components: vec![QuantityComponent {
                index: 0,
                expression: String::new(),
                weight: 1.0,
            }],
        }
    }

    pub fn vector_component(expr: impl Into<String>, component: usize) -> Self {
        Self {
            expression: expr.into(),
            quantity_type: QuantityType::VectorComponent(component),
            components: vec![QuantityComponent {
                index: component,
                expression: String::new(),
                weight: 1.0,
            }],
        }
    }

    pub fn tensor(expr: impl Into<String>, rank: usize) -> Self {
        Self {
            expression: expr.into(),
            quantity_type: QuantityType::Tensor(rank),
            components: Vec::new(),
        }
    }
}

/// Type of conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantityType {
    Scalar,
    VectorComponent(usize),
    Vector(usize),
    Tensor(usize),
    Density,
    Flux,
}

/// A component of a conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantityComponent {
    pub index: usize,
    pub expression: String,
    pub weight: f64,
}

/// A time series of conservation law values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationTimeSeries {
    pub law_id: String,
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub initial_value: f64,
    pub relative_errors: Vec<f64>,
}

impl ConservationTimeSeries {
    pub fn new(law_id: impl Into<String>, times: Vec<f64>, values: Vec<f64>) -> Self {
        let initial_value = values.first().copied().unwrap_or(0.0);
        let relative_errors: Vec<f64> = values
            .iter()
            .map(|v| {
                if initial_value.abs() > 1e-15 {
                    ((v - initial_value) / initial_value).abs()
                } else {
                    (v - initial_value).abs()
                }
            })
            .collect();
        Self {
            law_id: law_id.into(),
            times,
            values,
            initial_value,
            relative_errors,
        }
    }

    pub fn max_relative_error(&self) -> f64 {
        self.relative_errors
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
    }

    pub fn mean_relative_error(&self) -> f64 {
        if self.relative_errors.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.relative_errors.iter().sum();
        sum / self.relative_errors.len() as f64
    }

    pub fn drift_rate(&self) -> f64 {
        if self.times.len() < 2 {
            return 0.0;
        }
        let n = self.times.len();
        let dt = self.times[n - 1] - self.times[0];
        if dt.abs() < 1e-15 {
            return 0.0;
        }
        let total_change = (self.values[n - 1] - self.values[0]).abs();
        total_change / dt
    }

    pub fn estimate_drift_order(&self, dt: f64) -> f64 {
        if self.relative_errors.len() < 10 || dt <= 0.0 {
            return 0.0;
        }
        let max_err = self.max_relative_error();
        if max_err < 1e-15 {
            return f64::INFINITY;
        }
        -(max_err.ln()) / dt.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_conservation_law() {
        let law = ConservationLaw::energy("0.5*m*v^2 + V(x)");
        assert_eq!(law.kind, ConservationKind::Energy);
        assert!(law.associated_symmetry.is_some());
    }

    #[test]
    fn test_momentum_conservation_law() {
        let law = ConservationLaw::linear_momentum(0);
        assert_eq!(law.kind, ConservationKind::Momentum);
        assert!(law.name.contains("x"));
    }

    #[test]
    fn test_angular_momentum() {
        let law = ConservationLaw::angular_momentum(2);
        assert_eq!(law.kind, ConservationKind::AngularMomentum);
        assert!(law.name.contains("z"));
    }

    #[test]
    fn test_conservation_status() {
        let mut law = ConservationLaw::energy("H");
        assert!(!law.is_violated());
        assert!(!law.is_preserved());

        law.status = ConservationStatus::Violated {
            magnitude: 1e-3,
            order: 2,
            source_lines: vec!["line 42".to_string()],
        };
        assert!(law.is_violated());
        assert_eq!(law.violation_magnitude(), Some(1e-3));
    }

    #[test]
    fn test_time_series() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = vec![1.0, 1.001, 1.002, 1.003, 1.004];
        let ts = ConservationTimeSeries::new("energy", times, values);
        assert!((ts.drift_rate() - 0.001).abs() < 1e-10);
        assert!(ts.max_relative_error() < 0.005);
    }

    #[test]
    fn test_conservation_display() {
        let mut law = ConservationLaw::energy("H");
        law.status = ConservationStatus::Preserved { drift_rate: 1e-12 };
        let s = format!("{}", law);
        assert!(s.contains("✓"));
    }
}
