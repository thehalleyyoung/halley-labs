use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConservationKind {
    Energy,
    Momentum,
    AngularMomentum,
    Mass,
    Charge,
    Symplectic,
    Vorticity,
    Custom,
}

impl std::fmt::Display for ConservationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConservationKind::Energy => write!(f, "Energy"),
            ConservationKind::Momentum => write!(f, "Momentum"),
            ConservationKind::AngularMomentum => write!(f, "AngularMomentum"),
            ConservationKind::Mass => write!(f, "Mass"),
            ConservationKind::Charge => write!(f, "Charge"),
            ConservationKind::Symplectic => write!(f, "Symplectic"),
            ConservationKind::Vorticity => write!(f, "Vorticity"),
            ConservationKind::Custom => write!(f, "Custom"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservedQuantity {
    pub kind: ConservationKind,
    pub value: f64,
    pub name: String,
}

impl ConservedQuantity {
    pub fn new(kind: ConservationKind, value: f64) -> Self {
        let name = format!("{}", kind);
        Self {
            kind,
            value,
            name,
        }
    }

    /// Create a scalar conserved quantity (energy-like).
    pub fn scalar(value: f64) -> Self {
        Self {
            kind: ConservationKind::Energy,
            value,
            name: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationLaw {
    pub kind: ConservationKind,
    pub name: String,
    pub description: String,
}

impl ConservationLaw {
    pub fn new(kind: ConservationKind, name: &str) -> Self {
        Self {
            kind,
            name: name.to_string(),
            description: String::new(),
        }
    }

    pub fn energy() -> Self {
        Self::new(ConservationKind::Energy, "Total Energy")
    }

    pub fn momentum() -> Self {
        Self::new(ConservationKind::Momentum, "Total Momentum")
    }

    pub fn angular_momentum() -> Self {
        Self::new(ConservationKind::AngularMomentum, "Total Angular Momentum")
    }
}
