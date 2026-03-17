//! Phase space declarations and structures for conservation analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Describes the geometric structure of a phase space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PhaseSpaceKind {
    /// Euclidean space R^n
    Euclidean(usize),
    /// Symplectic manifold of dimension 2n
    Symplectic(usize),
    /// Riemannian manifold with metric
    Riemannian {
        dimension: usize,
        metric: MetricTensor,
    },
    /// Product of two phase spaces
    Product(Box<PhaseSpaceKind>, Box<PhaseSpaceKind>),
    /// Cotangent bundle T*Q
    CotangentBundle(usize),
    /// Custom phase space with user-specified structure
    Custom {
        name: String,
        dimension: usize,
        properties: Vec<PhaseSpaceProperty>,
    },
}

impl PhaseSpaceKind {
    pub fn dimension(&self) -> usize {
        match self {
            PhaseSpaceKind::Euclidean(n) => *n,
            PhaseSpaceKind::Symplectic(n) => 2 * n,
            PhaseSpaceKind::Riemannian { dimension, .. } => *dimension,
            PhaseSpaceKind::Product(a, b) => a.dimension() + b.dimension(),
            PhaseSpaceKind::CotangentBundle(n) => 2 * n,
            PhaseSpaceKind::Custom { dimension, .. } => *dimension,
        }
    }

    pub fn is_symplectic(&self) -> bool {
        matches!(self, PhaseSpaceKind::Symplectic(_) | PhaseSpaceKind::CotangentBundle(_))
    }

    pub fn is_euclidean(&self) -> bool {
        matches!(self, PhaseSpaceKind::Euclidean(_))
    }

    pub fn has_metric(&self) -> bool {
        matches!(
            self,
            PhaseSpaceKind::Riemannian { .. } | PhaseSpaceKind::Euclidean(_)
        )
    }

    pub fn canonical_coordinates(&self) -> Vec<Coordinate> {
        let dim = self.dimension();
        let mut coords = Vec::with_capacity(dim);
        match self {
            PhaseSpaceKind::Symplectic(n) | PhaseSpaceKind::CotangentBundle(n) => {
                for i in 0..*n {
                    coords.push(Coordinate {
                        index: i,
                        name: format!("q_{}", i),
                        kind: CoordinateKind::Position,
                        conjugate: Some(i + n),
                    });
                }
                for i in 0..*n {
                    coords.push(Coordinate {
                        index: i + n,
                        name: format!("p_{}", i),
                        kind: CoordinateKind::Momentum,
                        conjugate: Some(i),
                    });
                }
            }
            _ => {
                for i in 0..dim {
                    coords.push(Coordinate {
                        index: i,
                        name: format!("x_{}", i),
                        kind: CoordinateKind::Generic,
                        conjugate: None,
                    });
                }
            }
        }
        coords
    }
}

impl fmt::Display for PhaseSpaceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseSpaceKind::Euclidean(n) => write!(f, "R^{}", n),
            PhaseSpaceKind::Symplectic(n) => write!(f, "Symp({})", 2 * n),
            PhaseSpaceKind::Riemannian { dimension, .. } => write!(f, "Riem({})", dimension),
            PhaseSpaceKind::Product(a, b) => write!(f, "{} × {}", a, b),
            PhaseSpaceKind::CotangentBundle(n) => write!(f, "T*R^{}", n),
            PhaseSpaceKind::Custom { name, dimension, .. } => {
                write!(f, "{}({})", name, dimension)
            }
        }
    }
}

/// A metric tensor for Riemannian phase spaces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricTensor {
    pub dimension: usize,
    pub components: Vec<Vec<f64>>,
}

impl MetricTensor {
    pub fn euclidean(n: usize) -> Self {
        let mut components = vec![vec![0.0; n]; n];
        for i in 0..n {
            components[i][i] = 1.0;
        }
        Self {
            dimension: n,
            components,
        }
    }

    pub fn diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut components = vec![vec![0.0; n]; n];
        for (i, &d) in diag.iter().enumerate() {
            components[i][i] = d;
        }
        Self {
            dimension: n,
            components,
        }
    }

    pub fn is_positive_definite(&self) -> bool {
        let n = self.dimension;
        let mut a = self.components.clone();
        for i in 0..n {
            for j in 0..i {
                let s: f64 = (0..j).map(|k| a[i][k] * a[j][k]).sum();
                a[i][j] = (a[i][j] - s) / a[j][j];
            }
            let s: f64 = (0..i).map(|k| a[i][k] * a[i][k]).sum();
            let diag = a[i][i] - s;
            if diag <= 0.0 {
                return false;
            }
            a[i][i] = diag.sqrt();
        }
        true
    }

    pub fn determinant(&self) -> f64 {
        let n = self.dimension;
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return self.components[0][0];
        }
        if n == 2 {
            return self.components[0][0] * self.components[1][1]
                - self.components[0][1] * self.components[1][0];
        }
        let mut mat = self.components.clone();
        let mut det = 1.0;
        for col in 0..n {
            let mut pivot_row = None;
            for row in col..n {
                if mat[row][col].abs() > 1e-15 {
                    pivot_row = Some(row);
                    break;
                }
            }
            let pivot_row = match pivot_row {
                Some(r) => r,
                None => return 0.0,
            };
            if pivot_row != col {
                mat.swap(pivot_row, col);
                det = -det;
            }
            det *= mat[col][col];
            let pivot = mat[col][col];
            for row in (col + 1)..n {
                let factor = mat[row][col] / pivot;
                for j in col..n {
                    let val = mat[col][j];
                    mat[row][j] -= factor * val;
                }
            }
        }
        det
    }

    pub fn inverse(&self) -> Option<MetricTensor> {
        let n = self.dimension;
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = self.components[i][j];
            }
            aug[i][i + n] = 1.0;
        }
        for col in 0..n {
            let mut pivot = None;
            let mut max_val = 0.0f64;
            for row in col..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    pivot = Some(row);
                }
            }
            let pivot = pivot?;
            if max_val < 1e-15 {
                return None;
            }
            aug.swap(pivot, col);
            let div = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= div;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    let v = aug[col][j];
                    aug[row][j] -= factor * v;
                }
            }
        }
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][j + n];
            }
        }
        Some(MetricTensor {
            dimension: n,
            components: inv,
        })
    }
}

/// Properties of a phase space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PhaseSpaceProperty {
    Conservative,
    Hamiltonian,
    Dissipative,
    Stochastic,
    Bounded,
    Periodic(Vec<usize>),
}

/// A coordinate in phase space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Coordinate {
    pub index: usize,
    pub name: String,
    pub kind: CoordinateKind,
    pub conjugate: Option<usize>,
}

/// Classification of coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CoordinateKind {
    Position,
    Momentum,
    Velocity,
    Angle,
    AngularMomentum,
    Generic,
    Time,
    Parameter,
}

impl fmt::Display for CoordinateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoordinateKind::Position => write!(f, "position"),
            CoordinateKind::Momentum => write!(f, "momentum"),
            CoordinateKind::Velocity => write!(f, "velocity"),
            CoordinateKind::Angle => write!(f, "angle"),
            CoordinateKind::AngularMomentum => write!(f, "angular_momentum"),
            CoordinateKind::Generic => write!(f, "generic"),
            CoordinateKind::Time => write!(f, "time"),
            CoordinateKind::Parameter => write!(f, "parameter"),
        }
    }
}

/// A complete phase space declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSpace {
    pub name: String,
    pub kind: PhaseSpaceKind,
    pub coordinates: Vec<Coordinate>,
    pub constraints: Vec<PhaseSpaceConstraint>,
    pub variables: HashMap<String, VariableBinding>,
}

impl PhaseSpace {
    pub fn new(name: impl Into<String>, kind: PhaseSpaceKind) -> Self {
        let coordinates = kind.canonical_coordinates();
        Self {
            name: name.into(),
            kind,
            coordinates,
            constraints: Vec::new(),
            variables: HashMap::new(),
        }
    }

    pub fn euclidean(name: impl Into<String>, dim: usize) -> Self {
        Self::new(name, PhaseSpaceKind::Euclidean(dim))
    }

    pub fn symplectic(name: impl Into<String>, n: usize) -> Self {
        Self::new(name, PhaseSpaceKind::Symplectic(n))
    }

    pub fn cotangent_bundle(name: impl Into<String>, n: usize) -> Self {
        Self::new(name, PhaseSpaceKind::CotangentBundle(n))
    }

    pub fn dimension(&self) -> usize {
        self.kind.dimension()
    }

    pub fn bind_variable(&mut self, var_name: impl Into<String>, coord_index: usize) {
        let name = var_name.into();
        self.variables.insert(
            name.clone(),
            VariableBinding {
                variable_name: name,
                coordinate_index: coord_index,
                is_array: false,
                array_size: None,
            },
        );
    }

    pub fn bind_array_variable(
        &mut self,
        var_name: impl Into<String>,
        start_coord: usize,
        size: usize,
    ) {
        let name = var_name.into();
        self.variables.insert(
            name.clone(),
            VariableBinding {
                variable_name: name,
                coordinate_index: start_coord,
                is_array: true,
                array_size: Some(size),
            },
        );
    }

    pub fn add_constraint(&mut self, constraint: PhaseSpaceConstraint) {
        self.constraints.push(constraint);
    }

    pub fn position_coordinates(&self) -> Vec<&Coordinate> {
        self.coordinates
            .iter()
            .filter(|c| c.kind == CoordinateKind::Position)
            .collect()
    }

    pub fn momentum_coordinates(&self) -> Vec<&Coordinate> {
        self.coordinates
            .iter()
            .filter(|c| c.kind == CoordinateKind::Momentum)
            .collect()
    }

    pub fn validate(&self) -> crate::Result<()> {
        if self.coordinates.len() != self.kind.dimension() {
            return Err(crate::ConservationError::PhaseSpace(
                crate::error::PhaseSpaceError::DimensionMismatch {
                    expected: self.kind.dimension(),
                    actual: self.coordinates.len(),
                },
            ));
        }
        for var in self.variables.values() {
            if var.coordinate_index >= self.kind.dimension() {
                return Err(crate::ConservationError::PhaseSpace(
                    crate::error::PhaseSpaceError::InvalidCoordinate {
                        index: var.coordinate_index,
                        dimension: self.kind.dimension(),
                    },
                ));
            }
        }
        Ok(())
    }
}

/// A constraint on the phase space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSpaceConstraint {
    pub name: String,
    pub kind: ConstraintKind,
    pub involved_coordinates: Vec<usize>,
}

/// Types of phase space constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// f(q) = 0 holonomic constraint
    Holonomic(String),
    /// g(q, dq/dt) = 0 non-holonomic constraint
    NonHolonomic(String),
    /// Inequality constraint
    Inequality { expression: String, bound: f64 },
    /// Conservation constraint
    Conserved { quantity: String, value: f64 },
}

/// Binding of a program variable to phase space coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBinding {
    pub variable_name: String,
    pub coordinate_index: usize,
    pub is_array: bool,
    pub array_size: Option<usize>,
}

/// Symplectic form for Hamiltonian systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticForm {
    pub dimension: usize,
    pub matrix: Vec<Vec<f64>>,
}

impl SymplecticForm {
    pub fn canonical(n: usize) -> Self {
        let dim = 2 * n;
        let mut matrix = vec![vec![0.0; dim]; dim];
        for i in 0..n {
            matrix[i][i + n] = 1.0;
            matrix[i + n][i] = -1.0;
        }
        Self {
            dimension: dim,
            matrix,
        }
    }

    pub fn apply(&self, v1: &[f64], v2: &[f64]) -> f64 {
        let mut result = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                result += self.matrix[i][j] * v1[i] * v2[j];
            }
        }
        result
    }

    pub fn is_canonical(&self) -> bool {
        let n = self.dimension / 2;
        if self.dimension % 2 != 0 {
            return false;
        }
        for i in 0..n {
            if (self.matrix[i][i + n] - 1.0).abs() > 1e-12 {
                return false;
            }
            if (self.matrix[i + n][i] + 1.0).abs() > 1e-12 {
                return false;
            }
        }
        true
    }

    pub fn poisson_bracket_matrix(&self) -> Vec<Vec<f64>> {
        self.matrix.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_phase_space() {
        let ps = PhaseSpace::euclidean("test", 3);
        assert_eq!(ps.dimension(), 3);
        assert!(ps.kind.is_euclidean());
        assert!(!ps.kind.is_symplectic());
    }

    #[test]
    fn test_symplectic_phase_space() {
        let ps = PhaseSpace::symplectic("hamiltonian", 3);
        assert_eq!(ps.dimension(), 6);
        assert!(ps.kind.is_symplectic());
        assert_eq!(ps.position_coordinates().len(), 3);
        assert_eq!(ps.momentum_coordinates().len(), 3);
    }

    #[test]
    fn test_metric_tensor_positive_definite() {
        let m = MetricTensor::euclidean(3);
        assert!(m.is_positive_definite());
        assert!((m.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_metric_tensor_inverse() {
        let m = MetricTensor::diagonal(&[2.0, 3.0, 4.0]);
        let inv = m.inverse().unwrap();
        assert!((inv.components[0][0] - 0.5).abs() < 1e-12);
        assert!((inv.components[1][1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_symplectic_form() {
        let omega = SymplecticForm::canonical(3);
        assert!(omega.is_canonical());
        assert_eq!(omega.dimension, 6);
    }

    #[test]
    fn test_variable_binding() {
        let mut ps = PhaseSpace::symplectic("test", 2);
        ps.bind_variable("x", 0);
        ps.bind_variable("y", 1);
        ps.bind_variable("px", 2);
        ps.bind_variable("py", 3);
        assert!(ps.validate().is_ok());
    }

    #[test]
    fn test_phase_space_display() {
        assert_eq!(format!("{}", PhaseSpaceKind::Euclidean(3)), "R^3");
        assert_eq!(format!("{}", PhaseSpaceKind::Symplectic(3)), "Symp(6)");
    }

    #[test]
    fn test_product_space() {
        let kind = PhaseSpaceKind::Product(
            Box::new(PhaseSpaceKind::Euclidean(3)),
            Box::new(PhaseSpaceKind::Symplectic(2)),
        );
        assert_eq!(kind.dimension(), 7);
    }
}
