// regsynth-encoding: Constraint encoding engine
// Translates regulatory obligations into SMT/ILP constraint formats

pub mod smt_ast;
pub mod ilp_model;
pub mod obligation_encoder;
pub mod ilp_encoder;
pub mod temporal_unroller;
pub mod hard_soft_classifier;
pub mod provenance;
pub mod simplifier;

pub use smt_ast::*;
pub use obligation_encoder::{ObligationEncoder, RawObligation, ObligationKind as RawObligationKind, EncodedObligation, EncodedObligationSet};
pub use ilp_encoder::IlpEncoder;
pub use temporal_unroller::{TemporalUnroller, TemporalObligationSet, TemporalConstraintSet};
pub use hard_soft_classifier::{HardSoftClassifier, ConstraintClassification};
pub use provenance::ProvenanceMap;
pub use simplifier::ConstraintSimplifier;

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SMT Sorts ──────────────────────────────────────────────────────────────

/// SMT sort (type) for variables and expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    Real,
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => write!(f, "Bool"),
            Self::Int => write!(f, "Int"),
            Self::Real => write!(f, "Real"),
        }
    }
}

// ─── SMT Expressions ────────────────────────────────────────────────────────

/// SMT expression (term) tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtExpr {
    Var(String, SmtSort),
    BoolLit(bool),
    IntLit(i64),
    RealLit(f64),
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    Ge(Box<SmtExpr>, Box<SmtExpr>),
    Add(Vec<SmtExpr>),
    Sub(Box<SmtExpr>, Box<SmtExpr>),
    Mul(Vec<SmtExpr>),
    Neg(Box<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Apply(String, Vec<SmtExpr>),
}

// ─── SMT Constraint ─────────────────────────────────────────────────────────

/// Provenance: traces a constraint back to its regulatory source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub obligation_id: String,
    pub jurisdiction: String,
    pub article_ref: Option<String>,
    pub description: String,
}

/// A named SMT constraint with optional provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmtConstraint {
    pub id: String,
    pub expr: SmtExpr,
    pub provenance: Option<Provenance>,
}

// ─── ILP Types ──────────────────────────────────────────────────────────────

/// An ILP variable with bounds and integrality constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IlpVariable {
    pub name: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub is_integer: bool,
    pub is_binary: bool,
}

/// Type of ILP constraint (inequality direction or equality).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IlpConstraintType {
    Le, // <=
    Ge, // >=
    Eq, // ==
}

impl fmt::Display for IlpConstraintType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Le => write!(f, "<="),
            Self::Ge => write!(f, ">="),
            Self::Eq => write!(f, "="),
        }
    }
}

/// A single ILP constraint: sum(coeff_i * var_i) <op> rhs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IlpConstraint {
    pub id: String,
    pub coefficients: Vec<(String, f64)>,
    pub constraint_type: IlpConstraintType,
    pub rhs: f64,
    pub provenance: Option<Provenance>,
}

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

/// ILP objective function: sense * (sum(coeff_i * var_i) + constant).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IlpObjective {
    pub sense: ObjectiveSense,
    pub coefficients: Vec<(String, f64)>,
    pub constant: f64,
}

/// Complete ILP model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IlpModel {
    pub variables: Vec<IlpVariable>,
    pub constraints: Vec<IlpConstraint>,
    pub objective: IlpObjective,
}

// ─── Encoded Problem ────────────────────────────────────────────────────────

/// A fully encoded regulatory compliance problem ready for solving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedProblem {
    pub smt_constraints: Vec<SmtConstraint>,
    pub ilp_model: Option<IlpModel>,
    pub soft_constraints: Vec<(SmtConstraint, f64)>,
    pub objectives: Vec<IlpObjective>,
}

impl Default for EncodedProblem {
    fn default() -> Self {
        Self {
            smt_constraints: Vec::new(),
            ilp_model: None,
            soft_constraints: Vec::new(),
            objectives: Vec::new(),
        }
    }
}
