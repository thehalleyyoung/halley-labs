pub mod jurisdiction;
pub mod obligation;
pub mod temporal;
pub mod cost;
pub mod strategy;
pub mod constraint;
pub mod regulatory;
pub mod formalizability;
pub mod certificate;
pub mod diagnosis;
pub mod config;
pub mod error;

pub use jurisdiction::*;
pub use obligation::*;
pub use temporal::*;
pub use cost::*;
pub use strategy::*;
pub use constraint::*;
pub use formalizability::*;
pub use config::*;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Jurisdiction identifier (e.g., "EU", "US-CA", "UK").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Jurisdiction(pub String);

impl Jurisdiction {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
    /// Returns true if `self` is a parent of `other` in the jurisdiction lattice.
    /// E.g. "EU" is parent of "EU-DE".
    pub fn is_parent_of(&self, other: &Jurisdiction) -> bool {
        other.0.starts_with(&self.0) && other.0.len() > self.0.len()
    }
}

impl fmt::Display for Jurisdiction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Temporal interval with optional start/end dates.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemporalInterval {
    pub start: Option<chrono::NaiveDate>,
    pub end: Option<chrono::NaiveDate>,
}

impl TemporalInterval {
    pub fn unbounded() -> Self {
        Self { start: None, end: None }
    }
    pub fn new(start: Option<chrono::NaiveDate>, end: Option<chrono::NaiveDate>) -> Self {
        Self { start, end }
    }
    pub fn overlaps(&self, other: &TemporalInterval) -> bool {
        let s1 = self.start.unwrap_or(chrono::NaiveDate::MIN);
        let e1 = self.end.unwrap_or(chrono::NaiveDate::MAX);
        let s2 = other.start.unwrap_or(chrono::NaiveDate::MIN);
        let e2 = other.end.unwrap_or(chrono::NaiveDate::MAX);
        s1 <= e2 && s2 <= e1
    }
    pub fn intersection(&self, other: &TemporalInterval) -> Option<TemporalInterval> {
        if !self.overlaps(other) {
            return None;
        }
        let start = match (self.start, other.start) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        let end = match (self.end, other.end) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Some(TemporalInterval { start, end })
    }
}

/// Obligation kind: obligation, permission, or prohibition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObligationKind {
    Obligation,
    Permission,
    Prohibition,
}

impl fmt::Display for ObligationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Obligation => write!(f, "OBL"),
            Self::Permission => write!(f, "PERM"),
            Self::Prohibition => write!(f, "PROH"),
        }
    }
}

/// Risk level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    Minimal,
    Limited,
    High,
    Unacceptable,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimal => write!(f, "minimal"),
            Self::Limited => write!(f, "limited"),
            Self::High => write!(f, "high"),
            Self::Unacceptable => write!(f, "unacceptable"),
        }
    }
}

/// Formalizability grade from F1 (fully formalizable) to F5 (not formalizable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FormalizabilityGrade {
    F1,
    F2,
    F3,
    F4,
    F5,
}

impl FormalizabilityGrade {
    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::F1),
            2 => Some(Self::F2),
            3 => Some(Self::F3),
            4 => Some(Self::F4),
            5 => Some(Self::F5),
            _ => None,
        }
    }
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::F1 => 1,
            Self::F2 => 2,
            Self::F3 => 3,
            Self::F4 => 4,
            Self::F5 => 5,
        }
    }
    /// Compose two grades: take the worse (higher number).
    pub fn compose(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
}

impl fmt::Display for FormalizabilityGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F{}", self.as_u8())
    }
}

/// Domain of application.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Domain(pub String);

impl Domain {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Composition operators for obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositionOp {
    /// Conjunction (⊗): both must hold
    Conjunction,
    /// Disjunction (⊕): at least one must hold
    Disjunction,
    /// Override (▷): left overrides right where applicable
    Override,
    /// Exception (⊘): left minus right
    Exception,
}

impl fmt::Display for CompositionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conjunction => write!(f, "⊗"),
            Self::Disjunction => write!(f, "⊕"),
            Self::Override => write!(f, "▷"),
            Self::Exception => write!(f, "⊘"),
        }
    }
}

/// A cost value with currency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    pub amount: f64,
    pub currency: String,
}

/// An article reference (e.g., "EU-AI-Act Art. 6").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArticleRef {
    pub framework: String,
    pub article: String,
    pub paragraph: Option<String>,
}

impl fmt::Display for ArticleRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Art. {}", self.framework, self.article)?;
        if let Some(p) = &self.paragraph {
            write!(f, "({})", p)?;
        }
        Ok(())
    }
}

/// Unique identifier (wraps uuid).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub uuid::Uuid);

impl Id {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jurisdiction_parent() {
        let eu = Jurisdiction::new("EU");
        let eu_de = Jurisdiction::new("EU-DE");
        assert!(eu.is_parent_of(&eu_de));
        assert!(!eu_de.is_parent_of(&eu));
    }

    #[test]
    fn test_temporal_overlap() {
        let a = TemporalInterval::new(
            Some(chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
            Some(chrono::NaiveDate::from_ymd_opt(2024, 6, 30).unwrap()),
        );
        let b = TemporalInterval::new(
            Some(chrono::NaiveDate::from_ymd_opt(2024, 3, 1).unwrap()),
            Some(chrono::NaiveDate::from_ymd_opt(2024, 12, 31).unwrap()),
        );
        assert!(a.overlaps(&b));
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.start, Some(chrono::NaiveDate::from_ymd_opt(2024, 3, 1).unwrap()));
        assert_eq!(inter.end, Some(chrono::NaiveDate::from_ymd_opt(2024, 6, 30).unwrap()));
    }

    #[test]
    fn test_formalizability_compose() {
        assert_eq!(FormalizabilityGrade::F1.compose(FormalizabilityGrade::F3), FormalizabilityGrade::F3);
        assert_eq!(FormalizabilityGrade::F5.compose(FormalizabilityGrade::F2), FormalizabilityGrade::F5);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Minimal < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Unacceptable);
    }
}
