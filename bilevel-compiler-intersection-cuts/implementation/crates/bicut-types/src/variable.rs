//! Variable types, identifiers, metadata, and scoping for bilevel programs.
//!
//! Variables are the fundamental building blocks of a bilevel optimization
//! problem. Each variable has a type (continuous, integer, binary), belongs
//! to a scope (leader, follower, or shared), and carries optional bounds
//! and metadata.

use std::fmt;

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::error::{BicutError, BicutResult, ValidationError};

// ── Variable identifier ────────────────────────────────────────────

/// A unique, lightweight identifier for a variable within a problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VariableId(pub usize);

impl VariableId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

impl fmt::Display for VariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl From<usize> for VariableId {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

// ── Variable type ──────────────────────────────────────────────────

/// The domain of a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableType {
    /// Real-valued variable (default).
    Continuous,
    /// Integer-valued variable.
    Integer,
    /// Binary variable (0 or 1).
    Binary,
    /// Semi-continuous: 0 or in [lb, ub].
    SemiContinuous,
}

impl VariableType {
    /// Whether this type implies integrality.
    pub fn is_integer_type(&self) -> bool {
        matches!(self, VariableType::Integer | VariableType::Binary)
    }

    /// Whether this type is continuous.
    pub fn is_continuous(&self) -> bool {
        matches!(
            self,
            VariableType::Continuous | VariableType::SemiContinuous
        )
    }

    /// Natural lower bound for this type (0 for binary, -inf otherwise).
    pub fn natural_lb(&self) -> f64 {
        match self {
            VariableType::Binary => 0.0,
            VariableType::SemiContinuous => 0.0,
            _ => f64::NEG_INFINITY,
        }
    }

    /// Natural upper bound for this type (1 for binary, +inf otherwise).
    pub fn natural_ub(&self) -> f64 {
        match self {
            VariableType::Binary => 1.0,
            _ => f64::INFINITY,
        }
    }
}

impl Default for VariableType {
    fn default() -> Self {
        VariableType::Continuous
    }
}

impl fmt::Display for VariableType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariableType::Continuous => write!(f, "continuous"),
            VariableType::Integer => write!(f, "integer"),
            VariableType::Binary => write!(f, "binary"),
            VariableType::SemiContinuous => write!(f, "semi-continuous"),
        }
    }
}

// ── Variable scope ─────────────────────────────────────────────────

/// Which level of the bilevel hierarchy a variable belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableScope {
    /// Upper-level (leader) variable.
    Leader,
    /// Lower-level (follower) variable.
    Follower,
    /// Shared / coupling variable appearing in both levels.
    Shared,
}

impl VariableScope {
    /// Whether this variable appears in the upper-level problem.
    pub fn is_upper(&self) -> bool {
        matches!(self, VariableScope::Leader | VariableScope::Shared)
    }

    /// Whether this variable appears in the lower-level problem.
    pub fn is_lower(&self) -> bool {
        matches!(self, VariableScope::Follower | VariableScope::Shared)
    }
}

impl fmt::Display for VariableScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariableScope::Leader => write!(f, "leader"),
            VariableScope::Follower => write!(f, "follower"),
            VariableScope::Shared => write!(f, "shared"),
        }
    }
}

// ── Variable info ──────────────────────────────────────────────────

/// Complete information about a single variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableInfo {
    pub id: VariableId,
    pub name: String,
    pub var_type: VariableType,
    pub scope: VariableScope,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Optional objective coefficient (for quick access).
    pub obj_coeff: f64,
    /// Whether this variable has been fixed to a specific value.
    pub fixed_value: Option<f64>,
}

impl VariableInfo {
    pub fn new(
        id: VariableId,
        name: impl Into<String>,
        var_type: VariableType,
        scope: VariableScope,
    ) -> Self {
        let lb = var_type.natural_lb();
        let ub = var_type.natural_ub();
        Self {
            id,
            name: name.into(),
            var_type,
            scope,
            lower_bound: lb,
            upper_bound: ub,
            obj_coeff: 0.0,
            fixed_value: None,
        }
    }

    pub fn with_bounds(mut self, lb: f64, ub: f64) -> Self {
        self.lower_bound = lb;
        self.upper_bound = ub;
        self
    }

    pub fn with_obj_coeff(mut self, coeff: f64) -> Self {
        self.obj_coeff = coeff;
        self
    }

    pub fn fix_to(mut self, value: f64) -> Self {
        self.fixed_value = Some(value);
        self
    }

    /// Effective lower bound taking into account fixed value.
    pub fn effective_lb(&self) -> f64 {
        self.fixed_value.unwrap_or(self.lower_bound)
    }

    /// Effective upper bound taking into account fixed value.
    pub fn effective_ub(&self) -> f64 {
        self.fixed_value.unwrap_or(self.upper_bound)
    }

    /// Whether the variable is free (unbounded in both directions).
    pub fn is_free(&self) -> bool {
        self.fixed_value.is_none()
            && self.lower_bound == f64::NEG_INFINITY
            && self.upper_bound == f64::INFINITY
    }

    /// Whether this variable has finite bounds on both sides.
    pub fn is_bounded(&self) -> bool {
        self.lower_bound.is_finite() && self.upper_bound.is_finite()
    }

    /// The range of the variable (ub - lb). Returns +inf if unbounded.
    pub fn range(&self) -> f64 {
        if self.is_bounded() {
            self.upper_bound - self.lower_bound
        } else {
            f64::INFINITY
        }
    }

    /// Validate the variable info for consistency.
    pub fn validate(&self) -> BicutResult<()> {
        if self.lower_bound > self.upper_bound {
            return Err(BicutError::Validation(ValidationError::InvalidBounds {
                name: self.name.clone(),
                lb: self.lower_bound,
                ub: self.upper_bound,
            }));
        }
        if let Some(val) = self.fixed_value {
            if val < self.lower_bound - 1e-10 || val > self.upper_bound + 1e-10 {
                return Err(BicutError::Validation(ValidationError::InvalidBounds {
                    name: self.name.clone(),
                    lb: self.lower_bound,
                    ub: self.upper_bound,
                }));
            }
        }
        if self.var_type == VariableType::Binary {
            if self.lower_bound < -1e-10 || self.upper_bound > 1.0 + 1e-10 {
                return Err(BicutError::Validation(ValidationError::InvalidBounds {
                    name: self.name.clone(),
                    lb: self.lower_bound,
                    ub: self.upper_bound,
                }));
            }
        }
        if self.name.is_empty() {
            return Err(BicutError::Validation(ValidationError::DuplicateName {
                name: "(empty)".into(),
                context: "variable name".into(),
            }));
        }
        Ok(())
    }

    /// Check if a value is feasible for this variable given its type and bounds.
    pub fn is_feasible(&self, value: f64, tolerance: f64) -> bool {
        if value < self.lower_bound - tolerance || value > self.upper_bound + tolerance {
            return false;
        }
        match self.var_type {
            VariableType::Binary => {
                (value - 0.0).abs() <= tolerance || (value - 1.0).abs() <= tolerance
            }
            VariableType::Integer => (value - value.round()).abs() <= tolerance,
            VariableType::Continuous => true,
            VariableType::SemiContinuous => {
                value.abs() <= tolerance
                    || (value >= self.lower_bound - tolerance
                        && value <= self.upper_bound + tolerance)
            }
        }
    }
}

impl fmt::Display for VariableInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {}, [{}, {}])",
            self.name, self.var_type, self.scope, self.lower_bound, self.upper_bound
        )?;
        if let Some(v) = self.fixed_value {
            write!(f, " = {}", v)?;
        }
        Ok(())
    }
}

// ── Variable set ───────────────────────────────────────────────────

/// An ordered collection of variables with fast lookup by id and name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableSet {
    /// Variables stored in insertion order (index = id.0).
    variables: Vec<VariableInfo>,
    /// Name → id lookup.
    name_index: IndexMap<String, VariableId>,
}

impl VariableSet {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            name_index: IndexMap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            variables: Vec::with_capacity(cap),
            name_index: IndexMap::with_capacity(cap),
        }
    }

    /// Add a variable. The variable's id is overwritten to match its position.
    pub fn add(&mut self, mut info: VariableInfo) -> BicutResult<VariableId> {
        if self.name_index.contains_key(&info.name) {
            return Err(BicutError::Validation(ValidationError::DuplicateName {
                name: info.name.clone(),
                context: "variable set".into(),
            }));
        }
        let id = VariableId(self.variables.len());
        info.id = id;
        self.name_index.insert(info.name.clone(), id);
        self.variables.push(info);
        Ok(id)
    }

    /// Create and add a simple continuous variable.
    pub fn add_continuous(
        &mut self,
        name: impl Into<String>,
        scope: VariableScope,
        lb: f64,
        ub: f64,
    ) -> BicutResult<VariableId> {
        let name = name.into();
        let info = VariableInfo::new(VariableId(0), &name, VariableType::Continuous, scope)
            .with_bounds(lb, ub);
        self.add(info)
    }

    /// Create and add a binary variable.
    pub fn add_binary(
        &mut self,
        name: impl Into<String>,
        scope: VariableScope,
    ) -> BicutResult<VariableId> {
        let name = name.into();
        let info = VariableInfo::new(VariableId(0), &name, VariableType::Binary, scope);
        self.add(info)
    }

    /// Create and add an integer variable.
    pub fn add_integer(
        &mut self,
        name: impl Into<String>,
        scope: VariableScope,
        lb: f64,
        ub: f64,
    ) -> BicutResult<VariableId> {
        let name = name.into();
        let info = VariableInfo::new(VariableId(0), &name, VariableType::Integer, scope)
            .with_bounds(lb, ub);
        self.add(info)
    }

    pub fn len(&self) -> usize {
        self.variables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Get variable info by id.
    pub fn get(&self, id: VariableId) -> Option<&VariableInfo> {
        self.variables.get(id.0)
    }

    /// Get mutable variable info by id.
    pub fn get_mut(&mut self, id: VariableId) -> Option<&mut VariableInfo> {
        self.variables.get_mut(id.0)
    }

    /// Look up a variable by name.
    pub fn get_by_name(&self, name: &str) -> Option<&VariableInfo> {
        self.name_index.get(name).and_then(|id| self.get(*id))
    }

    /// Look up a variable id by name.
    pub fn id_of(&self, name: &str) -> Option<VariableId> {
        self.name_index.get(name).copied()
    }

    /// Iterate over all variables in order.
    pub fn iter(&self) -> impl Iterator<Item = &VariableInfo> {
        self.variables.iter()
    }

    /// Iterate with ids.
    pub fn iter_with_ids(&self) -> impl Iterator<Item = (VariableId, &VariableInfo)> {
        self.variables
            .iter()
            .enumerate()
            .map(|(i, v)| (VariableId(i), v))
    }

    /// All variable ids.
    pub fn ids(&self) -> impl Iterator<Item = VariableId> + '_ {
        (0..self.variables.len()).map(VariableId)
    }

    /// Filter variables by scope.
    pub fn filter_by_scope(&self, scope: VariableScope) -> Vec<&VariableInfo> {
        self.variables.iter().filter(|v| v.scope == scope).collect()
    }

    /// Filter variables by type.
    pub fn filter_by_type(&self, vtype: VariableType) -> Vec<&VariableInfo> {
        self.variables
            .iter()
            .filter(|v| v.var_type == vtype)
            .collect()
    }

    /// Get all leader variables.
    pub fn leader_vars(&self) -> Vec<&VariableInfo> {
        self.variables
            .iter()
            .filter(|v| v.scope.is_upper())
            .collect()
    }

    /// Get all follower variables.
    pub fn follower_vars(&self) -> Vec<&VariableInfo> {
        self.variables
            .iter()
            .filter(|v| v.scope.is_lower())
            .collect()
    }

    /// Get ids of leader variables.
    pub fn leader_ids(&self) -> Vec<VariableId> {
        self.variables
            .iter()
            .filter(|v| v.scope == VariableScope::Leader)
            .map(|v| v.id)
            .collect()
    }

    /// Get ids of follower variables.
    pub fn follower_ids(&self) -> Vec<VariableId> {
        self.variables
            .iter()
            .filter(|v| v.scope == VariableScope::Follower)
            .map(|v| v.id)
            .collect()
    }

    /// Number of leader-only variables.
    pub fn num_leader(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.scope == VariableScope::Leader)
            .count()
    }

    /// Number of follower-only variables.
    pub fn num_follower(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.scope == VariableScope::Follower)
            .count()
    }

    /// Number of shared variables.
    pub fn num_shared(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.scope == VariableScope::Shared)
            .count()
    }

    /// Whether any variable is integer or binary.
    pub fn has_integer_variables(&self) -> bool {
        self.variables.iter().any(|v| v.var_type.is_integer_type())
    }

    /// All lower bounds as a vector (ordered by id).
    pub fn lower_bounds(&self) -> Vec<f64> {
        self.variables.iter().map(|v| v.lower_bound).collect()
    }

    /// All upper bounds as a vector (ordered by id).
    pub fn upper_bounds(&self) -> Vec<f64> {
        self.variables.iter().map(|v| v.upper_bound).collect()
    }

    /// Validate all variables in the set.
    pub fn validate(&self) -> BicutResult<()> {
        for var in &self.variables {
            var.validate()?;
        }
        Ok(())
    }

    /// Build a sub-set containing only the given ids.
    pub fn subset(&self, ids: &[VariableId]) -> VariableSet {
        let mut sub = VariableSet::with_capacity(ids.len());
        for &id in ids {
            if let Some(info) = self.get(id) {
                let _ = sub.add(info.clone());
            }
        }
        sub
    }

    /// Return a compact representation of bounds for hashing.
    pub fn bounds_fingerprint(&self) -> Vec<(OrderedFloat<f64>, OrderedFloat<f64>)> {
        self.variables
            .iter()
            .map(|v| (OrderedFloat(v.lower_bound), OrderedFloat(v.upper_bound)))
            .collect()
    }
}

impl Default for VariableSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VariableSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VariableSet ({} variables):", self.len())?;
        for var in &self.variables {
            writeln!(f, "  {}", var)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_id_display() {
        let id = VariableId::new(42);
        assert_eq!(id.to_string(), "v42");
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn test_variable_type_defaults() {
        let vt = VariableType::default();
        assert_eq!(vt, VariableType::Continuous);
        assert!(!vt.is_integer_type());
        assert!(vt.is_continuous());
    }

    #[test]
    fn test_binary_natural_bounds() {
        let bt = VariableType::Binary;
        assert_eq!(bt.natural_lb(), 0.0);
        assert_eq!(bt.natural_ub(), 1.0);
        assert!(bt.is_integer_type());
    }

    #[test]
    fn test_variable_info_construction() {
        let info = VariableInfo::new(
            VariableId(0),
            "x1",
            VariableType::Continuous,
            VariableScope::Leader,
        )
        .with_bounds(-10.0, 10.0)
        .with_obj_coeff(3.5);
        assert_eq!(info.name, "x1");
        assert_eq!(info.lower_bound, -10.0);
        assert_eq!(info.obj_coeff, 3.5);
        assert!(info.is_bounded());
        assert!(!info.is_free());
        assert!((info.range() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_variable_info_validation() {
        let bad = VariableInfo::new(
            VariableId(0),
            "x",
            VariableType::Continuous,
            VariableScope::Leader,
        )
        .with_bounds(10.0, 5.0);
        assert!(bad.validate().is_err());

        let good = VariableInfo::new(
            VariableId(0),
            "x",
            VariableType::Binary,
            VariableScope::Follower,
        );
        assert!(good.validate().is_ok());
    }

    #[test]
    fn test_variable_feasibility() {
        let bin = VariableInfo::new(
            VariableId(0),
            "b",
            VariableType::Binary,
            VariableScope::Leader,
        );
        assert!(bin.is_feasible(0.0, 1e-6));
        assert!(bin.is_feasible(1.0, 1e-6));
        assert!(!bin.is_feasible(0.5, 1e-6));

        let int = VariableInfo::new(
            VariableId(0),
            "i",
            VariableType::Integer,
            VariableScope::Follower,
        )
        .with_bounds(0.0, 10.0);
        assert!(int.is_feasible(3.0, 1e-6));
        assert!(!int.is_feasible(3.5, 1e-6));
    }

    #[test]
    fn test_variable_set_add_and_lookup() {
        let mut vs = VariableSet::new();
        let id = vs
            .add_continuous("x", VariableScope::Leader, 0.0, 10.0)
            .unwrap();
        assert_eq!(id, VariableId(0));
        assert_eq!(vs.len(), 1);

        let info = vs.get(id).unwrap();
        assert_eq!(info.name, "x");

        let info2 = vs.get_by_name("x").unwrap();
        assert_eq!(info2.id, id);
    }

    #[test]
    fn test_variable_set_duplicate_name() {
        let mut vs = VariableSet::new();
        vs.add_continuous("x", VariableScope::Leader, 0.0, 1.0)
            .unwrap();
        let result = vs.add_continuous("x", VariableScope::Follower, 0.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_variable_set_filtering() {
        let mut vs = VariableSet::new();
        vs.add_continuous("x1", VariableScope::Leader, 0.0, 1.0)
            .unwrap();
        vs.add_continuous("y1", VariableScope::Follower, 0.0, 1.0)
            .unwrap();
        vs.add_binary("b1", VariableScope::Leader).unwrap();
        vs.add_continuous("s1", VariableScope::Shared, 0.0, 5.0)
            .unwrap();

        assert_eq!(vs.num_leader(), 2);
        assert_eq!(vs.num_follower(), 1);
        assert_eq!(vs.num_shared(), 1);
        assert!(vs.has_integer_variables());

        let bins = vs.filter_by_type(VariableType::Binary);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].name, "b1");
    }

    #[test]
    fn test_variable_set_subset() {
        let mut vs = VariableSet::new();
        let id0 = vs
            .add_continuous("a", VariableScope::Leader, 0.0, 1.0)
            .unwrap();
        let _id1 = vs
            .add_continuous("b", VariableScope::Follower, 0.0, 1.0)
            .unwrap();
        let id2 = vs
            .add_continuous("c", VariableScope::Leader, 0.0, 1.0)
            .unwrap();

        let sub = vs.subset(&[id0, id2]);
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn test_scope_predicates() {
        assert!(VariableScope::Leader.is_upper());
        assert!(!VariableScope::Leader.is_lower());
        assert!(VariableScope::Follower.is_lower());
        assert!(!VariableScope::Follower.is_upper());
        assert!(VariableScope::Shared.is_upper());
        assert!(VariableScope::Shared.is_lower());
    }
}
