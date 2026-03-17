//! Core primitive types for the MutSpec system.
//!
//! Provides the foundational type system for QF-LIA programs, including the
//! [`QfLiaType`] enum representing the type language, [`Value`] for runtime
//! values, [`Variable`] for named bindings, and [`FunctionSignature`] for
//! callable program elements.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// QfLiaType
// ---------------------------------------------------------------------------

/// Types supported by our quantifier-free linear integer arithmetic language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QfLiaType {
    /// 32-bit signed integer.
    Int,
    /// 64-bit signed integer.
    Long,
    /// Boolean truth value.
    Boolean,
    /// Array of integers (select/store theory).
    IntArray,
    /// No return value.
    Void,
}

impl QfLiaType {
    /// Returns `true` if this is a numeric type (`Int` or `Long`).
    pub fn is_numeric(&self) -> bool {
        matches!(self, QfLiaType::Int | QfLiaType::Long)
    }

    /// Returns `true` if this is a boolean type.
    pub fn is_boolean(&self) -> bool {
        matches!(self, QfLiaType::Boolean)
    }

    /// Returns `true` for array types.
    pub fn is_array(&self) -> bool {
        matches!(self, QfLiaType::IntArray)
    }

    /// Returns `true` if the type can carry a meaningful value.
    pub fn is_value_type(&self) -> bool {
        !matches!(self, QfLiaType::Void)
    }

    /// Returns `true` if a value of type `self` can be implicitly widened to `target`.
    pub fn can_widen_to(&self, target: &QfLiaType) -> bool {
        match (self, target) {
            (a, b) if a == b => true,
            (QfLiaType::Int, QfLiaType::Long) => true,
            _ => false,
        }
    }

    /// Returns the wider of two numeric types, or `None` if incompatible.
    pub fn unify(a: &QfLiaType, b: &QfLiaType) -> Option<QfLiaType> {
        if a == b {
            return Some(*a);
        }
        match (a, b) {
            (QfLiaType::Int, QfLiaType::Long) | (QfLiaType::Long, QfLiaType::Int) => {
                Some(QfLiaType::Long)
            }
            _ => None,
        }
    }

    /// Byte width used for SMT encoding.
    pub fn bit_width(&self) -> Option<u32> {
        match self {
            QfLiaType::Int => Some(32),
            QfLiaType::Long => Some(64),
            QfLiaType::Boolean => Some(1),
            _ => None,
        }
    }

    /// SMT-LIB sort name for this type.
    pub fn smt_sort(&self) -> &'static str {
        match self {
            QfLiaType::Int | QfLiaType::Long => "Int",
            QfLiaType::Boolean => "Bool",
            QfLiaType::IntArray => "(Array Int Int)",
            QfLiaType::Void => "Int",
        }
    }

    /// Parse a type name string into a [`QfLiaType`].
    pub fn from_name(name: &str) -> Option<QfLiaType> {
        match name {
            "int" => Some(QfLiaType::Int),
            "long" => Some(QfLiaType::Long),
            "boolean" | "bool" => Some(QfLiaType::Boolean),
            "int[]" | "intarray" => Some(QfLiaType::IntArray),
            "void" => Some(QfLiaType::Void),
            _ => None,
        }
    }

    /// Returns a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            QfLiaType::Int => "int",
            QfLiaType::Long => "long",
            QfLiaType::Boolean => "boolean",
            QfLiaType::IntArray => "int[]",
            QfLiaType::Void => "void",
        }
    }

    /// All non-void types in declaration order.
    pub fn all_value_types() -> &'static [QfLiaType] {
        &[
            QfLiaType::Int,
            QfLiaType::Long,
            QfLiaType::Boolean,
            QfLiaType::IntArray,
        ]
    }

    /// Returns true if this type is valid as a function return type.
    pub fn is_valid_return_type(&self) -> bool {
        true
    }

    /// Returns true if this type is valid as a parameter type.
    pub fn is_valid_param_type(&self) -> bool {
        !matches!(self, QfLiaType::Void)
    }
}

impl fmt::Display for QfLiaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------

/// The lexical scope in which a variable is visible.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// Global / module-level.
    Global,
    /// Function parameter.
    Parameter,
    /// Local to a function body.
    Local,
    /// Compiler-generated temporary.
    Temporary,
    /// SSA-renamed version of an original variable.
    Ssa { original: String, version: u32 },
}

impl Scope {
    pub fn ssa(original: impl Into<String>, version: u32) -> Self {
        Scope::Ssa {
            original: original.into(),
            version,
        }
    }

    pub fn ssa_info(&self) -> Option<(&str, u32)> {
        match self {
            Scope::Ssa { original, version } => Some((original.as_str(), *version)),
            _ => None,
        }
    }

    pub fn is_synthetic(&self) -> bool {
        matches!(self, Scope::Temporary | Scope::Ssa { .. })
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scope::Global => write!(f, "global"),
            Scope::Parameter => write!(f, "param"),
            Scope::Local => write!(f, "local"),
            Scope::Temporary => write!(f, "tmp"),
            Scope::Ssa { original, version } => write!(f, "ssa({original}#{version})"),
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Scope::Local
    }
}

// ---------------------------------------------------------------------------
// Variable
// ---------------------------------------------------------------------------

/// A named, typed variable binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub ty: QfLiaType,
    pub scope: Scope,
}

impl Variable {
    pub fn new(name: impl Into<String>, ty: QfLiaType, scope: Scope) -> Self {
        Self {
            name: name.into(),
            ty,
            scope,
        }
    }

    pub fn local(name: impl Into<String>, ty: QfLiaType) -> Self {
        Self::new(name, ty, Scope::Local)
    }

    pub fn param(name: impl Into<String>, ty: QfLiaType) -> Self {
        Self::new(name, ty, Scope::Parameter)
    }

    pub fn temp(name: impl Into<String>, ty: QfLiaType) -> Self {
        Self::new(name, ty, Scope::Temporary)
    }

    pub fn global(name: impl Into<String>, ty: QfLiaType) -> Self {
        Self::new(name, ty, Scope::Global)
    }

    pub fn ssa(
        name: impl Into<String>,
        ty: QfLiaType,
        original: impl Into<String>,
        version: u32,
    ) -> Self {
        Self {
            name: name.into(),
            ty,
            scope: Scope::Ssa {
                original: original.into(),
                version,
            },
        }
    }

    pub fn smt_name(&self) -> String {
        let sanitized = self
            .name
            .replace('.', "_dot_")
            .replace('#', "_v")
            .replace(' ', "_sp_");
        format!("|{}|", sanitized)
    }

    pub fn is_ssa(&self) -> bool {
        matches!(self.scope, Scope::Ssa { .. })
    }

    pub fn ssa_info(&self) -> Option<(&str, u32)> {
        self.scope.ssa_info()
    }

    pub fn is_temp(&self) -> bool {
        matches!(self.scope, Scope::Temporary)
    }

    pub fn next_ssa_version(&self, version: u32) -> Variable {
        let original = match &self.scope {
            Scope::Ssa { original, .. } => original.clone(),
            _ => self.name.clone(),
        };
        let new_name = format!("{}_{}", original, version);
        Variable::ssa(new_name, self.ty, original, version)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} [{}]", self.name, self.ty, self.scope)
    }
}

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// A concrete runtime value in the QF-LIA domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Value {
    Int(i32),
    Long(i64),
    Bool(bool),
    IntArray(Vec<i64>),
    Void,
}

impl Value {
    pub fn get_type(&self) -> QfLiaType {
        match self {
            Value::Int(_) => QfLiaType::Int,
            Value::Long(_) => QfLiaType::Long,
            Value::Bool(_) => QfLiaType::Boolean,
            Value::IntArray(_) => QfLiaType::IntArray,
            Value::Void => QfLiaType::Void,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v as i64),
            Value::Long(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[i64]> {
        match self {
            Value::IntArray(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(v) => *v != 0,
            Value::Long(v) => *v != 0,
            Value::IntArray(a) => !a.is_empty(),
            Value::Void => false,
        }
    }

    pub fn widen_to_long(&self) -> Option<Value> {
        match self {
            Value::Int(v) => Some(Value::Long(*v as i64)),
            Value::Long(_) => Some(self.clone()),
            _ => None,
        }
    }

    pub fn narrow_to_int(&self) -> Option<Value> {
        match self {
            Value::Long(v) if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 => {
                Some(Value::Int(*v as i32))
            }
            Value::Int(_) => Some(self.clone()),
            _ => None,
        }
    }

    pub fn array_select(&self, index: i64) -> Option<Value> {
        match self {
            Value::IntArray(arr) => arr.get(index as usize).map(|v| Value::Long(*v)),
            _ => None,
        }
    }

    pub fn array_store(&self, index: i64, value: i64) -> Option<Value> {
        match self {
            Value::IntArray(arr) => {
                let idx = index as usize;
                let mut new_arr = arr.clone();
                if idx < new_arr.len() {
                    new_arr[idx] = value;
                } else {
                    new_arr.resize(idx + 1, 0);
                    new_arr[idx] = value;
                }
                Some(Value::IntArray(new_arr))
            }
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            Value::IntArray(arr) => Some(arr.len()),
            _ => None,
        }
    }

    pub fn checked_add(&self, rhs: &Value) -> Option<Value> {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => a.checked_add(*b).map(Value::Int),
            (Value::Long(a), Value::Long(b)) => a.checked_add(*b).map(Value::Long),
            (Value::Int(a), Value::Long(b)) => (*a as i64).checked_add(*b).map(Value::Long),
            (Value::Long(a), Value::Int(b)) => a.checked_add(*b as i64).map(Value::Long),
            _ => None,
        }
    }

    pub fn checked_sub(&self, rhs: &Value) -> Option<Value> {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => a.checked_sub(*b).map(Value::Int),
            (Value::Long(a), Value::Long(b)) => a.checked_sub(*b).map(Value::Long),
            (Value::Int(a), Value::Long(b)) => (*a as i64).checked_sub(*b).map(Value::Long),
            (Value::Long(a), Value::Int(b)) => a.checked_sub(*b as i64).map(Value::Long),
            _ => None,
        }
    }

    pub fn checked_mul(&self, rhs: &Value) -> Option<Value> {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => a.checked_mul(*b).map(Value::Int),
            (Value::Long(a), Value::Long(b)) => a.checked_mul(*b).map(Value::Long),
            (Value::Int(a), Value::Long(b)) => (*a as i64).checked_mul(*b).map(Value::Long),
            (Value::Long(a), Value::Int(b)) => a.checked_mul(*b as i64).map(Value::Long),
            _ => None,
        }
    }

    pub fn checked_div(&self, rhs: &Value) -> Option<Value> {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => a.checked_div(*b).map(Value::Int),
            (Value::Long(a), Value::Long(b)) if *b != 0 => a.checked_div(*b).map(Value::Long),
            (Value::Int(a), Value::Long(b)) if *b != 0 => {
                (*a as i64).checked_div(*b).map(Value::Long)
            }
            (Value::Long(a), Value::Int(b)) if *b != 0 => a.checked_div(*b as i64).map(Value::Long),
            _ => None,
        }
    }

    pub fn checked_rem(&self, rhs: &Value) -> Option<Value> {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => a.checked_rem(*b).map(Value::Int),
            (Value::Long(a), Value::Long(b)) if *b != 0 => a.checked_rem(*b).map(Value::Long),
            (Value::Int(a), Value::Long(b)) if *b != 0 => {
                (*a as i64).checked_rem(*b).map(Value::Long)
            }
            (Value::Long(a), Value::Int(b)) if *b != 0 => a.checked_rem(*b as i64).map(Value::Long),
            _ => None,
        }
    }

    pub fn checked_neg(&self) -> Option<Value> {
        match self {
            Value::Int(v) => v.checked_neg().map(Value::Int),
            Value::Long(v) => v.checked_neg().map(Value::Long),
            _ => None,
        }
    }

    pub fn val_lt(&self, rhs: &Value) -> Option<Value> {
        match (self.as_i64(), rhs.as_i64()) {
            (Some(a), Some(b)) => Some(Value::Bool(a < b)),
            _ => None,
        }
    }

    pub fn val_le(&self, rhs: &Value) -> Option<Value> {
        match (self.as_i64(), rhs.as_i64()) {
            (Some(a), Some(b)) => Some(Value::Bool(a <= b)),
            _ => None,
        }
    }

    pub fn val_gt(&self, rhs: &Value) -> Option<Value> {
        match (self.as_i64(), rhs.as_i64()) {
            (Some(a), Some(b)) => Some(Value::Bool(a > b)),
            _ => None,
        }
    }

    pub fn val_ge(&self, rhs: &Value) -> Option<Value> {
        match (self.as_i64(), rhs.as_i64()) {
            (Some(a), Some(b)) => Some(Value::Bool(a >= b)),
            _ => None,
        }
    }

    pub fn val_eq(&self, rhs: &Value) -> Option<Value> {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Some(Value::Bool(a == b)),
            (Value::Long(a), Value::Long(b)) => Some(Value::Bool(a == b)),
            (Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(a == b)),
            (Value::Int(a), Value::Long(b)) => Some(Value::Bool(*a as i64 == *b)),
            (Value::Long(a), Value::Int(b)) => Some(Value::Bool(*a == *b as i64)),
            _ => None,
        }
    }

    pub fn val_ne(&self, rhs: &Value) -> Option<Value> {
        self.val_eq(rhs).map(|v| match v {
            Value::Bool(b) => Value::Bool(!b),
            other => other,
        })
    }

    pub fn logical_and(&self, rhs: &Value) -> Option<Value> {
        match (self.as_bool(), rhs.as_bool()) {
            (Some(a), Some(b)) => Some(Value::Bool(a && b)),
            _ => None,
        }
    }

    pub fn logical_or(&self, rhs: &Value) -> Option<Value> {
        match (self.as_bool(), rhs.as_bool()) {
            (Some(a), Some(b)) => Some(Value::Bool(a || b)),
            _ => None,
        }
    }

    pub fn logical_not(&self) -> Option<Value> {
        self.as_bool().map(|b| Value::Bool(!b))
    }

    pub fn default_for(ty: &QfLiaType) -> Value {
        match ty {
            QfLiaType::Int => Value::Int(0),
            QfLiaType::Long => Value::Long(0),
            QfLiaType::Boolean => Value::Bool(false),
            QfLiaType::IntArray => Value::IntArray(Vec::new()),
            QfLiaType::Void => Value::Void,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Long(v) => write!(f, "{v}L"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::IntArray(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Void => write!(f, "void"),
        }
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Long(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<Vec<i64>> for Value {
    fn from(v: Vec<i64>) -> Self {
        Value::IntArray(v)
    }
}

impl Add for Value {
    type Output = Option<Value>;
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs)
    }
}
impl Sub for Value {
    type Output = Option<Value>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
    }
}
impl Mul for Value {
    type Output = Option<Value>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(&rhs)
    }
}
impl Div for Value {
    type Output = Option<Value>;
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(&rhs)
    }
}
impl Rem for Value {
    type Output = Option<Value>;
    fn rem(self, rhs: Self) -> Self::Output {
        self.checked_rem(&rhs)
    }
}
impl Neg for Value {
    type Output = Option<Value>;
    fn neg(self) -> Self::Output {
        self.checked_neg()
    }
}

// ---------------------------------------------------------------------------
// FunctionSignature
// ---------------------------------------------------------------------------

/// Signature of a function in the target program.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<(String, QfLiaType)>,
    pub return_type: QfLiaType,
}

impl FunctionSignature {
    pub fn new(
        name: impl Into<String>,
        params: Vec<(String, QfLiaType)>,
        return_type: QfLiaType,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            return_type,
        }
    }

    pub fn arity(&self) -> usize {
        self.params.len()
    }

    pub fn param_type(&self, index: usize) -> Option<&QfLiaType> {
        self.params.get(index).map(|(_, ty)| ty)
    }

    pub fn param_type_by_name(&self, name: &str) -> Option<&QfLiaType> {
        self.params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ty)| ty)
    }

    pub fn param_names(&self) -> Vec<&str> {
        self.params.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn param_types(&self) -> Vec<QfLiaType> {
        self.params.iter().map(|(_, ty)| *ty).collect()
    }

    pub fn param_variables(&self) -> Vec<Variable> {
        self.params
            .iter()
            .map(|(n, ty)| Variable::param(n.clone(), *ty))
            .collect()
    }

    pub fn smt_declaration(&self) -> String {
        let param_sorts: Vec<&str> = self.params.iter().map(|(_, ty)| ty.smt_sort()).collect();
        format!(
            "(declare-fun {} ({}) {})",
            self.name,
            param_sorts.join(" "),
            self.return_type.smt_sort()
        )
    }

    pub fn is_void(&self) -> bool {
        self.return_type == QfLiaType::Void
    }
}

impl fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}(", self.return_type, self.name)?;
        for (i, (name, ty)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{ty} {name}")?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Type-checking utilities
// ---------------------------------------------------------------------------

pub fn type_compatible(expected: &QfLiaType, actual: &QfLiaType) -> bool {
    expected == actual || actual.can_widen_to(expected)
}

pub fn arith_result_type(lhs: &QfLiaType, rhs: &QfLiaType) -> Option<QfLiaType> {
    if lhs.is_numeric() && rhs.is_numeric() {
        QfLiaType::unify(lhs, rhs)
    } else {
        None
    }
}

pub fn relational_result_type(lhs: &QfLiaType, rhs: &QfLiaType) -> Option<QfLiaType> {
    if lhs.is_numeric() && rhs.is_numeric() {
        Some(QfLiaType::Boolean)
    } else {
        None
    }
}

pub fn equality_comparable(ty: &QfLiaType) -> bool {
    matches!(ty, QfLiaType::Int | QfLiaType::Long | QfLiaType::Boolean)
}

pub fn logical_result_type(lhs: &QfLiaType, rhs: &QfLiaType) -> Option<QfLiaType> {
    if lhs.is_boolean() && rhs.is_boolean() {
        Some(QfLiaType::Boolean)
    } else {
        None
    }
}

pub fn check_call_args(
    sig: &FunctionSignature,
    arg_types: &[QfLiaType],
) -> Result<(), Vec<(usize, QfLiaType, QfLiaType)>> {
    if sig.arity() != arg_types.len() {
        let mismatches: Vec<_> = (0..sig.arity().max(arg_types.len()))
            .map(|i| {
                let expected = sig.param_type(i).copied().unwrap_or(QfLiaType::Void);
                let actual = arg_types.get(i).copied().unwrap_or(QfLiaType::Void);
                (i, expected, actual)
            })
            .collect();
        return Err(mismatches);
    }
    let mismatches: Vec<_> = sig
        .params
        .iter()
        .zip(arg_types.iter())
        .enumerate()
        .filter(|(_, ((_, expected), actual))| !type_compatible(expected, actual))
        .map(|(i, ((_, expected), actual))| (i, *expected, *actual))
        .collect();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches)
    }
}

pub fn parse_type(s: &str) -> Option<QfLiaType> {
    QfLiaType::from_name(s.trim())
}

pub fn default_value(ty: &QfLiaType) -> Value {
    Value::default_for(ty)
}

pub fn coerce(value: &Value, target: &QfLiaType) -> Option<Value> {
    if value.get_type() == *target {
        return Some(value.clone());
    }
    match (value, target) {
        (Value::Int(v), QfLiaType::Long) => Some(Value::Long(*v as i64)),
        (Value::Long(v), QfLiaType::Int) => {
            if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                Some(Value::Int(*v as i32))
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_is_numeric() {
        assert!(QfLiaType::Int.is_numeric());
        assert!(QfLiaType::Long.is_numeric());
        assert!(!QfLiaType::Boolean.is_numeric());
        assert!(!QfLiaType::IntArray.is_numeric());
        assert!(!QfLiaType::Void.is_numeric());
    }

    #[test]
    fn test_type_display() {
        assert_eq!(QfLiaType::Int.to_string(), "int");
        assert_eq!(QfLiaType::Long.to_string(), "long");
        assert_eq!(QfLiaType::Boolean.to_string(), "boolean");
        assert_eq!(QfLiaType::IntArray.to_string(), "int[]");
        assert_eq!(QfLiaType::Void.to_string(), "void");
    }

    #[test]
    fn test_type_unify() {
        assert_eq!(
            QfLiaType::unify(&QfLiaType::Int, &QfLiaType::Int),
            Some(QfLiaType::Int)
        );
        assert_eq!(
            QfLiaType::unify(&QfLiaType::Int, &QfLiaType::Long),
            Some(QfLiaType::Long)
        );
        assert_eq!(
            QfLiaType::unify(&QfLiaType::Long, &QfLiaType::Int),
            Some(QfLiaType::Long)
        );
        assert_eq!(QfLiaType::unify(&QfLiaType::Int, &QfLiaType::Boolean), None);
    }

    #[test]
    fn test_type_can_widen() {
        assert!(QfLiaType::Int.can_widen_to(&QfLiaType::Long));
        assert!(QfLiaType::Int.can_widen_to(&QfLiaType::Int));
        assert!(!QfLiaType::Long.can_widen_to(&QfLiaType::Int));
        assert!(!QfLiaType::Boolean.can_widen_to(&QfLiaType::Int));
    }

    #[test]
    fn test_type_smt_sort() {
        assert_eq!(QfLiaType::Int.smt_sort(), "Int");
        assert_eq!(QfLiaType::Boolean.smt_sort(), "Bool");
        assert_eq!(QfLiaType::IntArray.smt_sort(), "(Array Int Int)");
    }

    #[test]
    fn test_type_from_name() {
        assert_eq!(QfLiaType::from_name("int"), Some(QfLiaType::Int));
        assert_eq!(QfLiaType::from_name("bool"), Some(QfLiaType::Boolean));
        assert_eq!(QfLiaType::from_name("boolean"), Some(QfLiaType::Boolean));
        assert_eq!(QfLiaType::from_name("int[]"), Some(QfLiaType::IntArray));
        assert_eq!(QfLiaType::from_name("nope"), None);
    }

    #[test]
    fn test_type_bit_width() {
        assert_eq!(QfLiaType::Int.bit_width(), Some(32));
        assert_eq!(QfLiaType::Long.bit_width(), Some(64));
        assert_eq!(QfLiaType::Boolean.bit_width(), Some(1));
        assert_eq!(QfLiaType::IntArray.bit_width(), None);
    }

    #[test]
    fn test_type_value_type() {
        assert!(QfLiaType::Int.is_value_type());
        assert!(!QfLiaType::Void.is_value_type());
    }

    #[test]
    fn test_all_value_types() {
        let types = QfLiaType::all_value_types();
        assert_eq!(types.len(), 4);
        assert!(!types.contains(&QfLiaType::Void));
    }

    #[test]
    fn test_variable_local() {
        let v = Variable::local("x", QfLiaType::Int);
        assert_eq!(v.name, "x");
        assert_eq!(v.ty, QfLiaType::Int);
        assert_eq!(v.scope, Scope::Local);
        assert!(!v.is_ssa());
    }

    #[test]
    fn test_variable_ssa() {
        let v = Variable::ssa("x_1", QfLiaType::Int, "x", 1);
        assert!(v.is_ssa());
        assert_eq!(v.ssa_info(), Some(("x", 1)));
    }

    #[test]
    fn test_variable_next_ssa() {
        let v = Variable::local("x", QfLiaType::Int);
        let v2 = v.next_ssa_version(1);
        assert_eq!(v2.name, "x_1");
        assert!(v2.is_ssa());
        let v3 = v2.next_ssa_version(2);
        assert_eq!(v3.name, "x_2");
    }

    #[test]
    fn test_variable_smt_name() {
        let v = Variable::local("my.var", QfLiaType::Int);
        assert_eq!(v.smt_name(), "|my_dot_var|");
    }

    #[test]
    fn test_variable_display() {
        let v = Variable::param("n", QfLiaType::Long);
        let s = v.to_string();
        assert!(s.contains("n"));
        assert!(s.contains("long"));
        assert!(s.contains("param"));
    }

    #[test]
    fn test_value_get_type() {
        assert_eq!(Value::Int(0).get_type(), QfLiaType::Int);
        assert_eq!(Value::Long(0).get_type(), QfLiaType::Long);
        assert_eq!(Value::Bool(true).get_type(), QfLiaType::Boolean);
        assert_eq!(Value::IntArray(vec![]).get_type(), QfLiaType::IntArray);
        assert_eq!(Value::Void.get_type(), QfLiaType::Void);
    }

    #[test]
    fn test_value_as_i64() {
        assert_eq!(Value::Int(42).as_i64(), Some(42));
        assert_eq!(Value::Long(100).as_i64(), Some(100));
        assert_eq!(Value::Bool(true).as_i64(), None);
    }

    #[test]
    fn test_value_display() {
        assert_eq!(Value::Int(42).to_string(), "42");
        assert_eq!(Value::Long(42).to_string(), "42L");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::IntArray(vec![1, 2, 3]).to_string(), "[1, 2, 3]");
        assert_eq!(Value::Void.to_string(), "void");
    }

    #[test]
    fn test_value_from() {
        let v: Value = 42i32.into();
        assert_eq!(v, Value::Int(42));
        let v: Value = 42i64.into();
        assert_eq!(v, Value::Long(42));
        let v: Value = true.into();
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn test_value_arithmetic() {
        assert_eq!(
            Value::Int(3).checked_add(&Value::Int(4)),
            Some(Value::Int(7))
        );
        assert_eq!(
            Value::Int(10).checked_sub(&Value::Int(3)),
            Some(Value::Int(7))
        );
        assert_eq!(
            Value::Int(3).checked_mul(&Value::Int(4)),
            Some(Value::Int(12))
        );
        assert_eq!(
            Value::Int(10).checked_div(&Value::Int(3)),
            Some(Value::Int(3))
        );
        assert_eq!(
            Value::Int(10).checked_rem(&Value::Int(3)),
            Some(Value::Int(1))
        );
    }

    #[test]
    fn test_value_mixed_arithmetic() {
        assert_eq!(
            Value::Int(3).checked_add(&Value::Long(4)),
            Some(Value::Long(7))
        );
        assert_eq!(
            Value::Long(10).checked_sub(&Value::Int(3)),
            Some(Value::Long(7))
        );
    }

    #[test]
    fn test_value_div_by_zero() {
        assert_eq!(Value::Int(10).checked_div(&Value::Int(0)), None);
        assert_eq!(Value::Long(10).checked_rem(&Value::Long(0)), None);
    }

    #[test]
    fn test_value_relational() {
        assert_eq!(
            Value::Int(3).val_lt(&Value::Int(5)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            Value::Int(5).val_lt(&Value::Int(3)),
            Some(Value::Bool(false))
        );
        assert_eq!(
            Value::Int(3).val_le(&Value::Int(3)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            Value::Int(5).val_gt(&Value::Int(3)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            Value::Int(3).val_ge(&Value::Int(5)),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_value_equality() {
        assert_eq!(
            Value::Int(3).val_eq(&Value::Int(3)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            Value::Int(3).val_ne(&Value::Int(4)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            Value::Bool(true).val_eq(&Value::Bool(false)),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_value_logical() {
        assert_eq!(
            Value::Bool(true).logical_and(&Value::Bool(false)),
            Some(Value::Bool(false))
        );
        assert_eq!(
            Value::Bool(true).logical_or(&Value::Bool(false)),
            Some(Value::Bool(true))
        );
        assert_eq!(Value::Bool(true).logical_not(), Some(Value::Bool(false)));
    }

    #[test]
    fn test_value_operator_impls() {
        let a = Value::Int(10);
        let b = Value::Int(3);
        assert_eq!(a.clone() + b.clone(), Some(Value::Int(13)));
        assert_eq!(a.clone() - b.clone(), Some(Value::Int(7)));
        assert_eq!(a.clone() * b.clone(), Some(Value::Int(30)));
        assert_eq!(a.clone() / b.clone(), Some(Value::Int(3)));
        assert_eq!(a.clone() % b.clone(), Some(Value::Int(1)));
        assert_eq!(-a, Some(Value::Int(-10)));
    }

    #[test]
    fn test_value_widen_narrow() {
        assert_eq!(Value::Int(42).widen_to_long(), Some(Value::Long(42)));
        assert_eq!(Value::Long(42).narrow_to_int(), Some(Value::Int(42)));
        assert_eq!(Value::Long(i64::MAX).narrow_to_int(), None);
        assert_eq!(Value::Bool(true).widen_to_long(), None);
    }

    #[test]
    fn test_value_array_ops() {
        let arr = Value::IntArray(vec![10, 20, 30]);
        assert_eq!(arr.array_select(1), Some(Value::Long(20)));
        assert_eq!(arr.array_select(5), None);
        assert_eq!(arr.array_len(), Some(3));
        let arr2 = arr.array_store(1, 99).unwrap();
        assert_eq!(arr2, Value::IntArray(vec![10, 99, 30]));
    }

    #[test]
    fn test_value_is_truthy() {
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Int(1).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(!Value::Void.is_truthy());
    }

    #[test]
    fn test_value_default_for() {
        assert_eq!(Value::default_for(&QfLiaType::Int), Value::Int(0));
        assert_eq!(Value::default_for(&QfLiaType::Boolean), Value::Bool(false));
        assert_eq!(
            Value::default_for(&QfLiaType::IntArray),
            Value::IntArray(vec![])
        );
    }

    #[test]
    fn test_function_signature_basic() {
        let sig = FunctionSignature::new(
            "max",
            vec![("a".into(), QfLiaType::Int), ("b".into(), QfLiaType::Int)],
            QfLiaType::Int,
        );
        assert_eq!(sig.arity(), 2);
        assert_eq!(sig.param_type(0), Some(&QfLiaType::Int));
        assert_eq!(sig.param_type_by_name("b"), Some(&QfLiaType::Int));
        assert_eq!(sig.param_type(5), None);
        assert!(!sig.is_void());
    }

    #[test]
    fn test_function_signature_display() {
        let sig = FunctionSignature::new(
            "add",
            vec![("x".into(), QfLiaType::Int), ("y".into(), QfLiaType::Int)],
            QfLiaType::Int,
        );
        let s = sig.to_string();
        assert!(s.contains("add"));
        assert!(s.contains("int x"));
    }

    #[test]
    fn test_function_signature_smt_decl() {
        let sig = FunctionSignature::new(
            "foo",
            vec![("a".into(), QfLiaType::Int)],
            QfLiaType::Boolean,
        );
        let decl = sig.smt_declaration();
        assert!(decl.contains("declare-fun"));
        assert!(decl.contains("foo"));
        assert!(decl.contains("Bool"));
    }

    #[test]
    fn test_function_signature_param_variables() {
        let sig = FunctionSignature::new(
            "f",
            vec![("a".into(), QfLiaType::Int), ("b".into(), QfLiaType::Long)],
            QfLiaType::Void,
        );
        let vars = sig.param_variables();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].scope, Scope::Parameter);
        assert_eq!(vars[1].ty, QfLiaType::Long);
        assert!(sig.is_void());
    }

    #[test]
    fn test_function_signature_param_names_types() {
        let sig = FunctionSignature::new(
            "g",
            vec![
                ("x".into(), QfLiaType::Int),
                ("y".into(), QfLiaType::Boolean),
            ],
            QfLiaType::Int,
        );
        assert_eq!(sig.param_names(), vec!["x", "y"]);
        assert_eq!(sig.param_types(), vec![QfLiaType::Int, QfLiaType::Boolean]);
    }

    #[test]
    fn test_type_compatible() {
        assert!(type_compatible(&QfLiaType::Int, &QfLiaType::Int));
        assert!(type_compatible(&QfLiaType::Long, &QfLiaType::Int));
        assert!(!type_compatible(&QfLiaType::Int, &QfLiaType::Long));
        assert!(!type_compatible(&QfLiaType::Boolean, &QfLiaType::Int));
    }

    #[test]
    fn test_arith_result_type() {
        assert_eq!(
            arith_result_type(&QfLiaType::Int, &QfLiaType::Int),
            Some(QfLiaType::Int)
        );
        assert_eq!(
            arith_result_type(&QfLiaType::Int, &QfLiaType::Long),
            Some(QfLiaType::Long)
        );
        assert_eq!(
            arith_result_type(&QfLiaType::Boolean, &QfLiaType::Int),
            None
        );
    }

    #[test]
    fn test_relational_result_type() {
        assert_eq!(
            relational_result_type(&QfLiaType::Int, &QfLiaType::Long),
            Some(QfLiaType::Boolean)
        );
        assert_eq!(
            relational_result_type(&QfLiaType::Boolean, &QfLiaType::Int),
            None
        );
    }

    #[test]
    fn test_logical_result_type() {
        assert_eq!(
            logical_result_type(&QfLiaType::Boolean, &QfLiaType::Boolean),
            Some(QfLiaType::Boolean)
        );
        assert_eq!(
            logical_result_type(&QfLiaType::Int, &QfLiaType::Boolean),
            None
        );
    }

    #[test]
    fn test_check_call_args_ok() {
        let sig = FunctionSignature::new(
            "f",
            vec![("a".into(), QfLiaType::Int), ("b".into(), QfLiaType::Long)],
            QfLiaType::Void,
        );
        assert!(check_call_args(&sig, &[QfLiaType::Int, QfLiaType::Long]).is_ok());
        assert!(check_call_args(&sig, &[QfLiaType::Int, QfLiaType::Int]).is_ok());
    }

    #[test]
    fn test_check_call_args_mismatch() {
        let sig = FunctionSignature::new("f", vec![("a".into(), QfLiaType::Int)], QfLiaType::Void);
        assert!(check_call_args(&sig, &[QfLiaType::Boolean]).is_err());
        assert!(check_call_args(&sig, &[]).is_err());
    }

    #[test]
    fn test_parse_type() {
        assert_eq!(parse_type("int"), Some(QfLiaType::Int));
        assert_eq!(parse_type("  long  "), Some(QfLiaType::Long));
        assert_eq!(parse_type("unknown"), None);
    }

    #[test]
    fn test_default_value() {
        assert_eq!(default_value(&QfLiaType::Int), Value::Int(0));
        assert_eq!(default_value(&QfLiaType::Boolean), Value::Bool(false));
    }

    #[test]
    fn test_coerce() {
        assert_eq!(
            coerce(&Value::Int(42), &QfLiaType::Long),
            Some(Value::Long(42))
        );
        assert_eq!(
            coerce(&Value::Long(42), &QfLiaType::Int),
            Some(Value::Int(42))
        );
        assert_eq!(coerce(&Value::Long(i64::MAX), &QfLiaType::Int), None);
        assert_eq!(coerce(&Value::Bool(true), &QfLiaType::Int), None);
    }

    #[test]
    fn test_equality_comparable() {
        assert!(equality_comparable(&QfLiaType::Int));
        assert!(equality_comparable(&QfLiaType::Boolean));
        assert!(!equality_comparable(&QfLiaType::IntArray));
        assert!(!equality_comparable(&QfLiaType::Void));
    }

    #[test]
    fn test_scope_display() {
        assert_eq!(Scope::Global.to_string(), "global");
        assert_eq!(Scope::Parameter.to_string(), "param");
        assert_eq!(Scope::Local.to_string(), "local");
        assert_eq!(Scope::Temporary.to_string(), "tmp");
        let ssa = Scope::Ssa {
            original: "x".into(),
            version: 3,
        };
        assert_eq!(ssa.to_string(), "ssa(x#3)");
    }

    #[test]
    fn test_scope_default() {
        assert_eq!(Scope::default(), Scope::Local);
    }

    #[test]
    fn test_scope_is_synthetic() {
        assert!(Scope::Temporary.is_synthetic());
        assert!(Scope::ssa("x", 1).is_synthetic());
        assert!(!Scope::Local.is_synthetic());
    }

    #[test]
    fn test_value_serialization() {
        let v = Value::Int(42);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_type_serialization() {
        let ty = QfLiaType::IntArray;
        let json = serde_json::to_string(&ty).unwrap();
        let ty2: QfLiaType = serde_json::from_str(&json).unwrap();
        assert_eq!(ty, ty2);
    }

    #[test]
    fn test_variable_serialization() {
        let v = Variable::local("x", QfLiaType::Int);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Variable = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_function_signature_serialization() {
        let sig = FunctionSignature::new(
            "test",
            vec![("a".into(), QfLiaType::Int)],
            QfLiaType::Boolean,
        );
        let json = serde_json::to_string(&sig).unwrap();
        let sig2: FunctionSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig, sig2);
    }
}
