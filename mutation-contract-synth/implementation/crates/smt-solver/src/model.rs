//! SMT model extraction and representation.
//!
//! After a `check-sat` returns SAT, the solver can produce a model — a mapping
//! from declared symbols to concrete values. This module defines the
//! representation and utilities for working with models.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ast::SmtSort;
use crate::sexp_parser::SExp;

// ---------------------------------------------------------------------------
// Model values
// ---------------------------------------------------------------------------

/// A concrete value from an SMT model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelValue {
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// Array represented as a finite map with a default.
    Array {
        entries: Vec<(ModelValue, ModelValue)>,
        default: Box<ModelValue>,
    },
    /// Uninterpreted constant.
    Uninterpreted(String),
    /// Unknown / unparseable value (raw s-expression text).
    Unknown(String),
}

impl ModelValue {
    /// Try to extract as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ModelValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ModelValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Check if this is an integer value.
    pub fn is_int(&self) -> bool {
        matches!(self, ModelValue::Int(_))
    }

    /// Check if this is a boolean value.
    pub fn is_bool(&self) -> bool {
        matches!(self, ModelValue::Bool(_))
    }

    /// Check if this is an array value.
    pub fn is_array(&self) -> bool {
        matches!(self, ModelValue::Array { .. })
    }

    /// Try to read an array entry.
    pub fn array_select(&self, index: &ModelValue) -> Option<&ModelValue> {
        match self {
            ModelValue::Array { entries, default } => {
                for (k, v) in entries {
                    if k == index {
                        return Some(v);
                    }
                }
                Some(default)
            }
            _ => None,
        }
    }

    /// Get the sort of this value.
    pub fn sort(&self) -> SmtSort {
        match self {
            ModelValue::Int(_) => SmtSort::Int,
            ModelValue::Bool(_) => SmtSort::Bool,
            ModelValue::Array { .. } => SmtSort::int_array(),
            ModelValue::Uninterpreted(_) | ModelValue::Unknown(_) => {
                SmtSort::Uninterpreted("Unknown".to_string())
            }
        }
    }
}

impl fmt::Display for ModelValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelValue::Int(v) => write!(f, "{}", v),
            ModelValue::Bool(v) => write!(f, "{}", v),
            ModelValue::Array {
                entries, default, ..
            } => {
                write!(f, "[")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} → {}", k, v)?;
                }
                write!(f, "; default → {}]", default)
            }
            ModelValue::Uninterpreted(s) => write!(f, "{}", s),
            ModelValue::Unknown(s) => write!(f, "?{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// An SMT model — a complete assignment of values to declared symbols.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmtModel {
    /// Symbol → value mappings.
    values: HashMap<String, ModelValue>,
    /// Original raw model text (for debugging).
    raw_text: Option<String>,
}

impl SmtModel {
    /// Create an empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a model from a map.
    pub fn from_map(values: HashMap<String, ModelValue>) -> Self {
        SmtModel {
            values,
            raw_text: None,
        }
    }

    /// Set the raw text (for debugging).
    pub fn with_raw_text(mut self, text: String) -> Self {
        self.raw_text = Some(text);
        self
    }

    /// Insert a value.
    pub fn insert(&mut self, name: String, value: ModelValue) {
        self.values.insert(name, value);
    }

    /// Get a value by name.
    pub fn get(&self, name: &str) -> Option<&ModelValue> {
        self.values.get(name)
    }

    /// Get an integer value by name.
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.values.get(name).and_then(|v| v.as_int())
    }

    /// Get a boolean value by name.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.values.get(name).and_then(|v| v.as_bool())
    }

    /// Check if a symbol has a value.
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Number of symbols in the model.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over all (name, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ModelValue)> {
        self.values.iter()
    }

    /// Get all symbol names.
    pub fn names(&self) -> Vec<&String> {
        self.values.keys().collect()
    }

    /// Filter to only integer values.
    pub fn int_values(&self) -> HashMap<String, i64> {
        self.values
            .iter()
            .filter_map(|(k, v)| v.as_int().map(|i| (k.clone(), i)))
            .collect()
    }

    /// Filter to only boolean values.
    pub fn bool_values(&self) -> HashMap<String, bool> {
        self.values
            .iter()
            .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
            .collect()
    }

    /// Merge another model into this one (other values take precedence).
    pub fn merge(&mut self, other: &SmtModel) {
        for (k, v) in &other.values {
            self.values.insert(k.clone(), v.clone());
        }
    }

    /// Get the raw text, if stored.
    pub fn raw_text(&self) -> Option<&str> {
        self.raw_text.as_deref()
    }

    /// Restrict model to a subset of names.
    pub fn restrict(&self, names: &[&str]) -> SmtModel {
        let values = names
            .iter()
            .filter_map(|n| self.values.get(*n).map(|v| (n.to_string(), v.clone())))
            .collect();
        SmtModel {
            values,
            raw_text: None,
        }
    }
}

impl fmt::Display for SmtModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model {{")?;
        let mut entries: Vec<_> = self.values.iter().collect();
        entries.sort_by_key(|(k, _)| k.clone());
        for (name, val) in entries {
            writeln!(f, "  {} = {}", name, val)?;
        }
        write!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Model parsing from S-expressions
// ---------------------------------------------------------------------------

/// Parse a model from S-expression output.
pub fn parse_model_from_sexp(sexp: &SExp) -> SmtModel {
    let mut model = SmtModel::new();
    match sexp {
        SExp::List(items) => {
            for item in items {
                if let Some((name, value)) = parse_define_fun(item) {
                    model.insert(name, value);
                }
            }
        }
        _ => {}
    }
    model
}

/// Parse a single `(define-fun name () Sort value)` from the model.
fn parse_define_fun(sexp: &SExp) -> Option<(String, ModelValue)> {
    match sexp {
        SExp::List(items) if items.len() >= 5 => {
            let head = match &items[0] {
                SExp::Atom(s) => s.as_str(),
                _ => return None,
            };
            if head != "define-fun" {
                return None;
            }
            let name = match &items[1] {
                SExp::Atom(s) => s.clone(),
                _ => return None,
            };
            let value = parse_model_value(&items[4]);
            Some((name, value))
        }
        _ => None,
    }
}

/// Parse a model value from an S-expression.
pub fn parse_model_value(sexp: &SExp) -> ModelValue {
    match sexp {
        SExp::Atom(s) => {
            if s == "true" {
                ModelValue::Bool(true)
            } else if s == "false" {
                ModelValue::Bool(false)
            } else if let Ok(v) = s.parse::<i64>() {
                ModelValue::Int(v)
            } else {
                ModelValue::Uninterpreted(s.clone())
            }
        }
        SExp::List(items) if items.len() == 2 => {
            // Possibly `(- N)` for negative integers.
            if let (SExp::Atom(op), SExp::Atom(num)) = (&items[0], &items[1]) {
                if op == "-" {
                    if let Ok(v) = num.parse::<i64>() {
                        return ModelValue::Int(-v);
                    }
                }
            }
            ModelValue::Unknown(format!("{}", sexp))
        }
        SExp::List(items) if items.len() >= 3 => {
            if let SExp::Atom(head) = &items[0] {
                if head == "store" && items.len() == 4 {
                    return parse_array_value(sexp);
                }
            }
            ModelValue::Unknown(format!("{}", sexp))
        }
        _ => ModelValue::Unknown(format!("{}", sexp)),
    }
}

/// Parse an array model value from nested `(store ...)` expressions.
fn parse_array_value(sexp: &SExp) -> ModelValue {
    let mut entries = Vec::new();
    let mut current = sexp;
    loop {
        match current {
            SExp::List(items) if items.len() == 4 => {
                if let SExp::Atom(head) = &items[0] {
                    if head == "store" {
                        let idx = parse_model_value(&items[2]);
                        let val = parse_model_value(&items[3]);
                        entries.push((idx, val));
                        current = &items[1];
                        continue;
                    }
                }
                break;
            }
            SExp::List(items) if items.len() == 2 => {
                // `((as const (Array Int Int)) default_val)`
                if let SExp::List(_) = &items[0] {
                    let default = parse_model_value(&items[1]);
                    entries.reverse();
                    return ModelValue::Array {
                        entries,
                        default: Box::new(default),
                    };
                }
                break;
            }
            _ => break,
        }
    }
    entries.reverse();
    ModelValue::Array {
        entries,
        default: Box::new(ModelValue::Int(0)),
    }
}

/// Construct a distinguishing-input model: given variable names and integer
/// values, build a model.
pub fn distinguishing_input(vars: &[(&str, i64)]) -> SmtModel {
    let values = vars
        .iter()
        .map(|(name, val)| (name.to_string(), ModelValue::Int(*val)))
        .collect();
    SmtModel::from_map(values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_value_int() {
        let v = ModelValue::Int(42);
        assert_eq!(v.as_int(), Some(42));
        assert!(v.is_int());
        assert!(!v.is_bool());
    }

    #[test]
    fn test_model_value_bool() {
        let v = ModelValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
        assert!(v.is_bool());
    }

    #[test]
    fn test_model_value_array() {
        let v = ModelValue::Array {
            entries: vec![
                (ModelValue::Int(0), ModelValue::Int(10)),
                (ModelValue::Int(1), ModelValue::Int(20)),
            ],
            default: Box::new(ModelValue::Int(0)),
        };
        assert!(v.is_array());
        assert_eq!(
            v.array_select(&ModelValue::Int(0)),
            Some(&ModelValue::Int(10))
        );
        assert_eq!(
            v.array_select(&ModelValue::Int(1)),
            Some(&ModelValue::Int(20))
        );
        // Default
        assert_eq!(
            v.array_select(&ModelValue::Int(99)),
            Some(&ModelValue::Int(0))
        );
    }

    #[test]
    fn test_model_insert_and_get() {
        let mut m = SmtModel::new();
        m.insert("x".to_string(), ModelValue::Int(5));
        m.insert("y".to_string(), ModelValue::Bool(false));
        assert_eq!(m.get_int("x"), Some(5));
        assert_eq!(m.get_bool("y"), Some(false));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_model_int_values() {
        let mut m = SmtModel::new();
        m.insert("x".to_string(), ModelValue::Int(1));
        m.insert("y".to_string(), ModelValue::Int(2));
        m.insert("b".to_string(), ModelValue::Bool(true));
        let ints = m.int_values();
        assert_eq!(ints.len(), 2);
        assert_eq!(ints["x"], 1);
        assert_eq!(ints["y"], 2);
    }

    #[test]
    fn test_model_merge() {
        let mut m1 = SmtModel::new();
        m1.insert("x".to_string(), ModelValue::Int(1));
        m1.insert("y".to_string(), ModelValue::Int(2));

        let mut m2 = SmtModel::new();
        m2.insert("y".to_string(), ModelValue::Int(99));
        m2.insert("z".to_string(), ModelValue::Int(3));

        m1.merge(&m2);
        assert_eq!(m1.get_int("x"), Some(1));
        assert_eq!(m1.get_int("y"), Some(99));
        assert_eq!(m1.get_int("z"), Some(3));
    }

    #[test]
    fn test_model_restrict() {
        let mut m = SmtModel::new();
        m.insert("x".to_string(), ModelValue::Int(1));
        m.insert("y".to_string(), ModelValue::Int(2));
        m.insert("z".to_string(), ModelValue::Int(3));
        let restricted = m.restrict(&["x", "z"]);
        assert_eq!(restricted.len(), 2);
        assert!(restricted.contains("x"));
        assert!(!restricted.contains("y"));
    }

    #[test]
    fn test_model_display() {
        let mut m = SmtModel::new();
        m.insert("x".to_string(), ModelValue::Int(42));
        let s = format!("{}", m);
        assert!(s.contains("x = 42"));
    }

    #[test]
    fn test_distinguishing_input() {
        let model = distinguishing_input(&[("x", 5), ("y", 10)]);
        assert_eq!(model.get_int("x"), Some(5));
        assert_eq!(model.get_int("y"), Some(10));
    }

    #[test]
    fn test_parse_model_value_atoms() {
        let sexp = SExp::Atom("42".to_string());
        assert_eq!(parse_model_value(&sexp), ModelValue::Int(42));

        let sexp = SExp::Atom("true".to_string());
        assert_eq!(parse_model_value(&sexp), ModelValue::Bool(true));
    }

    #[test]
    fn test_parse_negative_int() {
        let sexp = SExp::List(vec![
            SExp::Atom("-".to_string()),
            SExp::Atom("7".to_string()),
        ]);
        assert_eq!(parse_model_value(&sexp), ModelValue::Int(-7));
    }

    #[test]
    fn test_model_value_sort() {
        assert_eq!(ModelValue::Int(0).sort(), SmtSort::Int);
        assert_eq!(ModelValue::Bool(true).sort(), SmtSort::Bool);
    }
}
