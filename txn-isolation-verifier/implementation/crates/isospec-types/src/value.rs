//! Data value types for the transaction model.
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A database value that can be stored and compared.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    Bytes(Vec<u8>),
    Timestamp(i64),
    Array(Vec<Value>),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            Self::Integer(i) => Some(*i != 0),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "NULL",
            Self::Boolean(_) => "BOOLEAN",
            Self::Integer(_) => "INTEGER",
            Self::Float(_) => "FLOAT",
            Self::Text(_) => "TEXT",
            Self::Bytes(_) => "BYTES",
            Self::Timestamp(_) => "TIMESTAMP",
            Self::Array(_) => "ARRAY",
        }
    }

    pub fn sql_literal(&self) -> String {
        match self {
            Self::Null => "NULL".to_string(),
            Self::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
            Self::Integer(i) => i.to_string(),
            Self::Float(f) => format!("{:.6}", f),
            Self::Text(s) => format!("'{}'", s.replace('\'', "''")),
            Self::Bytes(b) => format!("X'{}'", hex_encode(b)),
            Self::Timestamp(ts) => format!("TIMESTAMP '{}'", ts),
            Self::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.sql_literal()).collect();
                format!("ARRAY[{}]", items.join(", "))
            }
        }
    }

    pub fn compatible_with(&self, other: &Value) -> bool {
        matches!(
            (self, other),
            (Self::Null, _) | (_, Self::Null)
            | (Self::Integer(_), Self::Integer(_))
            | (Self::Integer(_), Self::Float(_))
            | (Self::Float(_), Self::Integer(_))
            | (Self::Float(_), Self::Float(_))
            | (Self::Text(_), Self::Text(_))
            | (Self::Boolean(_), Self::Boolean(_))
            | (Self::Bytes(_), Self::Bytes(_))
            | (Self::Timestamp(_), Self::Timestamp(_))
        )
    }

    pub fn compare(&self, other: &Value) -> Option<Ordering> {
        match (self, other) {
            (Self::Null, Self::Null) => Some(Ordering::Equal),
            (Self::Null, _) => Some(Ordering::Less),
            (_, Self::Null) => Some(Ordering::Greater),
            (Self::Integer(a), Self::Integer(b)) => Some(a.cmp(b)),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::Integer(a), Self::Float(b)) => (*a as f64).partial_cmp(b),
            (Self::Float(a), Self::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (Self::Text(a), Self::Text(b)) => Some(a.cmp(b)),
            (Self::Boolean(a), Self::Boolean(b)) => Some(a.cmp(b)),
            (Self::Timestamp(a), Self::Timestamp(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Integer(a), Self::Integer(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::Timestamp(a), Self::Timestamp(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Null => {}
            Self::Boolean(b) => b.hash(state),
            Self::Integer(i) => i.hash(state),
            Self::Float(f) => f.to_bits().hash(state),
            Self::Text(s) => s.hash(state),
            Self::Bytes(b) => b.hash(state),
            Self::Timestamp(t) => t.hash(state),
            Self::Array(a) => a.hash(state),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(v) => write!(f, "{}", v),
            Self::Text(s) => write!(f, "'{}'", s),
            Self::Bytes(b) => write!(f, "0x{}", hex_encode(b)),
            Self::Timestamp(t) => write!(f, "TS({})", t),
            Self::Array(a) => {
                let items: Vec<String> = a.iter().map(|v| format!("{}", v)).collect();
                write!(f, "[{}]", items.join(", "))
            }
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// A row of values keyed by column name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub columns: indexmap::IndexMap<String, Value>,
}

impl Row {
    pub fn new() -> Self {
        Self {
            columns: indexmap::IndexMap::new(),
        }
    }

    pub fn with_column(mut self, name: impl Into<String>, value: Value) -> Self {
        self.columns.insert(name.into(), value);
        self
    }

    pub fn get(&self, column: &str) -> Option<&Value> {
        self.columns.get(column)
    }

    pub fn set(&mut self, column: impl Into<String>, value: Value) {
        self.columns.insert(column.into(), value);
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub fn merge(&mut self, other: &Row) {
        for (k, v) in &other.columns {
            self.columns.insert(k.clone(), v.clone());
        }
    }

    pub fn project(&self, columns: &[&str]) -> Row {
        let mut result = Row::new();
        for col in columns {
            if let Some(v) = self.columns.get(*col) {
                result.columns.insert(col.to_string(), v.clone());
            }
        }
        result
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

/// A version of a data item.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataVersion {
    pub item_id: crate::identifier::ItemId,
    pub version_number: u64,
    pub created_by: crate::identifier::TransactionId,
    pub value: Value,
    pub is_deleted: bool,
}

impl DataVersion {
    pub fn new(
        item_id: crate::identifier::ItemId,
        version_number: u64,
        created_by: crate::identifier::TransactionId,
        value: Value,
    ) -> Self {
        Self {
            item_id,
            version_number,
            created_by,
            value,
            is_deleted: false,
        }
    }

    pub fn deleted(
        item_id: crate::identifier::ItemId,
        version_number: u64,
        created_by: crate::identifier::TransactionId,
    ) -> Self {
        Self {
            item_id,
            version_number,
            created_by,
            value: Value::Null,
            is_deleted: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_types() {
        assert!(Value::Null.is_null());
        assert_eq!(Value::Integer(42).as_integer(), Some(42));
        assert_eq!(Value::Text("hello".into()).as_text(), Some("hello"));
        assert_eq!(Value::Boolean(true).as_bool(), Some(true));
    }

    #[test]
    fn test_value_comparison() {
        assert_eq!(
            Value::Integer(1).compare(&Value::Integer(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            Value::Text("a".into()).compare(&Value::Text("b".into())),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn test_value_sql_literal() {
        assert_eq!(Value::Null.sql_literal(), "NULL");
        assert_eq!(Value::Integer(42).sql_literal(), "42");
        assert_eq!(Value::Text("hello".into()).sql_literal(), "'hello'");
        assert_eq!(Value::Boolean(true).sql_literal(), "TRUE");
    }

    #[test]
    fn test_row_operations() {
        let row = Row::new()
            .with_column("id", Value::Integer(1))
            .with_column("name", Value::Text("Alice".into()));
        assert_eq!(row.len(), 2);
        assert_eq!(row.get("id"), Some(&Value::Integer(1)));
        let projected = row.project(&["name"]);
        assert_eq!(projected.len(), 1);
    }

    #[test]
    fn test_value_compatibility() {
        assert!(Value::Integer(1).compatible_with(&Value::Integer(2)));
        assert!(Value::Integer(1).compatible_with(&Value::Float(2.0)));
        assert!(!Value::Integer(1).compatible_with(&Value::Text("hi".into())));
        assert!(Value::Null.compatible_with(&Value::Integer(1)));
    }

    #[test]
    fn test_data_version() {
        use crate::identifier::*;
        let v = DataVersion::new(ItemId::new(1), 0, TransactionId::new(1), Value::Integer(10));
        assert!(!v.is_deleted);
        let del = DataVersion::deleted(ItemId::new(1), 1, TransactionId::new(2));
        assert!(del.is_deleted);
    }
}
