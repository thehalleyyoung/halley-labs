//! Column type definitions for schema modeling.
use serde::{Deserialize, Serialize};
use crate::value::Value;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    Boolean,
    SmallInt,
    Integer,
    BigInt,
    Float,
    Double,
    Decimal { precision: u8, scale: u8 },
    Char(u32),
    Varchar(u32),
    Text,
    Blob,
    Timestamp,
    Date,
    Time,
    Uuid,
}

impl DataType {
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::SmallInt | Self::Integer | Self::BigInt | Self::Float | Self::Double | Self::Decimal { .. })
    }
    pub fn is_string(self) -> bool {
        matches!(self, Self::Char(_) | Self::Varchar(_) | Self::Text)
    }
    pub fn is_orderable(self) -> bool {
        self.is_numeric() || self.is_string() || matches!(self, Self::Timestamp | Self::Date | Self::Time)
    }
    pub fn default_value(self) -> Value {
        match self {
            Self::Boolean => Value::Boolean(false),
            Self::SmallInt | Self::Integer | Self::BigInt => Value::Integer(0),
            Self::Float | Self::Double | Self::Decimal { .. } => Value::Float(0.0),
            Self::Char(_) | Self::Varchar(_) | Self::Text => Value::Text(String::new()),
            Self::Blob => Value::Bytes(Vec::new()),
            Self::Timestamp | Self::Date | Self::Time => Value::Timestamp(0),
            Self::Uuid => Value::Text("00000000-0000-0000-0000-000000000000".into()),
        }
    }
    pub fn byte_size_estimate(self) -> usize {
        match self {
            Self::Boolean => 1,
            Self::SmallInt => 2,
            Self::Integer => 4,
            Self::BigInt | Self::Float | Self::Double | Self::Timestamp | Self::Date | Self::Time => 8,
            Self::Decimal { .. } => 16,
            Self::Char(n) | Self::Varchar(n) => n as usize,
            Self::Text | Self::Blob => 256,
            Self::Uuid => 16,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::SmallInt => write!(f, "SMALLINT"),
            Self::Integer => write!(f, "INTEGER"),
            Self::BigInt => write!(f, "BIGINT"),
            Self::Float => write!(f, "FLOAT"),
            Self::Double => write!(f, "DOUBLE"),
            Self::Decimal { precision, scale } => write!(f, "DECIMAL({},{})", precision, scale),
            Self::Char(n) => write!(f, "CHAR({})", n),
            Self::Varchar(n) => write!(f, "VARCHAR({})", n),
            Self::Text => write!(f, "TEXT"),
            Self::Blob => write!(f, "BLOB"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Uuid => write!(f, "UUID"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub default: Option<Value>,
    pub auto_increment: bool,
    pub references: Option<ForeignKeyRef>,
}

impl ColumnDef {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable: true,
            primary_key: false,
            unique: false,
            default: None,
            auto_increment: false,
            references: None,
        }
    }
    pub fn not_null(mut self) -> Self { self.nullable = false; self }
    pub fn primary_key(mut self) -> Self { self.primary_key = true; self.nullable = false; self }
    pub fn unique(mut self) -> Self { self.unique = true; self }
    pub fn with_default(mut self, val: Value) -> Self { self.default = Some(val); self }
    pub fn auto_increment(mut self) -> Self { self.auto_increment = true; self }
    pub fn references(mut self, table: impl Into<String>, column: impl Into<String>) -> Self {
        self.references = Some(ForeignKeyRef { table: table.into(), column: column.into() });
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyRef {
    pub table: String,
    pub column: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_data_type_properties() {
        assert!(DataType::Integer.is_numeric());
        assert!(!DataType::Integer.is_string());
        assert!(DataType::Varchar(255).is_string());
        assert!(DataType::Integer.is_orderable());
    }
    #[test]
    fn test_column_def_builder() {
        let col = ColumnDef::new("id", DataType::Integer).primary_key().auto_increment();
        assert!(col.primary_key);
        assert!(!col.nullable);
        assert!(col.auto_increment);
    }
    #[test]
    fn test_data_type_display() {
        assert_eq!(format!("{}", DataType::Integer), "INTEGER");
        assert_eq!(format!("{}", DataType::Varchar(255)), "VARCHAR(255)");
        assert_eq!(format!("{}", DataType::Decimal { precision: 10, scale: 2 }), "DECIMAL(10,2)");
    }
}
