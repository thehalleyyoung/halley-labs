//! Avro schema parsing, compatibility checking, and diff computation.
//!
//! Implements the Apache Avro specification for schema representation,
//! resolution rules, and forward/backward compatibility analysis.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::{Result, SafeStepError};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Represents an Avro schema, mirroring the Avro specification types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "_avro_type")]
pub enum AvroSchema {
    Null,
    Boolean,
    Int,
    Long,
    Float,
    Double,
    String,
    Bytes,
    Record(AvroRecord),
    Enum(AvroEnum),
    Array { items: Box<AvroSchema> },
    Map { values: Box<AvroSchema> },
    Union(Vec<AvroSchema>),
    Fixed(AvroFixed),
    /// Named-type reference (used inside unions or self-referencing schemas).
    Ref(std::string::String),
}

/// An Avro record schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvroRecord {
    pub name: std::string::String,
    pub namespace: Option<std::string::String>,
    pub doc: Option<std::string::String>,
    pub aliases: Vec<std::string::String>,
    pub fields: Vec<AvroField>,
}

/// A single field inside an Avro record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvroField {
    pub name: std::string::String,
    #[serde(rename = "type")]
    pub type_: AvroSchema,
    pub default: Option<Value>,
    pub order: FieldOrder,
    pub aliases: Vec<std::string::String>,
    pub doc: Option<std::string::String>,
}

/// Sort order for record fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FieldOrder {
    Ascending,
    Descending,
    Ignore,
}

/// An Avro enum schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvroEnum {
    pub name: std::string::String,
    pub namespace: Option<std::string::String>,
    pub doc: Option<std::string::String>,
    pub aliases: Vec<std::string::String>,
    pub symbols: Vec<std::string::String>,
    pub default: Option<std::string::String>,
}

/// An Avro fixed-size binary schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvroFixed {
    pub name: std::string::String,
    pub namespace: Option<std::string::String>,
    pub size: usize,
    pub aliases: Vec<std::string::String>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

impl AvroSchema {
    /// Parse an Avro JSON schema string into an `AvroSchema`.
    pub fn parse(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| SafeStepError::schema(format!("Invalid JSON: {e}")))?;
        Self::from_value(&value)
    }

    /// Recursively build an `AvroSchema` from a `serde_json::Value`.
    pub fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::String(s) => Self::parse_type_string(s),
            Value::Object(map) => Self::parse_complex(map),
            Value::Array(arr) => Self::parse_union(arr),
            _ => Err(SafeStepError::schema(format!(
                "Unsupported Avro schema value: {value}"
            ))),
        }
    }

    /// Return the Avro type name as a string slice.
    pub fn type_name(&self) -> &str {
        match self {
            Self::Null => "null",
            Self::Boolean => "boolean",
            Self::Int => "int",
            Self::Long => "long",
            Self::Float => "float",
            Self::Double => "double",
            Self::String => "string",
            Self::Bytes => "bytes",
            Self::Record(_) => "record",
            Self::Enum(_) => "enum",
            Self::Array { .. } => "array",
            Self::Map { .. } => "map",
            Self::Union(_) => "union",
            Self::Fixed(_) => "fixed",
            Self::Ref(_) => "ref",
        }
    }

    /// True for null, boolean, int, long, float, double, string, bytes.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Self::Null
                | Self::Boolean
                | Self::Int
                | Self::Long
                | Self::Float
                | Self::Double
                | Self::String
                | Self::Bytes
        )
    }

    /// True for record, enum, array, map, union, fixed.
    pub fn is_complex(&self) -> bool {
        !self.is_primitive() && !matches!(self, Self::Ref(_))
    }

    /// True for record, enum, fixed (types that carry a name).
    pub fn is_named(&self) -> bool {
        matches!(self, Self::Record(_) | Self::Enum(_) | Self::Fixed(_))
    }

    /// Return the fully-qualified name for named types, or `None`.
    pub fn full_name(&self) -> Option<std::string::String> {
        match self {
            Self::Record(r) => Some(r.full_name()),
            Self::Enum(e) => Some(e.full_name()),
            Self::Fixed(f) => Some(f.full_name()),
            _ => None,
        }
    }

    // -- internal helpers ---------------------------------------------------

    fn parse_type_string(s: &str) -> Result<Self> {
        match s {
            "null" => Ok(Self::Null),
            "boolean" => Ok(Self::Boolean),
            "int" => Ok(Self::Int),
            "long" => Ok(Self::Long),
            "float" => Ok(Self::Float),
            "double" => Ok(Self::Double),
            "string" => Ok(Self::String),
            "bytes" => Ok(Self::Bytes),
            other => Ok(Self::Ref(other.to_owned())),
        }
    }

    fn parse_complex(map: &serde_json::Map<std::string::String, Value>) -> Result<Self> {
        let type_str = map
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| SafeStepError::schema("Complex schema missing 'type' field"))?;

        match type_str {
            // Primitive specified as {"type": "int"} etc.
            "null" => Ok(Self::Null),
            "boolean" => Ok(Self::Boolean),
            "int" => Ok(Self::Int),
            "long" => Ok(Self::Long),
            "float" => Ok(Self::Float),
            "double" => Ok(Self::Double),
            "string" => Ok(Self::String),
            "bytes" => Ok(Self::Bytes),
            "record" => Self::parse_record(map),
            "enum" => Self::parse_enum(map),
            "array" => Self::parse_array(map),
            "map" => Self::parse_map(map),
            "fixed" => Self::parse_fixed(map),
            other => Err(SafeStepError::schema(format!(
                "Unknown Avro type: '{other}'"
            ))),
        }
    }

    fn parse_record(map: &serde_json::Map<std::string::String, Value>) -> Result<Self> {
        let name = map
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| SafeStepError::schema("Record missing 'name'"))?
            .to_owned();

        let namespace = map.get("namespace").and_then(Value::as_str).map(str::to_owned);
        let doc = map.get("doc").and_then(Value::as_str).map(str::to_owned);
        let aliases = Self::parse_string_array(map.get("aliases"));

        let fields_val = map
            .get("fields")
            .and_then(Value::as_array)
            .ok_or_else(|| SafeStepError::schema("Record missing 'fields' array"))?;

        let mut fields = Vec::with_capacity(fields_val.len());
        for fv in fields_val {
            fields.push(AvroField::from_value(fv)?);
        }

        Ok(Self::Record(AvroRecord {
            name,
            namespace,
            doc,
            aliases,
            fields,
        }))
    }

    fn parse_enum(map: &serde_json::Map<std::string::String, Value>) -> Result<Self> {
        let name = map
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| SafeStepError::schema("Enum missing 'name'"))?
            .to_owned();

        let namespace = map.get("namespace").and_then(Value::as_str).map(str::to_owned);
        let doc = map.get("doc").and_then(Value::as_str).map(str::to_owned);
        let aliases = Self::parse_string_array(map.get("aliases"));

        let symbols = map
            .get("symbols")
            .and_then(Value::as_array)
            .ok_or_else(|| SafeStepError::schema("Enum missing 'symbols' array"))?
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect::<Vec<_>>();

        if symbols.is_empty() {
            return Err(SafeStepError::schema("Enum 'symbols' must not be empty"));
        }

        let default = map.get("default").and_then(Value::as_str).map(str::to_owned);

        Ok(Self::Enum(AvroEnum {
            name,
            namespace,
            doc,
            aliases,
            symbols,
            default,
        }))
    }

    fn parse_array(map: &serde_json::Map<std::string::String, Value>) -> Result<Self> {
        let items_val = map
            .get("items")
            .ok_or_else(|| SafeStepError::schema("Array missing 'items'"))?;
        let items = Self::from_value(items_val)?;
        Ok(Self::Array {
            items: Box::new(items),
        })
    }

    fn parse_map(map: &serde_json::Map<std::string::String, Value>) -> Result<Self> {
        let values_val = map
            .get("values")
            .ok_or_else(|| SafeStepError::schema("Map missing 'values'"))?;
        let values = Self::from_value(values_val)?;
        Ok(Self::Map {
            values: Box::new(values),
        })
    }

    fn parse_fixed(map: &serde_json::Map<std::string::String, Value>) -> Result<Self> {
        let name = map
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| SafeStepError::schema("Fixed missing 'name'"))?
            .to_owned();

        let namespace = map.get("namespace").and_then(Value::as_str).map(str::to_owned);
        let aliases = Self::parse_string_array(map.get("aliases"));

        let size = map
            .get("size")
            .and_then(Value::as_u64)
            .ok_or_else(|| SafeStepError::schema("Fixed missing 'size'"))? as usize;

        Ok(Self::Fixed(AvroFixed {
            name,
            namespace,
            size,
            aliases,
        }))
    }

    fn parse_union(arr: &[Value]) -> Result<Self> {
        let mut variants = Vec::with_capacity(arr.len());
        for v in arr {
            variants.push(Self::from_value(v)?);
        }
        Ok(Self::Union(variants))
    }

    fn parse_string_array(val: Option<&Value>) -> Vec<std::string::String> {
        val.and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// AvroField parsing
// ---------------------------------------------------------------------------

impl AvroField {
    fn from_value(value: &Value) -> Result<Self> {
        let map = value
            .as_object()
            .ok_or_else(|| SafeStepError::schema("Field must be a JSON object"))?;

        let name = map
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| SafeStepError::schema("Field missing 'name'"))?
            .to_owned();

        let type_val = map
            .get("type")
            .ok_or_else(|| SafeStepError::schema(format!("Field '{name}' missing 'type'")))?;
        let type_ = AvroSchema::from_value(type_val)?;

        let default = map.get("default").cloned();

        let order = match map.get("order").and_then(Value::as_str) {
            Some("descending") => FieldOrder::Descending,
            Some("ignore") => FieldOrder::Ignore,
            _ => FieldOrder::Ascending,
        };

        let aliases = AvroSchema::parse_string_array(map.get("aliases"));
        let doc = map.get("doc").and_then(Value::as_str).map(str::to_owned);

        Ok(Self {
            name,
            type_,
            default,
            order,
            aliases,
            doc,
        })
    }
}

// ---------------------------------------------------------------------------
// AvroRecord helpers
// ---------------------------------------------------------------------------

impl AvroRecord {
    /// Fully-qualified name: `namespace.name` if namespace is present.
    pub fn full_name(&self) -> std::string::String {
        match &self.namespace {
            Some(ns) => format!("{ns}.{}", self.name),
            None => self.name.clone(),
        }
    }

    /// Look up a field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&AvroField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

// ---------------------------------------------------------------------------
// AvroEnum helpers
// ---------------------------------------------------------------------------

impl AvroEnum {
    pub fn full_name(&self) -> std::string::String {
        match &self.namespace {
            Some(ns) => format!("{ns}.{}", self.name),
            None => self.name.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// AvroFixed helpers
// ---------------------------------------------------------------------------

impl AvroFixed {
    pub fn full_name(&self) -> std::string::String {
        match &self.namespace {
            Some(ns) => format!("{ns}.{}", self.name),
            None => self.name.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Compatibility checking (Avro spec §Schema Resolution)
// ---------------------------------------------------------------------------

/// Stateless helper for Avro schema compatibility analysis.
pub struct AvroCompatibility;

impl AvroCompatibility {
    /// A *writer* schema is **forward-compatible** with a *reader* when the
    /// reader can successfully decode data written with the writer schema.
    /// Equivalently, the reader can evolve independently.
    pub fn forward_compatible(reader: &AvroSchema, writer: &AvroSchema) -> bool {
        Self::compatibility_errors(reader, writer).is_empty()
    }

    /// A *reader* schema is **backward-compatible** with a *writer* when data
    /// written by the writer can be decoded by the reader.
    pub fn backward_compatible(reader: &AvroSchema, writer: &AvroSchema) -> bool {
        Self::compatibility_errors(reader, writer).is_empty()
    }

    /// Fully compatible means both directions work.
    pub fn full_compatible(schema1: &AvroSchema, schema2: &AvroSchema) -> bool {
        Self::compatibility_errors(schema1, schema2).is_empty()
            && Self::compatibility_errors(schema2, schema1).is_empty()
    }

    /// Return a list of human-readable error messages describing why the
    /// reader cannot decode data produced by the writer. An empty vec means
    /// the schemas are compatible.
    pub fn compatibility_errors(reader: &AvroSchema, writer: &AvroSchema) -> Vec<std::string::String> {
        let mut errors = Vec::new();
        Self::check_compat(reader, writer, "", &mut errors);
        errors
    }

    // -- recursive checker --------------------------------------------------

    fn check_compat(
        reader: &AvroSchema,
        writer: &AvroSchema,
        path: &str,
        errors: &mut Vec<std::string::String>,
    ) {
        // Identical schemas are always compatible.
        if reader == writer {
            return;
        }

        // Writer is a union → each branch must be readable.
        if let AvroSchema::Union(writer_variants) = writer {
            for (i, wv) in writer_variants.iter().enumerate() {
                let sub_path = format!("{path}/union[{i}]");
                if !Self::writer_branch_readable(reader, wv) {
                    errors.push(format!(
                        "{sub_path}: writer union branch '{}' not readable by reader",
                        wv.type_name()
                    ));
                }
            }
            return;
        }

        // Reader is a union → writer type must match at least one branch.
        if let AvroSchema::Union(reader_variants) = reader {
            let matched = reader_variants.iter().any(|rv| {
                Self::compatibility_errors(rv, writer).is_empty()
            });
            if !matched {
                errors.push(format!(
                    "{path}: writer type '{}' does not match any reader union branch",
                    writer.type_name()
                ));
            }
            return;
        }

        // Primitive promotions.
        if Self::is_promotable(writer, reader) {
            return;
        }

        // Type mismatch guard.
        if std::mem::discriminant(reader) != std::mem::discriminant(writer) {
            // Allow Ref to match a named type by name.
            if let AvroSchema::Ref(name) = reader {
                if writer.full_name().as_deref() == Some(name.as_str()) {
                    return;
                }
            }
            if let AvroSchema::Ref(name) = writer {
                if reader.full_name().as_deref() == Some(name.as_str()) {
                    return;
                }
            }
            errors.push(format!(
                "{path}: type changed from '{}' to '{}'",
                writer.type_name(),
                reader.type_name()
            ));
            return;
        }

        // Same-kind checks.
        match (reader, writer) {
            (AvroSchema::Record(rr), AvroSchema::Record(wr)) => {
                Self::check_record_compat(rr, wr, path, errors);
            }
            (AvroSchema::Enum(re), AvroSchema::Enum(we)) => {
                Self::check_enum_compat(re, we, path, errors);
            }
            (AvroSchema::Array { items: ri }, AvroSchema::Array { items: wi }) => {
                Self::check_compat(ri, wi, &format!("{path}/items"), errors);
            }
            (AvroSchema::Map { values: rv }, AvroSchema::Map { values: wv }) => {
                Self::check_compat(rv, wv, &format!("{path}/values"), errors);
            }
            (AvroSchema::Fixed(rf), AvroSchema::Fixed(wf)) => {
                if rf.size != wf.size {
                    errors.push(format!(
                        "{path}: fixed size changed from {} to {}",
                        wf.size, rf.size
                    ));
                }
            }
            _ => {
                // Primitives of the same kind are compatible (handled by
                // the equality check and promotion above).
            }
        }
    }

    /// Check record compatibility: fields present in the writer must be
    /// compatible in the reader; fields absent from the writer but present in
    /// the reader must have defaults.
    fn check_record_compat(
        reader: &AvroRecord,
        writer: &AvroRecord,
        path: &str,
        errors: &mut Vec<std::string::String>,
    ) {
        let reader_fields: IndexMap<&str, &AvroField> =
            reader.fields.iter().map(|f| (f.name.as_str(), f)).collect();
        let writer_fields: IndexMap<&str, &AvroField> =
            writer.fields.iter().map(|f| (f.name.as_str(), f)).collect();

        // Fields in the reader that are NOT in the writer must have a default.
        for (name, rf) in &reader_fields {
            if !writer_fields.contains_key(name) && rf.default.is_none() {
                errors.push(format!(
                    "{path}: reader field '{name}' has no default but is missing from writer"
                ));
            }
        }

        // Fields that exist in both must be type-compatible.
        for (name, wf) in &writer_fields {
            if let Some(rf) = reader_fields.get(name) {
                Self::check_compat(&rf.type_, &wf.type_, &format!("{path}/{name}"), errors);
            }
            // Fields only in the writer are ignored by the reader (allowed).
        }
    }

    /// Enum compat: every symbol in the writer must exist in the reader.
    fn check_enum_compat(
        reader: &AvroEnum,
        writer: &AvroEnum,
        path: &str,
        errors: &mut Vec<std::string::String>,
    ) {
        for sym in &writer.symbols {
            if !reader.symbols.contains(sym) {
                errors.push(format!(
                    "{path}: writer enum symbol '{sym}' missing from reader"
                ));
            }
        }
    }

    /// Returns true when a value written as `writer` can be promoted to
    /// `reader` per the Avro spec.
    fn is_promotable(writer: &AvroSchema, reader: &AvroSchema) -> bool {
        matches!(
            (writer, reader),
            (AvroSchema::Int, AvroSchema::Long)
                | (AvroSchema::Int, AvroSchema::Float)
                | (AvroSchema::Int, AvroSchema::Double)
                | (AvroSchema::Long, AvroSchema::Float)
                | (AvroSchema::Long, AvroSchema::Double)
                | (AvroSchema::Float, AvroSchema::Double)
        )
    }

    /// Check whether a single writer union branch is readable by the reader.
    fn writer_branch_readable(reader: &AvroSchema, writer_branch: &AvroSchema) -> bool {
        Self::compatibility_errors(reader, writer_branch).is_empty()
    }
}

// ---------------------------------------------------------------------------
// Diff
// ---------------------------------------------------------------------------

/// Compatibility level summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AvroCompatLevel {
    Full,
    Backward,
    Forward,
    None,
}

/// Describes a single field whose type changed between schema versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvroFieldChange {
    pub field_name: std::string::String,
    pub old_type: std::string::String,
    pub new_type: std::string::String,
    pub default_changed: bool,
}

/// The result of diffing two Avro schemas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvroDiffResult {
    pub schema_type_changed: bool,
    pub added_fields: Vec<std::string::String>,
    pub removed_fields: Vec<std::string::String>,
    pub modified_fields: Vec<AvroFieldChange>,
    pub added_symbols: Vec<std::string::String>,
    pub removed_symbols: Vec<std::string::String>,
    pub compatibility: AvroCompatLevel,
}

/// Stateless Avro schema differ.
pub struct AvroDiff;

impl AvroDiff {
    /// Compute the diff between `old` and `new` Avro schemas.
    pub fn diff(old: &AvroSchema, new: &AvroSchema) -> AvroDiffResult {
        let schema_type_changed =
            std::mem::discriminant(old) != std::mem::discriminant(new);

        let mut added_fields: Vec<std::string::String> = Vec::new();
        let mut removed_fields: Vec<std::string::String> = Vec::new();
        let mut modified_fields: Vec<AvroFieldChange> = Vec::new();
        let mut added_symbols: Vec<std::string::String> = Vec::new();
        let mut removed_symbols: Vec<std::string::String> = Vec::new();

        if !schema_type_changed {
            match (old, new) {
                (AvroSchema::Record(old_rec), AvroSchema::Record(new_rec)) => {
                    Self::diff_records(
                        old_rec,
                        new_rec,
                        &mut added_fields,
                        &mut removed_fields,
                        &mut modified_fields,
                    );
                }
                (AvroSchema::Enum(old_enum), AvroSchema::Enum(new_enum)) => {
                    Self::diff_enums(old_enum, new_enum, &mut added_symbols, &mut removed_symbols);
                }
                _ => {}
            }
        }

        let compatibility = Self::classify(old, new);

        AvroDiffResult {
            schema_type_changed,
            added_fields,
            removed_fields,
            modified_fields,
            added_symbols,
            removed_symbols,
            compatibility,
        }
    }

    fn diff_records(
        old_rec: &AvroRecord,
        new_rec: &AvroRecord,
        added: &mut Vec<std::string::String>,
        removed: &mut Vec<std::string::String>,
        modified: &mut Vec<AvroFieldChange>,
    ) {
        let old_map: IndexMap<&str, &AvroField> =
            old_rec.fields.iter().map(|f| (f.name.as_str(), f)).collect();
        let new_map: IndexMap<&str, &AvroField> =
            new_rec.fields.iter().map(|f| (f.name.as_str(), f)).collect();

        for (name, _) in &new_map {
            if !old_map.contains_key(name) {
                added.push((*name).to_owned());
            }
        }

        for (name, _) in &old_map {
            if !new_map.contains_key(name) {
                removed.push((*name).to_owned());
            }
        }

        for (name, old_f) in &old_map {
            if let Some(new_f) = new_map.get(name) {
                let type_changed = old_f.type_ != new_f.type_;
                let default_changed = old_f.default != new_f.default;
                if type_changed || default_changed {
                    modified.push(AvroFieldChange {
                        field_name: (*name).to_owned(),
                        old_type: old_f.type_.type_name().to_owned(),
                        new_type: new_f.type_.type_name().to_owned(),
                        default_changed,
                    });
                }
            }
        }
    }

    fn diff_enums(
        old_enum: &AvroEnum,
        new_enum: &AvroEnum,
        added: &mut Vec<std::string::String>,
        removed: &mut Vec<std::string::String>,
    ) {
        for sym in &new_enum.symbols {
            if !old_enum.symbols.contains(sym) {
                added.push(sym.clone());
            }
        }
        for sym in &old_enum.symbols {
            if !new_enum.symbols.contains(sym) {
                removed.push(sym.clone());
            }
        }
    }

    fn classify(old: &AvroSchema, new: &AvroSchema) -> AvroCompatLevel {
        let backward = AvroCompatibility::compatibility_errors(new, old).is_empty();
        let forward = AvroCompatibility::compatibility_errors(old, new).is_empty();
        match (backward, forward) {
            (true, true) => AvroCompatLevel::Full,
            (true, false) => AvroCompatLevel::Backward,
            (false, true) => AvroCompatLevel::Forward,
            (false, false) => AvroCompatLevel::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Display helpers (used in diffs & error messages)
// ---------------------------------------------------------------------------

impl std::fmt::Display for AvroSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Record(r) => write!(f, "record<{}>", r.full_name()),
            Self::Enum(e) => write!(f, "enum<{}>", e.full_name()),
            Self::Array { items } => write!(f, "array<{items}>"),
            Self::Map { values } => write!(f, "map<{values}>"),
            Self::Union(variants) => {
                let names: Vec<_> = variants.iter().map(|v| v.type_name()).collect();
                write!(f, "union<{}>", names.join(", "))
            }
            Self::Fixed(fx) => write!(f, "fixed<{}, {}>", fx.full_name(), fx.size),
            Self::Ref(name) => write!(f, "ref<{name}>"),
            other => write!(f, "{}", other.type_name()),
        }
    }
}

impl std::fmt::Display for AvroCompatLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "FULL"),
            Self::Backward => write!(f, "BACKWARD"),
            Self::Forward => write!(f, "FORWARD"),
            Self::None => write!(f, "NONE"),
        }
    }
}

impl Default for FieldOrder {
    fn default() -> Self {
        Self::Ascending
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- primitive parsing -------------------------------------------------

    #[test]
    fn parse_null() {
        let schema = AvroSchema::parse("\"null\"").unwrap();
        assert_eq!(schema, AvroSchema::Null);
        assert!(schema.is_primitive());
        assert!(!schema.is_complex());
        assert!(!schema.is_named());
        assert_eq!(schema.type_name(), "null");
    }

    #[test]
    fn parse_boolean() {
        let schema = AvroSchema::parse("\"boolean\"").unwrap();
        assert_eq!(schema, AvroSchema::Boolean);
        assert!(schema.is_primitive());
    }

    #[test]
    fn parse_int() {
        let schema = AvroSchema::parse("\"int\"").unwrap();
        assert_eq!(schema, AvroSchema::Int);
    }

    #[test]
    fn parse_long() {
        let schema = AvroSchema::parse("\"long\"").unwrap();
        assert_eq!(schema, AvroSchema::Long);
    }

    #[test]
    fn parse_float() {
        let schema = AvroSchema::parse("\"float\"").unwrap();
        assert_eq!(schema, AvroSchema::Float);
    }

    #[test]
    fn parse_double() {
        let schema = AvroSchema::parse("\"double\"").unwrap();
        assert_eq!(schema, AvroSchema::Double);
    }

    #[test]
    fn parse_string() {
        let schema = AvroSchema::parse("\"string\"").unwrap();
        assert_eq!(schema, AvroSchema::String);
    }

    #[test]
    fn parse_bytes() {
        let schema = AvroSchema::parse("\"bytes\"").unwrap();
        assert_eq!(schema, AvroSchema::Bytes);
    }

    #[test]
    fn parse_primitive_object_form() {
        let schema = AvroSchema::parse(r#"{"type": "int"}"#).unwrap();
        assert_eq!(schema, AvroSchema::Int);
    }

    // -- record parsing ----------------------------------------------------

    #[test]
    fn parse_record() {
        let json = r#"{
            "type": "record",
            "name": "User",
            "namespace": "com.example",
            "doc": "A user record",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"},
                {"name": "email", "type": ["null", "string"], "default": null}
            ]
        }"#;

        let schema = AvroSchema::parse(json).unwrap();
        assert!(schema.is_named());
        assert!(schema.is_complex());
        assert_eq!(schema.type_name(), "record");
        assert_eq!(schema.full_name(), Some("com.example.User".into()));

        if let AvroSchema::Record(rec) = &schema {
            assert_eq!(rec.name, "User");
            assert_eq!(rec.namespace.as_deref(), Some("com.example"));
            assert_eq!(rec.doc.as_deref(), Some("A user record"));
            assert_eq!(rec.field_count(), 3);

            let id_field = rec.field_by_name("id").unwrap();
            assert_eq!(id_field.type_, AvroSchema::Long);
            assert_eq!(id_field.order, FieldOrder::Ascending);

            let email_field = rec.field_by_name("email").unwrap();
            assert!(matches!(&email_field.type_, AvroSchema::Union(_)));
            assert_eq!(email_field.default, Some(Value::Null));
        } else {
            panic!("Expected Record");
        }
    }

    #[test]
    fn parse_record_field_aliases_and_doc() {
        let json = r#"{
            "type": "record",
            "name": "Event",
            "fields": [
                {
                    "name": "ts",
                    "type": "long",
                    "doc": "event timestamp",
                    "aliases": ["timestamp"],
                    "order": "ignore"
                }
            ]
        }"#;
        let schema = AvroSchema::parse(json).unwrap();
        if let AvroSchema::Record(rec) = &schema {
            let f = rec.field_by_name("ts").unwrap();
            assert_eq!(f.doc.as_deref(), Some("event timestamp"));
            assert_eq!(f.aliases, vec!["timestamp".to_owned()]);
            assert_eq!(f.order, FieldOrder::Ignore);
        } else {
            panic!("Expected Record");
        }
    }

    // -- enum parsing ------------------------------------------------------

    #[test]
    fn parse_enum() {
        let json = r#"{
            "type": "enum",
            "name": "Color",
            "namespace": "com.example",
            "symbols": ["RED", "GREEN", "BLUE"],
            "doc": "Primary colors"
        }"#;
        let schema = AvroSchema::parse(json).unwrap();
        assert_eq!(schema.type_name(), "enum");
        assert!(schema.is_named());
        assert_eq!(schema.full_name(), Some("com.example.Color".into()));

        if let AvroSchema::Enum(e) = &schema {
            assert_eq!(e.symbols, vec!["RED", "GREEN", "BLUE"]);
            assert_eq!(e.doc.as_deref(), Some("Primary colors"));
        } else {
            panic!("Expected Enum");
        }
    }

    #[test]
    fn parse_enum_with_default() {
        let json = r#"{
            "type": "enum",
            "name": "Status",
            "symbols": ["ACTIVE", "INACTIVE"],
            "default": "ACTIVE"
        }"#;
        let schema = AvroSchema::parse(json).unwrap();
        if let AvroSchema::Enum(e) = &schema {
            assert_eq!(e.default.as_deref(), Some("ACTIVE"));
        } else {
            panic!("Expected Enum");
        }
    }

    // -- array & map -------------------------------------------------------

    #[test]
    fn parse_array() {
        let json = r#"{"type": "array", "items": "string"}"#;
        let schema = AvroSchema::parse(json).unwrap();
        assert_eq!(schema.type_name(), "array");
        assert!(schema.is_complex());
        if let AvroSchema::Array { items } = &schema {
            assert_eq!(**items, AvroSchema::String);
        } else {
            panic!("Expected Array");
        }
    }

    #[test]
    fn parse_map() {
        let json = r#"{"type": "map", "values": "long"}"#;
        let schema = AvroSchema::parse(json).unwrap();
        assert_eq!(schema.type_name(), "map");
        if let AvroSchema::Map { values } = &schema {
            assert_eq!(**values, AvroSchema::Long);
        } else {
            panic!("Expected Map");
        }
    }

    // -- union -------------------------------------------------------------

    #[test]
    fn parse_union() {
        let json = r#"["null", "string"]"#;
        let schema = AvroSchema::parse(json).unwrap();
        assert_eq!(schema.type_name(), "union");
        assert!(schema.is_complex());
        if let AvroSchema::Union(variants) = &schema {
            assert_eq!(variants.len(), 2);
            assert_eq!(variants[0], AvroSchema::Null);
            assert_eq!(variants[1], AvroSchema::String);
        } else {
            panic!("Expected Union");
        }
    }

    #[test]
    fn parse_union_with_complex_types() {
        let json = r#"["null", {"type": "array", "items": "int"}, "string"]"#;
        let schema = AvroSchema::parse(json).unwrap();
        if let AvroSchema::Union(variants) = &schema {
            assert_eq!(variants.len(), 3);
            assert_eq!(variants[0], AvroSchema::Null);
            assert!(matches!(&variants[1], AvroSchema::Array { .. }));
            assert_eq!(variants[2], AvroSchema::String);
        } else {
            panic!("Expected Union");
        }
    }

    // -- fixed -------------------------------------------------------------

    #[test]
    fn parse_fixed() {
        let json = r#"{"type": "fixed", "name": "MD5", "size": 16}"#;
        let schema = AvroSchema::parse(json).unwrap();
        assert_eq!(schema.type_name(), "fixed");
        assert!(schema.is_named());
        if let AvroSchema::Fixed(f) = &schema {
            assert_eq!(f.name, "MD5");
            assert_eq!(f.size, 16);
        } else {
            panic!("Expected Fixed");
        }
    }

    // -- nested / complex ---------------------------------------------------

    #[test]
    fn parse_nested_record() {
        let json = r#"{
            "type": "record",
            "name": "Outer",
            "fields": [
                {
                    "name": "inner",
                    "type": {
                        "type": "record",
                        "name": "Inner",
                        "fields": [
                            {"name": "value", "type": "int"}
                        ]
                    }
                },
                {
                    "name": "tags",
                    "type": {"type": "array", "items": "string"}
                },
                {
                    "name": "metadata",
                    "type": {"type": "map", "values": "string"}
                }
            ]
        }"#;
        let schema = AvroSchema::parse(json).unwrap();
        if let AvroSchema::Record(rec) = &schema {
            assert_eq!(rec.field_count(), 3);

            let inner_field = rec.field_by_name("inner").unwrap();
            assert!(matches!(&inner_field.type_, AvroSchema::Record(_)));

            let tags_field = rec.field_by_name("tags").unwrap();
            assert!(matches!(&tags_field.type_, AvroSchema::Array { .. }));

            let meta_field = rec.field_by_name("metadata").unwrap();
            assert!(matches!(&meta_field.type_, AvroSchema::Map { .. }));
        } else {
            panic!("Expected Record");
        }
    }

    // -- backward compatibility --------------------------------------------

    #[test]
    fn backward_compat_added_field_with_default() {
        // Writer has {id, name}. Reader adds email with default → backward ok.
        let writer = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"}
            ]
        }"#).unwrap();

        let reader = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"},
                {"name": "email", "type": "string", "default": "unknown"}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
        assert!(AvroCompatibility::compatibility_errors(&reader, &writer).is_empty());
    }

    #[test]
    fn backward_compat_added_field_without_default_fails() {
        let writer = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"}
            ]
        }"#).unwrap();

        let reader = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"}
            ]
        }"#).unwrap();

        assert!(!AvroCompatibility::backward_compatible(&reader, &writer));
        let errs = AvroCompatibility::compatibility_errors(&reader, &writer);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("name"));
    }

    #[test]
    fn backward_compat_removed_field() {
        // Writer has {id, name, email}. Reader has {id, name} → reader
        // ignores extra writer fields → backward compatible.
        let writer = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"},
                {"name": "email", "type": "string"}
            ]
        }"#).unwrap();

        let reader = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    // -- forward compatibility ---------------------------------------------

    #[test]
    fn forward_compat_writer_adds_field() {
        // Writer adds a field the old reader doesn't know about → forward ok
        // (old reader ignores new field).
        let old_reader = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"}
            ]
        }"#).unwrap();

        let new_writer = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"}
            ]
        }"#).unwrap();

        // Forward: old reader can read new writer's data?
        // reader = old_reader, writer = new_writer
        assert!(AvroCompatibility::forward_compatible(&old_reader, &new_writer));
    }

    #[test]
    fn forward_compat_writer_removes_field_with_default_on_reader() {
        let old_reader = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string", "default": "unknown"}
            ]
        }"#).unwrap();

        let new_writer = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::forward_compatible(&old_reader, &new_writer));
    }

    #[test]
    fn forward_compat_writer_removes_field_without_default_fails() {
        let old_reader = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"}
            ]
        }"#).unwrap();

        let new_writer = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"}
            ]
        }"#).unwrap();

        assert!(!AvroCompatibility::forward_compatible(&old_reader, &new_writer));
    }

    // -- full compatibility ------------------------------------------------

    #[test]
    fn full_compat_identical_schemas() {
        let schema = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string"}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::full_compatible(&schema, &schema));
    }

    #[test]
    fn full_compat_added_field_with_default_on_both() {
        let v1 = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"}
            ]
        }"#).unwrap();

        // v2 adds `name` with a default → v1 reader ignores it (backward),
        // v2 reader fills in default (forward).
        // But v1→v2: v2 reader has 'name' field missing from v1 writer, has
        // default → ok. v2→v1: v1 reader doesn't have 'name', ignores it → ok.
        let v2 = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string", "default": ""}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::full_compatible(&v1, &v2));
    }

    // -- promotions --------------------------------------------------------

    #[test]
    fn promotion_int_to_long() {
        let writer = AvroSchema::Int;
        let reader = AvroSchema::Long;
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn promotion_int_to_float() {
        let writer = AvroSchema::Int;
        let reader = AvroSchema::Float;
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn promotion_int_to_double() {
        let writer = AvroSchema::Int;
        let reader = AvroSchema::Double;
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn promotion_long_to_double() {
        let writer = AvroSchema::Long;
        let reader = AvroSchema::Double;
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn promotion_float_to_double() {
        let writer = AvroSchema::Float;
        let reader = AvroSchema::Double;
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn no_demotion_long_to_int() {
        let writer = AvroSchema::Long;
        let reader = AvroSchema::Int;
        assert!(!AvroCompatibility::backward_compatible(&reader, &writer));
    }

    // -- incompatible changes ----------------------------------------------

    #[test]
    fn incompatible_type_change() {
        let old = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [{"name": "id", "type": "long"}]
        }"#).unwrap();

        let new = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [{"name": "id", "type": "string"}]
        }"#).unwrap();

        assert!(!AvroCompatibility::backward_compatible(&new, &old));
        let errs = AvroCompatibility::compatibility_errors(&new, &old);
        assert!(errs.iter().any(|e| e.contains("type changed")));
    }

    #[test]
    fn incompatible_primitive_to_record() {
        assert!(!AvroCompatibility::backward_compatible(
            &AvroSchema::String,
            &AvroSchema::Int
        ));
    }

    // -- enum compatibility ------------------------------------------------

    #[test]
    fn enum_backward_compat_added_symbol() {
        let writer = AvroSchema::parse(r#"{
            "type": "enum", "name": "Color",
            "symbols": ["RED", "GREEN"]
        }"#).unwrap();
        let reader = AvroSchema::parse(r#"{
            "type": "enum", "name": "Color",
            "symbols": ["RED", "GREEN", "BLUE"]
        }"#).unwrap();

        // Reader has all writer symbols → backward ok.
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn enum_backward_compat_removed_symbol_fails() {
        let writer = AvroSchema::parse(r#"{
            "type": "enum", "name": "Color",
            "symbols": ["RED", "GREEN", "BLUE"]
        }"#).unwrap();
        let reader = AvroSchema::parse(r#"{
            "type": "enum", "name": "Color",
            "symbols": ["RED", "GREEN"]
        }"#).unwrap();

        // Writer has BLUE but reader doesn't → backward fail.
        assert!(!AvroCompatibility::backward_compatible(&reader, &writer));
    }

    // -- array / map compatibility -----------------------------------------

    #[test]
    fn array_compat_promoted_items() {
        let writer = AvroSchema::Array {
            items: Box::new(AvroSchema::Int),
        };
        let reader = AvroSchema::Array {
            items: Box::new(AvroSchema::Long),
        };
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn map_compat_promoted_values() {
        let writer = AvroSchema::Map {
            values: Box::new(AvroSchema::Float),
        };
        let reader = AvroSchema::Map {
            values: Box::new(AvroSchema::Double),
        };
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    // -- union compatibility -----------------------------------------------

    #[test]
    fn union_writer_branch_must_be_in_reader() {
        let writer = AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::String]);
        let reader = AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::String]);
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn union_writer_has_extra_branch_fails() {
        let writer = AvroSchema::Union(vec![
            AvroSchema::Null,
            AvroSchema::String,
            AvroSchema::Int,
        ]);
        let reader = AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::String]);
        assert!(!AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn union_reader_wider_ok() {
        let writer = AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::String]);
        let reader = AvroSchema::Union(vec![
            AvroSchema::Null,
            AvroSchema::String,
            AvroSchema::Int,
        ]);
        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    // -- diff tests --------------------------------------------------------

    #[test]
    fn diff_record_added_field() {
        let old = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [{"name": "id", "type": "long"}]
        }"#).unwrap();
        let new = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "name", "type": "string", "default": ""}
            ]
        }"#).unwrap();

        let result = AvroDiff::diff(&old, &new);
        assert!(!result.schema_type_changed);
        assert_eq!(result.added_fields, vec!["name"]);
        assert!(result.removed_fields.is_empty());
        assert!(result.modified_fields.is_empty());
        assert_eq!(result.compatibility, AvroCompatLevel::Full);
    }

    #[test]
    fn diff_record_removed_field() {
        let old = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "email", "type": "string"}
            ]
        }"#).unwrap();
        let new = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [{"name": "id", "type": "long"}]
        }"#).unwrap();

        let result = AvroDiff::diff(&old, &new);
        assert_eq!(result.removed_fields, vec!["email"]);
        assert!(result.added_fields.is_empty());
    }

    #[test]
    fn diff_record_modified_field() {
        let old = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [{"name": "id", "type": "int"}]
        }"#).unwrap();
        let new = AvroSchema::parse(r#"{
            "type": "record", "name": "User",
            "fields": [{"name": "id", "type": "long"}]
        }"#).unwrap();

        let result = AvroDiff::diff(&old, &new);
        assert_eq!(result.modified_fields.len(), 1);
        assert_eq!(result.modified_fields[0].field_name, "id");
        assert_eq!(result.modified_fields[0].old_type, "int");
        assert_eq!(result.modified_fields[0].new_type, "long");
        // int→long is a valid promotion in both directions? No, long→int is
        // not promotable, so this is Backward only.
        assert_eq!(result.compatibility, AvroCompatLevel::Backward);
    }

    #[test]
    fn diff_enum_added_removed_symbols() {
        let old = AvroSchema::parse(r#"{
            "type": "enum", "name": "Color",
            "symbols": ["RED", "GREEN", "BLUE"]
        }"#).unwrap();
        let new = AvroSchema::parse(r#"{
            "type": "enum", "name": "Color",
            "symbols": ["RED", "GREEN", "YELLOW"]
        }"#).unwrap();

        let result = AvroDiff::diff(&old, &new);
        assert_eq!(result.added_symbols, vec!["YELLOW"]);
        assert_eq!(result.removed_symbols, vec!["BLUE"]);
        assert_eq!(result.compatibility, AvroCompatLevel::None);
    }

    #[test]
    fn diff_type_changed() {
        let old = AvroSchema::Int;
        let new = AvroSchema::String;
        let result = AvroDiff::diff(&old, &new);
        assert!(result.schema_type_changed);
        assert_eq!(result.compatibility, AvroCompatLevel::None);
    }

    // -- edge cases --------------------------------------------------------

    #[test]
    fn parse_error_invalid_json() {
        assert!(AvroSchema::parse("not json").is_err());
    }

    #[test]
    fn parse_error_missing_name() {
        assert!(AvroSchema::parse(r#"{"type": "record", "fields": []}"#).is_err());
    }

    #[test]
    fn parse_error_missing_fields() {
        assert!(AvroSchema::parse(r#"{"type": "record", "name": "X"}"#).is_err());
    }

    #[test]
    fn parse_error_empty_enum_symbols() {
        assert!(AvroSchema::parse(r#"{"type": "enum", "name": "E", "symbols": []}"#).is_err());
    }

    #[test]
    fn ref_type_string() {
        let schema = AvroSchema::parse("\"com.example.User\"").unwrap();
        assert_eq!(schema, AvroSchema::Ref("com.example.User".into()));
        assert!(!schema.is_primitive());
        assert!(!schema.is_named());
    }

    #[test]
    fn display_schemas() {
        assert_eq!(format!("{}", AvroSchema::Int), "int");
        assert_eq!(format!("{}", AvroSchema::String), "string");
        assert_eq!(
            format!(
                "{}",
                AvroSchema::Array {
                    items: Box::new(AvroSchema::Int)
                }
            ),
            "array<int>"
        );
        assert_eq!(
            format!(
                "{}",
                AvroSchema::Map {
                    values: Box::new(AvroSchema::String)
                }
            ),
            "map<string>"
        );
    }

    #[test]
    fn fixed_compat_different_sizes_fail() {
        let a = AvroSchema::Fixed(AvroFixed {
            name: "Hash".into(),
            namespace: None,
            size: 16,
            aliases: vec![],
        });
        let b = AvroSchema::Fixed(AvroFixed {
            name: "Hash".into(),
            namespace: None,
            size: 32,
            aliases: vec![],
        });
        assert!(!AvroCompatibility::backward_compatible(&a, &b));
    }

    #[test]
    fn full_compat_field_default_added_and_removed() {
        // Scenario: v1 has {a, b}. v2 has {a, c(default)}.
        // backward (v2 reads v1): c has default, b is ignored → ok.
        // forward (v1 reads v2): b missing from v2 writer, no default → fail.
        let v1 = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [
                {"name": "a", "type": "int"},
                {"name": "b", "type": "int"}
            ]
        }"#).unwrap();
        let v2 = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [
                {"name": "a", "type": "int"},
                {"name": "c", "type": "int", "default": 0}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::backward_compatible(&v2, &v1));
        // Forward: v1 reads v2 data — v1 has field b with no default, missing
        // from v2 writer → fail.
        assert!(!AvroCompatibility::forward_compatible(&v1, &v2));
        assert!(!AvroCompatibility::full_compatible(&v1, &v2));
    }

    #[test]
    fn diff_record_default_change() {
        let old = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [{"name": "x", "type": "int", "default": 0}]
        }"#).unwrap();
        let new = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [{"name": "x", "type": "int", "default": 42}]
        }"#).unwrap();

        let result = AvroDiff::diff(&old, &new);
        assert_eq!(result.modified_fields.len(), 1);
        assert!(result.modified_fields[0].default_changed);
        assert_eq!(result.modified_fields[0].old_type, "int");
        assert_eq!(result.modified_fields[0].new_type, "int");
    }

    #[test]
    fn promotion_in_record_field() {
        let writer = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [{"name": "val", "type": "int"}]
        }"#).unwrap();
        let reader = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [{"name": "val", "type": "long"}]
        }"#).unwrap();

        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }

    #[test]
    fn nested_array_compat() {
        let writer = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [
                {"name": "nums", "type": {"type": "array", "items": "int"}}
            ]
        }"#).unwrap();
        let reader = AvroSchema::parse(r#"{
            "type": "record", "name": "R",
            "fields": [
                {"name": "nums", "type": {"type": "array", "items": "long"}}
            ]
        }"#).unwrap();

        assert!(AvroCompatibility::backward_compatible(&reader, &writer));
    }
}
