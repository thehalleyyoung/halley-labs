//! OpenAPI schema diffing and breaking-change detection.
//!
//! Compares two [`OpenApiSchema`] instances and produces a detailed
//! [`SchemaDiff`] that classifies every change as breaking or non-breaking.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::openapi::{
    OpenApiOperation, OpenApiParameter, OpenApiPath, OpenApiSchema,
    SchemaObject,
};

// ---------------------------------------------------------------------------
// ChangeClassification
// ---------------------------------------------------------------------------

/// How a single schema change affects wire-level compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeClassification {
    Breaking,
    NonBreaking,
    Compatible,
    Unknown,
}

impl fmt::Display for ChangeClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Breaking => write!(f, "breaking"),
            Self::NonBreaking => write!(f, "non-breaking"),
            Self::Compatible => write!(f, "compatible"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl ChangeClassification {
    pub fn is_breaking(self) -> bool {
        matches!(self, Self::Breaking)
    }

    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Breaking, _) | (_, Self::Breaking) => Self::Breaking,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::NonBreaking, _) | (_, Self::NonBreaking) => Self::NonBreaking,
            _ => Self::Compatible,
        }
    }
}

// ---------------------------------------------------------------------------
// SchemaChange
// ---------------------------------------------------------------------------

/// A single atomic change observed between two schema objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaChange {
    TypeChanged {
        path: String,
        from: String,
        to: String,
    },
    FieldAdded {
        path: String,
        field_name: String,
        required: bool,
    },
    FieldRemoved {
        path: String,
        field_name: String,
        was_required: bool,
    },
    EnumValueAdded {
        path: String,
        value: String,
    },
    EnumValueRemoved {
        path: String,
        value: String,
    },
    ConstraintTightened {
        path: String,
        constraint: String,
        old_value: String,
        new_value: String,
    },
    ConstraintLoosened {
        path: String,
        constraint: String,
        old_value: String,
        new_value: String,
    },
    FormatChanged {
        path: String,
        from: Option<String>,
        to: Option<String>,
    },
    NullabilityChanged {
        path: String,
        was_nullable: bool,
        now_nullable: bool,
    },
    RequirednessChanged {
        path: String,
        field_name: String,
        was_required: bool,
        now_required: bool,
    },
    DescriptionChanged {
        path: String,
    },
    DefaultChanged {
        path: String,
    },
}

impl SchemaChange {
    pub fn path(&self) -> &str {
        match self {
            Self::TypeChanged { path, .. }
            | Self::FieldAdded { path, .. }
            | Self::FieldRemoved { path, .. }
            | Self::EnumValueAdded { path, .. }
            | Self::EnumValueRemoved { path, .. }
            | Self::ConstraintTightened { path, .. }
            | Self::ConstraintLoosened { path, .. }
            | Self::FormatChanged { path, .. }
            | Self::NullabilityChanged { path, .. }
            | Self::RequirednessChanged { path, .. }
            | Self::DescriptionChanged { path, .. }
            | Self::DefaultChanged { path, .. } => path.as_str(),
        }
    }
}

impl fmt::Display for SchemaChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeChanged { path, from, to } => {
                write!(f, "Type changed at {path}: {from} -> {to}")
            }
            Self::FieldAdded { path, field_name, required } => {
                let req = if *required { " (required)" } else { "" };
                write!(f, "Field added at {path}: {field_name}{req}")
            }
            Self::FieldRemoved { path, field_name, was_required } => {
                let req = if *was_required { " (was required)" } else { "" };
                write!(f, "Field removed at {path}: {field_name}{req}")
            }
            Self::EnumValueAdded { path, value } => {
                write!(f, "Enum value added at {path}: {value}")
            }
            Self::EnumValueRemoved { path, value } => {
                write!(f, "Enum value removed at {path}: {value}")
            }
            Self::ConstraintTightened { path, constraint, old_value, new_value } => {
                write!(f, "Constraint tightened at {path}: {constraint} {old_value}->{new_value}")
            }
            Self::ConstraintLoosened { path, constraint, old_value, new_value } => {
                write!(f, "Constraint loosened at {path}: {constraint} {old_value}->{new_value}")
            }
            Self::FormatChanged { path, from, to } => {
                write!(f, "Format changed at {path}: {:?} -> {:?}", from, to)
            }
            Self::NullabilityChanged { path, was_nullable, now_nullable } => {
                write!(f, "Nullability changed at {path}: {was_nullable} -> {now_nullable}")
            }
            Self::RequirednessChanged { path, field_name, was_required, now_required } => {
                write!(f, "Requiredness changed at {path}/{field_name}: {was_required} -> {now_required}")
            }
            Self::DescriptionChanged { path } => write!(f, "Description changed at {path}"),
            Self::DefaultChanged { path } => write!(f, "Default changed at {path}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ParameterChange
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterChange {
    Added {
        name: String,
        location: String,
        required: bool,
    },
    Removed {
        name: String,
        location: String,
    },
    TypeChanged {
        name: String,
        from: String,
        to: String,
    },
    RequiredChanged {
        name: String,
        from: bool,
        to: bool,
    },
    DeprecatedChanged {
        name: String,
        from: bool,
        to: bool,
    },
}

impl fmt::Display for ParameterChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Added { name, required, .. } => {
                let req = if *required { " (required)" } else { "" };
                write!(f, "Parameter added: {name}{req}")
            }
            Self::Removed { name, .. } => write!(f, "Parameter removed: {name}"),
            Self::TypeChanged { name, from, to } => {
                write!(f, "Parameter type changed {name}: {from} -> {to}")
            }
            Self::RequiredChanged { name, from, to } => {
                write!(f, "Parameter required changed {name}: {from} -> {to}")
            }
            Self::DeprecatedChanged { name, from, to } => {
                write!(f, "Parameter deprecated changed {name}: {from} -> {to}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OperationDiff / PathDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationDiff {
    pub method: String,
    pub parameter_changes: Vec<ParameterChange>,
    pub request_body_changes: Vec<SchemaChange>,
    pub response_changes: Vec<SchemaChange>,
    pub deprecated_changed: Option<(bool, bool)>,
}

impl OperationDiff {
    pub fn is_empty(&self) -> bool {
        self.parameter_changes.is_empty()
            && self.request_body_changes.is_empty()
            && self.response_changes.is_empty()
            && self.deprecated_changed.is_none()
    }

    pub fn has_breaking_changes(&self) -> bool {
        let bd = BreakingChangeDetector::new();
        self.parameter_changes.iter().any(|c| bd.is_parameter_change_breaking(c))
            || self.request_body_changes.iter().any(|c| bd.classify_schema_change(c).is_breaking())
            || self.response_changes.iter().any(|c| bd.classify_schema_change(c).is_breaking())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathDiff {
    pub path: String,
    pub added_operations: Vec<String>,
    pub removed_operations: Vec<String>,
    pub modified_operations: Vec<OperationDiff>,
}

impl PathDiff {
    pub fn is_empty(&self) -> bool {
        self.added_operations.is_empty()
            && self.removed_operations.is_empty()
            && self.modified_operations.is_empty()
    }

    pub fn has_breaking_changes(&self) -> bool {
        !self.removed_operations.is_empty()
            || self.modified_operations.iter().any(|od| od.has_breaking_changes())
    }
}

// ---------------------------------------------------------------------------
// SchemaDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaDiff {
    pub added_paths: Vec<String>,
    pub removed_paths: Vec<String>,
    pub modified_paths: Vec<PathDiff>,
    pub added_schemas: Vec<String>,
    pub removed_schemas: Vec<String>,
    pub modified_schemas: Vec<(String, Vec<SchemaChange>)>,
    pub breaking_changes: Vec<String>,
    pub non_breaking_changes: Vec<String>,
}

impl SchemaDiff {
    pub fn is_empty(&self) -> bool {
        self.added_paths.is_empty()
            && self.removed_paths.is_empty()
            && self.modified_paths.is_empty()
            && self.added_schemas.is_empty()
            && self.removed_schemas.is_empty()
            && self.modified_schemas.is_empty()
    }

    pub fn breaking_count(&self) -> usize {
        self.breaking_changes.len()
    }

    pub fn non_breaking_count(&self) -> usize {
        self.non_breaking_changes.len()
    }

    pub fn has_breaking_changes(&self) -> bool {
        !self.breaking_changes.is_empty()
    }

    pub fn total_changes(&self) -> usize {
        self.added_paths.len()
            + self.removed_paths.len()
            + self.modified_paths.len()
            + self.added_schemas.len()
            + self.removed_schemas.len()
            + self.modified_schemas.len()
    }

    pub fn summary(&self) -> String {
        format!(
            "Diff: +{}/-{} paths, +{}/-{} schemas, {} breaking, {} non-breaking",
            self.added_paths.len(),
            self.removed_paths.len(),
            self.added_schemas.len(),
            self.removed_schemas.len(),
            self.breaking_changes.len(),
            self.non_breaking_changes.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// OpenApiDiff — core diffing engine
// ---------------------------------------------------------------------------

pub struct OpenApiDiff;

impl OpenApiDiff {
    /// Compute a full diff between two OpenAPI schemas.
    pub fn diff(old: &OpenApiSchema, new: &OpenApiSchema) -> SchemaDiff {
        let mut result = SchemaDiff::default();
        let detector = BreakingChangeDetector::new();

        // --- paths ---
        for key in old.paths.keys() {
            if !new.paths.contains_key(key) {
                result.removed_paths.push(key.clone());
                result.breaking_changes.push(format!("Removed endpoint path: {key}"));
            }
        }
        for key in new.paths.keys() {
            if !old.paths.contains_key(key) {
                result.added_paths.push(key.clone());
                result.non_breaking_changes.push(format!("Added endpoint path: {key}"));
            }
        }
        let path_diffs = Self::diff_paths(&old.paths, &new.paths);
        for pd in &path_diffs {
            for rm in &pd.removed_operations {
                result.breaking_changes.push(format!(
                    "Removed operation {} on {}",
                    rm, pd.path
                ));
            }
            for od in &pd.modified_operations {
                for pc in &od.parameter_changes {
                    let cls = detector.is_parameter_change_breaking(pc);
                    if cls {
                        result.breaking_changes.push(format!("{pc}"));
                    } else {
                        result.non_breaking_changes.push(format!("{pc}"));
                    }
                }
                for sc in &od.request_body_changes {
                    if detector.classify_schema_change(sc).is_breaking() {
                        result.breaking_changes.push(format!("{sc}"));
                    } else {
                        result.non_breaking_changes.push(format!("{sc}"));
                    }
                }
                for sc in &od.response_changes {
                    if detector.classify_schema_change(sc).is_breaking() {
                        result.breaking_changes.push(format!("{sc}"));
                    } else {
                        result.non_breaking_changes.push(format!("{sc}"));
                    }
                }
            }
        }
        result.modified_paths = path_diffs;

        // --- component schemas ---
        for name in old.components.keys() {
            if !new.components.contains_key(name) {
                result.removed_schemas.push(name.clone());
            }
        }
        for name in new.components.keys() {
            if !old.components.contains_key(name) {
                result.added_schemas.push(name.clone());
            }
        }
        for (name, old_schema) in &old.components {
            if let Some(new_schema) = new.components.get(name) {
                let changes = Self::diff_schemas(old_schema, new_schema, name);
                if !changes.is_empty() {
                    result.modified_schemas.push((name.clone(), changes));
                }
            }
        }

        result
    }

    pub fn diff_paths(
        old_paths: &IndexMap<String, OpenApiPath>,
        new_paths: &IndexMap<String, OpenApiPath>,
    ) -> Vec<PathDiff> {
        let mut diffs = Vec::new();
        for (path, old_pi) in old_paths {
            if let Some(new_pi) = new_paths.get(path) {
                let pd = Self::diff_single_path(path, old_pi, new_pi);
                if !pd.is_empty() {
                    diffs.push(pd);
                }
            }
        }
        diffs
    }

    fn diff_single_path(path: &str, old: &OpenApiPath, new: &OpenApiPath) -> PathDiff {
        let old_methods = Self::path_method_map(old);
        let new_methods = Self::path_method_map(new);

        let mut pd = PathDiff {
            path: path.to_string(),
            ..Default::default()
        };

        for (method, _) in &old_methods {
            if !new_methods.contains_key(method) {
                pd.removed_operations.push(method.to_string());
            }
        }
        for (method, _) in &new_methods {
            if !old_methods.contains_key(method) {
                pd.added_operations.push(method.to_string());
            }
        }
        for (method, old_op) in &old_methods {
            if let Some(new_op) = new_methods.get(method) {
                let od = Self::diff_operations(method, old_op, new_op, path);
                if !od.is_empty() {
                    pd.modified_operations.push(od);
                }
            }
        }

        pd
    }

    fn path_method_map(pi: &OpenApiPath) -> IndexMap<String, &OpenApiOperation> {
        let mut map = IndexMap::new();
        for (method, op) in pi.operations() {
            map.insert(method.to_string(), op);
        }
        map
    }

    pub fn diff_operations(
        method: &str,
        old_op: &OpenApiOperation,
        new_op: &OpenApiOperation,
        context_path: &str,
    ) -> OperationDiff {
        let mut od = OperationDiff {
            method: method.to_string(),
            ..Default::default()
        };

        // Parameters
        od.parameter_changes = Self::diff_parameters(&old_op.parameters, &new_op.parameters);

        // Request body
        if let (Some(old_rb), Some(new_rb)) = (&old_op.request_body, &new_op.request_body) {
            for (media_type, old_mt) in &old_rb.content {
                if let Some(new_mt) = new_rb.content.get(media_type) {
                    if let (Some(old_s), Some(new_s)) = (&old_mt.schema, &new_mt.schema) {
                        let prefix = format!("{context_path}/{method}/requestBody/{media_type}");
                        od.request_body_changes
                            .extend(Self::diff_schemas(old_s, new_s, &prefix));
                    }
                }
            }
            if old_rb.required != new_rb.required {
                od.request_body_changes.push(SchemaChange::RequirednessChanged {
                    path: format!("{context_path}/{method}/requestBody"),
                    field_name: "body".to_string(),
                    was_required: old_rb.required,
                    now_required: new_rb.required,
                });
            }
        } else if old_op.request_body.is_some() && new_op.request_body.is_none() {
            od.request_body_changes.push(SchemaChange::FieldRemoved {
                path: format!("{context_path}/{method}"),
                field_name: "requestBody".to_string(),
                was_required: old_op
                    .request_body
                    .as_ref()
                    .map(|rb| rb.required)
                    .unwrap_or(false),
            });
        } else if old_op.request_body.is_none() && new_op.request_body.is_some() {
            od.request_body_changes.push(SchemaChange::FieldAdded {
                path: format!("{context_path}/{method}"),
                field_name: "requestBody".to_string(),
                required: new_op
                    .request_body
                    .as_ref()
                    .map(|rb| rb.required)
                    .unwrap_or(false),
            });
        }

        // Responses
        for (code, old_resp) in &old_op.responses {
            if let Some(new_resp) = new_op.responses.get(code) {
                for (media_type, old_mt) in &old_resp.content {
                    if let Some(new_mt) = new_resp.content.get(media_type) {
                        if let (Some(old_s), Some(new_s)) = (&old_mt.schema, &new_mt.schema) {
                            let prefix =
                                format!("{context_path}/{method}/responses/{code}/{media_type}");
                            od.response_changes
                                .extend(Self::diff_schemas(old_s, new_s, &prefix));
                        }
                    }
                }
            }
        }

        // Deprecated
        if old_op.deprecated != new_op.deprecated {
            od.deprecated_changed = Some((old_op.deprecated, new_op.deprecated));
        }

        od
    }

    fn diff_parameters(
        old_params: &[OpenApiParameter],
        new_params: &[OpenApiParameter],
    ) -> Vec<ParameterChange> {
        let mut changes = Vec::new();
        let old_map: IndexMap<String, &OpenApiParameter> = old_params
            .iter()
            .map(|p| (format!("{}:{:?}", p.name, p.location), p))
            .collect();
        let new_map: IndexMap<String, &OpenApiParameter> = new_params
            .iter()
            .map(|p| (format!("{}:{:?}", p.name, p.location), p))
            .collect();

        for (key, old_p) in &old_map {
            if !new_map.contains_key(key) {
                changes.push(ParameterChange::Removed {
                    name: old_p.name.clone(),
                    location: format!("{:?}", old_p.location),
                });
            }
        }
        for (key, new_p) in &new_map {
            if !old_map.contains_key(key) {
                changes.push(ParameterChange::Added {
                    name: new_p.name.clone(),
                    location: format!("{:?}", new_p.location),
                    required: new_p.required,
                });
            }
        }
        for (key, old_p) in &old_map {
            if let Some(new_p) = new_map.get(key) {
                if old_p.required != new_p.required {
                    changes.push(ParameterChange::RequiredChanged {
                        name: old_p.name.clone(),
                        from: old_p.required,
                        to: new_p.required,
                    });
                }
                let old_type = old_p
                    .schema
                    .as_ref()
                    .and_then(|s| s.type_.as_deref())
                    .unwrap_or("unknown");
                let new_type = new_p
                    .schema
                    .as_ref()
                    .and_then(|s| s.type_.as_deref())
                    .unwrap_or("unknown");
                if old_type != new_type {
                    changes.push(ParameterChange::TypeChanged {
                        name: old_p.name.clone(),
                        from: old_type.to_string(),
                        to: new_type.to_string(),
                    });
                }
                if old_p.deprecated != new_p.deprecated {
                    changes.push(ParameterChange::DeprecatedChanged {
                        name: old_p.name.clone(),
                        from: old_p.deprecated,
                        to: new_p.deprecated,
                    });
                }
            }
        }

        changes
    }

    pub fn diff_schemas(old: &SchemaObject, new: &SchemaObject, prefix: &str) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        let old_type = old.effective_type();
        let new_type = new.effective_type();
        if old_type != new_type {
            changes.push(SchemaChange::TypeChanged {
                path: prefix.to_string(),
                from: old_type.to_string(),
                to: new_type.to_string(),
            });
            return changes;
        }

        if old.format != new.format {
            changes.push(SchemaChange::FormatChanged {
                path: prefix.to_string(),
                from: old.format.clone(),
                to: new.format.clone(),
            });
        }

        if old.nullable != new.nullable {
            changes.push(SchemaChange::NullabilityChanged {
                path: prefix.to_string(),
                was_nullable: old.nullable,
                now_nullable: new.nullable,
            });
        }

        // Properties
        for (name, _old_prop) in &old.properties {
            if !new.properties.contains_key(name) {
                changes.push(SchemaChange::FieldRemoved {
                    path: prefix.to_string(),
                    field_name: name.clone(),
                    was_required: old.required.contains(name),
                });
            }
        }
        for (name, _new_prop) in &new.properties {
            if !old.properties.contains_key(name) {
                changes.push(SchemaChange::FieldAdded {
                    path: prefix.to_string(),
                    field_name: name.clone(),
                    required: new.required.contains(name),
                });
            }
        }
        for (name, old_prop) in &old.properties {
            if let Some(new_prop) = new.properties.get(name) {
                let child_prefix = format!("{prefix}/{name}");
                changes.extend(Self::diff_schemas(old_prop, new_prop, &child_prefix));
            }
        }

        // Requiredness changes for existing fields
        for name in &new.required {
            if !old.required.contains(name) && old.properties.contains_key(name) {
                changes.push(SchemaChange::RequirednessChanged {
                    path: prefix.to_string(),
                    field_name: name.clone(),
                    was_required: false,
                    now_required: true,
                });
            }
        }
        for name in &old.required {
            if !new.required.contains(name) && new.properties.contains_key(name) {
                changes.push(SchemaChange::RequirednessChanged {
                    path: prefix.to_string(),
                    field_name: name.clone(),
                    was_required: true,
                    now_required: false,
                });
            }
        }

        // Enum values
        let old_enums: Vec<String> = old.enum_values.iter().filter_map(|v| v.as_str().map(String::from)).collect();
        let new_enums: Vec<String> = new.enum_values.iter().filter_map(|v| v.as_str().map(String::from)).collect();
        for v in &old_enums {
            if !new_enums.contains(v) {
                changes.push(SchemaChange::EnumValueRemoved {
                    path: prefix.to_string(),
                    value: v.clone(),
                });
            }
        }
        for v in &new_enums {
            if !old_enums.contains(v) {
                changes.push(SchemaChange::EnumValueAdded {
                    path: prefix.to_string(),
                    value: v.clone(),
                });
            }
        }

        // Numeric constraints
        Self::diff_numeric_constraint(&mut changes, prefix, "minimum", old.minimum, new.minimum);
        Self::diff_numeric_constraint(&mut changes, prefix, "maximum", old.maximum, new.maximum);
        Self::diff_usize_constraint(&mut changes, prefix, "minLength", old.min_length, new.min_length);
        Self::diff_usize_constraint(&mut changes, prefix, "maxLength", old.max_length, new.max_length);
        Self::diff_usize_constraint(&mut changes, prefix, "minItems", old.min_items, new.min_items);
        Self::diff_usize_constraint(&mut changes, prefix, "maxItems", old.max_items, new.max_items);

        // Recurse into items
        if let (Some(old_items), Some(new_items)) = (&old.items, &new.items) {
            let child = format!("{prefix}/items");
            changes.extend(Self::diff_schemas(old_items, new_items, &child));
        }

        changes
    }

    fn diff_numeric_constraint(
        changes: &mut Vec<SchemaChange>,
        prefix: &str,
        name: &str,
        old_val: Option<f64>,
        new_val: Option<f64>,
    ) {
        match (old_val, new_val) {
            (Some(o), Some(n)) if (o - n).abs() > f64::EPSILON => {
                let tightened = (name.contains("min") && n > o) || (name.contains("max") && n < o);
                if tightened {
                    changes.push(SchemaChange::ConstraintTightened {
                        path: prefix.to_string(),
                        constraint: name.to_string(),
                        old_value: o.to_string(),
                        new_value: n.to_string(),
                    });
                } else {
                    changes.push(SchemaChange::ConstraintLoosened {
                        path: prefix.to_string(),
                        constraint: name.to_string(),
                        old_value: o.to_string(),
                        new_value: n.to_string(),
                    });
                }
            }
            (None, Some(n)) => {
                changes.push(SchemaChange::ConstraintTightened {
                    path: prefix.to_string(),
                    constraint: name.to_string(),
                    old_value: "none".to_string(),
                    new_value: n.to_string(),
                });
            }
            (Some(o), None) => {
                changes.push(SchemaChange::ConstraintLoosened {
                    path: prefix.to_string(),
                    constraint: name.to_string(),
                    old_value: o.to_string(),
                    new_value: "none".to_string(),
                });
            }
            _ => {}
        }
    }

    fn diff_usize_constraint(
        changes: &mut Vec<SchemaChange>,
        prefix: &str,
        name: &str,
        old_val: Option<usize>,
        new_val: Option<usize>,
    ) {
        Self::diff_numeric_constraint(
            changes,
            prefix,
            name,
            old_val.map(|v| v as f64),
            new_val.map(|v| v as f64),
        );
    }
}

// ---------------------------------------------------------------------------
// BreakingChangeDetector
// ---------------------------------------------------------------------------

pub struct BreakingChangeDetector {
    strict: bool,
}

impl BreakingChangeDetector {
    pub fn new() -> Self {
        Self { strict: false }
    }

    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Classify a single schema change.
    pub fn classify_schema_change(&self, change: &SchemaChange) -> ChangeClassification {
        match change {
            SchemaChange::TypeChanged { .. } => ChangeClassification::Breaking,
            SchemaChange::FieldRemoved { was_required, .. } => {
                if *was_required || self.strict {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::NonBreaking
                }
            }
            SchemaChange::FieldAdded { required, .. } => {
                if *required {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::NonBreaking
                }
            }
            SchemaChange::EnumValueRemoved { .. } => ChangeClassification::Breaking,
            SchemaChange::EnumValueAdded { .. } => {
                if self.strict {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::NonBreaking
                }
            }
            SchemaChange::ConstraintTightened { .. } => ChangeClassification::Breaking,
            SchemaChange::ConstraintLoosened { .. } => ChangeClassification::NonBreaking,
            SchemaChange::FormatChanged { .. } => {
                if self.strict {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::Unknown
                }
            }
            SchemaChange::NullabilityChanged { now_nullable, .. } => {
                if *now_nullable {
                    ChangeClassification::NonBreaking
                } else {
                    ChangeClassification::Breaking
                }
            }
            SchemaChange::RequirednessChanged { now_required, .. } => {
                if *now_required {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::NonBreaking
                }
            }
            SchemaChange::DescriptionChanged { .. } => ChangeClassification::Compatible,
            SchemaChange::DefaultChanged { .. } => ChangeClassification::NonBreaking,
        }
    }

    /// Classify a parameter-level change.
    pub fn is_parameter_change_breaking(&self, change: &ParameterChange) -> bool {
        match change {
            ParameterChange::Added { required, .. } => *required,
            ParameterChange::Removed { .. } => true,
            ParameterChange::TypeChanged { .. } => true,
            ParameterChange::RequiredChanged { to, .. } => *to,
            ParameterChange::DeprecatedChanged { .. } => false,
        }
    }

    /// Check whether a [`SchemaDiff`] contains any breaking changes.
    pub fn is_breaking(diff: &SchemaDiff) -> bool {
        diff.has_breaking_changes()
    }
}

impl Default for BreakingChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SchemaEvolution
// ---------------------------------------------------------------------------

/// Tracks multiple schema versions and computes pairwise compatibility.
pub struct SchemaEvolution {
    versions: Vec<(String, OpenApiSchema)>,
}

impl SchemaEvolution {
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    pub fn add_version(&mut self, version: &str, schema: OpenApiSchema) {
        self.versions.push((version.to_string(), schema));
    }

    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Check backward compatibility between two named versions.
    pub fn is_backward_compatible(&self, old_version: &str, new_version: &str) -> bool {
        let old = self.versions.iter().find(|(v, _)| v == old_version);
        let new = self.versions.iter().find(|(v, _)| v == new_version);
        match (old, new) {
            (Some((_, old_s)), Some((_, new_s))) => {
                let diff = OpenApiDiff::diff(old_s, new_s);
                !diff.has_breaking_changes()
            }
            _ => false,
        }
    }

    /// Compute a full compatibility matrix: (old, new, compatible).
    pub fn compatibility_matrix(&self) -> Vec<(String, String, bool)> {
        let mut matrix = Vec::new();
        for i in 0..self.versions.len() {
            for j in (i + 1)..self.versions.len() {
                let (v_old, s_old) = &self.versions[i];
                let (v_new, s_new) = &self.versions[j];
                let diff = OpenApiDiff::diff(s_old, s_new);
                matrix.push((v_old.clone(), v_new.clone(), !diff.has_breaking_changes()));
            }
        }
        matrix
    }

    /// Return the latest version that is backward-compatible with
    /// the given version, or `None` if none exists.
    pub fn latest_compatible_with(&self, version: &str) -> Option<&str> {
        let base_idx = self.versions.iter().position(|(v, _)| v == version)?;
        let base_schema = &self.versions[base_idx].1;
        for (v, s) in self.versions.iter().rev() {
            if v == version {
                continue;
            }
            let diff = OpenApiDiff::diff(base_schema, s);
            if !diff.has_breaking_changes() {
                return Some(v.as_str());
            }
        }
        None
    }

    /// Return an ordered list of breaking-change boundaries.
    pub fn breaking_boundaries(&self) -> Vec<(String, String)> {
        let mut boundaries = Vec::new();
        for i in 0..self.versions.len().saturating_sub(1) {
            let (v_old, s_old) = &self.versions[i];
            let (v_new, s_new) = &self.versions[i + 1];
            let diff = OpenApiDiff::diff(s_old, s_new);
            if diff.has_breaking_changes() {
                boundaries.push((v_old.clone(), v_new.clone()));
            }
        }
        boundaries
    }
}

impl Default for SchemaEvolution {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openapi::OpenApiSchema;

    fn v1_yaml() -> &'static str {
        concat!(
            "openapi: \"3.0.3\"\n",
            "info:\n  title: Test API\n  version: \"1.0.0\"\n",
            "paths:\n",
            "  /users:\n",
            "    get:\n",
            "      operationId: listUsers\n",
            "      parameters:\n",
            "        - name: limit\n          in: query\n          required: false\n          schema:\n            type: integer\n",
            "      responses:\n",
            "        \"200\":\n          description: OK\n          content:\n            application/json:\n              schema:\n                type: array\n                items:\n                  type: object\n                  properties:\n                    id:\n                      type: integer\n                    name:\n                      type: string\n",
            "    post:\n",
            "      operationId: createUser\n",
            "      requestBody:\n        required: true\n        content:\n          application/json:\n            schema:\n              type: object\n              properties:\n                id:\n                  type: integer\n                name:\n                  type: string\n",
            "      responses:\n        \"201\":\n          description: Created\n",
            "  /users/{id}:\n",
            "    get:\n",
            "      operationId: getUser\n",
            "      parameters:\n        - name: id\n          in: path\n          required: true\n          schema:\n            type: string\n",
            "      responses:\n        \"200\":\n          description: OK\n          content:\n            application/json:\n              schema:\n                type: object\n                properties:\n                  id:\n                    type: integer\n                  name:\n                    type: string\n",
            "components:\n  schemas:\n    User:\n      type: object\n      required:\n        - id\n        - name\n      properties:\n        id:\n          type: integer\n        name:\n          type: string\n        email:\n          type: string\n",
        )
    }

    fn v2_yaml() -> &'static str {
        concat!(
            "openapi: \"3.0.3\"\n",
            "info:\n  title: Test API\n  version: \"2.0.0\"\n",
            "paths:\n",
            "  /users:\n",
            "    get:\n",
            "      operationId: listUsers\n",
            "      parameters:\n",
            "        - name: limit\n          in: query\n          required: false\n          schema:\n            type: integer\n",
            "        - name: offset\n          in: query\n          required: false\n          schema:\n            type: integer\n",
            "      responses:\n",
            "        \"200\":\n          description: OK\n          content:\n            application/json:\n              schema:\n                type: array\n                items:\n                  type: object\n                  properties:\n                    id:\n                      type: integer\n                    name:\n                      type: string\n",
            "    post:\n",
            "      operationId: createUser\n",
            "      requestBody:\n        required: true\n        content:\n          application/json:\n            schema:\n              type: object\n              properties:\n                id:\n                  type: integer\n                name:\n                  type: string\n",
            "      responses:\n        \"201\":\n          description: Created\n",
            "  /users/{id}:\n",
            "    get:\n",
            "      operationId: getUser\n",
            "      parameters:\n        - name: id\n          in: path\n          required: true\n          schema:\n            type: string\n",
            "      responses:\n        \"200\":\n          description: OK\n          content:\n            application/json:\n              schema:\n                type: object\n                properties:\n                  id:\n                    type: integer\n                  name:\n                    type: string\n",
            "components:\n  schemas:\n    User:\n      type: object\n      required:\n        - id\n        - name\n      properties:\n        id:\n          type: integer\n        name:\n          type: string\n        email:\n          type: string\n        avatar_url:\n          type: string\n",
        )
    }

    fn v3_breaking_yaml() -> &'static str {
        concat!(
            "openapi: \"3.0.3\"\n",
            "info:\n  title: Test API\n  version: \"3.0.0\"\n",
            "paths:\n",
            "  /users:\n",
            "    get:\n",
            "      operationId: listUsers\n",
            "      parameters:\n",
            "        - name: limit\n          in: query\n          required: true\n          schema:\n            type: integer\n",
            "      responses:\n",
            "        \"200\":\n          description: OK\n          content:\n            application/json:\n              schema:\n                type: array\n                items:\n                  type: object\n                  properties:\n                    id:\n                      type: string\n                    name:\n                      type: string\n",
            "components:\n  schemas:\n    User:\n      type: object\n      required:\n        - id\n        - name\n        - email\n      properties:\n        id:\n          type: string\n        name:\n          type: string\n        email:\n          type: string\n",
        )
    }

    #[test]
    fn test_non_breaking_diff() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v2_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        assert!(!diff.has_breaking_changes());
        assert!(!diff.non_breaking_changes.is_empty());
    }

    #[test]
    fn test_breaking_diff_removed_endpoint() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        assert!(diff.has_breaking_changes());
        assert!(diff.removed_paths.contains(&"/users/{id}".to_string()));
    }

    #[test]
    fn test_breaking_diff_required_param() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        let breaking_strs: Vec<&str> = diff.breaking_changes.iter().map(|s| s.as_str()).collect();
        assert!(breaking_strs.iter().any(|s| s.contains("required")));
    }

    #[test]
    fn test_schema_type_change_detected() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        let has_type_change = diff
            .modified_schemas
            .iter()
            .flat_map(|(_, changes)| changes)
            .any(|c| matches!(c, SchemaChange::TypeChanged { .. }));
        assert!(has_type_change);
    }

    #[test]
    fn test_added_optional_field_non_breaking() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v2_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        let added = diff
            .modified_schemas
            .iter()
            .flat_map(|(_, changes)| changes)
            .find(|c| matches!(c, SchemaChange::FieldAdded { field_name, .. } if field_name == "avatar_url"));
        assert!(added.is_some());
        let detector = BreakingChangeDetector::new();
        assert_eq!(
            detector.classify_schema_change(added.unwrap()),
            ChangeClassification::NonBreaking
        );
    }

    #[test]
    fn test_schema_evolution_compatible() {
        let v1 = OpenApiSchema::parse(v1_yaml()).unwrap();
        let v2 = OpenApiSchema::parse(v2_yaml()).unwrap();
        let mut evo = SchemaEvolution::new();
        evo.add_version("1.0.0", v1);
        evo.add_version("2.0.0", v2);
        assert!(evo.is_backward_compatible("1.0.0", "2.0.0"));
    }

    #[test]
    fn test_schema_evolution_not_compatible() {
        let v1 = OpenApiSchema::parse(v1_yaml()).unwrap();
        let v3 = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let mut evo = SchemaEvolution::new();
        evo.add_version("1.0.0", v1);
        evo.add_version("3.0.0", v3);
        assert!(!evo.is_backward_compatible("1.0.0", "3.0.0"));
    }

    #[test]
    fn test_compatibility_matrix() {
        let v1 = OpenApiSchema::parse(v1_yaml()).unwrap();
        let v2 = OpenApiSchema::parse(v2_yaml()).unwrap();
        let v3 = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let mut evo = SchemaEvolution::new();
        evo.add_version("1.0.0", v1);
        evo.add_version("2.0.0", v2);
        evo.add_version("3.0.0", v3);
        let matrix = evo.compatibility_matrix();
        assert_eq!(matrix.len(), 3);
        let (_, _, compat_1_2) = &matrix[0];
        assert!(*compat_1_2);
        let (_, _, compat_1_3) = &matrix[1];
        assert!(!*compat_1_3);
    }

    #[test]
    fn test_breaking_boundaries() {
        let v1 = OpenApiSchema::parse(v1_yaml()).unwrap();
        let v2 = OpenApiSchema::parse(v2_yaml()).unwrap();
        let v3 = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let mut evo = SchemaEvolution::new();
        evo.add_version("1.0.0", v1);
        evo.add_version("2.0.0", v2);
        evo.add_version("3.0.0", v3);
        let boundaries = evo.breaking_boundaries();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], ("2.0.0".to_string(), "3.0.0".to_string()));
    }

    #[test]
    fn test_change_classification_merge() {
        assert_eq!(
            ChangeClassification::Breaking.merge(ChangeClassification::NonBreaking),
            ChangeClassification::Breaking
        );
        assert_eq!(
            ChangeClassification::NonBreaking.merge(ChangeClassification::Compatible),
            ChangeClassification::NonBreaking
        );
        assert_eq!(
            ChangeClassification::Compatible.merge(ChangeClassification::Compatible),
            ChangeClassification::Compatible
        );
    }

    #[test]
    fn test_diff_summary() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v3_breaking_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        let summary = diff.summary();
        assert!(summary.contains("breaking"));
    }

    #[test]
    fn test_parameter_changes() {
        let old = OpenApiSchema::parse(v1_yaml()).unwrap();
        let new = OpenApiSchema::parse(v2_yaml()).unwrap();
        let diff = OpenApiDiff::diff(&old, &new);
        let has_param_added = diff
            .modified_paths
            .iter()
            .flat_map(|pd| &pd.modified_operations)
            .any(|od| {
                od.parameter_changes
                    .iter()
                    .any(|pc| matches!(pc, ParameterChange::Added { name, .. } if name == "offset"))
            });
        assert!(has_param_added);
    }

    #[test]
    fn test_enum_diff() {
        let detector = BreakingChangeDetector::new();
        let removed = SchemaChange::EnumValueRemoved {
            path: "/test".into(),
            value: "old_val".into(),
        };
        let added = SchemaChange::EnumValueAdded {
            path: "/test".into(),
            value: "new_val".into(),
        };
        assert_eq!(
            detector.classify_schema_change(&removed),
            ChangeClassification::Breaking
        );
        assert_eq!(
            detector.classify_schema_change(&added),
            ChangeClassification::NonBreaking
        );
    }

    #[test]
    fn test_constraint_classification() {
        let detector = BreakingChangeDetector::new();
        let tightened = SchemaChange::ConstraintTightened {
            path: "/x".into(),
            constraint: "minimum".into(),
            old_value: "0".into(),
            new_value: "10".into(),
        };
        let loosened = SchemaChange::ConstraintLoosened {
            path: "/x".into(),
            constraint: "maximum".into(),
            old_value: "100".into(),
            new_value: "1000".into(),
        };
        assert_eq!(
            detector.classify_schema_change(&tightened),
            ChangeClassification::Breaking
        );
        assert_eq!(
            detector.classify_schema_change(&loosened),
            ChangeClassification::NonBreaking
        );
    }
}
