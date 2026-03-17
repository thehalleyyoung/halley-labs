//! Protobuf schema diffing and breaking-change detection.
//!
//! Compares two [`ProtobufSchema`] instances and produces a detailed
//! [`ProtoDiffResult`] with message, service, and enum diffs plus
//! wire-compatibility analysis.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::protobuf::{
    ProtoEnum, ProtoField, ProtoFieldType, ProtoMessage, ProtoMethod,
    ProtoService, ProtobufSchema,
};

// ---------------------------------------------------------------------------
// ChangeClassification (local copy matching openapi_diff)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeClassification {
    Breaking,
    NonBreaking,
    Compatible,
    Unknown,
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

// ---------------------------------------------------------------------------
// FieldChange
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldChange {
    Added {
        field_name: String,
        field_number: u32,
        type_name: String,
        label: String,
    },
    Removed {
        field_name: String,
        field_number: u32,
    },
    TypeChanged {
        field_name: String,
        field_number: u32,
        from: String,
        to: String,
    },
    LabelChanged {
        field_name: String,
        from: String,
        to: String,
    },
    NumberReused {
        old_field_name: String,
        new_field_name: String,
        number: u32,
    },
    NameChanged {
        old_name: String,
        new_name: String,
        field_number: u32,
    },
    DeprecatedChanged {
        field_name: String,
        from: bool,
        to: bool,
    },
}

impl fmt::Display for FieldChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Added { field_name, field_number, type_name, label } => {
                write!(f, "Field added: {label} {type_name} {field_name} = {field_number}")
            }
            Self::Removed { field_name, field_number } => {
                write!(f, "Field removed: {field_name} = {field_number}")
            }
            Self::TypeChanged { field_name, from, to, .. } => {
                write!(f, "Field type changed {field_name}: {from} -> {to}")
            }
            Self::LabelChanged { field_name, from, to } => {
                write!(f, "Field label changed {field_name}: {from} -> {to}")
            }
            Self::NumberReused { old_field_name, new_field_name, number } => {
                write!(f, "Field number {number} reused: {old_field_name} -> {new_field_name}")
            }
            Self::NameChanged { old_name, new_name, field_number } => {
                write!(f, "Field {field_number} renamed: {old_name} -> {new_name}")
            }
            Self::DeprecatedChanged { field_name, from, to } => {
                write!(f, "Field {field_name} deprecated changed: {from} -> {to}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MessageDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageDiff {
    pub name: String,
    pub added_fields: Vec<FieldChange>,
    pub removed_fields: Vec<FieldChange>,
    pub changed_fields: Vec<FieldChange>,
    pub reserved_added: Vec<String>,
    pub reserved_removed: Vec<String>,
}

impl MessageDiff {
    pub fn is_empty(&self) -> bool {
        self.added_fields.is_empty()
            && self.removed_fields.is_empty()
            && self.changed_fields.is_empty()
            && self.reserved_added.is_empty()
            && self.reserved_removed.is_empty()
    }

    pub fn has_breaking_changes(&self) -> bool {
        let detector = ProtoBreakingChange::new();
        !self.removed_fields.is_empty()
            || self.changed_fields.iter().any(|c| detector.classify(c).is_breaking())
    }

    pub fn total_changes(&self) -> usize {
        self.added_fields.len() + self.removed_fields.len() + self.changed_fields.len()
    }
}

// ---------------------------------------------------------------------------
// MethodDiff / ServiceDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodDiff {
    pub method_name: String,
    pub input_changed: Option<(String, String)>,
    pub output_changed: Option<(String, String)>,
    pub streaming_changed: Option<StreamingChange>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingChange {
    pub client_streaming_changed: Option<(bool, bool)>,
    pub server_streaming_changed: Option<(bool, bool)>,
}

impl MethodDiff {
    pub fn is_empty(&self) -> bool {
        self.input_changed.is_none()
            && self.output_changed.is_none()
            && self.streaming_changed.is_none()
    }

    pub fn is_breaking(&self) -> bool {
        self.input_changed.is_some()
            || self.output_changed.is_some()
            || self.streaming_changed.is_some()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServiceDiff {
    pub service_name: String,
    pub added_methods: Vec<String>,
    pub removed_methods: Vec<String>,
    pub changed_methods: Vec<MethodDiff>,
}

impl ServiceDiff {
    pub fn is_empty(&self) -> bool {
        self.added_methods.is_empty()
            && self.removed_methods.is_empty()
            && self.changed_methods.is_empty()
    }

    pub fn has_breaking_changes(&self) -> bool {
        !self.removed_methods.is_empty()
            || self.changed_methods.iter().any(|m| m.is_breaking())
    }
}

// ---------------------------------------------------------------------------
// EnumDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnumDiff {
    pub enum_name: String,
    pub added_values: Vec<(String, i32)>,
    pub removed_values: Vec<(String, i32)>,
    pub renamed_values: Vec<(String, String, i32)>,
}

impl EnumDiff {
    pub fn is_empty(&self) -> bool {
        self.added_values.is_empty()
            && self.removed_values.is_empty()
            && self.renamed_values.is_empty()
    }

    pub fn has_breaking_changes(&self) -> bool {
        !self.removed_values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ProtoDiffResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProtoDiffResult {
    pub message_changes: Vec<MessageDiff>,
    pub service_changes: Vec<ServiceDiff>,
    pub enum_changes: Vec<EnumDiff>,
    pub added_messages: Vec<String>,
    pub removed_messages: Vec<String>,
    pub added_services: Vec<String>,
    pub removed_services: Vec<String>,
    pub added_enums: Vec<String>,
    pub removed_enums: Vec<String>,
}

impl ProtoDiffResult {
    pub fn is_empty(&self) -> bool {
        self.message_changes.is_empty()
            && self.service_changes.is_empty()
            && self.enum_changes.is_empty()
            && self.added_messages.is_empty()
            && self.removed_messages.is_empty()
            && self.added_services.is_empty()
            && self.removed_services.is_empty()
            && self.added_enums.is_empty()
            && self.removed_enums.is_empty()
    }

    pub fn has_breaking_changes(&self) -> bool {
        !self.removed_messages.is_empty()
            || !self.removed_services.is_empty()
            || self.message_changes.iter().any(|m| m.has_breaking_changes())
            || self.service_changes.iter().any(|s| s.has_breaking_changes())
            || self.enum_changes.iter().any(|e| e.has_breaking_changes())
    }

    pub fn breaking_change_descriptions(&self) -> Vec<String> {
        let mut descs = Vec::new();
        for m in &self.removed_messages {
            descs.push(format!("Message removed: {m}"));
        }
        for s in &self.removed_services {
            descs.push(format!("Service removed: {s}"));
        }
        for md in &self.message_changes {
            for fc in &md.removed_fields {
                descs.push(format!("In message {}: {fc}", md.name));
            }
            for fc in &md.changed_fields {
                let det = ProtoBreakingChange::new();
                if det.classify(fc).is_breaking() {
                    descs.push(format!("In message {}: {fc}", md.name));
                }
            }
        }
        for sd in &self.service_changes {
            for m in &sd.removed_methods {
                descs.push(format!("In service {}: method removed: {m}", sd.service_name));
            }
            for md in &sd.changed_methods {
                if md.is_breaking() {
                    descs.push(format!("In service {}: method changed: {}", sd.service_name, md.method_name));
                }
            }
        }
        for ed in &self.enum_changes {
            for (name, _) in &ed.removed_values {
                descs.push(format!("In enum {}: value removed: {name}", ed.enum_name));
            }
        }
        descs
    }

    pub fn summary(&self) -> String {
        format!(
            "Proto diff: +{}/-{} messages, +{}/-{} services, +{}/-{} enums, {} msg changes, {} svc changes",
            self.added_messages.len(),
            self.removed_messages.len(),
            self.added_services.len(),
            self.removed_services.len(),
            self.added_enums.len(),
            self.removed_enums.len(),
            self.message_changes.len(),
            self.service_changes.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// ProtobufDiff — core diffing engine
// ---------------------------------------------------------------------------

pub struct ProtobufDiff;

impl ProtobufDiff {
    /// Compute a full diff between two protobuf schemas.
    pub fn diff(old: &ProtobufSchema, new: &ProtobufSchema) -> ProtoDiffResult {
        let mut result = ProtoDiffResult::default();

        // Messages
        let old_msg_names: Vec<&str> = old.messages.iter().map(|m| m.name.as_str()).collect();
        let new_msg_names: Vec<&str> = new.messages.iter().map(|m| m.name.as_str()).collect();

        for name in &old_msg_names {
            if !new_msg_names.contains(name) {
                result.removed_messages.push(name.to_string());
            }
        }
        for name in &new_msg_names {
            if !old_msg_names.contains(name) {
                result.added_messages.push(name.to_string());
            }
        }

        result.message_changes = Self::diff_messages(&old.messages, &new.messages);

        // Services
        let old_svc_names: Vec<&str> = old.services.iter().map(|s| s.name.as_str()).collect();
        let new_svc_names: Vec<&str> = new.services.iter().map(|s| s.name.as_str()).collect();

        for name in &old_svc_names {
            if !new_svc_names.contains(name) {
                result.removed_services.push(name.to_string());
            }
        }
        for name in &new_svc_names {
            if !old_svc_names.contains(name) {
                result.added_services.push(name.to_string());
            }
        }

        for old_svc in &old.services {
            if let Some(new_svc) = new.services.iter().find(|s| s.name == old_svc.name) {
                let sd = Self::diff_service(old_svc, new_svc);
                if !sd.is_empty() {
                    result.service_changes.push(sd);
                }
            }
        }

        // Enums
        let old_enum_names: Vec<&str> = old.enums.iter().map(|e| e.name.as_str()).collect();
        let new_enum_names: Vec<&str> = new.enums.iter().map(|e| e.name.as_str()).collect();

        for name in &old_enum_names {
            if !new_enum_names.contains(name) {
                result.removed_enums.push(name.to_string());
            }
        }
        for name in &new_enum_names {
            if !old_enum_names.contains(name) {
                result.added_enums.push(name.to_string());
            }
        }

        for old_enum in &old.enums {
            if let Some(new_enum) = new.enums.iter().find(|e| e.name == old_enum.name) {
                let ed = Self::diff_enum(old_enum, new_enum);
                if !ed.is_empty() {
                    result.enum_changes.push(ed);
                }
            }
        }

        result
    }

    pub fn diff_messages(old_msgs: &[ProtoMessage], new_msgs: &[ProtoMessage]) -> Vec<MessageDiff> {
        let mut diffs = Vec::new();
        for old_msg in old_msgs {
            if let Some(new_msg) = new_msgs.iter().find(|m| m.name == old_msg.name) {
                let md = Self::diff_single_message(old_msg, new_msg);
                if !md.is_empty() {
                    diffs.push(md);
                }
            }
        }
        diffs
    }

    fn diff_single_message(old: &ProtoMessage, new: &ProtoMessage) -> MessageDiff {
        let mut md = MessageDiff {
            name: old.name.clone(),
            ..Default::default()
        };

        let field_changes = Self::diff_fields(&old.fields, &new.fields);
        for fc in field_changes {
            match &fc {
                FieldChange::Added { .. } => md.added_fields.push(fc),
                FieldChange::Removed { .. } => md.removed_fields.push(fc),
                _ => md.changed_fields.push(fc),
            }
        }

        // Reserved ranges
        let old_reserved: Vec<String> = old
            .reserved_names
            .iter()
            .cloned()
            .chain(old.reserved_fields.iter().map(|r| format!("{}-{}", r.start, r.end)))
            .collect();
        let new_reserved: Vec<String> = new
            .reserved_names
            .iter()
            .cloned()
            .chain(new.reserved_fields.iter().map(|r| format!("{}-{}", r.start, r.end)))
            .collect();

        for r in &new_reserved {
            if !old_reserved.contains(r) {
                md.reserved_added.push(r.clone());
            }
        }
        for r in &old_reserved {
            if !new_reserved.contains(r) {
                md.reserved_removed.push(r.clone());
            }
        }

        md
    }

    pub fn diff_fields(old_fields: &[ProtoField], new_fields: &[ProtoField]) -> Vec<FieldChange> {
        let mut changes = Vec::new();

        let old_by_num: IndexMap<u32, &ProtoField> =
            old_fields.iter().map(|f| (f.number, f)).collect();
        let new_by_num: IndexMap<u32, &ProtoField> =
            new_fields.iter().map(|f| (f.number, f)).collect();
        let old_by_name: IndexMap<&str, &ProtoField> =
            old_fields.iter().map(|f| (f.name.as_str(), f)).collect();
        let new_by_name: IndexMap<&str, &ProtoField> =
            new_fields.iter().map(|f| (f.name.as_str(), f)).collect();

        // Removed fields
        for (num, old_f) in &old_by_num {
            if !new_by_num.contains_key(num) && !new_by_name.contains_key(old_f.name.as_str()) {
                changes.push(FieldChange::Removed {
                    field_name: old_f.name.clone(),
                    field_number: *num,
                });
            }
        }

        // Added fields
        for (num, new_f) in &new_by_num {
            if !old_by_num.contains_key(num) && !old_by_name.contains_key(new_f.name.as_str()) {
                changes.push(FieldChange::Added {
                    field_name: new_f.name.clone(),
                    field_number: *num,
                    type_name: format!("{:?}", new_f.type_),
                    label: format!("{:?}", new_f.label),
                });
            }
        }

        // Changed fields (matched by number)
        for (num, old_f) in &old_by_num {
            if let Some(new_f) = new_by_num.get(num) {
                // Name changed
                if old_f.name != new_f.name {
                    changes.push(FieldChange::NameChanged {
                        old_name: old_f.name.clone(),
                        new_name: new_f.name.clone(),
                        field_number: *num,
                    });
                }
                // Type changed
                if old_f.type_ != new_f.type_ {
                    changes.push(FieldChange::TypeChanged {
                        field_name: new_f.name.clone(),
                        field_number: *num,
                        from: format!("{:?}", old_f.type_),
                        to: format!("{:?}", new_f.type_),
                    });
                }
                // Label changed
                if old_f.label != new_f.label {
                    changes.push(FieldChange::LabelChanged {
                        field_name: new_f.name.clone(),
                        from: format!("{:?}", old_f.label),
                        to: format!("{:?}", new_f.label),
                    });
                }
                // Deprecated changed
                if old_f.deprecated != new_f.deprecated {
                    changes.push(FieldChange::DeprecatedChanged {
                        field_name: new_f.name.clone(),
                        from: old_f.deprecated,
                        to: new_f.deprecated,
                    });
                }
            }
        }

        // Number reuse detection: new field uses a number that belonged to a
        // differently-named old field that was removed.
        for (num, new_f) in &new_by_num {
            if let Some(old_f) = old_by_num.get(num) {
                if old_f.name != new_f.name
                    && !old_by_name.contains_key(new_f.name.as_str())
                    && !new_by_name.contains_key(old_f.name.as_str())
                {
                    changes.push(FieldChange::NumberReused {
                        old_field_name: old_f.name.clone(),
                        new_field_name: new_f.name.clone(),
                        number: *num,
                    });
                }
            }
        }

        changes
    }

    fn diff_service(old: &ProtoService, new: &ProtoService) -> ServiceDiff {
        let mut sd = ServiceDiff {
            service_name: old.name.clone(),
            ..Default::default()
        };

        let old_method_names: Vec<&str> = old.methods.iter().map(|m| m.name.as_str()).collect();
        let new_method_names: Vec<&str> = new.methods.iter().map(|m| m.name.as_str()).collect();

        for name in &old_method_names {
            if !new_method_names.contains(name) {
                sd.removed_methods.push(name.to_string());
            }
        }
        for name in &new_method_names {
            if !old_method_names.contains(name) {
                sd.added_methods.push(name.to_string());
            }
        }

        for old_m in &old.methods {
            if let Some(new_m) = new.methods.iter().find(|m| m.name == old_m.name) {
                let md = Self::diff_method(old_m, new_m);
                if !md.is_empty() {
                    sd.changed_methods.push(md);
                }
            }
        }

        sd
    }

    fn diff_method(old: &ProtoMethod, new: &ProtoMethod) -> MethodDiff {
        let input_changed = if old.input_type != new.input_type {
            Some((old.input_type.clone(), new.input_type.clone()))
        } else {
            None
        };
        let output_changed = if old.output_type != new.output_type {
            Some((old.output_type.clone(), new.output_type.clone()))
        } else {
            None
        };

        let cs_changed = if old.client_streaming != new.client_streaming {
            Some((old.client_streaming, new.client_streaming))
        } else {
            None
        };
        let ss_changed = if old.server_streaming != new.server_streaming {
            Some((old.server_streaming, new.server_streaming))
        } else {
            None
        };
        let streaming_changed = if cs_changed.is_some() || ss_changed.is_some() {
            Some(StreamingChange {
                client_streaming_changed: cs_changed,
                server_streaming_changed: ss_changed,
            })
        } else {
            None
        };

        MethodDiff {
            method_name: old.name.clone(),
            input_changed,
            output_changed,
            streaming_changed,
        }
    }

    fn diff_enum(old: &ProtoEnum, new: &ProtoEnum) -> EnumDiff {
        let mut ed = EnumDiff {
            enum_name: old.name.clone(),
            ..Default::default()
        };

        let old_vals: IndexMap<i32, &str> = old.values.iter().map(|v| (v.number, v.name.as_str())).collect();
        let new_vals: IndexMap<i32, &str> = new.values.iter().map(|v| (v.number, v.name.as_str())).collect();

        for (&num, &name) in &old_vals {
            if !new_vals.contains_key(&num) {
                ed.removed_values.push((name.to_string(), num));
            }
        }
        for (&num, &name) in &new_vals {
            if !old_vals.contains_key(&num) {
                ed.added_values.push((name.to_string(), num));
            }
        }
        for (&num, &old_name) in &old_vals {
            if let Some(&new_name) = new_vals.get(&num) {
                if old_name != new_name {
                    ed.renamed_values
                        .push((old_name.to_string(), new_name.to_string(), num));
                }
            }
        }

        ed
    }
}

// ---------------------------------------------------------------------------
// WireCompatibility
// ---------------------------------------------------------------------------

/// Checks wire-level type compatibility for protobuf field changes.
pub struct WireCompatibility;

impl WireCompatibility {
    /// Returns `true` if changing a field from `old_type` to `new_type` is
    /// wire-compatible (same wire encoding).
    pub fn is_wire_compatible(old_type: &str, new_type: &str) -> bool {
        if old_type == new_type {
            return true;
        }
        let pairs = Self::compatible_type_pairs();
        pairs.iter().any(|&(a, b)| {
            (a == old_type && b == new_type) || (a == new_type && b == old_type)
        })
    }

    /// Wire-compatible type pairs (same wire encoding).
    pub fn compatible_type_pairs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("int32", "int64"),
            ("int32", "uint32"),
            ("int64", "uint64"),
            ("uint32", "uint64"),
            ("sint32", "sint64"),
            ("fixed32", "sfixed32"),
            ("fixed64", "sfixed64"),
            ("bool", "int32"),
            ("bool", "int64"),
            ("bool", "uint32"),
            ("bool", "uint64"),
        ]
    }

    /// Check whether two ProtoFieldType values are wire-compatible.
    pub fn are_types_compatible(old: &ProtoFieldType, new: &ProtoFieldType) -> bool {
        if old == new {
            return true;
        }
        // Same wire type number = potentially compatible
        old.wire_type() == new.wire_type()
    }

    /// Return the wire type code for a named protobuf type.
    pub fn wire_type_for_name(type_name: &str) -> Option<u8> {
        let ft = ProtoFieldType::from_name(type_name);
        Some(ft.wire_type())
    }
}

// ---------------------------------------------------------------------------
// ProtoBreakingChange (detector)
// ---------------------------------------------------------------------------

/// Detects breaking changes in protobuf schema evolution.
pub struct ProtoBreakingChange {
    strict: bool,
}

impl ProtoBreakingChange {
    pub fn new() -> Self {
        Self { strict: false }
    }

    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Classify a single field change.
    pub fn classify(&self, change: &FieldChange) -> ChangeClassification {
        match change {
            FieldChange::Added { label, .. } => {
                if label.contains("Required") {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::NonBreaking
                }
            }
            FieldChange::Removed { .. } => ChangeClassification::Breaking,
            FieldChange::TypeChanged { from, to, .. } => {
                let from_clean = from.trim_matches(|c: char| !c.is_alphanumeric());
                let to_clean = to.trim_matches(|c: char| !c.is_alphanumeric());
                if WireCompatibility::is_wire_compatible(from_clean, to_clean) {
                    ChangeClassification::Compatible
                } else {
                    ChangeClassification::Breaking
                }
            }
            FieldChange::LabelChanged { from, to, .. } => {
                // repeated -> optional is breaking; optional -> repeated is breaking in proto2
                if from.contains("Required") || to.contains("Required") {
                    ChangeClassification::Breaking
                } else if self.strict {
                    ChangeClassification::Breaking
                } else {
                    ChangeClassification::NonBreaking
                }
            }
            FieldChange::NumberReused { .. } => ChangeClassification::Breaking,
            FieldChange::NameChanged { .. } => ChangeClassification::Compatible,
            FieldChange::DeprecatedChanged { .. } => ChangeClassification::NonBreaking,
        }
    }

    /// Check whether any diff result contains breaking changes.
    pub fn has_breaking(result: &ProtoDiffResult) -> bool {
        result.has_breaking_changes()
    }

    /// Compute overall classification for an entire diff.
    pub fn overall_classification(&self, result: &ProtoDiffResult) -> ChangeClassification {
        let mut overall = ChangeClassification::Compatible;

        if !result.removed_messages.is_empty()
            || !result.removed_services.is_empty()
        {
            return ChangeClassification::Breaking;
        }

        for md in &result.message_changes {
            for fc in &md.removed_fields {
                overall = overall.merge(self.classify(fc));
            }
            for fc in &md.changed_fields {
                overall = overall.merge(self.classify(fc));
            }
        }
        for sd in &result.service_changes {
            if !sd.removed_methods.is_empty() {
                overall = overall.merge(ChangeClassification::Breaking);
            }
            for md in &sd.changed_methods {
                if md.is_breaking() {
                    overall = overall.merge(ChangeClassification::Breaking);
                }
            }
        }
        for ed in &result.enum_changes {
            if !ed.removed_values.is_empty() {
                overall = overall.merge(ChangeClassification::Breaking);
            }
        }

        overall
    }
}

impl Default for ProtoBreakingChange {
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
    use crate::protobuf::*;

    fn make_field(name: &str, number: u32, ft: ProtoFieldType, label: FieldLabel) -> ProtoField {
        ProtoField::new(name, number, ft, label)
    }

    fn make_service(name: &str, methods: Vec<ProtoMethod>) -> ProtoService {
        ProtoService {
            name: name.to_string(),
            methods,
            options: IndexMap::new(),
        }
    }

    fn make_method(name: &str, input: &str, output: &str) -> ProtoMethod {
        ProtoMethod {
            name: name.to_string(),
            input_type: input.to_string(),
            output_type: output.to_string(),
            client_streaming: false,
            server_streaming: false,
            options: IndexMap::new(),
        }
    }

    fn make_enum(name: &str, values: &[(&str, i32)]) -> ProtoEnum {
        ProtoEnum {
            name: name.to_string(),
            values: values
                .iter()
                .map(|(n, v)| ProtoEnumValue {
                    name: n.to_string(),
                    number: *v,
                    options: IndexMap::new(),
                })
                .collect(),
            options: IndexMap::new(),
            allow_alias: false,
            reserved_numbers: vec![],
            reserved_names: vec![],
        }
    }

    fn empty_schema() -> ProtobufSchema {
        ProtobufSchema {
            syntax: "proto3".to_string(),
            package: None,
            imports: vec![],
            options: IndexMap::new(),
            messages: vec![],
            enums: vec![],
            services: vec![],
            extensions: vec![],
        }
    }

    #[test]
    fn test_no_changes() {
        let schema = empty_schema();
        let diff = ProtobufDiff::diff(&schema, &schema);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_added_message() {
        let old = empty_schema();
        let mut new = empty_schema();
        new.messages.push(ProtoMessage {
            name: "Foo".into(),
            fields: vec![],
            nested_messages: vec![],
            nested_enums: vec![],
            oneofs: vec![],
            reserved_fields: vec![],
            reserved_names: vec![],
            options: IndexMap::new(),
        });
        let diff = ProtobufDiff::diff(&old, &new);
        assert!(diff.added_messages.contains(&"Foo".to_string()));
        assert!(!diff.has_breaking_changes());
    }

    #[test]
    fn test_removed_message_is_breaking() {
        let mut old = empty_schema();
        old.messages.push(ProtoMessage {
            name: "Foo".into(),
            fields: vec![],
            nested_messages: vec![],
            nested_enums: vec![],
            oneofs: vec![],
            reserved_fields: vec![],
            reserved_names: vec![],
            options: IndexMap::new(),
        });
        let new = empty_schema();
        let diff = ProtobufDiff::diff(&old, &new);
        assert!(diff.removed_messages.contains(&"Foo".to_string()));
        assert!(diff.has_breaking_changes());
    }

    #[test]
    fn test_field_added() {
        let mut old = empty_schema();
        old.messages.push(ProtoMessage {
            name: "Msg".into(),
            fields: vec![make_field("a", 1, ProtoFieldType::String, FieldLabel::Optional)],
            nested_messages: vec![],
            nested_enums: vec![],
            oneofs: vec![],
            reserved_fields: vec![],
            reserved_names: vec![],
            options: IndexMap::new(),
        });
        let mut new = empty_schema();
        new.messages.push(ProtoMessage {
            name: "Msg".into(),
            fields: vec![
                make_field("a", 1, ProtoFieldType::String, FieldLabel::Optional),
                make_field("b", 2, ProtoFieldType::Int32, FieldLabel::Optional),
            ],
            nested_messages: vec![],
            nested_enums: vec![],
            oneofs: vec![],
            reserved_fields: vec![],
            reserved_names: vec![],
            options: IndexMap::new(),
        });
        let diff = ProtobufDiff::diff(&old, &new);
        assert_eq!(diff.message_changes.len(), 1);
        assert_eq!(diff.message_changes[0].added_fields.len(), 1);
        assert!(!diff.has_breaking_changes());
    }

    #[test]
    fn test_field_removed_is_breaking() {
        let mut old = empty_schema();
        old.messages.push(ProtoMessage {
            name: "Msg".into(),
            fields: vec![
                make_field("a", 1, ProtoFieldType::String, FieldLabel::Optional),
                make_field("b", 2, ProtoFieldType::Int32, FieldLabel::Optional),
            ],
            nested_messages: vec![],
            nested_enums: vec![],
            oneofs: vec![],
            reserved_fields: vec![],
            reserved_names: vec![],
            options: IndexMap::new(),
        });
        let mut new = empty_schema();
        new.messages.push(ProtoMessage {
            name: "Msg".into(),
            fields: vec![make_field("a", 1, ProtoFieldType::String, FieldLabel::Optional)],
            nested_messages: vec![],
            nested_enums: vec![],
            oneofs: vec![],
            reserved_fields: vec![],
            reserved_names: vec![],
            options: IndexMap::new(),
        });
        let diff = ProtobufDiff::diff(&old, &new);
        assert!(diff.has_breaking_changes());
    }

    #[test]
    fn test_type_change_incompatible() {
        let changes = ProtobufDiff::diff_fields(
            &[make_field("x", 1, ProtoFieldType::String, FieldLabel::Optional)],
            &[make_field("x", 1, ProtoFieldType::Int32, FieldLabel::Optional)],
        );
        let tc = changes.iter().find(|c| matches!(c, FieldChange::TypeChanged { .. }));
        assert!(tc.is_some());
        let det = ProtoBreakingChange::new();
        assert_eq!(det.classify(tc.unwrap()), ChangeClassification::Breaking);
    }

    #[test]
    fn test_wire_compatible_types() {
        assert!(WireCompatibility::is_wire_compatible("int32", "int64"));
        assert!(WireCompatibility::is_wire_compatible("int64", "int32"));
        assert!(WireCompatibility::is_wire_compatible("uint32", "uint64"));
        assert!(!WireCompatibility::is_wire_compatible("string", "int32"));
        assert!(!WireCompatibility::is_wire_compatible("bytes", "string"));
    }

    #[test]
    fn test_service_diff() {
        let mut old = empty_schema();
        old.services.push(make_service("Svc", vec![
            make_method("GetFoo", "FooReq", "FooResp"),
            make_method("GetBar", "BarReq", "BarResp"),
        ]));
        let mut new = empty_schema();
        new.services.push(make_service("Svc", vec![
            make_method("GetFoo", "FooReq", "FooResp"),
            make_method("GetBaz", "BazReq", "BazResp"),
        ]));
        let diff = ProtobufDiff::diff(&old, &new);
        assert_eq!(diff.service_changes.len(), 1);
        let sd = &diff.service_changes[0];
        assert!(sd.removed_methods.contains(&"GetBar".to_string()));
        assert!(sd.added_methods.contains(&"GetBaz".to_string()));
        assert!(sd.has_breaking_changes());
    }

    #[test]
    fn test_enum_diff() {
        let mut old = empty_schema();
        old.enums.push(make_enum("Status", &[("UNKNOWN", 0), ("ACTIVE", 1), ("DELETED", 2)]));
        let mut new = empty_schema();
        new.enums.push(make_enum("Status", &[("UNKNOWN", 0), ("ACTIVE", 1), ("SUSPENDED", 3)]));
        let diff = ProtobufDiff::diff(&old, &new);
        assert_eq!(diff.enum_changes.len(), 1);
        let ed = &diff.enum_changes[0];
        assert!(ed.removed_values.iter().any(|(n, _)| n == "DELETED"));
        assert!(ed.added_values.iter().any(|(n, _)| n == "SUSPENDED"));
        assert!(ed.has_breaking_changes());
    }

    #[test]
    fn test_method_type_change() {
        let mut old = empty_schema();
        old.services.push(make_service("Svc", vec![make_method("Do", "A", "B")]));
        let mut new = empty_schema();
        new.services.push(make_service("Svc", vec![make_method("Do", "C", "B")]));
        let diff = ProtobufDiff::diff(&old, &new);
        assert!(diff.service_changes[0].changed_methods[0].input_changed.is_some());
    }

    #[test]
    fn test_overall_classification() {
        let old = empty_schema();
        let new = empty_schema();
        let diff = ProtobufDiff::diff(&old, &new);
        let det = ProtoBreakingChange::new();
        assert_eq!(det.overall_classification(&diff), ChangeClassification::Compatible);
    }

    #[test]
    fn test_breaking_descriptions() {
        let mut old = empty_schema();
        old.services.push(make_service("Svc", vec![make_method("Dead", "A", "B")]));
        let new = empty_schema();
        let diff = ProtobufDiff::diff(&old, &new);
        let descs = diff.breaking_change_descriptions();
        assert!(!descs.is_empty());
    }

    #[test]
    fn test_field_number_reuse() {
        let changes = ProtobufDiff::diff_fields(
            &[make_field("old_name", 1, ProtoFieldType::String, FieldLabel::Optional)],
            &[make_field("new_name", 1, ProtoFieldType::String, FieldLabel::Optional)],
        );
        let reuse = changes.iter().find(|c| matches!(c, FieldChange::NumberReused { .. }));
        assert!(reuse.is_some());
        let det = ProtoBreakingChange::new();
        assert_eq!(det.classify(reuse.unwrap()), ChangeClassification::Breaking);
    }
}
