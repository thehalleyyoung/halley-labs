//! Unified schema abstraction across all API formats.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use safestep_types::{SafeStepError, Result};

use crate::openapi::OpenApiSchema;
use crate::protobuf::ProtobufSchema;
use crate::graphql::GraphqlSchema;
use crate::avro::AvroSchema;
use crate::confidence::ConfidenceScore;

// ---------------------------------------------------------------------------
// Unified types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SchemaFormat {
    OpenApi,
    Protobuf,
    GraphQL,
    Avro,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrimitiveKind {
    String,
    Integer,
    Long,
    Float,
    Double,
    Boolean,
    Bytes,
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedField {
    pub name: String,
    pub type_: UnifiedType,
    pub required: bool,
    pub description: Option<String>,
    pub deprecated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnifiedType {
    Primitive(PrimitiveKind),
    Object(Vec<UnifiedField>),
    Array(Box<UnifiedType>),
    Map(Box<UnifiedType>, Box<UnifiedType>),
    Union(Vec<UnifiedType>),
    Enum(Vec<String>),
    Ref(String),
    Unknown,
}

impl UnifiedType {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Primitive(_) => "primitive",
            Self::Object(_) => "object",
            Self::Array(_) => "array",
            Self::Map(_, _) => "map",
            Self::Union(_) => "union",
            Self::Enum(_) => "enum",
            Self::Ref(_) => "ref",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_primitive(&self) -> bool {
        matches!(self, Self::Primitive(_))
    }

    pub fn is_complex(&self) -> bool {
        matches!(self, Self::Object(_) | Self::Array(_) | Self::Map(_, _) | Self::Union(_))
    }

    pub fn field_count(&self) -> usize {
        match self {
            Self::Object(fields) => fields.len(),
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// UnifiedEndpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedEndpoint {
    pub path: String,
    pub method: String,
    pub input_schema: Option<UnifiedType>,
    pub output_schema: Option<UnifiedType>,
    pub parameters: Vec<UnifiedField>,
    pub deprecated: bool,
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// UnifiedSchema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSchema {
    pub format: SchemaFormat,
    pub name: String,
    pub version: String,
    pub endpoints: Vec<UnifiedEndpoint>,
    pub types: IndexMap<String, UnifiedType>,
}

impl UnifiedSchema {
    pub fn from_openapi(schema: &OpenApiSchema) -> Self {
        let mut endpoints = Vec::new();
        let mut types = IndexMap::new();

        for (path, pi) in &schema.paths {
            for (method, op) in pi.operations() {
                let input_schema = op.request_body.as_ref().and_then(|rb| {
                    rb.content.values().next().and_then(|mt| {
                        mt.schema.as_ref().map(|s| convert_openapi_schema(s))
                    })
                });

                let output_schema = op.responses.values().next().and_then(|resp| {
                    resp.content.values().next().and_then(|mt| {
                        mt.schema.as_ref().map(|s| convert_openapi_schema(s))
                    })
                });

                let parameters = op.parameters.iter().map(|p| {
                    UnifiedField {
                        name: p.name.clone(),
                        type_: p.schema.as_ref()
                            .map(|s| convert_openapi_schema(s))
                            .unwrap_or(UnifiedType::Primitive(PrimitiveKind::String)),
                        required: p.required,
                        description: p.description.clone(),
                        deprecated: p.deprecated,
                    }
                }).collect();

                endpoints.push(UnifiedEndpoint {
                    path: path.clone(),
                    method: method.to_uppercase(),
                    input_schema,
                    output_schema,
                    parameters,
                    deprecated: op.deprecated,
                    description: op.description.clone(),
                });
            }
        }

        for (name, schema_obj) in &schema.components {
            types.insert(name.clone(), convert_openapi_schema(schema_obj));
        }

        Self {
            format: SchemaFormat::OpenApi,
            name: schema.title.clone(),
            version: schema.version.clone(),
            endpoints,
            types,
        }
    }

    pub fn from_protobuf(schema: &ProtobufSchema) -> Self {
        let mut endpoints = Vec::new();
        let mut types = IndexMap::new();

        for svc in &schema.services {
            for method in &svc.methods {
                let input = UnifiedType::Ref(method.input_type.clone());
                let output = UnifiedType::Ref(method.output_type.clone());
                let streaming_suffix = match (method.client_streaming, method.server_streaming) {
                    (true, true) => " (bidi-stream)",
                    (true, false) => " (client-stream)",
                    (false, true) => " (server-stream)",
                    (false, false) => "",
                };
                endpoints.push(UnifiedEndpoint {
                    path: format!("/{}/{}", svc.name, method.name),
                    method: format!("RPC{}", streaming_suffix),
                    input_schema: Some(input),
                    output_schema: Some(output),
                    parameters: vec![],
                    deprecated: false,
                    description: None,
                });
            }
        }

        for msg in &schema.messages {
            types.insert(msg.name.clone(), convert_proto_message(msg));
        }
        for en in &schema.enums {
            types.insert(en.name.clone(), UnifiedType::Enum(
                en.values.iter().map(|v| v.name.clone()).collect(),
            ));
        }

        Self {
            format: SchemaFormat::Protobuf,
            name: schema.package.clone().unwrap_or_default(),
            version: String::new(),
            endpoints,
            types,
        }
    }

    pub fn from_graphql(schema: &GraphqlSchema) -> Self {
        let mut endpoints = Vec::new();
        let mut types = IndexMap::new();

        // Queries
        for f in schema.query_fields() {
            endpoints.push(UnifiedEndpoint {
                path: format!("/query/{}", f.name),
                method: "QUERY".to_string(),
                input_schema: if f.arguments.is_empty() { None } else {
                    Some(UnifiedType::Object(f.arguments.iter().map(|a| UnifiedField {
                        name: a.name.clone(),
                        type_: convert_graphql_field_type(&a.type_),
                        required: a.type_.is_non_null,
                        description: a.description.clone(),
                        deprecated: false,
                    }).collect()))
                },
                output_schema: Some(convert_graphql_field_type(&f.type_)),
                parameters: vec![],
                deprecated: f.is_deprecated,
                description: f.description.clone(),
            });
        }
        // Mutations
        for f in schema.mutation_fields() {
            endpoints.push(UnifiedEndpoint {
                path: format!("/mutation/{}", f.name),
                method: "MUTATION".to_string(),
                input_schema: if f.arguments.is_empty() { None } else {
                    Some(UnifiedType::Object(f.arguments.iter().map(|a| UnifiedField {
                        name: a.name.clone(),
                        type_: convert_graphql_field_type(&a.type_),
                        required: a.type_.is_non_null,
                        description: a.description.clone(),
                        deprecated: false,
                    }).collect()))
                },
                output_schema: Some(convert_graphql_field_type(&f.type_)),
                parameters: vec![],
                deprecated: f.is_deprecated,
                description: f.description.clone(),
            });
        }

        for (name, t) in &schema.types {
            let ut = match t.kind {
                crate::graphql::GraphqlTypeKind::Enum => {
                    UnifiedType::Enum(t.enum_values.iter().map(|v| v.name.clone()).collect())
                }
                crate::graphql::GraphqlTypeKind::Union => {
                    UnifiedType::Union(t.members.iter().map(|m| UnifiedType::Ref(m.clone())).collect())
                }
                crate::graphql::GraphqlTypeKind::Scalar => {
                    UnifiedType::Primitive(PrimitiveKind::String)
                }
                _ => {
                    UnifiedType::Object(t.fields.iter().map(|f| UnifiedField {
                        name: f.name.clone(),
                        type_: convert_graphql_field_type(&f.type_),
                        required: f.type_.is_non_null,
                        description: f.description.clone(),
                        deprecated: f.is_deprecated,
                    }).collect())
                }
            };
            types.insert(name.clone(), ut);
        }

        Self {
            format: SchemaFormat::GraphQL,
            name: String::new(),
            version: String::new(),
            endpoints,
            types,
        }
    }

    pub fn from_avro(schema: &AvroSchema) -> Self {
        let mut types = IndexMap::new();
        convert_avro_to_types(schema, &mut types);
        Self {
            format: SchemaFormat::Avro,
            name: schema.full_name().unwrap_or_default(),
            version: String::new(),
            endpoints: vec![],
            types,
        }
    }

    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    pub fn type_count(&self) -> usize {
        self.types.len()
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn convert_openapi_schema(s: &crate::openapi::SchemaObject) -> UnifiedType {
    if let Some(ref rp) = s.ref_path {
        let name = rp.strip_prefix("#/components/schemas/").unwrap_or(rp);
        return UnifiedType::Ref(name.to_string());
    }
    if !s.enum_values.is_empty() {
        return UnifiedType::Enum(
            s.enum_values.iter().map(|v| v.to_string().trim_matches('"').to_string()).collect(),
        );
    }
    if !s.one_of.is_empty() {
        return UnifiedType::Union(s.one_of.iter().map(|sub| convert_openapi_schema(sub)).collect());
    }
    if !s.any_of.is_empty() {
        return UnifiedType::Union(s.any_of.iter().map(|sub| convert_openapi_schema(sub)).collect());
    }
    match s.type_.as_deref() {
        Some("string") => UnifiedType::Primitive(PrimitiveKind::String),
        Some("integer") => UnifiedType::Primitive(PrimitiveKind::Integer),
        Some("number") => {
            if s.format.as_deref() == Some("float") {
                UnifiedType::Primitive(PrimitiveKind::Float)
            } else {
                UnifiedType::Primitive(PrimitiveKind::Double)
            }
        }
        Some("boolean") => UnifiedType::Primitive(PrimitiveKind::Boolean),
        Some("array") => {
            let items = s.items.as_ref()
                .map(|i| convert_openapi_schema(i))
                .unwrap_or(UnifiedType::Unknown);
            UnifiedType::Array(Box::new(items))
        }
        Some("object") | None if !s.properties.is_empty() => {
            let fields = s.properties.iter().map(|(name, prop)| UnifiedField {
                name: name.clone(),
                type_: convert_openapi_schema(prop),
                required: s.required.contains(name),
                description: prop.description.clone(),
                deprecated: false,
            }).collect();
            UnifiedType::Object(fields)
        }
        _ => {
            if !s.all_of.is_empty() {
                let merged = s.merge_all_of();
                return convert_openapi_schema(&merged);
            }
            UnifiedType::Unknown
        }
    }
}

fn convert_proto_message(msg: &crate::protobuf::ProtoMessage) -> UnifiedType {
    let fields = msg.fields.iter().map(|f| {
        let type_ = convert_proto_field_type(&f.type_);
        let type_ = if f.label == crate::protobuf::FieldLabel::Repeated {
            UnifiedType::Array(Box::new(type_))
        } else {
            type_
        };
        UnifiedField {
            name: f.name.clone(),
            type_,
            required: f.label == crate::protobuf::FieldLabel::Required,
            description: None,
            deprecated: f.deprecated,
        }
    }).collect();
    UnifiedType::Object(fields)
}

fn convert_proto_field_type(ft: &crate::protobuf::ProtoFieldType) -> UnifiedType {
    use crate::protobuf::ProtoFieldType;
    match ft {
        ProtoFieldType::Double => UnifiedType::Primitive(PrimitiveKind::Double),
        ProtoFieldType::Float => UnifiedType::Primitive(PrimitiveKind::Float),
        ProtoFieldType::Int32 | ProtoFieldType::Sint32 | ProtoFieldType::Fixed32 | ProtoFieldType::Sfixed32 | ProtoFieldType::Uint32 => {
            UnifiedType::Primitive(PrimitiveKind::Integer)
        }
        ProtoFieldType::Int64 | ProtoFieldType::Sint64 | ProtoFieldType::Fixed64 | ProtoFieldType::Sfixed64 | ProtoFieldType::Uint64 => {
            UnifiedType::Primitive(PrimitiveKind::Long)
        }
        ProtoFieldType::Bool => UnifiedType::Primitive(PrimitiveKind::Boolean),
        ProtoFieldType::String => UnifiedType::Primitive(PrimitiveKind::String),
        ProtoFieldType::Bytes => UnifiedType::Primitive(PrimitiveKind::Bytes),
        ProtoFieldType::Message(name) => UnifiedType::Ref(name.clone()),
        ProtoFieldType::Enum(name) => UnifiedType::Ref(name.clone()),
        ProtoFieldType::Map { key, value } => {
            UnifiedType::Map(
                Box::new(convert_proto_field_type(key)),
                Box::new(convert_proto_field_type(value)),
            )
        }
    }
}

fn convert_graphql_field_type(ft: &crate::graphql::GraphqlFieldType) -> UnifiedType {
    let base = match ft.name.as_str() {
        "String" | "ID" => UnifiedType::Primitive(PrimitiveKind::String),
        "Int" => UnifiedType::Primitive(PrimitiveKind::Integer),
        "Float" => UnifiedType::Primitive(PrimitiveKind::Double),
        "Boolean" => UnifiedType::Primitive(PrimitiveKind::Boolean),
        other => UnifiedType::Ref(other.to_string()),
    };
    if ft.is_list {
        UnifiedType::Array(Box::new(base))
    } else {
        base
    }
}

fn convert_avro_to_types(schema: &AvroSchema, types: &mut IndexMap<String, UnifiedType>) {
    match schema {
        AvroSchema::Record(rec) => {
            let fields = rec.fields.iter().map(|f| {
                let type_ = avro_to_unified_type(&f.type_);
                UnifiedField {
                    name: f.name.clone(),
                    type_,
                    required: f.default.is_none(),
                    description: f.doc.clone(),
                    deprecated: false,
                }
            }).collect();
            types.insert(rec.name.clone(), UnifiedType::Object(fields));
        }
        AvroSchema::Enum(en) => {
            types.insert(en.name.clone(), UnifiedType::Enum(en.symbols.clone()));
        }
        _ => {}
    }
}

fn avro_to_unified_type(schema: &AvroSchema) -> UnifiedType {
    match schema {
        AvroSchema::Null => UnifiedType::Primitive(PrimitiveKind::Null),
        AvroSchema::Boolean => UnifiedType::Primitive(PrimitiveKind::Boolean),
        AvroSchema::Int => UnifiedType::Primitive(PrimitiveKind::Integer),
        AvroSchema::Long => UnifiedType::Primitive(PrimitiveKind::Long),
        AvroSchema::Float => UnifiedType::Primitive(PrimitiveKind::Float),
        AvroSchema::Double => UnifiedType::Primitive(PrimitiveKind::Double),
        AvroSchema::String => UnifiedType::Primitive(PrimitiveKind::String),
        AvroSchema::Bytes => UnifiedType::Primitive(PrimitiveKind::Bytes),
        AvroSchema::Array { items } => UnifiedType::Array(Box::new(avro_to_unified_type(items))),
        AvroSchema::Map { values } => UnifiedType::Map(
            Box::new(UnifiedType::Primitive(PrimitiveKind::String)),
            Box::new(avro_to_unified_type(values)),
        ),
        AvroSchema::Union(variants) => {
            UnifiedType::Union(variants.iter().map(|v| avro_to_unified_type(v)).collect())
        }
        AvroSchema::Record(rec) => {
            UnifiedType::Object(rec.fields.iter().map(|f| UnifiedField {
                name: f.name.clone(),
                type_: avro_to_unified_type(&f.type_),
                required: f.default.is_none(),
                description: f.doc.clone(),
                deprecated: false,
            }).collect())
        }
        AvroSchema::Enum(en) => UnifiedType::Enum(en.symbols.clone()),
        AvroSchema::Fixed(f) => UnifiedType::Primitive(PrimitiveKind::Bytes),
        AvroSchema::Ref(name) => UnifiedType::Ref(name.clone()),
    }
}

// ---------------------------------------------------------------------------
// UnifiedDiff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDiffResult {
    pub added_endpoints: Vec<String>,
    pub removed_endpoints: Vec<String>,
    pub modified_endpoints: Vec<EndpointChange>,
    pub added_types: Vec<String>,
    pub removed_types: Vec<String>,
    pub modified_types: Vec<TypeChange>,
    pub is_breaking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointChange {
    pub path: String,
    pub method: String,
    pub kind: String,
    pub is_breaking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeChange {
    pub type_name: String,
    pub kind: String,
    pub is_breaking: bool,
}

pub struct UnifiedDiff;

impl UnifiedDiff {
    pub fn diff(old: &UnifiedSchema, new: &UnifiedSchema) -> UnifiedDiffResult {
        let mut added_endpoints = Vec::new();
        let mut removed_endpoints = Vec::new();
        let mut modified_endpoints = Vec::new();
        let mut added_types = Vec::new();
        let mut removed_types = Vec::new();
        let mut modified_types = Vec::new();
        let mut is_breaking = false;

        // Endpoint diff
        let old_ep_keys: HashMap<String, &UnifiedEndpoint> = old.endpoints.iter()
            .map(|e| (format!("{} {}", e.method, e.path), e)).collect();
        let new_ep_keys: HashMap<String, &UnifiedEndpoint> = new.endpoints.iter()
            .map(|e| (format!("{} {}", e.method, e.path), e)).collect();

        for key in new_ep_keys.keys() {
            if !old_ep_keys.contains_key(key) {
                added_endpoints.push(key.clone());
            }
        }
        for key in old_ep_keys.keys() {
            if !new_ep_keys.contains_key(key) {
                removed_endpoints.push(key.clone());
                is_breaking = true;
            }
        }

        for (key, old_ep) in &old_ep_keys {
            if let Some(new_ep) = new_ep_keys.get(key) {
                if old_ep.deprecated != new_ep.deprecated && new_ep.deprecated {
                    modified_endpoints.push(EndpointChange {
                        path: old_ep.path.clone(), method: old_ep.method.clone(),
                        kind: "deprecated".to_string(), is_breaking: false,
                    });
                }
                // Check required parameters
                for np in &new_ep.parameters {
                    let old_param = old_ep.parameters.iter().find(|p| p.name == np.name);
                    if old_param.is_none() && np.required {
                        is_breaking = true;
                        modified_endpoints.push(EndpointChange {
                            path: old_ep.path.clone(), method: old_ep.method.clone(),
                            kind: format!("added required param '{}'", np.name), is_breaking: true,
                        });
                    }
                }
                for op in &old_ep.parameters {
                    if !new_ep.parameters.iter().any(|p| p.name == op.name) {
                        is_breaking = true;
                        modified_endpoints.push(EndpointChange {
                            path: old_ep.path.clone(), method: old_ep.method.clone(),
                            kind: format!("removed param '{}'", op.name), is_breaking: true,
                        });
                    }
                }
            }
        }

        // Type diff
        for name in new.types.keys() {
            if !old.types.contains_key(name) {
                added_types.push(name.clone());
            }
        }
        for name in old.types.keys() {
            if !new.types.contains_key(name) {
                removed_types.push(name.clone());
                is_breaking = true;
            }
        }
        for (name, old_type) in &old.types {
            if let Some(new_type) = new.types.get(name) {
                if old_type.type_name() != new_type.type_name() {
                    is_breaking = true;
                    modified_types.push(TypeChange {
                        type_name: name.clone(),
                        kind: format!("type changed from {} to {}", old_type.type_name(), new_type.type_name()),
                        is_breaking: true,
                    });
                }
            }
        }

        UnifiedDiffResult {
            added_endpoints, removed_endpoints, modified_endpoints,
            added_types, removed_types, modified_types, is_breaking,
        }
    }
}

// ---------------------------------------------------------------------------
// CompatibilityClassifier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityPredicate {
    pub service_a: String,
    pub version_a: String,
    pub service_b: String,
    pub version_b: String,
    pub is_compatible: bool,
    pub confidence: f64,
    pub details: Vec<String>,
}

pub struct CompatibilityClassifier;

impl CompatibilityClassifier {
    pub fn classify(diff: &UnifiedDiffResult) -> Vec<CompatibilityPredicate> {
        let mut predicates = Vec::new();

        if diff.is_breaking {
            let mut details = Vec::new();
            for ep in &diff.removed_endpoints {
                details.push(format!("Removed endpoint: {}", ep));
            }
            for tc in &diff.modified_types {
                if tc.is_breaking {
                    details.push(format!("Breaking type change: {} - {}", tc.type_name, tc.kind));
                }
            }
            for ec in &diff.modified_endpoints {
                if ec.is_breaking {
                    details.push(format!("Breaking endpoint change: {} {} - {}", ec.method, ec.path, ec.kind));
                }
            }
            predicates.push(CompatibilityPredicate {
                service_a: String::new(),
                version_a: String::new(),
                service_b: String::new(),
                version_b: String::new(),
                is_compatible: false,
                confidence: 0.9,
                details,
            });
        } else {
            predicates.push(CompatibilityPredicate {
                service_a: String::new(),
                version_a: String::new(),
                service_b: String::new(),
                version_b: String::new(),
                is_compatible: true,
                confidence: 0.85,
                details: vec!["No breaking changes detected".to_string()],
            });
        }

        predicates
    }

    pub fn classify_with_context(
        diff: &UnifiedDiffResult,
        service_a: &str,
        version_a: &str,
        service_b: &str,
        version_b: &str,
    ) -> CompatibilityPredicate {
        let preds = Self::classify(diff);
        let base = &preds[0];
        CompatibilityPredicate {
            service_a: service_a.to_string(),
            version_a: version_a.to_string(),
            service_b: service_b.to_string(),
            version_b: version_b.to_string(),
            is_compatible: base.is_compatible,
            confidence: base.confidence,
            details: base.details.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// SchemaRegistry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryEntry {
    pub service: String,
    pub version: String,
    pub schema: UnifiedSchema,
}

#[derive(Debug, Clone, Default)]
pub struct SchemaRegistry {
    entries: HashMap<String, HashMap<String, UnifiedSchema>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMatrix {
    pub services: Vec<String>,
    pub versions: HashMap<String, Vec<String>>,
    pub compatibilities: Vec<CompatibilityPredicate>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, service: &str, version: &str, schema: UnifiedSchema) {
        self.entries
            .entry(service.to_string())
            .or_default()
            .insert(version.to_string(), schema);
    }

    pub fn get(&self, service: &str, version: &str) -> Option<&UnifiedSchema> {
        self.entries.get(service).and_then(|v| v.get(version))
    }

    pub fn services(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    pub fn versions(&self, service: &str) -> Vec<&str> {
        self.entries
            .get(service)
            .map(|v| v.keys().map(|k| k.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn total_entries(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    pub fn compute_all_compatibilities(&self) -> CompatibilityMatrix {
        let services: Vec<String> = self.entries.keys().cloned().collect();
        let mut versions: HashMap<String, Vec<String>> = HashMap::new();
        let mut compatibilities = Vec::new();

        for (svc, version_map) in &self.entries {
            let vers: Vec<String> = version_map.keys().cloned().collect();
            // Pairwise within same service
            for i in 0..vers.len() {
                for j in (i + 1)..vers.len() {
                    let s1 = &version_map[&vers[i]];
                    let s2 = &version_map[&vers[j]];
                    let diff = UnifiedDiff::diff(s1, s2);
                    compatibilities.push(CompatibilityClassifier::classify_with_context(
                        &diff, svc, &vers[i], svc, &vers[j],
                    ));
                }
            }
            versions.insert(svc.clone(), vers);
        }

        CompatibilityMatrix { services, versions, compatibilities }
    }

    pub fn is_compatible(&self, service: &str, v1: &str, v2: &str) -> Option<bool> {
        let s1 = self.get(service, v1)?;
        let s2 = self.get(service, v2)?;
        let diff = UnifiedDiff::diff(s1, s2);
        Some(!diff.is_breaking)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_openapi() -> OpenApiSchema {
        OpenApiSchema::parse(r#"
openapi: "3.0.0"
info:
  title: Test
  version: "1.0"
paths:
  /users:
    get:
      operationId: listUsers
      parameters:
        - name: limit
          in: query
          required: false
          schema:
            type: integer
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                type: array
                items:
                  type: object
                  properties:
                    id:
                      type: integer
                    name:
                      type: string
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
"#).unwrap()
    }

    #[test]
    fn test_from_openapi() {
        let schema = sample_openapi();
        let unified = UnifiedSchema::from_openapi(&schema);
        assert_eq!(unified.format, SchemaFormat::OpenApi);
        assert!(unified.endpoint_count() > 0);
        assert!(unified.type_count() > 0);
    }

    #[test]
    fn test_from_protobuf() {
        let proto = ProtobufSchema::parse_proto(r#"
syntax = "proto3";
package test;
message User {
    string name = 1;
    int32 age = 2;
}
service UserService {
    rpc GetUser(User) returns (User);
}
"#).unwrap();
        let unified = UnifiedSchema::from_protobuf(&proto);
        assert_eq!(unified.format, SchemaFormat::Protobuf);
        assert_eq!(unified.endpoint_count(), 1);
        assert!(unified.types.contains_key("User"));
    }

    #[test]
    fn test_from_graphql() {
        let gql = GraphqlSchema::parse(r#"
type Query { users: [User!]! }
type User { id: ID! name: String }
enum Status { ACTIVE INACTIVE }
"#).unwrap();
        let unified = UnifiedSchema::from_graphql(&gql);
        assert_eq!(unified.format, SchemaFormat::GraphQL);
        assert!(unified.endpoint_count() > 0);
        assert!(unified.types.contains_key("User"));
        assert!(unified.types.contains_key("Status"));
    }

    #[test]
    fn test_from_avro() {
        let avro = AvroSchema::parse(r#"{
            "type": "record",
            "name": "User",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "age", "type": "int"}
            ]
        }"#).unwrap();
        let unified = UnifiedSchema::from_avro(&avro);
        assert_eq!(unified.format, SchemaFormat::Avro);
        assert!(unified.types.contains_key("User"));
    }

    #[test]
    fn test_unified_diff_no_changes() {
        let s = UnifiedSchema::from_openapi(&sample_openapi());
        let diff = UnifiedDiff::diff(&s, &s);
        assert!(!diff.is_breaking);
        assert!(diff.added_endpoints.is_empty());
        assert!(diff.removed_endpoints.is_empty());
    }

    #[test]
    fn test_unified_diff_removed_endpoint() {
        let s1 = UnifiedSchema::from_openapi(&sample_openapi());
        let mut s2 = s1.clone();
        s2.endpoints.clear();
        let diff = UnifiedDiff::diff(&s1, &s2);
        assert!(diff.is_breaking);
        assert!(!diff.removed_endpoints.is_empty());
    }

    #[test]
    fn test_unified_diff_added_type() {
        let s1 = UnifiedSchema::from_openapi(&sample_openapi());
        let mut s2 = s1.clone();
        s2.types.insert("NewType".to_string(), UnifiedType::Primitive(PrimitiveKind::String));
        let diff = UnifiedDiff::diff(&s1, &s2);
        assert!(!diff.is_breaking);
        assert!(diff.added_types.contains(&"NewType".to_string()));
    }

    #[test]
    fn test_compatibility_classifier_breaking() {
        let s1 = UnifiedSchema::from_openapi(&sample_openapi());
        let mut s2 = s1.clone();
        s2.endpoints.clear();
        let diff = UnifiedDiff::diff(&s1, &s2);
        let preds = CompatibilityClassifier::classify(&diff);
        assert!(!preds.is_empty());
        assert!(!preds[0].is_compatible);
    }

    #[test]
    fn test_compatibility_classifier_compatible() {
        let s = UnifiedSchema::from_openapi(&sample_openapi());
        let diff = UnifiedDiff::diff(&s, &s);
        let preds = CompatibilityClassifier::classify(&diff);
        assert!(preds[0].is_compatible);
    }

    #[test]
    fn test_schema_registry() {
        let mut reg = SchemaRegistry::new();
        let s1 = UnifiedSchema::from_openapi(&sample_openapi());
        let s2 = s1.clone();
        reg.register("user-service", "1.0", s1);
        reg.register("user-service", "2.0", s2);
        assert_eq!(reg.total_entries(), 2);
        assert!(reg.get("user-service", "1.0").is_some());
        assert!(reg.is_compatible("user-service", "1.0", "2.0").unwrap());
    }

    #[test]
    fn test_compatibility_matrix() {
        let mut reg = SchemaRegistry::new();
        let s = UnifiedSchema::from_openapi(&sample_openapi());
        reg.register("svc", "1.0", s.clone());
        reg.register("svc", "2.0", s);
        let matrix = reg.compute_all_compatibilities();
        assert!(!matrix.compatibilities.is_empty());
        assert!(matrix.compatibilities[0].is_compatible);
    }

    #[test]
    fn test_unified_type_helpers() {
        let prim = UnifiedType::Primitive(PrimitiveKind::String);
        assert!(prim.is_primitive());
        assert!(!prim.is_complex());
        assert_eq!(prim.type_name(), "primitive");

        let obj = UnifiedType::Object(vec![
            UnifiedField { name: "a".into(), type_: UnifiedType::Primitive(PrimitiveKind::Integer), required: true, description: None, deprecated: false },
        ]);
        assert!(obj.is_complex());
        assert_eq!(obj.field_count(), 1);
    }

    #[test]
    fn test_predicate_with_context() {
        let s = UnifiedSchema::from_openapi(&sample_openapi());
        let diff = UnifiedDiff::diff(&s, &s);
        let pred = CompatibilityClassifier::classify_with_context(
            &diff, "svc-a", "1.0", "svc-b", "2.0",
        );
        assert_eq!(pred.service_a, "svc-a");
        assert_eq!(pred.version_a, "1.0");
        assert!(pred.is_compatible);
    }
}
