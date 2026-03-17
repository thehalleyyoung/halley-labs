//! OpenAPI 3.x schema parsing.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use safestep_types::{SafeStepError, Result};

// ---------------------------------------------------------------------------
// Core schema types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaObject {
    pub type_: Option<String>,
    pub format: Option<String>,
    pub properties: IndexMap<String, SchemaObject>,
    pub required: Vec<String>,
    pub additional_properties: Option<Box<SchemaObject>>,
    pub items: Option<Box<SchemaObject>>,
    pub all_of: Vec<SchemaObject>,
    pub any_of: Vec<SchemaObject>,
    pub one_of: Vec<SchemaObject>,
    pub enum_values: Vec<serde_json::Value>,
    pub description: Option<String>,
    pub nullable: bool,
    pub read_only: bool,
    pub write_only: bool,
    pub ref_path: Option<String>,
    pub default: Option<serde_json::Value>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub min_items: Option<usize>,
    pub max_items: Option<usize>,
    pub unique_items: bool,
}

impl SchemaObject {
    pub fn is_ref(&self) -> bool {
        self.ref_path.is_some()
    }

    pub fn is_primitive(&self) -> bool {
        matches!(
            self.type_.as_deref(),
            Some("string") | Some("integer") | Some("number") | Some("boolean")
        )
    }

    pub fn is_object(&self) -> bool {
        self.type_.as_deref() == Some("object") || !self.properties.is_empty()
    }

    pub fn is_array(&self) -> bool {
        self.type_.as_deref() == Some("array")
    }

    pub fn effective_type(&self) -> &str {
        if self.is_ref() {
            "ref"
        } else if let Some(t) = &self.type_ {
            t.as_str()
        } else if !self.properties.is_empty() {
            "object"
        } else if !self.all_of.is_empty() || !self.any_of.is_empty() || !self.one_of.is_empty() {
            "composed"
        } else {
            "unknown"
        }
    }

    pub fn has_constraints(&self) -> bool {
        self.minimum.is_some()
            || self.maximum.is_some()
            || self.min_length.is_some()
            || self.max_length.is_some()
            || self.pattern.is_some()
            || self.min_items.is_some()
            || self.max_items.is_some()
            || self.unique_items
            || !self.enum_values.is_empty()
    }

    pub fn merge_all_of(&self) -> SchemaObject {
        if self.all_of.is_empty() {
            return self.clone();
        }
        let mut merged = SchemaObject::default();
        merged.type_ = Some("object".to_string());
        for sub in &self.all_of {
            for (k, v) in &sub.properties {
                merged.properties.insert(k.clone(), v.clone());
            }
            for r in &sub.required {
                if !merged.required.contains(r) {
                    merged.required.push(r.clone());
                }
            }
            if merged.description.is_none() {
                merged.description.clone_from(&sub.description);
            }
        }
        for (k, v) in &self.properties {
            merged.properties.insert(k.clone(), v.clone());
        }
        for r in &self.required {
            if !merged.required.contains(r) {
                merged.required.push(r.clone());
            }
        }
        merged
    }

    pub fn resolve_ref(&self, components: &IndexMap<String, SchemaObject>) -> Result<SchemaObject> {
        if let Some(ref rp) = self.ref_path {
            let name = rp
                .strip_prefix("#/components/schemas/")
                .unwrap_or(rp.as_str());
            components
                .get(name)
                .cloned()
                .ok_or_else(|| SafeStepError::schema(format!("Unresolved $ref: {}", rp)))
        } else {
            Ok(self.clone())
        }
    }

    pub fn from_value(v: &serde_json::Value) -> Result<Self> {
        let mut s = SchemaObject::default();
        let obj = match v.as_object() {
            Some(o) => o,
            None => return Ok(s),
        };
        if let Some(r) = obj.get("$ref").and_then(|v| v.as_str()) {
            s.ref_path = Some(r.to_string());
            return Ok(s);
        }
        s.type_ = obj.get("type").and_then(|v| v.as_str()).map(|x| x.to_string());
        s.format = obj.get("format").and_then(|v| v.as_str()).map(|x| x.to_string());
        s.description = obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string());
        s.nullable = obj.get("nullable").and_then(|v| v.as_bool()).unwrap_or(false);
        s.read_only = obj.get("readOnly").and_then(|v| v.as_bool()).unwrap_or(false);
        s.write_only = obj.get("writeOnly").and_then(|v| v.as_bool()).unwrap_or(false);
        s.unique_items = obj.get("uniqueItems").and_then(|v| v.as_bool()).unwrap_or(false);
        s.default = obj.get("default").cloned();
        s.minimum = obj.get("minimum").and_then(|v| v.as_f64());
        s.maximum = obj.get("maximum").and_then(|v| v.as_f64());
        s.min_length = obj.get("minLength").and_then(|v| v.as_u64()).map(|n| n as usize);
        s.max_length = obj.get("maxLength").and_then(|v| v.as_u64()).map(|n| n as usize);
        s.pattern = obj.get("pattern").and_then(|v| v.as_str()).map(|x| x.to_string());
        s.min_items = obj.get("minItems").and_then(|v| v.as_u64()).map(|n| n as usize);
        s.max_items = obj.get("maxItems").and_then(|v| v.as_u64()).map(|n| n as usize);

        if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
            for (k, val) in props {
                s.properties.insert(k.clone(), SchemaObject::from_value(val)?);
            }
        }
        if let Some(req) = obj.get("required").and_then(|v| v.as_array()) {
            s.required = req.iter().filter_map(|v| v.as_str().map(|x| x.to_string())).collect();
        }
        if let Some(ap) = obj.get("additionalProperties") {
            if ap.is_object() {
                s.additional_properties = Some(Box::new(SchemaObject::from_value(ap)?));
            }
        }
        if let Some(items) = obj.get("items") {
            s.items = Some(Box::new(SchemaObject::from_value(items)?));
        }
        for (key, field) in [("allOf", &mut s.all_of), ("anyOf", &mut s.any_of), ("oneOf", &mut s.one_of)] {
            if let Some(arr) = obj.get(key).and_then(|v| v.as_array()) {
                for item in arr {
                    field.push(SchemaObject::from_value(item)?);
                }
            }
        }
        if let Some(ev) = obj.get("enum").and_then(|v| v.as_array()) {
            s.enum_values = ev.clone();
        }
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// Parameter / Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ParameterLocation {
    Query,
    Header,
    Path,
    Cookie,
}

impl ParameterLocation {
    fn from_str_val(s: &str) -> Self {
        match s {
            "header" => Self::Header,
            "path" => Self::Path,
            "cookie" => Self::Cookie,
            _ => Self::Query,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiParameter {
    pub name: String,
    pub location: ParameterLocation,
    pub required: bool,
    pub schema: Option<SchemaObject>,
    pub description: Option<String>,
    pub deprecated: bool,
}

impl OpenApiParameter {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("Parameter must be object"))?;
        Ok(Self {
            name: obj.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            location: ParameterLocation::from_str_val(
                obj.get("in").and_then(|v| v.as_str()).unwrap_or("query"),
            ),
            required: obj.get("required").and_then(|v| v.as_bool()).unwrap_or(false),
            schema: obj.get("schema").map(SchemaObject::from_value).transpose()?,
            description: obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
            deprecated: obj.get("deprecated").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaType {
    pub schema: Option<SchemaObject>,
    pub example: Option<serde_json::Value>,
}

impl MediaType {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("MediaType must be object"))?;
        Ok(Self {
            schema: obj.get("schema").map(SchemaObject::from_value).transpose()?,
            example: obj.get("example").cloned(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiRequestBody {
    pub description: Option<String>,
    pub required: bool,
    pub content: IndexMap<String, MediaType>,
}

impl OpenApiRequestBody {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("RequestBody must be object"))?;
        let mut content = IndexMap::new();
        if let Some(c) = obj.get("content").and_then(|v| v.as_object()) {
            for (k, val) in c {
                content.insert(k.clone(), MediaType::from_value(val)?);
            }
        }
        Ok(Self {
            description: obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
            required: obj.get("required").and_then(|v| v.as_bool()).unwrap_or(false),
            content,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiResponse {
    pub description: String,
    pub content: IndexMap<String, MediaType>,
    pub headers: IndexMap<String, OpenApiParameter>,
}

impl OpenApiResponse {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("Response must be object"))?;
        let mut content = IndexMap::new();
        if let Some(c) = obj.get("content").and_then(|v| v.as_object()) {
            for (k, val) in c {
                content.insert(k.clone(), MediaType::from_value(val)?);
            }
        }
        let mut headers = IndexMap::new();
        if let Some(h) = obj.get("headers").and_then(|v| v.as_object()) {
            for (k, val) in h {
                headers.insert(k.clone(), OpenApiParameter::from_value(val)?);
            }
        }
        Ok(Self {
            description: obj.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            content,
            headers,
        })
    }
}

// ---------------------------------------------------------------------------
// Operation / Path
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiOperation {
    pub operation_id: Option<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub parameters: Vec<OpenApiParameter>,
    pub request_body: Option<OpenApiRequestBody>,
    pub responses: IndexMap<String, OpenApiResponse>,
    pub deprecated: bool,
    pub security: Vec<IndexMap<String, Vec<String>>>,
}

impl OpenApiOperation {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("Operation must be object"))?;
        let mut params = Vec::new();
        if let Some(arr) = obj.get("parameters").and_then(|v| v.as_array()) {
            for p in arr {
                params.push(OpenApiParameter::from_value(p)?);
            }
        }
        let mut responses = IndexMap::new();
        if let Some(r) = obj.get("responses").and_then(|v| v.as_object()) {
            for (k, val) in r {
                responses.insert(k.clone(), OpenApiResponse::from_value(val)?);
            }
        }
        let tags = obj.get("tags").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|x| x.to_string())).collect())
            .unwrap_or_default();
        let security = obj.get("security").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|item| {
                item.as_object().map(|o| o.iter().map(|(k, v)| {
                    let scopes = v.as_array()
                        .map(|a| a.iter().filter_map(|s| s.as_str().map(|x| x.to_string())).collect())
                        .unwrap_or_default();
                    (k.clone(), scopes)
                }).collect())
            }).collect())
            .unwrap_or_default();

        Ok(Self {
            operation_id: obj.get("operationId").and_then(|v| v.as_str()).map(|x| x.to_string()),
            summary: obj.get("summary").and_then(|v| v.as_str()).map(|x| x.to_string()),
            description: obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
            tags,
            parameters: params,
            request_body: obj.get("requestBody").map(OpenApiRequestBody::from_value).transpose()?,
            responses,
            deprecated: obj.get("deprecated").and_then(|v| v.as_bool()).unwrap_or(false),
            security,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiPath {
    pub path: String,
    pub get: Option<OpenApiOperation>,
    pub post: Option<OpenApiOperation>,
    pub put: Option<OpenApiOperation>,
    pub delete: Option<OpenApiOperation>,
    pub patch: Option<OpenApiOperation>,
    pub options: Option<OpenApiOperation>,
    pub head: Option<OpenApiOperation>,
    pub parameters: Vec<OpenApiParameter>,
    pub summary: Option<String>,
    pub description: Option<String>,
}

impl OpenApiPath {
    pub fn operations(&self) -> Vec<(&str, &OpenApiOperation)> {
        let mut ops = Vec::new();
        if let Some(ref op) = self.get { ops.push(("get", op)); }
        if let Some(ref op) = self.post { ops.push(("post", op)); }
        if let Some(ref op) = self.put { ops.push(("put", op)); }
        if let Some(ref op) = self.delete { ops.push(("delete", op)); }
        if let Some(ref op) = self.patch { ops.push(("patch", op)); }
        if let Some(ref op) = self.options { ops.push(("options", op)); }
        if let Some(ref op) = self.head { ops.push(("head", op)); }
        ops
    }

    pub fn operation_count(&self) -> usize {
        self.operations().len()
    }

    pub fn operations_mut(&mut self) -> Vec<(&str, &mut OpenApiOperation)> {
        let mut ops: Vec<(&str, &mut OpenApiOperation)> = Vec::new();
        if let Some(ref mut op) = self.get { ops.push(("get", op)); }
        if let Some(ref mut op) = self.post { ops.push(("post", op)); }
        if let Some(ref mut op) = self.put { ops.push(("put", op)); }
        if let Some(ref mut op) = self.delete { ops.push(("delete", op)); }
        if let Some(ref mut op) = self.patch { ops.push(("patch", op)); }
        if let Some(ref mut op) = self.options { ops.push(("options", op)); }
        if let Some(ref mut op) = self.head { ops.push(("head", op)); }
        ops
    }

    fn from_value(path_str: &str, v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("Path must be object"))?;
        let mut params = Vec::new();
        if let Some(arr) = obj.get("parameters").and_then(|v| v.as_array()) {
            for p in arr {
                params.push(OpenApiParameter::from_value(p)?);
            }
        }
        Ok(Self {
            path: path_str.to_string(),
            get: obj.get("get").map(OpenApiOperation::from_value).transpose()?,
            post: obj.get("post").map(OpenApiOperation::from_value).transpose()?,
            put: obj.get("put").map(OpenApiOperation::from_value).transpose()?,
            delete: obj.get("delete").map(OpenApiOperation::from_value).transpose()?,
            patch: obj.get("patch").map(OpenApiOperation::from_value).transpose()?,
            options: obj.get("options").map(OpenApiOperation::from_value).transpose()?,
            head: obj.get("head").map(OpenApiOperation::from_value).transpose()?,
            parameters: params,
            summary: obj.get("summary").and_then(|v| v.as_str()).map(|x| x.to_string()),
            description: obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
        })
    }
}

// ---------------------------------------------------------------------------
// Security / Server / Tag
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecuritySchemeType { ApiKey, Http, OAuth2, OpenIdConnect }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScheme {
    pub type_: SecuritySchemeType,
    pub name: Option<String>,
    pub location: Option<String>,
    pub scheme: Option<String>,
    pub bearer_format: Option<String>,
    pub description: Option<String>,
}

impl SecurityScheme {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("SecurityScheme must be object"))?;
        let type_ = match obj.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "http" => SecuritySchemeType::Http,
            "oauth2" => SecuritySchemeType::OAuth2,
            "openIdConnect" => SecuritySchemeType::OpenIdConnect,
            _ => SecuritySchemeType::ApiKey,
        };
        Ok(Self {
            type_,
            name: obj.get("name").and_then(|v| v.as_str()).map(|x| x.to_string()),
            location: obj.get("in").and_then(|v| v.as_str()).map(|x| x.to_string()),
            scheme: obj.get("scheme").and_then(|v| v.as_str()).map(|x| x.to_string()),
            bearer_format: obj.get("bearerFormat").and_then(|v| v.as_str()).map(|x| x.to_string()),
            description: obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerVariable {
    pub default: String,
    pub description: Option<String>,
    pub enum_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub url: String,
    pub description: Option<String>,
    pub variables: IndexMap<String, ServerVariable>,
}

impl ServerInfo {
    fn from_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SafeStepError::schema("Server must be object"))?;
        let mut variables = IndexMap::new();
        if let Some(vars) = obj.get("variables").and_then(|v| v.as_object()) {
            for (k, val) in vars {
                let empty = serde_json::Map::new();
                let vo = val.as_object().unwrap_or(&empty);
                variables.insert(k.clone(), ServerVariable {
                    default: vo.get("default").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    description: vo.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
                    enum_values: vo.get("enum").and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(|x| x.to_string())).collect())
                        .unwrap_or_default(),
                });
            }
        }
        Ok(Self {
            url: obj.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            description: obj.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
            variables,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagInfo {
    pub name: String,
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// Top-level OpenApiSchema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiSchema {
    pub openapi_version: String,
    pub title: String,
    pub version: String,
    pub description: Option<String>,
    pub paths: IndexMap<String, OpenApiPath>,
    pub components: IndexMap<String, SchemaObject>,
    pub security_schemes: IndexMap<String, SecurityScheme>,
    pub servers: Vec<ServerInfo>,
    pub tags: Vec<TagInfo>,
}

impl OpenApiSchema {
    pub fn parse(yaml_str: &str) -> Result<Self> {
        let value: serde_json::Value = serde_yaml::from_str(yaml_str)
            .map_err(|e| SafeStepError::schema(format!("YAML parse error: {}", e)))?;
        Self::from_value(&value)
    }

    pub fn parse_json(json_str: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| SafeStepError::schema(format!("JSON parse error: {}", e)))?;
        Self::from_value(&value)
    }

    fn from_value(root: &serde_json::Value) -> Result<Self> {
        let obj = root.as_object().ok_or_else(|| SafeStepError::schema("Root must be object"))?;
        let openapi_version = obj.get("openapi").and_then(|v| v.as_str()).unwrap_or("3.0.0").to_string();
        let info = obj.get("info").and_then(|v| v.as_object());
        let title = info.and_then(|i| i.get("title")).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let version = info.and_then(|i| i.get("version")).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let description = info.and_then(|i| i.get("description")).and_then(|v| v.as_str()).map(|x| x.to_string());

        let mut paths = IndexMap::new();
        if let Some(p) = obj.get("paths").and_then(|v| v.as_object()) {
            for (k, val) in p {
                paths.insert(k.clone(), OpenApiPath::from_value(k, val)?);
            }
        }

        let mut components = IndexMap::new();
        if let Some(c) = obj.get("components").and_then(|v| v.as_object()) {
            if let Some(schemas) = c.get("schemas").and_then(|v| v.as_object()) {
                for (k, val) in schemas {
                    components.insert(k.clone(), SchemaObject::from_value(val)?);
                }
            }
        }

        let mut security_schemes = IndexMap::new();
        if let Some(c) = obj.get("components").and_then(|v| v.as_object()) {
            if let Some(ss) = c.get("securitySchemes").and_then(|v| v.as_object()) {
                for (k, val) in ss {
                    security_schemes.insert(k.clone(), SecurityScheme::from_value(val)?);
                }
            }
        }

        let mut servers = Vec::new();
        if let Some(arr) = obj.get("servers").and_then(|v| v.as_array()) {
            for s in arr { servers.push(ServerInfo::from_value(s)?); }
        }

        let tags = obj.get("tags").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|t| t.as_object().map(|o| TagInfo {
                name: o.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                description: o.get("description").and_then(|v| v.as_str()).map(|x| x.to_string()),
            })).collect())
            .unwrap_or_default();

        Ok(Self { openapi_version, title, version, description, paths, components, security_schemes, servers, tags })
    }

    pub fn endpoint_count(&self) -> usize {
        self.paths.values().map(|p| p.operation_count()).sum()
    }

    pub fn schema_count(&self) -> usize {
        self.components.len()
    }

    pub fn all_operations(&self) -> Vec<(String, String, &OpenApiOperation)> {
        let mut ops = Vec::new();
        for (path, pi) in &self.paths {
            for (method, op) in pi.operations() {
                ops.push((path.clone(), method.to_string(), op));
            }
        }
        ops
    }

    pub fn find_operation(&self, path: &str, method: &str) -> Option<&OpenApiOperation> {
        self.paths.get(path).and_then(|pi| match method.to_lowercase().as_str() {
            "get" => pi.get.as_ref(),
            "post" => pi.post.as_ref(),
            "put" => pi.put.as_ref(),
            "delete" => pi.delete.as_ref(),
            "patch" => pi.patch.as_ref(),
            "options" => pi.options.as_ref(),
            "head" => pi.head.as_ref(),
            _ => None,
        })
    }

    pub fn find_schema(&self, name: &str) -> Option<&SchemaObject> {
        self.components.get(name)
    }

    pub fn resolve_all_refs(&mut self) {
        let components = self.components.clone();
        for pi in self.paths.values_mut() {
            for (_, op) in pi.operations_mut() {
                for param in &mut op.parameters {
                    if let Some(ref mut s) = param.schema {
                        resolve_schema_refs(s, &components);
                    }
                }
                if let Some(ref mut rb) = op.request_body {
                    for mt in rb.content.values_mut() {
                        if let Some(ref mut s) = mt.schema {
                            resolve_schema_refs(s, &components);
                        }
                    }
                }
                for resp in op.responses.values_mut() {
                    for mt in resp.content.values_mut() {
                        if let Some(ref mut s) = mt.schema {
                            resolve_schema_refs(s, &components);
                        }
                    }
                }
            }
        }
    }
}

fn resolve_schema_refs(schema: &mut SchemaObject, components: &IndexMap<String, SchemaObject>) {
    if let Some(ref rp) = schema.ref_path {
        let name = rp.strip_prefix("#/components/schemas/").unwrap_or(rp.as_str());
        if let Some(resolved) = components.get(name) {
            *schema = resolved.clone();
        }
    }
    for sub in schema.properties.values_mut() {
        resolve_schema_refs(sub, components);
    }
    if let Some(ref mut items) = schema.items {
        resolve_schema_refs(items, components);
    }
    if let Some(ref mut ap) = schema.additional_properties {
        resolve_schema_refs(ap, components);
    }
    for sub in &mut schema.all_of { resolve_schema_refs(sub, components); }
    for sub in &mut schema.any_of { resolve_schema_refs(sub, components); }
    for sub in &mut schema.one_of { resolve_schema_refs(sub, components); }
}

// ---------------------------------------------------------------------------
// OpenApiParser builder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OpenApiParser {
    pub strict: bool,
    pub resolve_external: bool,
}

impl OpenApiParser {
    pub fn new() -> Self { Self { strict: false, resolve_external: false } }
    pub fn strict(mut self) -> Self { self.strict = true; self }
    pub fn with_external_refs(mut self) -> Self { self.resolve_external = true; self }

    pub fn parse_yaml(&self, yaml: &str) -> Result<OpenApiSchema> {
        let mut schema = OpenApiSchema::parse(yaml)?;
        if self.strict { self.validate(&schema)?; }
        schema.resolve_all_refs();
        Ok(schema)
    }

    pub fn parse_json(&self, json: &str) -> Result<OpenApiSchema> {
        let mut schema = OpenApiSchema::parse_json(json)?;
        if self.strict { self.validate(&schema)?; }
        schema.resolve_all_refs();
        Ok(schema)
    }

    fn validate(&self, schema: &OpenApiSchema) -> Result<()> {
        if schema.openapi_version.is_empty() {
            return Err(SafeStepError::schema("Missing openapi version"));
        }
        if !schema.openapi_version.starts_with("3.") {
            return Err(SafeStepError::schema(
                format!("Unsupported OpenAPI version: {}", schema.openapi_version),
            ));
        }
        if schema.title.is_empty() {
            return Err(SafeStepError::schema("Missing info.title"));
        }
        if schema.paths.is_empty() {
            return Err(SafeStepError::schema("No paths defined"));
        }
        for (path, pi) in &schema.paths {
            if !path.starts_with('/') {
                return Err(SafeStepError::schema(format!("Path must start with /: {}", path)));
            }
            if pi.operation_count() == 0 {
                return Err(SafeStepError::schema(format!("Path has no operations: {}", path)));
            }
        }
        Ok(())
    }
}

impl Default for OpenApiParser {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_YAML: &str = r##"
openapi: "3.0.3"
info:
  title: Pet Store
  version: "1.0.0"
  description: A pet store API
paths:
  /pets:
    get:
      operationId: listPets
      summary: List all pets
      parameters:
        - name: limit
          in: query
          required: false
          schema:
            type: integer
      responses:
        "200":
          description: A list of pets
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Pet"
    post:
      operationId: createPet
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/Pet"
      responses:
        "201":
          description: Pet created
  /pets/{petId}:
    get:
      operationId: getPet
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: A pet
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Pet"
components:
  schemas:
    Pet:
      type: object
      required:
        - id
        - name
      properties:
        id:
          type: integer
          format: int64
        name:
          type: string
        tag:
          type: string
  securitySchemes:
    api_key:
      type: apiKey
      name: api_key
      in: header
servers:
  - url: https://api.example.com/v1
    description: Production
tags:
  - name: pets
    description: Pet operations
"##;

    #[test]
    fn test_parse_minimal_yaml() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        assert_eq!(schema.openapi_version, "3.0.3");
        assert_eq!(schema.title, "Pet Store");
        assert_eq!(schema.version, "1.0.0");
        assert_eq!(schema.paths.len(), 2);
        assert_eq!(schema.components.len(), 1);
        assert_eq!(schema.endpoint_count(), 3);
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{"openapi":"3.0.0","info":{"title":"T","version":"0.1"},"paths":{"/h":{"get":{"operationId":"h","responses":{"200":{"description":"ok"}}}}}}"#;
        let schema = OpenApiSchema::parse_json(json).unwrap();
        assert_eq!(schema.title, "T");
        assert_eq!(schema.endpoint_count(), 1);
    }

    #[test]
    fn test_all_operations() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        let ops = schema.all_operations();
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn test_find_operation() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        assert!(schema.find_operation("/pets", "get").is_some());
        assert!(schema.find_operation("/pets", "delete").is_none());
    }

    #[test]
    fn test_find_schema() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        let pet = schema.find_schema("Pet").unwrap();
        assert!(pet.is_object());
        assert!(pet.required.contains(&"id".to_string()));
    }

    #[test]
    fn test_ref_resolution() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        let ref_obj = SchemaObject { ref_path: Some("#/components/schemas/Pet".into()), ..Default::default() };
        let resolved = ref_obj.resolve_ref(&schema.components).unwrap();
        assert!(resolved.properties.contains_key("id"));
    }

    #[test]
    fn test_parameter_parsing() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        let op = schema.find_operation("/pets", "get").unwrap();
        assert_eq!(op.parameters[0].name, "limit");
        assert_eq!(op.parameters[0].location, ParameterLocation::Query);
    }

    #[test]
    fn test_request_body() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        let op = schema.find_operation("/pets", "post").unwrap();
        assert!(op.request_body.as_ref().unwrap().required);
    }

    #[test]
    fn test_security_schemes() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        assert_eq!(schema.security_schemes.get("api_key").unwrap().type_, SecuritySchemeType::ApiKey);
    }

    #[test]
    fn test_servers() {
        let schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        assert_eq!(schema.servers[0].url, "https://api.example.com/v1");
    }

    #[test]
    fn test_strict_parser_rejects_v2() {
        let bad = r#"openapi: "2.0"
info:
  title: Old
  version: "1"
paths:
  /x:
    get:
      responses:
        "200":
          description: ok"#;
        assert!(OpenApiParser::new().strict().parse_yaml(bad).is_err());
    }

    #[test]
    fn test_resolve_all_refs() {
        let mut schema = OpenApiSchema::parse(MINIMAL_YAML).unwrap();
        schema.resolve_all_refs();
        let op = schema.find_operation("/pets/{petId}", "get").unwrap();
        let s = op.responses.get("200").unwrap().content.get("application/json").unwrap().schema.as_ref().unwrap();
        assert!(s.ref_path.is_none());
        assert!(s.properties.contains_key("id"));
    }

    #[test]
    fn test_schema_object_types() {
        let mut s = SchemaObject::default();
        assert!(!s.is_primitive());
        s.type_ = Some("string".into());
        assert!(s.is_primitive());
        s.type_ = Some("array".into());
        assert!(s.is_array());
    }

    #[test]
    fn test_merge_all_of() {
        let s = SchemaObject {
            all_of: vec![
                SchemaObject { properties: { let mut m = IndexMap::new(); m.insert("a".into(), SchemaObject::default()); m }, required: vec!["a".into()], ..Default::default() },
                SchemaObject { properties: { let mut m = IndexMap::new(); m.insert("b".into(), SchemaObject::default()); m }, ..Default::default() },
            ],
            ..Default::default()
        };
        let merged = s.merge_all_of();
        assert!(merged.properties.contains_key("a"));
        assert!(merged.properties.contains_key("b"));
    }

    #[test]
    fn test_effective_type() {
        assert_eq!(SchemaObject { type_: Some("string".into()), ..Default::default() }.effective_type(), "string");
        assert_eq!(SchemaObject { ref_path: Some("x".into()), ..Default::default() }.effective_type(), "ref");
        assert_eq!(SchemaObject { all_of: vec![SchemaObject::default()], ..Default::default() }.effective_type(), "composed");
        assert_eq!(SchemaObject::default().effective_type(), "unknown");
    }

    #[test]
    fn test_has_constraints() {
        assert!(SchemaObject { minimum: Some(0.0), ..Default::default() }.has_constraints());
        assert!(!SchemaObject::default().has_constraints());
    }
}
