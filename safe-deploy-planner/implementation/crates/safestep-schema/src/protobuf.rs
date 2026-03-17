//! Protobuf schema parsing and representation.
//!
//! Parses `.proto` files (proto2 and proto3 syntax) into a structured
//! [`ProtobufSchema`] representation. Supports messages, enums, services,
//! oneofs, map fields, reserved ranges, nested types, and extensions.

use std::fmt;

use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};

use safestep_types::{Result, SafeStepError};

// ---------------------------------------------------------------------------
// ProtoFieldType
// ---------------------------------------------------------------------------

/// Represents a protobuf field type including scalars, messages, enums, and maps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtoFieldType {
    Double,
    Float,
    Int32,
    Int64,
    Uint32,
    Uint64,
    Sint32,
    Sint64,
    Fixed32,
    Fixed64,
    Sfixed32,
    Sfixed64,
    Bool,
    String,
    Bytes,
    Message(std::string::String),
    Enum(std::string::String),
    Map {
        key: Box<ProtoFieldType>,
        value: Box<ProtoFieldType>,
    },
}

impl ProtoFieldType {
    /// Returns the protobuf wire type used for encoding.
    ///
    /// - 0 = varint (int32, int64, uint32, uint64, sint32, sint64, bool, enum)
    /// - 1 = 64-bit (fixed64, sfixed64, double)
    /// - 2 = length-delimited (string, bytes, message, map, repeated packed)
    /// - 5 = 32-bit (fixed32, sfixed32, float)
    pub fn wire_type(&self) -> u8 {
        match self {
            Self::Int32
            | Self::Int64
            | Self::Uint32
            | Self::Uint64
            | Self::Sint32
            | Self::Sint64
            | Self::Bool
            | Self::Enum(_) => 0,
            Self::Fixed64 | Self::Sfixed64 | Self::Double => 1,
            Self::String
            | Self::Bytes
            | Self::Message(_)
            | Self::Map { .. } => 2,
            Self::Fixed32 | Self::Sfixed32 | Self::Float => 5,
        }
    }

    /// Parse a type name string into a `ProtoFieldType`.
    pub fn from_name(name: &str) -> Self {
        match name {
            "double" => Self::Double,
            "float" => Self::Float,
            "int32" => Self::Int32,
            "int64" => Self::Int64,
            "uint32" => Self::Uint32,
            "uint64" => Self::Uint64,
            "sint32" => Self::Sint32,
            "sint64" => Self::Sint64,
            "fixed32" => Self::Fixed32,
            "fixed64" => Self::Fixed64,
            "sfixed32" => Self::Sfixed32,
            "sfixed64" => Self::Sfixed64,
            "bool" => Self::Bool,
            "string" => Self::String,
            "bytes" => Self::Bytes,
            other => Self::Message(other.to_string()),
        }
    }
}

impl fmt::Display for ProtoFieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Double => write!(f, "double"),
            Self::Float => write!(f, "float"),
            Self::Int32 => write!(f, "int32"),
            Self::Int64 => write!(f, "int64"),
            Self::Uint32 => write!(f, "uint32"),
            Self::Uint64 => write!(f, "uint64"),
            Self::Sint32 => write!(f, "sint32"),
            Self::Sint64 => write!(f, "sint64"),
            Self::Fixed32 => write!(f, "fixed32"),
            Self::Fixed64 => write!(f, "fixed64"),
            Self::Sfixed32 => write!(f, "sfixed32"),
            Self::Sfixed64 => write!(f, "sfixed64"),
            Self::Bool => write!(f, "bool"),
            Self::String => write!(f, "string"),
            Self::Bytes => write!(f, "bytes"),
            Self::Message(name) => write!(f, "{name}"),
            Self::Enum(name) => write!(f, "{name}"),
            Self::Map { key, value } => write!(f, "map<{key}, {value}>"),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldLabel
// ---------------------------------------------------------------------------

/// Protobuf field cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldLabel {
    Optional,
    Required,
    Repeated,
}

impl fmt::Display for FieldLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Optional => write!(f, "optional"),
            Self::Required => write!(f, "required"),
            Self::Repeated => write!(f, "repeated"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReservedRange
// ---------------------------------------------------------------------------

/// An inclusive range of reserved field numbers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservedRange {
    pub start: u32,
    pub end: u32,
}

impl ReservedRange {
    pub fn contains(&self, number: u32) -> bool {
        number >= self.start && number <= self.end
    }
}

// ---------------------------------------------------------------------------
// ProtoField
// ---------------------------------------------------------------------------

/// A single field within a protobuf message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoField {
    pub name: std::string::String,
    pub number: u32,
    pub type_: ProtoFieldType,
    pub label: FieldLabel,
    pub default_value: Option<std::string::String>,
    pub options: IndexMap<std::string::String, std::string::String>,
    pub oneof_index: Option<usize>,
    pub json_name: Option<std::string::String>,
    pub deprecated: bool,
}

impl ProtoField {
    /// Create a new field with sensible defaults.
    pub fn new(
        name: impl Into<std::string::String>,
        number: u32,
        type_: ProtoFieldType,
        label: FieldLabel,
    ) -> Self {
        let name = name.into();
        let json_name = Some(to_camel_case(&name));
        Self {
            name,
            number,
            type_,
            label,
            default_value: None,
            options: IndexMap::new(),
            oneof_index: None,
            json_name,
            deprecated: false,
        }
    }
}

/// Convert a snake_case name to lowerCamelCase (protobuf JSON name convention).
fn to_camel_case(s: &str) -> std::string::String {
    let mut result = std::string::String::with_capacity(s.len());
    let mut capitalize_next = false;
    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// ProtoEnumValue / ProtoEnum
// ---------------------------------------------------------------------------

/// A single value within a protobuf enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoEnumValue {
    pub name: std::string::String,
    pub number: i32,
    pub options: IndexMap<std::string::String, std::string::String>,
}

/// A protobuf enum definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoEnum {
    pub name: std::string::String,
    pub values: Vec<ProtoEnumValue>,
    pub options: IndexMap<std::string::String, std::string::String>,
    pub allow_alias: bool,
    pub reserved_numbers: Vec<ReservedRange>,
    pub reserved_names: Vec<std::string::String>,
}

impl ProtoEnum {
    pub fn value_by_name(&self, name: &str) -> Option<&ProtoEnumValue> {
        self.values.iter().find(|v| v.name == name)
    }

    pub fn value_by_number(&self, number: i32) -> Option<&ProtoEnumValue> {
        self.values.iter().find(|v| v.number == number)
    }
}

// ---------------------------------------------------------------------------
// ProtoOneof
// ---------------------------------------------------------------------------

/// A protobuf `oneof` group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoOneof {
    pub name: std::string::String,
    pub fields: Vec<ProtoField>,
}

// ---------------------------------------------------------------------------
// ProtoMessage
// ---------------------------------------------------------------------------

/// A protobuf message definition, potentially containing nested types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoMessage {
    pub name: std::string::String,
    pub fields: Vec<ProtoField>,
    pub nested_messages: Vec<ProtoMessage>,
    pub nested_enums: Vec<ProtoEnum>,
    pub oneofs: Vec<ProtoOneof>,
    pub reserved_fields: Vec<ReservedRange>,
    pub reserved_names: Vec<std::string::String>,
    pub options: IndexMap<std::string::String, std::string::String>,
}

impl ProtoMessage {
    pub fn field_by_number(&self, number: u32) -> Option<&ProtoField> {
        self.fields.iter().find(|f| f.number == number)
    }

    pub fn field_by_name(&self, name: &str) -> Option<&ProtoField> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn all_field_numbers(&self) -> Vec<u32> {
        self.fields.iter().map(|f| f.number).collect()
    }

    pub fn is_field_reserved(&self, number: u32) -> bool {
        self.reserved_fields.iter().any(|r| r.contains(number))
    }

    pub fn is_name_reserved(&self, name: &str) -> bool {
        self.reserved_names.iter().any(|n| n == name)
    }

    /// Collect all field numbers from the message and its nested messages.
    pub fn all_field_numbers_recursive(&self) -> Vec<u32> {
        let mut nums = self.all_field_numbers();
        for nested in &self.nested_messages {
            nums.extend(nested.all_field_numbers_recursive());
        }
        nums
    }
}

// ---------------------------------------------------------------------------
// ProtoMethod / ProtoService
// ---------------------------------------------------------------------------

/// A single RPC method within a protobuf service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoMethod {
    pub name: std::string::String,
    pub input_type: std::string::String,
    pub output_type: std::string::String,
    pub client_streaming: bool,
    pub server_streaming: bool,
    pub options: IndexMap<std::string::String, std::string::String>,
}

/// A protobuf service definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoService {
    pub name: std::string::String,
    pub methods: Vec<ProtoMethod>,
    pub options: IndexMap<std::string::String, std::string::String>,
}

impl ProtoService {
    pub fn method_by_name(&self, name: &str) -> Option<&ProtoMethod> {
        self.methods.iter().find(|m| m.name == name)
    }
}

// ---------------------------------------------------------------------------
// ProtoExtension
// ---------------------------------------------------------------------------

/// A protobuf extension block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtoExtension {
    pub extendee: std::string::String,
    pub fields: Vec<ProtoField>,
}

// ---------------------------------------------------------------------------
// ProtobufSchema
// ---------------------------------------------------------------------------

/// Top-level representation of a parsed `.proto` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtobufSchema {
    pub syntax: std::string::String,
    pub package: Option<std::string::String>,
    pub imports: Vec<std::string::String>,
    pub options: IndexMap<std::string::String, std::string::String>,
    pub messages: Vec<ProtoMessage>,
    pub enums: Vec<ProtoEnum>,
    pub services: Vec<ProtoService>,
    pub extensions: Vec<ProtoExtension>,
}

impl ProtobufSchema {
    /// Parse from JSON-encoded descriptor bytes.
    pub fn parse(descriptor_bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(descriptor_bytes).map_err(|e| {
            SafeStepError::encoding(
                format!("Failed to parse protobuf JSON descriptor: {e}"),
                "json",
            )
        })
    }

    /// Parse from `.proto` file text.
    pub fn parse_proto(proto_text: &str) -> Result<Self> {
        let mut parser = ProtobufParser::new();
        parser.parse(proto_text)
    }

    pub fn find_message(&self, name: &str) -> Option<&ProtoMessage> {
        find_message_recursive(&self.messages, name)
    }

    pub fn find_enum(&self, name: &str) -> Option<&ProtoEnum> {
        find_enum_recursive(&self.messages, &self.enums, name)
    }

    pub fn find_service(&self, name: &str) -> Option<&ProtoService> {
        self.services.iter().find(|s| s.name == name)
    }

    pub fn all_message_names(&self) -> Vec<std::string::String> {
        let mut names = Vec::new();
        collect_message_names(&self.messages, "", &mut names);
        names
    }

    pub fn all_field_numbers(&self, message_name: &str) -> Option<Vec<u32>> {
        self.find_message(message_name)
            .map(|m| m.all_field_numbers())
    }
}

fn find_message_recursive<'a>(
    messages: &'a [ProtoMessage],
    name: &str,
) -> Option<&'a ProtoMessage> {
    for msg in messages {
        if msg.name == name {
            return Some(msg);
        }
        if let Some(found) = find_message_recursive(&msg.nested_messages, name) {
            return Some(found);
        }
    }
    None
}

fn find_enum_recursive<'a>(
    messages: &'a [ProtoMessage],
    enums: &'a [ProtoEnum],
    name: &str,
) -> Option<&'a ProtoEnum> {
    for e in enums {
        if e.name == name {
            return Some(e);
        }
    }
    for msg in messages {
        if let Some(found) =
            find_enum_recursive(&msg.nested_messages, &msg.nested_enums, name)
        {
            return Some(found);
        }
    }
    None
}

fn collect_message_names(
    messages: &[ProtoMessage],
    prefix: &str,
    out: &mut Vec<std::string::String>,
) {
    for msg in messages {
        let fq = if prefix.is_empty() {
            msg.name.clone()
        } else {
            format!("{prefix}.{}", msg.name)
        };
        out.push(fq.clone());
        collect_message_names(&msg.nested_messages, &fq, out);
    }
}

// ---------------------------------------------------------------------------
// ProtobufParser
// ---------------------------------------------------------------------------

/// A parser that converts `.proto` text into a [`ProtobufSchema`].
pub struct ProtobufParser {
    syntax: std::string::String,
    package: Option<std::string::String>,
}

impl ProtobufParser {
    pub fn new() -> Self {
        Self {
            syntax: "proto3".to_string(),
            package: None,
        }
    }

    pub fn parse(&mut self, input: &str) -> Result<ProtobufSchema> {
        let cleaned = strip_comments(input);
        let tokens = tokenize(&cleaned)?;
        let mut pos = 0;
        let mut imports = Vec::new();
        let mut options = IndexMap::new();
        let mut messages = Vec::new();
        let mut enums = Vec::new();
        let mut services = Vec::new();
        let mut extensions = Vec::new();

        while pos < tokens.len() {
            match tokens[pos].as_str() {
                "syntax" => {
                    pos += 1;
                    expect_token(&tokens, &mut pos, "=")?;
                    let val = expect_string_literal(&tokens, &mut pos)?;
                    self.syntax = val;
                    expect_token(&tokens, &mut pos, ";")?;
                }
                "package" => {
                    pos += 1;
                    let mut pkg = std::string::String::new();
                    while pos < tokens.len() && tokens[pos] != ";" {
                        pkg.push_str(&tokens[pos]);
                        pos += 1;
                    }
                    self.package = Some(pkg);
                    expect_token(&tokens, &mut pos, ";")?;
                }
                "import" => {
                    pos += 1;
                    // skip "public" / "weak" modifiers
                    if pos < tokens.len()
                        && (tokens[pos] == "public" || tokens[pos] == "weak")
                    {
                        pos += 1;
                    }
                    let path = expect_string_literal(&tokens, &mut pos)?;
                    imports.push(path);
                    expect_token(&tokens, &mut pos, ";")?;
                }
                "option" => {
                    pos += 1;
                    let (key, val) = parse_option_kv(&tokens, &mut pos)?;
                    options.insert(key, val);
                    expect_token(&tokens, &mut pos, ";")?;
                }
                "message" => {
                    pos += 1;
                    let msg = parse_message(&tokens, &mut pos, &self.syntax)?;
                    messages.push(msg);
                }
                "enum" => {
                    pos += 1;
                    let en = parse_enum(&tokens, &mut pos)?;
                    enums.push(en);
                }
                "service" => {
                    pos += 1;
                    let svc = parse_service(&tokens, &mut pos)?;
                    services.push(svc);
                }
                "extend" => {
                    pos += 1;
                    let ext = parse_extension(&tokens, &mut pos, &self.syntax)?;
                    extensions.push(ext);
                }
                _ => {
                    // Skip unrecognised top-level tokens (e.g. edition)
                    pos += 1;
                }
            }
        }

        Ok(ProtobufSchema {
            syntax: self.syntax.clone(),
            package: self.package.clone(),
            imports,
            options,
            messages,
            enums,
            services,
            extensions,
        })
    }
}

impl Default for ProtobufParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Comment stripping
// ---------------------------------------------------------------------------

fn strip_comments(input: &str) -> std::string::String {
    let mut out = std::string::String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if i + 1 < len && chars[i] == '/' && chars[i + 1] == '/' {
            // single-line comment: skip to EOL
            while i < len && chars[i] != '\n' {
                i += 1;
            }
        } else if i + 1 < len && chars[i] == '/' && chars[i + 1] == '*' {
            // block comment
            i += 2;
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            if i + 1 < len {
                i += 2; // skip */
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

fn tokenize(input: &str) -> Result<Vec<std::string::String>> {
    let re = Regex::new(
        r#"(?x)
          "(?:[^"\\]|\\.)*"       # double-quoted string
        | '(?:[^'\\]|\\.)*'       # single-quoted string
        | [a-zA-Z_][\w.]*         # identifiers (including dotted)
        | \d+                     # integer literals
        | [{}()=;,<>]             # punctuation
        "#,
    )
    .map_err(|e| SafeStepError::schema(format!("Tokenizer regex error: {e}")))?;

    let tokens: Vec<std::string::String> =
        re.find_iter(input).map(|m| m.as_str().to_string()).collect();
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

fn expect_token(
    tokens: &[std::string::String],
    pos: &mut usize,
    expected: &str,
) -> Result<()> {
    if *pos >= tokens.len() {
        return Err(SafeStepError::schema(format!(
            "Unexpected end of input, expected '{expected}'"
        )));
    }
    if tokens[*pos] != expected {
        return Err(SafeStepError::schema(format!(
            "Expected '{expected}', found '{}'",
            tokens[*pos]
        )));
    }
    *pos += 1;
    Ok(())
}

fn expect_string_literal(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<std::string::String> {
    if *pos >= tokens.len() {
        return Err(SafeStepError::schema(
            "Unexpected end of input, expected string literal".to_string(),
        ));
    }
    let tok = &tokens[*pos];
    if (tok.starts_with('"') && tok.ends_with('"'))
        || (tok.starts_with('\'') && tok.ends_with('\''))
    {
        let inner = &tok[1..tok.len() - 1];
        *pos += 1;
        Ok(inner.to_string())
    } else {
        Err(SafeStepError::schema(format!(
            "Expected string literal, found '{tok}'"
        )))
    }
}

fn expect_identifier(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<std::string::String> {
    if *pos >= tokens.len() {
        return Err(SafeStepError::schema(
            "Unexpected end of input, expected identifier".to_string(),
        ));
    }
    let tok = tokens[*pos].clone();
    *pos += 1;
    Ok(tok)
}

fn expect_number(tokens: &[std::string::String], pos: &mut usize) -> Result<u32> {
    if *pos >= tokens.len() {
        return Err(SafeStepError::schema(
            "Unexpected end of input, expected number".to_string(),
        ));
    }
    let tok = &tokens[*pos];
    let n: u32 = tok.parse().map_err(|_| {
        SafeStepError::schema(format!("Expected number, found '{tok}'"))
    })?;
    *pos += 1;
    Ok(n)
}

fn expect_signed_number(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<i32> {
    if *pos >= tokens.len() {
        return Err(SafeStepError::schema(
            "Unexpected end of input, expected number".to_string(),
        ));
    }
    let tok = &tokens[*pos];
    let n: i32 = tok.parse().map_err(|_| {
        SafeStepError::schema(format!("Expected signed number, found '{tok}'"))
    })?;
    *pos += 1;
    Ok(n)
}

fn parse_option_kv(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<(std::string::String, std::string::String)> {
    // option key may be parenthesised: (foo.bar).baz
    let mut key = std::string::String::new();
    if *pos < tokens.len() && tokens[*pos] == "(" {
        *pos += 1; // skip (
        while *pos < tokens.len() && tokens[*pos] != ")" {
            key.push_str(&tokens[*pos]);
            *pos += 1;
        }
        if *pos < tokens.len() {
            *pos += 1; // skip )
        }
        // optional trailing dot-ident
        while *pos < tokens.len()
            && tokens[*pos] != "="
            && tokens[*pos] != ";"
        {
            key.push_str(&tokens[*pos]);
            *pos += 1;
        }
    } else {
        key = expect_identifier(tokens, pos)?;
    }
    expect_token(tokens, pos, "=")?;
    let val = if *pos < tokens.len()
        && (tokens[*pos].starts_with('"') || tokens[*pos].starts_with('\''))
    {
        expect_string_literal(tokens, pos)?
    } else {
        expect_identifier(tokens, pos)?
    };
    Ok((key, val))
}

// ---------------------------------------------------------------------------
// Message parsing
// ---------------------------------------------------------------------------

fn parse_message(
    tokens: &[std::string::String],
    pos: &mut usize,
    syntax: &str,
) -> Result<ProtoMessage> {
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "{")?;

    let mut fields = Vec::new();
    let mut nested_messages = Vec::new();
    let mut nested_enums = Vec::new();
    let mut oneofs: Vec<ProtoOneof> = Vec::new();
    let mut reserved_fields = Vec::new();
    let mut reserved_names = Vec::new();
    let mut options = IndexMap::new();

    while *pos < tokens.len() && tokens[*pos] != "}" {
        match tokens[*pos].as_str() {
            "message" => {
                *pos += 1;
                let nested = parse_message(tokens, pos, syntax)?;
                nested_messages.push(nested);
            }
            "enum" => {
                *pos += 1;
                let en = parse_enum(tokens, pos)?;
                nested_enums.push(en);
            }
            "oneof" => {
                *pos += 1;
                let oneof_idx = oneofs.len();
                let oneof = parse_oneof(tokens, pos, oneof_idx, syntax)?;
                for f in &oneof.fields {
                    fields.push(f.clone());
                }
                oneofs.push(oneof);
            }
            "reserved" => {
                *pos += 1;
                parse_reserved(tokens, pos, &mut reserved_fields, &mut reserved_names)?;
            }
            "option" => {
                *pos += 1;
                let (k, v) = parse_option_kv(tokens, pos)?;
                options.insert(k, v);
                expect_token(tokens, pos, ";")?;
            }
            "extensions" => {
                // skip extensions range declaration
                *pos += 1;
                while *pos < tokens.len() && tokens[*pos] != ";" {
                    *pos += 1;
                }
                if *pos < tokens.len() {
                    *pos += 1; // ;
                }
            }
            "map" => {
                let field = parse_map_field(tokens, pos)?;
                fields.push(field);
            }
            _ => {
                let field = parse_field(tokens, pos, syntax)?;
                fields.push(field);
            }
        }
    }

    expect_token(tokens, pos, "}")?;

    Ok(ProtoMessage {
        name,
        fields,
        nested_messages,
        nested_enums,
        oneofs,
        reserved_fields,
        reserved_names,
        options,
    })
}

// ---------------------------------------------------------------------------
// Field parsing
// ---------------------------------------------------------------------------

fn parse_field(
    tokens: &[std::string::String],
    pos: &mut usize,
    syntax: &str,
) -> Result<ProtoField> {
    let mut label = if syntax == "proto3" {
        FieldLabel::Optional
    } else {
        FieldLabel::Required
    };

    let first = &tokens[*pos];
    match first.as_str() {
        "optional" => {
            label = FieldLabel::Optional;
            *pos += 1;
        }
        "required" => {
            label = FieldLabel::Required;
            *pos += 1;
        }
        "repeated" => {
            label = FieldLabel::Repeated;
            *pos += 1;
        }
        _ => {}
    }

    let type_name = expect_identifier(tokens, pos)?;
    let type_ = ProtoFieldType::from_name(&type_name);
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "=")?;
    let number = expect_number(tokens, pos)?;

    let mut field_options = IndexMap::new();
    let mut deprecated = false;
    let mut default_value = None;
    let mut json_name = None;

    // inline options: [deprecated = true, json_name = "foo"]
    if *pos < tokens.len() && tokens[*pos] == "[" {
        *pos += 1;
        while *pos < tokens.len() && tokens[*pos] != "]" {
            if tokens[*pos] == "," {
                *pos += 1;
                continue;
            }
            let (k, v) = parse_option_kv(tokens, pos)?;
            match k.as_str() {
                "deprecated" => deprecated = v == "true",
                "default" => default_value = Some(v.clone()),
                "json_name" => json_name = Some(v.clone()),
                _ => {}
            }
            field_options.insert(k, v);
        }
        if *pos < tokens.len() {
            *pos += 1; // skip ]
        }
    }

    expect_token(tokens, pos, ";")?;

    if json_name.is_none() {
        json_name = Some(to_camel_case(&name));
    }

    Ok(ProtoField {
        name,
        number,
        type_,
        label,
        default_value,
        options: field_options,
        oneof_index: None,
        json_name,
        deprecated,
    })
}

// ---------------------------------------------------------------------------
// Map field parsing: map<KeyType, ValueType> name = number;
// ---------------------------------------------------------------------------

fn parse_map_field(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<ProtoField> {
    expect_token(tokens, pos, "map")?;
    expect_token(tokens, pos, "<")?;
    let key_name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, ",")?;
    let value_name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, ">")?;
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "=")?;
    let number = expect_number(tokens, pos)?;

    let mut field_options = IndexMap::new();
    let mut deprecated = false;

    if *pos < tokens.len() && tokens[*pos] == "[" {
        *pos += 1;
        while *pos < tokens.len() && tokens[*pos] != "]" {
            if tokens[*pos] == "," {
                *pos += 1;
                continue;
            }
            let (k, v) = parse_option_kv(tokens, pos)?;
            if k == "deprecated" {
                deprecated = v == "true";
            }
            field_options.insert(k, v);
        }
        if *pos < tokens.len() {
            *pos += 1; // ]
        }
    }

    expect_token(tokens, pos, ";")?;

    let key_type = ProtoFieldType::from_name(&key_name);
    let value_type = ProtoFieldType::from_name(&value_name);

    Ok(ProtoField {
        name: name.clone(),
        number,
        type_: ProtoFieldType::Map {
            key: Box::new(key_type),
            value: Box::new(value_type),
        },
        label: FieldLabel::Repeated,
        default_value: None,
        options: field_options,
        oneof_index: None,
        json_name: Some(to_camel_case(&name)),
        deprecated,
    })
}

// ---------------------------------------------------------------------------
// Oneof parsing
// ---------------------------------------------------------------------------

fn parse_oneof(
    tokens: &[std::string::String],
    pos: &mut usize,
    oneof_index: usize,
    syntax: &str,
) -> Result<ProtoOneof> {
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "{")?;

    let mut fields = Vec::new();
    while *pos < tokens.len() && tokens[*pos] != "}" {
        if tokens[*pos] == "option" {
            // oneof-level option; skip
            *pos += 1;
            while *pos < tokens.len() && tokens[*pos] != ";" {
                *pos += 1;
            }
            if *pos < tokens.len() {
                *pos += 1;
            }
            continue;
        }
        let mut field = parse_field(tokens, pos, syntax)?;
        field.oneof_index = Some(oneof_index);
        fields.push(field);
    }

    expect_token(tokens, pos, "}")?;

    Ok(ProtoOneof { name, fields })
}

// ---------------------------------------------------------------------------
// Reserved parsing
// ---------------------------------------------------------------------------

fn parse_reserved(
    tokens: &[std::string::String],
    pos: &mut usize,
    ranges: &mut Vec<ReservedRange>,
    names: &mut Vec<std::string::String>,
) -> Result<()> {
    // reserved can be: numbers/ranges or string names
    // e.g. reserved 1, 2, 5 to 10;  or  reserved "foo", "bar";
    if *pos < tokens.len()
        && (tokens[*pos].starts_with('"') || tokens[*pos].starts_with('\''))
    {
        // reserved names
        while *pos < tokens.len() && tokens[*pos] != ";" {
            if tokens[*pos] == "," {
                *pos += 1;
                continue;
            }
            let name = expect_string_literal(tokens, pos)?;
            names.push(name);
        }
    } else {
        // reserved numbers / ranges
        while *pos < tokens.len() && tokens[*pos] != ";" {
            if tokens[*pos] == "," {
                *pos += 1;
                continue;
            }
            let start = expect_number(tokens, pos)?;
            if *pos < tokens.len() && tokens[*pos] == "to" {
                *pos += 1; // skip "to"
                let end_tok = &tokens[*pos];
                let end = if end_tok == "max" {
                    *pos += 1;
                    536_870_911 // 2^29 - 1, max field number
                } else {
                    expect_number(tokens, pos)?
                };
                ranges.push(ReservedRange { start, end });
            } else {
                ranges.push(ReservedRange { start, end: start });
            }
        }
    }
    expect_token(tokens, pos, ";")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Enum parsing
// ---------------------------------------------------------------------------

fn parse_enum(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<ProtoEnum> {
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "{")?;

    let mut values = Vec::new();
    let mut options = IndexMap::new();
    let mut allow_alias = false;
    let mut reserved_numbers = Vec::new();
    let mut reserved_names = Vec::new();

    while *pos < tokens.len() && tokens[*pos] != "}" {
        match tokens[*pos].as_str() {
            "option" => {
                *pos += 1;
                let (k, v) = parse_option_kv(tokens, pos)?;
                if k == "allow_alias" && v == "true" {
                    allow_alias = true;
                }
                options.insert(k, v);
                expect_token(tokens, pos, ";")?;
            }
            "reserved" => {
                *pos += 1;
                parse_reserved(
                    tokens,
                    pos,
                    &mut reserved_numbers,
                    &mut reserved_names,
                )?;
            }
            _ => {
                let val_name = expect_identifier(tokens, pos)?;
                expect_token(tokens, pos, "=")?;
                let number = expect_signed_number(tokens, pos)?;

                let mut val_options = IndexMap::new();
                if *pos < tokens.len() && tokens[*pos] == "[" {
                    *pos += 1;
                    while *pos < tokens.len() && tokens[*pos] != "]" {
                        if tokens[*pos] == "," {
                            *pos += 1;
                            continue;
                        }
                        let (k, v) = parse_option_kv(tokens, pos)?;
                        val_options.insert(k, v);
                    }
                    if *pos < tokens.len() {
                        *pos += 1; // ]
                    }
                }

                expect_token(tokens, pos, ";")?;

                values.push(ProtoEnumValue {
                    name: val_name,
                    number,
                    options: val_options,
                });
            }
        }
    }

    expect_token(tokens, pos, "}")?;

    Ok(ProtoEnum {
        name,
        values,
        options,
        allow_alias,
        reserved_numbers,
        reserved_names,
    })
}

// ---------------------------------------------------------------------------
// Service parsing
// ---------------------------------------------------------------------------

fn parse_service(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<ProtoService> {
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "{")?;

    let mut methods = Vec::new();
    let mut options = IndexMap::new();

    while *pos < tokens.len() && tokens[*pos] != "}" {
        match tokens[*pos].as_str() {
            "rpc" => {
                *pos += 1;
                let method = parse_rpc_method(tokens, pos)?;
                methods.push(method);
            }
            "option" => {
                *pos += 1;
                let (k, v) = parse_option_kv(tokens, pos)?;
                options.insert(k, v);
                expect_token(tokens, pos, ";")?;
            }
            _ => {
                *pos += 1;
            }
        }
    }

    expect_token(tokens, pos, "}")?;

    Ok(ProtoService {
        name,
        methods,
        options,
    })
}

fn parse_rpc_method(
    tokens: &[std::string::String],
    pos: &mut usize,
) -> Result<ProtoMethod> {
    let name = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "(")?;

    let mut client_streaming = false;
    if *pos < tokens.len() && tokens[*pos] == "stream" {
        client_streaming = true;
        *pos += 1;
    }
    let input_type = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, ")")?;

    // "returns"
    expect_token(tokens, pos, "returns")?;
    expect_token(tokens, pos, "(")?;

    let mut server_streaming = false;
    if *pos < tokens.len() && tokens[*pos] == "stream" {
        server_streaming = true;
        *pos += 1;
    }
    let output_type = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, ")")?;

    let mut method_options = IndexMap::new();

    // method body or semicolon
    if *pos < tokens.len() && tokens[*pos] == "{" {
        *pos += 1;
        while *pos < tokens.len() && tokens[*pos] != "}" {
            if tokens[*pos] == "option" {
                *pos += 1;
                let (k, v) = parse_option_kv(tokens, pos)?;
                method_options.insert(k, v);
                expect_token(tokens, pos, ";")?;
            } else {
                *pos += 1;
            }
        }
        expect_token(tokens, pos, "}")?;
    } else {
        expect_token(tokens, pos, ";")?;
    }

    Ok(ProtoMethod {
        name,
        input_type,
        output_type,
        client_streaming,
        server_streaming,
        options: method_options,
    })
}

// ---------------------------------------------------------------------------
// Extension parsing
// ---------------------------------------------------------------------------

fn parse_extension(
    tokens: &[std::string::String],
    pos: &mut usize,
    syntax: &str,
) -> Result<ProtoExtension> {
    let extendee = expect_identifier(tokens, pos)?;
    expect_token(tokens, pos, "{")?;

    let mut fields = Vec::new();
    while *pos < tokens.len() && tokens[*pos] != "}" {
        let field = parse_field(tokens, pos, syntax)?;
        fields.push(field);
    }

    expect_token(tokens, pos, "}")?;

    Ok(ProtoExtension { extendee, fields })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_PROTO3: &str = r#"
syntax = "proto3";

package example.v1;

import "google/protobuf/timestamp.proto";

option java_package = "com.example.v1";

enum Status {
    UNKNOWN = 0;
    ACTIVE = 1;
    INACTIVE = 2;
}

message User {
    string name = 1;
    int32 age = 2;
    Status status = 3;
    repeated string tags = 4;
}
"#;

    #[test]
    fn test_parse_basic_proto3() {
        let schema = ProtobufSchema::parse_proto(BASIC_PROTO3).unwrap();
        assert_eq!(schema.syntax, "proto3");
        assert_eq!(schema.package.as_deref(), Some("example.v1"));
        assert_eq!(schema.imports.len(), 1);
        assert_eq!(schema.imports[0], "google/protobuf/timestamp.proto");
        assert_eq!(
            schema.options.get("java_package").map(|s| s.as_str()),
            Some("com.example.v1")
        );

        // Enum
        assert_eq!(schema.enums.len(), 1);
        let status = &schema.enums[0];
        assert_eq!(status.name, "Status");
        assert_eq!(status.values.len(), 3);
        assert_eq!(status.values[0].name, "UNKNOWN");
        assert_eq!(status.values[0].number, 0);
        assert_eq!(status.values[2].name, "INACTIVE");

        // Message
        assert_eq!(schema.messages.len(), 1);
        let user = &schema.messages[0];
        assert_eq!(user.name, "User");
        assert_eq!(user.fields.len(), 4);
        assert_eq!(user.fields[0].name, "name");
        assert_eq!(user.fields[0].type_, ProtoFieldType::String);
        assert_eq!(user.fields[1].name, "age");
        assert_eq!(user.fields[1].number, 2);
        assert_eq!(user.fields[3].label, FieldLabel::Repeated);
    }

    #[test]
    fn test_parse_service_with_streaming() {
        let proto = r#"
syntax = "proto3";

service Greeter {
    rpc SayHello(HelloRequest) returns (HelloResponse);
    rpc ServerStream(Request) returns (stream Response);
    rpc ClientStream(stream Request) returns (Response);
    rpc BiDiStream(stream Request) returns (stream Response);
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert_eq!(schema.services.len(), 1);
        let svc = &schema.services[0];
        assert_eq!(svc.name, "Greeter");
        assert_eq!(svc.methods.len(), 4);

        let say_hello = &svc.methods[0];
        assert_eq!(say_hello.name, "SayHello");
        assert_eq!(say_hello.input_type, "HelloRequest");
        assert_eq!(say_hello.output_type, "HelloResponse");
        assert!(!say_hello.client_streaming);
        assert!(!say_hello.server_streaming);

        let server_stream = &svc.methods[1];
        assert!(!server_stream.client_streaming);
        assert!(server_stream.server_streaming);

        let client_stream = &svc.methods[2];
        assert!(client_stream.client_streaming);
        assert!(!client_stream.server_streaming);

        let bidi = &svc.methods[3];
        assert!(bidi.client_streaming);
        assert!(bidi.server_streaming);
    }

    #[test]
    fn test_parse_map_fields() {
        let proto = r#"
syntax = "proto3";
message Config {
    map<string, int32> settings = 1;
    map<string, string> labels = 2;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let msg = &schema.messages[0];
        assert_eq!(msg.name, "Config");
        assert_eq!(msg.fields.len(), 2);

        let settings = &msg.fields[0];
        assert_eq!(settings.name, "settings");
        match &settings.type_ {
            ProtoFieldType::Map { key, value } => {
                assert_eq!(**key, ProtoFieldType::String);
                assert_eq!(**value, ProtoFieldType::Int32);
            }
            other => panic!("Expected Map, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_oneof() {
        let proto = r#"
syntax = "proto3";
message Event {
    string id = 1;
    oneof payload {
        string text = 2;
        int32 code = 3;
        bytes raw = 4;
    }
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let msg = &schema.messages[0];
        assert_eq!(msg.name, "Event");
        // 1 regular + 3 oneof fields in the fields vec
        assert_eq!(msg.fields.len(), 4);
        assert_eq!(msg.oneofs.len(), 1);
        assert_eq!(msg.oneofs[0].name, "payload");
        assert_eq!(msg.oneofs[0].fields.len(), 3);

        // Verify oneof_index is set
        let text_field = msg.field_by_name("text").unwrap();
        assert_eq!(text_field.oneof_index, Some(0));
        let code_field = msg.field_by_name("code").unwrap();
        assert_eq!(code_field.oneof_index, Some(0));
    }

    #[test]
    fn test_parse_reserved() {
        let proto = r#"
syntax = "proto3";
message Legacy {
    reserved 1, 2, 5 to 10;
    reserved "old_name", "deprecated_field";
    string current = 11;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let msg = &schema.messages[0];

        assert_eq!(msg.reserved_fields.len(), 3);
        assert_eq!(msg.reserved_fields[0], ReservedRange { start: 1, end: 1 });
        assert_eq!(msg.reserved_fields[1], ReservedRange { start: 2, end: 2 });
        assert_eq!(
            msg.reserved_fields[2],
            ReservedRange {
                start: 5,
                end: 10
            }
        );

        assert!(msg.is_field_reserved(1));
        assert!(msg.is_field_reserved(7));
        assert!(!msg.is_field_reserved(11));

        assert_eq!(msg.reserved_names.len(), 2);
        assert!(msg.is_name_reserved("old_name"));
        assert!(msg.is_name_reserved("deprecated_field"));
        assert!(!msg.is_name_reserved("current"));
    }

    #[test]
    fn test_nested_messages() {
        let proto = r#"
syntax = "proto3";
message Outer {
    string id = 1;
    message Inner {
        int32 value = 1;
        message DeepNested {
            bool flag = 1;
        }
    }
    Inner inner = 2;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let outer = &schema.messages[0];
        assert_eq!(outer.name, "Outer");
        assert_eq!(outer.nested_messages.len(), 1);

        let inner = &outer.nested_messages[0];
        assert_eq!(inner.name, "Inner");
        assert_eq!(inner.fields.len(), 1);
        assert_eq!(inner.nested_messages.len(), 1);

        let deep = &inner.nested_messages[0];
        assert_eq!(deep.name, "DeepNested");

        // find_message should locate nested types
        assert!(schema.find_message("Inner").is_some());
        assert!(schema.find_message("DeepNested").is_some());
        assert!(schema.find_message("NonExistent").is_none());

        // all_message_names should list with prefixes
        let names = schema.all_message_names();
        assert!(names.contains(&"Outer".to_string()));
        assert!(names.contains(&"Outer.Inner".to_string()));
        assert!(names.contains(&"Outer.Inner.DeepNested".to_string()));
    }

    #[test]
    fn test_field_lookup_methods() {
        let proto = r#"
syntax = "proto3";
message Lookup {
    string alpha = 1;
    int32 beta = 2;
    bool gamma = 3;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let msg = &schema.messages[0];

        assert_eq!(msg.field_by_number(1).unwrap().name, "alpha");
        assert_eq!(msg.field_by_number(2).unwrap().name, "beta");
        assert!(msg.field_by_number(99).is_none());

        assert_eq!(msg.field_by_name("gamma").unwrap().number, 3);
        assert!(msg.field_by_name("missing").is_none());

        let nums = msg.all_field_numbers();
        assert_eq!(nums, vec![1, 2, 3]);
    }

    #[test]
    fn test_invalid_syntax_errors() {
        // missing semicolon
        let bad = r#"syntax = "proto3" message Foo {}"#;
        let result = ProtobufSchema::parse_proto(bad);
        assert!(result.is_err());

        // malformed field
        let bad2 = r#"
syntax = "proto3";
message Foo {
    string name = ;
}
"#;
        let result2 = ProtobufSchema::parse_proto(bad2);
        assert!(result2.is_err());
    }

    #[test]
    fn test_proto2_syntax() {
        let proto = r#"
syntax = "proto2";
message OldMessage {
    required string id = 1;
    optional int32 count = 2;
    repeated string names = 3;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert_eq!(schema.syntax, "proto2");
        let msg = &schema.messages[0];
        assert_eq!(msg.fields[0].label, FieldLabel::Required);
        assert_eq!(msg.fields[1].label, FieldLabel::Optional);
        assert_eq!(msg.fields[2].label, FieldLabel::Repeated);
    }

    #[test]
    fn test_wire_types() {
        assert_eq!(ProtoFieldType::Int32.wire_type(), 0);
        assert_eq!(ProtoFieldType::Int64.wire_type(), 0);
        assert_eq!(ProtoFieldType::Bool.wire_type(), 0);
        assert_eq!(ProtoFieldType::Enum("Status".into()).wire_type(), 0);
        assert_eq!(ProtoFieldType::Double.wire_type(), 1);
        assert_eq!(ProtoFieldType::Fixed64.wire_type(), 1);
        assert_eq!(ProtoFieldType::String.wire_type(), 2);
        assert_eq!(ProtoFieldType::Bytes.wire_type(), 2);
        assert_eq!(
            ProtoFieldType::Message("Foo".into()).wire_type(),
            2
        );
        assert_eq!(ProtoFieldType::Float.wire_type(), 5);
        assert_eq!(ProtoFieldType::Fixed32.wire_type(), 5);
    }

    #[test]
    fn test_display_field_types() {
        assert_eq!(format!("{}", ProtoFieldType::Double), "double");
        assert_eq!(format!("{}", ProtoFieldType::String), "string");
        assert_eq!(
            format!("{}", ProtoFieldType::Message("Foo".into())),
            "Foo"
        );
        assert_eq!(
            format!(
                "{}",
                ProtoFieldType::Map {
                    key: Box::new(ProtoFieldType::String),
                    value: Box::new(ProtoFieldType::Int32),
                }
            ),
            "map<string, int32>"
        );
    }

    #[test]
    fn test_comments_stripped() {
        let proto = r#"
// This is a line comment
syntax = "proto3";

/* Block comment
   spanning multiple lines */
message Foo {
    string bar = 1; // inline comment
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert_eq!(schema.syntax, "proto3");
        assert_eq!(schema.messages.len(), 1);
        assert_eq!(schema.messages[0].fields[0].name, "bar");
    }

    #[test]
    fn test_json_descriptor_parse() {
        let json = serde_json::json!({
            "syntax": "proto3",
            "package": "test.v1",
            "imports": [],
            "options": {},
            "messages": [{
                "name": "Ping",
                "fields": [{
                    "name": "id",
                    "number": 1,
                    "type_": "String",
                    "label": "Optional",
                    "default_value": null,
                    "options": {},
                    "oneof_index": null,
                    "json_name": "id",
                    "deprecated": false
                }],
                "nested_messages": [],
                "nested_enums": [],
                "oneofs": [],
                "reserved_fields": [],
                "reserved_names": [],
                "options": {}
            }],
            "enums": [],
            "services": [],
            "extensions": []
        });
        let bytes = serde_json::to_vec(&json).unwrap();
        let schema = ProtobufSchema::parse(&bytes).unwrap();
        assert_eq!(schema.syntax, "proto3");
        assert_eq!(schema.messages[0].name, "Ping");
    }

    #[test]
    fn test_camel_case_conversion() {
        assert_eq!(to_camel_case("hello_world"), "helloWorld");
        assert_eq!(to_camel_case("foo"), "foo");
        assert_eq!(to_camel_case("one_two_three"), "oneTwoThree");
        assert_eq!(to_camel_case("already"), "already");
    }

    #[test]
    fn test_enum_value_lookup() {
        let proto = r#"
syntax = "proto3";
enum Color {
    RED = 0;
    GREEN = 1;
    BLUE = 2;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let color = &schema.enums[0];
        assert_eq!(color.value_by_name("GREEN").unwrap().number, 1);
        assert_eq!(color.value_by_number(2).unwrap().name, "BLUE");
        assert!(color.value_by_name("YELLOW").is_none());
    }

    #[test]
    fn test_service_method_lookup() {
        let proto = r#"
syntax = "proto3";
service Auth {
    rpc Login(LoginReq) returns (LoginResp);
    rpc Logout(LogoutReq) returns (LogoutResp);
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let svc = schema.find_service("Auth").unwrap();
        assert!(svc.method_by_name("Login").is_some());
        assert!(svc.method_by_name("Register").is_none());
    }

    #[test]
    fn test_service_with_options() {
        let proto = r#"
syntax = "proto3";
service MyService {
    rpc Do(Req) returns (Resp) {
        option deprecated = true;
    }
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let method = &schema.services[0].methods[0];
        assert_eq!(method.name, "Do");
        assert_eq!(
            method.options.get("deprecated").map(|s| s.as_str()),
            Some("true")
        );
    }

    #[test]
    fn test_all_field_numbers_via_schema() {
        let proto = r#"
syntax = "proto3";
message A { string x = 1; int32 y = 2; }
message B { bool z = 5; }
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let nums = schema.all_field_numbers("A").unwrap();
        assert_eq!(nums, vec![1, 2]);
        assert!(schema.all_field_numbers("C").is_none());
    }

    #[test]
    fn test_complex_proto() {
        let proto = r#"
syntax = "proto3";

package company.api.v2;

import "google/protobuf/any.proto";
import "google/protobuf/timestamp.proto";

option go_package = "company/api/v2";

enum Priority {
    UNSET = 0;
    LOW = 1;
    MEDIUM = 2;
    HIGH = 3;
    CRITICAL = 4;
}

message Task {
    string id = 1;
    string title = 2;
    string description = 3;
    Priority priority = 4;
    repeated string assignees = 5;
    map<string, string> metadata = 6;

    oneof schedule {
        string cron_expression = 7;
        int64 run_at_epoch = 8;
    }

    reserved 20, 21, 30 to 40;
    reserved "legacy_field";

    message SubTask {
        string name = 1;
        bool done = 2;
    }

    repeated SubTask sub_tasks = 9;
}

service TaskService {
    rpc CreateTask(Task) returns (Task);
    rpc ListTasks(ListRequest) returns (stream Task);
    rpc WatchTasks(stream WatchRequest) returns (stream Task);
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert_eq!(schema.syntax, "proto3");
        assert_eq!(schema.package.as_deref(), Some("company.api.v2"));
        assert_eq!(schema.imports.len(), 2);

        let task = schema.find_message("Task").unwrap();
        assert_eq!(task.fields.len(), 9); // includes oneof fields
        assert_eq!(task.oneofs.len(), 1);
        assert_eq!(task.nested_messages.len(), 1);
        assert!(task.is_field_reserved(20));
        assert!(task.is_field_reserved(35));
        assert!(!task.is_field_reserved(9));
        assert!(task.is_name_reserved("legacy_field"));

        let svc = schema.find_service("TaskService").unwrap();
        assert_eq!(svc.methods.len(), 3);
        let watch = svc.method_by_name("WatchTasks").unwrap();
        assert!(watch.client_streaming);
        assert!(watch.server_streaming);
    }

    #[test]
    fn test_enum_with_allow_alias() {
        let proto = r#"
syntax = "proto3";
enum AliasedEnum {
    option allow_alias = true;
    DEFAULT = 0;
    STARTED = 1;
    RUNNING = 1;
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let en = &schema.enums[0];
        assert!(en.allow_alias);
        assert_eq!(en.values.len(), 3);
    }

    #[test]
    fn test_field_with_inline_options() {
        let proto = r#"
syntax = "proto3";
message Annotated {
    string name = 1 [deprecated = true, json_name = "fullName"];
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let f = &schema.messages[0].fields[0];
        assert!(f.deprecated);
        assert_eq!(f.json_name.as_deref(), Some("fullName"));
    }

    #[test]
    fn test_multiple_messages_and_enums() {
        let proto = r#"
syntax = "proto3";

enum Direction { NORTH = 0; SOUTH = 1; EAST = 2; WEST = 3; }

message Point { double x = 1; double y = 2; }
message Line { Point start = 1; Point end = 2; Direction dir = 3; }
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert_eq!(schema.messages.len(), 2);
        assert_eq!(schema.enums.len(), 1);
        let names = schema.all_message_names();
        assert!(names.contains(&"Point".to_string()));
        assert!(names.contains(&"Line".to_string()));
    }

    #[test]
    fn test_find_enum() {
        let proto = r#"
syntax = "proto3";
message Wrapper {
    enum InnerEnum {
        ZERO = 0;
        ONE = 1;
    }
}
enum TopEnum { A = 0; B = 1; }
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert!(schema.find_enum("TopEnum").is_some());
        assert!(schema.find_enum("InnerEnum").is_some());
        assert!(schema.find_enum("Missing").is_none());
    }

    #[test]
    fn test_reserved_range_contains() {
        let range = ReservedRange { start: 5, end: 10 };
        assert!(!range.contains(4));
        assert!(range.contains(5));
        assert!(range.contains(7));
        assert!(range.contains(10));
        assert!(!range.contains(11));
    }

    #[test]
    fn test_proto_field_new_defaults() {
        let f = ProtoField::new("user_name", 1, ProtoFieldType::String, FieldLabel::Optional);
        assert_eq!(f.name, "user_name");
        assert_eq!(f.number, 1);
        assert_eq!(f.json_name.as_deref(), Some("userName"));
        assert!(!f.deprecated);
        assert!(f.default_value.is_none());
        assert!(f.options.is_empty());
    }

    #[test]
    fn test_all_field_numbers_recursive() {
        let proto = r#"
syntax = "proto3";
message Root {
    string a = 1;
    message Child {
        string b = 2;
        string c = 3;
    }
}
"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        let root = &schema.messages[0];
        let nums = root.all_field_numbers_recursive();
        assert!(nums.contains(&1));
        assert!(nums.contains(&2));
        assert!(nums.contains(&3));
    }

    #[test]
    fn test_empty_proto() {
        let proto = r#"syntax = "proto3";"#;
        let schema = ProtobufSchema::parse_proto(proto).unwrap();
        assert_eq!(schema.syntax, "proto3");
        assert!(schema.messages.is_empty());
        assert!(schema.enums.is_empty());
        assert!(schema.services.is_empty());
    }

    #[test]
    fn test_field_label_display() {
        assert_eq!(format!("{}", FieldLabel::Optional), "optional");
        assert_eq!(format!("{}", FieldLabel::Required), "required");
        assert_eq!(format!("{}", FieldLabel::Repeated), "repeated");
    }
}
