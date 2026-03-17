//! GraphQL schema parsing and diffing.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use safestep_types::{SafeStepError, Result};

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphqlTypeKind {
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    Scalar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlDirective {
    pub name: String,
    pub arguments: IndexMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlFieldType {
    pub name: String,
    pub is_non_null: bool,
    pub is_list: bool,
    pub list_item_non_null: bool,
}

impl GraphqlFieldType {
    pub fn display_type(&self) -> String {
        let base = if self.list_item_non_null {
            format!("{}!", self.name)
        } else {
            self.name.clone()
        };
        let wrapped = if self.is_list {
            format!("[{}]", base)
        } else {
            base
        };
        if self.is_non_null {
            format!("{}!", wrapped)
        } else {
            wrapped
        }
    }

    pub fn is_nullable(&self) -> bool {
        !self.is_non_null
    }

    pub fn is_scalar(&self) -> bool {
        matches!(
            self.name.as_str(),
            "String" | "Int" | "Float" | "Boolean" | "ID"
        )
    }

    fn parse(s: &str) -> Self {
        let s = s.trim();
        let (s, is_non_null) = if let Some(inner) = s.strip_suffix('!') {
            (inner, true)
        } else {
            (s, false)
        };
        let (inner, is_list) = if s.starts_with('[') && s.ends_with(']') {
            (&s[1..s.len() - 1], true)
        } else {
            (s, false)
        };
        let (name, list_item_non_null) = if let Some(n) = inner.strip_suffix('!') {
            (n.trim().to_string(), true)
        } else {
            (inner.trim().to_string(), false)
        };
        Self { name, is_non_null, is_list, list_item_non_null }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlArgument {
    pub name: String,
    pub type_: GraphqlFieldType,
    pub default_value: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlField {
    pub name: String,
    pub type_: GraphqlFieldType,
    pub arguments: Vec<GraphqlArgument>,
    pub directives: Vec<GraphqlDirective>,
    pub is_deprecated: bool,
    pub deprecation_reason: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlEnumValue {
    pub name: String,
    pub directives: Vec<GraphqlDirective>,
    pub is_deprecated: bool,
    pub deprecation_reason: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlType {
    pub name: String,
    pub kind: GraphqlTypeKind,
    pub fields: Vec<GraphqlField>,
    pub implements: Vec<String>,
    pub members: Vec<String>,
    pub enum_values: Vec<GraphqlEnumValue>,
    pub directives: Vec<GraphqlDirective>,
    pub description: Option<String>,
}

impl GraphqlType {
    pub fn field_by_name(&self, name: &str) -> Option<&GraphqlField> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    pub fn required_fields(&self) -> Vec<&GraphqlField> {
        self.fields.iter().filter(|f| f.type_.is_non_null).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlDirectiveDefinition {
    pub name: String,
    pub arguments: Vec<GraphqlArgument>,
    pub locations: Vec<String>,
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlSchema {
    pub types: IndexMap<String, GraphqlType>,
    pub query_type: Option<String>,
    pub mutation_type: Option<String>,
    pub subscription_type: Option<String>,
    pub directives: Vec<GraphqlDirectiveDefinition>,
}

impl GraphqlSchema {
    pub fn parse(sdl: &str) -> Result<Self> {
        GraphqlParser.parse(sdl)
    }

    pub fn find_type(&self, name: &str) -> Option<&GraphqlType> {
        self.types.get(name)
    }

    pub fn query_fields(&self) -> Vec<&GraphqlField> {
        self.query_type.as_ref()
            .and_then(|n| self.types.get(n))
            .map(|t| t.fields.iter().collect())
            .unwrap_or_default()
    }

    pub fn mutation_fields(&self) -> Vec<&GraphqlField> {
        self.mutation_type.as_ref()
            .and_then(|n| self.types.get(n))
            .map(|t| t.fields.iter().collect())
            .unwrap_or_default()
    }

    pub fn subscription_fields(&self) -> Vec<&GraphqlField> {
        self.subscription_type.as_ref()
            .and_then(|n| self.types.get(n))
            .map(|t| t.fields.iter().collect())
            .unwrap_or_default()
    }

    pub fn all_object_types(&self) -> Vec<&GraphqlType> {
        self.types.values().filter(|t| t.kind == GraphqlTypeKind::Object).collect()
    }

    pub fn all_input_types(&self) -> Vec<&GraphqlType> {
        self.types.values().filter(|t| t.kind == GraphqlTypeKind::InputObject).collect()
    }

    pub fn all_enum_types(&self) -> Vec<&GraphqlType> {
        self.types.values().filter(|t| t.kind == GraphqlTypeKind::Enum).collect()
    }

    pub fn all_interfaces(&self) -> Vec<&GraphqlType> {
        self.types.values().filter(|t| t.kind == GraphqlTypeKind::Interface).collect()
    }

    pub fn implements_interface(&self, type_name: &str, interface_name: &str) -> bool {
        self.types.get(type_name).map_or(false, |t| t.implements.iter().any(|i| i == interface_name))
    }

    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    pub fn field_count(&self) -> usize {
        self.types.values().map(|t| t.fields.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct GraphqlParser;

impl GraphqlParser {
    pub fn parse(&self, sdl: &str) -> Result<GraphqlSchema> {
        let cleaned = strip_comments(sdl);
        let mut types = IndexMap::new();
        let mut query_type: Option<String> = None;
        let mut mutation_type: Option<String> = None;
        let mut subscription_type: Option<String> = None;
        let mut directive_defs = Vec::new();

        let mut pos = 0;
        let chars: Vec<char> = cleaned.chars().collect();
        let len = chars.len();

        while pos < len {
            skip_ws(&chars, &mut pos);
            if pos >= len { break; }

            // Collect optional description (string literal)
            let description = try_parse_description(&chars, &mut pos);
            skip_ws(&chars, &mut pos);
            if pos >= len { break; }

            let keyword = read_word(&chars, &mut pos);
            skip_ws(&chars, &mut pos);

            match keyword.as_str() {
                "type" => {
                    let t = parse_object_type(&chars, &mut pos, GraphqlTypeKind::Object, description)?;
                    types.insert(t.name.clone(), t);
                }
                "interface" => {
                    let t = parse_object_type(&chars, &mut pos, GraphqlTypeKind::Interface, description)?;
                    types.insert(t.name.clone(), t);
                }
                "input" => {
                    let t = parse_object_type(&chars, &mut pos, GraphqlTypeKind::InputObject, description)?;
                    types.insert(t.name.clone(), t);
                }
                "union" => {
                    let t = parse_union(&chars, &mut pos, description)?;
                    types.insert(t.name.clone(), t);
                }
                "enum" => {
                    let t = parse_enum(&chars, &mut pos, description)?;
                    types.insert(t.name.clone(), t);
                }
                "scalar" => {
                    let name = read_word(&chars, &mut pos);
                    let directives = parse_directives_inline(&chars, &mut pos);
                    types.insert(name.clone(), GraphqlType {
                        name, kind: GraphqlTypeKind::Scalar, fields: vec![], implements: vec![],
                        members: vec![], enum_values: vec![], directives, description,
                    });
                }
                "schema" => {
                    parse_schema_def(&chars, &mut pos, &mut query_type, &mut mutation_type, &mut subscription_type);
                }
                "extend" => {
                    skip_ws(&chars, &mut pos);
                    let sub = read_word(&chars, &mut pos);
                    skip_ws(&chars, &mut pos);
                    if sub == "type" || sub == "interface" || sub == "input" {
                        let kind = match sub.as_str() {
                            "interface" => GraphqlTypeKind::Interface,
                            "input" => GraphqlTypeKind::InputObject,
                            _ => GraphqlTypeKind::Object,
                        };
                        let ext = parse_object_type(&chars, &mut pos, kind, None)?;
                        if let Some(existing) = types.get_mut(&ext.name) {
                            existing.fields.extend(ext.fields);
                            existing.implements.extend(ext.implements);
                            existing.directives.extend(ext.directives);
                        } else {
                            types.insert(ext.name.clone(), ext);
                        }
                    } else {
                        skip_block(&chars, &mut pos);
                    }
                }
                "directive" => {
                    let def = parse_directive_def(&chars, &mut pos, description);
                    directive_defs.push(def);
                }
                _ => {
                    // skip unknown tokens
                }
            }
        }

        // Infer schema types from well-known names
        if query_type.is_none() && types.contains_key("Query") {
            query_type = Some("Query".to_string());
        }
        if mutation_type.is_none() && types.contains_key("Mutation") {
            mutation_type = Some("Mutation".to_string());
        }
        if subscription_type.is_none() && types.contains_key("Subscription") {
            subscription_type = Some("Subscription".to_string());
        }

        Ok(GraphqlSchema { types, query_type, mutation_type, subscription_type, directives: directive_defs })
    }
}

// Helpers

fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for line in s.lines() {
        if let Some(idx) = line.find('#') {
            out.push_str(&line[..idx]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn read_word(chars: &[char], pos: &mut usize) -> String {
    skip_ws(chars, pos);
    let start = *pos;
    while *pos < chars.len() && (chars[*pos].is_alphanumeric() || chars[*pos] == '_') {
        *pos += 1;
    }
    chars[start..*pos].iter().collect()
}

fn read_until(chars: &[char], pos: &mut usize, stop: char) -> String {
    let start = *pos;
    while *pos < chars.len() && chars[*pos] != stop {
        *pos += 1;
    }
    let result: String = chars[start..*pos].iter().collect();
    if *pos < chars.len() { *pos += 1; }
    result
}

fn skip_block(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos] != '{' { *pos += 1; }
    if *pos < chars.len() { *pos += 1; }
    let mut depth = 1;
    while *pos < chars.len() && depth > 0 {
        if chars[*pos] == '{' { depth += 1; }
        if chars[*pos] == '}' { depth -= 1; }
        *pos += 1;
    }
}

fn try_parse_description(chars: &[char], pos: &mut usize) -> Option<String> {
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] == '"' {
        if *pos + 2 < chars.len() && chars[*pos + 1] == '"' && chars[*pos + 2] == '"' {
            *pos += 3;
            let start = *pos;
            while *pos + 2 < chars.len() {
                if chars[*pos] == '"' && chars[*pos + 1] == '"' && chars[*pos + 2] == '"' {
                    let desc: String = chars[start..*pos].iter().collect();
                    *pos += 3;
                    return Some(desc.trim().to_string());
                }
                *pos += 1;
            }
            let desc: String = chars[start..].iter().collect();
            *pos = chars.len();
            return Some(desc.trim().to_string());
        } else {
            *pos += 1;
            let start = *pos;
            while *pos < chars.len() && chars[*pos] != '"' { *pos += 1; }
            let desc: String = chars[start..*pos].iter().collect();
            if *pos < chars.len() { *pos += 1; }
            return Some(desc);
        }
    }
    None
}

fn parse_directives_inline(chars: &[char], pos: &mut usize) -> Vec<GraphqlDirective> {
    let mut directives = Vec::new();
    loop {
        skip_ws(chars, pos);
        if *pos >= chars.len() || chars[*pos] != '@' { break; }
        *pos += 1;
        let name = read_word(chars, pos);
        let mut arguments = IndexMap::new();
        skip_ws(chars, pos);
        if *pos < chars.len() && chars[*pos] == '(' {
            *pos += 1;
            loop {
                skip_ws(chars, pos);
                if *pos >= chars.len() || chars[*pos] == ')' { *pos += 1; break; }
                let arg_name = read_word(chars, pos);
                skip_ws(chars, pos);
                if *pos < chars.len() && chars[*pos] == ':' { *pos += 1; }
                skip_ws(chars, pos);
                let arg_val = read_arg_value(chars, pos);
                if !arg_name.is_empty() {
                    arguments.insert(arg_name, arg_val);
                }
                skip_ws(chars, pos);
                if *pos < chars.len() && chars[*pos] == ',' { *pos += 1; }
            }
        }
        directives.push(GraphqlDirective { name, arguments });
    }
    directives
}

fn read_arg_value(chars: &[char], pos: &mut usize) -> String {
    skip_ws(chars, pos);
    if *pos >= chars.len() { return String::new(); }
    if chars[*pos] == '"' {
        *pos += 1;
        let start = *pos;
        while *pos < chars.len() && chars[*pos] != '"' { *pos += 1; }
        let val: String = chars[start..*pos].iter().collect();
        if *pos < chars.len() { *pos += 1; }
        val
    } else {
        let start = *pos;
        while *pos < chars.len() && !chars[*pos].is_whitespace() && chars[*pos] != ')' && chars[*pos] != ',' {
            *pos += 1;
        }
        chars[start..*pos].iter().collect()
    }
}

fn read_type_ref(chars: &[char], pos: &mut usize) -> String {
    skip_ws(chars, pos);
    let start = *pos;
    while *pos < chars.len() && !chars[*pos].is_whitespace()
        && chars[*pos] != '@' && chars[*pos] != '{' && chars[*pos] != '}' && chars[*pos] != '(' {
        *pos += 1;
    }
    chars[start..*pos].iter().collect()
}

fn parse_object_type(
    chars: &[char],
    pos: &mut usize,
    kind: GraphqlTypeKind,
    description: Option<String>,
) -> Result<GraphqlType> {
    let name = read_word(chars, pos);
    if name.is_empty() {
        return Err(SafeStepError::schema("Expected type name"));
    }

    // implements
    let mut implements = Vec::new();
    skip_ws(chars, pos);
    let peek_start = *pos;
    let maybe_impl = read_word(chars, pos);
    if maybe_impl == "implements" {
        loop {
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == '{' || chars[*pos] == '@' { break; }
            if chars[*pos] == '&' { *pos += 1; continue; }
            let iface = read_word(chars, pos);
            if iface.is_empty() { break; }
            implements.push(iface);
        }
    } else {
        *pos = peek_start;
    }

    let directives = parse_directives_inline(chars, pos);

    // Parse body
    skip_ws(chars, pos);
    let mut fields = Vec::new();
    if *pos < chars.len() && chars[*pos] == '{' {
        *pos += 1;
        loop {
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == '}' { *pos += 1; break; }
            let desc = try_parse_description(chars, pos);
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == '}' { *pos += 1; break; }
            let fname = read_word(chars, pos);
            if fname.is_empty() { *pos += 1; continue; }

            // Arguments
            let mut arguments = Vec::new();
            skip_ws(chars, pos);
            if *pos < chars.len() && chars[*pos] == '(' {
                *pos += 1;
                loop {
                    skip_ws(chars, pos);
                    if *pos >= chars.len() || chars[*pos] == ')' { *pos += 1; break; }
                    let aname = read_word(chars, pos);
                    skip_ws(chars, pos);
                    if *pos < chars.len() && chars[*pos] == ':' { *pos += 1; }
                    let atype = read_type_ref(chars, pos);
                    skip_ws(chars, pos);
                    let default_value = if *pos < chars.len() && chars[*pos] == '=' {
                        *pos += 1;
                        Some(read_arg_value(chars, pos))
                    } else {
                        None
                    };
                    if !aname.is_empty() {
                        arguments.push(GraphqlArgument {
                            name: aname, type_: GraphqlFieldType::parse(&atype),
                            default_value, description: None,
                        });
                    }
                    skip_ws(chars, pos);
                    if *pos < chars.len() && chars[*pos] == ',' { *pos += 1; }
                }
            }

            skip_ws(chars, pos);
            if *pos < chars.len() && chars[*pos] == ':' { *pos += 1; }
            let ftype = read_type_ref(chars, pos);
            let fdirectives = parse_directives_inline(chars, pos);
            let is_deprecated = fdirectives.iter().any(|d| d.name == "deprecated");
            let deprecation_reason = fdirectives.iter()
                .find(|d| d.name == "deprecated")
                .and_then(|d| d.arguments.get("reason").cloned());

            fields.push(GraphqlField {
                name: fname, type_: GraphqlFieldType::parse(&ftype), arguments,
                directives: fdirectives, is_deprecated, deprecation_reason, description: desc,
            });
        }
    }

    Ok(GraphqlType {
        name, kind, fields, implements, members: vec![], enum_values: vec![],
        directives, description,
    })
}

fn parse_union(chars: &[char], pos: &mut usize, description: Option<String>) -> Result<GraphqlType> {
    let name = read_word(chars, pos);
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] == '=' { *pos += 1; }
    let mut members = Vec::new();
    loop {
        skip_ws(chars, pos);
        if *pos >= chars.len() { break; }
        if chars[*pos] == '|' { *pos += 1; continue; }
        let m = read_word(chars, pos);
        if m.is_empty() { break; }
        members.push(m);
        skip_ws(chars, pos);
        if *pos >= chars.len() || (chars[*pos] != '|' && !chars[*pos].is_alphanumeric()) { break; }
    }
    Ok(GraphqlType {
        name, kind: GraphqlTypeKind::Union, fields: vec![], implements: vec![],
        members, enum_values: vec![], directives: vec![], description,
    })
}

fn parse_enum(chars: &[char], pos: &mut usize, description: Option<String>) -> Result<GraphqlType> {
    let name = read_word(chars, pos);
    let directives = parse_directives_inline(chars, pos);
    skip_ws(chars, pos);
    let mut values = Vec::new();
    if *pos < chars.len() && chars[*pos] == '{' {
        *pos += 1;
        loop {
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == '}' { *pos += 1; break; }
            let desc = try_parse_description(chars, pos);
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == '}' { *pos += 1; break; }
            let vname = read_word(chars, pos);
            if vname.is_empty() { *pos += 1; continue; }
            let vdirs = parse_directives_inline(chars, pos);
            let is_dep = vdirs.iter().any(|d| d.name == "deprecated");
            let dep_reason = vdirs.iter().find(|d| d.name == "deprecated")
                .and_then(|d| d.arguments.get("reason").cloned());
            values.push(GraphqlEnumValue {
                name: vname, directives: vdirs, is_deprecated: is_dep,
                deprecation_reason: dep_reason, description: desc,
            });
        }
    }
    Ok(GraphqlType {
        name, kind: GraphqlTypeKind::Enum, fields: vec![], implements: vec![],
        members: vec![], enum_values: values, directives, description,
    })
}

fn parse_schema_def(
    chars: &[char], pos: &mut usize,
    query: &mut Option<String>, mutation: &mut Option<String>, subscription: &mut Option<String>,
) {
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] == '{' {
        *pos += 1;
        loop {
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == '}' { *pos += 1; break; }
            let key = read_word(chars, pos);
            skip_ws(chars, pos);
            if *pos < chars.len() && chars[*pos] == ':' { *pos += 1; }
            let val = read_word(chars, pos);
            match key.as_str() {
                "query" => *query = Some(val),
                "mutation" => *mutation = Some(val),
                "subscription" => *subscription = Some(val),
                _ => {}
            }
        }
    }
}

fn parse_directive_def(
    chars: &[char], pos: &mut usize, description: Option<String>,
) -> GraphqlDirectiveDefinition {
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] == '@' { *pos += 1; }
    let name = read_word(chars, pos);
    let mut arguments = Vec::new();
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] == '(' {
        *pos += 1;
        loop {
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] == ')' { *pos += 1; break; }
            let aname = read_word(chars, pos);
            skip_ws(chars, pos);
            if *pos < chars.len() && chars[*pos] == ':' { *pos += 1; }
            let atype = read_type_ref(chars, pos);
            skip_ws(chars, pos);
            let dv = if *pos < chars.len() && chars[*pos] == '=' {
                *pos += 1;
                Some(read_arg_value(chars, pos))
            } else { None };
            if !aname.is_empty() {
                arguments.push(GraphqlArgument {
                    name: aname, type_: GraphqlFieldType::parse(&atype),
                    default_value: dv, description: None,
                });
            }
            skip_ws(chars, pos);
            if *pos < chars.len() && chars[*pos] == ',' { *pos += 1; }
        }
    }
    // on LOCATIONS
    let mut locations = Vec::new();
    skip_ws(chars, pos);
    let peek = *pos;
    let w = read_word(chars, pos);
    if w == "on" {
        loop {
            skip_ws(chars, pos);
            if *pos >= chars.len() { break; }
            if chars[*pos] == '|' { *pos += 1; continue; }
            let loc = read_word(chars, pos);
            if loc.is_empty() { break; }
            locations.push(loc);
            skip_ws(chars, pos);
            if *pos >= chars.len() || chars[*pos] != '|' { break; }
        }
    } else {
        *pos = peek;
    }
    GraphqlDirectiveDefinition { name, arguments, locations, description }
}

// ---------------------------------------------------------------------------
// Diff types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlDiffResult {
    pub added_types: Vec<String>,
    pub removed_types: Vec<String>,
    pub modified_types: Vec<GraphqlTypeChange>,
    pub breaking_changes: Vec<GraphqlBreakingChange>,
    pub non_breaking_changes: Vec<GraphqlNonBreakingChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlTypeChange {
    pub type_name: String,
    pub added_fields: Vec<String>,
    pub removed_fields: Vec<String>,
    pub modified_fields: Vec<GraphqlFieldChange>,
    pub added_enum_values: Vec<String>,
    pub removed_enum_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlFieldChange {
    pub field_name: String,
    pub old_type: Option<String>,
    pub new_type: Option<String>,
    pub nullability_changed: bool,
    pub arguments_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlBreakingChange {
    pub kind: GraphqlBreakingKind,
    pub path: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphqlBreakingKind {
    RemovedType, RemovedField, ChangedFieldType, RemovedEnumValue,
    AddedRequiredArgument, ChangedArgumentType, RemovedArgument,
    ChangedInterfaceImplementation, ChangedUnionMembers, RemovedDirective,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphqlNonBreakingChange {
    pub kind: GraphqlNonBreakingKind,
    pub path: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphqlNonBreakingKind {
    AddedType, AddedField, AddedEnumValue, AddedOptionalArgument,
    DeprecatedField, AddedDirective, AddedUnionMember, AddedInterfaceImplementation,
}

// ---------------------------------------------------------------------------
// Diff implementation
// ---------------------------------------------------------------------------

pub struct GraphqlDiff;

impl GraphqlDiff {
    pub fn diff(old: &GraphqlSchema, new: &GraphqlSchema) -> GraphqlDiffResult {
        let mut added_types = Vec::new();
        let mut removed_types = Vec::new();
        let mut modified_types = Vec::new();
        let mut breaking = Vec::new();
        let mut non_breaking = Vec::new();

        // Added types
        for name in new.types.keys() {
            if !old.types.contains_key(name) {
                added_types.push(name.clone());
                non_breaking.push(GraphqlNonBreakingChange {
                    kind: GraphqlNonBreakingKind::AddedType,
                    path: name.clone(),
                    description: format!("Type '{}' was added", name),
                });
            }
        }

        // Removed types
        for name in old.types.keys() {
            if !new.types.contains_key(name) {
                removed_types.push(name.clone());
                breaking.push(GraphqlBreakingChange {
                    kind: GraphqlBreakingKind::RemovedType,
                    path: name.clone(),
                    description: format!("Type '{}' was removed", name),
                });
            }
        }

        // Modified types
        for (name, old_type) in &old.types {
            if let Some(new_type) = new.types.get(name) {
                let mut tc = GraphqlTypeChange {
                    type_name: name.clone(),
                    added_fields: vec![], removed_fields: vec![],
                    modified_fields: vec![], added_enum_values: vec![],
                    removed_enum_values: vec![],
                };
                let mut changed = false;

                // Fields
                for f in &old_type.fields {
                    if let Some(nf) = new_type.field_by_name(&f.name) {
                        let old_display = f.type_.display_type();
                        let new_display = nf.type_.display_type();
                        if old_display != new_display {
                            changed = true;
                            tc.modified_fields.push(GraphqlFieldChange {
                                field_name: f.name.clone(),
                                old_type: Some(old_display.clone()),
                                new_type: Some(new_display.clone()),
                                nullability_changed: f.type_.is_non_null != nf.type_.is_non_null,
                                arguments_changed: false,
                            });
                            breaking.push(GraphqlBreakingChange {
                                kind: GraphqlBreakingKind::ChangedFieldType,
                                path: format!("{}.{}", name, f.name),
                                description: format!("Field type changed from {} to {}", old_display, new_display),
                            });
                        }
                        // Check arguments
                        for arg in &nf.arguments {
                            let old_arg = f.arguments.iter().find(|a| a.name == arg.name);
                            if old_arg.is_none() && arg.type_.is_non_null && arg.default_value.is_none() {
                                changed = true;
                                breaking.push(GraphqlBreakingChange {
                                    kind: GraphqlBreakingKind::AddedRequiredArgument,
                                    path: format!("{}.{}({})", name, f.name, arg.name),
                                    description: format!("Required argument '{}' was added", arg.name),
                                });
                            } else if old_arg.is_none() {
                                non_breaking.push(GraphqlNonBreakingChange {
                                    kind: GraphqlNonBreakingKind::AddedOptionalArgument,
                                    path: format!("{}.{}({})", name, f.name, arg.name),
                                    description: format!("Optional argument '{}' was added", arg.name),
                                });
                            }
                        }
                        for arg in &f.arguments {
                            if !nf.arguments.iter().any(|a| a.name == arg.name) {
                                changed = true;
                                breaking.push(GraphqlBreakingChange {
                                    kind: GraphqlBreakingKind::RemovedArgument,
                                    path: format!("{}.{}({})", name, f.name, arg.name),
                                    description: format!("Argument '{}' was removed", arg.name),
                                });
                            }
                        }
                        // Deprecation
                        if !f.is_deprecated && nf.is_deprecated {
                            non_breaking.push(GraphqlNonBreakingChange {
                                kind: GraphqlNonBreakingKind::DeprecatedField,
                                path: format!("{}.{}", name, f.name),
                                description: format!("Field '{}' was deprecated", f.name),
                            });
                        }
                    } else {
                        changed = true;
                        tc.removed_fields.push(f.name.clone());
                        breaking.push(GraphqlBreakingChange {
                            kind: GraphqlBreakingKind::RemovedField,
                            path: format!("{}.{}", name, f.name),
                            description: format!("Field '{}' was removed from '{}'", f.name, name),
                        });
                    }
                }
                for f in &new_type.fields {
                    if !old_type.has_field(&f.name) {
                        changed = true;
                        tc.added_fields.push(f.name.clone());
                        non_breaking.push(GraphqlNonBreakingChange {
                            kind: GraphqlNonBreakingKind::AddedField,
                            path: format!("{}.{}", name, f.name),
                            description: format!("Field '{}' was added to '{}'", f.name, name),
                        });
                    }
                }

                // Enum values
                for ev in &old_type.enum_values {
                    if !new_type.enum_values.iter().any(|v| v.name == ev.name) {
                        changed = true;
                        tc.removed_enum_values.push(ev.name.clone());
                        breaking.push(GraphqlBreakingChange {
                            kind: GraphqlBreakingKind::RemovedEnumValue,
                            path: format!("{}.{}", name, ev.name),
                            description: format!("Enum value '{}' removed from '{}'", ev.name, name),
                        });
                    }
                }
                for ev in &new_type.enum_values {
                    if !old_type.enum_values.iter().any(|v| v.name == ev.name) {
                        changed = true;
                        tc.added_enum_values.push(ev.name.clone());
                        non_breaking.push(GraphqlNonBreakingChange {
                            kind: GraphqlNonBreakingKind::AddedEnumValue,
                            path: format!("{}.{}", name, ev.name),
                            description: format!("Enum value '{}' added to '{}'", ev.name, name),
                        });
                    }
                }

                // Union members
                for m in &new_type.members {
                    if !old_type.members.contains(m) {
                        non_breaking.push(GraphqlNonBreakingChange {
                            kind: GraphqlNonBreakingKind::AddedUnionMember,
                            path: format!("{}.{}", name, m),
                            description: format!("Union member '{}' added to '{}'", m, name),
                        });
                        changed = true;
                    }
                }
                for m in &old_type.members {
                    if !new_type.members.contains(m) {
                        breaking.push(GraphqlBreakingChange {
                            kind: GraphqlBreakingKind::ChangedUnionMembers,
                            path: format!("{}.{}", name, m),
                            description: format!("Union member '{}' removed from '{}'", m, name),
                        });
                        changed = true;
                    }
                }

                // Interface impl changes
                for i in &new_type.implements {
                    if !old_type.implements.contains(i) {
                        non_breaking.push(GraphqlNonBreakingChange {
                            kind: GraphqlNonBreakingKind::AddedInterfaceImplementation,
                            path: format!("{} implements {}", name, i),
                            description: format!("'{}' now implements '{}'", name, i),
                        });
                        changed = true;
                    }
                }
                for i in &old_type.implements {
                    if !new_type.implements.contains(i) {
                        breaking.push(GraphqlBreakingChange {
                            kind: GraphqlBreakingKind::ChangedInterfaceImplementation,
                            path: format!("{} implements {}", name, i),
                            description: format!("'{}' no longer implements '{}'", name, i),
                        });
                        changed = true;
                    }
                }

                if changed {
                    modified_types.push(tc);
                }
            }
        }

        GraphqlDiffResult { added_types, removed_types, modified_types, breaking_changes: breaking, non_breaking_changes: non_breaking }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SDL: &str = r#"
        type Query {
            users(limit: Int = 10): [User!]!
            user(id: ID!): User
        }

        type Mutation {
            createUser(input: CreateUserInput!): User!
        }

        type User implements Node {
            id: ID!
            name: String!
            email: String
            status: Status
        }

        interface Node {
            id: ID!
        }

        input CreateUserInput {
            name: String!
            email: String!
        }

        enum Status {
            ACTIVE
            INACTIVE
            PENDING
        }

        union SearchResult = User

        scalar DateTime
    "#;

    #[test]
    fn test_parse_basic() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        assert!(schema.types.contains_key("Query"));
        assert!(schema.types.contains_key("User"));
        assert!(schema.types.contains_key("Node"));
        assert!(schema.types.contains_key("CreateUserInput"));
        assert!(schema.types.contains_key("Status"));
        assert!(schema.types.contains_key("SearchResult"));
        assert!(schema.types.contains_key("DateTime"));
    }

    #[test]
    fn test_type_kinds() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        assert_eq!(schema.find_type("User").unwrap().kind, GraphqlTypeKind::Object);
        assert_eq!(schema.find_type("Node").unwrap().kind, GraphqlTypeKind::Interface);
        assert_eq!(schema.find_type("CreateUserInput").unwrap().kind, GraphqlTypeKind::InputObject);
        assert_eq!(schema.find_type("Status").unwrap().kind, GraphqlTypeKind::Enum);
        assert_eq!(schema.find_type("SearchResult").unwrap().kind, GraphqlTypeKind::Union);
        assert_eq!(schema.find_type("DateTime").unwrap().kind, GraphqlTypeKind::Scalar);
    }

    #[test]
    fn test_fields() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        let user = schema.find_type("User").unwrap();
        assert!(user.has_field("id"));
        assert!(user.has_field("name"));
        assert!(user.has_field("email"));
        let id_field = user.field_by_name("id").unwrap();
        assert!(id_field.type_.is_non_null);
    }

    #[test]
    fn test_implements() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        assert!(schema.implements_interface("User", "Node"));
    }

    #[test]
    fn test_enum_values() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        let status = schema.find_type("Status").unwrap();
        let names: Vec<_> = status.enum_values.iter().map(|v| v.name.as_str()).collect();
        assert!(names.contains(&"ACTIVE"));
        assert!(names.contains(&"INACTIVE"));
        assert!(names.contains(&"PENDING"));
    }

    #[test]
    fn test_union_members() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        let sr = schema.find_type("SearchResult").unwrap();
        assert!(sr.members.contains(&"User".to_string()));
    }

    #[test]
    fn test_query_fields() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        let fields = schema.query_fields();
        assert!(!fields.is_empty());
        assert!(fields.iter().any(|f| f.name == "users"));
    }

    #[test]
    fn test_field_arguments() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        let query = schema.find_type("Query").unwrap();
        let users = query.field_by_name("users").unwrap();
        assert!(!users.arguments.is_empty());
        assert_eq!(users.arguments[0].name, "limit");
    }

    #[test]
    fn test_field_type_display() {
        let ft = GraphqlFieldType { name: "User".into(), is_non_null: true, is_list: true, list_item_non_null: true };
        assert_eq!(ft.display_type(), "[User!]!");
    }

    #[test]
    fn test_schema_counts() {
        let schema = GraphqlSchema::parse(SDL).unwrap();
        assert!(schema.type_count() >= 7);
        assert!(schema.field_count() > 0);
    }

    #[test]
    fn test_diff_added_type() {
        let old = GraphqlSchema::parse("type Query { hello: String }").unwrap();
        let new = GraphqlSchema::parse("type Query { hello: String }\ntype User { id: ID! }").unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.added_types.contains(&"User".to_string()));
        assert!(diff.non_breaking_changes.iter().any(|c| c.kind == GraphqlNonBreakingKind::AddedType));
    }

    #[test]
    fn test_diff_removed_type() {
        let old = GraphqlSchema::parse("type Query { hello: String }\ntype User { id: ID! }").unwrap();
        let new = GraphqlSchema::parse("type Query { hello: String }").unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.removed_types.contains(&"User".to_string()));
        assert!(diff.breaking_changes.iter().any(|c| c.kind == GraphqlBreakingKind::RemovedType));
    }

    #[test]
    fn test_diff_removed_field() {
        let old = GraphqlSchema::parse("type Query { a: String\n b: Int }").unwrap();
        let new = GraphqlSchema::parse("type Query { a: String }").unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.breaking_changes.iter().any(|c| c.kind == GraphqlBreakingKind::RemovedField));
    }

    #[test]
    fn test_diff_added_field() {
        let old = GraphqlSchema::parse("type Query { a: String }").unwrap();
        let new = GraphqlSchema::parse("type Query { a: String\n b: Int }").unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.non_breaking_changes.iter().any(|c| c.kind == GraphqlNonBreakingKind::AddedField));
    }

    #[test]
    fn test_diff_removed_enum_value() {
        let old = GraphqlSchema::parse("enum Status { A\n B\n C }").unwrap();
        let new = GraphqlSchema::parse("enum Status { A\n B }").unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.breaking_changes.iter().any(|c| c.kind == GraphqlBreakingKind::RemovedEnumValue));
    }

    #[test]
    fn test_diff_added_enum_value() {
        let old = GraphqlSchema::parse("enum Status { A\n B }").unwrap();
        let new = GraphqlSchema::parse("enum Status { A\n B\n C }").unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.non_breaking_changes.iter().any(|c| c.kind == GraphqlNonBreakingKind::AddedEnumValue));
    }

    #[test]
    fn test_deprecated_field() {
        let old = GraphqlSchema::parse("type Query { old: String }").unwrap();
        let new = GraphqlSchema::parse(r#"type Query { old: String @deprecated(reason: "use new") }"#).unwrap();
        let diff = GraphqlDiff::diff(&old, &new);
        assert!(diff.non_breaking_changes.iter().any(|c| c.kind == GraphqlNonBreakingKind::DeprecatedField));
    }

    #[test]
    fn test_required_fields() {
        let schema = GraphqlSchema::parse("type User { id: ID!\n name: String!\n bio: String }").unwrap();
        let user = schema.find_type("User").unwrap();
        let req = user.required_fields();
        assert_eq!(req.len(), 2);
    }

    #[test]
    fn test_schema_def() {
        let sdl = r#"
            schema { query: RootQuery mutation: RootMutation }
            type RootQuery { hello: String }
            type RootMutation { update: Boolean }
        "#;
        let schema = GraphqlSchema::parse(sdl).unwrap();
        assert_eq!(schema.query_type.as_deref(), Some("RootQuery"));
        assert_eq!(schema.mutation_type.as_deref(), Some("RootMutation"));
    }

    #[test]
    fn test_extend_type() {
        let sdl = r#"
            type Query { hello: String }
            extend type Query { goodbye: String }
        "#;
        let schema = GraphqlSchema::parse(sdl).unwrap();
        let q = schema.find_type("Query").unwrap();
        assert!(q.has_field("hello"));
        assert!(q.has_field("goodbye"));
    }
}
