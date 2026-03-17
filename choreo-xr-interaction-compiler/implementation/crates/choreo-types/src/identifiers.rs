//! Identifier and symbol table types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identifier(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualifiedName {
    pub parts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub name: String,
    pub symbols: HashMap<String, Identifier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeStack {
    pub scopes: Vec<Scope>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable {
    pub scopes: ScopeStack,
}
