//! # shared-types
//!
//! Shared type definitions, AST, IR, errors, configuration, and traits for the
//! MutSpec mutation-contract-synth project.  MutSpec finds latent bugs via
//! formally grounded mutation-specification duality for loop-free first-order
//! imperative programs over QF-LIA (quantifier-free linear integer arithmetic).

pub mod ast;
pub mod config;
pub mod contracts;
pub mod display;
pub mod errors;
pub mod formats;
pub mod formula;
pub mod ir;
pub mod operators;
pub mod source_map;
pub mod types;

// ---- Re-exports for ergonomic imports ------------------------------------

pub use ast::{ArithOp, AstVisitor, Expression, Function, LogicOp, Program, RelOp, Statement};
pub use config::MutSpecConfig;
pub use contracts::{
    Contract, ContractClause, ContractProvenance, ContractStrength, Specification, SynthesisTier,
};
pub use display::{IndentWriter, TableFormatter};
pub use errors::{ErrorContext, MutSpecError, SourceLocation, SpanInfo};
pub use formats::{formatter_for, ContractFormat, ContractFormatter, FormatOptions};
pub use formula::{Formula, Predicate, Relation, Term};
pub use ir::{BasicBlock, IrExpr, IrFunction, IrProgram, IrStatement, PhiNode, SsaVar, Terminator};
pub use operators::{
    KillInfo, MutantDescriptor, MutantId, MutantStatus, MutationOperator, MutationSite,
};
pub use source_map::{MappingEntry, SourceMap, SourceRange};
pub use types::{FunctionSignature, QfLiaType, Scope, Value, Variable};

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, MutSpecError>;

// ---- Compatibility aliases for program-analysis crate --------------------
// These types map old names to current representations.

/// Block identifier (index into block list).
pub type BlockId = usize;

/// Variable identifier (string name).
pub type VarId = String;

/// Analysis error alias.
pub type AnalysisError = MutSpecError;

/// Analysis result alias.
pub type AnalysisResult<T> = std::result::Result<T, MutSpecError>;

/// Type alias mapping to QfLiaType.
pub type Type = QfLiaType;

/// Parameter: a named, typed function parameter.
pub type Parameter = Variable;

/// Unary operators for expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
    /// Logical NOT.
    Not,
    /// Bitwise NOT (treated as logical NOT in QF-LIA).
    BitwiseNot,
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::BitwiseNot => write!(f, "~"),
        }
    }
}

/// A mutation descriptor for error predicate computation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Mutation {
    /// Unique identifier.
    pub id: MutantId,
    /// The mutation operator applied.
    pub operator: MutationOperator,
    /// Target function name.
    pub function_name: String,
    /// Location in the source.
    pub site: MutationSite,
    /// Original expression text.
    pub original: String,
    /// Replacement expression text.
    pub replacement: String,
}

/// Symbol table for type checking.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    scopes: Vec<std::collections::HashMap<String, QfLiaType>>,
}

impl SymbolTable {
    /// Create a new empty symbol table with one scope.
    pub fn new() -> Self {
        Self {
            scopes: vec![std::collections::HashMap::new()],
        }
    }

    /// Push a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a variable in the current scope.
    pub fn define(&mut self, name: impl Into<String>, ty: QfLiaType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), ty);
        }
    }

    /// Look up a variable in all scopes (innermost first).
    pub fn lookup(&self, name: &str) -> Option<QfLiaType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(*ty);
            }
        }
        None
    }

    /// Check if a variable is defined in any scope.
    pub fn contains(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }
}
