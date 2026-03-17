//! SoniType DSL - Parser, AST, and Perceptual Type Checker
//!
//! This crate implements the front-end of the SoniType perceptual sonification
//! compiler. It provides:
//!
//! - **Lexer** (`token`, `lexer`): Tokenization of SoniType DSL source code.
//! - **Parser** (`parser`): Recursive-descent parser producing an AST.
//! - **AST** (`ast`): Abstract Syntax Tree representation.
//! - **Type System** (`type_system`): Perceptual type checking with psychoacoustic qualifiers.
//! - **Type Inference** (`type_inference`): Hindley-Milner style inference adapted for
//!   perceptual qualifier constraints.
//! - **Semantic Analysis** (`semantic`): Name resolution, scope analysis, and validation.
//! - **Desugaring** (`desugar`): Source-level transformations before type checking.
//! - **Pretty Printing** (`pretty_print`): Format AST back to source code.

pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod type_system;
pub mod type_inference;
pub mod semantic;
pub mod desugar;
pub mod pretty_print;

pub use ast::{Program, Declaration, Expr, Type as AstType};
pub use lexer::lex;
pub use parser::parse;
pub use type_system::{TypeChecker, TypedProgram};
pub use type_inference::InferenceEngine;
pub use semantic::SemanticAnalyzer;
pub use desugar::desugar_program;
pub use pretty_print::PrettyPrinter;
