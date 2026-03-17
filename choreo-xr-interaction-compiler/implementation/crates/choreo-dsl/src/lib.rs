//! DSL parser, lexer, AST, and spatial type checker for the Choreo
//! spatial-temporal choreography language.
//!
//! This crate provides the complete front-end pipeline for the Choreo compiler:
//!
//! 1. **Lexing** (`token`, `lexer`): Tokenizes source text into a token stream.
//! 2. **Parsing** (`parser`): Recursive descent parser producing an AST.
//! 3. **AST** (`ast`): The abstract syntax tree representation.
//! 4. **Semantic analysis** (`semantic`): Name resolution, scope analysis, cycle detection.
//! 5. **Desugaring** (`desugar`): Expands syntactic sugar to core representation.
//! 6. **Type checking** (`type_checker`): Spatial type checking with LP feasibility.
//! 7. **Pretty printing** (`pretty_printer`): AST back to formatted source.

pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod type_checker;
pub mod desugar;
pub mod pretty_printer;
pub mod semantic;

pub use token::{Token, TokenKind, DurationUnit, DistanceUnit};
pub use lexer::Lexer;
pub use ast::Program;
pub use parser::Parser;
pub use type_checker::TypeChecker;
pub use desugar::Desugarer;
pub use pretty_printer::PrettyPrinter;
pub use semantic::SemanticAnalyzer;
