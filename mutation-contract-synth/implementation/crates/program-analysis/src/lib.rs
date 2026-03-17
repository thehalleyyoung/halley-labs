//! Program analysis crate for the MutSpec mutation-contract synthesis engine.
//!
//! This crate provides:
//! - **Parsing**: Lexer and recursive-descent parser for the loop-free imperative language
//! - **IR Lowering**: AST to IR transformation with expression flattening
//! - **Control Flow Graphs**: CFG construction, dominators, path enumeration
//! - **SSA Transformation**: Static single assignment with phi nodes
//! - **Weakest Precondition**: WP computation engine for verification conditions
//! - **Error Predicates**: Extraction of error predicates from mutations
//! - **Data Flow Analysis**: Generic data-flow framework (reaching defs, liveness, etc.)
//! - **Type Checking**: Type inference/checking and QF-LIA fragment validation

pub mod cfg;
pub mod data_flow;
pub mod error_predicates;
pub mod ir_lowering;
pub mod parser;
pub mod ssa;
pub mod type_checker;
pub mod wp_engine;

pub use cfg::ControlFlowGraph;
pub use data_flow::{DataFlowAnalysis, LiveVariables, ReachingDefinitions};
pub use error_predicates::{ErrorPredicate, ErrorPredicateExtractor};
pub use ir_lowering::IrLowering;
pub use parser::{Lexer, Parser, Token};
pub use ssa::SsaTransform;
pub use type_checker::TypeChecker;
pub use wp_engine::WpEngine;
