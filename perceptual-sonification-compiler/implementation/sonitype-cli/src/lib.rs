//! # SoniType CLI
//!
//! Command-line interface and orchestration layer for the SoniType
//! perceptual sonification compiler. This crate ties together all
//! compiler phases—parsing, type checking, optimization, IR lowering,
//! code generation, and rendering—into a cohesive developer experience.

pub mod commands;
pub mod config;
pub mod diagnostics;
pub mod output;
pub mod pipeline;
pub mod progress;
pub mod repl;
