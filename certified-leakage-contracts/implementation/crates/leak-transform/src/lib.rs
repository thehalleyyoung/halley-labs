//! Program transformations and IR conversion for the Certified Leakage Contracts framework.
//!
//! This crate provides the binary analysis substrate: an intermediate representation (IR)
//! for speculative side-channel analysis on x86-64 crypto binaries, along with the lifting,
//! annotation, normalization, loop unrolling, CFG construction, and binary adaptation
//! infrastructure needed to prepare programs for leakage contract verification.

pub mod ir;
pub mod lifter;
pub mod annotator;
pub mod normalizer;
pub mod unroller;
pub mod cfg_builder;
pub mod adapter;
pub mod goblin_adapter;

pub use ir::{
    AnalysisIR, IRInstruction, IROperand, IREffect, IRBlock, IRFunction, IRProgram,
    IRBlockId, IRTerminator, CryptoKind,
};
pub use lifter::{InstructionLifter, X86Lifter, LiftResult, LiftError};
pub use annotator::{
    SecurityAnnotator, AnnotationSource, SecretRegions, TaintAnnotator,
    AnnotatedIR, AnnotationConfig,
};
pub use normalizer::{
    IRNormalizer, NormalizationPass, ConstantFolding, DeadCodeElimination,
    CopyPropagation, InstructionCanonicalization, NormalizationPipeline,
};
pub use unroller::{LoopUnroller, UnrollConfig, UnrollResult, LoopBound};
pub use cfg_builder::CfgBuilder;
pub use adapter::{
    BinaryAdapter, ElfAdapter, RawBytesAdapter, FunctionDiscovery, SymbolTable, Section,
};
