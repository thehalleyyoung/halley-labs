//! # leak-analysis
//!
//! Core analysis crate for the Certified Leakage Contracts framework.
//!
//! Implements the three abstract domains (D\_spec, D\_cache, D\_quant) and
//! the reduction operator ρ that composes them into a reduced product for
//! precise speculative side-channel leakage analysis on x86-64 binaries.
//!
//! ## Architecture
//!
//! The analysis is structured as a three-way reduced product:
//!
//! - **D\_spec** ([`spec_domain`]): Speculative Reachability Domain — tracks which
//!   program points are reachable under transient execution (branch misprediction,
//!   store-to-load forwarding, return stack buffer, etc).
//!
//! - **D\_cache** ([`cache_domain`]): Tainted Abstract Cache-State Domain — models
//!   LRU cache state with per-line taint annotations tracking secret dependence.
//!
//! - **D\_quant** ([`quant_domain`]): Quantitative Channel-Capacity Domain —
//!   bounds the information leakage in bits by counting distinguishable cache
//!   configurations restricted to tainted lines.
//!
//! - **ρ** ([`reduction`]): The reduction operator that sharpens the product:
//!   unreachable speculative paths remove cache taint, untainted cache sets
//!   zero their leakage contribution.
//!
//! ## Modules
//!
//! - [`product`]: Three-way reduced product combining all domains.
//! - [`transfer`]: Transfer functions for instruction-level updates.
//! - [`fixpoint`]: Worklist-based fixpoint computation over CFGs.
//! - [`regression`]: Regression detection comparing analysis results.

pub mod spec_domain;
pub mod cache_domain;
pub mod plru_domain;
pub mod quant_domain;
pub mod reduction;
pub mod product;
pub mod transfer;
pub mod fixpoint;
pub mod regression;

#[cfg(test)]
mod tests;

// Re-export primary types for convenient access.
pub use spec_domain::{SpecDomain, SpecState, SpecTag, MisspecKind, SpecWindow};
pub use cache_domain::{
    CacheDomain, AbstractCacheSet, AbstractCacheWay, CacheLineState,
    TaintAnnotation, TaintSource, AbstractAge,
};
pub use plru_domain::{
    PlruAbstractDomain, PlruAbstractSet, PlruTree, AbstractBit,
    PlruTightnessReport,
};
pub use quant_domain::{QuantDomain, QuantState, LeakageBits, SetLeakage};
pub use reduction::{
    ReductionOperator, ReductionResult, IterativeReduction, SinglePassReduction,
};
pub use product::ReducedProductState;
pub use transfer::{SpecTransfer, CacheTransfer, QuantTransfer, CombinedTransfer};
pub use fixpoint::{AnalysisEngine, AnalysisResult, AnalysisConfig};
pub use regression::{RegressionDetector, RegressionReport, ContractDelta};
