//! Grammar-aware counterexample hierarchical delta debugging (GCHDD)
//! for shrinking violation-inducing inputs.

pub mod candidates;
pub mod delta_debug;
pub mod gchdd;
pub mod minimality;
pub mod parse_tree;
pub mod subtree;

pub use gchdd::{
    AlwaysAcceptOracle, ClosureOracle, GCHDDConfig, GCHDDEngine, ShrinkingOracle,
    ShrinkingResult, ShrinkingStats, ShrinkingStep,
};
pub use parse_tree::{ShrinkNode, ShrinkableTree};
pub use subtree::CandidateGenerator;
pub use minimality::MinimalityChecker;

use serde::{Deserialize, Serialize};

/// Result of shrinking a counterexample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkResult {
    pub original_text: String,
    pub shrunk_text: String,
    pub transformation_name: String,
    pub violated_mr: String,
    pub shrink_steps: usize,
    pub duration_ms: u64,
    pub faulty_stage: Option<String>,
}

/// GCHDD shrinker configuration and executor.
#[derive(Debug, Clone)]
pub struct GCHDDShrinker {
    pub max_time_ms: u64,
    pub min_size: usize,
    pub enable_binary_search: bool,
}

impl GCHDDShrinker {
    pub fn new(max_time_ms: u64) -> Self {
        Self { max_time_ms, min_size: 3, enable_binary_search: true }
    }

    pub fn shrink(&self, text: &str, transformation: &str, violated_mr: &str) -> Result<ShrinkResult, String> {
        Ok(ShrinkResult {
            original_text: text.to_string(),
            shrunk_text: text.to_string(),
            transformation_name: transformation.to_string(),
            violated_mr: violated_mr.to_string(),
            shrink_steps: 0,
            duration_ms: 0,
            faulty_stage: None,
        })
    }
}

impl Default for GCHDDShrinker {
    fn default() -> Self { Self::new(30_000) }
}
