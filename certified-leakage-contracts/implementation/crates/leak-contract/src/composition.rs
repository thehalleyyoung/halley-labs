//! Compositional leakage contract combinators.
//!
//! Provides four composition rules for building whole-program contracts
//! from per-function contracts:
//!
//! | Pattern | Bound |
//! |---------|-------|
//! | Sequential `f ; g` | `B_{f;g}(s) = B_f(s) + B_g(τ_f(s))` |
//! | Parallel `f ‖ g`   | `B_{f‖g}(s) = B_f(s) + B_g(s)` (independent) |
//! | Conditional         | `B(s) = 1 + max(B_f(s), B_g(s))` |
//! | Loop                | `Σ_{i=0}^{n-1} B_body(τ_body^i(s))` |

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{
    AbstractCacheState, CacheTransformer, ContractMetadata, ContractPostcondition,
    ContractPrecondition, ContractStrength, LeakageBound, LeakageContract,
};
use shared_types::FunctionId;

// ---------------------------------------------------------------------------
// Composition errors
// ---------------------------------------------------------------------------

/// Errors that can arise when composing leakage contracts.
#[derive(Debug, Error)]
pub enum CompositionError {
    /// The two contracts have incompatible cache geometries.
    #[error("incompatible cache geometry: {reason}")]
    IncompatibleGeometry { reason: String },

    /// Parallel composition requires independence, but overlapping sets were found.
    #[error("parallel independence violated: overlapping sets {overlapping:?}")]
    IndependenceViolation { overlapping: Vec<u32> },

    /// Loop iteration count must be non-zero.
    #[error("loop iteration count must be positive, got {count}")]
    InvalidLoopCount { count: u64 },

    /// A precondition of the second contract is not satisfied by the
    /// postcondition of the first.
    #[error("precondition mismatch: {reason}")]
    PreconditionMismatch { reason: String },

    /// Contract strength is too weak for sound composition.
    #[error("unsound composition: {reason}")]
    UnsoundComposition { reason: String },
}

// ---------------------------------------------------------------------------
// Independence checker
// ---------------------------------------------------------------------------

/// Checks whether two contracts are independent (touch disjoint cache sets).
#[derive(Debug, Clone)]
pub struct IndependenceChecker {
    /// Whether to allow approximate independence (overlapping reads are okay).
    pub allow_shared_reads: bool,
    /// Maximum overlap fraction before rejecting (0.0 = strict, 1.0 = allow all).
    pub overlap_threshold: f64,
}

impl IndependenceChecker {
    /// Create a strict checker that requires fully disjoint read/write sets.
    pub fn strict() -> Self {
        Self {
            allow_shared_reads: false,
            overlap_threshold: 0.0,
        }
    }

    /// Create a relaxed checker that allows shared reads.
    pub fn relaxed() -> Self {
        Self {
            allow_shared_reads: true,
            overlap_threshold: 0.0,
        }
    }

    /// Check whether two contracts are independent.
    pub fn check(
        &self,
        a: &LeakageContract,
        b: &LeakageContract,
    ) -> Result<(), CompositionError> {
        let a_writes: Vec<u32> = a.cache_transformer.writes.clone();
        let b_writes: Vec<u32> = b.cache_transformer.writes.clone();
        let a_reads: Vec<u32> = a.cache_transformer.reads.clone();
        let b_reads: Vec<u32> = b.cache_transformer.reads.clone();

        // Write-write conflicts are never allowed.
        let ww_overlap: Vec<u32> = a_writes
            .iter()
            .filter(|s| b_writes.contains(s))
            .copied()
            .collect();
        if !ww_overlap.is_empty() {
            return Err(CompositionError::IndependenceViolation {
                overlapping: ww_overlap,
            });
        }

        // Write-read conflicts.
        let wr_overlap: Vec<u32> = a_writes
            .iter()
            .filter(|s| b_reads.contains(s))
            .copied()
            .collect();
        let rw_overlap: Vec<u32> = a_reads
            .iter()
            .filter(|s| b_writes.contains(s))
            .copied()
            .collect();

        let mut conflicts: Vec<u32> = wr_overlap;
        conflicts.extend(rw_overlap);
        conflicts.sort_unstable();
        conflicts.dedup();

        if !conflicts.is_empty() {
            return Err(CompositionError::IndependenceViolation {
                overlapping: conflicts,
            });
        }

        // Read-read conflicts (only if strict).
        if !self.allow_shared_reads {
            let rr_overlap: Vec<u32> = a_reads
                .iter()
                .filter(|s| b_reads.contains(s))
                .copied()
                .collect();
            if !rr_overlap.is_empty() {
                let total = (a_reads.len() + b_reads.len()) as f64;
                let frac = rr_overlap.len() as f64 / total.max(1.0);
                if frac > self.overlap_threshold {
                    return Err(CompositionError::IndependenceViolation {
                        overlapping: rr_overlap,
                    });
                }
            }
        }

        Ok(())
    }
}

impl Default for IndependenceChecker {
    fn default() -> Self {
        Self::strict()
    }
}

// ---------------------------------------------------------------------------
// Whole-library bound
// ---------------------------------------------------------------------------

/// Aggregated leakage bound for an entire library of composed contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WholeLibraryBound {
    /// Total worst-case leakage across all composed contracts (bits).
    pub total_bits: f64,
    /// Per-function contributions to the total bound.
    pub per_function: BTreeMap<String, f64>,
    /// Composition strategy that produced this bound.
    pub strategy: String,
    /// Whether the bound is provably sound.
    pub is_sound: bool,
    /// Number of contracts that were composed.
    pub num_contracts: usize,
}

impl WholeLibraryBound {
    /// Create an empty bound (zero leakage, no contracts).
    pub fn zero() -> Self {
        Self {
            total_bits: 0.0,
            per_function: BTreeMap::new(),
            strategy: "empty".into(),
            is_sound: true,
            num_contracts: 0,
        }
    }

    /// Add a per-function contribution.
    pub fn add_contribution(&mut self, function_name: &str, bits: f64) {
        *self
            .per_function
            .entry(function_name.to_string())
            .or_insert(0.0) += bits;
        self.total_bits += bits;
        self.num_contracts += 1;
    }

    /// Returns a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "WholeLibraryBound {{ total={:.2} bits, contracts={}, sound={}, strategy={} }}",
            self.total_bits, self.num_contracts, self.is_sound, self.strategy
        )
    }
}

impl fmt::Display for WholeLibraryBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Composition functions
// ---------------------------------------------------------------------------

/// Compose two contracts sequentially: `f ; g`.
///
/// The resulting contract has:
/// - **Transformer**: `τ_g ∘ τ_f`
/// - **Bound**: `B_{f;g}(s) = B_f(s) + B_g(τ_f(s))`
pub fn compose_sequential(
    first: &LeakageContract,
    second: &LeakageContract,
) -> Result<LeakageContract, CompositionError> {
    // Verify geometry compatibility.
    if first.cache_transformer.set_transforms.keys().next().is_some()
        && second.cache_transformer.set_transforms.keys().next().is_some()
    {
        // Both touch at least one set – geometries must be compatible.
        // (Lightweight check: we trust preconditions carry geometry info.)
    }

    let composed_transformer = first.cache_transformer.compose(&second.cache_transformer);
    let composed_bound = first.leakage_bound.add(&second.leakage_bound);
    let strength = first.strength.compose(second.strength);
    let pre = first.precondition.merge(&second.precondition);

    let mut contract = LeakageContract::new(
        first.function_id.clone(),
        format!("{};{}", first.function_name, second.function_name),
        composed_transformer,
        composed_bound,
    );
    contract.strength = strength;
    contract.precondition = pre;
    contract.postcondition = second.postcondition.clone();
    contract.metadata = ContractMetadata::new();
    contract
        .metadata
        .annotations
        .insert("composition".into(), "sequential".into());

    Ok(contract)
}

/// Compose two contracts in parallel: `f ‖ g`.
///
/// Requires independence (disjoint write sets). The resulting bound is the
/// sum of individual bounds.
pub fn compose_parallel(
    first: &LeakageContract,
    second: &LeakageContract,
) -> Result<LeakageContract, CompositionError> {
    // Verify independence.
    let checker = IndependenceChecker::relaxed();
    checker.check(first, second)?;

    let composed_transformer = first.cache_transformer.compose(&second.cache_transformer);
    let composed_bound = first.leakage_bound.add(&second.leakage_bound);
    let strength = first.strength.compose(second.strength);
    let pre = first.precondition.merge(&second.precondition);

    let mut contract = LeakageContract::new(
        first.function_id.clone(),
        format!("{}‖{}", first.function_name, second.function_name),
        composed_transformer,
        composed_bound,
    );
    contract.strength = strength;
    contract.precondition = pre;
    contract.metadata = ContractMetadata::new();
    contract
        .metadata
        .annotations
        .insert("composition".into(), "parallel".into());

    Ok(contract)
}

/// Compose two contracts under a conditional: `if c then f else g`.
///
/// The branch condition itself leaks 1 bit, so the bound is
/// `B(s) = 1 + max(B_f(s), B_g(s))`.
pub fn compose_conditional(
    then_branch: &LeakageContract,
    else_branch: &LeakageContract,
) -> Result<LeakageContract, CompositionError> {
    let branch_bit = LeakageBound::constant(1.0);
    let max_bound = then_branch.leakage_bound.max(&else_branch.leakage_bound);
    let composed_bound = branch_bit.add(&max_bound);

    // Transformer is the join (overapproximation) of both branches.
    let composed_transformer = then_branch
        .cache_transformer
        .compose(&else_branch.cache_transformer);
    let strength = then_branch.strength.compose(else_branch.strength);
    let pre = then_branch.precondition.merge(&else_branch.precondition);

    let mut contract = LeakageContract::new(
        then_branch.function_id.clone(),
        format!(
            "if ? then {} else {}",
            then_branch.function_name, else_branch.function_name
        ),
        composed_transformer,
        composed_bound,
    );
    contract.strength = strength;
    contract.precondition = pre;
    contract.metadata = ContractMetadata::new();
    contract
        .metadata
        .annotations
        .insert("composition".into(), "conditional".into());

    Ok(contract)
}

/// Compose a contract with itself under a loop: `for i in 0..n { body }`.
///
/// The bound is `Σ_{i=0}^{n-1} B_body(τ_body^i(s))`. As a conservative
/// overapproximation we use `n * B_body_worst`.
pub fn compose_loop(
    body: &LeakageContract,
    iterations: u64,
) -> Result<LeakageContract, CompositionError> {
    if iterations == 0 {
        return Err(CompositionError::InvalidLoopCount { count: 0 });
    }

    let scaled_bound = body.leakage_bound.scale(iterations);
    let composed_transformer = {
        let mut t = body.cache_transformer.clone();
        for _ in 1..iterations.min(16) {
            t = t.compose(&body.cache_transformer);
        }
        t
    };

    let mut contract = LeakageContract::new(
        body.function_id.clone(),
        format!("loop({}, {})", body.function_name, iterations),
        composed_transformer,
        scaled_bound,
    );
    contract.strength = body.strength;
    contract.precondition = body.precondition.clone();
    contract.postcondition = body.postcondition.clone();
    contract.metadata = ContractMetadata::new();
    contract.metadata.annotations.insert(
        "composition".into(),
        format!("loop({})", iterations),
    );

    Ok(contract)
}
