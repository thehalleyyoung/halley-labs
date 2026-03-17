//! Cache-aware SMT theory encoding.
//!
//! Provides [`CacheTheory`], which encodes cache microarchitectural semantics
//! (line fetch, set indexing, tag comparison, replacement policy) as SMT
//! expressions suitable for leakage verification.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::expr::{Expr, ExprId, ExprPool, Sort};
use crate::smtlib::SmtCommand;

// ---------------------------------------------------------------------------
// CacheTheory
// ---------------------------------------------------------------------------

/// Encodes cache microarchitectural behaviour as SMT constraints.
///
/// The theory models a set-associative cache with configurable geometry.
/// It emits axioms for cache-line indexing, tag comparison, and an abstract
/// replacement policy so that the SMT solver can reason about
/// observation-equivalence of memory-access traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTheory {
    /// Number of cache sets (log₂ gives index bits).
    pub num_sets: u32,
    /// Number of ways (associativity).
    pub num_ways: u32,
    /// Cache line size in bytes (log₂ gives offset bits).
    pub line_size: u32,
    /// Address width in bits (typically 64).
    pub addr_width: u32,
    /// Whether to model the replacement policy explicitly.
    pub model_replacement: bool,
}

impl CacheTheory {
    /// Create a theory for a typical L1 data cache.
    pub fn l1_default() -> Self {
        Self {
            num_sets: 64,
            num_ways: 8,
            line_size: 64,
            addr_width: 64,
            model_replacement: false,
        }
    }

    /// Create a theory with the given geometry.
    pub fn new(num_sets: u32, num_ways: u32, line_size: u32, addr_width: u32) -> Self {
        Self {
            num_sets,
            num_ways,
            line_size,
            addr_width,
            model_replacement: false,
        }
    }

    /// Number of offset bits (log₂ of line_size).
    pub fn offset_bits(&self) -> u32 {
        self.line_size.trailing_zeros()
    }

    /// Number of index bits (log₂ of num_sets).
    pub fn index_bits(&self) -> u32 {
        self.num_sets.trailing_zeros()
    }

    /// Number of tag bits.
    pub fn tag_bits(&self) -> u32 {
        self.addr_width - self.offset_bits() - self.index_bits()
    }

    /// Emit SMT declarations for cache-state sorts and helper functions.
    pub fn emit_declarations(&self, _pool: &mut ExprPool) -> Vec<SmtCommand> {
        // TODO: emit uninterpreted sorts for CacheState, declare
        // cache_lookup / cache_update functions.
        log::debug!("CacheTheory::emit_declarations stub");
        Vec::new()
    }

    /// Build an expression for extracting the cache-set index from an address.
    pub fn set_index_expr(&self, pool: &mut ExprPool, addr: ExprId) -> ExprId {
        let lo = self.offset_bits();
        let hi = lo + self.index_bits() - 1;
        pool.intern(Expr::BvExtract(hi, lo, addr))
    }

    /// Build an expression for extracting the tag from an address.
    pub fn tag_expr(&self, pool: &mut ExprPool, addr: ExprId) -> ExprId {
        let lo = self.offset_bits() + self.index_bits();
        let hi = self.addr_width - 1;
        pool.intern(Expr::BvExtract(hi, lo, addr))
    }

    /// Assert that two addresses map to the same cache set.
    pub fn same_set_constraint(&self, pool: &mut ExprPool, a: ExprId, b: ExprId) -> ExprId {
        let idx_a = self.set_index_expr(pool, a);
        let idx_b = self.set_index_expr(pool, b);
        pool.intern(Expr::Eq(idx_a, idx_b))
    }
}

impl Default for CacheTheory {
    fn default() -> Self {
        Self::l1_default()
    }
}

impl fmt::Display for CacheTheory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheTheory(sets={}, ways={}, line={}B, addr={}b)",
            self.num_sets, self.num_ways, self.line_size, self.addr_width
        )
    }
}
