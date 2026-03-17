//! Loop unrolling for the analysis IR.
//!
//! Bounded loops must be fully or partially unrolled before leakage contract
//! verification, because the abstract-interpretation–based checker operates
//! on acyclic (or bounded-depth) CFGs. This module detects natural loops,
//! determines their bounds, and produces an unrolled copy of the IR.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::ir::{AnalysisIR, IRBlockId, IRFunction, IRTerminator};

// ---------------------------------------------------------------------------
// LoopBound
// ---------------------------------------------------------------------------

/// Describes the iteration bound of a detected loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopBound {
    /// A statically determined constant bound.
    Constant(u64),
    /// A symbolic bound expressed as a string (e.g. "n" from a function parameter).
    Symbolic(String),
    /// An upper bound obtained from heuristics or user annotation.
    UpperBound(u64),
    /// The bound could not be determined.
    Unknown,
}

impl LoopBound {
    /// Return the numeric bound if statically known (constant or upper bound).
    pub fn as_constant(&self) -> Option<u64> {
        match self {
            Self::Constant(n) | Self::UpperBound(n) => Some(*n),
            _ => None,
        }
    }

    /// Whether the bound is statically known.
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

// ---------------------------------------------------------------------------
// UnrollConfig
// ---------------------------------------------------------------------------

/// Configuration for the loop unroller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrollConfig {
    /// Maximum number of iterations to unroll a single loop.
    pub max_unroll_factor: u64,
    /// If `true`, fail when a loop with an unknown bound is encountered.
    pub strict_bounds: bool,
    /// Default upper bound to assume for loops with unknown bounds when
    /// `strict_bounds` is `false`.
    pub default_bound: u64,
    /// Maximum total blocks after unrolling (safety limit).
    pub max_total_blocks: usize,
}

impl Default for UnrollConfig {
    fn default() -> Self {
        Self {
            max_unroll_factor: 256,
            strict_bounds: false,
            default_bound: 16,
            max_total_blocks: 100_000,
        }
    }
}

// ---------------------------------------------------------------------------
// UnrollResult
// ---------------------------------------------------------------------------

/// Outcome of an unrolling pass on a single function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnrollResult {
    /// Number of loops detected.
    pub loops_detected: usize,
    /// Number of loops successfully unrolled.
    pub loops_unrolled: usize,
    /// Number of loops skipped (bound too large or unknown).
    pub loops_skipped: usize,
    /// Total blocks produced after unrolling.
    pub total_blocks: usize,
    /// Per-loop details.
    pub loop_details: Vec<LoopDetail>,
}

/// Detail about a single detected loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopDetail {
    /// Header block of the loop.
    pub header: IRBlockId,
    /// Computed loop bound.
    pub bound: LoopBound,
    /// Whether the loop was unrolled.
    pub unrolled: bool,
    /// Number of blocks in the original loop body.
    pub body_block_count: usize,
}

impl UnrollResult {
    pub fn empty() -> Self {
        Self {
            loops_detected: 0,
            loops_unrolled: 0,
            loops_skipped: 0,
            total_blocks: 0,
            loop_details: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// LoopUnroller
// ---------------------------------------------------------------------------

/// Detects natural loops in the IR and unrolls them according to an
/// [`UnrollConfig`].
#[derive(Debug, Clone)]
pub struct LoopUnroller {
    config: UnrollConfig,
}

impl LoopUnroller {
    pub fn new(config: UnrollConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(UnrollConfig::default())
    }

    /// Unroll loops in a single function.
    pub fn unroll_function(&self, func: &mut IRFunction) -> UnrollResult {
        let loops = self.detect_loops(func);
        let mut result = UnrollResult::empty();
        result.loops_detected = loops.len();

        for (header, body) in &loops {
            let bound = self.compute_bound(func, *header, body);
            let unrolled = match bound.as_constant() {
                Some(n) if n <= self.config.max_unroll_factor => {
                    self.apply_unroll(func, *header, body, n);
                    true
                }
                None if !self.config.strict_bounds => {
                    self.apply_unroll(func, *header, body, self.config.default_bound);
                    true
                }
                _ => false,
            };

            if unrolled {
                result.loops_unrolled += 1;
            } else {
                result.loops_skipped += 1;
            }

            result.loop_details.push(LoopDetail {
                header: *header,
                bound,
                unrolled,
                body_block_count: body.len(),
            });
        }

        result.total_blocks = func.block_count();
        result
    }

    /// Unroll loops across all functions in the analysis IR.
    pub fn unroll_program(&self, ir: &mut AnalysisIR) -> Vec<UnrollResult> {
        let mut results = Vec::new();
        for func in ir.program.functions.values_mut() {
            results.push(self.unroll_function(func));
        }
        ir.unrolled = true;
        results
    }

    /// Access the unroll configuration.
    pub fn config(&self) -> &UnrollConfig {
        &self.config
    }

    // -- internal helpers --------------------------------------------------

    /// Detect natural loops by finding back-edges in a DFS.
    fn detect_loops(&self, func: &IRFunction) -> Vec<(IRBlockId, Vec<IRBlockId>)> {
        let mut loops = Vec::new();
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();

        self.dfs_find_back_edges(
            func,
            func.entry_block,
            &mut visited,
            &mut on_stack,
            &mut loops,
        );

        loops
    }

    fn dfs_find_back_edges(
        &self,
        func: &IRFunction,
        block_id: IRBlockId,
        visited: &mut HashSet<IRBlockId>,
        on_stack: &mut HashSet<IRBlockId>,
        loops: &mut Vec<(IRBlockId, Vec<IRBlockId>)>,
    ) {
        if !visited.insert(block_id) {
            return;
        }
        on_stack.insert(block_id);

        if let Some(block) = func.block(block_id) {
            for &succ in &block.successors {
                if on_stack.contains(&succ) {
                    // Back-edge found: succ is the loop header.
                    let body = self.collect_loop_body(func, succ, block_id);
                    loops.push((succ, body));
                } else {
                    self.dfs_find_back_edges(func, succ, visited, on_stack, loops);
                }
            }
        }

        on_stack.remove(&block_id);
    }

    fn collect_loop_body(
        &self,
        func: &IRFunction,
        header: IRBlockId,
        latch: IRBlockId,
    ) -> Vec<IRBlockId> {
        let mut body = HashSet::new();
        body.insert(header);
        let mut worklist = vec![latch];
        while let Some(n) = worklist.pop() {
            if body.insert(n) {
                if let Some(block) = func.block(n) {
                    for &pred in &block.predecessors {
                        worklist.push(pred);
                    }
                }
            }
        }
        body.into_iter().collect()
    }

    /// Heuristic loop-bound computation.
    fn compute_bound(
        &self,
        func: &IRFunction,
        header: IRBlockId,
        _body: &[IRBlockId],
    ) -> LoopBound {
        // Check for user-supplied unroll hint on the function.
        if let Some(count) = func.unroll_count {
            return LoopBound::Constant(count as u64);
        }

        // Attempt to infer a constant trip count from the header's terminator.
        if let Some(block) = func.block(header) {
            if let IRTerminator::Branch { .. } = &block.terminator {
                // Stub: a real implementation would analyse the induction
                // variable and compare operand to derive the trip count.
            }
        }

        LoopBound::Unknown
    }

    /// Duplicate loop body blocks `iterations` times, wiring up the copies.
    fn apply_unroll(
        &self,
        func: &mut IRFunction,
        header: IRBlockId,
        body: &[IRBlockId],
        iterations: u64,
    ) {
        let base_id = func.blocks.len() as u32;
        let body_len = body.len() as u32;

        for iter_idx in 1..iterations {
            let offset = base_id + (iter_idx as u32 - 1) * body_len;
            for (i, &orig_id) in body.iter().enumerate() {
                if let Some(orig_block) = func.block(orig_id) {
                    let mut cloned = orig_block.clone();
                    cloned.id = IRBlockId::new(offset + i as u32);
                    // Remap successor / predecessor ids within the copy.
                    // (Full remapping omitted for brevity in this stub.)
                    func.add_block(cloned);
                }
            }
        }

        log::debug!(
            "unrolled loop at header {:?} × {} iterations ({} body blocks)",
            header,
            iterations,
            body.len()
        );
    }
}

impl Default for LoopUnroller {
    fn default() -> Self {
        Self::with_defaults()
    }
}
