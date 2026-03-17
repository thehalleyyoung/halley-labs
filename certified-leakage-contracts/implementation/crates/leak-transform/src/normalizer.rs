//! IR normalization passes.
//!
//! Normalization simplifies and canonicalizes the analysis IR so that
//! downstream leakage analysis operates on a uniform representation.
//! Passes include constant folding, dead-code elimination, copy propagation,
//! and instruction canonicalization.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use shared_types::RegisterId;

use crate::ir::{AnalysisIR, IRFunction, IRInstruction, IROperand};

// ---------------------------------------------------------------------------
// NormalizationPass trait
// ---------------------------------------------------------------------------

/// A single normalization transformation applied to the IR.
pub trait NormalizationPass: fmt::Debug {
    /// Human-readable name of this pass.
    fn name(&self) -> &str;

    /// Apply the pass to a single function, returning the number of
    /// instructions modified or removed.
    fn run_on_function(&self, func: &mut IRFunction) -> usize;

    /// Apply the pass to the entire program.
    fn run_on_ir(&self, ir: &mut AnalysisIR) -> usize {
        let mut total = 0;
        for func in ir.program.functions.values_mut() {
            total += self.run_on_function(func);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// ConstantFolding
// ---------------------------------------------------------------------------

/// Evaluates instructions with all-constant operands at compile time.
#[derive(Debug, Clone, Default)]
pub struct ConstantFolding {
    /// Maximum bit-width to fold (default: 64).
    pub max_width: u32,
}

impl ConstantFolding {
    pub fn new() -> Self {
        Self { max_width: 64 }
    }

    fn try_fold_instruction(&self, instr: &IRInstruction) -> Option<IROperand> {
        // Only fold when all uses are immediates.
        if instr.uses.iter().all(|op| op.is_immediate()) && !instr.uses.is_empty() {
            // Stub: real implementation would evaluate the opcode.
            None
        } else {
            None
        }
    }
}

impl NormalizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "constant-folding"
    }

    fn run_on_function(&self, func: &mut IRFunction) -> usize {
        let mut count = 0;
        for block in func.blocks.values_mut() {
            for instr in &mut block.instructions {
                if let Some(_folded) = self.try_fold_instruction(instr) {
                    // Replace uses with the folded constant — stub.
                    count += 1;
                }
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// DeadCodeElimination
// ---------------------------------------------------------------------------

/// Removes instructions whose results are never used.
#[derive(Debug, Clone, Default)]
pub struct DeadCodeElimination {
    /// If `true`, also eliminate flag-only writes when flags are not read.
    pub eliminate_dead_flags: bool,
}

impl DeadCodeElimination {
    pub fn new() -> Self {
        Self {
            eliminate_dead_flags: true,
        }
    }

    fn is_dead(&self, instr: &IRInstruction, live: &HashSet<RegisterId>) -> bool {
        // An instruction is dead when none of its defs are live and it has
        // no memory or fence effects.
        if instr.accesses_memory() || instr.effects.iter().any(|e| {
            matches!(
                e,
                crate::ir::IREffect::Fence
                    | crate::ir::IREffect::Syscall
                    | crate::ir::IREffect::StackPush { .. }
                    | crate::ir::IREffect::StackPop { .. }
            )
        }) {
            return false;
        }
        instr
            .written_registers()
            .iter()
            .all(|r| !live.contains(r))
    }
}

impl NormalizationPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn run_on_function(&self, func: &mut IRFunction) -> usize {
        // Compute a conservative approximation of live registers by collecting
        // all registers read across the function.
        let live: HashSet<RegisterId> = func
            .blocks
            .values()
            .flat_map(|b| b.instructions.iter())
            .flat_map(|i| i.read_registers())
            .collect();

        let mut removed = 0;
        for block in func.blocks.values_mut() {
            let before = block.instructions.len();
            block.instructions.retain(|i| !self.is_dead(i, &live));
            removed += before - block.instructions.len();
        }
        removed
    }
}

// ---------------------------------------------------------------------------
// CopyPropagation
// ---------------------------------------------------------------------------

/// Replaces uses of a register that is a simple copy of another register
/// with the original source register.
#[derive(Debug, Clone, Default)]
pub struct CopyPropagation;

impl CopyPropagation {
    pub fn new() -> Self {
        Self
    }

    /// Detect whether `instr` is a simple register-to-register copy.
    fn is_copy(instr: &IRInstruction) -> Option<(RegisterId, RegisterId)> {
        use shared_types::Opcode;
        if instr.opcode != Opcode::MOV {
            return None;
        }
        if instr.defs.len() == 1 && instr.uses.len() == 1 {
            if let (
                IROperand::Register { id: dst, .. },
                IROperand::Register { id: src, .. },
            ) = (&instr.defs[0], &instr.uses[0])
            {
                return Some((*dst, *src));
            }
        }
        None
    }
}

impl NormalizationPass for CopyPropagation {
    fn name(&self) -> &str {
        "copy-propagation"
    }

    fn run_on_function(&self, func: &mut IRFunction) -> usize {
        // Build copy map.
        let mut copy_map: HashMap<RegisterId, RegisterId> = HashMap::new();
        for block in func.blocks.values() {
            for instr in &block.instructions {
                if let Some((dst, src)) = Self::is_copy(instr) {
                    copy_map.insert(dst, src);
                }
            }
        }

        // Resolve transitive copies.
        let resolved: HashMap<RegisterId, RegisterId> = copy_map
            .keys()
            .map(|&k| {
                let mut cur = k;
                while let Some(&next) = copy_map.get(&cur) {
                    if next == k {
                        break; // cycle
                    }
                    cur = next;
                }
                (k, cur)
            })
            .collect();

        // Rewrite uses.
        let mut count = 0;
        for block in func.blocks.values_mut() {
            for instr in &mut block.instructions {
                for u in &mut instr.uses {
                    if let IROperand::Register { id, .. } = u {
                        if let Some(&replacement) = resolved.get(id) {
                            if replacement != *id {
                                *id = replacement;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// InstructionCanonicalization
// ---------------------------------------------------------------------------

/// Rewrites instruction patterns into a canonical form
/// (e.g. `xor rax, rax` → zero-idiom; `sub rax, 0` → nop).
#[derive(Debug, Clone, Default)]
pub struct InstructionCanonicalization;

impl InstructionCanonicalization {
    pub fn new() -> Self {
        Self
    }
}

impl NormalizationPass for InstructionCanonicalization {
    fn name(&self) -> &str {
        "instruction-canonicalization"
    }

    fn run_on_function(&self, func: &mut IRFunction) -> usize {
        let mut count = 0;
        for block in func.blocks.values_mut() {
            for instr in &mut block.instructions {
                // Canonicalize `xor r, r` → mov r, 0
                if instr.opcode == shared_types::Opcode::XOR
                    && instr.uses.len() == 2
                    && instr.uses[0] == instr.uses[1]
                {
                    if let IROperand::Register { size_bits, .. } = &instr.uses[0] {
                        instr.opcode = shared_types::Opcode::MOV;
                        instr.uses = smallvec::smallvec![IROperand::imm(0, *size_bits)];
                        instr.mnemonic = "mov".to_string();
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// IRNormalizer
// ---------------------------------------------------------------------------

/// Applies a sequence of [`NormalizationPass`] implementations to the IR.
#[derive(Debug)]
pub struct IRNormalizer {
    passes: Vec<Box<dyn NormalizationPass>>,
}

impl IRNormalizer {
    /// Create a normalizer with an empty pass list.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    /// Create a normalizer pre-loaded with the standard pass pipeline.
    pub fn standard() -> Self {
        let mut n = Self::new();
        n.add_pass(Box::new(ConstantFolding::new()));
        n.add_pass(Box::new(CopyPropagation::new()));
        n.add_pass(Box::new(DeadCodeElimination::new()));
        n.add_pass(Box::new(InstructionCanonicalization::new()));
        n
    }

    /// Append a pass to the pipeline.
    pub fn add_pass(&mut self, pass: Box<dyn NormalizationPass>) {
        self.passes.push(pass);
    }

    /// Run all passes over `ir`, returning the total number of modifications.
    pub fn normalize(&self, ir: &mut AnalysisIR) -> usize {
        let mut total = 0;
        for pass in &self.passes {
            log::debug!("running normalization pass: {}", pass.name());
            total += pass.run_on_ir(ir);
        }
        ir.normalized = true;
        total
    }

    /// Number of configured passes.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

impl Default for IRNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NormalizationPipeline
// ---------------------------------------------------------------------------

/// Statistics from a normalization run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NormalizationStats {
    /// Number of changes per pass (pass name → count).
    pub per_pass: HashMap<String, usize>,
    /// Total modifications across all passes and iterations.
    pub total_changes: usize,
    /// Number of fixed-point iterations executed.
    pub iterations: usize,
}

/// A configurable, iterating normalization pipeline that re-runs passes
/// until a fixed point (or iteration limit) is reached.
#[derive(Debug)]
pub struct NormalizationPipeline {
    normalizer: IRNormalizer,
    /// Maximum iterations before stopping.
    pub max_iterations: usize,
}

impl NormalizationPipeline {
    /// Create a pipeline with the standard passes and a default iteration limit.
    pub fn new() -> Self {
        Self {
            normalizer: IRNormalizer::standard(),
            max_iterations: 10,
        }
    }

    /// Create a pipeline wrapping a custom [`IRNormalizer`].
    pub fn with_normalizer(normalizer: IRNormalizer, max_iterations: usize) -> Self {
        Self {
            normalizer,
            max_iterations,
        }
    }

    /// Run the pipeline to a fixed point, returning statistics.
    pub fn run(&self, ir: &mut AnalysisIR) -> NormalizationStats {
        let mut stats = NormalizationStats::default();

        for _ in 0..self.max_iterations {
            stats.iterations += 1;
            let changes = self.normalizer.normalize(ir);
            stats.total_changes += changes;
            if changes == 0 {
                break;
            }
        }

        stats
    }

    /// Access the inner normalizer.
    pub fn normalizer(&self) -> &IRNormalizer {
        &self.normalizer
    }
}

impl Default for NormalizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}
