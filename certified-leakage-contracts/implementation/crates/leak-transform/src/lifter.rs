//! Instruction lifting from machine code to the analysis IR.
//!
//! The lifter translates raw x86-64 instructions (from shared-types [`Instruction`])
//! into the richer [`IRInstruction`] representation, capturing operands, effects, and
//! crypto classification so that downstream passes can reason about side-channels.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use shared_types::{
    BasicBlock, Function, Instruction, Opcode,
    RegisterId, VirtualAddress,
};

use crate::ir::{
    AnalysisIR, CryptoKind, IRBlock, IRBlockId, IREffect, IRFunction, IRInstruction,
    IROperand, IRProgram, IRTerminator,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during instruction lifting.
#[derive(Debug, Error)]
pub enum LiftError {
    #[error("unsupported opcode: {opcode:?} at {address}")]
    UnsupportedOpcode {
        opcode: Opcode,
        address: VirtualAddress,
    },

    #[error("invalid operand encoding at {address}: {detail}")]
    InvalidOperand {
        address: VirtualAddress,
        detail: String,
    },

    #[error("function `{name}` has no entry block")]
    NoEntryBlock { name: String },

    #[error("empty control-flow graph for function `{name}`")]
    EmptyCfg { name: String },

    #[error("block {block_id:?} not found in CFG")]
    BlockNotFound { block_id: shared_types::BlockId },

    #[error("internal lifter error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// Result alias
// ---------------------------------------------------------------------------

/// Convenience result type for lifting operations.
pub type LiftResult<T> = Result<T, LiftError>;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Architecture-agnostic instruction lifter.
///
/// Implementors translate [`Instruction`] (or higher-level constructs such as
/// [`Function`] and [`ControlFlowGraph`]) into the analysis IR.
pub trait InstructionLifter {
    /// Lift a single shared-types instruction into one or more IR instructions.
    fn lift_instruction(&self, instr: &Instruction) -> LiftResult<Vec<IRInstruction>>;

    /// Lift an entire [`BasicBlock`] into an [`IRBlock`].
    fn lift_block(&self, block: &BasicBlock) -> LiftResult<IRBlock>;

    /// Lift a full [`Function`] (with its CFG) into an [`IRFunction`].
    fn lift_function(&self, func: &Function) -> LiftResult<IRFunction>;

    /// Lift a program consisting of multiple functions into an [`AnalysisIR`].
    fn lift_program(&self, functions: &[Function]) -> LiftResult<AnalysisIR> {
        let mut program = IRProgram::new("lifted");
        for func in functions {
            let ir_func = self.lift_function(func)?;
            program.add_function(ir_func);
        }
        Ok(AnalysisIR::new(program))
    }
}

// ---------------------------------------------------------------------------
// X86Lifter
// ---------------------------------------------------------------------------

/// Configuration for the x86-64 lifter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct X86LifterConfig {
    /// Whether to classify crypto instructions (AES-NI, PCLMULQDQ, …).
    pub detect_crypto: bool,
    /// Whether to model implicit flag writes.
    pub model_flags: bool,
    /// Whether to expand complex instructions into micro-ops.
    pub expand_complex: bool,
}

impl Default for X86LifterConfig {
    fn default() -> Self {
        Self {
            detect_crypto: true,
            model_flags: true,
            expand_complex: false,
        }
    }
}

/// x86-64 instruction lifter.
#[derive(Debug, Clone)]
pub struct X86Lifter {
    config: X86LifterConfig,
    /// Mapping from opcode to crypto classification, populated at construction.
    crypto_map: HashMap<Opcode, CryptoKind>,
}

impl X86Lifter {
    pub fn new() -> Self {
        Self::with_config(X86LifterConfig::default())
    }

    pub fn with_config(config: X86LifterConfig) -> Self {
        let crypto_map = Self::build_crypto_map();
        Self { config, crypto_map }
    }

    /// Classify an opcode as a cryptographic primitive (if applicable).
    pub fn classify_crypto(&self, opcode: Opcode) -> Option<CryptoKind> {
        self.crypto_map.get(&opcode).copied()
    }

    // -- internal helpers --------------------------------------------------

    fn build_crypto_map() -> HashMap<Opcode, CryptoKind> {
        let mut m = HashMap::new();
        m.insert(Opcode::AESENC, CryptoKind::AesRound);
        m.insert(Opcode::AESENCLAST, CryptoKind::AesRound);
        m.insert(Opcode::AESDEC, CryptoKind::AesRound);
        m.insert(Opcode::AESDECLAST, CryptoKind::AesRound);
        m.insert(Opcode::AESKEYGENASSIST, CryptoKind::AesKeySchedule);
        m.insert(Opcode::PCLMULQDQ, CryptoKind::CarrylessMultiply);
        m
    }

    fn lift_operand(
        &self,
        op: &shared_types::Operand,
    ) -> LiftResult<IROperand> {
        use shared_types::OperandKind;
        match &op.kind {
            OperandKind::Register(reg_id) => Ok(IROperand::reg(*reg_id, op.size_bits)),
            OperandKind::Immediate(imm) => Ok(IROperand::imm(imm.0, op.size_bits)),
            OperandKind::Memory(mem) => Ok(IROperand::Memory {
                base: mem.base,
                index: mem.index,
                scale: mem.scale,
                displacement: mem.displacement,
                size_bits: op.size_bits,
            }),
            OperandKind::RelativeOffset(off) => Ok(IROperand::RelativeOffset(*off)),
        }
    }

    fn instruction_effects(&self, instr: &Instruction) -> Vec<IREffect> {
        let mut effects = Vec::new();
        if instr.flags.reads_memory {
            effects.push(IREffect::MemoryRead {
                address_registers: instr.implicit_reads.iter().copied().collect(),
                size_bytes: 8, // conservative default
            });
        }
        if instr.flags.writes_memory {
            effects.push(IREffect::MemoryWrite {
                address_registers: instr.implicit_writes.iter().copied().collect(),
                size_bytes: 8,
            });
        }
        if instr.flags.is_speculation_barrier {
            effects.push(IREffect::Fence);
        }
        if instr.flags.modifies_flags && self.config.model_flags {
            effects.push(IREffect::FlagsWrite);
        }
        effects
    }

    fn terminator_for_block(&self, block: &BasicBlock) -> IRTerminator {
        if block.is_exit {
            return IRTerminator::Return;
        }
        match block.successors.len() {
            0 => IRTerminator::Return,
            1 => IRTerminator::Goto(IRBlockId::from_block_id(block.successors[0])),
            _ => {
                let true_target = IRBlockId::from_block_id(block.successors[0]);
                let false_target = IRBlockId::from_block_id(block.successors[1]);
                IRTerminator::Branch {
                    condition: RegisterId::RFLAGS,
                    true_target,
                    false_target,
                }
            }
        }
    }
}

impl Default for X86Lifter {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionLifter for X86Lifter {
    fn lift_instruction(&self, instr: &Instruction) -> LiftResult<Vec<IRInstruction>> {
        let mut defs = smallvec::SmallVec::new();
        let mut uses = smallvec::SmallVec::new();

        for op in &instr.operands {
            let ir_op = self.lift_operand(op)?;
            if op.is_write {
                defs.push(ir_op.clone());
            }
            if op.is_read {
                uses.push(ir_op);
            }
        }

        let effects: smallvec::SmallVec<[IREffect; 2]> =
            self.instruction_effects(instr).into_iter().collect();

        let crypto_kind = if self.config.detect_crypto {
            self.classify_crypto(instr.opcode)
        } else {
            None
        };

        let ir = IRInstruction {
            address: instr.address,
            opcode: instr.opcode,
            mnemonic: instr.mnemonic.clone(),
            defs,
            uses,
            effects,
            security_level: None,
            crypto_kind,
            original_length: instr.length,
        };

        Ok(vec![ir])
    }

    fn lift_block(&self, block: &BasicBlock) -> LiftResult<IRBlock> {
        let ir_block_id = IRBlockId::from_block_id(block.id);
        let mut ir_block = IRBlock::new(ir_block_id, block.start_address);

        for instr in &block.instructions {
            let ir_instrs = self.lift_instruction(instr)?;
            for ir in ir_instrs {
                ir_block.push_instruction(ir);
            }
        }

        ir_block.terminator = self.terminator_for_block(block);
        ir_block.is_entry = block.is_entry;
        ir_block.is_exit = block.is_exit;
        ir_block.loop_depth = block.loop_depth;
        ir_block.successors = block.successors.iter().map(|b| IRBlockId::from_block_id(*b)).collect();
        ir_block.predecessors = block.predecessors.iter().map(|b| IRBlockId::from_block_id(*b)).collect();

        Ok(ir_block)
    }

    fn lift_function(&self, func: &Function) -> LiftResult<IRFunction> {
        let entry_bid = func
            .cfg
            .entry
            .ok_or_else(|| LiftError::NoEntryBlock {
                name: func.name.clone(),
            })?;

        let entry_ir = IRBlockId::from_block_id(entry_bid);
        let mut ir_func = IRFunction::new(func.id, &func.name, func.entry_address, entry_ir);
        ir_func.is_crypto = func.is_crypto;
        ir_func.unroll_count = func.unroll_count;

        for (_, block) in &func.cfg.blocks {
            let ir_block = self.lift_block(block)?;
            ir_func.add_block(ir_block);
        }

        Ok(ir_func)
    }
}
