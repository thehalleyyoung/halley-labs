//! Analysis intermediate representation for the leakage contract verification pipeline.
//!
//! Provides the core IR types used throughout the lifting, annotation, normalization,
//! and verification stages. The IR is designed to faithfully represent x86-64 semantics
//! while abstracting away encoding details to facilitate side-channel analysis.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use shared_types::{
    BlockId, FunctionId, Opcode, RegisterId, SecurityLevel, VirtualAddress,
};

/// Identifies a block within the analysis IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IRBlockId(pub u32);

impl IRBlockId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Create an IRBlockId from a shared-types [`BlockId`].
    pub fn from_block_id(bid: BlockId) -> Self {
        Self(bid.0)
    }

    /// Convert back to a shared-types [`BlockId`].
    pub fn to_block_id(self) -> BlockId {
        BlockId(self.0)
    }
}

/// Cryptographic primitive classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CryptoKind {
    /// AES round operations (AESENC, AESDEC, etc.)
    AesRound,
    /// AES key schedule operations
    AesKeySchedule,
    /// Carry-less multiplication (PCLMULQDQ)
    CarrylessMultiply,
    /// SHA extension operations
    Sha,
    /// Constant-time comparison
    ConstantTimeCompare,
    /// Modular exponentiation pattern
    ModularExponentiation,
    /// Generic secret-dependent arithmetic
    SecretArithmetic,
    /// Unknown / unclassified cryptographic operation
    Unknown,
}

/// Side-effect produced by an IR instruction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IREffect {
    /// Read from memory at the given (possibly symbolic) address.
    MemoryRead {
        address_registers: SmallVec<[RegisterId; 2]>,
        size_bytes: u32,
    },
    /// Write to memory.
    MemoryWrite {
        address_registers: SmallVec<[RegisterId; 2]>,
        size_bytes: u32,
    },
    /// Modification of processor flags.
    FlagsWrite,
    /// Stack push.
    StackPush { size_bytes: u32 },
    /// Stack pop.
    StackPop { size_bytes: u32 },
    /// Fence / serialisation barrier.
    Fence,
    /// Cryptographic operation with a classified kind.
    Crypto(CryptoKind),
    /// System call or privileged instruction.
    Syscall,
    /// No observable side-effect.
    None,
}

/// An operand in the analysis IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IROperand {
    /// Register operand.
    Register {
        id: RegisterId,
        size_bits: u32,
    },
    /// Immediate / constant value.
    Immediate {
        value: i64,
        size_bits: u32,
    },
    /// Memory operand with base, optional index, scale, and displacement.
    Memory {
        base: Option<RegisterId>,
        index: Option<RegisterId>,
        scale: u8,
        displacement: i64,
        size_bits: u32,
    },
    /// A relative code offset (used in branches).
    RelativeOffset(i64),
}

impl IROperand {
    pub fn reg(id: RegisterId, size_bits: u32) -> Self {
        Self::Register { id, size_bits }
    }

    pub fn imm(value: i64, size_bits: u32) -> Self {
        Self::Immediate { value, size_bits }
    }

    pub fn mem_base(base: RegisterId, size_bits: u32) -> Self {
        Self::Memory {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: 0,
            size_bits,
        }
    }

    /// Collect all register ids referenced by this operand.
    pub fn referenced_registers(&self) -> SmallVec<[RegisterId; 2]> {
        match self {
            Self::Register { id, .. } => {
                let mut v = SmallVec::new();
                v.push(*id);
                v
            }
            Self::Memory { base, index, .. } => {
                let mut regs = SmallVec::new();
                if let Some(b) = base {
                    regs.push(*b);
                }
                if let Some(i) = index {
                    regs.push(*i);
                }
                regs
            }
            _ => SmallVec::new(),
        }
    }

    pub fn is_register(&self) -> bool {
        matches!(self, Self::Register { .. })
    }

    pub fn is_memory(&self) -> bool {
        matches!(self, Self::Memory { .. })
    }

    pub fn is_immediate(&self) -> bool {
        matches!(self, Self::Immediate { .. })
    }
}

/// A single instruction in the analysis IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRInstruction {
    /// Original virtual address in the binary.
    pub address: VirtualAddress,
    /// Original opcode from shared-types.
    pub opcode: Opcode,
    /// Human-readable mnemonic.
    pub mnemonic: String,
    /// Destination operands.
    pub defs: SmallVec<[IROperand; 2]>,
    /// Source operands.
    pub uses: SmallVec<[IROperand; 4]>,
    /// Observable side-effects.
    pub effects: SmallVec<[IREffect; 2]>,
    /// Security classification of this instruction's result.
    pub security_level: Option<SecurityLevel>,
    /// If this instruction is part of a crypto primitive, its kind.
    pub crypto_kind: Option<CryptoKind>,
    /// Length of the original encoded instruction in bytes.
    pub original_length: u8,
}

impl IRInstruction {
    pub fn new(address: VirtualAddress, opcode: Opcode, mnemonic: &str) -> Self {
        Self {
            address,
            opcode,
            mnemonic: mnemonic.to_string(),
            defs: SmallVec::new(),
            uses: SmallVec::new(),
            effects: SmallVec::new(),
            security_level: None,
            crypto_kind: None,
            original_length: 0,
        }
    }

    pub fn with_def(mut self, operand: IROperand) -> Self {
        self.defs.push(operand);
        self
    }

    pub fn with_use(mut self, operand: IROperand) -> Self {
        self.uses.push(operand);
        self
    }

    pub fn with_effect(mut self, effect: IREffect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Returns `true` if the instruction accesses memory.
    pub fn accesses_memory(&self) -> bool {
        self.effects.iter().any(|e| {
            matches!(e, IREffect::MemoryRead { .. } | IREffect::MemoryWrite { .. })
        })
    }

    /// Returns `true` if the instruction is classified as cryptographic.
    pub fn is_crypto(&self) -> bool {
        self.crypto_kind.is_some()
    }

    /// All registers read by this instruction.
    pub fn read_registers(&self) -> SmallVec<[RegisterId; 4]> {
        let mut regs = SmallVec::new();
        for u in &self.uses {
            regs.extend(u.referenced_registers());
        }
        regs
    }

    /// All registers written by this instruction.
    pub fn written_registers(&self) -> SmallVec<[RegisterId; 2]> {
        let mut regs = SmallVec::new();
        for d in &self.defs {
            if let IROperand::Register { id, .. } = d {
                regs.push(*id);
            }
        }
        regs
    }
}

/// How an IR block transfers control at its end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRTerminator {
    /// Unconditional jump to a single successor.
    Goto(IRBlockId),
    /// Conditional branch: (condition register, true-target, false-target).
    Branch {
        condition: RegisterId,
        true_target: IRBlockId,
        false_target: IRBlockId,
    },
    /// Indirect jump through a register.
    IndirectJump { target_reg: RegisterId },
    /// Function call (target block, return continuation block).
    Call {
        target: VirtualAddress,
        return_block: Option<IRBlockId>,
    },
    /// Return from the current function.
    Return,
    /// Unreachable / no terminator assigned yet.
    Unreachable,
}

/// A basic block in the analysis IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRBlock {
    /// Unique identifier.
    pub id: IRBlockId,
    /// Start address in the original binary.
    pub start_address: VirtualAddress,
    /// Ordered sequence of IR instructions.
    pub instructions: Vec<IRInstruction>,
    /// How control leaves this block.
    pub terminator: IRTerminator,
    /// Successor block ids.
    pub successors: SmallVec<[IRBlockId; 2]>,
    /// Predecessor block ids.
    pub predecessors: SmallVec<[IRBlockId; 4]>,
    /// Whether this is the entry block of its function.
    pub is_entry: bool,
    /// Whether this is an exit (return) block.
    pub is_exit: bool,
    /// Loop nesting depth (0 = not in a loop).
    pub loop_depth: u32,
}

impl IRBlock {
    pub fn new(id: IRBlockId, start_address: VirtualAddress) -> Self {
        Self {
            id,
            start_address,
            instructions: Vec::new(),
            terminator: IRTerminator::Unreachable,
            successors: SmallVec::new(),
            predecessors: SmallVec::new(),
            is_entry: false,
            is_exit: false,
            loop_depth: 0,
        }
    }

    pub fn push_instruction(&mut self, instr: IRInstruction) {
        self.instructions.push(instr);
    }

    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Returns `true` if any instruction in the block accesses memory.
    pub fn has_memory_access(&self) -> bool {
        self.instructions.iter().any(|i| i.accesses_memory())
    }

    /// Count of memory-accessing instructions.
    pub fn memory_access_count(&self) -> usize {
        self.instructions.iter().filter(|i| i.accesses_memory()).count()
    }
}

/// An IR-level function composed of blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRFunction {
    /// Shared-types function id.
    pub id: FunctionId,
    /// Symbol name.
    pub name: String,
    /// Entry point address.
    pub entry_address: VirtualAddress,
    /// Entry block of the function's IR CFG.
    pub entry_block: IRBlockId,
    /// All blocks belonging to this function, keyed by IRBlockId.
    pub blocks: indexmap::IndexMap<IRBlockId, IRBlock>,
    /// Whether this function has been classified as cryptographic.
    pub is_crypto: bool,
    /// Optional unroll factor applied during loop unrolling.
    pub unroll_count: Option<u32>,
}

impl IRFunction {
    pub fn new(id: FunctionId, name: &str, entry_address: VirtualAddress, entry_block: IRBlockId) -> Self {
        Self {
            id,
            name: name.to_string(),
            entry_address,
            entry_block,
            blocks: indexmap::IndexMap::new(),
            is_crypto: false,
            unroll_count: None,
        }
    }

    pub fn add_block(&mut self, block: IRBlock) {
        self.blocks.insert(block.id, block);
    }

    pub fn block(&self, id: IRBlockId) -> Option<&IRBlock> {
        self.blocks.get(&id)
    }

    pub fn block_mut(&mut self, id: IRBlockId) -> Option<&mut IRBlock> {
        self.blocks.get_mut(&id)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn instruction_count(&self) -> usize {
        self.blocks.values().map(|b| b.instruction_count()).sum()
    }

    /// Iterate over blocks in insertion order.
    pub fn blocks_iter(&self) -> impl Iterator<Item = &IRBlock> {
        self.blocks.values()
    }
}

/// A whole-program IR suitable for leakage analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRProgram {
    /// Program name or binary path.
    pub name: String,
    /// All functions in the program, keyed by FunctionId.
    pub functions: indexmap::IndexMap<FunctionId, IRFunction>,
    /// Optional entry function id.
    pub entry_function: Option<FunctionId>,
}

impl IRProgram {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            functions: indexmap::IndexMap::new(),
            entry_function: None,
        }
    }

    pub fn add_function(&mut self, func: IRFunction) {
        self.functions.insert(func.id, func);
    }

    pub fn function(&self, id: FunctionId) -> Option<&IRFunction> {
        self.functions.get(&id)
    }

    pub fn function_mut(&mut self, id: FunctionId) -> Option<&mut IRFunction> {
        self.functions.get_mut(&id)
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    pub fn total_instruction_count(&self) -> usize {
        self.functions.values().map(|f| f.instruction_count()).sum()
    }

    pub fn total_block_count(&self) -> usize {
        self.functions.values().map(|f| f.block_count()).sum()
    }
}

/// The top-level analysis IR container, wrapping an [`IRProgram`] with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisIR {
    /// The underlying program representation.
    pub program: IRProgram,
    /// Whether security annotations have been applied.
    pub annotated: bool,
    /// Whether normalization passes have been run.
    pub normalized: bool,
    /// Whether loops have been unrolled.
    pub unrolled: bool,
}

impl AnalysisIR {
    pub fn new(program: IRProgram) -> Self {
        Self {
            program,
            annotated: false,
            normalized: false,
            unrolled: false,
        }
    }

    /// Access the underlying program.
    pub fn program(&self) -> &IRProgram {
        &self.program
    }

    /// Mutably access the underlying program.
    pub fn program_mut(&mut self) -> &mut IRProgram {
        &mut self.program
    }

    pub fn function_count(&self) -> usize {
        self.program.function_count()
    }

    pub fn total_instruction_count(&self) -> usize {
        self.program.total_instruction_count()
    }
}
