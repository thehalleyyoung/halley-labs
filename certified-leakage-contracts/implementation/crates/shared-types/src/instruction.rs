//! Instruction representation for x86-64.

use serde::{Deserialize, Serialize};
use crate::address::VirtualAddress;
use crate::operand::Operand;
use crate::register::RegisterId;
use std::fmt;

/// Decoded x86-64 instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub address: VirtualAddress,
    pub length: u8,
    pub opcode: Opcode,
    pub kind: InstructionKind,
    pub operands: Vec<Operand>,
    pub flags: InstructionFlags,
    pub mnemonic: String,
    pub bytes: Vec<u8>,
    pub implicit_reads: Vec<RegisterId>,
    pub implicit_writes: Vec<RegisterId>,
}

impl Instruction {
    pub fn new(address: VirtualAddress, opcode: Opcode, kind: InstructionKind) -> Self {
        Self {
            address,
            length: 0,
            opcode,
            kind,
            operands: Vec::new(),
            flags: InstructionFlags::default(),
            mnemonic: String::new(),
            bytes: Vec::new(),
            implicit_reads: Vec::new(),
            implicit_writes: Vec::new(),
        }
    }

    pub fn end_address(&self) -> VirtualAddress {
        VirtualAddress(self.address.0 + self.length as u64)
    }

    pub fn is_branch(&self) -> bool {
        matches!(
            self.kind,
            InstructionKind::ConditionalBranch
                | InstructionKind::UnconditionalBranch
                | InstructionKind::IndirectBranch
        )
    }

    pub fn is_call(&self) -> bool {
        matches!(self.kind, InstructionKind::Call | InstructionKind::IndirectCall)
    }

    pub fn is_return(&self) -> bool {
        matches!(self.kind, InstructionKind::Return)
    }

    pub fn is_memory_access(&self) -> bool {
        self.flags.reads_memory || self.flags.writes_memory
    }

    pub fn is_fence(&self) -> bool {
        matches!(self.kind, InstructionKind::Fence)
    }

    pub fn is_nop(&self) -> bool {
        matches!(self.kind, InstructionKind::Nop)
    }

    pub fn is_speculation_barrier(&self) -> bool {
        matches!(self.opcode, Opcode::LFENCE | Opcode::CPUID | Opcode::SERIALIZE)
    }

    pub fn memory_operand(&self) -> Option<&Operand> {
        self.operands.iter().find(|o| o.is_memory())
    }

    pub fn read_registers(&self) -> Vec<RegisterId> {
        let mut regs = self.implicit_reads.clone();
        for op in &self.operands {
            op.collect_read_registers(&mut regs);
        }
        regs
    }

    pub fn written_registers(&self) -> Vec<RegisterId> {
        let mut regs = self.implicit_writes.clone();
        for op in &self.operands {
            op.collect_write_registers(&mut regs);
        }
        regs
    }

    pub fn has_secret_dependent_address(&self) -> bool {
        self.flags.secret_dependent_address
    }

    pub fn speculation_depth_cost(&self) -> u32 {
        match self.kind {
            InstructionKind::Arithmetic => 1,
            InstructionKind::Load => 4,
            InstructionKind::Store => 4,
            InstructionKind::ConditionalBranch => 1,
            InstructionKind::Multiply => 3,
            InstructionKind::Divide => 20,
            InstructionKind::VectorArith => 2,
            _ => 1,
        }
    }

    pub fn with_operand(mut self, op: Operand) -> Self {
        self.operands.push(op);
        self
    }

    pub fn with_mnemonic(mut self, m: &str) -> Self {
        self.mnemonic = m.to_string();
        self
    }

    pub fn with_length(mut self, len: u8) -> Self {
        self.length = len;
        self
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} {}", self.address, self.mnemonic, self.opcode)?;
        for (i, op) in self.operands.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, " {}", op)?;
        }
        Ok(())
    }
}

/// High-level instruction classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstructionKind {
    Nop,
    Arithmetic,
    Logic,
    Shift,
    Multiply,
    Divide,
    Load,
    Store,
    LoadStore,
    Move,
    ConditionalMove,
    Push,
    Pop,
    ConditionalBranch,
    UnconditionalBranch,
    IndirectBranch,
    Call,
    IndirectCall,
    Return,
    Fence,
    Syscall,
    VectorArith,
    VectorShuffle,
    VectorLoad,
    VectorStore,
    Crypto,
    Compare,
    Test,
    SetCondition,
    StringOp,
    BitManip,
    Convert,
    Lea,
    Exchange,
    Prefetch,
    Unknown,
}

impl InstructionKind {
    pub fn is_memory_op(self) -> bool {
        matches!(
            self,
            Self::Load | Self::Store | Self::LoadStore | Self::Push
            | Self::Pop | Self::VectorLoad | Self::VectorStore | Self::StringOp
        )
    }

    pub fn is_control_flow(self) -> bool {
        matches!(
            self,
            Self::ConditionalBranch | Self::UnconditionalBranch
            | Self::IndirectBranch | Self::Call | Self::IndirectCall | Self::Return
        )
    }

    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            Self::Arithmetic | Self::Logic | Self::Shift
            | Self::Multiply | Self::Divide | Self::VectorArith | Self::BitManip
        )
    }
}

/// x86-64 opcode enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Opcode {
    NOP, MOV, MOVZX, MOVSX, MOVSXD, LEA, CMOVZ, CMOVNZ, CMOVL, CMOVGE,
    CMOVLE, CMOVG, CMOVB, CMOVAE, CMOVBE, CMOVA, CMOVS, CMOVNS,
    ADD, SUB, ADC, SBB, INC, DEC, NEG, NOT,
    AND, OR, XOR, TEST, CMP,
    SHL, SHR, SAR, ROL, ROR, RCL, RCR, SHLD, SHRD,
    MUL, IMUL, DIV, IDIV,
    PUSH, POP, PUSHF, POPF,
    JMP, JZ, JNZ, JL, JGE, JLE, JG, JB, JAE, JBE, JA, JS, JNS,
    JO, JNO, JP, JNP, JCXZ, JECXZ, JRCXZ, LOOP, LOOPZ, LOOPNZ,
    CALL, RET, ENTER, LEAVE,
    LFENCE, SFENCE, MFENCE, CPUID, SERIALIZE,
    SYSCALL, SYSRET, INT,
    MOVD, MOVQ, MOVDQA, MOVDQU, MOVAPS, MOVUPS, MOVAPD, MOVUPD,
    MOVSS, MOVSD, MOVHPS, MOVLPS, MOVHLPS, MOVLHPS,
    PADDB, PADDW, PADDD, PADDQ, PSUBB, PSUBW, PSUBD, PSUBQ,
    PMULLW, PMULLD, PMULHW, PMULUDQ,
    PAND, PANDN, POR, PXOR,
    PSLLW, PSLLD, PSLLQ, PSRLW, PSRLD, PSRLQ, PSRAW, PSRAD,
    PCMPEQB, PCMPEQW, PCMPEQD, PCMPGTB, PCMPGTW, PCMPGTD,
    PSHUFB, PSHUFD, PSHUFLW, PSHUFHW, PUNPCKLBW, PUNPCKHBW,
    PUNPCKLWD, PUNPCKHWD, PUNPCKLDQ, PUNPCKHDQ, PUNPCKLQDQ, PUNPCKHQDQ,
    ADDSS, ADDSD, ADDPS, ADDPD, SUBSS, SUBSD, SUBPS, SUBPD,
    MULSS, MULSD, MULPS, MULPD, DIVSS, DIVSD, DIVPS, DIVPD,
    XORPS, XORPD, ANDPS, ANDPD, ORPS, ORPD,
    AESENC, AESENCLAST, AESDEC, AESDECLAST, AESKEYGENASSIST, AESIMC,
    PCLMULQDQ, SHA1RNDS4, SHA1NEXTE, SHA1MSG1, SHA1MSG2,
    SHA256RNDS2, SHA256MSG1, SHA256MSG2,
    SETZ, SETNZ, SETL, SETGE, SETLE, SETG, SETB, SETAE, SETBE, SETA,
    SETS, SETNS, SETO, SETNO, SETP, SETNP,
    CBW, CWDE, CDQE, CWD, CDQ, CQO,
    XCHG, CMPXCHG, LOCK_CMPXCHG, XADD,
    BSF, BSR, POPCNT, LZCNT, TZCNT, BSWAP, BT, BTS, BTR, BTC,
    MOVBE, PDEP, PEXT, BZHI, BLSR, BLSI, BLSMSK, ANDN,
    REP_MOVSB, REP_STOSB, REP_CMPSB, REP_SCASB,
    PREFETCHT0, PREFETCHT1, PREFETCHT2, PREFETCHNTA,
    CLC, STC, CMC, CLD, STD,
    RDTSC, RDTSCP, RDPMC,
    UD2,
    VINSERTI128, VEXTRACTI128, VPERM2I128,
    VPSHUFB, VPSHUFD, VPUNPCKLBW, VPUNPCKHBW,
    VPADDB, VPADDW, VPADDD, VPADDQ,
    VPSUBB, VPSUBW, VPSUBD, VPSUBQ,
    VPAND, VPANDN, VPOR, VPXOR,
    VPSLLW, VPSLLD, VPSLLQ, VPSRLW, VPSRLD, VPSRLQ,
    VMOVDQA, VMOVDQU, VMOVAPS, VMOVUPS,
    UNKNOWN,
}

impl Opcode {
    pub fn to_kind(self) -> InstructionKind {
        match self {
            Self::NOP => InstructionKind::Nop,
            Self::MOV | Self::MOVZX | Self::MOVSX | Self::MOVSXD => InstructionKind::Move,
            Self::LEA => InstructionKind::Lea,
            Self::CMOVZ | Self::CMOVNZ | Self::CMOVL | Self::CMOVGE
            | Self::CMOVLE | Self::CMOVG | Self::CMOVB | Self::CMOVAE
            | Self::CMOVBE | Self::CMOVA | Self::CMOVS | Self::CMOVNS => InstructionKind::ConditionalMove,
            Self::ADD | Self::SUB | Self::ADC | Self::SBB
            | Self::INC | Self::DEC | Self::NEG | Self::NOT => InstructionKind::Arithmetic,
            Self::AND | Self::OR | Self::XOR => InstructionKind::Logic,
            Self::SHL | Self::SHR | Self::SAR | Self::ROL | Self::ROR
            | Self::RCL | Self::RCR | Self::SHLD | Self::SHRD => InstructionKind::Shift,
            Self::MUL | Self::IMUL => InstructionKind::Multiply,
            Self::DIV | Self::IDIV => InstructionKind::Divide,
            Self::PUSH | Self::PUSHF => InstructionKind::Push,
            Self::POP | Self::POPF => InstructionKind::Pop,
            Self::JMP => InstructionKind::UnconditionalBranch,
            Self::JZ | Self::JNZ | Self::JL | Self::JGE | Self::JLE
            | Self::JG | Self::JB | Self::JAE | Self::JBE | Self::JA
            | Self::JS | Self::JNS | Self::JO | Self::JNO | Self::JP
            | Self::JNP | Self::JCXZ | Self::JECXZ | Self::JRCXZ
            | Self::LOOP | Self::LOOPZ | Self::LOOPNZ => InstructionKind::ConditionalBranch,
            Self::CALL => InstructionKind::Call,
            Self::RET | Self::LEAVE => InstructionKind::Return,
            Self::LFENCE | Self::SFENCE | Self::MFENCE | Self::CPUID | Self::SERIALIZE => InstructionKind::Fence,
            Self::SYSCALL | Self::SYSRET | Self::INT => InstructionKind::Syscall,
            Self::TEST | Self::CMP => InstructionKind::Compare,
            Self::SETZ | Self::SETNZ | Self::SETL | Self::SETGE | Self::SETLE
            | Self::SETG | Self::SETB | Self::SETAE | Self::SETBE | Self::SETA
            | Self::SETS | Self::SETNS | Self::SETO | Self::SETNO
            | Self::SETP | Self::SETNP => InstructionKind::SetCondition,
            Self::AESENC | Self::AESENCLAST | Self::AESDEC | Self::AESDECLAST
            | Self::AESKEYGENASSIST | Self::AESIMC | Self::PCLMULQDQ
            | Self::SHA1RNDS4 | Self::SHA1NEXTE | Self::SHA1MSG1 | Self::SHA1MSG2
            | Self::SHA256RNDS2 | Self::SHA256MSG1 | Self::SHA256MSG2 => InstructionKind::Crypto,
            Self::PADDB | Self::PADDW | Self::PADDD | Self::PADDQ
            | Self::PSUBB | Self::PSUBW | Self::PSUBD | Self::PSUBQ
            | Self::PMULLW | Self::PMULLD | Self::PMULHW | Self::PMULUDQ
            | Self::PAND | Self::PANDN | Self::POR | Self::PXOR
            | Self::PSLLW | Self::PSLLD | Self::PSLLQ | Self::PSRLW | Self::PSRLD | Self::PSRLQ
            | Self::PSRAW | Self::PSRAD | Self::PCMPEQB | Self::PCMPEQW
            | Self::PCMPEQD | Self::PCMPGTB | Self::PCMPGTW | Self::PCMPGTD
            | Self::ADDSS | Self::ADDSD | Self::ADDPS | Self::ADDPD
            | Self::SUBSS | Self::SUBSD | Self::SUBPS | Self::SUBPD
            | Self::MULSS | Self::MULSD | Self::MULPS | Self::MULPD
            | Self::DIVSS | Self::DIVSD | Self::DIVPS | Self::DIVPD
            | Self::XORPS | Self::XORPD | Self::ANDPS | Self::ANDPD
            | Self::ORPS | Self::ORPD
            | Self::VPADDB | Self::VPADDW | Self::VPADDD | Self::VPADDQ
            | Self::VPSUBB | Self::VPSUBW | Self::VPSUBD | Self::VPSUBQ
            | Self::VPAND | Self::VPANDN | Self::VPOR | Self::VPXOR
            | Self::VPSLLW | Self::VPSLLD | Self::VPSLLQ
            | Self::VPSRLW | Self::VPSRLD | Self::VPSRLQ => InstructionKind::VectorArith,
            Self::PSHUFB | Self::PSHUFD | Self::PSHUFLW | Self::PSHUFHW
            | Self::PUNPCKLBW | Self::PUNPCKHBW | Self::PUNPCKLWD
            | Self::PUNPCKHWD | Self::PUNPCKLDQ | Self::PUNPCKHDQ
            | Self::PUNPCKLQDQ | Self::PUNPCKHQDQ
            | Self::VPSHUFB | Self::VPSHUFD | Self::VPUNPCKLBW | Self::VPUNPCKHBW
            | Self::VINSERTI128 | Self::VEXTRACTI128 | Self::VPERM2I128 => InstructionKind::VectorShuffle,
            Self::MOVD | Self::MOVQ | Self::MOVDQA | Self::MOVDQU
            | Self::MOVAPS | Self::MOVUPS | Self::MOVAPD | Self::MOVUPD
            | Self::MOVSS | Self::MOVSD | Self::MOVHPS | Self::MOVLPS
            | Self::MOVHLPS | Self::MOVLHPS
            | Self::VMOVDQA | Self::VMOVDQU | Self::VMOVAPS | Self::VMOVUPS => InstructionKind::VectorLoad,
            Self::BSF | Self::BSR | Self::POPCNT | Self::LZCNT | Self::TZCNT
            | Self::BSWAP | Self::BT | Self::BTS | Self::BTR | Self::BTC
            | Self::MOVBE | Self::PDEP | Self::PEXT | Self::BZHI
            | Self::BLSR | Self::BLSI | Self::BLSMSK | Self::ANDN => InstructionKind::BitManip,
            Self::CBW | Self::CWDE | Self::CDQE | Self::CWD | Self::CDQ | Self::CQO => InstructionKind::Convert,
            Self::XCHG | Self::CMPXCHG | Self::LOCK_CMPXCHG | Self::XADD => InstructionKind::Exchange,
            Self::REP_MOVSB | Self::REP_STOSB | Self::REP_CMPSB | Self::REP_SCASB => InstructionKind::StringOp,
            Self::PREFETCHT0 | Self::PREFETCHT1 | Self::PREFETCHT2 | Self::PREFETCHNTA => InstructionKind::Prefetch,
            _ => InstructionKind::Unknown,
        }
    }

    pub fn is_branch(self) -> bool {
        self.to_kind().is_control_flow()
    }

    pub fn is_memory(self) -> bool {
        self.to_kind().is_memory_op()
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Instruction flags for analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstructionFlags {
    pub reads_memory: bool,
    pub writes_memory: bool,
    pub is_atomic: bool,
    pub has_lock_prefix: bool,
    pub has_rep_prefix: bool,
    pub secret_dependent_address: bool,
    pub secret_dependent_branch: bool,
    pub is_speculation_barrier: bool,
    pub modifies_flags: bool,
    pub reads_flags: bool,
    pub is_privileged: bool,
    pub may_fault: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_classification() {
        assert_eq!(Opcode::ADD.to_kind(), InstructionKind::Arithmetic);
        assert_eq!(Opcode::JZ.to_kind(), InstructionKind::ConditionalBranch);
        assert_eq!(Opcode::MOV.to_kind(), InstructionKind::Move);
        assert_eq!(Opcode::AESENC.to_kind(), InstructionKind::Crypto);
        assert_eq!(Opcode::LFENCE.to_kind(), InstructionKind::Fence);
    }

    #[test]
    fn test_instruction_builder() {
        let instr = Instruction::new(VirtualAddress(0x1000), Opcode::ADD, InstructionKind::Arithmetic)
            .with_mnemonic("add")
            .with_length(3);
        assert_eq!(instr.end_address(), VirtualAddress(0x1003));
        assert!(!instr.is_branch());
        assert!(instr.mnemonic == "add");
    }

    #[test]
    fn test_instruction_control_flow() {
        let branch = Instruction::new(VirtualAddress(0), Opcode::JZ, InstructionKind::ConditionalBranch);
        assert!(branch.is_branch());
        let call = Instruction::new(VirtualAddress(0), Opcode::CALL, InstructionKind::Call);
        assert!(call.is_call());
        let ret = Instruction::new(VirtualAddress(0), Opcode::RET, InstructionKind::Return);
        assert!(ret.is_return());
    }
}
