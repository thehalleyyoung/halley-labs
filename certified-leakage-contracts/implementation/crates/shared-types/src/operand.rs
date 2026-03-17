//! Operand types for x86-64 instructions.

use serde::{Deserialize, Serialize};
use crate::register::RegisterId;
use std::fmt;

/// An instruction operand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operand {
    pub kind: OperandKind,
    pub size_bits: u32,
    pub is_read: bool,
    pub is_write: bool,
}

impl Operand {
    pub fn reg(id: RegisterId, size: u32, read: bool, write: bool) -> Self {
        Self {
            kind: OperandKind::Register(id),
            size_bits: size,
            is_read: read,
            is_write: write,
        }
    }

    pub fn imm(value: i64, size: u32) -> Self {
        Self {
            kind: OperandKind::Immediate(ImmediateValue(value)),
            size_bits: size,
            is_read: true,
            is_write: false,
        }
    }

    pub fn mem(mem: MemoryOperand, size: u32, read: bool, write: bool) -> Self {
        Self {
            kind: OperandKind::Memory(mem),
            size_bits: size,
            is_read: read,
            is_write: write,
        }
    }

    pub fn is_register(&self) -> bool {
        matches!(self.kind, OperandKind::Register(_))
    }

    pub fn is_memory(&self) -> bool {
        matches!(self.kind, OperandKind::Memory(_))
    }

    pub fn is_immediate(&self) -> bool {
        matches!(self.kind, OperandKind::Immediate(_))
    }

    pub fn register_id(&self) -> Option<RegisterId> {
        match &self.kind {
            OperandKind::Register(id) => Some(*id),
            _ => None,
        }
    }

    pub fn memory_operand(&self) -> Option<&MemoryOperand> {
        match &self.kind {
            OperandKind::Memory(m) => Some(m),
            _ => None,
        }
    }

    pub fn immediate_value(&self) -> Option<i64> {
        match &self.kind {
            OperandKind::Immediate(v) => Some(v.0),
            _ => None,
        }
    }

    pub fn collect_read_registers(&self, regs: &mut Vec<RegisterId>) {
        if self.is_read {
            match &self.kind {
                OperandKind::Register(id) => regs.push(*id),
                OperandKind::Memory(m) => {
                    if let Some(base) = m.base {
                        regs.push(base);
                    }
                    if let Some(index) = m.index {
                        regs.push(index);
                    }
                }
                _ => {}
            }
        }
    }

    pub fn collect_write_registers(&self, regs: &mut Vec<RegisterId>) {
        if self.is_write {
            if let OperandKind::Register(id) = &self.kind {
                regs.push(*id);
            }
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            OperandKind::Register(r) => write!(f, "{}", r),
            OperandKind::Immediate(v) => write!(f, "0x{:x}", v.0),
            OperandKind::Memory(m) => write!(f, "{}", m),
            OperandKind::RelativeOffset(off) => write!(f, "rip+0x{:x}", off),
        }
    }
}

/// Operand kind discriminant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperandKind {
    Register(RegisterId),
    Immediate(ImmediateValue),
    Memory(MemoryOperand),
    RelativeOffset(i64),
}

/// An immediate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImmediateValue(pub i64);

impl ImmediateValue {
    pub fn as_u64(self) -> u64 {
        self.0 as u64
    }

    pub fn as_i64(self) -> i64 {
        self.0
    }
}

/// Memory addressing operand: [base + index*scale + displacement].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOperand {
    pub base: Option<RegisterId>,
    pub index: Option<RegisterId>,
    pub scale: u8,
    pub displacement: i64,
    pub segment: Option<RegisterId>,
}

impl MemoryOperand {
    pub fn base_only(base: RegisterId) -> Self {
        Self {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: 0,
            segment: None,
        }
    }

    pub fn base_disp(base: RegisterId, disp: i64) -> Self {
        Self {
            base: Some(base),
            index: None,
            scale: 1,
            displacement: disp,
            segment: None,
        }
    }

    pub fn base_index_scale(base: RegisterId, index: RegisterId, scale: u8) -> Self {
        Self {
            base: Some(base),
            index: Some(index),
            scale,
            displacement: 0,
            segment: None,
        }
    }

    pub fn full(base: RegisterId, index: RegisterId, scale: u8, disp: i64) -> Self {
        Self {
            base: Some(base),
            index: Some(index),
            scale,
            displacement: disp,
            segment: None,
        }
    }

    pub fn rip_relative(disp: i64) -> Self {
        Self {
            base: Some(RegisterId::RIP),
            index: None,
            scale: 1,
            displacement: disp,
            segment: None,
        }
    }

    pub fn absolute(addr: i64) -> Self {
        Self {
            base: None,
            index: None,
            scale: 1,
            displacement: addr,
            segment: None,
        }
    }

    pub fn uses_index(&self) -> bool {
        self.index.is_some()
    }

    pub fn is_rip_relative(&self) -> bool {
        self.base == Some(RegisterId::RIP)
    }

    pub fn is_stack_access(&self) -> bool {
        self.base == Some(RegisterId::RSP) || self.base == Some(RegisterId::RBP)
    }

    pub fn involved_registers(&self) -> Vec<RegisterId> {
        let mut regs = Vec::new();
        if let Some(b) = self.base {
            regs.push(b);
        }
        if let Some(i) = self.index {
            regs.push(i);
        }
        regs
    }
}

impl fmt::Display for MemoryOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let mut parts = Vec::new();
        if let Some(base) = self.base {
            parts.push(format!("{}", base));
        }
        if let Some(index) = self.index {
            if self.scale > 1 {
                parts.push(format!("{}*{}", index, self.scale));
            } else {
                parts.push(format!("{}", index));
            }
        }
        if self.displacement != 0 {
            if self.displacement > 0 {
                parts.push(format!("0x{:x}", self.displacement));
            } else {
                parts.push(format!("-0x{:x}", -self.displacement));
            }
        }
        write!(f, "{}]", parts.join("+"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operand_types() {
        let reg = Operand::reg(RegisterId::RAX, 64, true, false);
        assert!(reg.is_register());
        assert_eq!(reg.register_id(), Some(RegisterId::RAX));

        let imm = Operand::imm(42, 32);
        assert!(imm.is_immediate());
        assert_eq!(imm.immediate_value(), Some(42));

        let mem = Operand::mem(MemoryOperand::base_disp(RegisterId::RBP, -8), 64, true, false);
        assert!(mem.is_memory());
    }

    #[test]
    fn test_memory_operand_display() {
        let m = MemoryOperand::full(RegisterId::RBP, RegisterId::RCX, 4, 0x10);
        let s = format!("{}", m);
        assert!(s.contains("rbp"));
        assert!(s.contains("rcx"));
    }

    #[test]
    fn test_stack_access() {
        let m = MemoryOperand::base_disp(RegisterId::RSP, -8);
        assert!(m.is_stack_access());
        let m2 = MemoryOperand::base_only(RegisterId::RAX);
        assert!(!m2.is_stack_access());
    }
}
