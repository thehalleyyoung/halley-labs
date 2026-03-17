//! Register definitions for x86-64 architecture.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegisterId(pub u16);

impl RegisterId {
    pub const RAX: Self = Self(0);
    pub const RBX: Self = Self(1);
    pub const RCX: Self = Self(2);
    pub const RDX: Self = Self(3);
    pub const RSI: Self = Self(4);
    pub const RDI: Self = Self(5);
    pub const RBP: Self = Self(6);
    pub const RSP: Self = Self(7);
    pub const R8: Self = Self(8);
    pub const R9: Self = Self(9);
    pub const R10: Self = Self(10);
    pub const R11: Self = Self(11);
    pub const R12: Self = Self(12);
    pub const R13: Self = Self(13);
    pub const R14: Self = Self(14);
    pub const R15: Self = Self(15);
    pub const RIP: Self = Self(16);
    pub const RFLAGS: Self = Self(17);
    pub const XMM0: Self = Self(32);
    pub const XMM1: Self = Self(33);
    pub const XMM2: Self = Self(34);
    pub const XMM3: Self = Self(35);
    pub const XMM4: Self = Self(36);
    pub const XMM5: Self = Self(37);
    pub const XMM6: Self = Self(38);
    pub const XMM7: Self = Self(39);
    pub const XMM8: Self = Self(40);
    pub const XMM9: Self = Self(41);
    pub const XMM10: Self = Self(42);
    pub const XMM11: Self = Self(43);
    pub const XMM12: Self = Self(44);
    pub const XMM13: Self = Self(45);
    pub const XMM14: Self = Self(46);
    pub const XMM15: Self = Self(47);
    pub const YMM0: Self = Self(64);
    pub const YMM1: Self = Self(65);
    pub const YMM2: Self = Self(66);
    pub const YMM3: Self = Self(67);
    pub const YMM4: Self = Self(68);
    pub const YMM5: Self = Self(69);
    pub const YMM6: Self = Self(70);
    pub const YMM7: Self = Self(71);
    pub const FS_BASE: Self = Self(96);
    pub const GS_BASE: Self = Self(97);
    pub const CR0: Self = Self(128);
    pub const CR3: Self = Self(131);
    pub const CR4: Self = Self(132);
    pub const FLAGS_CF: Self = Self(200);
    pub const FLAGS_ZF: Self = Self(201);
    pub const FLAGS_SF: Self = Self(202);
    pub const FLAGS_OF: Self = Self(203);
    pub const FLAGS_PF: Self = Self(204);
    pub const FLAGS_AF: Self = Self(205);
    pub const INVALID: Self = Self(0xFFFF);

    pub fn new(id: u16) -> Self {
        Self(id)
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }

    pub fn class(self) -> RegisterClass {
        match self.0 {
            0..=17 => RegisterClass::GeneralPurpose,
            32..=47 => RegisterClass::Xmm,
            64..=79 => RegisterClass::Ymm,
            96..=97 => RegisterClass::Segment,
            128..=140 => RegisterClass::Control,
            200..=210 => RegisterClass::Flags,
            _ => RegisterClass::Unknown,
        }
    }

    pub fn bit_width(self) -> u32 {
        match self.class() {
            RegisterClass::GeneralPurpose => 64,
            RegisterClass::Xmm => 128,
            RegisterClass::Ymm => 256,
            RegisterClass::Segment => 64,
            RegisterClass::Control => 64,
            RegisterClass::Flags => 1,
            RegisterClass::Unknown => 64,
        }
    }

    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "rax", 1 => "rbx", 2 => "rcx", 3 => "rdx",
            4 => "rsi", 5 => "rdi", 6 => "rbp", 7 => "rsp",
            8 => "r8", 9 => "r9", 10 => "r10", 11 => "r11",
            12 => "r12", 13 => "r13", 14 => "r14", 15 => "r15",
            16 => "rip", 17 => "rflags",
            32 => "xmm0", 33 => "xmm1", 34 => "xmm2", 35 => "xmm3",
            36 => "xmm4", 37 => "xmm5", 38 => "xmm6", 39 => "xmm7",
            40 => "xmm8", 41 => "xmm9", 42 => "xmm10", 43 => "xmm11",
            44 => "xmm12", 45 => "xmm13", 46 => "xmm14", 47 => "xmm15",
            64 => "ymm0", 65 => "ymm1", 66 => "ymm2", 67 => "ymm3",
            68 => "ymm4", 69 => "ymm5", 70 => "ymm6", 71 => "ymm7",
            96 => "fs_base", 97 => "gs_base",
            128 => "cr0", 131 => "cr3", 132 => "cr4",
            200 => "cf", 201 => "zf", 202 => "sf", 203 => "of",
            204 => "pf", 205 => "af",
            _ => "unknown",
        }
    }

    pub fn is_gpr(self) -> bool {
        self.0 <= 15
    }

    pub fn is_xmm(self) -> bool {
        (32..=47).contains(&self.0)
    }

    pub fn is_ymm(self) -> bool {
        (64..=79).contains(&self.0)
    }

    pub fn is_flag(self) -> bool {
        (200..=210).contains(&self.0)
    }

    pub fn is_stack_pointer(self) -> bool {
        self == Self::RSP
    }

    pub fn is_frame_pointer(self) -> bool {
        self == Self::RBP
    }

    pub fn is_instruction_pointer(self) -> bool {
        self == Self::RIP
    }

    pub fn gpr_index(self) -> Option<usize> {
        if self.is_gpr() {
            Some(self.0 as usize)
        } else {
            None
        }
    }

    pub fn all_gprs() -> &'static [RegisterId] {
        &[
            Self::RAX, Self::RBX, Self::RCX, Self::RDX,
            Self::RSI, Self::RDI, Self::RBP, Self::RSP,
            Self::R8, Self::R9, Self::R10, Self::R11,
            Self::R12, Self::R13, Self::R14, Self::R15,
        ]
    }

    pub fn caller_saved() -> &'static [RegisterId] {
        &[
            Self::RAX, Self::RCX, Self::RDX,
            Self::RSI, Self::RDI,
            Self::R8, Self::R9, Self::R10, Self::R11,
        ]
    }

    pub fn callee_saved() -> &'static [RegisterId] {
        &[
            Self::RBX, Self::RBP,
            Self::R12, Self::R13, Self::R14, Self::R15,
        ]
    }
}

impl fmt::Display for RegisterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Register classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegisterClass {
    GeneralPurpose,
    Xmm,
    Ymm,
    Segment,
    Control,
    Flags,
    Unknown,
}

/// A named register combining identifier and sub-register access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Register {
    pub id: RegisterId,
    pub offset: u32,
    pub size: u32,
}

impl Register {
    pub fn full(id: RegisterId) -> Self {
        Self {
            id,
            offset: 0,
            size: id.bit_width(),
        }
    }

    pub fn sub(id: RegisterId, offset: u32, size: u32) -> Self {
        Self { id, offset, size }
    }

    pub fn low_byte(id: RegisterId) -> Self {
        Self { id, offset: 0, size: 8 }
    }

    pub fn low_word(id: RegisterId) -> Self {
        Self { id, offset: 0, size: 16 }
    }

    pub fn low_dword(id: RegisterId) -> Self {
        Self { id, offset: 0, size: 32 }
    }

    pub fn is_full_register(&self) -> bool {
        self.offset == 0 && self.size == self.id.bit_width()
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        if self.id != other.id {
            return false;
        }
        let self_end = self.offset + self.size;
        let other_end = other.offset + other.size;
        self.offset < other_end && other.offset < self_end
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.id == other.id
            && self.offset <= other.offset
            && self.offset + self.size >= other.offset + other.size
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_full_register() {
            write!(f, "{}", self.id)
        } else {
            write!(f, "{}[{}:{}]", self.id, self.offset, self.offset + self.size - 1)
        }
    }
}

/// Complete register file state.
#[derive(Debug, Clone)]
pub struct RegisterFile {
    gprs: [u64; 16],
    rip: u64,
    rflags: u64,
    xmm: [[u64; 2]; 16],
    ymm_high: [[u64; 2]; 8],
    fs_base: u64,
    gs_base: u64,
}

impl RegisterFile {
    pub fn new() -> Self {
        Self {
            gprs: [0; 16],
            rip: 0,
            rflags: 0x202,
            xmm: [[0; 2]; 16],
            ymm_high: [[0; 2]; 8],
            fs_base: 0,
            gs_base: 0,
        }
    }

    pub fn read_gpr(&self, id: RegisterId) -> u64 {
        match id.gpr_index() {
            Some(idx) => self.gprs[idx],
            None if id == RegisterId::RIP => self.rip,
            None if id == RegisterId::RFLAGS => self.rflags,
            _ => 0,
        }
    }

    pub fn write_gpr(&mut self, id: RegisterId, value: u64) {
        match id.gpr_index() {
            Some(idx) => self.gprs[idx] = value,
            None if id == RegisterId::RIP => self.rip = value,
            None if id == RegisterId::RFLAGS => self.rflags = value,
            _ => {}
        }
    }

    pub fn read_xmm(&self, index: usize) -> [u64; 2] {
        if index < 16 {
            self.xmm[index]
        } else {
            [0; 2]
        }
    }

    pub fn write_xmm(&mut self, index: usize, value: [u64; 2]) {
        if index < 16 {
            self.xmm[index] = value;
        }
    }

    pub fn rip(&self) -> u64 {
        self.rip
    }

    pub fn set_rip(&mut self, value: u64) {
        self.rip = value;
    }

    pub fn rsp(&self) -> u64 {
        self.gprs[RegisterId::RSP.0 as usize]
    }

    pub fn set_rsp(&mut self, value: u64) {
        self.gprs[RegisterId::RSP.0 as usize] = value;
    }

    pub fn flag_cf(&self) -> bool {
        self.rflags & 1 != 0
    }

    pub fn flag_zf(&self) -> bool {
        self.rflags & (1 << 6) != 0
    }

    pub fn flag_sf(&self) -> bool {
        self.rflags & (1 << 7) != 0
    }

    pub fn flag_of(&self) -> bool {
        self.rflags & (1 << 11) != 0
    }

    pub fn set_flag(&mut self, flag: RegisterId, value: bool) {
        let bit = match flag {
            RegisterId::FLAGS_CF => 0,
            RegisterId::FLAGS_ZF => 6,
            RegisterId::FLAGS_SF => 7,
            RegisterId::FLAGS_OF => 11,
            RegisterId::FLAGS_PF => 2,
            RegisterId::FLAGS_AF => 4,
            _ => return,
        };
        if value {
            self.rflags |= 1 << bit;
        } else {
            self.rflags &= !(1 << bit);
        }
    }
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_id_basics() {
        assert_eq!(RegisterId::RAX.name(), "rax");
        assert!(RegisterId::RAX.is_gpr());
        assert!(!RegisterId::XMM0.is_gpr());
        assert!(RegisterId::XMM0.is_xmm());
    }

    #[test]
    fn test_register_classes() {
        assert_eq!(RegisterId::RAX.class(), RegisterClass::GeneralPurpose);
        assert_eq!(RegisterId::XMM0.class(), RegisterClass::Xmm);
        assert_eq!(RegisterId::YMM0.class(), RegisterClass::Ymm);
        assert_eq!(RegisterId::FLAGS_CF.class(), RegisterClass::Flags);
    }

    #[test]
    fn test_register_overlap() {
        let r1 = Register::full(RegisterId::RAX);
        let r2 = Register::low_dword(RegisterId::RAX);
        assert!(r1.overlaps(&r2));
        assert!(r1.contains(&r2));
        assert!(!r2.contains(&r1));
    }

    #[test]
    fn test_register_file() {
        let mut rf = RegisterFile::new();
        rf.write_gpr(RegisterId::RAX, 42);
        assert_eq!(rf.read_gpr(RegisterId::RAX), 42);
        rf.set_flag(RegisterId::FLAGS_ZF, true);
        assert!(rf.flag_zf());
        rf.set_flag(RegisterId::FLAGS_ZF, false);
        assert!(!rf.flag_zf());
    }

    #[test]
    fn test_caller_callee_saved() {
        let caller = RegisterId::caller_saved();
        let callee = RegisterId::callee_saved();
        for r in caller {
            assert!(!callee.contains(r));
        }
    }
}
