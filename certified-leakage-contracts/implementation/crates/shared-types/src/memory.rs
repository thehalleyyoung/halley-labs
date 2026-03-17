//! Memory access types and memory map representation.

use serde::{Deserialize, Serialize};
use crate::address::{VirtualAddress, AddressRange};
use crate::register::RegisterId;
use std::fmt;
use std::collections::BTreeMap;

/// Kind of memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryAccessKind {
    Read,
    Write,
    ReadWrite,
    Execute,
    Prefetch,
}

/// A single memory access record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccess {
    pub address: VirtualAddress,
    pub size: u32,
    pub kind: MemoryAccessKind,
    pub is_secret_dependent: bool,
    pub is_speculative: bool,
    pub speculation_depth: u32,
    pub instruction_address: VirtualAddress,
    pub base_register: Option<RegisterId>,
    pub index_register: Option<RegisterId>,
}

impl MemoryAccess {
    pub fn read(addr: VirtualAddress, size: u32) -> Self {
        Self {
            address: addr,
            size,
            kind: MemoryAccessKind::Read,
            is_secret_dependent: false,
            is_speculative: false,
            speculation_depth: 0,
            instruction_address: VirtualAddress::ZERO,
            base_register: None,
            index_register: None,
        }
    }

    pub fn write(addr: VirtualAddress, size: u32) -> Self {
        Self {
            address: addr,
            size,
            kind: MemoryAccessKind::Write,
            is_secret_dependent: false,
            is_speculative: false,
            speculation_depth: 0,
            instruction_address: VirtualAddress::ZERO,
            base_register: None,
            index_register: None,
        }
    }

    pub fn with_secret(mut self) -> Self {
        self.is_secret_dependent = true;
        self
    }

    pub fn with_speculative(mut self, depth: u32) -> Self {
        self.is_speculative = true;
        self.speculation_depth = depth;
        self
    }

    pub fn end_address(&self) -> VirtualAddress {
        VirtualAddress(self.address.0 + self.size as u64)
    }

    pub fn address_range(&self) -> AddressRange {
        AddressRange::from_start_len(self.address, self.size as u64)
    }

    pub fn crosses_cache_line(&self, line_size_bits: u32) -> bool {
        let line_size = 1u64 << line_size_bits;
        let start_line = self.address.0 >> line_size_bits;
        let end_line = (self.address.0 + self.size as u64 - 1) >> line_size_bits;
        start_line != end_line
    }

    pub fn cache_lines_touched(&self, line_size_bits: u32) -> u32 {
        let line_size = 1u64 << line_size_bits;
        let start_line = self.address.0 / line_size;
        let end_line = (self.address.0 + self.size as u64 - 1) / line_size;
        (end_line - start_line + 1) as u32
    }
}

impl fmt::Display for MemoryAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            MemoryAccessKind::Read => "R",
            MemoryAccessKind::Write => "W",
            MemoryAccessKind::ReadWrite => "RW",
            MemoryAccessKind::Execute => "X",
            MemoryAccessKind::Prefetch => "PF",
        };
        write!(f, "{}({}, {}B", kind, self.address, self.size)?;
        if self.is_secret_dependent {
            write!(f, ", SECRET")?;
        }
        if self.is_speculative {
            write!(f, ", SPEC({})", self.speculation_depth)?;
        }
        write!(f, ")")
    }
}

/// Memory region permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryPermission {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl MemoryPermission {
    pub const RO: Self = Self { read: true, write: false, execute: false };
    pub const RW: Self = Self { read: true, write: true, execute: false };
    pub const RX: Self = Self { read: true, write: false, execute: true };
    pub const RWX: Self = Self { read: true, write: true, execute: true };
    pub const NONE: Self = Self { read: false, write: false, execute: false };

    pub fn allows(&self, kind: MemoryAccessKind) -> bool {
        match kind {
            MemoryAccessKind::Read | MemoryAccessKind::Prefetch => self.read,
            MemoryAccessKind::Write => self.write,
            MemoryAccessKind::ReadWrite => self.read && self.write,
            MemoryAccessKind::Execute => self.execute,
        }
    }
}

impl fmt::Display for MemoryPermission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}",
            if self.read { "r" } else { "-" },
            if self.write { "w" } else { "-" },
            if self.execute { "x" } else { "-" },
        )
    }
}

/// A named memory region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub name: String,
    pub range: AddressRange,
    pub permissions: MemoryPermission,
    pub is_secret: bool,
    pub is_stack: bool,
    pub is_heap: bool,
    pub content: Option<Vec<u8>>,
}

impl MemoryRegion {
    pub fn new(name: &str, start: u64, size: u64, perms: MemoryPermission) -> Self {
        Self {
            name: name.to_string(),
            range: AddressRange::from_start_len(VirtualAddress(start), size),
            permissions: perms,
            is_secret: false,
            is_stack: false,
            is_heap: false,
            content: None,
        }
    }

    pub fn secret(mut self) -> Self {
        self.is_secret = true;
        self
    }

    pub fn stack(mut self) -> Self {
        self.is_stack = true;
        self
    }

    pub fn contains(&self, addr: VirtualAddress) -> bool {
        self.range.contains(addr)
    }

    pub fn read_byte(&self, addr: VirtualAddress) -> Option<u8> {
        if !self.contains(addr) {
            return None;
        }
        self.content.as_ref().map(|c| {
            let offset = (addr.0 - self.range.start.0) as usize;
            if offset < c.len() { c[offset] } else { 0 }
        })
    }
}

/// Memory map of program regions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMap {
    pub regions: Vec<MemoryRegion>,
    secret_ranges: Vec<AddressRange>,
}

impl MemoryMap {
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            secret_ranges: Vec::new(),
        }
    }

    pub fn add_region(&mut self, region: MemoryRegion) {
        if region.is_secret {
            self.secret_ranges.push(region.range);
        }
        self.regions.push(region);
    }

    pub fn find_region(&self, addr: VirtualAddress) -> Option<&MemoryRegion> {
        self.regions.iter().find(|r| r.contains(addr))
    }

    pub fn is_secret_address(&self, addr: VirtualAddress) -> bool {
        self.secret_ranges.iter().any(|r| r.contains(addr))
    }

    pub fn is_readable(&self, addr: VirtualAddress) -> bool {
        self.find_region(addr).map_or(false, |r| r.permissions.read)
    }

    pub fn is_writable(&self, addr: VirtualAddress) -> bool {
        self.find_region(addr).map_or(false, |r| r.permissions.write)
    }

    pub fn is_executable(&self, addr: VirtualAddress) -> bool {
        self.find_region(addr).map_or(false, |r| r.permissions.execute)
    }

    pub fn secret_regions(&self) -> impl Iterator<Item = &MemoryRegion> {
        self.regions.iter().filter(|r| r.is_secret)
    }

    pub fn total_secret_size(&self) -> u64 {
        self.secret_ranges.iter().map(|r| r.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_access() {
        let access = MemoryAccess::read(VirtualAddress(0x1000), 8);
        assert_eq!(access.end_address(), VirtualAddress(0x1008));
        assert!(!access.crosses_cache_line(6)); // 64-byte lines
    }

    #[test]
    fn test_memory_access_cross_line() {
        let access = MemoryAccess::read(VirtualAddress(0x3E), 8);
        assert!(access.crosses_cache_line(6));
        assert_eq!(access.cache_lines_touched(6), 2);
    }

    #[test]
    fn test_memory_map() {
        let mut map = MemoryMap::new();
        map.add_region(MemoryRegion::new("code", 0x1000, 0x1000, MemoryPermission::RX));
        map.add_region(MemoryRegion::new("key", 0x3000, 0x100, MemoryPermission::RO).secret());

        assert!(map.is_executable(VirtualAddress(0x1500)));
        assert!(!map.is_writable(VirtualAddress(0x1500)));
        assert!(map.is_secret_address(VirtualAddress(0x3050)));
        assert!(!map.is_secret_address(VirtualAddress(0x1500)));
    }

    #[test]
    fn test_permissions() {
        assert!(MemoryPermission::RO.allows(MemoryAccessKind::Read));
        assert!(!MemoryPermission::RO.allows(MemoryAccessKind::Write));
        assert!(MemoryPermission::RWX.allows(MemoryAccessKind::Execute));
    }
}
