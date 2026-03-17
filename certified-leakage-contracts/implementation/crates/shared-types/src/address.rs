//! Address types for virtual and physical addresses, cache lines, sets, and tags.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub, BitAnd, Shr};

/// A virtual address in the program's address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VirtualAddress(pub u64);

impl VirtualAddress {
    pub const ZERO: Self = Self(0);

    pub fn new(addr: u64) -> Self {
        Self(addr)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn offset(self, off: i64) -> Self {
        if off >= 0 {
            Self(self.0.wrapping_add(off as u64))
        } else {
            Self(self.0.wrapping_sub((-off) as u64))
        }
    }

    pub fn page_number(self) -> u64 {
        self.0 >> 12
    }

    pub fn page_offset(self) -> u64 {
        self.0 & 0xFFF
    }

    pub fn cache_line(self, line_size_bits: u32) -> CacheLine {
        CacheLine(self.0 >> line_size_bits)
    }

    pub fn cache_set(self, config: &super::cache_config::CacheGeometry) -> CacheSet {
        let line_addr = self.0 >> config.line_size_bits;
        let set_index = line_addr & ((1u64 << config.set_index_bits) - 1);
        CacheSet(set_index as u32)
    }

    pub fn cache_tag(self, config: &super::cache_config::CacheGeometry) -> CacheTag {
        let tag_shift = config.line_size_bits + config.set_index_bits;
        CacheTag(self.0 >> tag_shift)
    }

    pub fn aligned_down(self, alignment: u64) -> Self {
        Self(self.0 & !(alignment - 1))
    }

    pub fn aligned_up(self, alignment: u64) -> Self {
        Self((self.0 + alignment - 1) & !(alignment - 1))
    }

    pub fn is_aligned(self, alignment: u64) -> bool {
        self.0 & (alignment - 1) == 0
    }

    pub fn distance_to(self, other: Self) -> u64 {
        if other.0 >= self.0 {
            other.0 - self.0
        } else {
            self.0 - other.0
        }
    }
}

impl fmt::Display for VirtualAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:016x}", self.0)
    }
}

impl Add<u64> for VirtualAddress {
    type Output = Self;
    fn add(self, rhs: u64) -> Self {
        Self(self.0.wrapping_add(rhs))
    }
}

impl Sub<u64> for VirtualAddress {
    type Output = Self;
    fn sub(self, rhs: u64) -> Self {
        Self(self.0.wrapping_sub(rhs))
    }
}

impl Sub for VirtualAddress {
    type Output = u64;
    fn sub(self, rhs: Self) -> u64 {
        self.0.wrapping_sub(rhs.0)
    }
}

/// A physical address (post-translation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PhysicalAddress(pub u64);

impl PhysicalAddress {
    pub fn new(addr: u64) -> Self {
        Self(addr)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for PhysicalAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "phys:0x{:016x}", self.0)
    }
}

/// An address range [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AddressRange {
    pub start: VirtualAddress,
    pub end: VirtualAddress,
}

impl AddressRange {
    pub fn new(start: VirtualAddress, end: VirtualAddress) -> Self {
        debug_assert!(start.0 <= end.0, "Invalid address range");
        Self { start, end }
    }

    pub fn from_start_len(start: VirtualAddress, len: u64) -> Self {
        Self {
            start,
            end: VirtualAddress(start.0 + len),
        }
    }

    pub fn len(&self) -> u64 {
        self.end.0 - self.start.0
    }

    pub fn is_empty(&self) -> bool {
        self.start.0 >= self.end.0
    }

    pub fn contains(&self, addr: VirtualAddress) -> bool {
        addr.0 >= self.start.0 && addr.0 < self.end.0
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let start = std::cmp::max(self.start.0, other.start.0);
        let end = std::cmp::min(self.end.0, other.end.0);
        if start < end {
            Some(Self {
                start: VirtualAddress(start),
                end: VirtualAddress(end),
            })
        } else {
            None
        }
    }

    pub fn union_if_adjacent(&self, other: &Self) -> Option<Self> {
        if self.overlaps(other) || self.end.0 == other.start.0 || other.end.0 == self.start.0 {
            Some(Self {
                start: VirtualAddress(std::cmp::min(self.start.0, other.start.0)),
                end: VirtualAddress(std::cmp::max(self.end.0, other.end.0)),
            })
        } else {
            None
        }
    }

    pub fn cache_lines(&self, line_size_bits: u32) -> impl Iterator<Item = CacheLine> {
        let line_size = 1u64 << line_size_bits;
        let first = self.start.0 >> line_size_bits;
        let last = (self.end.0.saturating_sub(1)) >> line_size_bits;
        (first..=last).map(CacheLine)
    }
}

impl fmt::Display for AddressRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

/// A cache line address (address >> line_size_bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheLine(pub u64);

impl CacheLine {
    pub fn new(line: u64) -> Self {
        Self(line)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn to_virtual_address(self, line_size_bits: u32) -> VirtualAddress {
        VirtualAddress(self.0 << line_size_bits)
    }
}

impl fmt::Display for CacheLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line:{}", self.0)
    }
}

/// A cache set index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheSet(pub u32);

impl CacheSet {
    pub fn new(set: u32) -> Self {
        Self(set)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for CacheSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "set:{}", self.0)
    }
}

/// A cache tag (high bits of address after set index and line offset are removed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheTag(pub u64);

impl CacheTag {
    pub fn new(tag: u64) -> Self {
        Self(tag)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CacheTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tag:0x{:x}", self.0)
    }
}

/// Decomposed cache address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheAddress {
    pub tag: CacheTag,
    pub set: CacheSet,
    pub line_offset: u32,
}

impl CacheAddress {
    pub fn from_virtual(addr: VirtualAddress, geom: &super::cache_config::CacheGeometry) -> Self {
        let line_offset = (addr.0 & ((1u64 << geom.line_size_bits) - 1)) as u32;
        let set_index = (addr.0 >> geom.line_size_bits) & ((1u64 << geom.set_index_bits) - 1);
        let tag = addr.0 >> (geom.line_size_bits + geom.set_index_bits);
        Self {
            tag: CacheTag(tag),
            set: CacheSet(set_index as u32),
            line_offset,
        }
    }

    pub fn to_virtual(&self, geom: &super::cache_config::CacheGeometry) -> VirtualAddress {
        let addr = (self.tag.0 << (geom.line_size_bits + geom.set_index_bits))
            | ((self.set.0 as u64) << geom.line_size_bits)
            | (self.line_offset as u64);
        VirtualAddress(addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_config::CacheGeometry;

    #[test]
    fn test_virtual_address_basic() {
        let addr = VirtualAddress::new(0x1000);
        assert_eq!(addr.as_u64(), 0x1000);
        assert_eq!(addr.page_number(), 1);
        assert_eq!(addr.page_offset(), 0);
    }

    #[test]
    fn test_virtual_address_offset() {
        let addr = VirtualAddress::new(0x1000);
        assert_eq!(addr.offset(0x100), VirtualAddress::new(0x1100));
        assert_eq!(addr.offset(-0x100), VirtualAddress::new(0x0F00));
    }

    #[test]
    fn test_virtual_address_alignment() {
        let addr = VirtualAddress::new(0x1234);
        assert!(!addr.is_aligned(0x1000));
        assert_eq!(addr.aligned_down(0x1000), VirtualAddress::new(0x1000));
        assert_eq!(addr.aligned_up(0x1000), VirtualAddress::new(0x2000));
    }

    #[test]
    fn test_address_range() {
        let range = AddressRange::new(VirtualAddress(0x1000), VirtualAddress(0x2000));
        assert_eq!(range.len(), 0x1000);
        assert!(range.contains(VirtualAddress(0x1500)));
        assert!(!range.contains(VirtualAddress(0x2000)));
        assert!(!range.contains(VirtualAddress(0x0500)));
    }

    #[test]
    fn test_address_range_overlap() {
        let r1 = AddressRange::new(VirtualAddress(0x1000), VirtualAddress(0x2000));
        let r2 = AddressRange::new(VirtualAddress(0x1800), VirtualAddress(0x2800));
        assert!(r1.overlaps(&r2));
        let intersection = r1.intersection(&r2).unwrap();
        assert_eq!(intersection.start, VirtualAddress(0x1800));
        assert_eq!(intersection.end, VirtualAddress(0x2000));
    }

    #[test]
    fn test_cache_address_decomposition() {
        let geom = CacheGeometry {
            line_size_bits: 6,
            set_index_bits: 6,
            num_ways: 8,
            num_sets: 64,
            total_size: 32768,
        };
        let addr = VirtualAddress::new(0x12345678);
        let cache_addr = CacheAddress::from_virtual(addr, &geom);
        let reconstructed = cache_addr.to_virtual(&geom);
        assert_eq!(addr, reconstructed);
    }

    #[test]
    fn test_cache_set_computation() {
        let geom = CacheGeometry {
            line_size_bits: 6,
            set_index_bits: 6,
            num_ways: 8,
            num_sets: 64,
            total_size: 32768,
        };
        let addr = VirtualAddress::new(0x1000);
        let set = addr.cache_set(&geom);
        assert!(set.0 < 64);
    }

    #[test]
    fn test_cache_lines_iterator() {
        let range = AddressRange::new(VirtualAddress(0x0), VirtualAddress(0x100));
        let lines: Vec<_> = range.cache_lines(6).collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], CacheLine(0));
        assert_eq!(lines[3], CacheLine(3));
    }
}
