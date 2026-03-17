//! Cache configuration and geometry types.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Complete cache configuration including all levels and policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1d: CacheLevel,
    pub l1i: Option<CacheLevel>,
    pub l2: Option<CacheLevel>,
    pub l3: Option<CacheLevel>,
    pub line_size: u32,
    pub speculation_window: u32,
    pub prefetch_enabled: bool,
}

impl CacheConfig {
    pub fn default_x86_64() -> Self {
        Self {
            l1d: CacheLevel {
                geometry: CacheGeometry::l1_default(),
                replacement: ReplacementPolicy::LRU,
                latency_cycles: 4,
                is_inclusive: false,
                is_shared: false,
            },
            l1i: Some(CacheLevel {
                geometry: CacheGeometry {
                    line_size_bits: 6,
                    set_index_bits: 6,
                    num_ways: 8,
                    num_sets: 64,
                    total_size: 32768,
                },
                replacement: ReplacementPolicy::LRU,
                latency_cycles: 4,
                is_inclusive: false,
                is_shared: false,
            }),
            l2: Some(CacheLevel {
                geometry: CacheGeometry {
                    line_size_bits: 6,
                    set_index_bits: 10,
                    num_ways: 4,
                    num_sets: 1024,
                    total_size: 262144,
                },
                replacement: ReplacementPolicy::LRU,
                latency_cycles: 12,
                is_inclusive: true,
                is_shared: false,
            }),
            l3: Some(CacheLevel {
                geometry: CacheGeometry {
                    line_size_bits: 6,
                    set_index_bits: 13,
                    num_ways: 16,
                    num_sets: 8192,
                    total_size: 8388608,
                },
                replacement: ReplacementPolicy::PLRU,
                latency_cycles: 40,
                is_inclusive: true,
                is_shared: true,
            }),
            line_size: 64,
            speculation_window: 200,
            prefetch_enabled: false,
        }
    }

    pub fn l1_only() -> Self {
        Self {
            l1d: CacheLevel {
                geometry: CacheGeometry::l1_default(),
                replacement: ReplacementPolicy::LRU,
                latency_cycles: 4,
                is_inclusive: false,
                is_shared: false,
            },
            l1i: None,
            l2: None,
            l3: None,
            line_size: 64,
            speculation_window: 200,
            prefetch_enabled: false,
        }
    }

    pub fn primary_geometry(&self) -> &CacheGeometry {
        &self.l1d.geometry
    }

    pub fn speculation_window(&self) -> u32 {
        self.speculation_window
    }

    pub fn total_cache_sets(&self) -> u32 {
        self.l1d.geometry.num_sets
    }

    pub fn associativity(&self) -> u32 {
        self.l1d.geometry.num_ways
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::default_x86_64()
    }
}

/// Configuration for a single cache level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevel {
    pub geometry: CacheGeometry,
    pub replacement: ReplacementPolicy,
    pub latency_cycles: u32,
    pub is_inclusive: bool,
    pub is_shared: bool,
}

/// Cache geometry parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheGeometry {
    pub line_size_bits: u32,
    pub set_index_bits: u32,
    pub num_ways: u32,
    pub num_sets: u32,
    pub total_size: u64,
}

impl CacheGeometry {
    pub fn l1_default() -> Self {
        Self {
            line_size_bits: 6,
            set_index_bits: 6,
            num_ways: 8,
            num_sets: 64,
            total_size: 32768,
        }
    }

    pub fn new(line_size_bits: u32, set_index_bits: u32, num_ways: u32) -> Self {
        let num_sets = 1u32 << set_index_bits;
        let line_size = 1u64 << line_size_bits;
        let total_size = line_size * (num_sets as u64) * (num_ways as u64);
        Self {
            line_size_bits,
            set_index_bits,
            num_ways,
            num_sets,
            total_size,
        }
    }

    pub fn line_size(&self) -> u64 {
        1u64 << self.line_size_bits
    }

    pub fn set_mask(&self) -> u64 {
        ((1u64 << self.set_index_bits) - 1) << self.line_size_bits
    }

    pub fn tag_shift(&self) -> u32 {
        self.line_size_bits + self.set_index_bits
    }

    pub fn max_tag_value(&self) -> u64 {
        (1u64 << (64 - self.tag_shift())) - 1
    }

    pub fn total_lines(&self) -> u32 {
        self.num_sets * self.num_ways
    }

    pub fn lattice_height(&self) -> u32 {
        self.num_sets * self.num_ways
    }

    pub fn validate(&self) -> bool {
        self.line_size_bits > 0
            && self.set_index_bits > 0
            && self.num_ways > 0
            && self.num_sets == (1u32 << self.set_index_bits)
            && self.total_size == self.line_size() * (self.num_sets as u64) * (self.num_ways as u64)
    }
}

impl fmt::Display for CacheGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}KB {}-way {}-set (line={}B)",
            self.total_size / 1024,
            self.num_ways,
            self.num_sets,
            self.line_size()
        )
    }
}

/// Cache replacement policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReplacementPolicy {
    LRU,
    PLRU,
    FIFO,
    Random,
    MRU,
    LIP,
    BIP,
}

impl ReplacementPolicy {
    pub fn is_deterministic(&self) -> bool {
        matches!(self, Self::LRU | Self::FIFO | Self::MRU)
    }

    pub fn is_lru_like(&self) -> bool {
        matches!(self, Self::LRU | Self::PLRU | Self::LIP | Self::BIP)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::LRU => "LRU",
            Self::PLRU => "Pseudo-LRU",
            Self::FIFO => "FIFO",
            Self::Random => "Random",
            Self::MRU => "MRU",
            Self::LIP => "LRU Insertion Policy",
            Self::BIP => "Bimodal Insertion Policy",
        }
    }
}

impl fmt::Display for ReplacementPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default_x86_64();
        assert_eq!(config.l1d.geometry.num_sets, 64);
        assert_eq!(config.l1d.geometry.num_ways, 8);
        assert_eq!(config.l1d.geometry.total_size, 32768);
    }

    #[test]
    fn test_geometry_new() {
        let geom = CacheGeometry::new(6, 6, 8);
        assert_eq!(geom.num_sets, 64);
        assert_eq!(geom.line_size(), 64);
        assert_eq!(geom.total_size, 64 * 64 * 8);
        assert!(geom.validate());
    }

    #[test]
    fn test_geometry_display() {
        let geom = CacheGeometry::l1_default();
        let s = format!("{}", geom);
        assert!(s.contains("32KB"));
        assert!(s.contains("8-way"));
    }

    #[test]
    fn test_replacement_policy() {
        assert!(ReplacementPolicy::LRU.is_deterministic());
        assert!(!ReplacementPolicy::Random.is_deterministic());
        assert!(ReplacementPolicy::PLRU.is_lru_like());
    }

    #[test]
    fn test_lattice_height() {
        let geom = CacheGeometry::l1_default();
        assert_eq!(geom.lattice_height(), 512);
    }
}
