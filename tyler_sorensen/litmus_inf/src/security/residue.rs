//! Memory residue channel analysis for GPU security.
//!
//! Detects information leakage through memory residue — data that persists
//! in GPU memory after a kernel finishes execution.

use std::collections::{HashMap, HashSet};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// ResiduePattern
// ---------------------------------------------------------------------------

/// Describes what data persists after kernel completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResiduePattern {
    pub id: String,
    pub memory_region: MemoryRegion,
    pub data_type: ResidueDataType,
    pub size_bytes: usize,
    pub persistence: Persistence,
    pub source_kernel: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryRegion {
    LocalMemory,
    SharedMemory,
    GlobalMemory,
    Registers,
    L1Cache,
    L2Cache,
    TextureCache,
    ConstantCache,
}

impl fmt::Display for MemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryRegion::LocalMemory => write!(f, "local"),
            MemoryRegion::SharedMemory => write!(f, "shared"),
            MemoryRegion::GlobalMemory => write!(f, "global"),
            MemoryRegion::Registers => write!(f, "registers"),
            MemoryRegion::L1Cache => write!(f, "L1"),
            MemoryRegion::L2Cache => write!(f, "L2"),
            MemoryRegion::TextureCache => write!(f, "texture"),
            MemoryRegion::ConstantCache => write!(f, "constant"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidueDataType {
    RawBytes,
    Weights,
    Activations,
    Gradients,
    Keys,
    UserData,
    IntermediateValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Persistence {
    /// Data survives until memory is reallocated.
    UntilReallocation,
    /// Data survives across kernel launches.
    AcrossKernels,
    /// Data survives across process boundaries.
    AcrossProcesses,
    /// Data is cleared by hardware.
    Cleared,
}

impl ResiduePattern {
    pub fn new(id: &str, region: MemoryRegion, data_type: ResidueDataType) -> Self {
        ResiduePattern {
            id: id.to_string(),
            memory_region: region,
            data_type,
            size_bytes: 0,
            persistence: Persistence::UntilReallocation,
            source_kernel: String::new(),
            description: String::new(),
        }
    }

    pub fn with_size(mut self, bytes: usize) -> Self {
        self.size_bytes = bytes;
        self
    }

    pub fn with_persistence(mut self, p: Persistence) -> Self {
        self.persistence = p;
        self
    }

    pub fn with_source(mut self, kernel: &str) -> Self {
        self.source_kernel = kernel.to_string();
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Risk score 0-10 based on region, persistence, and data type.
    pub fn risk_score(&self) -> f64 {
        let region_risk = match self.memory_region {
            MemoryRegion::LocalMemory => 8.0,
            MemoryRegion::SharedMemory => 7.0,
            MemoryRegion::Registers => 6.0,
            MemoryRegion::GlobalMemory => 5.0,
            MemoryRegion::L1Cache => 4.0,
            MemoryRegion::L2Cache => 3.0,
            MemoryRegion::TextureCache => 2.0,
            MemoryRegion::ConstantCache => 1.0,
        };
        let persist_factor = match self.persistence {
            Persistence::AcrossProcesses => 1.0,
            Persistence::AcrossKernels => 0.8,
            Persistence::UntilReallocation => 0.5,
            Persistence::Cleared => 0.1,
        };
        let data_factor = match self.data_type {
            ResidueDataType::Keys => 1.0,
            ResidueDataType::Weights => 0.8,
            ResidueDataType::Gradients => 0.7,
            ResidueDataType::UserData => 0.6,
            ResidueDataType::Activations => 0.5,
            ResidueDataType::IntermediateValues => 0.4,
            ResidueDataType::RawBytes => 0.3,
        };
        region_risk * persist_factor * data_factor
    }
}

// ---------------------------------------------------------------------------
// AllocationPattern
// ---------------------------------------------------------------------------

/// Models GPU memory allocation/deallocation patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    pub allocations: Vec<Allocation>,
    pub deallocations: Vec<Deallocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub id: usize,
    pub address: u64,
    pub size: usize,
    pub region: MemoryRegion,
    pub kernel: String,
    pub timestamp: u64,
    pub zeroed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deallocation {
    pub allocation_id: usize,
    pub timestamp: u64,
    pub wiped: bool,
}

impl AllocationPattern {
    pub fn new() -> Self {
        AllocationPattern { allocations: Vec::new(), deallocations: Vec::new() }
    }

    pub fn add_allocation(
        &mut self, address: u64, size: usize, region: MemoryRegion,
        kernel: &str, timestamp: u64, zeroed: bool,
    ) -> usize {
        let id = self.allocations.len();
        self.allocations.push(Allocation {
            id, address, size, region, kernel: kernel.to_string(),
            timestamp, zeroed,
        });
        id
    }

    pub fn add_deallocation(&mut self, allocation_id: usize, timestamp: u64, wiped: bool) {
        self.deallocations.push(Deallocation { allocation_id, timestamp, wiped });
    }

    /// Find allocations that were deallocated without being wiped.
    pub fn unwiped_deallocations(&self) -> Vec<(&Allocation, &Deallocation)> {
        self.deallocations.iter()
            .filter(|d| !d.wiped)
            .filter_map(|d| {
                self.allocations.get(d.allocation_id).map(|a| (a, d))
            })
            .collect()
    }

    /// Find memory reuse patterns (new allocation overlapping old unwiped one).
    pub fn reuse_patterns(&self) -> Vec<MemoryReuse> {
        let dealloced: HashSet<usize> = self.deallocations.iter()
            .map(|d| d.allocation_id)
            .collect();
        let unwiped: Vec<&Allocation> = self.deallocations.iter()
            .filter(|d| !d.wiped)
            .filter_map(|d| self.allocations.get(d.allocation_id))
            .collect();

        let mut reuses = Vec::new();
        for alloc in &self.allocations {
            if dealloced.contains(&alloc.id) { continue; }
            for &old in &unwiped {
                if alloc.address == old.address && alloc.id != old.id
                    && alloc.timestamp > old.timestamp && !alloc.zeroed
                {
                    reuses.push(MemoryReuse {
                        old_allocation: old.id,
                        new_allocation: alloc.id,
                        overlap_bytes: old.size.min(alloc.size),
                        old_kernel: old.kernel.clone(),
                        new_kernel: alloc.kernel.clone(),
                    });
                }
            }
        }
        reuses
    }

    /// Total allocated memory.
    pub fn total_allocated(&self) -> usize {
        self.allocations.iter().map(|a| a.size).sum()
    }

    /// Number of allocations that were zeroed.
    pub fn zeroed_count(&self) -> usize {
        self.allocations.iter().filter(|a| a.zeroed).count()
    }

    /// Memory utilization: fraction of allocations that are deallocated.
    pub fn deallocation_rate(&self) -> f64 {
        if self.allocations.is_empty() { return 0.0; }
        self.deallocations.len() as f64 / self.allocations.len() as f64
    }
}

/// A detected memory reuse where old data may leak.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReuse {
    pub old_allocation: usize,
    pub new_allocation: usize,
    pub overlap_bytes: usize,
    pub old_kernel: String,
    pub new_kernel: String,
}

// ---------------------------------------------------------------------------
// LeftoverLocalsDetector
// ---------------------------------------------------------------------------

/// Detects the LeftoverLocals vulnerability class:
/// local memory not cleared between kernel invocations.
#[derive(Debug, Clone)]
pub struct LeftoverLocalsDetector {
    pub patterns: Vec<LeftoverLocal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeftoverLocal {
    pub thread_id: usize,
    pub local_var_offset: usize,
    pub size_bytes: usize,
    pub written_by: String,
    pub readable_by: String,
    pub severity: f64,
}

impl LeftoverLocalsDetector {
    pub fn new() -> Self {
        LeftoverLocalsDetector { patterns: Vec::new() }
    }

    /// Analyze kernel pairs for leftover locals.
    pub fn analyze(&mut self, kernel_a: &KernelProfile, kernel_b: &KernelProfile) {
        // Find local variables written by A that overlap with reads by B
        for write in &kernel_a.local_writes {
            for read in &kernel_b.local_reads {
                if write.offset == read.offset && write.size == read.size {
                    self.patterns.push(LeftoverLocal {
                        thread_id: write.thread_id,
                        local_var_offset: write.offset,
                        size_bytes: write.size,
                        written_by: kernel_a.name.clone(),
                        readable_by: kernel_b.name.clone(),
                        severity: self.compute_severity(write, read),
                    });
                }
            }
        }
    }

    fn compute_severity(&self, write: &LocalAccess, read: &LocalAccess) -> f64 {
        let size_factor = (write.size as f64 / 256.0).min(1.0);
        let is_sensitive = if write.is_sensitive { 1.0 } else { 0.5 };
        size_factor * is_sensitive * 10.0
    }

    /// Get all detected patterns sorted by severity.
    pub fn get_results(&self) -> Vec<&LeftoverLocal> {
        let mut results: Vec<&LeftoverLocal> = self.patterns.iter().collect();
        results.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());
        results
    }

    /// Total number of detected patterns.
    pub fn count(&self) -> usize { self.patterns.len() }

    /// Maximum severity found.
    pub fn max_severity(&self) -> f64 {
        self.patterns.iter().map(|p| p.severity).fold(0.0f64, f64::max)
    }
}

/// Profile of a GPU kernel's local memory usage.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    pub name: String,
    pub local_writes: Vec<LocalAccess>,
    pub local_reads: Vec<LocalAccess>,
    pub shared_writes: Vec<SharedAccess>,
    pub shared_reads: Vec<SharedAccess>,
    pub local_memory_size: usize,
    pub shared_memory_size: usize,
}

impl KernelProfile {
    pub fn new(name: &str) -> Self {
        KernelProfile {
            name: name.to_string(),
            local_writes: Vec::new(),
            local_reads: Vec::new(),
            shared_writes: Vec::new(),
            shared_reads: Vec::new(),
            local_memory_size: 0,
            shared_memory_size: 0,
        }
    }

    pub fn add_local_write(&mut self, thread_id: usize, offset: usize, size: usize, sensitive: bool) {
        self.local_writes.push(LocalAccess { thread_id, offset, size, is_sensitive: sensitive });
    }

    pub fn add_local_read(&mut self, thread_id: usize, offset: usize, size: usize, sensitive: bool) {
        self.local_reads.push(LocalAccess { thread_id, offset, size, is_sensitive: sensitive });
    }
}

#[derive(Debug, Clone)]
pub struct LocalAccess {
    pub thread_id: usize,
    pub offset: usize,
    pub size: usize,
    pub is_sensitive: bool,
}

#[derive(Debug, Clone)]
pub struct SharedAccess {
    pub thread_id: usize,
    pub offset: usize,
    pub size: usize,
}

// ---------------------------------------------------------------------------
// CacheResidueAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes cache-based residue channels.
#[derive(Debug, Clone)]
pub struct CacheResidueAnalyzer {
    pub cache_sets: usize,
    pub ways: usize,
    pub line_size: usize,
}

impl CacheResidueAnalyzer {
    pub fn new(sets: usize, ways: usize, line_size: usize) -> Self {
        CacheResidueAnalyzer { cache_sets: sets, ways, line_size }
    }

    /// Default GPU L1 cache parameters.
    pub fn default_gpu_l1() -> Self {
        Self::new(64, 4, 128)
    }

    /// Map address to cache set.
    pub fn address_to_set(&self, address: u64) -> usize {
        ((address / self.line_size as u64) % self.cache_sets as u64) as usize
    }

    /// Analyze access pattern for cache residue.
    pub fn analyze_accesses(&self, victim_accesses: &[u64], attacker_probes: &[u64]) -> CacheResidueResult {
        // Which sets were touched by victim?
        let victim_sets: HashSet<usize> = victim_accesses.iter()
            .map(|&a| self.address_to_set(a))
            .collect();

        // Which sets does attacker probe?
        let probe_sets: HashSet<usize> = attacker_probes.iter()
            .map(|&a| self.address_to_set(a))
            .collect();

        // Overlap = sets where attacker can detect victim activity
        let overlap: HashSet<usize> = victim_sets.intersection(&probe_sets).copied().collect();

        // Information leakage through cache timing
        let bits_leaked = if overlap.is_empty() { 0.0 }
            else { (overlap.len() as f64).log2().max(0.0) + 1.0 };

        CacheResidueResult {
            victim_sets: victim_sets.len(),
            probe_sets: probe_sets.len(),
            overlap_sets: overlap.len(),
            bits_leaked,
            total_sets: self.cache_sets,
        }
    }

    /// Estimate cache-based covert channel bandwidth.
    pub fn estimate_bandwidth(&self, access_latency_ns: f64, probe_latency_ns: f64) -> f64 {
        // One bit per cache set per probe cycle
        let bits_per_probe = (self.cache_sets as f64).log2();
        let cycle_time = access_latency_ns + probe_latency_ns;
        if cycle_time > 0.0 {
            bits_per_probe * 1e9 / cycle_time // bits per second
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheResidueResult {
    pub victim_sets: usize,
    pub probe_sets: usize,
    pub overlap_sets: usize,
    pub bits_leaked: f64,
    pub total_sets: usize,
}

// ---------------------------------------------------------------------------
// ResidueChannelDetector
// ---------------------------------------------------------------------------

/// Top-level detector for memory residue channels.
#[derive(Debug, Clone)]
pub struct ResidueChannelDetector {
    pub patterns: Vec<ResiduePattern>,
    pub allocation_history: AllocationPattern,
}

impl ResidueChannelDetector {
    pub fn new() -> Self {
        ResidueChannelDetector {
            patterns: Vec::new(),
            allocation_history: AllocationPattern::new(),
        }
    }

    /// Add a detected residue pattern.
    pub fn add_pattern(&mut self, pattern: ResiduePattern) {
        self.patterns.push(pattern);
    }

    /// Scan allocation history for residue risks.
    pub fn scan_allocations(&mut self) {
        let reuses = self.allocation_history.reuse_patterns();
        for reuse in &reuses {
            let old = &self.allocation_history.allocations[reuse.old_allocation];
            self.patterns.push(
                ResiduePattern::new(
                    &format!("alloc-reuse-{}", reuse.old_allocation),
                    old.region,
                    ResidueDataType::RawBytes,
                )
                .with_size(reuse.overlap_bytes)
                .with_persistence(Persistence::UntilReallocation)
                .with_source(&reuse.old_kernel)
                .with_description(&format!(
                    "Memory from '{}' reused by '{}' without clearing ({} bytes)",
                    reuse.old_kernel, reuse.new_kernel, reuse.overlap_bytes
                ))
            );
        }
    }

    /// Generate a report.
    pub fn generate_report(&self) -> ResidueReport {
        let mut patterns = self.patterns.clone();
        patterns.sort_by(|a, b| b.risk_score().partial_cmp(&a.risk_score()).unwrap());

        let total_risk = patterns.iter().map(|p| p.risk_score()).sum::<f64>();
        let max_risk = patterns.iter().map(|p| p.risk_score()).fold(0.0f64, f64::max);

        ResidueReport {
            patterns,
            total_risk,
            max_risk,
            summary: String::new(),
        }
    }

    /// Number of detected patterns.
    pub fn count(&self) -> usize { self.patterns.len() }
}

// ---------------------------------------------------------------------------
// ResidueReport
// ---------------------------------------------------------------------------

/// Report of detected memory residue vulnerabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueReport {
    pub patterns: Vec<ResiduePattern>,
    pub total_risk: f64,
    pub max_risk: f64,
    pub summary: String,
}

impl ResidueReport {
    /// Severity based on maximum risk.
    pub fn severity(&self) -> &'static str {
        if self.max_risk > 7.0 { "CRITICAL" }
        else if self.max_risk > 5.0 { "HIGH" }
        else if self.max_risk > 3.0 { "MEDIUM" }
        else if self.max_risk > 1.0 { "LOW" }
        else { "NONE" }
    }

    /// Format as text report.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("=== Memory Residue Analysis ===\n"));
        s.push_str(&format!("Patterns found: {}\n", self.patterns.len()));
        s.push_str(&format!("Max risk: {:.1} ({})\n\n", self.max_risk, self.severity()));
        for (i, p) in self.patterns.iter().enumerate() {
            s.push_str(&format!("  [{}] {} ({}, {:.1} risk)\n",
                i + 1, p.id, p.memory_region, p.risk_score()));
            if !p.description.is_empty() {
                s.push_str(&format!("      {}\n", p.description));
            }
        }
        s
    }

    /// Format as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

impl fmt::Display for ResidueReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResidueReport({} patterns, max_risk={:.1})", self.patterns.len(), self.max_risk)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_pattern_risk() {
        let p = ResiduePattern::new("test", MemoryRegion::LocalMemory, ResidueDataType::Keys)
            .with_persistence(Persistence::AcrossProcesses);
        assert!(p.risk_score() > 5.0);
    }

    #[test]
    fn test_residue_pattern_low_risk() {
        let p = ResiduePattern::new("test", MemoryRegion::ConstantCache, ResidueDataType::RawBytes)
            .with_persistence(Persistence::Cleared);
        assert!(p.risk_score() < 1.0);
    }

    #[test]
    fn test_allocation_pattern_reuse() {
        let mut ap = AllocationPattern::new();
        let a0 = ap.add_allocation(0x1000, 256, MemoryRegion::LocalMemory, "kernel_a", 0, false);
        ap.add_deallocation(a0, 10, false);
        let _a1 = ap.add_allocation(0x1000, 256, MemoryRegion::LocalMemory, "kernel_b", 20, false);
        let reuses = ap.reuse_patterns();
        assert_eq!(reuses.len(), 1);
        assert_eq!(reuses[0].overlap_bytes, 256);
    }

    #[test]
    fn test_allocation_no_reuse_when_wiped() {
        let mut ap = AllocationPattern::new();
        let a0 = ap.add_allocation(0x1000, 256, MemoryRegion::LocalMemory, "kernel_a", 0, false);
        ap.add_deallocation(a0, 10, true); // wiped!
        let _a1 = ap.add_allocation(0x1000, 256, MemoryRegion::LocalMemory, "kernel_b", 20, false);
        let reuses = ap.reuse_patterns();
        assert_eq!(reuses.len(), 0);
    }

    #[test]
    fn test_leftover_locals_detector() {
        let mut ka = KernelProfile::new("victim");
        ka.add_local_write(0, 0, 128, true);
        ka.add_local_write(0, 128, 64, false);

        let mut kb = KernelProfile::new("attacker");
        kb.add_local_read(0, 0, 128, false);

        let mut detector = LeftoverLocalsDetector::new();
        detector.analyze(&ka, &kb);
        assert_eq!(detector.count(), 1);
        assert!(detector.max_severity() > 0.0);
    }

    #[test]
    fn test_cache_residue_analyzer() {
        let analyzer = CacheResidueAnalyzer::default_gpu_l1();
        let victim = vec![0x0, 0x80, 0x100, 0x180]; // 4 cache lines
        let attacker = vec![0x0, 0x100]; // 2 overlapping probes
        let result = analyzer.analyze_accesses(&victim, &attacker);
        assert!(result.overlap_sets > 0);
        assert!(result.bits_leaked > 0.0);
    }

    #[test]
    fn test_cache_bandwidth_estimation() {
        let analyzer = CacheResidueAnalyzer::default_gpu_l1();
        let bw = analyzer.estimate_bandwidth(100.0, 50.0);
        assert!(bw > 0.0);
    }

    #[test]
    fn test_residue_detector_scan() {
        let mut detector = ResidueChannelDetector::new();
        let a0 = detector.allocation_history.add_allocation(
            0x1000, 256, MemoryRegion::SharedMemory, "victim_kernel", 0, false
        );
        detector.allocation_history.add_deallocation(a0, 10, false);
        detector.allocation_history.add_allocation(
            0x1000, 256, MemoryRegion::SharedMemory, "attacker_kernel", 20, false
        );
        detector.scan_allocations();
        assert!(detector.count() > 0);
    }

    #[test]
    fn test_residue_report() {
        let mut detector = ResidueChannelDetector::new();
        detector.add_pattern(
            ResiduePattern::new("test1", MemoryRegion::LocalMemory, ResidueDataType::Keys)
                .with_persistence(Persistence::AcrossKernels)
        );
        let report = detector.generate_report();
        assert_eq!(report.patterns.len(), 1);
        assert!(report.max_risk > 0.0);
        let text = report.to_text();
        assert!(text.contains("Memory Residue"));
    }

    #[test]
    fn test_residue_report_json() {
        let report = ResidueReport {
            patterns: vec![],
            total_risk: 0.0,
            max_risk: 0.0,
            summary: String::new(),
        };
        let json = report.to_json();
        assert!(json.contains("patterns"));
    }

    #[test]
    fn test_allocation_stats() {
        let mut ap = AllocationPattern::new();
        ap.add_allocation(0x1000, 256, MemoryRegion::GlobalMemory, "k1", 0, true);
        ap.add_allocation(0x2000, 512, MemoryRegion::GlobalMemory, "k2", 1, false);
        assert_eq!(ap.total_allocated(), 768);
        assert_eq!(ap.zeroed_count(), 1);
    }

    #[test]
    fn test_unwiped_deallocations() {
        let mut ap = AllocationPattern::new();
        let a0 = ap.add_allocation(0x1000, 256, MemoryRegion::LocalMemory, "k1", 0, false);
        let a1 = ap.add_allocation(0x2000, 128, MemoryRegion::LocalMemory, "k2", 1, false);
        ap.add_deallocation(a0, 10, false);
        ap.add_deallocation(a1, 11, true);
        let unwiped = ap.unwiped_deallocations();
        assert_eq!(unwiped.len(), 1);
    }

    #[test]
    fn test_kernel_profile() {
        let mut kp = KernelProfile::new("test_kernel");
        kp.add_local_write(0, 0, 64, true);
        kp.add_local_read(0, 0, 64, false);
        assert_eq!(kp.local_writes.len(), 1);
        assert_eq!(kp.local_reads.len(), 1);
    }
}
