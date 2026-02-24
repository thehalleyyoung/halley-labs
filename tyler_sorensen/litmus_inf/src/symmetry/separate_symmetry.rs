//! Separate thread symmetry and memory symmetry computation.
//!
//! Key insight: computing joint thread×memory automorphism is expensive
//! (and the source of negative speedup). Computing them SEPARATELY is
//! much cheaper and still catches most symmetries.

use std::collections::{HashMap, HashSet, BTreeMap};
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, EventId, ThreadId, Address, OpType, Scope, ExecutionGraph,
};

fn optype_ord(o: &OpType) -> u8 {
    match o {
        OpType::Read => 0, OpType::Write => 1, OpType::Fence => 2, OpType::RMW => 3,
    }
}
fn scope_ord(s: &Scope) -> u8 {
    match s {
        Scope::CTA => 0, Scope::GPU => 1, Scope::System => 2, Scope::None => 3,
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparateSymmetryConfig {
    /// Maximum group size before giving up on thread symmetry.
    pub max_thread_group_size: usize,
    /// Maximum group size before giving up on memory symmetry.
    pub max_memory_group_size: usize,
    /// Whether to use thread symmetry.
    pub use_thread_symmetry: bool,
    /// Whether to use memory (address) symmetry.
    pub use_memory_symmetry: bool,
    /// Whether to use value symmetry.
    pub use_value_symmetry: bool,
}

impl Default for SeparateSymmetryConfig {
    fn default() -> Self {
        Self {
            max_thread_group_size: 1000,
            max_memory_group_size: 1000,
            use_thread_symmetry: true,
            use_memory_symmetry: true,
            use_value_symmetry: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparateSymmetryResult {
    /// Thread permutations that preserve the litmus test.
    pub thread_symmetries: Vec<Vec<usize>>,
    /// Memory address permutations that preserve the litmus test.
    pub memory_symmetries: Vec<Vec<u64>>,
    /// Combined reduction factor.
    pub reduction_factor: f64,
    /// Orbit count for threads.
    pub thread_orbit_count: usize,
    /// Orbit count for memory.
    pub memory_orbit_count: usize,
    /// Time to compute (ms).
    pub computation_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Thread Symmetry
// ---------------------------------------------------------------------------

/// Computes thread permutation symmetries.
///
/// Two threads are symmetric if they have the same instruction signature
/// (same sequence of opcodes, relative address references, orderings).
pub struct ThreadSymmetryComputer {
    config: SeparateSymmetryConfig,
}

impl ThreadSymmetryComputer {
    pub fn new(config: SeparateSymmetryConfig) -> Self {
        Self { config }
    }

    /// Compute a thread signature that captures its behavior abstractly.
    /// Two threads with the same signature are potentially symmetric.
    pub fn thread_signature(&self, events: &[Event], thread: ThreadId) -> Vec<(OpType, u64, Scope)> {
        let mut sig: Vec<(OpType, u64, Scope)> = events.iter()
            .filter(|e| e.thread == thread)
            .map(|e| (e.op_type, e.address, e.scope))
            .collect();
        sig.sort_by_key(|(o, a, s)| (optype_ord(o), *a, scope_ord(s)));
        sig
    }

    /// Find groups of threads with identical signatures.
    pub fn find_symmetric_groups(&self, events: &[Event], num_threads: usize) -> Vec<Vec<ThreadId>> {
        let mut sig_map: HashMap<Vec<(OpType, u64, Scope)>, Vec<ThreadId>> = HashMap::new();

        for t in 0..num_threads {
            let sig = self.thread_signature(events, t);
            sig_map.entry(sig).or_default().push(t);
        }

        sig_map.into_values()
            .filter(|group| group.len() > 1)
            .collect()
    }

    /// Generate thread permutations from symmetric groups.
    pub fn generate_permutations(
        &self,
        groups: &[Vec<ThreadId>],
        num_threads: usize,
    ) -> Vec<Vec<usize>> {
        let mut permutations = Vec::new();

        // Identity permutation
        let identity: Vec<usize> = (0..num_threads).collect();
        permutations.push(identity.clone());

        for group in groups {
            if group.len() > self.config.max_thread_group_size {
                continue;
            }
            // Generate all pairwise swaps within group
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    let mut perm = identity.clone();
                    perm[group[i]] = group[j];
                    perm[group[j]] = group[i];
                    permutations.push(perm);
                }
            }

            // For small groups, generate all permutations
            if group.len() <= 4 {
                let all_perms = Self::all_permutations_of(group);
                for p in all_perms {
                    let mut perm = identity.clone();
                    for (idx, &thread) in group.iter().enumerate() {
                        perm[thread] = p[idx];
                    }
                    if perm != identity {
                        permutations.push(perm);
                    }
                }
            }
        }

        permutations.sort();
        permutations.dedup();
        permutations
    }

    fn all_permutations_of(group: &[ThreadId]) -> Vec<Vec<ThreadId>> {
        if group.len() <= 1 {
            return vec![group.to_vec()];
        }

        let mut result = Vec::new();
        for i in 0..group.len() {
            let rest: Vec<ThreadId> = group.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &t)| t)
                .collect();
            for mut sub in Self::all_permutations_of(&rest) {
                sub.insert(0, group[i]);
                result.push(sub);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Memory Symmetry
// ---------------------------------------------------------------------------

/// Computes memory (address) permutation symmetries.
///
/// Two addresses are symmetric if swapping them preserves all litmus test
/// structure (same threads access them in the same pattern).
pub struct MemorySymmetryComputer {
    config: SeparateSymmetryConfig,
}

impl MemorySymmetryComputer {
    pub fn new(config: SeparateSymmetryConfig) -> Self {
        Self { config }
    }

    /// Compute address signature: which threads access it and how.
    pub fn address_signature(&self, events: &[Event], addr: Address) -> Vec<(ThreadId, OpType, Scope)> {
        let mut sig: Vec<(ThreadId, OpType, Scope)> = events.iter()
            .filter(|e| e.address == addr && e.op_type != OpType::Fence)
            .map(|e| (e.thread, e.op_type, e.scope))
            .collect();
        sig.sort_by_key(|(t, o, s)| (*t, optype_ord(o), scope_ord(s)));
        sig
    }

    /// Find groups of addresses with identical access patterns.
    pub fn find_symmetric_groups(&self, events: &[Event]) -> Vec<Vec<Address>> {
        let mut addrs: HashSet<Address> = HashSet::new();
        for e in events {
            if e.op_type != OpType::Fence {
                addrs.insert(e.address);
            }
        }

        let mut sig_map: HashMap<Vec<(ThreadId, OpType, Scope)>, Vec<Address>> = HashMap::new();
        for &addr in &addrs {
            let sig = self.address_signature(events, addr);
            sig_map.entry(sig).or_default().push(addr);
        }

        sig_map.into_values()
            .filter(|group| group.len() > 1)
            .collect()
    }

    /// Generate address permutations from symmetric groups.
    pub fn generate_permutations(
        &self,
        groups: &[Vec<Address>],
    ) -> Vec<Vec<u64>> {
        let mut permutations = Vec::new();

        for group in groups {
            if group.len() > self.config.max_memory_group_size {
                continue;
            }
            // Generate pairwise swaps
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    permutations.push(vec![group[i], group[j]]);
                }
            }
        }

        permutations
    }

    /// Compute the reduction factor from memory symmetry.
    pub fn reduction_factor(&self, groups: &[Vec<Address>]) -> f64 {
        let mut factor = 1.0;
        for group in groups {
            // |Sym(n)| = n!
            let n = group.len() as f64;
            let mut factorial = 1.0;
            for i in 2..=group.len() {
                factorial *= i as f64;
            }
            factor *= factorial;
        }
        factor
    }
}

// ---------------------------------------------------------------------------
// Combined
// ---------------------------------------------------------------------------

/// Compute separate symmetry reduction for an execution graph.
pub fn compute_separate_symmetry(
    events: &[Event],
    num_threads: usize,
    config: &SeparateSymmetryConfig,
) -> SeparateSymmetryResult {
    let start = std::time::Instant::now();

    let mut thread_symmetries = Vec::new();
    let mut thread_orbit_count = num_threads;
    let mut thread_reduction = 1.0f64;

    if config.use_thread_symmetry {
        let tsc = ThreadSymmetryComputer::new(config.clone());
        let groups = tsc.find_symmetric_groups(events, num_threads);
        thread_symmetries = tsc.generate_permutations(&groups, num_threads);
        thread_orbit_count = num_threads - groups.iter().map(|g| g.len() - 1).sum::<usize>();
        for group in &groups {
            let mut factorial = 1.0;
            for i in 2..=group.len() {
                factorial *= i as f64;
            }
            thread_reduction *= factorial;
        }
    }

    let mut memory_symmetries = Vec::new();
    let mut memory_orbit_count = 0;
    let mut memory_reduction = 1.0f64;

    if config.use_memory_symmetry {
        let msc = MemorySymmetryComputer::new(config.clone());
        let groups = msc.find_symmetric_groups(events);
        memory_symmetries = msc.generate_permutations(&groups);
        memory_reduction = msc.reduction_factor(&groups);
        let addrs: HashSet<Address> = events.iter()
            .filter(|e| e.op_type != OpType::Fence)
            .map(|e| e.address)
            .collect();
        memory_orbit_count = addrs.len() - groups.iter().map(|g| g.len() - 1).sum::<usize>();
    }

    let reduction_factor = thread_reduction * memory_reduction;

    SeparateSymmetryResult {
        thread_symmetries,
        memory_symmetries,
        reduction_factor,
        thread_orbit_count,
        memory_orbit_count,
        computation_time_ms: start.elapsed().as_millis() as u64,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_symmetric_events() -> (Vec<Event>, usize) {
        // 4-thread SB where threads 0,1 are symmetric and threads 2,3 are symmetric
        let events = vec![
            // Thread 0: W(x), R(y)
            Event::new(0, 0, OpType::Write, 0, 1),
            Event::new(1, 0, OpType::Read, 1, 0),
            // Thread 1: W(x), R(y) — same as thread 0
            Event::new(2, 1, OpType::Write, 0, 1),
            Event::new(3, 1, OpType::Read, 1, 0),
            // Thread 2: W(y), R(x)
            Event::new(4, 2, OpType::Write, 1, 1),
            Event::new(5, 2, OpType::Read, 0, 0),
            // Thread 3: W(y), R(x) — same as thread 2
            Event::new(6, 3, OpType::Write, 1, 1),
            Event::new(7, 3, OpType::Read, 0, 0),
        ];
        (events, 4)
    }

    #[test]
    fn test_thread_symmetry_detection() {
        let (events, num_threads) = make_symmetric_events();
        let config = SeparateSymmetryConfig::default();
        let tsc = ThreadSymmetryComputer::new(config);
        let groups = tsc.find_symmetric_groups(&events, num_threads);
        
        assert!(!groups.is_empty(), "Should find symmetric thread groups");
        // Threads 0,1 should be in one group, threads 2,3 in another
        assert!(groups.len() >= 2);
    }

    #[test]
    fn test_memory_symmetry_detection() {
        // Create events where addresses 0 and 1 are accessed identically
        let events = vec![
            Event::new(0, 0, OpType::Write, 0, 1),
            Event::new(1, 1, OpType::Read, 0, 0),
            Event::new(2, 0, OpType::Write, 1, 1),
            Event::new(3, 1, OpType::Read, 1, 0),
        ];
        let config = SeparateSymmetryConfig::default();
        let msc = MemorySymmetryComputer::new(config);
        let groups = msc.find_symmetric_groups(&events);
        
        assert!(!groups.is_empty(), "Should find symmetric address groups");
    }

    #[test]
    fn test_combined_reduction() {
        let (events, num_threads) = make_symmetric_events();
        let config = SeparateSymmetryConfig::default();
        let result = compute_separate_symmetry(&events, num_threads, &config);
        
        assert!(result.reduction_factor >= 1.0);
        assert!(!result.thread_symmetries.is_empty());
    }

    #[test]
    fn test_no_symmetry() {
        // All threads are different
        let events = vec![
            Event::new(0, 0, OpType::Write, 0, 1),
            Event::new(1, 1, OpType::Read, 1, 0),
            Event::new(2, 2, OpType::Write, 2, 1),
        ];
        let config = SeparateSymmetryConfig::default();
        let tsc = ThreadSymmetryComputer::new(config);
        let groups = tsc.find_symmetric_groups(&events, 3);
        assert!(groups.is_empty(), "No threads should be symmetric");
    }

    #[test]
    fn test_thread_permutation_generation() {
        let config = SeparateSymmetryConfig::default();
        let tsc = ThreadSymmetryComputer::new(config);
        let groups = vec![vec![0, 1], vec![2, 3]];
        let perms = tsc.generate_permutations(&groups, 4);
        assert!(perms.len() > 1); // At least identity + swaps
    }
}
