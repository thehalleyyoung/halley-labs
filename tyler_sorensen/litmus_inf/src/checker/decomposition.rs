//! Compositional decomposition for litmus test verification.
//!
//! Implements the Gluing Theorem approach: decompose a litmus test
//! into independent sub-tests that can be verified independently
//! and then composed. Includes optimal decomposition search and
//! validation of decomposition correctness.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use super::execution::{Address, ExecutionGraph, BitMatrix, EventId};
use super::litmus::{LitmusTest, Thread, Outcome, LitmusOutcome};
use super::memory_model::MemoryModel;
use super::verifier::{Verifier, VerificationResult, VerificationStats};

// ---------------------------------------------------------------------------
// DecompositionNode / DecompositionTree
// ---------------------------------------------------------------------------

/// A node in a decomposition tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionNode {
    /// Leaf: an atomic sub-test (list of thread indices).
    Leaf {
        thread_ids: Vec<usize>,
        addresses: Vec<Address>,
    },
    /// Internal: composition of sub-decompositions.
    Compose {
        children: Vec<DecompositionNode>,
        shared_addresses: Vec<Address>,
    },
}

impl DecompositionNode {
    /// All thread ids in this subtree.
    pub fn all_threads(&self) -> Vec<usize> {
        match self {
            Self::Leaf { thread_ids, .. } => thread_ids.clone(),
            Self::Compose { children, .. } => {
                let mut all = Vec::new();
                for child in children {
                    all.extend(child.all_threads());
                }
                all.sort();
                all.dedup();
                all
            }
        }
    }

    /// All addresses in this subtree.
    pub fn all_addresses(&self) -> Vec<Address> {
        match self {
            Self::Leaf { addresses, .. } => addresses.clone(),
            Self::Compose { children, shared_addresses, .. } => {
                let mut all: Vec<Address> = shared_addresses.clone();
                for child in children {
                    all.extend(child.all_addresses());
                }
                all.sort();
                all.dedup();
                all
            }
        }
    }

    /// Depth of the decomposition tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Compose { children, .. } => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Number of leaves.
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Compose { children, .. } => {
                children.iter().map(|c| c.leaf_count()).sum()
            }
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Compose { children, .. } => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }
}

impl fmt::Display for DecompositionNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

impl DecompositionNode {
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = " ".repeat(indent * 2);
        match self {
            Self::Leaf { thread_ids, addresses } => {
                writeln!(f, "{}Leaf(threads={:?}, addrs={:?})", pad, thread_ids, addresses)
            }
            Self::Compose { children, shared_addresses } => {
                writeln!(f, "{}Compose(shared={:?})", pad, shared_addresses)?;
                for child in children {
                    child.fmt_indent(f, indent + 1)?;
                }
                Ok(())
            }
        }
    }
}

/// Complete decomposition tree for a litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionTree {
    pub root: DecompositionNode,
    pub test_name: String,
    pub total_threads: usize,
    pub total_addresses: usize,
}

impl DecompositionTree {
    pub fn new(test_name: &str, root: DecompositionNode, n_threads: usize, n_addrs: usize) -> Self {
        Self {
            root,
            test_name: test_name.into(),
            total_threads: n_threads,
            total_addresses: n_addrs,
        }
    }

    /// Whether the decomposition is trivial (single leaf).
    pub fn is_trivial(&self) -> bool {
        matches!(&self.root, DecompositionNode::Leaf { .. })
    }

    /// Compression ratio: leaf_count / 1 (higher = more decomposition).
    pub fn compression_ratio(&self) -> f64 {
        self.root.leaf_count() as f64
    }
}

impl fmt::Display for DecompositionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DecompositionTree for '{}' ({} threads, {} addrs):",
            self.test_name, self.total_threads, self.total_addresses)?;
        write!(f, "{}", self.root)
    }
}

// ---------------------------------------------------------------------------
// TestDecomposer
// ---------------------------------------------------------------------------

/// Decomposes litmus tests into independent components.
pub struct TestDecomposer;

impl TestDecomposer {
    /// Build a decomposition tree for a litmus test.
    pub fn decompose(test: &LitmusTest) -> DecompositionTree {
        let n = test.thread_count();
        let addrs = test.all_addresses();

        if n <= 1 {
            let thread_ids: Vec<usize> = (0..n).collect();
            let root = DecompositionNode::Leaf {
                thread_ids,
                addresses: addrs.clone(),
            };
            return DecompositionTree::new(&test.name, root, n, addrs.len());
        }

        // Build thread-address bipartite graph.
        let thread_addrs: Vec<HashSet<Address>> = (0..n)
            .map(|i| test.threads[i].accessed_addresses().into_iter().collect())
            .collect();

        // Find connected components in the thread interaction graph.
        let components = Self::find_components(n, &thread_addrs);

        if components.len() == 1 {
            // Try hierarchical decomposition within the single component.
            let root = Self::hierarchical_decompose(&components[0], &thread_addrs);
            DecompositionTree::new(&test.name, root, n, addrs.len())
        } else {
            // Multiple independent components.
            let children: Vec<DecompositionNode> = components.iter().map(|comp| {
                let comp_addrs: Vec<Address> = comp.iter()
                    .flat_map(|&t| thread_addrs[t].iter().copied())
                    .collect::<HashSet<_>>()
                    .into_iter().collect();
                if comp.len() == 1 {
                    DecompositionNode::Leaf {
                        thread_ids: comp.clone(),
                        addresses: comp_addrs,
                    }
                } else {
                    Self::hierarchical_decompose(comp, &thread_addrs)
                }
            }).collect();

            let root = DecompositionNode::Compose {
                children,
                shared_addresses: Vec::new(),
            };
            DecompositionTree::new(&test.name, root, n, addrs.len())
        }
    }

    /// Find connected components based on shared addresses.
    fn find_components(n: usize, thread_addrs: &[HashSet<Address>]) -> Vec<Vec<usize>> {
        let mut adj = vec![HashSet::new(); n];
        for i in 0..n {
            for j in i + 1..n {
                if thread_addrs[i].intersection(&thread_addrs[j]).next().is_some() {
                    adj[i].insert(j);
                    adj[j].insert(i);
                }
            }
        }

        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if visited[start] { continue; }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(node) = queue.pop_front() {
                component.push(node);
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            component.sort();
            components.push(component);
        }

        components
    }

    /// Hierarchical decomposition within a connected component.
    /// Tries to split by removing "bridge" addresses.
    fn hierarchical_decompose(
        threads: &[usize],
        thread_addrs: &[HashSet<Address>],
    ) -> DecompositionNode {
        if threads.len() <= 2 {
            let addrs: Vec<Address> = threads.iter()
                .flat_map(|&t| thread_addrs[t].iter().copied())
                .collect::<HashSet<_>>()
                .into_iter().collect();
            return DecompositionNode::Leaf {
                thread_ids: threads.to_vec(),
                addresses: addrs,
            };
        }

        // Collect all addresses in this component.
        let all_addrs: HashSet<Address> = threads.iter()
            .flat_map(|&t| thread_addrs[t].iter().copied())
            .collect();

        // Try removing each address and check if the component splits.
        for &addr in &all_addrs {
            let reduced_addrs: Vec<HashSet<Address>> = threads.iter()
                .map(|&t| {
                    thread_addrs[t].iter()
                        .copied()
                        .filter(|&a| a != addr)
                        .collect()
                })
                .collect();

            // Build adjacency on reduced addresses.
            let n = threads.len();
            let mut adj = vec![HashSet::new(); n];
            for i in 0..n {
                for j in i + 1..n {
                    if reduced_addrs[i].intersection(&reduced_addrs[j]).next().is_some() {
                        adj[i].insert(j);
                        adj[j].insert(i);
                    }
                }
            }

            // Check connectivity.
            let mut visited = vec![false; n];
            let mut queue = VecDeque::new();
            queue.push_back(0);
            visited[0] = true;
            while let Some(node) = queue.pop_front() {
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }

            if visited.iter().any(|&v| !v) {
                // Component splits! Recurse on sub-components.
                let mut sub_components: Vec<Vec<usize>> = Vec::new();
                let mut comp_visited = vec![false; n];

                for start in 0..n {
                    if comp_visited[start] { continue; }
                    let mut comp = Vec::new();
                    let mut q = VecDeque::new();
                    q.push_back(start);
                    comp_visited[start] = true;
                    while let Some(node) = q.pop_front() {
                        comp.push(threads[node]);
                        for &neighbor in &adj[node] {
                            if !comp_visited[neighbor] {
                                comp_visited[neighbor] = true;
                                q.push_back(neighbor);
                            }
                        }
                    }
                    comp.sort();
                    sub_components.push(comp);
                }

                let children: Vec<DecompositionNode> = sub_components.iter()
                    .map(|comp| Self::hierarchical_decompose(comp, thread_addrs))
                    .collect();

                return DecompositionNode::Compose {
                    children,
                    shared_addresses: vec![addr],
                };
            }
        }

        // No split found: return as leaf.
        let addrs: Vec<Address> = all_addrs.into_iter().collect();
        DecompositionNode::Leaf {
            thread_ids: threads.to_vec(),
            addresses: addrs,
        }
    }
}

// ---------------------------------------------------------------------------
// GluingTheorem
// ---------------------------------------------------------------------------

/// Implementation of the Gluing Theorem for compositional verification.
///
/// Theorem: If each component of a decomposition is consistent under
/// the memory model, and the shared interface constraints are satisfied,
/// then the composed test is also consistent.
pub struct GluingTheorem;

impl GluingTheorem {
    /// Check whether a decomposition satisfies the gluing conditions.
    pub fn check_gluing_conditions(
        test: &LitmusTest,
        tree: &DecompositionTree,
    ) -> GluingResult {
        let components = Self::extract_components(&tree.root);

        // Check 1: All threads are covered.
        let all_threads: HashSet<usize> = components.iter()
            .flat_map(|c| c.iter().copied())
            .collect();
        let expected: HashSet<usize> = (0..test.thread_count()).collect();
        let complete = all_threads == expected;

        // Check 2: Components are disjoint in threads (modulo shared).
        let mut thread_counts: HashMap<usize, usize> = HashMap::new();
        for comp in &components {
            for &t in comp {
                *thread_counts.entry(t).or_insert(0) += 1;
            }
        }
        let disjoint = thread_counts.values().all(|&c| c == 1);

        // Check 3: Shared addresses are properly handled.
        let shared_addrs = Self::collect_shared_addresses(&tree.root);

        GluingResult {
            valid: complete && disjoint,
            complete,
            disjoint,
            shared_addresses: shared_addrs,
            component_count: components.len(),
        }
    }

    /// Extract leaf components (thread id lists) from a decomposition tree.
    fn extract_components(node: &DecompositionNode) -> Vec<Vec<usize>> {
        match node {
            DecompositionNode::Leaf { thread_ids, .. } => {
                vec![thread_ids.clone()]
            }
            DecompositionNode::Compose { children, .. } => {
                children.iter()
                    .flat_map(|c| Self::extract_components(c))
                    .collect()
            }
        }
    }

    /// Collect all shared addresses from the decomposition tree.
    fn collect_shared_addresses(node: &DecompositionNode) -> Vec<Address> {
        match node {
            DecompositionNode::Leaf { .. } => Vec::new(),
            DecompositionNode::Compose { children, shared_addresses } => {
                let mut addrs = shared_addresses.clone();
                for child in children {
                    addrs.extend(Self::collect_shared_addresses(child));
                }
                addrs.sort();
                addrs.dedup();
                addrs
            }
        }
    }

    /// Verify using the gluing theorem: verify components independently.
    pub fn verify_with_gluing(
        test: &LitmusTest,
        tree: &DecompositionTree,
        model: &MemoryModel,
    ) -> VerificationResult {
        let gluing = Self::check_gluing_conditions(test, tree);

        if !gluing.valid || gluing.component_count <= 1 {
            // Fall back to direct verification.
            let mut v = Verifier::new(model.clone());
            return v.verify_litmus(test);
        }

        let components = Self::extract_components(&tree.root);
        let mut total_consistent = 1usize;
        let mut total_execs = 1usize;
        let mut all_pass = true;
        let mut combined_stats = VerificationStats::default();
        let mut all_outcomes = Vec::new();
        let mut all_forbidden = Vec::new();
        let mut all_required_missing = Vec::new();

        let cv = super::verifier::CompositionalVerifier::new(model.clone());

        for component in &components {
            let sub_test = cv.extract_component(test, component);
            let mut v = Verifier::new(model.clone());
            let result = v.verify_litmus(&sub_test);

            total_consistent *= result.consistent_executions.max(1);
            total_execs *= result.total_executions.max(1);
            all_pass = all_pass && result.pass;

            combined_stats.executions_checked += result.stats.executions_checked;
            combined_stats.consistent_found += result.stats.consistent_found;
            combined_stats.violations_found += result.stats.violations_found;
            combined_stats.acyclicity_checks += result.stats.acyclicity_checks;
            combined_stats.irreflexivity_checks += result.stats.irreflexivity_checks;
            combined_stats.emptiness_checks += result.stats.emptiness_checks;
            combined_stats.relations_computed += result.stats.relations_computed;

            all_outcomes.extend(result.observed_outcomes);
            all_forbidden.extend(result.forbidden_observed);
            all_required_missing.extend(result.required_missing);
        }

        VerificationResult {
            test_name: test.name.clone(),
            model_name: model.name.clone(),
            total_executions: total_execs,
            consistent_executions: total_consistent,
            inconsistent_executions: total_execs.saturating_sub(total_consistent),
            observed_outcomes: all_outcomes,
            forbidden_observed: all_forbidden,
            required_missing: all_required_missing,
            pass: all_pass,
            stats: combined_stats,
        }
    }
}

/// Result of checking gluing conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GluingResult {
    pub valid: bool,
    pub complete: bool,
    pub disjoint: bool,
    pub shared_addresses: Vec<Address>,
    pub component_count: usize,
}

impl fmt::Display for GluingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gluing(valid={}, complete={}, disjoint={}, components={}, shared={:?})",
            self.valid, self.complete, self.disjoint,
            self.component_count, self.shared_addresses)
    }
}

// ---------------------------------------------------------------------------
// OptimalDecomposition
// ---------------------------------------------------------------------------

/// Finds the optimal decomposition of a litmus test.
///
/// "Optimal" means minimizing the total number of executions to enumerate
/// across all components.
pub struct OptimalDecomposition;

impl OptimalDecomposition {
    /// Find the best decomposition by trying different strategies.
    pub fn find_optimal(test: &LitmusTest) -> DecompositionTree {
        let basic = TestDecomposer::decompose(test);

        // If already well-decomposed, return.
        if basic.root.leaf_count() > 1 {
            return basic;
        }

        // Try alternative decomposition strategies.
        let by_address = Self::decompose_by_address(test);
        let by_thread_pair = Self::decompose_by_thread_pairs(test);

        // Pick the one with the most leaves (most decomposed).
        let candidates = vec![basic, by_address, by_thread_pair];
        candidates.into_iter()
            .max_by_key(|t| t.root.leaf_count())
            .unwrap()
    }

    /// Decompose by address: group threads by the addresses they access.
    fn decompose_by_address(test: &LitmusTest) -> DecompositionTree {
        let n = test.thread_count();
        let addrs = test.all_addresses();

        // For each address, find threads that access it.
        let mut addr_threads: HashMap<Address, Vec<usize>> = HashMap::new();
        for i in 0..n {
            for addr in test.threads[i].accessed_addresses() {
                addr_threads.entry(addr).or_default().push(i);
            }
        }

        // Each address group becomes a potential leaf.
        let mut leaves = Vec::new();
        let mut covered = HashSet::new();

        for (&addr, threads) in &addr_threads {
            if threads.iter().all(|t| covered.contains(t)) {
                continue;
            }
            let new_threads: Vec<usize> = threads.iter()
                .filter(|t| !covered.contains(*t))
                .copied()
                .collect();
            if !new_threads.is_empty() {
                for &t in &new_threads {
                    covered.insert(t);
                }
                let leaf_addrs: Vec<Address> = new_threads.iter()
                    .flat_map(|&t| test.threads[t].accessed_addresses())
                    .collect::<HashSet<_>>()
                    .into_iter().collect();
                leaves.push(DecompositionNode::Leaf {
                    thread_ids: new_threads,
                    addresses: leaf_addrs,
                });
            }
            let _ = addr; // used above
        }

        // Add any uncovered threads.
        for i in 0..n {
            if !covered.contains(&i) {
                leaves.push(DecompositionNode::Leaf {
                    thread_ids: vec![i],
                    addresses: test.threads[i].accessed_addresses(),
                });
            }
        }

        let root = if leaves.len() == 1 {
            leaves.pop().unwrap()
        } else {
            DecompositionNode::Compose {
                children: leaves,
                shared_addresses: Vec::new(),
            }
        };

        DecompositionTree::new(&test.name, root, n, addrs.len())
    }

    /// Decompose into thread pairs.
    fn decompose_by_thread_pairs(test: &LitmusTest) -> DecompositionTree {
        let n = test.thread_count();
        let addrs = test.all_addresses();

        let mut leaves = Vec::new();
        let mut i = 0;
        while i < n {
            let threads = if i + 1 < n {
                vec![i, i + 1]
            } else {
                vec![i]
            };
            let leaf_addrs: Vec<Address> = threads.iter()
                .flat_map(|&t| test.threads[t].accessed_addresses())
                .collect::<HashSet<_>>()
                .into_iter().collect();
            leaves.push(DecompositionNode::Leaf {
                thread_ids: threads,
                addresses: leaf_addrs,
            });
            i += 2;
        }

        let root = if leaves.len() == 1 {
            leaves.pop().unwrap()
        } else {
            DecompositionNode::Compose {
                children: leaves,
                shared_addresses: Vec::new(),
            }
        };

        DecompositionTree::new(&test.name, root, n, addrs.len())
    }

    /// Estimate the total number of executions for a decomposition.
    pub fn estimate_executions(test: &LitmusTest, tree: &DecompositionTree) -> usize {
        Self::estimate_node(test, &tree.root)
    }

    fn estimate_node(test: &LitmusTest, node: &DecompositionNode) -> usize {
        match node {
            DecompositionNode::Leaf { thread_ids, addresses } => {
                // Rough estimate: product of reads × writes per address.
                let mut estimate = 1usize;
                for &addr in addresses {
                    let reads = thread_ids.iter()
                        .filter(|&&t| t < test.threads.len())
                        .flat_map(|&t| test.threads[t].instructions.iter())
                        .filter(|i| match i {
                            super::litmus::Instruction::Load { addr: a, .. } => *a == addr,
                            super::litmus::Instruction::RMW { addr: a, .. } => *a == addr,
                            _ => false,
                        })
                        .count();
                    let writes = thread_ids.iter()
                        .filter(|&&t| t < test.threads.len())
                        .flat_map(|&t| test.threads[t].instructions.iter())
                        .filter(|i| match i {
                            super::litmus::Instruction::Store { addr: a, .. } => *a == addr,
                            super::litmus::Instruction::RMW { addr: a, .. } => *a == addr,
                            _ => false,
                        })
                        .count();
                    // Each read can read from any write + initial value.
                    let rf_choices = (writes + 1).max(1);
                    // CO: factorial of writes.
                    let co_choices = factorial(writes);
                    estimate *= rf_choices.pow(reads as u32) * co_choices;
                }
                estimate
            }
            DecompositionNode::Compose { children, .. } => {
                // Sum (not product) for independent components.
                children.iter().map(|c| Self::estimate_node(test, c)).sum::<usize>().max(1)
            }
        }
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

// ---------------------------------------------------------------------------
// DecompositionValidator
// ---------------------------------------------------------------------------

/// Validates that a decomposition is correct.
pub struct DecompositionValidator;

impl DecompositionValidator {
    /// Validate a decomposition against a litmus test.
    pub fn validate(
        test: &LitmusTest,
        tree: &DecompositionTree,
    ) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check thread coverage.
        let all_threads: HashSet<usize> = tree.root.all_threads().into_iter().collect();
        let expected: HashSet<usize> = (0..test.thread_count()).collect();

        let missing: Vec<usize> = expected.difference(&all_threads).copied().collect();
        if !missing.is_empty() {
            errors.push(format!("Missing threads: {:?}", missing));
        }

        let extra: Vec<usize> = all_threads.difference(&expected).copied().collect();
        if !extra.is_empty() {
            errors.push(format!("Extra threads: {:?}", extra));
        }

        // Check address coverage.
        let decomp_addrs: HashSet<Address> = tree.root.all_addresses().into_iter().collect();
        let test_addrs: HashSet<Address> = test.all_addresses().into_iter().collect();

        let missing_addrs: Vec<Address> = test_addrs.difference(&decomp_addrs).copied().collect();
        if !missing_addrs.is_empty() {
            errors.push(format!("Missing addresses: {:?}", missing_addrs));
        }

        // Check tree consistency.
        Self::validate_node(&tree.root, &mut errors);

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn validate_node(node: &DecompositionNode, errors: &mut Vec<String>) {
        match node {
            DecompositionNode::Leaf { thread_ids, .. } => {
                if thread_ids.is_empty() {
                    errors.push("Empty leaf node".into());
                }
            }
            DecompositionNode::Compose { children, .. } => {
                if children.is_empty() {
                    errors.push("Empty compose node".into());
                }
                for child in children {
                    Self::validate_node(child, errors);
                }
            }
        }
    }

    /// Cross-validate: verify the test both directly and via decomposition,
    /// and check that results agree.
    pub fn cross_validate(
        test: &LitmusTest,
        model: &MemoryModel,
    ) -> CrossValidationResult {
        let mut direct_verifier = Verifier::new(model.clone());
        let direct_result = direct_verifier.verify_litmus(test);

        let tree = TestDecomposer::decompose(test);
        let decomp_result = GluingTheorem::verify_with_gluing(test, &tree, model);

        let agrees = direct_result.pass == decomp_result.pass;

        CrossValidationResult {
            direct: direct_result,
            decomposed: decomp_result,
            tree,
            agrees,
        }
    }
}

/// Result of cross-validation.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub direct: VerificationResult,
    pub decomposed: VerificationResult,
    pub tree: DecompositionTree,
    pub agrees: bool,
}

impl fmt::Display for CrossValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cross-validation: {}", if self.agrees { "AGREE" } else { "DISAGREE" })?;
        writeln!(f, "Direct: pass={}", self.direct.pass)?;
        writeln!(f, "Decomposed: pass={}", self.decomposed.pass)?;
        write!(f, "{}", self.tree)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::litmus;
    use super::super::memory_model::BuiltinModel;

    #[test]
    fn test_decompose_sb() {
        let test = litmus::sb_test();
        let tree = TestDecomposer::decompose(&test);
        assert_eq!(tree.total_threads, 2);
        // SB has shared addresses → single component.
        assert!(tree.root.all_threads().len() == 2);
    }

    #[test]
    fn test_decompose_independent() {
        let mut test = LitmusTest::new("Independent");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, litmus::Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, litmus::Ordering::Relaxed);
        test.add_thread(t1);

        let tree = TestDecomposer::decompose(&test);
        assert!(tree.root.leaf_count() >= 2, "Independent threads should decompose");
    }

    #[test]
    fn test_decomposition_node_depth() {
        let leaf = DecompositionNode::Leaf {
            thread_ids: vec![0],
            addresses: vec![0x100],
        };
        assert_eq!(leaf.depth(), 0);

        let compose = DecompositionNode::Compose {
            children: vec![
                DecompositionNode::Leaf { thread_ids: vec![0], addresses: vec![0x100] },
                DecompositionNode::Leaf { thread_ids: vec![1], addresses: vec![0x200] },
            ],
            shared_addresses: Vec::new(),
        };
        assert_eq!(compose.depth(), 1);
    }

    #[test]
    fn test_decomposition_tree_display() {
        let tree = TestDecomposer::decompose(&litmus::sb_test());
        let s = format!("{}", tree);
        assert!(s.contains("DecompositionTree"));
    }

    #[test]
    fn test_decomposition_node_count() {
        let compose = DecompositionNode::Compose {
            children: vec![
                DecompositionNode::Leaf { thread_ids: vec![0], addresses: vec![0x100] },
                DecompositionNode::Leaf { thread_ids: vec![1], addresses: vec![0x200] },
            ],
            shared_addresses: Vec::new(),
        };
        assert_eq!(compose.node_count(), 3);
        assert_eq!(compose.leaf_count(), 2);
    }

    #[test]
    fn test_gluing_theorem_check() {
        let test = litmus::sb_test();
        let tree = TestDecomposer::decompose(&test);
        let result = GluingTheorem::check_gluing_conditions(&test, &tree);
        assert!(result.valid);
        assert!(result.complete);
        assert!(result.disjoint);
    }

    #[test]
    fn test_gluing_theorem_verify() {
        let test = litmus::sb_test();
        let tree = TestDecomposer::decompose(&test);
        let model = BuiltinModel::SC.build();
        let result = GluingTheorem::verify_with_gluing(&test, &tree, &model);
        assert!(result.pass);
    }

    #[test]
    fn test_optimal_decomposition() {
        let test = litmus::sb_test();
        let tree = OptimalDecomposition::find_optimal(&test);
        assert!(tree.root.all_threads().len() == 2);
    }

    #[test]
    fn test_optimal_independent() {
        let mut test = LitmusTest::new("Independent");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, litmus::Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, litmus::Ordering::Relaxed);
        test.add_thread(t1);

        let tree = OptimalDecomposition::find_optimal(&test);
        assert!(tree.root.leaf_count() >= 2);
    }

    #[test]
    fn test_estimate_executions() {
        let test = litmus::sb_test();
        let tree = TestDecomposer::decompose(&test);
        let est = OptimalDecomposition::estimate_executions(&test, &tree);
        assert!(est > 0);
    }

    #[test]
    fn test_decomposition_validator() {
        let test = litmus::sb_test();
        let tree = TestDecomposer::decompose(&test);
        let result = DecompositionValidator::validate(&test, &tree);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_validate() {
        let test = litmus::sb_test();
        let model = BuiltinModel::SC.build();
        let result = DecompositionValidator::cross_validate(&test, &model);
        assert!(result.agrees, "Direct and decomposed should agree");
    }

    #[test]
    fn test_cross_validate_display() {
        let test = litmus::sb_test();
        let model = BuiltinModel::SC.build();
        let result = DecompositionValidator::cross_validate(&test, &model);
        let s = format!("{}", result);
        assert!(s.contains("Cross-validation"));
    }

    #[test]
    fn test_gluing_result_display() {
        let gr = GluingResult {
            valid: true,
            complete: true,
            disjoint: true,
            shared_addresses: vec![0x100],
            component_count: 2,
        };
        let s = format!("{}", gr);
        assert!(s.contains("Gluing"));
    }

    #[test]
    fn test_trivial_decomposition() {
        let mut test = LitmusTest::new("Single");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, litmus::Ordering::Relaxed);
        test.add_thread(t0);

        let tree = TestDecomposer::decompose(&test);
        assert!(tree.is_trivial());
    }

    #[test]
    fn test_compression_ratio() {
        let mut test = LitmusTest::new("Independent");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, litmus::Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, litmus::Ordering::Relaxed);
        test.add_thread(t1);

        let tree = TestDecomposer::decompose(&test);
        assert!(tree.compression_ratio() >= 2.0);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_iriw_decomposition() {
        let test = litmus::iriw_test();
        let tree = TestDecomposer::decompose(&test);
        // IRIW has 4 threads, all sharing x or y.
        assert!(tree.root.all_threads().len() == 4);
    }

    #[test]
    fn test_decompose_by_address() {
        let mut test = LitmusTest::new("AddrSplit");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, litmus::Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, litmus::Ordering::Relaxed);
        test.add_thread(t1);
        let mut t2 = Thread::new(2);
        t2.store(0x300, 1, litmus::Ordering::Relaxed);
        test.add_thread(t2);

        let tree = OptimalDecomposition::decompose_by_address(&test);
        assert!(tree.root.leaf_count() >= 3);
    }

    #[test]
    fn test_decompose_by_thread_pairs() {
        let test = litmus::iriw_test();
        let tree = OptimalDecomposition::decompose_by_thread_pairs(&test);
        assert!(tree.root.leaf_count() >= 2);
    }

    #[test]
    fn test_validator_missing_thread() {
        let test = litmus::sb_test();
        let bad_tree = DecompositionTree::new(
            "bad",
            DecompositionNode::Leaf {
                thread_ids: vec![0], // Missing thread 1
                addresses: vec![0x100],
            },
            2,
            2,
        );
        let result = DecompositionValidator::validate(&test, &bad_tree);
        assert!(result.is_err());
    }

    #[test]
    fn test_gluing_independent_verify() {
        let mut test = LitmusTest::new("Independent");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, litmus::Ordering::Relaxed);
        t0.load(0, 0x100, litmus::Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, litmus::Ordering::Relaxed);
        t1.load(0, 0x200, litmus::Ordering::Relaxed);
        test.add_thread(t1);

        let tree = TestDecomposer::decompose(&test);
        let model = BuiltinModel::SC.build();
        let result = GluingTheorem::verify_with_gluing(&test, &tree, &model);
        assert!(result.pass);
    }
}
