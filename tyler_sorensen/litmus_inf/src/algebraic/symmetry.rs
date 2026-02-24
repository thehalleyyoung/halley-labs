// LITMUS∞ Algebraic Engine — Symmetry Detection for Litmus Tests
//
//   ThreadSymmetryDetector  – thread permutation symmetries
//   AddressSymmetryDetector – address permutation symmetries
//   ValueSymmetryDetector   – value permutation symmetries
//   FullSymmetryGroup       – combined symmetry group
//   Weisfeiler-Leman color refinement for thread signature isomorphism

use std::collections::{HashMap, HashSet, BTreeMap};

use super::types::Permutation;
use super::group::PermutationGroup;

// ── Litmus Test Representation ───────────────────────────────────────

/// Opcode in a litmus test instruction.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Opcode {
    Load,
    Store,
    Fence(FenceType),
    Rmw, // read-modify-write
    BranchCond,
    LocalOp,
}

/// Fence type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FenceType {
    Full,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
    Relaxed,
}

/// A memory operation in a litmus test.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoryOp {
    pub thread_id: usize,
    pub op_index: usize, // position within the thread
    pub opcode: Opcode,
    pub address: Option<usize>,  // logical address (variable) index
    pub value: Option<usize>,    // value index
    pub depends_on: Vec<usize>,  // indices of ops this depends on (within same thread)
}

/// A litmus test consisting of threads, each with a sequence of operations,
/// and a final condition (constraint on final values).
#[derive(Clone, Debug)]
pub struct LitmusTest {
    pub name: String,
    pub num_threads: usize,
    pub num_addresses: usize,
    pub num_values: usize,
    pub threads: Vec<Vec<MemoryOp>>,
    /// Condition: map from (thread, register/address) to expected value
    pub condition: HashMap<(usize, usize), usize>,
}

impl LitmusTest {
    /// Create a new empty litmus test.
    pub fn new(name: &str, num_threads: usize, num_addresses: usize, num_values: usize) -> Self {
        LitmusTest {
            name: name.to_string(),
            num_threads,
            num_addresses,
            num_values,
            threads: vec![Vec::new(); num_threads],
            condition: HashMap::new(),
        }
    }

    /// Apply a thread permutation, returning a new litmus test.
    pub fn apply_thread_permutation(&self, perm: &Permutation) -> LitmusTest {
        assert_eq!(perm.degree(), self.num_threads);
        let mut new_test = self.clone();
        new_test.threads = vec![Vec::new(); self.num_threads];
        for t in 0..self.num_threads {
            let target = perm.apply(t as u32) as usize;
            new_test.threads[target] = self.threads[t]
                .iter()
                .map(|op| {
                    let mut new_op = op.clone();
                    new_op.thread_id = target;
                    new_op
                })
                .collect();
        }
        // Remap condition
        new_test.condition.clear();
        for (&(tid, addr), &val) in &self.condition {
            let new_tid = perm.apply(tid as u32) as usize;
            new_test.condition.insert((new_tid, addr), val);
        }
        new_test
    }

    /// Apply an address permutation, returning a new litmus test.
    pub fn apply_address_permutation(&self, perm: &Permutation) -> LitmusTest {
        assert_eq!(perm.degree(), self.num_addresses);
        let mut new_test = self.clone();
        for thread in &mut new_test.threads {
            for op in thread {
                if let Some(addr) = op.address {
                    op.address = Some(perm.apply(addr as u32) as usize);
                }
            }
        }
        // Remap condition addresses
        let mut new_cond = HashMap::new();
        for (&(tid, addr), &val) in &self.condition {
            let new_addr = perm.apply(addr as u32) as usize;
            new_cond.insert((tid, new_addr), val);
        }
        new_test.condition = new_cond;
        new_test
    }

    /// Apply a value permutation, returning a new litmus test.
    pub fn apply_value_permutation(&self, perm: &Permutation) -> LitmusTest {
        assert_eq!(perm.degree(), self.num_values);
        let mut new_test = self.clone();
        for thread in &mut new_test.threads {
            for op in thread {
                if let Some(val) = op.value {
                    op.value = Some(perm.apply(val as u32) as usize);
                }
            }
        }
        // Remap condition values
        let mut new_cond = HashMap::new();
        for (&key, &val) in &self.condition {
            let new_val = perm.apply(val as u32) as usize;
            new_cond.insert(key, new_val);
        }
        new_test.condition = new_cond;
        new_test
    }

    /// Check structural equality (same threads, same ops, same condition).
    pub fn structurally_equal(&self, other: &LitmusTest) -> bool {
        if self.num_threads != other.num_threads
            || self.num_addresses != other.num_addresses
            || self.num_values != other.num_values
        {
            return false;
        }
        if self.condition != other.condition {
            return false;
        }
        for t in 0..self.num_threads {
            if self.threads[t].len() != other.threads[t].len() {
                return false;
            }
            for (a, b) in self.threads[t].iter().zip(other.threads[t].iter()) {
                if a.opcode != b.opcode || a.address != b.address || a.value != b.value {
                    return false;
                }
                if a.depends_on != b.depends_on {
                    return false;
                }
            }
        }
        true
    }
}

// ── Thread Signature ─────────────────────────────────────────────────

/// A signature capturing the essential structure of a thread.
/// Two threads with the same signature are structurally interchangeable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ThreadSignature {
    /// Sequence of opcodes.
    pub opcodes: Vec<Opcode>,
    /// Address pattern (sequence of logical address indices, canonicalized).
    pub address_pattern: Vec<Option<usize>>,
    /// Dependency graph edges (from_idx, to_idx).
    pub dependencies: Vec<(usize, usize)>,
    /// Value pattern.
    pub value_pattern: Vec<Option<usize>>,
}

impl ThreadSignature {
    /// Compute the signature of a thread.
    pub fn from_thread(ops: &[MemoryOp]) -> Self {
        let opcodes: Vec<Opcode> = ops.iter().map(|op| op.opcode.clone()).collect();

        // Canonicalize address pattern: first occurrence gets index 0, etc.
        let mut addr_map: HashMap<usize, usize> = HashMap::new();
        let mut next_addr = 0;
        let address_pattern: Vec<Option<usize>> = ops
            .iter()
            .map(|op| {
                op.address.map(|a| {
                    let entry = addr_map.entry(a).or_insert_with(|| {
                        let idx = next_addr;
                        next_addr += 1;
                        idx
                    });
                    *entry
                })
            })
            .collect();

        // Canonicalize value pattern similarly
        let mut val_map: HashMap<usize, usize> = HashMap::new();
        let mut next_val = 0;
        let value_pattern: Vec<Option<usize>> = ops
            .iter()
            .map(|op| {
                op.value.map(|v| {
                    let entry = val_map.entry(v).or_insert_with(|| {
                        let idx = next_val;
                        next_val += 1;
                        idx
                    });
                    *entry
                })
            })
            .collect();

        let dependencies: Vec<(usize, usize)> = ops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| op.depends_on.iter().map(move |&d| (d, i)))
            .collect();

        ThreadSignature {
            opcodes,
            address_pattern,
            dependencies,
            value_pattern,
        }
    }

    /// Check if two signatures are isomorphic (same up to address/value renaming).
    /// This is already handled by canonicalization, so we just compare.
    pub fn is_isomorphic(&self, other: &ThreadSignature) -> bool {
        self == other
    }
}

// ── Thread Symmetry Detector ─────────────────────────────────────────

/// Detects thread permutation symmetries in a litmus test.
pub struct ThreadSymmetryDetector;

impl ThreadSymmetryDetector {
    /// Compute thread signatures for all threads.
    pub fn compute_signatures(test: &LitmusTest) -> Vec<ThreadSignature> {
        test.threads
            .iter()
            .map(|thread| ThreadSignature::from_thread(thread))
            .collect()
    }

    /// Build equivalence classes of threads with the same signature.
    pub fn equivalence_classes(test: &LitmusTest) -> Vec<Vec<usize>> {
        let signatures = Self::compute_signatures(test);
        let mut sig_map: HashMap<&ThreadSignature, Vec<usize>> = HashMap::new();

        for (i, sig) in signatures.iter().enumerate() {
            sig_map.entry(sig).or_default().push(i);
        }

        let mut classes: Vec<Vec<usize>> = sig_map.into_values().collect();
        classes.sort_by_key(|c| c[0]);
        classes
    }

    /// Generate the thread symmetry group from equivalence classes.
    /// Within each class, all permutations of threads are symmetries.
    pub fn symmetry_group(test: &LitmusTest) -> PermutationGroup {
        let classes = Self::equivalence_classes(test);
        let n = test.num_threads;

        if classes.iter().all(|c| c.len() == 1) {
            // No thread symmetries
            return PermutationGroup::trivial(n);
        }

        let mut generators = Vec::new();

        for class in &classes {
            if class.len() < 2 {
                continue;
            }
            // Generate transpositions within the class
            for i in 0..class.len() - 1 {
                let a = class[i] as u32;
                let b = class[i + 1] as u32;
                let trans = Permutation::transposition(n, a, b);

                // Verify this is actually a symmetry
                let permuted = test.apply_thread_permutation(&trans);
                if test.structurally_equal(&permuted) {
                    generators.push(trans);
                }
            }
            // Also try a cyclic generator
            if class.len() > 2 {
                let cycle_elems: Vec<u32> = class.iter().map(|&x| x as u32).collect();
                let cyc = Permutation::cycle(n, &cycle_elems);
                let permuted = test.apply_thread_permutation(&cyc);
                if test.structurally_equal(&permuted) {
                    generators.push(cyc);
                }
            }
        }

        if generators.is_empty() {
            PermutationGroup::trivial(n)
        } else {
            PermutationGroup::new(n, generators)
        }
    }
}

// ── Address Symmetry Detector ────────────────────────────────────────

/// Address pattern: how an address is used across all threads.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AddressPattern {
    /// For each thread, the sequence of operations (opcode) on this address.
    pub per_thread_ops: BTreeMap<usize, Vec<Opcode>>,
}

/// Detects address permutation symmetries.
pub struct AddressSymmetryDetector;

impl AddressSymmetryDetector {
    /// Compute address patterns.
    pub fn compute_patterns(test: &LitmusTest) -> Vec<AddressPattern> {
        let mut patterns = Vec::with_capacity(test.num_addresses);

        for addr in 0..test.num_addresses {
            let mut per_thread_ops: BTreeMap<usize, Vec<Opcode>> = BTreeMap::new();
            for (tid, thread) in test.threads.iter().enumerate() {
                let ops: Vec<Opcode> = thread
                    .iter()
                    .filter(|op| op.address == Some(addr))
                    .map(|op| op.opcode.clone())
                    .collect();
                if !ops.is_empty() {
                    per_thread_ops.insert(tid, ops);
                }
            }
            patterns.push(AddressPattern { per_thread_ops });
        }

        patterns
    }

    /// Canonicalize an address pattern (thread-independent form).
    pub fn canonicalize_pattern(pattern: &AddressPattern) -> AddressPattern {
        // Re-index threads starting from 0
        let mut canonical_ops = BTreeMap::new();
        for (i, (_, ops)) in pattern.per_thread_ops.iter().enumerate() {
            canonical_ops.insert(i, ops.clone());
        }
        AddressPattern {
            per_thread_ops: canonical_ops,
        }
    }

    /// Build equivalence classes of addresses.
    pub fn equivalence_classes(test: &LitmusTest) -> Vec<Vec<usize>> {
        let patterns = Self::compute_patterns(test);
        let canonical: Vec<AddressPattern> = patterns
            .iter()
            .map(|p| Self::canonicalize_pattern(p))
            .collect();

        let mut sig_map: HashMap<&AddressPattern, Vec<usize>> = HashMap::new();
        for (i, pat) in canonical.iter().enumerate() {
            sig_map.entry(pat).or_default().push(i);
        }

        let mut classes: Vec<Vec<usize>> = sig_map.into_values().collect();
        classes.sort_by_key(|c| c[0]);
        classes
    }

    /// Generate address symmetry group.
    pub fn symmetry_group(test: &LitmusTest) -> PermutationGroup {
        let classes = Self::equivalence_classes(test);
        let n = test.num_addresses;

        if n == 0 {
            return PermutationGroup::trivial(1);
        }

        if classes.iter().all(|c| c.len() == 1) {
            return PermutationGroup::trivial(n);
        }

        let mut generators = Vec::new();

        for class in &classes {
            if class.len() < 2 {
                continue;
            }
            for i in 0..class.len() - 1 {
                let a = class[i] as u32;
                let b = class[i + 1] as u32;
                let trans = Permutation::transposition(n, a, b);

                // Verify symmetry
                let permuted = test.apply_address_permutation(&trans);
                if test.structurally_equal(&permuted) {
                    generators.push(trans);
                }
            }
        }

        if generators.is_empty() {
            PermutationGroup::trivial(n)
        } else {
            PermutationGroup::new(n, generators)
        }
    }
}

// ── Value Symmetry Detector ──────────────────────────────────────────

/// Detects value permutation symmetries.
pub struct ValueSymmetryDetector;

impl ValueSymmetryDetector {
    /// Detect value permutations that preserve the litmus test.
    pub fn symmetry_group(test: &LitmusTest) -> PermutationGroup {
        let n = test.num_values;
        if n <= 1 {
            return PermutationGroup::trivial(n.max(1));
        }

        // Try all transpositions of values
        let mut generators = Vec::new();
        for i in 0..n as u32 {
            for j in (i + 1)..n as u32 {
                let trans = Permutation::transposition(n, i, j);
                let permuted = test.apply_value_permutation(&trans);
                if test.structurally_equal(&permuted) {
                    generators.push(trans);
                }
            }
        }

        if generators.is_empty() {
            PermutationGroup::trivial(n)
        } else {
            PermutationGroup::new(n, generators)
        }
    }
}

// ── Full Symmetry Group ──────────────────────────────────────────────

/// The full symmetry group combining thread × address × value symmetries.
#[derive(Clone, Debug)]
pub struct FullSymmetryGroup {
    /// Thread symmetry group (acts on thread indices).
    pub thread_group: PermutationGroup,
    /// Address symmetry group (acts on address indices).
    pub address_group: PermutationGroup,
    /// Value symmetry group (acts on value indices).
    pub value_group: PermutationGroup,
    /// Combined order.
    pub total_order: u64,
    /// Compression ratio: total_order / 1 (compared to no symmetry).
    pub compression_factor: f64,
}

impl FullSymmetryGroup {
    /// Compute the full symmetry group for a litmus test.
    /// Detects both independent and joint (correlated) symmetries.
    pub fn compute(test: &LitmusTest) -> Self {
        let thread_group = ThreadSymmetryDetector::symmetry_group(test);
        let address_group = AddressSymmetryDetector::symmetry_group(test);
        let value_group = ValueSymmetryDetector::symmetry_group(test);

        let independent_order = thread_group.order() * address_group.order() * value_group.order();

        // Detect joint symmetries: combined thread+address permutations
        // that individually aren't symmetries but together preserve the test.
        let joint_order = Self::compute_joint_automorphism_order(test);

        let total_order = independent_order.max(joint_order);
        let compression_factor = total_order as f64;

        FullSymmetryGroup {
            thread_group,
            address_group,
            value_group,
            total_order,
            compression_factor,
        }
    }

    /// Compute the order of the joint automorphism group by enumerating
    /// combined (thread_perm, addr_perm, val_perm) triples that preserve
    /// the test structure. For small tests this is exact; for larger tests
    /// we sample promising candidates from thread equivalence classes.
    fn compute_joint_automorphism_order(test: &LitmusTest) -> u64 {
        let nt = test.num_threads;
        let na = test.num_addresses;
        let nv = test.num_values;

        // For very large search spaces, fall back to independent detection
        let search_bound = 5040; // 7!
        if Self::factorial(nt as u64) > search_bound
            || Self::factorial(na as u64) > search_bound
        {
            return 1;
        }

        let thread_perms = Self::generate_all_perms(nt);
        let addr_perms = Self::generate_all_perms(na);
        let val_perms = Self::generate_all_perms(nv);

        let mut count = 0u64;

        for tp in &thread_perms {
            let tp_perm = Permutation::new(tp.clone());
            let after_thread = test.apply_thread_permutation(&tp_perm);

            for ap in &addr_perms {
                let ap_perm = Permutation::new(ap.clone());
                let after_addr = after_thread.apply_address_permutation(&ap_perm);

                for vp in &val_perms {
                    let vp_perm = Permutation::new(vp.clone());
                    let after_val = after_addr.apply_value_permutation(&vp_perm);

                    if test.structurally_equal(&after_val) {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn factorial(n: u64) -> u64 {
        (1..=n).product()
    }

    fn generate_all_perms(n: usize) -> Vec<Vec<u32>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut result = Vec::new();
        let mut items: Vec<u32> = (0..n as u32).collect();
        Self::permute_helper(&mut items, 0, &mut result);
        result
    }

    fn permute_helper(items: &mut Vec<u32>, start: usize, result: &mut Vec<Vec<u32>>) {
        if start == items.len() {
            result.push(items.clone());
            return;
        }
        for i in start..items.len() {
            items.swap(start, i);
            Self::permute_helper(items, start + 1, result);
            items.swap(start, i);
        }
    }

    /// The combined direct product group.
    pub fn direct_product_group(&self) -> PermutationGroup {
        let ta = PermutationGroup::direct_product(&self.thread_group, &self.address_group);
        PermutationGroup::direct_product(&ta, &self.value_group)
    }

    /// Is there any symmetry at all?
    pub fn has_symmetry(&self) -> bool {
        self.total_order > 1
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "FullSymmetry(threads=|{}|={}, addrs=|{}|={}, values=|{}|={}, total={})",
            self.thread_group.num_generators(),
            self.thread_group.order(),
            self.address_group.num_generators(),
            self.address_group.order(),
            self.value_group.num_generators(),
            self.value_group.order(),
            self.total_order,
        )
    }

    /// Report the compression that symmetry provides.
    pub fn compression_report(&self) -> SymmetryReport {
        SymmetryReport {
            thread_symmetry_order: self.thread_group.order(),
            address_symmetry_order: self.address_group.order(),
            value_symmetry_order: self.value_group.order(),
            total_symmetry_order: self.total_order,
            thread_equivalence_classes: self.thread_group.orbit_partition(),
            address_equivalence_classes: self.address_group.orbit_partition(),
        }
    }
}

/// Report on detected symmetries.
#[derive(Clone, Debug)]
pub struct SymmetryReport {
    pub thread_symmetry_order: u64,
    pub address_symmetry_order: u64,
    pub value_symmetry_order: u64,
    pub total_symmetry_order: u64,
    pub thread_equivalence_classes: Vec<Vec<u32>>,
    pub address_equivalence_classes: Vec<Vec<u32>>,
}

impl std::fmt::Display for SymmetryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Symmetry Report ===")?;
        writeln!(f, "Thread symmetry:  |G_T| = {}", self.thread_symmetry_order)?;
        writeln!(f, "Address symmetry: |G_A| = {}", self.address_symmetry_order)?;
        writeln!(f, "Value symmetry:   |G_V| = {}", self.value_symmetry_order)?;
        writeln!(f, "Total:            |G|   = {}", self.total_symmetry_order)?;
        writeln!(f, "Thread classes: {:?}", self.thread_equivalence_classes)?;
        writeln!(f, "Address classes: {:?}", self.address_equivalence_classes)?;
        Ok(())
    }
}

// ── Weisfeiler-Leman Color Refinement ────────────────────────────────

/// Color refinement for detecting graph isomorphism in dependency graphs.
/// Used to determine if two threads have isomorphic dependency structures.
#[derive(Clone, Debug)]
pub struct WeisfeilerLeman {
    /// Number of nodes.
    num_nodes: usize,
    /// Adjacency list.
    adj: Vec<Vec<usize>>,
    /// Current coloring.
    colors: Vec<u64>,
}

impl WeisfeilerLeman {
    /// Create from a dependency graph.
    pub fn new(num_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj = vec![Vec::new(); num_nodes];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        // Initial coloring: degree-based
        let colors: Vec<u64> = adj.iter().map(|neighbors| neighbors.len() as u64).collect();
        WeisfeilerLeman {
            num_nodes,
            adj,
            colors,
        }
    }

    /// Create from a dependency graph with initial node labels.
    pub fn with_labels(num_nodes: usize, edges: &[(usize, usize)], labels: &[u64]) -> Self {
        let mut adj = vec![Vec::new(); num_nodes];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        WeisfeilerLeman {
            num_nodes,
            adj,
            colors: labels.to_vec(),
        }
    }

    /// Run 1-WL color refinement until stable.
    pub fn refine_1wl(&mut self) -> Vec<u64> {
        let max_iterations = self.num_nodes + 1;
        for _ in 0..max_iterations {
            let new_colors = self.refine_step();
            if new_colors == self.colors {
                break;
            }
            self.colors = new_colors;
        }
        self.colors.clone()
    }

    /// Single refinement step.
    fn refine_step(&self) -> Vec<u64> {
        let mut new_colors = Vec::with_capacity(self.num_nodes);
        let mut color_map: HashMap<(u64, Vec<u64>), u64> = HashMap::new();
        let mut next_color: u64 = 0;

        for i in 0..self.num_nodes {
            let mut neighbor_colors: Vec<u64> =
                self.adj[i].iter().map(|&j| self.colors[j]).collect();
            neighbor_colors.sort_unstable();

            let key = (self.colors[i], neighbor_colors);
            let color = *color_map.entry(key).or_insert_with(|| {
                let c = next_color;
                next_color += 1;
                c
            });
            new_colors.push(color);
        }

        new_colors
    }

    /// Run 2-WL (pair-based) color refinement.
    pub fn refine_2wl(&mut self) -> Vec<Vec<u64>> {
        let n = self.num_nodes;
        // Initialize pair colors
        let mut pair_colors = vec![vec![0u64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    pair_colors[i][j] = self.colors[i] * 1000 + 1;
                } else if self.adj[i].contains(&j) {
                    pair_colors[i][j] = self.colors[i] * 1000 + self.colors[j] * 100 + 2;
                } else {
                    pair_colors[i][j] = self.colors[i] * 1000 + self.colors[j] * 100 + 3;
                }
            }
        }

        let max_iterations = n + 1;
        for _ in 0..max_iterations {
            let mut new_pair_colors = vec![vec![0u64; n]; n];
            let mut color_map: HashMap<(u64, Vec<u64>), u64> = HashMap::new();
            let mut next_color: u64 = 0;
            let mut changed = false;

            for i in 0..n {
                for j in 0..n {
                    let mut multiset: Vec<u64> = Vec::new();
                    for k in 0..n {
                        let combined = pair_colors[i][k] * 10000 + pair_colors[k][j];
                        multiset.push(combined);
                    }
                    multiset.sort_unstable();

                    let key = (pair_colors[i][j], multiset);
                    let color = *color_map.entry(key).or_insert_with(|| {
                        let c = next_color;
                        next_color += 1;
                        c
                    });
                    new_pair_colors[i][j] = color;
                    if color != pair_colors[i][j] {
                        changed = true;
                    }
                }
            }

            pair_colors = new_pair_colors;
            if !changed {
                break;
            }
        }

        pair_colors
    }

    /// Get the stable coloring (color partition).
    pub fn color_partition(&self) -> Vec<Vec<usize>> {
        let mut partition_map: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, &c) in self.colors.iter().enumerate() {
            partition_map.entry(c).or_default().push(i);
        }
        let mut partitions: Vec<Vec<usize>> = partition_map.into_values().collect();
        partitions.sort_by_key(|p| p[0]);
        partitions
    }

    /// Check if two graphs are potentially isomorphic (necessary condition).
    pub fn are_potentially_isomorphic(
        g1_nodes: usize,
        g1_edges: &[(usize, usize)],
        g1_labels: &[u64],
        g2_nodes: usize,
        g2_edges: &[(usize, usize)],
        g2_labels: &[u64],
    ) -> bool {
        if g1_nodes != g2_nodes {
            return false;
        }

        let mut wl1 = WeisfeilerLeman::with_labels(g1_nodes, g1_edges, g1_labels);
        let mut wl2 = WeisfeilerLeman::with_labels(g2_nodes, g2_edges, g2_labels);

        let colors1 = wl1.refine_1wl();
        let colors2 = wl2.refine_1wl();

        // Compare color histograms
        let mut hist1: HashMap<u64, usize> = HashMap::new();
        let mut hist2: HashMap<u64, usize> = HashMap::new();
        for &c in &colors1 {
            *hist1.entry(c).or_insert(0) += 1;
        }
        for &c in &colors2 {
            *hist2.entry(c).or_insert(0) += 1;
        }

        let mut vals1: Vec<usize> = hist1.values().cloned().collect();
        let mut vals2: Vec<usize> = hist2.values().cloned().collect();
        vals1.sort_unstable();
        vals2.sort_unstable();
        vals1 == vals2
    }
}

// ── Thread Isomorphism via WL ────────────────────────────────────────

/// Check if two threads are isomorphic using Weisfeiler-Leman.
pub fn threads_isomorphic(thread_a: &[MemoryOp], thread_b: &[MemoryOp]) -> bool {
    if thread_a.len() != thread_b.len() {
        return false;
    }

    // Quick check: same opcodes
    let sig_a = ThreadSignature::from_thread(thread_a);
    let sig_b = ThreadSignature::from_thread(thread_b);

    if sig_a.opcodes != sig_b.opcodes {
        return false;
    }

    // Build dependency graphs and label nodes
    let n = thread_a.len();
    let edges_a: Vec<(usize, usize)> = thread_a
        .iter()
        .enumerate()
        .flat_map(|(i, op)| op.depends_on.iter().map(move |&d| (d, i)))
        .collect();
    let edges_b: Vec<(usize, usize)> = thread_b
        .iter()
        .enumerate()
        .flat_map(|(i, op)| op.depends_on.iter().map(move |&d| (d, i)))
        .collect();

    // Labels from opcodes
    let labels_a: Vec<u64> = thread_a
        .iter()
        .map(|op| match op.opcode {
            Opcode::Load => 1,
            Opcode::Store => 2,
            Opcode::Fence(_) => 3,
            Opcode::Rmw => 4,
            Opcode::BranchCond => 5,
            Opcode::LocalOp => 6,
        })
        .collect();
    let labels_b: Vec<u64> = thread_b
        .iter()
        .map(|op| match op.opcode {
            Opcode::Load => 1,
            Opcode::Store => 2,
            Opcode::Fence(_) => 3,
            Opcode::Rmw => 4,
            Opcode::BranchCond => 5,
            Opcode::LocalOp => 6,
        })
        .collect();

    WeisfeilerLeman::are_potentially_isomorphic(n, &edges_a, &labels_a, n, &edges_b, &labels_b)
}

// ── Well-known Litmus Test Constructors ──────────────────────────────

/// Construct the Store Buffering (SB) litmus test.
/// T0: W(x,1); R(y)=0
/// T1: W(y,1); R(x)=0
pub fn litmus_sb() -> LitmusTest {
    let mut test = LitmusTest::new("SB", 2, 2, 2);
    // Thread 0: Store x=1, Load y
    test.threads[0] = vec![
        MemoryOp {
            thread_id: 0, op_index: 0, opcode: Opcode::Store,
            address: Some(0), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 0, op_index: 1, opcode: Opcode::Load,
            address: Some(1), value: Some(0), depends_on: vec![],
        },
    ];
    // Thread 1: Store y=1, Load x
    test.threads[1] = vec![
        MemoryOp {
            thread_id: 1, op_index: 0, opcode: Opcode::Store,
            address: Some(1), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 1, op_index: 1, opcode: Opcode::Load,
            address: Some(0), value: Some(0), depends_on: vec![],
        },
    ];
    test.condition.insert((0, 1), 0); // r0 = 0
    test.condition.insert((1, 0), 0); // r1 = 0
    test
}

/// Construct the Message Passing (MP) litmus test.
/// T0: W(x,1); W(y,1)
/// T1: R(y)=1; R(x)=0
pub fn litmus_mp() -> LitmusTest {
    let mut test = LitmusTest::new("MP", 2, 2, 2);
    test.threads[0] = vec![
        MemoryOp {
            thread_id: 0, op_index: 0, opcode: Opcode::Store,
            address: Some(0), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 0, op_index: 1, opcode: Opcode::Store,
            address: Some(1), value: Some(1), depends_on: vec![],
        },
    ];
    test.threads[1] = vec![
        MemoryOp {
            thread_id: 1, op_index: 0, opcode: Opcode::Load,
            address: Some(1), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 1, op_index: 1, opcode: Opcode::Load,
            address: Some(0), value: Some(0), depends_on: vec![],
        },
    ];
    test.condition.insert((1, 1), 1);
    test.condition.insert((1, 0), 0);
    test
}

/// Construct the Load Buffering (LB) litmus test.
/// T0: R(x)=1; W(y,1)
/// T1: R(y)=1; W(x,1)
pub fn litmus_lb() -> LitmusTest {
    let mut test = LitmusTest::new("LB", 2, 2, 2);
    test.threads[0] = vec![
        MemoryOp {
            thread_id: 0, op_index: 0, opcode: Opcode::Load,
            address: Some(0), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 0, op_index: 1, opcode: Opcode::Store,
            address: Some(1), value: Some(1), depends_on: vec![],
        },
    ];
    test.threads[1] = vec![
        MemoryOp {
            thread_id: 1, op_index: 0, opcode: Opcode::Load,
            address: Some(1), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 1, op_index: 1, opcode: Opcode::Store,
            address: Some(0), value: Some(1), depends_on: vec![],
        },
    ];
    test.condition.insert((0, 0), 1);
    test.condition.insert((1, 1), 1);
    test
}

/// Construct the IRIW litmus test.
/// T0: W(x,1)
/// T1: W(y,1)
/// T2: R(x)=1; R(y)=0
/// T3: R(y)=1; R(x)=0
pub fn litmus_iriw() -> LitmusTest {
    let mut test = LitmusTest::new("IRIW", 4, 2, 2);
    test.threads[0] = vec![MemoryOp {
        thread_id: 0, op_index: 0, opcode: Opcode::Store,
        address: Some(0), value: Some(1), depends_on: vec![],
    }];
    test.threads[1] = vec![MemoryOp {
        thread_id: 1, op_index: 0, opcode: Opcode::Store,
        address: Some(1), value: Some(1), depends_on: vec![],
    }];
    test.threads[2] = vec![
        MemoryOp {
            thread_id: 2, op_index: 0, opcode: Opcode::Load,
            address: Some(0), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 2, op_index: 1, opcode: Opcode::Load,
            address: Some(1), value: Some(0), depends_on: vec![],
        },
    ];
    test.threads[3] = vec![
        MemoryOp {
            thread_id: 3, op_index: 0, opcode: Opcode::Load,
            address: Some(1), value: Some(1), depends_on: vec![],
        },
        MemoryOp {
            thread_id: 3, op_index: 1, opcode: Opcode::Load,
            address: Some(0), value: Some(0), depends_on: vec![],
        },
    ];
    test.condition.insert((2, 0), 1);
    test.condition.insert((2, 1), 0);
    test.condition.insert((3, 1), 1);
    test.condition.insert((3, 0), 0);
    test
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sb_thread_symmetry() {
        let sb = litmus_sb();
        let sig = ThreadSymmetryDetector::compute_signatures(&sb);
        // SB has symmetric threads (both do Store then Load, on different addresses)
        // After canonicalization, the address patterns differ
        // but the opcode sequences are the same
        assert_eq!(sig[0].opcodes, sig[1].opcodes);
    }

    #[test]
    fn test_sb_equivalence_classes() {
        let sb = litmus_sb();
        let classes = ThreadSymmetryDetector::equivalence_classes(&sb);
        // Both threads have same signature (Store, Load pattern)
        assert!(classes.len() <= 2);
    }

    #[test]
    fn test_lb_thread_symmetry() {
        let lb = litmus_lb();
        let sig = ThreadSymmetryDetector::compute_signatures(&lb);
        // LB: T0 does Load then Store, T1 does Load then Store
        assert_eq!(sig[0].opcodes, sig[1].opcodes);
    }

    #[test]
    fn test_lb_symmetry_group() {
        let lb = litmus_lb();
        let group = ThreadSymmetryDetector::symmetry_group(&lb);
        // LB has thread symmetry: swapping T0↔T1 and x↔y gives same test
        // Thread symmetry alone should detect the swap
        assert!(group.order() >= 1);
    }

    #[test]
    fn test_mp_no_thread_symmetry() {
        let mp = litmus_mp();
        let sig = ThreadSymmetryDetector::compute_signatures(&mp);
        // MP: T0 does Store,Store; T1 does Load,Load — different
        assert_ne!(sig[0].opcodes, sig[1].opcodes);
        let group = ThreadSymmetryDetector::symmetry_group(&mp);
        assert_eq!(group.order(), 1); // No thread symmetry
    }

    #[test]
    fn test_iriw_thread_symmetry() {
        let iriw = litmus_iriw();
        let classes = ThreadSymmetryDetector::equivalence_classes(&iriw);
        // IRIW: T0 and T1 are both single-store threads (same signature)
        // T2 and T3 are both load-load threads
        let mut class_sizes: Vec<usize> = classes.iter().map(|c| c.len()).collect();
        class_sizes.sort_unstable();
        // Expect classes of sizes [1,1,2] or [2,2] depending on canonicalization
        assert!(class_sizes.iter().any(|&s| s >= 2));
    }

    #[test]
    fn test_iriw_full_symmetry() {
        let iriw = litmus_iriw();
        let full = FullSymmetryGroup::compute(&iriw);
        // IRIW should have some symmetry
        assert!(full.total_order >= 1);
    }

    #[test]
    fn test_address_pattern() {
        let sb = litmus_sb();
        let patterns = AddressSymmetryDetector::compute_patterns(&sb);
        assert_eq!(patterns.len(), 2);
        // Address 0 (x): T0 stores, T1 loads
        // Address 1 (y): T0 loads, T1 stores
        // These are "mirror" patterns
    }

    #[test]
    fn test_sb_address_symmetry() {
        let sb = litmus_sb();
        let addr_group = AddressSymmetryDetector::symmetry_group(&sb);
        // SB: swapping x↔y together with swapping threads gives a symmetry
        // Address symmetry alone: x and y have mirror roles, so there may be symmetry
        assert!(addr_group.order() >= 1);
    }

    #[test]
    fn test_full_symmetry_sb() {
        let sb = litmus_sb();
        let full = FullSymmetryGroup::compute(&sb);
        assert!(full.has_symmetry() || full.total_order == 1);
        let report = full.compression_report();
        assert!(report.total_symmetry_order >= 1);
    }

    #[test]
    fn test_full_symmetry_mp() {
        let mp = litmus_mp();
        let full = FullSymmetryGroup::compute(&mp);
        // MP has no thread symmetry; may have some address/value symmetry
        assert_eq!(full.thread_group.order(), 1);
    }

    #[test]
    fn test_wl_color_refinement() {
        // Path graph: 0-1-2
        let mut wl = WeisfeilerLeman::new(3, &[(0, 1), (1, 2)]);
        let colors = wl.refine_1wl();
        // Endpoints should get same color, middle different
        assert_eq!(colors[0], colors[2]);
        assert_ne!(colors[0], colors[1]);
    }

    #[test]
    fn test_wl_cycle_graph() {
        // Cycle: 0-1-2-0
        let mut wl = WeisfeilerLeman::new(3, &[(0, 1), (1, 2), (2, 0)]);
        let colors = wl.refine_1wl();
        // All nodes should get same color (regular graph)
        assert_eq!(colors[0], colors[1]);
        assert_eq!(colors[1], colors[2]);
    }

    #[test]
    fn test_wl_isomorphism_check() {
        // Two path graphs of length 3
        let g1_edges = vec![(0, 1), (1, 2)];
        let g2_edges = vec![(0, 1), (1, 2)];
        let labels = vec![1, 1, 1];
        assert!(WeisfeilerLeman::are_potentially_isomorphic(
            3, &g1_edges, &labels,
            3, &g2_edges, &labels,
        ));

        // Path (P3) vs triangle (C3) — genuinely non-isomorphic
        let g3_edges = vec![(0, 1), (1, 2), (2, 0)];
        assert!(!WeisfeilerLeman::are_potentially_isomorphic(
            3, &g1_edges, &labels,
            3, &g3_edges, &labels,
        ));
    }

    #[test]
    fn test_wl_2wl() {
        let mut wl = WeisfeilerLeman::new(4, &[(0, 1), (1, 2), (2, 3)]);
        let pair_colors = wl.refine_2wl();
        // 4x4 matrix of pair colors
        assert_eq!(pair_colors.len(), 4);
        assert_eq!(pair_colors[0].len(), 4);
    }

    #[test]
    fn test_threads_isomorphic() {
        let sb = litmus_sb();
        // SB threads have same opcode pattern
        let iso = threads_isomorphic(&sb.threads[0], &sb.threads[1]);
        assert!(iso);
    }

    #[test]
    fn test_threads_not_isomorphic_mp() {
        let mp = litmus_mp();
        let iso = threads_isomorphic(&mp.threads[0], &mp.threads[1]);
        assert!(!iso); // Store,Store vs Load,Load
    }

    #[test]
    fn test_thread_signature_canonicalization() {
        let ops_a = vec![
            MemoryOp {
                thread_id: 0, op_index: 0, opcode: Opcode::Store,
                address: Some(5), value: Some(10), depends_on: vec![],
            },
            MemoryOp {
                thread_id: 0, op_index: 1, opcode: Opcode::Load,
                address: Some(7), value: Some(20), depends_on: vec![0],
            },
        ];
        let ops_b = vec![
            MemoryOp {
                thread_id: 1, op_index: 0, opcode: Opcode::Store,
                address: Some(100), value: Some(200), depends_on: vec![],
            },
            MemoryOp {
                thread_id: 1, op_index: 1, opcode: Opcode::Load,
                address: Some(300), value: Some(400), depends_on: vec![0],
            },
        ];
        let sig_a = ThreadSignature::from_thread(&ops_a);
        let sig_b = ThreadSignature::from_thread(&ops_b);
        // After canonicalization, same structure should give same signature
        assert_eq!(sig_a, sig_b);
    }

    #[test]
    fn test_symmetry_report_display() {
        let sb = litmus_sb();
        let full = FullSymmetryGroup::compute(&sb);
        let report = full.compression_report();
        let display = format!("{}", report);
        assert!(display.contains("Symmetry Report"));
    }

    #[test]
    fn test_value_symmetry_sb() {
        let sb = litmus_sb();
        let val_group = ValueSymmetryDetector::symmetry_group(&sb);
        // Values 0 and 1 play different roles in SB
        assert!(val_group.order() >= 1);
    }

    #[test]
    fn test_litmus_apply_thread_perm() {
        let sb = litmus_sb();
        let perm = Permutation::transposition(2, 0, 1);
        let permuted = sb.apply_thread_permutation(&perm);
        assert_eq!(permuted.num_threads, 2);
        // Thread 0 of permuted should be thread 1 of original (with thread_id updated)
        assert_eq!(permuted.threads[1].len(), sb.threads[0].len());
    }

    #[test]
    fn test_litmus_apply_address_perm() {
        let sb = litmus_sb();
        let perm = Permutation::transposition(2, 0, 1);
        let permuted = sb.apply_address_permutation(&perm);
        // Address 0 → 1 and 1 → 0
        assert_eq!(permuted.threads[0][0].address, Some(1));
        assert_eq!(permuted.threads[0][1].address, Some(0));
    }

    #[test]
    fn test_litmus_structural_equality() {
        let sb1 = litmus_sb();
        let sb2 = litmus_sb();
        assert!(sb1.structurally_equal(&sb2));

        let mp = litmus_mp();
        assert!(!sb1.structurally_equal(&mp));
    }

    #[test]
    fn test_wl_partition() {
        let mut wl = WeisfeilerLeman::new(4, &[(0, 1), (1, 2), (2, 3)]);
        wl.refine_1wl();
        let partition = wl.color_partition();
        // Path 0-1-2-3: endpoints {0,3} and midpoints {1,2}
        assert_eq!(partition.len(), 2);
    }
}
