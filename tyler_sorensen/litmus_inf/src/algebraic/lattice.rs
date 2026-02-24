//! Lattice theory for the LITMUS∞ algebraic engine.
//!
//! Implements finite lattices, complete lattices, Boolean lattices,
//! partition lattices, lattice homomorphisms, distributivity/modularity
//! checking, congruence lattices, and Knaster-Tarski fixed-point computation.

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use serde::{Serialize, Deserialize};

// ═══════════════════════════════════════════════════════════════════════════
// LatticeElement Trait
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for elements that can participate in a lattice structure.
pub trait LatticeElement: Clone + Eq + Hash + fmt::Debug {
    /// Meet (greatest lower bound / infimum).
    fn meet(&self, other: &Self) -> Self;
    /// Join (least upper bound / supremum).
    fn join(&self, other: &Self) -> Self;
    /// Partial order comparison in the lattice.
    fn lattice_leq(&self, other: &Self) -> bool {
        self.meet(other) == *self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FiniteLattice
// ═══════════════════════════════════════════════════════════════════════════

/// A finite lattice with precomputed meet and join tables.
#[derive(Debug, Clone)]
pub struct FiniteLattice {
    /// Number of elements.
    pub size: usize,
    /// Element labels (for display).
    pub labels: Vec<String>,
    /// Meet table: `meet_table[i][j]` is the index of meet(i, j).
    meet_table: Vec<Vec<usize>>,
    /// Join table: `join_table[i][j]` is the index of join(i, j).
    join_table: Vec<Vec<usize>>,
    /// Partial order: `leq[i][j]` = true iff i ≤ j.
    leq: Vec<Vec<bool>>,
    /// Index of the bottom element (0).
    pub bottom: usize,
    /// Index of the top element (1).
    pub top: usize,
}

impl FiniteLattice {
    /// Construct a finite lattice from a partial order given as pairs (i ≤ j).
    /// The partial order pairs should include reflexive pairs and be transitively closed,
    /// or the constructor will compute the transitive closure.
    pub fn from_partial_order(size: usize, pairs: &[(usize, usize)]) -> Option<Self> {
        if size == 0 { return None; }

        // Build and transitively close the partial order
        let mut leq = vec![vec![false; size]; size];
        for i in 0..size { leq[i][i] = true; } // reflexive
        for &(a, b) in pairs {
            if a < size && b < size {
                leq[a][b] = true;
            }
        }
        // Transitive closure (Floyd-Warshall)
        for k in 0..size {
            for i in 0..size {
                for j in 0..size {
                    if leq[i][k] && leq[k][j] {
                        leq[i][j] = true;
                    }
                }
            }
        }

        // Check antisymmetry
        for i in 0..size {
            for j in (i + 1)..size {
                if leq[i][j] && leq[j][i] {
                    return None; // not antisymmetric
                }
            }
        }

        // Compute meet table
        let mut meet_table = vec![vec![0usize; size]; size];
        for i in 0..size {
            for j in 0..size {
                // Meet(i,j) = greatest element that is ≤ both i and j
                let mut best: Option<usize> = None;
                for k in 0..size {
                    if leq[k][i] && leq[k][j] {
                        if best.is_none() || leq[best.unwrap()][k] {
                            best = Some(k);
                        }
                    }
                }
                match best {
                    Some(b) => meet_table[i][j] = b,
                    None => return None, // not a lattice
                }
            }
        }

        // Compute join table
        let mut join_table = vec![vec![0usize; size]; size];
        for i in 0..size {
            for j in 0..size {
                // Join(i,j) = least element that is ≥ both i and j
                let mut best = None;
                for k in 0..size {
                    if leq[i][k] && leq[j][k] {
                        if best.is_none() || leq[k][best.unwrap()] {
                            best = Some(k);
                        }
                    }
                }
                match best {
                    Some(b) => join_table[i][j] = b,
                    None => return None, // not a lattice
                }
            }
        }

        // Find bottom (element ≤ all others) and top (element ≥ all others)
        let bottom = (0..size).find(|&i| (0..size).all(|j| leq[i][j]));
        let top = (0..size).find(|&i| (0..size).all(|j| leq[j][i]));

        let bottom = bottom?;
        let top = top?;

        let labels: Vec<String> = (0..size).map(|i| format!("{}", i)).collect();

        Some(FiniteLattice { size, labels, meet_table, join_table, leq, bottom, top })
    }

    /// Set element labels.
    pub fn set_labels(&mut self, labels: Vec<String>) {
        assert_eq!(labels.len(), self.size);
        self.labels = labels;
    }

    /// Meet of two elements.
    pub fn meet(&self, a: usize, b: usize) -> usize {
        self.meet_table[a][b]
    }

    /// Join of two elements.
    pub fn join(&self, a: usize, b: usize) -> usize {
        self.join_table[a][b]
    }

    /// Check if a ≤ b in the lattice.
    pub fn leq(&self, a: usize, b: usize) -> bool {
        self.leq[a][b]
    }

    /// Check if a < b (strictly less).
    pub fn lt(&self, a: usize, b: usize) -> bool {
        a != b && self.leq[a][b]
    }

    /// Check if b covers a (a < b and no c with a < c < b).
    pub fn covers(&self, a: usize, b: usize) -> bool {
        if !self.lt(a, b) { return false; }
        for c in 0..self.size {
            if c != a && c != b && self.lt(a, c) && self.lt(c, b) {
                return false;
            }
        }
        true
    }

    /// Elements that cover a.
    pub fn upper_covers(&self, a: usize) -> Vec<usize> {
        (0..self.size).filter(|&b| self.covers(a, b)).collect()
    }

    /// Elements covered by a.
    pub fn lower_covers(&self, a: usize) -> Vec<usize> {
        (0..self.size).filter(|&b| self.covers(b, a)).collect()
    }

    /// Atoms: elements covering the bottom.
    pub fn atoms(&self) -> Vec<usize> {
        self.upper_covers(self.bottom)
    }

    /// Coatoms: elements covered by the top.
    pub fn coatoms(&self) -> Vec<usize> {
        self.lower_covers(self.top)
    }

    /// Interval [a, b] = {c : a ≤ c ≤ b}.
    pub fn interval(&self, a: usize, b: usize) -> Vec<usize> {
        if !self.leq(a, b) { return vec![]; }
        (0..self.size).filter(|&c| self.leq(a, c) && self.leq(c, b)).collect()
    }

    /// Complement of a: element b such that a ∧ b = ⊥ and a ∨ b = ⊤.
    pub fn complement(&self, a: usize) -> Option<usize> {
        for b in 0..self.size {
            if self.meet(a, b) == self.bottom && self.join(a, b) == self.top {
                return Some(b);
            }
        }
        None
    }

    /// Check if the lattice is complemented (every element has a complement).
    pub fn is_complemented(&self) -> bool {
        (0..self.size).all(|a| self.complement(a).is_some())
    }

    /// Check if the lattice is bounded (has top and bottom).
    pub fn is_bounded(&self) -> bool {
        true // By construction we always have top and bottom
    }

    /// Height of the lattice (length of longest chain from bottom to top).
    pub fn height(&self) -> usize {
        self.chain_length(self.bottom, self.top)
    }

    /// Length of longest chain from a to b.
    fn chain_length(&self, a: usize, b: usize) -> usize {
        if !self.leq(a, b) { return 0; }
        if a == b { return 0; }
        let covers = self.upper_covers(a);
        let mut max_len = 0;
        for &c in &covers {
            if self.leq(c, b) {
                max_len = max_len.max(1 + self.chain_length(c, b));
            }
        }
        max_len
    }

    /// Width of the lattice (maximum antichain size).
    pub fn width(&self) -> usize {
        // Dilworth's theorem: width = min number of chains to cover
        // Simple: try all subsets (exponential, only for small lattices)
        let mut max_antichain = 1;
        for mask in 1..(1u64 << self.size.min(20)) {
            let elems: Vec<usize> = (0..self.size)
                .filter(|&i| mask & (1 << i) != 0)
                .collect();
            let is_antichain = elems.iter().all(|&a| {
                elems.iter().all(|&b| a == b || (!self.leq(a, b) && !self.leq(b, a)))
            });
            if is_antichain {
                max_antichain = max_antichain.max(elems.len());
            }
        }
        max_antichain
    }

    /// Get all maximal chains from bottom to top.
    pub fn maximal_chains(&self) -> Vec<Vec<usize>> {
        let mut chains = Vec::new();
        let mut current = vec![self.bottom];
        self.find_chains(self.bottom, &mut current, &mut chains);
        chains
    }

    fn find_chains(&self, current: usize, chain: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if current == self.top {
            result.push(chain.clone());
            return;
        }
        let covers = self.upper_covers(current);
        for &c in &covers {
            chain.push(c);
            self.find_chains(c, chain, result);
            chain.pop();
        }
    }

    /// Covering relation as a list of pairs (for Hasse diagram).
    pub fn covering_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for a in 0..self.size {
            for b in 0..self.size {
                if self.covers(a, b) {
                    pairs.push((a, b));
                }
            }
        }
        pairs
    }

    /// Möbius function μ(a, b).
    pub fn mobius(&self, a: usize, b: usize) -> i64 {
        if !self.leq(a, b) { return 0; }
        if a == b { return 1; }
        let interval = self.interval(a, b);
        let mut mu = 0i64;
        for &c in &interval {
            if c != b {
                mu -= self.mobius(a, c);
            }
        }
        mu
    }

    /// Zeta polynomial: ζ(a, b) = 1 if a ≤ b, 0 otherwise.
    pub fn zeta(&self, a: usize, b: usize) -> i64 {
        if self.leq(a, b) { 1 } else { 0 }
    }
}

impl fmt::Display for FiniteLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Lattice (size={}, bottom={}, top={}):", self.size, self.labels[self.bottom], self.labels[self.top])?;
        writeln!(f, "Covering relations:")?;
        for (a, b) in self.covering_pairs() {
            writeln!(f, "  {} < {}", self.labels[a], self.labels[b])?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CompleteLattice — with arbitrary meets/joins and fixed-point computation
// ═══════════════════════════════════════════════════════════════════════════

/// A complete lattice: every subset has a meet and join.
/// For finite lattices this is automatic; this struct adds fixed-point operations.
#[derive(Debug, Clone)]
pub struct CompleteLattice {
    /// The underlying finite lattice.
    pub lattice: FiniteLattice,
}

impl CompleteLattice {
    /// Construct from a finite lattice.
    pub fn new(lattice: FiniteLattice) -> Self {
        CompleteLattice { lattice }
    }

    /// Meet of an arbitrary set of elements.
    pub fn meet_all(&self, elements: &[usize]) -> usize {
        if elements.is_empty() {
            return self.lattice.top; // empty meet = top
        }
        let mut result = elements[0];
        for &e in &elements[1..] {
            result = self.lattice.meet(result, e);
        }
        result
    }

    /// Join of an arbitrary set of elements.
    pub fn join_all(&self, elements: &[usize]) -> usize {
        if elements.is_empty() {
            return self.lattice.bottom; // empty join = bottom
        }
        let mut result = elements[0];
        for &e in &elements[1..] {
            result = self.lattice.join(result, e);
        }
        result
    }

    /// Knaster-Tarski least fixed point: lfp(f) = meet({x : f(x) ≤ x}).
    pub fn least_fixed_point<F: Fn(usize) -> usize>(&self, f: &F) -> usize {
        // Collect all pre-fixed points: {x : f(x) ≤ x}
        let pre_fixed: Vec<usize> = (0..self.lattice.size)
            .filter(|&x| self.lattice.leq(f(x), x))
            .collect();
        self.meet_all(&pre_fixed)
    }

    /// Knaster-Tarski greatest fixed point: gfp(f) = join({x : x ≤ f(x)}).
    pub fn greatest_fixed_point<F: Fn(usize) -> usize>(&self, f: &F) -> usize {
        // Collect all post-fixed points: {x : x ≤ f(x)}
        let post_fixed: Vec<usize> = (0..self.lattice.size)
            .filter(|&x| self.lattice.leq(x, f(x)))
            .collect();
        self.join_all(&post_fixed)
    }

    /// Iterative fixed-point computation from bottom (Kleene's theorem).
    pub fn iterative_lfp<F: Fn(usize) -> usize>(&self, f: &F) -> usize {
        let mut x = self.lattice.bottom;
        loop {
            let fx = f(x);
            if fx == x {
                return x;
            }
            x = fx;
        }
    }

    /// Iterative fixed-point computation from top.
    pub fn iterative_gfp<F: Fn(usize) -> usize>(&self, f: &F) -> usize {
        let mut x = self.lattice.top;
        loop {
            let fx = f(x);
            if fx == x {
                return x;
            }
            x = fx;
        }
    }

    /// All fixed points of a monotone function.
    pub fn all_fixed_points<F: Fn(usize) -> usize>(&self, f: &F) -> Vec<usize> {
        (0..self.lattice.size).filter(|&x| f(x) == x).collect()
    }

    /// Check if a function is monotone on this lattice.
    pub fn is_monotone<F: Fn(usize) -> usize>(&self, f: &F) -> bool {
        for a in 0..self.lattice.size {
            for b in 0..self.lattice.size {
                if self.lattice.leq(a, b) && !self.lattice.leq(f(a), f(b)) {
                    return false;
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BooleanLattice — power set lattice
// ═══════════════════════════════════════════════════════════════════════════

/// Boolean lattice (power set lattice) on n atoms.
/// Elements are subsets of {0, ..., n-1}, represented as bitmasks.
#[derive(Debug, Clone)]
pub struct BooleanLattice {
    /// Number of atoms.
    pub num_atoms: usize,
    /// The underlying finite lattice.
    pub lattice: FiniteLattice,
    /// Bitmask for each element index.
    pub element_mask: Vec<u64>,
    /// Map from bitmask to element index.
    mask_to_index: HashMap<u64, usize>,
}

impl BooleanLattice {
    /// Create a Boolean lattice on `n` atoms (n ≤ 16 for practical use).
    pub fn new(n: usize) -> Self {
        assert!(n <= 16, "Boolean lattice too large");
        let size = 1usize << n;
        let mut pairs = Vec::new();
        let mut element_mask: Vec<u64> = (0..size as u64).collect();
        let mut mask_to_index: HashMap<u64, usize> = HashMap::new();

        for i in 0..size {
            mask_to_index.insert(i as u64, i);
            for j in 0..size {
                if (i & j) == i { // i is subset of j
                    pairs.push((i, j));
                }
            }
        }

        let lattice = FiniteLattice::from_partial_order(size, &pairs)
            .expect("Boolean lattice should be valid");

        BooleanLattice { num_atoms: n, lattice, element_mask, mask_to_index }
    }

    /// Meet (intersection).
    pub fn meet(&self, a: usize, b: usize) -> usize {
        let mask = self.element_mask[a] & self.element_mask[b];
        self.mask_to_index[&mask]
    }

    /// Join (union).
    pub fn join(&self, a: usize, b: usize) -> usize {
        let mask = self.element_mask[a] | self.element_mask[b];
        self.mask_to_index[&mask]
    }

    /// Complement.
    pub fn complement(&self, a: usize) -> usize {
        let full = (1u64 << self.num_atoms) - 1;
        let mask = self.element_mask[a] ^ full;
        self.mask_to_index[&mask]
    }

    /// Atoms (singleton subsets).
    pub fn atoms(&self) -> Vec<usize> {
        (0..self.num_atoms).map(|i| self.mask_to_index[&(1u64 << i)]).collect()
    }

    /// Element from a set of atom indices.
    pub fn from_atoms(&self, atoms: &[usize]) -> usize {
        let mut mask = 0u64;
        for &a in atoms {
            mask |= 1u64 << a;
        }
        self.mask_to_index[&mask]
    }

    /// Get the atoms in an element.
    pub fn to_atoms(&self, elem: usize) -> Vec<usize> {
        let mask = self.element_mask[elem];
        (0..self.num_atoms).filter(|&i| mask & (1 << i) != 0).collect()
    }

    /// Size of the lattice (2^n).
    pub fn size(&self) -> usize {
        1 << self.num_atoms
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PartitionLattice — lattice of set partitions
// ═══════════════════════════════════════════════════════════════════════════

/// A set partition of {0, ..., n-1}.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetPartition {
    /// Block assignment: `block[i]` is the block index of element i.
    pub blocks: Vec<usize>,
    /// Number of elements.
    pub n: usize,
}

impl SetPartition {
    /// Create the discrete partition (each element in its own block).
    pub fn discrete(n: usize) -> Self {
        SetPartition { blocks: (0..n).collect(), n }
    }

    /// Create the single-block partition.
    pub fn single_block(n: usize) -> Self {
        SetPartition { blocks: vec![0; n], n }
    }

    /// Create from block assignment (normalizes block labels).
    pub fn from_blocks(blocks: Vec<usize>) -> Self {
        let n = blocks.len();
        let mut mapping = HashMap::new();
        let mut next_id = 0;
        let mut normalized = vec![0; n];
        for i in 0..n {
            let block = blocks[i];
            let id = *mapping.entry(block).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            normalized[i] = id;
        }
        SetPartition { blocks: normalized, n }
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        if self.blocks.is_empty() { return 0; }
        *self.blocks.iter().max().unwrap() + 1
    }

    /// Get elements in each block.
    pub fn block_elements(&self) -> Vec<Vec<usize>> {
        let nb = self.num_blocks();
        let mut result = vec![Vec::new(); nb];
        for (i, &b) in self.blocks.iter().enumerate() {
            result[b].push(i);
        }
        result
    }

    /// Meet of two partitions (finest common coarsening = intersection of equivalence relations).
    pub fn meet(&self, other: &SetPartition) -> SetPartition {
        assert_eq!(self.n, other.n);
        let n = self.n;
        // Two elements are in the same block iff they are in the same block in BOTH partitions
        let mut blocks = vec![0usize; n];
        let mut pair_to_block: HashMap<(usize, usize), usize> = HashMap::new();
        let mut next_block = 0;

        for i in 0..n {
            let key = (self.blocks[i], other.blocks[i]);
            let block = *pair_to_block.entry(key).or_insert_with(|| {
                let b = next_block;
                next_block += 1;
                b
            });
            blocks[i] = block;
        }
        SetPartition::from_blocks(blocks)
    }

    /// Join of two partitions (coarsest common refinement = join of equivalence relations).
    pub fn join(&self, other: &SetPartition) -> SetPartition {
        assert_eq!(self.n, other.n);
        let n = self.n;
        // Union-Find
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry { parent[rx] = ry; }
        }

        // Merge according to self
        for i in 0..n {
            for j in (i + 1)..n {
                if self.blocks[i] == self.blocks[j] {
                    union(&mut parent, i, j);
                }
            }
        }
        // Merge according to other
        for i in 0..n {
            for j in (i + 1)..n {
                if other.blocks[i] == other.blocks[j] {
                    union(&mut parent, i, j);
                }
            }
        }

        let blocks: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
        SetPartition::from_blocks(blocks)
    }

    /// Check if self refines other (self ≤ other in the partition lattice).
    pub fn refines(&self, other: &SetPartition) -> bool {
        assert_eq!(self.n, other.n);
        // self refines other iff every block of self is contained in some block of other
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.blocks[i] == self.blocks[j] && other.blocks[i] != other.blocks[j] {
                    return false;
                }
            }
        }
        true
    }
}

impl fmt::Display for SetPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let blocks = self.block_elements();
        write!(f, "{{")?;
        for (i, block) in blocks.iter().enumerate() {
            if i > 0 { write!(f, "|")?; }
            write!(f, "{{")?;
            for (j, &elem) in block.iter().enumerate() {
                if j > 0 { write!(f, ",")?; }
                write!(f, "{}", elem)?;
            }
            write!(f, "}}")?;
        }
        write!(f, "}}")
    }
}

/// Partition lattice: lattice of all set partitions of {0, ..., n-1}.
#[derive(Debug, Clone)]
pub struct PartitionLattice {
    /// Number of elements being partitioned.
    pub n: usize,
    /// All partitions (enumerated).
    pub partitions: Vec<SetPartition>,
}

impl PartitionLattice {
    /// Build the partition lattice for {0, ..., n-1}.
    /// Only practical for small n (≤ 6 or so due to Bell numbers).
    pub fn new(n: usize) -> Self {
        let partitions = Self::enumerate_all_partitions(n);
        PartitionLattice { n, partitions }
    }

    fn enumerate_all_partitions(n: usize) -> Vec<SetPartition> {
        if n == 0 {
            return vec![SetPartition::from_blocks(vec![])];
        }
        let mut result = Vec::new();
        let mut assignment = vec![0usize; n];
        Self::enumerate_recursive(n, 0, 0, &mut assignment, &mut result);
        result
    }

    fn enumerate_recursive(
        n: usize,
        pos: usize,
        max_block: usize,
        assignment: &mut Vec<usize>,
        result: &mut Vec<SetPartition>,
    ) {
        if pos == n {
            result.push(SetPartition::from_blocks(assignment.clone()));
            return;
        }
        for b in 0..=max_block {
            assignment[pos] = b;
            let new_max = if b == max_block { max_block + 1 } else { max_block };
            Self::enumerate_recursive(n, pos + 1, new_max, assignment, result);
        }
    }

    /// Number of partitions (Bell number B_n).
    pub fn size(&self) -> usize {
        self.partitions.len()
    }

    /// Bell number B_n.
    pub fn bell_number(n: usize) -> u64 {
        if n == 0 { return 1; }
        // Bell triangle
        let mut triangle = vec![vec![0u64; n + 1]; n + 1];
        triangle[0][0] = 1;
        for i in 1..=n {
            triangle[i][0] = triangle[i - 1][i - 1];
            for j in 1..=i {
                triangle[i][j] = triangle[i][j - 1] + triangle[i - 1][j - 1];
            }
        }
        triangle[n][0]
    }

    /// Find a partition by block assignment.
    pub fn find(&self, partition: &SetPartition) -> Option<usize> {
        self.partitions.iter().position(|p| p == partition)
    }

    /// Get the discrete partition index.
    pub fn discrete_index(&self) -> Option<usize> {
        self.find(&SetPartition::discrete(self.n))
    }

    /// Get the single-block partition index.
    pub fn single_block_index(&self) -> Option<usize> {
        self.find(&SetPartition::single_block(self.n))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lattice Homomorphism
// ═══════════════════════════════════════════════════════════════════════════

/// A homomorphism between finite lattices.
#[derive(Debug, Clone)]
pub struct LatticeHomomorphism {
    /// Source lattice size.
    pub source_size: usize,
    /// Target lattice size.
    pub target_size: usize,
    /// The mapping: `mapping[i]` is the image of element i.
    pub mapping: Vec<usize>,
}

impl LatticeHomomorphism {
    /// Create a lattice homomorphism.
    pub fn new(mapping: Vec<usize>) -> Self {
        let source_size = mapping.len();
        let target_size = mapping.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        LatticeHomomorphism { source_size, target_size, mapping }
    }

    /// Check if this preserves meet.
    pub fn preserves_meet(&self, source: &FiniteLattice, target: &FiniteLattice) -> bool {
        for a in 0..self.source_size {
            for b in 0..self.source_size {
                let meet_then_map = self.mapping[source.meet(a, b)];
                let map_then_meet = target.meet(self.mapping[a], self.mapping[b]);
                if meet_then_map != map_then_meet {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this preserves join.
    pub fn preserves_join(&self, source: &FiniteLattice, target: &FiniteLattice) -> bool {
        for a in 0..self.source_size {
            for b in 0..self.source_size {
                let join_then_map = self.mapping[source.join(a, b)];
                let map_then_join = target.join(self.mapping[a], self.mapping[b]);
                if join_then_map != map_then_join {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this is a valid lattice homomorphism (preserves both meet and join).
    pub fn is_valid(&self, source: &FiniteLattice, target: &FiniteLattice) -> bool {
        self.preserves_meet(source, target) && self.preserves_join(source, target)
    }

    /// Compute the kernel (preimage partition: elements with the same image).
    pub fn kernel(&self) -> Vec<Vec<usize>> {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &img) in self.mapping.iter().enumerate() {
            groups.entry(img).or_default().push(i);
        }
        groups.into_values().collect()
    }

    /// Image of the homomorphism.
    pub fn image(&self) -> HashSet<usize> {
        self.mapping.iter().copied().collect()
    }

    /// Check if injective.
    pub fn is_injective(&self) -> bool {
        let image: HashSet<usize> = self.mapping.iter().copied().collect();
        image.len() == self.source_size
    }

    /// Check if surjective.
    pub fn is_surjective(&self) -> bool {
        let image: HashSet<usize> = self.mapping.iter().copied().collect();
        image.len() == self.target_size
    }

    /// Check if this is an isomorphism.
    pub fn is_isomorphism(&self, source: &FiniteLattice, target: &FiniteLattice) -> bool {
        self.is_valid(source, target) && self.is_injective() && self.is_surjective()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Distributive and Modular Lattice Checks
// ═══════════════════════════════════════════════════════════════════════════

/// Lattice properties checker.
pub struct LatticeProperties;

impl LatticeProperties {
    /// Check distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c) for all a, b, c.
    pub fn is_distributive(lattice: &FiniteLattice) -> bool {
        for a in 0..lattice.size {
            for b in 0..lattice.size {
                for c in 0..lattice.size {
                    let lhs = lattice.meet(a, lattice.join(b, c));
                    let rhs = lattice.join(lattice.meet(a, b), lattice.meet(a, c));
                    if lhs != rhs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check modularity: a ≤ c implies a ∨ (b ∧ c) = (a ∨ b) ∧ c for all a, b, c.
    pub fn is_modular(lattice: &FiniteLattice) -> bool {
        for a in 0..lattice.size {
            for b in 0..lattice.size {
                for c in 0..lattice.size {
                    if lattice.leq(a, c) {
                        let lhs = lattice.join(a, lattice.meet(b, c));
                        let rhs = lattice.meet(lattice.join(a, b), c);
                        if lhs != rhs {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Find a non-distributive sublattice (N5 or M3) if one exists.
    pub fn find_non_distributive_sublattice(lattice: &FiniteLattice) -> Option<(usize, usize, usize)> {
        for a in 0..lattice.size {
            for b in 0..lattice.size {
                for c in 0..lattice.size {
                    let lhs = lattice.meet(a, lattice.join(b, c));
                    let rhs = lattice.join(lattice.meet(a, b), lattice.meet(a, c));
                    if lhs != rhs {
                        return Some((a, b, c));
                    }
                }
            }
        }
        None
    }

    /// Check if the lattice is atomic (every element is a join of atoms).
    pub fn is_atomic(lattice: &FiniteLattice) -> bool {
        let atoms = lattice.atoms();
        for e in 0..lattice.size {
            if e == lattice.bottom { continue; }
            // Check if e is a join of atoms below it
            let atoms_below: Vec<usize> = atoms.iter()
                .filter(|&&a| lattice.leq(a, e))
                .copied()
                .collect();
            if atoms_below.is_empty() {
                return false;
            }
            let joined = atoms_below.iter().fold(lattice.bottom, |acc, &a| lattice.join(acc, a));
            if joined != e {
                // Not exactly equal, but check if e is below the join
                // Actually for atomicity we need every element to be a join of atoms
                if !lattice.leq(e, joined) || !lattice.leq(joined, e) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the lattice is Boolean (complemented + distributive).
    pub fn is_boolean(lattice: &FiniteLattice) -> bool {
        lattice.is_complemented() && Self::is_distributive(lattice)
    }

    /// Check if the lattice is a chain (total order).
    pub fn is_chain(lattice: &FiniteLattice) -> bool {
        for a in 0..lattice.size {
            for b in 0..lattice.size {
                if !lattice.leq(a, b) && !lattice.leq(b, a) {
                    return false;
                }
            }
        }
        true
    }

    /// Check graded property: all maximal chains have the same length.
    pub fn is_graded(lattice: &FiniteLattice) -> bool {
        let chains = lattice.maximal_chains();
        if chains.is_empty() { return true; }
        let first_len = chains[0].len();
        chains.iter().all(|c| c.len() == first_len)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Congruence Lattice
// ═══════════════════════════════════════════════════════════════════════════

/// A congruence relation on a lattice (equivalence relation compatible with meet and join).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LatticeCongruence {
    /// Equivalence classes: `class_of[i]` is the class index of element i.
    pub class_of: Vec<usize>,
    /// Number of classes.
    pub num_classes: usize,
}

impl LatticeCongruence {
    /// The trivial congruence (each element in its own class).
    pub fn discrete(n: usize) -> Self {
        LatticeCongruence { class_of: (0..n).collect(), num_classes: n }
    }

    /// The total congruence (all elements in one class).
    pub fn total(n: usize) -> Self {
        LatticeCongruence { class_of: vec![0; n], num_classes: 1 }
    }

    /// Generate the smallest congruence containing (a, b).
    pub fn generated_by(lattice: &FiniteLattice, a: usize, b: usize) -> Self {
        let n = lattice.size;
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find(parent, parent[x]); }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) -> bool {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry { parent[rx] = ry; true } else { false }
        }

        union(&mut parent, a, b);

        // Close under meet and join
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                for j in 0..n {
                    if find(&mut parent, i) == find(&mut parent, j) {
                        // For all k, meet(i,k) ≡ meet(j,k) and join(i,k) ≡ join(j,k)
                        for k in 0..n {
                            if union(&mut parent, lattice.meet(i, k), lattice.meet(j, k)) {
                                changed = true;
                            }
                            if union(&mut parent, lattice.join(i, k), lattice.join(j, k)) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Normalize
        let mut mapping = HashMap::new();
        let mut next_id = 0;
        let mut class_of = vec![0; n];
        for i in 0..n {
            let root = find(&mut parent, i);
            let id = *mapping.entry(root).or_insert_with(|| { let id = next_id; next_id += 1; id });
            class_of[i] = id;
        }

        LatticeCongruence { class_of, num_classes: next_id }
    }

    /// Check if this is a valid congruence on the lattice.
    pub fn is_valid(&self, lattice: &FiniteLattice) -> bool {
        let n = lattice.size;
        for i in 0..n {
            for j in 0..n {
                if self.class_of[i] == self.class_of[j] {
                    for k in 0..n {
                        if self.class_of[lattice.meet(i, k)] != self.class_of[lattice.meet(j, k)] {
                            return false;
                        }
                        if self.class_of[lattice.join(i, k)] != self.class_of[lattice.join(j, k)] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Build the quotient lattice.
    pub fn quotient_lattice(&self, lattice: &FiniteLattice) -> Option<FiniteLattice> {
        let n = self.num_classes;
        // Representatives: pick the smallest element in each class
        let mut representatives = vec![0; n];
        for i in 0..lattice.size {
            if i == 0 || self.class_of[i] != self.class_of[i - 1] {
                // Find smallest in each class
            }
        }
        let mut class_members: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..lattice.size {
            class_members[self.class_of[i]].push(i);
        }
        for i in 0..n {
            representatives[i] = class_members[i][0];
        }

        // Build partial order on classes
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if lattice.leq(representatives[i], representatives[j]) {
                    pairs.push((i, j));
                }
            }
        }

        FiniteLattice::from_partial_order(n, &pairs)
    }
}

/// Enumerate all congruences of a lattice.
pub fn all_congruences(lattice: &FiniteLattice) -> Vec<LatticeCongruence> {
    let mut congruences = HashSet::new();
    congruences.insert(LatticeCongruence::discrete(lattice.size));
    congruences.insert(LatticeCongruence::total(lattice.size));

    // Generate congruences from each pair
    for a in 0..lattice.size {
        for b in (a + 1)..lattice.size {
            let cong = LatticeCongruence::generated_by(lattice, a, b);
            congruences.insert(cong);
        }
    }

    congruences.into_iter().collect()
}

/// Check if a lattice is simple (has only trivial congruences).
pub fn is_simple(lattice: &FiniteLattice) -> bool {
    let congs = all_congruences(lattice);
    congs.len() == 2 // only discrete and total
}

// ═══════════════════════════════════════════════════════════════════════════
// Lattice Construction Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a chain lattice (total order on n elements).
pub fn chain_lattice(n: usize) -> FiniteLattice {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in i..n {
            pairs.push((i, j));
        }
    }
    FiniteLattice::from_partial_order(n, &pairs).expect("Chain is always a valid lattice")
}

/// Build a diamond lattice M_n (bottom, n atoms, top).
pub fn diamond_lattice(n: usize) -> FiniteLattice {
    let size = n + 2; // bottom + n atoms + top
    let bottom = 0;
    let top = n + 1;
    let mut pairs = Vec::new();

    // bottom ≤ everything
    for i in 0..size { pairs.push((bottom, i)); }
    // everything ≤ top
    for i in 0..size { pairs.push((i, top)); }
    // reflexive
    for i in 0..size { pairs.push((i, i)); }

    FiniteLattice::from_partial_order(size, &pairs).expect("Diamond is a valid lattice")
}

/// Build the pentagon lattice N_5 (the smallest non-modular lattice).
pub fn pentagon_lattice() -> FiniteLattice {
    // Elements: 0=bottom, 1, 2, 3, 4=top
    // 0 < 1 < 3 < 4, 0 < 2 < 4, and 1 and 2 are incomparable
    let pairs = vec![
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 3), (1, 4),
        (2, 4),
        (3, 4),
    ];
    FiniteLattice::from_partial_order(5, &pairs).expect("N5 is a valid lattice")
}

/// Build the product of two lattices.
pub fn product_lattice(l1: &FiniteLattice, l2: &FiniteLattice) -> FiniteLattice {
    let n = l1.size * l2.size;
    let idx = |i: usize, j: usize| i * l2.size + j;

    let mut pairs = Vec::new();
    for i1 in 0..l1.size {
        for j1 in 0..l2.size {
            for i2 in 0..l1.size {
                for j2 in 0..l2.size {
                    if l1.leq(i1, i2) && l2.leq(j1, j2) {
                        pairs.push((idx(i1, j1), idx(i2, j2)));
                    }
                }
            }
        }
    }

    FiniteLattice::from_partial_order(n, &pairs).expect("Product of lattices is a lattice")
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain_3() -> FiniteLattice {
        chain_lattice(3)
    }

    fn make_diamond_3() -> FiniteLattice {
        diamond_lattice(3)
    }

    #[test]
    fn test_chain_lattice() {
        let l = make_chain_3();
        assert_eq!(l.bottom, 0);
        assert_eq!(l.top, 2);
        assert_eq!(l.meet(0, 1), 0);
        assert_eq!(l.meet(1, 2), 1);
        assert_eq!(l.join(0, 1), 1);
        assert_eq!(l.join(0, 2), 2);
    }

    #[test]
    fn test_chain_is_distributive() {
        let l = make_chain_3();
        assert!(LatticeProperties::is_distributive(&l));
    }

    #[test]
    fn test_chain_is_chain() {
        let l = make_chain_3();
        assert!(LatticeProperties::is_chain(&l));
    }

    #[test]
    fn test_diamond_lattice() {
        let l = make_diamond_3();
        assert_eq!(l.size, 5);
        assert_eq!(l.atoms().len(), 3);
    }

    #[test]
    fn test_diamond_is_modular() {
        let l = make_diamond_3();
        assert!(LatticeProperties::is_modular(&l));
    }

    #[test]
    fn test_pentagon_not_modular() {
        let l = pentagon_lattice();
        assert!(!LatticeProperties::is_modular(&l));
    }

    #[test]
    fn test_pentagon_not_distributive() {
        let l = pentagon_lattice();
        assert!(!LatticeProperties::is_distributive(&l));
    }

    #[test]
    fn test_interval() {
        let l = chain_lattice(5);
        let interval = l.interval(1, 3);
        assert_eq!(interval, vec![1, 2, 3]);
    }

    #[test]
    fn test_covering() {
        let l = chain_lattice(4);
        assert!(l.covers(0, 1));
        assert!(l.covers(1, 2));
        assert!(!l.covers(0, 2));
    }

    #[test]
    fn test_height() {
        let l = chain_lattice(5);
        assert_eq!(l.height(), 4);
    }

    #[test]
    fn test_mobius_chain() {
        let l = chain_lattice(3);
        assert_eq!(l.mobius(0, 0), 1);
        assert_eq!(l.mobius(0, 1), -1);
        assert_eq!(l.mobius(0, 2), 0);
    }

    #[test]
    fn test_complement() {
        let l = diamond_lattice(2); // M_2 is Boolean
        // atoms are 1 and 2
        let c1 = l.complement(1);
        assert!(c1.is_some());
    }

    #[test]
    fn test_boolean_lattice() {
        let bl = BooleanLattice::new(3);
        assert_eq!(bl.size(), 8);
        assert_eq!(bl.atoms().len(), 3);

        let a = bl.from_atoms(&[0, 1]);
        let b = bl.from_atoms(&[1, 2]);
        let m = bl.meet(a, b);
        let j = bl.join(a, b);
        assert_eq!(bl.to_atoms(m), vec![1]); // {0,1} ∩ {1,2} = {1}
        assert_eq!(bl.to_atoms(j), vec![0, 1, 2]); // {0,1} ∪ {1,2} = {0,1,2}
    }

    #[test]
    fn test_boolean_complement() {
        let bl = BooleanLattice::new(3);
        let a = bl.from_atoms(&[0, 2]);
        let c = bl.complement(a);
        assert_eq!(bl.to_atoms(c), vec![1]);
    }

    #[test]
    fn test_boolean_is_boolean() {
        let bl = BooleanLattice::new(3);
        assert!(LatticeProperties::is_boolean(&bl.lattice));
    }

    #[test]
    fn test_set_partition_meet() {
        let p1 = SetPartition::from_blocks(vec![0, 0, 1, 1]);
        let p2 = SetPartition::from_blocks(vec![0, 1, 0, 1]);
        let m = p1.meet(&p2);
        // Meet: same block in both => {0},{1},{2},{3}
        assert_eq!(m.num_blocks(), 4);
    }

    #[test]
    fn test_set_partition_join() {
        let p1 = SetPartition::from_blocks(vec![0, 0, 1, 1]);
        let p2 = SetPartition::from_blocks(vec![0, 1, 0, 1]);
        let j = p1.join(&p2);
        // Join: merge transitively => all in one block
        assert_eq!(j.num_blocks(), 1);
    }

    #[test]
    fn test_set_partition_refines() {
        let discrete = SetPartition::discrete(3);
        let single = SetPartition::single_block(3);
        assert!(discrete.refines(&single));
        assert!(!single.refines(&discrete));
    }

    #[test]
    fn test_bell_numbers() {
        assert_eq!(PartitionLattice::bell_number(0), 1);
        assert_eq!(PartitionLattice::bell_number(1), 1);
        assert_eq!(PartitionLattice::bell_number(2), 2);
        assert_eq!(PartitionLattice::bell_number(3), 5);
        assert_eq!(PartitionLattice::bell_number(4), 15);
        assert_eq!(PartitionLattice::bell_number(5), 52);
    }

    #[test]
    fn test_partition_lattice_size() {
        let pl = PartitionLattice::new(3);
        assert_eq!(pl.size(), 5);
    }

    #[test]
    fn test_lattice_homomorphism() {
        let l1 = chain_lattice(3); // 0 < 1 < 2
        let l2 = chain_lattice(2); // 0 < 1
        let hom = LatticeHomomorphism::new(vec![0, 0, 1]); // 0→0, 1→0, 2→1
        assert!(hom.preserves_meet(&l1, &l2));
        assert!(hom.preserves_join(&l1, &l2));
        assert!(hom.is_valid(&l1, &l2));
    }

    #[test]
    fn test_complete_lattice_lfp() {
        let l = chain_lattice(5); // 0 < 1 < 2 < 3 < 4
        let cl = CompleteLattice::new(l);
        // f(x) = min(x+1, 4)
        let f = |x: usize| -> usize { (x + 1).min(4) };
        // Pre-fixed points: f(x) ≤ x, i.e. x+1 ≤ x only for x=4
        let lfp = cl.least_fixed_point(&f);
        assert_eq!(lfp, 4);
    }

    #[test]
    fn test_complete_lattice_gfp() {
        let l = chain_lattice(5);
        let cl = CompleteLattice::new(l);
        // f(x) = max(x-1, 0)
        let f = |x: usize| -> usize { if x > 0 { x - 1 } else { 0 } };
        let gfp = cl.greatest_fixed_point(&f);
        assert_eq!(gfp, 0);
    }

    #[test]
    fn test_iterative_lfp() {
        let l = chain_lattice(5);
        let cl = CompleteLattice::new(l);
        // f(x) = min(x+1, 3)
        let f = |x: usize| -> usize { (x + 1).min(3) };
        let lfp = cl.iterative_lfp(&f);
        assert_eq!(lfp, 3);
    }

    #[test]
    fn test_monotone_check() {
        let l = chain_lattice(4);
        let cl = CompleteLattice::new(l);
        let f = |x: usize| -> usize { (x + 1).min(3) };
        assert!(cl.is_monotone(&f));
        // Non-monotone
        let g = |x: usize| -> usize { if x == 1 { 0 } else { x } };
        assert!(!cl.is_monotone(&g));
    }

    #[test]
    fn test_congruence_generated() {
        let l = chain_lattice(4);
        let cong = LatticeCongruence::generated_by(&l, 1, 2);
        assert!(cong.is_valid(&l));
    }

    #[test]
    fn test_congruence_trivial() {
        let l = chain_lattice(3);
        let d = LatticeCongruence::discrete(3);
        assert!(d.is_valid(&l));
        let t = LatticeCongruence::total(3);
        assert!(t.is_valid(&l));
    }

    #[test]
    fn test_product_lattice() {
        let l1 = chain_lattice(2);
        let l2 = chain_lattice(2);
        let prod = product_lattice(&l1, &l2);
        assert_eq!(prod.size, 4);
    }

    #[test]
    fn test_graded() {
        let l = chain_lattice(4);
        assert!(LatticeProperties::is_graded(&l));
        let d = diamond_lattice(3);
        assert!(LatticeProperties::is_graded(&d));
    }

    #[test]
    fn test_maximal_chains() {
        let l = chain_lattice(4);
        let chains = l.maximal_chains();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_width_chain() {
        let l = chain_lattice(5);
        assert_eq!(l.width(), 1); // chain has width 1
    }
}
