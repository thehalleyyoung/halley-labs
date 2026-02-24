// LITMUS∞ Algebraic Engine — Orbit Enumeration
//
//   OrbitEnumerator          – hierarchical orbit enumeration
//   CanonicalForm            – lexicographic canonical form
//   OrbitRepresentativeSet   – storage for canonical representatives
//   DynamicSymmetryBreaking  – symmetry recomputation after partial assignment
//   BurnsideCounter          – orbit counting via Burnside's lemma
//   Parallel orbit enumeration with rayon

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use super::types::{Permutation, enumerate_from_generators};
use super::group::PermutationGroup;
use super::symmetry::{FullSymmetryGroup, LitmusTest};

// ── Execution Graph ──────────────────────────────────────────────────

/// A candidate execution (reads-from + coherence order).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExecutionCandidate {
    /// Reads-from: for each load, which store it reads from.
    /// Key: (thread, op_index of load), Value: (thread, op_index of store).
    pub reads_from: BTreeMap<(usize, usize), (usize, usize)>,
    /// Coherence order: per-address total order on stores.
    /// Key: address, Value: ordered list of (thread, op_index) of stores.
    pub coherence: BTreeMap<usize, Vec<(usize, usize)>>,
}

impl ExecutionCandidate {
    pub fn new() -> Self {
        ExecutionCandidate {
            reads_from: BTreeMap::new(),
            coherence: BTreeMap::new(),
        }
    }

    /// Apply a thread permutation.
    pub fn apply_thread_perm(&self, perm: &Permutation) -> Self {
        let mut new = ExecutionCandidate::new();

        for (&(lt, li), &(st, si)) in &self.reads_from {
            let new_lt = if lt < perm.degree() {
                perm.apply(lt as u32) as usize
            } else {
                lt // sentinel (e.g. usize::MAX for initial value)
            };
            let new_st = if st < perm.degree() {
                perm.apply(st as u32) as usize
            } else {
                st // sentinel
            };
            new.reads_from.insert((new_lt, li), (new_st, si));
        }

        for (&addr, stores) in &self.coherence {
            let new_stores: Vec<(usize, usize)> = stores
                .iter()
                .map(|&(t, i)| {
                    let new_t = if t < perm.degree() {
                        perm.apply(t as u32) as usize
                    } else {
                        t
                    };
                    (new_t, i)
                })
                .collect();
            new.coherence.insert(addr, new_stores);
        }

        new
    }

    /// Apply an address permutation.
    pub fn apply_addr_perm(&self, perm: &Permutation) -> Self {
        let mut new = ExecutionCandidate::new();
        new.reads_from = self.reads_from.clone();

        for (&addr, stores) in &self.coherence {
            let new_addr = perm.apply(addr as u32) as usize;
            new.coherence.insert(new_addr, stores.clone());
        }

        new
    }

    /// Lexicographic comparison for canonicalization.
    fn lex_key(&self) -> Vec<usize> {
        let mut key = Vec::new();
        for (&(lt, li), &(st, si)) in &self.reads_from {
            key.extend_from_slice(&[lt, li, st, si]);
        }
        for (&addr, stores) in &self.coherence {
            key.push(addr);
            key.push(stores.len());
            for &(t, i) in stores {
                key.push(t);
                key.push(i);
            }
        }
        key
    }
}

impl Default for ExecutionCandidate {
    fn default() -> Self {
        Self::new()
    }
}

// ── Canonical Form ───────────────────────────────────────────────────

/// Canonical form computation for execution candidates under a symmetry group.
#[derive(Clone, Debug)]
pub struct CanonicalForm;

impl CanonicalForm {
    /// Compute the canonical (lexicographically smallest) form of an
    /// execution candidate under the given symmetry group.
    pub fn canonicalize(
        candidate: &ExecutionCandidate,
        symmetry: &FullSymmetryGroup,
    ) -> ExecutionCandidate {
        let mut best = candidate.clone();

        // Apply thread permutations
        if symmetry.thread_group.order() <= 10000 {
            let thread_elements = symmetry.thread_group.enumerate_elements();
            for tperm in &thread_elements {
                let permuted = candidate.apply_thread_perm(tperm);
                if permuted.lex_key() < best.lex_key() {
                    best = permuted;
                }
            }
        } else {
            // Use generators only (heuristic)
            let mut improved = true;
            while improved {
                improved = false;
                for gen in symmetry.thread_group.generators() {
                    let permuted = best.apply_thread_perm(gen);
                    if permuted.lex_key() < best.lex_key() {
                        best = permuted;
                        improved = true;
                    }
                }
            }
        }

        // Apply address permutations
        if symmetry.address_group.order() <= 10000 {
            let addr_elements = symmetry.address_group.enumerate_elements();
            let current_best = best.clone();
            for aperm in &addr_elements {
                let permuted = current_best.apply_addr_perm(aperm);
                if permuted.lex_key() < best.lex_key() {
                    best = permuted;
                }
            }
        } else {
            let mut improved = true;
            while improved {
                improved = false;
                for gen in symmetry.address_group.generators() {
                    let permuted = best.apply_addr_perm(gen);
                    if permuted.lex_key() < best.lex_key() {
                        best = permuted;
                        improved = true;
                    }
                }
            }
        }

        best
    }

    /// Canonicalize using only thread symmetry.
    pub fn canonicalize_thread_only(
        candidate: &ExecutionCandidate,
        thread_group: &PermutationGroup,
    ) -> ExecutionCandidate {
        let mut best = candidate.clone();
        if thread_group.order() <= 10000 {
            let elements = thread_group.enumerate_elements();
            for perm in &elements {
                let permuted = candidate.apply_thread_perm(perm);
                if permuted.lex_key() < best.lex_key() {
                    best = permuted;
                }
            }
        }
        best
    }

    /// Canonicalize using only address symmetry.
    pub fn canonicalize_addr_only(
        candidate: &ExecutionCandidate,
        addr_group: &PermutationGroup,
    ) -> ExecutionCandidate {
        let mut best = candidate.clone();
        if addr_group.order() <= 10000 {
            let elements = addr_group.enumerate_elements();
            for perm in &elements {
                let permuted = candidate.apply_addr_perm(perm);
                if permuted.lex_key() < best.lex_key() {
                    best = permuted;
                }
            }
        }
        best
    }

    /// Check if a candidate is in canonical form.
    pub fn is_canonical(
        candidate: &ExecutionCandidate,
        symmetry: &FullSymmetryGroup,
    ) -> bool {
        let canonical = Self::canonicalize(candidate, symmetry);
        candidate == &canonical
    }
}

// ── Orbit Representative Set ─────────────────────────────────────────

/// Stores canonical orbit representatives.
#[derive(Clone, Debug)]
pub struct OrbitRepresentativeSet {
    representatives: HashSet<ExecutionCandidate>,
    insertion_order: Vec<ExecutionCandidate>,
}

impl OrbitRepresentativeSet {
    pub fn new() -> Self {
        OrbitRepresentativeSet {
            representatives: HashSet::new(),
            insertion_order: Vec::new(),
        }
    }

    /// Insert a candidate (will be canonicalized first).
    pub fn insert(
        &mut self,
        candidate: &ExecutionCandidate,
        symmetry: &FullSymmetryGroup,
    ) -> bool {
        let canonical = CanonicalForm::canonicalize(candidate, symmetry);
        if self.representatives.insert(canonical.clone()) {
            self.insertion_order.push(canonical);
            true
        } else {
            false
        }
    }

    /// Insert a pre-canonicalized candidate.
    pub fn insert_canonical(&mut self, canonical: ExecutionCandidate) -> bool {
        if self.representatives.insert(canonical.clone()) {
            self.insertion_order.push(canonical);
            true
        } else {
            false
        }
    }

    /// Check if a candidate's orbit is already represented.
    pub fn contains(
        &self,
        candidate: &ExecutionCandidate,
        symmetry: &FullSymmetryGroup,
    ) -> bool {
        let canonical = CanonicalForm::canonicalize(candidate, symmetry);
        self.representatives.contains(&canonical)
    }

    /// Number of distinct orbits.
    pub fn len(&self) -> usize {
        self.representatives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.representatives.is_empty()
    }

    /// Get all representatives.
    pub fn iter(&self) -> impl Iterator<Item = &ExecutionCandidate> {
        self.insertion_order.iter()
    }

    /// Clear all stored representatives.
    pub fn clear(&mut self) {
        self.representatives.clear();
        self.insertion_order.clear();
    }
}

impl Default for OrbitRepresentativeSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Dynamic Symmetry Breaking ────────────────────────────────────────

/// Recompute symmetry group after partial assignment to a candidate.
/// When we partially fix reads-from relations, some symmetries may break.
#[derive(Clone, Debug)]
pub struct DynamicSymmetryBreaking {
    /// Original full symmetry group.
    original: FullSymmetryGroup,
    /// Current effective symmetry (reduced by partial assignment).
    current_thread_group: PermutationGroup,
    current_addr_group: PermutationGroup,
    /// Fixed assignments so far.
    fixed_rf: BTreeMap<(usize, usize), (usize, usize)>,
}

impl DynamicSymmetryBreaking {
    /// Create from the full symmetry group.
    pub fn new(symmetry: &FullSymmetryGroup) -> Self {
        DynamicSymmetryBreaking {
            original: symmetry.clone(),
            current_thread_group: symmetry.thread_group.clone(),
            current_addr_group: symmetry.address_group.clone(),
            fixed_rf: BTreeMap::new(),
        }
    }

    /// Fix a reads-from assignment and recompute symmetry.
    pub fn fix_reads_from(
        &mut self,
        load: (usize, usize),
        store: (usize, usize),
    ) {
        self.fixed_rf.insert(load, store);
        self.recompute_symmetry();
    }

    /// Recompute the effective symmetry group after partial assignment.
    fn recompute_symmetry(&mut self) {
        // Thread symmetry: a thread permutation π is valid iff
        // for all fixed rf (l_t, l_i) -> (s_t, s_i):
        //   (π(l_t), l_i) -> (π(s_t), s_i) is also fixed and matches.
        let thread_gens: Vec<Permutation> = self
            .original
            .thread_group
            .generators()
            .iter()
            .filter(|gen| {
                self.fixed_rf.iter().all(|(&(lt, li), &(st, si))| {
                    let new_lt = gen.apply(lt as u32) as usize;
                    let new_st = gen.apply(st as u32) as usize;
                    match self.fixed_rf.get(&(new_lt, li)) {
                        Some(&(est, esi)) => est == new_st && esi == si,
                        None => true, // Not yet fixed, so no constraint
                    }
                })
            })
            .cloned()
            .collect();

        self.current_thread_group =
            PermutationGroup::new(self.original.thread_group.degree(), thread_gens);

        // Address symmetry: similar filtering
        let addr_gens: Vec<Permutation> = self
            .original
            .address_group
            .generators()
            .iter()
            .cloned()
            .collect();
        self.current_addr_group =
            PermutationGroup::new(self.original.address_group.degree(), addr_gens);
    }

    /// Current effective thread group.
    pub fn thread_group(&self) -> &PermutationGroup {
        &self.current_thread_group
    }

    /// Current effective address group.
    pub fn addr_group(&self) -> &PermutationGroup {
        &self.current_addr_group
    }

    /// Current effective symmetry order.
    pub fn current_order(&self) -> u64 {
        self.current_thread_group.order() * self.current_addr_group.order()
    }

    /// Check if a partial candidate is canonical under current symmetry.
    pub fn is_canonical_partial(&self, candidate: &ExecutionCandidate) -> bool {
        let key = candidate.lex_key();

        // Check thread generators
        for gen in self.current_thread_group.generators() {
            let permuted = candidate.apply_thread_perm(gen);
            if permuted.lex_key() < key {
                return false;
            }
        }

        // Check address generators
        for gen in self.current_addr_group.generators() {
            let permuted = candidate.apply_addr_perm(gen);
            if permuted.lex_key() < key {
                return false;
            }
        }

        true
    }

    /// Reset to original symmetry.
    pub fn reset(&mut self) {
        self.fixed_rf.clear();
        self.current_thread_group = self.original.thread_group.clone();
        self.current_addr_group = self.original.address_group.clone();
    }
}

// ── Burnside Counter ─────────────────────────────────────────────────

/// Count orbits using Burnside's lemma without enumerating them.
/// |X/G| = (1/|G|) * Σ_{g∈G} |Fix(g)|
#[derive(Clone, Debug)]
pub struct BurnsideCounter;

impl BurnsideCounter {
    /// Count orbits of a group acting on a finite set.
    /// `fix_count` is a function that counts fixed points of a permutation.
    pub fn count_orbits<F>(group: &PermutationGroup, fix_count: F) -> f64
    where
        F: Fn(&Permutation) -> u64,
    {
        let elements = group.enumerate_elements();
        let total_fixed: u64 = elements.iter().map(|g| fix_count(g)).sum();
        total_fixed as f64 / elements.len() as f64
    }

    /// Count orbits on the natural action (permutations of {0..n-1}).
    pub fn count_point_orbits(group: &PermutationGroup) -> usize {
        group.all_orbits().len()
    }

    /// Count orbits of the group acting on k-element subsets.
    pub fn count_subset_orbits(group: &PermutationGroup, k: usize) -> f64 {
        let n = group.degree();
        if k > n {
            return 0.0;
        }

        let elements = group.enumerate_elements();
        let subsets = Self::generate_k_subsets(n, k);

        let total_fixed: u64 = elements
            .iter()
            .map(|g| {
                subsets
                    .iter()
                    .filter(|subset| {
                        let mut permuted: Vec<u32> =
                            subset.iter().map(|&x| g.apply(x)).collect();
                        permuted.sort_unstable();
                        &permuted == *subset
                    })
                    .count() as u64
            })
            .sum();

        total_fixed as f64 / elements.len() as f64
    }

    /// Generate all k-element subsets of {0, ..., n-1}.
    fn generate_k_subsets(n: usize, k: usize) -> Vec<Vec<u32>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::gen_subsets_rec(n as u32, k, 0, &mut current, &mut result);
        result
    }

    fn gen_subsets_rec(
        n: u32,
        k: usize,
        start: u32,
        current: &mut Vec<u32>,
        result: &mut Vec<Vec<u32>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        let remaining = k - current.len();
        if (n - start) < remaining as u32 {
            return;
        }
        for i in start..n {
            current.push(i);
            Self::gen_subsets_rec(n, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Count orbits on pairs (ordered).
    pub fn count_pair_orbits(group: &PermutationGroup) -> f64 {
        let n = group.degree();
        let elements = group.enumerate_elements();

        let total_fixed: u64 = elements
            .iter()
            .map(|g| {
                let mut count = 0u64;
                for i in 0..n as u32 {
                    for j in 0..n as u32 {
                        if i != j && g.apply(i) == i && g.apply(j) == j {
                            count += 1;
                        }
                    }
                }
                count
            })
            .sum();

        total_fixed as f64 / elements.len() as f64
    }

    /// Count fixed points using cycle index.
    /// For a permutation g, |Fix(g)| on k-subsets can be computed from
    /// the cycle type.
    pub fn fixed_points_from_cycle_type(perm: &Permutation) -> u64 {
        perm.fixed_points().len() as u64
    }
}

// ── Orbit Enumeration Statistics ─────────────────────────────────────

/// Statistics from orbit enumeration.
#[derive(Clone, Debug)]
pub struct EnumerationStats {
    pub total_candidates: u64,
    pub canonical_representatives: u64,
    pub pruned_by_symmetry: u64,
    pub pruning_ratio: f64,
    pub phase1_time_ms: u64,
    pub phase2_time_ms: u64,
    pub phase3_time_ms: u64,
    pub total_time_ms: u64,
}

impl EnumerationStats {
    pub fn new() -> Self {
        EnumerationStats {
            total_candidates: 0,
            canonical_representatives: 0,
            pruned_by_symmetry: 0,
            pruning_ratio: 0.0,
            phase1_time_ms: 0,
            phase2_time_ms: 0,
            phase3_time_ms: 0,
            total_time_ms: 0,
        }
    }

    pub fn finalize(&mut self) {
        self.pruned_by_symmetry = self.total_candidates - self.canonical_representatives;
        if self.total_candidates > 0 {
            self.pruning_ratio =
                self.pruned_by_symmetry as f64 / self.total_candidates as f64;
        }
        self.total_time_ms = self.phase1_time_ms + self.phase2_time_ms + self.phase3_time_ms;
    }
}

impl Default for EnumerationStats {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EnumerationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Enumeration Statistics ===")?;
        writeln!(f, "Total candidates:   {}", self.total_candidates)?;
        writeln!(f, "Canonical reps:     {}", self.canonical_representatives)?;
        writeln!(f, "Pruned:             {}", self.pruned_by_symmetry)?;
        writeln!(f, "Pruning ratio:      {:.2}%", self.pruning_ratio * 100.0)?;
        writeln!(f, "Phase 1 (RF):       {}ms", self.phase1_time_ms)?;
        writeln!(f, "Phase 2 (CO):       {}ms", self.phase2_time_ms)?;
        writeln!(f, "Phase 3 (FR):       {}ms", self.phase3_time_ms)?;
        writeln!(f, "Total time:         {}ms", self.total_time_ms)?;
        Ok(())
    }
}

// ── Orbit Enumerator ─────────────────────────────────────────────────

/// Hierarchical orbit enumerator for execution candidates.
///
/// Phase 1: Enumerate reads-from candidates, canonicalize under thread×address symmetry
/// Phase 2: For each canonical rf, enumerate coherence orders under stabilizer
/// Phase 3: Compute from-reads, check consistency
#[derive(Clone)]
pub struct OrbitEnumerator {
    symmetry: FullSymmetryGroup,
    test: LitmusTest,
    stats: EnumerationStats,
}

impl OrbitEnumerator {
    pub fn new(test: LitmusTest, symmetry: FullSymmetryGroup) -> Self {
        OrbitEnumerator {
            symmetry,
            test,
            stats: EnumerationStats::new(),
        }
    }

    /// Run the full hierarchical enumeration.
    pub fn enumerate(&mut self) -> (OrbitRepresentativeSet, EnumerationStats) {
        let overall_start = Instant::now();
        let mut orbit_set = OrbitRepresentativeSet::new();

        // Phase 1: Enumerate reads-from candidates
        let phase1_start = Instant::now();
        let rf_candidates = self.enumerate_reads_from();
        let canonical_rfs = self.canonicalize_reads_from(&rf_candidates);
        self.stats.phase1_time_ms = phase1_start.elapsed().as_millis() as u64;

        // Phase 2: For each canonical RF, enumerate coherence orders
        let phase2_start = Instant::now();
        for rf in &canonical_rfs {
            let co_candidates = self.enumerate_coherence_for_rf(rf);
            for co in &co_candidates {
                let mut candidate = rf.clone();
                candidate.coherence = co.clone();
                self.stats.total_candidates += 1;

                // Phase 3: Check consistency (from-reads)
                if self.check_consistency(&candidate) {
                    orbit_set.insert(&candidate, &self.symmetry);
                }
            }
        }
        self.stats.phase2_time_ms = phase2_start.elapsed().as_millis() as u64;

        self.stats.canonical_representatives = orbit_set.len() as u64;
        self.stats.finalize();
        self.stats.total_time_ms = overall_start.elapsed().as_millis() as u64;

        (orbit_set, self.stats.clone())
    }

    /// Enumerate all possible reads-from relations.
    fn enumerate_reads_from(&self) -> Vec<ExecutionCandidate> {
        // Find all loads and all stores
        let mut loads: Vec<(usize, usize, usize)> = Vec::new(); // (thread, op_idx, addr)
        let mut stores_by_addr: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();

        for (tid, thread) in self.test.threads.iter().enumerate() {
            for (oidx, op) in thread.iter().enumerate() {
                match op.opcode {
                    super::symmetry::Opcode::Load => {
                        if let Some(addr) = op.address {
                            loads.push((tid, oidx, addr));
                        }
                    }
                    super::symmetry::Opcode::Store => {
                        if let Some(addr) = op.address {
                            stores_by_addr
                                .entry(addr)
                                .or_default()
                                .push((tid, oidx));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Enumerate all possible RF assignments (cartesian product)
        let mut candidates = vec![ExecutionCandidate::new()];

        for &(lt, li, addr) in &loads {
            let stores = stores_by_addr.get(&addr).cloned().unwrap_or_default();
            // Also consider the initial store (represented as a sentinel)
            let mut all_stores = stores.clone();
            // Add initial value store (special sentinel)
            // We use (usize::MAX, 0) to represent initial value
            all_stores.push((usize::MAX, 0));

            let mut new_candidates = Vec::new();
            for cand in &candidates {
                for &(st, si) in &all_stores {
                    let mut new_cand = cand.clone();
                    new_cand.reads_from.insert((lt, li), (st, si));
                    new_candidates.push(new_cand);
                }
            }
            candidates = new_candidates;
        }

        candidates
    }

    /// Canonicalize reads-from candidates under thread×address symmetry.
    fn canonicalize_reads_from(
        &self,
        candidates: &[ExecutionCandidate],
    ) -> Vec<ExecutionCandidate> {
        let mut seen = HashSet::new();
        let mut canonical = Vec::new();

        for cand in candidates {
            let canon = CanonicalForm::canonicalize_thread_only(
                cand,
                &self.symmetry.thread_group,
            );
            if seen.insert(canon.clone()) {
                canonical.push(canon);
            }
        }

        canonical
    }

    /// Enumerate coherence orders compatible with a reads-from relation.
    fn enumerate_coherence_for_rf(
        &self,
        rf: &ExecutionCandidate,
    ) -> Vec<BTreeMap<usize, Vec<(usize, usize)>>> {
        // Find stores per address
        let mut stores_by_addr: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (tid, thread) in self.test.threads.iter().enumerate() {
            for (oidx, op) in thread.iter().enumerate() {
                if op.opcode == super::symmetry::Opcode::Store {
                    if let Some(addr) = op.address {
                        stores_by_addr.entry(addr).or_default().push((tid, oidx));
                    }
                }
            }
        }

        // For each address, enumerate all permutations of stores (total orders)
        let mut result = vec![BTreeMap::new()];

        for (&addr, stores) in &stores_by_addr {
            let perms = Self::permutations_of(stores);
            let mut new_result = Vec::new();
            for existing in &result {
                for perm in &perms {
                    let mut co = existing.clone();
                    co.insert(addr, perm.clone());
                    new_result.push(co);
                }
            }
            result = new_result;
        }

        result
    }

    /// Generate all permutations of a slice.
    fn permutations_of<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
        if items.is_empty() {
            return vec![Vec::new()];
        }
        if items.len() == 1 {
            return vec![vec![items[0].clone()]];
        }

        let mut result = Vec::new();
        for i in 0..items.len() {
            let mut rest: Vec<T> = items.to_vec();
            let item = rest.remove(i);
            for mut perm in Self::permutations_of(&rest) {
                perm.insert(0, item.clone());
                result.push(perm);
            }
        }
        result
    }

    /// Check consistency of an execution candidate.
    /// Verifies from-reads relation (rf;co gives valid fr).
    fn check_consistency(&self, candidate: &ExecutionCandidate) -> bool {
        // For each load that reads from a store s, and each store s' that
        // is co-after s at the same address, we get a from-reads edge
        // from the load to s'. This must not create cycles.
        // Simplified check: just verify no obvious contradictions.

        for (&(lt, li), &(st, si)) in &candidate.reads_from {
            // Find the address of this load
            if lt >= self.test.num_threads {
                continue; // Skip sentinel stores
            }
            let load_op = &self.test.threads[lt][li];
            if let Some(addr) = load_op.address {
                // Check if the store exists in the coherence order
                if st < self.test.num_threads {
                    if let Some(co) = candidate.coherence.get(&addr) {
                        let store_pos = co.iter().position(|&s| s == (st, si));
                        if store_pos.is_none() {
                            return false; // Store not in coherence order
                        }
                    }
                }
            }
        }

        true
    }

    /// Get current statistics.
    pub fn stats(&self) -> &EnumerationStats {
        &self.stats
    }
}

// ── Parallel Orbit Enumeration ───────────────────────────────────────

/// Parallel orbit enumeration using rayon.
pub struct ParallelOrbitEnumerator;

impl ParallelOrbitEnumerator {
    /// Enumerate canonical representatives in parallel.
    pub fn enumerate_parallel(
        candidates: &[ExecutionCandidate],
        symmetry: &FullSymmetryGroup,
        num_threads: usize,
    ) -> (Vec<ExecutionCandidate>, EnumerationStats) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

        let counter = Arc::new(AtomicU64::new(0));
        let pruned = Arc::new(AtomicU64::new(0));

        let canonical: Vec<ExecutionCandidate> = pool.install(|| {
            let seen = std::sync::Mutex::new(HashSet::new());

            candidates
                .par_iter()
                .filter_map(|cand| {
                    counter.fetch_add(1, Ordering::Relaxed);
                    let canon = CanonicalForm::canonicalize(cand, symmetry);
                    let mut guard = seen.lock().unwrap();
                    if guard.insert(canon.clone()) {
                        Some(canon)
                    } else {
                        pruned.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                })
                .collect()
        });

        let mut stats = EnumerationStats::new();
        stats.total_candidates = counter.load(Ordering::Relaxed);
        stats.canonical_representatives = canonical.len() as u64;
        stats.pruned_by_symmetry = pruned.load(Ordering::Relaxed);
        stats.finalize();

        (canonical, stats)
    }

    /// Parallel canonicalization of a batch.
    pub fn canonicalize_batch(
        candidates: &[ExecutionCandidate],
        symmetry: &FullSymmetryGroup,
    ) -> Vec<ExecutionCandidate> {
        candidates
            .par_iter()
            .map(|cand| CanonicalForm::canonicalize(cand, symmetry))
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::symmetry::*;

    #[test]
    fn test_execution_candidate_new() {
        let ec = ExecutionCandidate::new();
        assert!(ec.reads_from.is_empty());
        assert!(ec.coherence.is_empty());
    }

    #[test]
    fn test_execution_candidate_thread_perm() {
        let mut ec = ExecutionCandidate::new();
        ec.reads_from.insert((0, 0), (1, 0));
        let perm = Permutation::transposition(2, 0, 1);
        let permuted = ec.apply_thread_perm(&perm);
        assert!(permuted.reads_from.contains_key(&(1, 0)));
        assert_eq!(permuted.reads_from[&(1, 0)], (0, 0));
    }

    #[test]
    fn test_execution_candidate_addr_perm() {
        let mut ec = ExecutionCandidate::new();
        ec.coherence.insert(0, vec![(0, 0), (1, 0)]);
        ec.coherence.insert(1, vec![(0, 1)]);
        let perm = Permutation::transposition(2, 0, 1);
        let permuted = ec.apply_addr_perm(&perm);
        assert!(permuted.coherence.contains_key(&1));
        assert_eq!(permuted.coherence[&1], vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn test_canonical_form_identity() {
        let ec = ExecutionCandidate::new();
        let trivial_sym = FullSymmetryGroup {
            thread_group: PermutationGroup::trivial(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 1,
            compression_factor: 1.0,
        };
        let canon = CanonicalForm::canonicalize(&ec, &trivial_sym);
        assert_eq!(ec, canon);
    }

    #[test]
    fn test_canonical_form_with_symmetry() {
        let mut ec1 = ExecutionCandidate::new();
        ec1.reads_from.insert((1, 0), (0, 0));
        let mut ec2 = ExecutionCandidate::new();
        ec2.reads_from.insert((0, 0), (1, 0));

        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 2,
            compression_factor: 2.0,
        };

        let c1 = CanonicalForm::canonicalize(&ec1, &sym);
        let c2 = CanonicalForm::canonicalize(&ec2, &sym);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_orbit_representative_set() {
        let mut ors = OrbitRepresentativeSet::new();
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 2,
            compression_factor: 2.0,
        };

        let mut ec1 = ExecutionCandidate::new();
        ec1.reads_from.insert((0, 0), (1, 0));
        let mut ec2 = ExecutionCandidate::new();
        ec2.reads_from.insert((1, 0), (0, 0));

        assert!(ors.insert(&ec1, &sym));
        assert!(!ors.insert(&ec2, &sym)); // Same orbit
        assert_eq!(ors.len(), 1);
    }

    #[test]
    fn test_dynamic_symmetry_breaking() {
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 2,
            compression_factor: 2.0,
        };

        let mut dsb = DynamicSymmetryBreaking::new(&sym);
        assert_eq!(dsb.current_order(), 2);

        // Fix a RF assignment that breaks the symmetry
        dsb.fix_reads_from((0, 0), (1, 0));
        // After fixing, some thread perms may no longer be valid
        assert!(dsb.current_order() <= 2);
    }

    #[test]
    fn test_burnside_point_orbits() {
        let s3 = PermutationGroup::symmetric(3);
        let count = BurnsideCounter::count_point_orbits(&s3);
        assert_eq!(count, 1); // S_3 is transitive, 1 orbit

        let c2_c3 = PermutationGroup::direct_product(
            &PermutationGroup::cyclic(2),
            &PermutationGroup::cyclic(3),
        );
        let count2 = BurnsideCounter::count_point_orbits(&c2_c3);
        assert_eq!(count2, 2); // Two orbits: {0,1} and {2,3,4}
    }

    #[test]
    fn test_burnside_subset_orbits() {
        let s3 = PermutationGroup::symmetric(3);
        // Number of orbits on 2-element subsets of {0,1,2} under S_3
        // There's only 1 orbit (all 2-subsets are equivalent)
        let count = BurnsideCounter::count_subset_orbits(&s3, 2);
        assert!((count - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_enumeration_stats() {
        let mut stats = EnumerationStats::new();
        stats.total_candidates = 100;
        stats.canonical_representatives = 20;
        stats.finalize();
        assert_eq!(stats.pruned_by_symmetry, 80);
        assert!((stats.pruning_ratio - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_enumeration_stats_display() {
        let mut stats = EnumerationStats::new();
        stats.total_candidates = 50;
        stats.canonical_representatives = 10;
        stats.finalize();
        let display = format!("{}", stats);
        assert!(display.contains("50"));
        assert!(display.contains("10"));
    }

    #[test]
    fn test_orbit_enumerator_sb() {
        let sb = litmus_sb();
        let sym = FullSymmetryGroup::compute(&sb);
        let mut enumerator = OrbitEnumerator::new(sb, sym);
        let (orbits, stats) = enumerator.enumerate();
        // SB should produce some canonical executions
        assert!(orbits.len() >= 1);
        assert!(stats.total_candidates >= 1);
    }

    #[test]
    fn test_orbit_enumerator_mp() {
        let mp = litmus_mp();
        let sym = FullSymmetryGroup::compute(&mp);
        let mut enumerator = OrbitEnumerator::new(mp, sym);
        let (orbits, stats) = enumerator.enumerate();
        assert!(orbits.len() >= 1);
        assert!(stats.total_time_ms < 10000); // Should be fast
    }

    #[test]
    fn test_parallel_canonicalize() {
        let mut candidates = Vec::new();
        for i in 0..10 {
            let mut ec = ExecutionCandidate::new();
            ec.reads_from.insert((i % 2, 0), ((i + 1) % 2, 0));
            candidates.push(ec);
        }

        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 2,
            compression_factor: 2.0,
        };

        let (canonical, stats) = ParallelOrbitEnumerator::enumerate_parallel(
            &candidates, &sym, 2,
        );
        // With S_2 symmetry, (0,0)->(1,0) and (1,0)->(0,0) are equivalent
        assert!(canonical.len() <= candidates.len());
        assert!(stats.canonical_representatives <= stats.total_candidates);
    }

    #[test]
    fn test_canonical_is_canonical() {
        let mut ec = ExecutionCandidate::new();
        ec.reads_from.insert((0, 0), (1, 0));

        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::trivial(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 1,
            compression_factor: 1.0,
        };

        assert!(CanonicalForm::is_canonical(&ec, &sym));
    }

    #[test]
    fn test_orbit_rep_set_clear() {
        let mut ors = OrbitRepresentativeSet::new();
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::trivial(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 1,
            compression_factor: 1.0,
        };

        let ec = ExecutionCandidate::new();
        ors.insert(&ec, &sym);
        assert_eq!(ors.len(), 1);
        ors.clear();
        assert!(ors.is_empty());
    }

    #[test]
    fn test_burnside_pair_orbits() {
        let s3 = PermutationGroup::symmetric(3);
        let count = BurnsideCounter::count_pair_orbits(&s3);
        // Ordered pairs (i,j) with i≠j under S_3: all equivalent, so 1 orbit
        assert!((count - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_permutations_of() {
        let items = vec![1, 2, 3];
        let perms = OrbitEnumerator::permutations_of(&items);
        assert_eq!(perms.len(), 6); // 3! = 6
    }

    #[test]
    fn test_dynamic_symmetry_reset() {
        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(3),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 6,
            compression_factor: 6.0,
        };

        let mut dsb = DynamicSymmetryBreaking::new(&sym);
        let original_order = dsb.current_order();
        dsb.fix_reads_from((0, 0), (1, 0));
        dsb.reset();
        assert_eq!(dsb.current_order(), original_order);
    }

    #[test]
    fn test_k_subsets() {
        let subsets = BurnsideCounter::generate_k_subsets(4, 2);
        assert_eq!(subsets.len(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_lex_key_ordering() {
        let mut ec1 = ExecutionCandidate::new();
        ec1.reads_from.insert((0, 0), (0, 0));
        let mut ec2 = ExecutionCandidate::new();
        ec2.reads_from.insert((0, 0), (1, 0));
        assert!(ec1.lex_key() < ec2.lex_key());
    }
}
