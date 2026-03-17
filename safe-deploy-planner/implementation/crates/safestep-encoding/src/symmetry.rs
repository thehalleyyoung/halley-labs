use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Represents a replica state as (old_version_count, new_version_count)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaState {
    pub old_count: usize,
    pub new_count: usize,
    pub total: usize,
}

impl ReplicaState {
    pub fn new(old_count: usize, new_count: usize) -> Self {
        Self {
            old_count,
            new_count,
            total: old_count + new_count,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.old_count + self.new_count == self.total && self.total > 0
    }

    pub fn all_old(total: usize) -> Self {
        Self { old_count: total, new_count: 0, total }
    }

    pub fn all_new(total: usize) -> Self {
        Self { old_count: 0, new_count: total, total }
    }

    pub fn can_transition_to(&self, other: &ReplicaState) -> bool {
        self.total == other.total
            && (self.old_count as i64 - other.old_count as i64).abs() <= 1
            && (self.new_count as i64 - other.new_count as i64).abs() <= 1
    }

    pub fn maintains_minimum(&self, min_available: usize) -> bool {
        self.old_count + self.new_count >= min_available
    }

    pub fn progress(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.new_count as f64 / self.total as f64
    }
}

impl std::fmt::Display for ReplicaState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(old={}, new={}, total={})", self.old_count, self.new_count, self.total)
    }
}

/// Computes reduced state space for replicated services
pub struct ReplicaSymmetry;

impl ReplicaSymmetry {
    /// Enumerate all valid (old, new) pairs for a rolling update
    pub fn reduce(total_replicas: usize) -> Vec<ReplicaState> {
        let mut states = Vec::new();
        for old in 0..=total_replicas {
            let new = total_replicas - old;
            states.push(ReplicaState::new(old, new));
        }
        states
    }

    /// Enumerate states maintaining minimum availability
    pub fn reduce_with_minimum(total_replicas: usize, min_available: usize) -> Vec<ReplicaState> {
        Self::reduce(total_replicas)
            .into_iter()
            .filter(|s| s.maintains_minimum(min_available))
            .collect()
    }

    /// Count reduced states (O(L^2) instead of L^r)
    pub fn state_count(total_replicas: usize) -> usize {
        total_replicas + 1
    }

    /// Get valid transitions from a state respecting min_available
    pub fn valid_transitions(
        state: &ReplicaState,
        min_available: usize,
    ) -> Vec<ReplicaState> {
        let mut transitions = Vec::new();

        // Can convert one old replica to new
        if state.old_count > 0 {
            let next = ReplicaState::new(state.old_count - 1, state.new_count + 1);
            if next.maintains_minimum(min_available) {
                transitions.push(next);
            }
        }

        // Can also temporarily lose one replica during transition
        if state.old_count > 0 && state.total > min_available {
            // Remove old, then add new (two-step but modeled as one)
            let intermediate_available = state.old_count - 1 + state.new_count;
            if intermediate_available >= min_available {
                let next = ReplicaState::new(state.old_count - 1, state.new_count + 1);
                if !transitions.contains(&next) && next.maintains_minimum(min_available) {
                    transitions.push(next);
                }
            }
        }

        transitions
    }

    /// Compute the full transition graph for a rolling update
    pub fn transition_graph(
        total_replicas: usize,
        min_available: usize,
    ) -> Vec<(ReplicaState, ReplicaState)> {
        let states = Self::reduce_with_minimum(total_replicas, min_available);
        let mut edges = Vec::new();

        for state in &states {
            let transitions = Self::valid_transitions(state, min_available);
            for next in transitions {
                if states.contains(&next) {
                    edges.push((state.clone(), next));
                }
            }
        }

        edges
    }

    /// Check if a rolling update path exists from all-old to all-new
    pub fn rolling_update_possible(total_replicas: usize, min_available: usize) -> bool {
        if min_available > total_replicas {
            return false;
        }
        // A rolling update is always possible if min_available <= total - 1
        // because we can always replace one at a time
        min_available < total_replicas
    }

    /// Compute minimum steps for rolling update
    pub fn minimum_steps(total_replicas: usize, min_available: usize) -> Option<usize> {
        if !Self::rolling_update_possible(total_replicas, min_available) {
            return None;
        }
        // Each step converts one old to new
        Some(total_replicas)
    }

    /// Encode rolling update as SAT clauses
    pub fn encode_rolling_update(
        total_replicas: usize,
        min_available: usize,
        step_var_base: usize,
        num_steps: usize,
    ) -> Vec<Vec<i32>> {
        let mut clauses = Vec::new();
        let bits_needed = bits_for_value(total_replicas);

        // At each step, old_count is represented in binary
        // old_count[step] encoded as bits starting at step_var_base + step * bits_needed
        for step in 0..num_steps.saturating_sub(1) {
            let curr_base = step_var_base + step * bits_needed;
            let next_base = step_var_base + (step + 1) * bits_needed;

            // Transition: old_count decreases by at most 1
            // This is encoded as: next_old <= curr_old AND next_old >= curr_old - 1
            // Simplified: use comparison circuits
            for bit in 0..bits_needed {
                let curr_var = (curr_base + bit + 1) as i32;
                let next_var = (next_base + bit + 1) as i32;
                // If current bit is 0 and no borrow, next bit should be 0 or stay same
                // This is a simplified encoding
                clauses.push(vec![curr_var, -next_var]);
                clauses.push(vec![-curr_var, next_var, -((next_base + bits_needed + 1) as i32)]);
            }

            // Minimum availability: old_count + new_count >= min_available
            // Since old_count + new_count = total at all times during atomic transition
            // We just need total >= min_available, which is always true if total > min_available
            if total_replicas >= min_available {
                // Trivially satisfied, add tautological clause
                let v = (curr_base + 1) as i32;
                clauses.push(vec![v, -v]);
            }
        }

        clauses
    }
}

fn bits_for_value(max_val: usize) -> usize {
    if max_val == 0 {
        return 1;
    }
    (usize::BITS - max_val.leading_zeros()) as usize
}

/// Encodes replica-based rolling updates into SAT clauses.
///
/// Wraps `ReplicaSymmetry` with a stateful encoder interface that tracks
/// the total number of replicas, minimum availability, and step layout.
pub struct ReplicaEncoder {
    total_replicas: usize,
    min_available: usize,
}

impl ReplicaEncoder {
    /// Create a new replica encoder.
    pub fn new(total_replicas: usize, min_available: usize) -> Self {
        Self { total_replicas, min_available }
    }

    /// Encode the rolling update as SAT clauses.
    pub fn encode(&self, step_var_base: usize, num_steps: usize) -> Vec<Vec<i32>> {
        ReplicaSymmetry::encode_rolling_update(
            self.total_replicas,
            self.min_available,
            step_var_base,
            num_steps,
        )
    }

    /// Get the reduced state space.
    pub fn states(&self) -> Vec<ReplicaState> {
        ReplicaSymmetry::reduce_with_minimum(self.total_replicas, self.min_available)
    }

    /// Get valid transitions from a given state.
    pub fn transitions(&self, state: &ReplicaState) -> Vec<ReplicaState> {
        ReplicaSymmetry::valid_transitions(state, self.min_available)
    }

    /// Whether a rolling update is feasible.
    pub fn is_feasible(&self) -> bool {
        ReplicaSymmetry::rolling_update_possible(self.total_replicas, self.min_available)
    }

    /// Minimum steps required.
    pub fn min_steps(&self) -> Option<usize> {
        ReplicaSymmetry::minimum_steps(self.total_replicas, self.min_available)
    }
}

/// Detects symmetries in constraint graphs using color refinement
pub struct SymmetryDetector;

impl SymmetryDetector {
    /// Detect automorphisms using color refinement (WL algorithm)
    pub fn detect_automorphisms(
        adj: &[Vec<usize>],
        labels: &[usize],
    ) -> Vec<Vec<usize>> {
        let n = adj.len();
        if n == 0 {
            return vec![];
        }

        let colors = ColorRefinement::refine(adj, labels);
        let mut color_classes: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &c) in colors.iter().enumerate() {
            color_classes.entry(c).or_default().push(i);
        }

        // Generate permutations from color classes
        let mut permutations = Vec::new();
        let symmetric_classes: Vec<&Vec<usize>> = color_classes.values()
            .filter(|class| class.len() > 1)
            .collect();

        for class in symmetric_classes {
            if class.len() == 2 {
                // Simple swap
                let mut perm: Vec<usize> = (0..n).collect();
                perm[class[0]] = class[1];
                perm[class[1]] = class[0];

                // Verify it's actually an automorphism
                if Self::is_automorphism(adj, &perm) {
                    permutations.push(perm);
                }
            } else if class.len() <= 6 {
                // Try adjacent transpositions
                for i in 0..class.len() - 1 {
                    let mut perm: Vec<usize> = (0..n).collect();
                    perm[class[i]] = class[i + 1];
                    perm[class[i + 1]] = class[i];
                    if Self::is_automorphism(adj, &perm) {
                        permutations.push(perm);
                    }
                }
            }
        }

        permutations
    }

    /// Check if a permutation is a graph automorphism
    pub fn is_automorphism(adj: &[Vec<usize>], perm: &[usize]) -> bool {
        let n = adj.len();
        if perm.len() != n {
            return false;
        }

        for u in 0..n {
            let pu = perm[u];
            for &v in &adj[u] {
                let pv = perm[v];
                if !adj[pu].contains(&pv) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute orbit partition from permutation group
    pub fn orbit_partition(permutations: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            let mut x = x;
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }

        for perm in permutations {
            for (i, &pi) in perm.iter().enumerate() {
                if i != pi {
                    union(&mut parent, i, pi);
                }
            }
        }

        let mut orbits: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            orbits.entry(root).or_default().push(i);
        }

        let mut result: Vec<Vec<usize>> = orbits.into_values().collect();
        result.sort_by_key(|orbit| orbit[0]);
        result
    }
}

/// Generates symmetry-breaking clauses
pub struct SymmetryBreaker;

impl SymmetryBreaker {
    /// Generate lex-leader symmetry breaking constraints
    pub fn lex_leader_constraints(
        permutations: &[Vec<usize>],
        num_vars: usize,
    ) -> Vec<Vec<i32>> {
        let mut clauses = Vec::new();

        for perm in permutations {
            // For each permutation σ, add: x ≤_lex σ(x)
            // Encoded as: for the first position where they differ,
            // the original must be ≤ the permuted
            let mut is_identity = true;
            for i in 0..num_vars.min(perm.len()) {
                if perm[i] != i {
                    is_identity = false;
                    break;
                }
            }
            if is_identity {
                continue;
            }

            // Simple encoding: for first differing position
            for i in 0..num_vars.min(perm.len()) {
                if perm[i] != i && perm[i] < num_vars {
                    let x_i = (i + 1) as i32;
                    let x_pi = (perm[i] + 1) as i32;
                    // x_i => x_{σ(i)} (if original is true, permuted must be true)
                    clauses.push(vec![-x_i, x_pi]);
                    break;
                }
            }
        }

        clauses
    }

    /// Generate symmetry breaking for service replicas
    pub fn replica_symmetry_breaking(
        service_idx: usize,
        num_replicas: usize,
        var_base: usize,
    ) -> Vec<Vec<i32>> {
        let mut clauses = Vec::new();

        // Order replicas lexicographically: replica[i] <= replica[i+1]
        for i in 0..num_replicas.saturating_sub(1) {
            let var_i = (var_base + i + 1) as i32;
            let var_next = (var_base + i + 2) as i32;
            // If replica i is assigned version v, replica i+1 must be >= v
            clauses.push(vec![-var_i, var_next]);
        }

        let _ = service_idx; // Used for labeling in larger context
        clauses
    }
}

/// Color refinement (1-dimensional Weisfeiler-Leman)
pub struct ColorRefinement;

impl ColorRefinement {
    /// Refine vertex colors based on neighbor multisets
    pub fn refine(adj: &[Vec<usize>], initial_colors: &[usize]) -> Vec<usize> {
        let n = adj.len();
        let mut colors = initial_colors.to_vec();
        if colors.len() < n {
            colors.resize(n, 0);
        }

        let max_iterations = n + 1;
        for _ in 0..max_iterations {
            let mut new_colors = Vec::with_capacity(n);
            let mut color_map: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
            let mut next_color = 0usize;

            for v in 0..n {
                let mut neighbor_colors: Vec<usize> = adj[v].iter()
                    .map(|&u| colors[u])
                    .collect();
                neighbor_colors.sort_unstable();

                let key = (colors[v], neighbor_colors);
                let c = *color_map.entry(key).or_insert_with(|| {
                    let c = next_color;
                    next_color += 1;
                    c
                });
                new_colors.push(c);
            }

            if new_colors == colors {
                break; // Fixed point reached
            }
            colors = new_colors;
        }

        colors
    }

    /// Check if two graphs are potentially isomorphic based on color refinement
    pub fn possibly_isomorphic(
        adj1: &[Vec<usize>],
        adj2: &[Vec<usize>],
    ) -> bool {
        if adj1.len() != adj2.len() {
            return false;
        }

        let n = adj1.len();
        let initial = vec![0; n];
        let colors1 = Self::refine(adj1, &initial);
        let colors2 = Self::refine(adj2, &initial);

        let mut hist1: HashMap<usize, usize> = HashMap::new();
        let mut hist2: HashMap<usize, usize> = HashMap::new();

        for &c in &colors1 {
            *hist1.entry(c).or_insert(0) += 1;
        }
        for &c in &colors2 {
            *hist2.entry(c).or_insert(0) += 1;
        }

        // Compare color histograms
        let mut counts1: Vec<usize> = hist1.values().copied().collect();
        let mut counts2: Vec<usize> = hist2.values().copied().collect();
        counts1.sort_unstable();
        counts2.sort_unstable();
        counts1 == counts2
    }

    /// Compute the color histogram: maps color → count.
    pub fn color_histogram(adj: &[Vec<usize>], initial_colors: &[usize]) -> HashMap<usize, usize> {
        let colors = Self::refine(adj, initial_colors);
        let mut hist: HashMap<usize, usize> = HashMap::new();
        for &c in &colors {
            *hist.entry(c).or_insert(0) += 1;
        }
        hist
    }

    /// Number of distinct colors after refinement.
    pub fn num_colors(adj: &[Vec<usize>], initial_colors: &[usize]) -> usize {
        let colors = Self::refine(adj, initial_colors);
        let unique: HashSet<usize> = colors.iter().copied().collect();
        unique.len()
    }
}

// ---------------------------------------------------------------------------
// SymmetryStats
// ---------------------------------------------------------------------------

/// Statistics about detected symmetries.
#[derive(Debug, Clone)]
pub struct SymmetryStats {
    /// Number of vertices in the graph.
    pub num_vertices: usize,
    /// Number of automorphisms detected.
    pub num_automorphisms: usize,
    /// Number of orbits in the partition.
    pub num_orbits: usize,
    /// Sizes of the orbits (sorted descending).
    pub orbit_sizes: Vec<usize>,
    /// Number of distinct colors after WL refinement.
    pub num_colors: usize,
}

impl SymmetryStats {
    /// Compute symmetry statistics for a labeled graph.
    pub fn compute(adj: &[Vec<usize>], labels: &[usize]) -> Self {
        let n = adj.len();
        let autos = SymmetryDetector::detect_automorphisms(adj, labels);
        let orbits = SymmetryDetector::orbit_partition(&autos, n);
        let mut orbit_sizes: Vec<usize> = orbits.iter().map(|o| o.len()).collect();
        orbit_sizes.sort_unstable_by(|a, b| b.cmp(a));
        let num_colors = ColorRefinement::num_colors(adj, labels);

        Self {
            num_vertices: n,
            num_automorphisms: autos.len(),
            num_orbits: orbits.len(),
            orbit_sizes,
            num_colors,
        }
    }

    /// Symmetry reduction factor: n / num_orbits.
    pub fn reduction_factor(&self) -> f64 {
        if self.num_orbits == 0 {
            return 1.0;
        }
        self.num_vertices as f64 / self.num_orbits as f64
    }

    /// Whether the graph has any non-trivial symmetry.
    pub fn has_symmetry(&self) -> bool {
        self.num_automorphisms > 0
    }
}

// ---------------------------------------------------------------------------
// CanonicalForm
// ---------------------------------------------------------------------------

/// Computes a canonical ordering of vertices using color refinement.
pub struct CanonicalForm;

impl CanonicalForm {
    /// Compute a canonical vertex ordering based on color refinement.
    ///
    /// Vertices are ordered by (color, degree, index) which provides a
    /// deterministic ordering invariant under isomorphism (heuristic).
    pub fn canonical_ordering(adj: &[Vec<usize>]) -> Vec<usize> {
        let n = adj.len();
        let initial = vec![0usize; n];
        let colors = ColorRefinement::refine(adj, &initial);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| (colors[i], adj[i].len(), i));
        indices
    }

    /// Canonical adjacency matrix (upper triangle, row-major) as a bit string.
    pub fn canonical_hash(adj: &[Vec<usize>]) -> Vec<bool> {
        let ordering = Self::canonical_ordering(adj);
        let n = ordering.len();
        let mut inv = vec![0usize; n];
        for (pos, &v) in ordering.iter().enumerate() {
            inv[v] = pos;
        }

        let mut bits = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let vi = ordering[i];
                let vj = ordering[j];
                bits.push(adj[vi].contains(&vj));
            }
        }
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_state_basic() {
        let state = ReplicaState::new(3, 2);
        assert_eq!(state.total, 5);
        assert!(state.is_valid());
        assert!(state.maintains_minimum(4));
        assert!(!state.maintains_minimum(6));
    }

    #[test]
    fn test_replica_state_progress() {
        let state = ReplicaState::all_old(4);
        assert_eq!(state.progress(), 0.0);
        let state = ReplicaState::all_new(4);
        assert_eq!(state.progress(), 1.0);
        let state = ReplicaState::new(2, 2);
        assert_eq!(state.progress(), 0.5);
    }

    #[test]
    fn test_replica_transition() {
        let s1 = ReplicaState::new(3, 0);
        let s2 = ReplicaState::new(2, 1);
        assert!(s1.can_transition_to(&s2));
        let s3 = ReplicaState::new(1, 2);
        assert!(!s1.can_transition_to(&s3)); // Too big a jump
    }

    #[test]
    fn test_replica_symmetry_reduce() {
        let states = ReplicaSymmetry::reduce(3);
        assert_eq!(states.len(), 4); // (3,0), (2,1), (1,2), (0,3)
    }

    #[test]
    fn test_replica_symmetry_with_minimum() {
        let states = ReplicaSymmetry::reduce_with_minimum(4, 3);
        assert!(states.iter().all(|s| s.maintains_minimum(3)));
        // All (old+new=4) >= 3, so all 5 states pass
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn test_valid_transitions() {
        let state = ReplicaState::new(3, 0);
        let transitions = ReplicaSymmetry::valid_transitions(&state, 2);
        assert!(!transitions.is_empty());
        for t in &transitions {
            assert!(t.maintains_minimum(2));
        }
    }

    #[test]
    fn test_rolling_update_possible() {
        assert!(ReplicaSymmetry::rolling_update_possible(3, 2));
        assert!(ReplicaSymmetry::rolling_update_possible(5, 3));
        assert!(!ReplicaSymmetry::rolling_update_possible(3, 3));
        assert!(!ReplicaSymmetry::rolling_update_possible(3, 4));
    }

    #[test]
    fn test_minimum_steps() {
        assert_eq!(ReplicaSymmetry::minimum_steps(3, 2), Some(3));
        assert_eq!(ReplicaSymmetry::minimum_steps(5, 3), Some(5));
        assert_eq!(ReplicaSymmetry::minimum_steps(3, 3), None);
    }

    #[test]
    fn test_state_count() {
        assert_eq!(ReplicaSymmetry::state_count(3), 4);
        assert_eq!(ReplicaSymmetry::state_count(10), 11);
    }

    #[test]
    fn test_color_refinement_path() {
        let adj = vec![vec![1], vec![0, 2], vec![1]];
        let initial = vec![0, 0, 0];
        let colors = ColorRefinement::refine(&adj, &initial);
        // End vertices should have same color, different from middle
        assert_eq!(colors[0], colors[2]);
        assert_ne!(colors[0], colors[1]);
    }

    #[test]
    fn test_color_refinement_complete() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let initial = vec![0, 0, 0];
        let colors = ColorRefinement::refine(&adj, &initial);
        // All vertices in K3 should have same color (all symmetric)
        assert_eq!(colors[0], colors[1]);
        assert_eq!(colors[1], colors[2]);
    }

    #[test]
    fn test_possibly_isomorphic() {
        let adj1 = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // K3
        let adj2 = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // K3
        assert!(ColorRefinement::possibly_isomorphic(&adj1, &adj2));

        let adj3 = vec![vec![1], vec![0, 2], vec![1]]; // P3
        assert!(!ColorRefinement::possibly_isomorphic(&adj1, &adj3));
    }

    #[test]
    fn test_is_automorphism() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // K3
        let perm = vec![1, 0, 2]; // Swap 0 and 1
        assert!(SymmetryDetector::is_automorphism(&adj, &perm));

        let perm2 = vec![2, 1, 0]; // Swap 0 and 2
        assert!(SymmetryDetector::is_automorphism(&adj, &perm2));

        let path = vec![vec![1], vec![0, 2], vec![1]]; // P3
        let swap_ends = vec![2, 1, 0];
        assert!(SymmetryDetector::is_automorphism(&path, &swap_ends));
    }

    #[test]
    fn test_orbit_partition() {
        let perms = vec![vec![1, 0, 2]]; // Swap 0,1
        let orbits = SymmetryDetector::orbit_partition(&perms, 3);
        assert_eq!(orbits.len(), 2); // {0,1} and {2}
    }

    #[test]
    fn test_lex_leader_constraints() {
        let perms = vec![vec![1, 0, 2]];
        let clauses = SymmetryBreaker::lex_leader_constraints(&perms, 3);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_detect_automorphisms() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // K3
        let labels = vec![0, 0, 0];
        let autos = SymmetryDetector::detect_automorphisms(&adj, &labels);
        // K3 has 6 automorphisms, but we detect swap-based ones
        assert!(!autos.is_empty());
    }

    #[test]
    fn test_encode_rolling_update() {
        let clauses = ReplicaSymmetry::encode_rolling_update(3, 2, 0, 4);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_transition_graph() {
        let edges = ReplicaSymmetry::transition_graph(3, 2);
        assert!(!edges.is_empty());
        // Should have path from all-old to all-new
        let start = ReplicaState::all_old(3);
        let has_start_edge = edges.iter().any(|(from, _)| *from == start);
        assert!(has_start_edge);
    }

    #[test]
    fn test_replica_symmetry_breaking() {
        let clauses = SymmetryBreaker::replica_symmetry_breaking(0, 3, 10);
        assert_eq!(clauses.len(), 2); // 3-1 = 2 ordering constraints
    }

    #[test]
    fn test_bits_for_value() {
        assert_eq!(bits_for_value(0), 1);
        assert_eq!(bits_for_value(1), 1);
        assert_eq!(bits_for_value(2), 2);
        assert_eq!(bits_for_value(3), 2);
        assert_eq!(bits_for_value(4), 3);
        assert_eq!(bits_for_value(7), 3);
        assert_eq!(bits_for_value(8), 4);
    }

    #[test]
    fn test_replica_encoder_basic() {
        let enc = ReplicaEncoder::new(5, 3);
        assert!(enc.is_feasible());
        assert_eq!(enc.min_steps(), Some(5));
    }

    #[test]
    fn test_replica_encoder_infeasible() {
        let enc = ReplicaEncoder::new(3, 3);
        assert!(!enc.is_feasible());
        assert_eq!(enc.min_steps(), None);
    }

    #[test]
    fn test_replica_encoder_states() {
        let enc = ReplicaEncoder::new(4, 2);
        let states = enc.states();
        assert!(states.len() >= 3);
        assert!(states.iter().all(|s| s.maintains_minimum(2)));
    }

    #[test]
    fn test_replica_encoder_transitions() {
        let enc = ReplicaEncoder::new(4, 2);
        let start = ReplicaState::all_old(4);
        let transitions = enc.transitions(&start);
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_replica_encoder_encode() {
        let enc = ReplicaEncoder::new(3, 2);
        let clauses = enc.encode(0, 4);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_symmetry_stats_k3() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let labels = vec![0, 0, 0];
        let stats = SymmetryStats::compute(&adj, &labels);
        assert!(stats.has_symmetry());
        assert!(stats.num_automorphisms > 0);
        assert!(stats.reduction_factor() >= 1.0);
    }

    #[test]
    fn test_symmetry_stats_no_symmetry() {
        // Path graph with distinct labels
        let adj = vec![vec![1], vec![0, 2], vec![1]];
        let labels = vec![0, 1, 2];
        let stats = SymmetryStats::compute(&adj, &labels);
        assert_eq!(stats.num_vertices, 3);
    }

    #[test]
    fn test_canonical_ordering_k3() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let order = CanonicalForm::canonical_ordering(&adj);
        assert_eq!(order.len(), 3);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_canonical_hash_isomorphic() {
        // Two isomorphic graphs (same graph, different vertex labels)
        let adj1 = vec![vec![1], vec![0, 2], vec![1]]; // 0-1-2
        let adj2 = vec![vec![1], vec![0, 2], vec![1]]; // same
        let h1 = CanonicalForm::canonical_hash(&adj1);
        let h2 = CanonicalForm::canonical_hash(&adj2);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_canonical_hash_different() {
        let adj1 = vec![vec![1], vec![0]]; // edge 0-1
        let adj2 = vec![vec![], vec![]]; // no edges
        let h1 = CanonicalForm::canonical_hash(&adj1);
        let h2 = CanonicalForm::canonical_hash(&adj2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_color_histogram() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // K3
        let hist = ColorRefinement::color_histogram(&adj, &[0, 0, 0]);
        // All same color in K3
        assert_eq!(hist.len(), 1);
        assert_eq!(*hist.values().next().unwrap(), 3);
    }

    #[test]
    fn test_num_colors_path() {
        let adj = vec![vec![1], vec![0, 2], vec![1]];
        let nc = ColorRefinement::num_colors(&adj, &[0, 0, 0]);
        assert_eq!(nc, 2); // endpoints vs middle
    }
}
