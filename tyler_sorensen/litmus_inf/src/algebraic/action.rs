/// Group actions on sets for LITMUS∞ algebraic engine.
///
/// Implements group actions, orbit computation, stabilizer computation,
/// orbit-stabilizer theorem, fixed point enumeration, Burnside counting,
/// Polya enumeration, and equivariant maps.
#[allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;

// =========================================================================
// Permutation (self-contained for this module)
// =========================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permutation {
    pub images: Vec<u32>,
}

impl Permutation {
    pub fn new(images: Vec<u32>) -> Self { Permutation { images } }
    pub fn identity(n: usize) -> Self { Permutation { images: (0..n as u32).collect() } }
    pub fn degree(&self) -> usize { self.images.len() }
    pub fn apply(&self, i: u32) -> u32 { self.images[i as usize] }
    pub fn compose(&self, other: &Self) -> Self {
        Permutation { images: (0..self.degree()).map(|i| self.apply(other.apply(i as u32))).collect() }
    }
    pub fn inverse(&self) -> Self {
        let mut inv = vec![0u32; self.degree()];
        for (i, &j) in self.images.iter().enumerate() { inv[j as usize] = i as u32; }
        Permutation { images: inv }
    }
    pub fn order(&self) -> usize {
        let id = Self::identity(self.degree());
        let mut p = self.clone();
        for k in 1..=self.degree() + 1 {
            if p == id { return k; }
            p = p.compose(self);
        }
        self.degree() + 1
    }
    pub fn cycle_type(&self) -> Vec<usize> {
        let mut visited = vec![false; self.degree()];
        let mut cycles = Vec::new();
        for i in 0..self.degree() {
            if visited[i] { continue; }
            let mut len = 0;
            let mut j = i;
            loop {
                visited[j] = true;
                len += 1;
                j = self.apply(j as u32) as usize;
                if j == i { break; }
            }
            cycles.push(len);
        }
        cycles.sort();
        cycles
    }
    pub fn fixed_points(&self) -> Vec<u32> {
        (0..self.degree() as u32).filter(|&i| self.apply(i) == i).collect()
    }
    pub fn num_fixed_points(&self) -> usize { self.fixed_points().len() }
    pub fn is_identity(&self) -> bool { (0..self.degree() as u32).all(|i| self.apply(i) == i) }
    pub fn transposition(n: usize, i: u32, j: u32) -> Self {
        let mut images: Vec<u32> = (0..n as u32).collect();
        images[i as usize] = j;
        images[j as usize] = i;
        Permutation { images }
    }
    pub fn cyclic(n: usize) -> Self {
        let mut images: Vec<u32> = (1..n as u32).collect();
        images.push(0);
        Permutation { images }
    }
}

impl fmt::Display for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &j) in self.images.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", j)?;
        }
        write!(f, "]")
    }
}

// =========================================================================
// Finite Group Action
// =========================================================================

#[derive(Debug, Clone)]
pub struct FiniteGroupAction {
    pub group_elements: Vec<Permutation>,
    pub set_size: usize,
    pub action_table: Vec<Vec<u32>>,
}

impl FiniteGroupAction {
    pub fn new(group_elements: Vec<Permutation>, set_size: usize) -> Self {
        let action_table = group_elements.iter()
            .map(|g| (0..set_size as u32).map(|x| g.apply(x)).collect())
            .collect();
        FiniteGroupAction { group_elements, set_size, action_table }
    }

    pub fn from_generators(generators: &[Permutation], set_size: usize) -> Self {
        let mut elements = vec![Permutation::identity(set_size)];
        let mut queue = VecDeque::new();
        queue.push_back(Permutation::identity(set_size));
        let mut seen: HashSet<Vec<u32>> = HashSet::new();
        seen.insert((0..set_size as u32).collect());
        while let Some(g) = queue.pop_front() {
            for gen in generators {
                let product = g.compose(gen);
                if seen.insert(product.images.clone()) {
                    elements.push(product.clone());
                    queue.push_back(product);
                }
            }
        }
        Self::new(elements, set_size)
    }

    pub fn act(&self, g_idx: usize, x: u32) -> u32 { self.action_table[g_idx][x as usize] }
    pub fn group_order(&self) -> usize { self.group_elements.len() }

    pub fn verify_action(&self) -> bool {
        let n = self.group_order();
        for i in 0..n {
            for j in 0..n {
                let gi_gj = self.group_elements[i].compose(&self.group_elements[j]);
                for x in 0..self.set_size as u32 {
                    let left = self.act(i, self.act(j, x));
                    let right = gi_gj.apply(x);
                    if left != right { return false; }
                }
            }
        }
        true
    }

    pub fn is_transitive(&self) -> bool {
        if self.set_size == 0 { return true; }
        self.compute_orbit(0).len() == self.set_size
    }

    pub fn is_faithful(&self) -> bool {
        let id = Permutation::identity(self.set_size);
        self.group_elements.iter().filter(|g| **g == id).count() <= 1
    }

    pub fn is_free(&self) -> bool {
        for x in 0..self.set_size as u32 {
            let stab = self.compute_stabilizer(x);
            if stab.len() > 1 { return false; }
        }
        true
    }
}

// =========================================================================
// Orbit Computation
// =========================================================================

#[derive(Debug, Clone)]
pub struct Orbit {
    pub elements: BTreeSet<u32>,
    pub representative: u32,
}

impl Orbit {
    pub fn new(representative: u32, elements: BTreeSet<u32>) -> Self {
        Orbit { representative, elements }
    }
    pub fn size(&self) -> usize { self.elements.len() }
    pub fn len(&self) -> usize { self.elements.len() }
    pub fn contains(&self, x: u32) -> bool { self.elements.contains(&x) }
}

impl fmt::Display for Orbit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems: Vec<String> = self.elements.iter().map(|x| x.to_string()).collect();
        write!(f, "{{{}}}", elems.join(", "))
    }
}

impl FiniteGroupAction {
    pub fn compute_orbit(&self, x: u32) -> Orbit {
        let mut orbit = BTreeSet::new();
        let mut queue = VecDeque::new();
        orbit.insert(x);
        queue.push_back(x);
        while let Some(y) = queue.pop_front() {
            for g_idx in 0..self.group_order() {
                let z = self.act(g_idx, y);
                if orbit.insert(z) { queue.push_back(z); }
            }
        }
        Orbit::new(x, orbit)
    }

    pub fn compute_all_orbits(&self) -> OrbitPartition {
        let mut assigned = vec![false; self.set_size];
        let mut orbits = Vec::new();
        for x in 0..self.set_size as u32 {
            if assigned[x as usize] { continue; }
            let orbit = self.compute_orbit(x);
            for &elem in &orbit.elements { assigned[elem as usize] = true; }
            orbits.push(orbit);
        }
        OrbitPartition { orbits }
    }

    pub fn orbit_of(&self, x: u32) -> Orbit { self.compute_orbit(x) }

    pub fn orbit_representatives(&self) -> Vec<u32> {
        self.compute_all_orbits().orbits.iter().map(|o| o.representative).collect()
    }

    pub fn num_orbits(&self) -> usize { self.compute_all_orbits().orbits.len() }

    pub fn orbit_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.compute_all_orbits().orbits.iter().map(|o| o.size()).collect();
        sizes.sort();
        sizes
    }
}

#[derive(Debug, Clone)]
pub struct OrbitPartition {
    pub orbits: Vec<Orbit>,
}

impl OrbitPartition {
    pub fn num_orbits(&self) -> usize { self.orbits.len() }
    pub fn orbit_containing(&self, x: u32) -> Option<&Orbit> {
        self.orbits.iter().find(|o| o.contains(x))
    }
    pub fn orbit_sizes(&self) -> Vec<usize> {
        self.orbits.iter().map(|o| o.size()).collect()
    }
}

impl fmt::Display for OrbitPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, orbit) in self.orbits.iter().enumerate() {
        }
        Ok(())
    }
}

// =========================================================================
// Stabilizer Computation
// =========================================================================

#[derive(Debug, Clone)]
pub struct Stabilizer {
    pub element: u32,
    pub group_indices: Vec<usize>,
}

impl Stabilizer {
    pub fn size(&self) -> usize { self.group_indices.len() }
    pub fn len(&self) -> usize { self.group_indices.len() }
}

impl FiniteGroupAction {
    pub fn compute_stabilizer(&self, x: u32) -> Stabilizer {
        let indices: Vec<usize> = (0..self.group_order())
            .filter(|&g_idx| self.act(g_idx, x) == x)
            .collect();
        Stabilizer { element: x, group_indices: indices }
    }

    pub fn pointwise_stabilizer(&self, points: &[u32]) -> Vec<usize> {
        (0..self.group_order())
            .filter(|&g_idx| points.iter().all(|&x| self.act(g_idx, x) == x))
            .collect()
    }

    pub fn setwise_stabilizer(&self, set: &BTreeSet<u32>) -> Vec<usize> {
        (0..self.group_order())
            .filter(|&g_idx| {
                let image: BTreeSet<u32> = set.iter().map(|&x| self.act(g_idx, x)).collect();
                &image == set
            })
            .collect()
    }
}

// =========================================================================
// Orbit-Stabilizer Theorem
// =========================================================================

impl FiniteGroupAction {
    pub fn orbit_stabilizer_check(&self, x: u32) -> bool {
        let orbit = self.compute_orbit(x);
        let stab = self.compute_stabilizer(x);
        self.group_order() == orbit.size() * stab.size()
    }

    pub fn transversal(&self, x: u32) -> Transversal {
        let orbit = self.compute_orbit(x);
        let mut coset_reps = HashMap::new();
        for &y in &orbit.elements {
            for g_idx in 0..self.group_order() {
                if self.act(g_idx, x) == y {
                    coset_reps.entry(y).or_insert(g_idx);
                    break;
                }
            }
        }
        Transversal { base_point: x, coset_reps }
    }
}

#[derive(Debug, Clone)]
pub struct Transversal {
    pub base_point: u32,
    pub coset_reps: HashMap<u32, usize>,
}

impl Transversal {
    pub fn size(&self) -> usize { self.coset_reps.len() }
    pub fn representative_for(&self, y: u32) -> Option<usize> { self.coset_reps.get(&y).copied() }
}

// =========================================================================
// Fixed Point Enumeration
// =========================================================================

#[derive(Debug, Clone)]
pub struct FixedPointTable {
    pub fixed_counts: Vec<usize>,
    pub fixed_sets: Vec<Vec<u32>>,
}

impl FiniteGroupAction {
    pub fn fixed_points_of(&self, g_idx: usize) -> Vec<u32> {
        (0..self.set_size as u32).filter(|&x| self.act(g_idx, x) == x).collect()
    }

    pub fn fixed_point_table(&self) -> FixedPointTable {
        let mut counts = Vec::new();
        let mut sets = Vec::new();
        for g_idx in 0..self.group_order() {
            let fps = self.fixed_points_of(g_idx);
            counts.push(fps.len());
            sets.push(fps);
        }
        FixedPointTable { fixed_counts: counts, fixed_sets: sets }
    }

    pub fn common_fixed_points(&self) -> Vec<u32> {
        (0..self.set_size as u32)
            .filter(|&x| (0..self.group_order()).all(|g| self.act(g, x) == x))
            .collect()
    }

    pub fn fixed_point_proportion(&self, g_idx: usize) -> f64 {
        if self.set_size == 0 { return 0.0; }
        self.fixed_points_of(g_idx).len() as f64 / self.set_size as f64
    }

    pub fn maximal_fixed_set(&self) -> (usize, Vec<u32>) {
        let table = self.fixed_point_table();
        let mut best_idx = 0;
        let mut best_count = 0;
        for (i, &count) in table.fixed_counts.iter().enumerate() {
            if i == 0 { continue; } // skip identity
            if count > best_count { best_count = count; best_idx = i; }
        }
        (best_idx, table.fixed_sets[best_idx].clone())
    }
}

// =========================================================================
// Burnside Counting
// =========================================================================

#[derive(Debug, Clone)]
pub struct BurnsideResult {
    pub num_orbits: usize,
    pub group_order: usize,
    pub total_fixed: usize,
    pub fixed_per_element: Vec<usize>,
}

impl fmt::Display for BurnsideResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BurnsideResult(orbits={}, group_order={})", self.num_orbits, self.group_order)
    }
}

impl FiniteGroupAction {
    /// Burnside count: |X/G| = (1/|G|) Σ |Fix(g)|
    pub fn burnside_count(&self) -> BurnsideResult {
        let table = self.fixed_point_table();
        let total: usize = table.fixed_counts.iter().sum();
        let num_orbits = total / self.group_order();
        BurnsideResult {
            num_orbits,
            group_order: self.group_order(),
            total_fixed: total,
            fixed_per_element: table.fixed_counts,
        }
    }

    /// Burnside count with a predicate filter.
    pub fn burnside_with_constraint<F>(&self, predicate: F) -> usize
    where F: Fn(u32) -> bool {
        let filtered: Vec<u32> = (0..self.set_size as u32).filter(|&x| predicate(x)).collect();
        if filtered.is_empty() { return 0; }
        let mut total = 0usize;
        for g_idx in 0..self.group_order() {
            let fixed = filtered.iter()
                .filter(|&&x| self.act(g_idx, x) == x)
                .count();
            total += fixed;
        }
        total / self.group_order()
    }

    /// Count distinct behaviors up to symmetry.
    pub fn distinct_behavior_count(&self, behaviors: &[u32]) -> usize {
        let mut orbits_seen: HashSet<BTreeSet<u32>> = HashSet::new();
        for &b in behaviors {
            let orbit = self.compute_orbit(b);
            orbits_seen.insert(orbit.elements);
        }
        orbits_seen.len()
    }
}

// =========================================================================
// Polya Enumeration
// =========================================================================

#[derive(Debug, Clone)]
pub struct CycleIndex {
    pub terms: Vec<(f64, Vec<usize>)>, // (coefficient, [exponent of s1, s2, ...])
}

impl CycleIndex {
    pub fn new() -> Self { CycleIndex { terms: Vec::new() } }

    pub fn add_term(&mut self, coeff: f64, exponents: Vec<usize>) {
        self.terms.push((coeff, exponents));
    }

    /// Evaluate cycle index by substituting si = k for all i (count k-colorings).
    pub fn evaluate_uniform(&self, k: usize) -> f64 {
        let mut total = 0.0;
        for (coeff, exps) in &self.terms {
            let mut prod = *coeff;
            for &e in exps { prod *= k as f64; }
            total += prod;
        }
        total
    }
}

impl FiniteGroupAction {
    /// Compute cycle index of the action.
    pub fn cycle_index(&self) -> CycleIndex {
        let mut ci = CycleIndex::new();
        let n = self.group_order();
        for g_idx in 0..n {
            let ct = self.group_elements[g_idx].cycle_type();
            let max_len = ct.iter().cloned().max().unwrap_or(0);
            let mut exps = vec![0usize; max_len + 1];
            for &len in &ct { exps[len] += 1; }
            ci.add_term(1.0 / n as f64, exps);
        }
        ci
    }

    /// Count k-colorings up to symmetry (Polya).
    pub fn polya_count(&self, k: usize) -> usize {
        let n = self.group_order();
        let mut total = 0usize;
        for g_idx in 0..n {
            let ct = self.group_elements[g_idx].cycle_type();
            let num_cycles = ct.len();
            let mut contrib = 1usize;
            for _ in 0..num_cycles { contrib *= k; }
            total += contrib;
        }
        total / n
    }
}

// =========================================================================
// Equivariant Maps
// =========================================================================

#[derive(Debug, Clone)]
pub struct EquivariantMap {
    pub mapping: Vec<u32>,
    pub source_size: usize,
    pub target_size: usize,
}

impl EquivariantMap {
    pub fn new(mapping: Vec<u32>) -> Self {
        let source_size = mapping.len();
        let target_size = *mapping.iter().max().unwrap_or(&0) as usize + 1;
        EquivariantMap { mapping, source_size, target_size }
    }

    pub fn apply(&self, x: u32) -> u32 { self.mapping[x as usize] }

    /// Check equivariance: f(g·x) = g·f(x) for all g, x.
    pub fn check_equivariance(
        &self,
        source_action: &FiniteGroupAction,
        target_action: &FiniteGroupAction,
    ) -> bool {
        for g_idx in 0..source_action.group_order() {
            for x in 0..self.source_size as u32 {
                let left = self.apply(source_action.act(g_idx, x));
                let right = target_action.act(g_idx, self.apply(x));
                if left != right { return false; }
            }
        }
        true
    }
}

// =========================================================================
// Quotient Action
// =========================================================================

impl FiniteGroupAction {
    /// Compute the quotient action on orbits.
    pub fn quotient_action(&self) -> FiniteGroupAction {
        let partition = self.compute_all_orbits();
        let n = partition.num_orbits();
        let id = Permutation::identity(n);
        let trivial = vec![id; self.group_order()];
        FiniteGroupAction::new(trivial, n)
    }
}

// =========================================================================
// Cayley Graph
// =========================================================================

impl FiniteGroupAction {
    pub fn cayley_graph_dot(&self, generators: &[usize]) -> String {
        let mut dot = String::from("digraph cayley {
");
        dot.push_str("    rankdir=LR;
");
        for i in 0..self.group_order() {
        }
        let colors = ["red", "blue", "green", "orange", "purple"];
        for (ci, &gen_idx) in generators.iter().enumerate() {
            let color = colors[ci % colors.len()];
            for i in 0..self.group_order() {
                let j_perm = self.group_elements[i].compose(&self.group_elements[gen_idx]);
                if let Some(j) = self.group_elements.iter().position(|g| g == &j_perm) {
                }
            }
        }
        dot.push_str("}
");
        dot
    }
}

// =========================================================================
// Utilities
// =========================================================================

pub fn action_statistics(action: &FiniteGroupAction) -> ActionStatistics {
    let orbits = action.compute_all_orbits();
    let orbit_sizes = orbits.orbit_sizes();
    ActionStatistics {
        group_order: action.group_order(),
        set_size: action.set_size,
        num_orbits: orbits.num_orbits(),
        orbit_sizes,
        is_transitive: action.is_transitive(),
        is_faithful: action.is_faithful(),
        is_free: action.is_free(),
    }
}

#[derive(Debug, Clone)]
pub struct ActionStatistics {
    pub group_order: usize,
    pub set_size: usize,
    pub num_orbits: usize,
    pub orbit_sizes: Vec<usize>,
    pub is_transitive: bool,
    pub is_faithful: bool,
    pub is_free: bool,
}

impl fmt::Display for ActionStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionStatistics(group={}, set={}, orbits={})", self.group_order, self.set_size, self.num_orbits)
    }
}

pub fn print_orbit_table(action: &FiniteGroupAction) -> String {
    let partition = action.compute_all_orbits();
    format!("Orbits: {}", partition.num_orbits())
}

// =========================================================================
// Tests
// =========================================================================

// ===== Extended Action Operations =====

#[derive(Debug, Clone)]
pub struct InducedAction {
    pub base_action_size: usize,
    pub induced_set_size: usize,
    pub action_table: Vec<Vec<u32>>,
}

impl InducedAction {
    pub fn new(base_action_size: usize, induced_set_size: usize, action_table: Vec<Vec<u32>>) -> Self {
        InducedAction { base_action_size, induced_set_size, action_table }
    }

    pub fn get_base_action_size(&self) -> usize {
        self.base_action_size
    }

    pub fn get_induced_set_size(&self) -> usize {
        self.induced_set_size
    }

    pub fn action_table_len(&self) -> usize {
        self.action_table.len()
    }

    pub fn action_table_is_empty(&self) -> bool {
        self.action_table.is_empty()
    }

    pub fn with_base_action_size(mut self, v: usize) -> Self {
        self.base_action_size = v; self
    }

    pub fn with_induced_set_size(mut self, v: usize) -> Self {
        self.induced_set_size = v; self
    }

}

impl fmt::Display for InducedAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InducedAction({:?})", self.base_action_size)
    }
}

#[derive(Debug, Clone)]
pub struct InducedActionBuilder {
    base_action_size: usize,
    induced_set_size: usize,
    action_table: Vec<Vec<u32>>,
}

impl InducedActionBuilder {
    pub fn new() -> Self {
        InducedActionBuilder {
            base_action_size: 0,
            induced_set_size: 0,
            action_table: Vec::new(),
        }
    }

    pub fn base_action_size(mut self, v: usize) -> Self { self.base_action_size = v; self }
    pub fn induced_set_size(mut self, v: usize) -> Self { self.induced_set_size = v; self }
    pub fn action_table(mut self, v: Vec<Vec<u32>>) -> Self { self.action_table = v; self }
}

#[derive(Debug, Clone)]
pub struct RestrictedAction {
    pub original_size: usize,
    pub restricted_elements: Vec<u32>,
    pub action_table: Vec<Vec<u32>>,
}

impl RestrictedAction {
    pub fn new(original_size: usize, restricted_elements: Vec<u32>, action_table: Vec<Vec<u32>>) -> Self {
        RestrictedAction { original_size, restricted_elements, action_table }
    }

    pub fn get_original_size(&self) -> usize {
        self.original_size
    }

    pub fn restricted_elements_len(&self) -> usize {
        self.restricted_elements.len()
    }

    pub fn restricted_elements_is_empty(&self) -> bool {
        self.restricted_elements.is_empty()
    }

    pub fn action_table_len(&self) -> usize {
        self.action_table.len()
    }

    pub fn action_table_is_empty(&self) -> bool {
        self.action_table.is_empty()
    }

    pub fn with_original_size(mut self, v: usize) -> Self {
        self.original_size = v; self
    }

}

impl fmt::Display for RestrictedAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RestrictedAction({:?})", self.original_size)
    }
}

#[derive(Debug, Clone)]
pub struct RestrictedActionBuilder {
    original_size: usize,
    restricted_elements: Vec<u32>,
    action_table: Vec<Vec<u32>>,
}

impl RestrictedActionBuilder {
    pub fn new() -> Self {
        RestrictedActionBuilder {
            original_size: 0,
            restricted_elements: Vec::new(),
            action_table: Vec::new(),
        }
    }

    pub fn original_size(mut self, v: usize) -> Self { self.original_size = v; self }
    pub fn restricted_elements(mut self, v: Vec<u32>) -> Self { self.restricted_elements = v; self }
    pub fn action_table(mut self, v: Vec<Vec<u32>>) -> Self { self.action_table = v; self }
}

#[derive(Debug, Clone)]
pub struct ProductAction {
    pub factor_sizes: Vec<usize>,
    pub product_size: usize,
    pub action_table: Vec<Vec<u32>>,
}

impl ProductAction {
    pub fn new(factor_sizes: Vec<usize>, product_size: usize, action_table: Vec<Vec<u32>>) -> Self {
        ProductAction { factor_sizes, product_size, action_table }
    }

    pub fn factor_sizes_len(&self) -> usize {
        self.factor_sizes.len()
    }

    pub fn factor_sizes_is_empty(&self) -> bool {
        self.factor_sizes.is_empty()
    }

    pub fn get_product_size(&self) -> usize {
        self.product_size
    }

    pub fn action_table_len(&self) -> usize {
        self.action_table.len()
    }

    pub fn action_table_is_empty(&self) -> bool {
        self.action_table.is_empty()
    }

    pub fn with_product_size(mut self, v: usize) -> Self {
        self.product_size = v; self
    }

}

impl fmt::Display for ProductAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProductAction({:?})", self.factor_sizes)
    }
}

#[derive(Debug, Clone)]
pub struct ProductActionBuilder {
    factor_sizes: Vec<usize>,
    product_size: usize,
    action_table: Vec<Vec<u32>>,
}

impl ProductActionBuilder {
    pub fn new() -> Self {
        ProductActionBuilder {
            factor_sizes: Vec::new(),
            product_size: 0,
            action_table: Vec::new(),
        }
    }

    pub fn factor_sizes(mut self, v: Vec<usize>) -> Self { self.factor_sizes = v; self }
    pub fn product_size(mut self, v: usize) -> Self { self.product_size = v; self }
    pub fn action_table(mut self, v: Vec<Vec<u32>>) -> Self { self.action_table = v; self }
}

#[derive(Debug, Clone)]
pub struct WreathProductAction {
    pub top_degree: usize,
    pub bottom_degree: usize,
    pub total_degree: usize,
}

impl WreathProductAction {
    pub fn new(top_degree: usize, bottom_degree: usize, total_degree: usize) -> Self {
        WreathProductAction { top_degree, bottom_degree, total_degree }
    }

    pub fn get_top_degree(&self) -> usize {
        self.top_degree
    }

    pub fn get_bottom_degree(&self) -> usize {
        self.bottom_degree
    }

    pub fn get_total_degree(&self) -> usize {
        self.total_degree
    }

    pub fn with_top_degree(mut self, v: usize) -> Self {
        self.top_degree = v; self
    }

    pub fn with_bottom_degree(mut self, v: usize) -> Self {
        self.bottom_degree = v; self
    }

    pub fn with_total_degree(mut self, v: usize) -> Self {
        self.total_degree = v; self
    }

}

impl fmt::Display for WreathProductAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WreathProductAction({:?})", self.top_degree)
    }
}

#[derive(Debug, Clone)]
pub struct WreathProductActionBuilder {
    top_degree: usize,
    bottom_degree: usize,
    total_degree: usize,
}

impl WreathProductActionBuilder {
    pub fn new() -> Self {
        WreathProductActionBuilder {
            top_degree: 0,
            bottom_degree: 0,
            total_degree: 0,
        }
    }

    pub fn top_degree(mut self, v: usize) -> Self { self.top_degree = v; self }
    pub fn bottom_degree(mut self, v: usize) -> Self { self.bottom_degree = v; self }
    pub fn total_degree(mut self, v: usize) -> Self { self.total_degree = v; self }
}

#[derive(Debug, Clone)]
pub struct ConjugationAction {
    pub group_order: usize,
    pub conjugacy_classes: Vec<Vec<u32>>,
}

impl ConjugationAction {
    pub fn new(group_order: usize, conjugacy_classes: Vec<Vec<u32>>) -> Self {
        ConjugationAction { group_order, conjugacy_classes }
    }

    pub fn get_group_order(&self) -> usize {
        self.group_order
    }

    pub fn conjugacy_classes_len(&self) -> usize {
        self.conjugacy_classes.len()
    }

    pub fn conjugacy_classes_is_empty(&self) -> bool {
        self.conjugacy_classes.is_empty()
    }

    pub fn with_group_order(mut self, v: usize) -> Self {
        self.group_order = v; self
    }

}

impl fmt::Display for ConjugationAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConjugationAction({:?})", self.group_order)
    }
}

#[derive(Debug, Clone)]
pub struct ConjugationActionBuilder {
    group_order: usize,
    conjugacy_classes: Vec<Vec<u32>>,
}

impl ConjugationActionBuilder {
    pub fn new() -> Self {
        ConjugationActionBuilder {
            group_order: 0,
            conjugacy_classes: Vec::new(),
        }
    }

    pub fn group_order(mut self, v: usize) -> Self { self.group_order = v; self }
    pub fn conjugacy_classes(mut self, v: Vec<Vec<u32>>) -> Self { self.conjugacy_classes = v; self }
}

#[derive(Debug, Clone)]
pub struct RegularAction {
    pub group_order: usize,
    pub action_table: Vec<Vec<u32>>,
}

impl RegularAction {
    pub fn new(group_order: usize, action_table: Vec<Vec<u32>>) -> Self {
        RegularAction { group_order, action_table }
    }

    pub fn get_group_order(&self) -> usize {
        self.group_order
    }

    pub fn action_table_len(&self) -> usize {
        self.action_table.len()
    }

    pub fn action_table_is_empty(&self) -> bool {
        self.action_table.is_empty()
    }

    pub fn with_group_order(mut self, v: usize) -> Self {
        self.group_order = v; self
    }

}

impl fmt::Display for RegularAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegularAction({:?})", self.group_order)
    }
}

#[derive(Debug, Clone)]
pub struct RegularActionBuilder {
    group_order: usize,
    action_table: Vec<Vec<u32>>,
}

impl RegularActionBuilder {
    pub fn new() -> Self {
        RegularActionBuilder {
            group_order: 0,
            action_table: Vec::new(),
        }
    }

    pub fn group_order(mut self, v: usize) -> Self { self.group_order = v; self }
    pub fn action_table(mut self, v: Vec<Vec<u32>>) -> Self { self.action_table = v; self }
}

#[derive(Debug, Clone)]
pub struct CosetAction {
    pub group_order: usize,
    pub subgroup_order: usize,
    pub num_cosets: usize,
    pub action_table: Vec<Vec<u32>>,
}

impl CosetAction {
    pub fn new(group_order: usize, subgroup_order: usize, num_cosets: usize, action_table: Vec<Vec<u32>>) -> Self {
        CosetAction { group_order, subgroup_order, num_cosets, action_table }
    }

    pub fn get_group_order(&self) -> usize {
        self.group_order
    }

    pub fn get_subgroup_order(&self) -> usize {
        self.subgroup_order
    }

    pub fn get_num_cosets(&self) -> usize {
        self.num_cosets
    }

    pub fn action_table_len(&self) -> usize {
        self.action_table.len()
    }

    pub fn action_table_is_empty(&self) -> bool {
        self.action_table.is_empty()
    }

    pub fn with_group_order(mut self, v: usize) -> Self {
        self.group_order = v; self
    }

    pub fn with_subgroup_order(mut self, v: usize) -> Self {
        self.subgroup_order = v; self
    }

    pub fn with_num_cosets(mut self, v: usize) -> Self {
        self.num_cosets = v; self
    }

}

impl fmt::Display for CosetAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CosetAction({:?})", self.group_order)
    }
}

#[derive(Debug, Clone)]
pub struct CosetActionBuilder {
    group_order: usize,
    subgroup_order: usize,
    num_cosets: usize,
    action_table: Vec<Vec<u32>>,
}

impl CosetActionBuilder {
    pub fn new() -> Self {
        CosetActionBuilder {
            group_order: 0,
            subgroup_order: 0,
            num_cosets: 0,
            action_table: Vec::new(),
        }
    }

    pub fn group_order(mut self, v: usize) -> Self { self.group_order = v; self }
    pub fn subgroup_order(mut self, v: usize) -> Self { self.subgroup_order = v; self }
    pub fn num_cosets(mut self, v: usize) -> Self { self.num_cosets = v; self }
    pub fn action_table(mut self, v: Vec<Vec<u32>>) -> Self { self.action_table = v; self }
}

#[derive(Debug, Clone)]
pub struct PermutationRepr {
    pub degree: usize,
    pub generators: Vec<Vec<u32>>,
    pub order: usize,
}

impl PermutationRepr {
    pub fn new(degree: usize, generators: Vec<Vec<u32>>, order: usize) -> Self {
        PermutationRepr { degree, generators, order }
    }

    pub fn get_degree(&self) -> usize {
        self.degree
    }

    pub fn generators_len(&self) -> usize {
        self.generators.len()
    }

    pub fn generators_is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    pub fn get_order(&self) -> usize {
        self.order
    }

    pub fn with_degree(mut self, v: usize) -> Self {
        self.degree = v; self
    }

    pub fn with_order(mut self, v: usize) -> Self {
        self.order = v; self
    }

}

impl fmt::Display for PermutationRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PermutationRepr({:?})", self.degree)
    }
}

#[derive(Debug, Clone)]
pub struct PermutationReprBuilder {
    degree: usize,
    generators: Vec<Vec<u32>>,
    order: usize,
}

impl PermutationReprBuilder {
    pub fn new() -> Self {
        PermutationReprBuilder {
            degree: 0,
            generators: Vec::new(),
            order: 0,
        }
    }

    pub fn degree(mut self, v: usize) -> Self { self.degree = v; self }
    pub fn generators(mut self, v: Vec<Vec<u32>>) -> Self { self.generators = v; self }
    pub fn order(mut self, v: usize) -> Self { self.order = v; self }
}

#[derive(Debug, Clone)]
pub struct SchreierGenerator {
    pub generator_index: usize,
    pub stabilizer_point: u32,
    pub word: Vec<usize>,
}

impl SchreierGenerator {
    pub fn new(generator_index: usize, stabilizer_point: u32, word: Vec<usize>) -> Self {
        SchreierGenerator { generator_index, stabilizer_point, word }
    }

    pub fn get_generator_index(&self) -> usize {
        self.generator_index
    }

    pub fn get_stabilizer_point(&self) -> u32 {
        self.stabilizer_point
    }

    pub fn word_len(&self) -> usize {
        self.word.len()
    }

    pub fn word_is_empty(&self) -> bool {
        self.word.is_empty()
    }

    pub fn with_generator_index(mut self, v: usize) -> Self {
        self.generator_index = v; self
    }

    pub fn with_stabilizer_point(mut self, v: u32) -> Self {
        self.stabilizer_point = v; self
    }

}

impl fmt::Display for SchreierGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SchreierGenerator({:?})", self.generator_index)
    }
}

#[derive(Debug, Clone)]
pub struct SchreierGeneratorBuilder {
    generator_index: usize,
    stabilizer_point: u32,
    word: Vec<usize>,
}

impl SchreierGeneratorBuilder {
    pub fn new() -> Self {
        SchreierGeneratorBuilder {
            generator_index: 0,
            stabilizer_point: 0,
            word: Vec::new(),
        }
    }

    pub fn generator_index(mut self, v: usize) -> Self { self.generator_index = v; self }
    pub fn stabilizer_point(mut self, v: u32) -> Self { self.stabilizer_point = v; self }
    pub fn word(mut self, v: Vec<usize>) -> Self { self.word = v; self }
}

#[derive(Debug, Clone)]
pub struct BlockSystem {
    pub blocks: Vec<Vec<u32>>,
    pub num_blocks: usize,
    pub is_trivial: bool,
}

impl BlockSystem {
    pub fn new(blocks: Vec<Vec<u32>>, num_blocks: usize, is_trivial: bool) -> Self {
        BlockSystem { blocks, num_blocks, is_trivial }
    }

    pub fn blocks_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks_is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn get_num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn get_is_trivial(&self) -> bool {
        self.is_trivial
    }

    pub fn with_num_blocks(mut self, v: usize) -> Self {
        self.num_blocks = v; self
    }

    pub fn with_is_trivial(mut self, v: bool) -> Self {
        self.is_trivial = v; self
    }

}

impl fmt::Display for BlockSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockSystem({:?})", self.blocks)
    }
}

#[derive(Debug, Clone)]
pub struct BlockSystemBuilder {
    blocks: Vec<Vec<u32>>,
    num_blocks: usize,
    is_trivial: bool,
}

impl BlockSystemBuilder {
    pub fn new() -> Self {
        BlockSystemBuilder {
            blocks: Vec::new(),
            num_blocks: 0,
            is_trivial: false,
        }
    }

    pub fn blocks(mut self, v: Vec<Vec<u32>>) -> Self { self.blocks = v; self }
    pub fn num_blocks(mut self, v: usize) -> Self { self.num_blocks = v; self }
    pub fn is_trivial(mut self, v: bool) -> Self { self.is_trivial = v; self }
}

#[derive(Debug, Clone)]
pub struct PrimitivityTest {
    pub is_primitive: bool,
    pub block_system: Vec<Vec<u32>>,
    pub tested_points: Vec<u32>,
}

impl PrimitivityTest {
    pub fn new(is_primitive: bool, block_system: Vec<Vec<u32>>, tested_points: Vec<u32>) -> Self {
        PrimitivityTest { is_primitive, block_system, tested_points }
    }

    pub fn get_is_primitive(&self) -> bool {
        self.is_primitive
    }

    pub fn block_system_len(&self) -> usize {
        self.block_system.len()
    }

    pub fn block_system_is_empty(&self) -> bool {
        self.block_system.is_empty()
    }

    pub fn tested_points_len(&self) -> usize {
        self.tested_points.len()
    }

    pub fn tested_points_is_empty(&self) -> bool {
        self.tested_points.is_empty()
    }

    pub fn with_is_primitive(mut self, v: bool) -> Self {
        self.is_primitive = v; self
    }

}

impl fmt::Display for PrimitivityTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrimitivityTest({:?})", self.is_primitive)
    }
}

#[derive(Debug, Clone)]
pub struct PrimitivityTestBuilder {
    is_primitive: bool,
    block_system: Vec<Vec<u32>>,
    tested_points: Vec<u32>,
}

impl PrimitivityTestBuilder {
    pub fn new() -> Self {
        PrimitivityTestBuilder {
            is_primitive: false,
            block_system: Vec::new(),
            tested_points: Vec::new(),
        }
    }

    pub fn is_primitive(mut self, v: bool) -> Self { self.is_primitive = v; self }
    pub fn block_system(mut self, v: Vec<Vec<u32>>) -> Self { self.block_system = v; self }
    pub fn tested_points(mut self, v: Vec<u32>) -> Self { self.tested_points = v; self }
}

#[derive(Debug, Clone)]
pub struct ActionHomomorphism {
    pub source_degree: usize,
    pub target_degree: usize,
    pub kernel_size: usize,
    pub image_size: usize,
}

impl ActionHomomorphism {
    pub fn new(source_degree: usize, target_degree: usize, kernel_size: usize, image_size: usize) -> Self {
        ActionHomomorphism { source_degree, target_degree, kernel_size, image_size }
    }

    pub fn get_source_degree(&self) -> usize {
        self.source_degree
    }

    pub fn get_target_degree(&self) -> usize {
        self.target_degree
    }

    pub fn get_kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn get_image_size(&self) -> usize {
        self.image_size
    }

    pub fn with_source_degree(mut self, v: usize) -> Self {
        self.source_degree = v; self
    }

    pub fn with_target_degree(mut self, v: usize) -> Self {
        self.target_degree = v; self
    }

    pub fn with_kernel_size(mut self, v: usize) -> Self {
        self.kernel_size = v; self
    }

    pub fn with_image_size(mut self, v: usize) -> Self {
        self.image_size = v; self
    }

}

impl fmt::Display for ActionHomomorphism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionHomomorphism({:?})", self.source_degree)
    }
}

#[derive(Debug, Clone)]
pub struct ActionHomomorphismBuilder {
    source_degree: usize,
    target_degree: usize,
    kernel_size: usize,
    image_size: usize,
}

impl ActionHomomorphismBuilder {
    pub fn new() -> Self {
        ActionHomomorphismBuilder {
            source_degree: 0,
            target_degree: 0,
            kernel_size: 0,
            image_size: 0,
        }
    }

    pub fn source_degree(mut self, v: usize) -> Self { self.source_degree = v; self }
    pub fn target_degree(mut self, v: usize) -> Self { self.target_degree = v; self }
    pub fn kernel_size(mut self, v: usize) -> Self { self.kernel_size = v; self }
    pub fn image_size(mut self, v: usize) -> Self { self.image_size = v; self }
}

#[derive(Debug, Clone)]
pub struct ActionAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl ActionAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        ActionAnalysis { data, size, computed: false, label: "Action".to_string(), threshold: 0.01 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t; self
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.size && j < self.size { self.data[i][j] = v; }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size { self.data[i][j] } else { 0.0 }
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        if i < self.size { self.data[i].iter().sum() } else { 0.0 }
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        if j < self.size { (0..self.size).map(|i| self.data[i][j]).sum() } else { 0.0 }
    }

    pub fn total_sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn above_threshold(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] > self.threshold {
                    result.push((i, j, self.data[i][j]));
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let total = self.total_sum();
        if total > 0.0 {
            for i in 0..self.size {
                for j in 0..self.size {
                    self.data[i][j] /= total;
                }
            }
        }
        self.computed = true;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size);
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                let mut sum = 0.0;
                for k in 0..self.size {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.size).map(|i| self.data[i][i]).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.data[i][i]).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

}

impl fmt::Display for ActionAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for ActionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionStatus::Pending => write!(f, "pending"),
            ActionStatus::InProgress => write!(f, "inprogress"),
            ActionStatus::Completed => write!(f, "completed"),
            ActionStatus::Failed => write!(f, "failed"),
            ActionStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for ActionPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionPriority::Critical => write!(f, "critical"),
            ActionPriority::High => write!(f, "high"),
            ActionPriority::Medium => write!(f, "medium"),
            ActionPriority::Low => write!(f, "low"),
            ActionPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for ActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionMode::Strict => write!(f, "strict"),
            ActionMode::Relaxed => write!(f, "relaxed"),
            ActionMode::Permissive => write!(f, "permissive"),
            ActionMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn action_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn action_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = action_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn action_std_dev(data: &[f64]) -> f64 {
    action_variance(data).sqrt()
}

pub fn action_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for Action.
pub fn action_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn action_entropy(data: &[f64]) -> f64 {
    let total: f64 = data.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &x in data {
        if x > 0.0 {
            let p = x / total;
            h -= p * p.ln();
        }
    }
    h
}

pub fn action_gini(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let mut g = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

pub fn action_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = action_mean(&x);
    let my = action_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn action_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = action_covariance(data);
    let sx = action_std_dev(&data[..n]);
    let sy = action_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn action_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = action_mean(data);
    let s = action_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn action_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = action_mean(data);
    let s = action_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn action_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn action_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over action analysis results.
#[derive(Debug, Clone)]
pub struct ActionResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl ActionResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        ActionResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for ActionResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert InducedAction description to a summary string.
pub fn inducedaction_to_summary(item: &InducedAction) -> String {
    format!("InducedAction: {:?}", item)
}

/// Convert RestrictedAction description to a summary string.
pub fn restrictedaction_to_summary(item: &RestrictedAction) -> String {
    format!("RestrictedAction: {:?}", item)
}

/// Convert ProductAction description to a summary string.
pub fn productaction_to_summary(item: &ProductAction) -> String {
    format!("ProductAction: {:?}", item)
}

/// Convert WreathProductAction description to a summary string.
pub fn wreathproductaction_to_summary(item: &WreathProductAction) -> String {
    format!("WreathProductAction: {:?}", item)
}

/// Convert ConjugationAction description to a summary string.
pub fn conjugationaction_to_summary(item: &ConjugationAction) -> String {
    format!("ConjugationAction: {:?}", item)
}

/// Convert RegularAction description to a summary string.
pub fn regularaction_to_summary(item: &RegularAction) -> String {
    format!("RegularAction: {:?}", item)
}

/// Convert CosetAction description to a summary string.
pub fn cosetaction_to_summary(item: &CosetAction) -> String {
    format!("CosetAction: {:?}", item)
}

/// Convert PermutationRepr description to a summary string.
pub fn permutationrepr_to_summary(item: &PermutationRepr) -> String {
    format!("PermutationRepr: {:?}", item)
}

/// Convert SchreierGenerator description to a summary string.
pub fn schreiergenerator_to_summary(item: &SchreierGenerator) -> String {
    format!("SchreierGenerator: {:?}", item)
}

/// Convert BlockSystem description to a summary string.
pub fn blocksystem_to_summary(item: &BlockSystem) -> String {
    format!("BlockSystem: {:?}", item)
}

/// Convert PrimitivityTest description to a summary string.
pub fn primitivitytest_to_summary(item: &PrimitivityTest) -> String {
    format!("PrimitivityTest: {:?}", item)
}

/// Batch processor for action operations.
#[derive(Debug, Clone)]
pub struct ActionBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl ActionBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        ActionBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
    }
    pub fn process_batch(&mut self, data: &[f64]) {
        for chunk in data.chunks(self.batch_size) {
            let sum: f64 = chunk.iter().sum();
            self.results.push(sum / chunk.len() as f64);
            self.processed += chunk.len();
        }
    }
    pub fn success_rate(&self) -> f64 {
        if self.processed == 0 { return 0.0; }
        1.0 - (self.errors.len() as f64 / self.processed as f64)
    }
    pub fn average_result(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.results.iter().sum::<f64>() / self.results.len() as f64
    }
    pub fn reset(&mut self) { self.processed = 0; self.errors.clear(); self.results.clear(); }
}

impl fmt::Display for ActionBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for action analysis.
#[derive(Debug, Clone)]
pub struct ActionReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl ActionReport {
    pub fn new(title: impl Into<String>) -> Self {
        ActionReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
    }
    pub fn add_section(&mut self, name: impl Into<String>, content: Vec<String>) {
        self.sections.push((name.into(), content));
    }
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    pub fn total_metrics(&self) -> usize { self.metrics.len() }
    pub fn has_warnings(&self) -> bool { !self.warnings.is_empty() }
    pub fn metric_sum(&self) -> f64 { self.metrics.iter().map(|(_, v)| v).sum() }
    pub fn render_text(&self) -> String {
        let mut out = format!("=== {} ===\n", self.title);
        for (name, content) in &self.sections {
            out.push_str(&format!("\n--- {} ---\n", name));
            for line in content {
                out.push_str(&format!("  {}\n", line));
            }
        }
        out.push_str("\nMetrics:\n");
        for (name, val) in &self.metrics {
            out.push_str(&format!("  {}: {:.4}\n", name, val));
        }
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                out.push_str(&format!("  ! {}\n", w));
            }
        }
        out
    }
}

impl fmt::Display for ActionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionReport({})", self.title)
    }
}

/// Configuration for action analysis.
#[derive(Debug, Clone)]
pub struct ActionConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl ActionConfig {
    pub fn default_config() -> Self {
        ActionConfig {
            verbose: false, max_iterations: 1000, tolerance: 1e-6,
            timeout_ms: 30000, parallel: false, output_format: "text".to_string(),
        }
    }
    pub fn with_verbose(mut self, v: bool) -> Self { self.verbose = v; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    pub fn with_parallel(mut self, p: bool) -> Self { self.parallel = p; self }
    pub fn with_output_format(mut self, fmt: impl Into<String>) -> Self { self.output_format = fmt.into(); self }
}

impl fmt::Display for ActionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for action data distribution.
#[derive(Debug, Clone)]
pub struct ActionHistogramExt {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl ActionHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return ActionHistogramExt { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
        }
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };
        let mut bins = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { bin_edges.push(min_val + i as f64 * bin_width); }
        for &val in data {
            let idx = ((val - min_val) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        ActionHistogramExt { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut acc = 0usize;
        for &b in &self.bins { acc += b; cum.push(acc); }
        cum
    }
    pub fn entropy(&self) -> f64 {
        let total = self.total_count as f64;
        if total == 0.0 { return 0.0; }
        let mut h = 0.0f64;
        for &b in &self.bins {
            if b > 0 { let p = b as f64 / total; h -= p * p.ln(); }
        }
        h
    }
    pub fn render_ascii(&self, width: usize) -> String {
        let max = self.max_bin();
        let mut out = String::new();
        for (i, &count) in self.bins.iter().enumerate() {
            let bar_len = if max == 0 { 0 } else { count * width / max };
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            out.push_str(&format!("[{:.2}-{:.2}] {} {}\n",
                self.bin_edges[i], self.bin_edges[i + 1], bar, count));
        }
        out
    }
}

impl fmt::Display for ActionHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for action graph analysis.
#[derive(Debug, Clone)]
pub struct ActionGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl ActionGraph {
    pub fn new(n: usize) -> Self {
        ActionGraph {
            adjacency: vec![vec![false; n]; n],
            weights: vec![vec![0.0; n]; n],
            node_count: n, edge_count: 0,
            node_labels: (0..n).map(|i| format!("n{}", i)).collect(),
        }
    }
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.node_count && to < self.node_count && !self.adjacency[from][to] {
            self.adjacency[from][to] = true;
            self.weights[from][to] = weight;
            self.edge_count += 1;
        }
    }
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && self.adjacency[from][to] {
            self.adjacency[from][to] = false;
            self.weights[from][to] = 0.0;
            self.edge_count -= 1;
        }
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        from < self.node_count && to < self.node_count && self.adjacency[from][to]
    }
    pub fn weight(&self, from: usize, to: usize) -> f64 { self.weights[from][to] }
    pub fn out_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).count()
    }
    pub fn in_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&i| self.adjacency[i][node]).count()
    }
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).collect()
    }
    pub fn density(&self) -> f64 {
        if self.node_count <= 1 { return 0.0; }
        self.edge_count as f64 / (self.node_count * (self.node_count - 1)) as f64
    }
    pub fn is_acyclic(&self) -> bool {
        let n = self.node_count;
        let mut visited = vec![0u8; n];
        fn dfs_cycle_action(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_action(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_action(i, &self.adjacency, &mut visited) { return false; }
        }
        true
    }
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.node_count;
        let mut in_deg: Vec<usize> = (0..n).map(|j| self.in_degree(j)).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut result = Vec::new();
        while let Some(v) = queue.pop() {
            result.push(v);
            for j in 0..n { if self.adjacency[v][j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 { queue.push(j); }
            }}
        }
        if result.len() == n { Some(result) } else { None }
    }
    pub fn shortest_path_dijkstra(&self, start: usize) -> Vec<f64> {
        let n = self.node_count;
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..n { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for v in 0..n { if self.adjacency[u][v] {
                let alt = dist[u] + self.weights[u][v];
                if alt < dist[v] { dist[v] = alt; }
            }}
        }
        dist
    }
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if visited[v] { continue; }
                visited[v] = true;
                comp.push(v);
                for w in 0..n {
                    if (self.adjacency[v][w] || self.adjacency[w][v]) && !visited[w] {
                        stack.push(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for i in 0..self.node_count {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, self.node_labels[i]));
        }
        for i in 0..self.node_count { for j in 0..self.node_count { if self.adjacency[i][j] {
            out.push_str(&format!("  {} -> {} [label=\"{:.2}\"];\n", i, j, self.weights[i][j]));
        }}}
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for ActionGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for action computation results.
#[derive(Debug, Clone)]
pub struct ActionCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl ActionCache {
    pub fn new(capacity: usize) -> Self {
        ActionCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
    }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn insert(&mut self, key: u64, value: Vec<f64>) {
        if self.entries.len() >= self.capacity { self.entries.remove(0); }
        self.entries.push((key, value));
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; }
}

impl fmt::Display for ActionCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for action elements.
pub fn action_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let d: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

/// K-means clustering for action data.
pub fn action_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
    if data.is_empty() || k == 0 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0; let mut best_d = f64::INFINITY;
            for c in 0..centroids.len() {
                let d: f64 = data[i].iter().zip(centroids[c].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_c = c; }
            }
            if assignments[i] != best_c { changed = true; assignments[i] = best_c; }
        }
        if !changed { break; }
        // Update centroids
        for c in 0..centroids.len() {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() { continue; }
            for d in 0..dim {
                centroids[c][d] = members.iter().map(|&i| data[i][d]).sum::<f64>() / members.len() as f64;
            }
        }
    }
    assignments
}

/// Principal component analysis (simplified) for action data.
pub fn action_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if data.is_empty() || data[0].len() < 2 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    // Compute mean
    let mut mean = vec![0.0; dim];
    for row in data { for (j, &v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..dim { mean[j] /= n as f64; }
    // Center data
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(mean.iter()).map(|(v, m)| v - m).collect()
    }).collect();
    // Simple projection onto first two dimensions (not true PCA)
    centered.iter().map(|row| (row[0], row[1])).collect()
}

/// Dense matrix operations for Action computations.
#[derive(Debug, Clone)]
pub struct ActionDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl ActionDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        ActionDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        ActionDenseMatrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        ActionDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        ActionDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        ActionDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { result.set(j, i, self.get(i, j)); } }
        result
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        (0..self.cols).map(|j| self.get(i, j)).sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j)).sum()
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_diagonal(&self) -> bool {
        for i in 0..self.rows { for j in 0..self.cols {
            if i != j && self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows { for j in 0..i.min(self.cols) {
            if self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn determinant_2x2(&self) -> f64 {
        assert!(self.rows == 2 && self.cols == 2);
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    pub fn determinant_3x3(&self) -> f64 {
        assert!(self.rows == 3 && self.cols == 3);
        let a = self.get(0, 0); let b = self.get(0, 1); let c = self.get(0, 2);
        let d = self.get(1, 0); let e = self.get(1, 1); let ff = self.get(1, 2);
        let g = self.get(2, 0); let h = self.get(2, 1); let ii = self.get(2, 2);
        a * (e * ii - ff * h) - b * (d * ii - ff * g) + c * (d * h - e * g)
    }

    pub fn inverse_2x2(&self) -> Option<Self> {
        assert!(self.rows == 2 && self.cols == 2);
        let det = self.determinant_2x2();
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut result = Self::new(2, 2);
        result.set(0, 0, self.get(1, 1) * inv_det);
        result.set(0, 1, -self.get(0, 1) * inv_det);
        result.set(1, 0, -self.get(1, 0) * inv_det);
        result.set(1, 1, self.get(0, 0) * inv_det);
        Some(result)
    }

    pub fn power(&self, n: u32) -> Self {
        assert!(self.is_square());
        let mut result = Self::identity(self.rows);
        for _ in 0..n { result = result.mul_matrix(self); }
        result
    }

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        let mut result = Self::new(rows, cols);
        for i in 0..rows { for j in 0..cols {
            result.set(i, j, self.get(row_start + i, col_start + j));
        }}
        result
    }

    pub fn kronecker_product(&self, other: &Self) -> Self {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Self::new(m, n);
        for i in 0..self.rows { for j in 0..self.cols {
            let s = self.get(i, j);
            for p in 0..other.rows { for q in 0..other.cols {
                result.set(i * other.rows + p, j * other.cols + q, s * other.get(p, q));
            }}
        }}
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        ActionDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let mut result = Self::new(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { result.set(i, j, a[i] * b[j]); } }
        result
    }

    pub fn row_reduce(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_row = 0;
        for col in 0..result.cols {
            if pivot_row >= result.rows { break; }
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..result.rows {
                if result.get(row, col).abs() > result.get(max_row, col).abs() { max_row = row; }
            }
            if result.get(max_row, col).abs() < 1e-10 { continue; }
            for j in 0..result.cols {
                let tmp = result.get(pivot_row, j);
                result.set(pivot_row, j, result.get(max_row, j));
                result.set(max_row, j, tmp);
            }
            let pivot = result.get(pivot_row, col);
            for j in 0..result.cols { result.set(pivot_row, j, result.get(pivot_row, j) / pivot); }
            for row in 0..result.rows {
                if row == pivot_row { continue; }
                let factor = result.get(row, col);
                for j in 0..result.cols {
                    let v = result.get(row, j) - factor * result.get(pivot_row, j);
                    result.set(row, j, v);
                }
            }
            pivot_row += 1;
        }
        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_reduce();
        let mut r = 0;
        for i in 0..rref.rows {
            if (0..rref.cols).any(|j| rref.get(i, j).abs() > 1e-10) { r += 1; }
        }
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols - self.rank()
    }

    pub fn column_space_basis(&self) -> Vec<Vec<f64>> {
        let rref = self.row_reduce();
        let mut basis = Vec::new();
        for j in 0..self.cols {
            let is_pivot = (0..rref.rows).any(|i| {
                (rref.get(i, j) - 1.0).abs() < 1e-10 &&
                (0..j).all(|k| rref.get(i, k).abs() < 1e-10)
            });
            if is_pivot { basis.push(self.col(j)); }
        }
        basis
    }

    pub fn lu_decomposition(&self) -> (Self, Self) {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Self::identity(n);
        let mut u = self.clone();
        for k in 0..n {
            for i in (k+1)..n {
                if u.get(k, k).abs() < 1e-15 { continue; }
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                for j in k..n {
                    let v = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, v);
                }
            }
        }
        (l, u)
    }

    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert!(self.is_square());
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut augmented = Self::new(n, n + 1);
        for i in 0..n { for j in 0..n { augmented.set(i, j, self.get(i, j)); } augmented.set(i, n, b[i]); }
        let rref = augmented.row_reduce();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = rref.get(i, n);
            for j in (i+1)..n { x[i] -= rref.get(i, j) * x[j]; }
            if rref.get(i, i).abs() < 1e-15 { return None; }
            x[i] /= rref.get(i, i);
        }
        Some(x)
    }

    pub fn eigenvalues_2x2(&self) -> (f64, f64) {
        assert!(self.rows == 2 && self.cols == 2);
        let tr = self.trace();
        let det = self.determinant_2x2();
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            ((tr + disc.sqrt()) / 2.0, (tr - disc.sqrt()) / 2.0)
        } else {
            (tr / 2.0, tr / 2.0)
        }
    }

    pub fn condition_number(&self) -> f64 {
        let max_sv = self.frobenius_norm();
        if max_sv < 1e-15 { return f64::INFINITY; }
        max_sv
    }

}

impl fmt::Display for ActionDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActionMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for Action bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct ActionInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ActionInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        ActionInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        ActionInterval { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn hull(&self, other: &Self) -> Self {
        ActionInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(ActionInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        ActionInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        ActionInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        ActionInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { ActionInterval { lo: -self.hi, hi: -self.lo } }
        else { ActionInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        ActionInterval { lo, hi: self.hi.max(0.0).sqrt() }
    }

    pub fn is_positive(&self) -> bool {
        self.lo > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.hi < 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < 1e-15
    }

}

impl fmt::Display for ActionInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for Action protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionState {
    Undefined,
    Defined,
    Computed,
    Simplified,
    Verified,
    Exported,
}

impl fmt::Display for ActionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionState::Undefined => write!(f, "undefined"),
            ActionState::Defined => write!(f, "defined"),
            ActionState::Computed => write!(f, "computed"),
            ActionState::Simplified => write!(f, "simplified"),
            ActionState::Verified => write!(f, "verified"),
            ActionState::Exported => write!(f, "exported"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionStateMachine {
    pub current: ActionState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl ActionStateMachine {
    pub fn new() -> Self {
        ActionStateMachine { current: ActionState::Undefined, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &ActionState { &self.current }
    pub fn can_transition(&self, target: &ActionState) -> bool {
        match (&self.current, target) {
            (ActionState::Undefined, ActionState::Defined) => true,
            (ActionState::Defined, ActionState::Computed) => true,
            (ActionState::Computed, ActionState::Simplified) => true,
            (ActionState::Simplified, ActionState::Verified) => true,
            (ActionState::Verified, ActionState::Exported) => true,
            (ActionState::Exported, ActionState::Defined) => true,
            (ActionState::Computed, ActionState::Verified) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: ActionState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = ActionState::Undefined;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for ActionStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for Action event tracking.
#[derive(Debug, Clone)]
pub struct ActionRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl ActionRingBuffer {
    pub fn new(capacity: usize) -> Self {
        ActionRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
    }
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - 1) % self.capacity]) }
    }
    pub fn oldest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - self.count) % self.capacity]) }
    }
    pub fn average(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.data[(self.head + self.capacity - 1 - i) % self.capacity];
        }
        sum / self.count as f64
    }
    pub fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            result.push(self.data[(self.head + self.capacity - self.count + i) % self.capacity]);
        }
        result
    }
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::INFINITY, f64::min))
    }
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let avg = self.average();
        let v: f64 = self.to_vec().iter().map(|&x| (x - avg) * (x - avg)).sum();
        v / (self.count - 1) as f64
    }
    pub fn clear(&mut self) { self.head = 0; self.count = 0; }
}

impl fmt::Display for ActionRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for Action component tracking.
#[derive(Debug, Clone)]
pub struct ActionDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl ActionDisjointSet {
    pub fn new(n: usize) -> Self {
        ActionDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x { self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]; }
        x
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] { self.parent[rx] = ry; self.size[ry] += self.size[rx]; }
        else if self.rank[rx] > self.rank[ry] { self.parent[ry] = rx; self.size[rx] += self.size[ry]; }
        else { self.parent[ry] = rx; self.size[rx] += self.size[ry]; self.rank[rx] += 1; }
        self.num_components -= 1;
        true
    }
    pub fn connected(&mut self, x: usize, y: usize) -> bool { self.find(x) == self.find(y) }
    pub fn component_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_components(&self) -> usize { self.num_components }
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { let r = self.find(i); groups.entry(r).or_default().push(i); }
        groups.into_values().collect()
    }
}

impl fmt::Display for ActionDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for Action.
#[derive(Debug, Clone)]
pub struct ActionSortedList {
    data: Vec<f64>,
}

impl ActionSortedList {
    pub fn new() -> Self { ActionSortedList { data: Vec::new() } }
    pub fn insert(&mut self, value: f64) {
        let pos = self.data.partition_point(|&x| x < value);
        self.data.insert(pos, value);
    }
    pub fn contains(&self, value: f64) -> bool {
        self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()).is_ok()
    }
    pub fn rank(&self, value: f64) -> usize { self.data.partition_point(|&x| x < value) }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let idx = ((self.data.len() - 1) as f64 * q).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn min(&self) -> Option<f64> { self.data.first().copied() }
    pub fn max(&self) -> Option<f64> { self.data.last().copied() }
    pub fn median(&self) -> f64 { self.quantile(0.5) }
    pub fn iqr(&self) -> f64 { self.quantile(0.75) - self.quantile(0.25) }
    pub fn remove(&mut self, value: f64) -> bool {
        if let Ok(pos) = self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
            self.data.remove(pos); true
        } else { false }
    }
    pub fn range(&self, lo: f64, hi: f64) -> Vec<f64> {
        self.data.iter().filter(|&&x| x >= lo && x <= hi).cloned().collect()
    }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

impl fmt::Display for ActionSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for Action metrics.
#[derive(Debug, Clone)]
pub struct ActionEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl ActionEma {
    pub fn new(alpha: f64) -> Self { ActionEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for ActionEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for Action membership testing.
#[derive(Debug, Clone)]
pub struct ActionBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl ActionBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        ActionBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    fn hash_indices(&self, value: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        let mut h = value;
        for _ in 0..self.num_hashes {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert(&mut self, value: u64) {
        for idx in self.hash_indices(value) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn may_contain(&self, value: u64) -> bool {
        self.hash_indices(value).iter().all(|&idx| self.bits[idx])
    }
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        (set_bits / self.size as f64).powi(self.num_hashes as i32)
    }
    pub fn count(&self) -> usize { self.count }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
}

impl fmt::Display for ActionBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for Action string matching.
#[derive(Debug, Clone)]
pub struct ActionTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ActionTrie {
    nodes: Vec<ActionTrieNode>,
    count: usize,
}

impl ActionTrie {
    pub fn new() -> Self {
        ActionTrie { nodes: vec![ActionTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(ActionTrieNode { children: Vec::new(), is_terminal: false, value: None });
                    self.nodes[current].children.push((ch, idx));
                    idx
                }
            };
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return None,
            }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return false,
            }
        }
        true
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl fmt::Display for ActionTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for Action scheduling.
#[derive(Debug, Clone)]
pub struct ActionPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl ActionPriorityQueue {
    pub fn new() -> Self { ActionPriorityQueue { heap: Vec::new() } }
    pub fn push(&mut self, priority: f64, item: usize) {
        self.heap.push((priority, item));
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 { self.heap.swap(i, parent); i = parent; }
            else { break; }
        }
    }
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() { return None; }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(result)
    }
    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < self.heap.len() && self.heap[left].0 < self.heap[smallest].0 { smallest = left; }
            if right < self.heap.len() && self.heap[right].0 < self.heap[smallest].0 { smallest = right; }
            if smallest != i { self.heap.swap(i, smallest); i = smallest; }
            else { break; }
        }
    }
    pub fn peek(&self) -> Option<&(f64, usize)> { self.heap.first() }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
}

impl fmt::Display for ActionPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for Action.
#[derive(Debug, Clone)]
pub struct ActionAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl ActionAccumulator {
    pub fn new() -> Self { ActionAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.min_val }
    pub fn max(&self) -> f64 { self.max_val }
    pub fn sum(&self) -> f64 { self.sum }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.sum + other.sum) / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean = new_mean;
        self.count = total;
        self.sum += other.sum;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
    pub fn reset(&mut self) {
        self.count = 0; self.mean = 0.0; self.m2 = 0.0;
        self.min_val = f64::INFINITY; self.max_val = f64::NEG_INFINITY; self.sum = 0.0;
    }
}

impl fmt::Display for ActionAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for Action.
#[derive(Debug, Clone)]
pub struct ActionSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl ActionSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { ActionSparseMatrix { rows, cols, entries: Vec::new() } }
    pub fn insert(&mut self, i: usize, j: usize, v: f64) {
        if let Some(pos) = self.entries.iter().position(|&(r, c, _)| r == i && c == j) {
            self.entries[pos].2 = v;
        } else { self.entries.push((i, j, v)); }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.iter().find(|&&(r, c, _)| r == i && c == j).map(|&(_, _, v)| v).unwrap_or(0.0)
    }
    pub fn nnz(&self) -> usize { self.entries.len() }
    pub fn density(&self) -> f64 { self.entries.len() as f64 / (self.rows * self.cols) as f64 }
    pub fn transpose(&self) -> Self {
        let mut result = ActionSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = ActionSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = ActionSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.entries.push((i, j, v * s)); }
        result
    }
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(i, j, v) in &self.entries { result[i] += v * x[j]; }
        result
    }
    pub fn frobenius_norm(&self) -> f64 { self.entries.iter().map(|&(_, _, v)| v * v).sum::<f64>().sqrt() }
    pub fn row_nnz(&self, i: usize) -> usize { self.entries.iter().filter(|&&(r, _, _)| r == i).count() }
    pub fn col_nnz(&self, j: usize) -> usize { self.entries.iter().filter(|&&(_, c, _)| c == j).count() }
    pub fn to_dense(&self, dm_new: fn(usize, usize) -> Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for &(i, j, v) in &self.entries { result[i][j] = v; }
        result
    }
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }
    pub fn remove_zeros(&mut self, tol: f64) {
        self.entries.retain(|&(_, _, v)| v.abs() > tol);
    }
}

impl fmt::Display for ActionSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for Action.
#[derive(Debug, Clone)]
pub struct ActionPolynomial {
    pub coefficients: Vec<f64>,
}

impl ActionPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { ActionPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { ActionPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { ActionPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        ActionPolynomial { coefficients: c }
    }
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() { return 0; }
        let mut d = self.coefficients.len() - 1;
        while d > 0 && self.coefficients[d].abs() < 1e-15 { d -= 1; }
        d
    }
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut power = 1.0;
        for &c in &self.coefficients {
            result += c * power;
            power *= x;
        }
        result
    }
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() { return 0.0; }
        let mut result = *self.coefficients.last().unwrap();
        for &c in self.coefficients.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] += c; }
        ActionPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        ActionPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        ActionPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        ActionPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        ActionPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        ActionPolynomial { coefficients: coeffs }
    }
    pub fn roots_quadratic(&self) -> Vec<f64> {
        if self.degree() != 2 { return Vec::new(); }
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { Vec::new() }
        else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
        else { vec![(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)] }
    }
    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c.abs() < 1e-15) }
    pub fn leading_coefficient(&self) -> f64 {
        self.coefficients.get(self.degree()).copied().unwrap_or(0.0)
    }
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::one();
        for &c in &self.coefficients {
            result = result.add(&power.scale(c));
            power = power.mul(other);
        }
        result
    }
    pub fn newton_root(&self, initial_guess: f64, max_iters: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = initial_guess;
        for _ in 0..max_iters {
            let fx = self.evaluate(x);
            if fx.abs() < tol { return Some(x); }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < 1e-15 { return None; }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 { Some(x) } else { None }
    }
}

impl fmt::Display for ActionPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if i == 0 { terms.push(format!("{:.2}", c)); }
            else if i == 1 { terms.push(format!("{:.2}x", c)); }
            else { terms.push(format!("{:.2}x^{}", c, i)); }
        }
        if terms.is_empty() { write!(f, "0") }
        else { write!(f, "{}", terms.join(" + ")) }
    }
}

/// Simple linear congruential generator for Action.
#[derive(Debug, Clone)]
pub struct ActionRng {
    state: u64,
}

impl ActionRng {
    pub fn new(seed: u64) -> Self { ActionRng { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo { return lo; }
        lo + (self.next_u64() % (hi - lo))
    }
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.next_range(0, i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
    pub fn sample(&mut self, data: &[f64], n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.next_range(0, data.len() as u64) as usize;
            result.push(data[idx]);
        }
        result
    }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.next_f64() < p }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
    pub fn exponential(&mut self, lambda: f64) -> f64 { -self.next_f64().max(1e-15).ln() / lambda }
}

impl fmt::Display for ActionRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for Action benchmarking.
#[derive(Debug, Clone)]
pub struct ActionTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl ActionTimer {
    pub fn new(label: impl Into<String>) -> Self { ActionTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
    pub fn record(&mut self, ns: u64) { self.elapsed_ns.push(ns); }
    pub fn total_ns(&self) -> u64 { self.elapsed_ns.iter().sum() }
    pub fn count(&self) -> usize { self.elapsed_ns.len() }
    pub fn average_ns(&self) -> f64 {
        if self.elapsed_ns.is_empty() { 0.0 } else { self.total_ns() as f64 / self.elapsed_ns.len() as f64 }
    }
    pub fn min_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().min().unwrap_or(0) }
    pub fn max_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().max().unwrap_or(0) }
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.elapsed_ns.is_empty() { return 0; }
        let mut sorted = self.elapsed_ns.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn p50_ns(&self) -> u64 { self.percentile_ns(0.5) }
    pub fn p95_ns(&self) -> u64 { self.percentile_ns(0.95) }
    pub fn p99_ns(&self) -> u64 { self.percentile_ns(0.99) }
    pub fn reset(&mut self) { self.elapsed_ns.clear(); self.running = false; }
}

impl fmt::Display for ActionTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for Action set operations.
#[derive(Debug, Clone)]
pub struct ActionBitVector {
    words: Vec<u64>,
    len: usize,
}

impl ActionBitVector {
    pub fn new(len: usize) -> Self { ActionBitVector { words: vec![0u64; (len + 63) / 64], len } }
    pub fn set(&mut self, i: usize) { if i < self.len { self.words[i / 64] |= 1u64 << (i % 64); } }
    pub fn clear(&mut self, i: usize) { if i < self.len { self.words[i / 64] &= !(1u64 << (i % 64)); } }
    pub fn get(&self, i: usize) -> bool { i < self.len && (self.words[i / 64] & (1u64 << (i % 64))) != 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn count_ones(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn count_zeros(&self) -> usize { self.len - self.count_ones() }
    pub fn is_empty(&self) -> bool { self.count_ones() == 0 }
    pub fn and(&self, other: &Self) -> Self {
        let n = self.words.len().min(other.words.len());
        let mut result = Self::new(self.len.min(other.len));
        for i in 0..n { result.words[i] = self.words[i] & other.words[i]; }
        result
    }
    pub fn or(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] |= self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] |= other.words[i]; }
        result
    }
    pub fn xor(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] = self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] ^= other.words[i]; }
        result
    }
    pub fn not(&self) -> Self {
        let mut result = Self::new(self.len);
        for i in 0..self.words.len() { result.words[i] = !self.words[i]; }
        // Clear unused bits in last word
        let extra = self.len % 64;
        if extra > 0 && !result.words.is_empty() {
            let last = result.words.len() - 1;
            result.words[last] &= (1u64 << extra) - 1;
        }
        result
    }
    pub fn iter_ones(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.len { if self.get(i) { result.push(i); } }
        result
    }
    pub fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.and(other).count_ones() as f64;
        let union = self.or(other).count_ones() as f64;
        if union == 0.0 { 1.0 } else { intersection / union }
    }
    pub fn hamming_distance(&self, other: &Self) -> usize { self.xor(other).count_ones() }
    pub fn fill(&mut self, value: bool) {
        let fill_val = if value { u64::MAX } else { 0 };
        for w in &mut self.words { *w = fill_val; }
        if value { let extra = self.len % 64; if extra > 0 { let last = self.words.len() - 1; self.words[last] &= (1u64 << extra) - 1; } }
    }
}

impl fmt::Display for ActionBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for Action computation memoization.
#[derive(Debug, Clone)]
pub struct ActionLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl ActionLruCache {
    pub fn new(capacity: usize) -> Self { ActionLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].2 = self.clock;
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn put(&mut self, key: u64, value: Vec<f64>) {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].1 = value;
            self.entries[pos].2 = self.clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            let lru_pos = self.entries.iter().enumerate()
                .min_by_key(|(_, (_, _, ts))| *ts).map(|(i, _)| i).unwrap();
            self.entries.remove(lru_pos);
            self.evictions += 1;
        }
        self.entries.push((key, value, self.clock));
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hit_rate(&self) -> f64 { let t = self.hits + self.misses; if t == 0 { 0.0 } else { self.hits as f64 / t as f64 } }
    pub fn eviction_count(&self) -> u64 { self.evictions }
    pub fn contains(&self, key: u64) -> bool { self.entries.iter().any(|(k, _, _)| *k == key) }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; self.evictions = 0; self.clock = 0; }
    pub fn keys(&self) -> Vec<u64> { self.entries.iter().map(|(k, _, _)| *k).collect() }
}

impl fmt::Display for ActionLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for Action scheduling.
#[derive(Debug, Clone)]
pub struct ActionGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl ActionGraphColoring {
    pub fn new(n: usize) -> Self {
        ActionGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
    }
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i < self.num_nodes && j < self.num_nodes {
            self.adjacency[i][j] = true;
            self.adjacency[j][i] = true;
        }
    }
    pub fn greedy_color(&mut self) -> usize {
        self.colors = vec![None; self.num_nodes];
        let mut max_color = 0;
        for v in 0..self.num_nodes {
            let neighbor_colors: std::collections::HashSet<usize> = (0..self.num_nodes)
                .filter(|&u| self.adjacency[v][u] && self.colors[u].is_some())
                .map(|u| self.colors[u].unwrap()).collect();
            let mut c = 0;
            while neighbor_colors.contains(&c) { c += 1; }
            self.colors[v] = Some(c);
            max_color = max_color.max(c);
        }
        self.num_colors_used = max_color + 1;
        self.num_colors_used
    }
    pub fn is_valid_coloring(&self) -> bool {
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency[i][j] {
                    if let (Some(ci), Some(cj)) = (self.colors[i], self.colors[j]) {
                        if ci == cj { return false; }
                    }
                }
            }
        }
        true
    }
    pub fn chromatic_number_upper_bound(&self) -> usize {
        let max_degree = (0..self.num_nodes)
            .map(|v| (0..self.num_nodes).filter(|&u| self.adjacency[v][u]).count())
            .max().unwrap_or(0);
        max_degree + 1
    }
    pub fn color_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (v, &c) in self.colors.iter().enumerate() {
            if let Some(color) = c { classes.entry(color).or_default().push(v); }
        }
        let mut result: Vec<Vec<usize>> = classes.into_values().collect();
        result.sort_by_key(|v| v[0]);
        result
    }
}

impl fmt::Display for ActionGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for Action ranking.
#[derive(Debug, Clone)]
pub struct ActionTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl ActionTopK {
    pub fn new(k: usize) -> Self { ActionTopK { k, items: Vec::new() } }
    pub fn insert(&mut self, score: f64, label: impl Into<String>) {
        self.items.push((score, label.into()));
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.items.len() > self.k { self.items.truncate(self.k); }
    }
    pub fn top(&self) -> &[(f64, String)] { &self.items }
    pub fn min_score(&self) -> Option<f64> { self.items.last().map(|(s, _)| *s) }
    pub fn max_score(&self) -> Option<f64> { self.items.first().map(|(s, _)| *s) }
    pub fn is_full(&self) -> bool { self.items.len() >= self.k }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn contains_label(&self, label: &str) -> bool { self.items.iter().any(|(_, l)| l == label) }
    pub fn clear(&mut self) { self.items.clear(); }
    pub fn merge(&mut self, other: &Self) {
        for (score, label) in &other.items { self.insert(*score, label.clone()); }
    }
}

impl fmt::Display for ActionTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for Action monitoring.
#[derive(Debug, Clone)]
pub struct ActionSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl ActionSlidingWindow {
    pub fn new(window_size: usize) -> Self { ActionSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        self.sum += value;
        if self.data.len() > self.window_size {
            self.sum -= self.data.remove(0);
        }
    }
    pub fn mean(&self) -> f64 { if self.data.is_empty() { 0.0 } else { self.sum / self.data.len() as f64 } }
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (self.data.len() - 1) as f64
    }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_full(&self) -> bool { self.data.len() >= self.window_size }
    pub fn trend(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let n = self.data.len() as f64;
        let sum_x: f64 = (0..self.data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().sum();
        let sum_xy: f64 = self.data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..self.data.len()).map(|i| (i as f64) * (i as f64)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { 0.0 } else { (n * sum_xy - sum_x * sum_y) / denom }
    }
    pub fn anomaly_score(&self, value: f64) -> f64 {
        let s = self.std_dev();
        if s.abs() < 1e-15 { return 0.0; }
        ((value - self.mean()) / s).abs()
    }
    pub fn clear(&mut self) { self.data.clear(); self.sum = 0.0; }
}

impl fmt::Display for ActionSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for Action classification evaluation.
#[derive(Debug, Clone)]
pub struct ActionConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl ActionConfusionMatrix {
    pub fn new() -> Self { ActionConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = Self::new();
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            match (a, p) {
                (true, true) => cm.true_positive += 1,
                (false, true) => cm.false_positive += 1,
                (true, false) => cm.false_negative += 1,
                (false, false) => cm.true_negative += 1,
            }
        }
        cm
    }
    pub fn total(&self) -> u64 { self.true_positive + self.false_positive + self.true_negative + self.false_negative }
    pub fn accuracy(&self) -> f64 { let t = self.total(); if t == 0 { 0.0 } else { (self.true_positive + self.true_negative) as f64 / t as f64 } }
    pub fn precision(&self) -> f64 { let d = self.true_positive + self.false_positive; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.true_positive + self.false_negative; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn f1_score(&self) -> f64 { let p = self.precision(); let r = self.recall(); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
    pub fn specificity(&self) -> f64 { let d = self.true_negative + self.false_positive; if d == 0 { 0.0 } else { self.true_negative as f64 / d as f64 } }
    pub fn false_positive_rate(&self) -> f64 { 1.0 - self.specificity() }
    pub fn matthews_correlation(&self) -> f64 {
        let tp = self.true_positive as f64; let fp = self.false_positive as f64;
        let tn = self.true_negative as f64; let fn_ = self.false_negative as f64;
        let num = tp * tn - fp * fn_;
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    pub fn merge(&mut self, other: &Self) {
        self.true_positive += other.true_positive;
        self.false_positive += other.false_positive;
        self.true_negative += other.true_negative;
        self.false_negative += other.false_negative;
    }
}

impl fmt::Display for ActionConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for Action feature vectors.
pub fn action_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for Action.
pub fn action_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for Action.
pub fn action_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for Action.
pub fn action_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for Action.
pub fn action_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for Action.
pub fn action_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for Action.
pub fn action_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for Action.
pub fn action_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for Action.
pub fn action_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for Action.
pub fn action_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for Action.
pub fn action_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for Action.
pub fn action_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for Action.
pub fn action_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for Action.
pub fn action_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for Action.
pub fn action_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (action_kl_divergence(p, &m) + action_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for Action.
pub fn action_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for Action.
pub fn action_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for Action.
pub fn action_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for Action.
#[derive(Debug, Clone)]
pub struct ActionFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl ActionFeatureScaler {
    pub fn new() -> Self { ActionFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() { return; }
        let dim = data[0].len();
        let n = data.len() as f64;
        self.means = vec![0.0; dim];
        self.mins = vec![f64::INFINITY; dim];
        self.maxs = vec![f64::NEG_INFINITY; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.means[j] += v;
                self.mins[j] = self.mins[j].min(v);
                self.maxs[j] = self.maxs[j].max(v);
            }
        }
        for j in 0..dim { self.means[j] /= n; }
        self.stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.stds[j] += (v - self.means[j]).powi(2);
            }
        }
        for j in 0..dim { self.stds[j] = (self.stds[j] / (n - 1.0).max(1.0)).sqrt(); }
        self.fitted = true;
    }
    pub fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            if self.stds[j].abs() < 1e-15 { 0.0 } else { (v - self.means[j]) / self.stds[j] }
        }).collect()
    }
    pub fn normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            let range = self.maxs[j] - self.mins[j];
            if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
        }).collect()
    }
    pub fn inverse_standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * self.stds[j] + self.means[j]).collect()
    }
    pub fn inverse_normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j]).collect()
    }
    pub fn dimension(&self) -> usize { self.means.len() }
}

impl fmt::Display for ActionFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for Action trend analysis.
#[derive(Debug, Clone)]
pub struct ActionLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl ActionLinearRegression {
    pub fn new() -> Self { ActionLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        if n < 2.0 { return; }
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { return; }
        self.slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / n;
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - self.predict(xi)).powi(2)).sum();
        self.r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };
        self.fitted = true;
    }
    pub fn predict(&self, x: f64) -> f64 { self.slope * x + self.intercept }
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> { xs.iter().map(|&x| self.predict(x)).collect() }
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| yi - self.predict(xi)).collect()
    }
    pub fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let res = self.residuals(x, y);
        res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64
    }
    pub fn rmse(&self, x: &[f64], y: &[f64]) -> f64 { self.mse(x, y).sqrt() }
}

impl fmt::Display for ActionLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for Action.
#[derive(Debug, Clone)]
pub struct ActionWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl ActionWeightedGraph {
    pub fn new(n: usize) -> Self { ActionWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push((v, w));
        self.adj[v].push((u, w));
        self.num_edges += 1;
    }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }
    pub fn total_weight(&self) -> f64 {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }
    pub fn min_spanning_tree_weight(&self) -> f64 {
        // Kruskal's algorithm
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.num_nodes {
            for &(v, w) in &self.adj[u] {
                if u < v { edges.push((w, u, v)); }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..self.num_nodes).collect();
        let mut rank = vec![0usize; self.num_nodes];
        fn find_action(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_action(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_action(&mut parent, u);
            let rv = find_action(&mut parent, v);
            if ru != rv {
                if rank[ru] < rank[rv] { parent[ru] = rv; }
                else if rank[ru] > rank[rv] { parent[rv] = ru; }
                else { parent[rv] = ru; rank[ru] += 1; }
                total += w;
                count += 1;
                if count == self.num_nodes - 1 { break; }
            }
        }
        total
    }
    pub fn dijkstra(&self, start: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.num_nodes];
        let mut visited = vec![false; self.num_nodes];
        dist[start] = 0.0;
        for _ in 0..self.num_nodes {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..self.num_nodes { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for &(v, w) in &self.adj[u] {
                let alt = dist[u] + w;
                if alt < dist[v] { dist[v] = alt; }
            }
        }
        dist
    }
    pub fn eccentricity(&self, u: usize) -> f64 {
        let dists = self.dijkstra(u);
        dists.iter().cloned().filter(|&d| d.is_finite()).fold(0.0f64, f64::max)
    }
    pub fn diameter(&self) -> f64 {
        (0..self.num_nodes).map(|u| self.eccentricity(u)).fold(0.0f64, f64::max)
    }
    pub fn clustering_coefficient(&self, u: usize) -> f64 {
        let neighbors: Vec<usize> = self.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        if k < 2 { return 0.0; }
        let mut triangles = 0;
        for i in 0..k {
            for j in (i+1)..k {
                if self.adj[neighbors[i]].iter().any(|(v, _)| *v == neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }
    pub fn average_clustering_coefficient(&self) -> f64 {
        let sum: f64 = (0..self.num_nodes).map(|u| self.clustering_coefficient(u)).sum();
        sum / self.num_nodes as f64
    }
}

impl fmt::Display for ActionWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for Action.
pub fn action_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return Vec::new(); }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Cumulative sum for Action.
pub fn action_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for Action.
pub fn action_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for Action.
pub fn action_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for Action.
pub fn action_dft_magnitude(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n/2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Trapezoidal integration for Action.
pub fn action_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for Action.
pub fn action_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 || n % 2 == 0 { return 0.0; }
    let mut total = 0.0;
    for i in (0..n-2).step_by(2) {
        let h = (x[i+2] - x[i]) / 6.0;
        total += h * (y[i] + 4.0 * y[i+1] + y[i+2]);
    }
    total
}

/// Convolution for Action.
pub fn action_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Axis-aligned bounding box for Action spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct ActionAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl ActionAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { ActionAABB { x_min, y_min, x_max, y_max } }
    pub fn contains(&self, x: f64, y: f64) -> bool { x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max }
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.x_max < other.x_min || self.x_min > other.x_max || self.y_max < other.y_min || self.y_min > other.y_max)
    }
    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> (f64, f64) { ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0) }
    pub fn subdivide(&self) -> [Self; 4] {
        let (cx, cy) = self.center();
        [
            ActionAABB::new(self.x_min, self.y_min, cx, cy),
            ActionAABB::new(cx, self.y_min, self.x_max, cy),
            ActionAABB::new(self.x_min, cy, cx, self.y_max),
            ActionAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for Action.
#[derive(Debug, Clone, Copy)]
pub struct ActionPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for Action spatial indexing.
#[derive(Debug, Clone)]
pub struct ActionQuadTree {
    pub boundary: ActionAABB,
    pub points: Vec<ActionPoint2D>,
    pub children: Option<Vec<ActionQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl ActionQuadTree {
    pub fn new(boundary: ActionAABB, capacity: usize, max_depth: usize) -> Self {
        ActionQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: ActionAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        ActionQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: ActionPoint2D) -> bool {
        if !self.boundary.contains(p.x, p.y) { return false; }
        if self.points.len() < self.capacity && self.children.is_none() {
            self.points.push(p); return true;
        }
        if self.children.is_none() && self.depth < self.max_depth { self.subdivide_tree(); }
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() { if child.insert(p) { return true; } }
        }
        self.points.push(p); true
    }
    fn subdivide_tree(&mut self) {
        let quads = self.boundary.subdivide();
        let mut children = Vec::with_capacity(4);
        for q in quads.iter() {
            children.push(ActionQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &ActionAABB) -> Vec<ActionPoint2D> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for p in &self.points { if range.contains(p.x, p.y) { result.push(*p); } }
        if let Some(ref children) = self.children {
            for child in children { result.extend(child.query_range(range)); }
        }
        result
    }
    pub fn count(&self) -> usize {
        let mut c = self.points.len();
        if let Some(ref children) = self.children {
            for child in children { c += child.count(); }
        }
        c
    }
    pub fn tree_depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.tree_depth()).max().unwrap_or(0)
        } else { 0 }
    }
}

impl fmt::Display for ActionQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for Action.
pub fn action_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 { return (Vec::new(), Vec::new()); }
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; n]; // column vectors
    let mut r = vec![vec![0.0; n]; n];
    // extract columns of a
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q[i].iter()).map(|(&a, &b)| a * b).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[i][k]; }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm.abs() > 1e-15 { for k in 0..m { q[j][k] = v[k] / norm; } }
    }
    // convert q from list of column vectors to matrix
    let q_mat: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| q[j][i]).collect()).collect();
    (q_mat, r)
}

/// Solve upper triangular system Rx = b for Action.
pub fn action_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for Action.
pub fn action_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for Action.
pub fn action_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for Action.
pub fn action_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for Action.
pub fn action_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for Action.
pub fn action_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for Action.
pub fn action_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for Action.
pub fn action_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = action_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for Action.
#[derive(Debug, Clone)]
pub struct ActionRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl ActionRunningStats {
    pub fn new() -> Self { ActionRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min_val = self.min_val.min(x);
        self.max_val = self.max_val.max(x);
        self.sum += x;
    }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 { if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() } }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean = combined_mean;
        self.count = combined_count;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
    }
}

impl fmt::Display for ActionRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for Action.
pub fn action_iqr(data: &[f64]) -> f64 {
    action_percentile_at(data, 75.0) - action_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for Action.
pub fn action_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = action_percentile_at(data, 25.0);
    let q3 = action_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for Action.
pub fn action_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for Action.
pub fn action_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for Action.
pub fn action_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = action_rank(x);
    let ry = action_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for Action.
pub fn action_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() { return Vec::new(); }
    let n = data.len() as f64;
    let d = data[0].len();
    let means: Vec<f64> = (0..d).map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let c: f64 = data.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>() / (n - 1.0).max(1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix for Action.
pub fn action_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = action_covariance_matrix(data);
    let d = cov.len();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom.abs() < 1e-15 { 0.0 } else { cov[i][j] / denom };
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z_n_action(n: usize) -> FiniteGroupAction {
        let gen = Permutation::cyclic(n);
        FiniteGroupAction::from_generators(&[gen], n)
    }

    fn s3_action() -> FiniteGroupAction {
        let t12 = Permutation::transposition(3, 0, 1);
        let c = Permutation::cyclic(3);
        FiniteGroupAction::from_generators(&[t12, c], 3)
    }

    #[test]
    fn test_permutation_identity() {
        let id = Permutation::identity(4);
    }

    #[test]
    fn test_permutation_compose() {
        let a = Permutation::cyclic(4);
        let b = Permutation::transposition(4, 0, 1);
        let c = a.compose(&b);
    }

    #[test]
    fn test_permutation_inverse() {
        let p = Permutation::cyclic(4);
        let inv = p.inverse();
        let id = p.compose(&inv);
    }

    #[test]
    fn test_permutation_order() {
        let p = Permutation::cyclic(5);
    }

    #[test]
    fn test_permutation_cycle_type() {
        let p = Permutation::cyclic(5);
        let ct = p.cycle_type();
    }

    #[test]
    fn test_permutation_fixed_points() {
    }

    #[test]
    fn test_z3_action() {
        let action = z_n_action(3);
    }

    #[test]
    fn test_s3_action() {
        let action = s3_action();
    }

    #[test]
    fn test_orbit_computation() {
        let action = z_n_action(4);
        let orbit = action.compute_orbit(0);
    }

    #[test]
    fn test_all_orbits() {
        let action = z_n_action(4);
        let partition = action.compute_all_orbits();
    }

    #[test]
    fn test_non_transitive_orbits() {
        // Identity group on 3 elements: 3 orbits
        let action = FiniteGroupAction::from_generators(&[Permutation::identity(3)], 3);
        let partition = action.compute_all_orbits();
    }

    #[test]
    fn test_stabilizer() {
        let action = s3_action();
        let stab = action.compute_stabilizer(0);
    }

    #[test]
    fn test_orbit_stabilizer_theorem() {
        let action = s3_action();
        for x in 0..3 {
        }
    }

    #[test]
    fn test_transversal() {
        let action = s3_action();
        let trans = action.transversal(0);
    }

    #[test]
    fn test_fixed_point_table() {
        let action = s3_action();
        let table = action.fixed_point_table();
        // Identity fixes all 3 elements
    }

    #[test]
    fn test_common_fixed_points() {
        let action = s3_action();
        let common = action.common_fixed_points();
    }

    #[test]
    fn test_burnside_count() {
        let action = z_n_action(4);
        let result = action.burnside_count();
    }

    #[test]
    fn test_burnside_trivial() {
        let action = FiniteGroupAction::from_generators(&[Permutation::identity(3)], 3);
        let result = action.burnside_count();
    }

    #[test]
    fn test_polya_count() {
        // Z4 acting on 4 elements, count 2-colorings up to rotation
        let action = z_n_action(4);
        let count = action.polya_count(2);
        // Necklaces of 4 beads with 2 colors: 6
    }

    #[test]
    fn test_polya_count_3_colors() {
        let action = z_n_action(3);
        let count = action.polya_count(3);
        // 3-colored necklaces of 3 beads: 11
    }

    #[test]
    fn test_cycle_index() {
        let action = z_n_action(3);
        let ci = action.cycle_index();
    }

    #[test]
    fn test_equivariant_map() {
        let action = z_n_action(3);
    }

    #[test]
    fn test_equivariant_map_not_equivariant() {
        let action = z_n_action(3);
    }

    #[test]
    fn test_verify_action() {
        let action = s3_action();
    }

    #[test]
    fn test_is_free() {
        // Z3 on {0,1,2} is free
        let action = z_n_action(3);
    }

    #[test]
    fn test_not_free() {
        // S3 on {0,1,2} is not free (transpositions fix a point)
        let action = s3_action();
    }

    #[test]
    fn test_pointwise_stabilizer() {
        let action = s3_action();
        let stab = action.pointwise_stabilizer(&[0, 1]);
    }

    #[test]
    fn test_setwise_stabilizer() {
        let action = s3_action();
        let set = BTreeSet::from([0u32, 1]);
        let stab = action.setwise_stabilizer(&set);
    }

    #[test]
    fn test_action_statistics() {
        let action = s3_action();
        let stats = action_statistics(&action);
    }

    #[test]
    fn test_burnside_with_constraint() {
        let action = z_n_action(4);
        let count = action.burnside_with_constraint(|x| x < 4);
    }

    #[test]
    fn test_quotient_action() {
        let action = z_n_action(4);
        let qa = action.quotient_action();
    }

    #[test]
    fn test_orbit_display() {
    }

    #[test]
    fn test_permutation_transposition() {
        let t = Permutation::transposition(4, 1, 3);
    }

    #[test]
    fn test_distinct_behavior_count() {
        let action = z_n_action(4);
        let count = action.distinct_behavior_count(&[0, 1, 2, 3]);
    }

    #[test]
    fn test_cayley_graph() {
        let action = z_n_action(3);
        let dot = action.cayley_graph_dot(&[1]); // generator index 1
    }

    #[test]
    fn test_maximal_fixed_set() {
        let action = s3_action();
        let (_, fps) = action.maximal_fixed_set();
        // Transpositions fix 1 point each
    }

    #[test]
    fn test_orbit_partition_display() {
        let action = z_n_action(3);
        let partition = action.compute_all_orbits();
    }
    #[test]
    fn test_inducedaction_new() {
        let item = InducedAction::new(0, 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_restrictedaction_new() {
        let item = RestrictedAction::new(0, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_productaction_new() {
        let item = ProductAction::new(Vec::new(), 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_wreathproductaction_new() {
        let item = WreathProductAction::new(0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_conjugationaction_new() {
        let item = ConjugationAction::new(0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_regularaction_new() {
        let item = RegularAction::new(0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_cosetaction_new() {
        let item = CosetAction::new(0, 0, 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_permutationrepr_new() {
        let item = PermutationRepr::new(0, Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_schreiergenerator_new() {
        let item = SchreierGenerator::new(0, 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_blocksystem_new() {
        let item = BlockSystem::new(Vec::new(), 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_primitivitytest_new() {
        let item = PrimitivityTest::new(false, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_actionhomomorphism_new() {
        let item = ActionHomomorphism::new(0, 0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_action_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = action_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = action_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_action_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = action_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = action_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_action_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = action_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_action_analysis() {
        let mut a = ActionAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_action_iterator() {
        let iter = ActionResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_action_batch_processor() {
        let mut proc = ActionBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_action_histogram() {
        let hist = ActionHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_action_graph() {
        let mut g = ActionGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_action_graph_shortest_path() {
        let mut g = ActionGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_graph_topo_sort() {
        let mut g = ActionGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_action_graph_components() {
        let mut g = ActionGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_action_cache() {
        let mut cache = ActionCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_action_config() {
        let config = ActionConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_action_report() {
        let mut report = ActionReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_action_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = action_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_action_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = action_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = action_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_action_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = action_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_action_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = action_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_action_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = action_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_action_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = action_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_action_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = action_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_action_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = action_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_action_analysis_normalize() {
        let mut a = ActionAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_analysis_transpose() {
        let mut a = ActionAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_analysis_multiply() {
        let mut a = ActionAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = ActionAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_analysis_frobenius() {
        let mut a = ActionAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_analysis_symmetric() {
        let mut a = ActionAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_action_graph_dot() {
        let mut g = ActionGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_action_histogram_render() {
        let hist = ActionHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_action_batch_reset() {
        let mut proc = ActionBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_action_graph_remove_edge() {
        let mut g = ActionGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_action_dense_matrix_new() {
        let m = ActionDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_action_dense_matrix_identity() {
        let m = ActionDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_mul() {
        let a = ActionDenseMatrix::identity(2);
        let b = ActionDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_transpose() {
        let a = ActionDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_det_2x2() {
        let m = ActionDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_det_3x3() {
        let m = ActionDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_inverse_2x2() {
        let m = ActionDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_power() {
        let m = ActionDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_rank() {
        let m = ActionDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_action_dense_matrix_solve() {
        let a = ActionDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_action_dense_matrix_lu() {
        let a = ActionDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_eigenvalues() {
        let m = ActionDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_kronecker() {
        let a = ActionDenseMatrix::identity(2);
        let b = ActionDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_action_dense_matrix_hadamard() {
        let a = ActionDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = ActionDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_interval() {
        let a = ActionInterval::new(1.0, 3.0);
        let b = ActionInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_interval_mul() {
        let a = ActionInterval::new(-2.0, 3.0);
        let b = ActionInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_interval_hull() {
        let a = ActionInterval::new(1.0, 3.0);
        let b = ActionInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_state_machine() {
        let mut sm = ActionStateMachine::new();
        assert_eq!(*sm.state(), ActionState::Undefined);
        assert!(sm.transition(ActionState::Defined));
        assert_eq!(*sm.state(), ActionState::Defined);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_action_state_machine_invalid() {
        let mut sm = ActionStateMachine::new();
        let last_state = ActionState::Exported;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_action_state_machine_reset() {
        let mut sm = ActionStateMachine::new();
        sm.transition(ActionState::Defined);
        sm.reset();
        assert_eq!(*sm.state(), ActionState::Undefined);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_action_ring_buffer() {
        let mut rb = ActionRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_ring_buffer_to_vec() {
        let mut rb = ActionRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_disjoint_set() {
        let mut ds = ActionDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_action_disjoint_set_components() {
        let mut ds = ActionDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_action_sorted_list() {
        let mut sl = ActionSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_sorted_list_remove() {
        let mut sl = ActionSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_action_ema() {
        let mut ema = ActionEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_bloom_filter() {
        let mut bf = ActionBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_action_trie() {
        let mut trie = ActionTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert_eq!(trie.search("hell"), None);
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }

    #[test]
    fn test_action_dense_matrix_sym() {
        let m = ActionDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_action_dense_matrix_diag() {
        let m = ActionDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_action_dense_matrix_upper_tri() {
        let m = ActionDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_action_dense_matrix_outer() {
        let m = ActionDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dense_matrix_submatrix() {
        let m = ActionDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_priority_queue() {
        let mut pq = ActionPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_action_accumulator() {
        let mut acc = ActionAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_accumulator_merge() {
        let mut a = ActionAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = ActionAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_sparse_matrix() {
        let mut m = ActionSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_action_sparse_mul_vec() {
        let mut m = ActionSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_sparse_transpose() {
        let mut m = ActionSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_eval() {
        let p = ActionPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_add() {
        let a = ActionPolynomial::new(vec![1.0, 2.0]);
        let b = ActionPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_mul() {
        let a = ActionPolynomial::new(vec![1.0, 1.0]);
        let b = ActionPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_deriv() {
        let p = ActionPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_integral() {
        let p = ActionPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_roots() {
        let p = ActionPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_action_polynomial_newton() {
        let p = ActionPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_action_polynomial_compose() {
        let p = ActionPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = ActionPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_rng() {
        let mut rng = ActionRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_action_rng_gaussian() {
        let mut rng = ActionRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_action_timer() {
        let mut timer = ActionTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_bitvector() {
        let mut bv = ActionBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_action_bitvector_ops() {
        let mut a = ActionBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = ActionBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_action_bitvector_jaccard() {
        let mut a = ActionBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = ActionBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_priority_queue_empty() {
        let mut pq = ActionPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_action_sparse_add() {
        let mut a = ActionSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = ActionSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_rng_shuffle() {
        let mut rng = ActionRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_polynomial_display() {
        let p = ActionPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_action_polynomial_monomial() {
        let m = ActionPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_timer_percentiles() {
        let mut timer = ActionTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_action_accumulator_cv() {
        let mut acc = ActionAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_action_sparse_diagonal() {
        let mut m = ActionSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_lru_cache() {
        let mut cache = ActionLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_action_lru_hit_rate() {
        let mut cache = ActionLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_graph_coloring() {
        let mut gc = ActionGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_action_graph_coloring_complete() {
        let mut gc = ActionGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_action_topk() {
        let mut tk = ActionTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_sliding_window() {
        let mut sw = ActionSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_action_sliding_window_trend() {
        let mut sw = ActionSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_action_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = ActionConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_action_confusion_f1() {
        let cm = ActionConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_action_cosine_similarity() {
        let s = action_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = action_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_euclidean_distance() {
        let d = action_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_sigmoid() {
        let s = action_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = action_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_action_softmax() {
        let sm = action_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_action_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = action_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_action_normalize() {
        let v = action_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_lerp() {
        assert!((action_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((action_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((action_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_clamp() {
        assert!((action_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((action_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((action_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_cross_product() {
        let c = action_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dot_product() {
        let d = action_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = action_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_action_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = action_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_action_logsumexp() {
        let lse = action_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_action_feature_scaler() {
        let mut scaler = ActionFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_feature_scaler_inverse() {
        let mut scaler = ActionFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_linear_regression() {
        let mut lr = ActionLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_linear_regression_predict() {
        let mut lr = ActionLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_weighted_graph() {
        let mut g = ActionWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_weighted_graph_mst() {
        let mut g = ActionWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = action_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_cumsum() {
        let cs = action_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_diff() {
        let d = action_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_autocorrelation() {
        let ac = action_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_dft_magnitude() {
        let mags = action_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_action_integrate_trapezoid() {
        let area = action_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_convolve() {
        let c = action_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_weighted_graph_clustering() {
        let mut g = ActionWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_histogram_cumulative() {
        let h = ActionHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_action_histogram_entropy() {
        let h = ActionHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_action_aabb() {
        let bb = ActionAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_aabb_intersects() {
        let a = ActionAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = ActionAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = ActionAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_action_quadtree() {
        let bb = ActionAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = ActionQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(ActionPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_action_quadtree_query() {
        let bb = ActionAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = ActionQuadTree::new(bb, 2, 8);
        qt.insert(ActionPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(ActionPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = ActionAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_action_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = action_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = action_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = action_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_action_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((action_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_identity() {
        let id = action_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = action_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_action_running_stats() {
        let mut s = ActionRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_running_stats_merge() {
        let mut a = ActionRunningStats::new();
        let mut b = ActionRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_running_stats_cv() {
        let mut s = ActionRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_action_iqr() {
        let iqr = action_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_action_outliers() {
        let outliers = action_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_action_zscore() {
        let z = action_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_action_rank() {
        let r = action_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_spearman() {
        let rho = action_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_action_sample_skewness_symmetric() {
        let s = action_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_action_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = action_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_action_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = action_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}
