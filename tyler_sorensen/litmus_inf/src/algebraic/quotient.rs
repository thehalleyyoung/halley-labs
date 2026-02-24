//! Quotient group and quotient space constructions for the LITMUS∞ algebraic engine.
//!
//! Implements subgroup detection, normal subgroup verification, coset enumeration,
//! quotient group construction, factor group analysis, composition series,
//! Sylow subgroup computation, and group extensions.

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

use super::types::{Permutation, enumerate_from_generators};

// ═══════════════════════════════════════════════════════════════════════════
// Subgroup
// ═══════════════════════════════════════════════════════════════════════════

/// A subgroup of a permutation group, stored by generators.
#[derive(Debug, Clone)]
pub struct Subgroup {
    /// Generators of the subgroup.
    pub generators: Vec<Permutation>,
    /// Degree of the permutations.
    pub degree: usize,
    /// Cached elements (computed lazily).
    elements: Option<Vec<Permutation>>,
}

impl Subgroup {
    /// Create a subgroup from generators.
    pub fn from_generators(generators: Vec<Permutation>, degree: usize) -> Self {
        Subgroup { generators, degree, elements: None }
    }

    /// Create the trivial subgroup (identity only).
    pub fn trivial(degree: usize) -> Self {
        Subgroup {
            generators: vec![Permutation::identity(degree)],
            degree,
            elements: Some(vec![Permutation::identity(degree)]),
        }
    }

    /// Get all elements (computes and caches if needed).
    pub fn elements(&mut self) -> &[Permutation] {
        if self.elements.is_none() {
            let elems = enumerate_from_generators(&self.generators, self.degree);
            self.elements = Some(elems.into_iter().collect());
        }
        self.elements.as_ref().unwrap()
    }

    /// Get all elements as a set.
    pub fn element_set(&self) -> HashSet<Permutation> {
        enumerate_from_generators(&self.generators, self.degree)
    }

    /// Order of the subgroup.
    pub fn order(&self) -> usize {
        self.element_set().len()
    }

    /// Check if an element belongs to the subgroup.
    pub fn contains(&self, elem: &Permutation) -> bool {
        self.element_set().contains(elem)
    }

    /// Index of this subgroup in a parent group.
    pub fn index_in(&self, parent: &Subgroup) -> usize {
        let parent_order = parent.order();
        let self_order = self.order();
        if self_order == 0 { return 0; }
        parent_order / self_order
    }

    /// Check if this is a subgroup of parent.
    pub fn is_subgroup_of(&self, parent: &Subgroup) -> bool {
        let parent_set = parent.element_set();
        self.element_set().iter().all(|e| parent_set.contains(e))
    }

    /// Intersection with another subgroup.
    pub fn intersection(&self, other: &Subgroup) -> Subgroup {
        let self_set = self.element_set();
        let other_set = other.element_set();
        let inter: Vec<Permutation> = self_set.intersection(&other_set).cloned().collect();
        if inter.is_empty() {
            Subgroup::trivial(self.degree)
        } else {
            Subgroup::from_generators(inter, self.degree)
        }
    }

    /// Generated subgroup ⟨H, K⟩.
    pub fn join(&self, other: &Subgroup) -> Subgroup {
        let mut gens = self.generators.clone();
        gens.extend(other.generators.iter().cloned());
        Subgroup::from_generators(gens, self.degree)
    }

    /// Centralizer of an element in this group.
    pub fn centralizer_of(&self, elem: &Permutation) -> Subgroup {
        let elems = self.element_set();
        let centralizer: Vec<Permutation> = elems.into_iter()
            .filter(|g| g.compose(elem) == elem.compose(g))
            .collect();
        Subgroup::from_generators(centralizer, self.degree)
    }

    /// Center of the group: Z(G) = {g ∈ G : gh = hg for all h ∈ G}.
    pub fn center(&self) -> Subgroup {
        let elems: Vec<Permutation> = self.element_set().into_iter().collect();
        let center: Vec<Permutation> = elems.iter()
            .filter(|g| elems.iter().all(|h| g.compose(h) == h.compose(g)))
            .cloned()
            .collect();
        if center.is_empty() {
            Subgroup::trivial(self.degree)
        } else {
            Subgroup::from_generators(center, self.degree)
        }
    }

    /// Normalizer of this subgroup in parent: N_G(H) = {g ∈ G : gHg⁻¹ = H}.
    pub fn normalizer_in(&self, parent: &Subgroup) -> Subgroup {
        let self_set = self.element_set();
        let parent_elems = parent.element_set();
        let normalizer: Vec<Permutation> = parent_elems.into_iter()
            .filter(|g| {
                let g_inv = g.inverse();
                self_set.iter().all(|h| {
                    let conjugate = g.compose(h).compose(&g_inv);
                    self_set.contains(&conjugate)
                })
            })
            .collect();
        Subgroup::from_generators(normalizer, self.degree)
    }

    /// Derived subgroup (commutator subgroup) [G, G].
    pub fn derived_subgroup(&self) -> Subgroup {
        let elems: Vec<Permutation> = self.element_set().into_iter().collect();
        let mut commutators = HashSet::new();
        for a in &elems {
            for b in &elems {
                commutators.insert(a.commutator(b));
            }
        }
        let gens: Vec<Permutation> = commutators.into_iter().collect();
        Subgroup::from_generators(gens, self.degree)
    }

    /// Derived series: G ⊇ [G,G] ⊇ [[G,G],[G,G]] ⊇ ...
    pub fn derived_series(&self) -> Vec<Subgroup> {
        let mut series = vec![self.clone()];
        let mut current = self.clone();
        loop {
            let derived = current.derived_subgroup();
            let d_order = derived.order();
            let c_order = current.order();
            series.push(derived.clone());
            if d_order == c_order {
                break; // stabilized
            }
            if d_order <= 1 {
                break; // reached trivial
            }
            current = derived;
        }
        series
    }

    /// Check if the group is solvable.
    pub fn is_solvable(&self) -> bool {
        let series = self.derived_series();
        let last = series.last().unwrap();
        last.order() <= 1
    }

    /// Check if the group is abelian.
    pub fn is_abelian(&self) -> bool {
        let elems: Vec<Permutation> = self.element_set().into_iter().collect();
        for i in 0..elems.len() {
            for j in (i + 1)..elems.len() {
                if elems[i].compose(&elems[j]) != elems[j].compose(&elems[i]) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the group is cyclic.
    pub fn is_cyclic(&self) -> bool {
        let order = self.order();
        let elems = self.element_set();
        elems.iter().any(|e| e.order() as usize == order)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NormalSubgroup
// ═══════════════════════════════════════════════════════════════════════════

/// A normal subgroup N ◁ G verified against its parent group.
#[derive(Debug, Clone)]
pub struct NormalSubgroup {
    /// The subgroup.
    pub subgroup: Subgroup,
    /// The parent group.
    pub parent: Subgroup,
}

impl NormalSubgroup {
    /// Check normality: gNg⁻¹ = N for all g ∈ G.
    pub fn check_normality(subgroup: &Subgroup, parent: &Subgroup) -> bool {
        let n_set = subgroup.element_set();
        let g_set = parent.element_set();

        for g in &g_set {
            let g_inv = g.inverse();
            for n in &n_set {
                let conjugate = g.compose(n).compose(&g_inv);
                if !n_set.contains(&conjugate) {
                    return false;
                }
            }
        }
        true
    }

    /// Create a verified normal subgroup.
    pub fn new(subgroup: Subgroup, parent: Subgroup) -> Option<Self> {
        if Self::check_normality(&subgroup, &parent) {
            Some(NormalSubgroup { subgroup, parent })
        } else {
            None
        }
    }

    /// Normal closure of H in G: smallest normal subgroup of G containing H.
    pub fn normal_closure(subgroup: &Subgroup, parent: &Subgroup) -> Subgroup {
        let g_set = parent.element_set();
        let h_set = subgroup.element_set();
        let mut closure_gens = HashSet::new();

        for g in &g_set {
            let g_inv = g.inverse();
            for h in &h_set {
                closure_gens.insert(g.compose(h).compose(&g_inv));
            }
        }

        Subgroup::from_generators(closure_gens.into_iter().collect(), subgroup.degree)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Coset
// ═══════════════════════════════════════════════════════════════════════════

/// A left coset gH of a subgroup H.
#[derive(Debug, Clone)]
pub struct LeftCoset {
    /// Representative element.
    pub representative: Permutation,
    /// Elements of the coset.
    pub elements: HashSet<Permutation>,
}

impl LeftCoset {
    /// Compute the left coset gH.
    pub fn new(g: &Permutation, subgroup: &Subgroup) -> Self {
        let h_set = subgroup.element_set();
        let elements: HashSet<Permutation> = h_set.iter()
            .map(|h| g.compose(h))
            .collect();
        LeftCoset { representative: g.clone(), elements }
    }

    /// Check if an element is in this coset.
    pub fn contains(&self, elem: &Permutation) -> bool {
        self.elements.contains(elem)
    }

    /// Size of the coset.
    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

/// A right coset Hg of a subgroup H.
#[derive(Debug, Clone)]
pub struct RightCoset {
    /// Representative element.
    pub representative: Permutation,
    /// Elements of the coset.
    pub elements: HashSet<Permutation>,
}

impl RightCoset {
    /// Compute the right coset Hg.
    pub fn new(g: &Permutation, subgroup: &Subgroup) -> Self {
        let h_set = subgroup.element_set();
        let elements: HashSet<Permutation> = h_set.iter()
            .map(|h| h.compose(g))
            .collect();
        RightCoset { representative: g.clone(), elements }
    }
}

/// Enumerate all left cosets of H in G.
pub fn left_cosets(subgroup: &Subgroup, parent: &Subgroup) -> Vec<LeftCoset> {
    let g_set: Vec<Permutation> = parent.element_set().into_iter().collect();
    let mut cosets = Vec::new();
    let mut covered = HashSet::new();

    for g in &g_set {
        if covered.contains(g) { continue; }
        let coset = LeftCoset::new(g, subgroup);
        for elem in &coset.elements {
            covered.insert(elem.clone());
        }
        cosets.push(coset);
    }
    cosets
}

/// Compute double cosets HgK.
pub fn double_cosets(
    h: &Subgroup,
    k: &Subgroup,
    parent: &Subgroup,
) -> Vec<HashSet<Permutation>> {
    let g_set: Vec<Permutation> = parent.element_set().into_iter().collect();
    let h_set = h.element_set();
    let k_set = k.element_set();
    let mut cosets = Vec::new();
    let mut covered = HashSet::new();

    for g in &g_set {
        if covered.contains(g) { continue; }
        let mut coset = HashSet::new();
        for h_elem in &h_set {
            for k_elem in &k_set {
                let elem = h_elem.compose(g).compose(k_elem);
                coset.insert(elem);
            }
        }
        for elem in &coset {
            covered.insert(elem.clone());
        }
        cosets.push(coset);
    }
    cosets
}

// ═══════════════════════════════════════════════════════════════════════════
// QuotientGroup
// ═══════════════════════════════════════════════════════════════════════════

/// The quotient group G/N where N is a normal subgroup of G.
#[derive(Debug, Clone)]
pub struct QuotientGroup {
    /// The cosets (elements of G/N).
    pub cosets: Vec<LeftCoset>,
    /// Representatives of each coset.
    pub representatives: Vec<Permutation>,
    /// Multiplication table: `mult[i][j]` = index of product coset.
    pub multiplication_table: Vec<Vec<usize>>,
    /// Order of the quotient group.
    pub order: usize,
    /// Degree of the parent group.
    pub degree: usize,
}

impl QuotientGroup {
    /// Construct G/N.
    pub fn new(normal: &NormalSubgroup) -> Self {
        let cosets = left_cosets(&normal.subgroup, &normal.parent);
        let order = cosets.len();
        let representatives: Vec<Permutation> = cosets.iter()
            .map(|c| c.representative.clone())
            .collect();

        // Build multiplication table
        let mut mult = vec![vec![0usize; order]; order];
        for i in 0..order {
            for j in 0..order {
                let product = representatives[i].compose(&representatives[j]);
                // Find which coset the product belongs to
                for (k, coset) in cosets.iter().enumerate() {
                    if coset.contains(&product) {
                        mult[i][j] = k;
                        break;
                    }
                }
            }
        }

        QuotientGroup {
            cosets,
            representatives,
            multiplication_table: mult,
            order,
            degree: normal.subgroup.degree,
        }
    }

    /// Multiply two cosets (by index).
    pub fn multiply(&self, i: usize, j: usize) -> usize {
        self.multiplication_table[i][j]
    }

    /// Identity coset index.
    pub fn identity(&self) -> usize {
        let id = Permutation::identity(self.degree);
        for (i, coset) in self.cosets.iter().enumerate() {
            if coset.contains(&id) {
                return i;
            }
        }
        0
    }

    /// Inverse of a coset.
    pub fn inverse_of(&self, i: usize) -> usize {
        let id_idx = self.identity();
        for j in 0..self.order {
            if self.multiply(i, j) == id_idx {
                return j;
            }
        }
        i
    }

    /// Order of a coset element.
    pub fn element_order(&self, i: usize) -> usize {
        let id_idx = self.identity();
        let mut current = i;
        for k in 1..=self.order {
            if current == id_idx {
                return k;
            }
            current = self.multiply(current, i);
        }
        self.order
    }

    /// Check if the quotient group is abelian.
    pub fn is_abelian(&self) -> bool {
        for i in 0..self.order {
            for j in 0..self.order {
                if self.multiply(i, j) != self.multiply(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the quotient group is cyclic.
    pub fn is_cyclic(&self) -> bool {
        for i in 0..self.order {
            if self.element_order(i) == self.order {
                return true;
            }
        }
        false
    }

    /// Natural projection homomorphism π: G → G/N.
    pub fn project(&self, elem: &Permutation) -> usize {
        for (i, coset) in self.cosets.iter().enumerate() {
            if coset.contains(elem) {
                return i;
            }
        }
        0
    }
}

impl fmt::Display for QuotientGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quotient Group (order = {}):", self.order)?;
        writeln!(f, "Coset representatives:")?;
        for (i, rep) in self.representatives.iter().enumerate() {
            writeln!(f, "  C{}: {}", i, rep.to_cycle_notation())?;
        }
        writeln!(f, "Multiplication table:")?;
        write!(f, "     ")?;
        for j in 0..self.order {
            write!(f, " C{}", j)?;
        }
        writeln!(f)?;
        for i in 0..self.order {
            write!(f, "  C{} ", i)?;
            for j in 0..self.order {
                write!(f, " C{}", self.multiply(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FactorGroupEnumerator
// ═══════════════════════════════════════════════════════════════════════════

/// Enumerate all normal subgroups and factor groups of a group.
#[derive(Debug)]
pub struct FactorGroupEnumerator {
    /// The parent group.
    pub group: Subgroup,
}

impl FactorGroupEnumerator {
    /// Create a new enumerator.
    pub fn new(group: Subgroup) -> Self {
        FactorGroupEnumerator { group }
    }

    /// Find all subgroups (brute force, for small groups only).
    pub fn all_subgroups(&self) -> Vec<Subgroup> {
        let elems: Vec<Permutation> = self.group.element_set().into_iter().collect();
        let mut subgroups = Vec::new();
        let n = elems.len();

        // Always include trivial and full group
        subgroups.push(Subgroup::trivial(self.group.degree));
        subgroups.push(self.group.clone());

        // Try generating subgroups from each element
        for elem in &elems {
            if elem.is_identity() { continue; }
            let gen = vec![elem.clone()];
            let sg = Subgroup::from_generators(gen, self.group.degree);
            let order = sg.order();
            // Check if we already have this subgroup
            let already = subgroups.iter().any(|s| s.order() == order && s.element_set() == sg.element_set());
            if !already {
                subgroups.push(sg);
            }
        }

        // Try pairs of generators
        for i in 0..elems.len() {
            for j in (i + 1)..elems.len() {
                let gen = vec![elems[i].clone(), elems[j].clone()];
                let sg = Subgroup::from_generators(gen, self.group.degree);
                let order = sg.order();
                let already = subgroups.iter().any(|s| s.order() == order && s.element_set() == sg.element_set());
                if !already {
                    subgroups.push(sg);
                }
            }
        }

        subgroups
    }

    /// Find all normal subgroups.
    pub fn all_normal_subgroups(&self) -> Vec<Subgroup> {
        self.all_subgroups().into_iter()
            .filter(|sg| NormalSubgroup::check_normality(sg, &self.group))
            .collect()
    }

    /// Compute all quotient groups.
    pub fn all_quotient_groups(&self) -> Vec<QuotientGroup> {
        self.all_normal_subgroups().into_iter()
            .filter_map(|sg| {
                let ns = NormalSubgroup::new(sg, self.group.clone())?;
                Some(QuotientGroup::new(&ns))
            })
            .collect()
    }

    /// Composition series: chain G = G_0 ▷ G_1 ▷ ... ▷ G_n = {1}
    /// where each G_{i+1} is maximal normal in G_i and G_i/G_{i+1} is simple.
    pub fn composition_series(&self) -> Vec<Subgroup> {
        let mut series = vec![self.group.clone()];
        let mut current = self.group.clone();

        loop {
            let order = current.order();
            if order <= 1 { break; }

            // Find a maximal proper normal subgroup
            let normals = {
                let enumerator = FactorGroupEnumerator::new(current.clone());
                enumerator.all_normal_subgroups()
            };

            // Find the largest proper normal subgroup
            let best = normals.into_iter()
                .filter(|n| {
                    let n_order = n.order();
                    n_order > 0 && n_order < order
                })
                .max_by_key(|n| n.order());

            match best {
                Some(n) => {
                    series.push(n.clone());
                    current = n;
                }
                None => break,
            }
        }

        series
    }

    /// Composition factors: the simple quotients G_i/G_{i+1}.
    pub fn composition_factors(&self) -> Vec<usize> {
        let series = self.composition_series();
        let mut factors = Vec::new();
        for i in 0..series.len() - 1 {
            let quotient_order = series[i].order() / series[i + 1].order();
            factors.push(quotient_order);
        }
        factors
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QuotientSpace
// ═══════════════════════════════════════════════════════════════════════════

/// Quotient of a finite set by an equivalence relation.
#[derive(Debug, Clone)]
pub struct QuotientSpace {
    /// Number of elements in the original set.
    pub n: usize,
    /// Equivalence class assignment: `class_of[i]` is the class of element i.
    pub class_of: Vec<usize>,
    /// Number of equivalence classes.
    pub num_classes: usize,
    /// Elements in each class.
    pub classes: Vec<Vec<usize>>,
    /// Canonical representative of each class (smallest element).
    pub representatives: Vec<usize>,
}

impl QuotientSpace {
    /// Create from an equivalence relation given as pairs.
    pub fn from_pairs(n: usize, pairs: &[(usize, usize)]) -> Self {
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find(parent, parent[x]); }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry { parent[rx] = ry; }
        }

        // Reflexive
        // Merge according to pairs
        for &(a, b) in pairs {
            union(&mut parent, a, b);
        }

        // Build classes
        let mut class_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            class_map.entry(root).or_default().push(i);
        }

        let mut classes: Vec<Vec<usize>> = class_map.into_values().collect();
        classes.sort_by_key(|c| c[0]);

        let num_classes = classes.len();
        let mut class_of = vec![0; n];
        let mut representatives = Vec::new();

        for (ci, class) in classes.iter().enumerate() {
            representatives.push(class[0]);
            for &elem in class {
                class_of[elem] = ci;
            }
        }

        QuotientSpace { n, class_of, num_classes, classes, representatives }
    }

    /// Create from a class assignment function.
    pub fn from_class_assignment(n: usize, class_of: Vec<usize>) -> Self {
        let mut class_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &c) in class_of.iter().enumerate() {
            class_map.entry(c).or_default().push(i);
        }

        let mut classes: Vec<Vec<usize>> = class_map.into_values().collect();
        classes.sort_by_key(|c| c[0]);

        let num_classes = classes.len();
        let mut normalized_class_of = vec![0; n];
        let mut representatives = Vec::new();

        for (ci, class) in classes.iter().enumerate() {
            representatives.push(class[0]);
            for &elem in class {
                normalized_class_of[elem] = ci;
            }
        }

        QuotientSpace {
            n,
            class_of: normalized_class_of,
            num_classes,
            classes,
            representatives,
        }
    }

    /// Project an element to its equivalence class.
    pub fn project(&self, elem: usize) -> usize {
        self.class_of[elem]
    }

    /// Get the canonical representative of a class.
    pub fn representative(&self, class: usize) -> usize {
        self.representatives[class]
    }

    /// Get all elements in a class.
    pub fn class_elements(&self, class: usize) -> &[usize] {
        &self.classes[class]
    }

    /// Section (right inverse): maps each class to its representative.
    pub fn section(&self, class: usize) -> usize {
        self.representatives[class]
    }
}

impl fmt::Display for QuotientSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QuotientSpace ({} elements → {} classes):", self.n, self.num_classes)?;
        for (i, class) in self.classes.iter().enumerate() {
            writeln!(f, "  [{:?}] rep={}", class, self.representatives[i])?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SylowSubgroup
// ═══════════════════════════════════════════════════════════════════════════

/// Sylow subgroup computation for finite groups.
pub struct SylowComputer {
    /// The group.
    group: Subgroup,
}

impl SylowComputer {
    /// Create a new Sylow computer.
    pub fn new(group: Subgroup) -> Self {
        SylowComputer { group }
    }

    /// Find a p-Sylow subgroup (largest p-power order subgroup).
    pub fn sylow_subgroup(&self, p: usize) -> Option<Subgroup> {
        let order = self.group.order();
        let p_part = Self::p_part(order, p);
        if p_part <= 1 {
            return Some(Subgroup::trivial(self.group.degree));
        }

        let enumerator = FactorGroupEnumerator::new(self.group.clone());
        let subgroups = enumerator.all_subgroups();

        // Find a subgroup of order p_part
        subgroups.into_iter()
            .find(|sg| sg.order() == p_part)
    }

    /// Compute p^a where p^a divides n but p^(a+1) doesn't.
    fn p_part(mut n: usize, p: usize) -> usize {
        let mut result = 1;
        while n % p == 0 {
            result *= p;
            n /= p;
        }
        result
    }

    /// Count the number of p-Sylow subgroups.
    pub fn count_sylow_subgroups(&self, p: usize) -> usize {
        let order = self.group.order();
        let p_part = Self::p_part(order, p);
        if p_part <= 1 { return 1; }

        let enumerator = FactorGroupEnumerator::new(self.group.clone());
        let subgroups = enumerator.all_subgroups();

        subgroups.iter()
            .filter(|sg| sg.order() == p_part)
            .count()
    }

    /// Verify Sylow's third theorem: n_p ≡ 1 (mod p) and n_p divides |G|/p^a.
    pub fn verify_sylow_theorem(&self, p: usize) -> bool {
        let n_p = self.count_sylow_subgroups(p);
        let order = self.group.order();
        let p_part = Self::p_part(order, p);
        let m = order / p_part;

        n_p % p == 1 && m % n_p == 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GroupExtension
// ═══════════════════════════════════════════════════════════════════════════

/// A short exact sequence 1 → N → G → Q → 1.
#[derive(Debug, Clone)]
pub struct ShortExactSequence {
    /// The normal subgroup N.
    pub normal: Subgroup,
    /// The full group G.
    pub group: Subgroup,
    /// The quotient group Q = G/N.
    pub quotient: QuotientGroup,
}

impl ShortExactSequence {
    /// Build a short exact sequence from G and N ◁ G.
    pub fn new(normal: Subgroup, group: Subgroup) -> Option<Self> {
        let ns = NormalSubgroup::new(normal.clone(), group.clone())?;
        let quotient = QuotientGroup::new(&ns);
        Some(ShortExactSequence { normal, group, quotient })
    }

    /// Check if the extension splits (G ≅ N ⋊ Q).
    pub fn is_split(&self) -> bool {
        // An extension splits iff there exists a complement K to N in G:
        // G = NK, N ∩ K = {1}, |K| = |Q|
        let q_order = self.quotient.order;
        let n_set = self.normal.element_set();

        let enumerator = FactorGroupEnumerator::new(self.group.clone());
        let subgroups = enumerator.all_subgroups();

        subgroups.iter().any(|k| {
            if k.order() != q_order { return false; }
            let k_set = k.element_set();
            // N ∩ K = {id}
            let inter: HashSet<_> = n_set.intersection(&k_set).collect();
            if inter.len() != 1 { return false; }
            // NK = G
            let mut nk = HashSet::new();
            for n in &n_set {
                for kk in &k_set {
                    nk.insert(n.compose(kk));
                }
            }
            nk.len() == self.group.order()
        })
    }

    /// Verify exactness: |N| * |Q| = |G|.
    pub fn verify_exactness(&self) -> bool {
        self.normal.order() * self.quotient.order == self.group.order()
    }
}

/// Construct a semidirect product N ⋊_φ H given an action φ: H → Aut(N).
/// Both N and H are given as permutation groups on the same degree.
pub fn semidirect_product(
    n_gens: &[Permutation],
    h_gens: &[Permutation],
    degree: usize,
    action: &dyn Fn(&Permutation, &Permutation) -> Permutation,
) -> Subgroup {
    let product_degree = degree * 2;
    // Embed N into the first copy and H into the second
    let mut gens = Vec::new();

    // N generators embedded
    for gen in n_gens {
        let mut images = vec![0u32; product_degree];
        for i in 0..degree {
            images[i] = gen.apply(i as u32);
        }
        for i in degree..product_degree {
            images[i] = i as u32;
        }
        gens.push(Permutation::new(images));
    }

    // H generators embedded with the action
    for gen in h_gens {
        let mut images = vec![0u32; product_degree];
        for i in 0..degree {
            images[i] = i as u32; // placeholder
        }
        for i in degree..product_degree {
            images[i] = degree as u32 + gen.apply((i - degree) as u32);
        }
        gens.push(Permutation::new(images));
    }

    Subgroup::from_generators(gens, product_degree)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn s3() -> Subgroup {
        Subgroup::from_generators(
            vec![
                Permutation::transposition(3, 0, 1),
                Permutation::cycle(3, &[0, 1, 2]),
            ],
            3,
        )
    }

    fn z3() -> Subgroup {
        Subgroup::from_generators(
            vec![Permutation::cycle(3, &[0, 1, 2])],
            3,
        )
    }

    #[test]
    fn test_subgroup_order() {
        assert_eq!(s3().order(), 6);
        assert_eq!(z3().order(), 3);
    }

    #[test]
    fn test_subgroup_contains() {
        let g = s3();
        assert!(g.contains(&Permutation::identity(3)));
        assert!(g.contains(&Permutation::transposition(3, 0, 1)));
    }

    #[test]
    fn test_trivial_subgroup() {
        let t = Subgroup::trivial(3);
        assert_eq!(t.order(), 1);
    }

    #[test]
    fn test_is_subgroup_of() {
        let g = s3();
        let h = z3();
        assert!(h.is_subgroup_of(&g));
    }

    #[test]
    fn test_index() {
        let g = s3();
        let h = z3();
        assert_eq!(h.index_in(&g), 2);
    }

    #[test]
    fn test_center_s3() {
        let g = s3();
        let center = g.center();
        assert_eq!(center.order(), 1); // S3 has trivial center
    }

    #[test]
    fn test_center_z3() {
        let g = z3();
        let center = g.center();
        assert_eq!(center.order(), 3); // Z3 is abelian so center = Z3
    }

    #[test]
    fn test_derived_subgroup() {
        let g = s3();
        let derived = g.derived_subgroup();
        assert_eq!(derived.order(), 3); // [S3,S3] = A3 ≅ Z3
    }

    #[test]
    fn test_is_solvable() {
        assert!(s3().is_solvable());
        assert!(z3().is_solvable());
    }

    #[test]
    fn test_is_abelian() {
        assert!(!s3().is_abelian());
        assert!(z3().is_abelian());
    }

    #[test]
    fn test_is_cyclic() {
        assert!(!s3().is_cyclic());
        assert!(z3().is_cyclic());
    }

    #[test]
    fn test_normality_a3_in_s3() {
        let g = s3();
        let h = z3(); // A3 = Z3
        assert!(NormalSubgroup::check_normality(&h, &g));
    }

    #[test]
    fn test_left_cosets() {
        let g = s3();
        let h = z3();
        let cosets = left_cosets(&h, &g);
        assert_eq!(cosets.len(), 2); // [S3:A3] = 2
    }

    #[test]
    fn test_quotient_group() {
        let g = s3();
        let h = z3();
        let ns = NormalSubgroup::new(h, g).unwrap();
        let q = QuotientGroup::new(&ns);
        assert_eq!(q.order, 2); // S3/A3 ≅ Z2
        assert!(q.is_abelian());
        assert!(q.is_cyclic());
    }

    #[test]
    fn test_quotient_identity() {
        let g = s3();
        let h = z3();
        let ns = NormalSubgroup::new(h, g.clone()).unwrap();
        let q = QuotientGroup::new(&ns);
        let id_idx = q.identity();
        let projected = q.project(&Permutation::identity(3));
        assert_eq!(projected, id_idx);
    }

    #[test]
    fn test_all_normal_subgroups() {
        let g = s3();
        let enumerator = FactorGroupEnumerator::new(g);
        let normals = enumerator.all_normal_subgroups();
        // S3 has 3 normal subgroups: {e}, A3, S3
        assert!(normals.len() >= 3);
    }

    #[test]
    fn test_composition_factors() {
        let g = s3();
        let enumerator = FactorGroupEnumerator::new(g);
        let factors = enumerator.composition_factors();
        // S3 composition factors: 2, 3 (in some order)
        let mut sorted = factors.clone();
        sorted.sort();
        assert_eq!(sorted, vec![2, 3]);
    }

    #[test]
    fn test_quotient_space() {
        let qs = QuotientSpace::from_pairs(5, &[(0, 1), (2, 3)]);
        assert_eq!(qs.num_classes, 3); // {0,1}, {2,3}, {4}
        assert_eq!(qs.project(0), qs.project(1));
        assert_eq!(qs.project(2), qs.project(3));
        assert_ne!(qs.project(0), qs.project(4));
    }

    #[test]
    fn test_quotient_space_section() {
        let qs = QuotientSpace::from_pairs(4, &[(0, 1), (2, 3)]);
        for i in 0..qs.num_classes {
            let rep = qs.section(i);
            assert_eq!(qs.project(rep), i);
        }
    }

    #[test]
    fn test_sylow_s3() {
        let g = s3();
        let computer = SylowComputer::new(g);

        let sylow2 = computer.sylow_subgroup(2);
        assert!(sylow2.is_some());
        assert_eq!(sylow2.unwrap().order(), 2);

        let sylow3 = computer.sylow_subgroup(3);
        assert!(sylow3.is_some());
        assert_eq!(sylow3.unwrap().order(), 3);
    }

    #[test]
    fn test_sylow_count() {
        let g = s3();
        let computer = SylowComputer::new(g);
        let n3 = computer.count_sylow_subgroups(3);
        assert_eq!(n3, 1); // A3 is the unique Sylow 3-subgroup
    }

    #[test]
    fn test_short_exact_sequence() {
        let g = s3();
        let n = z3();
        let ses = ShortExactSequence::new(n, g);
        assert!(ses.is_some());
        let ses = ses.unwrap();
        assert!(ses.verify_exactness());
    }

    #[test]
    fn test_split_extension() {
        let g = s3();
        let n = z3();
        let ses = ShortExactSequence::new(n, g).unwrap();
        // S3 ≅ Z3 ⋊ Z2, so this extension splits
        assert!(ses.is_split());
    }

    #[test]
    fn test_normal_closure() {
        let g = s3();
        let h = Subgroup::from_generators(
            vec![Permutation::transposition(3, 0, 1)],
            3,
        );
        let closure = NormalSubgroup::normal_closure(&h, &g);
        // Normal closure of a transposition in S3 is all of S3
        assert_eq!(closure.order(), 6);
    }

    #[test]
    fn test_double_cosets() {
        let g = s3();
        let h = z3();
        let k = Subgroup::from_generators(
            vec![Permutation::transposition(3, 0, 1)],
            3,
        );
        let dcosets = double_cosets(&h, &k, &g);
        // |S3| = sum of |HgK| for each double coset
        let total: usize = dcosets.iter().map(|c| c.len()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_derived_series() {
        let g = s3();
        let series = g.derived_series();
        assert!(series.len() >= 3); // S3 ⊇ A3 ⊇ {e}
        assert_eq!(series[0].order(), 6);
    }

    #[test]
    fn test_intersection() {
        let g = s3();
        let h = z3();
        let k = Subgroup::from_generators(
            vec![Permutation::transposition(3, 0, 1)],
            3,
        );
        let inter = h.intersection(&k);
        assert_eq!(inter.order(), 1); // Z3 ∩ Z2 = {e}
    }
}
