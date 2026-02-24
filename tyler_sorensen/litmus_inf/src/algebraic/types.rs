// LITMUS∞ Algebraic Engine — Core Algebraic Types
//
//   Permutation   – vector mapping backed by compact storage
//   PermutationGroup (stub, full impl in group.rs)
//   GroupElement operations (compose, inverse, identity)
//   Orbit of an element under a group action
//   Coset representation
//   GroupAction<T> trait

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};

// ── Permutation ──────────────────────────────────────────────────────

/// A permutation on {0, 1, …, n-1} stored as a vector mapping.
/// `images[i]` is the image of `i` under the permutation.
#[derive(Clone, Eq)]
pub struct Permutation {
    images: Vec<u32>,
}

impl Permutation {
    /// Create a new permutation from a vector of images.
    /// `images[i]` is the image of element `i`.
    /// Panics if the images do not form a valid permutation.
    pub fn new(images: Vec<u32>) -> Self {
        debug_assert!(Self::is_valid_perm(&images), "Invalid permutation");
        Permutation { images }
    }

    /// Create a permutation from a slice of images.
    pub fn from_slice(images: &[u32]) -> Self {
        Self::new(images.to_vec())
    }

    /// Try to create a permutation, returning None if invalid.
    pub fn try_new(images: Vec<u32>) -> Option<Self> {
        if Self::is_valid_perm(&images) {
            Some(Permutation { images })
        } else {
            None
        }
    }

    /// The identity permutation on n elements.
    pub fn identity(n: usize) -> Self {
        Permutation {
            images: (0..n as u32).collect(),
        }
    }

    /// The degree (number of elements permuted).
    pub fn degree(&self) -> usize {
        self.images.len()
    }

    /// Image of element `i` under this permutation.
    #[inline]
    pub fn apply(&self, i: u32) -> u32 {
        self.images[i as usize]
    }

    /// Image of element `i`, with bounds checking returning None.
    pub fn try_apply(&self, i: u32) -> Option<u32> {
        self.images.get(i as usize).copied()
    }

    /// Apply this permutation to a vector (permute the elements).
    pub fn apply_to_vec<T: Clone>(&self, v: &[T]) -> Vec<T> {
        assert_eq!(v.len(), self.degree());
        let mut result = v.to_vec();
        for i in 0..self.degree() {
            result[self.images[i] as usize] = v[i].clone();
        }
        result
    }

    /// Compose: self ∘ other  (apply other first, then self).
    pub fn compose(&self, other: &Permutation) -> Permutation {
        assert_eq!(self.degree(), other.degree());
        let n = self.degree();
        let mut images = vec![0u32; n];
        for i in 0..n {
            images[i] = self.images[other.images[i] as usize];
        }
        Permutation { images }
    }

    /// Inverse permutation.
    pub fn inverse(&self) -> Permutation {
        let n = self.degree();
        let mut inv = vec![0u32; n];
        for i in 0..n {
            inv[self.images[i] as usize] = i as u32;
        }
        Permutation { images: inv }
    }

    /// Check whether this is the identity.
    pub fn is_identity(&self) -> bool {
        self.images.iter().enumerate().all(|(i, &v)| v == i as u32)
    }

    /// The order of this permutation (smallest k > 0 s.t. self^k = id).
    pub fn order(&self) -> u64 {
        // Order = lcm of cycle lengths.
        let cycles = self.cycle_decomposition();
        let mut ord: u64 = 1;
        for c in &cycles {
            let len = c.len() as u64;
            ord = lcm(ord, len);
        }
        ord
    }

    /// Power: self^k.
    pub fn pow(&self, mut k: i64) -> Permutation {
        let n = self.degree();
        if k == 0 {
            return Permutation::identity(n);
        }
        let base = if k < 0 {
            k = -k;
            self.inverse()
        } else {
            self.clone()
        };
        let mut result = Permutation::identity(n);
        let mut acc = base;
        let mut k = k as u64;
        while k > 0 {
            if k & 1 == 1 {
                result = result.compose(&acc);
            }
            acc = acc.compose(&acc);
            k >>= 1;
        }
        result
    }

    /// Cycle decomposition: returns a vector of cycles, each cycle
    /// is a vector of elements. Fixed points are returned as 1-cycles.
    pub fn cycle_decomposition(&self) -> Vec<Vec<u32>> {
        let n = self.degree();
        let mut visited = vec![false; n];
        let mut cycles = Vec::new();
        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut j = i as u32;
            loop {
                visited[j as usize] = true;
                cycle.push(j);
                j = self.images[j as usize];
                if j == i as u32 {
                    break;
                }
            }
            cycles.push(cycle);
        }
        cycles
    }

    /// Cycle type: sorted vector of cycle lengths (descending).
    pub fn cycle_type(&self) -> Vec<usize> {
        let mut lengths: Vec<usize> = self
            .cycle_decomposition()
            .iter()
            .map(|c| c.len())
            .collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths
    }

    /// The set of fixed points.
    pub fn fixed_points(&self) -> Vec<u32> {
        (0..self.degree() as u32)
            .filter(|&i| self.images[i as usize] == i)
            .collect()
    }

    /// The support: elements that are moved.
    pub fn support(&self) -> Vec<u32> {
        (0..self.degree() as u32)
            .filter(|&i| self.images[i as usize] != i)
            .collect()
    }

    /// Parity: true = even, false = odd.
    pub fn is_even(&self) -> bool {
        let cycles = self.cycle_decomposition();
        let transposition_count: usize = cycles.iter().map(|c| c.len() - 1).sum();
        transposition_count % 2 == 0
    }

    /// Sign: +1 for even, -1 for odd.
    pub fn sign(&self) -> i32 {
        if self.is_even() {
            1
        } else {
            -1
        }
    }

    /// Create a transposition (i j) on n elements.
    pub fn transposition(n: usize, i: u32, j: u32) -> Self {
        assert!((i as usize) < n && (j as usize) < n);
        let mut images: Vec<u32> = (0..n as u32).collect();
        images[i as usize] = j;
        images[j as usize] = i;
        Permutation { images }
    }

    /// Create a k-cycle on n elements.
    pub fn cycle(n: usize, elements: &[u32]) -> Self {
        assert!(elements.iter().all(|&e| (e as usize) < n));
        let mut images: Vec<u32> = (0..n as u32).collect();
        if elements.is_empty() {
            return Permutation { images };
        }
        for i in 0..elements.len() - 1 {
            images[elements[i] as usize] = elements[i + 1];
        }
        images[elements[elements.len() - 1] as usize] = elements[0];
        Permutation { images }
    }

    /// Conjugation: other^{-1} · self · other.
    pub fn conjugate_by(&self, other: &Permutation) -> Permutation {
        other.inverse().compose(self).compose(other)
    }

    /// Commutator: [self, other] = self^{-1} · other^{-1} · self · other.
    pub fn commutator(&self, other: &Permutation) -> Permutation {
        self.inverse()
            .compose(&other.inverse())
            .compose(self)
            .compose(other)
    }

    /// The raw images slice.
    pub fn images(&self) -> &[u32] {
        &self.images
    }

    /// Extend this permutation to act on n elements (n >= current degree).
    /// New elements are fixed points.
    pub fn extend_to(&self, n: usize) -> Permutation {
        assert!(n >= self.degree());
        let mut images = self.images.clone();
        for i in self.degree()..n {
            images.push(i as u32);
        }
        Permutation { images }
    }

    /// Check validity of a permutation mapping.
    fn is_valid_perm(images: &[u32]) -> bool {
        let n = images.len();
        let mut seen = vec![false; n];
        for &v in images {
            if v as usize >= n {
                return false;
            }
            if seen[v as usize] {
                return false;
            }
            seen[v as usize] = true;
        }
        true
    }

    /// Cycle notation string (e.g. "(0 1 2)(3 4)").
    pub fn to_cycle_notation(&self) -> String {
        let cycles = self.cycle_decomposition();
        let non_trivial: Vec<_> = cycles.iter().filter(|c| c.len() > 1).collect();
        if non_trivial.is_empty() {
            return "()".to_string();
        }
        let mut s = String::new();
        for c in &non_trivial {
            s.push('(');
            for (i, &e) in c.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push_str(&e.to_string());
            }
            s.push(')');
        }
        s
    }
}

impl PartialEq for Permutation {
    fn eq(&self, other: &Self) -> bool {
        self.images == other.images
    }
}

impl Hash for Permutation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.images.hash(state);
    }
}

impl fmt::Debug for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Perm{}", self.to_cycle_notation())
    }
}

impl fmt::Display for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_cycle_notation())
    }
}

// ── GroupElement ──────────────────────────────────────────────────────

/// Trait for group element operations.
pub trait GroupElement: Clone + Eq + Hash {
    /// The identity element.
    fn identity(context: &GroupContext) -> Self;

    /// Compose self with other (group operation).
    fn compose_with(&self, other: &Self) -> Self;

    /// Inverse element.
    fn invert(&self) -> Self;

    /// Check if this is the identity.
    fn is_identity_element(&self) -> bool;

    /// Order of this element.
    fn element_order(&self) -> u64;
}

/// Context for group construction (e.g., degree for permutation groups).
#[derive(Clone, Debug)]
pub struct GroupContext {
    pub degree: usize,
}

impl GroupElement for Permutation {
    fn identity(ctx: &GroupContext) -> Self {
        Permutation::identity(ctx.degree)
    }

    fn compose_with(&self, other: &Self) -> Self {
        self.compose(other)
    }

    fn invert(&self) -> Self {
        self.inverse()
    }

    fn is_identity_element(&self) -> bool {
        self.is_identity()
    }

    fn element_order(&self) -> u64 {
        self.order()
    }
}

// ── Orbit ────────────────────────────────────────────────────────────

/// The orbit of an element under a group action.
#[derive(Clone, Debug)]
pub struct Orbit<T: Clone + Eq + Hash> {
    /// Representative element (the seed).
    pub representative: T,
    /// All elements in the orbit.
    pub elements: HashSet<T>,
    /// Transversal: for each element in the orbit, a group element
    /// mapping the representative to that element.
    pub transversal: HashMap<T, Permutation>,
}

impl<T: Clone + Eq + Hash> Orbit<T> {
    /// Size of the orbit.
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Check if an element is in this orbit.
    pub fn contains(&self, elem: &T) -> bool {
        self.elements.contains(elem)
    }

    /// Get the transversal element that maps rep -> elem.
    pub fn transversal_element(&self, elem: &T) -> Option<&Permutation> {
        self.transversal.get(elem)
    }
}

/// Compute the orbit of `seed` under a group given by generators,
/// using the action function.
pub fn compute_orbit<T, F>(
    seed: T,
    generators: &[Permutation],
    action: F,
    degree: usize,
) -> Orbit<T>
where
    T: Clone + Eq + Hash,
    F: Fn(&T, &Permutation) -> T,
{
    let mut elements = HashSet::new();
    let mut transversal = HashMap::new();
    let mut queue = VecDeque::new();

    elements.insert(seed.clone());
    transversal.insert(seed.clone(), Permutation::identity(degree));
    queue.push_back(seed.clone());

    while let Some(current) = queue.pop_front() {
        let current_perm = transversal[&current].clone();
        for gen in generators {
            let next = action(&current, gen);
            if !elements.contains(&next) {
                elements.insert(next.clone());
                transversal.insert(next.clone(), current_perm.compose(gen));
                queue.push_back(next);
            }
        }
    }

    Orbit {
        representative: seed,
        elements,
        transversal,
    }
}

/// Compute the orbit of a point (u32) under a permutation group.
pub fn compute_point_orbit(point: u32, generators: &[Permutation], degree: usize) -> Orbit<u32> {
    compute_orbit(point, generators, |&p, g| g.apply(p), degree)
}

// ── Coset ────────────────────────────────────────────────────────────

/// A coset gH of a subgroup H in a group G.
#[derive(Clone, Debug)]
pub struct Coset {
    /// The coset representative.
    pub representative: Permutation,
    /// The subgroup H (stored as generators).
    pub subgroup_generators: Vec<Permutation>,
    /// The degree.
    pub degree: usize,
}

impl Coset {
    /// Create a new coset.
    pub fn new(representative: Permutation, subgroup_generators: Vec<Permutation>) -> Self {
        let degree = representative.degree();
        Coset {
            representative,
            subgroup_generators,
            degree,
        }
    }

    /// Check if an element belongs to this coset (requires membership testing).
    pub fn contains(&self, element: &Permutation) -> bool {
        // g^{-1} · element should be in the subgroup
        let diff = self.representative.inverse().compose(element);
        // Naive check: see if diff can be expressed as a product of generators
        // For correctness, this should use the Schreier-Sims membership test,
        // which is implemented in group.rs.
        if diff.is_identity() {
            return true;
        }
        // Fallback: enumerate small subgroups
        let elements = enumerate_from_generators(&self.subgroup_generators, self.degree);
        elements.contains(&diff)
    }

    /// Enumerate all elements of this coset (for small subgroups).
    pub fn elements(&self) -> HashSet<Permutation> {
        let subgroup = enumerate_from_generators(&self.subgroup_generators, self.degree);
        subgroup
            .into_iter()
            .map(|h| self.representative.compose(&h))
            .collect()
    }
}

/// Enumerate all elements generated by a set of generators (for small groups).
pub fn enumerate_from_generators(generators: &[Permutation], degree: usize) -> HashSet<Permutation> {
    let mut elements = HashSet::new();
    let id = Permutation::identity(degree);
    elements.insert(id.clone());
    let mut queue = VecDeque::new();
    queue.push_back(id);

    while let Some(current) = queue.pop_front() {
        for gen in generators {
            let product = current.compose(gen);
            if !elements.contains(&product) {
                elements.insert(product.clone());
                queue.push_back(product);
            }
            let product_inv = current.compose(&gen.inverse());
            if !elements.contains(&product_inv) {
                elements.insert(product_inv.clone());
                queue.push_back(product_inv);
            }
        }
    }

    elements
}

// ── GroupAction trait ─────────────────────────────────────────────────

/// A group action of a permutation group on a set of type T.
pub trait GroupAction<T: Clone + Eq + Hash> {
    /// Apply a group element to an object.
    fn act(&self, element: &Permutation, object: &T) -> T;

    /// Compute the orbit of an object under this action.
    fn orbit(&self, object: &T, generators: &[Permutation], degree: usize) -> Orbit<T> {
        compute_orbit(object.clone(), generators, |o, g| self.act(g, o), degree)
    }

    /// Compute the stabilizer generators of an object (naive approach).
    fn stabilizer_generators(
        &self,
        object: &T,
        group_elements: &[Permutation],
    ) -> Vec<Permutation> {
        group_elements
            .iter()
            .filter(|g| self.act(g, object) == *object)
            .cloned()
            .collect()
    }

    /// Check if two objects are in the same orbit.
    fn same_orbit(
        &self,
        a: &T,
        b: &T,
        generators: &[Permutation],
        degree: usize,
    ) -> bool {
        let orbit_a = self.orbit(a, generators, degree);
        orbit_a.contains(b)
    }
}

/// Natural action of S_n on {0, ..., n-1}.
pub struct NaturalAction;

impl GroupAction<u32> for NaturalAction {
    fn act(&self, element: &Permutation, &point: &u32) -> u32 {
        element.apply(point)
    }
}

/// Action on sets: permutation acts on subsets of {0, ..., n-1}.
pub struct SetAction;

impl GroupAction<Vec<u32>> for SetAction {
    fn act(&self, element: &Permutation, set: &Vec<u32>) -> Vec<u32> {
        let mut result: Vec<u32> = set.iter().map(|&x| element.apply(x)).collect();
        result.sort_unstable();
        result
    }
}

/// Action on tuples: permutation acts on ordered tuples.
pub struct TupleAction;

impl GroupAction<Vec<u32>> for TupleAction {
    fn act(&self, element: &Permutation, tuple: &Vec<u32>) -> Vec<u32> {
        tuple.iter().map(|&x| element.apply(x)).collect()
    }
}

// ── BitPacked Permutation ────────────────────────────────────────────

/// Bit-packed permutation using bitvec for memory-efficient storage.
/// Each element uses ceil(log2(n)) bits.
#[derive(Clone)]
pub struct PackedPermutation {
    data: bitvec::vec::BitVec,
    degree: usize,
    bits_per_element: usize,
}

impl PackedPermutation {
    /// Create a packed permutation from a regular Permutation.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let degree = perm.degree();
        let bits_per_element = if degree <= 1 {
            1
        } else {
            (degree as f64).log2().ceil() as usize
        };
        let total_bits = degree * bits_per_element;
        let mut data = bitvec::vec::BitVec::with_capacity(total_bits);
        data.resize(total_bits, false);

        for i in 0..degree {
            let val = perm.apply(i as u32) as usize;
            for b in 0..bits_per_element {
                let bit = (val >> b) & 1 == 1;
                data.set(i * bits_per_element + b, bit);
            }
        }

        PackedPermutation {
            data,
            degree,
            bits_per_element,
        }
    }

    /// Convert back to a regular Permutation.
    pub fn to_permutation(&self) -> Permutation {
        let mut images = Vec::with_capacity(self.degree);
        for i in 0..self.degree {
            let mut val: u32 = 0;
            for b in 0..self.bits_per_element {
                if self.data[i * self.bits_per_element + b] {
                    val |= 1 << b;
                }
            }
            images.push(val);
        }
        Permutation::new(images)
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.data.len() + 7) / 8
    }

    /// Degree.
    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl fmt::Debug for PackedPermutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedPerm(n={}, {}B)", self.degree, self.memory_bytes())
    }
}

// ── Utility ──────────────────────────────────────────────────────────

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd(a, b) * b
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = Permutation::identity(5);
        assert!(id.is_identity());
        assert_eq!(id.degree(), 5);
        for i in 0..5 {
            assert_eq!(id.apply(i), i);
        }
    }

    #[test]
    fn test_transposition() {
        let t = Permutation::transposition(5, 1, 3);
        assert_eq!(t.apply(0), 0);
        assert_eq!(t.apply(1), 3);
        assert_eq!(t.apply(3), 1);
        assert_eq!(t.apply(2), 2);
        assert_eq!(t.apply(4), 4);
        assert_eq!(t.order(), 2);
        assert!(!t.is_even());
    }

    #[test]
    fn test_cycle() {
        let c = Permutation::cycle(5, &[0, 1, 2]);
        assert_eq!(c.apply(0), 1);
        assert_eq!(c.apply(1), 2);
        assert_eq!(c.apply(2), 0);
        assert_eq!(c.apply(3), 3);
        assert_eq!(c.order(), 3);
        assert!(c.is_even());
    }

    #[test]
    fn test_compose_identity() {
        let p = Permutation::cycle(4, &[0, 1, 2, 3]);
        let id = Permutation::identity(4);
        assert_eq!(p.compose(&id), p);
        assert_eq!(id.compose(&p), p);
    }

    #[test]
    fn test_inverse() {
        let p = Permutation::cycle(5, &[0, 2, 4]);
        let inv = p.inverse();
        let composed = p.compose(&inv);
        assert!(composed.is_identity());
        let composed2 = inv.compose(&p);
        assert!(composed2.is_identity());
    }

    #[test]
    fn test_compose_associativity() {
        let a = Permutation::cycle(4, &[0, 1, 2]);
        let b = Permutation::transposition(4, 2, 3);
        let c = Permutation::cycle(4, &[0, 3]);
        let ab_c = a.compose(&b).compose(&c);
        let a_bc = a.compose(&b.compose(&c));
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_order() {
        // (0 1)(2 3 4) has order lcm(2,3) = 6
        let p = Permutation::new(vec![1, 0, 3, 4, 2]);
        assert_eq!(p.order(), 6);
    }

    #[test]
    fn test_pow() {
        let p = Permutation::cycle(5, &[0, 1, 2]);
        assert_eq!(p.pow(3), Permutation::identity(5));
        assert_eq!(p.pow(-1), p.inverse());
        assert_eq!(p.pow(0), Permutation::identity(5));
        assert_eq!(p.pow(1), p);
    }

    #[test]
    fn test_cycle_decomposition() {
        let p = Permutation::new(vec![1, 0, 3, 4, 2]);
        let cycles = p.cycle_decomposition();
        let lengths: Vec<usize> = {
            let mut v: Vec<usize> = cycles.iter().map(|c| c.len()).collect();
            v.sort_unstable_by(|a, b| b.cmp(a));
            v
        };
        assert_eq!(lengths, vec![3, 2]);
    }

    #[test]
    fn test_cycle_type() {
        let p = Permutation::new(vec![1, 0, 3, 4, 2]);
        assert_eq!(p.cycle_type(), vec![3, 2]);
    }

    #[test]
    fn test_support_and_fixed() {
        let t = Permutation::transposition(5, 1, 3);
        let supp = t.support();
        let fixed = t.fixed_points();
        assert_eq!(supp.len(), 2);
        assert!(supp.contains(&1));
        assert!(supp.contains(&3));
        assert_eq!(fixed.len(), 3);
    }

    #[test]
    fn test_sign() {
        let id = Permutation::identity(4);
        assert_eq!(id.sign(), 1);
        let t = Permutation::transposition(4, 0, 1);
        assert_eq!(t.sign(), -1);
        let three_cycle = Permutation::cycle(4, &[0, 1, 2]);
        assert_eq!(three_cycle.sign(), 1); // even (2 transpositions)
    }

    #[test]
    fn test_conjugate() {
        let a = Permutation::cycle(4, &[0, 1, 2]);
        let b = Permutation::transposition(4, 0, 3);
        let conj = a.conjugate_by(&b);
        // Conjugation preserves cycle type
        assert_eq!(a.cycle_type(), conj.cycle_type());
    }

    #[test]
    fn test_commutator() {
        let a = Permutation::cycle(3, &[0, 1, 2]);
        let b = a.clone();
        let comm = a.commutator(&b);
        assert!(comm.is_identity()); // Element commutes with itself
    }

    #[test]
    fn test_orbit_point() {
        // S_3 generators: (0 1 2) and (0 1)
        let gens = vec![
            Permutation::cycle(3, &[0, 1, 2]),
            Permutation::transposition(3, 0, 1),
        ];
        let orbit = compute_point_orbit(0, &gens, 3);
        assert_eq!(orbit.size(), 3); // S_3 acts transitively
    }

    #[test]
    fn test_orbit_stabilizer_theorem() {
        // |G| = |Orbit(x)| * |Stab(x)|
        // S_3 has order 6, orbits of size 3, so stabilizer has order 2
        let gens = vec![
            Permutation::cycle(3, &[0, 1, 2]),
            Permutation::transposition(3, 0, 1),
        ];
        let all_elements = enumerate_from_generators(&gens, 3);
        let group_order = all_elements.len();
        assert_eq!(group_order, 6);

        let orbit = compute_point_orbit(0, &gens, 3);
        let stab: Vec<_> = all_elements
            .iter()
            .filter(|g| g.apply(0) == 0)
            .collect();
        assert_eq!(group_order, orbit.size() * stab.len());
    }

    #[test]
    fn test_natural_action() {
        let action = NaturalAction;
        let g = Permutation::cycle(4, &[0, 1, 2]);
        assert_eq!(action.act(&g, &0), 1);
        assert_eq!(action.act(&g, &1), 2);
        assert_eq!(action.act(&g, &2), 0);
        assert_eq!(action.act(&g, &3), 3);
    }

    #[test]
    fn test_set_action() {
        let action = SetAction;
        let g = Permutation::cycle(5, &[0, 1, 2]);
        let set = vec![0, 2, 4];
        let result = action.act(&g, &set);
        assert_eq!(result, vec![0, 1, 4]); // {0->1, 2->0, 4->4} = {0,1,4}
    }

    #[test]
    fn test_coset_elements() {
        let rep = Permutation::transposition(3, 0, 1);
        let subgroup_gens = vec![Permutation::cycle(3, &[0, 1, 2])];
        let coset = Coset::new(rep, subgroup_gens);
        let elems = coset.elements();
        assert_eq!(elems.len(), 3); // |<(012)>| = 3
    }

    #[test]
    fn test_packed_permutation_roundtrip() {
        let perm = Permutation::new(vec![3, 0, 2, 1, 4]);
        let packed = PackedPermutation::from_permutation(&perm);
        let unpacked = packed.to_permutation();
        assert_eq!(perm, unpacked);
    }

    #[test]
    fn test_packed_permutation_small() {
        let perm = Permutation::identity(2);
        let packed = PackedPermutation::from_permutation(&perm);
        assert_eq!(packed.degree(), 2);
        let unpacked = packed.to_permutation();
        assert_eq!(perm, unpacked);
    }

    #[test]
    fn test_extend_to() {
        let p = Permutation::cycle(3, &[0, 1, 2]);
        let ext = p.extend_to(5);
        assert_eq!(ext.degree(), 5);
        assert_eq!(ext.apply(0), 1);
        assert_eq!(ext.apply(3), 3);
        assert_eq!(ext.apply(4), 4);
    }

    #[test]
    fn test_apply_to_vec() {
        let p = Permutation::new(vec![2, 0, 1]);
        let v = vec!["a", "b", "c"];
        let result = p.apply_to_vec(&v);
        // p sends 0->2, 1->0, 2->1
        // So result[2] = v[0] = "a", result[0] = v[1] = "b", result[1] = v[2] = "c"
        assert_eq!(result, vec!["b", "c", "a"]);
    }

    #[test]
    fn test_enumerate_from_generators_s3() {
        let gens = vec![
            Permutation::cycle(3, &[0, 1, 2]),
            Permutation::transposition(3, 0, 1),
        ];
        let elements = enumerate_from_generators(&gens, 3);
        assert_eq!(elements.len(), 6); // |S_3| = 6
    }

    #[test]
    fn test_group_element_trait() {
        let ctx = GroupContext { degree: 4 };
        let id = Permutation::identity(ctx.degree);
        assert!(id.is_identity_element());
        let p = Permutation::cycle(4, &[0, 1, 2]);
        let inv = p.invert();
        let product = p.compose_with(&inv);
        assert!(product.is_identity_element());
        assert_eq!(p.element_order(), 3);
    }

    #[test]
    fn test_to_cycle_notation() {
        let id = Permutation::identity(3);
        assert_eq!(id.to_cycle_notation(), "()");
        let t = Permutation::transposition(4, 1, 3);
        assert_eq!(t.to_cycle_notation(), "(1 3)");
    }

    #[test]
    fn test_try_new_invalid() {
        assert!(Permutation::try_new(vec![0, 0, 2]).is_none());
        assert!(Permutation::try_new(vec![0, 5]).is_none());
        assert!(Permutation::try_new(vec![1, 0]).is_some());
    }

    #[test]
    fn test_symmetric_group_axioms() {
        // Verify all group axioms for S_3
        let gens = vec![
            Permutation::cycle(3, &[0, 1, 2]),
            Permutation::transposition(3, 0, 1),
        ];
        let elements: Vec<_> = enumerate_from_generators(&gens, 3).into_iter().collect();
        let id = Permutation::identity(3);

        // Closure
        for a in &elements {
            for b in &elements {
                let ab = a.compose(b);
                assert!(elements.contains(&ab), "Closure violated");
            }
        }

        // Identity
        for a in &elements {
            assert_eq!(a.compose(&id), *a);
            assert_eq!(id.compose(a), *a);
        }

        // Inverse
        for a in &elements {
            let inv = a.inverse();
            assert!(elements.contains(&inv), "Inverse not in group");
            assert!(a.compose(&inv).is_identity());
        }

        // Associativity (spot check)
        for i in 0..elements.len().min(4) {
            for j in 0..elements.len().min(4) {
                for k in 0..elements.len().min(4) {
                    let a = &elements[i];
                    let b = &elements[j];
                    let c = &elements[k];
                    assert_eq!(
                        a.compose(b).compose(c),
                        a.compose(&b.compose(c)),
                        "Associativity violated"
                    );
                }
            }
        }
    }
}
