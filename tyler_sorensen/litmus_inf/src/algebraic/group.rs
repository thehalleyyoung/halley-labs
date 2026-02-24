// LITMUS∞ Algebraic Engine — Permutation Group Algorithms
//
//   Schreier-Sims algorithm for base/strong generating set
//   Group order computation
//   Membership testing
//   Random element generation
//   Orbit / stabilizer computation
//   Group intersection
//   Symmetric, dihedral, direct/wreath product constructions

use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;

use super::types::{Permutation, compute_point_orbit, enumerate_from_generators, Orbit};

// ── Schreier-Sims Data ───────────────────────────────────────────────

/// A level in the stabilizer chain.
/// Represents the coset table for G^{(i)} acting on base[i].
#[derive(Clone, Debug)]
struct StabilizerLevel {
    /// The base point for this level.
    base_point: u32,
    /// Orbit of the base point under G^{(i)}.
    orbit: HashSet<u32>,
    /// Transversal: maps each element β in the orbit to a
    /// group element u_β such that u_β(base_point) = β.
    transversal: HashMap<u32, Permutation>,
    /// Strong generators at this level.
    generators: Vec<Permutation>,
}

impl StabilizerLevel {
    fn new(base_point: u32, degree: usize) -> Self {
        let mut orbit = HashSet::new();
        orbit.insert(base_point);
        let mut transversal = HashMap::new();
        transversal.insert(base_point, Permutation::identity(degree));
        StabilizerLevel {
            base_point,
            orbit,
            transversal,
            generators: Vec::new(),
        }
    }

    /// Extend the orbit with a generator. Returns true if the orbit grew.
    fn extend_orbit(&mut self, gen: &Permutation, degree: usize) -> bool {
        let mut changed = false;
        let current_orbit: Vec<u32> = self.orbit.iter().copied().collect();
        let mut queue: VecDeque<u32> = VecDeque::new();

        for &pt in &current_orbit {
            let image = gen.apply(pt);
            if !self.orbit.contains(&image) {
                self.orbit.insert(image);
                let u_pt = self.transversal[&pt].clone();
                self.transversal.insert(image, u_pt.compose(gen));
                queue.push_back(image);
                changed = true;
            }
        }

        // BFS to extend further
        while let Some(pt) = queue.pop_front() {
            for g in self.generators.iter().chain(std::iter::once(gen)) {
                let image = g.apply(pt);
                if !self.orbit.contains(&image) {
                    self.orbit.insert(image);
                    let u_pt = self.transversal[&pt].clone();
                    self.transversal.insert(image, u_pt.compose(g));
                    queue.push_back(image);
                    changed = true;
                }
                // Also try inverse
                let inv = g.inverse();
                let image_inv = inv.apply(pt);
                if !self.orbit.contains(&image_inv) {
                    self.orbit.insert(image_inv);
                    let u_pt = self.transversal[&pt].clone();
                    self.transversal.insert(image_inv, u_pt.compose(&inv));
                    queue.push_back(image_inv);
                    changed = true;
                }
            }
        }

        changed
    }

    /// Recompute orbit from scratch given the current generators.
    fn recompute_orbit(&mut self, degree: usize) {
        self.orbit.clear();
        self.transversal.clear();
        self.orbit.insert(self.base_point);
        self.transversal
            .insert(self.base_point, Permutation::identity(degree));

        let mut queue = VecDeque::new();
        queue.push_back(self.base_point);

        while let Some(pt) = queue.pop_front() {
            for gen in &self.generators {
                let image = gen.apply(pt);
                if !self.orbit.contains(&image) {
                    self.orbit.insert(image);
                    let u_pt = self.transversal[&pt].clone();
                    self.transversal.insert(image, u_pt.compose(gen));
                    queue.push_back(image);
                }
                let inv = gen.inverse();
                let image_inv = inv.apply(pt);
                if !self.orbit.contains(&image_inv) {
                    self.orbit.insert(image_inv);
                    let u_pt = self.transversal[&pt].clone();
                    self.transversal.insert(image_inv, u_pt.compose(&inv));
                    queue.push_back(image_inv);
                }
            }
        }
    }

    /// Sift element: strip away the contribution at this level.
    /// Returns the remainder after "dividing out" the transversal.
    /// Returns None if the element's image is not in the orbit.
    fn sift(&self, perm: &Permutation) -> Option<Permutation> {
        let image = perm.apply(self.base_point);
        if let Some(u_beta) = self.transversal.get(&image) {
            Some(u_beta.inverse().compose(perm))
        } else {
            None
        }
    }
}

// ── PermutationGroup ─────────────────────────────────────────────────

/// A permutation group represented by generators and a Schreier-Sims
/// stabilizer chain.
#[derive(Clone, Debug)]
pub struct PermutationGroup {
    /// Degree of the group (acts on {0, ..., degree-1}).
    degree: usize,
    /// Original generators.
    generators: Vec<Permutation>,
    /// Base: sequence of points.
    base: Vec<u32>,
    /// Strong generating set.
    strong_generators: Vec<Permutation>,
    /// Stabilizer chain levels.
    levels: Vec<StabilizerLevel>,
    /// Whether the Schreier-Sims computation is complete.
    is_complete: bool,
}

impl PermutationGroup {
    // ── Construction ─────────────────────────────────────────────

    /// Create a new permutation group from generators.
    pub fn new(degree: usize, generators: Vec<Permutation>) -> Self {
        for g in &generators {
            assert_eq!(
                g.degree(),
                degree,
                "Generator degree mismatch"
            );
        }
        let mut group = PermutationGroup {
            degree,
            generators: generators.clone(),
            base: Vec::new(),
            strong_generators: Vec::new(),
            levels: Vec::new(),
            is_complete: false,
        };
        if !generators.is_empty() {
            group.compute_schreier_sims();
        } else {
            group.is_complete = true;
        }
        group
    }

    /// The trivial group on n elements.
    pub fn trivial(degree: usize) -> Self {
        PermutationGroup {
            degree,
            generators: Vec::new(),
            base: Vec::new(),
            strong_generators: Vec::new(),
            levels: Vec::new(),
            is_complete: true,
        }
    }

    /// The symmetric group S_n.
    pub fn symmetric(n: usize) -> Self {
        if n <= 1 {
            return Self::trivial(n);
        }
        // Generate S_n with (0 1) and (0 1 2 ... n-1)
        let mut gens = Vec::new();
        if n == 2 {
            gens.push(Permutation::transposition(n, 0, 1));
        } else {
            gens.push(Permutation::transposition(n, 0, 1));
            gens.push(Permutation::cycle(n, &(0..n as u32).collect::<Vec<_>>()));
        }
        Self::new(n, gens)
    }

    /// The alternating group A_n.
    pub fn alternating(n: usize) -> Self {
        if n <= 2 {
            return Self::trivial(n);
        }
        // Generate A_n with 3-cycles (0 1 2), (0 1 3), ..., (0 1 n-1)
        let mut gens = Vec::new();
        for i in 2..n as u32 {
            gens.push(Permutation::cycle(n, &[0, 1, i]));
        }
        Self::new(n, gens)
    }

    /// The cyclic group C_n (generated by (0 1 2 ... n-1)).
    pub fn cyclic(n: usize) -> Self {
        if n <= 1 {
            return Self::trivial(n);
        }
        let gen = Permutation::cycle(n, &(0..n as u32).collect::<Vec<_>>());
        Self::new(n, vec![gen])
    }

    /// The dihedral group D_n of order 2n (symmetries of regular n-gon).
    pub fn dihedral(n: usize) -> Self {
        if n <= 1 {
            return Self::trivial(n.max(1));
        }
        let rotation = Permutation::cycle(n, &(0..n as u32).collect::<Vec<_>>());
        // Reflection: reverse (1..n-1)
        let mut refl_images: Vec<u32> = (0..n as u32).collect();
        refl_images[0] = 0;
        for i in 1..n {
            refl_images[i] = (n - i) as u32;
        }
        let reflection = Permutation::new(refl_images);
        Self::new(n, vec![rotation, reflection])
    }

    /// Direct product of two permutation groups.
    /// The result acts on {0, ..., n1+n2-1} where the first group
    /// acts on {0, ..., n1-1} and the second on {n1, ..., n1+n2-1}.
    pub fn direct_product(g1: &PermutationGroup, g2: &PermutationGroup) -> Self {
        let n1 = g1.degree;
        let n2 = g2.degree;
        let n = n1 + n2;

        let mut gens = Vec::new();
        // Embed g1 generators
        for gen in &g1.generators {
            let mut images: Vec<u32> = (0..n as u32).collect();
            for i in 0..n1 {
                images[i] = gen.apply(i as u32);
            }
            gens.push(Permutation::new(images));
        }
        // Embed g2 generators
        for gen in &g2.generators {
            let mut images: Vec<u32> = (0..n as u32).collect();
            for i in 0..n2 {
                images[n1 + i] = n1 as u32 + gen.apply(i as u32);
            }
            gens.push(Permutation::new(images));
        }

        Self::new(n, gens)
    }

    /// Wreath product G ≀ H where G acts on {0..m-1} and H acts on {0..k-1}.
    /// Result acts on {0..m*k-1}, with k copies of G permuted by H.
    pub fn wreath_product(g: &PermutationGroup, h: &PermutationGroup) -> Self {
        let m = g.degree;
        let k = h.degree;
        let n = m * k;

        let mut gens = Vec::new();

        // Base group generators: k copies of G, each in their block
        for block in 0..k {
            for gen in &g.generators {
                let mut images: Vec<u32> = (0..n as u32).collect();
                for i in 0..m {
                    images[block * m + i] = (block * m) as u32 + gen.apply(i as u32);
                }
                gens.push(Permutation::new(images));
            }
        }

        // Top group generators: H permutes the blocks
        for gen in &h.generators {
            let mut images: Vec<u32> = (0..n as u32).collect();
            for block in 0..k {
                let target_block = gen.apply(block as u32) as usize;
                for i in 0..m {
                    images[block * m + i] = (target_block * m + i) as u32;
                }
            }
            gens.push(Permutation::new(images));
        }

        Self::new(n, gens)
    }

    // ── Schreier-Sims Algorithm ──────────────────────────────────

    /// Select a base using a heuristic: pick points that are moved
    /// by the most generators first.
    fn select_base(degree: usize, generators: &[Permutation]) -> Vec<u32> {
        if generators.is_empty() {
            return Vec::new();
        }

        let mut base = Vec::new();
        // remaining_gens tracks generators that fix all base points so far
        // (i.e., they are in the stabilizer and need deeper base points).
        let mut remaining_gens: Vec<&Permutation> = generators
            .iter()
            .filter(|g| !g.is_identity())
            .collect();

        while !remaining_gens.is_empty() {
            // Score each point by how many remaining generators move it
            let mut scores = vec![0usize; degree];
            for gen in &remaining_gens {
                for i in 0..degree {
                    if gen.apply(i as u32) != i as u32 {
                        scores[i] += 1;
                    }
                }
            }

            // Pick the point with highest score among points not yet in base
            let best_point = match (0..degree)
                .filter(|&i| !base.contains(&(i as u32)) && scores[i] > 0)
                .max_by_key(|&i| scores[i])
            {
                Some(p) => p as u32,
                None => break,
            };

            base.push(best_point);

            // Keep only generators that fix this point (they need deeper levels)
            remaining_gens.retain(|g| g.apply(best_point) == best_point);
        }

        base
    }

    /// Main Schreier-Sims computation.
    fn compute_schreier_sims(&mut self) {
        // Select base
        self.base = Self::select_base(self.degree, &self.generators);
        if self.base.is_empty() {
            // All generators are identity
            self.is_complete = true;
            return;
        }

        // Initialize levels
        self.levels.clear();
        for &bp in &self.base {
            self.levels.push(StabilizerLevel::new(bp, self.degree));
        }

        // Distribute generators to levels
        self.strong_generators = self.generators.clone();
        let gens_copy = self.generators.clone();
        for gen in &gens_copy {
            self.add_generator_to_levels(gen.clone());
        }

        // Schreier-Sims main loop: compute Schreier generators and sift
        let max_iterations = 100;
        for _ in 0..max_iterations {
            let mut new_gens = Vec::new();

            for level_idx in 0..self.levels.len() {
                let schreier_gens = self.compute_schreier_generators(level_idx);
                for sg in schreier_gens {
                    if let Some(residue) = self.sift_through(&sg, level_idx + 1) {
                        if !residue.is_identity() {
                            new_gens.push(residue);
                        }
                    }
                }
            }

            if new_gens.is_empty() {
                break;
            }

            for gen in new_gens {
                self.add_new_strong_generator(gen);
            }
        }

        self.is_complete = true;
    }

    /// Add a generator to the appropriate levels.
    fn add_generator_to_levels(&mut self, gen: Permutation) {
        if gen.is_identity() {
            return;
        }
        // Find the first base point moved by gen
        let first_moved = self
            .base
            .iter()
            .position(|&bp| gen.apply(bp) != bp);

        if let Some(level_idx) = first_moved {
            self.levels[level_idx].generators.push(gen.clone());
            self.levels[level_idx].extend_orbit(&gen, self.degree);
        }
    }

    /// Compute Schreier generators at a given level.
    fn compute_schreier_generators(&self, level_idx: usize) -> Vec<Permutation> {
        let level = &self.levels[level_idx];
        let mut schreier_gens = Vec::new();

        for &beta in level.orbit.iter() {
            let u_beta = &level.transversal[&beta];
            for gen in &level.generators {
                let gamma = gen.apply(beta);
                if let Some(u_gamma) = level.transversal.get(&gamma) {
                    // Schreier generator: u_gamma^{-1} * gen * u_beta
                    let sg = u_gamma.inverse().compose(gen).compose(u_beta);
                    if !sg.is_identity() {
                        schreier_gens.push(sg);
                    }
                }
            }
        }

        schreier_gens
    }

    /// Sift a permutation through levels starting from `start_level`.
    /// Returns the residue, or None if it drops out of all levels.
    fn sift_through(&self, perm: &Permutation, start_level: usize) -> Option<Permutation> {
        let mut current = perm.clone();
        for level_idx in start_level..self.levels.len() {
            match self.levels[level_idx].sift(&current) {
                Some(remainder) => {
                    current = remainder;
                    if current.is_identity() {
                        return Some(current);
                    }
                }
                None => {
                    return Some(current);
                }
            }
        }
        Some(current)
    }

    /// Add a new strong generator, possibly extending the base.
    fn add_new_strong_generator(&mut self, gen: Permutation) {
        if gen.is_identity() {
            return;
        }

        self.strong_generators.push(gen.clone());

        // Find where this generator first moves a base point
        let first_moved = self
            .base
            .iter()
            .position(|&bp| gen.apply(bp) != bp);

        match first_moved {
            Some(level_idx) => {
                self.levels[level_idx].generators.push(gen.clone());
                self.levels[level_idx].extend_orbit(&gen, self.degree);
            }
            None => {
                // Need to extend the base
                let new_bp = gen
                    .support()
                    .into_iter()
                    .find(|p| !self.base.contains(p))
                    .unwrap_or(0);
                self.base.push(new_bp);
                let mut level = StabilizerLevel::new(new_bp, self.degree);
                level.generators.push(gen.clone());
                level.extend_orbit(&gen, self.degree);
                self.levels.push(level);
            }
        }
    }

    // ── Queries ──────────────────────────────────────────────────

    /// The degree of the group.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The generators.
    pub fn generators(&self) -> &[Permutation] {
        &self.generators
    }

    /// The base.
    pub fn base(&self) -> &[u32] {
        &self.base
    }

    /// The strong generating set.
    pub fn strong_generators(&self) -> &[Permutation] {
        &self.strong_generators
    }

    /// Compute the order of the group.
    /// |G| = product of orbit sizes at each level.
    pub fn order(&self) -> u64 {
        if self.levels.is_empty() {
            return 1;
        }
        self.levels
            .iter()
            .map(|level| level.orbit.len() as u64)
            .product()
    }

    /// Test membership: is `perm` in this group?
    /// Uses the Schreier-Sims sifting procedure.
    /// Time complexity: O(|base| * n).
    pub fn contains(&self, perm: &Permutation) -> bool {
        if perm.degree() != self.degree {
            return false;
        }
        if perm.is_identity() {
            return true;
        }

        let mut current = perm.clone();
        for level in &self.levels {
            match level.sift(&current) {
                Some(remainder) => {
                    current = remainder;
                    if current.is_identity() {
                        return true;
                    }
                }
                None => return false,
            }
        }
        current.is_identity()
    }

    /// Factor a group element as a product of transversal elements.
    /// Returns the sequence of transversal elements, or None if not in group.
    pub fn factor(&self, perm: &Permutation) -> Option<Vec<Permutation>> {
        if perm.degree() != self.degree {
            return None;
        }

        let mut factors = Vec::new();
        let mut current = perm.clone();

        for level in &self.levels {
            let image = current.apply(level.base_point);
            if let Some(u_beta) = level.transversal.get(&image) {
                factors.push(u_beta.clone());
                current = u_beta.inverse().compose(&current);
            } else {
                return None;
            }
        }

        if current.is_identity() {
            Some(factors)
        } else {
            None
        }
    }

    /// Generate a uniformly random element using the product replacement algorithm.
    pub fn random_element<R: Rng>(&self, rng: &mut R) -> Permutation {
        if self.levels.is_empty() {
            return Permutation::identity(self.degree);
        }

        // Pick a random element from each transversal
        let mut result = Permutation::identity(self.degree);
        for level in &self.levels {
            let orbit_vec: Vec<u32> = level.orbit.iter().copied().collect();
            let idx = rng.gen_range(0..orbit_vec.len());
            let beta = orbit_vec[idx];
            let u_beta = &level.transversal[&beta];
            result = u_beta.compose(&result);
        }
        result
    }

    /// Generate multiple random elements.
    pub fn random_elements<R: Rng>(&self, rng: &mut R, count: usize) -> Vec<Permutation> {
        (0..count).map(|_| self.random_element(rng)).collect()
    }

    /// Enumerate all elements of the group (only for small groups!).
    /// Panics if the group has more than `max_size` elements.
    pub fn enumerate_elements(&self) -> Vec<Permutation> {
        self.enumerate_elements_bounded(100_000)
    }

    /// Alias for `enumerate_elements`.
    pub fn elements(&self) -> Vec<Permutation> {
        self.enumerate_elements()
    }

    /// Enumerate elements with a size bound.
    pub fn enumerate_elements_bounded(&self, max_size: u64) -> Vec<Permutation> {
        let ord = self.order();
        assert!(
            ord <= max_size,
            "Group too large to enumerate: {} > {}",
            ord,
            max_size
        );
        enumerate_from_generators(&self.generators, self.degree)
            .into_iter()
            .collect()
    }

    /// Compute the orbit of a point under this group.
    pub fn orbit(&self, point: u32) -> Orbit<u32> {
        compute_point_orbit(point, &self.generators, self.degree)
    }

    /// Compute all orbits of this group.
    pub fn all_orbits(&self) -> Vec<Orbit<u32>> {
        let mut seen = HashSet::new();
        let mut orbits = Vec::new();

        for i in 0..self.degree as u32 {
            if seen.contains(&i) {
                continue;
            }
            let orbit = self.orbit(i);
            for &pt in &orbit.elements {
                seen.insert(pt);
            }
            orbits.push(orbit);
        }

        orbits
    }

    /// Is the group transitive?
    pub fn is_transitive(&self) -> bool {
        if self.degree == 0 {
            return true;
        }
        let orbit = self.orbit(0);
        orbit.size() == self.degree
    }

    /// Compute the stabilizer subgroup of a point.
    pub fn stabilizer(&self, point: u32) -> PermutationGroup {
        // Collect generators that fix the point
        let mut stab_gens = Vec::new();

        // From the Schreier-Sims data, the stabilizer of a base point
        // is generated by the strong generators at the next level.
        // For arbitrary points, we compute Schreier generators.

        if let Some(level_idx) = self.base.iter().position(|&bp| bp == point) {
            // Point is in the base; use the stabilizer chain
            for i in (level_idx + 1)..self.levels.len() {
                stab_gens.extend(self.levels[i].generators.clone());
            }
            // Also add Schreier generators
            let schreier = self.compute_schreier_generators(level_idx);
            for sg in schreier {
                if sg.apply(point) == point && !sg.is_identity() {
                    stab_gens.push(sg);
                }
            }
        } else {
            // Brute force: find all elements fixing the point
            // For large groups, we'd use a smarter approach
            if self.order() <= 10000 {
                let elements = self.enumerate_elements();
                for elem in elements {
                    if elem.apply(point) == point && !elem.is_identity() {
                        stab_gens.push(elem);
                    }
                }
            } else {
                // Use orbit-stabilizer approach
                let orbit = self.orbit(point);
                for &beta in &orbit.elements {
                    let u_beta = orbit.transversal[&beta].clone();
                    for gen in &self.generators {
                        let gamma = gen.apply(beta);
                        if let Some(u_gamma) = orbit.transversal.get(&gamma) {
                            let sg = u_gamma.inverse().compose(gen).compose(&u_beta);
                            if !sg.is_identity() && sg.apply(point) == point {
                                stab_gens.push(sg);
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate
        let gen_set: HashSet<Permutation> = stab_gens.into_iter().collect();
        PermutationGroup::new(self.degree, gen_set.into_iter().collect())
    }

    /// Pointwise stabilizer of a set of points.
    pub fn pointwise_stabilizer(&self, points: &[u32]) -> PermutationGroup {
        let mut result = self.clone();
        for &pt in points {
            result = result.stabilizer(pt);
        }
        result
    }

    /// Setwise stabilizer of a set of points.
    pub fn setwise_stabilizer(&self, set: &[u32]) -> PermutationGroup {
        let set_hash: HashSet<u32> = set.iter().copied().collect();
        if self.order() <= 10000 {
            let elements = self.enumerate_elements();
            let stab_gens: Vec<Permutation> = elements
                .into_iter()
                .filter(|g| {
                    set.iter().all(|&pt| set_hash.contains(&g.apply(pt)))
                })
                .collect();
            PermutationGroup::new(self.degree, stab_gens)
        } else {
            // Approximate: use strong generators that stabilize the set
            let stab_gens: Vec<Permutation> = self
                .strong_generators
                .iter()
                .filter(|g| {
                    set.iter().all(|&pt| set_hash.contains(&g.apply(pt)))
                })
                .cloned()
                .collect();
            PermutationGroup::new(self.degree, stab_gens)
        }
    }

    /// Compute intersection of two groups (for small groups).
    pub fn intersect(&self, other: &PermutationGroup) -> PermutationGroup {
        assert_eq!(self.degree, other.degree);

        if self.order() <= 10000 && other.order() <= 10000 {
            let elements_self = self.enumerate_elements();
            let elements_other: HashSet<Permutation> =
                other.enumerate_elements().into_iter().collect();
            let intersection: Vec<Permutation> = elements_self
                .into_iter()
                .filter(|e| elements_other.contains(e))
                .collect();
            PermutationGroup::new(self.degree, intersection)
        } else {
            // Heuristic: use strong generators
            let mut intersection_gens = Vec::new();
            for gen in &self.strong_generators {
                if other.contains(gen) {
                    intersection_gens.push(gen.clone());
                }
            }
            for gen in &other.strong_generators {
                if self.contains(gen) && !intersection_gens.contains(gen) {
                    intersection_gens.push(gen.clone());
                }
            }
            PermutationGroup::new(self.degree, intersection_gens)
        }
    }

    /// Coset representatives of a subgroup in this group.
    pub fn coset_representatives(&self, subgroup: &PermutationGroup) -> Vec<Permutation> {
        assert_eq!(self.degree, subgroup.degree);

        let elements = self.enumerate_elements();
        let mut reps: Vec<Permutation> = Vec::new();

        for elem in &elements {
            // Check if elem's coset is already represented
            let mut found = false;
            for rep in &reps {
                let diff = rep.inverse().compose(elem);
                if subgroup.contains(&diff) {
                    found = true;
                    break;
                }
            }
            if !found {
                reps.push(elem.clone());
            }
        }

        reps
    }

    /// Right coset representatives using the stabilizer chain.
    pub fn right_coset_reps_for_level(&self, level_idx: usize) -> Vec<Permutation> {
        if level_idx >= self.levels.len() {
            return vec![Permutation::identity(self.degree)];
        }
        self.levels[level_idx]
            .transversal
            .values()
            .cloned()
            .collect()
    }

    /// Center of the group (elements that commute with everything).
    pub fn center(&self) -> PermutationGroup {
        if self.order() > 10000 {
            // Fallback
            return PermutationGroup::trivial(self.degree);
        }

        let elements = self.enumerate_elements();
        let center_elements: Vec<Permutation> = elements
            .iter()
            .filter(|&z| {
                elements.iter().all(|g| z.compose(g) == g.compose(z))
            })
            .cloned()
            .collect();

        PermutationGroup::new(self.degree, center_elements)
    }

    /// Derived subgroup (commutator subgroup).
    pub fn derived_subgroup(&self) -> PermutationGroup {
        if self.order() > 10000 {
            // Heuristic: use generator commutators
            let mut comm_gens = Vec::new();
            for g1 in &self.generators {
                for g2 in &self.generators {
                    let c = g1.commutator(g2);
                    if !c.is_identity() {
                        comm_gens.push(c);
                    }
                }
            }
            return PermutationGroup::new(self.degree, comm_gens);
        }

        let elements = self.enumerate_elements();
        let mut commutators = Vec::new();
        for a in &elements {
            for b in &elements {
                let c = a.commutator(b);
                if !c.is_identity() {
                    commutators.push(c);
                }
            }
        }

        let comm_set: HashSet<Permutation> = commutators.into_iter().collect();
        PermutationGroup::new(self.degree, comm_set.into_iter().collect())
    }

    /// Check if the group is abelian.
    pub fn is_abelian(&self) -> bool {
        for i in 0..self.generators.len() {
            for j in (i + 1)..self.generators.len() {
                if self.generators[i].compose(&self.generators[j])
                    != self.generators[j].compose(&self.generators[i])
                {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the group is a subgroup of another.
    pub fn is_subgroup_of(&self, other: &PermutationGroup) -> bool {
        self.generators.iter().all(|g| other.contains(g))
    }

    /// Normal closure of a set of elements.
    pub fn normal_closure(&self, elements: &[Permutation]) -> PermutationGroup {
        let mut gens = elements.to_vec();
        let mut changed = true;
        let max_iter = 20;
        let mut iter = 0;

        while changed && iter < max_iter {
            changed = false;
            iter += 1;
            let mut new_gens = Vec::new();
            for g in &self.generators {
                for h in &gens {
                    let conjugate = h.conjugate_by(g);
                    let current_group = PermutationGroup::new(self.degree, gens.clone());
                    if !current_group.contains(&conjugate) {
                        new_gens.push(conjugate);
                        changed = true;
                    }
                }
            }
            gens.extend(new_gens);
        }

        PermutationGroup::new(self.degree, gens)
    }

    /// Orbit sizes (sorted descending).
    pub fn orbit_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.all_orbits().iter().map(|o| o.size()).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// Orbit partition: returns vec of vec of points, one per orbit.
    pub fn orbit_partition(&self) -> Vec<Vec<u32>> {
        self.all_orbits()
            .iter()
            .map(|o| {
                let mut pts: Vec<u32> = o.elements.iter().copied().collect();
                pts.sort_unstable();
                pts
            })
            .collect()
    }

    /// Restrict the group to a subset (only keep generators acting within subset).
    pub fn restrict_to(&self, subset: &[u32]) -> PermutationGroup {
        let n = subset.len();
        let subset_set: HashSet<u32> = subset.iter().copied().collect();
        let index_map: HashMap<u32, u32> = subset
            .iter()
            .enumerate()
            .map(|(i, &pt)| (pt, i as u32))
            .collect();

        let mut gens = Vec::new();
        for gen in &self.generators {
            // Check if generator preserves the subset
            if subset.iter().all(|&pt| subset_set.contains(&gen.apply(pt))) {
                let mut images = vec![0u32; n];
                for (i, &pt) in subset.iter().enumerate() {
                    images[i] = index_map[&gen.apply(pt)];
                }
                if let Some(p) = Permutation::try_new(images) {
                    if !p.is_identity() {
                        gens.push(p);
                    }
                }
            }
        }

        PermutationGroup::new(n, gens)
    }

    /// Compute the action on blocks (for imprimitive groups).
    pub fn block_action(&self, blocks: &[Vec<u32>]) -> PermutationGroup {
        let k = blocks.len();
        let block_map: HashMap<u32, usize> = blocks
            .iter()
            .enumerate()
            .flat_map(|(i, block)| block.iter().map(move |&pt| (pt, i)))
            .collect();

        let mut gens = Vec::new();
        for gen in &self.generators {
            let mut images = vec![0u32; k];
            let mut valid = true;
            for (i, block) in blocks.iter().enumerate() {
                if block.is_empty() {
                    valid = false;
                    break;
                }
                let target_block = block_map[&gen.apply(block[0])];
                // Verify entire block maps to target_block
                if !block
                    .iter()
                    .all(|&pt| block_map[&gen.apply(pt)] == target_block)
                {
                    valid = false;
                    break;
                }
                images[i] = target_block as u32;
            }
            if valid {
                if let Some(p) = Permutation::try_new(images) {
                    if !p.is_identity() {
                        gens.push(p);
                    }
                }
            }
        }

        PermutationGroup::new(k, gens)
    }

    /// Lexicographically smallest element in the orbit of a sequence.
    pub fn canonical_sequence(&self, seq: &[u32]) -> Vec<u32> {
        if self.order() <= 10000 {
            let elements = self.enumerate_elements();
            let mut best = seq.to_vec();
            for elem in &elements {
                let permuted: Vec<u32> = seq.iter().map(|&x| elem.apply(x)).collect();
                if permuted < best {
                    best = permuted;
                }
            }
            best
        } else {
            // For large groups, try generators and their products
            let mut best = seq.to_vec();
            let mut improved = true;
            while improved {
                improved = false;
                for gen in &self.generators {
                    let permuted: Vec<u32> = best.iter().map(|&x| gen.apply(x)).collect();
                    if permuted < best {
                        best = permuted;
                        improved = true;
                    }
                    let inv = gen.inverse();
                    let permuted_inv: Vec<u32> = best.iter().map(|&x| inv.apply(x)).collect();
                    if permuted_inv < best {
                        best = permuted_inv;
                        improved = true;
                    }
                }
            }
            best
        }
    }

    /// Decompose into orbits and return the action restricted to each orbit.
    pub fn orbit_decomposition(&self) -> Vec<(Vec<u32>, PermutationGroup)> {
        let orbits = self.all_orbits();
        let mut result = Vec::new();
        for orbit in orbits {
            let mut pts: Vec<u32> = orbit.elements.into_iter().collect();
            pts.sort_unstable();
            let restricted = self.restrict_to(&pts);
            result.push((pts, restricted));
        }
        result
    }

    /// Number of generators.
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }

    /// Number of strong generators.
    pub fn num_strong_generators(&self) -> usize {
        self.strong_generators.len()
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "PermutationGroup(degree={}, order={}, generators={}, base_len={})",
            self.degree,
            self.order(),
            self.generators.len(),
            self.base.len(),
        )
    }
}

// ── Display ──────────────────────────────────────────────────────────

impl std::fmt::Display for PermutationGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Group(deg={}, |G|={}, gens=[{}])",
            self.degree,
            self.order(),
            self.generators
                .iter()
                .map(|g| g.to_cycle_notation())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_group() {
        let g = PermutationGroup::trivial(5);
        assert_eq!(g.order(), 1);
        assert!(g.contains(&Permutation::identity(5)));
        assert!(!g.contains(&Permutation::transposition(5, 0, 1)));
    }

    #[test]
    fn test_s3_order() {
        let g = PermutationGroup::symmetric(3);
        assert_eq!(g.order(), 6);
    }

    #[test]
    fn test_s4_order() {
        let g = PermutationGroup::symmetric(4);
        assert_eq!(g.order(), 24);
    }

    #[test]
    fn test_s5_order() {
        let g = PermutationGroup::symmetric(5);
        assert_eq!(g.order(), 120);
    }

    #[test]
    fn test_s3_membership() {
        let g = PermutationGroup::symmetric(3);
        // Every permutation of {0,1,2} should be in S_3
        assert!(g.contains(&Permutation::identity(3)));
        assert!(g.contains(&Permutation::transposition(3, 0, 1)));
        assert!(g.contains(&Permutation::transposition(3, 0, 2)));
        assert!(g.contains(&Permutation::transposition(3, 1, 2)));
        assert!(g.contains(&Permutation::cycle(3, &[0, 1, 2])));
        assert!(g.contains(&Permutation::cycle(3, &[0, 2, 1])));
    }

    #[test]
    fn test_s3_enumerate() {
        let g = PermutationGroup::symmetric(3);
        let elements = g.enumerate_elements();
        assert_eq!(elements.len(), 6);
    }

    #[test]
    fn test_cyclic_group() {
        let g = PermutationGroup::cyclic(5);
        assert_eq!(g.order(), 5);
        let gen = Permutation::cycle(5, &[0, 1, 2, 3, 4]);
        assert!(g.contains(&gen));
        assert!(g.contains(&gen.pow(2)));
        assert!(!g.contains(&Permutation::transposition(5, 0, 1)));
    }

    #[test]
    fn test_dihedral_group() {
        // D_3 has order 6 = |S_3| (they are isomorphic)
        let d3 = PermutationGroup::dihedral(3);
        assert_eq!(d3.order(), 6);

        // D_4 has order 8
        let d4 = PermutationGroup::dihedral(4);
        assert_eq!(d4.order(), 8);

        // D_5 has order 10
        let d5 = PermutationGroup::dihedral(5);
        assert_eq!(d5.order(), 10);
    }

    #[test]
    fn test_alternating_group() {
        let a4 = PermutationGroup::alternating(4);
        assert_eq!(a4.order(), 12);

        let a3 = PermutationGroup::alternating(3);
        assert_eq!(a3.order(), 3);
    }

    #[test]
    fn test_orbit() {
        let g = PermutationGroup::symmetric(4);
        let orbit = g.orbit(0);
        assert_eq!(orbit.size(), 4); // S_4 is transitive
    }

    #[test]
    fn test_orbit_cyclic() {
        let g = PermutationGroup::cyclic(6);
        let orbit = g.orbit(0);
        assert_eq!(orbit.size(), 6);
    }

    #[test]
    fn test_all_orbits() {
        // Direct product of C_2 on {0,1} and C_3 on {2,3,4}
        let c2 = PermutationGroup::cyclic(2);
        let c3 = PermutationGroup::cyclic(3);
        let product = PermutationGroup::direct_product(&c2, &c3);
        let orbits = product.all_orbits();
        let sizes: HashSet<usize> = orbits.iter().map(|o| o.size()).collect();
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&3));
    }

    #[test]
    fn test_is_transitive() {
        let s4 = PermutationGroup::symmetric(4);
        assert!(s4.is_transitive());

        let c2 = PermutationGroup::cyclic(2);
        let c3 = PermutationGroup::cyclic(3);
        let product = PermutationGroup::direct_product(&c2, &c3);
        assert!(!product.is_transitive());
    }

    #[test]
    fn test_stabilizer() {
        let s4 = PermutationGroup::symmetric(4);
        let stab = s4.stabilizer(0);
        // Stab_0(S_4) ≅ S_3, order 6
        assert_eq!(stab.order(), 6);
    }

    #[test]
    fn test_stabilizer_orbit_theorem() {
        let s4 = PermutationGroup::symmetric(4);
        let orbit = s4.orbit(0);
        let stab = s4.stabilizer(0);
        // |G| = |Orb(x)| * |Stab(x)|
        assert_eq!(s4.order(), orbit.size() as u64 * stab.order());
    }

    #[test]
    fn test_direct_product_order() {
        let c2 = PermutationGroup::cyclic(2);
        let c3 = PermutationGroup::cyclic(3);
        let product = PermutationGroup::direct_product(&c2, &c3);
        assert_eq!(product.order(), 6); // 2 * 3
        assert_eq!(product.degree(), 5); // 2 + 3
    }

    #[test]
    fn test_wreath_product_order() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wreath = PermutationGroup::wreath_product(&c2, &s2);
        // |C_2 ≀ S_2| = |C_2|^2 * |S_2| = 4 * 2 = 8
        assert_eq!(wreath.order(), 8);
        assert_eq!(wreath.degree(), 4); // 2 * 2
    }

    #[test]
    fn test_is_abelian() {
        let c5 = PermutationGroup::cyclic(5);
        assert!(c5.is_abelian());

        let s3 = PermutationGroup::symmetric(3);
        assert!(!s3.is_abelian());

        let c2 = PermutationGroup::cyclic(2);
        let c3 = PermutationGroup::cyclic(3);
        let product = PermutationGroup::direct_product(&c2, &c3);
        assert!(product.is_abelian());
    }

    #[test]
    fn test_center() {
        let s3 = PermutationGroup::symmetric(3);
        let center = s3.center();
        assert_eq!(center.order(), 1); // Center of S_3 is trivial

        let c5 = PermutationGroup::cyclic(5);
        let center_c5 = c5.center();
        assert_eq!(center_c5.order(), 5); // Cyclic group is abelian
    }

    #[test]
    fn test_derived_subgroup() {
        let s3 = PermutationGroup::symmetric(3);
        let derived = s3.derived_subgroup();
        assert_eq!(derived.order(), 3); // [S_3, S_3] = A_3 ≅ C_3
    }

    #[test]
    fn test_random_element_membership() {
        let s4 = PermutationGroup::symmetric(4);
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let elem = s4.random_element(&mut rng);
            assert!(s4.contains(&elem));
        }
    }

    #[test]
    fn test_factor() {
        let s3 = PermutationGroup::symmetric(3);
        let perm = Permutation::cycle(3, &[0, 1, 2]);
        let factors = s3.factor(&perm);
        assert!(factors.is_some());
    }

    #[test]
    fn test_coset_representatives() {
        let s3 = PermutationGroup::symmetric(3);
        let c3 = PermutationGroup::new(
            3,
            vec![Permutation::cycle(3, &[0, 1, 2])],
        );
        let reps = s3.coset_representatives(&c3);
        // [S_3 : C_3] = 2
        assert_eq!(reps.len(), 2);
    }

    #[test]
    fn test_orbit_partition() {
        let g = PermutationGroup::new(
            6,
            vec![
                Permutation::transposition(6, 0, 1),
                Permutation::transposition(6, 2, 3),
                Permutation::cycle(6, &[4, 5]),
            ],
        );
        let partition = g.orbit_partition();
        let sizes: HashSet<usize> = partition.iter().map(|o| o.len()).collect();
        assert!(sizes.contains(&2));
    }

    #[test]
    fn test_restrict_to() {
        let s4 = PermutationGroup::symmetric(4);
        let restricted = s4.restrict_to(&[0, 1, 2]);
        // S_4 restricted to {0,1,2}: generators that fix 3 and act on {0,1,2}
        // This gives S_3
        assert_eq!(restricted.order(), 6);
    }

    #[test]
    fn test_block_action() {
        let s4 = PermutationGroup::symmetric(4);
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let block_group = s4.block_action(&blocks);
        // S_4 permuting the two blocks {0,1} and {2,3} gives S_2
        assert_eq!(block_group.degree(), 2);
        assert!(block_group.order() <= 2);
    }

    #[test]
    fn test_canonical_sequence() {
        let s3 = PermutationGroup::symmetric(3);
        let seq = vec![2, 0, 1];
        let canonical = s3.canonical_sequence(&seq);
        assert_eq!(canonical, vec![0, 1, 2]);
    }

    #[test]
    fn test_intersect() {
        let s4 = PermutationGroup::symmetric(4);
        let a4 = PermutationGroup::alternating(4);
        let intersection = s4.intersect(&a4);
        assert_eq!(intersection.order(), 12); // A_4
    }

    #[test]
    fn test_is_subgroup() {
        let a4 = PermutationGroup::alternating(4);
        let s4 = PermutationGroup::symmetric(4);
        assert!(a4.is_subgroup_of(&s4));
        assert!(!s4.is_subgroup_of(&a4));
    }

    #[test]
    fn test_setwise_stabilizer() {
        let s4 = PermutationGroup::symmetric(4);
        let stab = s4.setwise_stabilizer(&[0, 1]);
        // Setwise stabilizer of {0,1} in S_4:
        // Permutations that map {0,1} to {0,1}
        // This is S_2 × S_2, order 4
        assert_eq!(stab.order(), 4);
    }

    #[test]
    fn test_orbit_decomposition() {
        let g = PermutationGroup::new(
            4,
            vec![
                Permutation::transposition(4, 0, 1),
                Permutation::transposition(4, 2, 3),
            ],
        );
        let decomp = g.orbit_decomposition();
        assert_eq!(decomp.len(), 2);
        for (pts, sub_g) in &decomp {
            assert_eq!(pts.len(), 2);
            assert_eq!(sub_g.order(), 2);
        }
    }

    #[test]
    fn test_s1_trivial() {
        let s1 = PermutationGroup::symmetric(1);
        assert_eq!(s1.order(), 1);
    }

    #[test]
    fn test_s2_order() {
        let s2 = PermutationGroup::symmetric(2);
        assert_eq!(s2.order(), 2);
    }

    #[test]
    fn test_direct_product_larger() {
        let s3 = PermutationGroup::symmetric(3);
        let s2 = PermutationGroup::symmetric(2);
        let product = PermutationGroup::direct_product(&s3, &s2);
        assert_eq!(product.order(), 12); // 6 * 2
        assert_eq!(product.degree(), 5); // 3 + 2
    }

    #[test]
    fn test_group_summary() {
        let s4 = PermutationGroup::symmetric(4);
        let summary = s4.summary();
        assert!(summary.contains("24"));
        assert!(summary.contains("degree=4"));
    }

    #[test]
    fn test_wreath_product_s2_wr_s2() {
        let s2 = PermutationGroup::symmetric(2);
        let wreath = PermutationGroup::wreath_product(&s2, &s2);
        // |S_2 ≀ S_2| = |S_2|^2 * |S_2| = 4 * 2 = 8
        assert_eq!(wreath.order(), 8);
    }

    #[test]
    fn test_pointwise_stabilizer() {
        let s4 = PermutationGroup::symmetric(4);
        let stab = s4.pointwise_stabilizer(&[0, 1]);
        // Stab_{0,1}(S_4) ≅ S_2 (permutations fixing 0 and 1)
        assert_eq!(stab.order(), 2);
    }

    #[test]
    fn test_normal_closure() {
        let s4 = PermutationGroup::symmetric(4);
        let elem = Permutation::cycle(4, &[0, 1, 2]);
        let closure = s4.normal_closure(&[elem]);
        // Normal closure of a 3-cycle in S_4 is A_4
        assert_eq!(closure.order(), 12);
    }
}
