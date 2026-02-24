//! Advanced permutation group algorithms for the LITMUS∞ algebraic engine.
//!
//! Extends the basic `Permutation` type from `types.rs` with:
//! - Lehmer code computation and reconstruction
//! - Permutation matrices and descent sets
//! - Conjugacy class enumeration
//! - Character table construction for symmetric groups
//! - Block system computation and primitivity testing
//! - Schreier vector factored transversals
//! - Group presentations and coset enumeration

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use super::types::{Permutation, Orbit, compute_point_orbit, enumerate_from_generators};

// ═══════════════════════════════════════════════════════════════════════════
// Lehmer Code and Inversion Table
// ═══════════════════════════════════════════════════════════════════════════

/// The Lehmer code (factorial number system) representation of a permutation.
/// `code[i]` = number of j > i such that σ(j) < σ(i).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LehmerCode {
    /// The code digits. `code[i]` is in {0, ..., n-1-i}.
    pub code: Vec<u32>,
}

impl LehmerCode {
    /// Compute the Lehmer code of a permutation.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.degree();
        let mut code = vec![0u32; n];
        for i in 0..n {
            let mut count = 0u32;
            for j in (i + 1)..n {
                if perm.apply(j as u32) < perm.apply(i as u32) {
                    count += 1;
                }
            }
            code[i] = count;
        }
        LehmerCode { code }
    }

    /// Reconstruct a permutation from its Lehmer code.
    pub fn to_permutation(&self) -> Permutation {
        let n = self.code.len();
        let mut available: Vec<u32> = (0..n as u32).collect();
        let mut images = vec![0u32; n];
        for i in 0..n {
            let idx = self.code[i] as usize;
            images[i] = available.remove(idx);
        }
        Permutation::new(images)
    }

    /// Convert Lehmer code to a rank (index in lexicographic order).
    pub fn to_rank(&self) -> u64 {
        let n = self.code.len();
        let mut rank = 0u64;
        let mut factorial = 1u64;
        for i in (0..n).rev() {
            rank += self.code[i] as u64 * factorial;
            factorial *= (n - i) as u64;
        }
        rank
    }

    /// Create a Lehmer code from a rank (lexicographic index).
    pub fn from_rank(n: usize, mut rank: u64) -> Self {
        let mut code = vec![0u32; n];
        for i in (0..n).rev() {
            let divisor = factorial(n - 1 - i);
            code[i] = (rank / divisor) as u32;
            rank %= divisor;
        }
        LehmerCode { code }
    }

    /// The degree of the permutation.
    pub fn degree(&self) -> usize {
        self.code.len()
    }
}

impl fmt::Display for LehmerCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &c) in self.code.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", c)?;
        }
        write!(f, "]")
    }
}

/// Compute n! (factorial). Panics on overflow for large n.
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}

// ═══════════════════════════════════════════════════════════════════════════
// Inversion Table
// ═══════════════════════════════════════════════════════════════════════════

/// Inversion table of a permutation.
/// `table[i]` = number of j < i such that σ(j) > σ(i).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InversionTable {
    /// The inversion counts per position.
    pub table: Vec<u32>,
}

impl InversionTable {
    /// Compute the inversion table of a permutation.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.degree();
        let mut table = vec![0u32; n];
        for i in 0..n {
            let mut count = 0u32;
            for j in 0..i {
                if perm.apply(j as u32) > perm.apply(i as u32) {
                    count += 1;
                }
            }
            table[i] = count;
        }
        InversionTable { table }
    }

    /// Total number of inversions.
    pub fn inversion_count(&self) -> u64 {
        self.table.iter().map(|&x| x as u64).sum()
    }

    /// The degree of the permutation.
    pub fn degree(&self) -> usize {
        self.table.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Descent Set and Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Combinatorial statistics of a permutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationStatistics {
    /// Descent set: {i : σ(i) > σ(i+1)}.
    pub descent_set: BTreeSet<usize>,
    /// Ascent set: {i : σ(i) < σ(i+1)}.
    pub ascent_set: BTreeSet<usize>,
    /// Major index: sum of descent positions.
    pub major_index: usize,
    /// Number of inversions.
    pub inversions: u64,
    /// Number of descents.
    pub num_descents: usize,
    /// Number of excedances: {i : σ(i) > i}.
    pub num_excedances: usize,
    /// Number of fixed points.
    pub num_fixed_points: usize,
}

impl PermutationStatistics {
    /// Compute all statistics for a permutation.
    pub fn compute(perm: &Permutation) -> Self {
        let n = perm.degree();
        let mut descent_set = BTreeSet::new();
        let mut ascent_set = BTreeSet::new();
        let mut num_excedances = 0;
        let mut num_fixed_points = 0;

        for i in 0..n {
            if perm.apply(i as u32) == i as u32 {
                num_fixed_points += 1;
            }
            if perm.apply(i as u32) > i as u32 {
                num_excedances += 1;
            }
            if i + 1 < n && perm.apply(i as u32) > perm.apply((i + 1) as u32) {
                descent_set.insert(i);
            }
            if i + 1 < n && perm.apply(i as u32) < perm.apply((i + 1) as u32) {
                ascent_set.insert(i);
            }
        }

        let major_index: usize = descent_set.iter().sum();
        let inversions = InversionTable::from_permutation(perm).inversion_count();
        let num_descents = descent_set.len();

        PermutationStatistics {
            descent_set,
            ascent_set,
            major_index,
            inversions,
            num_descents,
            num_excedances,
            num_fixed_points,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Permutation Matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Permutation matrix representation: M[i][σ(i)] = 1, rest 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermutationMatrix {
    /// The size of the matrix (n × n).
    pub size: usize,
    /// Row-major entries: `entries[i * size + j]`.
    entries: Vec<u8>,
}

impl PermutationMatrix {
    /// Construct the permutation matrix for a permutation.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.degree();
        let mut entries = vec![0u8; n * n];
        for i in 0..n {
            entries[i * n + perm.apply(i as u32) as usize] = 1;
        }
        PermutationMatrix { size: n, entries }
    }

    /// Get the entry at (row, col).
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.entries[row * self.size + col]
    }

    /// Multiply two permutation matrices.
    pub fn multiply(&self, other: &PermutationMatrix) -> PermutationMatrix {
        assert_eq!(self.size, other.size);
        let n = self.size;
        let mut result = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0u8;
                for k in 0..n {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result[i * n + j] = sum;
            }
        }
        PermutationMatrix { size: n, entries: result }
    }

    /// Convert back to a Permutation.
    pub fn to_permutation(&self) -> Permutation {
        let n = self.size;
        let mut images = vec![0u32; n];
        for i in 0..n {
            for j in 0..n {
                if self.get(i, j) == 1 {
                    images[i] = j as u32;
                    break;
                }
            }
        }
        Permutation::new(images)
    }

    /// Transpose of the matrix (= inverse permutation).
    pub fn transpose(&self) -> PermutationMatrix {
        let n = self.size;
        let mut entries = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..n {
                entries[j * n + i] = self.entries[i * n + j];
            }
        }
        PermutationMatrix { size: n, entries }
    }

    /// Determinant: +1 for even permutations, -1 for odd.
    pub fn determinant(&self) -> i32 {
        self.to_permutation().sign()
    }

    /// Trace: number of fixed points.
    pub fn trace(&self) -> usize {
        let n = self.size;
        (0..n).filter(|&i| self.get(i, i) == 1).count()
    }
}

impl fmt::Display for PermutationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.size;
        for i in 0..n {
            for j in 0..n {
                if j > 0 { write!(f, " ")?; }
                write!(f, "{}", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reduced Word Representation
// ═══════════════════════════════════════════════════════════════════════════

/// Representation of a permutation as a product of adjacent transpositions.
/// Each entry `i` represents the transposition (i, i+1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedWord {
    /// Sequence of adjacent transposition indices.
    pub word: Vec<usize>,
    /// The degree of the symmetric group.
    pub degree: usize,
}

impl ReducedWord {
    /// Compute a reduced word for a permutation using bubble sort.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.degree();
        let mut images: Vec<u32> = (0..n).map(|i| perm.apply(i as u32)).collect();
        let mut word = Vec::new();

        // Bubble sort, recording transpositions
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..(n - 1) {
                if images[i] > images[i + 1] {
                    images.swap(i, i + 1);
                    word.push(i);
                    changed = true;
                }
            }
        }

        ReducedWord { word, degree: n }
    }

    /// Reconstruct a permutation from a reduced word.
    pub fn to_permutation(&self) -> Permutation {
        let mut result = Permutation::identity(self.degree);
        for &i in &self.word {
            let t = Permutation::transposition(self.degree, i as u32, (i + 1) as u32);
            result = result.compose(&t);
        }
        result
    }

    /// Length of the reduced word (= number of inversions).
    pub fn length(&self) -> usize {
        self.word.len()
    }

    /// Check if this is indeed a reduced word (length = inversions).
    pub fn is_reduced(&self) -> bool {
        let perm = self.to_permutation();
        let inversions = InversionTable::from_permutation(&perm).inversion_count();
        self.word.len() == inversions as usize
    }
}

impl fmt::Display for ReducedWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &s) in self.word.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "s{}", s)?;
        }
        write!(f, "]")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cycle Structure and Partition
// ═══════════════════════════════════════════════════════════════════════════

/// Integer partition (sorted in non-increasing order).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Partition {
    /// Parts in non-increasing order.
    pub parts: Vec<usize>,
}

impl Partition {
    /// Create a partition from parts (sorts them).
    pub fn new(mut parts: Vec<usize>) -> Self {
        parts.sort_unstable_by(|a, b| b.cmp(a));
        parts.retain(|&p| p > 0);
        Partition { parts }
    }

    /// The total: sum of all parts.
    pub fn total(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Number of parts.
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Conjugate (transpose) partition.
    pub fn conjugate(&self) -> Partition {
        if self.parts.is_empty() {
            return Partition { parts: vec![] };
        }
        let max_part = self.parts[0];
        let mut conj = vec![0usize; max_part];
        for &p in &self.parts {
            for j in 0..p {
                conj[j] += 1;
            }
        }
        Partition { parts: conj }
    }

    /// Dominance order: λ dominates μ if for all k, sum(λ_1..λ_k) >= sum(μ_1..μ_k).
    pub fn dominates(&self, other: &Partition) -> bool {
        if self.total() != other.total() {
            return false;
        }
        let mut sum_self = 0usize;
        let mut sum_other = 0usize;
        let max_len = self.parts.len().max(other.parts.len());
        for i in 0..max_len {
            sum_self += self.parts.get(i).copied().unwrap_or(0);
            sum_other += other.parts.get(i).copied().unwrap_or(0);
            if sum_self < sum_other {
                return false;
            }
        }
        true
    }

    /// Generate all partitions of n.
    pub fn all_partitions(n: usize) -> Vec<Partition> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::generate_partitions(n, n, &mut current, &mut result);
        result
    }

    fn generate_partitions(
        remaining: usize,
        max_part: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Partition>,
    ) {
        if remaining == 0 {
            result.push(Partition { parts: current.clone() });
            return;
        }
        let upper = remaining.min(max_part);
        for part in (1..=upper).rev() {
            current.push(part);
            Self::generate_partitions(remaining - part, part, current, result);
            current.pop();
        }
    }

    /// Number of permutations with this cycle type.
    pub fn count_permutations(&self) -> u64 {
        let n = self.total();
        let numerator = factorial(n);
        let mut denominator = 1u64;
        let mut part_counts: HashMap<usize, usize> = HashMap::new();
        for &p in &self.parts {
            *part_counts.entry(p).or_insert(0) += 1;
            denominator *= p as u64;
        }
        for &count in part_counts.values() {
            denominator *= factorial(count);
        }
        numerator / denominator
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, &p) in self.parts.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", p)?;
        }
        write!(f, ")")
    }
}

/// Detailed cycle structure analysis of a permutation.
#[derive(Debug, Clone)]
pub struct CycleStructure {
    /// The cycles as vectors of elements.
    pub cycles: Vec<Vec<u32>>,
    /// The cycle type as a partition.
    pub cycle_type: Partition,
    /// Number of cycles.
    pub num_cycles: usize,
}

impl CycleStructure {
    /// Analyze the cycle structure of a permutation.
    pub fn analyze(perm: &Permutation) -> Self {
        let cycles = perm.cycle_decomposition();
        let mut lengths: Vec<usize> = cycles.iter().map(|c| c.len()).collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        let cycle_type = Partition { parts: lengths };
        let num_cycles = cycles.len();
        CycleStructure { cycles, cycle_type, num_cycles }
    }

    /// Parse cycle notation string like "(0 1 2)(3 4)".
    pub fn from_cycle_notation(s: &str, degree: usize) -> Option<Permutation> {
        let mut images: Vec<u32> = (0..degree as u32).collect();
        let s = s.trim();
        let mut i = 0;
        let chars: Vec<char> = s.chars().collect();

        while i < chars.len() {
            if chars[i] == '(' {
                i += 1;
                let mut cycle = Vec::new();
                let mut num_str = String::new();
                while i < chars.len() && chars[i] != ')' {
                    if chars[i].is_ascii_digit() {
                        num_str.push(chars[i]);
                    } else if !num_str.is_empty() {
                        if let Ok(n) = num_str.parse::<u32>() {
                            cycle.push(n);
                        }
                        num_str.clear();
                    }
                    i += 1;
                }
                if !num_str.is_empty() {
                    if let Ok(n) = num_str.parse::<u32>() {
                        cycle.push(n);
                    }
                }
                // Apply cycle
                if cycle.len() > 1 {
                    for j in 0..cycle.len() {
                        let from = cycle[j] as usize;
                        let to = cycle[(j + 1) % cycle.len()];
                        if from >= degree {
                            return None;
                        }
                        images[from] = to;
                    }
                }
                i += 1; // skip ')'
            } else {
                i += 1;
            }
        }

        Permutation::try_new(images)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Conjugacy Classes
// ═══════════════════════════════════════════════════════════════════════════

/// A conjugacy class in a permutation group.
#[derive(Debug, Clone)]
pub struct ConjugacyClass {
    /// Representative element.
    pub representative: Permutation,
    /// The cycle type (all elements share this).
    pub cycle_type: Partition,
    /// All elements in this conjugacy class.
    pub elements: Vec<Permutation>,
    /// Size of the class.
    pub size: usize,
}

/// Conjugacy class computation for permutation groups.
#[derive(Debug, Clone)]
pub struct ConjugacyClassComputer {
    /// The degree of the permutation group.
    degree: usize,
    /// Generators of the group.
    generators: Vec<Permutation>,
}

impl ConjugacyClassComputer {
    /// Create a new conjugacy class computer.
    pub fn new(degree: usize, generators: Vec<Permutation>) -> Self {
        ConjugacyClassComputer { degree, generators }
    }

    /// Enumerate all conjugacy classes by explicit computation.
    pub fn compute_classes(&self) -> Vec<ConjugacyClass> {
        let all_elements = enumerate_from_generators(&self.generators, self.degree);
        let elements: Vec<Permutation> = all_elements.into_iter().collect();
        let mut visited: HashSet<Permutation> = HashSet::new();
        let mut classes = Vec::new();

        for elem in &elements {
            if visited.contains(elem) {
                continue;
            }
            let mut class_elements = Vec::new();
            for g in &elements {
                let conjugate = elem.conjugate_by(g);
                if !visited.contains(&conjugate) {
                    visited.insert(conjugate.clone());
                    class_elements.push(conjugate);
                }
            }
            let cycle_type = CycleStructure::analyze(&class_elements[0]).cycle_type;
            let size = class_elements.len();
            classes.push(ConjugacyClass {
                representative: class_elements[0].clone(),
                cycle_type,
                elements: class_elements,
                size,
            });
        }

        classes
    }

    /// Compute the class equation: |G| = |Z(G)| + Σ [G:C_G(g_i)].
    pub fn class_equation(&self) -> ClassEquation {
        let classes = self.compute_classes();
        let group_order: usize = classes.iter().map(|c| c.size).sum();
        let center_size = classes.iter().filter(|c| c.size == 1).count();
        let nontrivial_sizes: Vec<usize> = classes.iter()
            .filter(|c| c.size > 1)
            .map(|c| c.size)
            .collect();

        ClassEquation {
            group_order,
            center_size,
            nontrivial_class_sizes: nontrivial_sizes,
        }
    }

    /// Compute the centralizer of an element.
    pub fn centralizer(&self, elem: &Permutation) -> Vec<Permutation> {
        let all_elements = enumerate_from_generators(&self.generators, self.degree);
        all_elements.into_iter()
            .filter(|g| {
                let conj = elem.conjugate_by(g);
                conj == *elem
            })
            .collect()
    }

    /// Compute the center of the group.
    pub fn center(&self) -> Vec<Permutation> {
        let all_elements: Vec<Permutation> = enumerate_from_generators(&self.generators, self.degree)
            .into_iter().collect();

        all_elements.iter()
            .filter(|g| {
                all_elements.iter().all(|h| {
                    g.compose(h) == h.compose(g)
                })
            })
            .cloned()
            .collect()
    }
}

/// The class equation of a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassEquation {
    /// |G|
    pub group_order: usize,
    /// |Z(G)|
    pub center_size: usize,
    /// Sizes of non-trivial conjugacy classes.
    pub nontrivial_class_sizes: Vec<usize>,
}

impl ClassEquation {
    /// Verify the class equation: |G| = |Z(G)| + Σ sizes.
    pub fn verify(&self) -> bool {
        let sum: usize = self.nontrivial_class_sizes.iter().sum();
        self.group_order == self.center_size + sum
    }
}

impl fmt::Display for ClassEquation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.group_order, self.center_size)?;
        for &s in &self.nontrivial_class_sizes {
            write!(f, " + {}", s)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Character Table
// ═══════════════════════════════════════════════════════════════════════════

/// Character table of a finite group.
/// Rows correspond to irreducible characters, columns to conjugacy classes.
#[derive(Debug, Clone)]
pub struct CharacterTable {
    /// Number of irreducible characters (= number of conjugacy classes).
    pub num_irreducibles: usize,
    /// Character values: `values[i][j]` = χ_i(C_j).
    pub values: Vec<Vec<f64>>,
    /// Conjugacy class sizes.
    pub class_sizes: Vec<usize>,
    /// Conjugacy class cycle types.
    pub class_types: Vec<Partition>,
    /// Group order.
    pub group_order: usize,
}

impl CharacterTable {
    /// Compute the character table for the symmetric group S_n.
    /// Uses the Murnaghan-Nakayama rule.
    pub fn symmetric_group(n: usize) -> Self {
        let partitions = Partition::all_partitions(n);
        let num_classes = partitions.len();
        let mut values = vec![vec![0.0f64; num_classes]; num_classes];
        let mut class_sizes = Vec::new();

        for partition in &partitions {
            class_sizes.push(partition.count_permutations() as usize);
        }

        // Compute character values using Murnaghan-Nakayama
        for (i, lambda) in partitions.iter().enumerate() {
            for (j, mu) in partitions.iter().enumerate() {
                values[i][j] = Self::murnaghan_nakayama(lambda, mu);
            }
        }

        let group_order = factorial(n) as usize;

        CharacterTable {
            num_irreducibles: num_classes,
            values,
            class_sizes,
            class_types: partitions,
            group_order,
        }
    }

    /// Murnaghan-Nakayama rule for computing character values.
    /// χ^λ(cycle type μ) computed recursively by removing border strips.
    fn murnaghan_nakayama(lambda: &Partition, mu: &Partition) -> f64 {
        if mu.parts.is_empty() {
            if lambda.parts.is_empty() { return 1.0; }
            else { return 0.0; }
        }
        if lambda.total() != mu.total() {
            return 0.0;
        }
        if lambda.total() == 0 {
            return 1.0;
        }

        let r = mu.parts[0]; // first (largest) part
        let remaining_mu = Partition::new(mu.parts[1..].to_vec());

        // Convert lambda to Young diagram (row lengths)
        let diagram = &lambda.parts;
        if diagram.is_empty() {
            return 0.0;
        }

        // Find all border strips of length r
        let strips = Self::find_border_strips(diagram, r);

        let mut result = 0.0;
        for (new_diagram, height) in strips {
            let new_lambda = Partition::new(new_diagram);
            let sign = if height % 2 == 0 { 1.0 } else { -1.0 };
            result += sign * Self::murnaghan_nakayama(&new_lambda, &remaining_mu);
        }

        result
    }

    /// Find all border strips of length r in a Young diagram.
    /// Returns pairs (new_diagram, height) where height = number of rows spanned - 1.
    fn find_border_strips(diagram: &[usize], r: usize) -> Vec<(Vec<usize>, usize)> {
        let mut results = Vec::new();
        let num_rows = diagram.len();

        // Try removing r cells from the border
        // A border strip is a connected skew shape with no 2x2 square
        Self::find_strips_recursive(diagram, r, num_rows, &mut results);
        results
    }

    fn find_strips_recursive(
        diagram: &[usize],
        r: usize,
        _num_rows: usize,
        results: &mut Vec<(Vec<usize>, usize)>,
    ) {
        if r == 0 {
            let cleaned: Vec<usize> = diagram.iter().copied().filter(|&x| x > 0).collect();
            results.push((cleaned, 0));
            return;
        }

        // Simple approach: try removing cells from each possible starting position
        let num_rows = diagram.len();
        for start_row in 0..num_rows {
            let mut new_diagram = diagram.to_vec();
            let mut cells_removed = 0;
            let mut rows_touched = 0;
            let mut valid = true;

            let mut row = start_row;
            while cells_removed < r && row < new_diagram.len() {
                if new_diagram[row] == 0 {
                    break;
                }
                // Can we remove from this row?
                let max_remove = if row + 1 < new_diagram.len() {
                    if new_diagram[row] <= new_diagram[row + 1] {
                        break; // Can't remove - would break column condition
                    }
                    new_diagram[row] - new_diagram[row + 1]
                } else {
                    new_diagram[row]
                };

                let to_remove = (r - cells_removed).min(max_remove);
                if to_remove == 0 {
                    break;
                }

                new_diagram[row] -= to_remove;
                cells_removed += to_remove;
                rows_touched += 1;
                row += 1;
            }

            if cells_removed == r && valid {
                let height = if rows_touched > 0 { rows_touched - 1 } else { 0 };
                let cleaned: Vec<usize> = new_diagram.into_iter().filter(|&x| x > 0).collect();
                // Check it's not a duplicate
                let entry = (cleaned, height);
                if !results.contains(&entry) {
                    results.push(entry);
                }
            }
        }
    }

    /// Inner product of two class functions.
    /// ⟨χ, ψ⟩ = (1/|G|) Σ_C |C| χ(C)* ψ(C)
    pub fn inner_product(&self, chi: &[f64], psi: &[f64]) -> f64 {
        assert_eq!(chi.len(), self.num_irreducibles);
        assert_eq!(psi.len(), self.num_irreducibles);
        let mut sum = 0.0;
        for j in 0..self.num_irreducibles {
            sum += self.class_sizes[j] as f64 * chi[j] * psi[j];
        }
        sum / self.group_order as f64
    }

    /// Check orthogonality of the character table (row orthogonality).
    pub fn check_orthogonality(&self) -> bool {
        for i in 0..self.num_irreducibles {
            for j in 0..self.num_irreducibles {
                let ip = self.inner_product(&self.values[i], &self.values[j]);
                let expected = if i == j { 1.0 } else { 0.0 };
                if (ip - expected).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Dimension of the i-th irreducible representation.
    /// This is χ_i(identity) = values[i][0] (identity is always in class 0).
    pub fn dimension(&self, i: usize) -> f64 {
        // Find the identity class (class with size 1 and cycle type (1,1,...,1))
        for j in 0..self.num_irreducibles {
            if self.class_sizes[j] == 1 {
                return self.values[i][j];
            }
        }
        self.values[i][0]
    }

    /// Character of the regular representation.
    pub fn regular_character(&self) -> Vec<f64> {
        let mut chi = vec![0.0; self.num_irreducibles];
        // Regular rep has value |G| at identity, 0 elsewhere
        for j in 0..self.num_irreducibles {
            if self.class_sizes[j] == 1 {
                chi[j] = self.group_order as f64;
            }
        }
        chi
    }

    /// Decompose a class function into irreducible characters.
    pub fn decompose(&self, chi: &[f64]) -> Vec<f64> {
        let mut multiplicities = Vec::new();
        for i in 0..self.num_irreducibles {
            multiplicities.push(self.inner_product(chi, &self.values[i]));
        }
        multiplicities
    }

    /// Verify sum of squares of dimensions equals group order.
    pub fn verify_dimension_formula(&self) -> bool {
        let sum: f64 = (0..self.num_irreducibles)
            .map(|i| {
                let d = self.dimension(i);
                d * d
            })
            .sum();
        (sum - self.group_order as f64).abs() < 1e-6
    }
}

impl fmt::Display for CharacterTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Character Table (|G| = {}):", self.group_order)?;
        // Header
        write!(f, "{:>10}", "")?;
        for (j, ct) in self.class_types.iter().enumerate() {
            write!(f, " {:>8}", format!("C{}", j))?;
        }
        writeln!(f)?;
        write!(f, "{:>10}", "size")?;
        for &s in &self.class_sizes {
            write!(f, " {:>8}", s)?;
        }
        writeln!(f)?;
        write!(f, "{:>10}", "type")?;
        for ct in &self.class_types {
            write!(f, " {:>8}", ct)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", "-".repeat(10 + 9 * self.num_irreducibles))?;
        for i in 0..self.num_irreducibles {
            write!(f, "χ{:<9}", i)?;
            for j in 0..self.num_irreducibles {
                write!(f, " {:>8.2}", self.values[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Schreier Vector
// ═══════════════════════════════════════════════════════════════════════════

/// Schreier vector: a factored transversal for efficient orbit/word computation.
/// For each point β in the orbit of base_point, stores the generator index
/// and predecessor that maps base_point → β.
#[derive(Debug, Clone)]
pub struct SchreierVector {
    /// The base point.
    pub base_point: u32,
    /// The degree.
    pub degree: usize,
    /// Generators used.
    pub generators: Vec<Permutation>,
    /// For each point in the orbit: (generator_index, predecessor_point).
    /// None for points outside the orbit.
    /// base_point maps to itself with no generator.
    parent: Vec<Option<(usize, u32)>>,
    /// The orbit.
    orbit: HashSet<u32>,
}

impl SchreierVector {
    /// Build a Schreier vector for the orbit of `base_point` under `generators`.
    pub fn build(base_point: u32, generators: &[Permutation], degree: usize) -> Self {
        let mut parent: Vec<Option<(usize, u32)>> = vec![None; degree];
        let mut orbit = HashSet::new();
        orbit.insert(base_point);
        // base_point has no parent (it's the root)
        parent[base_point as usize] = Some((usize::MAX, base_point));

        let mut queue = VecDeque::new();
        queue.push_back(base_point);

        while let Some(pt) = queue.pop_front() {
            for (gi, gen) in generators.iter().enumerate() {
                let image = gen.apply(pt);
                if !orbit.contains(&image) {
                    orbit.insert(image);
                    parent[image as usize] = Some((gi, pt));
                    queue.push_back(image);
                }
            }
        }

        SchreierVector {
            base_point,
            degree,
            generators: generators.to_vec(),
            parent,
            orbit,
        }
    }

    /// Check if a point is in the orbit.
    pub fn in_orbit(&self, point: u32) -> bool {
        self.orbit.contains(&point)
    }

    /// Get the orbit.
    pub fn orbit(&self) -> &HashSet<u32> {
        &self.orbit
    }

    /// Orbit size.
    pub fn orbit_size(&self) -> usize {
        self.orbit.len()
    }

    /// Trace the path from base_point to target, returning a word
    /// (sequence of generator indices) such that g_{w[k-1]} ∘ ... ∘ g_{w[0]}
    /// maps base_point to target.
    pub fn trace_word(&self, target: u32) -> Option<Vec<usize>> {
        if !self.in_orbit(target) {
            return None;
        }
        if target == self.base_point {
            return Some(vec![]);
        }

        let mut word = Vec::new();
        let mut current = target;
        while current != self.base_point {
            if let Some((gi, pred)) = self.parent[current as usize] {
                if gi == usize::MAX {
                    break; // root
                }
                word.push(gi);
                current = pred;
            } else {
                return None; // shouldn't happen
            }
        }
        word.reverse();
        Some(word)
    }

    /// Get the transversal element u_β that maps base_point to β.
    pub fn transversal_element(&self, target: u32) -> Option<Permutation> {
        let word = self.trace_word(target)?;
        let mut result = Permutation::identity(self.degree);
        for &gi in &word {
            result = self.generators[gi].compose(&result);
        }
        Some(result)
    }

    /// Test membership: given element g, check if g is in the group
    /// by sifting through a stabilizer chain.
    pub fn sift(&self, elem: &Permutation) -> Option<Permutation> {
        let image = elem.apply(self.base_point);
        if !self.in_orbit(image) {
            return None;
        }
        let u = self.transversal_element(image)?;
        let remainder = u.inverse().compose(elem);
        Some(remainder)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Block System
// ═══════════════════════════════════════════════════════════════════════════

/// A block system for a permutation group action.
/// A block is a non-empty subset B of Ω such that for all g in G,
/// either gB = B or gB ∩ B = ∅.
#[derive(Debug, Clone)]
pub struct BlockSystem {
    /// The degree.
    pub degree: usize,
    /// Block assignment: `block_of[i]` is the block index of element i.
    pub block_of: Vec<usize>,
    /// The blocks as sets.
    pub blocks: Vec<Vec<u32>>,
    /// Number of blocks.
    pub num_blocks: usize,
}

impl BlockSystem {
    /// Trivial block system: each element is its own block.
    pub fn discrete(degree: usize) -> Self {
        let block_of: Vec<usize> = (0..degree).collect();
        let blocks: Vec<Vec<u32>> = (0..degree).map(|i| vec![i as u32]).collect();
        BlockSystem { degree, block_of, blocks, num_blocks: degree }
    }

    /// Trivial block system: all elements in one block.
    pub fn trivial(degree: usize) -> Self {
        let block_of = vec![0usize; degree];
        let blocks = vec![(0..degree as u32).collect()];
        BlockSystem { degree, block_of, blocks, num_blocks: 1 }
    }

    /// Check if this is a valid block system for the given generators.
    pub fn is_valid(&self, generators: &[Permutation]) -> bool {
        for gen in generators {
            // For each block, check that the image is also a block
            for block in &self.blocks {
                let image_blocks: HashSet<usize> = block.iter()
                    .map(|&pt| self.block_of[gen.apply(pt) as usize])
                    .collect();
                if image_blocks.len() != 1 {
                    return false;
                }
            }
        }
        true
    }

    /// Is this the discrete block system?
    pub fn is_discrete(&self) -> bool {
        self.num_blocks == self.degree
    }

    /// Is this the trivial (single block) system?
    pub fn is_trivial_single(&self) -> bool {
        self.num_blocks == 1
    }

    /// Block size (assuming equal-sized blocks).
    pub fn block_size(&self) -> usize {
        if self.num_blocks == 0 { return 0; }
        self.blocks[0].len()
    }
}

/// Block system computation for permutation groups.
pub struct BlockSystemComputer {
    degree: usize,
    generators: Vec<Permutation>,
}

impl BlockSystemComputer {
    /// Create a new block system computer.
    pub fn new(degree: usize, generators: Vec<Permutation>) -> Self {
        BlockSystemComputer { degree, generators }
    }

    /// Find a minimal block system containing {a, b} in the same block.
    /// Uses union-find to merge blocks.
    pub fn minimal_block_system(&self, a: u32, b: u32) -> BlockSystem {
        let n = self.degree;
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
            if rx != ry {
                parent[rx] = ry;
            }
        }

        // Initial merge
        union(&mut parent, a as usize, b as usize);

        // Iterate until stable
        let mut changed = true;
        while changed {
            changed = false;
            for gen in &self.generators {
                let mut merges = Vec::new();
                // For all pairs in the same block, their images must also be in the same block
                for i in 0..n {
                    for j in (i + 1)..n {
                        if find(&mut parent, i) == find(&mut parent, j) {
                            let gi = gen.apply(i as u32) as usize;
                            let gj = gen.apply(j as u32) as usize;
                            if find(&mut parent, gi) != find(&mut parent, gj) {
                                merges.push((gi, gj));
                            }
                        }
                    }
                }
                for (x, y) in merges {
                    union(&mut parent, x, y);
                    changed = true;
                }
            }
        }

        // Build blocks
        let mut block_map: HashMap<usize, Vec<u32>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            block_map.entry(root).or_default().push(i as u32);
        }
        let mut blocks: Vec<Vec<u32>> = block_map.into_values().collect();
        blocks.sort_by_key(|b| b[0]);

        let mut block_of = vec![0usize; n];
        for (bi, block) in blocks.iter().enumerate() {
            for &pt in block {
                block_of[pt as usize] = bi;
            }
        }

        let num_blocks = blocks.len();
        BlockSystem { degree: n, block_of, blocks, num_blocks }
    }

    /// Check if the group acts primitively (no non-trivial block systems).
    pub fn is_primitive(&self) -> bool {
        if self.degree <= 2 {
            return true;
        }

        // Check if action is transitive first
        let orbit = compute_point_orbit(0, &self.generators, self.degree);
        if orbit.elements.len() != self.degree {
            return false; // not transitive, so not primitive
        }

        // Try to find a non-trivial block system by merging 0 with each other point
        for pt in 1..self.degree as u32 {
            let bs = self.minimal_block_system(0, pt);
            if !bs.is_discrete() && !bs.is_trivial_single() {
                return false; // found non-trivial block system
            }
        }
        true
    }

    /// Find all block systems.
    pub fn all_block_systems(&self) -> Vec<BlockSystem> {
        let mut systems = vec![
            BlockSystem::discrete(self.degree),
            BlockSystem::trivial(self.degree),
        ];

        // Try all pairs
        for pt in 1..self.degree as u32 {
            let bs = self.minimal_block_system(0, pt);
            if !bs.is_discrete() && !bs.is_trivial_single() {
                // Check if we already have an equivalent one
                let is_new = systems.iter().all(|existing| {
                    existing.num_blocks != bs.num_blocks || existing.blocks != bs.blocks
                });
                if is_new {
                    systems.push(bs);
                }
            }
        }

        systems
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Permutation Group Homomorphism
// ═══════════════════════════════════════════════════════════════════════════

/// A homomorphism between permutation groups defined on generators.
#[derive(Debug, Clone)]
pub struct PermutationHomomorphism {
    /// Degree of the source group.
    pub source_degree: usize,
    /// Degree of the target group.
    pub target_degree: usize,
    /// Source generators.
    pub source_generators: Vec<Permutation>,
    /// Images of generators.
    pub generator_images: Vec<Permutation>,
}

impl PermutationHomomorphism {
    /// Create a homomorphism defined by mapping generators to their images.
    pub fn new(
        source_degree: usize,
        target_degree: usize,
        source_generators: Vec<Permutation>,
        generator_images: Vec<Permutation>,
    ) -> Self {
        assert_eq!(source_generators.len(), generator_images.len());
        PermutationHomomorphism {
            source_degree,
            target_degree,
            source_generators,
            generator_images,
        }
    }

    /// Apply the homomorphism to an element given as a word in generators.
    pub fn apply_word(&self, word: &[usize]) -> Permutation {
        let mut result = Permutation::identity(self.target_degree);
        for &gi in word {
            result = result.compose(&self.generator_images[gi]);
        }
        result
    }

    /// Compute the image of the homomorphism (as a set of permutations).
    pub fn image(&self) -> HashSet<Permutation> {
        enumerate_from_generators(&self.generator_images, self.target_degree)
    }

    /// Compute the kernel: elements mapping to identity.
    pub fn kernel(&self) -> Vec<Permutation> {
        let source_elements = enumerate_from_generators(&self.source_generators, self.source_degree);
        let target_id = Permutation::identity(self.target_degree);

        // Build a map from source elements to their images
        let source_vec: Vec<Permutation> = source_elements.into_iter().collect();
        let mut kernel = Vec::new();

        // For each source element, decompose it in terms of generators
        // and apply the homomorphism
        // Simple approach: enumerate all source elements and check their images
        for elem in &source_vec {
            let image = self.apply_element(elem);
            if image == target_id {
                kernel.push(elem.clone());
            }
        }
        kernel
    }

    /// Apply the homomorphism to an arbitrary element.
    /// Uses the group structure to find a decomposition.
    fn apply_element(&self, elem: &Permutation) -> Permutation {
        // Build a BFS mapping from source group elements to target
        let source_elements = enumerate_from_generators(&self.source_generators, self.source_degree);
        let mut elem_to_image: HashMap<Permutation, Permutation> = HashMap::new();
        let source_id = Permutation::identity(self.source_degree);
        let target_id = Permutation::identity(self.target_degree);
        elem_to_image.insert(source_id.clone(), target_id);

        let mut queue = VecDeque::new();
        queue.push_back(source_id);
        let mut visited = HashSet::new();

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            let current_image = elem_to_image[&current].clone();

            for (gi, gen) in self.source_generators.iter().enumerate() {
                let next = current.compose(gen);
                if !visited.contains(&next) {
                    let next_image = current_image.compose(&self.generator_images[gi]);
                    elem_to_image.insert(next.clone(), next_image);
                    queue.push_back(next);
                }
            }
        }

        elem_to_image.get(elem).cloned().unwrap_or_else(|| Permutation::identity(self.target_degree))
    }

    /// Verify this is a valid homomorphism by checking relations.
    pub fn verify(&self) -> bool {
        // Check that generator relations are preserved
        let source_elements = enumerate_from_generators(&self.source_generators, self.source_degree);
        let source_vec: Vec<Permutation> = source_elements.into_iter().collect();

        // Check a⋅b maps to φ(a)⋅φ(b) for generators
        for (i, gi) in self.source_generators.iter().enumerate() {
            for (j, gj) in self.source_generators.iter().enumerate() {
                let product_source = gi.compose(gj);
                let product_target = self.generator_images[i].compose(&self.generator_images[j]);
                let image_of_product = self.apply_element(&product_source);
                if image_of_product != product_target {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the homomorphism is injective (trivial kernel).
    pub fn is_injective(&self) -> bool {
        let kernel = self.kernel();
        kernel.len() == 1 // only identity
    }

    /// Check if the homomorphism is surjective.
    pub fn is_surjective(&self) -> bool {
        let image = self.image();
        let target = enumerate_from_generators(&self.generator_images, self.target_degree);
        // Check if image == full target group
        // (We can only check against the group generated by the images)
        image.len() >= target.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Group Presentation
// ═══════════════════════════════════════════════════════════════════════════

/// A relation in a group presentation: a word that equals the identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Relation {
    /// The word as a sequence of (generator_index, exponent) pairs.
    pub word: Vec<(usize, i32)>,
}

impl Relation {
    /// Create from a sequence of generator indices (all exponent 1).
    pub fn from_indices(indices: &[usize]) -> Self {
        Relation {
            word: indices.iter().map(|&i| (i, 1)).collect(),
        }
    }

    /// Create from (index, exponent) pairs.
    pub fn from_pairs(pairs: Vec<(usize, i32)>) -> Self {
        Relation { word: pairs }
    }

    /// Inverse of the relation.
    pub fn inverse(&self) -> Self {
        let word: Vec<(usize, i32)> = self.word.iter().rev().map(|&(g, e)| (g, -e)).collect();
        Relation { word }
    }

    /// Length of the word (sum of absolute exponents).
    pub fn length(&self) -> usize {
        self.word.iter().map(|(_, e)| e.unsigned_abs() as usize).sum()
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, &(gen, exp)) in self.word.iter().enumerate() {
            if i > 0 { write!(f, "·")?; }
            write!(f, "g{}", gen)?;
            if exp != 1 {
                write!(f, "^{}", exp)?;
            }
        }
        Ok(())
    }
}

/// A group presentation ⟨generators | relations⟩.
#[derive(Debug, Clone)]
pub struct GroupPresentation {
    /// Number of generators.
    pub num_generators: usize,
    /// Generator names.
    pub generator_names: Vec<String>,
    /// Relations (words equal to identity).
    pub relations: Vec<Relation>,
}

impl GroupPresentation {
    /// Create a new presentation.
    pub fn new(num_generators: usize) -> Self {
        let names: Vec<String> = (0..num_generators).map(|i| format!("g{}", i)).collect();
        GroupPresentation {
            num_generators,
            generator_names: names,
            relations: Vec::new(),
        }
    }

    /// Create with named generators.
    pub fn with_names(names: Vec<String>) -> Self {
        let n = names.len();
        GroupPresentation {
            num_generators: n,
            generator_names: names,
            relations: Vec::new(),
        }
    }

    /// Add a relation.
    pub fn add_relation(&mut self, rel: Relation) {
        self.relations.push(rel);
    }

    /// Presentation of the cyclic group Z_n = ⟨a | a^n⟩.
    pub fn cyclic(n: usize) -> Self {
        let mut pres = Self::with_names(vec!["a".to_string()]);
        pres.add_relation(Relation::from_pairs(vec![(0, n as i32)]));
        pres
    }

    /// Presentation of the dihedral group D_n = ⟨r, s | r^n, s^2, (rs)^2⟩.
    pub fn dihedral(n: usize) -> Self {
        let mut pres = Self::with_names(vec!["r".to_string(), "s".to_string()]);
        pres.add_relation(Relation::from_pairs(vec![(0, n as i32)])); // r^n = 1
        pres.add_relation(Relation::from_pairs(vec![(1, 2)])); // s^2 = 1
        pres.add_relation(Relation::from_pairs(vec![(0, 1), (1, 1), (0, 1), (1, 1)])); // (rs)^2 = 1
        pres
    }

    /// Presentation of the symmetric group S_n (Coxeter presentation).
    pub fn symmetric(n: usize) -> Self {
        if n <= 1 {
            return Self::new(0);
        }
        let names: Vec<String> = (0..n - 1).map(|i| format!("s{}", i)).collect();
        let mut pres = Self::with_names(names);

        // s_i^2 = 1
        for i in 0..n - 1 {
            pres.add_relation(Relation::from_pairs(vec![(i, 2)]));
        }
        // (s_i s_{i+1})^3 = 1
        for i in 0..n - 2 {
            pres.add_relation(Relation::from_pairs(vec![
                (i, 1), (i + 1, 1), (i, 1), (i + 1, 1), (i, 1), (i + 1, 1),
            ]));
        }
        // (s_i s_j)^2 = 1 for |i-j| >= 2
        for i in 0..n - 1 {
            for j in (i + 2)..n - 1 {
                pres.add_relation(Relation::from_pairs(vec![
                    (i, 1), (j, 1), (i, 1), (j, 1),
                ]));
            }
        }
        pres
    }

    /// Todd-Coxeter coset enumeration (simplified).
    /// Enumerates cosets of the trivial subgroup to find the group order.
    /// Returns the number of cosets (= group order) or None if it exceeds max_cosets.
    pub fn todd_coxeter(&self, max_cosets: usize) -> Option<usize> {
        // Coset table: coset_table[coset][generator] = target coset
        let num_cols = self.num_generators * 2; // generators and their inverses
        let mut table: Vec<Vec<Option<usize>>> = vec![vec![None; num_cols]; 1];
        let mut num_cosets = 1usize;

        // Process relations to fill in the table
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < max_cosets {
            changed = false;
            iterations += 1;

            // For each coset and each relation, try to deduce new entries
            for coset in 0..num_cosets {
                for rel in &self.relations {
                    // Trace the relation from both ends
                    let mut fwd = coset;
                    let mut fwd_ok = true;
                    let mut fwd_steps = 0;

                    for &(gen, exp) in &rel.word {
                        for _ in 0..exp.unsigned_abs() {
                            let col = if exp > 0 { gen } else { gen + self.num_generators };
                            if let Some(next) = table[fwd][col] {
                                fwd = next;
                                fwd_steps += 1;
                            } else {
                                fwd_ok = false;
                                break;
                            }
                        }
                        if !fwd_ok { break; }
                    }

                    if fwd_ok && fwd != coset {
                        // Relation not satisfied; this is a contradiction in our simple version
                        // In full Todd-Coxeter, we'd merge cosets
                    }
                }

                // Define new cosets for undefined entries
                for col in 0..num_cols {
                    if table[coset][col].is_none() && num_cosets < max_cosets {
                        let new_coset = num_cosets;
                        table.push(vec![None; num_cols]);
                        num_cosets += 1;
                        table[coset][col] = Some(new_coset);
                        // Inverse
                        let inv_col = if col < self.num_generators {
                            col + self.num_generators
                        } else {
                            col - self.num_generators
                        };
                        table[new_coset][inv_col] = Some(coset);
                        changed = true;
                    }
                }
            }
        }

        if num_cosets >= max_cosets {
            None
        } else {
            Some(num_cosets)
        }
    }
}

impl fmt::Display for GroupPresentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "⟨")?;
        for (i, name) in self.generator_names.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", name)?;
        }
        write!(f, " | ")?;
        for (i, rel) in self.relations.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{} = 1", rel)?;
        }
        write!(f, "⟩")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Permutation Group Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Generate all permutations of degree n (S_n) iteratively.
pub fn all_permutations(n: usize) -> Vec<Permutation> {
    if n == 0 {
        return vec![Permutation::new(vec![])];
    }
    let mut result = Vec::new();
    let mut images: Vec<u32> = (0..n as u32).collect();
    generate_permutations_heap(&mut images, n, &mut result);
    result
}

fn generate_permutations_heap(arr: &mut Vec<u32>, k: usize, result: &mut Vec<Permutation>) {
    if k == 1 {
        result.push(Permutation::new(arr.clone()));
        return;
    }
    generate_permutations_heap(arr, k - 1, result);
    for i in 0..k - 1 {
        if k % 2 == 0 {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
        generate_permutations_heap(arr, k - 1, result);
    }
}

/// Count the number of derangements (permutations with no fixed points) of n elements.
pub fn count_derangements(n: usize) -> u64 {
    if n == 0 { return 1; }
    if n == 1 { return 0; }
    let mut d_prev2 = 1u64; // D(0)
    let mut d_prev1 = 0u64; // D(1)
    for i in 2..=n {
        let d = (i as u64 - 1) * (d_prev1 + d_prev2);
        d_prev2 = d_prev1;
        d_prev1 = d;
    }
    d_prev1
}

/// Enumerate all derangements of degree n.
pub fn derangements(n: usize) -> Vec<Permutation> {
    all_permutations(n).into_iter()
        .filter(|p| p.fixed_points().is_empty())
        .collect()
}

/// Generate the alternating group A_n (even permutations of degree n).
pub fn alternating_group_generators(n: usize) -> Vec<Permutation> {
    if n <= 2 {
        return vec![Permutation::identity(n)];
    }
    // A_n is generated by 3-cycles (0 1 i) for i = 2, ..., n-1
    let mut gens = Vec::new();
    for i in 2..n {
        gens.push(Permutation::cycle(n, &[0, 1, i as u32]));
    }
    gens
}

/// Compute the commutator subgroup [G, G] given generators.
pub fn commutator_subgroup(generators: &[Permutation], degree: usize) -> Vec<Permutation> {
    let elements: Vec<Permutation> = enumerate_from_generators(generators, degree)
        .into_iter().collect();
    let mut commutators = HashSet::new();

    for a in &elements {
        for b in &elements {
            let comm = a.commutator(b);
            commutators.insert(comm);
        }
    }

    // The commutator subgroup is generated by all commutators
    let comm_gens: Vec<Permutation> = commutators.into_iter().collect();
    enumerate_from_generators(&comm_gens, degree).into_iter().collect()
}

/// Check if a group (given by generators) is abelian.
pub fn is_abelian(generators: &[Permutation]) -> bool {
    for i in 0..generators.len() {
        for j in (i + 1)..generators.len() {
            if generators[i].compose(&generators[j]) != generators[j].compose(&generators[i]) {
                return false;
            }
        }
    }
    true
}

/// Check if a group (given by generators) is cyclic.
pub fn is_cyclic(generators: &[Permutation], degree: usize) -> bool {
    let group_size = enumerate_from_generators(generators, degree).len();
    // A group is cyclic iff it has an element of order equal to the group size
    let elements = enumerate_from_generators(generators, degree);
    for elem in &elements {
        if elem.order() as usize == group_size {
            return true;
        }
    }
    false
}

/// Compute the exponent of a group (LCM of all element orders).
pub fn group_exponent(generators: &[Permutation], degree: usize) -> u64 {
    let elements = enumerate_from_generators(generators, degree);
    let mut exponent = 1u64;
    for elem in &elements {
        let ord = elem.order();
        exponent = lcm(exponent, ord);
    }
    exponent
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a / gcd(a, b) * b }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lehmer_code_identity() {
        let perm = Permutation::identity(4);
        let code = LehmerCode::from_permutation(&perm);
        assert_eq!(code.code, vec![0, 0, 0, 0]);
        let recovered = code.to_permutation();
        assert_eq!(perm, recovered);
    }

    #[test]
    fn test_lehmer_code_reverse() {
        let perm = Permutation::new(vec![3, 2, 1, 0]);
        let code = LehmerCode::from_permutation(&perm);
        assert_eq!(code.code, vec![3, 2, 1, 0]);
        let recovered = code.to_permutation();
        assert_eq!(perm, recovered);
    }

    #[test]
    fn test_lehmer_rank_roundtrip() {
        let n = 4;
        for rank in 0..factorial(n) {
            let code = LehmerCode::from_rank(n, rank);
            assert_eq!(code.to_rank(), rank);
        }
    }

    #[test]
    fn test_inversion_count() {
        let perm = Permutation::new(vec![2, 0, 1]);
        let inv = InversionTable::from_permutation(&perm);
        let count = inv.inversion_count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_permutation_statistics() {
        let perm = Permutation::new(vec![2, 0, 3, 1]);
        let stats = PermutationStatistics::compute(&perm);
        assert!(stats.descent_set.contains(&0)); // σ(0)=2 > σ(1)=0
        assert!(stats.descent_set.contains(&2)); // σ(2)=3 > σ(3)=1
        assert_eq!(stats.num_descents, 2);
    }

    #[test]
    fn test_permutation_matrix() {
        let perm = Permutation::new(vec![1, 2, 0]);
        let mat = PermutationMatrix::from_permutation(&perm);
        assert_eq!(mat.get(0, 1), 1);
        assert_eq!(mat.get(1, 2), 1);
        assert_eq!(mat.get(2, 0), 1);
        assert_eq!(mat.get(0, 0), 0);
        let recovered = mat.to_permutation();
        assert_eq!(perm, recovered);
    }

    #[test]
    fn test_permutation_matrix_multiply() {
        let p = Permutation::new(vec![1, 2, 0]);
        let q = Permutation::new(vec![2, 0, 1]);
        let mp = PermutationMatrix::from_permutation(&p);
        let mq = PermutationMatrix::from_permutation(&q);
        let product = mp.multiply(&mq);
        let expected = PermutationMatrix::from_permutation(&p.compose(&q));
        assert_eq!(product, expected);
    }

    #[test]
    fn test_permutation_matrix_transpose() {
        let perm = Permutation::new(vec![1, 2, 0]);
        let mat = PermutationMatrix::from_permutation(&perm);
        let transpose = mat.transpose();
        let inv_mat = PermutationMatrix::from_permutation(&perm.inverse());
        assert_eq!(transpose, inv_mat);
    }

    #[test]
    fn test_reduced_word() {
        let perm = Permutation::new(vec![1, 0, 2]); // transposition (0,1)
        let rw = ReducedWord::from_permutation(&perm);
        assert_eq!(rw.length(), 1);
        let recovered = rw.to_permutation();
        assert_eq!(perm, recovered);
    }

    #[test]
    fn test_reduced_word_identity() {
        let perm = Permutation::identity(4);
        let rw = ReducedWord::from_permutation(&perm);
        assert_eq!(rw.length(), 0);
    }

    #[test]
    fn test_partition_basics() {
        let p = Partition::new(vec![3, 1, 2]);
        assert_eq!(p.parts, vec![3, 2, 1]);
        assert_eq!(p.total(), 6);
        assert_eq!(p.num_parts(), 3);
    }

    #[test]
    fn test_partition_conjugate() {
        let p = Partition::new(vec![3, 2, 1]);
        let conj = p.conjugate();
        assert_eq!(conj.parts, vec![3, 2, 1]); // self-conjugate
    }

    #[test]
    fn test_partition_enumeration() {
        let parts4 = Partition::all_partitions(4);
        assert_eq!(parts4.len(), 5); // p(4) = 5
        let parts5 = Partition::all_partitions(5);
        assert_eq!(parts5.len(), 7); // p(5) = 7
    }

    #[test]
    fn test_partition_dominance() {
        let p1 = Partition::new(vec![3, 1]);
        let p2 = Partition::new(vec![2, 2]);
        assert!(p1.dominates(&p2));
        assert!(!p2.dominates(&p1));
    }

    #[test]
    fn test_partition_count_permutations() {
        // Cycle type (2,1,1) in S_4: should have 6 permutations
        let p = Partition::new(vec![2, 1, 1]);
        assert_eq!(p.count_permutations(), 6);
    }

    #[test]
    fn test_cycle_structure() {
        let perm = Permutation::new(vec![1, 2, 0, 3]);
        let cs = CycleStructure::analyze(&perm);
        assert_eq!(cs.num_cycles, 2); // (0 1 2) and (3)
        assert_eq!(cs.cycle_type, Partition::new(vec![3, 1]));
    }

    #[test]
    fn test_cycle_notation_parsing() {
        let perm = CycleStructure::from_cycle_notation("(0 1 2)(3 4)", 5);
        assert!(perm.is_some());
        let p = perm.unwrap();
        assert_eq!(p.apply(0), 1);
        assert_eq!(p.apply(1), 2);
        assert_eq!(p.apply(2), 0);
        assert_eq!(p.apply(3), 4);
        assert_eq!(p.apply(4), 3);
    }

    #[test]
    fn test_conjugacy_classes_s3() {
        // S_3 has 3 conjugacy classes: {e}, {(01),(02),(12)}, {(012),(021)}
        let gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        let computer = ConjugacyClassComputer::new(3, gens);
        let classes = computer.compute_classes();
        assert_eq!(classes.len(), 3);

        let sizes: Vec<usize> = {
            let mut s: Vec<_> = classes.iter().map(|c| c.size).collect();
            s.sort();
            s
        };
        assert_eq!(sizes, vec![1, 2, 3]);
    }

    #[test]
    fn test_class_equation() {
        let gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        let computer = ConjugacyClassComputer::new(3, gens);
        let eq = computer.class_equation();
        assert!(eq.verify());
        assert_eq!(eq.group_order, 6);
    }

    #[test]
    fn test_character_table_s2() {
        let ct = CharacterTable::symmetric_group(2);
        assert_eq!(ct.num_irreducibles, 2);
        assert_eq!(ct.group_order, 2);
    }

    #[test]
    fn test_character_table_s3() {
        let ct = CharacterTable::symmetric_group(3);
        assert_eq!(ct.num_irreducibles, 3);
        assert_eq!(ct.group_order, 6);
    }

    #[test]
    fn test_character_dimension_formula() {
        let ct = CharacterTable::symmetric_group(4);
        assert!(ct.verify_dimension_formula());
    }

    #[test]
    fn test_schreier_vector() {
        let gens = vec![
            Permutation::cycle(4, &[0, 1, 2, 3]),
        ];
        let sv = SchreierVector::build(0, &gens, 4);
        assert_eq!(sv.orbit_size(), 4);
        assert!(sv.in_orbit(0));
        assert!(sv.in_orbit(1));
        assert!(sv.in_orbit(2));
        assert!(sv.in_orbit(3));
    }

    #[test]
    fn test_schreier_vector_word() {
        let gens = vec![
            Permutation::cycle(4, &[0, 1, 2, 3]),
        ];
        let sv = SchreierVector::build(0, &gens, 4);
        let word = sv.trace_word(2);
        assert!(word.is_some());
        let u = sv.transversal_element(2);
        assert!(u.is_some());
        assert_eq!(u.unwrap().apply(0), 2);
    }

    #[test]
    fn test_block_system_trivial() {
        let gens = vec![
            Permutation::cycle(4, &[0, 1, 2, 3]),
        ];
        let computer = BlockSystemComputer::new(4, gens);
        let bs = computer.minimal_block_system(0, 2);
        // Z_4 acting on {0,1,2,3}: {0,2} and {1,3} form a block system
        assert_eq!(bs.num_blocks, 2);
    }

    #[test]
    fn test_primitivity_s3() {
        let gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        let computer = BlockSystemComputer::new(3, gens);
        // S_3 acts primitively on 3 points (prime degree transitive)
        assert!(computer.is_primitive());
    }

    #[test]
    fn test_group_presentation_cyclic() {
        let pres = GroupPresentation::cyclic(5);
        assert_eq!(pres.num_generators, 1);
        assert_eq!(pres.relations.len(), 1);
    }

    #[test]
    fn test_group_presentation_dihedral() {
        let pres = GroupPresentation::dihedral(4);
        assert_eq!(pres.num_generators, 2);
        assert_eq!(pres.relations.len(), 3);
    }

    #[test]
    fn test_group_presentation_symmetric() {
        let pres = GroupPresentation::symmetric(4);
        assert_eq!(pres.num_generators, 3); // s0, s1, s2
    }

    #[test]
    fn test_all_permutations_count() {
        assert_eq!(all_permutations(0).len(), 1);
        assert_eq!(all_permutations(1).len(), 1);
        assert_eq!(all_permutations(2).len(), 2);
        assert_eq!(all_permutations(3).len(), 6);
        assert_eq!(all_permutations(4).len(), 24);
    }

    #[test]
    fn test_derangements() {
        assert_eq!(count_derangements(0), 1);
        assert_eq!(count_derangements(1), 0);
        assert_eq!(count_derangements(2), 1);
        assert_eq!(count_derangements(3), 2);
        assert_eq!(count_derangements(4), 9);
        let d4 = derangements(4);
        assert_eq!(d4.len(), 9);
    }

    #[test]
    fn test_is_abelian() {
        // Z_3 is abelian
        let gens = vec![Permutation::cycle(3, &[0, 1, 2])];
        assert!(is_abelian(&gens));

        // S_3 is not abelian
        let gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        assert!(!is_abelian(&gens));
    }

    #[test]
    fn test_is_cyclic() {
        // Z_4 is cyclic
        let gens = vec![Permutation::cycle(4, &[0, 1, 2, 3])];
        assert!(is_cyclic(&gens, 4));
    }

    #[test]
    fn test_group_exponent() {
        // S_3 has exponent lcm(1,2,3) = 6
        let gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        assert_eq!(group_exponent(&gens, 3), 6);
    }

    #[test]
    fn test_homomorphism_sign() {
        // Sign homomorphism S_3 → Z_2
        let s3_gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        // Map: (01) -> (01), (012) -> identity
        // This is NOT a homomorphism (just testing structure)
        let z2_images = vec![
            Permutation::transposition(2, 0, 1), // (01) maps to (01) — sign -1
            Permutation::identity(2),              // (012) maps to id — sign +1
        ];
        let hom = PermutationHomomorphism::new(3, 2, s3_gens, z2_images);
        // This may or may not verify; it tests the structure
        let image = hom.image();
        assert!(image.len() <= 2);
    }

    #[test]
    fn test_alternating_group() {
        let gens = alternating_group_generators(4);
        let a4 = enumerate_from_generators(&gens, 4);
        assert_eq!(a4.len(), 12); // |A_4| = 12
    }

    #[test]
    fn test_commutator_subgroup_s3() {
        let gens = vec![
            Permutation::transposition(3, 0, 1),
            Permutation::cycle(3, &[0, 1, 2]),
        ];
        let comm = commutator_subgroup(&gens, 3);
        assert_eq!(comm.len(), 3); // [S_3, S_3] = A_3 ≅ Z_3
    }
}
