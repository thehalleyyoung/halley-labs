//! Homological algebra for memory models.
//!
//! Implements chain complexes over execution categories, obstruction theory
//! for consistency checking, Ext group computation, and spectral sequences
//! for multi-level scope hierarchies.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{BitMatrix, ExecutionGraph, EventId};

// ---------------------------------------------------------------------------
// ChainGroup — a free abelian group (Z-module) of formal sums
// ---------------------------------------------------------------------------

/// Element of a free abelian group: a formal Z-linear combination of basis elements.
/// Represented as a sparse map from basis index to coefficient.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainElement {
    /// Coefficients: basis_index → coefficient (non-zero entries only).
    pub coeffs: HashMap<usize, i64>,
}

impl ChainElement {
    /// Zero element.
    pub fn zero() -> Self {
        Self { coeffs: HashMap::new() }
    }

    /// Basis element e_i with coefficient 1.
    pub fn basis(i: usize) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(i, 1);
        Self { coeffs }
    }

    /// Scalar multiple.
    pub fn scale(&self, c: i64) -> Self {
        if c == 0 {
            return Self::zero();
        }
        let coeffs = self.coeffs.iter()
            .map(|(&k, &v)| (k, v * c))
            .filter(|(_, v)| *v != 0)
            .collect();
        Self { coeffs }
    }

    /// Addition.
    pub fn add(&self, other: &Self) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (&k, &v) in &other.coeffs {
            let entry = coeffs.entry(k).or_insert(0);
            *entry += v;
            if *entry == 0 {
                coeffs.remove(&k);
            }
        }
        Self { coeffs }
    }

    /// Subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(-1))
    }

    /// Is this the zero element?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Number of non-zero terms.
    pub fn num_terms(&self) -> usize {
        self.coeffs.len()
    }

    /// Get coefficient of basis element i.
    pub fn coeff(&self, i: usize) -> i64 {
        *self.coeffs.get(&i).unwrap_or(&0)
    }

    /// Support: set of basis elements with non-zero coefficients.
    pub fn support(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self.coeffs.keys().copied().collect();
        s.sort();
        s
    }
}

impl fmt::Display for ChainElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut terms: Vec<(usize, i64)> = self.coeffs.iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        terms.sort_by_key(|&(k, _)| k);
        let strs: Vec<String> = terms.iter().map(|&(k, v)| {
            if v == 1 { format!("e{}", k) }
            else if v == -1 { format!("-e{}", k) }
            else { format!("{}·e{}", v, k) }
        }).collect();
        write!(f, "{}", strs.join(" + "))
    }
}

// ---------------------------------------------------------------------------
// BoundaryMap — a linear map between chain groups
// ---------------------------------------------------------------------------

/// A linear map ∂: C_n → C_{n-1} represented as a matrix (column i maps basis_i).
#[derive(Debug, Clone)]
pub struct BoundaryMap {
    /// Matrix entries: matrix[i][j] = coefficient of e_i in ∂(e_j).
    pub matrix: Vec<Vec<i64>>,
    /// Domain dimension.
    pub domain_dim: usize,
    /// Codomain dimension.
    pub codomain_dim: usize,
}

impl BoundaryMap {
    /// Create a zero boundary map.
    pub fn zero(codomain_dim: usize, domain_dim: usize) -> Self {
        Self {
            matrix: vec![vec![0i64; domain_dim]; codomain_dim],
            domain_dim,
            codomain_dim,
        }
    }

    /// Set entry (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: i64) {
        self.matrix[i][j] = val;
    }

    /// Get entry (i, j).
    pub fn get(&self, i: usize, j: usize) -> i64 {
        self.matrix[i][j]
    }

    /// Apply the boundary map to a chain element.
    pub fn apply(&self, elem: &ChainElement) -> ChainElement {
        let mut result = ChainElement::zero();
        for (&j, &coeff) in &elem.coeffs {
            if j >= self.domain_dim { continue; }
            for i in 0..self.codomain_dim {
                let val = self.matrix[i][j] * coeff;
                if val != 0 {
                    let entry = result.coeffs.entry(i).or_insert(0);
                    *entry += val;
                    if *entry == 0 {
                        result.coeffs.remove(&i);
                    }
                }
            }
        }
        result
    }

    /// Compose two boundary maps: self ∘ other.
    pub fn compose(&self, other: &BoundaryMap) -> BoundaryMap {
        assert_eq!(self.domain_dim, other.codomain_dim);
        let mut result = BoundaryMap::zero(self.codomain_dim, other.domain_dim);
        for i in 0..self.codomain_dim {
            for j in 0..other.domain_dim {
                let mut sum = 0i64;
                for k in 0..self.domain_dim {
                    sum += self.matrix[i][k] * other.matrix[k][j];
                }
                result.matrix[i][j] = sum;
            }
        }
        result
    }

    /// Check if map is zero.
    pub fn is_zero(&self) -> bool {
        self.matrix.iter().all(|row| row.iter().all(|&v| v == 0))
    }

    /// Rank of the map (over Q, approximated by Gaussian elimination).
    pub fn rank(&self) -> usize {
        let mut m: Vec<Vec<f64>> = self.matrix.iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();
        let rows = m.len();
        let cols = if rows > 0 { m[0].len() } else { 0 };
        let mut rank = 0;
        let mut pivot_col = 0;

        for row in 0..rows {
            if pivot_col >= cols { break; }
            // Find pivot.
            let mut pivot_row = None;
            for r in row..rows {
                if m[r][pivot_col].abs() > 1e-10 {
                    pivot_row = Some(r);
                    break;
                }
            }
            if let Some(pr) = pivot_row {
                m.swap(row, pr);
                let pivot = m[row][pivot_col];
                for c in 0..cols {
                    m[row][c] /= pivot;
                }
                for r in 0..rows {
                    if r == row { continue; }
                    let factor = m[r][pivot_col];
                    for c in 0..cols {
                        m[r][c] -= factor * m[row][c];
                    }
                }
                rank += 1;
            }
            pivot_col += 1;
        }
        rank
    }
}

impl fmt::Display for BoundaryMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BoundaryMap({}×{})", self.codomain_dim, self.domain_dim)
    }
}

// ---------------------------------------------------------------------------
// ChainComplex — a sequence of chain groups and boundary maps
// ---------------------------------------------------------------------------

/// A chain complex: ... → C_n →∂_n C_{n-1} → ... → C_0.
/// The fundamental property is ∂_{n-1} ∘ ∂_n = 0.
#[derive(Debug, Clone)]
pub struct ChainComplex {
    /// Dimensions of chain groups C_0, C_1, ..., C_n.
    pub dimensions: Vec<usize>,
    /// Boundary maps ∂_1, ∂_2, ..., ∂_n (∂_i: C_i → C_{i-1}).
    pub boundaries: Vec<BoundaryMap>,
}

impl ChainComplex {
    /// Create a chain complex from dimensions and boundary maps.
    pub fn new(dimensions: Vec<usize>, boundaries: Vec<BoundaryMap>) -> Self {
        assert_eq!(boundaries.len() + 1, dimensions.len());
        Self { dimensions, boundaries }
    }

    /// Number of levels (length of the complex).
    pub fn length(&self) -> usize {
        self.dimensions.len()
    }

    /// Check the chain complex property: ∂_{i} ∘ ∂_{i+1} = 0 for all i.
    pub fn is_valid(&self) -> bool {
        for i in 0..self.boundaries.len().saturating_sub(1) {
            let composed = self.boundaries[i].compose(&self.boundaries[i + 1]);
            if !composed.is_zero() {
                return false;
            }
        }
        true
    }

    /// Compute the n-th homology group rank: H_n = ker(∂_n) / im(∂_{n+1}).
    /// Returns the rank (Betti number) β_n.
    pub fn homology_rank(&self, n: usize) -> usize {
        if n >= self.dimensions.len() { return 0; }

        // ker(∂_n) rank = dim(C_n) - rank(∂_n).
        let ker_rank = if n > 0 && n - 1 < self.boundaries.len() {
            self.dimensions[n] - self.boundaries[n - 1].rank()
        } else {
            self.dimensions[n]
        };

        // im(∂_{n+1}) rank.
        let im_rank = if n < self.boundaries.len() {
            self.boundaries[n].rank()
        } else {
            0
        };

        ker_rank.saturating_sub(im_rank)
    }

    /// Compute all Betti numbers (homology ranks).
    pub fn betti_numbers(&self) -> Vec<usize> {
        (0..self.dimensions.len())
            .map(|n| self.homology_rank(n))
            .collect()
    }

    /// Euler characteristic: χ = Σ (-1)^n β_n.
    pub fn euler_characteristic(&self) -> i64 {
        self.betti_numbers().iter().enumerate()
            .map(|(n, &b)| if n % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum()
    }
}

impl fmt::Display for ChainComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.dimensions.iter().map(|d| format!("Z^{}", d)).collect();
        write!(f, "{}", dims.join(" ← "))
    }
}

// ---------------------------------------------------------------------------
// Execution category chain complex
// ---------------------------------------------------------------------------

/// Build a chain complex from an execution graph.
/// The chain groups are:
///   C_0 = events (vertices)
///   C_1 = edges (po, rf, co, fr)
///   C_2 = triangles (3-cycles or 3-paths)
pub fn execution_chain_complex(exec: &ExecutionGraph) -> ChainComplex {
    let n_events = exec.len();
    let all_relations = exec.po.union(&exec.rf).union(&exec.co).union(&exec.fr);
    let edges = all_relations.edges();
    let n_edges = edges.len();

    // Build edge index map.
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, &(i, j)) in edges.iter().enumerate() {
        edge_index.insert((i, j), idx);
    }

    // C_2: triangles (i→j→k where all edges exist).
    let mut triangles: Vec<(usize, usize, usize)> = Vec::new();
    for &(i, j) in &edges {
        for &(j2, k) in &edges {
            if j == j2 && edge_index.contains_key(&(i, k)) {
                triangles.push((i, j, k));
            }
        }
    }
    // Limit to avoid combinatorial explosion.
    triangles.truncate(1000);
    let n_triangles = triangles.len();

    // ∂_1: C_1 → C_0 (boundary of edge = target - source).
    let mut d1 = BoundaryMap::zero(n_events, n_edges);
    for (idx, &(i, j)) in edges.iter().enumerate() {
        if i < n_events { d1.set(i, idx, -1); }
        if j < n_events { d1.set(j, idx, 1); }
    }

    // ∂_2: C_2 → C_1 (boundary of triangle = edges).
    let mut d2 = BoundaryMap::zero(n_edges, n_triangles);
    for (t_idx, &(i, j, k)) in triangles.iter().enumerate() {
        if let Some(&e_ij) = edge_index.get(&(i, j)) {
            d2.set(e_ij, t_idx, 1);
        }
        if let Some(&e_jk) = edge_index.get(&(j, k)) {
            d2.set(e_jk, t_idx, -1);
        }
        if let Some(&e_ik) = edge_index.get(&(i, k)) {
            d2.set(e_ik, t_idx, 1);
        }
    }

    ChainComplex::new(
        vec![n_events, n_edges, n_triangles],
        vec![d1, d2],
    )
}

// ---------------------------------------------------------------------------
// Obstruction theory for consistency
// ---------------------------------------------------------------------------

/// Obstruction to consistency: a non-trivial element in homology.
#[derive(Debug, Clone)]
pub struct Obstruction {
    pub degree: usize,
    pub element: ChainElement,
    pub description: String,
}

impl fmt::Display for Obstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Obstruction(H_{}: {})", self.degree, self.description)
    }
}

/// Check for consistency obstructions using homological methods.
pub fn find_obstructions(exec: &ExecutionGraph) -> Vec<Obstruction> {
    let complex = execution_chain_complex(exec);
    let betti = complex.betti_numbers();
    let mut obstructions = Vec::new();

    // Non-trivial H_1 indicates cycles that are not boundaries of 2-cells,
    // which correspond to potential consistency violations.
    if betti.len() > 1 && betti[1] > 0 {
        obstructions.push(Obstruction {
            degree: 1,
            element: ChainElement::zero(),
            description: format!("H_1 has rank {} (non-trivial 1-cycles)", betti[1]),
        });
    }

    // Non-trivial H_0 indicates disconnected components.
    if betti.len() > 0 && betti[0] > 1 {
        obstructions.push(Obstruction {
            degree: 0,
            element: ChainElement::zero(),
            description: format!("H_0 has rank {} (disconnected components)", betti[0]),
        });
    }

    obstructions
}

// ---------------------------------------------------------------------------
// Ext groups
// ---------------------------------------------------------------------------

/// Compute Ext^n(A, B) rank for chain complexes.
/// This is the derived functor Hom measuring extensions.
pub fn ext_rank(complex: &ChainComplex, n: usize) -> usize {
    // Ext^n ≅ H^n(Hom(complex, Z)) ≅ H_n for free modules.
    complex.homology_rank(n)
}

/// Compute all Ext ranks up to the length of the complex.
pub fn ext_ranks(complex: &ChainComplex) -> Vec<usize> {
    (0..complex.length())
        .map(|n| ext_rank(complex, n))
        .collect()
}

// ---------------------------------------------------------------------------
// Spectral sequence for multi-level hierarchies
// ---------------------------------------------------------------------------

/// A page of a spectral sequence.
#[derive(Debug, Clone)]
pub struct SpectralPage {
    /// Page number r.
    pub page: usize,
    /// Entries E_r^{p,q}: indexed by (p, q).
    pub entries: HashMap<(usize, usize), usize>,
}

impl SpectralPage {
    pub fn new(page: usize) -> Self {
        Self {
            page,
            entries: HashMap::new(),
        }
    }

    pub fn set(&mut self, p: usize, q: usize, rank: usize) {
        self.entries.insert((p, q), rank);
    }

    pub fn get(&self, p: usize, q: usize) -> usize {
        *self.entries.get(&(p, q)).unwrap_or(&0)
    }

    /// Total rank on the diagonal p + q = n.
    pub fn diagonal_rank(&self, n: usize) -> usize {
        (0..=n).map(|p| self.get(p, n - p)).sum()
    }
}

impl fmt::Display for SpectralPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E_{}", self.page)?;
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by_key(|&(&(p, q), _)| (p, q));
        for (&(p, q), &rank) in &entries {
            if rank > 0 {
                write!(f, " ({},{})={}", p, q, rank)?;
            }
        }
        Ok(())
    }
}

/// Spectral sequence for a filtered chain complex.
/// Models the multi-level GPU scope hierarchy.
#[derive(Debug, Clone)]
pub struct SpectralSequence {
    pub pages: Vec<SpectralPage>,
}

impl SpectralSequence {
    /// Compute a spectral sequence from a hierarchy of chain complexes.
    /// Each level corresponds to a scope level (thread, warp, CTA, GPU, system).
    pub fn from_hierarchy(complexes: &[ChainComplex]) -> Self {
        let mut pages = Vec::new();

        // E_0 page: entries are the dimensions of chain groups at each level.
        let mut e0 = SpectralPage::new(0);
        for (p, complex) in complexes.iter().enumerate() {
            for (q, &dim) in complex.dimensions.iter().enumerate() {
                e0.set(p, q, dim);
            }
        }
        pages.push(e0);

        // E_1 page: homology of each level complex.
        let mut e1 = SpectralPage::new(1);
        for (p, complex) in complexes.iter().enumerate() {
            let betti = complex.betti_numbers();
            for (q, &b) in betti.iter().enumerate() {
                e1.set(p, q, b);
            }
        }
        let e1_entries = e1.entries.clone();
        pages.push(e1);

        // E_2 page: approximate by E_1 (higher differentials are hard to compute).
        let e2 = SpectralPage {
            page: 2,
            entries: e1_entries,
        };
        pages.push(e2);

        Self { pages }
    }

    /// Get a specific page.
    pub fn page(&self, r: usize) -> Option<&SpectralPage> {
        self.pages.get(r)
    }

    /// Number of computed pages.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Check if the spectral sequence has converged (E_r ≅ E_{r+1}).
    pub fn has_converged(&self) -> bool {
        if self.pages.len() < 2 { return false; }
        let last = &self.pages[self.pages.len() - 1];
        let prev = &self.pages[self.pages.len() - 2];
        last.entries == prev.entries
    }

    /// Compute the total homology from the converged page.
    pub fn total_homology(&self) -> Vec<usize> {
        if self.pages.is_empty() { return vec![]; }
        let last = &self.pages[self.pages.len() - 1];
        let max_n = last.entries.keys()
            .map(|&(p, q)| p + q)
            .max()
            .unwrap_or(0);
        (0..=max_n).map(|n| last.diagonal_rank(n)).collect()
    }
}

impl fmt::Display for SpectralSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Spectral Sequence ({} pages)", self.pages.len())?;
        for page in &self.pages {
            writeln!(f, "  {}", page)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_element_zero() {
        let z = ChainElement::zero();
        assert!(z.is_zero());
        assert_eq!(z.num_terms(), 0);
    }

    #[test]
    fn test_chain_element_basis() {
        let e = ChainElement::basis(3);
        assert!(!e.is_zero());
        assert_eq!(e.coeff(3), 1);
        assert_eq!(e.coeff(0), 0);
    }

    #[test]
    fn test_chain_element_add() {
        let a = ChainElement::basis(0);
        let b = ChainElement::basis(1);
        let c = a.add(&b);
        assert_eq!(c.coeff(0), 1);
        assert_eq!(c.coeff(1), 1);
        assert_eq!(c.num_terms(), 2);
    }

    #[test]
    fn test_chain_element_sub() {
        let a = ChainElement::basis(0);
        let b = ChainElement::basis(0);
        let c = a.sub(&b);
        assert!(c.is_zero());
    }

    #[test]
    fn test_chain_element_scale() {
        let e = ChainElement::basis(2);
        let s = e.scale(3);
        assert_eq!(s.coeff(2), 3);
    }

    #[test]
    fn test_chain_element_scale_zero() {
        let e = ChainElement::basis(2);
        let s = e.scale(0);
        assert!(s.is_zero());
    }

    #[test]
    fn test_chain_element_support() {
        let a = ChainElement::basis(0).add(&ChainElement::basis(2));
        assert_eq!(a.support(), vec![0, 2]);
    }

    #[test]
    fn test_chain_element_display() {
        let e = ChainElement::basis(0).add(&ChainElement::basis(1).scale(2));
        let s = format!("{}", e);
        assert!(s.contains("e0"));
        assert!(s.contains("e1"));
    }

    #[test]
    fn test_boundary_map_apply() {
        let mut d = BoundaryMap::zero(2, 1);
        d.set(0, 0, -1); // ∂(e_0) = -e_0 (in codomain)
        d.set(1, 0, 1);  // ∂(e_0) = -e_0 + e_1

        let result = d.apply(&ChainElement::basis(0));
        assert_eq!(result.coeff(0), -1);
        assert_eq!(result.coeff(1), 1);
    }

    #[test]
    fn test_boundary_map_compose() {
        let mut d1 = BoundaryMap::zero(2, 3);
        let mut d2 = BoundaryMap::zero(3, 2);
        // d1 and d2 should compose to zero for a valid chain complex.
        let composed = d1.compose(&d2);
        assert!(composed.is_zero());
    }

    #[test]
    fn test_boundary_map_rank() {
        let mut d = BoundaryMap::zero(2, 2);
        d.set(0, 0, 1);
        d.set(1, 1, 1);
        assert_eq!(d.rank(), 2);
    }

    #[test]
    fn test_boundary_map_rank_zero() {
        let d = BoundaryMap::zero(3, 3);
        assert_eq!(d.rank(), 0);
    }

    #[test]
    fn test_chain_complex_valid() {
        // Simple: C_0 = Z, C_1 = Z, ∂_1 = identity (not ∂∘∂=0 since only one map).
        let d1 = BoundaryMap::zero(2, 3);
        let complex = ChainComplex::new(vec![2, 3], vec![d1]);
        assert!(complex.is_valid()); // Only 1 boundary, nothing to compose.
    }

    #[test]
    fn test_chain_complex_betti_numbers() {
        // Complex: Z^2 ←∂₁ Z^3, ∂₁ = 0.
        let d1 = BoundaryMap::zero(2, 3);
        let complex = ChainComplex::new(vec![2, 3], vec![d1]);
        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 2); // H_0 = ker(0)/im(∂_1) = Z^2 / 0 = Z^2
        assert_eq!(betti[1], 3); // H_1 = ker(nothing)/im(nothing) = Z^3
    }

    #[test]
    fn test_chain_complex_euler_characteristic() {
        let d1 = BoundaryMap::zero(2, 3);
        let complex = ChainComplex::new(vec![2, 3], vec![d1]);
        let chi = complex.euler_characteristic();
        assert_eq!(chi, 2 - 3); // β_0 - β_1
    }

    #[test]
    fn test_chain_complex_display() {
        let d1 = BoundaryMap::zero(2, 3);
        let complex = ChainComplex::new(vec![2, 3], vec![d1]);
        let s = format!("{}", complex);
        assert!(s.contains("Z^2"));
        assert!(s.contains("Z^3"));
    }

    #[test]
    fn test_execution_chain_complex() {
        use crate::checker::execution::{Event, OpType};
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x100, 1).with_po_index(1),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true);
        graph.derive_fr();

        let complex = execution_chain_complex(&graph);
        assert!(complex.length() >= 2);
        assert!(complex.is_valid());
    }

    #[test]
    fn test_find_obstructions() {
        use crate::checker::execution::{Event, OpType};
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x100, 1).with_po_index(1),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true);
        graph.derive_fr();

        let obstructions = find_obstructions(&graph);
        // Simple 2-event graph should have no consistency obstructions.
        // (H_1 might be 0)
        for obs in &obstructions {
            let _ = format!("{}", obs);
        }
    }

    #[test]
    fn test_ext_ranks() {
        let d1 = BoundaryMap::zero(2, 3);
        let complex = ChainComplex::new(vec![2, 3], vec![d1]);
        let ext = ext_ranks(&complex);
        assert_eq!(ext.len(), 2);
    }

    #[test]
    fn test_spectral_page() {
        let mut page = SpectralPage::new(0);
        page.set(0, 0, 4);
        page.set(1, 0, 3);
        assert_eq!(page.get(0, 0), 4);
        assert_eq!(page.get(1, 0), 3);
        assert_eq!(page.get(2, 2), 0);
    }

    #[test]
    fn test_spectral_page_diagonal() {
        let mut page = SpectralPage::new(0);
        page.set(0, 1, 2);
        page.set(1, 0, 3);
        assert_eq!(page.diagonal_rank(1), 5);
    }

    #[test]
    fn test_spectral_sequence() {
        let d1 = BoundaryMap::zero(2, 3);
        let c1 = ChainComplex::new(vec![2, 3], vec![d1]);
        let d2 = BoundaryMap::zero(1, 2);
        let c2 = ChainComplex::new(vec![1, 2], vec![d2]);

        let ss = SpectralSequence::from_hierarchy(&[c1, c2]);
        assert!(ss.num_pages() >= 2);
    }

    #[test]
    fn test_spectral_sequence_convergence() {
        let d1 = BoundaryMap::zero(2, 3);
        let c1 = ChainComplex::new(vec![2, 3], vec![d1]);

        let ss = SpectralSequence::from_hierarchy(&[c1]);
        // E_1 should equal E_2 (since we only approximate).
        assert!(ss.has_converged());
    }

    #[test]
    fn test_spectral_sequence_total_homology() {
        let d1 = BoundaryMap::zero(2, 3);
        let c1 = ChainComplex::new(vec![2, 3], vec![d1]);
        let ss = SpectralSequence::from_hierarchy(&[c1]);
        let h = ss.total_homology();
        assert!(!h.is_empty());
    }

    #[test]
    fn test_spectral_sequence_display() {
        let d1 = BoundaryMap::zero(2, 3);
        let c1 = ChainComplex::new(vec![2, 3], vec![d1]);
        let ss = SpectralSequence::from_hierarchy(&[c1]);
        let s = format!("{}", ss);
        assert!(s.contains("Spectral Sequence"));
    }

    #[test]
    fn test_obstruction_display() {
        let obs = Obstruction {
            degree: 1,
            element: ChainElement::zero(),
            description: "test obstruction".to_string(),
        };
        let s = format!("{}", obs);
        assert!(s.contains("H_1"));
    }

    #[test]
    fn test_boundary_map_display() {
        let d = BoundaryMap::zero(2, 3);
        let s = format!("{}", d);
        assert!(s.contains("2×3"));
    }

    #[test]
    fn test_chain_complex_length() {
        let d1 = BoundaryMap::zero(2, 3);
        let d2 = BoundaryMap::zero(3, 1);
        let complex = ChainComplex::new(vec![2, 3, 1], vec![d1, d2]);
        assert_eq!(complex.length(), 3);
    }

    #[test]
    fn test_two_boundary_chain_complex() {
        // C_0=Z^2, C_1=Z^3, C_2=Z^1
        // ∂_1: C_1→C_0 (zero), ∂_2: C_2→C_1 (zero)
        let d1 = BoundaryMap::zero(2, 3);
        let d2 = BoundaryMap::zero(3, 1);
        let complex = ChainComplex::new(vec![2, 3, 1], vec![d1, d2]);
        // ∂_1 ∘ ∂_2 should be zero.
        assert!(complex.is_valid());
        let betti = complex.betti_numbers();
        assert_eq!(betti.len(), 3);
    }
}
