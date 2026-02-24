// LITMUS∞ Algebraic Engine — Wreath Product Decomposition
//
//   WreathProduct construction for hierarchical symmetries
//   GPU-specific: warp-level, CTA-level, GPU-level decomposition
//   HierarchicalOrbitEnumeration using wreath product

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::types::Permutation;
use super::group::PermutationGroup;
use super::symmetry::{FullSymmetryGroup, LitmusTest};
use super::orbit::{
    ExecutionCandidate, OrbitRepresentativeSet, CanonicalForm, EnumerationStats,
};

// ── Wreath Product ───────────────────────────────────────────────────

/// A wreath product G ≀ H decomposition.
///
/// If G acts on {0, ..., m-1} and H acts on {0, ..., k-1}, then
/// G ≀ H acts on {0, ..., m*k - 1} where:
///   - The base group is G^k (k copies of G, one per block)
///   - The top group H permutes the blocks
#[derive(Clone, Debug)]
pub struct WreathProduct {
    /// The base group G (acts within each block).
    pub base_group: PermutationGroup,
    /// The top group H (permutes blocks).
    pub top_group: PermutationGroup,
    /// Block size m.
    pub block_size: usize,
    /// Number of blocks k.
    pub num_blocks: usize,
    /// Total degree m * k.
    pub degree: usize,
    /// The full wreath product group.
    pub full_group: PermutationGroup,
}

impl WreathProduct {
    /// Construct G ≀ H.
    pub fn new(base: &PermutationGroup, top: &PermutationGroup) -> Self {
        let m = base.degree();
        let k = top.degree();
        let n = m * k;

        let full = PermutationGroup::wreath_product(base, top);

        WreathProduct {
            base_group: base.clone(),
            top_group: top.clone(),
            block_size: m,
            num_blocks: k,
            degree: n,
            full_group: full,
        }
    }

    /// Order of the wreath product: |G|^k * |H|.
    pub fn order(&self) -> u64 {
        let base_order = self.base_group.order();
        let top_order = self.top_group.order();
        let mut result = top_order;
        for _ in 0..self.num_blocks {
            result = result.saturating_mul(base_order);
        }
        result
    }

    /// Get the block containing element i.
    pub fn block_of(&self, i: usize) -> usize {
        i / self.block_size
    }

    /// Get the position within a block.
    pub fn position_in_block(&self, i: usize) -> usize {
        i % self.block_size
    }

    /// Get the element at block b, position p.
    pub fn element_at(&self, block: usize, position: usize) -> usize {
        block * self.block_size + position
    }

    /// Create a base group element: apply a permutation to a single block.
    pub fn base_element(&self, block: usize, perm: &Permutation) -> Permutation {
        assert_eq!(perm.degree(), self.block_size);
        assert!(block < self.num_blocks);

        let mut images: Vec<u32> = (0..self.degree as u32).collect();
        let offset = block * self.block_size;
        for i in 0..self.block_size {
            images[offset + i] = (offset as u32) + perm.apply(i as u32);
        }
        Permutation::new(images)
    }

    /// Create a top group element: permute blocks according to perm.
    pub fn top_element(&self, perm: &Permutation) -> Permutation {
        assert_eq!(perm.degree(), self.num_blocks);

        let mut images: Vec<u32> = (0..self.degree as u32).collect();
        for block in 0..self.num_blocks {
            let target = perm.apply(block as u32) as usize;
            for i in 0..self.block_size {
                images[block * self.block_size + i] =
                    (target * self.block_size + i) as u32;
            }
        }
        Permutation::new(images)
    }

    /// Decompose an element of the wreath product into base and top components.
    /// Returns (base_perms, top_perm) where base_perms[i] is the permutation
    /// applied to block i, and top_perm is the block permutation.
    pub fn decompose(&self, perm: &Permutation) -> (Vec<Permutation>, Permutation) {
        assert_eq!(perm.degree(), self.degree);

        // Determine the top permutation: which block does each block map to?
        let mut top_images = vec![0u32; self.num_blocks];
        for block in 0..self.num_blocks {
            let elem = block * self.block_size;
            let image = perm.apply(elem as u32) as usize;
            top_images[block] = self.block_of(image) as u32;
        }
        let top_perm = Permutation::new(top_images);

        // Determine base permutations for each block
        let mut base_perms = Vec::new();
        for block in 0..self.num_blocks {
            let target_block = top_perm.apply(block as u32) as usize;
            let mut base_images = vec![0u32; self.block_size];
            for i in 0..self.block_size {
                let elem = block * self.block_size + i;
                let image = perm.apply(elem as u32) as usize;
                let pos = self.position_in_block(image);
                base_images[i] = pos as u32;
            }
            base_perms.push(Permutation::new(base_images));
        }

        (base_perms, top_perm)
    }

    /// Reconstruct an element from base and top components.
    pub fn reconstruct(
        &self,
        base_perms: &[Permutation],
        top_perm: &Permutation,
    ) -> Permutation {
        assert_eq!(base_perms.len(), self.num_blocks);
        assert_eq!(top_perm.degree(), self.num_blocks);

        let mut images: Vec<u32> = vec![0; self.degree];
        for block in 0..self.num_blocks {
            let target_block = top_perm.apply(block as u32) as usize;
            for i in 0..self.block_size {
                let base_image = base_perms[block].apply(i as u32) as usize;
                images[block * self.block_size + i] =
                    (target_block * self.block_size + base_image) as u32;
            }
        }
        Permutation::new(images)
    }

    /// Check if a permutation belongs to this wreath product.
    pub fn contains(&self, perm: &Permutation) -> bool {
        self.full_group.contains(perm)
    }
}

impl fmt::Display for WreathProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WreathProduct({} ≀ {}, blocks={}×{}, |G≀H|={})",
            self.base_group,
            self.top_group,
            self.num_blocks,
            self.block_size,
            self.order(),
        )
    }
}

// ── GPU Hierarchy ────────────────────────────────────────────────────

/// GPU execution hierarchy levels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuLevel {
    /// Individual thread within a warp.
    Thread,
    /// Warp (typically 32 threads).
    Warp,
    /// CTA / thread block.
    Cta,
    /// GPU device.
    Gpu,
}

/// GPU-specific hierarchical symmetry decomposition.
///
/// A GPU has a hierarchy: GPU → CTAs → Warps → Threads.
/// If threads within a warp are symmetric, warps within a CTA are symmetric,
/// etc., we get a wreath product:
///   G_thread ≀ G_warp ≀ G_cta ≀ G_gpu
#[derive(Clone, Debug)]
pub struct GpuHierarchicalSymmetry {
    /// Thread-level symmetry within a warp.
    pub thread_symmetry: PermutationGroup,
    /// Warp-level symmetry within a CTA.
    pub warp_symmetry: PermutationGroup,
    /// CTA-level symmetry within the GPU.
    pub cta_symmetry: PermutationGroup,

    /// Number of threads per warp.
    pub threads_per_warp: usize,
    /// Number of warps per CTA.
    pub warps_per_cta: usize,
    /// Number of CTAs.
    pub num_ctas: usize,

    /// Full hierarchical wreath product.
    pub full_wreath: Option<WreathProduct>,
}

impl GpuHierarchicalSymmetry {
    /// Construct from the GPU hierarchy parameters and symmetries.
    pub fn new(
        threads_per_warp: usize,
        warps_per_cta: usize,
        num_ctas: usize,
        thread_sym: PermutationGroup,
        warp_sym: PermutationGroup,
        cta_sym: PermutationGroup,
    ) -> Self {
        // Build the hierarchical wreath product
        // First: thread_sym ≀ warp_sym
        let warp_level = if thread_sym.order() > 1 && warp_sym.order() > 1 {
            Some(WreathProduct::new(&thread_sym, &warp_sym))
        } else {
            None
        };

        // Then: (thread_sym ≀ warp_sym) ≀ cta_sym
        let full_wreath = if let Some(ref wl) = warp_level {
            if cta_sym.order() > 1 {
                Some(WreathProduct::new(&wl.full_group, &cta_sym))
            } else {
                Some(wl.clone())
            }
        } else {
            None
        };

        GpuHierarchicalSymmetry {
            thread_symmetry: thread_sym,
            warp_symmetry: warp_sym,
            cta_symmetry: cta_sym,
            threads_per_warp,
            warps_per_cta,
            num_ctas,
            full_wreath,
        }
    }

    /// Total number of threads.
    pub fn total_threads(&self) -> usize {
        self.threads_per_warp * self.warps_per_cta * self.num_ctas
    }

    /// Total symmetry order.
    pub fn total_order(&self) -> u64 {
        let t = self.thread_symmetry.order();
        let w = self.warp_symmetry.order();
        let c = self.cta_symmetry.order();

        // |G_t ≀ G_w ≀ G_c| = |G_t|^(warps*ctas) * |G_w|^ctas * |G_c|
        let mut result = c;
        for _ in 0..self.num_ctas {
            result = result.saturating_mul(w);
        }
        let total_warps = self.warps_per_cta * self.num_ctas;
        for _ in 0..total_warps {
            result = result.saturating_mul(t);
        }
        result
    }

    /// Compression factor at each level.
    pub fn level_factors(&self) -> HashMap<GpuLevel, f64> {
        let mut factors = HashMap::new();
        factors.insert(GpuLevel::Thread, self.thread_symmetry.order() as f64);
        factors.insert(GpuLevel::Warp, self.warp_symmetry.order() as f64);
        factors.insert(GpuLevel::Cta, self.cta_symmetry.order() as f64);
        factors.insert(GpuLevel::Gpu, self.total_order() as f64);
        factors
    }

    /// Get the global thread index from (cta, warp, thread).
    pub fn global_thread_id(&self, cta: usize, warp: usize, thread: usize) -> usize {
        cta * self.warps_per_cta * self.threads_per_warp
            + warp * self.threads_per_warp
            + thread
    }

    /// Decompose a global thread ID into (cta, warp, thread).
    pub fn decompose_thread_id(&self, global_id: usize) -> (usize, usize, usize) {
        let threads_per_cta = self.warps_per_cta * self.threads_per_warp;
        let cta = global_id / threads_per_cta;
        let remainder = global_id % threads_per_cta;
        let warp = remainder / self.threads_per_warp;
        let thread = remainder % self.threads_per_warp;
        (cta, warp, thread)
    }

    /// Summary.
    pub fn summary(&self) -> String {
        format!(
            "GpuHierarchy({}×{}×{} = {} threads, |G|={})",
            self.num_ctas,
            self.warps_per_cta,
            self.threads_per_warp,
            self.total_threads(),
            self.total_order(),
        )
    }
}

// ── Hierarchical Orbit Enumeration ───────────────────────────────────

/// Hierarchical orbit enumeration using wreath product decomposition.
///
/// Exploits the hierarchical structure to enumerate orbits level by level:
/// 1. Enumerate orbits at the warp level (within each warp)
/// 2. Enumerate orbits at the CTA level (combining warp orbits)
/// 3. Enumerate orbits at the GPU level (combining CTA orbits)
#[derive(Clone)]
pub struct HierarchicalOrbitEnumeration {
    gpu_hierarchy: GpuHierarchicalSymmetry,
    symmetry: FullSymmetryGroup,
}

impl HierarchicalOrbitEnumeration {
    pub fn new(
        gpu_hierarchy: GpuHierarchicalSymmetry,
        symmetry: FullSymmetryGroup,
    ) -> Self {
        HierarchicalOrbitEnumeration {
            gpu_hierarchy,
            symmetry,
        }
    }

    /// Run hierarchical enumeration.
    pub fn enumerate(
        &self,
        candidates: &[ExecutionCandidate],
    ) -> (OrbitRepresentativeSet, HierarchicalStats) {
        let mut stats = HierarchicalStats::new();
        let mut result = OrbitRepresentativeSet::new();

        stats.total_input = candidates.len() as u64;

        // Level 1: Canonicalize under thread-level symmetry
        let thread_canonical = self.canonicalize_level(
            candidates,
            &self.gpu_hierarchy.thread_symmetry,
            CanonicalizationLevel::Thread,
        );
        stats.after_thread_level = thread_canonical.len() as u64;

        // Level 2: Canonicalize under warp-level symmetry
        let warp_canonical = self.canonicalize_level(
            &thread_canonical,
            &self.gpu_hierarchy.warp_symmetry,
            CanonicalizationLevel::Warp,
        );
        stats.after_warp_level = warp_canonical.len() as u64;

        // Level 3: Canonicalize under CTA-level symmetry
        let cta_canonical = self.canonicalize_level(
            &warp_canonical,
            &self.gpu_hierarchy.cta_symmetry,
            CanonicalizationLevel::Cta,
        );
        stats.after_cta_level = cta_canonical.len() as u64;

        // Insert all canonical representatives
        for cand in &cta_canonical {
            result.insert_canonical(cand.clone());
        }

        stats.final_orbits = result.len() as u64;
        (result, stats)
    }

    /// Canonicalize at a specific level.
    fn canonicalize_level(
        &self,
        candidates: &[ExecutionCandidate],
        group: &PermutationGroup,
        level: CanonicalizationLevel,
    ) -> Vec<ExecutionCandidate> {
        if group.order() <= 1 {
            return candidates.to_vec();
        }

        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for cand in candidates {
            let canonical = match level {
                CanonicalizationLevel::Thread | CanonicalizationLevel::Warp => {
                    CanonicalForm::canonicalize_thread_only(cand, group)
                }
                CanonicalizationLevel::Cta => {
                    CanonicalForm::canonicalize_thread_only(cand, group)
                }
            };

            if seen.insert(canonical.clone()) {
                result.push(canonical);
            }
        }

        result
    }

    /// Get the GPU hierarchy.
    pub fn hierarchy(&self) -> &GpuHierarchicalSymmetry {
        &self.gpu_hierarchy
    }
}

#[derive(Clone, Debug)]
enum CanonicalizationLevel {
    Thread,
    Warp,
    Cta,
}

/// Statistics from hierarchical enumeration.
#[derive(Clone, Debug)]
pub struct HierarchicalStats {
    pub total_input: u64,
    pub after_thread_level: u64,
    pub after_warp_level: u64,
    pub after_cta_level: u64,
    pub final_orbits: u64,
}

impl HierarchicalStats {
    pub fn new() -> Self {
        HierarchicalStats {
            total_input: 0,
            after_thread_level: 0,
            after_warp_level: 0,
            after_cta_level: 0,
            final_orbits: 0,
        }
    }

    /// Thread-level compression ratio.
    pub fn thread_compression(&self) -> f64 {
        if self.after_thread_level > 0 {
            self.total_input as f64 / self.after_thread_level as f64
        } else {
            1.0
        }
    }

    /// Warp-level compression ratio.
    pub fn warp_compression(&self) -> f64 {
        if self.after_warp_level > 0 {
            self.after_thread_level as f64 / self.after_warp_level as f64
        } else {
            1.0
        }
    }

    /// CTA-level compression ratio.
    pub fn cta_compression(&self) -> f64 {
        if self.after_cta_level > 0 {
            self.after_warp_level as f64 / self.after_cta_level as f64
        } else {
            1.0
        }
    }

    /// Total compression.
    pub fn total_compression(&self) -> f64 {
        if self.final_orbits > 0 {
            self.total_input as f64 / self.final_orbits as f64
        } else {
            1.0
        }
    }
}

impl Default for HierarchicalStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for HierarchicalStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Hierarchical Enumeration Stats ===")?;
        writeln!(f, "Input candidates:    {}", self.total_input)?;
        writeln!(
            f,
            "After thread level:  {} ({:.2}x)",
            self.after_thread_level,
            self.thread_compression()
        )?;
        writeln!(
            f,
            "After warp level:    {} ({:.2}x)",
            self.after_warp_level,
            self.warp_compression()
        )?;
        writeln!(
            f,
            "After CTA level:     {} ({:.2}x)",
            self.after_cta_level,
            self.cta_compression()
        )?;
        writeln!(
            f,
            "Final orbits:        {} ({:.2}x total)",
            self.final_orbits,
            self.total_compression()
        )?;
        Ok(())
    }
}

// ── Multi-level Wreath Decomposition ─────────────────────────────────

/// Build a multi-level wreath product from a sequence of groups.
/// G_1 ≀ G_2 ≀ ... ≀ G_k
pub fn multi_level_wreath(groups: &[PermutationGroup]) -> Option<WreathProduct> {
    if groups.is_empty() {
        return None;
    }
    if groups.len() == 1 {
        // Single group: wreath with trivial top
        let trivial = PermutationGroup::trivial(1);
        return Some(WreathProduct::new(&groups[0], &trivial));
    }

    // Build from bottom up
    let mut current = WreathProduct::new(&groups[0], &groups[1]);
    for i in 2..groups.len() {
        current = WreathProduct::new(&current.full_group, &groups[i]);
    }
    Some(current)
}

/// Detect if a litmus test has hierarchical (wreath-product) structure.
pub fn detect_hierarchical_structure(
    test: &LitmusTest,
) -> Option<GpuHierarchicalSymmetry> {
    let n = test.num_threads;

    // Try common GPU configurations
    let configs = [
        (2, 2, 1), // 4 threads: 2 warps of 2
        (2, 1, 2), // 4 threads: 2 CTAs of 2
        (2, 2, 2), // 8 threads
        (4, 2, 1), // 8 threads: 2 warps of 4
        (2, 4, 1), // 8 threads: 4 warps of 2
    ];

    for &(tpw, wpc, nc) in &configs {
        if tpw * wpc * nc == n {
            // Try this configuration
            let thread_sym = detect_within_block_symmetry(test, tpw, wpc * nc);
            let warp_sym = detect_block_permutation_symmetry(test, tpw, wpc, nc);
            let cta_sym = detect_cta_symmetry(test, tpw * wpc, nc);

            if thread_sym.order() > 1 || warp_sym.order() > 1 || cta_sym.order() > 1 {
                return Some(GpuHierarchicalSymmetry::new(
                    tpw, wpc, nc, thread_sym, warp_sym, cta_sym,
                ));
            }
        }
    }

    None
}

/// Detect symmetry within blocks of a given size.
fn detect_within_block_symmetry(
    test: &LitmusTest,
    block_size: usize,
    num_blocks: usize,
) -> PermutationGroup {
    if block_size <= 1 {
        return PermutationGroup::trivial(block_size);
    }

    // Check if threads within each block are symmetric
    let mut gens = Vec::new();

    // Try adjacent transpositions within the first block
    for i in 0..block_size - 1 {
        let perm = Permutation::transposition(test.num_threads, i as u32, (i + 1) as u32);
        let permuted = test.apply_thread_permutation(&perm);
        if test.structurally_equal(&permuted) {
            // This works within the first block; check if it's consistent across blocks
            gens.push(Permutation::transposition(block_size, i as u32, (i + 1) as u32));
        }
    }

    if gens.is_empty() {
        PermutationGroup::trivial(block_size)
    } else {
        PermutationGroup::new(block_size, gens)
    }
}

/// Detect symmetry in permuting blocks (warps within a CTA).
fn detect_block_permutation_symmetry(
    test: &LitmusTest,
    block_size: usize,
    num_blocks: usize,
    _num_ctas: usize,
) -> PermutationGroup {
    if num_blocks <= 1 {
        return PermutationGroup::trivial(num_blocks);
    }

    let n = test.num_threads;
    let mut gens = Vec::new();

    // Try swapping adjacent blocks
    for b in 0..num_blocks - 1 {
        let mut images: Vec<u32> = (0..n as u32).collect();
        for i in 0..block_size {
            let a = b * block_size + i;
            let c = (b + 1) * block_size + i;
            images[a] = c as u32;
            images[c] = a as u32;
        }
        let perm = Permutation::new(images);
        let permuted = test.apply_thread_permutation(&perm);
        if test.structurally_equal(&permuted) {
            gens.push(Permutation::transposition(num_blocks, b as u32, (b + 1) as u32));
        }
    }

    if gens.is_empty() {
        PermutationGroup::trivial(num_blocks)
    } else {
        PermutationGroup::new(num_blocks, gens)
    }
}

/// Detect CTA-level symmetry.
fn detect_cta_symmetry(
    test: &LitmusTest,
    threads_per_cta: usize,
    num_ctas: usize,
) -> PermutationGroup {
    if num_ctas <= 1 {
        return PermutationGroup::trivial(num_ctas);
    }

    let n = test.num_threads;
    let mut gens = Vec::new();

    // Try swapping adjacent CTAs
    for c in 0..num_ctas - 1 {
        let mut images: Vec<u32> = (0..n as u32).collect();
        for i in 0..threads_per_cta {
            let a = c * threads_per_cta + i;
            let b = (c + 1) * threads_per_cta + i;
            images[a] = b as u32;
            images[b] = a as u32;
        }
        let perm = Permutation::new(images);
        let permuted = test.apply_thread_permutation(&perm);
        if test.structurally_equal(&permuted) {
            gens.push(Permutation::transposition(num_ctas, c as u32, (c + 1) as u32));
        }
    }

    if gens.is_empty() {
        PermutationGroup::trivial(num_ctas)
    } else {
        PermutationGroup::new(num_ctas, gens)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::symmetry::*;

    #[test]
    fn test_wreath_product_construction() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);

        assert_eq!(wp.block_size, 2);
        assert_eq!(wp.num_blocks, 2);
        assert_eq!(wp.degree, 4);
        // |C_2 ≀ S_2| = |C_2|^2 * |S_2| = 4 * 2 = 8
        assert_eq!(wp.order(), 8);
    }

    #[test]
    fn test_wreath_product_s2_wr_s3() {
        let s2 = PermutationGroup::symmetric(2);
        let s3 = PermutationGroup::symmetric(3);
        let wp = WreathProduct::new(&s2, &s3);

        assert_eq!(wp.block_size, 2);
        assert_eq!(wp.num_blocks, 3);
        assert_eq!(wp.degree, 6);
        // |S_2 ≀ S_3| = |S_2|^3 * |S_3| = 8 * 6 = 48
        assert_eq!(wp.order(), 48);
    }

    #[test]
    fn test_wreath_block_of() {
        let c2 = PermutationGroup::cyclic(2);
        let s3 = PermutationGroup::symmetric(3);
        let wp = WreathProduct::new(&c2, &s3);

        assert_eq!(wp.block_of(0), 0);
        assert_eq!(wp.block_of(1), 0);
        assert_eq!(wp.block_of(2), 1);
        assert_eq!(wp.block_of(3), 1);
        assert_eq!(wp.block_of(4), 2);
        assert_eq!(wp.block_of(5), 2);
    }

    #[test]
    fn test_wreath_position_in_block() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);

        assert_eq!(wp.position_in_block(0), 0);
        assert_eq!(wp.position_in_block(1), 1);
        assert_eq!(wp.position_in_block(2), 0);
        assert_eq!(wp.position_in_block(3), 1);
    }

    #[test]
    fn test_wreath_element_at() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);

        assert_eq!(wp.element_at(0, 0), 0);
        assert_eq!(wp.element_at(0, 1), 1);
        assert_eq!(wp.element_at(1, 0), 2);
        assert_eq!(wp.element_at(1, 1), 3);
    }

    #[test]
    fn test_wreath_base_element() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);

        // Apply (0 1) to block 0
        let perm = Permutation::transposition(2, 0, 1);
        let base_elem = wp.base_element(0, &perm);
        assert_eq!(base_elem.apply(0), 1);
        assert_eq!(base_elem.apply(1), 0);
        assert_eq!(base_elem.apply(2), 2);
        assert_eq!(base_elem.apply(3), 3);
    }

    #[test]
    fn test_wreath_top_element() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);

        // Swap blocks
        let perm = Permutation::transposition(2, 0, 1);
        let top_elem = wp.top_element(&perm);
        assert_eq!(top_elem.apply(0), 2);
        assert_eq!(top_elem.apply(1), 3);
        assert_eq!(top_elem.apply(2), 0);
        assert_eq!(top_elem.apply(3), 1);
    }

    #[test]
    fn test_wreath_decompose_reconstruct() {
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&s2, &s2);

        // Create a wreath product element
        let base0 = Permutation::transposition(2, 0, 1); // Swap in block 0
        let base1 = Permutation::identity(2);             // Identity in block 1
        let top = Permutation::transposition(2, 0, 1);    // Swap blocks

        let elem = wp.reconstruct(&[base0.clone(), base1.clone()], &top);

        // Decompose it back
        let (dec_base, dec_top) = wp.decompose(&elem);
        let reconstructed = wp.reconstruct(&dec_base, &dec_top);

        assert_eq!(elem, reconstructed);
    }

    #[test]
    fn test_wreath_contains() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);

        let base_elem = wp.base_element(0, &Permutation::transposition(2, 0, 1));
        assert!(wp.contains(&base_elem));

        let top_elem = wp.top_element(&Permutation::transposition(2, 0, 1));
        assert!(wp.contains(&top_elem));

        let id = Permutation::identity(4);
        assert!(wp.contains(&id));
    }

    #[test]
    fn test_gpu_hierarchy_construction() {
        let thread_sym = PermutationGroup::symmetric(2);
        let warp_sym = PermutationGroup::symmetric(2);
        let cta_sym = PermutationGroup::trivial(1);

        let gpu = GpuHierarchicalSymmetry::new(
            2, 2, 1, thread_sym, warp_sym, cta_sym,
        );

        assert_eq!(gpu.total_threads(), 4);
        assert!(gpu.total_order() >= 1);
    }

    #[test]
    fn test_gpu_hierarchy_thread_id() {
        let gpu = GpuHierarchicalSymmetry::new(
            4, 2, 2,
            PermutationGroup::trivial(4),
            PermutationGroup::trivial(2),
            PermutationGroup::trivial(2),
        );

        assert_eq!(gpu.total_threads(), 16);
        assert_eq!(gpu.global_thread_id(0, 0, 0), 0);
        assert_eq!(gpu.global_thread_id(0, 0, 3), 3);
        assert_eq!(gpu.global_thread_id(0, 1, 0), 4);
        assert_eq!(gpu.global_thread_id(1, 0, 0), 8);
    }

    #[test]
    fn test_gpu_hierarchy_decompose_id() {
        let gpu = GpuHierarchicalSymmetry::new(
            4, 2, 2,
            PermutationGroup::trivial(4),
            PermutationGroup::trivial(2),
            PermutationGroup::trivial(2),
        );

        assert_eq!(gpu.decompose_thread_id(0), (0, 0, 0));
        assert_eq!(gpu.decompose_thread_id(3), (0, 0, 3));
        assert_eq!(gpu.decompose_thread_id(4), (0, 1, 0));
        assert_eq!(gpu.decompose_thread_id(8), (1, 0, 0));
        assert_eq!(gpu.decompose_thread_id(15), (1, 1, 3));
    }

    #[test]
    fn test_gpu_level_factors() {
        let gpu = GpuHierarchicalSymmetry::new(
            2, 2, 1,
            PermutationGroup::symmetric(2),
            PermutationGroup::symmetric(2),
            PermutationGroup::trivial(1),
        );

        let factors = gpu.level_factors();
        assert_eq!(*factors.get(&GpuLevel::Thread).unwrap() as u64, 2);
        assert_eq!(*factors.get(&GpuLevel::Warp).unwrap() as u64, 2);
        assert_eq!(*factors.get(&GpuLevel::Cta).unwrap() as u64, 1);
    }

    #[test]
    fn test_gpu_summary() {
        let gpu = GpuHierarchicalSymmetry::new(
            2, 2, 2,
            PermutationGroup::trivial(2),
            PermutationGroup::trivial(2),
            PermutationGroup::trivial(2),
        );
        let summary = gpu.summary();
        assert!(summary.contains("8 threads"));
    }

    #[test]
    fn test_multi_level_wreath() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let s3 = PermutationGroup::symmetric(3);

        let wreath = multi_level_wreath(&[c2, s2, s3]);
        assert!(wreath.is_some());
    }

    #[test]
    fn test_multi_level_wreath_empty() {
        let result = multi_level_wreath(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_multi_level_wreath_single() {
        let c3 = PermutationGroup::cyclic(3);
        let result = multi_level_wreath(&[c3]);
        assert!(result.is_some());
    }

    #[test]
    fn test_hierarchical_orbit_enumeration() {
        let gpu = GpuHierarchicalSymmetry::new(
            2, 1, 1,
            PermutationGroup::symmetric(2),
            PermutationGroup::trivial(1),
            PermutationGroup::trivial(1),
        );

        let sym = FullSymmetryGroup {
            thread_group: PermutationGroup::symmetric(2),
            address_group: PermutationGroup::trivial(2),
            value_group: PermutationGroup::trivial(2),
            total_order: 2,
            compression_factor: 2.0,
        };

        let hoe = HierarchicalOrbitEnumeration::new(gpu, sym);

        // Create some candidates
        let mut c1 = ExecutionCandidate::new();
        c1.reads_from.insert((0, 0), (1, 0));
        let mut c2 = ExecutionCandidate::new();
        c2.reads_from.insert((1, 0), (0, 0));

        let (orbits, stats) = hoe.enumerate(&[c1, c2]);
        assert!(orbits.len() >= 1);
        assert_eq!(stats.total_input, 2);
    }

    #[test]
    fn test_hierarchical_stats_display() {
        let mut stats = HierarchicalStats::new();
        stats.total_input = 100;
        stats.after_thread_level = 50;
        stats.after_warp_level = 30;
        stats.after_cta_level = 20;
        stats.final_orbits = 20;
        let display = format!("{}", stats);
        assert!(display.contains("100"));
        assert!(display.contains("20"));
    }

    #[test]
    fn test_hierarchical_stats_compression() {
        let mut stats = HierarchicalStats::new();
        stats.total_input = 100;
        stats.after_thread_level = 50;
        stats.after_warp_level = 25;
        stats.after_cta_level = 10;
        stats.final_orbits = 10;

        assert!((stats.thread_compression() - 2.0).abs() < 0.01);
        assert!((stats.warp_compression() - 2.0).abs() < 0.01);
        assert!((stats.cta_compression() - 2.5).abs() < 0.01);
        assert!((stats.total_compression() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_wreath_product_display() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);
        let display = format!("{}", wp);
        assert!(display.contains("WreathProduct"));
        assert!(display.contains("8"));
    }

    #[test]
    fn test_detect_hierarchical_iriw() {
        let iriw = litmus_iriw();
        let result = detect_hierarchical_structure(&iriw);
        // IRIW has 4 threads, might match (2,2,1) or (2,1,2)
        // Whether hierarchical structure is detected depends on the symmetries
        // Just check it doesn't panic
        if let Some(hier) = result {
            assert_eq!(hier.total_threads(), 4);
        }
    }

    #[test]
    fn test_wreath_identity_decomposition() {
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&s2, &s2);

        let id = Permutation::identity(4);
        let (bases, top) = wp.decompose(&id);

        assert!(top.is_identity());
        for b in &bases {
            assert!(b.is_identity());
        }
    }

    #[test]
    fn test_wreath_product_order_c3_wr_s2() {
        let c3 = PermutationGroup::cyclic(3);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c3, &s2);
        // |C_3 ≀ S_2| = 3^2 * 2 = 18
        assert_eq!(wp.order(), 18);
    }

    #[test]
    fn test_wreath_full_group_order() {
        let c2 = PermutationGroup::cyclic(2);
        let s2 = PermutationGroup::symmetric(2);
        let wp = WreathProduct::new(&c2, &s2);
        assert_eq!(wp.full_group.order(), 8);
    }
}
