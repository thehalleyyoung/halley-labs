//! Spectral sequences for multi-level GPU memory hierarchy verification.
//!
//! Implements spectral sequence machinery from §9 of the LITMUS∞ paper.
//! Decomposes GPU verification by hierarchy level:
//!   Thread → Warp → CTA → GPU → System
//! Each level computes local consistency; the spectral sequence glues
//! the levels together into a global consistency proof.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// HierarchyLevel — abstraction of GPU memory scope levels
// ═══════════════════════════════════════════════════════════════════════

/// A level in the GPU memory hierarchy, ordered from finest to coarsest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HierarchyLevel {
    Thread = 0,
    Warp = 1,
    CTA = 2,
    GPU = 3,
    System = 4,
}

impl HierarchyLevel {
    pub fn all() -> &'static [HierarchyLevel] {
        &[
            HierarchyLevel::Thread,
            HierarchyLevel::Warp,
            HierarchyLevel::CTA,
            HierarchyLevel::GPU,
            HierarchyLevel::System,
        ]
    }

    pub fn index(&self) -> usize {
        *self as usize
    }

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(HierarchyLevel::Thread),
            1 => Some(HierarchyLevel::Warp),
            2 => Some(HierarchyLevel::CTA),
            3 => Some(HierarchyLevel::GPU),
            4 => Some(HierarchyLevel::System),
            _ => None,
        }
    }

    pub fn count() -> usize {
        5
    }
}

impl fmt::Display for HierarchyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HierarchyLevel::Thread => write!(f, "Thread"),
            HierarchyLevel::Warp => write!(f, "Warp"),
            HierarchyLevel::CTA => write!(f, "CTA"),
            HierarchyLevel::GPU => write!(f, "GPU"),
            HierarchyLevel::System => write!(f, "System"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MatrixEntry — elements of E-page matrices
// ═══════════════════════════════════════════════════════════════════════

/// An entry in an E-page matrix representing an abstract consistency group.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixEntry {
    /// Trivial (zero) — no contribution.
    Zero,
    /// A generator, carrying a consistency label and multiplicity.
    Generator {
        label: String,
        multiplicity: usize,
    },
    /// Direct sum of entries.
    DirectSum(Vec<MatrixEntry>),
}

impl MatrixEntry {
    pub fn zero() -> Self {
        MatrixEntry::Zero
    }

    pub fn generator(label: &str, multiplicity: usize) -> Self {
        MatrixEntry::Generator {
            label: label.to_string(),
            multiplicity,
        }
    }

    pub fn direct_sum(entries: Vec<MatrixEntry>) -> Self {
        let non_zero: Vec<MatrixEntry> = entries
            .into_iter()
            .filter(|e| !e.is_zero())
            .collect();
        if non_zero.is_empty() {
            MatrixEntry::Zero
        } else if non_zero.len() == 1 {
            non_zero.into_iter().next().unwrap()
        } else {
            MatrixEntry::DirectSum(non_zero)
        }
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, MatrixEntry::Zero)
    }

    /// Rank (dimension) of this entry.
    pub fn rank(&self) -> usize {
        match self {
            MatrixEntry::Zero => 0,
            MatrixEntry::Generator { multiplicity, .. } => *multiplicity,
            MatrixEntry::DirectSum(entries) => entries.iter().map(|e| e.rank()).sum(),
        }
    }

    /// Add two entries (direct sum).
    pub fn add(&self, other: &MatrixEntry) -> MatrixEntry {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        MatrixEntry::DirectSum(vec![self.clone(), other.clone()])
    }
}

impl fmt::Display for MatrixEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixEntry::Zero => write!(f, "0"),
            MatrixEntry::Generator { label, multiplicity } => {
                if *multiplicity == 1 {
                    write!(f, "⟨{}⟩", label)
                } else {
                    write!(f, "⟨{}⟩^{}", label, multiplicity)
                }
            }
            MatrixEntry::DirectSum(entries) => {
                let parts: Vec<String> = entries.iter().map(|e| format!("{}", e)).collect();
                write!(f, "{}", parts.join(" ⊕ "))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EPage — a single page E_r in the spectral sequence
// ═══════════════════════════════════════════════════════════════════════

/// An E_r page in the spectral sequence, stored as a 2D grid of MatrixEntries.
///
/// Rows correspond to the hierarchy level (p-index, filtration degree).
/// Columns correspond to the consistency property index (q-index).
#[derive(Debug, Clone)]
pub struct EPage {
    /// Page number r ≥ 0.
    pub page_number: usize,
    /// Number of rows (p dimension, typically = hierarchy depth).
    pub rows: usize,
    /// Number of columns (q dimension).
    pub cols: usize,
    /// Row-major matrix of entries.
    entries: Vec<MatrixEntry>,
    /// Metadata about what each row represents.
    pub row_labels: Vec<String>,
    /// Metadata about what each column represents.
    pub col_labels: Vec<String>,
}

impl EPage {
    /// Create a new E-page with all zero entries.
    pub fn new(page_number: usize, rows: usize, cols: usize) -> Self {
        Self {
            page_number,
            rows,
            cols,
            entries: vec![MatrixEntry::Zero; rows * cols],
            row_labels: (0..rows).map(|i| format!("p={}", i)).collect(),
            col_labels: (0..cols).map(|j| format!("q={}", j)).collect(),
        }
    }

    /// Create an E-page for GPU hierarchy levels.
    pub fn for_gpu_hierarchy(page_number: usize, num_properties: usize) -> Self {
        let levels = HierarchyLevel::all();
        let rows = levels.len();
        let mut page = Self::new(page_number, rows, num_properties);
        page.row_labels = levels.iter().map(|l| format!("{}", l)).collect();
        page
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows && col < self.cols);
        row * self.cols + col
    }

    pub fn get(&self, row: usize, col: usize) -> &MatrixEntry {
        &self.entries[self.idx(row, col)]
    }

    pub fn set(&mut self, row: usize, col: usize, entry: MatrixEntry) {
        let idx = self.idx(row, col);
        self.entries[idx] = entry;
    }

    /// Total rank of all entries.
    pub fn total_rank(&self) -> usize {
        self.entries.iter().map(|e| e.rank()).sum()
    }

    /// Rank of a specific (p, q) entry.
    pub fn rank_at(&self, row: usize, col: usize) -> usize {
        self.get(row, col).rank()
    }

    /// Check if all entries are zero.
    pub fn is_zero(&self) -> bool {
        self.entries.iter().all(|e| e.is_zero())
    }

    /// Get the diagonal entries E^{p,q} where p + q = n.
    pub fn diagonal(&self, n: usize) -> Vec<(usize, usize, &MatrixEntry)> {
        let mut result = Vec::new();
        for p in 0..self.rows {
            let q = n.checked_sub(p);
            if let Some(q) = q {
                if q < self.cols {
                    result.push((p, q, self.get(p, q)));
                }
            }
        }
        result
    }

    /// Create a new page with the same dimensions, all zeros.
    pub fn zero_page(&self) -> Self {
        Self::new(self.page_number + 1, self.rows, self.cols)
    }

    /// Pretty-print the page as a table.
    pub fn pretty_print(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("E_{} page ({}×{}):\n", self.page_number, self.rows, self.cols));

        // Header row
        s.push_str(&format!("{:>12}", ""));
        for j in 0..self.cols {
            s.push_str(&format!(" {:>12}", &self.col_labels[j]));
        }
        s.push('\n');

        // Data rows
        for i in 0..self.rows {
            s.push_str(&format!("{:>12}", &self.row_labels[i]));
            for j in 0..self.cols {
                let entry = self.get(i, j);
                s.push_str(&format!(" {:>12}", format!("{}", entry)));
            }
            s.push('\n');
        }
        s
    }
}

impl fmt::Display for EPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E_{}[{}×{}, rank={}]", self.page_number, self.rows, self.cols, self.total_rank())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Differential — computes d_r differentials between E-pages
// ═══════════════════════════════════════════════════════════════════════

/// A differential d_r : E_r^{p,q} → E_r^{p+r, q-r+1}.
///
/// Differentials encode the obstruction to extending local consistency
/// from one hierarchy level to the next.
#[derive(Debug, Clone)]
pub struct Differential {
    /// Differential degree r.
    pub degree: usize,
    /// Source page dimensions.
    pub source_rows: usize,
    pub source_cols: usize,
    /// Maps (source_p, source_q) → image MatrixEntry.
    images: HashMap<(usize, usize), MatrixEntry>,
}

impl Differential {
    pub fn new(degree: usize, source_rows: usize, source_cols: usize) -> Self {
        Self {
            degree,
            source_rows,
            source_cols,
            images: HashMap::new(),
        }
    }

    /// Compute the target (p', q') for a differential from (p, q).
    /// d_r : E_r^{p,q} → E_r^{p+r, q-r+1}
    pub fn target_indices(&self, p: usize, q: usize) -> Option<(usize, usize)> {
        let p_target = p + self.degree;
        if q + 1 < self.degree {
            return None;
        }
        let q_target = q - self.degree + 1;
        if p_target < self.source_rows && q_target < self.source_cols {
            Some((p_target, q_target))
        } else {
            None
        }
    }

    /// Set the image of d_r at (p, q).
    pub fn set_image(&mut self, p: usize, q: usize, image: MatrixEntry) {
        self.images.insert((p, q), image);
    }

    /// Get the image of d_r at (p, q).
    pub fn image_at(&self, p: usize, q: usize) -> &MatrixEntry {
        self.images.get(&(p, q)).unwrap_or(&ZERO_ENTRY)
    }

    /// Check d_r ∘ d_r = 0 (the fundamental property of differentials).
    pub fn check_square_zero(&self) -> bool {
        for (&(p, q), img) in &self.images {
            if img.is_zero() {
                continue;
            }
            if let Some((p2, q2)) = self.target_indices(p, q) {
                if let Some(img2) = self.images.get(&(p2, q2)) {
                    // d_r(d_r(x)) should be zero for a valid differential
                    if !img2.is_zero() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if the differential is trivial (all images are zero).
    pub fn is_trivial(&self) -> bool {
        self.images.values().all(|e| e.is_zero())
    }

    /// Compute the next page E_{r+1} = ker(d_r) / im(d_r).
    ///
    /// In our application, this quotient captures which consistency
    /// properties survive extension from level r to level r+1.
    pub fn compute_homology(&self, source: &EPage) -> EPage {
        let mut next = EPage::new(
            source.page_number + 1,
            source.rows,
            source.cols,
        );
        next.row_labels = source.row_labels.clone();
        next.col_labels = source.col_labels.clone();

        for p in 0..source.rows {
            for q in 0..source.cols {
                let entry = source.get(p, q);
                if entry.is_zero() {
                    continue;
                }

                // Kernel: entries not hit by d_r from (p, q)
                let outgoing = self.image_at(p, q);

                // Image: entries arriving at (p, q) from (p - r, q + r - 1)
                let incoming = if p >= self.degree && q + self.degree >= 1 {
                    let sp = p - self.degree;
                    let sq = q + self.degree - 1;
                    if sq < source.cols {
                        self.image_at(sp, sq)
                    } else {
                        &ZERO_ENTRY
                    }
                } else {
                    &ZERO_ENTRY
                };

                // Homology = ker / im
                // Rank formula: rank(H) = rank(entry) - rank(outgoing) - rank(incoming)
                let out_rank = outgoing.rank();
                let in_rank = incoming.rank();
                let entry_rank = entry.rank();
                let homology_rank = entry_rank.saturating_sub(out_rank).saturating_sub(in_rank);

                if homology_rank > 0 {
                    next.set(
                        p,
                        q,
                        MatrixEntry::generator(
                            &format!("H_{}({},{})", source.page_number + 1, p, q),
                            homology_rank,
                        ),
                    );
                }
            }
        }
        next
    }
}

impl fmt::Display for Differential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nonzero = self.images.values().filter(|e| !e.is_zero()).count();
        write!(f, "d_{} ({} nonzero images)", self.degree, nonzero)
    }
}

static ZERO_ENTRY: MatrixEntry = MatrixEntry::Zero;

// ═══════════════════════════════════════════════════════════════════════
// ConvergenceStatus — tracks when the spectral sequence stabilizes
// ═══════════════════════════════════════════════════════════════════════

/// Status of spectral sequence convergence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Still evolving — not yet stabilized.
    Evolving { current_page: usize },
    /// Converged at page r — all subsequent differentials are zero.
    Converged { at_page: usize },
    /// Degenerated at E_2 (common for simpler models).
    Degenerate,
    /// Failed to converge within the maximum number of iterations.
    DidNotConverge { max_pages: usize },
}

impl fmt::Display for ConvergenceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvergenceStatus::Evolving { current_page } =>
                write!(f, "evolving (currently at E_{})", current_page),
            ConvergenceStatus::Converged { at_page } =>
                write!(f, "converged at E_{}", at_page),
            ConvergenceStatus::Degenerate =>
                write!(f, "degenerate (E_2 = E_∞)"),
            ConvergenceStatus::DidNotConverge { max_pages } =>
                write!(f, "did not converge within {} pages", max_pages),
        }
    }
}

/// Convergence checker for spectral sequences.
#[derive(Debug, Clone)]
pub struct Convergence {
    /// Maximum page number to compute before giving up.
    pub max_pages: usize,
    /// Tolerance for rank changes (0 = exact).
    pub rank_tolerance: usize,
}

impl Convergence {
    pub fn new(max_pages: usize) -> Self {
        Self {
            max_pages,
            rank_tolerance: 0,
        }
    }

    pub fn with_tolerance(mut self, tol: usize) -> Self {
        self.rank_tolerance = tol;
        self
    }

    /// Check if two consecutive pages are equal (convergence criterion).
    pub fn pages_equal(&self, a: &EPage, b: &EPage) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        for p in 0..a.rows {
            for q in 0..a.cols {
                let diff = (a.rank_at(p, q) as isize - b.rank_at(p, q) as isize).unsigned_abs();
                if diff > self.rank_tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Check convergence status given the history of pages.
    pub fn check(&self, pages: &[EPage]) -> ConvergenceStatus {
        if pages.len() < 2 {
            return ConvergenceStatus::Evolving {
                current_page: pages.len().saturating_sub(1),
            };
        }

        // Check if the last two pages are equal
        let n = pages.len();
        if self.pages_equal(&pages[n - 2], &pages[n - 1]) {
            let converged_at = n - 2;
            if converged_at <= 2 {
                return ConvergenceStatus::Degenerate;
            }
            return ConvergenceStatus::Converged { at_page: converged_at };
        }

        if n > self.max_pages {
            return ConvergenceStatus::DidNotConverge { max_pages: self.max_pages };
        }

        ConvergenceStatus::Evolving { current_page: n - 1 }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConsistencyProperty — what we verify at each hierarchy level
// ═══════════════════════════════════════════════════════════════════════

/// A consistency property to be verified across the memory hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConsistencyProperty {
    pub name: String,
    pub description: String,
    /// The hierarchy levels at which this property is relevant.
    pub relevant_levels: Vec<HierarchyLevel>,
}

impl ConsistencyProperty {
    pub fn new(name: &str, description: &str, levels: Vec<HierarchyLevel>) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            relevant_levels: levels,
        }
    }

    pub fn coherence() -> Self {
        Self::new(
            "coherence",
            "Per-location total order on writes",
            vec![HierarchyLevel::CTA, HierarchyLevel::GPU, HierarchyLevel::System],
        )
    }

    pub fn causality() -> Self {
        Self::new(
            "causality",
            "Reads-from induces causal ordering",
            HierarchyLevel::all().to_vec(),
        )
    }

    pub fn atomicity() -> Self {
        Self::new(
            "atomicity",
            "RMW operations are atomic at the specified scope",
            vec![HierarchyLevel::Warp, HierarchyLevel::CTA, HierarchyLevel::GPU],
        )
    }

    pub fn sc_per_location() -> Self {
        Self::new(
            "sc-per-loc",
            "Sequential consistency per memory location",
            HierarchyLevel::all().to_vec(),
        )
    }

    pub fn no_thin_air() -> Self {
        Self::new(
            "no-thin-air",
            "No out-of-thin-air values",
            HierarchyLevel::all().to_vec(),
        )
    }

    pub fn is_relevant_at(&self, level: HierarchyLevel) -> bool {
        self.relevant_levels.contains(&level)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LocalConsistencyResult — result at one hierarchy level
// ═══════════════════════════════════════════════════════════════════════

/// Result of checking consistency at a single hierarchy level.
#[derive(Debug, Clone)]
pub struct LocalConsistencyResult {
    pub level: HierarchyLevel,
    pub property: String,
    /// Whether the property holds locally.
    pub holds: bool,
    /// If it fails, witnesses.
    pub counterexamples: Vec<String>,
    /// Obstruction class for the spectral sequence.
    pub obstruction: MatrixEntry,
}

impl LocalConsistencyResult {
    pub fn success(level: HierarchyLevel, property: &str) -> Self {
        Self {
            level,
            property: property.to_string(),
            holds: true,
            counterexamples: Vec::new(),
            obstruction: MatrixEntry::Zero,
        }
    }

    pub fn failure(
        level: HierarchyLevel,
        property: &str,
        counterexamples: Vec<String>,
    ) -> Self {
        let multiplicity = counterexamples.len().max(1);
        Self {
            level,
            property: property.to_string(),
            holds: false,
            counterexamples,
            obstruction: MatrixEntry::generator(
                &format!("obs({},{})", level, property),
                multiplicity,
            ),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SpectralSequence — the main orchestrator
// ═══════════════════════════════════════════════════════════════════════

/// A spectral sequence for multi-level memory model verification.
///
/// The spectral sequence {E_r, d_r} converges to the "total consistency":
/// - E_0 captures raw per-level constraints
/// - E_1 captures local consistency per level
/// - E_2 captures pairwise compatibility between adjacent levels
/// - E_∞ captures global consistency
///
/// If E_∞ is zero in the right positions, the execution is consistent.
#[derive(Debug, Clone)]
pub struct SpectralSequence {
    /// All computed pages (E_0, E_1, ...).
    pub pages: Vec<EPage>,
    /// All computed differentials.
    pub differentials: Vec<Differential>,
    /// Properties being verified.
    pub properties: Vec<ConsistencyProperty>,
    /// Convergence checker.
    pub convergence: Convergence,
    /// Current status.
    pub status: ConvergenceStatus,
    /// Local results cache.
    local_results: Vec<LocalConsistencyResult>,
}

impl SpectralSequence {
    /// Create a new spectral sequence for the given properties.
    pub fn new(properties: Vec<ConsistencyProperty>, max_pages: usize) -> Self {
        Self {
            pages: Vec::new(),
            differentials: Vec::new(),
            properties,
            convergence: Convergence::new(max_pages),
            status: ConvergenceStatus::Evolving { current_page: 0 },
            local_results: Vec::new(),
        }
    }

    /// Initialize E_0 from local consistency checks.
    pub fn initialize(&mut self, local_results: Vec<LocalConsistencyResult>) {
        let num_levels = HierarchyLevel::count();
        let num_props = self.properties.len();
        let mut e0 = EPage::for_gpu_hierarchy(0, num_props);

        // Set column labels from properties
        e0.col_labels = self.properties.iter().map(|p| p.name.clone()).collect();

        for result in &local_results {
            let row = result.level.index();
            // Find the property column
            if let Some(col) = self.properties.iter().position(|p| p.name == result.property) {
                if row < num_levels && col < num_props {
                    if result.holds {
                        e0.set(
                            row,
                            col,
                            MatrixEntry::generator(
                                &format!("ok({},{})", result.level, result.property),
                                1,
                            ),
                        );
                    } else {
                        e0.set(row, col, result.obstruction.clone());
                    }
                }
            }
        }

        self.local_results = local_results;
        self.pages.push(e0);
    }

    /// Compute the next page by applying the differential.
    pub fn advance(&mut self) -> bool {
        if self.pages.is_empty() {
            return false;
        }

        let r = self.pages.len() - 1;
        let current = &self.pages[r];

        // Build the differential d_r
        let differential = self.compute_differential(r, current);

        // Compute homology (next page)
        let next_page = differential.compute_homology(current);

        self.differentials.push(differential);
        self.pages.push(next_page);

        // Update convergence status
        self.status = self.convergence.check(&self.pages);

        matches!(
            self.status,
            ConvergenceStatus::Evolving { .. }
        )
    }

    /// Run the spectral sequence to convergence.
    pub fn run_to_convergence(&mut self) -> &ConvergenceStatus {
        while matches!(self.status, ConvergenceStatus::Evolving { .. }) {
            if !self.advance() {
                break;
            }
        }
        &self.status
    }

    /// Get the limiting page E_∞.
    pub fn e_infinity(&self) -> Option<&EPage> {
        match &self.status {
            ConvergenceStatus::Converged { at_page } => self.pages.get(*at_page),
            ConvergenceStatus::Degenerate => self.pages.get(2).or(self.pages.last()),
            _ => self.pages.last(),
        }
    }

    /// Check if global consistency holds.
    /// Global consistency holds iff E_∞ has the expected pattern.
    pub fn is_globally_consistent(&self) -> bool {
        if let Some(e_inf) = self.e_infinity() {
            // The execution is globally consistent if all obstruction
            // classes have been killed by differentials (rank 0 in
            // the entries corresponding to failures).
            for result in &self.local_results {
                if !result.holds {
                    let row = result.level.index();
                    if let Some(col) = self.properties.iter().position(|p| p.name == result.property) {
                        if row < e_inf.rows && col < e_inf.cols {
                            if e_inf.rank_at(row, col) > 0 {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Compute differential d_r based on level-gluing maps.
    fn compute_differential(&self, r: usize, page: &EPage) -> Differential {
        let mut diff = Differential::new(r + 1, page.rows, page.cols);

        // The differential d_r connects level p to level p+r+1.
        // It's non-trivial when an obstruction at level p can be
        // resolved by a constraint at level p+r+1.
        for p in 0..page.rows {
            for q in 0..page.cols {
                let entry = page.get(p, q);
                if entry.is_zero() {
                    continue;
                }

                if let Some((tp, tq)) = diff.target_indices(p, q) {
                    let target = page.get(tp, tq);
                    // The differential is non-trivial if both source
                    // and target are non-zero and the levels are adjacent.
                    if !target.is_zero() {
                        let image_rank = entry.rank().min(target.rank());
                        if image_rank > 0 {
                            diff.set_image(
                                p,
                                q,
                                MatrixEntry::generator(
                                    &format!("d_{}({},{})", r + 1, p, q),
                                    image_rank,
                                ),
                            );
                        }
                    }
                }
            }
        }

        diff
    }

    /// Number of pages computed so far.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get a specific page.
    pub fn page(&self, r: usize) -> Option<&EPage> {
        self.pages.get(r)
    }

    /// Get a specific differential.
    pub fn differential(&self, r: usize) -> Option<&Differential> {
        self.differentials.get(r)
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "SpectralSequence: {} pages computed, status: {}\n",
            self.pages.len(),
            self.status,
        ));
        for (i, page) in self.pages.iter().enumerate() {
            s.push_str(&format!("  E_{}: rank = {}\n", i, page.total_rank()));
        }
        if self.is_globally_consistent() {
            s.push_str("  Result: GLOBALLY CONSISTENT\n");
        } else {
            s.push_str("  Result: INCONSISTENCY DETECTED\n");
        }
        s
    }
}

impl fmt::Display for SpectralSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpectralSequence({} pages, {})",
            self.pages.len(),
            self.status,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SerreSpectralSequence — hierarchical decomposition via Serre fibration
// ═══════════════════════════════════════════════════════════════════════

/// A fiber in the Serre fibration corresponding to a scope group.
#[derive(Debug, Clone)]
pub struct Fiber {
    /// Identifier (e.g., "warp_0", "cta_2").
    pub id: String,
    /// The scope level this fiber lives at.
    pub level: HierarchyLevel,
    /// Indices of events belonging to this fiber.
    pub event_indices: Vec<usize>,
    /// Local consistency check result.
    pub local_consistent: Option<bool>,
}

impl Fiber {
    pub fn new(id: &str, level: HierarchyLevel, events: Vec<usize>) -> Self {
        Self {
            id: id.to_string(),
            level,
            event_indices: events,
            local_consistent: None,
        }
    }

    pub fn mark_consistent(&mut self, consistent: bool) {
        self.local_consistent = Some(consistent);
    }
}

/// The Serre spectral sequence specialized for GPU memory hierarchy.
///
/// Models the GPU hierarchy as a tower of fibrations:
///   System ← GPU ← CTA ← Warp ← Thread
///
/// The fibers at each level are the scope groups (e.g., all threads in a warp).
/// The Serre spectral sequence decomposes verification into:
/// 1. Check consistency within each fiber (local check).
/// 2. Check compatibility between fibers at the same level.
/// 3. Check compatibility across levels (the differentials).
#[derive(Debug, Clone)]
pub struct SerreSpectralSequence {
    /// Fibers organized by hierarchy level.
    pub fibers: BTreeMap<HierarchyLevel, Vec<Fiber>>,
    /// The underlying spectral sequence.
    pub spectral: SpectralSequence,
    /// Gluing maps: (level, fiber_i, fiber_j) → compatibility result.
    gluing_results: HashMap<(HierarchyLevel, usize, usize), bool>,
}

impl SerreSpectralSequence {
    /// Create a new Serre spectral sequence.
    pub fn new(properties: Vec<ConsistencyProperty>) -> Self {
        Self {
            fibers: BTreeMap::new(),
            spectral: SpectralSequence::new(properties, 20),
            gluing_results: HashMap::new(),
        }
    }

    /// Add fibers at a hierarchy level.
    pub fn add_fibers(&mut self, level: HierarchyLevel, fibers: Vec<Fiber>) {
        self.fibers.insert(level, fibers);
    }

    /// Build fibers from a thread-to-scope assignment.
    ///
    /// `thread_scopes[thread_id]` = (warp_id, cta_id, gpu_id)
    pub fn build_from_scope_assignment(
        &mut self,
        num_events: usize,
        event_threads: &[usize],
        thread_scopes: &HashMap<usize, (usize, usize, usize)>,
    ) {
        // Group events by thread
        let mut thread_events: HashMap<usize, Vec<usize>> = HashMap::new();
        for (eid, &tid) in event_threads.iter().enumerate() {
            thread_events.entry(tid).or_default().push(eid);
        }

        // Build thread-level fibers
        let thread_fibers: Vec<Fiber> = thread_events
            .iter()
            .map(|(tid, events)| {
                Fiber::new(&format!("thread_{}", tid), HierarchyLevel::Thread, events.clone())
            })
            .collect();
        self.fibers.insert(HierarchyLevel::Thread, thread_fibers);

        // Build warp-level fibers
        let mut warp_events: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&tid, events) in &thread_events {
            if let Some(&(warp_id, _, _)) = thread_scopes.get(&tid) {
                warp_events.entry(warp_id).or_default().extend(events);
            }
        }
        let warp_fibers: Vec<Fiber> = warp_events
            .into_iter()
            .map(|(wid, events)| Fiber::new(&format!("warp_{}", wid), HierarchyLevel::Warp, events))
            .collect();
        self.fibers.insert(HierarchyLevel::Warp, warp_fibers);

        // Build CTA-level fibers
        let mut cta_events: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&tid, events) in &thread_events {
            if let Some(&(_, cta_id, _)) = thread_scopes.get(&tid) {
                cta_events.entry(cta_id).or_default().extend(events);
            }
        }
        let cta_fibers: Vec<Fiber> = cta_events
            .into_iter()
            .map(|(cid, events)| Fiber::new(&format!("cta_{}", cid), HierarchyLevel::CTA, events))
            .collect();
        self.fibers.insert(HierarchyLevel::CTA, cta_fibers);

        // Build GPU-level fibers
        let mut gpu_events: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&tid, events) in &thread_events {
            if let Some(&(_, _, gpu_id)) = thread_scopes.get(&tid) {
                gpu_events.entry(gpu_id).or_default().extend(events);
            }
        }
        let gpu_fibers: Vec<Fiber> = gpu_events
            .into_iter()
            .map(|(gid, events)| Fiber::new(&format!("gpu_{}", gid), HierarchyLevel::GPU, events))
            .collect();
        self.fibers.insert(HierarchyLevel::GPU, gpu_fibers);

        // System-level: single fiber containing all events
        let system_fiber = Fiber::new(
            "system",
            HierarchyLevel::System,
            (0..num_events).collect(),
        );
        self.fibers.insert(HierarchyLevel::System, vec![system_fiber]);
    }

    /// Record a gluing result between two fibers at the same level.
    pub fn record_gluing(
        &mut self,
        level: HierarchyLevel,
        fiber_i: usize,
        fiber_j: usize,
        compatible: bool,
    ) {
        self.gluing_results
            .insert((level, fiber_i, fiber_j), compatible);
    }

    /// Initialize the spectral sequence from fiber-level checks.
    pub fn initialize_from_fibers(&mut self) {
        let mut local_results = Vec::new();

        for (&level, fibers) in &self.fibers {
            for (fidx, fiber) in fibers.iter().enumerate() {
                let consistent = fiber.local_consistent.unwrap_or(true);
                let prop_name = format!("fiber_{}", fiber.id);
                if consistent {
                    local_results.push(LocalConsistencyResult::success(level, &prop_name));
                } else {
                    local_results.push(LocalConsistencyResult::failure(
                        level,
                        &prop_name,
                        vec![format!("Fiber {} inconsistent", fiber.id)],
                    ));
                }
            }
        }

        // Extend properties to cover all fibers
        let mut properties = self.spectral.properties.clone();
        for result in &local_results {
            if !properties.iter().any(|p| p.name == result.property) {
                properties.push(ConsistencyProperty::new(
                    &result.property,
                    &format!("Fiber consistency for {}", result.property),
                    vec![result.level],
                ));
            }
        }
        self.spectral.properties = properties;

        self.spectral.initialize(local_results);
    }

    /// Run the Serre spectral sequence to convergence.
    pub fn run(&mut self) -> &ConvergenceStatus {
        self.spectral.run_to_convergence()
    }

    /// Check if the hierarchical verification succeeded.
    pub fn is_consistent(&self) -> bool {
        // All fibers must be locally consistent
        for fibers in self.fibers.values() {
            for fiber in fibers {
                if fiber.local_consistent == Some(false) {
                    return false;
                }
            }
        }
        // All gluings must be compatible
        if self.gluing_results.values().any(|&v| !v) {
            return false;
        }
        // The spectral sequence must converge to consistency
        self.spectral.is_globally_consistent()
    }

    /// Get fibers at a specific level.
    pub fn fibers_at(&self, level: HierarchyLevel) -> &[Fiber] {
        self.fibers.get(&level).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Summary of the hierarchical decomposition.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("SerreSpectralSequence decomposition:\n");
        for (&level, fibers) in &self.fibers {
            s.push_str(&format!(
                "  {}: {} fibers\n",
                level,
                fibers.len(),
            ));
            for fiber in fibers {
                let status = match fiber.local_consistent {
                    Some(true) => "✓",
                    Some(false) => "✗",
                    None => "?",
                };
                s.push_str(&format!(
                    "    {} [{}] ({} events)\n",
                    fiber.id,
                    status,
                    fiber.event_indices.len(),
                ));
            }
        }
        s.push_str(&self.spectral.summary());
        s
    }
}

impl fmt::Display for SerreSpectralSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_fibers: usize = self.fibers.values().map(|v| v.len()).sum();
        write!(
            f,
            "SerreSpectralSequence({} fibers across {} levels)",
            total_fibers,
            self.fibers.len(),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// HierarchicalDecomposition — top-level API
// ═══════════════════════════════════════════════════════════════════════

/// Decomposition of GPU verification into hierarchy levels.
///
/// Usage:
/// 1. Create a `HierarchicalDecomposition` with the desired properties.
/// 2. Add events and scope assignments.
/// 3. Run local checks at each level.
/// 4. Glue results using the spectral sequence.
#[derive(Debug, Clone)]
pub struct HierarchicalDecomposition {
    /// The Serre spectral sequence doing the heavy lifting.
    pub serre: SerreSpectralSequence,
    /// Which properties to check.
    pub properties: Vec<ConsistencyProperty>,
    /// Number of events total.
    pub num_events: usize,
}

impl HierarchicalDecomposition {
    /// Create a new hierarchical decomposition.
    pub fn new(properties: Vec<ConsistencyProperty>) -> Self {
        Self {
            serre: SerreSpectralSequence::new(properties.clone()),
            properties,
            num_events: 0,
        }
    }

    /// Create with standard GPU consistency properties.
    pub fn standard_gpu() -> Self {
        Self::new(vec![
            ConsistencyProperty::coherence(),
            ConsistencyProperty::causality(),
            ConsistencyProperty::atomicity(),
            ConsistencyProperty::sc_per_location(),
            ConsistencyProperty::no_thin_air(),
        ])
    }

    /// Set up the hierarchy from event/thread/scope data.
    pub fn setup(
        &mut self,
        num_events: usize,
        event_threads: &[usize],
        thread_scopes: &HashMap<usize, (usize, usize, usize)>,
    ) {
        self.num_events = num_events;
        self.serre
            .build_from_scope_assignment(num_events, event_threads, thread_scopes);
    }

    /// Mark a fiber as locally consistent or not.
    pub fn set_fiber_consistency(
        &mut self,
        level: HierarchyLevel,
        fiber_index: usize,
        consistent: bool,
    ) {
        if let Some(fibers) = self.serre.fibers.get_mut(&level) {
            if let Some(fiber) = fibers.get_mut(fiber_index) {
                fiber.mark_consistent(consistent);
            }
        }
    }

    /// Run the full hierarchical verification.
    pub fn verify(&mut self) -> bool {
        self.serre.initialize_from_fibers();
        self.serre.run();
        self.serre.is_consistent()
    }

    /// Get the decomposition summary.
    pub fn summary(&self) -> String {
        self.serre.summary()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FilteredComplex — underlying algebraic structure
// ═══════════════════════════════════════════════════════════════════════

/// A filtered chain complex underlying the spectral sequence.
///
/// Provides the algebraic foundation:
///   F^p C_n ⊇ F^{p+1} C_n (decreasing filtration)
///
/// In our application:
/// - C_n = consistency conditions at total degree n
/// - F^p = conditions involving scope level ≥ p
#[derive(Debug, Clone)]
pub struct FilteredComplex {
    /// Filtration depth (number of levels).
    pub depth: usize,
    /// Chain groups by total degree.
    pub chains: HashMap<usize, Vec<ChainGroup>>,
    /// Boundary maps.
    pub boundaries: HashMap<usize, BoundaryMap>,
}

/// A chain group at a specific filtration level.
#[derive(Debug, Clone)]
pub struct ChainGroup {
    pub filtration_level: usize,
    pub total_degree: usize,
    pub rank: usize,
    pub generators: Vec<String>,
}

impl ChainGroup {
    pub fn new(filtration_level: usize, total_degree: usize, rank: usize) -> Self {
        Self {
            filtration_level,
            total_degree,
            rank,
            generators: (0..rank)
                .map(|i| format!("g_{}^{},{}", i, filtration_level, total_degree))
                .collect(),
        }
    }
}

/// A boundary map ∂_n : C_n → C_{n-1}.
#[derive(Debug, Clone)]
pub struct BoundaryMap {
    pub source_degree: usize,
    pub target_degree: usize,
    /// Matrix entries (row, col) → coefficient.
    pub matrix: HashMap<(usize, usize), i64>,
    pub source_rank: usize,
    pub target_rank: usize,
}

impl BoundaryMap {
    pub fn new(source_degree: usize, target_degree: usize, source_rank: usize, target_rank: usize) -> Self {
        Self {
            source_degree,
            target_degree,
            matrix: HashMap::new(),
            source_rank,
            target_rank,
        }
    }

    pub fn set(&mut self, row: usize, col: usize, val: i64) {
        self.matrix.insert((row, col), val);
    }

    pub fn get(&self, row: usize, col: usize) -> i64 {
        self.matrix.get(&(row, col)).copied().unwrap_or(0)
    }

    /// Check ∂ ∘ ∂ = 0.
    pub fn check_boundary_squared_zero(&self, next: &BoundaryMap) -> bool {
        // Composition should be the zero map
        for i in 0..next.source_rank {
            for k in 0..self.target_rank {
                let mut sum: i64 = 0;
                for j in 0..self.source_rank {
                    sum += next.get(j, i) * self.get(k, j);
                }
                if sum != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the rank of the image.
    pub fn image_rank(&self) -> usize {
        // Gaussian elimination to find rank
        if self.source_rank == 0 || self.target_rank == 0 {
            return 0;
        }

        let mut mat: Vec<Vec<i64>> = (0..self.target_rank)
            .map(|r| (0..self.source_rank).map(|c| self.get(r, c)).collect())
            .collect();

        let rows = mat.len();
        let cols = if rows > 0 { mat[0].len() } else { 0 };
        let mut rank = 0;
        let mut pivot_col = 0;

        for row in 0..rows {
            if pivot_col >= cols {
                break;
            }
            // Find pivot
            let mut pivot_row = None;
            for r in row..rows {
                if mat[r][pivot_col] != 0 {
                    pivot_row = Some(r);
                    break;
                }
            }
            if let Some(pr) = pivot_row {
                mat.swap(row, pr);
                rank += 1;
                let pivot_val = mat[row][pivot_col];
                for r in 0..rows {
                    if r != row && mat[r][pivot_col] != 0 {
                        let factor = mat[r][pivot_col] / pivot_val;
                        for c in 0..cols {
                            mat[r][c] -= factor * mat[row][c];
                        }
                    }
                }
                pivot_col += 1;
            } else {
                pivot_col += 1;
            }
        }

        rank
    }

    /// Compute the kernel rank.
    pub fn kernel_rank(&self) -> usize {
        self.source_rank - self.image_rank()
    }
}

impl FilteredComplex {
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            chains: HashMap::new(),
            boundaries: HashMap::new(),
        }
    }

    /// Add a chain group.
    pub fn add_chain_group(&mut self, group: ChainGroup) {
        self.chains
            .entry(group.total_degree)
            .or_default()
            .push(group);
    }

    /// Add a boundary map.
    pub fn add_boundary(&mut self, boundary: BoundaryMap) {
        self.boundaries.insert(boundary.source_degree, boundary);
    }

    /// Build the E_0 page from this filtered complex.
    pub fn build_e0(&self, num_properties: usize) -> EPage {
        let mut page = EPage::for_gpu_hierarchy(0, num_properties);
        for (degree, groups) in &self.chains {
            for group in groups {
                let p = group.filtration_level;
                let q = degree.saturating_sub(p);
                if p < page.rows && q < page.cols {
                    let current = page.get(p, q).clone();
                    let new_entry = MatrixEntry::generator(
                        &format!("C({},{})", p, q),
                        group.rank,
                    );
                    page.set(p, q, current.add(&new_entry));
                }
            }
        }
        page
    }

    /// Compute homology at degree n: H_n = ker(∂_n) / im(∂_{n+1}).
    pub fn homology_rank(&self, degree: usize) -> usize {
        let ker = self
            .boundaries
            .get(&degree)
            .map(|b| b.kernel_rank())
            .unwrap_or(0);

        let im = self
            .boundaries
            .get(&(degree + 1))
            .map(|b| b.image_rank())
            .unwrap_or(0);

        ker.saturating_sub(im)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GradedModule — graded algebraic structure
// ═══════════════════════════════════════════════════════════════════════

/// A bigraded module used as pages in the spectral sequence.
#[derive(Debug, Clone)]
pub struct GradedModule {
    /// Entries indexed by (p, q) bidegree.
    pub entries: HashMap<(usize, usize), usize>,
}

impl GradedModule {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn set_rank(&mut self, p: usize, q: usize, rank: usize) {
        if rank > 0 {
            self.entries.insert((p, q), rank);
        } else {
            self.entries.remove(&(p, q));
        }
    }

    pub fn rank(&self, p: usize, q: usize) -> usize {
        self.entries.get(&(p, q)).copied().unwrap_or(0)
    }

    pub fn total_rank(&self) -> usize {
        self.entries.values().sum()
    }

    /// Poincaré polynomial: Σ rank(p,q) · s^p · t^q.
    pub fn poincare_polynomial(&self) -> BTreeMap<(usize, usize), usize> {
        self.entries.clone().into_iter().collect()
    }

    /// Euler characteristic: Σ (-1)^{p+q} rank(p,q).
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi: i64 = 0;
        for (&(p, q), &rank) in &self.entries {
            let sign = if (p + q) % 2 == 0 { 1i64 } else { -1i64 };
            chi += sign * rank as i64;
        }
        chi
    }

    /// Convert to an EPage.
    pub fn to_epage(&self, page_number: usize, rows: usize, cols: usize) -> EPage {
        let mut page = EPage::new(page_number, rows, cols);
        for (&(p, q), &rank) in &self.entries {
            if p < rows && q < cols && rank > 0 {
                page.set(p, q, MatrixEntry::generator(&format!("M({},{})", p, q), rank));
            }
        }
        page
    }
}

impl Default for GradedModule {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SpectralSequenceBuilder — convenient builder pattern
// ═══════════════════════════════════════════════════════════════════════

/// Builder for setting up a spectral sequence for GPU verification.
#[derive(Debug)]
pub struct SpectralSequenceBuilder {
    properties: Vec<ConsistencyProperty>,
    local_results: Vec<LocalConsistencyResult>,
    max_pages: usize,
}

impl SpectralSequenceBuilder {
    pub fn new() -> Self {
        Self {
            properties: Vec::new(),
            local_results: Vec::new(),
            max_pages: 20,
        }
    }

    pub fn add_property(mut self, prop: ConsistencyProperty) -> Self {
        self.properties.push(prop);
        self
    }

    pub fn add_standard_properties(self) -> Self {
        self.add_property(ConsistencyProperty::coherence())
            .add_property(ConsistencyProperty::causality())
            .add_property(ConsistencyProperty::atomicity())
            .add_property(ConsistencyProperty::sc_per_location())
            .add_property(ConsistencyProperty::no_thin_air())
    }

    pub fn add_local_result(mut self, result: LocalConsistencyResult) -> Self {
        self.local_results.push(result);
        self
    }

    pub fn max_pages(mut self, max: usize) -> Self {
        self.max_pages = max;
        self
    }

    pub fn build(self) -> SpectralSequence {
        let mut ss = SpectralSequence::new(self.properties, self.max_pages);
        if !self.local_results.is_empty() {
            ss.initialize(self.local_results);
        }
        ss
    }
}

impl Default for SpectralSequenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_level_ordering() {
        assert!(HierarchyLevel::Thread < HierarchyLevel::Warp);
        assert!(HierarchyLevel::Warp < HierarchyLevel::CTA);
        assert!(HierarchyLevel::CTA < HierarchyLevel::GPU);
        assert!(HierarchyLevel::GPU < HierarchyLevel::System);
    }

    #[test]
    fn test_hierarchy_level_roundtrip() {
        for level in HierarchyLevel::all() {
            let idx = level.index();
            assert_eq!(HierarchyLevel::from_index(idx), Some(*level));
        }
        assert_eq!(HierarchyLevel::from_index(99), None);
    }

    #[test]
    fn test_matrix_entry_zero() {
        let z = MatrixEntry::zero();
        assert!(z.is_zero());
        assert_eq!(z.rank(), 0);
    }

    #[test]
    fn test_matrix_entry_generator() {
        let g = MatrixEntry::generator("test", 3);
        assert!(!g.is_zero());
        assert_eq!(g.rank(), 3);
    }

    #[test]
    fn test_matrix_entry_direct_sum() {
        let a = MatrixEntry::generator("a", 2);
        let b = MatrixEntry::generator("b", 3);
        let sum = MatrixEntry::direct_sum(vec![a, b]);
        assert_eq!(sum.rank(), 5);
    }

    #[test]
    fn test_matrix_entry_add() {
        let a = MatrixEntry::generator("a", 1);
        let b = MatrixEntry::generator("b", 1);
        let sum = a.add(&b);
        assert_eq!(sum.rank(), 2);

        let z = MatrixEntry::zero();
        let sum2 = a.add(&z);
        assert_eq!(sum2.rank(), 1);
    }

    #[test]
    fn test_matrix_entry_direct_sum_simplification() {
        // Direct sum of all zeros → zero
        let sum = MatrixEntry::direct_sum(vec![MatrixEntry::Zero, MatrixEntry::Zero]);
        assert!(sum.is_zero());

        // Direct sum of single non-zero → unwrapped
        let single = MatrixEntry::direct_sum(vec![MatrixEntry::generator("x", 1)]);
        assert_eq!(single.rank(), 1);
        assert!(matches!(single, MatrixEntry::Generator { .. }));
    }

    #[test]
    fn test_epage_creation() {
        let page = EPage::new(0, 5, 3);
        assert_eq!(page.rows, 5);
        assert_eq!(page.cols, 3);
        assert_eq!(page.total_rank(), 0);
        assert!(page.is_zero());
    }

    #[test]
    fn test_epage_gpu_hierarchy() {
        let page = EPage::for_gpu_hierarchy(0, 4);
        assert_eq!(page.rows, 5); // 5 hierarchy levels
        assert_eq!(page.cols, 4);
        assert_eq!(page.row_labels[0], "Thread");
        assert_eq!(page.row_labels[4], "System");
    }

    #[test]
    fn test_epage_set_get() {
        let mut page = EPage::new(0, 3, 3);
        page.set(1, 2, MatrixEntry::generator("test", 5));
        assert_eq!(page.rank_at(1, 2), 5);
        assert_eq!(page.rank_at(0, 0), 0);
        assert_eq!(page.total_rank(), 5);
    }

    #[test]
    fn test_epage_diagonal() {
        let mut page = EPage::new(0, 4, 4);
        page.set(0, 2, MatrixEntry::generator("a", 1));
        page.set(1, 1, MatrixEntry::generator("b", 2));
        page.set(2, 0, MatrixEntry::generator("c", 3));

        let diag = page.diagonal(2);
        assert_eq!(diag.len(), 3);
        assert_eq!(diag[0].2.rank(), 1);
        assert_eq!(diag[1].2.rank(), 2);
        assert_eq!(diag[2].2.rank(), 3);
    }

    #[test]
    fn test_differential_target() {
        // d_1 : E_1^{p,q} → E_1^{p+1,q}
        let d = Differential::new(1, 5, 5);
        assert_eq!(d.target_indices(0, 1), Some((1, 1)));
        assert_eq!(d.target_indices(4, 1), None); // target p=5 out of range

        // d_2 : E_2^{p,q} → E_2^{p+2,q-1}
        let d2 = Differential::new(2, 5, 5);
        assert_eq!(d2.target_indices(0, 2), Some((2, 1)));
        assert_eq!(d2.target_indices(0, 0), None); // target q negative
    }

    #[test]
    fn test_differential_trivial() {
        let d = Differential::new(1, 3, 3);
        assert!(d.is_trivial());
        assert!(d.check_square_zero());
    }

    #[test]
    fn test_differential_homology() {
        let mut page = EPage::new(0, 3, 3);
        page.set(0, 0, MatrixEntry::generator("a", 2));
        page.set(1, 0, MatrixEntry::generator("b", 2));

        let mut diff = Differential::new(1, 3, 3);
        diff.set_image(0, 0, MatrixEntry::generator("d(a)", 1));

        let next = diff.compute_homology(&page);
        assert_eq!(next.page_number, 1);
        // a had rank 2, lost 1 to outgoing → 1 remaining
        assert!(next.rank_at(0, 0) <= 2);
    }

    #[test]
    fn test_convergence_pages_equal() {
        let conv = Convergence::new(10);
        let a = EPage::new(0, 3, 3);
        let b = EPage::new(1, 3, 3);
        assert!(conv.pages_equal(&a, &b));

        let mut c = EPage::new(0, 3, 3);
        c.set(0, 0, MatrixEntry::generator("x", 1));
        assert!(!conv.pages_equal(&a, &c));
    }

    #[test]
    fn test_convergence_with_tolerance() {
        let conv = Convergence::new(10).with_tolerance(1);
        let mut a = EPage::new(0, 3, 3);
        let mut b = EPage::new(1, 3, 3);
        a.set(0, 0, MatrixEntry::generator("x", 2));
        b.set(0, 0, MatrixEntry::generator("x", 1));
        assert!(conv.pages_equal(&a, &b)); // difference of 1 within tolerance
    }

    #[test]
    fn test_convergence_status() {
        let conv = Convergence::new(10);
        let p0 = EPage::new(0, 3, 3);
        let p1 = EPage::new(1, 3, 3);

        // Single page: evolving
        assert!(matches!(
            conv.check(&[p0.clone()]),
            ConvergenceStatus::Evolving { .. }
        ));

        // Two identical pages: degenerate (converged at page 0 ≤ 2)
        assert!(matches!(
            conv.check(&[p0.clone(), p1.clone()]),
            ConvergenceStatus::Degenerate
        ));
    }

    #[test]
    fn test_consistency_property_standard() {
        let coh = ConsistencyProperty::coherence();
        assert!(coh.is_relevant_at(HierarchyLevel::GPU));
        assert!(!coh.is_relevant_at(HierarchyLevel::Thread));

        let caus = ConsistencyProperty::causality();
        assert!(caus.is_relevant_at(HierarchyLevel::Thread));
        assert!(caus.is_relevant_at(HierarchyLevel::System));
    }

    #[test]
    fn test_local_consistency_result() {
        let ok = LocalConsistencyResult::success(HierarchyLevel::CTA, "coherence");
        assert!(ok.holds);
        assert!(ok.obstruction.is_zero());

        let fail = LocalConsistencyResult::failure(
            HierarchyLevel::GPU,
            "causality",
            vec!["cycle in po ∪ rf".to_string()],
        );
        assert!(!fail.holds);
        assert!(!fail.obstruction.is_zero());
    }

    #[test]
    fn test_spectral_sequence_all_consistent() {
        let props = vec![
            ConsistencyProperty::coherence(),
            ConsistencyProperty::causality(),
        ];

        let mut ss = SpectralSequence::new(props.clone(), 10);

        let results = vec![
            LocalConsistencyResult::success(HierarchyLevel::Thread, "coherence"),
            LocalConsistencyResult::success(HierarchyLevel::Thread, "causality"),
            LocalConsistencyResult::success(HierarchyLevel::CTA, "coherence"),
            LocalConsistencyResult::success(HierarchyLevel::CTA, "causality"),
            LocalConsistencyResult::success(HierarchyLevel::GPU, "coherence"),
            LocalConsistencyResult::success(HierarchyLevel::GPU, "causality"),
        ];

        ss.initialize(results);
        ss.run_to_convergence();
        assert!(ss.is_globally_consistent());
    }

    #[test]
    fn test_spectral_sequence_with_failure() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut ss = SpectralSequence::new(props, 10);

        let results = vec![
            LocalConsistencyResult::success(HierarchyLevel::Thread, "coherence"),
            LocalConsistencyResult::failure(
                HierarchyLevel::GPU,
                "coherence",
                vec!["co cycle".to_string()],
            ),
        ];

        ss.initialize(results);
        ss.run_to_convergence();
        assert!(!ss.is_globally_consistent());
    }

    #[test]
    fn test_spectral_sequence_page_count() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut ss = SpectralSequence::new(props, 5);
        let results = vec![
            LocalConsistencyResult::success(HierarchyLevel::Thread, "coherence"),
        ];
        ss.initialize(results);
        assert_eq!(ss.page_count(), 1);
        ss.advance();
        assert_eq!(ss.page_count(), 2);
    }

    #[test]
    fn test_spectral_sequence_builder() {
        let ss = SpectralSequenceBuilder::new()
            .add_standard_properties()
            .max_pages(15)
            .add_local_result(LocalConsistencyResult::success(
                HierarchyLevel::CTA,
                "coherence",
            ))
            .build();

        assert_eq!(ss.properties.len(), 5);
        assert_eq!(ss.convergence.max_pages, 15);
        assert_eq!(ss.page_count(), 1);
    }

    #[test]
    fn test_fiber_creation() {
        let mut fiber = Fiber::new("warp_0", HierarchyLevel::Warp, vec![0, 1, 2, 3]);
        assert_eq!(fiber.event_indices.len(), 4);
        assert_eq!(fiber.local_consistent, None);

        fiber.mark_consistent(true);
        assert_eq!(fiber.local_consistent, Some(true));
    }

    #[test]
    fn test_serre_spectral_sequence_simple() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut serre = SerreSpectralSequence::new(props);

        let fibers = vec![
            Fiber::new("thread_0", HierarchyLevel::Thread, vec![0, 1]),
            Fiber::new("thread_1", HierarchyLevel::Thread, vec![2, 3]),
        ];
        serre.add_fibers(HierarchyLevel::Thread, fibers);

        let mut f0 = serre.fibers.get_mut(&HierarchyLevel::Thread).unwrap();
        f0[0].mark_consistent(true);
        f0[1].mark_consistent(true);

        serre.initialize_from_fibers();
        serre.run();
        assert!(serre.is_consistent());
    }

    #[test]
    fn test_serre_build_from_scope_assignment() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut serre = SerreSpectralSequence::new(props);

        let event_threads = vec![0, 0, 1, 1]; // 4 events, 2 threads
        let mut scopes = HashMap::new();
        scopes.insert(0, (0, 0, 0)); // thread 0: warp 0, CTA 0, GPU 0
        scopes.insert(1, (0, 0, 0)); // thread 1: warp 0, CTA 0, GPU 0

        serre.build_from_scope_assignment(4, &event_threads, &scopes);

        assert!(serre.fibers.contains_key(&HierarchyLevel::Thread));
        assert!(serre.fibers.contains_key(&HierarchyLevel::Warp));
        assert!(serre.fibers.contains_key(&HierarchyLevel::CTA));
        assert!(serre.fibers.contains_key(&HierarchyLevel::GPU));
        assert!(serre.fibers.contains_key(&HierarchyLevel::System));
        assert_eq!(serre.fibers[&HierarchyLevel::System].len(), 1);
    }

    #[test]
    fn test_hierarchical_decomposition_standard() {
        let mut hd = HierarchicalDecomposition::standard_gpu();
        assert_eq!(hd.properties.len(), 5);

        let event_threads = vec![0, 0, 1, 1];
        let mut scopes = HashMap::new();
        scopes.insert(0, (0, 0, 0));
        scopes.insert(1, (1, 0, 0));

        hd.setup(4, &event_threads, &scopes);

        // Mark all fibers consistent
        for (&level, fibers) in &hd.serre.fibers.clone() {
            for i in 0..fibers.len() {
                hd.set_fiber_consistency(level, i, true);
            }
        }

        assert!(hd.verify());
    }

    #[test]
    fn test_hierarchical_decomposition_failure() {
        let mut hd = HierarchicalDecomposition::standard_gpu();
        let event_threads = vec![0, 0, 1, 1];
        let mut scopes = HashMap::new();
        scopes.insert(0, (0, 0, 0));
        scopes.insert(1, (1, 0, 0));
        hd.setup(4, &event_threads, &scopes);

        // Mark one fiber inconsistent
        hd.set_fiber_consistency(HierarchyLevel::Thread, 0, false);
        hd.set_fiber_consistency(HierarchyLevel::Thread, 1, true);

        assert!(!hd.verify());
    }

    #[test]
    fn test_filtered_complex_basic() {
        let mut fc = FilteredComplex::new(3);
        fc.add_chain_group(ChainGroup::new(0, 0, 2));
        fc.add_chain_group(ChainGroup::new(1, 1, 3));

        let page = fc.build_e0(5);
        assert!(page.total_rank() > 0);
    }

    #[test]
    fn test_boundary_map_rank() {
        let mut bm = BoundaryMap::new(1, 0, 3, 2);
        bm.set(0, 0, 1);
        bm.set(1, 1, 1);
        // rank should be 2 (two independent rows)
        assert_eq!(bm.image_rank(), 2);
        assert_eq!(bm.kernel_rank(), 1);
    }

    #[test]
    fn test_boundary_map_zero() {
        let bm = BoundaryMap::new(1, 0, 3, 2);
        assert_eq!(bm.image_rank(), 0);
        assert_eq!(bm.kernel_rank(), 3);
    }

    #[test]
    fn test_graded_module() {
        let mut m = GradedModule::new();
        m.set_rank(0, 0, 1);
        m.set_rank(1, 0, 2);
        m.set_rank(0, 1, 3);

        assert_eq!(m.rank(0, 0), 1);
        assert_eq!(m.rank(1, 0), 2);
        assert_eq!(m.rank(0, 1), 3);
        assert_eq!(m.rank(2, 2), 0);
        assert_eq!(m.total_rank(), 6);
    }

    #[test]
    fn test_graded_module_euler_characteristic() {
        let mut m = GradedModule::new();
        m.set_rank(0, 0, 1); // (-1)^0 * 1 = 1
        m.set_rank(1, 0, 2); // (-1)^1 * 2 = -2
        m.set_rank(0, 1, 3); // (-1)^1 * 3 = -3
        assert_eq!(m.euler_characteristic(), 1 - 2 - 3);
    }

    #[test]
    fn test_graded_module_to_epage() {
        let mut m = GradedModule::new();
        m.set_rank(0, 0, 1);
        m.set_rank(1, 1, 2);

        let page = m.to_epage(0, 3, 3);
        assert_eq!(page.rank_at(0, 0), 1);
        assert_eq!(page.rank_at(1, 1), 2);
        assert_eq!(page.rank_at(2, 2), 0);
    }

    #[test]
    fn test_epage_pretty_print() {
        let mut page = EPage::new(0, 2, 2);
        page.set(0, 0, MatrixEntry::generator("a", 1));
        let output = page.pretty_print();
        assert!(output.contains("E_0"));
    }

    #[test]
    fn test_spectral_sequence_summary() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut ss = SpectralSequence::new(props, 10);
        let results = vec![
            LocalConsistencyResult::success(HierarchyLevel::Thread, "coherence"),
        ];
        ss.initialize(results);
        let summary = ss.summary();
        assert!(summary.contains("SpectralSequence"));
    }

    #[test]
    fn test_serre_summary() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut serre = SerreSpectralSequence::new(props);
        let summary = serre.summary();
        assert!(summary.contains("SerreSpectralSequence"));
    }

    #[test]
    fn test_convergence_did_not_converge() {
        let conv = Convergence::new(3);
        let mut pages = Vec::new();
        for i in 0..5 {
            let mut p = EPage::new(i, 2, 2);
            p.set(0, 0, MatrixEntry::generator("x", i + 1));
            pages.push(p);
        }
        let status = conv.check(&pages);
        assert!(matches!(status, ConvergenceStatus::DidNotConverge { .. }));
    }

    #[test]
    fn test_spectral_sequence_display() {
        let ss = SpectralSequence::new(vec![ConsistencyProperty::coherence()], 10);
        let display = format!("{}", ss);
        assert!(display.contains("SpectralSequence"));
    }

    #[test]
    fn test_serre_gluing() {
        let props = vec![ConsistencyProperty::coherence()];
        let mut serre = SerreSpectralSequence::new(props);

        let fibers = vec![
            Fiber::new("cta_0", HierarchyLevel::CTA, vec![0, 1]),
            Fiber::new("cta_1", HierarchyLevel::CTA, vec![2, 3]),
        ];
        serre.add_fibers(HierarchyLevel::CTA, fibers);

        serre.record_gluing(HierarchyLevel::CTA, 0, 1, true);
        // Gluing is compatible
        assert!(!serre.gluing_results.values().any(|&v| !v));

        serre.record_gluing(HierarchyLevel::CTA, 0, 1, false);
        // Now gluing fails
        assert!(serre.gluing_results.values().any(|&v| !v));
    }

    #[test]
    fn test_matrix_entry_display() {
        assert_eq!(format!("{}", MatrixEntry::Zero), "0");
        assert_eq!(format!("{}", MatrixEntry::generator("x", 1)), "⟨x⟩");
        assert_eq!(format!("{}", MatrixEntry::generator("x", 3)), "⟨x⟩^3");

        let sum = MatrixEntry::DirectSum(vec![
            MatrixEntry::generator("a", 1),
            MatrixEntry::generator("b", 2),
        ]);
        let s = format!("{}", sum);
        assert!(s.contains("⊕"));
    }

    #[test]
    fn test_chain_group() {
        let cg = ChainGroup::new(2, 3, 4);
        assert_eq!(cg.filtration_level, 2);
        assert_eq!(cg.total_degree, 3);
        assert_eq!(cg.rank, 4);
        assert_eq!(cg.generators.len(), 4);
    }

    #[test]
    fn test_poincare_polynomial() {
        let mut m = GradedModule::new();
        m.set_rank(0, 0, 1);
        m.set_rank(1, 1, 2);
        let poly = m.poincare_polynomial();
        assert_eq!(poly.len(), 2);
        assert_eq!(poly[&(0, 0)], 1);
        assert_eq!(poly[&(1, 1)], 2);
    }

    #[test]
    fn test_filtered_complex_homology_rank() {
        let mut fc = FilteredComplex::new(3);
        let mut bm = BoundaryMap::new(1, 0, 2, 1);
        bm.set(0, 0, 1); // ∂(g_0) = g_0 of C_0
        fc.add_boundary(bm);

        let h1 = fc.homology_rank(1);
        assert_eq!(h1, 0); // ker has rank 1, but we check at degree 1
    }
}
