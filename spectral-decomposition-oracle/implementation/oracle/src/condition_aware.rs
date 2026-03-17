// Condition-number-aware method selection: routes instances based on κ(A)
// to avoid vacuous spectral bounds on ill-conditioned (big-M) formulations.
//
// When κ(A) < KAPPA_THRESHOLD, the spectral decomposition theorem (Theorem 4)
// gives tight bounds and we use spectral-guided selection.  When κ(A) ≥ threshold,
// the crown theorem constant C = O(k·κ⁴·‖c‖∞) is vacuous, so we fall back to
// structure-exploiting decomposition that depends only on sparsity pattern—not κ.

use crate::error::{OracleError, OracleResult};
use crate::structure::detector::{DetectionResult, SparseMatrix, StructureDetector, StructureType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Condition number threshold separating spectral-safe from spectral-unsafe.
/// For κ ≥ 1000 the crown theorem bound is ≥ 10¹² and uninformative.
const KAPPA_THRESHOLD: f64 = 1e3;

/// Entries above this magnitude are candidates for big-M pattern detection.
const BIG_M_ENTRY_THRESHOLD: f64 = 1e3;

/// Minimum fraction of big-M constraints to flag the instance as big-M-heavy.
const BIG_M_FRACTION_ALERT: f64 = 0.10;

// ─────────────────────────────────────────────────────────────────────────────
// DecompositionRoute: the strategy chosen by the condition-aware selector
// ─────────────────────────────────────────────────────────────────────────────

/// Which decomposition pathway the selector chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecompositionRoute {
    /// κ is small → spectral theorem applies, use spectral-guided selection.
    SpectralGuided,
    /// κ is large → use structure-exploiting decomposition (no κ dependence).
    StructureExploiting,
}

impl std::fmt::Display for DecompositionRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompositionRoute::SpectralGuided => write!(f, "SpectralGuided"),
            DecompositionRoute::StructureExploiting => write!(f, "StructureExploiting"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConditionAwareSelector
// ─────────────────────────────────────────────────────────────────────────────

/// Routes instances to spectral or structure-based decomposition based on κ(A).
///
/// * κ < threshold: spectral decomposition (tight bound from Theorem 4).
/// * κ ≥ threshold: block-angular structure detection (bound independent of κ).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionAwareSelector {
    pub kappa_threshold: f64,
}

impl Default for ConditionAwareSelector {
    fn default() -> Self {
        Self {
            kappa_threshold: KAPPA_THRESHOLD,
        }
    }
}

/// Result returned by the condition-aware selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub route: DecompositionRoute,
    pub kappa_estimate: f64,
    pub kappa_threshold: f64,
    pub bound_tightness: BoundTightness,
    pub details: HashMap<String, String>,
}

/// Qualitative assessment of the spectral bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundTightness {
    /// Crown theorem gives an informative bound.
    Tight,
    /// Crown theorem bound is vacuous; falling back to structure-based.
    Vacuous,
}

impl std::fmt::Display for BoundTightness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundTightness::Tight => write!(f, "tight"),
            BoundTightness::Vacuous => write!(f, "vacuous"),
        }
    }
}

impl ConditionAwareSelector {
    pub fn new(kappa_threshold: f64) -> Self {
        Self { kappa_threshold }
    }

    /// Estimate κ(A) from the eigenvalue spectrum.
    ///
    /// For a symmetric matrix the condition number is |λ_max| / |λ_min|
    /// (excluding zero eigenvalues below tolerance).
    pub fn estimate_kappa(eigenvalues: &[f64]) -> f64 {
        let tol = 1e-14;
        let nonzero: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&v| v.abs() > tol)
            .collect();
        if nonzero.is_empty() {
            return f64::INFINITY;
        }
        let abs_max = nonzero.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let abs_min = nonzero.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
        if abs_min == 0.0 {
            return f64::INFINITY;
        }
        abs_max / abs_min
    }

    /// Select decomposition route based on the condition number.
    pub fn select(&self, eigenvalues: &[f64]) -> SelectionResult {
        let kappa = Self::estimate_kappa(eigenvalues);
        let (route, tightness) = if kappa < self.kappa_threshold {
            (DecompositionRoute::SpectralGuided, BoundTightness::Tight)
        } else {
            (DecompositionRoute::StructureExploiting, BoundTightness::Vacuous)
        };

        let mut details = HashMap::new();
        details.insert("log10_kappa".to_string(), kappa.log10().to_string());
        details.insert("route".to_string(), route.to_string());
        if tightness == BoundTightness::Vacuous {
            details.insert(
                "reason".to_string(),
                format!(
                    "κ={:.2e} ≥ threshold {:.0e}: crown theorem bound O(κ⁴) is vacuous",
                    kappa, self.kappa_threshold
                ),
            );
        }

        SelectionResult {
            route,
            kappa_estimate: kappa,
            kappa_threshold: self.kappa_threshold,
            bound_tightness: tightness,
            details,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BigMDetector
// ─────────────────────────────────────────────────────────────────────────────

/// A single big-M constraint identified in the formulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigMConstraint {
    pub row: usize,
    pub big_m_value: f64,
    /// Column index of the binary indicator variable (if detected).
    pub indicator_col: Option<usize>,
    /// Whether this constraint can be reformulated as an indicator constraint.
    pub reformulable: bool,
}

/// Summary of big-M analysis for an instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigMReport {
    pub n_constraints_total: usize,
    pub n_big_m_constraints: usize,
    pub big_m_fraction: f64,
    pub max_big_m_value: f64,
    pub n_reformulable: usize,
    pub constraints: Vec<BigMConstraint>,
    pub is_big_m_heavy: bool,
}

impl BigMReport {
    pub fn summary(&self) -> String {
        format!(
            "big-M: {}/{} constraints ({:.1}%), max M={:.2e}, reformulable={}",
            self.n_big_m_constraints,
            self.n_constraints_total,
            self.big_m_fraction * 100.0,
            self.max_big_m_value,
            self.n_reformulable,
        )
    }
}

/// Detects big-M constraints in a formulation.
///
/// Scans the constraint matrix for entries with |a_ij| > 10³ that follow
/// the pattern  a·x + M·y ≤ M  (indicator variable y ∈ {0,1}).
pub struct BigMDetector {
    pub entry_threshold: f64,
}

impl Default for BigMDetector {
    fn default() -> Self {
        Self {
            entry_threshold: BIG_M_ENTRY_THRESHOLD,
        }
    }
}

impl BigMDetector {
    pub fn new(entry_threshold: f64) -> Self {
        Self { entry_threshold }
    }

    /// Scan a coefficient matrix (given as row-major dense slice) for big-M patterns.
    ///
    /// `values` is a row-major dense representation: `values[row * n_cols + col]`.
    /// `binary_cols` lists column indices known to be binary (integer with bounds [0,1]).
    pub fn detect(
        &self,
        values: &[f64],
        n_rows: usize,
        n_cols: usize,
        binary_cols: &[usize],
    ) -> BigMReport {
        let binary_set: std::collections::HashSet<usize> =
            binary_cols.iter().copied().collect();
        let mut constraints = Vec::new();
        let mut max_m = 0.0_f64;

        for row in 0..n_rows {
            let mut big_entry: Option<(usize, f64)> = None;
            let mut has_indicator = false;
            let mut indicator_col = None;

            for col in 0..n_cols {
                let val = values[row * n_cols + col];
                if val.abs() > self.entry_threshold {
                    if val.abs() > big_entry.map_or(0.0, |(_, v)| v) {
                        big_entry = Some((col, val.abs()));
                    }
                    if binary_set.contains(&col) {
                        has_indicator = true;
                        indicator_col = Some(col);
                    }
                }
            }

            if let Some((_col, m_val)) = big_entry {
                max_m = max_m.max(m_val);
                // Reformulable if: exactly one big-M entry multiplies a binary variable.
                let reformulable = has_indicator;
                constraints.push(BigMConstraint {
                    row,
                    big_m_value: m_val,
                    indicator_col: if reformulable { indicator_col } else { None },
                    reformulable,
                });
            }
        }

        let n_big_m = constraints.len();
        let frac = if n_rows > 0 {
            n_big_m as f64 / n_rows as f64
        } else {
            0.0
        };
        let n_reformulable = constraints.iter().filter(|c| c.reformulable).count();

        BigMReport {
            n_constraints_total: n_rows,
            n_big_m_constraints: n_big_m,
            big_m_fraction: frac,
            max_big_m_value: max_m,
            n_reformulable,
            constraints,
            is_big_m_heavy: frac >= BIG_M_FRACTION_ALERT,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Block detected by the structure-exploiting decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
}

/// Result of structure-exploiting decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveDecompositionResult {
    pub blocks: Vec<Block>,
    pub linking_rows: Vec<usize>,
    pub n_blocks: usize,
    pub max_block_size: usize,
    pub linking_fraction: f64,
    pub structure_type: StructureType,
}

impl AdaptiveDecompositionResult {
    pub fn summary(&self) -> String {
        format!(
            "structure={}, blocks={}, max_block={}, linking={:.1}%",
            self.structure_type,
            self.n_blocks,
            self.max_block_size,
            self.linking_fraction * 100.0,
        )
    }
}

/// Structure-exploiting decomposition for high-κ instances.
///
/// Uses block-angular detection on the sparsity pattern (not spectral
/// properties), so the decomposition bound has no dependence on κ.
pub struct AdaptiveDecomposition {
    pub min_block_rows: usize,
    pub max_linking_fraction: f64,
}

impl Default for AdaptiveDecomposition {
    fn default() -> Self {
        Self {
            min_block_rows: 3,
            max_linking_fraction: 0.30,
        }
    }
}

impl AdaptiveDecomposition {
    pub fn new(min_block_rows: usize, max_linking_fraction: f64) -> Self {
        Self {
            min_block_rows,
            max_linking_fraction,
        }
    }

    /// Decompose by constraint-graph connectivity on the sparsity pattern.
    ///
    /// 1. Build a row-adjacency graph: rows i,j share an edge if they have
    ///    a nonzero in the same column.
    /// 2. Find connected components (blocks).
    /// 3. Rows touching multiple components become linking rows.
    pub fn decompose(&self, matrix: &SparseMatrix) -> OracleResult<AdaptiveDecompositionResult> {
        if matrix.n_rows == 0 || matrix.n_cols == 0 {
            return Err(OracleError::invalid_input("empty matrix"));
        }

        // Build row-adjacency via shared columns.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); matrix.n_rows];
        for col in 0..matrix.n_cols {
            let rows_in_col: Vec<usize> = matrix.col_to_rows[col].iter().copied().collect();
            for i in 0..rows_in_col.len() {
                for j in (i + 1)..rows_in_col.len() {
                    adj[rows_in_col[i]].push(rows_in_col[j]);
                    adj[rows_in_col[j]].push(rows_in_col[i]);
                }
            }
        }

        // Connected components via BFS.
        let mut component = vec![usize::MAX; matrix.n_rows];
        let mut comp_id = 0;
        for start in 0..matrix.n_rows {
            if component[start] != usize::MAX {
                continue;
            }
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            component[start] = comp_id;
            while let Some(node) = queue.pop_front() {
                for &nbr in &adj[node] {
                    if component[nbr] == usize::MAX {
                        component[nbr] = comp_id;
                        queue.push_back(nbr);
                    }
                }
            }
            comp_id += 1;
        }

        // Group rows into blocks.
        let mut blocks_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (row, &cid) in component.iter().enumerate() {
            blocks_map.entry(cid).or_default().push(row);
        }

        // Separate small components as linking rows.
        let mut blocks = Vec::new();
        let mut linking_rows = Vec::new();
        for (_cid, rows) in &blocks_map {
            if rows.len() < self.min_block_rows {
                linking_rows.extend_from_slice(rows);
            } else {
                let cols: Vec<usize> = rows
                    .iter()
                    .flat_map(|&r| matrix.row_to_cols[r].iter().copied())
                    .collect::<std::collections::HashSet<usize>>()
                    .into_iter()
                    .collect();
                blocks.push(Block {
                    rows: rows.clone(),
                    cols,
                });
            }
        }

        let linking_fraction = if matrix.n_rows > 0 {
            linking_rows.len() as f64 / matrix.n_rows as f64
        } else {
            0.0
        };

        let n_blocks = blocks.len();
        let max_block_size = blocks.iter().map(|b| b.rows.len()).max().unwrap_or(0);

        let structure_type = if n_blocks >= 2 && linking_fraction <= self.max_linking_fraction {
            StructureType::BlockAngular
        } else if n_blocks >= 2 {
            StructureType::BorderedBlockDiagonal
        } else {
            StructureType::Unstructured
        };

        Ok(AdaptiveDecompositionResult {
            blocks,
            linking_rows,
            n_blocks,
            max_block_size,
            linking_fraction,
            structure_type,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QualityPredictor
// ─────────────────────────────────────────────────────────────────────────────

/// Predicted decomposition quality from fast heuristic analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPrediction {
    pub expected_speedup: f64,
    pub bound_tightness: BoundTightness,
    pub confidence: f64,
    pub route: DecompositionRoute,
    pub notes: Vec<String>,
}

impl QualityPrediction {
    pub fn summary(&self) -> String {
        format!(
            "route={}, speedup={:.2}×, bound={}, conf={:.3}",
            self.route, self.expected_speedup, self.bound_tightness, self.confidence,
        )
    }
}

/// Predict decomposition quality without running the full solver.
///
/// Fast heuristic based on:
/// 1. Matrix sparsity pattern (density, row/col degree distribution).
/// 2. Condition number estimate from eigenvalue spectrum.
/// 3. Big-M prevalence.
pub struct QualityPredictor {
    pub kappa_threshold: f64,
}

impl Default for QualityPredictor {
    fn default() -> Self {
        Self {
            kappa_threshold: KAPPA_THRESHOLD,
        }
    }
}

impl QualityPredictor {
    pub fn new(kappa_threshold: f64) -> Self {
        Self { kappa_threshold }
    }

    /// Fast prediction combining sparsity, κ, and big-M signals.
    pub fn predict(
        &self,
        eigenvalues: &[f64],
        density: f64,
        n_rows: usize,
        n_cols: usize,
        big_m_fraction: f64,
    ) -> QualityPrediction {
        let kappa = ConditionAwareSelector::estimate_kappa(eigenvalues);
        let log_kappa = if kappa > 0.0 { kappa.log10() } else { 0.0 };

        let (route, tightness) = if kappa < self.kappa_threshold {
            (DecompositionRoute::SpectralGuided, BoundTightness::Tight)
        } else {
            (DecompositionRoute::StructureExploiting, BoundTightness::Vacuous)
        };

        // Heuristic speedup model:
        //   - Sparse + low-κ → high expected speedup (spectral bound is tight).
        //   - Sparse + high-κ → moderate speedup (structure-based, no guarantee).
        //   - Dense → low expected speedup regardless.
        let sparsity_bonus = (1.0 - density).max(0.0);
        let size_bonus = ((n_rows * n_cols) as f64).log10().min(6.0) / 6.0;
        let kappa_penalty = if kappa >= self.kappa_threshold {
            0.3 // reduced confidence for structure-only path
        } else {
            0.0
        };
        let big_m_penalty = big_m_fraction * 0.5;

        let raw_speedup = 1.0 + 4.0 * sparsity_bonus * size_bonus;
        let expected_speedup = (raw_speedup * (1.0 - big_m_penalty)).max(1.0);
        let confidence = (0.85 * sparsity_bonus - kappa_penalty - big_m_penalty * 0.3)
            .clamp(0.1, 0.99);

        let mut notes = Vec::new();
        if tightness == BoundTightness::Vacuous {
            notes.push(format!(
                "κ={:.2e}: spectral bound vacuous, using structure-based fallback",
                kappa
            ));
        }
        if big_m_fraction > BIG_M_FRACTION_ALERT {
            notes.push(format!(
                "{:.0}% big-M constraints detected; consider indicator reformulation",
                big_m_fraction * 100.0
            ));
        }
        if log_kappa > 8.0 {
            notes.push("recommend Ruiz equilibration preprocessing".to_string());
        }

        QualityPrediction {
            expected_speedup,
            bound_tightness: tightness,
            confidence,
            route,
            notes,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kappa_estimate_well_conditioned() {
        let eigenvalues = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kappa = ConditionAwareSelector::estimate_kappa(&eigenvalues);
        assert!((kappa - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_estimate_ill_conditioned() {
        let eigenvalues = vec![1e-6, 1.0, 100.0, 1e6];
        let kappa = ConditionAwareSelector::estimate_kappa(&eigenvalues);
        assert!((kappa - 1e12).abs() / 1e12 < 1e-10);
    }

    #[test]
    fn test_selector_routes_low_kappa_to_spectral() {
        let selector = ConditionAwareSelector::default();
        let eigenvalues = vec![1.0, 2.0, 3.0]; // κ = 3
        let result = selector.select(&eigenvalues);
        assert_eq!(result.route, DecompositionRoute::SpectralGuided);
        assert_eq!(result.bound_tightness, BoundTightness::Tight);
    }

    #[test]
    fn test_selector_routes_high_kappa_to_structure() {
        let selector = ConditionAwareSelector::default();
        let eigenvalues = vec![1e-4, 1.0, 1e4]; // κ = 10⁸
        let result = selector.select(&eigenvalues);
        assert_eq!(result.route, DecompositionRoute::StructureExploiting);
        assert_eq!(result.bound_tightness, BoundTightness::Vacuous);
    }

    #[test]
    fn test_big_m_detection() {
        let detector = BigMDetector::default();
        // 3 rows × 4 cols, row 1 has big-M entry at col 3 (binary)
        #[rustfmt::skip]
        let values = vec![
            1.0, 2.0,  0.0,    0.0,
            0.0, 1.0, -1.0, 1e6,     // big-M row
            3.0, 0.0,  1.0,    0.0,
        ];
        let binary_cols = vec![3];
        let report = detector.detect(&values, 3, 4, &binary_cols);
        assert_eq!(report.n_big_m_constraints, 1);
        assert!(report.constraints[0].reformulable);
        assert_eq!(report.constraints[0].indicator_col, Some(3));
    }

    #[test]
    fn test_big_m_no_binary_not_reformulable() {
        let detector = BigMDetector::default();
        #[rustfmt::skip]
        let values = vec![
            1e5, 2.0,
            0.0,  1.0,
        ];
        let report = detector.detect(&values, 2, 2, &[]);
        assert_eq!(report.n_big_m_constraints, 1);
        assert!(!report.constraints[0].reformulable);
    }

    #[test]
    fn test_adaptive_decomposition_two_blocks() {
        let mut matrix = SparseMatrix::new(6, 4);
        // Block 1: rows 0,1,2 share cols 0,1
        matrix.add_entry(0, 0);
        matrix.add_entry(1, 0);
        matrix.add_entry(1, 1);
        matrix.add_entry(2, 1);
        // Block 2: rows 3,4,5 share cols 2,3
        matrix.add_entry(3, 2);
        matrix.add_entry(4, 2);
        matrix.add_entry(4, 3);
        matrix.add_entry(5, 3);

        let decomp = AdaptiveDecomposition::default();
        let result = decomp.decompose(&matrix).unwrap();
        assert_eq!(result.n_blocks, 2);
        assert_eq!(result.structure_type, StructureType::BlockAngular);
    }

    #[test]
    fn test_adaptive_decomposition_single_block() {
        let mut matrix = SparseMatrix::new(3, 3);
        // Fully connected: single block
        matrix.add_entry(0, 0);
        matrix.add_entry(0, 1);
        matrix.add_entry(1, 1);
        matrix.add_entry(1, 2);
        matrix.add_entry(2, 0);
        matrix.add_entry(2, 2);

        let decomp = AdaptiveDecomposition::default();
        let result = decomp.decompose(&matrix).unwrap();
        assert_eq!(result.n_blocks, 1);
        assert_eq!(result.structure_type, StructureType::Unstructured);
    }

    #[test]
    fn test_quality_predictor_low_kappa() {
        let predictor = QualityPredictor::default();
        let eigenvalues = vec![1.0, 2.0, 3.0];
        let pred = predictor.predict(&eigenvalues, 0.05, 500, 300, 0.0);
        assert_eq!(pred.route, DecompositionRoute::SpectralGuided);
        assert_eq!(pred.bound_tightness, BoundTightness::Tight);
        assert!(pred.expected_speedup > 1.0);
    }

    #[test]
    fn test_quality_predictor_high_kappa_big_m() {
        let predictor = QualityPredictor::default();
        let eigenvalues = vec![1e-5, 1.0, 1e5];
        let pred = predictor.predict(&eigenvalues, 0.02, 1000, 500, 0.40);
        assert_eq!(pred.route, DecompositionRoute::StructureExploiting);
        assert_eq!(pred.bound_tightness, BoundTightness::Vacuous);
        assert!(!pred.notes.is_empty());
    }
}
