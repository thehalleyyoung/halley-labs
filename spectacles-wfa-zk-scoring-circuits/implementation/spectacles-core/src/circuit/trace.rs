// Execution trace module for the STARK prover.
//
// An execution trace is a two-dimensional matrix of field elements where each
// row represents one step of the computation and each column corresponds to a
// register / wire.  The prover commits to this trace via a Merkle tree and the
// verifier can query individual rows together with authentication paths.

use super::goldilocks::GoldilocksField;
use super::merkle::MerkleTree;

use super::goldilocks::{ntt, intt, evaluate_on_coset};
use super::merkle::{Digest, MerkleProof, blake3_hash};

use std::collections::HashSet;
use std::fmt;

// ═══════════════════════════════════════════════════════════════
// TraceError
// ═══════════════════════════════════════════════════════════════

/// Errors that can arise while constructing or manipulating a trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    /// Two dimensions that should agree do not.
    DimensionMismatch { expected: usize, got: usize },
    /// A row index is out of bounds.
    InvalidRow { row: usize, num_rows: usize },
    /// A column index is out of bounds.
    InvalidColumn { col: usize, num_cols: usize },
    /// The trace height is not a power of two where one is required.
    NotPowerOfTwo { height: usize },
    /// The trace is empty (zero rows or zero columns).
    EmptyTrace,
    /// Generic deserialization failure.
    DeserializationError(String),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            TraceError::InvalidRow { row, num_rows } => {
                write!(f, "invalid row index {} (trace has {} rows)", row, num_rows)
            }
            TraceError::InvalidColumn { col, num_cols } => {
                write!(f, "invalid column index {} (trace has {} columns)", col, num_cols)
            }
            TraceError::NotPowerOfTwo { height } => {
                write!(f, "trace height {} is not a power of two", height)
            }
            TraceError::EmptyTrace => write!(f, "trace is empty"),
            TraceError::DeserializationError(msg) => {
                write!(f, "deserialization error: {}", msg)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ExecutionTrace
// ═══════════════════════════════════════════════════════════════

/// A two-dimensional matrix of Goldilocks field elements stored in row-major
/// order.  `rows[r][c]` is the element at row `r`, column `c`.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Row-major data: `rows[row][col]`.
    pub rows: Vec<Vec<GoldilocksField>>,
    /// Number of columns (registers).
    pub width: usize,
    /// Number of rows (steps).
    pub length: usize,
    /// Optional human-readable column names.
    column_names: Vec<String>,
}

// ─────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────

impl ExecutionTrace {
    /// Create a trace filled with zeros.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        let rows = vec![vec![GoldilocksField::ZERO; num_cols]; num_rows];
        let column_names = (0..num_cols).map(|i| format!("col_{}", i)).collect();
        Self { rows, length: num_rows, width: num_cols, column_names }
    }

    /// Create an empty trace with given dimensions (width-first argument
    /// order for backward compatibility).
    pub fn zeros(width: usize, length: usize) -> Self {
        Self::new(length, width)
    }

    /// Build a trace from a vector of rows.  All rows must have the same
    /// length.
    pub fn from_rows(rows: Vec<Vec<GoldilocksField>>) -> Self {
        assert!(!rows.is_empty(), "from_rows: cannot construct empty trace");
        let num_cols = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(), num_cols,
                "from_rows: row {} has length {} but expected {}",
                i, row.len(), num_cols
            );
        }
        let num_rows = rows.len();
        let column_names = (0..num_cols).map(|i| format!("col_{}", i)).collect();
        Self { rows, length: num_rows, width: num_cols, column_names }
    }

    /// Build a trace from column-major data.  Each inner `Vec` is one full
    /// column and all columns must have equal length.
    pub fn from_columns(cols: Vec<Vec<GoldilocksField>>) -> Self {
        assert!(!cols.is_empty(), "from_columns: cannot construct empty trace");
        let num_rows = cols[0].len();
        for (i, col) in cols.iter().enumerate() {
            assert_eq!(
                col.len(), num_rows,
                "from_columns: column {} has length {} but expected {}",
                i, col.len(), num_rows
            );
        }
        let num_cols = cols.len();
        let mut rows = vec![vec![GoldilocksField::ZERO; num_cols]; num_rows];
        for c in 0..num_cols {
            for r in 0..num_rows {
                rows[r][c] = cols[c][r];
            }
        }
        let column_names = (0..num_cols).map(|i| format!("col_{}", i)).collect();
        Self { rows, length: num_rows, width: num_cols, column_names }
    }

    // ─────────────────────────────────────────────────────────
    // Element access
    // ─────────────────────────────────────────────────────────

    /// Read a single cell.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> GoldilocksField {
        self.rows[row][col]
    }

    /// Write a single cell.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: GoldilocksField) {
        self.rows[row][col] = val;
    }

    /// Reference to an entire row.
    #[inline]
    pub fn get_row(&self, row: usize) -> &Vec<GoldilocksField> {
        &self.rows[row]
    }

    /// Get a row as a slice (backward-compatible alias).
    #[inline]
    pub fn row(&self, idx: usize) -> &[GoldilocksField] {
        &self.rows[idx]
    }

    /// Extract an entire column (copies the data).
    pub fn get_column(&self, col: usize) -> Vec<GoldilocksField> {
        self.rows.iter().map(|row| row[col]).collect()
    }

    /// Extract an entire column (backward-compatible alias).
    pub fn column(&self, col: usize) -> Vec<GoldilocksField> {
        self.get_column(col)
    }

    /// Overwrite a full row.
    pub fn set_row(&mut self, row: usize, values: Vec<GoldilocksField>) {
        assert_eq!(
            values.len(), self.width,
            "set_row: length {} does not match trace width {}",
            values.len(), self.width
        );
        self.rows[row] = values;
    }

    /// Overwrite a full column.
    pub fn set_column(&mut self, col: usize, values: Vec<GoldilocksField>) {
        assert_eq!(
            values.len(), self.length,
            "set_column: length {} does not match trace height {}",
            values.len(), self.length
        );
        for r in 0..self.length {
            self.rows[r][col] = values[r];
        }
    }

    /// Assign a human-readable name to a column.
    pub fn set_column_name(&mut self, col: usize, name: &str) {
        assert!(col < self.width, "set_column_name: index out of bounds");
        self.column_names[col] = name.to_string();
    }

    /// Trace width (number of columns).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Trace height (number of rows).
    #[inline]
    pub fn height(&self) -> usize {
        self.length
    }

    /// Number of rows (backward-compatible alias).
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.length
    }

    /// Number of columns (backward-compatible alias).
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.width
    }

    /// Get column names.
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Get raw data reference.
    pub fn raw_data(&self) -> &Vec<Vec<GoldilocksField>> {
        &self.rows
    }

    /// Append a row (backward-compatible alias).
    pub fn push_row(&mut self, row: Vec<GoldilocksField>) {
        assert_eq!(row.len(), self.width, "push_row: width mismatch");
        self.rows.push(row);
        self.length += 1;
    }

    /// Check if the trace length is a power of 2 (backward-compatible alias).
    pub fn is_power_of_two(&self) -> bool {
        self.length > 0 && self.length.is_power_of_two()
    }
}

// ═══════════════════════════════════════════════════════════════
// Padding
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Pad the trace to the next power of two height by repeating the last
    /// row.  If the trace is already a power-of-two height this is a no-op.
    pub fn pad_to_power_of_two(&mut self) {
        if self.length == 0 {
            return;
        }
        let target = self.length.next_power_of_two();
        if target == self.length {
            return;
        }
        let last_row = self.rows[self.length - 1].clone();
        while self.rows.len() < target {
            self.rows.push(last_row.clone());
        }
        self.length = target;
    }

    /// Pad with all-zero rows up to `target_rows`.
    pub fn pad_with_zeros(&mut self, target_rows: usize) {
        if target_rows <= self.length {
            return;
        }
        let zero_row = vec![GoldilocksField::ZERO; self.width];
        while self.rows.len() < target_rows {
            self.rows.push(zero_row.clone());
        }
        self.length = target_rows;
    }

    /// Pad with a constant value in every cell up to `target_rows`.
    pub fn pad_with_value(&mut self, target_rows: usize, value: GoldilocksField) {
        if target_rows <= self.length {
            return;
        }
        let fill_row = vec![value; self.width];
        while self.rows.len() < target_rows {
            self.rows.push(fill_row.clone());
        }
        self.length = target_rows;
    }

    /// Check whether the height is a power of two.
    pub fn is_power_of_two_height(&self) -> bool {
        self.length > 0 && self.length.is_power_of_two()
    }
}

// ═══════════════════════════════════════════════════════════════
// Trace Commitment
// ═══════════════════════════════════════════════════════════════

/// The result of committing to an execution trace via a Merkle tree.
#[derive(Debug, Clone)]
pub struct TraceCommitment {
    /// The Merkle tree built over the trace rows.
    pub merkle_tree: MerkleTree,
    /// Root hash of the commitment.
    pub root_hash: Digest,
    /// A clone of the committed trace for answering queries.
    pub committed_trace: ExecutionTrace,
}

impl ExecutionTrace {
    /// Commit to the trace by building a Merkle tree over its rows.
    pub fn commit(&self) -> TraceCommitment {
        let tree = MerkleTree::from_field_rows(&self.rows);
        let root = tree.root();
        TraceCommitment {
            merkle_tree: tree,
            root_hash: root,
            committed_trace: self.clone(),
        }
    }

    /// Query a single row from a commitment, returning the row data and a
    /// Merkle authentication path.
    pub fn query_row(
        &self,
        commitment: &TraceCommitment,
        row_index: usize,
    ) -> (Vec<GoldilocksField>, MerkleProof) {
        assert!(row_index < self.length, "query_row: index out of bounds");
        let row_data = self.rows[row_index].clone();
        let proof = commitment.merkle_tree.prove(row_index);
        (row_data, proof)
    }

    /// Verify that a row query is consistent with a committed root.
    pub fn verify_row_query(
        root: &Digest,
        _row_index: usize,
        row_data: &[GoldilocksField],
        proof: &MerkleProof,
    ) -> bool {
        MerkleTree::verify_field_row(root, row_data, proof)
    }
}

// ═══════════════════════════════════════════════════════════════
// Trace Validation
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Validate that internal dimensions are consistent.
    pub fn validate_dimensions(&self) -> Result<(), TraceError> {
        if self.length == 0 || self.width == 0 {
            return Err(TraceError::EmptyTrace);
        }
        if self.rows.len() != self.length {
            return Err(TraceError::DimensionMismatch {
                expected: self.length,
                got: self.rows.len(),
            });
        }
        for (i, row) in self.rows.iter().enumerate() {
            if row.len() != self.width {
                return Err(TraceError::DimensionMismatch {
                    expected: self.width,
                    got: row.len(),
                });
            }
            if i >= self.length {
                return Err(TraceError::InvalidRow {
                    row: i,
                    num_rows: self.length,
                });
            }
        }
        if self.column_names.len() != self.width {
            return Err(TraceError::DimensionMismatch {
                expected: self.width,
                got: self.column_names.len(),
            });
        }
        Ok(())
    }

    /// Check that a specific cell equals an expected boundary value.
    pub fn validate_boundary(
        &self,
        col: usize,
        row: usize,
        expected_val: GoldilocksField,
    ) -> bool {
        if row >= self.length || col >= self.width {
            return false;
        }
        self.rows[row][col] == expected_val
    }

    /// Check that a transition predicate holds between every pair of
    /// consecutive rows.  `check(current_row, next_row)` must return `true`.
    pub fn validate_transition<F>(&self, check: F) -> bool
    where
        F: Fn(&[GoldilocksField], &[GoldilocksField]) -> bool,
    {
        if self.length <= 1 {
            return true;
        }
        for i in 0..self.length - 1 {
            if !check(&self.rows[i], &self.rows[i + 1]) {
                return false;
            }
        }
        true
    }

    /// Return `true` if every element in column `col` is zero.
    pub fn check_all_zeros_column(&self, col: usize) -> bool {
        assert!(col < self.width, "check_all_zeros_column: column out of bounds");
        for r in 0..self.length {
            if !self.rows[r][col].is_zero() {
                return false;
            }
        }
        true
    }

    /// Return the index of the first row where the transition check fails,
    /// or `None` if all transitions pass.
    pub fn find_first_violation<F>(&self, check: F) -> Option<usize>
    where
        F: Fn(&[GoldilocksField], &[GoldilocksField]) -> bool,
    {
        if self.length <= 1 {
            return None;
        }
        for i in 0..self.length - 1 {
            if !check(&self.rows[i], &self.rows[i + 1]) {
                return Some(i);
            }
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════
// Low Degree Extension (LDE)
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Perform a low-degree extension of the trace.
    ///
    /// Each column is treated as evaluations of a polynomial of degree
    /// `< length` over the roots-of-unity domain of size `length`.
    /// We recover the coefficient form via INTT, then evaluate on a larger
    /// coset domain of size `length * blowup_factor`.
    pub fn low_degree_extend(&self, blowup_factor: usize) -> ExecutionTrace {
        assert!(self.length > 0, "LDE on empty trace");
        assert!(
            self.length.is_power_of_two(),
            "LDE requires power-of-two trace height"
        );
        assert!(
            blowup_factor.is_power_of_two() && blowup_factor >= 2,
            "blowup_factor must be a power of two >= 2"
        );

        let extended_len = self.length * blowup_factor;
        let coset_shift = GoldilocksField::new(7);

        let mut ext_cols: Vec<Vec<GoldilocksField>> = Vec::with_capacity(self.width);
        for c in 0..self.width {
            let col = self.get_column(c);
            let extended = extend_column(&col, blowup_factor, coset_shift);
            ext_cols.push(extended);
        }

        let mut ext_rows = vec![vec![GoldilocksField::ZERO; self.width]; extended_len];
        for c in 0..self.width {
            for r in 0..extended_len {
                ext_rows[r][c] = ext_cols[c][r];
            }
        }

        let column_names: Vec<String> = self
            .column_names
            .iter()
            .map(|n| format!("{}_lde", n))
            .collect();

        ExecutionTrace {
            rows: ext_rows,
            length: extended_len,
            width: self.width,
            column_names,
        }
    }
}

/// Extend a single column using INTT → coset NTT.
pub fn extend_column(
    col_data: &[GoldilocksField],
    blowup: usize,
    coset_shift: GoldilocksField,
) -> Vec<GoldilocksField> {
    let n = col_data.len();
    assert!(n.is_power_of_two(), "column length must be power of 2");
    let mut coeffs = col_data.to_vec();
    intt(&mut coeffs);
    let extended_len = n * blowup;
    evaluate_on_coset(&coeffs, coset_shift, extended_len)
}

// ═══════════════════════════════════════════════════════════════
// Sub-operations (slicing, merging, appending)
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Extract a sub-trace containing only the specified columns.
    pub fn sub_trace(&self, cols: &[usize]) -> ExecutionTrace {
        assert!(!cols.is_empty(), "sub_trace: must select at least one column");
        for &c in cols {
            assert!(c < self.width, "sub_trace: column {} out of bounds", c);
        }
        let mut rows = Vec::with_capacity(self.length);
        for r in 0..self.length {
            let row: Vec<GoldilocksField> = cols.iter().map(|&c| self.rows[r][c]).collect();
            rows.push(row);
        }
        let column_names: Vec<String> = cols
            .iter()
            .map(|&c| self.column_names[c].clone())
            .collect();
        ExecutionTrace {
            rows,
            length: self.length,
            width: cols.len(),
            column_names,
        }
    }

    /// Append a single column to the right side of the trace.
    pub fn append_column(&mut self, col_data: Vec<GoldilocksField>, name: &str) {
        assert_eq!(
            col_data.len(), self.length,
            "append_column: column length {} != trace height {}",
            col_data.len(), self.length
        );
        for r in 0..self.length {
            self.rows[r].push(col_data[r]);
        }
        self.width += 1;
        self.column_names.push(name.to_string());
    }

    /// Append multiple columns at once.
    pub fn append_columns(&mut self, cols: Vec<Vec<GoldilocksField>>, names: &[String]) {
        assert_eq!(cols.len(), names.len());
        for (i, col) in cols.iter().enumerate() {
            assert_eq!(
                col.len(), self.length,
                "append_columns: column {} has length {} but expected {}",
                i, col.len(), self.length
            );
        }
        for r in 0..self.length {
            for col in &cols {
                self.rows[r].push(col[r]);
            }
        }
        self.width += cols.len();
        self.column_names.extend(names.iter().cloned());
    }

    /// Horizontally concatenate multiple traces (same height, columns side by
    /// side).
    pub fn merge_traces(traces: &[&ExecutionTrace]) -> ExecutionTrace {
        assert!(!traces.is_empty(), "merge_traces: need at least one trace");
        let h = traces[0].length;
        for (i, t) in traces.iter().enumerate() {
            assert_eq!(
                t.length, h,
                "merge_traces: trace {} has {} rows but expected {}",
                i, t.length, h
            );
        }
        let total_cols: usize = traces.iter().map(|t| t.width).sum();
        let mut rows = Vec::with_capacity(h);
        for r in 0..h {
            let mut row = Vec::with_capacity(total_cols);
            for t in traces {
                row.extend_from_slice(&t.rows[r]);
            }
            rows.push(row);
        }
        let mut column_names = Vec::with_capacity(total_cols);
        for t in traces {
            column_names.extend(t.column_names.iter().cloned());
        }
        ExecutionTrace { rows, length: h, width: total_cols, column_names }
    }

    /// Vertically stack traces (same width, rows concatenated).
    pub fn stack_traces(traces: &[&ExecutionTrace]) -> ExecutionTrace {
        assert!(!traces.is_empty(), "stack_traces: need at least one trace");
        let w = traces[0].width;
        for (i, t) in traces.iter().enumerate() {
            assert_eq!(
                t.width, w,
                "stack_traces: trace {} has {} cols but expected {}",
                i, t.width, w
            );
        }
        let total_rows: usize = traces.iter().map(|t| t.length).sum();
        let mut rows = Vec::with_capacity(total_rows);
        for t in traces {
            rows.extend(t.rows.iter().cloned());
        }
        let column_names = traces[0].column_names.clone();
        ExecutionTrace { rows, length: total_rows, width: w, column_names }
    }

    /// Return a window consisting of the current row and the next row
    /// (wrapping around at the end).
    pub fn window_at(&self, row: usize) -> TraceWindow {
        assert!(row < self.length, "window_at: row out of bounds");
        let next = (row + 1) % self.length;
        TraceWindow {
            current_row: self.rows[row].clone(),
            next_row: self.rows[next].clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceWindow
// ═══════════════════════════════════════════════════════════════

/// A view of two consecutive rows of the execution trace (current and next).
#[derive(Debug, Clone)]
pub struct TraceWindow {
    pub current_row: Vec<GoldilocksField>,
    pub next_row: Vec<GoldilocksField>,
}

impl TraceWindow {
    #[inline]
    pub fn get_current(&self, col: usize) -> GoldilocksField {
        self.current_row[col]
    }
    #[inline]
    pub fn get_next(&self, col: usize) -> GoldilocksField {
        self.next_row[col]
    }
    pub fn width(&self) -> usize {
        self.current_row.len()
    }
}

// ═══════════════════════════════════════════════════════════════
// Trace Statistics & Visualisation
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ColumnStat {
    pub min_val: u64,
    pub max_val: u64,
    pub zero_count: usize,
    pub distinct_count: usize,
}

#[derive(Debug, Clone)]
pub struct TraceStats {
    pub column_stats: Vec<ColumnStat>,
    pub num_rows: usize,
    pub num_cols: usize,
    pub total_cells: usize,
    pub total_zeros: usize,
}

impl ExecutionTrace {
    pub fn statistics(&self) -> TraceStats {
        let mut column_stats = Vec::with_capacity(self.width);
        let mut total_zeros: usize = 0;

        for c in 0..self.width {
            let mut min_val: u64 = u64::MAX;
            let mut max_val: u64 = 0;
            let mut zero_count: usize = 0;
            let mut distinct: HashSet<u64> = HashSet::new();

            for r in 0..self.length {
                let v = self.rows[r][c].to_canonical();
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
                if v == 0 { zero_count += 1; }
                distinct.insert(v);
            }
            if self.length == 0 { min_val = 0; }
            total_zeros += zero_count;
            column_stats.push(ColumnStat { min_val, max_val, zero_count, distinct_count: distinct.len() });
        }

        TraceStats {
            column_stats,
            num_rows: self.length,
            num_cols: self.width,
            total_cells: self.length * self.width,
            total_zeros,
        }
    }

    pub fn display_ascii(&self, max_rows: usize, max_cols: usize) -> String {
        let rows_to_show = max_rows.min(self.length);
        let cols_to_show = max_cols.min(self.width);
        let truncated_rows = self.length > rows_to_show;
        let truncated_cols = self.width > cols_to_show;

        let mut widths: Vec<usize> = Vec::with_capacity(cols_to_show);
        for c in 0..cols_to_show {
            let header_w = self.column_names[c].len();
            let max_data_w = (0..rows_to_show)
                .map(|r| format!("{}", self.rows[r][c].to_canonical()).len())
                .max()
                .unwrap_or(1);
            widths.push(header_w.max(max_data_w));
        }

        let mut out = String::new();
        out.push_str(&format!("ExecutionTrace  {}×{}", self.length, self.width));
        if truncated_rows || truncated_cols {
            out.push_str(&format!(" (showing {}×{})", rows_to_show, cols_to_show));
        }
        out.push('\n');

        let row_idx_width = format!("{}", rows_to_show).len().max(3);
        out.push_str(&format!("{:>width$} |", "row", width = row_idx_width));
        for c in 0..cols_to_show {
            out.push_str(&format!(" {:>width$}", self.column_names[c], width = widths[c]));
            if c + 1 < cols_to_show { out.push_str(" |"); }
        }
        if truncated_cols { out.push_str(" | ..."); }
        out.push('\n');

        let sep_len: usize = row_idx_width + 2 + widths.iter().sum::<usize>()
            + (cols_to_show.saturating_sub(1)) * 3 + 1;
        out.push_str(&"-".repeat(sep_len));
        out.push('\n');

        for r in 0..rows_to_show {
            out.push_str(&format!("{:>width$} |", r, width = row_idx_width));
            for c in 0..cols_to_show {
                let val = self.rows[r][c].to_canonical();
                out.push_str(&format!(" {:>width$}", val, width = widths[c]));
                if c + 1 < cols_to_show { out.push_str(" |"); }
            }
            if truncated_cols { out.push_str(" | ..."); }
            out.push('\n');
        }
        if truncated_rows {
            out.push_str(&format!("{:>width$} | ...\n", "...", width = row_idx_width));
        }
        out
    }

    pub fn display_column_summary(&self) -> String {
        let stats = self.statistics();
        let mut out = String::new();
        out.push_str(&format!("Trace column summary ({} rows × {} cols)\n", self.length, self.width));
        out.push_str(&format!("{:<20} {:>20} {:>20} {:>10} {:>10}\n", "Column", "Min", "Max", "Zeros", "Distinct"));
        out.push_str(&"-".repeat(82));
        out.push('\n');
        for (i, cs) in stats.column_stats.iter().enumerate() {
            out.push_str(&format!(
                "{:<20} {:>20} {:>20} {:>10} {:>10}\n",
                self.column_names[i], cs.min_val, cs.max_val, cs.zero_count, cs.distinct_count
            ));
        }
        out
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        for (i, name) in self.column_names.iter().enumerate() {
            if i > 0 { out.push(','); }
            out.push_str(name);
        }
        out.push('\n');
        for r in 0..self.length {
            for c in 0..self.width {
                if c > 0 { out.push(','); }
                out.push_str(&format!("{}", self.rows[r][c].to_canonical()));
            }
            out.push('\n');
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════
// Partial Trace / Debug Helpers
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn partial_trace(&self, start_row: usize, end_row: usize) -> ExecutionTrace {
        assert!(
            start_row <= end_row && end_row <= self.length,
            "partial_trace: invalid range [{}..{}), trace has {} rows",
            start_row, end_row, self.length
        );
        let rows: Vec<Vec<GoldilocksField>> = self.rows[start_row..end_row].to_vec();
        let h = end_row - start_row;
        ExecutionTrace { rows, length: h, width: self.width, column_names: self.column_names.clone() }
    }

    pub fn debug_row(&self, row: usize) -> String {
        assert!(row < self.length, "debug_row: row out of bounds");
        let mut out = format!("Row {}:", row);
        for c in 0..self.width {
            out.push_str(&format!(" {}={}", self.column_names[c], self.rows[row][c].to_canonical()));
        }
        out
    }

    pub fn debug_window(&self, row: usize) -> String {
        assert!(row < self.length, "debug_window: row out of bounds");
        let next = (row + 1) % self.length;
        let mut out = format!("Window at row {}:\n", row);
        out.push_str(&format!("  current({}): ", row));
        for c in 0..self.width {
            out.push_str(&format!("{}={} ", self.column_names[c], self.rows[row][c].to_canonical()));
        }
        out.push('\n');
        out.push_str(&format!("  next({}):    ", next));
        for c in 0..self.width {
            out.push_str(&format!("{}={} ", self.column_names[c], self.rows[next][c].to_canonical()));
        }
        out.push('\n');
        out
    }
}

// ═══════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Serialize the trace to a compact binary representation.
    ///
    /// Layout (all little-endian):
    ///   magic   : 4 bytes  "STRC"
    ///   version : 4 bytes  (1)
    ///   num_rows: 8 bytes
    ///   num_cols: 8 bytes
    ///   for each column name:
    ///       name_len : 4 bytes
    ///       name     : name_len bytes (UTF-8)
    ///   data    : num_rows * num_cols * 8 bytes (row-major, LE u64)
    pub fn serialize_to_bytes(&self) -> Vec<u8> {
        let name_bytes_total: usize = self.column_names.iter().map(|n| 4 + n.len()).sum();
        let capacity = 4 + 4 + 8 + 8 + name_bytes_total + self.length * self.width * 8;
        let mut buf = Vec::with_capacity(capacity);

        buf.extend_from_slice(b"STRC");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(self.length as u64).to_le_bytes());
        buf.extend_from_slice(&(self.width as u64).to_le_bytes());

        for name in &self.column_names {
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
        }

        for r in 0..self.length {
            for c in 0..self.width {
                buf.extend_from_slice(&self.rows[r][c].to_bytes_le());
            }
        }
        buf
    }

    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, TraceError> {
        if bytes.len() < 24 {
            return Err(TraceError::DeserializationError("input too short for header".into()));
        }
        if &bytes[0..4] != b"STRC" {
            return Err(TraceError::DeserializationError("invalid magic bytes".into()));
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 1 {
            return Err(TraceError::DeserializationError(format!("unsupported version {}", version)));
        }

        let num_rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let num_cols = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        if num_rows == 0 || num_cols == 0 {
            return Err(TraceError::EmptyTrace);
        }

        let mut pos = 24usize;
        let mut column_names = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            if pos + 4 > bytes.len() {
                return Err(TraceError::DeserializationError("truncated column name length".into()));
            }
            let name_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + name_len > bytes.len() {
                return Err(TraceError::DeserializationError("truncated column name data".into()));
            }
            let name = String::from_utf8(bytes[pos..pos + name_len].to_vec())
                .map_err(|e| TraceError::DeserializationError(e.to_string()))?;
            column_names.push(name);
            pos += name_len;
        }

        let data_bytes_needed = num_rows * num_cols * 8;
        if pos + data_bytes_needed > bytes.len() {
            return Err(TraceError::DeserializationError("truncated data section".into()));
        }

        let mut rows = Vec::with_capacity(num_rows);
        for _r in 0..num_rows {
            let mut row = Vec::with_capacity(num_cols);
            for _c in 0..num_cols {
                let val = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                row.push(GoldilocksField::new(val));
                pos += 8;
            }
            rows.push(row);
        }

        Ok(Self { rows, length: num_rows, width: num_cols, column_names })
    }

    /// Compute a 32-byte hash of the entire trace.
    pub fn hash_trace(&self) -> [u8; 32] {
        let bytes = self.serialize_to_bytes();
        blake3_hash(&bytes)
    }
}

// ═══════════════════════════════════════════════════════════════
// Batch / Polynomial Operations
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Interpolate the column as a polynomial over the roots-of-unity domain
    /// and evaluate at an arbitrary point `x`.
    pub fn evaluate_polynomial_column(
        &self, col: usize, x: GoldilocksField,
    ) -> GoldilocksField {
        let coeffs = self.column_to_polynomial(col);
        GoldilocksField::eval_poly(&coeffs, x)
    }

    /// Recover the coefficient representation of the polynomial whose
    /// evaluations at the roots of unity are the column values.
    pub fn column_to_polynomial(&self, col: usize) -> Vec<GoldilocksField> {
        assert!(col < self.width, "column_to_polynomial: column out of bounds");
        assert!(self.length.is_power_of_two(), "column_to_polynomial requires power-of-two height");
        let mut evals = self.get_column(col);
        intt(&mut evals);
        evals
    }

    /// Given polynomial coefficients, evaluate on the standard roots-of-unity
    /// domain of size `num_rows` and return the evaluations.
    pub fn polynomial_to_column(
        coeffs: &[GoldilocksField], num_rows: usize,
    ) -> Vec<GoldilocksField> {
        assert!(num_rows.is_power_of_two(), "num_rows must be power of 2");
        let mut padded = vec![GoldilocksField::ZERO; num_rows];
        let copy_len = coeffs.len().min(num_rows);
        padded[..copy_len].copy_from_slice(&coeffs[..copy_len]);
        ntt(&mut padded);
        padded
    }

    /// Evaluate every column polynomial at a single point `x`.
    pub fn evaluate_all_columns_at(&self, x: GoldilocksField) -> Vec<GoldilocksField> {
        (0..self.width).map(|c| self.evaluate_polynomial_column(c, x)).collect()
    }

    /// Batch-interpolate: convert every column to coefficient form.
    pub fn all_columns_to_polynomials(&self) -> Vec<Vec<GoldilocksField>> {
        (0..self.width).map(|c| self.column_to_polynomial(c)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Additional Trace Utilities
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Create a trace filled with a constant value.
    pub fn filled(num_rows: usize, num_cols: usize, val: GoldilocksField) -> Self {
        let rows = vec![vec![val; num_cols]; num_rows];
        let column_names = (0..num_cols).map(|i| format!("col_{}", i)).collect();
        Self { rows, length: num_rows, width: num_cols, column_names }
    }

    /// Create a trace where cell (r, c) = r*num_cols + c.
    pub fn identity_like(num_rows: usize, num_cols: usize) -> Self {
        let mut t = Self::new(num_rows, num_cols);
        for r in 0..num_rows {
            for c in 0..num_cols {
                t.rows[r][c] = GoldilocksField::new((r * num_cols + c) as u64);
            }
        }
        t
    }

    /// Transpose: rows ↔ columns.
    pub fn transpose(&self) -> ExecutionTrace {
        let mut new_rows = vec![vec![GoldilocksField::ZERO; self.length]; self.width];
        for r in 0..self.length {
            for c in 0..self.width {
                new_rows[c][r] = self.rows[r][c];
            }
        }
        let column_names = (0..self.length).map(|i| format!("row_{}", i)).collect();
        ExecutionTrace { rows: new_rows, length: self.width, width: self.length, column_names }
    }

    /// Map a function over every cell.
    pub fn map<F>(&self, f: F) -> ExecutionTrace
    where F: Fn(GoldilocksField) -> GoldilocksField,
    {
        let new_rows = self.rows.iter()
            .map(|row| row.iter().map(|&v| f(v)).collect())
            .collect();
        ExecutionTrace { rows: new_rows, length: self.length, width: self.width, column_names: self.column_names.clone() }
    }

    /// Element-wise addition of two traces.
    pub fn add_trace(&self, other: &ExecutionTrace) -> ExecutionTrace {
        assert_eq!(self.length, other.length);
        assert_eq!(self.width, other.width);
        let new_rows = (0..self.length)
            .map(|r| (0..self.width).map(|c| self.rows[r][c].add_elem(other.rows[r][c])).collect())
            .collect();
        ExecutionTrace { rows: new_rows, length: self.length, width: self.width, column_names: self.column_names.clone() }
    }

    /// Element-wise subtraction.
    pub fn sub_trace_elementwise(&self, other: &ExecutionTrace) -> ExecutionTrace {
        assert_eq!(self.length, other.length);
        assert_eq!(self.width, other.width);
        let new_rows = (0..self.length)
            .map(|r| (0..self.width).map(|c| self.rows[r][c].sub_elem(other.rows[r][c])).collect())
            .collect();
        ExecutionTrace { rows: new_rows, length: self.length, width: self.width, column_names: self.column_names.clone() }
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, scalar: GoldilocksField) -> ExecutionTrace {
        self.map(|v| v.mul_elem(scalar))
    }

    /// Check equality of two traces.
    pub fn equals(&self, other: &ExecutionTrace) -> bool {
        if self.length != other.length || self.width != other.width { return false; }
        for r in 0..self.length {
            for c in 0..self.width {
                if self.rows[r][c] != other.rows[r][c] { return false; }
            }
        }
        true
    }

    /// Hamming distance between two equal-size traces.
    pub fn hamming_distance(&self, other: &ExecutionTrace) -> usize {
        assert_eq!(self.length, other.length);
        assert_eq!(self.width, other.width);
        let mut dist = 0usize;
        for r in 0..self.length {
            for c in 0..self.width {
                if self.rows[r][c] != other.rows[r][c] { dist += 1; }
            }
        }
        dist
    }

    /// Compute a hash digest for each row.
    pub fn row_hashes(&self) -> Vec<Digest> {
        self.rows.iter()
            .map(|row| {
                let mut bytes = Vec::with_capacity(row.len() * 8);
                for elem in row { bytes.extend_from_slice(&elem.to_bytes_le()); }
                blake3_hash(&bytes)
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Constraint Composition Helpers
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Evaluate a linear combination of columns at a given row.
    pub fn linear_combination_at(
        &self, row: usize, col_indices: &[usize], coeffs: &[GoldilocksField],
    ) -> GoldilocksField {
        assert_eq!(col_indices.len(), coeffs.len());
        let mut acc = GoldilocksField::ZERO;
        for (&ci, &coeff) in col_indices.iter().zip(coeffs.iter()) {
            acc = acc.add_elem(self.rows[row][ci].mul_elem(coeff));
        }
        acc
    }

    /// Compute a new column as a linear combination of existing columns.
    pub fn derive_column_linear(
        &self, col_indices: &[usize], coeffs: &[GoldilocksField],
    ) -> Vec<GoldilocksField> {
        assert_eq!(col_indices.len(), coeffs.len());
        (0..self.length).map(|r| self.linear_combination_at(r, col_indices, coeffs)).collect()
    }

    /// Compute a column that is the product of two columns.
    pub fn derive_column_product(&self, col_a: usize, col_b: usize) -> Vec<GoldilocksField> {
        (0..self.length).map(|r| self.rows[r][col_a].mul_elem(self.rows[r][col_b])).collect()
    }

    /// Compute a "constraint polynomial" column.
    pub fn compute_constraint_column<F>(&self, constraint_fn: F) -> Vec<GoldilocksField>
    where F: Fn(&[GoldilocksField], &[GoldilocksField]) -> GoldilocksField,
    {
        let mut col = Vec::with_capacity(self.length);
        for r in 0..self.length {
            let next = (r + 1) % self.length;
            col.push(constraint_fn(&self.rows[r], &self.rows[next]));
        }
        col
    }

    /// Verify that a constraint polynomial column is zero everywhere.
    pub fn verify_constraint_column<F>(&self, constraint_fn: F) -> bool
    where F: Fn(&[GoldilocksField], &[GoldilocksField]) -> GoldilocksField,
    {
        for r in 0..self.length {
            let next = (r + 1) % self.length;
            if !constraint_fn(&self.rows[r], &self.rows[next]).is_zero() { return false; }
        }
        true
    }

    /// Find rows where a constraint evaluates to nonzero.
    pub fn find_constraint_violations<F>(&self, constraint_fn: F) -> Vec<usize>
    where F: Fn(&[GoldilocksField], &[GoldilocksField]) -> GoldilocksField,
    {
        let mut violations = Vec::new();
        for r in 0..self.length {
            let next = (r + 1) % self.length;
            if !constraint_fn(&self.rows[r], &self.rows[next]).is_zero() { violations.push(r); }
        }
        violations
    }
}

// ═══════════════════════════════════════════════════════════════
// Permutation & Lookup Argument Helpers
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    /// Running-product column for a permutation argument.
    pub fn running_product_column(&self, col: usize, beta: GoldilocksField) -> Vec<GoldilocksField> {
        let mut rp = Vec::with_capacity(self.length);
        let mut acc = GoldilocksField::ONE;
        for r in 0..self.length {
            acc = acc.mul_elem(self.rows[r][col].add_elem(beta));
            rp.push(acc);
        }
        rp
    }

    /// Running-sum column for log-derivative lookups.
    pub fn running_sum_column(&self, col: usize, beta: GoldilocksField) -> Vec<GoldilocksField> {
        let mut rs = Vec::with_capacity(self.length);
        let mut acc = GoldilocksField::ZERO;
        for r in 0..self.length {
            let denom = self.rows[r][col].add_elem(beta);
            if let Some(inv) = denom.inv() { acc = acc.add_elem(inv); }
            rs.push(acc);
        }
        rs
    }

    /// Multi-column running product.
    pub fn multi_column_running_product(
        &self, cols: &[usize], alpha: GoldilocksField, beta: GoldilocksField,
    ) -> Vec<GoldilocksField> {
        let mut rp = Vec::with_capacity(self.length);
        let mut acc = GoldilocksField::ONE;
        for r in 0..self.length {
            let mut lc = GoldilocksField::ZERO;
            let mut alpha_power = GoldilocksField::ONE;
            for &c in cols {
                lc = lc.add_elem(self.rows[r][c].mul_elem(alpha_power));
                alpha_power = alpha_power.mul_elem(alpha);
            }
            acc = acc.mul_elem(lc.add_elem(beta));
            rp.push(acc);
        }
        rp
    }
}

// ═══════════════════════════════════════════════════════════════
// Quotient Helpers
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn vanishing_poly_evals(
        domain_size: usize, lde_size: usize, coset_shift: GoldilocksField,
    ) -> Vec<GoldilocksField> {
        assert!(domain_size.is_power_of_two());
        assert!(lde_size.is_power_of_two());
        let omega_lde = GoldilocksField::root_of_unity(lde_size);
        let mut evals = Vec::with_capacity(lde_size);
        let mut point = coset_shift;
        for _ in 0..lde_size {
            let z = point.pow(domain_size as u64).sub_elem(GoldilocksField::ONE);
            evals.push(z);
            point = point.mul_elem(omega_lde);
        }
        evals
    }

    pub fn quotient_column(
        constraint_evals: &[GoldilocksField], vanishing_evals: &[GoldilocksField],
    ) -> Vec<GoldilocksField> {
        assert_eq!(constraint_evals.len(), vanishing_evals.len());
        constraint_evals.iter().zip(vanishing_evals.iter())
            .map(|(&c, &z)| {
                if z.is_zero() { GoldilocksField::ZERO }
                else { c.mul_elem(z.inv_or_panic()) }
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Trace Builders (common AIR patterns)
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn fibonacci_trace(a: GoldilocksField, b: GoldilocksField, steps: usize) -> Self {
        assert!(steps >= 2);
        let n = steps.next_power_of_two();
        let mut trace = Self::new(n, 2);
        trace.set_column_name(0, "fib_prev");
        trace.set_column_name(1, "fib_curr");
        trace.set(0, 0, a);
        trace.set(0, 1, b);
        for i in 1..steps {
            let prev = trace.get(i - 1, 0);
            let curr = trace.get(i - 1, 1);
            let next_val = prev.add_elem(curr);
            trace.set(i, 0, curr);
            trace.set(i, 1, next_val);
        }
        if n > steps {
            let last_0 = trace.get(steps - 1, 0);
            let last_1 = trace.get(steps - 1, 1);
            for i in steps..n { trace.set(i, 0, last_0); trace.set(i, 1, last_1); }
        }
        trace
    }

    pub fn counter_trace(n: usize) -> Self {
        let size = n.next_power_of_two();
        let mut trace = Self::new(size, 1);
        trace.set_column_name(0, "counter");
        for i in 0..n { trace.set(i, 0, GoldilocksField::new(i as u64)); }
        trace
    }

    pub fn repeated_squaring_trace(base: GoldilocksField, steps: usize) -> Self {
        let n = steps.next_power_of_two();
        let mut trace = Self::new(n, 1);
        trace.set_column_name(0, "value");
        trace.set(0, 0, base);
        for i in 1..steps {
            let prev = trace.get(i - 1, 0);
            trace.set(i, 0, prev.square());
        }
        if n > steps {
            let last = trace.get(steps - 1, 0);
            for i in steps..n { trace.set(i, 0, last); }
        }
        trace
    }

    pub fn wfa_step_trace(
        states: &[u64], symbols: &[u64], weights_num: &[u64], weights_den: &[u64],
    ) -> Self {
        let steps = states.len();
        assert_eq!(steps, symbols.len());
        assert_eq!(steps, weights_num.len());
        assert_eq!(steps, weights_den.len());
        assert!(steps >= 1);

        let n = steps.next_power_of_two();
        let mut trace = Self::new(n, 6);
        trace.set_column_name(0, "state");
        trace.set_column_name(1, "symbol");
        trace.set_column_name(2, "w_num");
        trace.set_column_name(3, "w_den");
        trace.set_column_name(4, "acc_num");
        trace.set_column_name(5, "acc_den");

        trace.set(0, 0, GoldilocksField::new(states[0]));
        trace.set(0, 1, GoldilocksField::new(symbols[0]));
        trace.set(0, 2, GoldilocksField::new(weights_num[0]));
        trace.set(0, 3, GoldilocksField::new(weights_den[0]));
        trace.set(0, 4, GoldilocksField::new(weights_num[0]));
        trace.set(0, 5, GoldilocksField::new(weights_den[0]));

        for i in 1..steps {
            trace.set(i, 0, GoldilocksField::new(states[i]));
            trace.set(i, 1, GoldilocksField::new(symbols[i]));
            trace.set(i, 2, GoldilocksField::new(weights_num[i]));
            trace.set(i, 3, GoldilocksField::new(weights_den[i]));

            let prev_num = trace.get(i - 1, 4);
            let prev_den = trace.get(i - 1, 5);
            let w_num = GoldilocksField::new(weights_num[i]);
            let w_den = GoldilocksField::new(weights_den[i]);

            let new_num = prev_num.mul_elem(w_den).add_elem(w_num.mul_elem(prev_den));
            let new_den = prev_den.mul_elem(w_den);

            trace.set(i, 4, new_num);
            trace.set(i, 5, new_den);
        }

        if n > steps {
            let last_row = trace.rows[steps - 1].clone();
            for i in steps..n { trace.rows[i] = last_row.clone(); }
        }
        trace
    }
}

// ═══════════════════════════════════════════════════════════════
// Randomized Trace Extension (DEEP-ALI / composition)
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn extend_with_random_linear_combinations(
        &mut self, num_extra: usize, alphas: &[Vec<GoldilocksField>],
    ) {
        assert_eq!(alphas.len(), num_extra);
        for k in 0..num_extra {
            assert_eq!(alphas[k].len(), self.width);
            let new_col = self.derive_column_linear(
                &(0..self.width).collect::<Vec<_>>(), &alphas[k],
            );
            for r in 0..self.length { self.rows[r].push(new_col[r]); }
            self.column_names.push(format!("rlc_{}", k));
        }
        self.width += num_extra;
    }

    pub fn boundary_quotient_column(
        &self, col: usize, row: usize, val: GoldilocksField,
        domain_size: usize, coset_shift: GoldilocksField,
    ) -> Vec<GoldilocksField> {
        let omega = GoldilocksField::root_of_unity(domain_size);
        let omega_row = omega.pow(row as u64);
        let omega_lde = GoldilocksField::root_of_unity(self.length);

        let mut result = Vec::with_capacity(self.length);
        let mut point = coset_shift;
        for r in 0..self.length {
            let numerator = self.rows[r][col].sub_elem(val);
            let denominator = point.sub_elem(omega_row);
            if denominator.is_zero() { result.push(GoldilocksField::ZERO); }
            else { result.push(numerator.mul_elem(denominator.inv_or_panic())); }
            point = point.mul_elem(omega_lde);
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// Display impl
// ═══════════════════════════════════════════════════════════════

impl fmt::Display for ExecutionTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_ascii(16, 8))
    }
}

impl fmt::Display for TraceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraceStats {{ rows: {}, cols: {}, total_cells: {}, total_zeros: {} }}",
            self.num_rows, self.num_cols, self.total_cells, self.total_zeros)
    }
}

// ═══════════════════════════════════════════════════════════════
// Iterator support
// ═══════════════════════════════════════════════════════════════

pub struct TraceRowIter<'a> {
    trace: &'a ExecutionTrace,
    index: usize,
}

impl<'a> Iterator for TraceRowIter<'a> {
    type Item = &'a Vec<GoldilocksField>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.trace.length {
            let row = &self.trace.rows[self.index];
            self.index += 1;
            Some(row)
        } else { None }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.trace.length - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TraceRowIter<'a> {}

pub struct TraceWindowIter<'a> {
    trace: &'a ExecutionTrace,
    index: usize,
}

impl<'a> Iterator for TraceWindowIter<'a> {
    type Item = TraceWindow;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.trace.length {
            let w = self.trace.window_at(self.index);
            self.index += 1;
            Some(w)
        } else { None }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.trace.length - self.index;
        (remaining, Some(remaining))
    }
}

impl ExecutionTrace {
    pub fn iter_rows(&self) -> TraceRowIter<'_> {
        TraceRowIter { trace: self, index: 0 }
    }
    pub fn iter_windows(&self) -> TraceWindowIter<'_> {
        TraceWindowIter { trace: self, index: 0 }
    }
}

// ═══════════════════════════════════════════════════════════════
// Degree-bound checking
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn check_degree_bound(&self, col: usize, max_degree: usize) -> bool {
        let coeffs = self.column_to_polynomial(col);
        for i in max_degree..coeffs.len() {
            if !coeffs[i].is_zero() { return false; }
        }
        true
    }

    pub fn column_degree(&self, col: usize) -> usize {
        let coeffs = self.column_to_polynomial(col);
        for i in (0..coeffs.len()).rev() {
            if !coeffs[i].is_zero() { return i; }
        }
        0
    }
}

// ═══════════════════════════════════════════════════════════════
// Coset domain helpers
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn evaluation_domain(&self) -> Vec<GoldilocksField> {
        assert!(self.length.is_power_of_two());
        GoldilocksField::roots_of_unity(self.length)
    }

    pub fn coset_domain(&self, coset_shift: GoldilocksField) -> Vec<GoldilocksField> {
        assert!(self.length.is_power_of_two());
        let roots = GoldilocksField::roots_of_unity(self.length);
        roots.iter().map(|&r| r.mul_elem(coset_shift)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Multi-column interpolation & composition polynomial
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn compose_columns(&self, alpha: GoldilocksField) -> Vec<GoldilocksField> {
        assert!(self.length.is_power_of_two());
        let polys = self.all_columns_to_polynomials();
        let mut result = vec![GoldilocksField::ZERO; self.length];
        let mut alpha_power = GoldilocksField::ONE;
        for poly in &polys {
            for (i, &coeff) in poly.iter().enumerate() {
                result[i] = result[i].add_elem(coeff.mul_elem(alpha_power));
            }
            alpha_power = alpha_power.mul_elem(alpha);
        }
        result
    }

    pub fn evaluate_composition(
        &self, alpha: GoldilocksField, x: GoldilocksField,
    ) -> GoldilocksField {
        let comp = self.compose_columns(alpha);
        GoldilocksField::eval_poly(&comp, x)
    }
}

// ═══════════════════════════════════════════════════════════════
// Trace-level Merkle helpers for FRI
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn commit_columns(&self) -> Vec<(MerkleTree, Digest)> {
        (0..self.width).map(|c| {
            let col = self.get_column(c);
            let leaves: Vec<Vec<u8>> = col.iter().map(|v| v.to_bytes_le().to_vec()).collect();
            let tree = MerkleTree::from_leaves(&leaves);
            let root = tree.root();
            (tree, root)
        }).collect()
    }

    pub fn query_column_element(
        col_tree: &MerkleTree, col_data: &[GoldilocksField], index: usize,
    ) -> (GoldilocksField, MerkleProof) {
        let val = col_data[index];
        let proof = col_tree.prove(index);
        (val, proof)
    }

    pub fn verify_column_element(
        root: &Digest, value: GoldilocksField, proof: &MerkleProof,
    ) -> bool {
        let bytes = value.to_bytes_le();
        MerkleTree::verify(root, &bytes, proof)
    }
}

// ═══════════════════════════════════════════════════════════════
// DEEP composition helpers (out-of-domain sampling)
// ═══════════════════════════════════════════════════════════════

impl ExecutionTrace {
    pub fn deep_query(
        &self, z: GoldilocksField,
    ) -> (Vec<GoldilocksField>, Vec<GoldilocksField>) {
        let omega = GoldilocksField::root_of_unity(self.length);
        let z_next = z.mul_elem(omega);
        let evals_z = self.evaluate_all_columns_at(z);
        let evals_z_next = self.evaluate_all_columns_at(z_next);
        (evals_z, evals_z_next)
    }

    pub fn deep_composition_column(
        lde_trace: &ExecutionTrace,
        evals_at_z: &[GoldilocksField],
        evals_at_z_next: &[GoldilocksField],
        z: GoldilocksField,
        alpha: GoldilocksField,
        domain_size: usize,
        coset_shift: GoldilocksField,
    ) -> Vec<GoldilocksField> {
        let lde_size = lde_trace.length;
        let omega_lde = GoldilocksField::root_of_unity(lde_size);
        let omega_trace = GoldilocksField::root_of_unity(domain_size);
        let z_next = z.mul_elem(omega_trace);

        let num_cols = lde_trace.width;
        let mut result = vec![GoldilocksField::ZERO; lde_size];

        let mut alpha_power = GoldilocksField::ONE;
        for c in 0..num_cols {
            let val_z = evals_at_z[c];
            let val_z_next = evals_at_z_next[c];

            let mut point = coset_shift;
            for r in 0..lde_size {
                let trace_val = lde_trace.rows[r][c];

                let num1 = trace_val.sub_elem(val_z);
                let den1 = point.sub_elem(z);
                let term1 = if den1.is_zero() { GoldilocksField::ZERO }
                            else { num1.mul_elem(den1.inv_or_panic()) };

                let num2 = trace_val.sub_elem(val_z_next);
                let den2 = point.sub_elem(z_next);
                let term2 = if den2.is_zero() { GoldilocksField::ZERO }
                            else { alpha.mul_elem(num2.mul_elem(den2.inv_or_panic())) };

                let contribution = term1.add_elem(term2).mul_elem(alpha_power);
                result[r] = result[r].add_elem(contribution);

                point = point.mul_elem(omega_lde);
            }
            alpha_power = alpha_power.mul_elem(alpha);
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// AnomalyType
// ═══════════════════════════════════════════════════════════════

/// Classification of anomalies detected in a trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyType {
    /// A zero value appeared in a column that is otherwise non-zero.
    ZeroInNonZeroColumn,
    /// A large discontinuous jump between consecutive rows.
    DiscontinuousTransition,
    /// A value lies outside an expected range.
    ValueOutOfRange,
    /// Two consecutive rows are identical (possible padding leak).
    RepeatedRow,
}

impl fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnomalyType::ZeroInNonZeroColumn => write!(f, "ZeroInNonZeroColumn"),
            AnomalyType::DiscontinuousTransition => write!(f, "DiscontinuousTransition"),
            AnomalyType::ValueOutOfRange => write!(f, "ValueOutOfRange"),
            AnomalyType::RepeatedRow => write!(f, "RepeatedRow"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceAnomaly
// ═══════════════════════════════════════════════════════════════

/// A single anomaly detected within a trace.
#[derive(Debug, Clone)]
pub struct TraceAnomaly {
    pub row: usize,
    pub col: usize,
    pub anomaly_type: AnomalyType,
    pub details: String,
}

impl fmt::Display for TraceAnomaly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[row={}, col={}] {}: {}", self.row, self.col, self.anomaly_type, self.details)
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceAnalysis
// ═══════════════════════════════════════════════════════════════

/// Result of analyzing a trace.
#[derive(Debug, Clone)]
pub struct TraceAnalysis {
    pub num_zero_cols: usize,
    pub num_constant_cols: usize,
    pub periodic_cols: Vec<(usize, usize)>,
    pub col_entropies: Vec<f64>,
    pub anomalies: Vec<TraceAnomaly>,
    pub overall_density: f64,
}

impl TraceAnalysis {
    /// Produce a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "TraceAnalysis {{ zero_cols: {}, constant_cols: {}, periodic_cols: {}, anomalies: {}, density: {:.4} }}",
            self.num_zero_cols,
            self.num_constant_cols,
            self.periodic_cols.len(),
            self.anomalies.len(),
            self.overall_density,
        )
    }

    /// Serialize to a JSON-like string.
    pub fn to_json(&self) -> String {
        let periodic: Vec<String> = self.periodic_cols.iter()
            .map(|(c, p)| format!("{{\"col\":{},\"period\":{}}}", c, p))
            .collect();
        let entropies: Vec<String> = self.col_entropies.iter()
            .map(|e| format!("{:.6}", e))
            .collect();
        let anomalies_json: Vec<String> = self.anomalies.iter()
            .map(|a| format!(
                "{{\"row\":{},\"col\":{},\"type\":\"{}\",\"details\":\"{}\"}}",
                a.row, a.col, a.anomaly_type, a.details
            ))
            .collect();
        format!(
            "{{\"num_zero_cols\":{},\"num_constant_cols\":{},\"periodic_cols\":[{}],\"col_entropies\":[{}],\"anomalies\":[{}],\"overall_density\":{:.6}}}",
            self.num_zero_cols,
            self.num_constant_cols,
            periodic.join(","),
            entropies.join(","),
            anomalies_json.join(","),
            self.overall_density,
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceAnalyzer
// ═══════════════════════════════════════════════════════════════

/// Analyzes trace properties such as periodicity, entropy, and anomalies.
pub struct TraceAnalyzer;

impl TraceAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        TraceAnalyzer
    }

    /// Perform a full analysis of a trace.
    pub fn analyze(&self, trace: &ExecutionTrace) -> TraceAnalysis {
        let zero_cols = self.find_zero_columns(trace);
        let constant_cols = self.find_constant_columns(trace);
        let periodic_cols = self.find_periodic_columns(trace);
        let col_entropies: Vec<f64> = (0..trace.width)
            .map(|c| self.column_entropy(trace, c))
            .collect();
        let anomalies = self.detect_anomalies(trace);

        let total_cells = trace.length * trace.width;
        let nonzero_count = if total_cells == 0 {
            0
        } else {
            let mut count = 0usize;
            for r in 0..trace.length {
                for c in 0..trace.width {
                    if !trace.rows[r][c].is_zero() {
                        count += 1;
                    }
                }
            }
            count
        };
        let overall_density = if total_cells == 0 {
            0.0
        } else {
            nonzero_count as f64 / total_cells as f64
        };

        TraceAnalysis {
            num_zero_cols: zero_cols.len(),
            num_constant_cols: constant_cols.len(),
            periodic_cols,
            col_entropies,
            anomalies,
            overall_density,
        }
    }

    /// Find columns that are periodic, returning (col_index, period).
    pub fn find_periodic_columns(&self, trace: &ExecutionTrace) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        if trace.length <= 1 {
            return result;
        }
        for c in 0..trace.width {
            let col = trace.get_column(c);
            // Try periods from 1 up to length/2
            for period in 1..=trace.length / 2 {
                if trace.length % period != 0 {
                    continue;
                }
                let mut is_periodic = true;
                for r in period..trace.length {
                    if col[r] != col[r % period] {
                        is_periodic = false;
                        break;
                    }
                }
                if is_periodic {
                    // period=1 means constant; skip if we only want non-trivial periodicity
                    // but we still report it
                    result.push((c, period));
                    break; // smallest period found
                }
            }
        }
        result
    }

    /// Find columns where every element is the same value.
    pub fn find_constant_columns(&self, trace: &ExecutionTrace) -> Vec<usize> {
        let mut result = Vec::new();
        for c in 0..trace.width {
            if trace.length == 0 {
                result.push(c);
                continue;
            }
            let first = trace.rows[0][c];
            let mut constant = true;
            for r in 1..trace.length {
                if trace.rows[r][c] != first {
                    constant = false;
                    break;
                }
            }
            if constant {
                result.push(c);
            }
        }
        result
    }

    /// Find columns where every element is zero.
    pub fn find_zero_columns(&self, trace: &ExecutionTrace) -> Vec<usize> {
        let mut result = Vec::new();
        for c in 0..trace.width {
            let mut all_zero = true;
            for r in 0..trace.length {
                if !trace.rows[r][c].is_zero() {
                    all_zero = false;
                    break;
                }
            }
            if all_zero {
                result.push(c);
            }
        }
        result
    }

    /// Compute the Shannon entropy (in bits) of a column's value distribution.
    pub fn column_entropy(&self, trace: &ExecutionTrace, col: usize) -> f64 {
        if trace.length == 0 {
            return 0.0;
        }
        let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for r in 0..trace.length {
            *counts.entry(trace.rows[r][col].to_canonical()).or_insert(0) += 1;
        }
        let n = trace.length as f64;
        let mut entropy = 0.0f64;
        for &count in counts.values() {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Compute similarity between two rows as the fraction of matching cells.
    pub fn row_similarity(&self, trace: &ExecutionTrace, row_a: usize, row_b: usize) -> f64 {
        if trace.width == 0 {
            return 1.0;
        }
        let mut matches = 0usize;
        for c in 0..trace.width {
            if trace.rows[row_a][c] == trace.rows[row_b][c] {
                matches += 1;
            }
        }
        matches as f64 / trace.width as f64
    }

    /// Compute the autocorrelation of a column at a given lag.
    /// Returns a value in [-1, 1] based on normalized correlation of (value, value+lag).
    pub fn column_autocorrelation(&self, trace: &ExecutionTrace, col: usize, lag: usize) -> f64 {
        if trace.length <= lag || trace.length == 0 {
            return 0.0;
        }
        let n = trace.length - lag;
        let vals: Vec<f64> = (0..trace.length)
            .map(|r| trace.rows[r][col].to_canonical() as f64)
            .collect();
        let mean: f64 = vals.iter().sum::<f64>() / trace.length as f64;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..n {
            num += (vals[i] - mean) * (vals[i + lag] - mean);
        }
        for v in &vals {
            den += (v - mean) * (v - mean);
        }
        if den.abs() < 1e-30 {
            return 1.0; // constant column → perfect correlation
        }
        num / den
    }

    /// Detect anomalies in the trace.
    pub fn detect_anomalies(&self, trace: &ExecutionTrace) -> Vec<TraceAnomaly> {
        let mut anomalies = Vec::new();
        if trace.length == 0 || trace.width == 0 {
            return anomalies;
        }

        // Detect zeros in otherwise non-zero columns
        for c in 0..trace.width {
            let mut zero_rows = Vec::new();
            let mut nonzero_count = 0usize;
            for r in 0..trace.length {
                if trace.rows[r][c].is_zero() {
                    zero_rows.push(r);
                } else {
                    nonzero_count += 1;
                }
            }
            // If the column is mostly non-zero but has a few zeros, flag them
            if nonzero_count > 0 && zero_rows.len() > 0 && zero_rows.len() * 4 < trace.length {
                for &r in &zero_rows {
                    anomalies.push(TraceAnomaly {
                        row: r,
                        col: c,
                        anomaly_type: AnomalyType::ZeroInNonZeroColumn,
                        details: format!("zero at row {} in column with {} nonzero values", r, nonzero_count),
                    });
                }
            }
        }

        // Detect repeated rows
        for r in 1..trace.length {
            if trace.rows[r] == trace.rows[r - 1] {
                anomalies.push(TraceAnomaly {
                    row: r,
                    col: 0,
                    anomaly_type: AnomalyType::RepeatedRow,
                    details: format!("row {} is identical to row {}", r, r - 1),
                });
            }
        }

        // Detect discontinuous transitions (large jumps in value)
        for c in 0..trace.width {
            for r in 1..trace.length {
                let prev = trace.rows[r - 1][c].to_canonical();
                let curr = trace.rows[r][c].to_canonical();
                let diff = if curr >= prev { curr - prev } else { prev - curr };
                // Flag if the jump is more than half the field
                if diff > GoldilocksField::MODULUS / 2 {
                    anomalies.push(TraceAnomaly {
                        row: r,
                        col: c,
                        anomaly_type: AnomalyType::DiscontinuousTransition,
                        details: format!("large jump from {} to {} at row {}", prev, curr, r),
                    });
                }
            }
        }

        anomalies
    }
}

// ═══════════════════════════════════════════════════════════════
// ColumnEncoding
// ═══════════════════════════════════════════════════════════════

/// Encoding strategy used for a compressed column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnEncoding {
    /// Raw uncompressed bytes.
    Raw,
    /// Run-length encoded.
    RunLength,
    /// Delta encoded (differences between consecutive elements).
    Delta,
    /// Column is a single constant value.
    Constant(GoldilocksField),
}

// ═══════════════════════════════════════════════════════════════
// CompressedColumn
// ═══════════════════════════════════════════════════════════════

/// A single column in compressed form.
#[derive(Debug, Clone)]
pub struct CompressedColumn {
    pub encoding: ColumnEncoding,
    pub data: Vec<u8>,
}

// ═══════════════════════════════════════════════════════════════
// CompressedTrace
// ═══════════════════════════════════════════════════════════════

/// A compressed representation of an execution trace.
#[derive(Debug, Clone)]
pub struct CompressedTrace {
    pub columns: Vec<CompressedColumn>,
    pub num_rows: usize,
    pub num_cols: usize,
    pub original_size_bytes: usize,
}

impl CompressedTrace {
    /// Total size of compressed data in bytes.
    pub fn size_bytes(&self) -> usize {
        self.columns.iter().map(|c| c.data.len()).sum()
    }

    /// Check if the compressed trace has enough data to decompress.
    pub fn decompressible(&self) -> bool {
        self.num_cols == self.columns.len() && self.num_rows > 0
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceCompressor
// ═══════════════════════════════════════════════════════════════

/// Compresses and decompresses execution traces for storage.
pub struct TraceCompressor;

impl TraceCompressor {
    /// Compress a trace, choosing the best encoding per column.
    pub fn compress(trace: &ExecutionTrace) -> CompressedTrace {
        let original_size_bytes = trace.length * trace.width * 8;
        let mut columns = Vec::with_capacity(trace.width);

        for c in 0..trace.width {
            let col = trace.get_column(c);

            // Check if constant
            let first = col[0];
            let is_constant = col.iter().all(|&v| v == first);
            if is_constant {
                columns.push(CompressedColumn {
                    encoding: ColumnEncoding::Constant(first),
                    data: first.to_bytes_le().to_vec(),
                });
                continue;
            }

            // Try run-length encoding
            let rle = Self::run_length_encode(&col);
            let rle_bytes = rle.len() * 16; // 8 bytes value + 8 bytes count

            // Try delta encoding
            let delta = Self::delta_encode(&col);
            let delta_bytes = delta.len() * 8;

            let raw_bytes = col.len() * 8;

            // Pick smallest
            if rle_bytes < delta_bytes && rle_bytes < raw_bytes {
                let mut data = Vec::with_capacity(rle_bytes);
                for (val, count) in &rle {
                    data.extend_from_slice(&val.to_bytes_le());
                    data.extend_from_slice(&(*count as u64).to_le_bytes());
                }
                columns.push(CompressedColumn {
                    encoding: ColumnEncoding::RunLength,
                    data,
                });
            } else if delta_bytes < raw_bytes {
                let mut data = Vec::with_capacity(delta_bytes);
                for v in &delta {
                    data.extend_from_slice(&v.to_bytes_le());
                }
                columns.push(CompressedColumn {
                    encoding: ColumnEncoding::Delta,
                    data,
                });
            } else {
                let mut data = Vec::with_capacity(raw_bytes);
                for v in &col {
                    data.extend_from_slice(&v.to_bytes_le());
                }
                columns.push(CompressedColumn {
                    encoding: ColumnEncoding::Raw,
                    data,
                });
            }
        }

        CompressedTrace {
            columns,
            num_rows: trace.length,
            num_cols: trace.width,
            original_size_bytes,
        }
    }

    /// Decompress a compressed trace back to an ExecutionTrace.
    pub fn decompress(compressed: &CompressedTrace) -> ExecutionTrace {
        let mut cols_data: Vec<Vec<GoldilocksField>> = Vec::with_capacity(compressed.num_cols);

        for cc in &compressed.columns {
            let col = match &cc.encoding {
                ColumnEncoding::Constant(val) => {
                    vec![*val; compressed.num_rows]
                }
                ColumnEncoding::Raw => {
                    let mut col = Vec::with_capacity(compressed.num_rows);
                    let mut pos = 0;
                    while pos + 8 <= cc.data.len() {
                        let val = u64::from_le_bytes(cc.data[pos..pos + 8].try_into().unwrap());
                        col.push(GoldilocksField::new(val));
                        pos += 8;
                    }
                    col
                }
                ColumnEncoding::RunLength => {
                    let mut encoded = Vec::new();
                    let mut pos = 0;
                    while pos + 16 <= cc.data.len() {
                        let val = u64::from_le_bytes(cc.data[pos..pos + 8].try_into().unwrap());
                        let count = u64::from_le_bytes(cc.data[pos + 8..pos + 16].try_into().unwrap()) as usize;
                        encoded.push((GoldilocksField::new(val), count));
                        pos += 16;
                    }
                    Self::run_length_decode(&encoded)
                }
                ColumnEncoding::Delta => {
                    let mut encoded = Vec::new();
                    let mut pos = 0;
                    while pos + 8 <= cc.data.len() {
                        let val = u64::from_le_bytes(cc.data[pos..pos + 8].try_into().unwrap());
                        encoded.push(GoldilocksField::new(val));
                        pos += 8;
                    }
                    Self::delta_decode(&encoded)
                }
            };
            cols_data.push(col);
        }

        // Build row-major trace from column data
        let mut rows = vec![vec![GoldilocksField::ZERO; compressed.num_cols]; compressed.num_rows];
        for c in 0..compressed.num_cols {
            for r in 0..compressed.num_rows.min(cols_data[c].len()) {
                rows[r][c] = cols_data[c][r];
            }
        }

        let column_names = (0..compressed.num_cols).map(|i| format!("col_{}", i)).collect();
        ExecutionTrace {
            rows,
            length: compressed.num_rows,
            width: compressed.num_cols,
            column_names,
        }
    }

    /// Run-length encode a column.
    pub fn run_length_encode(column: &[GoldilocksField]) -> Vec<(GoldilocksField, usize)> {
        if column.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut current = column[0];
        let mut count = 1usize;
        for i in 1..column.len() {
            if column[i] == current {
                count += 1;
            } else {
                result.push((current, count));
                current = column[i];
                count = 1;
            }
        }
        result.push((current, count));
        result
    }

    /// Decode a run-length encoded column.
    pub fn run_length_decode(encoded: &[(GoldilocksField, usize)]) -> Vec<GoldilocksField> {
        let total: usize = encoded.iter().map(|(_, c)| c).sum();
        let mut result = Vec::with_capacity(total);
        for &(val, count) in encoded {
            for _ in 0..count {
                result.push(val);
            }
        }
        result
    }

    /// Delta encode a column: store first value, then differences.
    pub fn delta_encode(column: &[GoldilocksField]) -> Vec<GoldilocksField> {
        if column.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(column.len());
        result.push(column[0]);
        for i in 1..column.len() {
            result.push(column[i].sub_elem(column[i - 1]));
        }
        result
    }

    /// Decode a delta-encoded column.
    pub fn delta_decode(encoded: &[GoldilocksField]) -> Vec<GoldilocksField> {
        if encoded.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(encoded.len());
        result.push(encoded[0]);
        for i in 1..encoded.len() {
            result.push(result[i - 1].add_elem(encoded[i]));
        }
        result
    }

    /// Estimate the compressed size in bytes without actually compressing.
    pub fn estimate_compressed_size(trace: &ExecutionTrace) -> usize {
        let mut total = 0usize;
        for c in 0..trace.width {
            let col = trace.get_column(c);
            let first = col[0];
            let is_constant = col.iter().all(|&v| v == first);
            if is_constant {
                total += 8; // just one value
                continue;
            }
            let rle = Self::run_length_encode(&col);
            let rle_size = rle.len() * 16;
            let delta_size = col.len() * 8;
            let raw_size = col.len() * 8;
            total += rle_size.min(delta_size).min(raw_size);
        }
        total
    }

    /// Compute the compression ratio (original_size / compressed_size).
    pub fn compression_ratio(trace: &ExecutionTrace) -> f64 {
        let original = (trace.length * trace.width * 8) as f64;
        let compressed = Self::estimate_compressed_size(trace) as f64;
        if compressed == 0.0 {
            return 1.0;
        }
        original / compressed
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceInterpolator
// ═══════════════════════════════════════════════════════════════

/// Polynomial operations on trace columns.
pub struct TraceInterpolator;

impl TraceInterpolator {
    /// Convert a column to polynomial coefficient form via INTT.
    pub fn column_as_polynomial(trace: &ExecutionTrace, col: usize) -> Vec<GoldilocksField> {
        assert!(col < trace.width, "column_as_polynomial: column out of bounds");
        assert!(trace.length.is_power_of_two(), "column_as_polynomial: requires power-of-two height");
        let mut evals = trace.get_column(col);
        intt(&mut evals);
        evals
    }

    /// Convert polynomial coefficients back to evaluations via NTT.
    pub fn polynomial_to_column(coeffs: &[GoldilocksField], num_rows: usize) -> Vec<GoldilocksField> {
        assert!(num_rows.is_power_of_two(), "polynomial_to_column: num_rows must be power of 2");
        let mut padded = vec![GoldilocksField::ZERO; num_rows];
        let copy_len = coeffs.len().min(num_rows);
        padded[..copy_len].copy_from_slice(&coeffs[..copy_len]);
        ntt(&mut padded);
        padded
    }

    /// Evaluate a column's polynomial at an arbitrary point.
    pub fn evaluate_column_at_point(
        trace: &ExecutionTrace, col: usize, x: GoldilocksField,
    ) -> GoldilocksField {
        let coeffs = Self::column_as_polynomial(trace, col);
        GoldilocksField::eval_poly(&coeffs, x)
    }

    /// Extrapolate a column by evaluating its polynomial at additional points beyond
    /// the original domain.
    pub fn extrapolate_column(
        trace: &ExecutionTrace, col: usize, extra_rows: usize,
    ) -> Vec<GoldilocksField> {
        let coeffs = Self::column_as_polynomial(trace, col);
        let omega = GoldilocksField::root_of_unity(trace.length);
        let mut result = Vec::with_capacity(trace.length + extra_rows);
        // Original evaluations
        for r in 0..trace.length {
            result.push(trace.rows[r][col]);
        }
        // Extra evaluations at omega^(length), omega^(length+1), ...
        for i in 0..extra_rows {
            let x = omega.pow((trace.length + i) as u64);
            result.push(GoldilocksField::eval_poly(&coeffs, x));
        }
        result
    }

    /// Find the degree of the polynomial interpolating a column.
    pub fn column_degree(trace: &ExecutionTrace, col: usize) -> usize {
        let coeffs = Self::column_as_polynomial(trace, col);
        for i in (0..coeffs.len()).rev() {
            if !coeffs[i].is_zero() {
                return i;
            }
        }
        0
    }

    /// Batch interpolate multiple columns at once.
    pub fn batch_interpolate(trace: &ExecutionTrace, cols: &[usize]) -> Vec<Vec<GoldilocksField>> {
        cols.iter()
            .map(|&c| Self::column_as_polynomial(trace, c))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// CellDifference
// ═══════════════════════════════════════════════════════════════

/// A single cell where two traces differ.
#[derive(Debug, Clone)]
pub struct CellDifference {
    pub row: usize,
    pub col: usize,
    pub value_a: GoldilocksField,
    pub value_b: GoldilocksField,
}

// ═══════════════════════════════════════════════════════════════
// TraceDiff
// ═══════════════════════════════════════════════════════════════

/// The result of diffing two traces.
#[derive(Debug, Clone)]
pub struct TraceDiff {
    pub equal: bool,
    pub num_differences: usize,
    pub differences: Vec<CellDifference>,
}

impl TraceDiff {
    /// Produce a human-readable summary of the diff.
    pub fn summary(&self) -> String {
        if self.equal {
            "Traces are identical".to_string()
        } else {
            let mut out = format!("Traces differ in {} cells:\n", self.num_differences);
            let show = self.differences.len().min(20);
            for d in &self.differences[..show] {
                out.push_str(&format!(
                    "  [row={}, col={}] {} vs {}\n",
                    d.row, d.col,
                    d.value_a.to_canonical(),
                    d.value_b.to_canonical()
                ));
            }
            if self.differences.len() > 20 {
                out.push_str(&format!("  ... and {} more\n", self.differences.len() - 20));
            }
            out
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceDiffer
// ═══════════════════════════════════════════════════════════════

/// Compare two traces cell-by-cell.
pub struct TraceDiffer;

impl TraceDiffer {
    /// Compute the full diff between two traces.
    pub fn diff(a: &ExecutionTrace, b: &ExecutionTrace) -> TraceDiff {
        if a.length != b.length || a.width != b.width {
            // Different dimensions: compute differences up to the common area
            let rows = a.length.min(b.length);
            let cols = a.width.min(b.width);
            let mut differences = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    if a.rows[r][c] != b.rows[r][c] {
                        differences.push(CellDifference {
                            row: r,
                            col: c,
                            value_a: a.rows[r][c],
                            value_b: b.rows[r][c],
                        });
                    }
                }
            }
            // Count extra cells as differences too
            let extra = (a.length.max(b.length) - rows) * a.width.max(b.width)
                + rows * (a.width.max(b.width) - cols);
            let total = differences.len() + extra;
            return TraceDiff {
                equal: false,
                num_differences: total,
                differences,
            };
        }

        let mut differences = Vec::new();
        for r in 0..a.length {
            for c in 0..a.width {
                if a.rows[r][c] != b.rows[r][c] {
                    differences.push(CellDifference {
                        row: r,
                        col: c,
                        value_a: a.rows[r][c],
                        value_b: b.rows[r][c],
                    });
                }
            }
        }
        let equal = differences.is_empty();
        let num_differences = differences.len();
        TraceDiff { equal, num_differences, differences }
    }

    /// Check if two traces are identical.
    pub fn is_identical(a: &ExecutionTrace, b: &ExecutionTrace) -> bool {
        if a.length != b.length || a.width != b.width {
            return false;
        }
        for r in 0..a.length {
            for c in 0..a.width {
                if a.rows[r][c] != b.rows[r][c] {
                    return false;
                }
            }
        }
        true
    }

    /// Find the cell with the maximum absolute difference.
    pub fn max_difference(
        a: &ExecutionTrace, b: &ExecutionTrace,
    ) -> Option<(usize, usize, GoldilocksField)> {
        if a.length != b.length || a.width != b.width {
            return None;
        }
        let mut max_row = 0;
        let mut max_col = 0;
        let mut max_diff = GoldilocksField::ZERO;
        let mut max_diff_val: u64 = 0;
        let mut found = false;

        for r in 0..a.length {
            for c in 0..a.width {
                let diff = a.rows[r][c].sub_elem(b.rows[r][c]);
                let diff_val = diff.to_canonical();
                // Use min(diff_val, modulus - diff_val) for "absolute" distance
                let abs_diff = diff_val.min(GoldilocksField::MODULUS - diff_val);
                if abs_diff > max_diff_val {
                    max_diff_val = abs_diff;
                    max_diff = diff;
                    max_row = r;
                    max_col = c;
                    found = true;
                }
            }
        }
        if found && max_diff_val > 0 {
            Some((max_row, max_col, max_diff))
        } else {
            None
        }
    }

    /// Diff a single column between two traces.
    pub fn diff_column(
        a: &ExecutionTrace, b: &ExecutionTrace, col: usize,
    ) -> Vec<(usize, GoldilocksField, GoldilocksField)> {
        let rows = a.length.min(b.length);
        let mut diffs = Vec::new();
        for r in 0..rows {
            if a.rows[r][col] != b.rows[r][col] {
                diffs.push((r, a.rows[r][col], b.rows[r][col]));
            }
        }
        diffs
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceBuilder
// ═══════════════════════════════════════════════════════════════

/// Builder pattern for constructing execution traces incrementally.
pub struct TraceBuilder {
    num_cols: usize,
    rows: Vec<Vec<GoldilocksField>>,
    column_names: Option<Vec<String>>,
}

impl TraceBuilder {
    /// Create a new builder for a trace with `num_cols` columns.
    pub fn new(num_cols: usize) -> Self {
        TraceBuilder {
            num_cols,
            rows: Vec::new(),
            column_names: None,
        }
    }

    /// Add a row to the trace being built.
    pub fn add_row(&mut self, row: Vec<GoldilocksField>) -> &mut Self {
        assert_eq!(
            row.len(), self.num_cols,
            "TraceBuilder::add_row: row length {} != expected {}",
            row.len(), self.num_cols
        );
        self.rows.push(row);
        self
    }

    /// Set human-readable column names.
    pub fn set_column_names(&mut self, names: Vec<String>) -> &mut Self {
        assert_eq!(
            names.len(), self.num_cols,
            "TraceBuilder::set_column_names: names length {} != expected {}",
            names.len(), self.num_cols
        );
        self.column_names = Some(names);
        self
    }

    /// Create a builder pre-populated from a state machine execution.
    /// Each state vector becomes a row; inputs are placed in an extra column.
    pub fn from_state_machine(
        states: &[Vec<GoldilocksField>], inputs: &[GoldilocksField],
    ) -> Self {
        assert!(!states.is_empty(), "from_state_machine: no states");
        let state_width = states[0].len();
        let num_cols = state_width + 1; // states + input column
        let mut builder = TraceBuilder::new(num_cols);
        for (i, state) in states.iter().enumerate() {
            assert_eq!(state.len(), state_width);
            let mut row = state.clone();
            let input = if i < inputs.len() {
                inputs[i]
            } else {
                GoldilocksField::ZERO
            };
            row.push(input);
            builder.add_row(row);
        }
        let mut names: Vec<String> = (0..state_width).map(|i| format!("state_{}", i)).collect();
        names.push("input".to_string());
        builder.set_column_names(names);
        builder
    }

    /// Build the trace. Panics if no rows were added.
    pub fn build(self) -> ExecutionTrace {
        assert!(!self.rows.is_empty(), "TraceBuilder::build: no rows added");
        let column_names = self.column_names.unwrap_or_else(|| {
            (0..self.num_cols).map(|i| format!("col_{}", i)).collect()
        });
        ExecutionTrace {
            length: self.rows.len(),
            width: self.num_cols,
            rows: self.rows,
            column_names,
        }
    }

    /// Build the trace and automatically pad to the next power of two.
    pub fn build_padded(self) -> ExecutionTrace {
        let mut trace = self.build();
        trace.pad_to_power_of_two();
        trace
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceTransformer
// ═══════════════════════════════════════════════════════════════

/// Transform traces by reordering or modifying their structure.
pub struct TraceTransformer;

impl TraceTransformer {
    /// Reorder columns according to a permutation.
    pub fn permute_columns(trace: &ExecutionTrace, permutation: &[usize]) -> ExecutionTrace {
        assert_eq!(permutation.len(), trace.width, "permute_columns: permutation length mismatch");
        for &p in permutation {
            assert!(p < trace.width, "permute_columns: index {} out of bounds", p);
        }
        let mut rows = Vec::with_capacity(trace.length);
        for r in 0..trace.length {
            let row: Vec<GoldilocksField> = permutation.iter().map(|&c| trace.rows[r][c]).collect();
            rows.push(row);
        }
        let column_names: Vec<String> = permutation.iter()
            .map(|&c| trace.column_names[c].clone())
            .collect();
        ExecutionTrace {
            rows,
            length: trace.length,
            width: trace.width,
            column_names,
        }
    }

    /// Reorder rows according to a permutation.
    pub fn permute_rows(trace: &ExecutionTrace, permutation: &[usize]) -> ExecutionTrace {
        assert_eq!(permutation.len(), trace.length, "permute_rows: permutation length mismatch");
        for &p in permutation {
            assert!(p < trace.length, "permute_rows: index {} out of bounds", p);
        }
        let rows: Vec<Vec<GoldilocksField>> = permutation.iter()
            .map(|&r| trace.rows[r].clone())
            .collect();
        ExecutionTrace {
            rows,
            length: trace.length,
            width: trace.width,
            column_names: trace.column_names.clone(),
        }
    }

    /// Interleave two traces by alternating rows: a[0], b[0], a[1], b[1], ...
    pub fn interleave_traces(a: &ExecutionTrace, b: &ExecutionTrace) -> ExecutionTrace {
        assert_eq!(a.width, b.width, "interleave_traces: width mismatch");
        assert_eq!(a.length, b.length, "interleave_traces: height mismatch");
        let new_len = a.length + b.length;
        let mut rows = Vec::with_capacity(new_len);
        for i in 0..a.length {
            rows.push(a.rows[i].clone());
            rows.push(b.rows[i].clone());
        }
        ExecutionTrace {
            rows,
            length: new_len,
            width: a.width,
            column_names: a.column_names.clone(),
        }
    }

    /// Reverse the order of rows.
    pub fn reverse_rows(trace: &ExecutionTrace) -> ExecutionTrace {
        let mut rows = trace.rows.clone();
        rows.reverse();
        ExecutionTrace {
            rows,
            length: trace.length,
            width: trace.width,
            column_names: trace.column_names.clone(),
        }
    }

    /// Cyclically rotate columns to the right by `shift`.
    pub fn rotate_columns(trace: &ExecutionTrace, shift: usize) -> ExecutionTrace {
        if trace.width == 0 {
            return trace.clone();
        }
        let effective_shift = shift % trace.width;
        if effective_shift == 0 {
            return trace.clone();
        }
        let mut perm: Vec<usize> = Vec::with_capacity(trace.width);
        for i in 0..trace.width {
            perm.push((i + trace.width - effective_shift) % trace.width);
        }
        Self::permute_columns(trace, &perm)
    }

    /// Apply a linear transformation matrix to each row.
    /// `matrix` is width×width, and each row of the trace is multiplied by it.
    pub fn apply_linear_transform(
        trace: &ExecutionTrace, matrix: &Vec<Vec<GoldilocksField>>,
    ) -> ExecutionTrace {
        assert_eq!(matrix.len(), trace.width, "apply_linear_transform: matrix row count mismatch");
        for row in matrix {
            assert_eq!(row.len(), trace.width, "apply_linear_transform: matrix col count mismatch");
        }
        let mut new_rows = Vec::with_capacity(trace.length);
        for r in 0..trace.length {
            let mut new_row = vec![GoldilocksField::ZERO; trace.width];
            for c in 0..trace.width {
                let mut sum = GoldilocksField::ZERO;
                for k in 0..trace.width {
                    sum = sum.add_elem(trace.rows[r][k].mul_elem(matrix[c][k]));
                }
                new_row[c] = sum;
            }
            new_rows.push(new_row);
        }
        ExecutionTrace {
            rows: new_rows,
            length: trace.length,
            width: trace.width,
            column_names: trace.column_names.clone(),
        }
    }

    /// Filter rows by a predicate. Predicate receives (row_index, row_data).
    pub fn filter_rows(
        trace: &ExecutionTrace,
        predicate: impl Fn(usize, &[GoldilocksField]) -> bool,
    ) -> ExecutionTrace {
        let mut rows = Vec::new();
        for r in 0..trace.length {
            if predicate(r, &trace.rows[r]) {
                rows.push(trace.rows[r].clone());
            }
        }
        let new_len = rows.len();
        ExecutionTrace {
            rows,
            length: new_len,
            width: trace.width,
            column_names: trace.column_names.clone(),
        }
    }

    /// Map a function over every element of a single column.
    pub fn map_column(
        trace: &ExecutionTrace,
        col: usize,
        f: impl Fn(GoldilocksField) -> GoldilocksField,
    ) -> ExecutionTrace {
        assert!(col < trace.width, "map_column: column out of bounds");
        let mut new_rows = trace.rows.clone();
        for r in 0..trace.length {
            new_rows[r][col] = f(new_rows[r][col]);
        }
        ExecutionTrace {
            rows: new_rows,
            length: trace.length,
            width: trace.width,
            column_names: trace.column_names.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ValidationError
// ═══════════════════════════════════════════════════════════════

/// An error found during trace validation.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub error_type: String,
    pub row: Option<usize>,
    pub col: Option<usize>,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row_str = self.row.map(|r| format!(" row={}", r)).unwrap_or_default();
        let col_str = self.col.map(|c| format!(" col={}", c)).unwrap_or_default();
        write!(f, "[{}{}{}] {}", self.error_type, row_str, col_str, self.message)
    }
}

// ═══════════════════════════════════════════════════════════════
// TraceValidator
// ═══════════════════════════════════════════════════════════════

/// Comprehensive validation of trace properties.
pub struct TraceValidator;

impl TraceValidator {
    /// Run all validations on a trace.
    pub fn validate_full(trace: &ExecutionTrace) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Check dimensions
        if trace.length == 0 || trace.width == 0 {
            errors.push(ValidationError {
                error_type: "EmptyTrace".into(),
                row: None,
                col: None,
                message: "trace has zero rows or zero columns".into(),
            });
            return errors;
        }

        // Check row lengths
        for (r, row) in trace.rows.iter().enumerate() {
            if row.len() != trace.width {
                errors.push(ValidationError {
                    error_type: "DimensionMismatch".into(),
                    row: Some(r),
                    col: None,
                    message: format!("row {} has length {} but expected {}", r, row.len(), trace.width),
                });
            }
        }

        // Check height matches
        if trace.rows.len() != trace.length {
            errors.push(ValidationError {
                error_type: "DimensionMismatch".into(),
                row: None,
                col: None,
                message: format!("rows.len()={} but length={}", trace.rows.len(), trace.length),
            });
        }

        // Check column names length
        if trace.column_names.len() != trace.width {
            errors.push(ValidationError {
                error_type: "DimensionMismatch".into(),
                row: None,
                col: None,
                message: format!(
                    "column_names length {} != width {}",
                    trace.column_names.len(), trace.width
                ),
            });
        }

        // Check power of two
        if let Err(e) = Self::validate_power_of_two(trace) {
            errors.push(e);
        }

        errors
    }

    /// Validate that trace height is a power of two.
    pub fn validate_power_of_two(trace: &ExecutionTrace) -> Result<(), ValidationError> {
        if trace.length == 0 || !trace.length.is_power_of_two() {
            Err(ValidationError {
                error_type: "NotPowerOfTwo".into(),
                row: None,
                col: None,
                message: format!("trace height {} is not a power of two", trace.length),
            })
        } else {
            Ok(())
        }
    }

    /// Validate that all values in a column are within [min, max].
    pub fn validate_column_range(
        trace: &ExecutionTrace, col: usize,
        min: GoldilocksField, max: GoldilocksField,
    ) -> Result<(), ValidationError> {
        assert!(col < trace.width);
        let min_val = min.to_canonical();
        let max_val = max.to_canonical();
        for r in 0..trace.length {
            let v = trace.rows[r][col].to_canonical();
            if v < min_val || v > max_val {
                return Err(ValidationError {
                    error_type: "ValueOutOfRange".into(),
                    row: Some(r),
                    col: Some(col),
                    message: format!("value {} not in [{}, {}]", v, min_val, max_val),
                });
            }
        }
        Ok(())
    }

    /// Validate that a column contains only 0 or 1.
    pub fn validate_boolean_column(
        trace: &ExecutionTrace, col: usize,
    ) -> Result<(), ValidationError> {
        assert!(col < trace.width);
        for r in 0..trace.length {
            let v = trace.rows[r][col].to_canonical();
            if v != 0 && v != 1 {
                return Err(ValidationError {
                    error_type: "NotBoolean".into(),
                    row: Some(r),
                    col: Some(col),
                    message: format!("value {} is not boolean (0 or 1)", v),
                });
            }
        }
        Ok(())
    }

    /// Validate that a column is a permutation of 0..N-1.
    pub fn validate_permutation_column(
        trace: &ExecutionTrace, col: usize,
    ) -> Result<(), ValidationError> {
        assert!(col < trace.width);
        let n = trace.length;
        let mut seen = vec![false; n];
        for r in 0..n {
            let v = trace.rows[r][col].to_canonical() as usize;
            if v >= n {
                return Err(ValidationError {
                    error_type: "InvalidPermutation".into(),
                    row: Some(r),
                    col: Some(col),
                    message: format!("value {} >= trace length {}", v, n),
                });
            }
            if seen[v] {
                return Err(ValidationError {
                    error_type: "InvalidPermutation".into(),
                    row: Some(r),
                    col: Some(col),
                    message: format!("value {} appears more than once", v),
                });
            }
            seen[v] = true;
        }
        Ok(())
    }

    /// Validate that a column is monotonically non-decreasing.
    pub fn validate_monotonic_column(
        trace: &ExecutionTrace, col: usize,
    ) -> Result<(), ValidationError> {
        assert!(col < trace.width);
        for r in 1..trace.length {
            let prev = trace.rows[r - 1][col].to_canonical();
            let curr = trace.rows[r][col].to_canonical();
            if curr < prev {
                return Err(ValidationError {
                    error_type: "NotMonotonic".into(),
                    row: Some(r),
                    col: Some(col),
                    message: format!("value {} < previous value {} at row {}", curr, prev, r),
                });
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn gf(v: u64) -> GoldilocksField { GoldilocksField::new(v) }

    fn make_small_trace() -> ExecutionTrace {
        ExecutionTrace::from_rows(vec![
            vec![gf(1), gf(2), gf(3)],
            vec![gf(4), gf(5), gf(6)],
            vec![gf(7), gf(8), gf(9)],
            vec![gf(10), gf(11), gf(12)],
        ])
    }

    // ── creation ─────────────────────────────────────────────

    #[test]
    fn test_new_trace() {
        let t = ExecutionTrace::new(8, 3);
        assert_eq!(t.width(), 3);
        assert_eq!(t.height(), 8);
        for r in 0..8 { for c in 0..3 { assert_eq!(t.get(r, c), GoldilocksField::ZERO); } }
    }

    #[test]
    fn test_zeros_compat() {
        let t = ExecutionTrace::zeros(3, 8);
        assert_eq!(t.width, 3);
        assert_eq!(t.length, 8);
    }

    #[test]
    fn test_from_rows() {
        let t = make_small_trace();
        assert_eq!(t.width(), 3);
        assert_eq!(t.height(), 4);
        assert_eq!(t.get(0, 0), gf(1));
        assert_eq!(t.get(3, 2), gf(12));
    }

    #[test]
    fn test_from_columns() {
        let cols = vec![
            vec![gf(1), gf(4), gf(7), gf(10)],
            vec![gf(2), gf(5), gf(8), gf(11)],
            vec![gf(3), gf(6), gf(9), gf(12)],
        ];
        let t = ExecutionTrace::from_columns(cols);
        assert_eq!(t.get(0, 0), gf(1));
        assert_eq!(t.get(1, 1), gf(5));
        assert_eq!(t.get(3, 2), gf(12));
    }

    // ── get / set ────────────────────────────────────────────

    #[test]
    fn test_get_set() {
        let mut t = ExecutionTrace::new(4, 2);
        t.set(1, 0, gf(42));
        assert_eq!(t.get(1, 0), gf(42));
        assert_eq!(t.get(0, 0), gf(0));
    }

    #[test]
    fn test_get_row() {
        let t = make_small_trace();
        assert_eq!(t.get_row(2), &vec![gf(7), gf(8), gf(9)]);
    }

    #[test]
    fn test_get_column() {
        let t = make_small_trace();
        assert_eq!(t.get_column(1), vec![gf(2), gf(5), gf(8), gf(11)]);
    }

    #[test]
    fn test_set_row() {
        let mut t = make_small_trace();
        t.set_row(0, vec![gf(100), gf(200), gf(300)]);
        assert_eq!(t.get(0, 0), gf(100));
        assert_eq!(t.get(0, 2), gf(300));
    }

    #[test]
    fn test_set_column() {
        let mut t = make_small_trace();
        t.set_column(2, vec![gf(30), gf(60), gf(90), gf(120)]);
        assert_eq!(t.get(0, 2), gf(30));
        assert_eq!(t.get(3, 2), gf(120));
    }

    #[test]
    fn test_set_column_name() {
        let mut t = make_small_trace();
        t.set_column_name(0, "state");
        assert_eq!(t.column_names()[0], "state");
    }

    // ── padding ──────────────────────────────────────────────

    #[test]
    fn test_pad_to_power_of_two_noop() {
        let mut t = make_small_trace();
        t.pad_to_power_of_two();
        assert_eq!(t.height(), 4);
    }

    #[test]
    fn test_pad_to_power_of_two() {
        let mut t = ExecutionTrace::from_rows(vec![
            vec![gf(1), gf(2)], vec![gf(3), gf(4)], vec![gf(5), gf(6)],
        ]);
        t.pad_to_power_of_two();
        assert_eq!(t.height(), 4);
        assert_eq!(t.get(3, 0), gf(5));
        assert_eq!(t.get(3, 1), gf(6));
    }

    #[test]
    fn test_pad_with_zeros() {
        let mut t = make_small_trace();
        t.pad_with_zeros(8);
        assert_eq!(t.height(), 8);
        assert_eq!(t.get(7, 0), gf(0));
    }

    #[test]
    fn test_pad_with_value() {
        let mut t = make_small_trace();
        t.pad_with_value(8, gf(99));
        assert_eq!(t.height(), 8);
        assert_eq!(t.get(5, 1), gf(99));
    }

    #[test]
    fn test_is_power_of_two_height() {
        assert!(make_small_trace().is_power_of_two_height());
        assert!(!ExecutionTrace::from_rows(vec![vec![gf(1)], vec![gf(2)], vec![gf(3)]]).is_power_of_two_height());
    }

    // ── commitment ───────────────────────────────────────────

    #[test]
    fn test_commit_and_query() {
        let t = make_small_trace();
        let commitment = t.commit();
        assert_ne!(commitment.root_hash, [0u8; 32]);
        for row_idx in 0..4 {
            let (row_data, proof) = t.query_row(&commitment, row_idx);
            assert_eq!(row_data, *t.get_row(row_idx));
            assert!(ExecutionTrace::verify_row_query(&commitment.root_hash, row_idx, &row_data, &proof));
        }
    }

    #[test]
    fn test_commitment_tamper_detection() {
        let t = make_small_trace();
        let commitment = t.commit();
        let (_row_data, proof) = t.query_row(&commitment, 0);
        let bad_data = vec![gf(999), gf(999), gf(999)];
        assert!(!ExecutionTrace::verify_row_query(&commitment.root_hash, 0, &bad_data, &proof));
    }

    // ── validation ───────────────────────────────────────────

    #[test]
    fn test_validate_dimensions() {
        assert!(make_small_trace().validate_dimensions().is_ok());
    }

    #[test]
    fn test_validate_boundary() {
        let t = make_small_trace();
        assert!(t.validate_boundary(0, 0, gf(1)));
        assert!(!t.validate_boundary(0, 0, gf(999)));
        assert!(!t.validate_boundary(0, 100, gf(1)));
    }

    #[test]
    fn test_validate_transition() {
        let t = make_small_trace();
        assert!(t.validate_transition(|cur, nxt| nxt[0] == cur[0].add_elem(gf(3))));
        assert!(!t.validate_transition(|cur, nxt| nxt[0] == cur[0].add_elem(gf(1))));
    }

    #[test]
    fn test_check_all_zeros_column() {
        let mut t = ExecutionTrace::new(4, 2);
        assert!(t.check_all_zeros_column(0));
        t.set(2, 0, gf(1));
        assert!(!t.check_all_zeros_column(0));
        assert!(t.check_all_zeros_column(1));
    }

    #[test]
    fn test_find_first_violation() {
        let t = make_small_trace();
        assert_eq!(t.find_first_violation(|cur, nxt| nxt[0] == cur[0].add_elem(gf(3))), None);
        assert_eq!(t.find_first_violation(|cur, nxt| nxt[0] == cur[0]), Some(0));
    }

    // ── LDE ──────────────────────────────────────────────────

    #[test]
    fn test_low_degree_extend() {
        let t = make_small_trace();
        let lde = t.low_degree_extend(2);
        assert_eq!(lde.height(), 8);
        assert_eq!(lde.width(), 3);
    }

    #[test]
    fn test_extend_column_preserves_polynomial() {
        let coeffs = vec![gf(1), gf(2), gf(3), gf(0)];
        let mut evals = coeffs.clone();
        ntt(&mut evals);
        let coset_shift = gf(7);
        let extended = extend_column(&evals, 2, coset_shift);
        assert_eq!(extended.len(), 8);
        let omega8 = GoldilocksField::root_of_unity(8);
        let mut point = coset_shift;
        for i in 0..8 {
            let expected = GoldilocksField::eval_poly(&coeffs, point);
            assert_eq!(extended[i], expected, "LDE mismatch at index {}", i);
            point = point.mul_elem(omega8);
        }
    }

    // ── sub-operations ───────────────────────────────────────

    #[test]
    fn test_sub_trace() {
        let t = make_small_trace();
        let st = t.sub_trace(&[0, 2]);
        assert_eq!(st.width(), 2);
        assert_eq!(st.get(0, 0), gf(1));
        assert_eq!(st.get(0, 1), gf(3));
    }

    #[test]
    fn test_append_column() {
        let mut t = make_small_trace();
        t.append_column(vec![gf(100), gf(200), gf(300), gf(400)], "extra");
        assert_eq!(t.width(), 4);
        assert_eq!(t.get(0, 3), gf(100));
        assert_eq!(t.column_names()[3], "extra");
    }

    #[test]
    fn test_append_columns() {
        let mut t = make_small_trace();
        t.append_columns(
            vec![vec![gf(10), gf(20), gf(30), gf(40)], vec![gf(11), gf(21), gf(31), gf(41)]],
            &["c1".into(), "c2".into()],
        );
        assert_eq!(t.width(), 5);
        assert_eq!(t.get(1, 3), gf(20));
        assert_eq!(t.get(1, 4), gf(21));
    }

    #[test]
    fn test_merge_traces() {
        let t1 = make_small_trace();
        let t2 = ExecutionTrace::new(4, 2);
        let merged = ExecutionTrace::merge_traces(&[&t1, &t2]);
        assert_eq!(merged.width(), 5);
        assert_eq!(merged.get(0, 0), gf(1));
        assert_eq!(merged.get(0, 3), gf(0));
    }

    #[test]
    fn test_stack_traces() {
        let t1 = make_small_trace();
        let t2 = make_small_trace();
        let stacked = ExecutionTrace::stack_traces(&[&t1, &t2]);
        assert_eq!(stacked.height(), 8);
        assert_eq!(stacked.get(4, 0), gf(1));
    }

    #[test]
    fn test_window_at() {
        let t = make_small_trace();
        let w = t.window_at(1);
        assert_eq!(w.get_current(0), gf(4));
        assert_eq!(w.get_next(0), gf(7));
        let w_last = t.window_at(3);
        assert_eq!(w_last.get_current(0), gf(10));
        assert_eq!(w_last.get_next(0), gf(1));
    }

    // ── statistics ───────────────────────────────────────────

    #[test]
    fn test_statistics() {
        let t = make_small_trace();
        let stats = t.statistics();
        assert_eq!(stats.num_rows, 4);
        assert_eq!(stats.num_cols, 3);
        assert_eq!(stats.column_stats[0].min_val, 1);
        assert_eq!(stats.column_stats[0].max_val, 10);
        assert_eq!(stats.column_stats[0].zero_count, 0);
        assert_eq!(stats.column_stats[0].distinct_count, 4);
    }

    #[test]
    fn test_display_ascii() {
        let s = make_small_trace().display_ascii(10, 10);
        assert!(s.contains("ExecutionTrace"));
        assert!(s.contains("4×3"));
    }

    #[test]
    fn test_display_column_summary() {
        assert!(make_small_trace().display_column_summary().contains("col_0"));
    }

    #[test]
    fn test_to_csv() {
        let csv = make_small_trace().to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].contains("col_0"));
    }

    // ── partial trace / debug ────────────────────────────────

    #[test]
    fn test_partial_trace() {
        let pt = make_small_trace().partial_trace(1, 3);
        assert_eq!(pt.height(), 2);
        assert_eq!(pt.get(0, 0), gf(4));
    }

    #[test]
    fn test_debug_row() {
        let s = make_small_trace().debug_row(0);
        assert!(s.contains("Row 0"));
        assert!(s.contains("col_0=1"));
    }

    #[test]
    fn test_debug_window() {
        let s = make_small_trace().debug_window(0);
        assert!(s.contains("Window at row 0"));
        assert!(s.contains("current(0)"));
    }

    // ── serialization ────────────────────────────────────────

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let t = make_small_trace();
        let bytes = t.serialize_to_bytes();
        let t2 = ExecutionTrace::deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(t2.height(), t.height());
        assert_eq!(t2.width(), t.width());
        for r in 0..t.height() { for c in 0..t.width() { assert_eq!(t2.get(r, c), t.get(r, c)); } }
        assert_eq!(t2.column_names(), t.column_names());
    }

    #[test]
    fn test_deserialize_invalid_magic() {
        assert!(ExecutionTrace::deserialize_from_bytes(b"BADXxxxxxxxx").is_err());
    }

    #[test]
    fn test_deserialize_truncated() {
        assert!(ExecutionTrace::deserialize_from_bytes(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_hash_trace_deterministic() {
        let t = make_small_trace();
        let h1 = t.hash_trace();
        let h2 = t.hash_trace();
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 32]);
    }

    #[test]
    fn test_hash_trace_different_data() {
        let t1 = make_small_trace();
        let mut t2 = make_small_trace();
        t2.set(0, 0, gf(999));
        assert_ne!(t1.hash_trace(), t2.hash_trace());
    }

    // ── polynomial operations ────────────────────────────────

    #[test]
    fn test_column_to_polynomial_roundtrip() {
        let t = make_small_trace();
        let coeffs = t.column_to_polynomial(0);
        let evals = ExecutionTrace::polynomial_to_column(&coeffs, 4);
        let col = t.get_column(0);
        for i in 0..4 { assert_eq!(evals[i], col[i]); }
    }

    #[test]
    fn test_evaluate_polynomial_column() {
        let t = make_small_trace();
        let x = gf(42);
        let coeffs = t.column_to_polynomial(0);
        assert_eq!(t.evaluate_polynomial_column(0, x), GoldilocksField::eval_poly(&coeffs, x));
    }

    #[test]
    fn test_polynomial_to_column() {
        let coeffs = vec![gf(1), gf(2), gf(3), gf(0)];
        let evals = ExecutionTrace::polynomial_to_column(&coeffs, 4);
        let omega = GoldilocksField::root_of_unity(4);
        let mut w = GoldilocksField::ONE;
        for i in 0..4 {
            assert_eq!(evals[i], GoldilocksField::eval_poly(&coeffs, w));
            w = w.mul_elem(omega);
        }
    }

    // ── windowing / iteration ────────────────────────────────

    #[test]
    fn test_iter_rows() {
        let t = make_small_trace();
        let rows: Vec<_> = t.iter_rows().collect();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0], t.get_row(0));
    }

    #[test]
    fn test_iter_windows() {
        let t = make_small_trace();
        let windows: Vec<_> = t.iter_windows().collect();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0].get_current(0), gf(1));
        assert_eq!(windows[3].get_next(0), gf(1));
    }

    // ── additional helpers ───────────────────────────────────

    #[test]
    fn test_fibonacci_trace() {
        let t = ExecutionTrace::fibonacci_trace(gf(1), gf(1), 8);
        assert_eq!(t.height(), 8);
        assert_eq!(t.get(0, 0), gf(1));
        assert_eq!(t.get(0, 1), gf(1));
        assert_eq!(t.get(1, 1), gf(2));
        assert_eq!(t.get(3, 1), gf(5));
        assert!(t.validate_transition(|cur, nxt| nxt[0] == cur[1] && nxt[1] == cur[0].add_elem(cur[1])));
    }

    #[test]
    fn test_counter_trace() {
        let t = ExecutionTrace::counter_trace(4);
        for i in 0..4 { assert_eq!(t.get(i, 0), gf(i as u64)); }
    }

    #[test]
    fn test_repeated_squaring_trace() {
        let t = ExecutionTrace::repeated_squaring_trace(gf(2), 4);
        assert_eq!(t.get(0, 0), gf(2));
        assert_eq!(t.get(1, 0), gf(4));
        assert_eq!(t.get(2, 0), gf(16));
        assert_eq!(t.get(3, 0), gf(256));
    }

    #[test]
    fn test_transpose() {
        let tt = make_small_trace().transpose();
        assert_eq!(tt.height(), 3);
        assert_eq!(tt.width(), 4);
        assert_eq!(tt.get(0, 0), gf(1));
        assert_eq!(tt.get(2, 3), gf(12));
    }

    #[test]
    fn test_scale() {
        let scaled = make_small_trace().scale(gf(2));
        assert_eq!(scaled.get(0, 0), gf(2));
        assert_eq!(scaled.get(0, 1), gf(4));
    }

    #[test]
    fn test_add_trace() {
        let t = make_small_trace();
        let sum = t.add_trace(&make_small_trace());
        assert_eq!(sum.get(0, 0), gf(2));
        assert_eq!(sum.get(1, 1), gf(10));
    }

    #[test]
    fn test_equals() {
        assert!(make_small_trace().equals(&make_small_trace()));
        let mut t3 = make_small_trace();
        t3.set(0, 0, gf(999));
        assert!(!make_small_trace().equals(&t3));
    }

    #[test]
    fn test_hamming_distance() {
        let t1 = make_small_trace();
        let mut t2 = make_small_trace();
        assert_eq!(t1.hamming_distance(&t2), 0);
        t2.set(0, 0, gf(999));
        t2.set(1, 1, gf(999));
        assert_eq!(t1.hamming_distance(&t2), 2);
    }

    #[test]
    fn test_linear_combination() {
        let t = make_small_trace();
        assert_eq!(t.linear_combination_at(0, &[0, 1, 2], &[gf(1), gf(2), gf(3)]), gf(14));
    }

    #[test]
    fn test_derive_column_product() {
        let t = make_small_trace();
        let prod = t.derive_column_product(0, 1);
        assert_eq!(prod[0], gf(2));
        assert_eq!(prod[1], gf(20));
    }

    #[test]
    fn test_running_product() {
        let t = ExecutionTrace::counter_trace(4);
        let rp = t.running_product_column(0, gf(10));
        assert_eq!(rp[0], gf(10));
        assert_eq!(rp[1], gf(110));
        assert_eq!(rp[2], gf(1320));
        assert_eq!(rp[3], gf(17160));
    }

    #[test]
    fn test_degree_bound() {
        let coeffs = vec![gf(5), gf(3), gf(0), gf(0)];
        let mut evals = coeffs.clone();
        ntt(&mut evals);
        let t = ExecutionTrace::from_columns(vec![evals]);
        assert!(t.check_degree_bound(0, 2));
        assert!(!t.check_degree_bound(0, 1));
        assert_eq!(t.column_degree(0), 1);
    }

    #[test]
    fn test_compose_columns() {
        let t = make_small_trace();
        let alpha = gf(3);
        let comp = t.compose_columns(alpha);
        let x = gf(123);
        assert_eq!(GoldilocksField::eval_poly(&comp, x), t.evaluate_composition(alpha, x));
    }

    #[test]
    fn test_constraint_column() {
        let t = make_small_trace();
        let col = t.compute_constraint_column(|cur, nxt| nxt[0].sub_elem(cur[0]).sub_elem(gf(3)));
        assert_eq!(col[0], gf(0));
        assert_eq!(col[1], gf(0));
        assert_eq!(col[2], gf(0));
    }

    #[test]
    fn test_vanishing_poly_evals() {
        let z = ExecutionTrace::vanishing_poly_evals(4, 8, gf(7));
        assert_eq!(z.len(), 8);
    }

    #[test]
    fn test_evaluation_domain() {
        let domain = make_small_trace().evaluation_domain();
        assert_eq!(domain.len(), 4);
        assert_eq!(domain[0], GoldilocksField::ONE);
    }

    #[test]
    fn test_wfa_step_trace() {
        let t = ExecutionTrace::wfa_step_trace(&[0,1,2,3], &[0,1,0,1], &[1,2,3,4], &[1,1,1,1]);
        assert_eq!(t.width(), 6);
        assert_eq!(t.get(0, 4), gf(1));
        assert_eq!(t.get(1, 4), gf(3));
    }

    #[test]
    fn test_commit_columns() {
        let commitments = make_small_trace().commit_columns();
        assert_eq!(commitments.len(), 3);
        for (_, root) in &commitments { assert_ne!(*root, [0u8; 32]); }
    }

    #[test]
    fn test_column_element_query_verify() {
        let t = make_small_trace();
        let commitments = t.commit_columns();
        let col0 = t.get_column(0);
        let (tree0, root0) = &commitments[0];
        let (val, proof) = ExecutionTrace::query_column_element(tree0, &col0, 2);
        assert_eq!(val, gf(7));
        assert!(ExecutionTrace::verify_column_element(root0, val, &proof));
        assert!(!ExecutionTrace::verify_column_element(root0, gf(999), &proof));
    }

    #[test]
    fn test_deep_query() {
        let t = make_small_trace();
        let z = gf(42);
        let (evals_z, evals_z_next) = t.deep_query(z);
        assert_eq!(evals_z.len(), 3);
        let c0 = t.column_to_polynomial(0);
        assert_eq!(evals_z[0], GoldilocksField::eval_poly(&c0, z));
    }

    #[test]
    fn test_map() {
        let doubled = make_small_trace().map(|v| v.add_elem(v));
        assert_eq!(doubled.get(0, 0), gf(2));
    }

    #[test]
    fn test_filled() {
        let t = ExecutionTrace::filled(4, 3, gf(7));
        for r in 0..4 { for c in 0..3 { assert_eq!(t.get(r, c), gf(7)); } }
    }

    #[test]
    fn test_identity_like() {
        let t = ExecutionTrace::identity_like(4, 3);
        assert_eq!(t.get(0, 0), gf(0));
        assert_eq!(t.get(0, 1), gf(1));
        assert_eq!(t.get(1, 0), gf(3));
    }

    #[test]
    fn test_row_hashes() {
        let hashes = make_small_trace().row_hashes();
        assert_eq!(hashes.len(), 4);
        for i in 0..4 { for j in (i+1)..4 { assert_ne!(hashes[i], hashes[j]); } }
    }

    #[test]
    fn test_find_constraint_violations() {
        let t = make_small_trace();
        let violations = t.find_constraint_violations(|cur, nxt| nxt[0].sub_elem(cur[0]).sub_elem(gf(3)));
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0], 3);
    }

    #[test]
    fn test_sub_trace_elementwise() {
        let diff = make_small_trace().sub_trace_elementwise(&make_small_trace());
        for r in 0..diff.height() { for c in 0..diff.width() { assert_eq!(diff.get(r, c), gf(0)); } }
    }

    #[test]
    fn test_coset_domain() {
        let domain = make_small_trace().coset_domain(gf(7));
        assert_eq!(domain.len(), 4);
        assert_eq!(domain[0], gf(7));
    }

    #[test]
    fn test_quotient_column() {
        let q = ExecutionTrace::quotient_column(
            &[gf(10), gf(20), gf(30), gf(40)],
            &[gf(2), gf(4), gf(5), gf(10)],
        );
        assert_eq!(q[0], gf(10).mul_elem(gf(2).inv_or_panic()));
    }

    #[test]
    fn test_verify_constraint_column() {
        let t = ExecutionTrace::counter_trace(4);
        assert!(!t.verify_constraint_column(|cur, nxt| nxt[0].sub_elem(cur[0]).sub_elem(gf(1))));
        assert!(t.verify_constraint_column(|_, _| gf(0)));
    }

    #[test]
    fn test_multi_column_running_product() {
        let rp = make_small_trace().multi_column_running_product(&[0, 1], gf(2), gf(5));
        assert_eq!(rp.len(), 4);
        assert_eq!(rp[0], gf(10));
    }

    #[test]
    fn test_display_trace() {
        assert!(format!("{}", make_small_trace()).contains("ExecutionTrace"));
    }

    #[test]
    fn test_trace_error_display() {
        assert!(format!("{}", TraceError::EmptyTrace).contains("empty"));
    }

    #[test]
    fn test_large_trace_serialize_roundtrip() {
        let mut t = ExecutionTrace::new(16, 4);
        for r in 0..16 { for c in 0..4 { t.set(r, c, gf((r * 4 + c + 1) as u64)); } }
        let t2 = ExecutionTrace::deserialize_from_bytes(&t.serialize_to_bytes()).unwrap();
        assert!(t.equals(&t2));
    }

    #[test]
    fn test_lde_column_consistency() {
        let coeffs = vec![gf(1), gf(0), gf(0), gf(0)];
        let mut evals = coeffs.clone();
        ntt(&mut evals);
        let t = ExecutionTrace::from_columns(vec![evals]);
        let lde = t.low_degree_extend(4);
        for r in 0..lde.height() { assert_eq!(lde.get(r, 0), gf(1)); }
    }

    #[test]
    fn test_all_columns_to_polynomials() {
        let t = make_small_trace();
        let polys = t.all_columns_to_polynomials();
        assert_eq!(polys.len(), 3);
        for c in 0..3 {
            let mut evals = polys[c].clone();
            ntt(&mut evals);
            let col = t.get_column(c);
            for i in 0..4 { assert_eq!(evals[i], col[i]); }
        }
    }

    #[test]
    fn test_evaluate_all_columns_at() {
        let t = make_small_trace();
        let x = gf(17);
        let vals = t.evaluate_all_columns_at(x);
        for c in 0..3 { assert_eq!(vals[c], t.evaluate_polynomial_column(c, x)); }
    }

    #[test]
    fn test_boundary_quotient_basic() {
        let lde = ExecutionTrace::counter_trace(4).low_degree_extend(2);
        assert_eq!(lde.boundary_quotient_column(0, 0, gf(0), 4, gf(7)).len(), 8);
    }

    #[test]
    fn test_deep_composition_column() {
        let t = make_small_trace();
        let lde = t.low_degree_extend(2);
        let (ez, ezn) = t.deep_query(gf(42));
        let dc = ExecutionTrace::deep_composition_column(&lde, &ez, &ezn, gf(42), gf(3), 4, gf(7));
        assert_eq!(dc.len(), 8);
    }

    #[test]
    fn test_running_sum_column() {
        let t = ExecutionTrace::counter_trace(4);
        let rs = t.running_sum_column(0, gf(100));
        assert_eq!(rs[0], gf(100).inv_or_panic());
    }

    #[test]
    fn test_extend_with_random_linear_combinations() {
        let mut t = make_small_trace();
        t.extend_with_random_linear_combinations(1, &[vec![gf(1), gf(2), gf(3)]]);
        assert_eq!(t.width(), 4);
        assert_eq!(t.get(0, 3), gf(14));
    }

    #[test]
    fn test_backward_compat_push_row() {
        let mut t = ExecutionTrace::zeros(2, 4);
        assert_eq!(t.length, 4);
        assert_eq!(t.width, 2);
        t.push_row(vec![gf(1), gf(2)]);
        assert_eq!(t.length, 5);
    }

    #[test]
    fn test_backward_compat_row_column() {
        let t = make_small_trace();
        assert_eq!(t.row(0), &[gf(1), gf(2), gf(3)]);
        assert_eq!(t.column(0), vec![gf(1), gf(4), gf(7), gf(10)]);
        assert_eq!(t.num_rows(), 4);
        assert_eq!(t.num_cols(), 3);
        assert!(t.is_power_of_two());
    }

    // ═══════════════════════════════════════════════════════════
    // TraceAnalyzer tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_analyzer_find_zero_columns() {
        let mut t = ExecutionTrace::new(4, 3);
        // col 0: all zero, col 1: has values, col 2: all zero
        for r in 0..4 { t.set(r, 1, gf((r + 1) as u64)); }
        let analyzer = TraceAnalyzer::new();
        let zeros = analyzer.find_zero_columns(&t);
        assert_eq!(zeros, vec![0, 2]);
    }

    #[test]
    fn test_analyzer_find_constant_columns() {
        let mut t = ExecutionTrace::new(4, 3);
        for r in 0..4 {
            t.set(r, 0, gf(42));
            t.set(r, 1, gf(42));
            t.set(r, 2, gf((r + 1) as u64));
        }
        let analyzer = TraceAnalyzer::new();
        let consts = analyzer.find_constant_columns(&t);
        assert_eq!(consts, vec![0, 1]);
    }

    #[test]
    fn test_analyzer_find_periodic_columns() {
        let mut t = ExecutionTrace::new(4, 2);
        // col 0: period 2 → [1,2,1,2]
        t.set(0, 0, gf(1)); t.set(1, 0, gf(2));
        t.set(2, 0, gf(1)); t.set(3, 0, gf(2));
        // col 1: not periodic → [1,2,3,4]
        for r in 0..4 { t.set(r, 1, gf((r + 1) as u64)); }
        let analyzer = TraceAnalyzer::new();
        let periodic = analyzer.find_periodic_columns(&t);
        assert!(periodic.iter().any(|&(c, p)| c == 0 && p == 2));
    }

    #[test]
    fn test_analyzer_column_entropy() {
        let analyzer = TraceAnalyzer::new();
        // Constant column → entropy 0
        let t = ExecutionTrace::filled(4, 1, gf(5));
        assert!((analyzer.column_entropy(&t, 0) - 0.0).abs() < 1e-10);
        // All distinct → entropy = log2(4) = 2.0
        let t2 = ExecutionTrace::from_columns(vec![vec![gf(1), gf(2), gf(3), gf(4)]]);
        assert!((analyzer.column_entropy(&t2, 0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_row_similarity() {
        let t = make_small_trace();
        let analyzer = TraceAnalyzer::new();
        // Same row → similarity 1.0
        assert!((analyzer.row_similarity(&t, 0, 0) - 1.0).abs() < 1e-10);
        // Different rows → similarity 0.0
        assert!((analyzer.row_similarity(&t, 0, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_column_autocorrelation() {
        let analyzer = TraceAnalyzer::new();
        // Constant column → autocorrelation 1.0
        let t = ExecutionTrace::filled(4, 1, gf(5));
        assert!((analyzer.column_autocorrelation(&t, 0, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_detect_anomalies_repeated_rows() {
        let t = ExecutionTrace::from_rows(vec![
            vec![gf(1), gf(2)],
            vec![gf(1), gf(2)], // repeated
            vec![gf(3), gf(4)],
            vec![gf(5), gf(6)],
        ]);
        let analyzer = TraceAnalyzer::new();
        let anomalies = analyzer.detect_anomalies(&t);
        assert!(anomalies.iter().any(|a| a.anomaly_type == AnomalyType::RepeatedRow && a.row == 1));
    }

    #[test]
    fn test_analyzer_analyze_full() {
        let t = make_small_trace();
        let analyzer = TraceAnalyzer::new();
        let analysis = analyzer.analyze(&t);
        assert_eq!(analysis.num_zero_cols, 0);
        assert_eq!(analysis.num_constant_cols, 0);
        assert_eq!(analysis.col_entropies.len(), 3);
        assert!(analysis.overall_density > 0.0);
    }

    #[test]
    fn test_trace_analysis_summary_and_json() {
        let t = make_small_trace();
        let analyzer = TraceAnalyzer::new();
        let analysis = analyzer.analyze(&t);
        let summary = analysis.summary();
        assert!(summary.contains("TraceAnalysis"));
        let json = analysis.to_json();
        assert!(json.contains("num_zero_cols"));
        assert!(json.contains("overall_density"));
    }

    // ═══════════════════════════════════════════════════════════
    // AnomalyType / TraceAnomaly tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_anomaly_type_display() {
        assert_eq!(format!("{}", AnomalyType::ZeroInNonZeroColumn), "ZeroInNonZeroColumn");
        assert_eq!(format!("{}", AnomalyType::RepeatedRow), "RepeatedRow");
        assert_eq!(format!("{}", AnomalyType::DiscontinuousTransition), "DiscontinuousTransition");
        assert_eq!(format!("{}", AnomalyType::ValueOutOfRange), "ValueOutOfRange");
    }

    #[test]
    fn test_trace_anomaly_display() {
        let a = TraceAnomaly {
            row: 3,
            col: 1,
            anomaly_type: AnomalyType::RepeatedRow,
            details: "test detail".to_string(),
        };
        let s = format!("{}", a);
        assert!(s.contains("row=3"));
        assert!(s.contains("col=1"));
        assert!(s.contains("RepeatedRow"));
        assert!(s.contains("test detail"));
    }

    // ═══════════════════════════════════════════════════════════
    // TraceCompressor tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_run_length_encode_decode() {
        let col = vec![gf(1), gf(1), gf(1), gf(2), gf(2), gf(3)];
        let encoded = TraceCompressor::run_length_encode(&col);
        assert_eq!(encoded, vec![(gf(1), 3), (gf(2), 2), (gf(3), 1)]);
        let decoded = TraceCompressor::run_length_decode(&encoded);
        assert_eq!(decoded, col);
    }

    #[test]
    fn test_run_length_encode_empty() {
        let col: Vec<GoldilocksField> = vec![];
        let encoded = TraceCompressor::run_length_encode(&col);
        assert!(encoded.is_empty());
        let decoded = TraceCompressor::run_length_decode(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_delta_encode_decode() {
        let col = vec![gf(10), gf(13), gf(15), gf(20)];
        let encoded = TraceCompressor::delta_encode(&col);
        assert_eq!(encoded[0], gf(10));
        assert_eq!(encoded[1], gf(3));
        assert_eq!(encoded[2], gf(2));
        assert_eq!(encoded[3], gf(5));
        let decoded = TraceCompressor::delta_decode(&encoded);
        assert_eq!(decoded, col);
    }

    #[test]
    fn test_delta_encode_decode_empty() {
        let col: Vec<GoldilocksField> = vec![];
        let encoded = TraceCompressor::delta_encode(&col);
        assert!(encoded.is_empty());
        let decoded = TraceCompressor::delta_decode(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let t = make_small_trace();
        let compressed = TraceCompressor::compress(&t);
        assert!(compressed.decompressible());
        assert_eq!(compressed.num_rows, 4);
        assert_eq!(compressed.num_cols, 3);
        let decompressed = TraceCompressor::decompress(&compressed);
        assert!(t.equals(&decompressed));
    }

    #[test]
    fn test_compress_constant_trace() {
        let t = ExecutionTrace::filled(8, 3, gf(42));
        let compressed = TraceCompressor::compress(&t);
        // Constant columns should be very small
        for cc in &compressed.columns {
            assert_eq!(cc.encoding, ColumnEncoding::Constant(gf(42)));
            assert_eq!(cc.data.len(), 8);
        }
        let decompressed = TraceCompressor::decompress(&compressed);
        assert!(t.equals(&decompressed));
    }

    #[test]
    fn test_compressed_trace_size_bytes() {
        let t = make_small_trace();
        let compressed = TraceCompressor::compress(&t);
        assert!(compressed.size_bytes() > 0);
        assert!(compressed.size_bytes() <= compressed.original_size_bytes);
    }

    #[test]
    fn test_estimate_compressed_size() {
        let t = ExecutionTrace::filled(4, 2, gf(1));
        let est = TraceCompressor::estimate_compressed_size(&t);
        assert_eq!(est, 16); // 2 constant columns × 8 bytes each
    }

    #[test]
    fn test_compression_ratio() {
        let t = ExecutionTrace::filled(8, 4, gf(1));
        let ratio = TraceCompressor::compression_ratio(&t);
        assert!(ratio > 1.0); // Constant trace should compress well
    }

    // ═══════════════════════════════════════════════════════════
    // TraceInterpolator tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_interpolator_column_as_polynomial() {
        let t = make_small_trace();
        let coeffs = TraceInterpolator::column_as_polynomial(&t, 0);
        assert_eq!(coeffs.len(), 4);
        // Verify by NTT back
        let mut evals = coeffs.clone();
        ntt(&mut evals);
        assert_eq!(evals, t.get_column(0));
    }

    #[test]
    fn test_interpolator_polynomial_to_column() {
        let coeffs = vec![gf(1), gf(2), gf(0), gf(0)];
        let col = TraceInterpolator::polynomial_to_column(&coeffs, 4);
        assert_eq!(col.len(), 4);
        // Verify evaluation at root of unity matches
        let mut expected = coeffs.clone();
        ntt(&mut expected);
        assert_eq!(col, expected);
    }

    #[test]
    fn test_interpolator_evaluate_column_at_point() {
        let t = make_small_trace();
        let x = gf(17);
        let val = TraceInterpolator::evaluate_column_at_point(&t, 0, x);
        let coeffs = TraceInterpolator::column_as_polynomial(&t, 0);
        let expected = GoldilocksField::eval_poly(&coeffs, x);
        assert_eq!(val, expected);
    }

    #[test]
    fn test_interpolator_extrapolate_column() {
        let t = make_small_trace();
        let ext = TraceInterpolator::extrapolate_column(&t, 0, 2);
        assert_eq!(ext.len(), 6);
        // First 4 should match original
        for i in 0..4 { assert_eq!(ext[i], t.get(i, 0)); }
    }

    #[test]
    fn test_interpolator_column_degree() {
        // Constant column has degree 0
        let t = ExecutionTrace::filled(4, 1, gf(5));
        assert_eq!(TraceInterpolator::column_degree(&t, 0), 0);
    }

    #[test]
    fn test_interpolator_batch_interpolate() {
        let t = make_small_trace();
        let polys = TraceInterpolator::batch_interpolate(&t, &[0, 2]);
        assert_eq!(polys.len(), 2);
        assert_eq!(polys[0].len(), 4);
        assert_eq!(polys[1].len(), 4);
    }

    // ═══════════════════════════════════════════════════════════
    // TraceDiffer tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_differ_identical_traces() {
        let t = make_small_trace();
        assert!(TraceDiffer::is_identical(&t, &t));
        let diff = TraceDiffer::diff(&t, &t);
        assert!(diff.equal);
        assert_eq!(diff.num_differences, 0);
    }

    #[test]
    fn test_differ_different_traces() {
        let t1 = make_small_trace();
        let mut t2 = make_small_trace();
        t2.set(0, 0, gf(99));
        assert!(!TraceDiffer::is_identical(&t1, &t2));
        let diff = TraceDiffer::diff(&t1, &t2);
        assert!(!diff.equal);
        assert_eq!(diff.num_differences, 1);
        assert_eq!(diff.differences[0].row, 0);
        assert_eq!(diff.differences[0].col, 0);
    }

    #[test]
    fn test_differ_max_difference() {
        let t1 = make_small_trace();
        let t2 = make_small_trace();
        assert!(TraceDiffer::max_difference(&t1, &t2).is_none());

        let mut t3 = make_small_trace();
        t3.set(2, 1, gf(1000));
        let max_diff = TraceDiffer::max_difference(&t1, &t3);
        assert!(max_diff.is_some());
        let (r, c, _) = max_diff.unwrap();
        assert_eq!(r, 2);
        assert_eq!(c, 1);
    }

    #[test]
    fn test_differ_diff_column() {
        let t1 = make_small_trace();
        let mut t2 = make_small_trace();
        t2.set(1, 0, gf(99));
        t2.set(3, 0, gf(88));
        let diffs = TraceDiffer::diff_column(&t1, &t2, 0);
        assert_eq!(diffs.len(), 2);
        assert_eq!(diffs[0].0, 1); // row 1
        assert_eq!(diffs[1].0, 3); // row 3
    }

    #[test]
    fn test_trace_diff_summary_equal() {
        let t = make_small_trace();
        let diff = TraceDiffer::diff(&t, &t);
        assert!(diff.summary().contains("identical"));
    }

    #[test]
    fn test_trace_diff_summary_different() {
        let t1 = make_small_trace();
        let mut t2 = make_small_trace();
        t2.set(0, 0, gf(99));
        let diff = TraceDiffer::diff(&t1, &t2);
        assert!(diff.summary().contains("differ"));
    }

    #[test]
    fn test_differ_different_dimensions() {
        let t1 = ExecutionTrace::new(4, 3);
        let t2 = ExecutionTrace::new(4, 2);
        assert!(!TraceDiffer::is_identical(&t1, &t2));
        let diff = TraceDiffer::diff(&t1, &t2);
        assert!(!diff.equal);
    }

    // ═══════════════════════════════════════════════════════════
    // TraceBuilder tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_builder_basic() {
        let mut builder = TraceBuilder::new(2);
        builder.add_row(vec![gf(1), gf(2)]);
        builder.add_row(vec![gf(3), gf(4)]);
        builder.add_row(vec![gf(5), gf(6)]);
        builder.add_row(vec![gf(7), gf(8)]);
        let t = builder.build();
        assert_eq!(t.width(), 2);
        assert_eq!(t.height(), 4);
        assert_eq!(t.get(0, 0), gf(1));
        assert_eq!(t.get(3, 1), gf(8));
    }

    #[test]
    fn test_builder_with_names() {
        let mut builder = TraceBuilder::new(2);
        builder.set_column_names(vec!["a".into(), "b".into()]);
        builder.add_row(vec![gf(1), gf(2)]);
        builder.add_row(vec![gf(3), gf(4)]);
        let t = builder.build();
        assert_eq!(t.column_names()[0], "a");
        assert_eq!(t.column_names()[1], "b");
    }

    #[test]
    fn test_builder_build_padded() {
        let mut builder = TraceBuilder::new(2);
        builder.add_row(vec![gf(1), gf(2)]);
        builder.add_row(vec![gf(3), gf(4)]);
        builder.add_row(vec![gf(5), gf(6)]);
        let t = builder.build_padded();
        assert_eq!(t.height(), 4); // padded from 3 to 4
        assert!(t.is_power_of_two());
    }

    #[test]
    fn test_builder_from_state_machine() {
        let states = vec![
            vec![gf(0), gf(1)],
            vec![gf(1), gf(2)],
            vec![gf(2), gf(3)],
            vec![gf(3), gf(4)],
        ];
        let inputs = vec![gf(10), gf(20), gf(30), gf(40)];
        let builder = TraceBuilder::from_state_machine(&states, &inputs);
        let t = builder.build();
        assert_eq!(t.width(), 3); // 2 state cols + 1 input col
        assert_eq!(t.height(), 4);
        assert_eq!(t.get(0, 0), gf(0));
        assert_eq!(t.get(0, 2), gf(10));
        assert_eq!(t.column_names()[2], "input");
    }

    #[test]
    #[should_panic]
    fn test_builder_empty_build_panics() {
        let builder = TraceBuilder::new(2);
        builder.build();
    }

    #[test]
    #[should_panic]
    fn test_builder_wrong_row_width_panics() {
        let mut builder = TraceBuilder::new(2);
        builder.add_row(vec![gf(1), gf(2), gf(3)]);
    }

    // ═══════════════════════════════════════════════════════════
    // TraceTransformer tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_transformer_permute_columns() {
        let t = make_small_trace();
        let p = TraceTransformer::permute_columns(&t, &[2, 0, 1]);
        assert_eq!(p.get(0, 0), gf(3)); // was col 2
        assert_eq!(p.get(0, 1), gf(1)); // was col 0
        assert_eq!(p.get(0, 2), gf(2)); // was col 1
    }

    #[test]
    fn test_transformer_permute_rows() {
        let t = make_small_trace();
        let p = TraceTransformer::permute_rows(&t, &[3, 2, 1, 0]);
        assert_eq!(p.get(0, 0), gf(10)); // was row 3
        assert_eq!(p.get(3, 0), gf(1));  // was row 0
    }

    #[test]
    fn test_transformer_interleave() {
        let a = ExecutionTrace::from_rows(vec![
            vec![gf(1), gf(2)],
            vec![gf(3), gf(4)],
        ]);
        let b = ExecutionTrace::from_rows(vec![
            vec![gf(10), gf(20)],
            vec![gf(30), gf(40)],
        ]);
        let interleaved = TraceTransformer::interleave_traces(&a, &b);
        assert_eq!(interleaved.height(), 4);
        assert_eq!(interleaved.get(0, 0), gf(1));
        assert_eq!(interleaved.get(1, 0), gf(10));
        assert_eq!(interleaved.get(2, 0), gf(3));
        assert_eq!(interleaved.get(3, 0), gf(30));
    }

    #[test]
    fn test_transformer_reverse_rows() {
        let t = make_small_trace();
        let rev = TraceTransformer::reverse_rows(&t);
        assert_eq!(rev.get(0, 0), gf(10));
        assert_eq!(rev.get(3, 0), gf(1));
    }

    #[test]
    fn test_transformer_rotate_columns() {
        let t = make_small_trace();
        let rotated = TraceTransformer::rotate_columns(&t, 1);
        // Shift right by 1: [col0,col1,col2] → [col2,col0,col1]
        assert_eq!(rotated.get(0, 0), gf(3)); // was col 2
        assert_eq!(rotated.get(0, 1), gf(1)); // was col 0
        assert_eq!(rotated.get(0, 2), gf(2)); // was col 1
    }

    #[test]
    fn test_transformer_rotate_columns_zero() {
        let t = make_small_trace();
        let rotated = TraceTransformer::rotate_columns(&t, 0);
        assert!(t.equals(&rotated));
    }

    #[test]
    fn test_transformer_rotate_columns_full() {
        let t = make_small_trace();
        let rotated = TraceTransformer::rotate_columns(&t, 3);
        assert!(t.equals(&rotated));
    }

    #[test]
    fn test_transformer_apply_linear_transform_identity() {
        let t = ExecutionTrace::from_rows(vec![
            vec![gf(1), gf(2)],
            vec![gf(3), gf(4)],
            vec![gf(5), gf(6)],
            vec![gf(7), gf(8)],
        ]);
        let identity = vec![
            vec![gf(1), gf(0)],
            vec![gf(0), gf(1)],
        ];
        let result = TraceTransformer::apply_linear_transform(&t, &identity);
        assert!(t.equals(&result));
    }

    #[test]
    fn test_transformer_apply_linear_transform_swap() {
        let t = ExecutionTrace::from_rows(vec![
            vec![gf(1), gf(2)],
            vec![gf(3), gf(4)],
            vec![gf(5), gf(6)],
            vec![gf(7), gf(8)],
        ]);
        let swap = vec![
            vec![gf(0), gf(1)],
            vec![gf(1), gf(0)],
        ];
        let result = TraceTransformer::apply_linear_transform(&t, &swap);
        assert_eq!(result.get(0, 0), gf(2));
        assert_eq!(result.get(0, 1), gf(1));
    }

    #[test]
    fn test_transformer_filter_rows() {
        let t = make_small_trace();
        // Keep only rows where col 0 > 5
        let filtered = TraceTransformer::filter_rows(&t, |_, row| {
            row[0].to_canonical() > 5
        });
        assert_eq!(filtered.height(), 2); // rows with 7 and 10
        assert_eq!(filtered.get(0, 0), gf(7));
        assert_eq!(filtered.get(1, 0), gf(10));
    }

    #[test]
    fn test_transformer_filter_rows_none() {
        let t = make_small_trace();
        let filtered = TraceTransformer::filter_rows(&t, |_, _| false);
        assert_eq!(filtered.height(), 0);
    }

    #[test]
    fn test_transformer_map_column() {
        let t = make_small_trace();
        let mapped = TraceTransformer::map_column(&t, 0, |v| v.mul_elem(gf(2)));
        assert_eq!(mapped.get(0, 0), gf(2));
        assert_eq!(mapped.get(1, 0), gf(8));
        // Other columns unchanged
        assert_eq!(mapped.get(0, 1), gf(2));
    }

    // ═══════════════════════════════════════════════════════════
    // TraceValidator tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_validator_validate_full_good_trace() {
        let t = make_small_trace(); // 4 rows, power of 2
        let errors = TraceValidator::validate_full(&t);
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_validator_validate_full_non_power_of_two() {
        let t = ExecutionTrace::from_rows(vec![
            vec![gf(1)],
            vec![gf(2)],
            vec![gf(3)],
        ]);
        let errors = TraceValidator::validate_full(&t);
        assert!(errors.iter().any(|e| e.error_type == "NotPowerOfTwo"));
    }

    #[test]
    fn test_validator_power_of_two() {
        let t = make_small_trace();
        assert!(TraceValidator::validate_power_of_two(&t).is_ok());

        let t3 = ExecutionTrace::from_rows(vec![vec![gf(1)], vec![gf(2)], vec![gf(3)]]);
        assert!(TraceValidator::validate_power_of_two(&t3).is_err());
    }

    #[test]
    fn test_validator_column_range() {
        let t = make_small_trace();
        assert!(TraceValidator::validate_column_range(&t, 0, gf(1), gf(10)).is_ok());
        assert!(TraceValidator::validate_column_range(&t, 0, gf(2), gf(10)).is_err()); // row 0 has 1
    }

    #[test]
    fn test_validator_boolean_column() {
        let t = ExecutionTrace::from_columns(vec![
            vec![gf(0), gf(1), gf(1), gf(0)],
        ]);
        assert!(TraceValidator::validate_boolean_column(&t, 0).is_ok());

        let t2 = ExecutionTrace::from_columns(vec![
            vec![gf(0), gf(2), gf(1), gf(0)],
        ]);
        assert!(TraceValidator::validate_boolean_column(&t2, 0).is_err());
    }

    #[test]
    fn test_validator_permutation_column() {
        let t = ExecutionTrace::from_columns(vec![
            vec![gf(2), gf(0), gf(3), gf(1)],
        ]);
        assert!(TraceValidator::validate_permutation_column(&t, 0).is_ok());

        // Duplicate value → invalid
        let t2 = ExecutionTrace::from_columns(vec![
            vec![gf(0), gf(0), gf(1), gf(2)],
        ]);
        assert!(TraceValidator::validate_permutation_column(&t2, 0).is_err());

        // Value out of range
        let t3 = ExecutionTrace::from_columns(vec![
            vec![gf(0), gf(1), gf(2), gf(10)],
        ]);
        assert!(TraceValidator::validate_permutation_column(&t3, 0).is_err());
    }

    #[test]
    fn test_validator_monotonic_column() {
        let t = ExecutionTrace::from_columns(vec![
            vec![gf(1), gf(3), gf(5), gf(7)],
        ]);
        assert!(TraceValidator::validate_monotonic_column(&t, 0).is_ok());

        let t2 = ExecutionTrace::from_columns(vec![
            vec![gf(1), gf(5), gf(3), gf(7)],
        ]);
        assert!(TraceValidator::validate_monotonic_column(&t2, 0).is_err());
    }

    #[test]
    fn test_validation_error_display() {
        let e = ValidationError {
            error_type: "TestError".into(),
            row: Some(5),
            col: Some(2),
            message: "test message".into(),
        };
        let s = format!("{}", e);
        assert!(s.contains("TestError"));
        assert!(s.contains("row=5"));
        assert!(s.contains("col=2"));
        assert!(s.contains("test message"));
    }

    #[test]
    fn test_validation_error_display_no_row_col() {
        let e = ValidationError {
            error_type: "TestError".into(),
            row: None,
            col: None,
            message: "test".into(),
        };
        let s = format!("{}", e);
        assert!(s.contains("TestError"));
        assert!(!s.contains("row="));
    }

    // ═══════════════════════════════════════════════════════════
    // CompressedTrace tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_compressed_trace_decompressible() {
        let ct = CompressedTrace {
            columns: vec![
                CompressedColumn { encoding: ColumnEncoding::Raw, data: vec![0u8; 32] },
            ],
            num_rows: 4,
            num_cols: 1,
            original_size_bytes: 32,
        };
        assert!(ct.decompressible());
    }

    #[test]
    fn test_compressed_trace_not_decompressible() {
        let ct = CompressedTrace {
            columns: vec![],
            num_rows: 4,
            num_cols: 1,
            original_size_bytes: 32,
        };
        assert!(!ct.decompressible());
    }

    #[test]
    fn test_column_encoding_variants() {
        let _r = ColumnEncoding::Raw;
        let _rl = ColumnEncoding::RunLength;
        let _d = ColumnEncoding::Delta;
        let _c = ColumnEncoding::Constant(gf(42));
        assert_eq!(ColumnEncoding::Constant(gf(1)), ColumnEncoding::Constant(gf(1)));
        assert_ne!(ColumnEncoding::Raw, ColumnEncoding::Delta);
    }

    // ═══════════════════════════════════════════════════════════
    // TraceDiff / CellDifference struct tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_cell_difference_fields() {
        let cd = CellDifference {
            row: 1,
            col: 2,
            value_a: gf(10),
            value_b: gf(20),
        };
        assert_eq!(cd.row, 1);
        assert_eq!(cd.col, 2);
        assert_eq!(cd.value_a, gf(10));
        assert_eq!(cd.value_b, gf(20));
    }

    // ═══════════════════════════════════════════════════════════
    // Additional integration tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_compress_decompress_counter_trace() {
        let t = ExecutionTrace::counter_trace(8);
        let compressed = TraceCompressor::compress(&t);
        let decompressed = TraceCompressor::decompress(&compressed);
        assert!(t.equals(&decompressed));
    }

    #[test]
    fn test_analyzer_with_zero_column_trace() {
        let t = ExecutionTrace::new(4, 2); // all zeros
        let analyzer = TraceAnalyzer::new();
        let analysis = analyzer.analyze(&t);
        assert_eq!(analysis.num_zero_cols, 2);
        assert_eq!(analysis.num_constant_cols, 2);
        assert!((analysis.overall_density - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_chain() {
        let t = {
            let mut b = TraceBuilder::new(2);
            b.add_row(vec![gf(1), gf(2)])
             .add_row(vec![gf(3), gf(4)])
             .add_row(vec![gf(5), gf(6)])
             .add_row(vec![gf(7), gf(8)]);
            b.build()
        };
        assert_eq!(t.width(), 2);
        assert_eq!(t.height(), 4);
    }

    #[test]
    fn test_transformer_permute_columns_identity() {
        let t = make_small_trace();
        let p = TraceTransformer::permute_columns(&t, &[0, 1, 2]);
        assert!(t.equals(&p));
    }

    #[test]
    fn test_transformer_permute_rows_identity() {
        let t = make_small_trace();
        let p = TraceTransformer::permute_rows(&t, &[0, 1, 2, 3]);
        assert!(t.equals(&p));
    }

    #[test]
    fn test_transformer_reverse_rows_twice_identity() {
        let t = make_small_trace();
        let rev = TraceTransformer::reverse_rows(&TraceTransformer::reverse_rows(&t));
        assert!(t.equals(&rev));
    }

    #[test]
    fn test_differ_is_identical_symmetry() {
        let t1 = make_small_trace();
        let mut t2 = make_small_trace();
        t2.set(0, 0, gf(99));
        assert!(!TraceDiffer::is_identical(&t1, &t2));
        assert!(!TraceDiffer::is_identical(&t2, &t1));
    }

    #[test]
    fn test_validator_validate_full_empty_trace() {
        let t = ExecutionTrace {
            rows: vec![],
            length: 0,
            width: 0,
            column_names: vec![],
        };
        let errors = TraceValidator::validate_full(&t);
        assert!(errors.iter().any(|e| e.error_type == "EmptyTrace"));
    }

    #[test]
    fn test_interpolator_roundtrip() {
        let t = make_small_trace();
        let coeffs = TraceInterpolator::column_as_polynomial(&t, 0);
        let evals = TraceInterpolator::polynomial_to_column(&coeffs, 4);
        let original = t.get_column(0);
        assert_eq!(evals, original);
    }

    #[test]
    fn test_validator_monotonic_equal_values() {
        let t = ExecutionTrace::from_columns(vec![
            vec![gf(3), gf(3), gf(5), gf(5)],
        ]);
        assert!(TraceValidator::validate_monotonic_column(&t, 0).is_ok());
    }

    #[test]
    fn test_analyzer_anomaly_zero_in_nonzero() {
        // Create a trace with a mostly nonzero column that has one zero
        let t = ExecutionTrace::from_columns(vec![
            vec![gf(1), gf(2), gf(3), gf(4), gf(5), gf(6), gf(7), gf(0)],
        ]);
        let analyzer = TraceAnalyzer::new();
        let anomalies = analyzer.detect_anomalies(&t);
        assert!(anomalies.iter().any(|a| a.anomaly_type == AnomalyType::ZeroInNonZeroColumn));
    }

    #[test]
    fn test_compress_decompress_fibonacci() {
        let t = ExecutionTrace::fibonacci_trace(gf(1), gf(1), 8);
        let compressed = TraceCompressor::compress(&t);
        let decompressed = TraceCompressor::decompress(&compressed);
        assert!(t.equals(&decompressed));
    }

    #[test]
    fn test_transformer_map_column_identity() {
        let t = make_small_trace();
        let mapped = TraceTransformer::map_column(&t, 1, |v| v);
        assert!(t.equals(&mapped));
    }

    #[test]
    fn test_builder_from_state_machine_fewer_inputs() {
        let states = vec![
            vec![gf(0)],
            vec![gf(1)],
            vec![gf(2)],
            vec![gf(3)],
        ];
        let inputs = vec![gf(10), gf(20)]; // fewer inputs than states
        let builder = TraceBuilder::from_state_machine(&states, &inputs);
        let t = builder.build();
        assert_eq!(t.width(), 2);
        assert_eq!(t.get(0, 1), gf(10));
        assert_eq!(t.get(1, 1), gf(20));
        assert_eq!(t.get(2, 1), gf(0)); // default
    }

    #[test]
    fn test_differ_diff_column_no_differences() {
        let t = make_small_trace();
        let diffs = TraceDiffer::diff_column(&t, &t, 0);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compressed_trace_size_vs_original() {
        let t = ExecutionTrace::filled(16, 4, gf(1));
        let compressed = TraceCompressor::compress(&t);
        assert!(compressed.size_bytes() < compressed.original_size_bytes);
    }

    #[test]
    fn test_trace_analysis_json_parseable() {
        let t = make_small_trace();
        let analyzer = TraceAnalyzer::new();
        let analysis = analyzer.analyze(&t);
        let json = analysis.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn test_analyzer_periodic_constant_is_period_1() {
        let t = ExecutionTrace::filled(4, 1, gf(5));
        let analyzer = TraceAnalyzer::new();
        let periodic = analyzer.find_periodic_columns(&t);
        // A constant column is periodic with period 1
        assert!(periodic.iter().any(|&(c, p)| c == 0 && p == 1));
    }

    #[test]
    fn test_transformer_filter_keep_all() {
        let t = make_small_trace();
        let filtered = TraceTransformer::filter_rows(&t, |_, _| true);
        assert!(t.equals(&filtered));
    }

    #[test]
    fn test_interpolator_constant_column_degree_zero() {
        let t = ExecutionTrace::filled(4, 2, gf(7));
        assert_eq!(TraceInterpolator::column_degree(&t, 0), 0);
        assert_eq!(TraceInterpolator::column_degree(&t, 1), 0);
    }

    #[test]
    fn test_differ_max_difference_different_dims() {
        let t1 = ExecutionTrace::new(4, 2);
        let t2 = ExecutionTrace::new(4, 3);
        assert!(TraceDiffer::max_difference(&t1, &t2).is_none());
    }
}
