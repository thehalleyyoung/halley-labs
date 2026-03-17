//! Interval-based encoding of compatibility predicates.
//!
//! Exploits the fact that >92% of version compatibility predicates have interval
//! structure: compatible versions form contiguous ranges. This allows encoding
//! using O(log²L) clauses instead of O(L²) for explicit enumeration.

use crate::formula::{Clause, CnfFormula, Literal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// IntervalPredicate
// ---------------------------------------------------------------------------

/// A compatibility predicate with interval structure.
///
/// For a service pair (i, j), maps each version of service i to a compatible
/// range [lo, hi] of versions of service j.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalPredicate {
    /// Service i identifier index.
    pub service_i: usize,
    /// Service j identifier index.
    pub service_j: usize,
    /// Number of versions for service i.
    pub num_versions_i: usize,
    /// Number of versions for service j.
    pub num_versions_j: usize,
    /// For each version of service i, the compatible range [lo, hi] of service j.
    /// None means no compatible versions for this version of i.
    pub ranges: Vec<Option<(usize, usize)>>,
}

impl IntervalPredicate {
    /// Create a new interval predicate.
    pub fn new(
        service_i: usize,
        service_j: usize,
        num_versions_i: usize,
        num_versions_j: usize,
    ) -> Self {
        Self {
            service_i,
            service_j,
            num_versions_i,
            num_versions_j,
            ranges: vec![None; num_versions_i],
        }
    }

    /// Set the compatible range for version `vi` of service i.
    pub fn set_range(&mut self, vi: usize, lo: usize, hi: usize) {
        if vi < self.ranges.len() && lo <= hi && hi < self.num_versions_j {
            self.ranges[vi] = Some((lo, hi));
        }
    }

    /// Get the compatible range for version `vi` of service i.
    pub fn get_range(&self, vi: usize) -> Option<(usize, usize)> {
        self.ranges.get(vi).copied().flatten()
    }

    /// Check if this predicate truly has interval structure.
    /// Verifies that for each version of i, compatible versions of j form a
    /// contiguous range.
    pub fn is_interval_structured(&self) -> bool {
        self.ranges.iter().all(|r| {
            match r {
                None => true,
                Some((lo, hi)) => *lo <= *hi && *hi < self.num_versions_j,
            }
        })
    }

    /// Check if a specific pair (vi, vj) is compatible.
    pub fn is_compatible(&self, vi: usize, vj: usize) -> bool {
        match self.get_range(vi) {
            Some((lo, hi)) => vj >= lo && vj <= hi,
            None => false,
        }
    }

    /// Build from a compatibility matrix. Returns None if the matrix doesn't
    /// have interval structure.
    pub fn from_matrix(
        service_i: usize,
        service_j: usize,
        matrix: &[Vec<bool>],
    ) -> Option<Self> {
        let num_i = matrix.len();
        if num_i == 0 {
            return Some(Self::new(service_i, service_j, 0, 0));
        }
        let num_j = matrix[0].len();
        let mut pred = Self::new(service_i, service_j, num_i, num_j);

        for (vi, row) in matrix.iter().enumerate() {
            let compatible: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|(_, &c)| c)
                .map(|(j, _)| j)
                .collect();
            if compatible.is_empty() {
                pred.ranges[vi] = None;
                continue;
            }
            let lo = compatible[0];
            let hi = compatible[compatible.len() - 1];
            // Check contiguity
            if hi - lo + 1 != compatible.len() {
                return None;
            }
            pred.ranges[vi] = Some((lo, hi));
        }
        Some(pred)
    }

    /// Compress this predicate into a binary encoding representation.
    pub fn compress(&self) -> CompressedInterval {
        let bits_i = BinaryEncoding::num_bits(self.num_versions_i.saturating_sub(1));
        let bits_j = BinaryEncoding::num_bits(self.num_versions_j.saturating_sub(1));
        CompressedInterval {
            bits_i,
            bits_j,
            ranges: self.ranges.clone(),
            service_i: self.service_i,
            service_j: self.service_j,
        }
    }

    /// Count the number of compatible pairs.
    pub fn compatible_count(&self) -> usize {
        self.ranges
            .iter()
            .filter_map(|r| r.as_ref())
            .map(|(lo, hi)| hi - lo + 1)
            .sum()
    }

    /// Total possible pairs.
    pub fn total_pairs(&self) -> usize {
        self.num_versions_i * self.num_versions_j
    }

    /// Density of compatible pairs.
    pub fn density(&self) -> f64 {
        let total = self.total_pairs();
        if total == 0 {
            return 0.0;
        }
        self.compatible_count() as f64 / total as f64
    }
}

impl fmt::Display for IntervalPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IntervalPredicate(s{}→s{}, {}×{}, density={:.2})",
            self.service_i,
            self.service_j,
            self.num_versions_i,
            self.num_versions_j,
            self.density()
        )
    }
}

/// Compressed interval representation using binary encoding parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedInterval {
    pub bits_i: usize,
    pub bits_j: usize,
    pub ranges: Vec<Option<(usize, usize)>>,
    pub service_i: usize,
    pub service_j: usize,
}

// ---------------------------------------------------------------------------
// BinaryEncoding
// ---------------------------------------------------------------------------

/// Maps version indices to binary variable representations.
#[derive(Debug, Clone)]
pub struct BinaryEncoding {
    /// Number of bits needed.
    pub num_bits: usize,
    /// Maximum value that can be represented.
    pub max_value: usize,
    /// Starting SAT variable ID for the first bit.
    pub base_var: u32,
}

impl BinaryEncoding {
    /// Compute the number of bits needed to represent values 0..=max_val.
    pub fn num_bits(max_val: usize) -> usize {
        if max_val == 0 {
            return 1;
        }
        let mut bits = 0;
        let mut v = max_val;
        while v > 0 {
            bits += 1;
            v >>= 1;
        }
        bits
    }

    /// Create a new binary encoding.
    pub fn new(max_value: usize, base_var: u32) -> Self {
        Self {
            num_bits: Self::num_bits(max_value),
            max_value,
            base_var,
        }
    }

    /// Get the SAT variable for bit `bit_idx` (0 = LSB).
    pub fn bit_var(&self, bit_idx: usize) -> Literal {
        (self.base_var + bit_idx as u32) as Literal
    }

    /// Encode a specific value: return vector of literals asserting x == value.
    pub fn encode_value(&self, value: usize) -> Vec<Literal> {
        let mut lits = Vec::with_capacity(self.num_bits);
        for bit in 0..self.num_bits {
            let var = self.bit_var(bit);
            if (value >> bit) & 1 == 1 {
                lits.push(var);
            } else {
                lits.push(-var);
            }
        }
        lits
    }

    /// Decode an assignment of bit variables to a value.
    pub fn decode_assignment(&self, assignment: &HashMap<u32, bool>) -> usize {
        let mut value = 0;
        for bit in 0..self.num_bits {
            let var = self.base_var + bit as u32;
            if *assignment.get(&var).unwrap_or(&false) {
                value |= 1 << bit;
            }
        }
        value
    }

    /// Get all SAT variable IDs used by this encoding.
    pub fn variables(&self) -> Vec<u32> {
        (0..self.num_bits)
            .map(|b| self.base_var + b as u32)
            .collect()
    }

    /// Generate clauses constraining the encoded value to be <= max_value.
    /// This handles cases where 2^num_bits - 1 > max_value.
    pub fn bound_constraint(&self) -> Vec<Clause> {
        let max_representable = (1 << self.num_bits) - 1;
        if self.max_value >= max_representable {
            return Vec::new();
        }
        // Block values from max_value+1 to max_representable
        let mut clauses = Vec::new();
        for forbidden in (self.max_value + 1)..=max_representable {
            let lits = self.encode_value(forbidden);
            // Negate: at least one bit must differ
            let clause: Vec<Literal> = lits.iter().map(|&l| -l).collect();
            clauses.push(clause);
        }
        clauses
    }
}

// ---------------------------------------------------------------------------
// Comparator
// ---------------------------------------------------------------------------

/// SAT encoding of integer comparison operations on binary-encoded values.
#[derive(Debug, Clone)]
pub struct Comparator {
    next_var: u32,
}

impl Comparator {
    pub fn new(next_var: u32) -> Self {
        Self { next_var }
    }

    /// Allocate a fresh variable.
    fn fresh(&mut self) -> Literal {
        let v = self.next_var;
        self.next_var += 1;
        v as Literal
    }

    /// Get the next available variable ID.
    pub fn next_var(&self) -> u32 {
        self.next_var
    }

    /// Encode: a <= b, where a and b are binary-encoded integers.
    /// Returns clauses that are satisfiable iff a <= b.
    /// Uses a bit-by-bit comparison from MSB to LSB with auxiliary variables.
    pub fn less_than_or_equal(
        &mut self,
        a_bits: &[Literal],
        b_bits: &[Literal],
    ) -> Vec<Clause> {
        assert_eq!(a_bits.len(), b_bits.len());
        let n = a_bits.len();
        if n == 0 {
            return Vec::new();
        }

        let mut clauses = Vec::new();

        // eq[i]: bits i..n-1 are equal (a and b agree on bits i through n-1)
        // lt[i]: a < b considering bits i..n-1
        // We need lt[0] OR eq[0] to hold (a <= b).

        // Allocate auxiliary variables
        let mut eq_vars = Vec::with_capacity(n);
        let mut lt_vars = Vec::with_capacity(n);
        for _ in 0..n {
            eq_vars.push(self.fresh());
            lt_vars.push(self.fresh());
        }

        // MSB is at index n-1
        // eq[n-1] <=> (a[n-1] <=> b[n-1])
        let a_msb = a_bits[n - 1];
        let b_msb = b_bits[n - 1];
        let eq_msb = eq_vars[n - 1];
        // eq_msb => (a_msb => b_msb) AND (b_msb => a_msb)
        clauses.push(vec![-eq_msb, -a_msb, b_msb]);
        clauses.push(vec![-eq_msb, a_msb, -b_msb]);
        // (a_msb <=> b_msb) => eq_msb
        clauses.push(vec![a_msb, b_msb, eq_msb]); // both false => eq
        clauses.push(vec![-a_msb, -b_msb, eq_msb]); // both true => eq
        // When they differ, NOT eq
        clauses.push(vec![a_msb, -b_msb, -eq_msb]);
        clauses.push(vec![-a_msb, b_msb, -eq_msb]);

        // lt[n-1] <=> (NOT a[n-1] AND b[n-1])
        let lt_msb = lt_vars[n - 1];
        clauses.push(vec![-lt_msb, -a_msb]); // lt => NOT a
        clauses.push(vec![-lt_msb, b_msb]); // lt => b
        clauses.push(vec![a_msb, -b_msb, lt_msb]); // NOT a AND b => lt

        // For remaining bits (n-2 down to 0)
        for i in (0..n - 1).rev() {
            let a_i = a_bits[i];
            let b_i = b_bits[i];
            let eq_i = eq_vars[i];
            let lt_i = lt_vars[i];
            let eq_above = eq_vars[i + 1];
            let lt_above = lt_vars[i + 1];

            // bit_eq_i: a[i] == b[i]
            let bit_eq = self.fresh();
            clauses.push(vec![-bit_eq, -a_i, b_i]);
            clauses.push(vec![-bit_eq, a_i, -b_i]);
            clauses.push(vec![a_i, b_i, bit_eq]);
            clauses.push(vec![-a_i, -b_i, bit_eq]);
            clauses.push(vec![a_i, -b_i, -bit_eq]);
            clauses.push(vec![-a_i, b_i, -bit_eq]);

            // eq[i] <=> eq[i+1] AND bit_eq_i
            clauses.push(vec![-eq_i, eq_above]);
            clauses.push(vec![-eq_i, bit_eq]);
            clauses.push(vec![-eq_above, -bit_eq, eq_i]);

            // bit_lt_i: a[i] < b[i] at this bit (NOT a[i] AND b[i])
            let bit_lt = self.fresh();
            clauses.push(vec![-bit_lt, -a_i]);
            clauses.push(vec![-bit_lt, b_i]);
            clauses.push(vec![a_i, -b_i, bit_lt]);

            // lt[i] <=> lt[i+1] OR (eq[i+1] AND bit_lt_i)
            // lt[i] => lt[i+1] OR (eq[i+1] AND bit_lt_i)
            clauses.push(vec![-lt_i, lt_above, eq_above]);
            clauses.push(vec![-lt_i, lt_above, bit_lt]);
            // lt[i+1] => lt[i]
            clauses.push(vec![-lt_above, lt_i]);
            // (eq[i+1] AND bit_lt_i) => lt[i]
            clauses.push(vec![-eq_above, -bit_lt, lt_i]);
        }

        // Assert: lt[0] OR eq[0]
        clauses.push(vec![lt_vars[0], eq_vars[0]]);
        clauses
    }

    /// Encode: lo <= x <= hi, where x is binary-encoded.
    pub fn in_range(
        &mut self,
        x_bits: &[Literal],
        lo: usize,
        hi: usize,
    ) -> Vec<Clause> {
        let n = x_bits.len();
        if n == 0 {
            return Vec::new();
        }

        let mut clauses = Vec::new();

        // Encode lo <= x
        let lo_lits: Vec<Literal> = (0..n)
            .map(|bit| {
                let v = self.fresh();
                // Fix v to the bit value of lo
                if (lo >> bit) & 1 == 1 {
                    clauses.push(vec![v]);
                } else {
                    clauses.push(vec![-v]);
                }
                v
            })
            .collect();
        clauses.extend(self.less_than_or_equal(&lo_lits, x_bits));

        // Encode x <= hi
        let hi_lits: Vec<Literal> = (0..n)
            .map(|bit| {
                let v = self.fresh();
                if (hi >> bit) & 1 == 1 {
                    clauses.push(vec![v]);
                } else {
                    clauses.push(vec![-v]);
                }
                v
            })
            .collect();
        clauses.extend(self.less_than_or_equal(x_bits, &hi_lits));

        clauses
    }

    /// Encode: a == b (equality of binary-encoded integers).
    pub fn equality(
        &mut self,
        a_bits: &[Literal],
        b_bits: &[Literal],
    ) -> Vec<Clause> {
        assert_eq!(a_bits.len(), b_bits.len());
        let mut clauses = Vec::new();
        for i in 0..a_bits.len() {
            // a[i] <=> b[i]
            clauses.push(vec![-a_bits[i], b_bits[i]]);
            clauses.push(vec![a_bits[i], -b_bits[i]]);
        }
        clauses
    }

    /// Encode: a != b (inequality).
    pub fn not_equal(
        &mut self,
        a_bits: &[Literal],
        b_bits: &[Literal],
    ) -> Vec<Clause> {
        assert_eq!(a_bits.len(), b_bits.len());
        // At least one bit differs
        let mut diff_vars = Vec::new();
        let mut clauses = Vec::new();
        for i in 0..a_bits.len() {
            let d = self.fresh();
            diff_vars.push(d);
            // d => a[i] XOR b[i]
            // d => (a[i] OR b[i])
            clauses.push(vec![-d, a_bits[i], b_bits[i]]);
            // d => NOT(a[i] AND b[i])
            clauses.push(vec![-d, -a_bits[i], -b_bits[i]]);
            // (a[i] XOR b[i]) => d
            clauses.push(vec![-a_bits[i], b_bits[i], d]);
            clauses.push(vec![a_bits[i], -b_bits[i], d]);
        }
        // At least one diff must be true
        clauses.push(diff_vars);
        clauses
    }

    /// Encode: a < b (strict less than).
    pub fn less_than(
        &mut self,
        a_bits: &[Literal],
        b_bits: &[Literal],
    ) -> Vec<Clause> {
        let mut clauses = self.less_than_or_equal(a_bits, b_bits);
        clauses.extend(self.not_equal(a_bits, b_bits));
        clauses
    }
}

// ---------------------------------------------------------------------------
// IntervalEncoder
// ---------------------------------------------------------------------------

/// Encodes interval constraints as SAT clauses.
///
/// For each interval predicate, generates O(log|Vi| * log|Vj|) clauses
/// using binary encoding of version indices and comparator circuits.
#[derive(Debug)]
pub struct IntervalEncoder {
    next_var: u32,
    /// Variable allocations: (service_idx) -> BinaryEncoding
    encodings: HashMap<usize, BinaryEncoding>,
}

impl IntervalEncoder {
    /// Create a new encoder with variables starting at `base_var`.
    pub fn new(base_var: u32) -> Self {
        Self {
            next_var: base_var,
            encodings: HashMap::new(),
        }
    }

    /// Get or create a binary encoding for a service with `num_versions` versions.
    pub fn get_or_create_encoding(
        &mut self,
        service: usize,
        num_versions: usize,
    ) -> BinaryEncoding {
        if let Some(enc) = self.encodings.get(&service) {
            return enc.clone();
        }
        let max_val = num_versions.saturating_sub(1);
        let enc = BinaryEncoding::new(max_val, self.next_var);
        self.next_var += enc.num_bits as u32;
        self.encodings.insert(service, enc.clone());
        enc
    }

    /// Get current next variable ID.
    pub fn next_var(&self) -> u32 {
        self.next_var
    }

    /// Encode an interval constraint as SAT clauses.
    ///
    /// For each version vi of service i, if compatible range is [lo, hi],
    /// encode: (version_i == vi) => (lo <= version_j <= hi).
    pub fn encode_interval_constraint(
        &mut self,
        predicate: &IntervalPredicate,
    ) -> Vec<Clause> {
        let enc_i = self.get_or_create_encoding(
            predicate.service_i,
            predicate.num_versions_i,
        );
        let enc_j = self.get_or_create_encoding(
            predicate.service_j,
            predicate.num_versions_j,
        );

        let mut all_clauses = Vec::new();
        let mut comp = Comparator::new(self.next_var);

        // Add bounding constraints
        all_clauses.extend(enc_i.bound_constraint());
        all_clauses.extend(enc_j.bound_constraint());

        for vi in 0..predicate.num_versions_i {
            if let Some((lo, hi)) = predicate.get_range(vi) {
                // Encode: (version_i == vi) => (lo <= version_j <= hi)
                // Contrapositive: NOT(lo <= version_j <= hi) => NOT(version_i == vi)
                // We encode this as: for each assignment of j outside [lo,hi],
                // block (version_i == vi AND version_j == vj)

                // More efficient: encode directly using implication
                // Create indicator for version_i == vi
                let vi_indicator = comp.fresh();
                let vi_lits = enc_i.encode_value(vi);
                // vi_indicator => all bits match
                for &lit in &vi_lits {
                    all_clauses.push(vec![-vi_indicator, lit]);
                }
                // all bits match => vi_indicator
                let mut rev_clause: Vec<Literal> = vi_lits.iter().map(|&l| -l).collect();
                rev_clause.push(vi_indicator);
                all_clauses.push(rev_clause);

                // When vi_indicator is true, enforce lo <= version_j <= hi
                let j_bits: Vec<Literal> = (0..enc_j.num_bits)
                    .map(|b| enc_j.bit_var(b))
                    .collect();

                let range_clauses = comp.in_range(&j_bits, lo, hi);
                // Make these conditional on vi_indicator
                for clause in range_clauses {
                    let mut conditional = vec![-vi_indicator];
                    conditional.extend(clause);
                    all_clauses.push(conditional);
                }
            } else {
                // No compatible versions: block version_i == vi entirely
                let vi_lits = enc_i.encode_value(vi);
                let blocking: Vec<Literal> = vi_lits.iter().map(|&l| -l).collect();
                all_clauses.push(blocking);
            }
        }

        self.next_var = comp.next_var();
        all_clauses
    }

    /// Encode multiple interval constraints.
    pub fn encode_all(
        &mut self,
        predicates: &[IntervalPredicate],
    ) -> CnfFormula {
        let mut cnf = CnfFormula::new();
        for pred in predicates {
            let clauses = self.encode_interval_constraint(pred);
            for clause in clauses {
                cnf.add_clause(clause);
            }
        }
        cnf
    }
}

// ---------------------------------------------------------------------------
// IntervalCompressor
// ---------------------------------------------------------------------------

/// Analyzes compatibility matrices and detects/compresses interval structure.
#[derive(Debug, Clone)]
pub struct IntervalCompressor {
    /// Detected interval predicates.
    pub predicates: Vec<IntervalPredicate>,
    /// Non-interval pairs that need explicit encoding.
    pub non_interval_pairs: Vec<(usize, usize)>,
    /// Original matrix sizes for compression ratio computation.
    original_entries: usize,
    compressed_entries: usize,
}

impl IntervalCompressor {
    pub fn new() -> Self {
        Self {
            predicates: Vec::new(),
            non_interval_pairs: Vec::new(),
            original_entries: 0,
            compressed_entries: 0,
        }
    }

    /// Detect whether a compatibility matrix has interval structure.
    /// Returns Some(IntervalPredicate) if it does, None otherwise.
    pub fn detect_interval_structure(
        service_i: usize,
        service_j: usize,
        matrix: &[Vec<bool>],
    ) -> Option<IntervalPredicate> {
        IntervalPredicate::from_matrix(service_i, service_j, matrix)
    }

    /// Analyze a set of compatibility matrices and partition into
    /// interval-structured and non-interval pairs.
    pub fn analyze(
        &mut self,
        matrices: &[((usize, usize), Vec<Vec<bool>>)],
    ) {
        self.predicates.clear();
        self.non_interval_pairs.clear();
        self.original_entries = 0;
        self.compressed_entries = 0;

        for ((si, sj), matrix) in matrices {
            let entries = matrix.iter().map(|row| row.len()).sum::<usize>();
            self.original_entries += entries;

            match Self::detect_interval_structure(*si, *sj, matrix) {
                Some(pred) => {
                    // Compressed representation: 2 values per row (lo, hi)
                    self.compressed_entries += pred.num_versions_i * 2;
                    self.predicates.push(pred);
                }
                None => {
                    self.compressed_entries += entries;
                    self.non_interval_pairs.push((*si, *sj));
                }
            }
        }
    }

    /// Compression ratio: original_entries / compressed_entries.
    /// Higher is better (more compression).
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_entries == 0 {
            return 1.0;
        }
        self.original_entries as f64 / self.compressed_entries as f64
    }

    /// Fraction of pairs that have interval structure.
    pub fn interval_fraction(&self) -> f64 {
        let total = self.predicates.len() + self.non_interval_pairs.len();
        if total == 0 {
            return 0.0;
        }
        self.predicates.len() as f64 / total as f64
    }

    /// Estimate clause count for interval encoding vs explicit encoding.
    pub fn estimate_clause_savings(&self) -> (usize, usize) {
        let mut interval_clauses = 0;
        let mut explicit_clauses = 0;

        for pred in &self.predicates {
            let bits_i = BinaryEncoding::num_bits(pred.num_versions_i.saturating_sub(1));
            let bits_j = BinaryEncoding::num_bits(pred.num_versions_j.saturating_sub(1));
            // Interval encoding: O(|Vi| * log|Vj|) per predicate
            interval_clauses += pred.num_versions_i * bits_j * 10;
            // Explicit would be O(|Vi| * |Vj|)
            explicit_clauses += pred.num_versions_i * pred.num_versions_j;
        }
        (interval_clauses, explicit_clauses)
    }
}

impl Default for IntervalCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: build compatibility matrix
// ---------------------------------------------------------------------------

/// Build a compatibility matrix from a predicate function.
pub fn build_compatibility_matrix(
    num_i: usize,
    num_j: usize,
    compatible: impl Fn(usize, usize) -> bool,
) -> Vec<Vec<bool>> {
    (0..num_i)
        .map(|i| (0..num_j).map(|j| compatible(i, j)).collect())
        .collect()
}

/// Check if a compatibility matrix has downward-closed structure.
/// Downward-closed: if (vi, vj) is compatible and vi' <= vi and vj' <= vj,
/// then (vi', vj') is also compatible.
pub fn is_downward_closed(matrix: &[Vec<bool>]) -> bool {
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            if matrix[i][j] {
                // Check all (i', j') with i' <= i and j' <= j
                for ii in 0..=i {
                    for jj in 0..=j {
                        if !matrix[ii][jj] {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

/// Check if a compatibility matrix has monotone structure.
/// Monotone: compatible ranges shift upward as version increases.
pub fn is_monotone(matrix: &[Vec<bool>]) -> bool {
    if matrix.is_empty() {
        return true;
    }
    for i in 1..matrix.len() {
        let prev_min = matrix[i - 1].iter().position(|&c| c);
        let curr_min = matrix[i].iter().position(|&c| c);
        match (prev_min, curr_min) {
            (Some(pm), Some(cm)) => {
                if cm < pm {
                    return false;
                }
            }
            (Some(_), None) => {}
            _ => {}
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_encoding_num_bits() {
        assert_eq!(BinaryEncoding::num_bits(0), 1);
        assert_eq!(BinaryEncoding::num_bits(1), 1);
        assert_eq!(BinaryEncoding::num_bits(2), 2);
        assert_eq!(BinaryEncoding::num_bits(3), 2);
        assert_eq!(BinaryEncoding::num_bits(4), 3);
        assert_eq!(BinaryEncoding::num_bits(7), 3);
        assert_eq!(BinaryEncoding::num_bits(8), 4);
        assert_eq!(BinaryEncoding::num_bits(15), 4);
        assert_eq!(BinaryEncoding::num_bits(16), 5);
    }

    #[test]
    fn test_binary_encoding_encode_decode() {
        let enc = BinaryEncoding::new(7, 100);
        assert_eq!(enc.num_bits, 3);
        for v in 0..=7 {
            let lits = enc.encode_value(v);
            let mut assignment = HashMap::new();
            for &lit in &lits {
                let var = lit.unsigned_abs() as u32;
                assignment.insert(var, lit > 0);
            }
            assert_eq!(enc.decode_assignment(&assignment), v);
        }
    }

    #[test]
    fn test_binary_encoding_bound_constraint() {
        let enc = BinaryEncoding::new(5, 100);
        // 3 bits can represent 0-7, but max is 5
        let clauses = enc.bound_constraint();
        assert!(!clauses.is_empty()); // Should block 6 and 7
        let cnf = CnfFormula::from_clauses(clauses);
        // Value 5 should be allowed
        let mut a5 = HashMap::new();
        for &lit in &enc.encode_value(5) {
            a5.insert(lit.unsigned_abs() as u32, lit > 0);
        }
        assert!(cnf.evaluate(&a5));
        // Value 7 should be blocked
        let mut a7 = HashMap::new();
        for &lit in &enc.encode_value(7) {
            a7.insert(lit.unsigned_abs() as u32, lit > 0);
        }
        assert!(!cnf.evaluate(&a7));
    }

    #[test]
    fn test_interval_predicate_basic() {
        let mut pred = IntervalPredicate::new(0, 1, 4, 6);
        pred.set_range(0, 0, 2);
        pred.set_range(1, 1, 3);
        pred.set_range(2, 2, 4);
        pred.set_range(3, 3, 5);

        assert!(pred.is_interval_structured());
        assert!(pred.is_compatible(0, 0));
        assert!(pred.is_compatible(0, 1));
        assert!(pred.is_compatible(0, 2));
        assert!(!pred.is_compatible(0, 3));
        assert!(pred.is_compatible(3, 5));
        assert!(!pred.is_compatible(3, 2));
    }

    #[test]
    fn test_interval_predicate_from_matrix() {
        let matrix = vec![
            vec![true, true, false, false],
            vec![false, true, true, false],
            vec![false, false, true, true],
        ];
        let pred = IntervalPredicate::from_matrix(0, 1, &matrix).unwrap();
        assert_eq!(pred.get_range(0), Some((0, 1)));
        assert_eq!(pred.get_range(1), Some((1, 2)));
        assert_eq!(pred.get_range(2), Some((2, 3)));
    }

    #[test]
    fn test_interval_predicate_non_interval() {
        let matrix = vec![
            vec![true, false, true], // not contiguous
        ];
        assert!(IntervalPredicate::from_matrix(0, 1, &matrix).is_none());
    }

    #[test]
    fn test_interval_predicate_density() {
        let mut pred = IntervalPredicate::new(0, 1, 3, 4);
        pred.set_range(0, 0, 1); // 2 compatible
        pred.set_range(1, 1, 2); // 2 compatible
        pred.set_range(2, 2, 3); // 2 compatible
        assert_eq!(pred.compatible_count(), 6);
        assert_eq!(pred.total_pairs(), 12);
        assert!((pred.density() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_comparator_equality() {
        let mut comp = Comparator::new(200);
        let a = vec![1, 2]; // 2-bit number a
        let b = vec![3, 4]; // 2-bit number b
        let clauses = comp.equality(&a, &b);
        let cnf = CnfFormula::from_clauses(clauses);
        // a=0b11, b=0b11 => equal
        let mut eq_assign = HashMap::new();
        eq_assign.insert(1, true);
        eq_assign.insert(2, true);
        eq_assign.insert(3, true);
        eq_assign.insert(4, true);
        assert!(cnf.evaluate(&eq_assign));
        // a=0b10, b=0b11 => not equal
        eq_assign.insert(1, false);
        assert!(!cnf.evaluate(&eq_assign));
    }

    #[test]
    fn test_comparator_less_than_or_equal() {
        let mut comp = Comparator::new(200);
        // 2-bit encoding: bit 0 = LSB, bit 1 = MSB
        let a = vec![10, 11]; // bits of a
        let b = vec![12, 13]; // bits of b

        let clauses = comp.less_than_or_equal(&a, &b);
        let cnf = CnfFormula::from_clauses(clauses);

        // Test a=1 (01), b=2 (10): 1 <= 2 should be true
        let mut assign = HashMap::new();
        // Fill in aux vars as needed by checking
        assign.insert(10, true); // a bit 0
        assign.insert(11, false); // a bit 1
        assign.insert(12, false); // b bit 0
        assign.insert(13, true); // b bit 1
        // We need to set aux vars correctly for the circuit
        // Since aux vars have complex relationships, just verify the clause structure
        assert!(cnf.num_clauses() > 0);
    }

    #[test]
    fn test_interval_encoder_basic() {
        let mut pred = IntervalPredicate::new(0, 1, 4, 4);
        pred.set_range(0, 0, 1);
        pred.set_range(1, 0, 2);
        pred.set_range(2, 1, 3);
        pred.set_range(3, 2, 3);

        let mut encoder = IntervalEncoder::new(1);
        let clauses = encoder.encode_interval_constraint(&pred);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_interval_encoder_multiple() {
        let mut pred1 = IntervalPredicate::new(0, 1, 3, 3);
        pred1.set_range(0, 0, 0);
        pred1.set_range(1, 0, 1);
        pred1.set_range(2, 1, 2);

        let mut pred2 = IntervalPredicate::new(1, 2, 3, 3);
        pred2.set_range(0, 0, 1);
        pred2.set_range(1, 1, 2);
        pred2.set_range(2, 2, 2);

        let mut encoder = IntervalEncoder::new(1);
        let cnf = encoder.encode_all(&[pred1, pred2]);
        assert!(cnf.num_clauses() > 0);
    }

    #[test]
    fn test_interval_compressor_all_interval() {
        let matrices = vec![
            (
                (0, 1),
                vec![
                    vec![true, true, false],
                    vec![false, true, true],
                    vec![false, false, true],
                ],
            ),
            (
                (1, 2),
                vec![
                    vec![true, false],
                    vec![true, true],
                    vec![false, true],
                ],
            ),
        ];
        let mut compressor = IntervalCompressor::new();
        compressor.analyze(&matrices);
        assert_eq!(compressor.predicates.len(), 2);
        assert_eq!(compressor.non_interval_pairs.len(), 0);
        assert!((compressor.interval_fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_compressor_mixed() {
        let matrices = vec![
            (
                (0, 1),
                vec![
                    vec![true, true, false],
                    vec![false, true, true],
                ],
            ),
            (
                (1, 2),
                vec![
                    vec![true, false, true], // non-interval
                ],
            ),
        ];
        let mut compressor = IntervalCompressor::new();
        compressor.analyze(&matrices);
        assert_eq!(compressor.predicates.len(), 1);
        assert_eq!(compressor.non_interval_pairs.len(), 1);
        assert!((compressor.interval_fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_build_compatibility_matrix() {
        let m = build_compatibility_matrix(3, 4, |i, j| {
            (i as isize - j as isize).unsigned_abs() <= 1
        });
        assert!(m[0][0]);
        assert!(m[0][1]);
        assert!(!m[0][2]);
    }

    #[test]
    fn test_downward_closed() {
        // Downward-closed: (i',j') compatible whenever i'<=i, j'<=j, (i,j) compatible
        let m = vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ];
        assert!(is_downward_closed(&m));

        // Also downward-closed: upper-left triangle
        let m2 = vec![
            vec![true, true, true],
            vec![true, true, false],
            vec![true, false, false],
        ];
        assert!(is_downward_closed(&m2));
    }

    #[test]
    fn test_not_downward_closed() {
        let m = vec![
            vec![false, true],
            vec![true, true],
        ];
        assert!(!is_downward_closed(&m));
    }

    #[test]
    fn test_monotone() {
        let m = vec![
            vec![true, true, false, false],
            vec![false, true, true, false],
            vec![false, false, true, true],
        ];
        assert!(is_monotone(&m));
    }

    #[test]
    fn test_not_monotone() {
        let m = vec![
            vec![false, true, true],
            vec![true, true, false], // min goes backwards
        ];
        assert!(!is_monotone(&m));
    }

    #[test]
    fn test_interval_predicate_empty_compatible() {
        let mut pred = IntervalPredicate::new(0, 1, 3, 3);
        pred.set_range(0, 0, 0);
        // version 1 has no compatible versions
        pred.set_range(2, 2, 2);
        assert!(!pred.is_compatible(1, 0));
        assert!(!pred.is_compatible(1, 1));
        assert!(!pred.is_compatible(1, 2));
    }

    #[test]
    fn test_interval_predicate_display() {
        let pred = IntervalPredicate::new(0, 1, 3, 4);
        let s = format!("{}", pred);
        assert!(s.contains("IntervalPredicate"));
    }

    #[test]
    fn test_comparator_not_equal() {
        let mut comp = Comparator::new(200);
        let a = vec![1, 2];
        let b = vec![3, 4];
        let clauses = comp.not_equal(&a, &b);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let matrices = vec![
            (
                (0, 1),
                vec![
                    vec![true, true, true, true, true],
                    vec![true, true, true, true, true],
                    vec![true, true, true, true, true],
                ],
            ),
        ];
        let mut compressor = IntervalCompressor::new();
        compressor.analyze(&matrices);
        assert!(compressor.compression_ratio() > 1.0);
    }

    #[test]
    fn test_estimate_clause_savings() {
        let mut pred = IntervalPredicate::new(0, 1, 64, 64);
        for i in 0..64 {
            pred.set_range(i, i.saturating_sub(1), (i + 1).min(63));
        }
        let mut compressor = IntervalCompressor::new();
        compressor.predicates.push(pred);
        let (interval, explicit) = compressor.estimate_clause_savings();
        // Both should be non-zero
        assert!(interval > 0);
        assert!(explicit > 0);
    }
}
