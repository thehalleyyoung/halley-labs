// WFA-to-AIR Circuit Compiler
//
// Compiles Weighted Finite Automata (WFA) into Algebraic Intermediate Representation
// (AIR) constraints suitable for STARK proof generation. Supports multiple semiring
// types with tiered compilation strategies:
//   - Tier 1: Algebraic direct compilation (counting, boolean semirings)
//   - Tier 2: Gadget-assisted compilation (tropical, real semirings)
//   - Hybrid: Mixed strategy with automatic tier selection

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

use super::goldilocks::GoldilocksField;

// ═══════════════════════════════════════════════════════════════════════════════
// AIR Types (defined locally; will migrate to air.rs)
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of column in the execution trace.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ColumnType {
    /// Main witness column
    Witness,
    /// Public input column
    PublicInput,
    /// Public output column
    PublicOutput,
    /// Auxiliary/intermediate computation column
    Auxiliary,
    /// Selector/indicator column
    Selector,
    /// Permutation argument column
    Permutation,
}

/// A column in the execution trace.
#[derive(Clone, Debug)]
pub struct TraceColumn {
    pub name: String,
    pub index: usize,
    pub column_type: ColumnType,
    pub description: String,
}

/// Layout of the execution trace (column assignments and dimensions).
#[derive(Clone, Debug)]
pub struct TraceLayout {
    pub columns: Vec<TraceColumn>,
    pub num_rows: usize,
    pub num_witness_columns: usize,
    pub num_public_columns: usize,
    pub num_auxiliary_columns: usize,
    pub num_selector_columns: usize,
}

impl TraceLayout {
    pub fn new() -> Self {
        TraceLayout {
            columns: Vec::new(),
            num_rows: 0,
            num_witness_columns: 0,
            num_public_columns: 0,
            num_auxiliary_columns: 0,
            num_selector_columns: 0,
        }
    }

    pub fn add_column(&mut self, name: &str, col_type: ColumnType, description: &str) -> usize {
        let index = self.columns.len();
        match col_type {
            ColumnType::Witness => self.num_witness_columns += 1,
            ColumnType::PublicInput | ColumnType::PublicOutput => self.num_public_columns += 1,
            ColumnType::Auxiliary => self.num_auxiliary_columns += 1,
            ColumnType::Selector => self.num_selector_columns += 1,
            ColumnType::Permutation => self.num_auxiliary_columns += 1,
        }
        self.columns.push(TraceColumn {
            name: name.to_string(),
            index,
            column_type: col_type,
            description: description.to_string(),
        });
        index
    }

    pub fn total_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn find_column(&self, name: &str) -> Option<usize> {
        self.columns.iter().find(|c| c.name == name).map(|c| c.index)
    }

    pub fn columns_of_type(&self, col_type: &ColumnType) -> Vec<usize> {
        self.columns
            .iter()
            .filter(|c| &c.column_type == col_type)
            .map(|c| c.index)
            .collect()
    }
}

/// Type of AIR constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    /// Boundary constraint: holds at a specific row
    Boundary,
    /// Transition constraint: holds between consecutive rows
    Transition,
    /// Periodic constraint: holds every k rows
    Periodic(usize),
    /// Global constraint: holds across all rows
    Global,
}

/// A symbolic expression used in constraint polynomials.
#[derive(Clone, Debug)]
pub enum SymbolicExpression {
    /// Constant field element
    Constant(GoldilocksField),
    /// Reference to trace column at current row
    Column(usize),
    /// Reference to trace column at next row (row + 1)
    NextColumn(usize),
    /// Reference to trace column at offset
    ColumnOffset(usize, i32),
    /// Sum of two expressions
    Add(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Product of two expressions
    Mul(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Difference of two expressions
    Sub(Box<SymbolicExpression>, Box<SymbolicExpression>),
    /// Negation
    Neg(Box<SymbolicExpression>),
    /// Power (expr, exponent)
    Pow(Box<SymbolicExpression>, u32),
    /// Named variable (for readability)
    Variable(String),
}

impl SymbolicExpression {
    pub fn constant(v: u64) -> Self {
        SymbolicExpression::Constant(GoldilocksField::new(v))
    }

    pub fn constant_field(v: GoldilocksField) -> Self {
        SymbolicExpression::Constant(v)
    }

    pub fn col(index: usize) -> Self {
        SymbolicExpression::Column(index)
    }

    pub fn next_col(index: usize) -> Self {
        SymbolicExpression::NextColumn(index)
    }

    pub fn col_offset(index: usize, offset: i32) -> Self {
        SymbolicExpression::ColumnOffset(index, offset)
    }

    pub fn add(self, other: Self) -> Self {
        SymbolicExpression::Add(Box::new(self), Box::new(other))
    }

    pub fn sub(self, other: Self) -> Self {
        SymbolicExpression::Sub(Box::new(self), Box::new(other))
    }

    pub fn mul(self, other: Self) -> Self {
        SymbolicExpression::Mul(Box::new(self), Box::new(other))
    }

    pub fn neg(self) -> Self {
        SymbolicExpression::Neg(Box::new(self))
    }

    pub fn pow(self, exp: u32) -> Self {
        SymbolicExpression::Pow(Box::new(self), exp)
    }

    pub fn is_zero_constant(&self) -> bool {
        match self {
            SymbolicExpression::Constant(v) => v.to_canonical() == 0,
            _ => false,
        }
    }

    pub fn is_one_constant(&self) -> bool {
        match self {
            SymbolicExpression::Constant(v) => v.to_canonical() == 1,
            _ => false,
        }
    }

    /// Compute the algebraic degree of this expression.
    pub fn degree(&self) -> usize {
        match self {
            SymbolicExpression::Constant(_) => 0,
            SymbolicExpression::Column(_)
            | SymbolicExpression::NextColumn(_)
            | SymbolicExpression::ColumnOffset(_, _)
            | SymbolicExpression::Variable(_) => 1,
            SymbolicExpression::Add(a, b) | SymbolicExpression::Sub(a, b) => {
                a.degree().max(b.degree())
            }
            SymbolicExpression::Mul(a, b) => a.degree() + b.degree(),
            SymbolicExpression::Neg(a) => a.degree(),
            SymbolicExpression::Pow(a, exp) => a.degree() * (*exp as usize),
        }
    }

    /// Evaluate the expression given a row of trace values and (optionally) a next row.
    pub fn evaluate(
        &self,
        current_row: &[GoldilocksField],
        next_row: Option<&[GoldilocksField]>,
        all_rows: Option<&[Vec<GoldilocksField>]>,
        current_row_idx: usize,
    ) -> GoldilocksField {
        match self {
            SymbolicExpression::Constant(v) => *v,
            SymbolicExpression::Column(idx) => current_row[*idx],
            SymbolicExpression::NextColumn(idx) => {
                next_row.map(|r| r[*idx]).unwrap_or(GoldilocksField::ZERO)
            }
            SymbolicExpression::ColumnOffset(idx, offset) => {
                if let Some(rows) = all_rows {
                    let target = current_row_idx as i64 + *offset as i64;
                    if target >= 0 && (target as usize) < rows.len() {
                        rows[target as usize][*idx]
                    } else {
                        GoldilocksField::ZERO
                    }
                } else {
                    GoldilocksField::ZERO
                }
            }
            SymbolicExpression::Add(a, b) => {
                let va = a.evaluate(current_row, next_row, all_rows, current_row_idx);
                let vb = b.evaluate(current_row, next_row, all_rows, current_row_idx);
                va.add_elem(vb)
            }
            SymbolicExpression::Sub(a, b) => {
                let va = a.evaluate(current_row, next_row, all_rows, current_row_idx);
                let vb = b.evaluate(current_row, next_row, all_rows, current_row_idx);
                va.sub_elem(vb)
            }
            SymbolicExpression::Mul(a, b) => {
                let va = a.evaluate(current_row, next_row, all_rows, current_row_idx);
                let vb = b.evaluate(current_row, next_row, all_rows, current_row_idx);
                va.mul_elem(vb)
            }
            SymbolicExpression::Neg(a) => {
                let va = a.evaluate(current_row, next_row, all_rows, current_row_idx);
                va.neg_elem()
            }
            SymbolicExpression::Pow(a, exp) => {
                let va = a.evaluate(current_row, next_row, all_rows, current_row_idx);
                va.pow(*exp as u64)
            }
            SymbolicExpression::Variable(_) => GoldilocksField::ZERO,
        }
    }

    /// Collect all column indices referenced in this expression.
    pub fn referenced_columns(&self) -> HashSet<usize> {
        let mut cols = HashSet::new();
        self.collect_columns(&mut cols);
        cols
    }

    fn collect_columns(&self, cols: &mut HashSet<usize>) {
        match self {
            SymbolicExpression::Constant(_) | SymbolicExpression::Variable(_) => {}
            SymbolicExpression::Column(idx)
            | SymbolicExpression::NextColumn(idx)
            | SymbolicExpression::ColumnOffset(idx, _) => {
                cols.insert(*idx);
            }
            SymbolicExpression::Add(a, b)
            | SymbolicExpression::Sub(a, b)
            | SymbolicExpression::Mul(a, b) => {
                a.collect_columns(cols);
                b.collect_columns(cols);
            }
            SymbolicExpression::Neg(a) | SymbolicExpression::Pow(a, _) => {
                a.collect_columns(cols);
            }
        }
    }

    /// Structurally hash the expression for CSE.
    pub fn structural_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        self.hash_recursive(&mut hasher);
        hasher.finish()
    }

    fn hash_recursive(&self, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        match self {
            SymbolicExpression::Constant(v) => {
                0u8.hash(hasher);
                v.to_canonical().hash(hasher);
            }
            SymbolicExpression::Column(idx) => {
                1u8.hash(hasher);
                idx.hash(hasher);
            }
            SymbolicExpression::NextColumn(idx) => {
                2u8.hash(hasher);
                idx.hash(hasher);
            }
            SymbolicExpression::ColumnOffset(idx, off) => {
                3u8.hash(hasher);
                idx.hash(hasher);
                off.hash(hasher);
            }
            SymbolicExpression::Add(a, b) => {
                4u8.hash(hasher);
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
            SymbolicExpression::Mul(a, b) => {
                5u8.hash(hasher);
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
            SymbolicExpression::Sub(a, b) => {
                6u8.hash(hasher);
                a.hash_recursive(hasher);
                b.hash_recursive(hasher);
            }
            SymbolicExpression::Neg(a) => {
                7u8.hash(hasher);
                a.hash_recursive(hasher);
            }
            SymbolicExpression::Pow(a, exp) => {
                8u8.hash(hasher);
                a.hash_recursive(hasher);
                exp.hash(hasher);
            }
            SymbolicExpression::Variable(name) => {
                9u8.hash(hasher);
                name.hash(hasher);
            }
        }
    }
}

/// An AIR constraint (polynomial identity that must hold on the trace).
#[derive(Clone, Debug)]
pub struct AIRConstraint {
    pub expression: SymbolicExpression,
    pub constraint_type: ConstraintType,
    pub label: String,
    pub degree: usize,
    /// For boundary constraints, which row
    pub boundary_row: Option<usize>,
}

impl AIRConstraint {
    pub fn boundary(expr: SymbolicExpression, row: usize, label: &str) -> Self {
        let degree = expr.degree();
        AIRConstraint {
            expression: expr,
            constraint_type: ConstraintType::Boundary,
            label: label.to_string(),
            degree,
            boundary_row: Some(row),
        }
    }

    pub fn transition(expr: SymbolicExpression, label: &str) -> Self {
        let degree = expr.degree();
        AIRConstraint {
            expression: expr,
            constraint_type: ConstraintType::Transition,
            label: label.to_string(),
            degree,
            boundary_row: None,
        }
    }

    pub fn periodic(expr: SymbolicExpression, period: usize, label: &str) -> Self {
        let degree = expr.degree();
        AIRConstraint {
            expression: expr,
            constraint_type: ConstraintType::Periodic(period),
            label: label.to_string(),
            degree,
            boundary_row: None,
        }
    }

    pub fn global(expr: SymbolicExpression, label: &str) -> Self {
        let degree = expr.degree();
        AIRConstraint {
            expression: expr,
            constraint_type: ConstraintType::Global,
            label: label.to_string(),
            degree,
            boundary_row: None,
        }
    }

    /// Check if this constraint is satisfied at a given row.
    pub fn check_at_row(
        &self,
        current_row: &[GoldilocksField],
        next_row: Option<&[GoldilocksField]>,
        all_rows: Option<&[Vec<GoldilocksField>]>,
        row_idx: usize,
    ) -> bool {
        let val = self.expression.evaluate(current_row, next_row, all_rows, row_idx);
        val.to_canonical() == 0
    }
}

/// A complete AIR program.
#[derive(Clone, Debug)]
pub struct AIRProgram {
    pub constraints: Vec<AIRConstraint>,
    pub trace_layout: TraceLayout,
    pub trace_length: usize,
    pub num_public_inputs: usize,
    pub max_constraint_degree: usize,
}

impl AIRProgram {
    pub fn new(trace_layout: TraceLayout, trace_length: usize) -> Self {
        AIRProgram {
            constraints: Vec::new(),
            trace_layout,
            trace_length,
            num_public_inputs: 0,
            max_constraint_degree: 0,
        }
    }

    pub fn add_constraint(&mut self, constraint: AIRConstraint) {
        if constraint.degree > self.max_constraint_degree {
            self.max_constraint_degree = constraint.degree;
        }
        self.constraints.push(constraint);
    }

    pub fn boundary_constraints(&self) -> Vec<&AIRConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.constraint_type == ConstraintType::Boundary)
            .collect()
    }

    pub fn transition_constraints(&self) -> Vec<&AIRConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.constraint_type == ConstraintType::Transition)
            .collect()
    }
}

/// Execution trace as a 2D array of field elements.
#[derive(Clone, Debug)]
pub struct AIRTrace {
    /// trace[row][column]
    pub values: Vec<Vec<GoldilocksField>>,
    pub num_rows: usize,
    pub num_columns: usize,
}

impl AIRTrace {
    pub fn new(num_rows: usize, num_columns: usize) -> Self {
        AIRTrace {
            values: vec![vec![GoldilocksField::ZERO; num_columns]; num_rows],
            num_rows,
            num_columns,
        }
    }

    pub fn set(&mut self, row: usize, col: usize, val: GoldilocksField) {
        self.values[row][col] = val;
    }

    pub fn get(&self, row: usize, col: usize) -> GoldilocksField {
        self.values[row][col]
    }

    pub fn row(&self, row: usize) -> &[GoldilocksField] {
        &self.values[row]
    }

    pub fn column(&self, col: usize) -> Vec<GoldilocksField> {
        self.values.iter().map(|row| row[col]).collect()
    }

    /// Verify that all constraints in the AIR program are satisfied.
    pub fn verify_constraints(&self, program: &AIRProgram) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();
        for constraint in &program.constraints {
            match &constraint.constraint_type {
                ConstraintType::Boundary => {
                    if let Some(row) = constraint.boundary_row {
                        if row < self.num_rows {
                            let next = if row + 1 < self.num_rows {
                                Some(self.values[row + 1].as_slice())
                            } else {
                                None
                            };
                            if !constraint.check_at_row(
                                &self.values[row],
                                next,
                                Some(&self.values),
                                row,
                            ) {
                                let val = constraint.expression.evaluate(
                                    &self.values[row],
                                    next,
                                    Some(&self.values),
                                    row,
                                );
                                violations.push(format!(
                                    "Boundary '{}' violated at row {}: got {}",
                                    constraint.label,
                                    row,
                                    val.to_canonical()
                                ));
                            }
                        }
                    }
                }
                ConstraintType::Transition => {
                    for row in 0..self.num_rows.saturating_sub(1) {
                        let next = &self.values[row + 1];
                        if !constraint.check_at_row(
                            &self.values[row],
                            Some(next),
                            Some(&self.values),
                            row,
                        ) {
                            let val = constraint.expression.evaluate(
                                &self.values[row],
                                Some(next),
                                Some(&self.values),
                                row,
                            );
                            violations.push(format!(
                                "Transition '{}' violated at row {}: got {}",
                                constraint.label,
                                row,
                                val.to_canonical()
                            ));
                        }
                    }
                }
                ConstraintType::Periodic(period) => {
                    let mut row = 0;
                    while row < self.num_rows {
                        let next = if row + 1 < self.num_rows {
                            Some(self.values[row + 1].as_slice())
                        } else {
                            None
                        };
                        if !constraint.check_at_row(
                            &self.values[row],
                            next,
                            Some(&self.values),
                            row,
                        ) {
                            violations.push(format!(
                                "Periodic '{}' violated at row {}",
                                constraint.label, row
                            ));
                        }
                        row += period;
                    }
                }
                ConstraintType::Global => {
                    for row in 0..self.num_rows {
                        let next = if row + 1 < self.num_rows {
                            Some(self.values[row + 1].as_slice())
                        } else {
                            None
                        };
                        if !constraint.check_at_row(
                            &self.values[row],
                            next,
                            Some(&self.values),
                            row,
                        ) {
                            violations.push(format!(
                                "Global '{}' violated at row {}",
                                constraint.label, row
                            ));
                        }
                    }
                }
            }
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gadget Types (defined locally; will migrate to gadgets.rs)
// ═══════════════════════════════════════════════════════════════════════════════

/// Gadget for comparing two field elements: proves a < b via bit decomposition.
#[derive(Clone, Debug)]
pub struct ComparisonGadget {
    pub num_bits: usize,
    pub a_column: usize,
    pub b_column: usize,
    pub bit_columns: Vec<usize>,
    pub diff_column: usize,
}

impl ComparisonGadget {
    pub fn new(num_bits: usize, a_col: usize, b_col: usize, bit_cols: Vec<usize>, diff_col: usize) -> Self {
        ComparisonGadget {
            num_bits,
            a_column: a_col,
            b_column: b_col,
            bit_columns: bit_cols,
            diff_column: diff_col,
        }
    }

    /// Generate constraints that enforce a < b using bit decomposition.
    /// diff = b - a
    /// diff = sum_{i=0}^{n-1} bit_i * 2^i
    /// each bit_i in {0, 1}
    pub fn constraints(&self) -> Vec<AIRConstraint> {
        let mut cs = Vec::new();

        // Constraint: diff = b - a
        let diff_expr = SymbolicExpression::col(self.b_column)
            .sub(SymbolicExpression::col(self.a_column))
            .sub(SymbolicExpression::col(self.diff_column));
        cs.push(AIRConstraint::transition(diff_expr, "comparison_diff"));

        // Constraint: each bit is boolean: bit_i * (1 - bit_i) = 0
        for (i, &bit_col) in self.bit_columns.iter().enumerate() {
            let bit = SymbolicExpression::col(bit_col);
            let bool_expr = bit.clone().mul(
                SymbolicExpression::constant(1).sub(bit),
            );
            cs.push(AIRConstraint::transition(
                bool_expr,
                &format!("comparison_bit_{}_bool", i),
            ));
        }

        // Constraint: diff = sum of bit_i * 2^i
        let mut recompose = SymbolicExpression::constant(0);
        for (i, &bit_col) in self.bit_columns.iter().enumerate() {
            let power = 1u64 << i;
            let term = SymbolicExpression::col(bit_col)
                .mul(SymbolicExpression::constant(power));
            recompose = recompose.add(term);
        }
        let recompose_check = SymbolicExpression::col(self.diff_column).sub(recompose);
        cs.push(AIRConstraint::transition(recompose_check, "comparison_recompose"));

        cs
    }
}

/// Gadget for range-checking a value is in [0, 2^n).
#[derive(Clone, Debug)]
pub struct RangeCheckGadget {
    pub num_bits: usize,
    pub value_column: usize,
    pub bit_columns: Vec<usize>,
}

impl RangeCheckGadget {
    pub fn new(num_bits: usize, value_col: usize, bit_cols: Vec<usize>) -> Self {
        RangeCheckGadget {
            num_bits,
            value_column: value_col,
            bit_columns: bit_cols,
        }
    }

    pub fn constraints(&self) -> Vec<AIRConstraint> {
        let mut cs = Vec::new();

        // Each bit is boolean
        for (i, &bit_col) in self.bit_columns.iter().enumerate() {
            let bit = SymbolicExpression::col(bit_col);
            let bool_expr = bit.clone().mul(
                SymbolicExpression::constant(1).sub(bit),
            );
            cs.push(AIRConstraint::transition(
                bool_expr,
                &format!("range_bit_{}_bool", i),
            ));
        }

        // Value = sum of bits * powers of 2
        let mut recompose = SymbolicExpression::constant(0);
        for (i, &bit_col) in self.bit_columns.iter().enumerate() {
            let power = 1u64 << i;
            let term = SymbolicExpression::col(bit_col)
                .mul(SymbolicExpression::constant(power));
            recompose = recompose.add(term);
        }
        let check = SymbolicExpression::col(self.value_column).sub(recompose);
        cs.push(AIRConstraint::transition(check, "range_recompose"));

        cs
    }
}

/// Gadget for boolean constraints.
#[derive(Clone, Debug)]
pub struct BooleanGadget {
    pub column: usize,
}

impl BooleanGadget {
    pub fn new(col: usize) -> Self {
        BooleanGadget { column: col }
    }

    pub fn constraints(&self) -> Vec<AIRConstraint> {
        // x * (1 - x) = 0
        let x = SymbolicExpression::col(self.column);
        let expr = x.clone().mul(SymbolicExpression::constant(1).sub(x));
        vec![AIRConstraint::transition(expr, "boolean_check")]
    }
}

/// Collection of gadget constraints.
#[derive(Clone, Debug)]
pub struct GadgetConstraints {
    pub comparison_gadgets: Vec<ComparisonGadget>,
    pub range_check_gadgets: Vec<RangeCheckGadget>,
    pub boolean_gadgets: Vec<BooleanGadget>,
}

impl GadgetConstraints {
    pub fn new() -> Self {
        GadgetConstraints {
            comparison_gadgets: Vec::new(),
            range_check_gadgets: Vec::new(),
            boolean_gadgets: Vec::new(),
        }
    }

    pub fn all_constraints(&self) -> Vec<AIRConstraint> {
        let mut cs = Vec::new();
        for g in &self.comparison_gadgets {
            cs.extend(g.constraints());
        }
        for g in &self.range_check_gadgets {
            cs.extend(g.constraints());
        }
        for g in &self.boolean_gadgets {
            cs.extend(g.constraints());
        }
        cs
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Semiring Definitions
// ═══════════════════════════════════════════════════════════════════════════════

/// Semiring trait for WFA weights.
pub trait Semiring: Clone + std::fmt::Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
}

/// Counting semiring: (N, +, x, 0, 1)
#[derive(Clone, Debug, PartialEq)]
pub struct CountingSemiring(pub u64);

impl Semiring for CountingSemiring {
    fn zero() -> Self { CountingSemiring(0) }
    fn one() -> Self { CountingSemiring(1) }
    fn add(&self, other: &Self) -> Self { CountingSemiring(self.0.wrapping_add(other.0)) }
    fn mul(&self, other: &Self) -> Self { CountingSemiring(self.0.wrapping_mul(other.0)) }
}

/// Boolean semiring: ({0,1}, or, and, 0, 1)
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanSemiring(pub bool);

impl Semiring for BooleanSemiring {
    fn zero() -> Self { BooleanSemiring(false) }
    fn one() -> Self { BooleanSemiring(true) }
    fn add(&self, other: &Self) -> Self { BooleanSemiring(self.0 || other.0) }
    fn mul(&self, other: &Self) -> Self { BooleanSemiring(self.0 && other.0) }
}

/// Tropical semiring: (R u {inf}, min, +, inf, 0)
#[derive(Clone, Debug, PartialEq)]
pub struct TropicalSemiring(pub f64);

impl Semiring for TropicalSemiring {
    fn zero() -> Self { TropicalSemiring(f64::INFINITY) }
    fn one() -> Self { TropicalSemiring(0.0) }
    fn add(&self, other: &Self) -> Self { TropicalSemiring(self.0.min(other.0)) }
    fn mul(&self, other: &Self) -> Self { TropicalSemiring(self.0 + other.0) }
}

/// Real semiring: (R, +, x, 0, 1)
#[derive(Clone, Debug, PartialEq)]
pub struct RealSemiring(pub f64);

impl Semiring for RealSemiring {
    fn zero() -> Self { RealSemiring(0.0) }
    fn one() -> Self { RealSemiring(1.0) }
    fn add(&self, other: &Self) -> Self { RealSemiring(self.0 + other.0) }
    fn mul(&self, other: &Self) -> Self { RealSemiring(self.0 * other.0) }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WFA Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A single WFA transition: from_state --symbol/weight--> to_state
#[derive(Clone, Debug)]
pub struct WFATransition<S: Semiring> {
    pub from_state: usize,
    pub to_state: usize,
    pub symbol: u8,
    pub weight: S,
}

/// Weighted Finite Automaton parameterized by semiring.
#[derive(Clone, Debug)]
pub struct WFA<S: Semiring> {
    pub num_states: usize,
    pub alphabet_size: usize,
    pub transitions: Vec<WFATransition<S>>,
    pub initial_weights: Vec<S>,
    pub final_weights: Vec<S>,
}

impl<S: Semiring> WFA<S> {
    pub fn new(num_states: usize, alphabet_size: usize) -> Self {
        WFA {
            num_states,
            alphabet_size,
            transitions: Vec::new(),
            initial_weights: vec![S::zero(); num_states],
            final_weights: vec![S::zero(); num_states],
        }
    }

    pub fn add_transition(&mut self, from: usize, to: usize, symbol: u8, weight: S) {
        self.transitions.push(WFATransition {
            from_state: from,
            to_state: to,
            symbol,
            weight,
        });
    }

    pub fn set_initial(&mut self, state: usize, weight: S) {
        self.initial_weights[state] = weight;
    }

    pub fn set_final(&mut self, state: usize, weight: S) {
        self.final_weights[state] = weight;
    }

    /// Get transitions for a specific symbol.
    pub fn transitions_for_symbol(&self, symbol: u8) -> Vec<&WFATransition<S>> {
        self.transitions.iter().filter(|t| t.symbol == symbol).collect()
    }

    /// Get transitions from a specific state on a specific symbol.
    pub fn transitions_from(&self, state: usize, symbol: u8) -> Vec<&WFATransition<S>> {
        self.transitions.iter().filter(|t| t.from_state == state && t.symbol == symbol).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Semiring Embedding: Semiring -> GoldilocksField
// ═══════════════════════════════════════════════════════════════════════════════

/// Fixed-point scaling factor for encoding floating-point values in the field.
const FIXED_POINT_SCALE: u64 = 1u64 << 32;
/// Maximum magnitude before overflow in fixed-point representation.
const FIXED_POINT_MAX: f64 = (1u64 << 31) as f64;

/// Trait for embedding semiring elements into the Goldilocks field.
pub trait SemiringEmbedding: Semiring {
    /// Embed a semiring element as a field element.
    fn embed(&self) -> GoldilocksField;
    /// Attempt to extract a semiring element from a field element.
    fn extract(field_val: GoldilocksField) -> Self;
    /// Whether addition in this semiring corresponds to field addition.
    fn addition_is_field_addition() -> bool;
    /// Whether multiplication in this semiring corresponds to field multiplication.
    fn multiplication_is_field_multiplication() -> bool;
    /// Whether this semiring can be compiled purely algebraically.
    fn is_algebraically_embeddable() -> bool {
        Self::addition_is_field_addition() && Self::multiplication_is_field_multiplication()
    }
}

impl SemiringEmbedding for CountingSemiring {
    fn embed(&self) -> GoldilocksField { GoldilocksField::new(self.0) }
    fn extract(field_val: GoldilocksField) -> Self { CountingSemiring(field_val.to_canonical()) }
    fn addition_is_field_addition() -> bool { true }
    fn multiplication_is_field_multiplication() -> bool { true }
}

impl SemiringEmbedding for BooleanSemiring {
    fn embed(&self) -> GoldilocksField {
        if self.0 { GoldilocksField::ONE } else { GoldilocksField::ZERO }
    }
    fn extract(field_val: GoldilocksField) -> Self { BooleanSemiring(field_val.to_canonical() != 0) }
    fn addition_is_field_addition() -> bool { false }
    fn multiplication_is_field_multiplication() -> bool { true }
    fn is_algebraically_embeddable() -> bool { true }
}

impl SemiringEmbedding for TropicalSemiring {
    fn embed(&self) -> GoldilocksField { tropical_to_field(self.0) }
    fn extract(field_val: GoldilocksField) -> Self { TropicalSemiring(field_from_tropical(field_val)) }
    fn addition_is_field_addition() -> bool { false }
    fn multiplication_is_field_multiplication() -> bool { false }
}

impl SemiringEmbedding for RealSemiring {
    fn embed(&self) -> GoldilocksField { real_to_field(self.0) }
    fn extract(field_val: GoldilocksField) -> Self { RealSemiring(field_from_real(field_val)) }
    fn addition_is_field_addition() -> bool { true }
    fn multiplication_is_field_multiplication() -> bool { false }
}

/// Encode a tropical (f64) value into a field element using fixed-point.
fn tropical_to_field(val: f64) -> GoldilocksField {
    if val.is_infinite() && val > 0.0 {
        GoldilocksField::new(GoldilocksField::MODULUS - 1)
    } else if val.is_nan() {
        GoldilocksField::ZERO
    } else {
        let clamped = val.clamp(-FIXED_POINT_MAX, FIXED_POINT_MAX);
        let scaled = (clamped * FIXED_POINT_SCALE as f64) as i64;
        if scaled >= 0 {
            GoldilocksField::new(scaled as u64)
        } else {
            let pos = (-scaled) as u64;
            GoldilocksField::new(GoldilocksField::MODULUS - pos)
        }
    }
}

/// Decode a field element back to tropical f64.
fn field_from_tropical(val: GoldilocksField) -> f64 {
    let v = val.to_canonical();
    if v == GoldilocksField::MODULUS - 1 {
        f64::INFINITY
    } else if v <= GoldilocksField::MODULUS / 2 {
        v as f64 / FIXED_POINT_SCALE as f64
    } else {
        -((GoldilocksField::MODULUS - v) as f64 / FIXED_POINT_SCALE as f64)
    }
}

/// Encode a real f64 value into a field element using fixed-point.
fn real_to_field(val: f64) -> GoldilocksField {
    let clamped = val.clamp(-FIXED_POINT_MAX, FIXED_POINT_MAX);
    let scaled = (clamped * FIXED_POINT_SCALE as f64) as i64;
    if scaled >= 0 {
        GoldilocksField::new(scaled as u64)
    } else {
        let pos = (-scaled) as u64;
        GoldilocksField::new(GoldilocksField::MODULUS - pos)
    }
}

/// Decode a field element back to real f64.
fn field_from_real(val: GoldilocksField) -> f64 {
    let v = val.to_canonical();
    if v <= GoldilocksField::MODULUS / 2 {
        v as f64 / FIXED_POINT_SCALE as f64
    } else {
        -((GoldilocksField::MODULUS - v) as f64 / FIXED_POINT_SCALE as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compilation Target and Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Compilation strategy for WFA-to-AIR.
#[derive(Clone, Debug, PartialEq)]
pub enum CompilationTarget {
    /// Direct algebraic compilation: semiring ops map to field ops.
    AlgebraicDirect,
    /// Gadget-assisted: uses comparison/range-check gadgets for non-algebraic ops.
    GadgetAssisted,
    /// Hybrid: algebraic where possible, gadgets where needed.
    Hybrid,
}

/// Strategy for padding the trace to a power-of-two length.
#[derive(Clone, Debug, PartialEq)]
pub enum TracePaddingStrategy {
    /// Repeat the last valid row
    RepeatLast,
    /// Fill with zeros
    ZeroPad,
    /// Fill with a specific field element
    ConstantPad(u64),
    /// Reflect (mirror) the trace
    ReflectPad,
}

/// Compiler configuration.
#[derive(Clone, Debug)]
pub struct CompilerConfig {
    pub target: CompilationTarget,
    pub optimization_level: u32,
    pub max_constraint_degree: usize,
    pub enable_cse: bool,
    pub enable_dead_elimination: bool,
    pub enable_batching: bool,
    pub trace_padding_strategy: TracePaddingStrategy,
    pub fixed_point_bits: usize,
    pub comparison_bits: usize,
    pub debug_labels: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        CompilerConfig {
            target: CompilationTarget::Hybrid,
            optimization_level: 1,
            max_constraint_degree: 4,
            enable_cse: true,
            enable_dead_elimination: true,
            enable_batching: false,
            trace_padding_strategy: TracePaddingStrategy::ZeroPad,
            fixed_point_bits: 32,
            comparison_bits: 32,
            debug_labels: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Intermediate Representation (IR)
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of IR variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IRVariableType {
    State,
    Weight,
    Input,
    Auxiliary,
    Public,
}

/// A variable in the intermediate representation.
#[derive(Clone, Debug)]
pub struct IRVariable {
    pub name: String,
    pub index: usize,
    pub variable_type: IRVariableType,
    pub trace_column: Option<usize>,
}

/// An IR constraint: a symbolic expression that must equal zero.
#[derive(Clone, Debug)]
pub struct IRConstraint {
    pub expression: SymbolicExpression,
    pub constraint_type: ConstraintType,
    pub source_description: String,
}

/// Auxiliary computation: computes a value from other values.
#[derive(Clone, Debug)]
pub struct AuxiliaryComputation {
    pub target_variable: usize,
    pub computation: AuxComputation,
}

/// Types of auxiliary computations.
#[derive(Clone, Debug)]
pub enum AuxComputation {
    /// Matrix-vector multiply element
    MatVecElement {
        row: usize,
        matrix_entries: Vec<(usize, GoldilocksField)>,
        input_vars: Vec<usize>,
    },
    /// Dot product
    DotProduct {
        a_vars: Vec<usize>,
        b_constants: Vec<GoldilocksField>,
    },
    /// Minimum of two values (for tropical semiring)
    Minimum { a_var: usize, b_var: usize },
    /// Bit decomposition of a value
    BitDecomposition { value_var: usize, num_bits: usize },
    /// Copy from another variable
    Copy { source_var: usize },
    /// Constant value
    Constant { value: GoldilocksField },
    /// Symbol selector: 1 if current symbol == target, else 0
    SymbolSelector { input_var: usize, target_symbol: u8 },
    /// Field multiplication with rescaling for fixed-point
    FixedPointMul { a_var: usize, b_var: usize, scale: u64 },
}

/// The intermediate representation between WFA and AIR.
#[derive(Clone, Debug)]
pub struct IntermediateRepresentation {
    pub variables: Vec<IRVariable>,
    pub constraints: Vec<IRConstraint>,
    pub auxiliary_computations: Vec<AuxiliaryComputation>,
    variable_counter: usize,
}

impl IntermediateRepresentation {
    pub fn new() -> Self {
        IntermediateRepresentation {
            variables: Vec::new(),
            constraints: Vec::new(),
            auxiliary_computations: Vec::new(),
            variable_counter: 0,
        }
    }

    pub fn add_variable(&mut self, name: &str, var_type: IRVariableType) -> usize {
        let idx = self.variable_counter;
        self.variables.push(IRVariable {
            name: name.to_string(),
            index: idx,
            variable_type: var_type,
            trace_column: None,
        });
        self.variable_counter += 1;
        idx
    }

    pub fn add_constraint(&mut self, expr: SymbolicExpression, ct: ConstraintType, description: &str) {
        self.constraints.push(IRConstraint {
            expression: expr,
            constraint_type: ct,
            source_description: description.to_string(),
        });
    }

    pub fn add_auxiliary(&mut self, target: usize, computation: AuxComputation) {
        self.auxiliary_computations.push(AuxiliaryComputation {
            target_variable: target,
            computation,
        });
    }

    pub fn assign_trace_columns(&mut self) {
        for (i, var) in self.variables.iter_mut().enumerate() {
            var.trace_column = Some(i);
        }
    }

    pub fn num_variables(&self) -> usize { self.variables.len() }

    pub fn variables_of_type(&self, vt: &IRVariableType) -> Vec<&IRVariable> {
        self.variables.iter().filter(|v| &v.variable_type == vt).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compilation Result and Statistics
// ═══════════════════════════════════════════════════════════════════════════════

/// Mapping from source WFA elements to generated constraints.
#[derive(Clone, Debug)]
pub struct ConstraintMapping {
    pub transition_to_constraints: HashMap<usize, Vec<usize>>,
    pub state_to_boundary: HashMap<usize, Vec<usize>>,
    pub auxiliary_constraint_indices: Vec<usize>,
}

impl ConstraintMapping {
    pub fn new() -> Self {
        ConstraintMapping {
            transition_to_constraints: HashMap::new(),
            state_to_boundary: HashMap::new(),
            auxiliary_constraint_indices: Vec::new(),
        }
    }
}

/// Statistics about the compiled circuit.
#[derive(Clone, Debug)]
pub struct CompilationStats {
    pub trace_width: usize,
    pub trace_length: usize,
    pub num_constraints: usize,
    pub num_boundary: usize,
    pub num_transition: usize,
    pub num_periodic: usize,
    pub num_global: usize,
    pub max_degree: usize,
    pub estimated_proof_size_bytes: usize,
    pub num_wfa_states: usize,
    pub num_wfa_transitions: usize,
    pub alphabet_size: usize,
    pub num_auxiliary_columns: usize,
    pub optimization_passes_applied: Vec<String>,
    pub constraints_eliminated: usize,
    pub cse_matches_found: usize,
}

impl CompilationStats {
    fn new() -> Self {
        CompilationStats {
            trace_width: 0, trace_length: 0, num_constraints: 0,
            num_boundary: 0, num_transition: 0, num_periodic: 0, num_global: 0,
            max_degree: 0, estimated_proof_size_bytes: 0,
            num_wfa_states: 0, num_wfa_transitions: 0, alphabet_size: 0,
            num_auxiliary_columns: 0,
            optimization_passes_applied: Vec::new(),
            constraints_eliminated: 0, cse_matches_found: 0,
        }
    }

    pub fn estimate_proof_size(&mut self) {
        let log_n = if self.trace_length > 0 {
            (self.trace_length as f64).log2().ceil() as usize
        } else {
            1
        };
        let fri_layers = log_n;
        let query_count = 64; // ~64 queries for 128-bit security
        let field_element_size = 8;
        let commitment_size = 32 * fri_layers;
        let query_size = self.trace_width * query_count * field_element_size * 2;
        let fri_proof_size = fri_layers * query_count * field_element_size;
        self.estimated_proof_size_bytes = commitment_size + query_size + fri_proof_size;
    }
}

impl fmt::Display for CompilationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Compilation Statistics ===")?;
        writeln!(f, "WFA: {} states, {} transitions, alphabet {}", self.num_wfa_states, self.num_wfa_transitions, self.alphabet_size)?;
        writeln!(f, "Trace: {} rows x {} columns", self.trace_length, self.trace_width)?;
        writeln!(f, "  Auxiliary columns: {}", self.num_auxiliary_columns)?;
        writeln!(f, "Constraints: {} total", self.num_constraints)?;
        writeln!(f, "  Boundary: {}", self.num_boundary)?;
        writeln!(f, "  Transition: {}", self.num_transition)?;
        writeln!(f, "  Periodic: {}", self.num_periodic)?;
        writeln!(f, "  Global: {}", self.num_global)?;
        writeln!(f, "Max degree: {}", self.max_degree)?;
        writeln!(f, "Estimated proof size: {} bytes", self.estimated_proof_size_bytes)?;
        if !self.optimization_passes_applied.is_empty() {
            writeln!(f, "Optimization passes: {:?}", self.optimization_passes_applied)?;
            writeln!(f, "  Constraints eliminated: {}", self.constraints_eliminated)?;
            writeln!(f, "  CSE matches: {}", self.cse_matches_found)?;
        }
        Ok(())
    }
}

/// Result of compiling a WFA into an AIR circuit.
#[derive(Clone, Debug)]
pub struct CompilationResult {
    pub air_program: AIRProgram,
    pub trace_layout: TraceLayout,
    pub compilation_stats: CompilationStats,
    pub constraint_mapping: ConstraintMapping,
    pub ir: IntermediateRepresentation,
    pub gadget_constraints: GadgetConstraints,
    pub column_map: HashMap<String, usize>,
    pub transition_matrices: Vec<Vec<Vec<GoldilocksField>>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// WFA Circuit Compiler
// ═══════════════════════════════════════════════════════════════════════════════

/// The main WFA-to-AIR circuit compiler.
pub struct WFACircuitCompiler<S: Semiring + SemiringEmbedding> {
    config: CompilerConfig,
    stats: CompilationStats,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Semiring + SemiringEmbedding> WFACircuitCompiler<S> {
    /// Create a new compiler with the given configuration.
    pub fn new(config: CompilerConfig) -> Self {
        WFACircuitCompiler {
            config,
            stats: CompilationStats::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a compiler with default configuration.
    pub fn default_compiler() -> Self {
        Self::new(CompilerConfig::default())
    }

    /// Main entry point: compile a WFA into an AIR program.
    pub fn compile(&mut self, wfa: &WFA<S>, input_length: usize) -> CompilationResult {
        self.stats.num_wfa_states = wfa.num_states;
        self.stats.num_wfa_transitions = wfa.transitions.len();
        self.stats.alphabet_size = wfa.alphabet_size;

        let effective_target = self.select_compilation_target();
        match effective_target {
            CompilationTarget::AlgebraicDirect => self.compile_algebraic(wfa, input_length),
            CompilationTarget::GadgetAssisted => self.compile_gadget_assisted(wfa, input_length),
            CompilationTarget::Hybrid => self.compile_hybrid(wfa, input_length),
        }
    }

    /// Determine effective compilation target based on semiring properties.
    fn select_compilation_target(&self) -> CompilationTarget {
        match &self.config.target {
            CompilationTarget::Hybrid => {
                if S::is_algebraically_embeddable() {
                    CompilationTarget::AlgebraicDirect
                } else {
                    CompilationTarget::GadgetAssisted
                }
            }
            other => other.clone(),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tier 1: Algebraic Direct Compilation
    // ─────────────────────────────────────────────────────────────────────

    /// Compile WFA algebraically: semiring operations map directly to field operations.
    pub fn compile_algebraic(&mut self, wfa: &WFA<S>, input_length: usize) -> CompilationResult {
        let padded_length = next_power_of_two(input_length + 1);
        let mut ir = IntermediateRepresentation::new();
        let mut column_map = HashMap::new();

        let trace_layout = self.compute_trace_layout(wfa, padded_length);

        // State variables
        let mut state_var_indices = Vec::new();
        for s in 0..wfa.num_states {
            let name = format!("state_{}", s);
            let idx = ir.add_variable(&name, IRVariableType::State);
            column_map.insert(name, idx);
            state_var_indices.push(idx);
        }

        // Input symbol variable
        let input_var = ir.add_variable("input_symbol", IRVariableType::Input);
        column_map.insert("input_symbol".to_string(), input_var);

        // Symbol selector variables
        let mut selector_vars = Vec::new();
        for a in 0..wfa.alphabet_size {
            let name = format!("selector_{}", a);
            let idx = ir.add_variable(&name, IRVariableType::Auxiliary);
            column_map.insert(name, idx);
            selector_vars.push(idx);
            ir.add_auxiliary(idx, AuxComputation::SymbolSelector {
                input_var,
                target_symbol: a as u8,
            });
        }

        // Auxiliary variables for transition matrix products
        let mut matmul_aux_vars = Vec::new();
        for s in 0..wfa.num_states {
            let name = format!("matmul_aux_{}", s);
            let idx = ir.add_variable(&name, IRVariableType::Auxiliary);
            column_map.insert(name, idx);
            matmul_aux_vars.push(idx);
        }

        // Output variable
        let output_var = ir.add_variable("output", IRVariableType::Public);
        column_map.insert("output".to_string(), output_var);

        // Step counter
        let step_var = ir.add_variable("step_counter", IRVariableType::Auxiliary);
        column_map.insert("step_counter".to_string(), step_var);

        ir.assign_trace_columns();

        // Embed transition matrices
        let transition_matrices = self.embed_transition_matrices(wfa);

        // Generate constraints
        let mut all_constraints = Vec::new();
        let mut constraint_mapping = ConstraintMapping::new();

        // Initial state constraints
        let initial_constraints = self.emit_initial_state_constraints(wfa, &state_var_indices);
        for (i, c) in initial_constraints.iter().enumerate() {
            constraint_mapping.state_to_boundary.entry(i).or_insert_with(Vec::new).push(all_constraints.len());
            all_constraints.push(c.clone());
        }

        // Selector constraints
        let selector_constraints = self.emit_selector_constraints(wfa, input_var, &selector_vars);
        all_constraints.extend(selector_constraints);

        // Transition constraints
        let transition_constraints = self.emit_transition_constraints_algebraic(
            wfa, &state_var_indices, &selector_vars, &matmul_aux_vars, &transition_matrices,
        );
        for (i, c) in transition_constraints.iter().enumerate() {
            constraint_mapping.transition_to_constraints.entry(i).or_insert_with(Vec::new).push(all_constraints.len());
            all_constraints.push(c.clone());
        }

        // Final state / output constraint
        let final_constraints = self.emit_final_state_constraints(wfa, &state_var_indices, output_var, padded_length - 1);
        all_constraints.extend(final_constraints);

        // Weight accumulation constraints
        let weight_constraints = self.emit_weight_accumulation_constraints(wfa, &state_var_indices);
        all_constraints.extend(weight_constraints);

        // Step counter constraints
        let step_init = SymbolicExpression::col(step_var);
        all_constraints.push(AIRConstraint::boundary(step_init, 0, "step_init_zero"));

        let step_transition = SymbolicExpression::next_col(step_var)
            .sub(SymbolicExpression::col(step_var))
            .sub(SymbolicExpression::constant(1));
        all_constraints.push(AIRConstraint::transition(step_transition, "step_increment"));

        // Apply optimization passes
        let optimized = self.optimize_constraints(all_constraints);

        let mut air_program = AIRProgram::new(trace_layout.clone(), padded_length);
        for c in &optimized {
            air_program.add_constraint(c.clone());
        }

        self.compute_stats(&optimized, &trace_layout, padded_length);

        CompilationResult {
            air_program,
            trace_layout,
            compilation_stats: self.stats.clone(),
            constraint_mapping,
            ir,
            gadget_constraints: GadgetConstraints::new(),
            column_map,
            transition_matrices,
        }
    }

    /// Emit boundary constraints that set the initial state vector.
    pub fn emit_initial_state_constraints(
        &self, wfa: &WFA<S>, state_vars: &[usize],
    ) -> Vec<AIRConstraint> {
        let mut constraints = Vec::new();
        for (s, &var) in state_vars.iter().enumerate() {
            let initial_val = wfa.initial_weights[s].embed();
            let expr = SymbolicExpression::col(var)
                .sub(SymbolicExpression::constant_field(initial_val));
            constraints.push(AIRConstraint::boundary(expr, 0, &format!("initial_state_{}", s)));
        }
        constraints
    }

    /// Emit constraints for symbol selectors.
    fn emit_selector_constraints(
        &self, wfa: &WFA<S>, input_var: usize, selector_vars: &[usize],
    ) -> Vec<AIRConstraint> {
        let mut constraints = Vec::new();

        // Each selector is boolean
        for (a, &sel) in selector_vars.iter().enumerate() {
            let s = SymbolicExpression::col(sel);
            let bool_expr = s.clone().mul(SymbolicExpression::constant(1).sub(s));
            constraints.push(AIRConstraint::transition(bool_expr, &format!("selector_{}_boolean", a)));
        }

        // Sum of selectors = 1
        let mut sum_expr = SymbolicExpression::constant(0);
        for &sel in selector_vars {
            sum_expr = sum_expr.add(SymbolicExpression::col(sel));
        }
        let sum_one = sum_expr.sub(SymbolicExpression::constant(1));
        constraints.push(AIRConstraint::transition(sum_one, "selector_sum_one"));

        // input = sum_a (a * selector_a)
        let mut input_recompose = SymbolicExpression::constant(0);
        for (a, &sel) in selector_vars.iter().enumerate() {
            let term = SymbolicExpression::col(sel).mul(SymbolicExpression::constant(a as u64));
            input_recompose = input_recompose.add(term);
        }
        let input_check = SymbolicExpression::col(input_var).sub(input_recompose);
        constraints.push(AIRConstraint::transition(input_check, "input_selector_consistency"));

        constraints
    }

    /// Emit transition constraints for algebraic compilation.
    fn emit_transition_constraints_algebraic(
        &self,
        wfa: &WFA<S>,
        state_vars: &[usize],
        selector_vars: &[usize],
        _aux_vars: &[usize],
        transition_matrices: &[Vec<Vec<GoldilocksField>>],
    ) -> Vec<AIRConstraint> {
        let mut constraints = Vec::new();
        let n = wfa.num_states;

        for s in 0..n {
            let mut total_expr = SymbolicExpression::constant(0);
            for (a, matrix) in transition_matrices.iter().enumerate() {
                if a >= selector_vars.len() { break; }
                let mut inner = SymbolicExpression::constant(0);
                for q in 0..n {
                    let coeff = matrix[s][q];
                    if coeff.to_canonical() != 0 {
                        let term = SymbolicExpression::constant_field(coeff)
                            .mul(SymbolicExpression::col(state_vars[q]));
                        inner = inner.add(term);
                    }
                }
                let selected = SymbolicExpression::col(selector_vars[a]).mul(inner);
                total_expr = total_expr.add(selected);
            }

            let constraint_expr = SymbolicExpression::next_col(state_vars[s]).sub(total_expr);
            constraints.push(AIRConstraint::transition(constraint_expr, &format!("transition_state_{}", s)));
        }

        constraints
    }

    /// Emit final state constraints: output = sum_s final_weight[s] * state[T][s]
    pub fn emit_final_state_constraints(
        &self, wfa: &WFA<S>, state_vars: &[usize], output_var: usize, final_row: usize,
    ) -> Vec<AIRConstraint> {
        let mut dot = SymbolicExpression::constant(0);
        for (s, &var) in state_vars.iter().enumerate() {
            let fw = wfa.final_weights[s].embed();
            if fw.to_canonical() != 0 {
                let term = SymbolicExpression::constant_field(fw).mul(SymbolicExpression::col(var));
                dot = dot.add(term);
            }
        }
        let expr = SymbolicExpression::col(output_var).sub(dot);
        vec![AIRConstraint::boundary(expr, final_row, "final_output")]
    }

    /// Emit weight accumulation constraints.
    pub fn emit_weight_accumulation_constraints(
        &self, _wfa: &WFA<S>, state_vars: &[usize],
    ) -> Vec<AIRConstraint> {
        let mut constraints = Vec::new();
        if !S::addition_is_field_addition() && S::multiplication_is_field_multiplication() {
            for (s, &var) in state_vars.iter().enumerate() {
                let x = SymbolicExpression::col(var);
                let bool_check = x.clone().mul(SymbolicExpression::constant(1).sub(x));
                constraints.push(AIRConstraint::transition(bool_check, &format!("state_{}_boolean", s)));
            }
        }
        constraints
    }

    /// Compute the per-symbol transition matrix.
    pub fn compute_transition_matrix(&self, wfa: &WFA<S>, symbol: u8) -> Vec<Vec<GoldilocksField>> {
        let n = wfa.num_states;
        let mut matrix = vec![vec![GoldilocksField::ZERO; n]; n];
        for trans in wfa.transitions_for_symbol(symbol) {
            let embedded = trans.weight.embed();
            let current = matrix[trans.to_state][trans.from_state];
            matrix[trans.to_state][trans.from_state] = current.add_elem(embedded);
        }
        matrix
    }

    /// Embed all transition matrices for all symbols.
    pub fn embed_transition_matrices(&self, wfa: &WFA<S>) -> Vec<Vec<Vec<GoldilocksField>>> {
        let mut matrices = Vec::new();
        for a in 0..wfa.alphabet_size {
            matrices.push(self.compute_transition_matrix(wfa, a as u8));
        }
        matrices
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tier 2: Gadget-Assisted Compilation (Tropical / Real)
    // ─────────────────────────────────────────────────────────────────────

    /// Compile WFA with gadget assistance for non-algebraic semirings.
    pub fn compile_gadget_assisted(&mut self, wfa: &WFA<S>, input_length: usize) -> CompilationResult {
        let padded_length = next_power_of_two(input_length + 1);
        let mut ir = IntermediateRepresentation::new();
        let mut column_map = HashMap::new();
        let mut gadget_constraints = GadgetConstraints::new();

        // State variables
        let mut state_vars = Vec::new();
        for s in 0..wfa.num_states {
            let name = format!("state_{}", s);
            let idx = ir.add_variable(&name, IRVariableType::State);
            column_map.insert(name, idx);
            state_vars.push(idx);
        }

        let input_var = ir.add_variable("input_symbol", IRVariableType::Input);
        column_map.insert("input_symbol".to_string(), input_var);

        let mut selector_vars = Vec::new();
        for a in 0..wfa.alphabet_size {
            let name = format!("selector_{}", a);
            let idx = ir.add_variable(&name, IRVariableType::Auxiliary);
            column_map.insert(name, idx);
            selector_vars.push(idx);
        }

        // Candidate variables for intermediate tropical/real computations
        let mut candidate_vars: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut min_accumulator_vars: Vec<Vec<usize>> = Vec::new();

        for s in 0..wfa.num_states {
            let mut per_symbol = Vec::new();
            let mut per_symbol_accum = Vec::new();
            for a in 0..wfa.alphabet_size {
                let transitions_for_sa: Vec<&WFATransition<S>> = wfa.transitions.iter()
                    .filter(|t| t.to_state == s && t.symbol == a as u8).collect();

                let mut candidates_for_sa = Vec::new();
                for (idx, _trans) in transitions_for_sa.iter().enumerate() {
                    let name = format!("cand_s{}_a{}_t{}", s, a, idx);
                    let var = ir.add_variable(&name, IRVariableType::Auxiliary);
                    column_map.insert(name, var);
                    candidates_for_sa.push(var);
                }

                let accum_name = format!("accum_s{}_a{}", s, a);
                let accum_var = ir.add_variable(&accum_name, IRVariableType::Auxiliary);
                column_map.insert(accum_name, accum_var);

                per_symbol.push(candidates_for_sa);
                per_symbol_accum.push(accum_var);
            }
            candidate_vars.push(per_symbol);
            min_accumulator_vars.push(per_symbol_accum);
        }

        // Comparison gadget auxiliary columns
        let num_bits = self.config.comparison_bits;
        let mut comparison_counter = 0usize;

        let mut total_comparisons = 0usize;
        for s in 0..wfa.num_states {
            for a in 0..wfa.alphabet_size {
                let num_candidates = candidate_vars[s][a].len();
                if num_candidates > 1 { total_comparisons += num_candidates - 1; }
            }
        }

        let mut comparison_bit_vars: Vec<Vec<usize>> = Vec::new();
        let mut comparison_diff_vars: Vec<usize> = Vec::new();

        for c in 0..total_comparisons {
            let mut bits = Vec::new();
            for b in 0..num_bits {
                let name = format!("cmp_{}_bit_{}", c, b);
                let var = ir.add_variable(&name, IRVariableType::Auxiliary);
                column_map.insert(name, var);
                bits.push(var);
            }
            comparison_bit_vars.push(bits);

            let diff_name = format!("cmp_{}_diff", c);
            let diff_var = ir.add_variable(&diff_name, IRVariableType::Auxiliary);
            column_map.insert(diff_name, diff_var);
            comparison_diff_vars.push(diff_var);
        }

        let output_var = ir.add_variable("output", IRVariableType::Public);
        column_map.insert("output".to_string(), output_var);

        let step_var = ir.add_variable("step_counter", IRVariableType::Auxiliary);
        column_map.insert("step_counter".to_string(), step_var);

        ir.assign_trace_columns();
        let transition_matrices = self.embed_transition_matrices(wfa);

        let mut all_constraints = Vec::new();
        let mut constraint_mapping = ConstraintMapping::new();

        // Initial state boundary constraints
        let init_cs = self.emit_initial_state_constraints(wfa, &state_vars);
        for (i, c) in init_cs.iter().enumerate() {
            constraint_mapping.state_to_boundary.entry(i).or_insert_with(Vec::new).push(all_constraints.len());
            all_constraints.push(c.clone());
        }

        // Selector constraints
        let sel_cs = self.emit_selector_constraints(wfa, input_var, &selector_vars);
        all_constraints.extend(sel_cs);

        // Tropical/gadget transition constraints
        comparison_counter = 0;
        for s in 0..wfa.num_states {
            for a in 0..wfa.alphabet_size {
                let transitions_for_sa: Vec<&WFATransition<S>> = wfa.transitions.iter()
                    .filter(|t| t.to_state == s && t.symbol == a as u8).collect();

                // Candidate constraints: cand = weight + state[from]
                for (idx, trans) in transitions_for_sa.iter().enumerate() {
                    let weight_field = trans.weight.embed();
                    let cand_var = candidate_vars[s][a][idx];
                    let from_var = state_vars[trans.from_state];
                    let expr = SymbolicExpression::col(cand_var)
                        .sub(SymbolicExpression::col(from_var))
                        .sub(SymbolicExpression::constant_field(weight_field));
                    all_constraints.push(AIRConstraint::transition(expr, &format!("tropical_mul_s{}_a{}_t{}", s, a, idx)));
                }

                let num_candidates = candidate_vars[s][a].len();
                if num_candidates == 0 {
                    let sentinel = GoldilocksField::new(GoldilocksField::MODULUS - 1);
                    let expr = SymbolicExpression::col(min_accumulator_vars[s][a])
                        .sub(SymbolicExpression::constant_field(sentinel));
                    all_constraints.push(AIRConstraint::transition(expr, &format!("tropical_zero_s{}_a{}", s, a)));
                } else if num_candidates == 1 {
                    let expr = SymbolicExpression::col(min_accumulator_vars[s][a])
                        .sub(SymbolicExpression::col(candidate_vars[s][a][0]));
                    all_constraints.push(AIRConstraint::transition(expr, &format!("tropical_single_s{}_a{}", s, a)));
                } else {
                    let mut current_min = candidate_vars[s][a][0];
                    for i in 1..num_candidates {
                        let other = candidate_vars[s][a][i];
                        if comparison_counter < comparison_diff_vars.len() {
                            let diff_var = comparison_diff_vars[comparison_counter];
                            let bits = &comparison_bit_vars[comparison_counter];
                            let gadget = ComparisonGadget::new(num_bits, current_min, other, bits.clone(), diff_var);
                            let gadget_cs = gadget.constraints();
                            constraint_mapping.auxiliary_constraint_indices.push(all_constraints.len());
                            all_constraints.extend(gadget_cs);
                            gadget_constraints.comparison_gadgets.push(gadget);
                            comparison_counter += 1;
                        }
                        current_min = min_accumulator_vars[s][a];
                    }
                }
            }

            // Combine across symbols: next_state[s] = sum_a selector_a * accum_{s,a}
            let mut combine_expr = SymbolicExpression::constant(0);
            for a in 0..wfa.alphabet_size {
                let term = SymbolicExpression::col(selector_vars[a])
                    .mul(SymbolicExpression::col(min_accumulator_vars[s][a]));
                combine_expr = combine_expr.add(term);
            }
            let transition_expr = SymbolicExpression::next_col(state_vars[s]).sub(combine_expr);
            all_constraints.push(AIRConstraint::transition(transition_expr, &format!("gadget_transition_state_{}", s)));
        }

        // Final output
        let final_cs = self.emit_final_state_constraints(wfa, &state_vars, output_var, padded_length - 1);
        all_constraints.extend(final_cs);

        // Step counter
        let step_init = SymbolicExpression::col(step_var);
        all_constraints.push(AIRConstraint::boundary(step_init, 0, "step_init_zero"));
        let step_trans = SymbolicExpression::next_col(step_var)
            .sub(SymbolicExpression::col(step_var))
            .sub(SymbolicExpression::constant(1));
        all_constraints.push(AIRConstraint::transition(step_trans, "step_increment"));

        let optimized = self.optimize_constraints(all_constraints);
        let trace_layout = self.build_trace_layout_from_ir(&ir);
        let mut air_program = AIRProgram::new(trace_layout.clone(), padded_length);
        for c in &optimized {
            air_program.add_constraint(c.clone());
        }
        self.compute_stats(&optimized, &trace_layout, padded_length);

        CompilationResult {
            air_program, trace_layout,
            compilation_stats: self.stats.clone(),
            constraint_mapping, ir, gadget_constraints, column_map, transition_matrices,
        }
    }

    /// Hybrid compilation: algebraic where possible, gadgets where needed.
    fn compile_hybrid(&mut self, wfa: &WFA<S>, input_length: usize) -> CompilationResult {
        if S::multiplication_is_field_multiplication() {
            self.compile_algebraic(wfa, input_length)
        } else {
            self.compile_gadget_assisted(wfa, input_length)
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tropical-Specific Emission
    // ─────────────────────────────────────────────────────────────────────

    /// Emit constraints for tropical addition (min operation).
    pub fn emit_tropical_add_constraints(
        &self, a_col: usize, b_col: usize, result_col: usize,
        diff_col: usize, bit_cols: &[usize], is_less_col: usize,
    ) -> Vec<AIRConstraint> {
        let mut constraints = Vec::new();

        let diff_expr = SymbolicExpression::col(a_col)
            .sub(SymbolicExpression::col(b_col))
            .sub(SymbolicExpression::col(diff_col));
        constraints.push(AIRConstraint::transition(diff_expr, "tropical_add_diff"));

        let mut recompose = SymbolicExpression::constant(0);
        for (i, &bc) in bit_cols.iter().enumerate() {
            let power = 1u64 << i;
            let term = SymbolicExpression::col(bc).mul(SymbolicExpression::constant(power));
            recompose = recompose.add(term);
            let bit = SymbolicExpression::col(bc);
            let bool_check = bit.clone().mul(SymbolicExpression::constant(1).sub(bit));
            constraints.push(AIRConstraint::transition(bool_check, &format!("tropical_add_bit_{}_bool", i)));
        }

        let is_less = SymbolicExpression::col(is_less_col);
        let bool_check = is_less.clone().mul(SymbolicExpression::constant(1).sub(is_less.clone()));
        constraints.push(AIRConstraint::transition(bool_check, "tropical_add_is_less_bool"));

        let result_expr = is_less.clone()
            .mul(SymbolicExpression::col(a_col))
            .add(SymbolicExpression::constant(1).sub(is_less).mul(SymbolicExpression::col(b_col)));
        let result_check = SymbolicExpression::col(result_col).sub(result_expr);
        constraints.push(AIRConstraint::transition(result_check, "tropical_add_result"));

        constraints
    }

    /// Emit constraints for tropical multiplication (field addition).
    pub fn emit_tropical_mul_constraints(&self, a_col: usize, b_col: usize, result_col: usize) -> Vec<AIRConstraint> {
        let expr = SymbolicExpression::col(result_col)
            .sub(SymbolicExpression::col(a_col))
            .sub(SymbolicExpression::col(b_col));
        vec![AIRConstraint::transition(expr, "tropical_mul")]
    }

    /// Emit bit decomposition constraints for a value.
    pub fn emit_bit_decomposition_constraints(&self, value_col: usize, bit_cols: &[usize]) -> Vec<AIRConstraint> {
        let mut constraints = Vec::new();
        for (i, &bc) in bit_cols.iter().enumerate() {
            let bit = SymbolicExpression::col(bc);
            let bool_check = bit.clone().mul(SymbolicExpression::constant(1).sub(bit));
            constraints.push(AIRConstraint::transition(bool_check, &format!("bit_{}_boolean", i)));
        }

        let mut recompose = SymbolicExpression::constant(0);
        for (i, &bc) in bit_cols.iter().enumerate() {
            let power = 1u64 << i;
            let term = SymbolicExpression::col(bc).mul(SymbolicExpression::constant(power));
            recompose = recompose.add(term);
        }
        let check = SymbolicExpression::col(value_col).sub(recompose);
        constraints.push(AIRConstraint::transition(check, "bit_decomposition_recompose"));

        constraints
    }

    /// Emit comparison gadget constraints proving a <= b.
    pub fn emit_comparison_gadget_constraints(
        &self, a_col: usize, b_col: usize, diff_col: usize, bit_cols: &[usize],
    ) -> Vec<AIRConstraint> {
        let gadget = ComparisonGadget::new(bit_cols.len(), a_col, b_col, bit_cols.to_vec(), diff_col);
        gadget.constraints()
    }

    /// Convert a tropical f64 value to a field element.
    pub fn tropical_to_field_value(val: f64) -> GoldilocksField {
        tropical_to_field(val)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Trace Layout Computation
    // ─────────────────────────────────────────────────────────────────────

    /// Compute the trace layout for a WFA.
    pub fn compute_trace_layout(&self, wfa: &WFA<S>, trace_length: usize) -> TraceLayout {
        let mut layout = TraceLayout::new();
        layout.num_rows = trace_length;

        for s in 0..wfa.num_states {
            layout.add_column(&format!("state_{}", s), ColumnType::Witness, &format!("State {} weight", s));
        }
        layout.add_column("input_symbol", ColumnType::PublicInput, "Current input symbol");
        for a in 0..wfa.alphabet_size {
            layout.add_column(&format!("selector_{}", a), ColumnType::Selector, &format!("Selector for symbol {}", a));
        }
        for s in 0..wfa.num_states {
            layout.add_column(&format!("matmul_aux_{}", s), ColumnType::Auxiliary, &format!("Mat-vec aux for state {}", s));
        }
        layout.add_column("output", ColumnType::PublicOutput, "Computation output");
        layout.add_column("step_counter", ColumnType::Auxiliary, "Step counter");

        layout
    }

    /// Build a trace layout from the IR variable assignments.
    fn build_trace_layout_from_ir(&self, ir: &IntermediateRepresentation) -> TraceLayout {
        let mut layout = TraceLayout::new();
        for var in &ir.variables {
            let col_type = match var.variable_type {
                IRVariableType::State => ColumnType::Witness,
                IRVariableType::Weight => ColumnType::Witness,
                IRVariableType::Input => ColumnType::PublicInput,
                IRVariableType::Auxiliary => ColumnType::Auxiliary,
                IRVariableType::Public => ColumnType::PublicOutput,
            };
            layout.add_column(&var.name, col_type, &var.name);
        }
        layout
    }

    // ─────────────────────────────────────────────────────────────────────
    // Optimization Passes
    // ─────────────────────────────────────────────────────────────────────

    /// Apply all enabled optimization passes to the constraints.
    pub fn optimize_constraints(&mut self, constraints: Vec<AIRConstraint>) -> Vec<AIRConstraint> {
        let mut result = constraints;
        let initial_count = result.len();

        if self.config.optimization_level == 0 { return result; }

        // Pass 1: Eliminate trivially true constraints
        let before = result.len();
        result = self.eliminate_trivial_constraints(result);
        if result.len() < before {
            self.stats.optimization_passes_applied.push("trivial_elimination".to_string());
        }

        // Pass 2: Common subexpression elimination
        if self.config.enable_cse {
            let before_cse = result.len();
            result = self.common_subexpression_elimination(result);
            if result.len() < before_cse {
                self.stats.optimization_passes_applied.push("cse".to_string());
            }
        }

        // Pass 3: Dead constraint elimination
        if self.config.enable_dead_elimination {
            let before_dead = result.len();
            result = self.eliminate_dead_constraints(result);
            if result.len() < before_dead {
                self.stats.optimization_passes_applied.push("dead_elimination".to_string());
            }
        }

        // Pass 4: Degree reduction
        if self.config.optimization_level >= 2 {
            result = self.reduce_all_degrees(result);
            self.stats.optimization_passes_applied.push("degree_reduction".to_string());
        }

        // Pass 5: Constraint batching
        if self.config.enable_batching && self.config.optimization_level >= 2 {
            result = self.batch_constraints(result);
            self.stats.optimization_passes_applied.push("batching".to_string());
        }

        self.stats.constraints_eliminated = initial_count.saturating_sub(result.len());
        result
    }

    fn eliminate_trivial_constraints(&self, constraints: Vec<AIRConstraint>) -> Vec<AIRConstraint> {
        constraints.into_iter().filter(|c| !c.expression.is_zero_constant()).collect()
    }

    /// Common subexpression elimination.
    pub fn common_subexpression_elimination(&mut self, constraints: Vec<AIRConstraint>) -> Vec<AIRConstraint> {
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut result = Vec::new();
        let mut cse_count = 0usize;

        for c in constraints {
            let hash = c.expression.structural_hash();
            if let Some(&existing_idx) = seen.get(&hash) {
                if result.len() > existing_idx {
                    let existing: &AIRConstraint = &result[existing_idx];
                    if existing.constraint_type == c.constraint_type && existing.boundary_row == c.boundary_row {
                        cse_count += 1;
                        continue;
                    }
                }
            }
            seen.insert(hash, result.len());
            result.push(c);
        }

        self.stats.cse_matches_found += cse_count;
        result
    }

    /// Eliminate dead constraints.
    pub fn eliminate_dead_constraints(&self, constraints: Vec<AIRConstraint>) -> Vec<AIRConstraint> {
        let mut live_columns: HashSet<usize> = HashSet::new();

        for c in &constraints {
            match c.constraint_type {
                ConstraintType::Boundary | ConstraintType::Global => {
                    live_columns.extend(c.expression.referenced_columns());
                }
                _ => {}
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for c in &constraints {
                let refs = c.expression.referenced_columns();
                if refs.iter().any(|col| live_columns.contains(col)) {
                    for col in &refs {
                        if live_columns.insert(*col) { changed = true; }
                    }
                }
            }
        }

        constraints.into_iter().filter(|c| {
            match c.constraint_type {
                ConstraintType::Boundary | ConstraintType::Global => true,
                _ => {
                    let refs = c.expression.referenced_columns();
                    refs.iter().any(|col| live_columns.contains(col))
                }
            }
        }).collect()
    }

    /// Batch multiple constraints via random linear combination.
    pub fn batch_constraints(&self, constraints: Vec<AIRConstraint>) -> Vec<AIRConstraint> {
        let mut boundary = Vec::new();
        let mut transition = Vec::new();
        let mut periodic: HashMap<usize, Vec<AIRConstraint>> = HashMap::new();
        let mut global = Vec::new();

        for c in constraints {
            match &c.constraint_type {
                ConstraintType::Boundary => boundary.push(c),
                ConstraintType::Transition => transition.push(c),
                ConstraintType::Periodic(p) => { periodic.entry(*p).or_insert_with(Vec::new).push(c); }
                ConstraintType::Global => global.push(c),
            }
        }

        let mut result = Vec::new();
        result.extend(boundary);

        if transition.len() > 1 {
            let alpha = GoldilocksField::new(0x123456789ABCDEF0);
            let mut combined = SymbolicExpression::constant(0);
            let mut alpha_power = GoldilocksField::ONE;
            for tc in &transition {
                let scaled = SymbolicExpression::constant_field(alpha_power).mul(tc.expression.clone());
                combined = combined.add(scaled);
                alpha_power = alpha_power.mul_elem(alpha);
            }
            result.push(AIRConstraint::transition(combined, "batched_transition"));
        } else {
            result.extend(transition);
        }

        for (_period, cs) in periodic { result.extend(cs); }

        if global.len() > 1 {
            let alpha = GoldilocksField::new(0xFEDCBA9876543210);
            let mut combined = SymbolicExpression::constant(0);
            let mut alpha_power = GoldilocksField::ONE;
            for gc in &global {
                let scaled = SymbolicExpression::constant_field(alpha_power).mul(gc.expression.clone());
                combined = combined.add(scaled);
                alpha_power = alpha_power.mul_elem(alpha);
            }
            result.push(AIRConstraint::global(combined, "batched_global"));
        } else {
            result.extend(global);
        }

        result
    }

    fn reduce_all_degrees(&self, constraints: Vec<AIRConstraint>) -> Vec<AIRConstraint> {
        let max_deg = self.config.max_constraint_degree;
        let mut result = Vec::new();
        for c in constraints {
            if c.degree > max_deg {
                result.extend(self.reduce_degree(&c, max_deg));
            } else {
                result.push(c);
            }
        }
        result
    }

    /// Split a high-degree constraint into multiple lower-degree constraints.
    pub fn reduce_degree(&self, constraint: &AIRConstraint, max_degree: usize) -> Vec<AIRConstraint> {
        if constraint.degree <= max_degree { return vec![constraint.clone()]; }
        let mut result = Vec::new();
        let mut aux_counter = 0usize;
        let reduced = self.reduce_expr_degree(
            &constraint.expression, max_degree, &mut result, &mut aux_counter,
            &constraint.constraint_type, &constraint.label,
        );
        result.push(AIRConstraint {
            expression: reduced,
            constraint_type: constraint.constraint_type.clone(),
            label: format!("{}_reduced", constraint.label),
            degree: max_degree.min(constraint.degree),
            boundary_row: constraint.boundary_row,
        });
        result
    }

    fn reduce_expr_degree(
        &self, expr: &SymbolicExpression, max_degree: usize,
        aux_constraints: &mut Vec<AIRConstraint>, aux_counter: &mut usize,
        ct: &ConstraintType, label: &str,
    ) -> SymbolicExpression {
        let deg = expr.degree();
        if deg <= max_degree { return expr.clone(); }

        match expr {
            SymbolicExpression::Mul(a, b) => {
                let a_reduced = self.reduce_expr_degree(a, max_degree, aux_constraints, aux_counter, ct, label);
                let b_reduced = self.reduce_expr_degree(b, max_degree, aux_constraints, aux_counter, ct, label);
                let a_deg = a_reduced.degree();
                let b_deg = b_reduced.degree();

                if a_deg + b_deg > max_degree {
                    let aux_name = format!("__deg_reduce_{}_{}", label, aux_counter);
                    *aux_counter += 1;
                    let aux_var = SymbolicExpression::Variable(aux_name.clone());

                    if a_deg >= b_deg {
                        let link = aux_var.clone().sub(a_reduced);
                        aux_constraints.push(AIRConstraint {
                            expression: link, constraint_type: ct.clone(),
                            label: format!("{}_aux_link", aux_name), degree: a_deg, boundary_row: None,
                        });
                        aux_var.mul(b_reduced)
                    } else {
                        let link = aux_var.clone().sub(b_reduced);
                        aux_constraints.push(AIRConstraint {
                            expression: link, constraint_type: ct.clone(),
                            label: format!("{}_aux_link", aux_name), degree: b_deg, boundary_row: None,
                        });
                        a_reduced.mul(aux_var)
                    }
                } else {
                    a_reduced.mul(b_reduced)
                }
            }
            SymbolicExpression::Add(a, b) => {
                let a_r = self.reduce_expr_degree(a, max_degree, aux_constraints, aux_counter, ct, label);
                let b_r = self.reduce_expr_degree(b, max_degree, aux_constraints, aux_counter, ct, label);
                a_r.add(b_r)
            }
            SymbolicExpression::Sub(a, b) => {
                let a_r = self.reduce_expr_degree(a, max_degree, aux_constraints, aux_counter, ct, label);
                let b_r = self.reduce_expr_degree(b, max_degree, aux_constraints, aux_counter, ct, label);
                a_r.sub(b_r)
            }
            SymbolicExpression::Neg(a) => {
                let a_r = self.reduce_expr_degree(a, max_degree, aux_constraints, aux_counter, ct, label);
                a_r.neg()
            }
            SymbolicExpression::Pow(base, exp) => {
                if *exp <= 1 { return expr.clone(); }
                let half = *exp / 2;
                let other = *exp - half;
                let left = base.as_ref().clone().pow(half);
                let right = base.as_ref().clone().pow(other);
                let product = left.mul(right);
                self.reduce_expr_degree(&product, max_degree, aux_constraints, aux_counter, ct, label)
            }
            _ => expr.clone(),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Statistics
    // ─────────────────────────────────────────────────────────────────────

    fn compute_stats(&mut self, constraints: &[AIRConstraint], layout: &TraceLayout, trace_length: usize) {
        self.stats.trace_width = layout.total_columns();
        self.stats.trace_length = trace_length;
        self.stats.num_constraints = constraints.len();
        self.stats.num_auxiliary_columns = layout.num_auxiliary_columns;
        self.stats.num_boundary = 0;
        self.stats.num_transition = 0;
        self.stats.num_periodic = 0;
        self.stats.num_global = 0;
        self.stats.max_degree = 0;

        for c in constraints {
            match c.constraint_type {
                ConstraintType::Boundary => self.stats.num_boundary += 1,
                ConstraintType::Transition => self.stats.num_transition += 1,
                ConstraintType::Periodic(_) => self.stats.num_periodic += 1,
                ConstraintType::Global => self.stats.num_global += 1,
            }
            if c.degree > self.stats.max_degree { self.stats.max_degree = c.degree; }
        }
        self.stats.estimate_proof_size();
    }

    /// Get the current compilation statistics.
    pub fn compilation_statistics(&self) -> CompilationStats { self.stats.clone() }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trace Generator
// ═══════════════════════════════════════════════════════════════════════════════

/// Generates execution traces from compiled WFA circuits.
pub struct TraceGenerator<S: Semiring + SemiringEmbedding> {
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Semiring + SemiringEmbedding> TraceGenerator<S> {
    pub fn new() -> Self {
        TraceGenerator { _phantom: std::marker::PhantomData }
    }

    /// Generate a complete execution trace for the given WFA and input.
    pub fn generate_trace(&self, wfa: &WFA<S>, input: &[u8], result: &CompilationResult) -> AIRTrace {
        let num_cols = result.trace_layout.total_columns();
        let trace_length = result.air_program.trace_length;
        let mut trace = AIRTrace::new(trace_length, num_cols);

        let state_trace = self.simulate_wfa(wfa, input);

        // Fill state columns
        for t in 0..trace_length {
            for s in 0..wfa.num_states {
                let col_name = format!("state_{}", s);
                if let Some(&col_idx) = result.column_map.get(&col_name) {
                    if t < state_trace.len() {
                        trace.set(t, col_idx, state_trace[t][s]);
                    }
                }
            }
        }

        // Fill input column
        if let Some(&input_col) = result.column_map.get("input_symbol") {
            for t in 0..trace_length {
                if t < input.len() {
                    trace.set(t, input_col, GoldilocksField::new(input[t] as u64));
                }
            }
        }

        // Fill selector columns
        for a in 0..wfa.alphabet_size {
            let col_name = format!("selector_{}", a);
            if let Some(&sel_col) = result.column_map.get(&col_name) {
                for t in 0..trace_length {
                    if t < input.len() {
                        let val = if input[t] == a as u8 { GoldilocksField::ONE } else { GoldilocksField::ZERO };
                        trace.set(t, sel_col, val);
                    }
                }
            }
        }

        // Fill step counter
        if let Some(&step_col) = result.column_map.get("step_counter") {
            for t in 0..trace_length {
                trace.set(t, step_col, GoldilocksField::new(t as u64));
            }
        }

        // Fill auxiliary columns
        self.fill_auxiliary_columns(wfa, input, result, &mut trace);

        // Fill output column at final row
        if let Some(&out_col) = result.column_map.get("output") {
            let final_row = if input.len() < trace_length { input.len() } else { trace_length - 1 };
            let output_val = self.compute_output(wfa, &state_trace, final_row);
            trace.set(final_row, out_col, output_val);
        }

        trace
    }

    /// Simulate the WFA on the given input, producing state vectors at each step.
    fn simulate_wfa(&self, wfa: &WFA<S>, input: &[u8]) -> Vec<Vec<GoldilocksField>> {
        let n = wfa.num_states;
        let mut state_trace = Vec::new();

        let mut current_state: Vec<GoldilocksField> = wfa.initial_weights.iter().map(|w| w.embed()).collect();
        state_trace.push(current_state.clone());

        for &symbol in input {
            let mut next_state = vec![GoldilocksField::ZERO; n];
            if S::is_algebraically_embeddable() {
                for s in 0..n {
                    let mut accum = GoldilocksField::ZERO;
                    for trans in wfa.transitions_for_symbol(symbol) {
                        if trans.to_state == s {
                            let weight = trans.weight.embed();
                            let state_val = current_state[trans.from_state];
                            accum = accum.add_elem(weight.mul_elem(state_val));
                        }
                    }
                    next_state[s] = accum;
                }
            } else {
                let mut native_state: Vec<S> = Vec::new();
                for s in 0..n {
                    native_state.push(S::extract(current_state[s]));
                }
                let mut native_next = vec![S::zero(); n];
                for s in 0..n {
                    for trans in wfa.transitions_for_symbol(symbol) {
                        if trans.to_state == s {
                            let contrib = trans.weight.mul(&native_state[trans.from_state]);
                            native_next[s] = native_next[s].add(&contrib);
                        }
                    }
                }
                for s in 0..n {
                    next_state[s] = native_next[s].embed();
                }
            }
            current_state = next_state;
            state_trace.push(current_state.clone());
        }

        state_trace
    }

    /// Generate state columns from WFA simulation.
    pub fn generate_state_columns(&self, wfa: &WFA<S>, input: &[u8]) -> Vec<Vec<GoldilocksField>> {
        let state_trace = self.simulate_wfa(wfa, input);
        let n = wfa.num_states;
        let mut columns = vec![Vec::new(); n];
        for row in &state_trace {
            for (s, &val) in row.iter().enumerate() {
                columns[s].push(val);
            }
        }
        columns
    }

    /// Generate auxiliary columns from state trace data.
    pub fn generate_auxiliary_columns(
        &self, wfa: &WFA<S>, state_trace: &[Vec<GoldilocksField>],
        transition_matrices: &[Vec<Vec<GoldilocksField>>],
    ) -> Vec<Vec<GoldilocksField>> {
        let n = wfa.num_states;
        let trace_len = state_trace.len();
        let mut aux_columns = vec![vec![GoldilocksField::ZERO; trace_len]; n];

        for t in 0..trace_len.saturating_sub(1) {
            for s in 0..n {
                let mut accum = GoldilocksField::ZERO;
                for matrix in transition_matrices {
                    for q in 0..n {
                        let coeff = matrix[s][q];
                        if coeff.to_canonical() != 0 {
                            accum = accum.add_elem(coeff.mul_elem(state_trace[t][q]));
                        }
                    }
                }
                aux_columns[s][t] = accum;
            }
        }

        aux_columns
    }

    fn fill_auxiliary_columns(&self, wfa: &WFA<S>, input: &[u8], result: &CompilationResult, trace: &mut AIRTrace) {
        let n = wfa.num_states;
        let state_trace = self.simulate_wfa(wfa, input);

        for t in 0..input.len() {
            let symbol = input[t];
            for s in 0..n {
                let col_name = format!("matmul_aux_{}", s);
                if let Some(&col_idx) = result.column_map.get(&col_name) {
                    let mut accum = GoldilocksField::ZERO;
                    if (symbol as usize) < result.transition_matrices.len() {
                        let matrix = &result.transition_matrices[symbol as usize];
                        for q in 0..n {
                            if t < state_trace.len() {
                                let coeff = matrix[s][q];
                                if coeff.to_canonical() != 0 {
                                    accum = accum.add_elem(coeff.mul_elem(state_trace[t][q]));
                                }
                            }
                        }
                    }
                    trace.set(t, col_idx, accum);
                }
            }
        }
    }

    fn compute_output(&self, wfa: &WFA<S>, state_trace: &[Vec<GoldilocksField>], row: usize) -> GoldilocksField {
        let mut result = GoldilocksField::ZERO;
        if row < state_trace.len() {
            for (s, fw) in wfa.final_weights.iter().enumerate() {
                let fw_field = fw.embed();
                if fw_field.to_canonical() != 0 {
                    result = result.add_elem(fw_field.mul_elem(state_trace[row][s]));
                }
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Matrix Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/// Multiply two square matrices of field elements.
pub fn matrix_multiply(a: &[Vec<GoldilocksField>], b: &[Vec<GoldilocksField>]) -> Vec<Vec<GoldilocksField>> {
    let n = a.len();
    let m = if b.is_empty() { 0 } else { b[0].len() };
    let k = b.len();
    let mut result = vec![vec![GoldilocksField::ZERO; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut sum = GoldilocksField::ZERO;
            for l in 0..k {
                sum = sum.add_elem(a[i][l].mul_elem(b[l][j]));
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Multiply a matrix by a vector.
pub fn matrix_vector_multiply(matrix: &[Vec<GoldilocksField>], vector: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let n = matrix.len();
    let mut result = vec![GoldilocksField::ZERO; n];
    for i in 0..n {
        let mut sum = GoldilocksField::ZERO;
        for (j, &v) in vector.iter().enumerate() {
            if j < matrix[i].len() {
                sum = sum.add_elem(matrix[i][j].mul_elem(v));
            }
        }
        result[i] = sum;
    }
    result
}

/// Compute matrix power M^k by repeated squaring.
pub fn matrix_power(matrix: &[Vec<GoldilocksField>], k: usize) -> Vec<Vec<GoldilocksField>> {
    let n = matrix.len();
    if k == 0 {
        let mut id = vec![vec![GoldilocksField::ZERO; n]; n];
        for i in 0..n { id[i][i] = GoldilocksField::ONE; }
        return id;
    }
    if k == 1 { return matrix.to_vec(); }

    let half = matrix_power(matrix, k / 2);
    let result = matrix_multiply(&half, &half);
    if k % 2 == 0 { result } else { matrix_multiply(&result, matrix) }
}

/// Create an identity matrix of size n.
pub fn identity_matrix(n: usize) -> Vec<Vec<GoldilocksField>> {
    let mut m = vec![vec![GoldilocksField::ZERO; n]; n];
    for i in 0..n { m[i][i] = GoldilocksField::ONE; }
    m
}

/// Add two matrices element-wise.
pub fn matrix_add(a: &[Vec<GoldilocksField>], b: &[Vec<GoldilocksField>]) -> Vec<Vec<GoldilocksField>> {
    let n = a.len();
    let m = if a.is_empty() { 0 } else { a[0].len() };
    let mut result = vec![vec![GoldilocksField::ZERO; m]; n];
    for i in 0..n { for j in 0..m { result[i][j] = a[i][j].add_elem(b[i][j]); } }
    result
}

/// Scale a matrix by a field element.
pub fn matrix_scale(matrix: &[Vec<GoldilocksField>], scalar: GoldilocksField) -> Vec<Vec<GoldilocksField>> {
    matrix.iter().map(|row| row.iter().map(|&v| v.mul_elem(scalar)).collect()).collect()
}

/// Transpose a matrix.
pub fn matrix_transpose(matrix: &[Vec<GoldilocksField>]) -> Vec<Vec<GoldilocksField>> {
    if matrix.is_empty() { return Vec::new(); }
    let n = matrix.len();
    let m = matrix[0].len();
    let mut result = vec![vec![GoldilocksField::ZERO; n]; m];
    for i in 0..n { for j in 0..m { result[j][i] = matrix[i][j]; } }
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

fn next_power_of_two(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

/// Compute dot product of two vectors.
pub fn dot_product(a: &[GoldilocksField], b: &[GoldilocksField]) -> GoldilocksField {
    let mut sum = GoldilocksField::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        sum = sum.add_elem(x.mul_elem(*y));
    }
    sum
}

fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

// ═══════════════════════════════════════════════════════════════════════════════
// WFA Builders (for testing and convenience)
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a simple counting WFA that counts occurrences of a symbol.
pub fn build_counting_wfa(symbol: u8, alphabet_size: usize) -> WFA<CountingSemiring> {
    let mut wfa = WFA::new(2, alphabet_size);
    wfa.set_initial(0, CountingSemiring(1));
    wfa.set_initial(1, CountingSemiring(0));
    wfa.set_final(1, CountingSemiring(1));

    for a in 0..alphabet_size {
        wfa.add_transition(0, 0, a as u8, CountingSemiring(1));
    }
    wfa.add_transition(0, 1, symbol, CountingSemiring(1));
    for a in 0..alphabet_size {
        wfa.add_transition(1, 1, a as u8, CountingSemiring(1));
    }
    wfa
}

/// Build a simple boolean WFA that accepts strings containing a specific symbol.
pub fn build_boolean_wfa(symbol: u8, alphabet_size: usize) -> WFA<BooleanSemiring> {
    let mut wfa = WFA::new(2, alphabet_size);
    wfa.set_initial(0, BooleanSemiring(true));
    wfa.set_initial(1, BooleanSemiring(false));
    wfa.set_final(0, BooleanSemiring(false));
    wfa.set_final(1, BooleanSemiring(true));

    for a in 0..alphabet_size {
        if a as u8 == symbol {
            wfa.add_transition(0, 1, a as u8, BooleanSemiring(true));
        } else {
            wfa.add_transition(0, 0, a as u8, BooleanSemiring(true));
        }
    }
    for a in 0..alphabet_size {
        wfa.add_transition(1, 1, a as u8, BooleanSemiring(true));
    }
    wfa
}

/// Build a tropical WFA for shortest-path computation.
pub fn build_tropical_shortest_path_wfa(
    num_states: usize, edges: &[(usize, usize, u8, f64)], alphabet_size: usize,
) -> WFA<TropicalSemiring> {
    let mut wfa = WFA::new(num_states, alphabet_size);
    wfa.set_initial(0, TropicalSemiring(0.0));
    for s in 1..num_states { wfa.set_initial(s, TropicalSemiring(f64::INFINITY)); }
    wfa.set_final(num_states - 1, TropicalSemiring(0.0));
    for s in 0..num_states - 1 { wfa.set_final(s, TropicalSemiring(f64::INFINITY)); }

    for &(from, to, sym, weight) in edges {
        wfa.add_transition(from, to, sym, TropicalSemiring(weight));
    }
    wfa
}

/// Build a simple real-weighted WFA.
pub fn build_real_wfa(
    num_states: usize, alphabet_size: usize,
    transitions: &[(usize, usize, u8, f64)], initial: &[f64], final_w: &[f64],
) -> WFA<RealSemiring> {
    let mut wfa = WFA::new(num_states, alphabet_size);
    for (s, &w) in initial.iter().enumerate() { wfa.set_initial(s, RealSemiring(w)); }
    for (s, &w) in final_w.iter().enumerate() { wfa.set_final(s, RealSemiring(w)); }
    for &(from, to, sym, weight) in transitions {
        wfa.add_transition(from, to, sym, RealSemiring(weight));
    }
    wfa
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint Verifier
// ═══════════════════════════════════════════════════════════════════════════════

/// Verify that all constraints in a compilation result are satisfied by a trace.
pub fn verify_compilation(result: &CompilationResult, trace: &AIRTrace) -> Result<(), Vec<String>> {
    trace.verify_constraints(&result.air_program)
}

/// Result of verifying a single constraint.
#[derive(Clone, Debug)]
pub struct ConstraintVerificationResult {
    pub label: String,
    pub constraint_type: String,
    pub satisfied: bool,
    pub violation_row: Option<usize>,
    pub violation_value: u64,
}

impl fmt::Display for ConstraintVerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.satisfied {
            write!(f, "ok {} ({})", self.label, self.constraint_type)
        } else {
            write!(f, "FAIL {} ({}) violated at row {:?}, value = {}",
                self.label, self.constraint_type, self.violation_row, self.violation_value)
        }
    }
}

/// Detailed constraint verification with per-constraint status.
pub fn verify_constraints_detailed(result: &CompilationResult, trace: &AIRTrace) -> Vec<ConstraintVerificationResult> {
    let mut results = Vec::new();
    for constraint in &result.air_program.constraints {
        let mut satisfied = true;
        let mut violation_row = None;
        let mut violation_value = GoldilocksField::ZERO;

        match &constraint.constraint_type {
            ConstraintType::Boundary => {
                if let Some(row) = constraint.boundary_row {
                    if row < trace.num_rows {
                        let next = if row + 1 < trace.num_rows { Some(trace.values[row + 1].as_slice()) } else { None };
                        let val = constraint.expression.evaluate(&trace.values[row], next, Some(&trace.values), row);
                        if val.to_canonical() != 0 {
                            satisfied = false;
                            violation_row = Some(row);
                            violation_value = val;
                        }
                    }
                }
            }
            ConstraintType::Transition => {
                for row in 0..trace.num_rows.saturating_sub(1) {
                    let next = &trace.values[row + 1];
                    let val = constraint.expression.evaluate(&trace.values[row], Some(next), Some(&trace.values), row);
                    if val.to_canonical() != 0 {
                        satisfied = false;
                        violation_row = Some(row);
                        violation_value = val;
                        break;
                    }
                }
            }
            ConstraintType::Periodic(period) => {
                let mut row = 0;
                while row < trace.num_rows {
                    let next = if row + 1 < trace.num_rows { Some(trace.values[row + 1].as_slice()) } else { None };
                    let val = constraint.expression.evaluate(&trace.values[row], next, Some(&trace.values), row);
                    if val.to_canonical() != 0 {
                        satisfied = false;
                        violation_row = Some(row);
                        violation_value = val;
                        break;
                    }
                    row += period;
                }
            }
            ConstraintType::Global => {
                for row in 0..trace.num_rows {
                    let next = if row + 1 < trace.num_rows { Some(trace.values[row + 1].as_slice()) } else { None };
                    let val = constraint.expression.evaluate(&trace.values[row], next, Some(&trace.values), row);
                    if val.to_canonical() != 0 {
                        satisfied = false;
                        violation_row = Some(row);
                        violation_value = val;
                        break;
                    }
                }
            }
        }

        results.push(ConstraintVerificationResult {
            label: constraint.label.clone(),
            constraint_type: format!("{:?}", constraint.constraint_type),
            satisfied,
            violation_row,
            violation_value: violation_value.to_canonical(),
        });
    }
    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// Expression Builder DSL
// ═══════════════════════════════════════════════════════════════════════════════

/// Convenience builder for constructing AIR constraint expressions.
pub struct ExpressionBuilder {
    next_aux_col: usize,
}

impl ExpressionBuilder {
    pub fn new(start_col: usize) -> Self { ExpressionBuilder { next_aux_col: start_col } }

    pub fn col(&self, idx: usize) -> SymbolicExpression { SymbolicExpression::col(idx) }
    pub fn next(&self, idx: usize) -> SymbolicExpression { SymbolicExpression::next_col(idx) }
    pub fn constant(&self, val: u64) -> SymbolicExpression { SymbolicExpression::constant(val) }
    pub fn field_constant(&self, val: GoldilocksField) -> SymbolicExpression { SymbolicExpression::constant_field(val) }

    pub fn alloc_aux(&mut self) -> usize {
        let idx = self.next_aux_col;
        self.next_aux_col += 1;
        idx
    }

    /// Build a dot product expression: sum_i coeffs[i] * col(cols[i])
    pub fn dot_product_expr(&self, coeffs: &[GoldilocksField], cols: &[usize]) -> SymbolicExpression {
        let mut expr = SymbolicExpression::constant(0);
        for (&coeff, &col) in coeffs.iter().zip(cols.iter()) {
            if coeff.to_canonical() != 0 {
                let term = SymbolicExpression::constant_field(coeff).mul(SymbolicExpression::col(col));
                expr = expr.add(term);
            }
        }
        expr
    }

    pub fn matvec_element_expr(&self, matrix_row: &[GoldilocksField], input_cols: &[usize]) -> SymbolicExpression {
        self.dot_product_expr(matrix_row, input_cols)
    }

    pub fn assert_linear_combination(&self, result_col: usize, coeffs: &[GoldilocksField], input_cols: &[usize]) -> SymbolicExpression {
        let lc = self.dot_product_expr(coeffs, input_cols);
        SymbolicExpression::col(result_col).sub(lc)
    }

    pub fn selected_sum(&self, selector_cols: &[usize], value_exprs: &[SymbolicExpression]) -> SymbolicExpression {
        let mut expr = SymbolicExpression::constant(0);
        for (&sel, val) in selector_cols.iter().zip(value_exprs.iter()) {
            let term = SymbolicExpression::col(sel).mul(val.clone());
            expr = expr.add(term);
        }
        expr
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compilation Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/// End-to-end compilation and trace generation pipeline.
pub struct CompilationPipeline<S: Semiring + SemiringEmbedding> {
    compiler: WFACircuitCompiler<S>,
    trace_generator: TraceGenerator<S>,
}

impl<S: Semiring + SemiringEmbedding> CompilationPipeline<S> {
    pub fn new(config: CompilerConfig) -> Self {
        CompilationPipeline {
            compiler: WFACircuitCompiler::new(config),
            trace_generator: TraceGenerator::new(),
        }
    }

    pub fn with_default_config() -> Self { Self::new(CompilerConfig::default()) }

    pub fn compile_and_trace(&mut self, wfa: &WFA<S>, input: &[u8]) -> (CompilationResult, AIRTrace) {
        let result = self.compiler.compile(wfa, input.len());
        let trace = self.trace_generator.generate_trace(wfa, input, &result);
        (result, trace)
    }

    pub fn compile_trace_verify(&mut self, wfa: &WFA<S>, input: &[u8]) -> Result<(CompilationResult, AIRTrace), Vec<String>> {
        let (result, trace) = self.compile_and_trace(wfa, input);
        verify_compilation(&result, &trace)?;
        Ok((result, trace))
    }

    pub fn stats(&self) -> CompilationStats { self.compiler.compilation_statistics() }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint Graph Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Analyze the constraint dependency graph.
pub struct ConstraintGraph {
    pub adjacency: Vec<HashSet<usize>>,
    pub column_usage: HashMap<usize, Vec<usize>>,
    pub num_constraints: usize,
}

impl ConstraintGraph {
    pub fn build(constraints: &[AIRConstraint]) -> Self {
        let n = constraints.len();
        let mut column_usage: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut adjacency = vec![HashSet::new(); n];

        for (i, c) in constraints.iter().enumerate() {
            let cols = c.expression.referenced_columns();
            for col in cols { column_usage.entry(col).or_insert_with(Vec::new).push(i); }
        }

        for (_col, constraint_indices) in &column_usage {
            for &i in constraint_indices {
                for &j in constraint_indices {
                    if i != j { adjacency[i].insert(j); }
                }
            }
        }

        ConstraintGraph { adjacency, column_usage, num_constraints: n }
    }

    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_constraints];
        let mut components = Vec::new();

        for start in 0..self.num_constraints {
            if visited[start] { continue; }
            let mut component = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] { continue; }
                visited[node] = true;
                component.push(node);
                for &neighbor in &self.adjacency[node] {
                    if !visited[neighbor] { stack.push(neighbor); }
                }
            }
            component.sort();
            components.push(component);
        }
        components
    }

    pub fn max_degree(&self) -> usize { self.adjacency.iter().map(|s| s.len()).max().unwrap_or(0) }
    pub fn num_edges(&self) -> usize { self.adjacency.iter().map(|s| s.len()).sum::<usize>() / 2 }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Symbolic Expression Simplifier
// ═══════════════════════════════════════════════════════════════════════════════

/// Simplify a symbolic expression using algebraic identities.
pub fn simplify_expression(expr: &SymbolicExpression) -> SymbolicExpression {
    match expr {
        SymbolicExpression::Add(a, b) => {
            let sa = simplify_expression(a);
            let sb = simplify_expression(b);
            if sa.is_zero_constant() { return sb; }
            if sb.is_zero_constant() { return sa; }
            if let (SymbolicExpression::Constant(va), SymbolicExpression::Constant(vb)) = (&sa, &sb) {
                return SymbolicExpression::Constant(va.add_elem(*vb));
            }
            sa.add(sb)
        }
        SymbolicExpression::Sub(a, b) => {
            let sa = simplify_expression(a);
            let sb = simplify_expression(b);
            if sb.is_zero_constant() { return sa; }
            if let (SymbolicExpression::Constant(va), SymbolicExpression::Constant(vb)) = (&sa, &sb) {
                return SymbolicExpression::Constant(va.sub_elem(*vb));
            }
            sa.sub(sb)
        }
        SymbolicExpression::Mul(a, b) => {
            let sa = simplify_expression(a);
            let sb = simplify_expression(b);
            if sa.is_zero_constant() || sb.is_zero_constant() { return SymbolicExpression::constant(0); }
            if sa.is_one_constant() { return sb; }
            if sb.is_one_constant() { return sa; }
            if let (SymbolicExpression::Constant(va), SymbolicExpression::Constant(vb)) = (&sa, &sb) {
                return SymbolicExpression::Constant(va.mul_elem(*vb));
            }
            sa.mul(sb)
        }
        SymbolicExpression::Neg(a) => {
            let sa = simplify_expression(a);
            if let SymbolicExpression::Neg(inner) = &sa { return *inner.clone(); }
            if let SymbolicExpression::Constant(v) = &sa { return SymbolicExpression::Constant(v.neg_elem()); }
            sa.neg()
        }
        SymbolicExpression::Pow(base, exp) => {
            if *exp == 0 { return SymbolicExpression::constant(1); }
            let sb = simplify_expression(base);
            if *exp == 1 { return sb; }
            if let SymbolicExpression::Constant(v) = &sb { return SymbolicExpression::Constant(v.pow(*exp as u64)); }
            sb.pow(*exp)
        }
        other => other.clone(),
    }
}

/// Deeply simplify an expression (multiple passes until fixed point).
pub fn deep_simplify(expr: &SymbolicExpression) -> SymbolicExpression {
    let mut current = expr.clone();
    for _ in 0..10 {
        let simplified = simplify_expression(&current);
        let h1 = current.structural_hash();
        let h2 = simplified.structural_hash();
        if h1 == h2 { break; }
        current = simplified;
    }
    current
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-WFA Composition Compiler
// ═══════════════════════════════════════════════════════════════════════════════

/// Compile multiple WFAs into a single AIR program.
pub struct MultiWFACompiler<S: Semiring + SemiringEmbedding> {
    config: CompilerConfig,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Semiring + SemiringEmbedding> MultiWFACompiler<S> {
    pub fn new(config: CompilerConfig) -> Self {
        MultiWFACompiler { config, _phantom: std::marker::PhantomData }
    }

    /// Compile a product (intersection) of multiple WFAs.
    pub fn compile_product(&mut self, wfas: &[WFA<S>], input_length: usize) -> CompilationResult {
        let product = self.build_product_wfa(wfas);
        let mut compiler = WFACircuitCompiler::<S>::new(self.config.clone());
        compiler.compile(&product, input_length)
    }

    fn build_product_wfa(&self, wfas: &[WFA<S>]) -> WFA<S> {
        if wfas.is_empty() { return WFA::new(0, 0); }
        if wfas.len() == 1 { return wfas[0].clone(); }
        let mut result = self.product_of_two(&wfas[0], &wfas[1]);
        for i in 2..wfas.len() { result = self.product_of_two(&result, &wfas[i]); }
        result
    }

    fn product_of_two(&self, a: &WFA<S>, b: &WFA<S>) -> WFA<S> {
        let na = a.num_states;
        let nb = b.num_states;
        let n_product = na * nb;
        let alphabet = a.alphabet_size.max(b.alphabet_size);
        let mut product = WFA::new(n_product, alphabet);

        let state_idx = |i: usize, j: usize| -> usize { i * nb + j };

        for i in 0..na {
            for j in 0..nb {
                let w = a.initial_weights[i].mul(&b.initial_weights[j]);
                product.set_initial(state_idx(i, j), w);
            }
        }
        for i in 0..na {
            for j in 0..nb {
                let w = a.final_weights[i].mul(&b.final_weights[j]);
                product.set_final(state_idx(i, j), w);
            }
        }
        for ta in &a.transitions {
            for tb in &b.transitions {
                if ta.symbol == tb.symbol {
                    let from = state_idx(ta.from_state, tb.from_state);
                    let to = state_idx(ta.to_state, tb.to_state);
                    let w = ta.weight.mul(&tb.weight);
                    product.add_transition(from, to, ta.symbol, w);
                }
            }
        }
        product
    }

    /// Compile a union (sum) of multiple WFAs.
    pub fn compile_union(&mut self, wfas: &[WFA<S>], input_length: usize) -> CompilationResult {
        let union_wfa = self.build_union_wfa(wfas);
        let mut compiler = WFACircuitCompiler::<S>::new(self.config.clone());
        compiler.compile(&union_wfa, input_length)
    }

    fn build_union_wfa(&self, wfas: &[WFA<S>]) -> WFA<S> {
        if wfas.is_empty() { return WFA::new(0, 0); }
        let total_states: usize = wfas.iter().map(|w| w.num_states).sum();
        let max_alphabet = wfas.iter().map(|w| w.alphabet_size).max().unwrap_or(0);
        let mut union = WFA::new(total_states, max_alphabet);
        let mut state_offset = 0usize;
        for wfa in wfas {
            for s in 0..wfa.num_states { union.set_initial(state_offset + s, wfa.initial_weights[s].clone()); }
            for s in 0..wfa.num_states { union.set_final(state_offset + s, wfa.final_weights[s].clone()); }
            for t in &wfa.transitions {
                union.add_transition(state_offset + t.from_state, state_offset + t.to_state, t.symbol, t.weight.clone());
            }
            state_offset += wfa.num_states;
        }
        union
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compiler Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/// Stages of the compiler pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum CompilerStage {
    Parse,
    TypeCheck,
    LowerToIR,
    OptimizeIR,
    EmitConstraints,
    OptimizeConstraints,
    LayoutTrace,
    Finalize,
}

impl CompilerStage {
    /// Human-readable name of the stage.
    pub fn name(&self) -> &'static str {
        match self {
            CompilerStage::Parse => "Parse",
            CompilerStage::TypeCheck => "TypeCheck",
            CompilerStage::LowerToIR => "LowerToIR",
            CompilerStage::OptimizeIR => "OptimizeIR",
            CompilerStage::EmitConstraints => "EmitConstraints",
            CompilerStage::OptimizeConstraints => "OptimizeConstraints",
            CompilerStage::LayoutTrace => "LayoutTrace",
            CompilerStage::Finalize => "Finalize",
        }
    }
}

impl fmt::Display for CompilerStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A configurable, staged compilation pipeline.
pub struct CompilerPipeline {
    stages: Vec<CompilerStage>,
}

impl CompilerPipeline {
    /// Create the default pipeline with all stages.
    pub fn default_pipeline() -> Self {
        CompilerPipeline {
            stages: vec![
                CompilerStage::Parse,
                CompilerStage::TypeCheck,
                CompilerStage::LowerToIR,
                CompilerStage::OptimizeIR,
                CompilerStage::EmitConstraints,
                CompilerStage::OptimizeConstraints,
                CompilerStage::LayoutTrace,
                CompilerStage::Finalize,
            ],
        }
    }

    /// Create a pipeline with a single stage.
    pub fn with_stage(stage: CompilerStage) -> Self {
        CompilerPipeline {
            stages: vec![stage],
        }
    }

    /// Run the pipeline on a WFA, producing a compilation result.
    pub fn run<S: Semiring + SemiringEmbedding>(
        &self,
        wfa: &WFA<S>,
        input_length: usize,
    ) -> CompilationResult {
        let mut compiler = WFACircuitCompiler::<S>::new(CompilerConfig::default());

        // Execute each stage. The underlying compiler handles the real work;
        // we just drive it and optionally skip / reorder stages.
        for stage in &self.stages {
            match stage {
                CompilerStage::Parse
                | CompilerStage::TypeCheck
                | CompilerStage::LowerToIR
                | CompilerStage::OptimizeIR
                | CompilerStage::EmitConstraints
                | CompilerStage::OptimizeConstraints
                | CompilerStage::LayoutTrace
                | CompilerStage::Finalize => {
                    // All stages are handled by the monolithic compile() call
                    // below; individual stage hooks are for diagnostics only.
                }
            }
        }

        compiler.compile(wfa, input_length)
    }

    /// List the names of the stages in order.
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name().to_string()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compiler Diagnostics
// ═══════════════════════════════════════════════════════════════════════════════

/// Record of one compiler stage execution.
#[derive(Clone, Debug)]
struct StageDiagnostic {
    name: String,
    duration_us: u64,
    items_processed: usize,
}

/// Record of constraint counts at a particular stage.
#[derive(Clone, Debug)]
struct ConstraintCountRecord {
    stage: String,
    count: usize,
}

/// Record of an optimization pass.
#[derive(Clone, Debug)]
struct OptimizationRecord {
    name: String,
    before: usize,
    after: usize,
}

/// Diagnostics tracker for the compilation process.
pub struct CompilerDiagnostics {
    stages: Vec<StageDiagnostic>,
    constraint_counts: Vec<ConstraintCountRecord>,
    optimizations: Vec<OptimizationRecord>,
}

impl CompilerDiagnostics {
    /// Create a new empty diagnostics tracker.
    pub fn new() -> Self {
        CompilerDiagnostics {
            stages: Vec::new(),
            constraint_counts: Vec::new(),
            optimizations: Vec::new(),
        }
    }

    /// Record that a stage completed.
    pub fn record_stage(&mut self, name: &str, duration_us: u64, items_processed: usize) {
        self.stages.push(StageDiagnostic {
            name: name.to_string(),
            duration_us,
            items_processed,
        });
    }

    /// Record a constraint count at a particular stage.
    pub fn record_constraint_count(&mut self, stage: &str, count: usize) {
        self.constraint_counts.push(ConstraintCountRecord {
            stage: stage.to_string(),
            count,
        });
    }

    /// Record an optimization pass result.
    pub fn record_optimization(&mut self, name: &str, before: usize, after: usize) {
        self.optimizations.push(OptimizationRecord {
            name: name.to_string(),
            before,
            after,
        });
    }

    /// Produce a human-readable summary.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Compiler Diagnostics ===\n");
        out.push_str(&format!("Total stages: {}\n", self.stages.len()));
        for s in &self.stages {
            out.push_str(&format!(
                "  {} : {} µs, {} items\n",
                s.name, s.duration_us, s.items_processed
            ));
        }
        if !self.constraint_counts.is_empty() {
            out.push_str("Constraint counts:\n");
            for cc in &self.constraint_counts {
                out.push_str(&format!("  {} : {}\n", cc.stage, cc.count));
            }
        }
        if !self.optimizations.is_empty() {
            out.push_str("Optimizations:\n");
            for o in &self.optimizations {
                let pct = if o.before > 0 {
                    100.0 * (1.0 - o.after as f64 / o.before as f64)
                } else {
                    0.0
                };
                out.push_str(&format!(
                    "  {} : {} -> {} ({:.1}% reduction)\n",
                    o.name, o.before, o.after, pct
                ));
            }
        }
        out.push_str(&format!("Total time: {} µs\n", self.total_time_us()));
        out
    }

    /// Produce a JSON representation.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str("  \"stages\": [\n");
        for (i, s) in self.stages.iter().enumerate() {
            out.push_str(&format!(
                "    {{\"name\": \"{}\", \"duration_us\": {}, \"items_processed\": {}}}{}",
                s.name,
                s.duration_us,
                s.items_processed,
                if i + 1 < self.stages.len() { ",\n" } else { "\n" }
            ));
        }
        out.push_str("  ],\n");
        out.push_str(&format!("  \"total_time_us\": {}\n", self.total_time_us()));
        out.push_str("}");
        out
    }

    /// Total time across all recorded stages.
    pub fn total_time_us(&self) -> u64 {
        self.stages.iter().map(|s| s.duration_us).sum()
    }

    /// Return the optimization with the best (largest) percentage reduction,
    /// or `None` if no optimizations were recorded.
    pub fn best_optimization(&self) -> Option<(String, f64)> {
        self.optimizations
            .iter()
            .filter(|o| o.before > 0)
            .map(|o| {
                let pct = 100.0 * (1.0 - o.after as f64 / o.before as f64);
                (o.name.clone(), pct)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

impl Default for CompilerDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint Analyzer
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of analyzing a set of AIR constraints.
#[derive(Clone, Debug)]
pub struct ConstraintAnalysis {
    pub total: usize,
    pub by_type: HashMap<String, usize>,
    pub by_degree: HashMap<usize, usize>,
    pub max_degree: usize,
    pub avg_degree: f64,
    pub column_coverage: f64,
    pub redundant_pairs: Vec<(usize, usize)>,
}

/// Utilities for analyzing AIR constraint properties.
pub struct ConstraintAnalyzer;

impl ConstraintAnalyzer {
    /// Full analysis of a constraint set.
    pub fn analyze(constraints: &[AIRConstraint]) -> ConstraintAnalysis {
        let total = constraints.len();
        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut by_degree: HashMap<usize, usize> = HashMap::new();
        let mut max_degree = 0usize;
        let mut degree_sum = 0usize;

        for c in constraints {
            let type_name = match &c.constraint_type {
                ConstraintType::Boundary => "Boundary",
                ConstraintType::Transition => "Transition",
                ConstraintType::Periodic(_) => "Periodic",
                ConstraintType::Global => "Global",
            };
            *by_type.entry(type_name.to_string()).or_insert(0) += 1;
            *by_degree.entry(c.degree).or_insert(0) += 1;
            if c.degree > max_degree {
                max_degree = c.degree;
            }
            degree_sum += c.degree;
        }

        let avg_degree = if total > 0 {
            degree_sum as f64 / total as f64
        } else {
            0.0
        };

        let redundant_pairs = Self::find_redundant_pairs(constraints);

        ConstraintAnalysis {
            total,
            by_type,
            by_degree,
            max_degree,
            avg_degree,
            column_coverage: 0.0, // computed externally with num_cols
            redundant_pairs,
        }
    }

    /// Histogram of constraint degrees.
    pub fn degree_histogram(constraints: &[AIRConstraint]) -> HashMap<usize, usize> {
        let mut hist = HashMap::new();
        for c in constraints {
            *hist.entry(c.degree).or_insert(0) += 1;
        }
        hist
    }

    /// Count how many constraints reference each column.
    pub fn column_usage(constraints: &[AIRConstraint]) -> HashMap<usize, usize> {
        let mut usage: HashMap<usize, usize> = HashMap::new();
        for c in constraints {
            let cols = Self::collect_columns(&c.expression);
            for col in cols {
                *usage.entry(col).or_insert(0) += 1;
            }
        }
        usage
    }

    /// Fraction of columns that appear in at least one constraint.
    pub fn constraint_density(constraints: &[AIRConstraint], num_cols: usize) -> f64 {
        if num_cols == 0 {
            return 0.0;
        }
        let used = Self::column_usage(constraints).len();
        used as f64 / num_cols as f64
    }

    /// Find pairs of constraints with identical labels (likely duplicates).
    pub fn find_redundant_pairs(constraints: &[AIRConstraint]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..constraints.len() {
            for j in (i + 1)..constraints.len() {
                if constraints[i].label == constraints[j].label
                    && constraints[i].degree == constraints[j].degree
                    && constraints[i].constraint_type == constraints[j].constraint_type
                {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Estimate verification cost (sum of degrees).
    pub fn estimate_verification_cost(constraints: &[AIRConstraint]) -> usize {
        constraints.iter().map(|c| c.degree.max(1)).sum()
    }

    /// Collect all column indices referenced by an expression.
    fn collect_columns(expr: &SymbolicExpression) -> HashSet<usize> {
        let mut cols = HashSet::new();
        Self::walk_columns(expr, &mut cols);
        cols
    }

    fn walk_columns(expr: &SymbolicExpression, cols: &mut HashSet<usize>) {
        match expr {
            SymbolicExpression::Column(c) => { cols.insert(*c); }
            SymbolicExpression::NextColumn(c) => { cols.insert(*c); }
            SymbolicExpression::ColumnOffset(c, _) => { cols.insert(*c); }
            SymbolicExpression::Add(a, b) | SymbolicExpression::Mul(a, b) | SymbolicExpression::Sub(a, b) => {
                Self::walk_columns(a, cols);
                Self::walk_columns(b, cols);
            }
            SymbolicExpression::Neg(a) | SymbolicExpression::Pow(a, _) => {
                Self::walk_columns(a, cols);
            }
            SymbolicExpression::Constant(_) | SymbolicExpression::Variable(_) => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WFA Analyzer
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of analyzing a WFA before compilation.
#[derive(Clone, Debug)]
pub struct WFAAnalysis {
    pub num_states: usize,
    pub alphabet_size: usize,
    pub num_transitions: usize,
    pub density: f64,
    pub is_deterministic: bool,
    pub has_epsilon: bool,
    pub max_weight: f64,
    pub min_weight: f64,
    pub recommended_target: CompilationTarget,
}

/// Analyzer for WFA properties that inform compilation decisions.
pub struct WFAAnalyzer;

impl WFAAnalyzer {
    /// Analyze a WFA over the counting semiring.
    pub fn analyze_counting(wfa: &WFA<CountingSemiring>) -> WFAAnalysis {
        let max_w = wfa.transitions.iter().map(|t| t.weight.0 as f64).fold(0.0f64, f64::max);
        let min_w = wfa.transitions.iter().map(|t| t.weight.0 as f64).fold(f64::MAX, f64::min);
        Self::build_analysis(wfa.num_states, wfa.alphabet_size, &wfa.transitions, max_w, if wfa.transitions.is_empty() { 0.0 } else { min_w }, CompilationTarget::AlgebraicDirect)
    }

    /// Analyze a WFA over the tropical semiring.
    pub fn analyze_tropical(wfa: &WFA<TropicalSemiring>) -> WFAAnalysis {
        let max_w = wfa.transitions.iter().map(|t| t.weight.0).fold(f64::NEG_INFINITY, f64::max);
        let min_w = wfa.transitions.iter().map(|t| t.weight.0).fold(f64::INFINITY, f64::min);
        Self::build_analysis(wfa.num_states, wfa.alphabet_size, &wfa.transitions, if max_w.is_infinite() { 0.0 } else { max_w }, if min_w.is_infinite() { 0.0 } else { min_w }, CompilationTarget::GadgetAssisted)
    }

    /// Recommend a compilation target based on analysis.
    pub fn recommend_target(wfa_analysis: &WFAAnalysis) -> CompilationTarget {
        wfa_analysis.recommended_target.clone()
    }

    /// Estimate trace size for the given WFA and input length.
    pub fn estimate_trace_size(wfa_analysis: &WFAAnalysis, input_length: usize) -> usize {
        // Trace rows = input_length + 1 (for the initial state).
        // Trace width ≈ num_states + alphabet_size + auxiliary.
        let rows = input_length + 1;
        let cols = wfa_analysis.num_states + wfa_analysis.alphabet_size + 4;
        rows * cols
    }

    /// Estimate the number of constraints for a given WFA analysis.
    pub fn estimate_constraint_count(wfa_analysis: &WFAAnalysis) -> usize {
        // Boundary: num_states (initial) + num_states (final).
        // Transition: num_transitions.
        // Auxiliary: ~2 per state for selectors.
        let boundary = wfa_analysis.num_states * 2;
        let transition = wfa_analysis.num_transitions;
        let auxiliary = wfa_analysis.num_states * 2;
        boundary + transition + auxiliary
    }

    fn build_analysis<S: Semiring>(
        num_states: usize,
        alphabet_size: usize,
        transitions: &[WFATransition<S>],
        max_weight: f64,
        min_weight: f64,
        recommended_target: CompilationTarget,
    ) -> WFAAnalysis {
        let num_transitions = transitions.len();
        let max_possible = num_states * num_states * alphabet_size;
        let density = if max_possible > 0 {
            num_transitions as f64 / max_possible as f64
        } else {
            0.0
        };

        // Check determinism: for each (state, symbol) pair there is at most
        // one transition.
        let mut is_deterministic = true;
        let mut seen = HashSet::new();
        for t in transitions {
            if !seen.insert((t.from_state, t.symbol)) {
                is_deterministic = false;
                break;
            }
        }

        WFAAnalysis {
            num_states,
            alphabet_size,
            num_transitions,
            density,
            is_deterministic,
            has_epsilon: false, // epsilon transitions not modeled
            max_weight,
            min_weight,
            recommended_target,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Circuit Optimizer
// ═══════════════════════════════════════════════════════════════════════════════

/// Post-compilation circuit optimizations.
pub struct CircuitOptimizer;

impl CircuitOptimizer {
    /// Propagate constant expressions.
    ///
    /// If a constraint is trivially satisfied (e.g., `Constant(0) == 0`),
    /// remove it. If a sub-expression is a constant, simplify.
    pub fn constant_propagation(constraints: &[AIRConstraint]) -> Vec<AIRConstraint> {
        constraints
            .iter()
            .filter(|c| {
                // Remove constraints that are identically zero.
                !Self::is_trivially_zero(&c.expression)
            })
            .cloned()
            .collect()
    }

    /// Strength reduction: replace `x * x` with `x^2`, etc.
    pub fn strength_reduction(constraints: &[AIRConstraint]) -> Vec<AIRConstraint> {
        constraints
            .iter()
            .map(|c| {
                let mut new_c = c.clone();
                new_c.expression = Self::reduce_strength(&c.expression);
                new_c
            })
            .collect()
    }

    /// Inline auxiliary columns that are used in only one constraint.
    pub fn inline_auxiliary(
        constraints: &[AIRConstraint],
        trace_layout: &TraceLayout,
    ) -> Vec<AIRConstraint> {
        // Count usage of each auxiliary column.
        let aux_cols: HashSet<usize> = trace_layout
            .columns
            .iter()
            .filter(|c| c.column_type == ColumnType::Auxiliary)
            .map(|c| c.index)
            .collect();

        let usage = ConstraintAnalyzer::column_usage(constraints);
        let _single_use: HashSet<usize> = aux_cols
            .iter()
            .filter(|c| usage.get(c).copied().unwrap_or(0) <= 1)
            .copied()
            .collect();

        // For now, return constraints unchanged (full inlining requires
        // expression substitution which is complex).
        constraints.to_vec()
    }

    /// Remove columns from the AIR program that no constraint references.
    pub fn remove_unused_columns(air: &AIRProgram) -> AIRProgram {
        let usage = ConstraintAnalyzer::column_usage(&air.constraints);
        let used_cols: HashSet<usize> = usage.keys().copied().collect();

        let mut new_layout = TraceLayout::new();
        new_layout.num_rows = air.trace_layout.num_rows;
        for col in &air.trace_layout.columns {
            if used_cols.contains(&col.index)
                || col.column_type == ColumnType::PublicInput
                || col.column_type == ColumnType::PublicOutput
            {
                new_layout.add_column(&col.name, col.column_type.clone(), &col.description);
            }
        }

        let mut new_air = AIRProgram::new(new_layout, air.trace_length);
        new_air.num_public_inputs = air.num_public_inputs;
        for c in &air.constraints {
            new_air.add_constraint(c.clone());
        }
        new_air
    }

    /// Apply all optimization passes.
    pub fn optimize_fully(air: &AIRProgram) -> AIRProgram {
        let c1 = Self::constant_propagation(&air.constraints);
        let c2 = Self::strength_reduction(&c1);

        let mut optimized = AIRProgram::new(air.trace_layout.clone(), air.trace_length);
        optimized.num_public_inputs = air.num_public_inputs;
        for c in c2 {
            optimized.add_constraint(c);
        }
        Self::remove_unused_columns(&optimized)
    }

    fn is_trivially_zero(expr: &SymbolicExpression) -> bool {
        match expr {
            SymbolicExpression::Constant(v) => v.to_canonical() == 0,
            _ => false,
        }
    }

    fn reduce_strength(expr: &SymbolicExpression) -> SymbolicExpression {
        match expr {
            SymbolicExpression::Mul(a, b) => {
                // Detect x * x → x^2
                if Self::expr_eq(a, b) {
                    return SymbolicExpression::Pow(a.clone(), 2);
                }
                SymbolicExpression::Mul(
                    Box::new(Self::reduce_strength(a)),
                    Box::new(Self::reduce_strength(b)),
                )
            }
            SymbolicExpression::Add(a, b) => {
                SymbolicExpression::Add(
                    Box::new(Self::reduce_strength(a)),
                    Box::new(Self::reduce_strength(b)),
                )
            }
            SymbolicExpression::Sub(a, b) => {
                SymbolicExpression::Sub(
                    Box::new(Self::reduce_strength(a)),
                    Box::new(Self::reduce_strength(b)),
                )
            }
            SymbolicExpression::Neg(a) => {
                SymbolicExpression::Neg(Box::new(Self::reduce_strength(a)))
            }
            SymbolicExpression::Pow(a, e) => {
                SymbolicExpression::Pow(Box::new(Self::reduce_strength(a)), *e)
            }
            other => other.clone(),
        }
    }

    /// Simple structural equality for expressions.
    fn expr_eq(a: &SymbolicExpression, b: &SymbolicExpression) -> bool {
        match (a, b) {
            (SymbolicExpression::Column(x), SymbolicExpression::Column(y)) => x == y,
            (SymbolicExpression::NextColumn(x), SymbolicExpression::NextColumn(y)) => x == y,
            (SymbolicExpression::Constant(x), SymbolicExpression::Constant(y)) => {
                x.to_canonical() == y.to_canonical()
            }
            (SymbolicExpression::Variable(x), SymbolicExpression::Variable(y)) => x == y,
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trace Layout Optimizer
// ═══════════════════════════════════════════════════════════════════════════════

/// Optimizer for trace column layouts.
pub struct TraceLayoutOptimizer;

impl TraceLayoutOptimizer {
    /// Optimize the layout by reordering columns for better locality.
    pub fn optimize(
        layout: &TraceLayout,
        constraints: &[AIRConstraint],
    ) -> TraceLayout {
        // Sort columns by usage frequency (most-used first for cache friendliness).
        let usage = ConstraintAnalyzer::column_usage(constraints);
        let mut col_order: Vec<(usize, usize)> = layout
            .columns
            .iter()
            .map(|c| (c.index, *usage.get(&c.index).unwrap_or(&0)))
            .collect();
        col_order.sort_by(|a, b| b.1.cmp(&a.1));

        let mut new_layout = TraceLayout::new();
        new_layout.num_rows = layout.num_rows;
        for (old_idx, _) in &col_order {
            let col = &layout.columns[*old_idx];
            new_layout.add_column(&col.name, col.column_type.clone(), &col.description);
        }
        new_layout
    }

    /// Minimize the number of trace columns by removing unused ones.
    pub fn minimize_width(
        layout: &TraceLayout,
        constraints: &[AIRConstraint],
    ) -> TraceLayout {
        let usage = ConstraintAnalyzer::column_usage(constraints);
        let mut new_layout = TraceLayout::new();
        new_layout.num_rows = layout.num_rows;
        for col in &layout.columns {
            if usage.contains_key(&col.index)
                || col.column_type == ColumnType::PublicInput
                || col.column_type == ColumnType::PublicOutput
            {
                new_layout.add_column(&col.name, col.column_type.clone(), &col.description);
            }
        }
        new_layout
    }

    /// Group columns by their access pattern in constraints.
    ///
    /// Columns that appear together in the same constraint are placed
    /// adjacent in the layout for improved cache behaviour.
    pub fn group_by_access_pattern(
        layout: &TraceLayout,
        constraints: &[AIRConstraint],
    ) -> TraceLayout {
        // Build a co-occurrence matrix.
        let n = layout.columns.len();
        let mut cooccurrence = vec![vec![0usize; n]; n];
        for c in constraints {
            let cols: Vec<usize> = ConstraintAnalyzer::column_usage(&[c.clone()])
                .keys()
                .copied()
                .filter(|&idx| idx < n)
                .collect();
            for i in 0..cols.len() {
                for j in (i + 1)..cols.len() {
                    cooccurrence[cols[i]][cols[j]] += 1;
                    cooccurrence[cols[j]][cols[i]] += 1;
                }
            }
        }

        // Greedy ordering: start with the most-used column, then always pick
        // the column with the highest co-occurrence to the last placed.
        let usage = ConstraintAnalyzer::column_usage(constraints);
        let mut ordered: Vec<usize> = Vec::with_capacity(n);
        let mut remaining: HashSet<usize> = (0..n).collect();

        // Seed with the most-used column.
        let seed = remaining
            .iter()
            .copied()
            .max_by_key(|c| usage.get(c).copied().unwrap_or(0))
            .unwrap_or(0);
        ordered.push(seed);
        remaining.remove(&seed);

        while !remaining.is_empty() {
            let last = *ordered.last().unwrap();
            let next = remaining
                .iter()
                .copied()
                .max_by_key(|&c| cooccurrence[last][c])
                .unwrap();
            ordered.push(next);
            remaining.remove(&next);
        }

        let mut new_layout = TraceLayout::new();
        new_layout.num_rows = layout.num_rows;
        for &idx in &ordered {
            if idx < layout.columns.len() {
                let col = &layout.columns[idx];
                new_layout.add_column(&col.name, col.column_type.clone(), &col.description);
            }
        }
        new_layout
    }

    /// Estimate cache efficiency as a score in [0, 1].
    ///
    /// Higher means columns that are accessed together tend to be adjacent.
    pub fn estimate_cache_efficiency(
        layout: &TraceLayout,
        constraints: &[AIRConstraint],
    ) -> f64 {
        if layout.columns.is_empty() || constraints.is_empty() {
            return 1.0;
        }

        // Build column index → position map.
        let pos: HashMap<usize, usize> = layout
            .columns
            .iter()
            .enumerate()
            .map(|(p, c)| (c.index, p))
            .collect();

        let mut total_span = 0usize;
        let mut constraint_count = 0usize;
        for c in constraints {
            let cols: Vec<usize> = ConstraintAnalyzer::column_usage(&[c.clone()])
                .keys()
                .copied()
                .collect();
            if cols.len() < 2 {
                continue;
            }
            let positions: Vec<usize> = cols.iter().filter_map(|c| pos.get(c).copied()).collect();
            if positions.len() < 2 {
                continue;
            }
            let min_p = *positions.iter().min().unwrap();
            let max_p = *positions.iter().max().unwrap();
            total_span += max_p - min_p;
            constraint_count += 1;
        }

        if constraint_count == 0 {
            return 1.0;
        }

        let avg_span = total_span as f64 / constraint_count as f64;
        let max_span = layout.columns.len() as f64;
        1.0 - (avg_span / max_span).min(1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compilation Report
// ═══════════════════════════════════════════════════════════════════════════════

/// A detailed report of a compilation run.
#[derive(Clone, Debug)]
pub struct CompilationReport {
    pub trace_width: usize,
    pub trace_length: usize,
    pub num_constraints: usize,
    pub max_degree: usize,
    pub num_boundary: usize,
    pub num_transition: usize,
    pub estimated_proof_size: usize,
    pub num_states: usize,
    pub num_transitions: usize,
}

impl CompilationReport {
    /// Build a report from a compilation result.
    pub fn from_result(result: &CompilationResult) -> Self {
        CompilationReport {
            trace_width: result.compilation_stats.trace_width,
            trace_length: result.compilation_stats.trace_length,
            num_constraints: result.compilation_stats.num_constraints,
            max_degree: result.compilation_stats.max_degree,
            num_boundary: result.compilation_stats.num_boundary,
            num_transition: result.compilation_stats.num_transition,
            estimated_proof_size: result.compilation_stats.estimated_proof_size_bytes,
            num_states: result.compilation_stats.num_wfa_states,
            num_transitions: result.compilation_stats.num_wfa_transitions,
        }
    }

    /// Render the report as plain text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Compilation Report ===\n");
        out.push_str(&format!("WFA: {} states, {} transitions\n", self.num_states, self.num_transitions));
        out.push_str(&format!("Trace: {} rows x {} columns\n", self.trace_length, self.trace_width));
        out.push_str(&format!("Constraints: {} total (max degree {})\n", self.num_constraints, self.max_degree));
        out.push_str(&format!("  Boundary: {}\n", self.num_boundary));
        out.push_str(&format!("  Transition: {}\n", self.num_transition));
        out.push_str(&format!("Estimated proof size: {} bytes\n", self.estimated_proof_size));
        out
    }

    /// Render the report as JSON.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"trace_width\":{},\"trace_length\":{},\"num_constraints\":{},\"max_degree\":{},\
             \"num_boundary\":{},\"num_transition\":{},\"estimated_proof_size\":{},\
             \"num_states\":{},\"num_transitions\":{}}}",
            self.trace_width,
            self.trace_length,
            self.num_constraints,
            self.max_degree,
            self.num_boundary,
            self.num_transition,
            self.estimated_proof_size,
            self.num_states,
            self.num_transitions,
        )
    }

    /// Render the report as HTML.
    pub fn to_html(&self) -> String {
        let mut out = String::new();
        out.push_str("<div class=\"compilation-report\">\n");
        out.push_str("<h2>Compilation Report</h2>\n");
        out.push_str("<table>\n");
        out.push_str(&format!("<tr><td>WFA States</td><td>{}</td></tr>\n", self.num_states));
        out.push_str(&format!("<tr><td>WFA Transitions</td><td>{}</td></tr>\n", self.num_transitions));
        out.push_str(&format!("<tr><td>Trace Width</td><td>{}</td></tr>\n", self.trace_width));
        out.push_str(&format!("<tr><td>Trace Length</td><td>{}</td></tr>\n", self.trace_length));
        out.push_str(&format!("<tr><td>Constraints</td><td>{}</td></tr>\n", self.num_constraints));
        out.push_str(&format!("<tr><td>Max Degree</td><td>{}</td></tr>\n", self.max_degree));
        out.push_str(&format!("<tr><td>Boundary</td><td>{}</td></tr>\n", self.num_boundary));
        out.push_str(&format!("<tr><td>Transition</td><td>{}</td></tr>\n", self.num_transition));
        out.push_str(&format!("<tr><td>Est. Proof Size</td><td>{} bytes</td></tr>\n", self.estimated_proof_size));
        out.push_str("</table>\n");
        out.push_str("</div>\n");
        out
    }

    /// Compare multiple reports side-by-side.
    pub fn comparison(reports: &[CompilationReport]) -> String {
        if reports.is_empty() {
            return "No reports to compare.\n".to_string();
        }
        let mut out = String::new();
        out.push_str("=== Report Comparison ===\n");
        out.push_str(&format!("{:<20}", "Metric"));
        for (i, _) in reports.iter().enumerate() {
            out.push_str(&format!("{:<15}", format!("Run {}", i)));
        }
        out.push('\n');

        let metrics: Vec<(&str, Box<dyn Fn(&CompilationReport) -> String>)> = vec![
            ("States", Box::new(|r: &CompilationReport| r.num_states.to_string())),
            ("Transitions", Box::new(|r: &CompilationReport| r.num_transitions.to_string())),
            ("Trace Width", Box::new(|r: &CompilationReport| r.trace_width.to_string())),
            ("Trace Length", Box::new(|r: &CompilationReport| r.trace_length.to_string())),
            ("Constraints", Box::new(|r: &CompilationReport| r.num_constraints.to_string())),
            ("Max Degree", Box::new(|r: &CompilationReport| r.max_degree.to_string())),
            ("Proof Size", Box::new(|r: &CompilationReport| r.estimated_proof_size.to_string())),
        ];

        for (name, getter) in &metrics {
            out.push_str(&format!("{:<20}", name));
            for r in reports {
                out.push_str(&format!("{:<15}", getter(r)));
            }
            out.push('\n');
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Semiring Tests ──────────────────────────────────────────────────

    #[test]
    fn test_counting_semiring_properties() {
        let zero = CountingSemiring::zero();
        let one = CountingSemiring::one();
        let a = CountingSemiring(5);
        let b = CountingSemiring(3);

        assert_eq!(a.add(&zero), a);
        assert_eq!(zero.add(&a), a);
        assert_eq!(a.mul(&one), a);
        assert_eq!(one.mul(&a), a);
        assert_eq!(a.mul(&zero), zero);
        assert_eq!(a.add(&b), b.add(&a));
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.add(&b), CountingSemiring(8));
        assert_eq!(a.mul(&b), CountingSemiring(15));
    }

    #[test]
    fn test_boolean_semiring_properties() {
        let zero = BooleanSemiring::zero();
        let one = BooleanSemiring::one();
        assert_eq!(zero.add(&zero), zero);
        assert_eq!(zero.add(&one), one);
        assert_eq!(one.add(&zero), one);
        assert_eq!(one.add(&one), one);
        assert_eq!(zero.mul(&zero), zero);
        assert_eq!(zero.mul(&one), zero);
        assert_eq!(one.mul(&zero), zero);
        assert_eq!(one.mul(&one), one);
    }

    #[test]
    fn test_tropical_semiring_properties() {
        let zero = TropicalSemiring::zero();
        let one = TropicalSemiring::one();
        let a = TropicalSemiring(3.0);
        let b = TropicalSemiring(5.0);

        assert_eq!(a.add(&b), a); // min(3, 5) = 3
        assert_eq!(a.add(&zero), a);
        assert_eq!(zero.add(&a), a);
        assert_eq!(a.mul(&b), TropicalSemiring(8.0));
        assert_eq!(a.mul(&one), a);
    }

    #[test]
    fn test_real_semiring_properties() {
        let zero = RealSemiring::zero();
        let one = RealSemiring::one();
        let a = RealSemiring(2.5);
        let b = RealSemiring(4.0);
        assert_eq!(a.add(&zero), a);
        assert_eq!(a.mul(&one), a);
        assert_eq!(a.mul(&zero), zero);
        assert_eq!(a.add(&b), RealSemiring(6.5));
        assert_eq!(a.mul(&b), RealSemiring(10.0));
    }

    // ─── Embedding Tests ────────────────────────────────────────────────

    #[test]
    fn test_counting_embedding_roundtrip() {
        for val in [0u64, 1, 42, 1000, 999999] {
            let s = CountingSemiring(val);
            let field = s.embed();
            let extracted = CountingSemiring::extract(field);
            assert_eq!(extracted.0, val);
        }
    }

    #[test]
    fn test_boolean_embedding_roundtrip() {
        let t = BooleanSemiring(true);
        let f = BooleanSemiring(false);
        assert_eq!(t.embed(), GoldilocksField::ONE);
        assert_eq!(f.embed(), GoldilocksField::ZERO);
        assert!(BooleanSemiring::extract(GoldilocksField::ONE).0);
        assert!(!BooleanSemiring::extract(GoldilocksField::ZERO).0);
    }

    #[test]
    fn test_tropical_embedding_roundtrip() {
        for val in [0.0, 1.0, -1.0, 3.14, 100.0, -50.5] {
            let s = TropicalSemiring(val);
            let field = s.embed();
            let extracted = TropicalSemiring::extract(field);
            assert!((extracted.0 - val).abs() < 1e-6, "Tropical roundtrip failed for {}: got {}", val, extracted.0);
        }
        let inf = TropicalSemiring(f64::INFINITY);
        assert!(TropicalSemiring::extract(inf.embed()).0.is_infinite());
    }

    #[test]
    fn test_real_embedding_roundtrip() {
        for val in [0.0, 1.0, -1.0, 2.5, -3.75, 100.125] {
            let s = RealSemiring(val);
            let field = s.embed();
            let extracted = RealSemiring::extract(field);
            assert!((extracted.0 - val).abs() < 1e-6, "Real roundtrip failed for {}: got {}", val, extracted.0);
        }
    }

    // ─── WFA Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_wfa_construction() {
        let mut wfa = WFA::<CountingSemiring>::new(3, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 2, 1, CountingSemiring(1));
        wfa.set_final(2, CountingSemiring(1));

        assert_eq!(wfa.num_states, 3);
        assert_eq!(wfa.alphabet_size, 2);
        assert_eq!(wfa.transitions.len(), 2);
        assert_eq!(wfa.transitions_for_symbol(0).len(), 1);
        assert_eq!(wfa.transitions_for_symbol(1).len(), 1);
        assert_eq!(wfa.transitions_from(0, 0).len(), 1);
        assert_eq!(wfa.transitions_from(0, 1).len(), 0);
    }

    #[test]
    fn test_counting_wfa_builder() {
        let wfa = build_counting_wfa(1, 2);
        assert_eq!(wfa.num_states, 2);
        assert_eq!(wfa.alphabet_size, 2);
        assert!(wfa.transitions.len() > 0);
    }

    #[test]
    fn test_boolean_wfa_builder() {
        let wfa = build_boolean_wfa(1, 2);
        assert_eq!(wfa.num_states, 2);
        assert_eq!(wfa.alphabet_size, 2);
    }

    // ─── Symbolic Expression Tests ──────────────────────────────────────

    #[test]
    fn test_expression_degree() {
        assert_eq!(SymbolicExpression::constant(5).degree(), 0);
        assert_eq!(SymbolicExpression::col(0).degree(), 1);
        assert_eq!(SymbolicExpression::col(0).mul(SymbolicExpression::col(1)).degree(), 2);
        assert_eq!(SymbolicExpression::col(0).mul(SymbolicExpression::col(1)).mul(SymbolicExpression::col(2)).degree(), 3);
        assert_eq!(SymbolicExpression::col(0).add(SymbolicExpression::col(1)).degree(), 1);
        assert_eq!(SymbolicExpression::col(0).pow(4).degree(), 4);
    }

    #[test]
    fn test_expression_evaluation() {
        let row = vec![GoldilocksField::new(3), GoldilocksField::new(7), GoldilocksField::new(5)];
        let next = vec![GoldilocksField::new(10), GoldilocksField::new(20), GoldilocksField::new(30)];

        let expr = SymbolicExpression::col(0).add(SymbolicExpression::col(1));
        assert_eq!(expr.evaluate(&row, Some(&next), None, 0).to_canonical(), 10);

        let expr2 = SymbolicExpression::col(0).mul(SymbolicExpression::col(1));
        assert_eq!(expr2.evaluate(&row, Some(&next), None, 0).to_canonical(), 21);

        assert_eq!(SymbolicExpression::next_col(0).evaluate(&row, Some(&next), None, 0).to_canonical(), 10);

        let expr4 = SymbolicExpression::col(0).sub(SymbolicExpression::constant(3));
        assert_eq!(expr4.evaluate(&row, Some(&next), None, 0).to_canonical(), 0);
    }

    #[test]
    fn test_expression_referenced_columns() {
        let expr = SymbolicExpression::col(0).mul(SymbolicExpression::col(2)).add(SymbolicExpression::next_col(1));
        let cols = expr.referenced_columns();
        assert!(cols.contains(&0));
        assert!(cols.contains(&1));
        assert!(cols.contains(&2));
        assert_eq!(cols.len(), 3);
    }

    #[test]
    fn test_expression_structural_hash() {
        let a = SymbolicExpression::col(0).add(SymbolicExpression::col(1));
        let b = SymbolicExpression::col(0).add(SymbolicExpression::col(1));
        let c = SymbolicExpression::col(1).add(SymbolicExpression::col(0));
        assert_eq!(a.structural_hash(), b.structural_hash());
        assert_ne!(a.structural_hash(), c.structural_hash());
    }

    // ─── Simplification Tests ───────────────────────────────────────────

    #[test]
    fn test_simplify_zero_add() {
        let expr = SymbolicExpression::constant(0).add(SymbolicExpression::col(3));
        match simplify_expression(&expr) {
            SymbolicExpression::Column(3) => {}
            other => panic!("Expected Column(3), got {:?}", other),
        }
    }

    #[test]
    fn test_simplify_zero_mul() {
        let expr = SymbolicExpression::constant(0).mul(SymbolicExpression::col(3));
        assert!(simplify_expression(&expr).is_zero_constant());
    }

    #[test]
    fn test_simplify_one_mul() {
        let expr = SymbolicExpression::constant(1).mul(SymbolicExpression::col(5));
        match simplify_expression(&expr) {
            SymbolicExpression::Column(5) => {}
            other => panic!("Expected Column(5), got {:?}", other),
        }
    }

    #[test]
    fn test_simplify_constant_fold() {
        let expr = SymbolicExpression::constant(3).add(SymbolicExpression::constant(7));
        match simplify_expression(&expr) {
            SymbolicExpression::Constant(v) => assert_eq!(v.to_canonical(), 10),
            other => panic!("Expected constant 10, got {:?}", other),
        }
    }

    #[test]
    fn test_simplify_double_neg() {
        let expr = SymbolicExpression::col(2).neg().neg();
        match simplify_expression(&expr) {
            SymbolicExpression::Column(2) => {}
            other => panic!("Expected Column(2), got {:?}", other),
        }
    }

    #[test]
    fn test_simplify_pow_zero() {
        assert!(simplify_expression(&SymbolicExpression::col(0).pow(0)).is_one_constant());
    }

    #[test]
    fn test_simplify_pow_one() {
        match simplify_expression(&SymbolicExpression::col(0).pow(1)) {
            SymbolicExpression::Column(0) => {}
            other => panic!("Expected Column(0), got {:?}", other),
        }
    }

    // ─── Matrix Utility Tests ───────────────────────────────────────────

    #[test]
    fn test_matrix_multiply_identity() {
        let n = 3;
        let id = identity_matrix(n);
        let m = vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)],
            vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)],
            vec![GoldilocksField::new(7), GoldilocksField::new(8), GoldilocksField::new(9)],
        ];
        let result = matrix_multiply(&m, &id);
        for i in 0..n { for j in 0..n { assert_eq!(result[i][j], m[i][j]); } }
    }

    #[test]
    fn test_matrix_multiply_2x2() {
        let a = vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ];
        let b = vec![
            vec![GoldilocksField::new(5), GoldilocksField::new(6)],
            vec![GoldilocksField::new(7), GoldilocksField::new(8)],
        ];
        let result = matrix_multiply(&a, &b);
        assert_eq!(result[0][0].to_canonical(), 19);
        assert_eq!(result[0][1].to_canonical(), 22);
        assert_eq!(result[1][0].to_canonical(), 43);
        assert_eq!(result[1][1].to_canonical(), 50);
    }

    #[test]
    fn test_matrix_vector_multiply_basic() {
        let m = vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ];
        let v = vec![GoldilocksField::new(5), GoldilocksField::new(6)];
        let result = matrix_vector_multiply(&m, &v);
        assert_eq!(result[0].to_canonical(), 17);
        assert_eq!(result[1].to_canonical(), 39);
    }

    #[test]
    fn test_matrix_power() {
        let m = vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::new(0)],
        ];
        let m0 = matrix_power(&m, 0);
        assert_eq!(m0[0][0].to_canonical(), 1);
        assert_eq!(m0[0][1].to_canonical(), 0);
        assert_eq!(m0[1][0].to_canonical(), 0);
        assert_eq!(m0[1][1].to_canonical(), 1);

        let m2 = matrix_power(&m, 2);
        assert_eq!(m2[0][0].to_canonical(), 2);
        assert_eq!(m2[0][1].to_canonical(), 1);
        assert_eq!(m2[1][0].to_canonical(), 1);
        assert_eq!(m2[1][1].to_canonical(), 1);

        let m6 = matrix_power(&m, 6);
        assert_eq!(m6[0][0].to_canonical(), 13);
        assert_eq!(m6[0][1].to_canonical(), 8);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)],
            vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)],
        ];
        let t = matrix_transpose(&m);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert_eq!(t[0][0].to_canonical(), 1);
        assert_eq!(t[0][1].to_canonical(), 4);
        assert_eq!(t[2][1].to_canonical(), 6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)];
        let b = vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)];
        assert_eq!(dot_product(&a, &b).to_canonical(), 32);
    }

    // ─── Trace Layout Tests ─────────────────────────────────────────────

    #[test]
    fn test_trace_layout_construction() {
        let mut layout = TraceLayout::new();
        let c0 = layout.add_column("state_0", ColumnType::Witness, "State 0");
        let c1 = layout.add_column("state_1", ColumnType::Witness, "State 1");
        let c2 = layout.add_column("input", ColumnType::PublicInput, "Input");
        let c3 = layout.add_column("aux", ColumnType::Auxiliary, "Auxiliary");

        assert_eq!(c0, 0);
        assert_eq!(c1, 1);
        assert_eq!(c2, 2);
        assert_eq!(c3, 3);
        assert_eq!(layout.total_columns(), 4);
        assert_eq!(layout.num_witness_columns, 2);
        assert_eq!(layout.num_public_columns, 1);
        assert_eq!(layout.num_auxiliary_columns, 1);
        assert_eq!(layout.find_column("state_0"), Some(0));
        assert_eq!(layout.find_column("missing"), None);
        assert_eq!(layout.columns_of_type(&ColumnType::Witness), vec![0, 1]);
    }

    // ─── AIR Constraint Tests ───────────────────────────────────────────

    #[test]
    fn test_air_constraint_boundary() {
        let expr = SymbolicExpression::col(0).sub(SymbolicExpression::constant(42));
        let constraint = AIRConstraint::boundary(expr, 0, "init");
        assert_eq!(constraint.constraint_type, ConstraintType::Boundary);
        assert_eq!(constraint.boundary_row, Some(0));
        assert_eq!(constraint.degree, 1);

        assert!(constraint.check_at_row(&[GoldilocksField::new(42)], None, None, 0));
        assert!(!constraint.check_at_row(&[GoldilocksField::new(43)], None, None, 0));
    }

    #[test]
    fn test_air_constraint_transition() {
        let expr = SymbolicExpression::next_col(0).sub(SymbolicExpression::col(0)).sub(SymbolicExpression::constant(1));
        let constraint = AIRConstraint::transition(expr, "increment");

        assert!(constraint.check_at_row(&[GoldilocksField::new(5)], Some(&[GoldilocksField::new(6)]), None, 0));
        assert!(!constraint.check_at_row(&[GoldilocksField::new(5)], Some(&[GoldilocksField::new(7)]), None, 0));
    }

    // ─── AIR Trace Tests ────────────────────────────────────────────────

    #[test]
    fn test_air_trace_verify_increment() {
        let mut trace = AIRTrace::new(4, 1);
        for i in 0..4 { trace.set(i, 0, GoldilocksField::new(i as u64)); }

        let mut program = AIRProgram::new(TraceLayout::new(), 4);
        program.add_constraint(AIRConstraint::boundary(SymbolicExpression::col(0), 0, "init_zero"));
        let inc = SymbolicExpression::next_col(0).sub(SymbolicExpression::col(0)).sub(SymbolicExpression::constant(1));
        program.add_constraint(AIRConstraint::transition(inc, "increment"));

        assert!(trace.verify_constraints(&program).is_ok());
    }

    #[test]
    fn test_air_trace_verify_failure() {
        let mut trace = AIRTrace::new(4, 1);
        trace.set(0, 0, GoldilocksField::new(0));
        trace.set(1, 0, GoldilocksField::new(1));
        trace.set(2, 0, GoldilocksField::new(5));
        trace.set(3, 0, GoldilocksField::new(3));

        let mut program = AIRProgram::new(TraceLayout::new(), 4);
        let inc = SymbolicExpression::next_col(0).sub(SymbolicExpression::col(0)).sub(SymbolicExpression::constant(1));
        program.add_constraint(AIRConstraint::transition(inc, "increment"));
        assert!(trace.verify_constraints(&program).is_err());
    }

    // ─── IR Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_intermediate_representation() {
        let mut ir = IntermediateRepresentation::new();
        let s0 = ir.add_variable("state_0", IRVariableType::State);
        let s1 = ir.add_variable("state_1", IRVariableType::State);
        let input = ir.add_variable("input", IRVariableType::Input);
        let aux = ir.add_variable("aux", IRVariableType::Auxiliary);

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(input, 2);
        assert_eq!(aux, 3);
        assert_eq!(ir.num_variables(), 4);
        assert_eq!(ir.variables_of_type(&IRVariableType::State).len(), 2);

        ir.assign_trace_columns();
        assert_eq!(ir.variables[0].trace_column, Some(0));
        assert_eq!(ir.variables[3].trace_column, Some(3));
    }

    // ─── Compiler Config Tests ──────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = CompilerConfig::default();
        assert_eq!(config.target, CompilationTarget::Hybrid);
        assert_eq!(config.optimization_level, 1);
        assert_eq!(config.max_constraint_degree, 4);
        assert!(config.enable_cse);
        assert!(config.enable_dead_elimination);
        assert!(!config.enable_batching);
    }

    // ─── Transition Matrix Tests ────────────────────────────────────────

    #[test]
    fn test_transition_matrix_counting() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(2));
        wfa.add_transition(1, 1, 0, CountingSemiring(3));
        wfa.add_transition(0, 0, 1, CountingSemiring(4));
        wfa.add_transition(1, 0, 1, CountingSemiring(5));

        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let m0 = compiler.compute_transition_matrix(&wfa, 0);
        assert_eq!(m0[0][0].to_canonical(), 1);
        assert_eq!(m0[1][0].to_canonical(), 2);
        assert_eq!(m0[1][1].to_canonical(), 3);
        assert_eq!(m0[0][1].to_canonical(), 0);

        let m1 = compiler.compute_transition_matrix(&wfa, 1);
        assert_eq!(m1[0][0].to_canonical(), 4);
        assert_eq!(m1[0][1].to_canonical(), 5);
    }

    #[test]
    fn test_embed_all_transition_matrices() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 3);
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 0, 1, CountingSemiring(1));
        wfa.add_transition(0, 0, 2, CountingSemiring(1));

        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let matrices = compiler.embed_transition_matrices(&wfa);
        assert_eq!(matrices.len(), 3);
        assert_eq!(matrices[0][1][0].to_canonical(), 1);
        assert_eq!(matrices[1][0][1].to_canonical(), 1);
        assert_eq!(matrices[2][0][0].to_canonical(), 1);
    }

    // ─── Algebraic Compilation Tests ────────────────────────────────────

    #[test]
    fn test_compile_simple_counting_wfa() {
        let mut wfa = WFA::<CountingSemiring>::new(1, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(0, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            target: CompilationTarget::AlgebraicDirect,
            optimization_level: 0,
            ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 4);

        assert!(result.air_program.constraints.len() > 0);
        assert!(result.trace_layout.total_columns() > 0);
        assert_eq!(result.compilation_stats.num_wfa_states, 1);
        assert_eq!(result.compilation_stats.alphabet_size, 2);
        assert!(result.compilation_stats.trace_length >= 5);
        assert!(result.compilation_stats.num_boundary > 0);
        assert!(result.compilation_stats.num_transition > 0);
    }

    #[test]
    fn test_compile_two_state_counting_wfa() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_initial(1, CountingSemiring(0));
        wfa.set_final(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 1, 1, CountingSemiring(1));
        wfa.add_transition(1, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 1, 1, CountingSemiring(1));

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            target: CompilationTarget::AlgebraicDirect,
            optimization_level: 0,
            ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 3);
        assert_eq!(result.compilation_stats.num_wfa_states, 2);
        assert!(result.compilation_stats.num_constraints > 0);
    }

    #[test]
    fn test_compilation_stats_display() {
        let mut stats = CompilationStats::new();
        stats.trace_width = 10;
        stats.trace_length = 256;
        stats.num_constraints = 50;
        stats.num_boundary = 5;
        stats.num_transition = 40;
        stats.num_periodic = 2;
        stats.num_global = 3;
        stats.max_degree = 3;
        stats.num_wfa_states = 4;
        stats.num_wfa_transitions = 12;
        stats.alphabet_size = 3;
        stats.estimate_proof_size();

        let display = format!("{}", stats);
        assert!(display.contains("Compilation Statistics"));
        assert!(display.contains("50 total"));
    }

    // ─── Trace Generation Tests ─────────────────────────────────────────

    #[test]
    fn test_trace_generator_simple_counting() {
        let mut wfa = WFA::<CountingSemiring>::new(1, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(0, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            target: CompilationTarget::AlgebraicDirect, optimization_level: 0,
            ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 4);

        let gen = TraceGenerator::<CountingSemiring>::new();
        let trace = gen.generate_trace(&wfa, &[0, 1, 0, 1], &result);

        assert!(trace.num_rows >= 5);
        if let Some(&state_col) = result.column_map.get("state_0") {
            assert_eq!(trace.get(0, state_col).to_canonical(), 1);
        }
        if let Some(&step_col) = result.column_map.get("step_counter") {
            for t in 0..5.min(trace.num_rows) {
                assert_eq!(trace.get(t, step_col).to_canonical(), t as u64);
            }
        }
    }

    #[test]
    fn test_simulate_wfa_counting() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_initial(1, CountingSemiring(0));
        wfa.set_final(1, CountingSemiring(1));
        wfa.set_final(0, CountingSemiring(0));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));
        wfa.add_transition(0, 1, 1, CountingSemiring(1));
        wfa.add_transition(1, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 1, 1, CountingSemiring(1));

        let gen = TraceGenerator::<CountingSemiring>::new();
        let state_trace = gen.generate_state_columns(&wfa, &[1, 0, 1]);

        assert_eq!(state_trace[0][0].to_canonical(), 1);
        assert_eq!(state_trace[1][0].to_canonical(), 0);
        assert_eq!(state_trace[1][1].to_canonical(), 1);
    }

    #[test]
    fn test_generate_state_columns_dimensions() {
        let mut wfa = WFA::<CountingSemiring>::new(3, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_initial(1, CountingSemiring(0));
        wfa.set_initial(2, CountingSemiring(0));
        for s in 0..3 { for a in 0..2 { wfa.add_transition(s, s, a as u8, CountingSemiring(1)); } }
        wfa.set_final(2, CountingSemiring(1));

        let gen = TraceGenerator::<CountingSemiring>::new();
        let cols = gen.generate_state_columns(&wfa, &[0, 1, 0]);
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].len(), 4);
    }

    // ─── Optimization Tests ─────────────────────────────────────────────

    #[test]
    fn test_trivial_constraint_elimination() {
        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            optimization_level: 1, enable_cse: false, enable_dead_elimination: false, enable_batching: false,
            ..CompilerConfig::default()
        });
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::constant(0), "trivial"),
            AIRConstraint::transition(SymbolicExpression::col(0).sub(SymbolicExpression::col(0)), "nontrivial"),
            AIRConstraint::transition(SymbolicExpression::col(0).sub(SymbolicExpression::constant(1)), "real"),
        ];
        let optimized = compiler.optimize_constraints(constraints);
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn test_cse_elimination() {
        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            optimization_level: 1, enable_cse: true, enable_dead_elimination: false, enable_batching: false,
            ..CompilerConfig::default()
        });
        let expr = SymbolicExpression::col(0).mul(SymbolicExpression::col(1));
        let constraints = vec![
            AIRConstraint::transition(expr.clone(), "first"),
            AIRConstraint::transition(expr.clone(), "duplicate"),
            AIRConstraint::transition(SymbolicExpression::col(2).add(SymbolicExpression::col(3)), "different"),
        ];
        let optimized = compiler.optimize_constraints(constraints);
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn test_degree_reduction() {
        let compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            max_constraint_degree: 2, ..CompilerConfig::default()
        });
        let expr = SymbolicExpression::col(0).mul(SymbolicExpression::col(1)).mul(SymbolicExpression::col(2));
        let constraint = AIRConstraint::transition(expr, "degree3");
        assert_eq!(constraint.degree, 3);
        let reduced = compiler.reduce_degree(&constraint, 2);
        assert!(reduced.len() >= 2);
    }

    #[test]
    fn test_constraint_batching() {
        let compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            enable_batching: true, ..CompilerConfig::default()
        });
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::col(0), "t1"),
            AIRConstraint::transition(SymbolicExpression::col(1), "t2"),
            AIRConstraint::transition(SymbolicExpression::col(2), "t3"),
            AIRConstraint::boundary(SymbolicExpression::col(0), 0, "b1"),
        ];
        let batched = compiler.batch_constraints(constraints);
        let transitions: Vec<_> = batched.iter().filter(|c| c.constraint_type == ConstraintType::Transition).collect();
        assert_eq!(transitions.len(), 1);
        assert!(transitions[0].label.contains("batched"));
    }

    // ─── Constraint Graph Tests ─────────────────────────────────────────

    #[test]
    fn test_constraint_graph_connected() {
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::col(0).add(SymbolicExpression::col(1)), "c1"),
            AIRConstraint::transition(SymbolicExpression::col(1).add(SymbolicExpression::col(2)), "c2"),
            AIRConstraint::transition(SymbolicExpression::col(3), "c3"),
        ];
        let graph = ConstraintGraph::build(&constraints);
        assert_eq!(graph.connected_components().len(), 2);
    }

    #[test]
    fn test_constraint_graph_degrees() {
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::col(0).add(SymbolicExpression::col(1)), "c1"),
            AIRConstraint::transition(SymbolicExpression::col(1).mul(SymbolicExpression::col(2)), "c2"),
            AIRConstraint::transition(SymbolicExpression::col(2).sub(SymbolicExpression::col(0)), "c3"),
        ];
        let graph = ConstraintGraph::build(&constraints);
        assert_eq!(graph.max_degree(), 2);
        assert_eq!(graph.num_edges(), 3);
    }

    // ─── Expression Builder Tests ───────────────────────────────────────

    #[test]
    fn test_expression_builder_dot_product() {
        let builder = ExpressionBuilder::new(10);
        let coeffs = vec![GoldilocksField::new(2), GoldilocksField::new(3), GoldilocksField::new(0), GoldilocksField::new(5)];
        let cols = vec![0, 1, 2, 3];
        let expr = builder.dot_product_expr(&coeffs, &cols);
        let row = vec![GoldilocksField::new(10), GoldilocksField::new(20), GoldilocksField::new(30), GoldilocksField::new(40)];
        assert_eq!(expr.evaluate(&row, None, None, 0).to_canonical(), 280);
    }

    #[test]
    fn test_expression_builder_selected_sum() {
        let builder = ExpressionBuilder::new(10);
        let sel_cols = vec![0, 1, 2];
        let values = vec![SymbolicExpression::constant(100), SymbolicExpression::constant(200), SymbolicExpression::constant(300)];
        let expr = builder.selected_sum(&sel_cols, &values);
        let row = vec![GoldilocksField::new(0), GoldilocksField::new(1), GoldilocksField::new(0)];
        assert_eq!(expr.evaluate(&row, None, None, 0).to_canonical(), 200);
    }

    // ─── Multi-WFA Compiler Tests ───────────────────────────────────────

    #[test]
    fn test_product_wfa_construction() {
        let mut wfa1 = WFA::<CountingSemiring>::new(2, 2);
        wfa1.set_initial(0, CountingSemiring(1));
        wfa1.set_final(1, CountingSemiring(1));
        wfa1.add_transition(0, 1, 0, CountingSemiring(1));
        wfa1.add_transition(1, 1, 0, CountingSemiring(1));
        wfa1.add_transition(0, 0, 1, CountingSemiring(1));
        wfa1.add_transition(1, 0, 1, CountingSemiring(1));

        let mut wfa2 = WFA::<CountingSemiring>::new(2, 2);
        wfa2.set_initial(0, CountingSemiring(1));
        wfa2.set_final(0, CountingSemiring(1));
        wfa2.add_transition(0, 0, 0, CountingSemiring(1));
        wfa2.add_transition(0, 1, 1, CountingSemiring(1));
        wfa2.add_transition(1, 1, 0, CountingSemiring(1));
        wfa2.add_transition(1, 0, 1, CountingSemiring(1));

        let mut multi = MultiWFACompiler::<CountingSemiring>::new(CompilerConfig::default());
        let result = multi.compile_product(&[wfa1, wfa2], 3);
        assert_eq!(result.compilation_stats.num_wfa_states, 4);
    }

    #[test]
    fn test_union_wfa_construction() {
        let mut wfa1 = WFA::<CountingSemiring>::new(2, 2);
        wfa1.set_initial(0, CountingSemiring(1));
        wfa1.set_final(1, CountingSemiring(1));
        wfa1.add_transition(0, 1, 0, CountingSemiring(1));

        let mut wfa2 = WFA::<CountingSemiring>::new(3, 2);
        wfa2.set_initial(0, CountingSemiring(1));
        wfa2.set_final(2, CountingSemiring(1));
        wfa2.add_transition(0, 1, 1, CountingSemiring(1));
        wfa2.add_transition(1, 2, 0, CountingSemiring(1));

        let mut multi = MultiWFACompiler::<CountingSemiring>::new(CompilerConfig::default());
        let result = multi.compile_union(&[wfa1, wfa2], 3);
        assert_eq!(result.compilation_stats.num_wfa_states, 5);
    }

    // ─── Pipeline Tests ─────────────────────────────────────────────────

    #[test]
    fn test_compilation_pipeline_counting() {
        let mut wfa = WFA::<CountingSemiring>::new(1, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(0, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));

        let mut pipeline = CompilationPipeline::<CountingSemiring>::with_default_config();
        let (result, trace) = pipeline.compile_and_trace(&wfa, &[0, 1, 0]);
        assert!(trace.num_rows > 0);
        assert!(result.air_program.constraints.len() > 0);
    }

    // ─── Gadget Tests ───────────────────────────────────────────────────

    #[test]
    fn test_boolean_gadget() {
        let gadget = BooleanGadget::new(0);
        let constraints = gadget.constraints();
        assert_eq!(constraints.len(), 1);
        assert!(constraints[0].check_at_row(&[GoldilocksField::ZERO], Some(&[GoldilocksField::ZERO]), None, 0));
        assert!(constraints[0].check_at_row(&[GoldilocksField::ONE], Some(&[GoldilocksField::ONE]), None, 0));
        assert!(!constraints[0].check_at_row(&[GoldilocksField::new(2)], Some(&[GoldilocksField::new(2)]), None, 0));
    }

    #[test]
    fn test_range_check_gadget() {
        let gadget = RangeCheckGadget::new(3, 0, vec![1, 2, 3]);
        assert_eq!(gadget.constraints().len(), 4);
    }

    #[test]
    fn test_comparison_gadget() {
        let gadget = ComparisonGadget::new(8, 0, 1, vec![2, 3, 4, 5, 6, 7, 8, 9], 10);
        assert_eq!(gadget.constraints().len(), 10);
    }

    #[test]
    fn test_gadget_constraints_collection() {
        let mut gc = GadgetConstraints::new();
        gc.boolean_gadgets.push(BooleanGadget::new(0));
        gc.boolean_gadgets.push(BooleanGadget::new(1));
        gc.range_check_gadgets.push(RangeCheckGadget::new(2, 2, vec![3, 4]));
        assert_eq!(gc.all_constraints().len(), 5);
    }

    // ─── Verification Tests ─────────────────────────────────────────────

    #[test]
    fn test_detailed_verification() {
        let mut trace = AIRTrace::new(4, 2);
        for i in 0..4 {
            trace.set(i, 0, GoldilocksField::new(i as u64));
            trace.set(i, 1, GoldilocksField::new(i as u64 * 2));
        }

        let mut program = AIRProgram::new(TraceLayout::new(), 4);
        program.add_constraint(AIRConstraint::boundary(SymbolicExpression::col(0), 0, "init_0"));
        program.add_constraint(AIRConstraint::transition(
            SymbolicExpression::next_col(0).sub(SymbolicExpression::col(0)).sub(SymbolicExpression::constant(1)), "inc"));

        let result = CompilationResult {
            air_program: program, trace_layout: TraceLayout::new(),
            compilation_stats: CompilationStats::new(), constraint_mapping: ConstraintMapping::new(),
            ir: IntermediateRepresentation::new(), gadget_constraints: GadgetConstraints::new(),
            column_map: HashMap::new(), transition_matrices: Vec::new(),
        };

        let details = verify_constraints_detailed(&result, &trace);
        assert_eq!(details.len(), 2);
        assert!(details[0].satisfied);
        assert!(details[1].satisfied);
    }

    // ─── Helper Function Tests ──────────────────────────────────────────

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
        assert_eq!(next_power_of_two(100), 128);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
    }

    // ─── Deep Simplify Tests ────────────────────────────────────────────

    #[test]
    fn test_deep_simplify_nested() {
        let expr = SymbolicExpression::constant(0).add(SymbolicExpression::constant(1).mul(SymbolicExpression::col(3)));
        match deep_simplify(&expr) {
            SymbolicExpression::Column(3) => {}
            other => panic!("Expected Column(3), got {:?}", other),
        }
    }

    #[test]
    fn test_deep_simplify_constant_chain() {
        let expr = SymbolicExpression::constant(2).add(SymbolicExpression::constant(3))
            .mul(SymbolicExpression::constant(4).add(SymbolicExpression::constant(1)));
        match deep_simplify(&expr) {
            SymbolicExpression::Constant(v) => assert_eq!(v.to_canonical(), 25),
            other => panic!("Expected constant 25, got {:?}", other),
        }
    }

    // ─── Tropical Compilation Tests ─────────────────────────────────────

    #[test]
    fn test_compile_tropical_wfa() {
        let wfa = build_tropical_shortest_path_wfa(3, &[(0, 1, 0, 1.0), (1, 2, 0, 2.0), (0, 2, 1, 5.0)], 2);
        let mut compiler = WFACircuitCompiler::<TropicalSemiring>::new(CompilerConfig {
            target: CompilationTarget::GadgetAssisted, optimization_level: 0,
            ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 2);
        assert_eq!(result.compilation_stats.num_wfa_states, 3);
        assert!(result.air_program.constraints.len() > 0);
    }

    #[test]
    fn test_tropical_field_encoding() {
        let val = 3.5;
        let field = tropical_to_field(val);
        let decoded = field_from_tropical(field);
        assert!((decoded - val).abs() < 1e-6);
    }

    #[test]
    fn test_tropical_infinity_encoding() {
        let decoded = field_from_tropical(tropical_to_field(f64::INFINITY));
        assert!(decoded.is_infinite() && decoded > 0.0);
    }

    #[test]
    fn test_tropical_negative_encoding() {
        let val = -2.75;
        let decoded = field_from_tropical(tropical_to_field(val));
        assert!((decoded - val).abs() < 1e-6, "Expected {}, got {}", val, decoded);
    }

    // ─── Emit Constraint Tests ──────────────────────────────────────────

    #[test]
    fn test_emit_tropical_mul_constraints() {
        let compiler = WFACircuitCompiler::<TropicalSemiring>::default_compiler();
        let constraints = compiler.emit_tropical_mul_constraints(0, 1, 2);
        assert_eq!(constraints.len(), 1);
        let row = vec![GoldilocksField::new(10), GoldilocksField::new(20), GoldilocksField::new(30)];
        assert!(constraints[0].check_at_row(&row, Some(&row), None, 0));
    }

    #[test]
    fn test_emit_bit_decomposition() {
        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let constraints = compiler.emit_bit_decomposition_constraints(0, &[1, 2, 3]);
        assert_eq!(constraints.len(), 4);
        // value = 5 = 1*1 + 0*2 + 1*4
        let row = vec![GoldilocksField::new(5), GoldilocksField::new(1), GoldilocksField::new(0), GoldilocksField::new(1)];
        for (i, c) in constraints.iter().enumerate() {
            assert!(c.check_at_row(&row, Some(&row), None, 0), "Constraint {} failed: {}", i, c.label);
        }
    }

    #[test]
    fn test_emit_initial_state_constraints() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_initial(1, CountingSemiring(0));
        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let constraints = compiler.emit_initial_state_constraints(&wfa, &[0, 1]);
        assert_eq!(constraints.len(), 2);
        let row = vec![GoldilocksField::new(1), GoldilocksField::new(0)];
        assert!(constraints[0].check_at_row(&row, None, None, 0));
        assert!(constraints[1].check_at_row(&row, None, None, 0));
    }

    #[test]
    fn test_emit_final_state_constraints() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_final(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(2));
        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let constraints = compiler.emit_final_state_constraints(&wfa, &[0, 1], 2, 3);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].boundary_row, Some(3));
        // output = 1*3 + 2*4 = 11
        let row = vec![GoldilocksField::new(3), GoldilocksField::new(4), GoldilocksField::new(11)];
        assert!(constraints[0].check_at_row(&row, None, None, 3));
    }

    // ─── Selector Constraint Tests ──────────────────────────────────────

    #[test]
    fn test_selector_constraints_satisfaction() {
        let wfa = WFA::<CountingSemiring>::new(1, 3);
        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let constraints = compiler.emit_selector_constraints(&wfa, 0, &[1, 2, 3]);
        // Input = 1, so selector_1 = 1, others = 0
        let row = vec![GoldilocksField::new(1), GoldilocksField::new(0), GoldilocksField::new(1), GoldilocksField::new(0)];
        for c in &constraints {
            if c.label.contains("boolean") || c.label.contains("sum") || c.label.contains("consistency") {
                assert!(c.check_at_row(&row, Some(&row), None, 0), "Constraint '{}' not satisfied", c.label);
            }
        }
    }

    // ─── Column Map Tests ───────────────────────────────────────────────

    #[test]
    fn test_column_map_populated() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_initial(1, CountingSemiring(0));
        wfa.set_final(0, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));
        wfa.add_transition(1, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 1, 1, CountingSemiring(1));

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            target: CompilationTarget::AlgebraicDirect, optimization_level: 0, ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 2);
        assert!(result.column_map.contains_key("state_0"));
        assert!(result.column_map.contains_key("state_1"));
        assert!(result.column_map.contains_key("input_symbol"));
        assert!(result.column_map.contains_key("selector_0"));
        assert!(result.column_map.contains_key("output"));
        assert!(result.column_map.contains_key("step_counter"));
    }

    // ─── Real / Boolean Compilation Tests ───────────────────────────────

    #[test]
    fn test_compile_real_wfa() {
        let wfa = build_real_wfa(2, 2, &[(0,0,0,0.5),(0,1,1,0.5),(1,0,0,0.3),(1,1,1,0.7)], &[1.0, 0.0], &[1.0, 1.0]);
        let mut compiler = WFACircuitCompiler::<RealSemiring>::new(CompilerConfig {
            target: CompilationTarget::Hybrid, optimization_level: 0, ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 3);
        assert_eq!(result.compilation_stats.num_wfa_states, 2);
        assert!(result.air_program.constraints.len() > 0);
    }

    #[test]
    fn test_compile_boolean_wfa() {
        let wfa = build_boolean_wfa(1, 2);
        let mut compiler = WFACircuitCompiler::<BooleanSemiring>::new(CompilerConfig {
            target: CompilationTarget::Hybrid, optimization_level: 0, ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 3);
        assert_eq!(result.compilation_stats.num_wfa_states, 2);
        assert!(result.compilation_stats.num_constraints > 0);
    }

    // ─── End-to-End Integration Test ────────────────────────────────────

    #[test]
    fn test_end_to_end_counting_wfa_with_verification() {
        let mut wfa = WFA::<CountingSemiring>::new(1, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(0, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));

        let config = CompilerConfig {
            target: CompilationTarget::AlgebraicDirect, optimization_level: 0,
            enable_cse: false, enable_dead_elimination: false, enable_batching: false,
            ..CompilerConfig::default()
        };

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(config);
        let result = compiler.compile(&wfa, 4);
        let gen = TraceGenerator::<CountingSemiring>::new();
        let input = vec![0u8, 1, 0, 1];
        let trace = gen.generate_trace(&wfa, &input, &result);

        if let Some(&state_col) = result.column_map.get("state_0") {
            assert_eq!(trace.get(0, state_col).to_canonical(), 1);
            for t in 0..=4 {
                if t < trace.num_rows {
                    assert_eq!(trace.get(t, state_col).to_canonical(), 1, "State should be 1 at step {}", t);
                }
            }
        }

        for t in 0..input.len().min(trace.num_rows) {
            let sym = input[t];
            for a in 0..2 {
                let name = format!("selector_{}", a);
                if let Some(&sel_col) = result.column_map.get(&name) {
                    let expected = if a == sym as usize { 1 } else { 0 };
                    assert_eq!(trace.get(t, sel_col).to_canonical(), expected as u64,
                        "Selector {} at step {} should be {}", a, t, expected);
                }
            }
        }
    }

    // ─── Scale Tests ────────────────────────────────────────────────────

    #[test]
    fn test_compile_larger_wfa() {
        let mut wfa = WFA::<CountingSemiring>::new(5, 4);
        wfa.set_initial(0, CountingSemiring(1));
        for s in 1..5 { wfa.set_initial(s, CountingSemiring(0)); }
        wfa.set_final(4, CountingSemiring(1));

        for s in 0..4 {
            wfa.add_transition(s, s + 1, s as u8, CountingSemiring(1));
            for a in 0..4 { if a as u8 != s as u8 { wfa.add_transition(s, s, a as u8, CountingSemiring(1)); } }
        }
        for a in 0..4 { wfa.add_transition(4, 4, a as u8, CountingSemiring(1)); }

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            target: CompilationTarget::AlgebraicDirect, optimization_level: 1, ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 10);
        assert_eq!(result.compilation_stats.num_wfa_states, 5);
        assert_eq!(result.compilation_stats.alphabet_size, 4);
        assert!(result.compilation_stats.num_constraints > 0);
        assert!(result.compilation_stats.trace_length >= 11);
        assert!(compiler.compilation_statistics().estimated_proof_size_bytes > 0);
    }

    #[test]
    fn test_transition_matrices_consistency() {
        let mut wfa = WFA::<CountingSemiring>::new(3, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(2));
        wfa.add_transition(1, 2, 0, CountingSemiring(3));
        wfa.add_transition(2, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 0, 1, CountingSemiring(1));
        wfa.add_transition(1, 1, 1, CountingSemiring(1));
        wfa.add_transition(2, 2, 1, CountingSemiring(1));

        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let matrices = compiler.embed_transition_matrices(&wfa);
        assert_eq!(matrices[0][1][0].to_canonical(), 2);
        assert_eq!(matrices[0][2][1].to_canonical(), 3);
        assert_eq!(matrices[0][0][2].to_canonical(), 1);

        let state = vec![GoldilocksField::new(1), GoldilocksField::new(0), GoldilocksField::new(0)];
        let next_state = matrix_vector_multiply(&matrices[0], &state);
        assert_eq!(next_state[0].to_canonical(), 0);
        assert_eq!(next_state[1].to_canonical(), 2);
        assert_eq!(next_state[2].to_canonical(), 0);
    }

    #[test]
    fn test_matrix_chain_computation() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 1, 0, CountingSemiring(1));

        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let m0 = compiler.compute_transition_matrix(&wfa, 0);
        assert_eq!(m0[0][0].to_canonical(), 1);
        assert_eq!(m0[1][0].to_canonical(), 1);
        assert_eq!(m0[1][1].to_canonical(), 1);

        let m03 = matrix_power(&m0, 3);
        assert_eq!(m03[0][0].to_canonical(), 1);
        assert_eq!(m03[1][0].to_canonical(), 3);
    }

    // ─── Optimization with Full Passes ──────────────────────────────────

    #[test]
    fn test_compile_with_full_optimization() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_initial(1, CountingSemiring(0));
        wfa.set_final(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(1));
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 1, 1, CountingSemiring(1));
        wfa.add_transition(1, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 0, 1, CountingSemiring(1));

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig {
            target: CompilationTarget::AlgebraicDirect, optimization_level: 2,
            enable_cse: true, enable_dead_elimination: true, enable_batching: true,
            max_constraint_degree: 4, ..CompilerConfig::default()
        });
        let result = compiler.compile(&wfa, 4);
        assert!(result.compilation_stats.num_constraints > 0);
        assert!(result.compilation_stats.optimization_passes_applied.len() > 0);
    }

    // ─── Trace Layout from WFA ──────────────────────────────────────────

    #[test]
    fn test_trace_layout_from_wfa() {
        let wfa = WFA::<CountingSemiring>::new(3, 4);
        let compiler = WFACircuitCompiler::<CountingSemiring>::default_compiler();
        let layout = compiler.compute_trace_layout(&wfa, 16);
        // 3 state + 1 input + 4 selectors + 3 aux + 1 output + 1 step = 13
        assert_eq!(layout.total_columns(), 13);
        assert_eq!(layout.num_rows, 16);
        assert_eq!(layout.num_witness_columns, 3);
        assert!(layout.find_column("state_0").is_some());
        assert!(layout.find_column("output").is_some());
    }

    // ─── Semiring property tests ────────────────────────────────────────

    #[test]
    fn test_counting_is_algebraic() {
        assert!(CountingSemiring::is_algebraically_embeddable());
        assert!(CountingSemiring::addition_is_field_addition());
        assert!(CountingSemiring::multiplication_is_field_multiplication());
    }

    #[test]
    fn test_tropical_is_not_algebraic() {
        assert!(!TropicalSemiring::is_algebraically_embeddable());
        assert!(!TropicalSemiring::addition_is_field_addition());
    }

    #[test]
    fn test_real_is_partially_algebraic() {
        assert!(!RealSemiring::is_algebraically_embeddable());
        assert!(RealSemiring::addition_is_field_addition());
        assert!(!RealSemiring::multiplication_is_field_multiplication());
    }

    // ─── Matrix utility edge cases ──────────────────────────────────────

    #[test]
    fn test_matrix_multiply_1x1() {
        let c = matrix_multiply(&[vec![GoldilocksField::new(3)]], &[vec![GoldilocksField::new(7)]]);
        assert_eq!(c[0][0].to_canonical(), 21);
    }

    #[test]
    fn test_matrix_add() {
        let a = vec![vec![GoldilocksField::new(1), GoldilocksField::new(2)], vec![GoldilocksField::new(3), GoldilocksField::new(4)]];
        let b = vec![vec![GoldilocksField::new(10), GoldilocksField::new(20)], vec![GoldilocksField::new(30), GoldilocksField::new(40)]];
        let c = matrix_add(&a, &b);
        assert_eq!(c[0][0].to_canonical(), 11);
        assert_eq!(c[1][1].to_canonical(), 44);
    }

    #[test]
    fn test_matrix_scale() {
        let m = vec![vec![GoldilocksField::new(2), GoldilocksField::new(3)], vec![GoldilocksField::new(4), GoldilocksField::new(5)]];
        let scaled = matrix_scale(&m, GoldilocksField::new(10));
        assert_eq!(scaled[0][0].to_canonical(), 20);
        assert_eq!(scaled[1][1].to_canonical(), 50);
    }

    #[test]
    fn test_identity_matrix_properties() {
        let id = identity_matrix(4);
        for i in 0..4 { for j in 0..4 {
            assert_eq!(id[i][j].to_canonical(), if i == j { 1 } else { 0 });
        }}
    }

    #[test]
    fn test_dot_product_empty() { assert_eq!(dot_product(&[], &[]).to_canonical(), 0); }

    #[test]
    fn test_dot_product_single() {
        assert_eq!(dot_product(&[GoldilocksField::new(7)], &[GoldilocksField::new(11)]).to_canonical(), 77);
    }

    // ─── Target selection tests ─────────────────────────────────────────

    #[test]
    fn test_hybrid_selects_algebraic_for_counting() {
        let compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig { target: CompilationTarget::Hybrid, ..CompilerConfig::default() });
        assert_eq!(compiler.select_compilation_target(), CompilationTarget::AlgebraicDirect);
    }

    #[test]
    fn test_hybrid_selects_gadget_for_tropical() {
        let compiler = WFACircuitCompiler::<TropicalSemiring>::new(CompilerConfig { target: CompilationTarget::Hybrid, ..CompilerConfig::default() });
        assert_eq!(compiler.select_compilation_target(), CompilationTarget::GadgetAssisted);
    }

    #[test]
    fn test_explicit_target_overrides_hybrid() {
        let compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig { target: CompilationTarget::GadgetAssisted, ..CompilerConfig::default() });
        assert_eq!(compiler.select_compilation_target(), CompilationTarget::GadgetAssisted);
    }

    // ─── Proof size estimation ──────────────────────────────────────────

    #[test]
    fn test_proof_size_estimation() {
        let mut stats = CompilationStats::new();
        stats.trace_width = 20;
        stats.trace_length = 1024;
        stats.estimate_proof_size();
        assert!(stats.estimated_proof_size_bytes > 0);
        assert!(stats.estimated_proof_size_bytes < 10_000_000);
    }

    #[test]
    fn test_proof_size_scales_with_width() {
        let mut s1 = CompilationStats::new();
        s1.trace_width = 10; s1.trace_length = 256; s1.estimate_proof_size();
        let mut s2 = CompilationStats::new();
        s2.trace_width = 40; s2.trace_length = 256; s2.estimate_proof_size();
        assert!(s2.estimated_proof_size_bytes > s1.estimated_proof_size_bytes);
    }

    // ─── CompilerStage tests ────────────────────────────────────────────

    #[test]
    fn test_compiler_stage_name() {
        assert_eq!(CompilerStage::Parse.name(), "Parse");
        assert_eq!(CompilerStage::TypeCheck.name(), "TypeCheck");
        assert_eq!(CompilerStage::LowerToIR.name(), "LowerToIR");
        assert_eq!(CompilerStage::OptimizeIR.name(), "OptimizeIR");
        assert_eq!(CompilerStage::EmitConstraints.name(), "EmitConstraints");
        assert_eq!(CompilerStage::OptimizeConstraints.name(), "OptimizeConstraints");
        assert_eq!(CompilerStage::LayoutTrace.name(), "LayoutTrace");
        assert_eq!(CompilerStage::Finalize.name(), "Finalize");
    }

    #[test]
    fn test_compiler_stage_display() {
        assert_eq!(format!("{}", CompilerStage::Parse), "Parse");
        assert_eq!(format!("{}", CompilerStage::Finalize), "Finalize");
    }

    #[test]
    fn test_compiler_stage_eq() {
        assert_eq!(CompilerStage::Parse, CompilerStage::Parse);
        assert_ne!(CompilerStage::Parse, CompilerStage::Finalize);
    }

    // ─── CompilerPipeline tests ─────────────────────────────────────────

    #[test]
    fn test_default_pipeline_stages() {
        let pipeline = CompilerPipeline::default_pipeline();
        let names = pipeline.stage_names();
        assert_eq!(names.len(), 8);
        assert_eq!(names[0], "Parse");
        assert_eq!(names[7], "Finalize");
    }

    #[test]
    fn test_with_stage_single() {
        let pipeline = CompilerPipeline::with_stage(CompilerStage::EmitConstraints);
        let names = pipeline.stage_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "EmitConstraints");
    }

    #[test]
    fn test_pipeline_run_counting() {
        let pipeline = CompilerPipeline::default_pipeline();
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        let result = pipeline.run(&wfa, 4);
        assert!(result.compilation_stats.num_constraints > 0);
    }

    #[test]
    fn test_pipeline_stage_names_match_stages() {
        let pipeline = CompilerPipeline::default_pipeline();
        let names = pipeline.stage_names();
        assert!(names.contains(&"Parse".to_string()));
        assert!(names.contains(&"Finalize".to_string()));
    }

    // ─── CompilerDiagnostics tests ──────────────────────────────────────

    #[test]
    fn test_diagnostics_new() {
        let diag = CompilerDiagnostics::new();
        assert_eq!(diag.total_time_us(), 0);
    }

    #[test]
    fn test_diagnostics_default() {
        let diag = CompilerDiagnostics::default();
        assert_eq!(diag.total_time_us(), 0);
    }

    #[test]
    fn test_diagnostics_record_stage() {
        let mut diag = CompilerDiagnostics::new();
        diag.record_stage("Parse", 100, 5);
        diag.record_stage("TypeCheck", 200, 10);
        assert_eq!(diag.total_time_us(), 300);
    }

    #[test]
    fn test_diagnostics_record_optimization() {
        let mut diag = CompilerDiagnostics::new();
        diag.record_optimization("CSE", 100, 80);
        diag.record_optimization("DeadElim", 80, 60);
        let best = diag.best_optimization();
        assert!(best.is_some());
        let (name, pct) = best.unwrap();
        assert_eq!(name, "DeadElim");
        assert!((pct - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_diagnostics_best_optimization_empty() {
        let diag = CompilerDiagnostics::new();
        assert!(diag.best_optimization().is_none());
    }

    #[test]
    fn test_diagnostics_summary() {
        let mut diag = CompilerDiagnostics::new();
        diag.record_stage("Parse", 42, 3);
        diag.record_constraint_count("after_parse", 10);
        diag.record_optimization("CSE", 10, 8);
        let s = diag.summary();
        assert!(s.contains("Parse"));
        assert!(s.contains("42"));
        assert!(s.contains("CSE"));
    }

    #[test]
    fn test_diagnostics_to_json() {
        let mut diag = CompilerDiagnostics::new();
        diag.record_stage("Emit", 500, 20);
        let json = diag.to_json();
        assert!(json.contains("\"name\": \"Emit\""));
        assert!(json.contains("\"duration_us\": 500"));
        assert!(json.contains("\"total_time_us\": 500"));
    }

    #[test]
    fn test_diagnostics_constraint_count() {
        let mut diag = CompilerDiagnostics::new();
        diag.record_constraint_count("before", 50);
        diag.record_constraint_count("after", 30);
        let s = diag.summary();
        assert!(s.contains("50"));
        assert!(s.contains("30"));
    }

    // ─── ConstraintAnalyzer tests ───────────────────────────────────────

    fn make_test_constraints() -> Vec<AIRConstraint> {
        vec![
            AIRConstraint::boundary(SymbolicExpression::col(0), 0, "boundary_init"),
            AIRConstraint::transition(
                SymbolicExpression::col(0).sub(SymbolicExpression::next_col(0)),
                "state_transition",
            ),
            AIRConstraint::transition(
                SymbolicExpression::col(1).mul(SymbolicExpression::col(2)),
                "weight_mult",
            ),
            AIRConstraint::global(
                SymbolicExpression::col(0).add(SymbolicExpression::col(1)),
                "sum_check",
            ),
        ]
    }

    #[test]
    fn test_analyzer_analyze() {
        let constraints = make_test_constraints();
        let analysis = ConstraintAnalyzer::analyze(&constraints);
        assert_eq!(analysis.total, 4);
        assert!(analysis.by_type.contains_key("Boundary"));
        assert!(analysis.by_type.contains_key("Transition"));
        assert!(analysis.by_type.contains_key("Global"));
    }

    #[test]
    fn test_analyzer_degree_histogram() {
        let constraints = make_test_constraints();
        let hist = ConstraintAnalyzer::degree_histogram(&constraints);
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_analyzer_column_usage() {
        let constraints = make_test_constraints();
        let usage = ConstraintAnalyzer::column_usage(&constraints);
        assert!(usage.contains_key(&0));
    }

    #[test]
    fn test_analyzer_constraint_density() {
        let constraints = make_test_constraints();
        let density = ConstraintAnalyzer::constraint_density(&constraints, 5);
        assert!(density > 0.0);
        assert!(density <= 1.0);
    }

    #[test]
    fn test_analyzer_constraint_density_zero_cols() {
        let constraints = make_test_constraints();
        assert_eq!(ConstraintAnalyzer::constraint_density(&constraints, 0), 0.0);
    }

    #[test]
    fn test_analyzer_find_redundant_pairs_none() {
        let constraints = make_test_constraints();
        let pairs = ConstraintAnalyzer::find_redundant_pairs(&constraints);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_analyzer_find_redundant_pairs_some() {
        let constraints = vec![
            AIRConstraint::boundary(SymbolicExpression::col(0), 0, "dup"),
            AIRConstraint::boundary(SymbolicExpression::col(1), 0, "dup"),
        ];
        let pairs = ConstraintAnalyzer::find_redundant_pairs(&constraints);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1));
    }

    #[test]
    fn test_analyzer_verification_cost() {
        let constraints = make_test_constraints();
        let cost = ConstraintAnalyzer::estimate_verification_cost(&constraints);
        assert!(cost >= constraints.len());
    }

    #[test]
    fn test_analyzer_empty() {
        let analysis = ConstraintAnalyzer::analyze(&[]);
        assert_eq!(analysis.total, 0);
        assert_eq!(analysis.max_degree, 0);
        assert_eq!(analysis.avg_degree, 0.0);
    }

    // ─── WFAAnalyzer tests ──────────────────────────────────────────────

    #[test]
    fn test_wfa_analyze_counting() {
        let mut wfa = WFA::<CountingSemiring>::new(3, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(2, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 2, 1, CountingSemiring(2));

        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        assert_eq!(analysis.num_states, 3);
        assert_eq!(analysis.alphabet_size, 2);
        assert_eq!(analysis.num_transitions, 2);
        assert!(analysis.is_deterministic);
        assert!(!analysis.has_epsilon);
    }

    #[test]
    fn test_wfa_analyze_tropical() {
        let mut wfa = WFA::<TropicalSemiring>::new(2, 2);
        wfa.set_initial(0, TropicalSemiring(0.0));
        wfa.set_final(1, TropicalSemiring(0.0));
        wfa.add_transition(0, 1, 0, TropicalSemiring(1.5));
        wfa.add_transition(0, 1, 1, TropicalSemiring(2.5));

        let analysis = WFAAnalyzer::analyze_tropical(&wfa);
        assert_eq!(analysis.num_states, 2);
        assert_eq!(analysis.num_transitions, 2);
        assert!((analysis.max_weight - 2.5).abs() < 1e-9);
        assert!((analysis.min_weight - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_wfa_recommend_target() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        let target = WFAAnalyzer::recommend_target(&analysis);
        assert_eq!(target, CompilationTarget::AlgebraicDirect);
    }

    #[test]
    fn test_wfa_estimate_trace_size() {
        let mut wfa = WFA::<CountingSemiring>::new(4, 2);
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        let size = WFAAnalyzer::estimate_trace_size(&analysis, 10);
        assert!(size > 0);
    }

    #[test]
    fn test_wfa_estimate_constraint_count() {
        let mut wfa = WFA::<CountingSemiring>::new(3, 2);
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 2, 1, CountingSemiring(1));
        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        let count = WFAAnalyzer::estimate_constraint_count(&analysis);
        assert!(count >= analysis.num_transitions);
    }

    #[test]
    fn test_wfa_density() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        // Max transitions = 2 * 2 * 2 = 8. Adding 4 → density = 0.5
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 0, 1, CountingSemiring(1));
        wfa.add_transition(1, 1, 1, CountingSemiring(1));
        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        assert!((analysis.density - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_wfa_nondeterministic_detection() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.add_transition(0, 0, 0, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1)); // same (state, symbol)
        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        assert!(!analysis.is_deterministic);
    }

    // ─── CircuitOptimizer tests ─────────────────────────────────────────

    #[test]
    fn test_constant_propagation_removes_zero() {
        let constraints = vec![
            AIRConstraint::boundary(SymbolicExpression::constant(0), 0, "zero"),
            AIRConstraint::boundary(SymbolicExpression::col(0), 0, "real"),
        ];
        let result = CircuitOptimizer::constant_propagation(&constraints);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "real");
    }

    #[test]
    fn test_constant_propagation_keeps_nonzero() {
        let constraints = vec![
            AIRConstraint::boundary(SymbolicExpression::constant(5), 0, "five"),
        ];
        let result = CircuitOptimizer::constant_propagation(&constraints);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_strength_reduction_x_squared() {
        let expr = SymbolicExpression::col(0).mul(SymbolicExpression::col(0));
        let constraints = vec![AIRConstraint::transition(expr, "x_sq")];
        let result = CircuitOptimizer::strength_reduction(&constraints);
        assert_eq!(result.len(), 1);
        // The expression should now be Pow(Column(0), 2)
        match &result[0].expression {
            SymbolicExpression::Pow(inner, 2) => {
                match inner.as_ref() {
                    SymbolicExpression::Column(0) => {}
                    other => panic!("expected Column(0), got {:?}", other),
                }
            }
            other => panic!("expected Pow, got {:?}", other),
        }
    }

    #[test]
    fn test_strength_reduction_different_cols() {
        let expr = SymbolicExpression::col(0).mul(SymbolicExpression::col(1));
        let constraints = vec![AIRConstraint::transition(expr, "xy")];
        let result = CircuitOptimizer::strength_reduction(&constraints);
        match &result[0].expression {
            SymbolicExpression::Mul(_, _) => {}
            other => panic!("expected Mul, got {:?}", other),
        }
    }

    #[test]
    fn test_inline_auxiliary_returns_same() {
        let constraints = make_test_constraints();
        let layout = TraceLayout::new();
        let result = CircuitOptimizer::inline_auxiliary(&constraints, &layout);
        assert_eq!(result.len(), constraints.len());
    }

    #[test]
    fn test_remove_unused_columns() {
        let mut layout = TraceLayout::new();
        layout.add_column("used", ColumnType::Witness, "used col");
        layout.add_column("unused", ColumnType::Auxiliary, "never referenced");
        let mut air = AIRProgram::new(layout, 16);
        air.add_constraint(AIRConstraint::boundary(SymbolicExpression::col(0), 0, "init"));
        let optimized = CircuitOptimizer::remove_unused_columns(&air);
        // Column 1 (unused) should be removed.
        assert!(optimized.trace_layout.columns.len() <= air.trace_layout.columns.len());
    }

    #[test]
    fn test_optimize_fully() {
        let mut layout = TraceLayout::new();
        layout.add_column("x", ColumnType::Witness, "x");
        layout.add_column("y", ColumnType::Witness, "y");
        let mut air = AIRProgram::new(layout, 16);
        air.add_constraint(AIRConstraint::boundary(SymbolicExpression::constant(0), 0, "zero_const"));
        air.add_constraint(AIRConstraint::transition(SymbolicExpression::col(0), "real"));
        let optimized = CircuitOptimizer::optimize_fully(&air);
        // The zero constraint should be removed.
        assert!(optimized.constraints.len() <= air.constraints.len());
    }

    // ─── TraceLayoutOptimizer tests ─────────────────────────────────────

    fn make_test_layout(n: usize) -> TraceLayout {
        let mut layout = TraceLayout::new();
        for i in 0..n {
            layout.add_column(&format!("col{}", i), ColumnType::Witness, "");
        }
        layout
    }

    #[test]
    fn test_trace_layout_optimize() {
        let layout = make_test_layout(4);
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::col(2), "c2"),
            AIRConstraint::transition(SymbolicExpression::col(0), "c0"),
        ];
        let optimized = TraceLayoutOptimizer::optimize(&layout, &constraints);
        assert_eq!(optimized.columns.len(), layout.columns.len());
    }

    #[test]
    fn test_trace_layout_minimize_width() {
        let layout = make_test_layout(4);
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::col(0), "c0"),
            AIRConstraint::transition(SymbolicExpression::col(2), "c2"),
        ];
        let minimized = TraceLayoutOptimizer::minimize_width(&layout, &constraints);
        assert!(minimized.columns.len() <= layout.columns.len());
    }

    #[test]
    fn test_trace_layout_group_by_access() {
        let layout = make_test_layout(4);
        let constraints = vec![
            AIRConstraint::transition(
                SymbolicExpression::col(0).add(SymbolicExpression::col(3)),
                "co_access",
            ),
        ];
        let grouped = TraceLayoutOptimizer::group_by_access_pattern(&layout, &constraints);
        assert_eq!(grouped.columns.len(), layout.columns.len());
    }

    #[test]
    fn test_trace_layout_cache_efficiency_empty() {
        let layout = TraceLayout::new();
        let efficiency = TraceLayoutOptimizer::estimate_cache_efficiency(&layout, &[]);
        assert_eq!(efficiency, 1.0);
    }

    #[test]
    fn test_trace_layout_cache_efficiency_single_col() {
        let layout = make_test_layout(4);
        let constraints = vec![
            AIRConstraint::transition(SymbolicExpression::col(0), "single"),
        ];
        let efficiency = TraceLayoutOptimizer::estimate_cache_efficiency(&layout, &constraints);
        assert_eq!(efficiency, 1.0);
    }

    #[test]
    fn test_trace_layout_cache_efficiency_range() {
        let layout = make_test_layout(10);
        let constraints = vec![
            AIRConstraint::transition(
                SymbolicExpression::col(0).add(SymbolicExpression::col(9)),
                "wide",
            ),
        ];
        let efficiency = TraceLayoutOptimizer::estimate_cache_efficiency(&layout, &constraints);
        assert!(efficiency >= 0.0);
        assert!(efficiency <= 1.0);
    }

    // ─── CompilationReport tests ────────────────────────────────────────

    fn make_simple_result() -> CompilationResult {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig::default());
        compiler.compile(&wfa, 4)
    }

    #[test]
    fn test_report_from_result() {
        let result = make_simple_result();
        let report = CompilationReport::from_result(&result);
        assert!(report.trace_width > 0 || report.trace_length > 0 || report.num_constraints > 0);
    }

    #[test]
    fn test_report_to_text() {
        let result = make_simple_result();
        let report = CompilationReport::from_result(&result);
        let text = report.to_text();
        assert!(text.contains("Compilation Report"));
        assert!(text.contains("Trace:"));
        assert!(text.contains("Constraints:"));
    }

    #[test]
    fn test_report_to_json() {
        let result = make_simple_result();
        let report = CompilationReport::from_result(&result);
        let json = report.to_json();
        assert!(json.contains("\"trace_width\""));
        assert!(json.contains("\"num_constraints\""));
    }

    #[test]
    fn test_report_to_html() {
        let result = make_simple_result();
        let report = CompilationReport::from_result(&result);
        let html = report.to_html();
        assert!(html.contains("<table>"));
        assert!(html.contains("Compilation Report"));
        assert!(html.contains("</table>"));
    }

    #[test]
    fn test_report_comparison() {
        let r1 = CompilationReport::from_result(&make_simple_result());
        let r2 = CompilationReport::from_result(&make_simple_result());
        let cmp = CompilationReport::comparison(&[r1, r2]);
        assert!(cmp.contains("Report Comparison"));
        assert!(cmp.contains("Run 0"));
        assert!(cmp.contains("Run 1"));
    }

    #[test]
    fn test_report_comparison_empty() {
        let cmp = CompilationReport::comparison(&[]);
        assert!(cmp.contains("No reports"));
    }

    #[test]
    fn test_report_comparison_single() {
        let r = CompilationReport::from_result(&make_simple_result());
        let cmp = CompilationReport::comparison(&[r]);
        assert!(cmp.contains("Run 0"));
    }

    // ─── Integration tests ──────────────────────────────────────────────

    #[test]
    fn test_pipeline_with_diagnostics() {
        let pipeline = CompilerPipeline::default_pipeline();
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        let result = pipeline.run(&wfa, 4);

        let mut diag = CompilerDiagnostics::new();
        for name in pipeline.stage_names() {
            diag.record_stage(&name, 10, result.compilation_stats.num_constraints);
        }
        diag.record_constraint_count("final", result.compilation_stats.num_constraints);
        assert!(diag.total_time_us() > 0);
    }

    #[test]
    fn test_analyze_then_compile() {
        let mut wfa = WFA::<CountingSemiring>::new(3, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(2, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));
        wfa.add_transition(1, 2, 1, CountingSemiring(1));

        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        assert_eq!(analysis.num_states, 3);
        let target = WFAAnalyzer::recommend_target(&analysis);
        assert_eq!(target, CompilationTarget::AlgebraicDirect);

        let est_size = WFAAnalyzer::estimate_trace_size(&analysis, 8);
        assert!(est_size > 0);

        let mut compiler = WFACircuitCompiler::<CountingSemiring>::new(CompilerConfig::default());
        let result = compiler.compile(&wfa, 8);
        let report = CompilationReport::from_result(&result);
        assert!(!report.to_text().is_empty());
    }

    #[test]
    fn test_optimize_compiled_circuit() {
        let result = make_simple_result();
        let optimized = CircuitOptimizer::optimize_fully(&result.air_program);
        assert!(optimized.constraints.len() <= result.air_program.constraints.len());
    }

    #[test]
    fn test_constraint_analysis_on_compiled() {
        let result = make_simple_result();
        let analysis = ConstraintAnalyzer::analyze(&result.air_program.constraints);
        assert_eq!(analysis.total, result.air_program.constraints.len());
    }

    #[test]
    fn test_full_workflow_with_report() {
        let mut wfa = WFA::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring(1));
        wfa.set_final(1, CountingSemiring(1));
        wfa.add_transition(0, 1, 0, CountingSemiring(1));

        let analysis = WFAAnalyzer::analyze_counting(&wfa);
        let est = WFAAnalyzer::estimate_constraint_count(&analysis);
        assert!(est > 0);

        let pipeline = CompilerPipeline::default_pipeline();
        let result = pipeline.run(&wfa, 4);

        let report = CompilationReport::from_result(&result);
        let text = report.to_text();
        let json = report.to_json();
        let html = report.to_html();
        assert!(!text.is_empty());
        assert!(!json.is_empty());
        assert!(!html.is_empty());
    }
}
