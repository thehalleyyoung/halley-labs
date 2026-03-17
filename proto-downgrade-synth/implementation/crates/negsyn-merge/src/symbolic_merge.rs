//! Symbolic state merging — merges symbolic execution states by combining
//! path constraints via disjunction and creating ITE expressions for
//! differing symbolic values.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use negsyn_types::{
    BinOp, ConcreteValue, MergeConfig, MergeError, MergeResult, MemoryPermissions, MemoryRegion,
    PathConstraint, SymSort, SymbolicMemory, SymbolicState, SymbolicValue, UnOp,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a PathConstraint from a single condition expression.
fn path_constraint_from(cond: SymbolicValue) -> PathConstraint {
    PathConstraint {
        conditions: vec![cond.clone()],
        constraint: Some(cond),
        is_negated: false,
        parent_id: None,
    }
}

/// Convert SymbolicMemory to a name-keyed BTreeMap for merge operations.
fn memory_to_btreemap(memory: &SymbolicMemory) -> BTreeMap<String, MemoryRegion> {
    memory
        .regions
        .iter()
        .map(|r| (r.name.clone(), r.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Constraint merge
// ---------------------------------------------------------------------------

/// Merges path constraints from two execution paths via disjunction.
///
/// Given constraint sets Φ₁ = {c₁₁ ∧ c₁₂ ∧ ...} and Φ₂ = {c₂₁ ∧ c₂₂ ∧ ...},
/// the merged constraint is (∧Φ₁) ∨ (∧Φ₂).
pub struct ConstraintMerge {
    enable_simplification: bool,
    max_constraint_nodes: usize,
}

impl ConstraintMerge {
    pub fn new(config: &MergeConfig) -> Self {
        Self {
            enable_simplification: config.enable_constraint_simplification,
            max_constraint_nodes: config.max_merged_constraints as usize,
        }
    }

    /// Build the disjunction of two constraint sets.
    pub fn merge_constraints(
        &self,
        left: &[PathConstraint],
        right: &[PathConstraint],
    ) -> MergeResult<Vec<PathConstraint>> {
        let left_conj = Self::conjunction(left);
        let right_conj = Self::conjunction(right);

        let disjunction = left_conj.or_expr(right_conj);

        let simplified = if self.enable_simplification {
            self.simplify(disjunction)
        } else {
            disjunction
        };

        if simplified.node_count() > self.max_constraint_nodes {
            return Err(MergeError::ComplexityExceeded {
                reason: "merged constraint too complex".to_string(),
                complexity: simplified.node_count() as u32,
                limit: self.max_constraint_nodes as u32,
            });
        }

        Ok(vec![path_constraint_from(simplified)])
    }

    /// Build a conjunction from a slice of path constraints.
    pub fn conjunction(constraints: &[PathConstraint]) -> SymbolicValue {
        if constraints.is_empty() {
            return SymbolicValue::Concrete(ConcreteValue::Bool(true));
        }
        let mut result = constraints[0].effective_constraint();
        for c in &constraints[1..] {
            result = result.and_expr(c.effective_constraint());
        }
        result
    }

    /// Extract the condition distinguishing left from right path.
    /// This is the left conjunction, used as the ITE condition.
    pub fn distinguishing_condition(left: &[PathConstraint]) -> SymbolicValue {
        Self::conjunction(left)
    }

    /// Simplify a symbolic boolean expression.
    pub fn simplify(&self, expr: SymbolicValue) -> SymbolicValue {
        self.simplify_recursive(expr, 0)
    }

    fn simplify_recursive(&self, expr: SymbolicValue, depth: usize) -> SymbolicValue {
        if depth > 64 {
            return expr;
        }

        match expr {
            // NOT(NOT(x)) => x
            SymbolicValue::UnaryOp { op: UnOp::Not, operand: inner } => match *inner {
                SymbolicValue::UnaryOp { op: UnOp::Not, operand: x } => self.simplify_recursive(*x, depth + 1),
                SymbolicValue::Concrete(ConcreteValue::Bool(b)) => SymbolicValue::Concrete(ConcreteValue::Bool(!b)),
                other => {
                    let s = self.simplify_recursive(other, depth + 1);
                    match s {
                        SymbolicValue::UnaryOp { op: UnOp::Not, operand: x } => *x,
                        SymbolicValue::Concrete(ConcreteValue::Bool(b)) => SymbolicValue::Concrete(ConcreteValue::Bool(!b)),
                        other => SymbolicValue::unary(UnOp::Not, other),
                    }
                }
            },

            // AND simplifications
            SymbolicValue::BinaryOp { op: BinOp::LogicAnd, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (SymbolicValue::Concrete(ConcreteValue::Bool(true)), _) => sb,
                    (_, SymbolicValue::Concrete(ConcreteValue::Bool(true))) => sa,
                    (SymbolicValue::Concrete(ConcreteValue::Bool(false)), _) => SymbolicValue::Concrete(ConcreteValue::Bool(false)),
                    (_, SymbolicValue::Concrete(ConcreteValue::Bool(false))) => SymbolicValue::Concrete(ConcreteValue::Bool(false)),
                    _ if sa == sb => sa,
                    _ => SymbolicValue::binary(BinOp::LogicAnd, sa, sb),
                }
            }

            // OR simplifications
            SymbolicValue::BinaryOp { op: BinOp::LogicOr, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (SymbolicValue::Concrete(ConcreteValue::Bool(false)), _) => sb,
                    (_, SymbolicValue::Concrete(ConcreteValue::Bool(false))) => sa,
                    (SymbolicValue::Concrete(ConcreteValue::Bool(true)), _) => SymbolicValue::Concrete(ConcreteValue::Bool(true)),
                    (_, SymbolicValue::Concrete(ConcreteValue::Bool(true))) => SymbolicValue::Concrete(ConcreteValue::Bool(true)),
                    _ if sa == sb => sa,
                    // a OR NOT(a) => true
                    (_, SymbolicValue::UnaryOp { op: UnOp::Not, operand: inner }) if sa == **inner => {
                        SymbolicValue::Concrete(ConcreteValue::Bool(true))
                    }
                    (SymbolicValue::UnaryOp { op: UnOp::Not, operand: inner }, _) if **inner == sb => {
                        SymbolicValue::Concrete(ConcreteValue::Bool(true))
                    }
                    _ => SymbolicValue::binary(BinOp::LogicOr, sa, sb),
                }
            }

            // ITE simplifications
            SymbolicValue::Ite { condition: c, then_val: t, else_val: e } => {
                let sc = self.simplify_recursive(*c, depth + 1);
                let st = self.simplify_recursive(*t, depth + 1);
                let se = self.simplify_recursive(*e, depth + 1);
                match &sc {
                    SymbolicValue::Concrete(ConcreteValue::Bool(true)) => st,
                    SymbolicValue::Concrete(ConcreteValue::Bool(false)) => se,
                    _ if st == se => st,
                    _ => SymbolicValue::ite(sc, st, se),
                }
            }

            // EQ simplifications
            SymbolicValue::BinaryOp { op: BinOp::Eq, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                if sa == sb {
                    SymbolicValue::Concrete(ConcreteValue::Bool(true))
                } else {
                    match (&sa, &sb) {
                        (SymbolicValue::Concrete(a_val), SymbolicValue::Concrete(b_val)) => {
                            SymbolicValue::Concrete(ConcreteValue::Bool(a_val == b_val))
                        }
                        _ => SymbolicValue::binary(BinOp::Eq, sa, sb),
                    }
                }
            }

            // Add constant folding
            SymbolicValue::BinaryOp { op: BinOp::Add, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (SymbolicValue::Concrete(ConcreteValue::Int(0)), _) => sb,
                    (_, SymbolicValue::Concrete(ConcreteValue::Int(0))) => sa,
                    (SymbolicValue::Concrete(ConcreteValue::Int(a)), SymbolicValue::Concrete(ConcreteValue::Int(b))) => {
                        SymbolicValue::Concrete(ConcreteValue::Int(a.wrapping_add(*b)))
                    }
                    _ => SymbolicValue::binary(BinOp::Add, sa, sb),
                }
            }

            // Sub constant folding
            SymbolicValue::BinaryOp { op: BinOp::Sub, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (_, SymbolicValue::Concrete(ConcreteValue::Int(0))) => sa,
                    (SymbolicValue::Concrete(ConcreteValue::Int(a)), SymbolicValue::Concrete(ConcreteValue::Int(b))) => {
                        SymbolicValue::Concrete(ConcreteValue::Int(a.wrapping_sub(*b)))
                    }
                    _ if sa == sb => SymbolicValue::Concrete(ConcreteValue::Int(0)),
                    _ => SymbolicValue::binary(BinOp::Sub, sa, sb),
                }
            }

            // Mul constant folding
            SymbolicValue::BinaryOp { op: BinOp::Mul, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (SymbolicValue::Concrete(ConcreteValue::Int(0)), _) | (_, SymbolicValue::Concrete(ConcreteValue::Int(0))) => {
                        SymbolicValue::Concrete(ConcreteValue::Int(0))
                    }
                    (SymbolicValue::Concrete(ConcreteValue::Int(1)), _) => sb,
                    (_, SymbolicValue::Concrete(ConcreteValue::Int(1))) => sa,
                    (SymbolicValue::Concrete(ConcreteValue::Int(a)), SymbolicValue::Concrete(ConcreteValue::Int(b))) => {
                        SymbolicValue::Concrete(ConcreteValue::Int(a.wrapping_mul(*b)))
                    }
                    _ => SymbolicValue::binary(BinOp::Mul, sa, sb),
                }
            }

            // Lt constant folding
            SymbolicValue::BinaryOp { op: BinOp::Lt, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (SymbolicValue::Concrete(ConcreteValue::Int(a)), SymbolicValue::Concrete(ConcreteValue::Int(b))) => {
                        SymbolicValue::Concrete(ConcreteValue::Bool(a < b))
                    }
                    _ if sa == sb => SymbolicValue::Concrete(ConcreteValue::Bool(false)),
                    _ => SymbolicValue::binary(BinOp::Lt, sa, sb),
                }
            }

            // Le constant folding
            SymbolicValue::BinaryOp { op: BinOp::Le, left: a, right: b } => {
                let sa = self.simplify_recursive(*a, depth + 1);
                let sb = self.simplify_recursive(*b, depth + 1);
                match (&sa, &sb) {
                    (SymbolicValue::Concrete(ConcreteValue::Int(a)), SymbolicValue::Concrete(ConcreteValue::Int(b))) => {
                        SymbolicValue::Concrete(ConcreteValue::Bool(a <= b))
                    }
                    _ if sa == sb => SymbolicValue::Concrete(ConcreteValue::Bool(true)),
                    _ => SymbolicValue::binary(BinOp::Le, sa, sb),
                }
            }

            other => other,
        }
    }

    /// Check if a constraint is trivially satisfiable.
    pub fn is_trivially_sat(constraint: &SymbolicValue) -> bool {
        matches!(constraint, SymbolicValue::Concrete(ConcreteValue::Bool(true)))
    }

    /// Check if a constraint is trivially unsatisfiable.
    pub fn is_trivially_unsat(constraint: &SymbolicValue) -> bool {
        matches!(constraint, SymbolicValue::Concrete(ConcreteValue::Bool(false)))
    }
}

// ---------------------------------------------------------------------------
// Value merge
// ---------------------------------------------------------------------------

/// Merges individual symbolic values using ITE expressions.
pub struct ValueMerge {
    max_ite_depth: u32,
}

impl ValueMerge {
    pub fn new(config: &MergeConfig) -> Self {
        Self {
            max_ite_depth: config.max_ite_depth,
        }
    }

    /// Merge two symbolic values under a condition.
    /// If values are equal, returns the value directly (no ITE).
    /// Otherwise returns ITE(condition, left_val, right_val).
    pub fn merge_values(
        &self,
        condition: &SymbolicValue,
        left: &SymbolicValue,
        right: &SymbolicValue,
    ) -> MergeResult<SymbolicValue> {
        if left == right {
            return Ok(left.clone());
        }

        let depth = self.ite_nesting_depth(left).max(self.ite_nesting_depth(right));
        if depth >= self.max_ite_depth {
            return Err(MergeError::ComplexityExceeded {
                reason: "ITE nesting depth exceeded".to_string(),
                complexity: depth,
                limit: self.max_ite_depth,
            });
        }

        Ok(SymbolicValue::ite(
            condition.clone(),
            left.clone(),
            right.clone(),
        ))
    }

    /// Merge two optional values.
    pub fn merge_optional(
        &self,
        condition: &SymbolicValue,
        left: Option<&SymbolicValue>,
        right: Option<&SymbolicValue>,
    ) -> MergeResult<Option<SymbolicValue>> {
        match (left, right) {
            (Some(l), Some(r)) => Ok(Some(self.merge_values(condition, l, r)?)),
            (Some(l), None) => Ok(Some(l.clone())),
            (None, Some(r)) => Ok(Some(r.clone())),
            (None, None) => Ok(None),
        }
    }

    /// Count the ITE nesting depth of an expression.
    fn ite_nesting_depth(&self, expr: &SymbolicValue) -> u32 {
        match expr {
            SymbolicValue::Ite { condition, then_val, else_val } => {
                1 + self
                    .ite_nesting_depth(condition)
                    .max(self.ite_nesting_depth(then_val))
                    .max(self.ite_nesting_depth(else_val))
            }
            SymbolicValue::UnaryOp { operand, .. } => self.ite_nesting_depth(operand),
            SymbolicValue::BinaryOp { left, right, .. }
            | SymbolicValue::Select { array: left, index: right } => {
                self.ite_nesting_depth(left).max(self.ite_nesting_depth(right))
            }
            SymbolicValue::Store { array, index, value } => self
                .ite_nesting_depth(array)
                .max(self.ite_nesting_depth(index))
                .max(self.ite_nesting_depth(value)),
            _ => 0,
        }
    }

    /// Flatten nested ITE expressions with the same condition.
    pub fn flatten_ite(&self, expr: &SymbolicValue) -> SymbolicValue {
        match expr {
            SymbolicValue::Ite { condition: c, then_val: t, else_val: e } => {
                let ft = self.flatten_ite(t);
                let fe = self.flatten_ite(e);

                // ITE(c, ITE(c, a, b), e) => ITE(c, a, e)
                if let SymbolicValue::Ite { condition: inner_c, then_val: inner_t, .. } = &ft {
                    if **inner_c == **c {
                        return SymbolicValue::ite((**c).clone(), (**inner_t).clone(), fe);
                    }
                }

                // ITE(c, t, ITE(c, a, b)) => ITE(c, t, b)
                if let SymbolicValue::Ite { condition: inner_c, else_val: inner_e, .. } = &fe {
                    if **inner_c == **c {
                        return SymbolicValue::ite((**c).clone(), ft, (**inner_e).clone());
                    }
                }

                SymbolicValue::ite((**c).clone(), ft, fe)
            }
            other => other.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory merge
// ---------------------------------------------------------------------------

/// Merges symbolic memory regions from two execution states.
pub struct MemoryMerge {
    value_merge: ValueMerge,
}

impl MemoryMerge {
    pub fn new(config: &MergeConfig) -> Self {
        Self {
            value_merge: ValueMerge::new(config),
        }
    }

    /// Merge two memory region maps under a distinguishing condition.
    pub fn merge_regions(
        &self,
        condition: &SymbolicValue,
        left: &BTreeMap<String, MemoryRegion>,
        right: &BTreeMap<String, MemoryRegion>,
    ) -> MergeResult<(BTreeMap<String, MemoryRegion>, MemoryMergeStats)> {
        let mut merged = BTreeMap::new();
        let mut stats = MemoryMergeStats::default();

        let all_names: BTreeSet<&String> = left.keys().chain(right.keys()).collect();

        for name in all_names {
            match (left.get(name), right.get(name)) {
                (Some(lr), Some(rr)) => {
                    let (region, region_stats) = self.merge_single_region(condition, lr, rr)?;
                    stats.merge_with(&region_stats);
                    stats.regions_merged += 1;
                    merged.insert(name.clone(), region);
                }
                (Some(lr), None) => {
                    stats.left_only_regions += 1;
                    merged.insert(name.clone(), lr.clone());
                }
                (None, Some(rr)) => {
                    stats.right_only_regions += 1;
                    merged.insert(name.clone(), rr.clone());
                }
                (None, None) => unreachable!(),
            }
        }

        Ok((merged, stats))
    }

    /// Merge two memory regions cell-by-cell.
    fn merge_single_region(
        &self,
        condition: &SymbolicValue,
        left: &MemoryRegion,
        right: &MemoryRegion,
    ) -> MergeResult<(MemoryRegion, MemoryMergeStats)> {
        let perms = MemoryPermissions {
            read: true,
            write: left.is_writable() || right.is_writable(),
            execute: false,
        };
        let mut merged = MemoryRegion::new(
            left.name.clone(),
            left.base().min(right.base()),
            left.size.max(right.size),
            perms,
        );

        let mut stats = MemoryMergeStats::default();

        let all_offsets: BTreeSet<u64> = left
            .contents()
            .keys()
            .chain(right.contents().keys())
            .copied()
            .collect();

        for offset in all_offsets {
            match (left.contents().get(&offset), right.contents().get(&offset)) {
                (Some(lv), Some(rv)) => {
                    if lv == rv {
                        stats.identical_cells += 1;
                        merged.write(offset, lv.clone());
                    } else {
                        stats.ite_cells += 1;
                        let merged_val = self.value_merge.merge_values(condition, lv, rv)?;
                        merged.write(offset, merged_val);
                    }
                }
                (Some(lv), None) => {
                    stats.left_only_cells += 1;
                    merged.write(offset, lv.clone());
                }
                (None, Some(rv)) => {
                    stats.right_only_cells += 1;
                    merged.write(offset, rv.clone());
                }
                (None, None) => unreachable!(),
            }
        }

        Ok((merged, stats))
    }

    /// Estimate the number of ITE nodes that would be created.
    pub fn estimate_ite_count(
        &self,
        left: &BTreeMap<String, MemoryRegion>,
        right: &BTreeMap<String, MemoryRegion>,
    ) -> usize {
        let mut count = 0;
        let all_names: BTreeSet<&String> = left.keys().chain(right.keys()).collect();

        for name in all_names {
            if let (Some(lr), Some(rr)) = (left.get(name), right.get(name)) {
                let all_offsets: BTreeSet<u64> = lr
                    .contents()
                    .keys()
                    .chain(rr.contents().keys())
                    .copied()
                    .collect();

                for offset in all_offsets {
                    if let (Some(lv), Some(rv)) = (lr.contents().get(&offset), rr.contents().get(&offset))
                    {
                        if lv != rv {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }
}

/// Statistics from a memory merge operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMergeStats {
    pub regions_merged: usize,
    pub left_only_regions: usize,
    pub right_only_regions: usize,
    pub identical_cells: usize,
    pub ite_cells: usize,
    pub left_only_cells: usize,
    pub right_only_cells: usize,
}

impl MemoryMergeStats {
    pub fn total_cells(&self) -> usize {
        self.identical_cells + self.ite_cells + self.left_only_cells + self.right_only_cells
    }

    pub fn merge_with(&mut self, other: &MemoryMergeStats) {
        self.regions_merged += other.regions_merged;
        self.left_only_regions += other.left_only_regions;
        self.right_only_regions += other.right_only_regions;
        self.identical_cells += other.identical_cells;
        self.ite_cells += other.ite_cells;
        self.left_only_cells += other.left_only_cells;
        self.right_only_cells += other.right_only_cells;
    }
}

// ---------------------------------------------------------------------------
// Phi node insertion
// ---------------------------------------------------------------------------

/// Handles phi-node insertion for merged control flow.
///
/// When two execution paths merge, variables that differ between the paths
/// need phi-nodes (implemented as ITE expressions over the path condition).
pub struct PhiNodeInsertion {
    value_merge: ValueMerge,
    inserted_phis: Vec<PhiNode>,
}

/// A phi node recording the merge of a variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiNode {
    pub variable: String,
    pub condition: SymbolicValue,
    pub left_value: SymbolicValue,
    pub right_value: SymbolicValue,
    pub merged_value: SymbolicValue,
}

impl PhiNodeInsertion {
    pub fn new(config: &MergeConfig) -> Self {
        Self {
            value_merge: ValueMerge::new(config),
            inserted_phis: Vec::new(),
        }
    }

    /// Insert phi nodes for all differing variables between two register maps.
    pub fn insert_phis(
        &mut self,
        condition: &SymbolicValue,
        left: &BTreeMap<String, SymbolicValue>,
        right: &BTreeMap<String, SymbolicValue>,
    ) -> MergeResult<BTreeMap<String, SymbolicValue>> {
        let mut merged = BTreeMap::new();
        let all_vars: BTreeSet<&String> = left.keys().chain(right.keys()).collect();

        for var in all_vars {
            let lv = left.get(var);
            let rv = right.get(var);

            match (lv, rv) {
                (Some(l), Some(r)) if l == r => {
                    merged.insert(var.clone(), l.clone());
                }
                (Some(l), Some(r)) => {
                    let merged_val = self.value_merge.merge_values(condition, l, r)?;
                    self.inserted_phis.push(PhiNode {
                        variable: var.clone(),
                        condition: condition.clone(),
                        left_value: l.clone(),
                        right_value: r.clone(),
                        merged_value: merged_val.clone(),
                    });
                    merged.insert(var.clone(), merged_val);
                }
                (Some(l), None) => {
                    let default_val = SymbolicValue::Concrete(ConcreteValue::Int(0));
                    let merged_val = self.value_merge.merge_values(condition, l, &default_val)?;
                    self.inserted_phis.push(PhiNode {
                        variable: var.clone(),
                        condition: condition.clone(),
                        left_value: l.clone(),
                        right_value: default_val,
                        merged_value: merged_val.clone(),
                    });
                    merged.insert(var.clone(), merged_val);
                }
                (None, Some(r)) => {
                    let default_val = SymbolicValue::Concrete(ConcreteValue::Int(0));
                    let merged_val = self.value_merge.merge_values(condition, &default_val, r)?;
                    self.inserted_phis.push(PhiNode {
                        variable: var.clone(),
                        condition: condition.clone(),
                        left_value: default_val,
                        right_value: r.clone(),
                        merged_value: merged_val.clone(),
                    });
                    merged.insert(var.clone(), merged_val);
                }
                (None, None) => unreachable!(),
            }
        }

        Ok(merged)
    }

    pub fn phi_count(&self) -> usize {
        self.inserted_phis.len()
    }

    pub fn phis(&self) -> &[PhiNode] {
        &self.inserted_phis
    }

    pub fn clear(&mut self) {
        self.inserted_phis.clear();
    }

    /// Get all variables that needed phi nodes.
    pub fn phi_variables(&self) -> Vec<&str> {
        self.inserted_phis
            .iter()
            .map(|p| p.variable.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Symbolic merger (top-level orchestrator)
// ---------------------------------------------------------------------------

/// Top-level symbolic state merger that orchestrates constraint merge,
/// value merge, memory merge, and phi-node insertion.
pub struct SymbolicMerger {
    constraint_merge: ConstraintMerge,
    value_merge: ValueMerge,
    memory_merge: MemoryMerge,
    phi_insertion: PhiNodeInsertion,
    config: MergeConfig,
}

impl SymbolicMerger {
    pub fn new(config: MergeConfig) -> Self {
        let constraint_merge = ConstraintMerge::new(&config);
        let value_merge = ValueMerge::new(&config);
        let memory_merge = MemoryMerge::new(&config);
        let phi_insertion = PhiNodeInsertion::new(&config);
        Self {
            constraint_merge,
            value_merge,
            memory_merge,
            phi_insertion,
            config,
        }
    }

    /// Perform a full symbolic merge of two states.
    pub fn merge(
        &mut self,
        merged_id: u64,
        left: &SymbolicState,
        right: &SymbolicState,
    ) -> MergeResult<(SymbolicState, SymbolicMergeStats)> {
        let mut stats = SymbolicMergeStats::default();

        // 1. Merge constraints via disjunction
        let merged_constraints = self
            .constraint_merge
            .merge_constraints(
                std::slice::from_ref(&left.pc),
                std::slice::from_ref(&right.pc),
            )?;
        stats.constraint_nodes = merged_constraints
            .iter()
            .map(|c| c.constraint.as_ref().map_or(0, |v| v.node_count()))
            .sum();

        // 2. Compute distinguishing condition
        let condition = ConstraintMerge::distinguishing_condition(
            std::slice::from_ref(&left.pc),
        );

        // 3. Insert phi nodes for registers (convert HashMap to BTreeMap)
        self.phi_insertion.clear();
        let left_regs: BTreeMap<String, SymbolicValue> =
            left.registers.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let right_regs: BTreeMap<String, SymbolicValue> =
            right.registers.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let merged_registers_btree = self
            .phi_insertion
            .insert_phis(&condition, &left_regs, &right_regs)?;
        stats.phi_nodes = self.phi_insertion.phi_count();

        // 4. Merge memory (convert SymbolicMemory to BTreeMap)
        let left_mem = memory_to_btreemap(&left.memory);
        let right_mem = memory_to_btreemap(&right.memory);
        let (merged_memory_btree, mem_stats) =
            self.memory_merge
                .merge_regions(&condition, &left_mem, &right_mem)?;
        stats.memory_stats = mem_stats;

        // 5. Build merged state
        let merged_pc = if let Some(first) = merged_constraints.first() {
            first.clone()
        } else {
            PathConstraint::new()
        };
        let merged_constraints_values: Vec<SymbolicValue> = merged_pc.conditions.clone();
        let merged_registers: HashMap<String, SymbolicValue> =
            merged_registers_btree.into_iter().collect();
        let mut merged_sym_memory = SymbolicMemory::new();
        for (_, region) in merged_memory_btree {
            merged_sym_memory.add_region(region);
        }

        let merged = SymbolicState {
            id: merged_id,
            program_counter: left.program_counter,
            registers: merged_registers,
            memory: merged_sym_memory,
            path_constraint: merged_pc.clone(),
            pc: merged_pc,
            constraints: merged_constraints_values,
            negotiation: left.negotiation.clone(),
            depth: left.depth.max(right.depth) + 1,
            is_feasible: true,
            parent_id: None,
        };

        Ok((merged, stats))
    }

    /// Estimate the cost of merging two states without actually merging.
    pub fn estimate_merge_size(
        &self,
        left: &SymbolicState,
        right: &SymbolicState,
    ) -> usize {
        let constraint_size = left
            .constraints
            .iter()
            .chain(right.constraints.iter())
            .map(|c| c.node_count())
            .sum::<usize>();

        let register_diffs = left
            .registers
            .keys()
            .chain(right.registers.keys())
            .collect::<BTreeSet<_>>()
            .len();

        let left_mem = memory_to_btreemap(&left.memory);
        let right_mem = memory_to_btreemap(&right.memory);
        let memory_ites = self
            .memory_merge
            .estimate_ite_count(&left_mem, &right_mem);

        constraint_size + register_diffs + memory_ites
    }

    pub fn constraint_merger(&self) -> &ConstraintMerge {
        &self.constraint_merge
    }

    pub fn phi_insertion(&self) -> &PhiNodeInsertion {
        &self.phi_insertion
    }
}

/// Statistics from a full symbolic merge.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolicMergeStats {
    pub constraint_nodes: usize,
    pub phi_nodes: usize,
    pub memory_stats: MemoryMergeStats,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> MergeConfig {
        MergeConfig::default()
    }

    fn make_state(id: u64) -> SymbolicState {
        SymbolicState::new(id, 0x1000)
    }

    fn sym_var(name: &str) -> SymbolicValue {
        SymbolicValue::var(name, SymSort::Int)
    }

    fn int_val(n: i64) -> SymbolicValue {
        SymbolicValue::Concrete(ConcreteValue::Int(n))
    }

    #[test]
    fn test_constraint_merge_empty() {
        let cm = ConstraintMerge::new(&make_config());
        let result = cm.merge_constraints(&[], &[]).unwrap();
        assert_eq!(result.len(), 1);
        // OR(true, true) => should simplify to true
        assert!(matches!(
            result[0].constraint,
            Some(SymbolicValue::Concrete(ConcreteValue::Bool(true)))
        ));
    }

    #[test]
    fn test_constraint_merge_with_constraints() {
        let cm = ConstraintMerge::new(&make_config());
        let c1 = vec![path_constraint_from(SymbolicValue::binary(
            BinOp::Eq,
            sym_var("x"),
            int_val(1),
        ))];
        let c2 = vec![path_constraint_from(SymbolicValue::binary(
            BinOp::Eq,
            sym_var("x"),
            int_val(2),
        ))];

        let result = cm.merge_constraints(&c1, &c2).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(
            result[0].constraint,
            Some(SymbolicValue::BinaryOp { op: BinOp::LogicOr, .. })
        ));
    }

    #[test]
    fn test_simplify_double_negation() {
        let cm = ConstraintMerge::new(&make_config());
        let expr = SymbolicValue::unary(UnOp::Not, SymbolicValue::unary(UnOp::Not, sym_var("x")));
        let simplified = cm.simplify(expr);
        assert!(matches!(simplified, SymbolicValue::Variable(_)));
    }

    #[test]
    fn test_simplify_and_true() {
        let cm = ConstraintMerge::new(&make_config());
        let expr = SymbolicValue::Concrete(ConcreteValue::Bool(true)).and_expr(sym_var("x"));
        let simplified = cm.simplify(expr);
        assert!(matches!(simplified, SymbolicValue::Variable(_)));
    }

    #[test]
    fn test_simplify_or_false() {
        let cm = ConstraintMerge::new(&make_config());
        let expr = SymbolicValue::Concrete(ConcreteValue::Bool(false)).or_expr(sym_var("x"));
        let simplified = cm.simplify(expr);
        assert!(matches!(simplified, SymbolicValue::Variable(_)));
    }

    #[test]
    fn test_simplify_or_complement() {
        let cm = ConstraintMerge::new(&make_config());
        let x = sym_var("x");
        let expr = x.clone().or_expr(SymbolicValue::unary(UnOp::Not, x));
        let simplified = cm.simplify(expr);
        assert_eq!(simplified, SymbolicValue::Concrete(ConcreteValue::Bool(true)));
    }

    #[test]
    fn test_simplify_constant_fold_add() {
        let cm = ConstraintMerge::new(&make_config());
        let expr = SymbolicValue::binary(BinOp::Add, int_val(3), int_val(4));
        let simplified = cm.simplify(expr);
        assert_eq!(simplified, int_val(7));
    }

    #[test]
    fn test_simplify_ite_same_branches() {
        let cm = ConstraintMerge::new(&make_config());
        let expr = SymbolicValue::ite(sym_var("cond"), int_val(42), int_val(42));
        let simplified = cm.simplify(expr);
        assert_eq!(simplified, int_val(42));
    }

    #[test]
    fn test_value_merge_identical() {
        let vm = ValueMerge::new(&make_config());
        let cond = sym_var("cond");
        let val = int_val(42);
        let result = vm.merge_values(&cond, &val, &val).unwrap();
        assert_eq!(result, int_val(42));
    }

    #[test]
    fn test_value_merge_different() {
        let vm = ValueMerge::new(&make_config());
        let cond = sym_var("cond");
        let left = int_val(1);
        let right = int_val(2);
        let result = vm.merge_values(&cond, &left, &right).unwrap();
        assert!(matches!(result, SymbolicValue::Ite { .. }));
    }

    #[test]
    fn test_flatten_ite() {
        let vm = ValueMerge::new(&make_config());
        let cond = sym_var("c");
        let inner = SymbolicValue::ite(cond.clone(), int_val(1), int_val(2));
        let outer = SymbolicValue::ite(cond.clone(), inner, int_val(3));
        let flattened = vm.flatten_ite(&outer);
        // Should flatten: ITE(c, ITE(c, 1, 2), 3) => ITE(c, 1, 3)
        if let SymbolicValue::Ite { then_val: t, else_val: e, .. } = &flattened {
            assert_eq!(**t, int_val(1));
            assert_eq!(**e, int_val(3));
        } else {
            panic!("Expected ITE");
        }
    }

    #[test]
    fn test_memory_merge_identical() {
        let mm = MemoryMerge::new(&make_config());
        let cond = sym_var("cond");

        let mut left = BTreeMap::new();
        let mut r1 = MemoryRegion::new("stack", 0, 64, MemoryPermissions::rw());
        r1.content.insert(0, int_val(42));
        left.insert("stack".to_string(), r1);

        let mut right = BTreeMap::new();
        let mut r2 = MemoryRegion::new("stack", 0, 64, MemoryPermissions::rw());
        r2.content.insert(0, int_val(42));
        right.insert("stack".to_string(), r2);

        let (merged, stats) = mm.merge_regions(&cond, &left, &right).unwrap();
        assert_eq!(stats.identical_cells, 1);
        assert_eq!(stats.ite_cells, 0);
        assert_eq!(
            merged.get("stack").unwrap().content.get(&0),
            Some(&int_val(42))
        );
    }

    #[test]
    fn test_memory_merge_different() {
        let mm = MemoryMerge::new(&make_config());
        let cond = sym_var("cond");

        let mut left = BTreeMap::new();
        let mut r1 = MemoryRegion::new("stack", 0, 64, MemoryPermissions::rw());
        r1.content.insert(0, int_val(1));
        left.insert("stack".to_string(), r1);

        let mut right = BTreeMap::new();
        let mut r2 = MemoryRegion::new("stack", 0, 64, MemoryPermissions::rw());
        r2.content.insert(0, int_val(2));
        right.insert("stack".to_string(), r2);

        let (merged, stats) = mm.merge_regions(&cond, &left, &right).unwrap();
        assert_eq!(stats.ite_cells, 1);
        assert!(matches!(
            merged.get("stack").unwrap().content.get(&0),
            Some(SymbolicValue::Ite { .. })
        ));
    }

    #[test]
    fn test_phi_node_insertion() {
        let config = make_config();
        let cond = sym_var("cond");
        let mut phi = PhiNodeInsertion::new(&config);

        let mut left = BTreeMap::new();
        left.insert("rax".to_string(), int_val(10));
        left.insert("rbx".to_string(), int_val(20));

        let mut right = BTreeMap::new();
        right.insert("rax".to_string(), int_val(10)); // same
        right.insert("rbx".to_string(), int_val(30)); // different

        let merged = phi.insert_phis(&cond, &left, &right).unwrap();
        assert_eq!(phi.phi_count(), 1); // only rbx needs phi
        assert_eq!(merged.get("rax"), Some(&int_val(10)));
        assert!(matches!(
            merged.get("rbx"),
            Some(SymbolicValue::Ite { .. })
        ));
    }

    #[test]
    fn test_symbolic_merger_full() {
        let config = make_config();
        let mut merger = SymbolicMerger::new(config);

        let mut s1 = make_state(1);
        s1.registers
            .insert("rax".to_string(), int_val(1));
        let eq1 = SymbolicValue::binary(BinOp::Eq, sym_var("x"), int_val(1));
        s1.constraints.push(eq1.clone());
        s1.pc = path_constraint_from(eq1);

        let mut s2 = make_state(2);
        s2.registers
            .insert("rax".to_string(), int_val(2));
        let eq2 = SymbolicValue::binary(BinOp::Eq, sym_var("x"), int_val(2));
        s2.constraints.push(eq2.clone());
        s2.pc = path_constraint_from(eq2);

        let (merged, stats) = merger.merge(100, &s1, &s2).unwrap();
        assert_eq!(merged.id, 100);
        assert!(stats.phi_nodes > 0);
    }

    #[test]
    fn test_estimate_merge_size() {
        let config = make_config();
        let merger = SymbolicMerger::new(config);

        let mut s1 = make_state(1);
        s1.registers
            .insert("rax".to_string(), int_val(1));

        let mut s2 = make_state(2);
        s2.registers
            .insert("rax".to_string(), int_val(2));

        let size = merger.estimate_merge_size(&s1, &s2);
        assert!(size > 0);
    }

    #[test]
    fn test_constraint_merge_complexity_limit() {
        let mut config = make_config();
        config.max_merged_constraints = 5; // very low limit

        let cm = ConstraintMerge::new(&config);
        // Build a deeply nested constraint
        let mut expr = sym_var("x");
        for i in 2..10 {
            expr = expr.and_expr(sym_var(&format!("x{}", i)));
        }
        let c1 = vec![path_constraint_from(expr.clone())];
        let c2 = vec![path_constraint_from(expr)];

        let result = cm.merge_constraints(&c1, &c2);
        assert!(result.is_err());
    }
}
