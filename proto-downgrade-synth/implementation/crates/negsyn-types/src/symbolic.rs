//! Symbolic execution types for protocol analysis.
//!
//! Provides symbolic values, path constraints, symbolic state management,
//! execution trees, and protocol-aware state merging.

use crate::error::NegSynthResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

// ── Symbolic Value ───────────────────────────────────────────────────────

/// A symbolic value in the symbolic execution engine.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolicValue {
    /// A concrete value.
    Concrete(ConcreteValue),
    /// A symbolic variable.
    Variable(SymVar),
    /// Binary operation.
    BinaryOp {
        op: BinOp,
        left: Box<SymbolicValue>,
        right: Box<SymbolicValue>,
    },
    /// Unary operation.
    UnaryOp {
        op: UnOp,
        operand: Box<SymbolicValue>,
    },
    /// If-then-else (conditional).
    Ite {
        condition: Box<SymbolicValue>,
        then_val: Box<SymbolicValue>,
        else_val: Box<SymbolicValue>,
    },
    /// Array/memory select: select(array, index).
    Select {
        array: Box<SymbolicValue>,
        index: Box<SymbolicValue>,
    },
    /// Array/memory store: store(array, index, value).
    Store {
        array: Box<SymbolicValue>,
        index: Box<SymbolicValue>,
        value: Box<SymbolicValue>,
    },
}

/// Concrete runtime values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConcreteValue {
    Bool(bool),
    Int(i64),
    BitVec { value: u64, width: u32 },
    Bytes(Vec<u8>),
}

/// A symbolic variable with name and type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymVar {
    pub name: String,
    pub sort: SymSort,
    pub generation: u32,
}

/// Sorts for symbolic values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymSort {
    Bool,
    Int,
    BitVec(u32),
    Array { index: Box<SymSort>, element: Box<SymSort> },
    Bytes,
}

/// Binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    // Arithmetic
    Add, Sub, Mul, Div, Rem,
    // Bitwise
    And, Or, Xor, Shl, Shr,
    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,
    // Logical
    LogicAnd, LogicOr, Implies,
    // Bitvector
    BvAdd, BvSub, BvMul, BvUdiv, BvSdiv,
    BvAnd, BvOr, BvXor, BvShl, BvLshr, BvAshr,
    BvUlt, BvUle, BvUgt, BvUge, BvSlt, BvSle, BvSgt, BvSge,
    Concat,
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
    BvNot,
    BvNeg,
    ZeroExt(u32),
    SignExt(u32),
    Extract { high: u32, low: u32 },
}

impl SymbolicValue {
    // ── Constructors ─────────────────────────────────────────────

    pub fn bool_const(v: bool) -> Self {
        SymbolicValue::Concrete(ConcreteValue::Bool(v))
    }

    pub fn int_const(v: i64) -> Self {
        SymbolicValue::Concrete(ConcreteValue::Int(v))
    }

    pub fn bv_const(value: u64, width: u32) -> Self {
        SymbolicValue::Concrete(ConcreteValue::BitVec { value, width })
    }

    pub fn var(name: impl Into<String>, sort: SymSort) -> Self {
        SymbolicValue::Variable(SymVar {
            name: name.into(),
            sort,
            generation: 0,
        })
    }

    pub fn binary(op: BinOp, left: SymbolicValue, right: SymbolicValue) -> Self {
        SymbolicValue::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn unary(op: UnOp, operand: SymbolicValue) -> Self {
        SymbolicValue::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    pub fn ite(cond: SymbolicValue, then_v: SymbolicValue, else_v: SymbolicValue) -> Self {
        SymbolicValue::Ite {
            condition: Box::new(cond),
            then_val: Box::new(then_v),
            else_val: Box::new(else_v),
        }
    }

    pub fn select(array: SymbolicValue, index: SymbolicValue) -> Self {
        SymbolicValue::Select {
            array: Box::new(array),
            index: Box::new(index),
        }
    }

    pub fn store(array: SymbolicValue, index: SymbolicValue, value: SymbolicValue) -> Self {
        SymbolicValue::Store {
            array: Box::new(array),
            index: Box::new(index),
            value: Box::new(value),
        }
    }

    /// Convenience method: conjunction of self and another.
    pub fn and_expr(self, other: SymbolicValue) -> Self {
        Self::binary(BinOp::LogicAnd, self, other)
    }

    /// Convenience method: disjunction of self and another.
    pub fn or_expr(self, other: SymbolicValue) -> Self {
        Self::binary(BinOp::LogicOr, self, other)
    }

    // ── Queries ──────────────────────────────────────────────────

    pub fn is_concrete(&self) -> bool {
        matches!(self, SymbolicValue::Concrete(_))
    }

    pub fn is_symbolic(&self) -> bool {
        !self.is_concrete()
    }

    pub fn depth(&self) -> usize {
        match self {
            SymbolicValue::Concrete(_) | SymbolicValue::Variable(_) => 0,
            SymbolicValue::BinaryOp { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
            SymbolicValue::UnaryOp { operand, .. } => 1 + operand.depth(),
            SymbolicValue::Ite {
                condition,
                then_val,
                else_val,
            } => {
                1 + condition
                    .depth()
                    .max(then_val.depth())
                    .max(else_val.depth())
            }
            SymbolicValue::Select { array, index } => {
                1 + array.depth().max(index.depth())
            }
            SymbolicValue::Store {
                array,
                index,
                value,
            } => 1 + array.depth().max(index.depth()).max(value.depth()),
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            SymbolicValue::Concrete(_) | SymbolicValue::Variable(_) => 1,
            SymbolicValue::BinaryOp { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
            SymbolicValue::UnaryOp { operand, .. } => 1 + operand.node_count(),
            SymbolicValue::Ite {
                condition,
                then_val,
                else_val,
            } => {
                1 + condition.node_count()
                    + then_val.node_count()
                    + else_val.node_count()
            }
            SymbolicValue::Select { array, index } => {
                1 + array.node_count() + index.node_count()
            }
            SymbolicValue::Store {
                array,
                index,
                value,
            } => 1 + array.node_count() + index.node_count() + value.node_count(),
        }
    }

    /// Collect all free variables in this expression.
    pub fn free_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            SymbolicValue::Variable(v) => {
                vars.insert(v.name.clone());
            }
            SymbolicValue::BinaryOp { left, right, .. } => {
                left.collect_vars(vars);
                right.collect_vars(vars);
            }
            SymbolicValue::UnaryOp { operand, .. } => operand.collect_vars(vars),
            SymbolicValue::Ite {
                condition,
                then_val,
                else_val,
            } => {
                condition.collect_vars(vars);
                then_val.collect_vars(vars);
                else_val.collect_vars(vars);
            }
            SymbolicValue::Select { array, index } => {
                array.collect_vars(vars);
                index.collect_vars(vars);
            }
            SymbolicValue::Store {
                array,
                index,
                value,
            } => {
                array.collect_vars(vars);
                index.collect_vars(vars);
                value.collect_vars(vars);
            }
            _ => {}
        }
    }

    /// Simplify the symbolic value using basic algebraic rules.
    pub fn simplify(&self) -> SymbolicValue {
        match self {
            SymbolicValue::BinaryOp { op, left, right } => {
                let sl = left.simplify();
                let sr = right.simplify();
                Self::simplify_binary(*op, sl, sr)
            }
            SymbolicValue::UnaryOp { op, operand } => {
                let so = operand.simplify();
                Self::simplify_unary(*op, so)
            }
            SymbolicValue::Ite {
                condition,
                then_val,
                else_val,
            } => {
                let sc = condition.simplify();
                let st = then_val.simplify();
                let se = else_val.simplify();
                // If condition is concrete, simplify
                if let SymbolicValue::Concrete(ConcreteValue::Bool(b)) = &sc {
                    if *b { st } else { se }
                } else if st == se {
                    st
                } else {
                    SymbolicValue::ite(sc, st, se)
                }
            }
            other => other.clone(),
        }
    }

    fn simplify_binary(op: BinOp, left: SymbolicValue, right: SymbolicValue) -> SymbolicValue {
        // Constant folding
        if let (
            SymbolicValue::Concrete(ConcreteValue::Int(a)),
            SymbolicValue::Concrete(ConcreteValue::Int(b)),
        ) = (&left, &right)
        {
            match op {
                BinOp::Add => return SymbolicValue::int_const(a.wrapping_add(*b)),
                BinOp::Sub => return SymbolicValue::int_const(a.wrapping_sub(*b)),
                BinOp::Mul => return SymbolicValue::int_const(a.wrapping_mul(*b)),
                BinOp::Eq => return SymbolicValue::bool_const(a == b),
                BinOp::Ne => return SymbolicValue::bool_const(a != b),
                BinOp::Lt => return SymbolicValue::bool_const(a < b),
                BinOp::Le => return SymbolicValue::bool_const(a <= b),
                BinOp::Gt => return SymbolicValue::bool_const(a > b),
                BinOp::Ge => return SymbolicValue::bool_const(a >= b),
                _ => {}
            }
        }

        // Boolean constant folding
        if let (
            SymbolicValue::Concrete(ConcreteValue::Bool(a)),
            SymbolicValue::Concrete(ConcreteValue::Bool(b)),
        ) = (&left, &right)
        {
            match op {
                BinOp::LogicAnd => return SymbolicValue::bool_const(*a && *b),
                BinOp::LogicOr => return SymbolicValue::bool_const(*a || *b),
                BinOp::Implies => return SymbolicValue::bool_const(!*a || *b),
                BinOp::Eq => return SymbolicValue::bool_const(a == b),
                _ => {}
            }
        }

        // Identity rules
        match op {
            BinOp::Add if right == SymbolicValue::int_const(0) => return left,
            BinOp::Add if left == SymbolicValue::int_const(0) => return right,
            BinOp::Mul if right == SymbolicValue::int_const(1) => return left,
            BinOp::Mul if left == SymbolicValue::int_const(1) => return right,
            BinOp::Mul if right == SymbolicValue::int_const(0) => {
                return SymbolicValue::int_const(0)
            }
            BinOp::Mul if left == SymbolicValue::int_const(0) => {
                return SymbolicValue::int_const(0)
            }
            BinOp::LogicAnd if left == SymbolicValue::bool_const(true) => return right,
            BinOp::LogicAnd if right == SymbolicValue::bool_const(true) => return left,
            BinOp::LogicAnd
                if left == SymbolicValue::bool_const(false)
                    || right == SymbolicValue::bool_const(false) =>
            {
                return SymbolicValue::bool_const(false)
            }
            BinOp::LogicOr if left == SymbolicValue::bool_const(false) => return right,
            BinOp::LogicOr if right == SymbolicValue::bool_const(false) => return left,
            BinOp::LogicOr
                if left == SymbolicValue::bool_const(true)
                    || right == SymbolicValue::bool_const(true) =>
            {
                return SymbolicValue::bool_const(true)
            }
            // x == x is true
            BinOp::Eq if left == right => return SymbolicValue::bool_const(true),
            _ => {}
        }

        SymbolicValue::binary(op, left, right)
    }

    fn simplify_unary(op: UnOp, operand: SymbolicValue) -> SymbolicValue {
        match (&op, &operand) {
            (UnOp::Not, SymbolicValue::Concrete(ConcreteValue::Bool(b))) => {
                SymbolicValue::bool_const(!b)
            }
            (UnOp::Neg, SymbolicValue::Concrete(ConcreteValue::Int(n))) => {
                SymbolicValue::int_const(-n)
            }
            // Double negation elimination
            (UnOp::Not, SymbolicValue::UnaryOp { op: UnOp::Not, operand: inner }) => {
                inner.as_ref().clone()
            }
            (UnOp::Neg, SymbolicValue::UnaryOp { op: UnOp::Neg, operand: inner }) => {
                inner.as_ref().clone()
            }
            _ => SymbolicValue::unary(op, operand),
        }
    }

    /// Substitute a variable with a value.
    pub fn substitute(&self, var_name: &str, replacement: &SymbolicValue) -> SymbolicValue {
        match self {
            SymbolicValue::Variable(v) if v.name == var_name => replacement.clone(),
            SymbolicValue::BinaryOp { op, left, right } => SymbolicValue::binary(
                *op,
                left.substitute(var_name, replacement),
                right.substitute(var_name, replacement),
            ),
            SymbolicValue::UnaryOp { op, operand } => {
                SymbolicValue::unary(*op, operand.substitute(var_name, replacement))
            }
            SymbolicValue::Ite {
                condition,
                then_val,
                else_val,
            } => SymbolicValue::ite(
                condition.substitute(var_name, replacement),
                then_val.substitute(var_name, replacement),
                else_val.substitute(var_name, replacement),
            ),
            SymbolicValue::Select { array, index } => SymbolicValue::select(
                array.substitute(var_name, replacement),
                index.substitute(var_name, replacement),
            ),
            SymbolicValue::Store {
                array,
                index,
                value,
            } => SymbolicValue::store(
                array.substitute(var_name, replacement),
                index.substitute(var_name, replacement),
                value.substitute(var_name, replacement),
            ),
            other => other.clone(),
        }
    }
}

impl fmt::Display for SymbolicValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicValue::Concrete(ConcreteValue::Bool(b)) => write!(f, "{}", b),
            SymbolicValue::Concrete(ConcreteValue::Int(n)) => write!(f, "{}", n),
            SymbolicValue::Concrete(ConcreteValue::BitVec { value, width }) => {
                write!(f, "#x{:0width$x}", value, width = (*width as usize + 3) / 4)
            }
            SymbolicValue::Concrete(ConcreteValue::Bytes(b)) => write!(f, "0x{}", hex::encode(b)),
            SymbolicValue::Variable(v) => write!(f, "{}_{}", v.name, v.generation),
            SymbolicValue::BinaryOp { op, left, right } => {
                write!(f, "({:?} {} {})", op, left, right)
            }
            SymbolicValue::UnaryOp { op, operand } => write!(f, "({:?} {})", op, operand),
            SymbolicValue::Ite {
                condition,
                then_val,
                else_val,
            } => write!(f, "(ite {} {} {})", condition, then_val, else_val),
            SymbolicValue::Select { array, index } => {
                write!(f, "(select {} {})", array, index)
            }
            SymbolicValue::Store {
                array,
                index,
                value,
            } => write!(f, "(store {} {} {})", array, index, value),
        }
    }
}

// ── Path Constraint ──────────────────────────────────────────────────────

/// A conjunction of symbolic conditions along an execution path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathConstraint {
    pub conditions: Vec<SymbolicValue>,
    /// The primary constraint expression (alias for the conjunction).
    pub constraint: Option<SymbolicValue>,
    /// Whether the constraint is negated.
    pub is_negated: bool,
    /// Parent path constraint ID (for tree structure).
    pub parent_id: Option<u64>,
}

impl PathConstraint {
    pub fn empty() -> Self {
        PathConstraint {
            conditions: Vec::new(),
            constraint: None,
            is_negated: false,
            parent_id: None,
        }
    }

    /// Create a new path constraint (alias for `empty()`).
    pub fn new() -> Self {
        Self::empty()
    }

    /// Add a condition to the path constraint.
    pub fn add(&mut self, condition: SymbolicValue) {
        self.conditions.push(condition);
    }

    /// Number of conditions.
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Conjoin all conditions into a single symbolic value.
    pub fn as_conjunction(&self) -> SymbolicValue {
        if self.conditions.is_empty() {
            return SymbolicValue::bool_const(true);
        }
        let mut result = self.conditions[0].clone();
        for cond in &self.conditions[1..] {
            result = SymbolicValue::binary(BinOp::LogicAnd, result, cond.clone());
        }
        result
    }

    /// Fork: create two path constraints for a branch.
    pub fn fork(&self, condition: &SymbolicValue) -> (PathConstraint, PathConstraint) {
        let mut true_branch = self.clone();
        true_branch.add(condition.clone());

        let mut false_branch = self.clone();
        false_branch.add(SymbolicValue::unary(UnOp::Not, condition.clone()));

        (true_branch, false_branch)
    }

    /// Simplify all conditions in the path constraint.
    pub fn simplify(&mut self) {
        self.conditions = self
            .conditions
            .iter()
            .map(|c| c.simplify())
            .filter(|c| *c != SymbolicValue::bool_const(true))
            .collect();
    }

    /// Check if the constraint is trivially unsatisfiable.
    pub fn is_trivially_unsat(&self) -> bool {
        self.conditions
            .iter()
            .any(|c| *c == SymbolicValue::bool_const(false))
    }

    /// All free variables across all conditions.
    pub fn free_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for cond in &self.conditions {
            vars.extend(cond.free_variables());
        }
        vars
    }

    /// Return the effective constraint (conjunction, possibly negated).
    pub fn effective_constraint(&self) -> SymbolicValue {
        let conj = self.as_conjunction();
        if self.is_negated {
            SymbolicValue::unary(UnOp::Not, conj)
        } else {
            conj
        }
    }
}

impl fmt::Display for PathConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.conditions.is_empty() {
            write!(f, "true")
        } else {
            let parts: Vec<String> = self.conditions.iter().map(|c| format!("{}", c)).collect();
            write!(f, "(∧ {})", parts.join(" "))
        }
    }
}

// ── Symbolic Memory ──────────────────────────────────────────────────────

/// A region of symbolic memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub name: String,
    pub base_address: u64,
    pub size: u64,
    pub content: BTreeMap<u64, SymbolicValue>,
    pub permissions: MemoryPermissions,
}

impl MemoryRegion {
    /// Alias for `base_address`.
    pub fn base(&self) -> u64 {
        self.base_address
    }

    /// Alias for `content`.
    pub fn contents(&self) -> &BTreeMap<u64, SymbolicValue> {
        &self.content
    }

    /// Whether this region is writable.
    pub fn is_writable(&self) -> bool {
        self.permissions.write
    }

    /// Write a value (alias for `store`).
    pub fn write(&mut self, addr: u64, value: SymbolicValue) -> bool {
        self.store(addr, value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl MemoryPermissions {
    pub fn rwx() -> Self { MemoryPermissions { read: true, write: true, execute: true } }
    pub fn rw() -> Self { MemoryPermissions { read: true, write: true, execute: false } }
    pub fn ro() -> Self { MemoryPermissions { read: true, write: false, execute: false } }
    pub fn rx() -> Self { MemoryPermissions { read: true, write: false, execute: true } }
}

impl MemoryRegion {
    pub fn new(name: impl Into<String>, base: u64, size: u64, perms: MemoryPermissions) -> Self {
        MemoryRegion {
            name: name.into(),
            base_address: base,
            size,
            content: BTreeMap::new(),
            permissions: perms,
        }
    }

    pub fn contains_address(&self, addr: u64) -> bool {
        addr >= self.base_address && addr < self.base_address + self.size
    }

    pub fn load(&self, addr: u64, width: u32) -> Option<SymbolicValue> {
        if !self.contains_address(addr) || !self.permissions.read {
            return None;
        }
        self.content.get(&addr).cloned().or_else(|| {
            Some(SymbolicValue::var(
                format!("mem_{}_{:#x}", self.name, addr),
                SymSort::BitVec(width),
            ))
        })
    }

    pub fn store(&mut self, addr: u64, value: SymbolicValue) -> bool {
        if !self.contains_address(addr) || !self.permissions.write {
            return false;
        }
        self.content.insert(addr, value);
        true
    }
}

/// Symbolic memory manager with multiple regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicMemory {
    pub regions: Vec<MemoryRegion>,
    write_count: u64,
}

impl SymbolicMemory {
    pub fn new() -> Self {
        SymbolicMemory {
            regions: Vec::new(),
            write_count: 0,
        }
    }

    pub fn add_region(&mut self, region: MemoryRegion) {
        self.regions.push(region);
    }

    /// Find the region containing an address.
    fn find_region(&self, addr: u64) -> Option<usize> {
        self.regions.iter().position(|r| r.contains_address(addr))
    }

    fn find_region_mut(&mut self, addr: u64) -> Option<&mut MemoryRegion> {
        self.regions.iter_mut().find(|r| r.contains_address(addr))
    }

    /// Symbolic load from memory.
    pub fn load(&self, addr: u64, width: u32) -> SymbolicValue {
        if let Some(idx) = self.find_region(addr) {
            self.regions[idx].load(addr, width).unwrap_or_else(|| {
                SymbolicValue::var(format!("mem_{:#x}", addr), SymSort::BitVec(width))
            })
        } else {
            SymbolicValue::var(format!("unmapped_{:#x}", addr), SymSort::BitVec(width))
        }
    }

    /// Symbolic store to memory.
    pub fn store(&mut self, addr: u64, value: SymbolicValue) -> bool {
        if let Some(region) = self.find_region_mut(addr) {
            let ok = region.store(addr, value);
            if ok {
                self.write_count += 1;
            }
            ok
        } else {
            false
        }
    }

    /// Total number of writes performed.
    pub fn write_count(&self) -> u64 {
        self.write_count
    }

    /// Concrete byte values in a region.
    pub fn concrete_bytes(&self, addr: u64, len: usize) -> Vec<Option<u8>> {
        (0..len)
            .map(|i| {
                let a = addr + i as u64;
                if let Some(idx) = self.find_region(a) {
                    if let Some(SymbolicValue::Concrete(ConcreteValue::BitVec { value, .. })) =
                        self.regions[idx].content.get(&a)
                    {
                        Some(*value as u8)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return all region names (keys).
    pub fn keys(&self) -> Vec<&str> {
        self.regions.iter().map(|r| r.name.as_str()).collect()
    }
}

impl Default for SymbolicMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ── Symbolic State ───────────────────────────────────────────────────────

/// A symbolic execution state: program location + symbolic store + constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicState {
    pub id: u64,
    pub program_counter: u64,
    pub registers: HashMap<String, SymbolicValue>,
    pub memory: SymbolicMemory,
    pub path_constraint: PathConstraint,
    pub depth: u32,
    pub is_feasible: bool,
    /// Protocol negotiation state associated with this symbolic state.
    pub negotiation: crate::protocol::NegotiationState,
    /// Alias for path_constraint used by downstream crates.
    pub pc: PathConstraint,
    /// Parent state id (for tree tracking).
    pub parent_id: Option<u64>,
    /// Alias for path_constraint conditions used by downstream crates.
    pub constraints: Vec<SymbolicValue>,
}

impl SymbolicState {
    pub fn new(id: u64, pc: u64) -> Self {
        let path_constraint = PathConstraint::empty();
        SymbolicState {
            id,
            program_counter: pc,
            registers: HashMap::new(),
            memory: SymbolicMemory::new(),
            constraints: Vec::new(),
            pc: path_constraint.clone(),
            path_constraint,
            depth: 0,
            is_feasible: true,
            negotiation: crate::protocol::NegotiationState::initial(),
            parent_id: None,
        }
    }

    pub fn get_register(&self, name: &str) -> SymbolicValue {
        self.registers
            .get(name)
            .cloned()
            .unwrap_or_else(|| SymbolicValue::var(name, SymSort::BitVec(64)))
    }

    pub fn set_register(&mut self, name: impl Into<String>, value: SymbolicValue) {
        self.registers.insert(name.into(), value);
    }

    /// Fork this state into two branches on a symbolic condition.
    pub fn fork(&self, condition: &SymbolicValue, next_id: &mut u64) -> (SymbolicState, SymbolicState) {
        let (true_pc, false_pc) = self.path_constraint.fork(condition);

        let mut true_state = self.clone();
        true_state.id = *next_id;
        true_state.parent_id = Some(self.id);
        *next_id += 1;
        true_state.path_constraint = true_pc.clone();
        true_state.pc = true_pc.clone();
        true_state.constraints = true_pc.conditions.clone();
        true_state.depth += 1;

        let mut false_state = self.clone();
        false_state.id = *next_id;
        false_state.parent_id = Some(self.id);
        *next_id += 1;
        false_state.path_constraint = false_pc.clone();
        false_state.pc = false_pc.clone();
        false_state.constraints = false_pc.conditions.clone();
        false_state.depth += 1;

        (true_state, false_state)
    }

    /// Number of path constraints.
    pub fn constraint_count(&self) -> usize {
        self.path_constraint.len()
    }

    /// All free variables in this state (registers + constraints).
    pub fn free_variables(&self) -> HashSet<String> {
        let mut vars = self.path_constraint.free_variables();
        for val in self.registers.values() {
            vars.extend(val.free_variables());
        }
        vars
    }
}

// ── Execution Tree ───────────────────────────────────────────────────────

/// Node in the symbolic execution tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionNode {
    pub state_id: u64,
    pub program_counter: u64,
    pub children: Vec<ExecutionEdge>,
    pub status: NodeStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEdge {
    pub condition: Option<SymbolicValue>,
    pub child_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Completed,
    Infeasible,
    Merged,
    DepthExceeded,
    Error,
}

/// A tree tracking the entire symbolic execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTree {
    pub nodes: HashMap<u64, ExecutionNode>,
    pub root: u64,
    next_id: u64,
}

impl ExecutionTree {
    pub fn new(root_pc: u64) -> Self {
        let root_node = ExecutionNode {
            state_id: 0,
            program_counter: root_pc,
            children: Vec::new(),
            status: NodeStatus::Active,
        };
        let mut nodes = HashMap::new();
        nodes.insert(0, root_node);
        ExecutionTree {
            nodes,
            root: 0,
            next_id: 1,
        }
    }

    /// Add a child node.
    pub fn add_child(
        &mut self,
        parent_id: u64,
        pc: u64,
        condition: Option<SymbolicValue>,
    ) -> u64 {
        let child_id = self.next_id;
        self.next_id += 1;

        let child = ExecutionNode {
            state_id: child_id,
            program_counter: pc,
            children: Vec::new(),
            status: NodeStatus::Active,
        };
        self.nodes.insert(child_id, child);

        if let Some(parent) = self.nodes.get_mut(&parent_id) {
            parent.children.push(ExecutionEdge {
                condition,
                child_id,
            });
        }

        child_id
    }

    /// Set the status of a node.
    pub fn set_status(&mut self, node_id: u64, status: NodeStatus) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.status = status;
        }
    }

    /// Count nodes by status.
    pub fn count_by_status(&self, status: NodeStatus) -> usize {
        self.nodes.values().filter(|n| n.status == status).count()
    }

    /// Total nodes in the tree.
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Active (unexplored) leaf nodes.
    pub fn active_leaves(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.status == NodeStatus::Active && n.children.is_empty())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        self.depth_from(self.root)
    }

    fn depth_from(&self, node_id: u64) -> usize {
        if let Some(node) = self.nodes.get(&node_id) {
            if node.children.is_empty() {
                0
            } else {
                1 + node
                    .children
                    .iter()
                    .map(|e| self.depth_from(e.child_id))
                    .max()
                    .unwrap_or(0)
            }
        } else {
            0
        }
    }

    /// Collect path from root to a given node.
    pub fn path_to(&self, target_id: u64) -> Option<Vec<u64>> {
        let mut path = Vec::new();
        if self.find_path(self.root, target_id, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    fn find_path(&self, current: u64, target: u64, path: &mut Vec<u64>) -> bool {
        path.push(current);
        if current == target {
            return true;
        }
        if let Some(node) = self.nodes.get(&current) {
            for edge in &node.children {
                if self.find_path(edge.child_id, target, path) {
                    return true;
                }
            }
        }
        path.pop();
        false
    }
}

// ── Mergeable State Trait ────────────────────────────────────────────────

/// Trait for protocol-aware state merging.
pub trait MergeableState: Clone {
    /// Whether two states are merge-compatible (same program point, compatible types).
    fn is_merge_compatible(&self, other: &Self) -> bool;

    /// Merge two states into one, introducing symbolic choices.
    fn merge(&self, other: &Self) -> Option<Self>;

    /// Widening operator for loop convergence.
    fn widen(&self, other: &Self) -> Option<Self>;

    /// A key that groups merge-compatible states.
    fn merge_key(&self) -> u64;
}

impl MergeableState for SymbolicState {
    fn is_merge_compatible(&self, other: &Self) -> bool {
        self.program_counter == other.program_counter
    }

    fn merge(&self, other: &Self) -> Option<Self> {
        if !self.is_merge_compatible(other) {
            return None;
        }

        let merge_cond = SymbolicValue::var(
            format!("merge_{}_{}", self.id, other.id),
            SymSort::Bool,
        );

        let mut merged = self.clone();
        merged.depth = self.depth.max(other.depth);

        // Merge registers: create ITE for differing values
        let all_regs: HashSet<String> = self
            .registers
            .keys()
            .chain(other.registers.keys())
            .cloned()
            .collect();

        for reg in all_regs {
            let v1 = self.get_register(&reg);
            let v2 = other.get_register(&reg);
            if v1 != v2 {
                let merged_val = SymbolicValue::ite(merge_cond.clone(), v1, v2);
                merged.set_register(reg, merged_val);
            }
        }

        // Merge path constraints: disjunction
        let c1 = self.path_constraint.as_conjunction();
        let c2 = other.path_constraint.as_conjunction();
        merged.path_constraint = PathConstraint::empty();
        merged.path_constraint.add(SymbolicValue::binary(BinOp::LogicOr, c1, c2));

        Some(merged)
    }

    fn widen(&self, other: &Self) -> Option<Self> {
        // Widening drops constraints that are not common
        if !self.is_merge_compatible(other) {
            return None;
        }
        let mut widened = self.merge(other)?;
        // Drop all path constraints as widening approximation
        widened.path_constraint = PathConstraint::empty();
        Some(widened)
    }

    fn merge_key(&self) -> u64 {
        self.program_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_value_constructors() {
        let b = SymbolicValue::bool_const(true);
        assert!(b.is_concrete());

        let v = SymbolicValue::var("x", SymSort::BitVec(32));
        assert!(v.is_symbolic());
    }

    #[test]
    fn test_symbolic_depth_and_size() {
        let leaf = SymbolicValue::int_const(42);
        assert_eq!(leaf.depth(), 0);
        assert_eq!(leaf.node_count(), 1);

        let expr = SymbolicValue::binary(
            BinOp::Add,
            SymbolicValue::var("x", SymSort::Int),
            SymbolicValue::int_const(1),
        );
        assert_eq!(expr.depth(), 1);
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_constant_folding() {
        let expr = SymbolicValue::binary(
            BinOp::Add,
            SymbolicValue::int_const(3),
            SymbolicValue::int_const(4),
        );
        let simplified = expr.simplify();
        assert_eq!(simplified, SymbolicValue::int_const(7));
    }

    #[test]
    fn test_identity_simplification() {
        let x = SymbolicValue::var("x", SymSort::Int);
        let expr = SymbolicValue::binary(BinOp::Add, x.clone(), SymbolicValue::int_const(0));
        let simplified = expr.simplify();
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_boolean_simplification() {
        let expr = SymbolicValue::binary(
            BinOp::LogicAnd,
            SymbolicValue::bool_const(true),
            SymbolicValue::var("p", SymSort::Bool),
        );
        let simplified = expr.simplify();
        assert_eq!(simplified, SymbolicValue::var("p", SymSort::Bool));
    }

    #[test]
    fn test_ite_simplification() {
        let expr = SymbolicValue::ite(
            SymbolicValue::bool_const(true),
            SymbolicValue::int_const(1),
            SymbolicValue::int_const(2),
        );
        assert_eq!(expr.simplify(), SymbolicValue::int_const(1));

        let x = SymbolicValue::var("x", SymSort::Int);
        let same = SymbolicValue::ite(
            SymbolicValue::var("c", SymSort::Bool),
            x.clone(),
            x.clone(),
        );
        assert_eq!(same.simplify(), x);
    }

    #[test]
    fn test_double_negation() {
        let x = SymbolicValue::var("x", SymSort::Bool);
        let dbl_neg = SymbolicValue::unary(UnOp::Not, SymbolicValue::unary(UnOp::Not, x.clone()));
        assert_eq!(dbl_neg.simplify(), x);
    }

    #[test]
    fn test_free_variables() {
        let expr = SymbolicValue::binary(
            BinOp::Add,
            SymbolicValue::var("x", SymSort::Int),
            SymbolicValue::binary(
                BinOp::Mul,
                SymbolicValue::var("y", SymSort::Int),
                SymbolicValue::int_const(2),
            ),
        );
        let vars = expr.free_variables();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_substitution() {
        let expr = SymbolicValue::binary(
            BinOp::Add,
            SymbolicValue::var("x", SymSort::Int),
            SymbolicValue::int_const(1),
        );
        let result = expr.substitute("x", &SymbolicValue::int_const(5));
        let simplified = result.simplify();
        assert_eq!(simplified, SymbolicValue::int_const(6));
    }

    #[test]
    fn test_path_constraint() {
        let mut pc = PathConstraint::empty();
        assert!(pc.is_empty());

        pc.add(SymbolicValue::bool_const(true));
        pc.add(SymbolicValue::var("p", SymSort::Bool));
        assert_eq!(pc.len(), 2);

        pc.simplify();
        // `true` should be removed
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn test_path_constraint_fork() {
        let pc = PathConstraint::empty();
        let cond = SymbolicValue::var("c", SymSort::Bool);
        let (true_branch, false_branch) = pc.fork(&cond);

        assert_eq!(true_branch.len(), 1);
        assert_eq!(false_branch.len(), 1);
    }

    #[test]
    fn test_trivially_unsat() {
        let mut pc = PathConstraint::empty();
        pc.add(SymbolicValue::bool_const(false));
        assert!(pc.is_trivially_unsat());

        let pc2 = PathConstraint::empty();
        assert!(!pc2.is_trivially_unsat());
    }

    #[test]
    fn test_memory_region() {
        let mut region = MemoryRegion::new("stack", 0x1000, 0x1000, MemoryPermissions::rw());
        assert!(region.contains_address(0x1000));
        assert!(!region.contains_address(0x2000));

        assert!(region.store(0x1000, SymbolicValue::bv_const(42, 32)));
        let loaded = region.load(0x1000, 32);
        assert_eq!(loaded, Some(SymbolicValue::bv_const(42, 32)));
    }

    #[test]
    fn test_symbolic_memory() {
        let mut mem = SymbolicMemory::new();
        mem.add_region(MemoryRegion::new("heap", 0x4000, 0x4000, MemoryPermissions::rw()));

        assert!(mem.store(0x4000, SymbolicValue::bv_const(0xFF, 8)));
        let val = mem.load(0x4000, 8);
        assert_eq!(val, SymbolicValue::bv_const(0xFF, 8));

        // Unmapped address returns symbolic variable
        let unmap = mem.load(0x9999, 32);
        assert!(unmap.is_symbolic());
    }

    #[test]
    fn test_symbolic_state_fork() {
        let state = SymbolicState::new(0, 0x1000);
        let cond = SymbolicValue::var("branch", SymSort::Bool);
        let mut next_id = 1u64;
        let (t, f) = state.fork(&cond, &mut next_id);

        assert_ne!(t.id, f.id);
        assert_eq!(t.program_counter, f.program_counter);
        assert_eq!(t.path_constraint.len(), 1);
        assert_eq!(f.path_constraint.len(), 1);
    }

    #[test]
    fn test_execution_tree() {
        let mut tree = ExecutionTree::new(0x1000);
        assert_eq!(tree.total_nodes(), 1);

        let c1 = tree.add_child(0, 0x1004, None);
        let c2 = tree.add_child(0, 0x1008, Some(SymbolicValue::var("c", SymSort::Bool)));
        assert_eq!(tree.total_nodes(), 3);

        tree.set_status(c1, NodeStatus::Completed);
        assert_eq!(tree.count_by_status(NodeStatus::Completed), 1);
        assert_eq!(tree.count_by_status(NodeStatus::Active), 2);

        let path = tree.path_to(c2);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![0, c2]);
    }

    #[test]
    fn test_mergeable_state() {
        let mut s1 = SymbolicState::new(0, 0x1000);
        s1.set_register("rax", SymbolicValue::int_const(1));

        let mut s2 = SymbolicState::new(1, 0x1000);
        s2.set_register("rax", SymbolicValue::int_const(2));

        assert!(s1.is_merge_compatible(&s2));
        let merged = s1.merge(&s2);
        assert!(merged.is_some());
        let m = merged.unwrap();
        let rax = m.get_register("rax");
        assert!(rax.is_symbolic()); // Should be an ITE
    }

    #[test]
    fn test_non_compatible_merge() {
        let s1 = SymbolicState::new(0, 0x1000);
        let s2 = SymbolicState::new(1, 0x2000);
        assert!(!s1.is_merge_compatible(&s2));
        assert!(s1.merge(&s2).is_none());
    }
}
