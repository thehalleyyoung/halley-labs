//! SMT expression AST with hash-consing, builder methods, and SMT-LIB2 display.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Sorts
// ---------------------------------------------------------------------------

/// SMT sort (type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sort {
    Bool,
    BitVec(u32),
    Int,
    Real,
    Array(Box<Sort>, Box<Sort>),
    /// Abstract sort representing a cache state snapshot.
    CacheState,
    /// User-declared uninterpreted sort.
    Uninterpreted(String),
}

impl Sort {
    pub fn bv(width: u32) -> Self {
        Sort::BitVec(width)
    }

    pub fn array(key: Sort, val: Sort) -> Self {
        Sort::Array(Box::new(key), Box::new(val))
    }

    /// Width in bits if this is a bitvector sort, else `None`.
    pub fn bv_width(&self) -> Option<u32> {
        match self {
            Sort::BitVec(w) => Some(*w),
            _ => None,
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool)
    }

    pub fn is_bitvec(&self) -> bool {
        matches!(self, Sort::BitVec(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Sort::Int)
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real)
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Sort::Array(_, _))
    }

    /// Return the element sorts for an array sort.
    pub fn array_sorts(&self) -> Option<(&Sort, &Sort)> {
        match self {
            Sort::Array(k, v) => Some((k, v)),
            _ => None,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool => write!(f, "Bool"),
            Sort::BitVec(w) => write!(f, "(_ BitVec {})", w),
            Sort::Int => write!(f, "Int"),
            Sort::Real => write!(f, "Real"),
            Sort::Array(k, v) => write!(f, "(Array {} {})", k, v),
            Sort::CacheState => write!(f, "CacheState"),
            Sort::Uninterpreted(name) => write!(f, "{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A concrete value returned from a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    BitVec { width: u32, value: u64 },
    Int(i64),
    Real(f64),
    Uninterpreted(String),
}

impl Value {
    pub fn bv(width: u32, value: u64) -> Self {
        Value::BitVec { width, value }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::BitVec { value, .. } => Some(*value),
            Value::Int(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::BitVec { value, .. } => Some(*value as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Real(r) => Some(*r),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::BitVec { width, value } => {
                write!(f, "#b")?;
                for i in (0..*width).rev() {
                    write!(f, "{}", (value >> i) & 1)?;
                }
                Ok(())
            }
            Value::Int(v) => write!(f, "{}", v),
            Value::Real(v) => write!(f, "{}", v),
            Value::Uninterpreted(s) => write!(f, "{}", s),
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Bool(b) => b.hash(state),
            Value::BitVec { width, value } => {
                width.hash(state);
                value.hash(state);
            }
            Value::Int(v) => v.hash(state),
            Value::Real(v) => v.to_bits().hash(state),
            Value::Uninterpreted(s) => s.hash(state),
        }
    }
}

// ---------------------------------------------------------------------------
// ExprId – lightweight handle into ExprPool
// ---------------------------------------------------------------------------

/// Opaque identifier for an expression in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExprId(pub u64);

impl ExprId {
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// QualifiedVar – for quantified binders
// ---------------------------------------------------------------------------

/// A variable with its sort for use in quantified expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SortedVar {
    pub name: String,
    pub sort: Sort,
}

impl SortedVar {
    pub fn new(name: impl Into<String>, sort: Sort) -> Self {
        Self {
            name: name.into(),
            sort,
        }
    }
}

impl fmt::Display for SortedVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {})", self.name, self.sort)
    }
}

// ---------------------------------------------------------------------------
// Expr – the main AST
// ---------------------------------------------------------------------------

/// SMT expression node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expr {
    // Literals / variables
    Const(Value),
    Var(String, Sort),

    // Boolean connectives
    Not(ExprId),
    And(Vec<ExprId>),
    Or(Vec<ExprId>),
    Implies(ExprId, ExprId),
    Iff(ExprId, ExprId),

    // Core
    Eq(ExprId, ExprId),
    Distinct(Vec<ExprId>),
    Ite(ExprId, ExprId, ExprId),

    // Bitvector arithmetic
    BvAdd(ExprId, ExprId),
    BvSub(ExprId, ExprId),
    BvMul(ExprId, ExprId),
    BvAnd(ExprId, ExprId),
    BvOr(ExprId, ExprId),
    BvXor(ExprId, ExprId),
    BvNot(ExprId),
    BvNeg(ExprId),
    BvShl(ExprId, ExprId),
    BvLshr(ExprId, ExprId),
    BvAshr(ExprId, ExprId),

    // Bitvector comparison
    BvUlt(ExprId, ExprId),
    BvUle(ExprId, ExprId),
    BvUgt(ExprId, ExprId),
    BvUge(ExprId, ExprId),
    BvSlt(ExprId, ExprId),
    BvSle(ExprId, ExprId),

    // Bitvector manipulation
    BvExtract(u32, u32, ExprId),
    BvConcat(ExprId, ExprId),
    BvZext(u32, ExprId),
    BvSext(u32, ExprId),

    // Array theory
    Select(ExprId, ExprId),
    Store(ExprId, ExprId, ExprId),

    // Integer arithmetic
    IntAdd(ExprId, ExprId),
    IntSub(ExprId, ExprId),
    IntMul(ExprId, ExprId),
    IntDiv(ExprId, ExprId),
    IntMod(ExprId, ExprId),
    IntLe(ExprId, ExprId),
    IntLt(ExprId, ExprId),
    IntGe(ExprId, ExprId),
    IntGt(ExprId, ExprId),
    IntNeg(ExprId),

    // Real arithmetic
    RealAdd(ExprId, ExprId),
    RealSub(ExprId, ExprId),
    RealMul(ExprId, ExprId),
    RealDiv(ExprId, ExprId),
    RealLe(ExprId, ExprId),
    RealLt(ExprId, ExprId),
    RealGe(ExprId, ExprId),
    RealNeg(ExprId),

    // Quantifiers
    Forall(Vec<SortedVar>, ExprId),
    Exists(Vec<SortedVar>, ExprId),

    // Let bindings
    Let(Vec<(String, ExprId)>, ExprId),

    // Uninterpreted function application
    Apply(String, Vec<ExprId>),
}

impl Expr {
    /// Recursively collect all child ExprIds.
    pub fn children(&self) -> Vec<ExprId> {
        match self {
            Expr::Const(_) | Expr::Var(_, _) => vec![],
            Expr::Not(a)
            | Expr::BvNot(a)
            | Expr::BvNeg(a)
            | Expr::IntNeg(a)
            | Expr::RealNeg(a) => vec![*a],
            Expr::BvExtract(_, _, a) | Expr::BvZext(_, a) | Expr::BvSext(_, a) => vec![*a],
            Expr::And(v) | Expr::Or(v) | Expr::Distinct(v) => v.clone(),
            Expr::Implies(a, b)
            | Expr::Iff(a, b)
            | Expr::Eq(a, b)
            | Expr::BvAdd(a, b)
            | Expr::BvSub(a, b)
            | Expr::BvMul(a, b)
            | Expr::BvAnd(a, b)
            | Expr::BvOr(a, b)
            | Expr::BvXor(a, b)
            | Expr::BvShl(a, b)
            | Expr::BvLshr(a, b)
            | Expr::BvAshr(a, b)
            | Expr::BvUlt(a, b)
            | Expr::BvUle(a, b)
            | Expr::BvUgt(a, b)
            | Expr::BvUge(a, b)
            | Expr::BvSlt(a, b)
            | Expr::BvSle(a, b)
            | Expr::BvConcat(a, b)
            | Expr::Select(a, b)
            | Expr::IntAdd(a, b)
            | Expr::IntSub(a, b)
            | Expr::IntMul(a, b)
            | Expr::IntDiv(a, b)
            | Expr::IntMod(a, b)
            | Expr::IntLe(a, b)
            | Expr::IntLt(a, b)
            | Expr::IntGe(a, b)
            | Expr::IntGt(a, b)
            | Expr::RealAdd(a, b)
            | Expr::RealSub(a, b)
            | Expr::RealMul(a, b)
            | Expr::RealDiv(a, b)
            | Expr::RealLe(a, b)
            | Expr::RealLt(a, b)
            | Expr::RealGe(a, b) => vec![*a, *b],
            Expr::Ite(c, t, e) | Expr::Store(c, t, e) => vec![*c, *t, *e],
            Expr::Forall(_, body) | Expr::Exists(_, body) => vec![*body],
            Expr::Let(bindings, body) => {
                let mut ids: Vec<ExprId> = bindings.iter().map(|(_, id)| *id).collect();
                ids.push(*body);
                ids
            }
            Expr::Apply(_, args) => args.clone(),
        }
    }

    /// Is this a leaf node (no children)?
    pub fn is_leaf(&self) -> bool {
        matches!(self, Expr::Const(_) | Expr::Var(_, _))
    }

    /// Is this a boolean-sorted expression?
    pub fn is_boolean(&self) -> bool {
        matches!(
            self,
            Expr::Const(Value::Bool(_))
                | Expr::Not(_)
                | Expr::And(_)
                | Expr::Or(_)
                | Expr::Implies(_, _)
                | Expr::Iff(_, _)
                | Expr::Eq(_, _)
                | Expr::Distinct(_)
                | Expr::BvUlt(_, _)
                | Expr::BvUle(_, _)
                | Expr::BvUgt(_, _)
                | Expr::BvUge(_, _)
                | Expr::BvSlt(_, _)
                | Expr::BvSle(_, _)
                | Expr::IntLe(_, _)
                | Expr::IntLt(_, _)
                | Expr::IntGe(_, _)
                | Expr::IntGt(_, _)
                | Expr::RealLe(_, _)
                | Expr::RealLt(_, _)
                | Expr::RealGe(_, _)
                | Expr::Forall(_, _)
                | Expr::Exists(_, _)
        )
    }
}

// ---------------------------------------------------------------------------
// SMT-LIB2 display for Expr
// ---------------------------------------------------------------------------

/// Helper that recursively renders an expression given an `ExprPool`.
pub struct ExprDisplay<'a> {
    pub pool: &'a ExprPool,
    pub id: ExprId,
}

impl<'a> fmt::Display for ExprDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let expr = &self.pool.exprs[&self.id];
        match expr {
            Expr::Const(Value::Bool(true)) => write!(f, "true"),
            Expr::Const(Value::Bool(false)) => write!(f, "false"),
            Expr::Const(Value::BitVec { width, value }) => {
                write!(f, "(_ bv{} {})", value, width)
            }
            Expr::Const(Value::Int(v)) => {
                if *v < 0 {
                    write!(f, "(- {})", -v)
                } else {
                    write!(f, "{}", v)
                }
            }
            Expr::Const(Value::Real(v)) => write!(f, "{:.6}", v),
            Expr::Const(Value::Uninterpreted(s)) => write!(f, "{}", s),
            Expr::Var(name, _) => write!(f, "{}", name),
            Expr::Not(a) => write!(f, "(not {})", self.pool.display(*a)),
            Expr::And(args) => {
                if args.is_empty() {
                    write!(f, "true")
                } else if args.len() == 1 {
                    write!(f, "{}", self.pool.display(args[0]))
                } else {
                    write!(f, "(and")?;
                    for a in args {
                        write!(f, " {}", self.pool.display(*a))?;
                    }
                    write!(f, ")")
                }
            }
            Expr::Or(args) => {
                if args.is_empty() {
                    write!(f, "false")
                } else if args.len() == 1 {
                    write!(f, "{}", self.pool.display(args[0]))
                } else {
                    write!(f, "(or")?;
                    for a in args {
                        write!(f, " {}", self.pool.display(*a))?;
                    }
                    write!(f, ")")
                }
            }
            Expr::Implies(a, b) => {
                write!(f, "(=> {} {})", self.pool.display(*a), self.pool.display(*b))
            }
            Expr::Iff(a, b) => {
                write!(f, "(= {} {})", self.pool.display(*a), self.pool.display(*b))
            }
            Expr::Eq(a, b) => {
                write!(f, "(= {} {})", self.pool.display(*a), self.pool.display(*b))
            }
            Expr::Distinct(args) => {
                write!(f, "(distinct")?;
                for a in args {
                    write!(f, " {}", self.pool.display(*a))?;
                }
                write!(f, ")")
            }
            Expr::Ite(c, t, e) => {
                write!(
                    f,
                    "(ite {} {} {})",
                    self.pool.display(*c),
                    self.pool.display(*t),
                    self.pool.display(*e)
                )
            }
            Expr::BvAdd(a, b) => write!(f, "(bvadd {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvSub(a, b) => write!(f, "(bvsub {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvMul(a, b) => write!(f, "(bvmul {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvAnd(a, b) => write!(f, "(bvand {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvOr(a, b) => write!(f, "(bvor {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvXor(a, b) => write!(f, "(bvxor {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvNot(a) => write!(f, "(bvnot {})", self.pool.display(*a)),
            Expr::BvNeg(a) => write!(f, "(bvneg {})", self.pool.display(*a)),
            Expr::BvShl(a, b) => write!(f, "(bvshl {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvLshr(a, b) => write!(f, "(bvlshr {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvAshr(a, b) => write!(f, "(bvashr {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvUlt(a, b) => write!(f, "(bvult {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvUle(a, b) => write!(f, "(bvule {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvUgt(a, b) => write!(f, "(bvugt {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvUge(a, b) => write!(f, "(bvuge {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvSlt(a, b) => write!(f, "(bvslt {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvSle(a, b) => write!(f, "(bvsle {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::BvExtract(hi, lo, a) => {
                write!(f, "((_ extract {} {}) {})", hi, lo, self.pool.display(*a))
            }
            Expr::BvConcat(a, b) => {
                write!(f, "(concat {} {})", self.pool.display(*a), self.pool.display(*b))
            }
            Expr::BvZext(n, a) => {
                write!(f, "((_ zero_extend {}) {})", n, self.pool.display(*a))
            }
            Expr::BvSext(n, a) => {
                write!(f, "((_ sign_extend {}) {})", n, self.pool.display(*a))
            }
            Expr::Select(arr, idx) => {
                write!(f, "(select {} {})", self.pool.display(*arr), self.pool.display(*idx))
            }
            Expr::Store(arr, idx, val) => {
                write!(
                    f,
                    "(store {} {} {})",
                    self.pool.display(*arr),
                    self.pool.display(*idx),
                    self.pool.display(*val)
                )
            }
            Expr::IntAdd(a, b) => write!(f, "(+ {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntSub(a, b) => write!(f, "(- {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntMul(a, b) => write!(f, "(* {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntDiv(a, b) => write!(f, "(div {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntMod(a, b) => write!(f, "(mod {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntLe(a, b) => write!(f, "(<= {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntLt(a, b) => write!(f, "(< {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntGe(a, b) => write!(f, "(>= {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntGt(a, b) => write!(f, "(> {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::IntNeg(a) => write!(f, "(- {})", self.pool.display(*a)),
            Expr::RealAdd(a, b) => write!(f, "(+ {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealSub(a, b) => write!(f, "(- {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealMul(a, b) => write!(f, "(* {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealDiv(a, b) => write!(f, "(/ {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealLe(a, b) => write!(f, "(<= {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealLt(a, b) => write!(f, "(< {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealGe(a, b) => write!(f, "(>= {} {})", self.pool.display(*a), self.pool.display(*b)),
            Expr::RealNeg(a) => write!(f, "(- {})", self.pool.display(*a)),
            Expr::Forall(vars, body) => {
                write!(f, "(forall (")?;
                for (i, v) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ") {})", self.pool.display(*body))
            }
            Expr::Exists(vars, body) => {
                write!(f, "(exists (")?;
                for (i, v) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ") {})", self.pool.display(*body))
            }
            Expr::Let(bindings, body) => {
                write!(f, "(let (")?;
                for (i, (name, val)) in bindings.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", name, self.pool.display(*val))?;
                }
                write!(f, ") {})", self.pool.display(*body))
            }
            Expr::Apply(func, args) => {
                if args.is_empty() {
                    write!(f, "{}", func)
                } else {
                    write!(f, "({}", func)?;
                    for a in args {
                        write!(f, " {}", self.pool.display(*a))?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExprPool – hash-consing expression store
// ---------------------------------------------------------------------------

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(0);

/// Hash-consing expression pool for efficient expression construction and
/// structural sharing.  Every distinct `Expr` gets a unique `ExprId`.
#[derive(Debug, Clone)]
pub struct ExprPool {
    pool_id: u64,
    next_id: u64,
    pub(crate) exprs: IndexMap<ExprId, Expr>,
    dedup: HashMap<Expr, ExprId>,
    /// Cached sort information for each expression.
    sorts: HashMap<ExprId, Sort>,
    /// Named assertions for tracking.
    named: HashMap<String, ExprId>,
}

impl Default for ExprPool {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprPool {
    pub fn new() -> Self {
        Self {
            pool_id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            next_id: 0,
            exprs: IndexMap::new(),
            dedup: HashMap::new(),
            sorts: HashMap::new(),
            named: HashMap::new(),
        }
    }

    /// Pool identifier (useful when multiple pools coexist).
    pub fn pool_id(&self) -> u64 {
        self.pool_id
    }

    /// Number of unique expressions.
    pub fn len(&self) -> usize {
        self.exprs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

    /// Insert an expression, returning a deduplicated `ExprId`.
    pub fn intern(&mut self, expr: Expr) -> ExprId {
        if let Some(&id) = self.dedup.get(&expr) {
            return id;
        }
        let id = ExprId(self.next_id);
        self.next_id += 1;
        self.dedup.insert(expr.clone(), id);
        self.exprs.insert(id, expr);
        id
    }

    /// Look up the Expr for an id.
    pub fn get(&self, id: ExprId) -> Option<&Expr> {
        self.exprs.get(&id)
    }

    /// Look up the Expr for an id (panicking version).
    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[&id]
    }

    /// Create a display wrapper.
    pub fn display(&self, id: ExprId) -> ExprDisplay<'_> {
        ExprDisplay { pool: self, id }
    }

    /// Render an expression to its SMT-LIB2 string form.
    pub fn to_smtlib(&self, id: ExprId) -> String {
        format!("{}", self.display(id))
    }

    /// Store a cached sort for an expression.
    pub fn set_sort(&mut self, id: ExprId, sort: Sort) {
        self.sorts.insert(id, sort);
    }

    /// Get the cached sort for an expression.
    pub fn get_sort(&self, id: ExprId) -> Option<&Sort> {
        self.sorts.get(&id)
    }

    /// Name an assertion for unsat-core extraction.
    pub fn name_expr(&mut self, name: impl Into<String>, id: ExprId) {
        self.named.insert(name.into(), id);
    }

    /// Get a named expression.
    pub fn get_named(&self, name: &str) -> Option<ExprId> {
        self.named.get(name).copied()
    }

    /// Iterate all named expressions.
    pub fn named_exprs(&self) -> impl Iterator<Item = (&String, &ExprId)> {
        self.named.iter()
    }

    // -----------------------------------------------------------------------
    // Builder helpers – constants
    // -----------------------------------------------------------------------

    pub fn bool_const(&mut self, val: bool) -> ExprId {
        self.intern(Expr::Const(Value::Bool(val)))
    }

    pub fn true_expr(&mut self) -> ExprId {
        self.bool_const(true)
    }

    pub fn false_expr(&mut self) -> ExprId {
        self.bool_const(false)
    }

    pub fn bv_const(&mut self, width: u32, value: u64) -> ExprId {
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        self.intern(Expr::Const(Value::BitVec {
            width,
            value: value & mask,
        }))
    }

    pub fn int_const(&mut self, value: i64) -> ExprId {
        self.intern(Expr::Const(Value::Int(value)))
    }

    pub fn real_const(&mut self, value: f64) -> ExprId {
        self.intern(Expr::Const(Value::Real(value)))
    }

    pub fn bv_zero(&mut self, width: u32) -> ExprId {
        self.bv_const(width, 0)
    }

    pub fn bv_ones(&mut self, width: u32) -> ExprId {
        let val = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        self.bv_const(width, val)
    }

    // -----------------------------------------------------------------------
    // Builder helpers – variables
    // -----------------------------------------------------------------------

    pub fn var(&mut self, name: impl Into<String>, sort: Sort) -> ExprId {
        let name = name.into();
        let id = self.intern(Expr::Var(name, sort.clone()));
        self.set_sort(id, sort);
        id
    }

    pub fn bool_var(&mut self, name: impl Into<String>) -> ExprId {
        self.var(name, Sort::Bool)
    }

    pub fn bv_var(&mut self, name: impl Into<String>, width: u32) -> ExprId {
        self.var(name, Sort::BitVec(width))
    }

    pub fn int_var(&mut self, name: impl Into<String>) -> ExprId {
        self.var(name, Sort::Int)
    }

    pub fn real_var(&mut self, name: impl Into<String>) -> ExprId {
        self.var(name, Sort::Real)
    }

    pub fn array_var(&mut self, name: impl Into<String>, key: Sort, val: Sort) -> ExprId {
        self.var(name, Sort::Array(Box::new(key), Box::new(val)))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – boolean
    // -----------------------------------------------------------------------

    pub fn not(&mut self, a: ExprId) -> ExprId {
        self.intern(Expr::Not(a))
    }

    pub fn and(&mut self, args: Vec<ExprId>) -> ExprId {
        if args.is_empty() {
            return self.true_expr();
        }
        if args.len() == 1 {
            return args[0];
        }
        self.intern(Expr::And(args))
    }

    pub fn and2(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.and(vec![a, b])
    }

    pub fn or(&mut self, args: Vec<ExprId>) -> ExprId {
        if args.is_empty() {
            return self.false_expr();
        }
        if args.len() == 1 {
            return args[0];
        }
        self.intern(Expr::Or(args))
    }

    pub fn or2(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.or(vec![a, b])
    }

    pub fn implies(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Implies(a, b))
    }

    pub fn iff(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Iff(a, b))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – core
    // -----------------------------------------------------------------------

    pub fn eq(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Eq(a, b))
    }

    pub fn distinct(&mut self, args: Vec<ExprId>) -> ExprId {
        self.intern(Expr::Distinct(args))
    }

    pub fn ite(&mut self, cond: ExprId, then_: ExprId, else_: ExprId) -> ExprId {
        self.intern(Expr::Ite(cond, then_, else_))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – bitvector
    // -----------------------------------------------------------------------

    pub fn bvadd(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvAdd(a, b))
    }

    pub fn bvsub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvSub(a, b))
    }

    pub fn bvmul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvMul(a, b))
    }

    pub fn bvand(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvAnd(a, b))
    }

    pub fn bvor(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvOr(a, b))
    }

    pub fn bvxor(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvXor(a, b))
    }

    pub fn bvnot(&mut self, a: ExprId) -> ExprId {
        self.intern(Expr::BvNot(a))
    }

    pub fn bvneg(&mut self, a: ExprId) -> ExprId {
        self.intern(Expr::BvNeg(a))
    }

    pub fn bvshl(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvShl(a, b))
    }

    pub fn bvlshr(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvLshr(a, b))
    }

    pub fn bvashr(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvAshr(a, b))
    }

    pub fn bvult(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvUlt(a, b))
    }

    pub fn bvule(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvUle(a, b))
    }

    pub fn bvugt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvUgt(a, b))
    }

    pub fn bvuge(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvUge(a, b))
    }

    pub fn bvslt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvSlt(a, b))
    }

    pub fn bvsle(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvSle(a, b))
    }

    pub fn bvextract(&mut self, hi: u32, lo: u32, a: ExprId) -> ExprId {
        self.intern(Expr::BvExtract(hi, lo, a))
    }

    pub fn bvconcat(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::BvConcat(a, b))
    }

    pub fn bvzext(&mut self, extra_bits: u32, a: ExprId) -> ExprId {
        self.intern(Expr::BvZext(extra_bits, a))
    }

    pub fn bvsext(&mut self, extra_bits: u32, a: ExprId) -> ExprId {
        self.intern(Expr::BvSext(extra_bits, a))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – array
    // -----------------------------------------------------------------------

    pub fn select(&mut self, arr: ExprId, idx: ExprId) -> ExprId {
        self.intern(Expr::Select(arr, idx))
    }

    pub fn store(&mut self, arr: ExprId, idx: ExprId, val: ExprId) -> ExprId {
        self.intern(Expr::Store(arr, idx, val))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – integer arithmetic
    // -----------------------------------------------------------------------

    pub fn int_add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntAdd(a, b))
    }

    pub fn int_sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntSub(a, b))
    }

    pub fn int_mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntMul(a, b))
    }

    pub fn int_div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntDiv(a, b))
    }

    pub fn int_mod(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntMod(a, b))
    }

    pub fn int_le(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntLe(a, b))
    }

    pub fn int_lt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntLt(a, b))
    }

    pub fn int_ge(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntGe(a, b))
    }

    pub fn int_gt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::IntGt(a, b))
    }

    pub fn int_neg(&mut self, a: ExprId) -> ExprId {
        self.intern(Expr::IntNeg(a))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – real arithmetic
    // -----------------------------------------------------------------------

    pub fn real_add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealAdd(a, b))
    }

    pub fn real_sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealSub(a, b))
    }

    pub fn real_mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealMul(a, b))
    }

    pub fn real_div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealDiv(a, b))
    }

    pub fn real_le(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealLe(a, b))
    }

    pub fn real_lt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealLt(a, b))
    }

    pub fn real_ge(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::RealGe(a, b))
    }

    pub fn real_neg(&mut self, a: ExprId) -> ExprId {
        self.intern(Expr::RealNeg(a))
    }

    // -----------------------------------------------------------------------
    // Builder helpers – quantifiers and binders
    // -----------------------------------------------------------------------

    pub fn forall(&mut self, vars: Vec<SortedVar>, body: ExprId) -> ExprId {
        self.intern(Expr::Forall(vars, body))
    }

    pub fn exists(&mut self, vars: Vec<SortedVar>, body: ExprId) -> ExprId {
        self.intern(Expr::Exists(vars, body))
    }

    pub fn let_expr(
        &mut self,
        bindings: Vec<(String, ExprId)>,
        body: ExprId,
    ) -> ExprId {
        self.intern(Expr::Let(bindings, body))
    }

    pub fn apply(&mut self, func: impl Into<String>, args: Vec<ExprId>) -> ExprId {
        self.intern(Expr::Apply(func.into(), args))
    }

    // -----------------------------------------------------------------------
    // Traversal helpers
    // -----------------------------------------------------------------------

    /// Post-order traversal of all reachable nodes from `root`.
    pub fn post_order(&self, root: ExprId) -> Vec<ExprId> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self.post_order_inner(root, &mut visited, &mut result);
        result
    }

    fn post_order_inner(
        &self,
        id: ExprId,
        visited: &mut std::collections::HashSet<ExprId>,
        result: &mut Vec<ExprId>,
    ) {
        if !visited.insert(id) {
            return;
        }
        if let Some(expr) = self.exprs.get(&id) {
            for child in expr.children() {
                self.post_order_inner(child, visited, result);
            }
        }
        result.push(id);
    }

    /// Count the total number of nodes reachable from `root`.
    pub fn dag_size(&self, root: ExprId) -> usize {
        self.post_order(root).len()
    }

    /// Collect all free variables in the expression.
    pub fn free_vars(&self, root: ExprId) -> Vec<(String, Sort)> {
        let mut vars = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for id in self.post_order(root) {
            if let Some(Expr::Var(name, sort)) = self.exprs.get(&id) {
                if seen.insert(name.clone()) {
                    vars.push((name.clone(), sort.clone()));
                }
            }
        }
        vars
    }

    /// Substitute variables in an expression. Returns a new root ExprId.
    pub fn substitute(&mut self, root: ExprId, subst: &HashMap<String, ExprId>) -> ExprId {
        let order = self.post_order(root);
        let mut mapping: HashMap<ExprId, ExprId> = HashMap::new();

        for id in order {
            let expr = self.exprs[&id].clone();
            let new_id = match &expr {
                Expr::Var(name, _) => {
                    if let Some(&replacement) = subst.get(name) {
                        replacement
                    } else {
                        id
                    }
                }
                other => {
                    let children = other.children();
                    if children.iter().all(|c| mapping.get(c).copied().unwrap_or(*c) == *c) {
                        id
                    } else {
                        let new_expr = self.remap_children(other, &mapping);
                        self.intern(new_expr)
                    }
                }
            };
            mapping.insert(id, new_id);
        }

        mapping.get(&root).copied().unwrap_or(root)
    }

    /// Remap children of an expression using a mapping.
    fn remap_children(&self, expr: &Expr, mapping: &HashMap<ExprId, ExprId>) -> Expr {
        let remap = |id: &ExprId| mapping.get(id).copied().unwrap_or(*id);
        let remap_vec = |v: &[ExprId]| v.iter().map(remap).collect();

        match expr {
            Expr::Const(v) => Expr::Const(v.clone()),
            Expr::Var(n, s) => Expr::Var(n.clone(), s.clone()),
            Expr::Not(a) => Expr::Not(remap(a)),
            Expr::And(v) => Expr::And(remap_vec(v)),
            Expr::Or(v) => Expr::Or(remap_vec(v)),
            Expr::Implies(a, b) => Expr::Implies(remap(a), remap(b)),
            Expr::Iff(a, b) => Expr::Iff(remap(a), remap(b)),
            Expr::Eq(a, b) => Expr::Eq(remap(a), remap(b)),
            Expr::Distinct(v) => Expr::Distinct(remap_vec(v)),
            Expr::Ite(c, t, e) => Expr::Ite(remap(c), remap(t), remap(e)),
            Expr::BvAdd(a, b) => Expr::BvAdd(remap(a), remap(b)),
            Expr::BvSub(a, b) => Expr::BvSub(remap(a), remap(b)),
            Expr::BvMul(a, b) => Expr::BvMul(remap(a), remap(b)),
            Expr::BvAnd(a, b) => Expr::BvAnd(remap(a), remap(b)),
            Expr::BvOr(a, b) => Expr::BvOr(remap(a), remap(b)),
            Expr::BvXor(a, b) => Expr::BvXor(remap(a), remap(b)),
            Expr::BvNot(a) => Expr::BvNot(remap(a)),
            Expr::BvNeg(a) => Expr::BvNeg(remap(a)),
            Expr::BvShl(a, b) => Expr::BvShl(remap(a), remap(b)),
            Expr::BvLshr(a, b) => Expr::BvLshr(remap(a), remap(b)),
            Expr::BvAshr(a, b) => Expr::BvAshr(remap(a), remap(b)),
            Expr::BvUlt(a, b) => Expr::BvUlt(remap(a), remap(b)),
            Expr::BvUle(a, b) => Expr::BvUle(remap(a), remap(b)),
            Expr::BvUgt(a, b) => Expr::BvUgt(remap(a), remap(b)),
            Expr::BvUge(a, b) => Expr::BvUge(remap(a), remap(b)),
            Expr::BvSlt(a, b) => Expr::BvSlt(remap(a), remap(b)),
            Expr::BvSle(a, b) => Expr::BvSle(remap(a), remap(b)),
            Expr::BvExtract(hi, lo, a) => Expr::BvExtract(*hi, *lo, remap(a)),
            Expr::BvConcat(a, b) => Expr::BvConcat(remap(a), remap(b)),
            Expr::BvZext(n, a) => Expr::BvZext(*n, remap(a)),
            Expr::BvSext(n, a) => Expr::BvSext(*n, remap(a)),
            Expr::Select(a, b) => Expr::Select(remap(a), remap(b)),
            Expr::Store(a, b, c) => Expr::Store(remap(a), remap(b), remap(c)),
            Expr::IntAdd(a, b) => Expr::IntAdd(remap(a), remap(b)),
            Expr::IntSub(a, b) => Expr::IntSub(remap(a), remap(b)),
            Expr::IntMul(a, b) => Expr::IntMul(remap(a), remap(b)),
            Expr::IntDiv(a, b) => Expr::IntDiv(remap(a), remap(b)),
            Expr::IntMod(a, b) => Expr::IntMod(remap(a), remap(b)),
            Expr::IntLe(a, b) => Expr::IntLe(remap(a), remap(b)),
            Expr::IntLt(a, b) => Expr::IntLt(remap(a), remap(b)),
            Expr::IntGe(a, b) => Expr::IntGe(remap(a), remap(b)),
            Expr::IntGt(a, b) => Expr::IntGt(remap(a), remap(b)),
            Expr::IntNeg(a) => Expr::IntNeg(remap(a)),
            Expr::RealAdd(a, b) => Expr::RealAdd(remap(a), remap(b)),
            Expr::RealSub(a, b) => Expr::RealSub(remap(a), remap(b)),
            Expr::RealMul(a, b) => Expr::RealMul(remap(a), remap(b)),
            Expr::RealDiv(a, b) => Expr::RealDiv(remap(a), remap(b)),
            Expr::RealLe(a, b) => Expr::RealLe(remap(a), remap(b)),
            Expr::RealLt(a, b) => Expr::RealLt(remap(a), remap(b)),
            Expr::RealGe(a, b) => Expr::RealGe(remap(a), remap(b)),
            Expr::RealNeg(a) => Expr::RealNeg(remap(a)),
            Expr::Forall(vars, body) => Expr::Forall(vars.clone(), remap(body)),
            Expr::Exists(vars, body) => Expr::Exists(vars.clone(), remap(body)),
            Expr::Let(bindings, body) => {
                let new_bindings: Vec<_> = bindings
                    .iter()
                    .map(|(n, id)| (n.clone(), remap(id)))
                    .collect();
                Expr::Let(new_bindings, remap(body))
            }
            Expr::Apply(f, args) => Expr::Apply(f.clone(), remap_vec(args)),
        }
    }

    /// Simplify common patterns (constant folding, identity, etc.).
    pub fn simplify(&mut self, id: ExprId) -> ExprId {
        let expr = self.exprs[&id].clone();
        match &expr {
            Expr::Not(a) => {
                let sa = self.simplify(*a);
                if let Some(Expr::Const(Value::Bool(b))) = self.exprs.get(&sa) {
                    return self.bool_const(!b);
                }
                if let Some(Expr::Not(inner)) = self.exprs.get(&sa) {
                    return *inner;
                }
                self.not(sa)
            }
            Expr::And(args) => {
                let mut simplified = Vec::new();
                for a in args {
                    let sa = self.simplify(*a);
                    if let Some(Expr::Const(Value::Bool(false))) = self.exprs.get(&sa) {
                        return self.false_expr();
                    }
                    if let Some(Expr::Const(Value::Bool(true))) = self.exprs.get(&sa) {
                        continue;
                    }
                    simplified.push(sa);
                }
                self.and(simplified)
            }
            Expr::Or(args) => {
                let mut simplified = Vec::new();
                for a in args {
                    let sa = self.simplify(*a);
                    if let Some(Expr::Const(Value::Bool(true))) = self.exprs.get(&sa) {
                        return self.true_expr();
                    }
                    if let Some(Expr::Const(Value::Bool(false))) = self.exprs.get(&sa) {
                        continue;
                    }
                    simplified.push(sa);
                }
                self.or(simplified)
            }
            Expr::Ite(c, t, e) => {
                let sc = self.simplify(*c);
                if let Some(Expr::Const(Value::Bool(b))) = self.exprs.get(&sc) {
                    return if *b {
                        self.simplify(*t)
                    } else {
                        self.simplify(*e)
                    };
                }
                let st = self.simplify(*t);
                let se = self.simplify(*e);
                if st == se {
                    return st;
                }
                self.ite(sc, st, se)
            }
            Expr::Eq(a, b) => {
                let sa = self.simplify(*a);
                let sb = self.simplify(*b);
                if sa == sb {
                    return self.true_expr();
                }
                self.eq(sa, sb)
            }
            _ => id,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_display() {
        assert_eq!(Sort::Bool.to_string(), "Bool");
        assert_eq!(Sort::BitVec(32).to_string(), "(_ BitVec 32)");
        assert_eq!(Sort::Int.to_string(), "Int");
        assert_eq!(Sort::Real.to_string(), "Real");
        assert_eq!(
            Sort::Array(Box::new(Sort::BitVec(64)), Box::new(Sort::BitVec(8))).to_string(),
            "(Array (_ BitVec 64) (_ BitVec 8))"
        );
        assert_eq!(Sort::CacheState.to_string(), "CacheState");
    }

    #[test]
    fn test_sort_predicates() {
        assert!(Sort::Bool.is_bool());
        assert!(Sort::BitVec(32).is_bitvec());
        assert_eq!(Sort::BitVec(32).bv_width(), Some(32));
        assert!(Sort::Int.is_int());
        assert!(Sort::Real.is_real());
        let arr = Sort::array(Sort::Int, Sort::Bool);
        assert!(arr.is_array());
        let (k, v) = arr.array_sorts().unwrap();
        assert_eq!(*k, Sort::Int);
        assert_eq!(*v, Sort::Bool);
    }

    #[test]
    fn test_value_display() {
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Bool(false).to_string(), "false");
        assert_eq!(Value::Int(42).to_string(), "42");
        assert_eq!(Value::bv(4, 0b1010).to_string(), "#b1010");
    }

    #[test]
    fn test_hash_consing() {
        let mut pool = ExprPool::new();
        let a = pool.bv_const(32, 42);
        let b = pool.bv_const(32, 42);
        assert_eq!(a, b);
        assert_eq!(pool.len(), 1);

        let c = pool.bv_const(32, 43);
        assert_ne!(a, c);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_builder_boolean() {
        let mut pool = ExprPool::new();
        let x = pool.bool_var("x");
        let y = pool.bool_var("y");
        let conj = pool.and2(x, y);
        let smtlib = pool.to_smtlib(conj);
        assert_eq!(smtlib, "(and x y)");

        let disj = pool.or2(x, y);
        assert_eq!(pool.to_smtlib(disj), "(or x y)");

        let neg = pool.not(x);
        assert_eq!(pool.to_smtlib(neg), "(not x)");

        let imp = pool.implies(x, y);
        assert_eq!(pool.to_smtlib(imp), "(=> x y)");
    }

    #[test]
    fn test_builder_bitvector() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 32);
        let y = pool.bv_var("y", 32);
        let sum = pool.bvadd(x, y);
        assert_eq!(pool.to_smtlib(sum), "(bvadd x y)");

        let ext = pool.bvextract(7, 0, x);
        assert_eq!(pool.to_smtlib(ext), "((_ extract 7 0) x)");

        let zx = pool.bvzext(32, x);
        assert_eq!(pool.to_smtlib(zx), "((_ zero_extend 32) x)");

        let cat = pool.bvconcat(x, y);
        assert_eq!(pool.to_smtlib(cat), "(concat x y)");
    }

    #[test]
    fn test_builder_array() {
        let mut pool = ExprPool::new();
        let arr = pool.array_var("mem", Sort::BitVec(64), Sort::BitVec(8));
        let idx = pool.bv_var("addr", 64);
        let val = pool.bv_var("byte", 8);

        let sel = pool.select(arr, idx);
        assert_eq!(pool.to_smtlib(sel), "(select mem addr)");

        let sto = pool.store(arr, idx, val);
        assert_eq!(pool.to_smtlib(sto), "(store mem addr byte)");
    }

    #[test]
    fn test_builder_quantifier() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 64);
        let zero = pool.bv_zero(64);
        let body = pool.bvuge(x, zero);
        let fa = pool.forall(
            vec![SortedVar::new("x", Sort::BitVec(64))],
            body,
        );
        assert_eq!(
            pool.to_smtlib(fa),
            "(forall ((x (_ BitVec 64))) (bvuge x (_ bv0 64)))"
        );
    }

    #[test]
    fn test_builder_let() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 32);
        let y = pool.bv_var("y", 32);
        let sum = pool.bvadd(x, y);
        let body = pool.bvmul(pool.bv_var("tmp", 32), pool.bv_var("tmp", 32));
        let l = pool.let_expr(vec![("tmp".into(), sum)], body);
        assert_eq!(
            pool.to_smtlib(l),
            "(let ((tmp (bvadd x y))) (bvmul tmp tmp))"
        );
    }

    #[test]
    fn test_builder_apply() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 64);
        let y = pool.bv_var("y", 64);
        let app = pool.apply("cache_hit", vec![x, y]);
        assert_eq!(pool.to_smtlib(app), "(cache_hit x y)");
    }

    #[test]
    fn test_free_vars() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 32);
        let y = pool.bv_var("y", 32);
        let z = pool.bv_var("z", 32);
        let sum = pool.bvadd(x, y);
        let prod = pool.bvmul(sum, z);
        let fv = pool.free_vars(prod);
        let names: Vec<_> = fv.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));
        assert!(names.contains(&"z"));
        assert_eq!(fv.len(), 3);
    }

    #[test]
    fn test_simplify_const_and() {
        let mut pool = ExprPool::new();
        let t = pool.true_expr();
        let x = pool.bool_var("x");
        let conj = pool.and2(t, x);
        let simplified = pool.simplify(conj);
        assert_eq!(pool.to_smtlib(simplified), "x");
    }

    #[test]
    fn test_simplify_false_and() {
        let mut pool = ExprPool::new();
        let f = pool.false_expr();
        let x = pool.bool_var("x");
        let conj = pool.and2(f, x);
        let simplified = pool.simplify(conj);
        assert_eq!(pool.to_smtlib(simplified), "false");
    }

    #[test]
    fn test_simplify_true_or() {
        let mut pool = ExprPool::new();
        let t = pool.true_expr();
        let x = pool.bool_var("x");
        let disj = pool.or2(x, t);
        let simplified = pool.simplify(disj);
        assert_eq!(pool.to_smtlib(simplified), "true");
    }

    #[test]
    fn test_simplify_ite_true() {
        let mut pool = ExprPool::new();
        let t = pool.true_expr();
        let x = pool.bool_var("x");
        let y = pool.bool_var("y");
        let ite = pool.ite(t, x, y);
        let simplified = pool.simplify(ite);
        assert_eq!(pool.to_smtlib(simplified), "x");
    }

    #[test]
    fn test_simplify_eq_same() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 32);
        let eq = pool.eq(x, x);
        let simplified = pool.simplify(eq);
        assert_eq!(pool.to_smtlib(simplified), "true");
    }

    #[test]
    fn test_simplify_double_not() {
        let mut pool = ExprPool::new();
        let x = pool.bool_var("x");
        let nn = pool.not(pool.not(x));
        let simplified = pool.simplify(nn);
        assert_eq!(pool.to_smtlib(simplified), "x");
    }

    #[test]
    fn test_substitute() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 32);
        let y = pool.bv_var("y", 32);
        let sum = pool.bvadd(x, y);

        let forty_two = pool.bv_const(32, 42);
        let mut subst = HashMap::new();
        subst.insert("x".to_string(), forty_two);
        let result = pool.substitute(sum, &subst);
        assert_eq!(pool.to_smtlib(result), "(bvadd (_ bv42 32) y)");
    }

    #[test]
    fn test_dag_size() {
        let mut pool = ExprPool::new();
        let x = pool.bv_var("x", 32);
        let y = pool.bv_var("y", 32);
        let sum = pool.bvadd(x, y);
        let prod = pool.bvmul(sum, x);
        // prod -> sum, x; sum -> x, y. Unique: x, y, sum, prod = 4
        assert_eq!(pool.dag_size(prod), 4);
    }

    #[test]
    fn test_named_exprs() {
        let mut pool = ExprPool::new();
        let x = pool.bool_var("x");
        pool.name_expr("assertion_1", x);
        assert_eq!(pool.get_named("assertion_1"), Some(x));
        assert_eq!(pool.get_named("nonexistent"), None);
    }

    #[test]
    fn test_int_arithmetic_display() {
        let mut pool = ExprPool::new();
        let a = pool.int_var("a");
        let b = pool.int_var("b");
        let sum = pool.int_add(a, b);
        assert_eq!(pool.to_smtlib(sum), "(+ a b)");

        let prod = pool.int_mul(a, b);
        assert_eq!(pool.to_smtlib(prod), "(* a b)");

        let le = pool.int_le(a, b);
        assert_eq!(pool.to_smtlib(le), "(<= a b)");
    }

    #[test]
    fn test_real_arithmetic_display() {
        let mut pool = ExprPool::new();
        let x = pool.real_var("x");
        let y = pool.real_var("y");
        let sum = pool.real_add(x, y);
        assert_eq!(pool.to_smtlib(sum), "(+ x y)");

        let div = pool.real_div(x, y);
        assert_eq!(pool.to_smtlib(div), "(/ x y)");
    }

    #[test]
    fn test_distinct() {
        let mut pool = ExprPool::new();
        let a = pool.bv_var("a", 32);
        let b = pool.bv_var("b", 32);
        let c = pool.bv_var("c", 32);
        let d = pool.distinct(vec![a, b, c]);
        assert_eq!(pool.to_smtlib(d), "(distinct a b c)");
    }

    #[test]
    fn test_children() {
        let pool = ExprPool::new();

        let leaf = Expr::Const(Value::Bool(true));
        assert!(leaf.children().is_empty());
        assert!(leaf.is_leaf());

        let not = Expr::Not(ExprId(0));
        assert_eq!(not.children().len(), 1);
        assert!(!not.is_leaf());

        let and = Expr::And(vec![ExprId(0), ExprId(1), ExprId(2)]);
        assert_eq!(and.children().len(), 3);

        let ite = Expr::Ite(ExprId(0), ExprId(1), ExprId(2));
        assert_eq!(ite.children().len(), 3);

        drop(pool);
    }

    #[test]
    fn test_expr_is_boolean() {
        assert!(Expr::Const(Value::Bool(true)).is_boolean());
        assert!(Expr::Not(ExprId(0)).is_boolean());
        assert!(Expr::BvUlt(ExprId(0), ExprId(1)).is_boolean());
        assert!(Expr::IntLe(ExprId(0), ExprId(1)).is_boolean());
        assert!(!Expr::BvAdd(ExprId(0), ExprId(1)).is_boolean());
    }
}
