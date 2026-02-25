//! QCTL_F formula syntax: AST, parsing, pretty-printing, simplification.
//!
//! Defines the abstract syntax tree for quantitative coalgebraic temporal logic
//! formulas parameterized by a behavioral functor. Supports state formulas,
//! path formulas with quantitative extensions, graded modalities, and formula
//! manipulation utilities.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use ordered_float::OrderedFloat;
use uuid::Uuid;

// ───────────────────────────────────────────────────────────────────────────────
// Core identifier types (local aliases)
// ───────────────────────────────────────────────────────────────────────────────

/// An atomic proposition name.
pub type PropName = String;

/// Unique identifier for a formula node (useful for subformula DAGs).
pub type FormulaId = String;

// ───────────────────────────────────────────────────────────────────────────────
// Comparison / boolean operators
// ───────────────────────────────────────────────────────────────────────────────

/// Comparison operators for quantitative bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Ge,  // ≥
    Gt,  // >
    Le,  // ≤
    Lt,  // <
    Eq,  // =
}

impl ComparisonOp {
    /// Evaluate the comparison on two f64 values.
    pub fn evaluate(self, lhs: f64, rhs: f64) -> bool {
        match self {
            ComparisonOp::Ge => lhs >= rhs - 1e-12,
            ComparisonOp::Gt => lhs > rhs + 1e-12,
            ComparisonOp::Le => lhs <= rhs + 1e-12,
            ComparisonOp::Lt => lhs < rhs - 1e-12,
            ComparisonOp::Eq => (lhs - rhs).abs() < 1e-9,
        }
    }

    /// Negate the comparison operator.
    pub fn negate(self) -> Self {
        match self {
            ComparisonOp::Ge => ComparisonOp::Lt,
            ComparisonOp::Gt => ComparisonOp::Le,
            ComparisonOp::Le => ComparisonOp::Gt,
            ComparisonOp::Lt => ComparisonOp::Ge,
            ComparisonOp::Eq => ComparisonOp::Eq, // ≠ not in our logic; keep as Eq
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            ComparisonOp::Ge => ">=",
            ComparisonOp::Gt => ">",
            ComparisonOp::Le => "<=",
            ComparisonOp::Lt => "<",
            ComparisonOp::Eq => "=",
        }
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Boolean connective kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoolOp {
    And,
    Or,
    Implies,
    Iff,
}

impl fmt::Display for BoolOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolOp::And => write!(f, "∧"),
            BoolOp::Or => write!(f, "∨"),
            BoolOp::Implies => write!(f, "→"),
            BoolOp::Iff => write!(f, "↔"),
        }
    }
}

/// Path quantifier: universal or existential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathQuantifier {
    /// For all paths (A)
    All,
    /// There exists a path (E)
    Exists,
}

impl fmt::Display for PathQuantifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathQuantifier::All => write!(f, "A"),
            PathQuantifier::Exists => write!(f, "E"),
        }
    }
}

/// Temporal operator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalOp {
    /// NeXt step
    X,
    /// Globally (always)
    G,
    /// Finally (eventually)
    F,
    /// Until
    U,
    /// Weak-until / Release
    R,
}

impl fmt::Display for TemporalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalOp::X => write!(f, "X"),
            TemporalOp::G => write!(f, "G"),
            TemporalOp::F => write!(f, "F"),
            TemporalOp::U => write!(f, "U"),
            TemporalOp::R => write!(f, "R"),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Formula AST
// ───────────────────────────────────────────────────────────────────────────────

/// A QCTL_F formula — the main AST type.
///
/// This is a unified enum that covers state formulas (evaluated at a state),
/// path formulas (evaluated on paths), and quantitative extensions (probabilistic
/// bounds, graded modalities).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Formula {
    // ── atomic ──
    /// Boolean constant: ⊤ or ⊥
    Bool(bool),

    /// Atomic proposition: p
    Atom(PropName),

    // ── boolean connectives ──
    /// Negation: ¬φ
    Not(Box<Formula>),

    /// Binary boolean connective: φ₁ op φ₂
    BoolBin {
        op: BoolOp,
        lhs: Box<Formula>,
        rhs: Box<Formula>,
    },

    // ── CTL path formulas ──
    /// AX φ, EX φ  (next)
    Next {
        quantifier: PathQuantifier,
        inner: Box<Formula>,
    },

    /// AG φ, EG φ  (globally)
    Globally {
        quantifier: PathQuantifier,
        inner: Box<Formula>,
    },

    /// AF φ, EF φ  (finally)
    Finally {
        quantifier: PathQuantifier,
        inner: Box<Formula>,
    },

    /// A[φ₁ U φ₂], E[φ₁ U φ₂]  (until)
    Until {
        quantifier: PathQuantifier,
        hold: Box<Formula>,
        goal: Box<Formula>,
    },

    /// A[φ₁ R φ₂], E[φ₁ R φ₂]  (release)
    Release {
        quantifier: PathQuantifier,
        trigger: Box<Formula>,
        invariant: Box<Formula>,
    },

    // ── quantitative extensions ──
    /// Probabilistic bound: P[⊳p](φ)  where ⊳ ∈ {≥,>,≤,<,=}
    ProbBound {
        op: ComparisonOp,
        threshold: f64,
        inner: Box<Formula>,
    },

    /// Quantitative expectation: E[⊳v](X φ)
    ExpBound {
        op: ComparisonOp,
        threshold: f64,
        temporal: TemporalOp,
        inner: Box<Formula>,
    },

    /// Graded modality parameterized by functor grade k: ⟨k⟩φ
    GradedModality {
        grade: u32,
        inner: Box<Formula>,
    },

    /// Bounded until: A/E[φ₁ U≤n φ₂]  with step bound
    BoundedUntil {
        quantifier: PathQuantifier,
        hold: Box<Formula>,
        goal: Box<Formula>,
        bound: u32,
    },

    /// Fixed point: μX.φ  or  νX.φ
    FixedPoint {
        is_least: bool,
        variable: String,
        body: Box<Formula>,
    },

    /// Fixed point variable reference
    Var(String),

    /// Quantitative value literal in [0,1]
    QVal(OrderedFloat<f64>),
}

// ───────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ───────────────────────────────────────────────────────────────────────────────

impl Formula {
    // ── atoms & constants ──
    pub fn top() -> Self { Formula::Bool(true) }
    pub fn bot() -> Self { Formula::Bool(false) }

    pub fn atom(name: impl Into<String>) -> Self {
        Formula::Atom(name.into())
    }

    pub fn qval(v: f64) -> Self {
        Formula::QVal(OrderedFloat(v.clamp(0.0, 1.0)))
    }

    pub fn var(name: impl Into<String>) -> Self {
        Formula::Var(name.into())
    }

    // ── boolean ──
    pub fn not(inner: Formula) -> Self {
        Formula::Not(Box::new(inner))
    }

    pub fn and(lhs: Formula, rhs: Formula) -> Self {
        Formula::BoolBin { op: BoolOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    pub fn or(lhs: Formula, rhs: Formula) -> Self {
        Formula::BoolBin { op: BoolOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    pub fn implies(lhs: Formula, rhs: Formula) -> Self {
        Formula::BoolBin { op: BoolOp::Implies, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    pub fn iff(lhs: Formula, rhs: Formula) -> Self {
        Formula::BoolBin { op: BoolOp::Iff, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    // ── temporal ──
    pub fn ax(inner: Formula) -> Self {
        Formula::Next { quantifier: PathQuantifier::All, inner: Box::new(inner) }
    }
    pub fn ex(inner: Formula) -> Self {
        Formula::Next { quantifier: PathQuantifier::Exists, inner: Box::new(inner) }
    }
    pub fn ag(inner: Formula) -> Self {
        Formula::Globally { quantifier: PathQuantifier::All, inner: Box::new(inner) }
    }
    pub fn eg(inner: Formula) -> Self {
        Formula::Globally { quantifier: PathQuantifier::Exists, inner: Box::new(inner) }
    }
    pub fn af(inner: Formula) -> Self {
        Formula::Finally { quantifier: PathQuantifier::All, inner: Box::new(inner) }
    }
    pub fn ef(inner: Formula) -> Self {
        Formula::Finally { quantifier: PathQuantifier::Exists, inner: Box::new(inner) }
    }
    pub fn au(hold: Formula, goal: Formula) -> Self {
        Formula::Until { quantifier: PathQuantifier::All, hold: Box::new(hold), goal: Box::new(goal) }
    }
    pub fn eu(hold: Formula, goal: Formula) -> Self {
        Formula::Until { quantifier: PathQuantifier::Exists, hold: Box::new(hold), goal: Box::new(goal) }
    }
    pub fn ar(trigger: Formula, invariant: Formula) -> Self {
        Formula::Release { quantifier: PathQuantifier::All, trigger: Box::new(trigger), invariant: Box::new(invariant) }
    }
    pub fn er(trigger: Formula, invariant: Formula) -> Self {
        Formula::Release { quantifier: PathQuantifier::Exists, trigger: Box::new(trigger), invariant: Box::new(invariant) }
    }

    // ── quantitative ──
    pub fn prob_ge(p: f64, inner: Formula) -> Self {
        Formula::ProbBound { op: ComparisonOp::Ge, threshold: p, inner: Box::new(inner) }
    }
    pub fn prob_le(p: f64, inner: Formula) -> Self {
        Formula::ProbBound { op: ComparisonOp::Le, threshold: p, inner: Box::new(inner) }
    }
    pub fn exp_bound(op: ComparisonOp, v: f64, temporal: TemporalOp, inner: Formula) -> Self {
        Formula::ExpBound { op, threshold: v, temporal, inner: Box::new(inner) }
    }
    pub fn graded(k: u32, inner: Formula) -> Self {
        Formula::GradedModality { grade: k, inner: Box::new(inner) }
    }
    pub fn bounded_until(q: PathQuantifier, hold: Formula, goal: Formula, n: u32) -> Self {
        Formula::BoundedUntil { quantifier: q, hold: Box::new(hold), goal: Box::new(goal), bound: n }
    }
    pub fn mu(var: impl Into<String>, body: Formula) -> Self {
        Formula::FixedPoint { is_least: true, variable: var.into(), body: Box::new(body) }
    }
    pub fn nu(var: impl Into<String>, body: Formula) -> Self {
        Formula::FixedPoint { is_least: false, variable: var.into(), body: Box::new(body) }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// StateFormula / PathFormula / QuantFormula – classified wrapper types
// ───────────────────────────────────────────────────────────────────────────────

/// Classification of a formula into state vs path vs quantitative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormulaClass {
    State,
    Path,
    Quantitative,
    FixedPt,
}

/// Thin wrapper: a formula known to be a state formula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateFormula(pub Formula);

/// Thin wrapper: a formula known to be a path formula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathFormula(pub Formula);

/// Thin wrapper: a formula known to be quantitative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantFormula(pub Formula);

impl Formula {
    /// Classify a formula node.
    pub fn classify(&self) -> FormulaClass {
        match self {
            Formula::Bool(_) | Formula::Atom(_) | Formula::Not(_) | Formula::BoolBin { .. } =>
                FormulaClass::State,
            Formula::Next { .. } | Formula::Globally { .. } | Formula::Finally { .. }
            | Formula::Until { .. } | Formula::Release { .. } | Formula::BoundedUntil { .. } =>
                FormulaClass::Path,
            Formula::ProbBound { .. } | Formula::ExpBound { .. }
            | Formula::GradedModality { .. } | Formula::QVal(_) =>
                FormulaClass::Quantitative,
            Formula::FixedPoint { .. } | Formula::Var(_) =>
                FormulaClass::FixedPt,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// FormulaInfo — structural metrics
// ───────────────────────────────────────────────────────────────────────────────

/// Structural metrics about a formula.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaInfo {
    /// Total number of AST nodes.
    pub size: usize,
    /// Maximum nesting depth.
    pub depth: usize,
    /// Set of atomic proposition names referenced.
    pub atoms: BTreeSet<String>,
    /// Number of temporal operators used.
    pub temporal_count: usize,
    /// Number of quantitative extensions.
    pub quant_count: usize,
    /// Free variables (from fixed-point binders).
    pub free_vars: BTreeSet<String>,
    /// Alternation depth (nesting of μ inside ν and vice-versa).
    pub alternation_depth: usize,
}

impl Formula {
    /// Compute structural info about this formula.
    pub fn info(&self) -> FormulaInfo {
        let mut info = FormulaInfo {
            size: 0,
            depth: 0,
            atoms: BTreeSet::new(),
            temporal_count: 0,
            quant_count: 0,
            free_vars: BTreeSet::new(),
            alternation_depth: 0,
        };
        let bound: HashSet<String> = HashSet::new();
        self.collect_info(&mut info, 0, &bound);
        info.alternation_depth = self.compute_alternation_depth(&HashSet::new(), true);
        info
    }

    fn collect_info(&self, info: &mut FormulaInfo, depth: usize, bound: &HashSet<String>) {
        info.size += 1;
        if depth > info.depth {
            info.depth = depth;
        }
        match self {
            Formula::Bool(_) => {}
            Formula::Atom(name) => { info.atoms.insert(name.clone()); }
            Formula::QVal(_) => { info.quant_count += 1; }
            Formula::Var(v) => {
                if !bound.contains(v) {
                    info.free_vars.insert(v.clone());
                }
            }
            Formula::Not(inner) => {
                inner.collect_info(info, depth + 1, bound);
            }
            Formula::BoolBin { lhs, rhs, .. } => {
                lhs.collect_info(info, depth + 1, bound);
                rhs.collect_info(info, depth + 1, bound);
            }
            Formula::Next { inner, .. } | Formula::Globally { inner, .. }
            | Formula::Finally { inner, .. } => {
                info.temporal_count += 1;
                inner.collect_info(info, depth + 1, bound);
            }
            Formula::Until { hold, goal, .. } | Formula::Release { trigger: hold, invariant: goal, .. } => {
                info.temporal_count += 1;
                hold.collect_info(info, depth + 1, bound);
                goal.collect_info(info, depth + 1, bound);
            }
            Formula::BoundedUntil { hold, goal, .. } => {
                info.temporal_count += 1;
                hold.collect_info(info, depth + 1, bound);
                goal.collect_info(info, depth + 1, bound);
            }
            Formula::ProbBound { inner, .. } => {
                info.quant_count += 1;
                inner.collect_info(info, depth + 1, bound);
            }
            Formula::ExpBound { inner, .. } => {
                info.quant_count += 1;
                info.temporal_count += 1;
                inner.collect_info(info, depth + 1, bound);
            }
            Formula::GradedModality { inner, .. } => {
                info.quant_count += 1;
                inner.collect_info(info, depth + 1, bound);
            }
            Formula::FixedPoint { variable, body, .. } => {
                let mut bound2 = bound.clone();
                bound2.insert(variable.clone());
                body.collect_info(info, depth + 1, &bound2);
            }
        }
    }

    fn compute_alternation_depth(&self, _bound: &HashSet<String>, _is_least_context: bool) -> usize {
        match self {
            Formula::FixedPoint { is_least, body, .. } => {
                let inner_ad = body.compute_alternation_depth(&HashSet::new(), *is_least);
                // If the outermost is μ and we find ν inside (or vice-versa), add 1
                inner_ad + 1
            }
            Formula::Not(inner) => inner.compute_alternation_depth(_bound, _is_least_context),
            Formula::BoolBin { lhs, rhs, .. } => {
                let l = lhs.compute_alternation_depth(_bound, _is_least_context);
                let r = rhs.compute_alternation_depth(_bound, _is_least_context);
                l.max(r)
            }
            Formula::Next { inner, .. } | Formula::Globally { inner, .. }
            | Formula::Finally { inner, .. } | Formula::ProbBound { inner, .. }
            | Formula::ExpBound { inner, .. } | Formula::GradedModality { inner, .. } => {
                inner.compute_alternation_depth(_bound, _is_least_context)
            }
            Formula::Until { hold, goal, .. } | Formula::Release { trigger: hold, invariant: goal, .. }
            | Formula::BoundedUntil { hold, goal, .. } => {
                let l = hold.compute_alternation_depth(_bound, _is_least_context);
                let r = goal.compute_alternation_depth(_bound, _is_least_context);
                l.max(r)
            }
            _ => 0,
        }
    }

    /// Total number of AST nodes.
    pub fn size(&self) -> usize {
        self.info().size
    }

    /// Maximum nesting depth.
    pub fn depth(&self) -> usize {
        self.info().depth
    }

    /// Collect all atomic propositions.
    pub fn atoms(&self) -> BTreeSet<String> {
        self.info().atoms
    }

    /// Extract all subformulas (including self), depth-first.
    pub fn subformulas(&self) -> Vec<&Formula> {
        let mut result = Vec::new();
        self.collect_subformulas(&mut result);
        result
    }

    fn collect_subformulas<'a>(&'a self, out: &mut Vec<&'a Formula>) {
        out.push(self);
        match self {
            Formula::Bool(_) | Formula::Atom(_) | Formula::QVal(_) | Formula::Var(_) => {}
            Formula::Not(inner) => inner.collect_subformulas(out),
            Formula::BoolBin { lhs, rhs, .. } => {
                lhs.collect_subformulas(out);
                rhs.collect_subformulas(out);
            }
            Formula::Next { inner, .. } | Formula::Globally { inner, .. }
            | Formula::Finally { inner, .. } | Formula::ProbBound { inner, .. }
            | Formula::ExpBound { inner, .. } | Formula::GradedModality { inner, .. } => {
                inner.collect_subformulas(out);
            }
            Formula::Until { hold, goal, .. } | Formula::Release { trigger: hold, invariant: goal, .. }
            | Formula::BoundedUntil { hold, goal, .. } => {
                hold.collect_subformulas(out);
                goal.collect_subformulas(out);
            }
            Formula::FixedPoint { body, .. } => body.collect_subformulas(out),
        }
    }

    /// Substitute a variable with a formula.
    pub fn substitute(&self, var: &str, replacement: &Formula) -> Formula {
        match self {
            Formula::Var(v) if v == var => replacement.clone(),
            Formula::Var(_) | Formula::Bool(_) | Formula::Atom(_) | Formula::QVal(_) => self.clone(),
            Formula::Not(inner) => Formula::not(inner.substitute(var, replacement)),
            Formula::BoolBin { op, lhs, rhs } => Formula::BoolBin {
                op: *op,
                lhs: Box::new(lhs.substitute(var, replacement)),
                rhs: Box::new(rhs.substitute(var, replacement)),
            },
            Formula::Next { quantifier, inner } => Formula::Next {
                quantifier: *quantifier,
                inner: Box::new(inner.substitute(var, replacement)),
            },
            Formula::Globally { quantifier, inner } => Formula::Globally {
                quantifier: *quantifier,
                inner: Box::new(inner.substitute(var, replacement)),
            },
            Formula::Finally { quantifier, inner } => Formula::Finally {
                quantifier: *quantifier,
                inner: Box::new(inner.substitute(var, replacement)),
            },
            Formula::Until { quantifier, hold, goal } => Formula::Until {
                quantifier: *quantifier,
                hold: Box::new(hold.substitute(var, replacement)),
                goal: Box::new(goal.substitute(var, replacement)),
            },
            Formula::Release { quantifier, trigger, invariant } => Formula::Release {
                quantifier: *quantifier,
                trigger: Box::new(trigger.substitute(var, replacement)),
                invariant: Box::new(invariant.substitute(var, replacement)),
            },
            Formula::BoundedUntil { quantifier, hold, goal, bound } => Formula::BoundedUntil {
                quantifier: *quantifier,
                hold: Box::new(hold.substitute(var, replacement)),
                goal: Box::new(goal.substitute(var, replacement)),
                bound: *bound,
            },
            Formula::ProbBound { op, threshold, inner } => Formula::ProbBound {
                op: *op,
                threshold: *threshold,
                inner: Box::new(inner.substitute(var, replacement)),
            },
            Formula::ExpBound { op, threshold, temporal, inner } => Formula::ExpBound {
                op: *op,
                threshold: *threshold,
                temporal: *temporal,
                inner: Box::new(inner.substitute(var, replacement)),
            },
            Formula::GradedModality { grade, inner } => Formula::GradedModality {
                grade: *grade,
                inner: Box::new(inner.substitute(var, replacement)),
            },
            Formula::FixedPoint { is_least, variable, body } => {
                if variable == var {
                    // The binding shadows; do not recurse
                    self.clone()
                } else {
                    Formula::FixedPoint {
                        is_least: *is_least,
                        variable: variable.clone(),
                        body: Box::new(body.substitute(var, replacement)),
                    }
                }
            }
        }
    }

    /// Check structural equality (using derived PartialEq).
    pub fn structurally_equal(&self, other: &Formula) -> bool {
        self == other
    }

    /// Negate the formula using push-through NNF transformation.
    pub fn nnf_negate(&self) -> Formula {
        match self {
            Formula::Bool(b) => Formula::Bool(!b),
            Formula::Atom(_) | Formula::QVal(_) | Formula::Var(_) => Formula::not(self.clone()),
            Formula::Not(inner) => *inner.clone(),
            Formula::BoolBin { op, lhs, rhs } => match op {
                BoolOp::And => Formula::or(lhs.nnf_negate(), rhs.nnf_negate()),
                BoolOp::Or => Formula::and(lhs.nnf_negate(), rhs.nnf_negate()),
                BoolOp::Implies => Formula::and(*lhs.clone(), rhs.nnf_negate()),
                BoolOp::Iff => Formula::or(
                    Formula::and(*lhs.clone(), rhs.nnf_negate()),
                    Formula::and(lhs.nnf_negate(), *rhs.clone()),
                ),
            },
            Formula::Next { quantifier, inner } => {
                let dual = match quantifier {
                    PathQuantifier::All => PathQuantifier::Exists,
                    PathQuantifier::Exists => PathQuantifier::All,
                };
                Formula::Next { quantifier: dual, inner: Box::new(inner.nnf_negate()) }
            }
            Formula::Globally { quantifier, inner } => {
                let dual = match quantifier {
                    PathQuantifier::All => PathQuantifier::Exists,
                    PathQuantifier::Exists => PathQuantifier::All,
                };
                Formula::Finally { quantifier: dual, inner: Box::new(inner.nnf_negate()) }
            }
            Formula::Finally { quantifier, inner } => {
                let dual = match quantifier {
                    PathQuantifier::All => PathQuantifier::Exists,
                    PathQuantifier::Exists => PathQuantifier::All,
                };
                Formula::Globally { quantifier: dual, inner: Box::new(inner.nnf_negate()) }
            }
            Formula::Until { quantifier, hold, goal } => {
                let dual = match quantifier {
                    PathQuantifier::All => PathQuantifier::Exists,
                    PathQuantifier::Exists => PathQuantifier::All,
                };
                Formula::Release {
                    quantifier: dual,
                    trigger: Box::new(hold.nnf_negate()),
                    invariant: Box::new(goal.nnf_negate()),
                }
            }
            Formula::Release { quantifier, trigger, invariant } => {
                let dual = match quantifier {
                    PathQuantifier::All => PathQuantifier::Exists,
                    PathQuantifier::Exists => PathQuantifier::All,
                };
                Formula::Until {
                    quantifier: dual,
                    hold: Box::new(trigger.nnf_negate()),
                    goal: Box::new(invariant.nnf_negate()),
                }
            }
            Formula::ProbBound { op, threshold, inner } => {
                Formula::ProbBound {
                    op: op.negate(),
                    threshold: *threshold,
                    inner: inner.clone(),
                }
            }
            _ => Formula::not(self.clone()),
        }
    }

    /// Convert to negation normal form (push negations to atoms).
    pub fn to_nnf(&self) -> Formula {
        match self {
            Formula::Not(inner) => inner.nnf_negate(),
            Formula::BoolBin { op, lhs, rhs } => {
                let l = lhs.to_nnf();
                let r = rhs.to_nnf();
                match op {
                    BoolOp::Implies => Formula::or(l.nnf_negate(), r),
                    BoolOp::Iff => Formula::and(
                        Formula::or(l.nnf_negate(), r.clone()),
                        Formula::or(l, r.nnf_negate()),
                    ),
                    _ => Formula::BoolBin { op: *op, lhs: Box::new(l), rhs: Box::new(r) },
                }
            }
            Formula::Next { quantifier, inner } => Formula::Next {
                quantifier: *quantifier,
                inner: Box::new(inner.to_nnf()),
            },
            Formula::Globally { quantifier, inner } => Formula::Globally {
                quantifier: *quantifier,
                inner: Box::new(inner.to_nnf()),
            },
            Formula::Finally { quantifier, inner } => Formula::Finally {
                quantifier: *quantifier,
                inner: Box::new(inner.to_nnf()),
            },
            Formula::Until { quantifier, hold, goal } => Formula::Until {
                quantifier: *quantifier,
                hold: Box::new(hold.to_nnf()),
                goal: Box::new(goal.to_nnf()),
            },
            Formula::Release { quantifier, trigger, invariant } => Formula::Release {
                quantifier: *quantifier,
                trigger: Box::new(trigger.to_nnf()),
                invariant: Box::new(invariant.to_nnf()),
            },
            Formula::BoundedUntil { quantifier, hold, goal, bound } => Formula::BoundedUntil {
                quantifier: *quantifier,
                hold: Box::new(hold.to_nnf()),
                goal: Box::new(goal.to_nnf()),
                bound: *bound,
            },
            Formula::ProbBound { op, threshold, inner } => Formula::ProbBound {
                op: *op,
                threshold: *threshold,
                inner: Box::new(inner.to_nnf()),
            },
            Formula::ExpBound { op, threshold, temporal, inner } => Formula::ExpBound {
                op: *op,
                threshold: *threshold,
                temporal: *temporal,
                inner: Box::new(inner.to_nnf()),
            },
            Formula::GradedModality { grade, inner } => Formula::GradedModality {
                grade: *grade,
                inner: Box::new(inner.to_nnf()),
            },
            Formula::FixedPoint { is_least, variable, body } => Formula::FixedPoint {
                is_least: *is_least,
                variable: variable.clone(),
                body: Box::new(body.to_nnf()),
            },
            other => other.clone(),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Display (pretty-printing)
// ───────────────────────────────────────────────────────────────────────────────

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        FormulaPrinter::new().write(self, f)
    }
}

/// Pretty-printer with configurable style.
#[derive(Debug, Clone)]
pub struct FormulaPrinter {
    /// Use Unicode symbols (∧, ∨, etc.) vs ASCII (&&, ||, etc.).
    pub unicode: bool,
    /// Maximum line width for wrapping (0 = no wrapping).
    pub max_width: usize,
}

impl FormulaPrinter {
    pub fn new() -> Self {
        Self { unicode: true, max_width: 0 }
    }

    pub fn ascii() -> Self {
        Self { unicode: false, max_width: 0 }
    }

    pub fn to_string(&self, formula: &Formula) -> String {
        let mut buf = String::new();
        self.write(formula, &mut buf).unwrap();
        buf
    }

    fn write<W: fmt::Write>(&self, formula: &Formula, w: &mut W) -> fmt::Result {
        match formula {
            Formula::Bool(true) => write!(w, "{}", if self.unicode { "⊤" } else { "true" }),
            Formula::Bool(false) => write!(w, "{}", if self.unicode { "⊥" } else { "false" }),
            Formula::Atom(name) => write!(w, "{}", name),
            Formula::QVal(v) => write!(w, "{:.4}", v.into_inner()),
            Formula::Var(v) => write!(w, "{}", v),
            Formula::Not(inner) => {
                write!(w, "{}", if self.unicode { "¬" } else { "!" })?;
                if inner.needs_parens() {
                    write!(w, "(")?;
                    self.write(inner, w)?;
                    write!(w, ")")
                } else {
                    self.write(inner, w)
                }
            }
            Formula::BoolBin { op, lhs, rhs } => {
                let op_str = if self.unicode {
                    match op {
                        BoolOp::And => "∧",
                        BoolOp::Or => "∨",
                        BoolOp::Implies => "→",
                        BoolOp::Iff => "↔",
                    }
                } else {
                    match op {
                        BoolOp::And => "&&",
                        BoolOp::Or => "||",
                        BoolOp::Implies => "->",
                        BoolOp::Iff => "<->",
                    }
                };
                write!(w, "(")?;
                self.write(lhs, w)?;
                write!(w, " {} ", op_str)?;
                self.write(rhs, w)?;
                write!(w, ")")
            }
            Formula::Next { quantifier, inner } => {
                write!(w, "{}X ", quantifier)?;
                self.write(inner, w)
            }
            Formula::Globally { quantifier, inner } => {
                write!(w, "{}G ", quantifier)?;
                self.write(inner, w)
            }
            Formula::Finally { quantifier, inner } => {
                write!(w, "{}F ", quantifier)?;
                self.write(inner, w)
            }
            Formula::Until { quantifier, hold, goal } => {
                write!(w, "{}[", quantifier)?;
                self.write(hold, w)?;
                write!(w, " U ")?;
                self.write(goal, w)?;
                write!(w, "]")
            }
            Formula::Release { quantifier, trigger, invariant } => {
                write!(w, "{}[", quantifier)?;
                self.write(trigger, w)?;
                write!(w, " R ")?;
                self.write(invariant, w)?;
                write!(w, "]")
            }
            Formula::BoundedUntil { quantifier, hold, goal, bound } => {
                write!(w, "{}[", quantifier)?;
                self.write(hold, w)?;
                write!(w, " U≤{} ", bound)?;
                self.write(goal, w)?;
                write!(w, "]")
            }
            Formula::ProbBound { op, threshold, inner } => {
                write!(w, "P[{}{:.4}](", op, threshold)?;
                self.write(inner, w)?;
                write!(w, ")")
            }
            Formula::ExpBound { op, threshold, temporal, inner } => {
                write!(w, "E[{}{:.4}]{} ", op, threshold, temporal)?;
                self.write(inner, w)
            }
            Formula::GradedModality { grade, inner } => {
                write!(w, "⟨{}⟩", grade)?;
                self.write(inner, w)
            }
            Formula::FixedPoint { is_least, variable, body } => {
                write!(w, "{}{}.(",
                    if *is_least { "μ" } else { "ν" },
                    variable)?;
                self.write(body, w)?;
                write!(w, ")")
            }
        }
    }
}

impl Formula {
    fn needs_parens(&self) -> bool {
        matches!(self, Formula::BoolBin { .. } | Formula::Until { .. }
            | Formula::Release { .. } | Formula::BoundedUntil { .. })
    }
}

impl Default for FormulaPrinter {
    fn default() -> Self { Self::new() }
}

// ───────────────────────────────────────────────────────────────────────────────
// FormulaParser — recursive descent from string
// ───────────────────────────────────────────────────────────────────────────────

/// Recursive-descent parser for QCTL_F formula strings.
///
/// Grammar (informal):
/// ```text
/// formula  := atom | "true" | "false" | "!" formula | "(" formula ")"
///           | formula boolop formula
///           | path_quant temp_op formula
///           | path_quant "[" formula "U" formula "]"
///           | "P" "[" cmp number "]" "(" formula ")"
///           | "mu" var "." formula | "nu" var "." formula
/// boolop   := "&&" | "||" | "->" | "<->"
/// cmp      := ">=" | ">" | "<=" | "<" | "="
/// path_q   := "A" | "E"
/// temp_op  := "X" | "G" | "F"
/// ```
pub struct FormulaParser {
    tokens: Vec<Token>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    LBrack,
    RBrack,
    Not,
    And,
    Or,
    Arrow,
    Biarrow,
    PathA,
    PathE,
    TempX,
    TempG,
    TempF,
    TempU,
    TempR,
    ProbP,
    ExpE,
    Mu,
    Nu,
    Dot,
    Comma,
    Cmp(ComparisonOp),
    Number(f64),
    Ident(String),
    True,
    False,
    Graded(u32),
    BoundedU(u32),
    Eof,
}

impl FormulaParser {
    /// Parse a formula from a string.
    pub fn parse(input: &str) -> Result<Formula, String> {
        let tokens = Self::tokenize(input)?;
        let mut parser = FormulaParser { tokens, pos: 0 };
        let formula = parser.parse_formula()?;
        if !parser.at_end() {
            return Err(format!("Unexpected token at position {}: {:?}", parser.pos, parser.peek()));
        }
        Ok(formula)
    }

    fn tokenize(input: &str) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            match chars[i] {
                ' ' | '\t' | '\n' | '\r' => { i += 1; }
                '(' => { tokens.push(Token::LParen); i += 1; }
                ')' => { tokens.push(Token::RParen); i += 1; }
                '[' => { tokens.push(Token::LBrack); i += 1; }
                ']' => { tokens.push(Token::RBrack); i += 1; }
                '!' | '¬' => { tokens.push(Token::Not); i += 1; }
                '.' => { tokens.push(Token::Dot); i += 1; }
                ',' => { tokens.push(Token::Comma); i += 1; }
                '&' => {
                    if i + 1 < chars.len() && chars[i + 1] == '&' {
                        tokens.push(Token::And);
                        i += 2;
                    } else {
                        return Err(format!("Expected '&&' at position {}", i));
                    }
                }
                '|' => {
                    if i + 1 < chars.len() && chars[i + 1] == '|' {
                        tokens.push(Token::Or);
                        i += 2;
                    } else {
                        return Err(format!("Expected '||' at position {}", i));
                    }
                }
                '-' => {
                    if i + 1 < chars.len() && chars[i + 1] == '>' {
                        tokens.push(Token::Arrow);
                        i += 2;
                    } else if i + 1 < chars.len() && chars[i+1].is_ascii_digit() {
                        // Negative number
                        let start = i;
                        i += 1;
                        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                            i += 1;
                        }
                        let num_str: String = chars[start..i].iter().collect();
                        let val: f64 = num_str.parse().map_err(|e| format!("Bad number: {}", e))?;
                        tokens.push(Token::Number(val));
                    } else {
                        return Err(format!("Unexpected '-' at position {}", i));
                    }
                }
                '<' => {
                    if i + 2 < chars.len() && chars[i + 1] == '-' && chars[i + 2] == '>' {
                        tokens.push(Token::Biarrow);
                        i += 3;
                    } else if i + 1 < chars.len() && chars[i + 1] == '=' {
                        tokens.push(Token::Cmp(ComparisonOp::Le));
                        i += 2;
                    } else {
                        tokens.push(Token::Cmp(ComparisonOp::Lt));
                        i += 1;
                    }
                }
                '>' => {
                    if i + 1 < chars.len() && chars[i + 1] == '=' {
                        tokens.push(Token::Cmp(ComparisonOp::Ge));
                        i += 2;
                    } else {
                        tokens.push(Token::Cmp(ComparisonOp::Gt));
                        i += 1;
                    }
                }
                '=' => {
                    tokens.push(Token::Cmp(ComparisonOp::Eq));
                    i += 1;
                }
                '∧' => { tokens.push(Token::And); i += 1; }
                '∨' => { tokens.push(Token::Or); i += 1; }
                '→' => { tokens.push(Token::Arrow); i += 1; }
                '↔' => { tokens.push(Token::Biarrow); i += 1; }
                'μ' => { tokens.push(Token::Mu); i += 1; }
                'ν' => { tokens.push(Token::Nu); i += 1; }
                '⊤' => { tokens.push(Token::True); i += 1; }
                '⊥' => { tokens.push(Token::False); i += 1; }
                c if c.is_ascii_digit() => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let val: f64 = num_str.parse().map_err(|e| format!("Bad number: {}", e))?;
                    tokens.push(Token::Number(val));
                }
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    let word: String = chars[start..i].iter().collect();
                    match word.as_str() {
                        "true" => tokens.push(Token::True),
                        "false" => tokens.push(Token::False),
                        "A" => tokens.push(Token::PathA),
                        "E" => tokens.push(Token::PathE),
                        "X" => tokens.push(Token::TempX),
                        "G" => tokens.push(Token::TempG),
                        "F" => tokens.push(Token::TempF),
                        "U" => tokens.push(Token::TempU),
                        "R" => tokens.push(Token::TempR),
                        "P" => tokens.push(Token::ProbP),
                        "mu" => tokens.push(Token::Mu),
                        "nu" => tokens.push(Token::Nu),
                        "AX" => { tokens.push(Token::PathA); tokens.push(Token::TempX); }
                        "EX" => { tokens.push(Token::PathE); tokens.push(Token::TempX); }
                        "AG" => { tokens.push(Token::PathA); tokens.push(Token::TempG); }
                        "EG" => { tokens.push(Token::PathE); tokens.push(Token::TempG); }
                        "AF" => { tokens.push(Token::PathA); tokens.push(Token::TempF); }
                        "EF" => { tokens.push(Token::PathE); tokens.push(Token::TempF); }
                        "AU" => { tokens.push(Token::PathA); tokens.push(Token::TempU); }
                        "EU" => { tokens.push(Token::PathE); tokens.push(Token::TempU); }
                        _ => tokens.push(Token::Ident(word)),
                    }
                }
                other => {
                    // skip unknown unicode
                    i += 1;
                    if other == '⟨' || other == '⟩' {
                        // graded modality markers, skip
                    }
                }
            }
        }
        tokens.push(Token::Eof);
        Ok(tokens)
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        let tok = self.advance();
        if &tok == expected {
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?} at position {}", expected, tok, self.pos - 1))
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.peek(), Token::Eof)
    }

    fn parse_formula(&mut self) -> Result<Formula, String> {
        self.parse_iff()
    }

    fn parse_iff(&mut self) -> Result<Formula, String> {
        let mut lhs = self.parse_implies()?;
        while matches!(self.peek(), Token::Biarrow) {
            self.advance();
            let rhs = self.parse_implies()?;
            lhs = Formula::iff(lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_implies(&mut self) -> Result<Formula, String> {
        let mut lhs = self.parse_or()?;
        while matches!(self.peek(), Token::Arrow) {
            self.advance();
            let rhs = self.parse_or()?;
            lhs = Formula::implies(lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> Result<Formula, String> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Token::Or) {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Formula::or(lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Formula, String> {
        let mut lhs = self.parse_unary()?;
        while matches!(self.peek(), Token::And) {
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Formula::and(lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Formula, String> {
        match self.peek().clone() {
            Token::Not => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(Formula::not(inner))
            }
            Token::PathA | Token::PathE => {
                let q = if matches!(self.peek(), Token::PathA) {
                    PathQuantifier::All
                } else {
                    PathQuantifier::Exists
                };
                self.advance();
                self.parse_path_formula(q)
            }
            Token::ProbP => {
                self.advance();
                self.parse_prob_bound()
            }
            Token::Mu => {
                self.advance();
                self.parse_fixedpoint(true)
            }
            Token::Nu => {
                self.advance();
                self.parse_fixedpoint(false)
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_path_formula(&mut self, q: PathQuantifier) -> Result<Formula, String> {
        match self.peek().clone() {
            Token::TempX => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(match q {
                    PathQuantifier::All => Formula::ax(inner),
                    PathQuantifier::Exists => Formula::ex(inner),
                })
            }
            Token::TempG => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(match q {
                    PathQuantifier::All => Formula::ag(inner),
                    PathQuantifier::Exists => Formula::eg(inner),
                })
            }
            Token::TempF => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(match q {
                    PathQuantifier::All => Formula::af(inner),
                    PathQuantifier::Exists => Formula::ef(inner),
                })
            }
            Token::LBrack => {
                self.advance();
                let hold = self.parse_formula()?;
                match self.peek().clone() {
                    Token::TempU => {
                        self.advance();
                        let goal = self.parse_formula()?;
                        self.expect(&Token::RBrack)?;
                        Ok(match q {
                            PathQuantifier::All => Formula::au(hold, goal),
                            PathQuantifier::Exists => Formula::eu(hold, goal),
                        })
                    }
                    Token::TempR => {
                        self.advance();
                        let inv = self.parse_formula()?;
                        self.expect(&Token::RBrack)?;
                        Ok(match q {
                            PathQuantifier::All => Formula::ar(hold, inv),
                            PathQuantifier::Exists => Formula::er(hold, inv),
                        })
                    }
                    other => Err(format!("Expected U or R in path formula, got {:?}", other)),
                }
            }
            _ => {
                // Try treating as state formula with the quantifier as an identifier
                Err(format!("Expected temporal operator after path quantifier {:?}, got {:?}", q, self.peek()))
            }
        }
    }

    fn parse_prob_bound(&mut self) -> Result<Formula, String> {
        self.expect(&Token::LBrack)?;
        let op = match self.advance() {
            Token::Cmp(op) => op,
            other => return Err(format!("Expected comparison in P[...], got {:?}", other)),
        };
        let threshold = match self.advance() {
            Token::Number(n) => n,
            other => return Err(format!("Expected number in P[...], got {:?}", other)),
        };
        self.expect(&Token::RBrack)?;
        self.expect(&Token::LParen)?;
        let inner = self.parse_formula()?;
        self.expect(&Token::RParen)?;
        Ok(Formula::ProbBound { op, threshold, inner: Box::new(inner) })
    }

    fn parse_fixedpoint(&mut self, is_least: bool) -> Result<Formula, String> {
        let var = match self.advance() {
            Token::Ident(name) => name,
            other => return Err(format!("Expected variable name after μ/ν, got {:?}", other)),
        };
        self.expect(&Token::Dot)?;
        let body = self.parse_formula()?;
        Ok(Formula::FixedPoint { is_least, variable: var, body: Box::new(body) })
    }

    fn parse_primary(&mut self) -> Result<Formula, String> {
        match self.peek().clone() {
            Token::True => { self.advance(); Ok(Formula::top()) }
            Token::False => { self.advance(); Ok(Formula::bot()) }
            Token::Number(v) => { self.advance(); Ok(Formula::qval(v)) }
            Token::Ident(name) => { self.advance(); Ok(Formula::atom(name)) }
            Token::LParen => {
                self.advance();
                let inner = self.parse_formula()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(format!("Unexpected token: {:?}", other)),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// FormulaSimplifier
// ───────────────────────────────────────────────────────────────────────────────

/// Simplifies QCTL_F formulas via algebraic rewriting.
pub struct FormulaSimplifier {
    max_iterations: usize,
}

impl FormulaSimplifier {
    pub fn new() -> Self {
        Self { max_iterations: 100 }
    }

    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self { max_iterations }
    }

    /// Apply simplification rules until a fixed point is reached.
    pub fn simplify(&self, formula: &Formula) -> Formula {
        let mut current = formula.clone();
        for _ in 0..self.max_iterations {
            let next = self.simplify_once(&current);
            if next == current {
                break;
            }
            current = next;
        }
        current
    }

    fn simplify_once(&self, formula: &Formula) -> Formula {
        match formula {
            // ¬¬φ => φ
            Formula::Not(inner) => match inner.as_ref() {
                Formula::Not(inner2) => self.simplify_once(inner2),
                Formula::Bool(b) => Formula::Bool(!b),
                _ => Formula::not(self.simplify_once(inner)),
            },

            // φ ∧ ⊤ => φ, φ ∧ ⊥ => ⊥, φ ∧ φ => φ
            Formula::BoolBin { op: BoolOp::And, lhs, rhs } => {
                let l = self.simplify_once(lhs);
                let r = self.simplify_once(rhs);
                match (&l, &r) {
                    (Formula::Bool(true), _) => r,
                    (_, Formula::Bool(true)) => l,
                    (Formula::Bool(false), _) | (_, Formula::Bool(false)) => Formula::bot(),
                    _ if l == r => l,
                    _ => Formula::and(l, r),
                }
            }

            // φ ∨ ⊥ => φ, φ ∨ ⊤ => ⊤, φ ∨ φ => φ
            Formula::BoolBin { op: BoolOp::Or, lhs, rhs } => {
                let l = self.simplify_once(lhs);
                let r = self.simplify_once(rhs);
                match (&l, &r) {
                    (Formula::Bool(false), _) => r,
                    (_, Formula::Bool(false)) => l,
                    (Formula::Bool(true), _) | (_, Formula::Bool(true)) => Formula::top(),
                    _ if l == r => l,
                    _ => Formula::or(l, r),
                }
            }

            // φ → ψ => ¬φ ∨ ψ  (kept as implication but simplified)
            Formula::BoolBin { op: BoolOp::Implies, lhs, rhs } => {
                let l = self.simplify_once(lhs);
                let r = self.simplify_once(rhs);
                match (&l, &r) {
                    (Formula::Bool(false), _) => Formula::top(),
                    (Formula::Bool(true), _) => r,
                    (_, Formula::Bool(true)) => Formula::top(),
                    (_, Formula::Bool(false)) => Formula::not(l),
                    _ => Formula::implies(l, r),
                }
            }

            Formula::BoolBin { op: BoolOp::Iff, lhs, rhs } => {
                let l = self.simplify_once(lhs);
                let r = self.simplify_once(rhs);
                match (&l, &r) {
                    (Formula::Bool(true), _) => r,
                    (_, Formula::Bool(true)) => l,
                    (Formula::Bool(false), _) => Formula::not(r),
                    (_, Formula::Bool(false)) => Formula::not(l),
                    _ if l == r => Formula::top(),
                    _ => Formula::iff(l, r),
                }
            }

            // P[>=0](φ) => ⊤, P[<=1](φ) => ⊤
            Formula::ProbBound { op, threshold, inner } => {
                let inner_s = self.simplify_once(inner);
                match (op, *threshold) {
                    (ComparisonOp::Ge, t) if t <= 0.0 => Formula::top(),
                    (ComparisonOp::Le, t) if t >= 1.0 => Formula::top(),
                    (ComparisonOp::Gt, t) if t >= 1.0 => Formula::bot(),
                    (ComparisonOp::Lt, t) if t <= 0.0 => Formula::bot(),
                    _ => Formula::ProbBound { op: *op, threshold: *threshold, inner: Box::new(inner_s) },
                }
            }

            // AG ⊤ => ⊤, EF ⊥ => ⊥
            Formula::Globally { quantifier, inner } => {
                let inner_s = self.simplify_once(inner);
                match &inner_s {
                    Formula::Bool(true) => Formula::top(),
                    _ => Formula::Globally { quantifier: *quantifier, inner: Box::new(inner_s) },
                }
            }

            Formula::Finally { quantifier, inner } => {
                let inner_s = self.simplify_once(inner);
                match &inner_s {
                    Formula::Bool(false) => Formula::bot(),
                    Formula::Bool(true) => Formula::top(),
                    _ => Formula::Finally { quantifier: *quantifier, inner: Box::new(inner_s) },
                }
            }

            Formula::Next { quantifier, inner } => {
                let inner_s = self.simplify_once(inner);
                Formula::Next { quantifier: *quantifier, inner: Box::new(inner_s) }
            }

            Formula::Until { quantifier, hold, goal } => {
                let h = self.simplify_once(hold);
                let g = self.simplify_once(goal);
                // φ U ⊤ => ⊤
                match &g {
                    Formula::Bool(true) => Formula::top(),
                    Formula::Bool(false) => Formula::bot(),
                    _ => Formula::Until { quantifier: *quantifier, hold: Box::new(h), goal: Box::new(g) },
                }
            }

            Formula::Release { quantifier, trigger, invariant } => {
                let t = self.simplify_once(trigger);
                let inv = self.simplify_once(invariant);
                Formula::Release { quantifier: *quantifier, trigger: Box::new(t), invariant: Box::new(inv) }
            }

            Formula::BoundedUntil { quantifier, hold, goal, bound } => {
                let h = self.simplify_once(hold);
                let g = self.simplify_once(goal);
                if *bound == 0 {
                    g
                } else {
                    Formula::BoundedUntil {
                        quantifier: *quantifier,
                        hold: Box::new(h),
                        goal: Box::new(g),
                        bound: *bound,
                    }
                }
            }

            Formula::ExpBound { op, threshold, temporal, inner } => {
                let inner_s = self.simplify_once(inner);
                Formula::ExpBound { op: *op, threshold: *threshold, temporal: *temporal, inner: Box::new(inner_s) }
            }

            Formula::GradedModality { grade, inner } => {
                let inner_s = self.simplify_once(inner);
                Formula::GradedModality { grade: *grade, inner: Box::new(inner_s) }
            }

            Formula::FixedPoint { is_least, variable, body } => {
                let body_s = self.simplify_once(body);
                // If the body doesn't reference the variable, the fixed point is trivial
                if !body_s.atoms().contains(variable) && !contains_var(&body_s, variable) {
                    body_s
                } else {
                    Formula::FixedPoint {
                        is_least: *is_least,
                        variable: variable.clone(),
                        body: Box::new(body_s),
                    }
                }
            }

            other => other.clone(),
        }
    }
}

impl Default for FormulaSimplifier {
    fn default() -> Self { Self::new() }
}

/// Check if a formula contains a free variable reference.
fn contains_var(formula: &Formula, var: &str) -> bool {
    match formula {
        Formula::Var(v) => v == var,
        Formula::Bool(_) | Formula::Atom(_) | Formula::QVal(_) => false,
        Formula::Not(inner) => contains_var(inner, var),
        Formula::BoolBin { lhs, rhs, .. } => contains_var(lhs, var) || contains_var(rhs, var),
        Formula::Next { inner, .. } | Formula::Globally { inner, .. }
        | Formula::Finally { inner, .. } | Formula::ProbBound { inner, .. }
        | Formula::ExpBound { inner, .. } | Formula::GradedModality { inner, .. } => {
            contains_var(inner, var)
        }
        Formula::Until { hold, goal, .. } | Formula::Release { trigger: hold, invariant: goal, .. }
        | Formula::BoundedUntil { hold, goal, .. } => {
            contains_var(hold, var) || contains_var(goal, var)
        }
        Formula::FixedPoint { variable, body, .. } => {
            if variable == var { false } else { contains_var(body, var) }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Derived CTL equivalences / rewriting
// ───────────────────────────────────────────────────────────────────────────────

impl Formula {
    /// Rewrite AF, EF, AG, EG in terms of fixed-point operators.
    /// AF φ = μX.(φ ∨ AX X)
    /// EF φ = μX.(φ ∨ EX X)
    /// AG φ = νX.(φ ∧ AX X)
    /// EG φ = νX.(φ ∧ EX X)
    pub fn to_fixedpoint(&self) -> Formula {
        let fresh = format!("_fp_{}", &Uuid::new_v4().to_string()[..8]);
        match self {
            Formula::Finally { quantifier, inner } => {
                let inner_fp = inner.to_fixedpoint();
                Formula::mu(
                    fresh.clone(),
                    Formula::or(
                        inner_fp,
                        match quantifier {
                            PathQuantifier::All => Formula::ax(Formula::var(&fresh)),
                            PathQuantifier::Exists => Formula::ex(Formula::var(&fresh)),
                        },
                    ),
                )
            }
            Formula::Globally { quantifier, inner } => {
                let inner_fp = inner.to_fixedpoint();
                Formula::nu(
                    fresh.clone(),
                    Formula::and(
                        inner_fp,
                        match quantifier {
                            PathQuantifier::All => Formula::ax(Formula::var(&fresh)),
                            PathQuantifier::Exists => Formula::ex(Formula::var(&fresh)),
                        },
                    ),
                )
            }
            Formula::Until { quantifier, hold, goal } => {
                let hold_fp = hold.to_fixedpoint();
                let goal_fp = goal.to_fixedpoint();
                Formula::mu(
                    fresh.clone(),
                    Formula::or(
                        goal_fp,
                        Formula::and(
                            hold_fp,
                            match quantifier {
                                PathQuantifier::All => Formula::ax(Formula::var(&fresh)),
                                PathQuantifier::Exists => Formula::ex(Formula::var(&fresh)),
                            },
                        ),
                    ),
                )
            }
            Formula::Not(inner) => Formula::not(inner.to_fixedpoint()),
            Formula::BoolBin { op, lhs, rhs } => Formula::BoolBin {
                op: *op,
                lhs: Box::new(lhs.to_fixedpoint()),
                rhs: Box::new(rhs.to_fixedpoint()),
            },
            Formula::Next { quantifier, inner } => Formula::Next {
                quantifier: *quantifier,
                inner: Box::new(inner.to_fixedpoint()),
            },
            Formula::ProbBound { op, threshold, inner } => Formula::ProbBound {
                op: *op,
                threshold: *threshold,
                inner: Box::new(inner.to_fixedpoint()),
            },
            Formula::ExpBound { op, threshold, temporal, inner } => Formula::ExpBound {
                op: *op,
                threshold: *threshold,
                temporal: *temporal,
                inner: Box::new(inner.to_fixedpoint()),
            },
            Formula::GradedModality { grade, inner } => Formula::GradedModality {
                grade: *grade,
                inner: Box::new(inner.to_fixedpoint()),
            },
            Formula::Release { quantifier, trigger, invariant } => {
                let t_fp = trigger.to_fixedpoint();
                let i_fp = invariant.to_fixedpoint();
                Formula::nu(
                    fresh.clone(),
                    Formula::and(
                        i_fp,
                        Formula::or(
                            t_fp,
                            match quantifier {
                                PathQuantifier::All => Formula::ax(Formula::var(&fresh)),
                                PathQuantifier::Exists => Formula::ex(Formula::var(&fresh)),
                            },
                        ),
                    ),
                )
            }
            Formula::BoundedUntil { quantifier, hold, goal, bound } => {
                // Unroll: φ U≤0 ψ = ψ, φ U≤(n+1) ψ = ψ ∨ (φ ∧ QX(φ U≤n ψ))
                self.unroll_bounded_until(*quantifier, hold, goal, *bound)
            }
            other => other.clone(),
        }
    }

    fn unroll_bounded_until(
        &self,
        q: PathQuantifier,
        hold: &Formula,
        goal: &Formula,
        bound: u32,
    ) -> Formula {
        if bound == 0 {
            return goal.to_fixedpoint();
        }
        let inner = Formula::bounded_until(q, hold.clone(), goal.clone(), bound - 1);
        let inner_fp = inner.to_fixedpoint();
        let hold_fp = hold.to_fixedpoint();
        let goal_fp = goal.to_fixedpoint();
        Formula::or(
            goal_fp,
            Formula::and(
                hold_fp,
                match q {
                    PathQuantifier::All => Formula::ax(inner_fp),
                    PathQuantifier::Exists => Formula::ex(inner_fp),
                },
            ),
        )
    }

    /// Map over all subformulas bottom-up.
    pub fn map_bottom_up<F: Fn(Formula) -> Formula>(&self, f: &F) -> Formula {
        let mapped = match self {
            Formula::Bool(_) | Formula::Atom(_) | Formula::QVal(_) | Formula::Var(_) => self.clone(),
            Formula::Not(inner) => Formula::not(inner.map_bottom_up(f)),
            Formula::BoolBin { op, lhs, rhs } => Formula::BoolBin {
                op: *op,
                lhs: Box::new(lhs.map_bottom_up(f)),
                rhs: Box::new(rhs.map_bottom_up(f)),
            },
            Formula::Next { quantifier, inner } => Formula::Next {
                quantifier: *quantifier,
                inner: Box::new(inner.map_bottom_up(f)),
            },
            Formula::Globally { quantifier, inner } => Formula::Globally {
                quantifier: *quantifier,
                inner: Box::new(inner.map_bottom_up(f)),
            },
            Formula::Finally { quantifier, inner } => Formula::Finally {
                quantifier: *quantifier,
                inner: Box::new(inner.map_bottom_up(f)),
            },
            Formula::Until { quantifier, hold, goal } => Formula::Until {
                quantifier: *quantifier,
                hold: Box::new(hold.map_bottom_up(f)),
                goal: Box::new(goal.map_bottom_up(f)),
            },
            Formula::Release { quantifier, trigger, invariant } => Formula::Release {
                quantifier: *quantifier,
                trigger: Box::new(trigger.map_bottom_up(f)),
                invariant: Box::new(invariant.map_bottom_up(f)),
            },
            Formula::BoundedUntil { quantifier, hold, goal, bound } => Formula::BoundedUntil {
                quantifier: *quantifier,
                hold: Box::new(hold.map_bottom_up(f)),
                goal: Box::new(goal.map_bottom_up(f)),
                bound: *bound,
            },
            Formula::ProbBound { op, threshold, inner } => Formula::ProbBound {
                op: *op,
                threshold: *threshold,
                inner: Box::new(inner.map_bottom_up(f)),
            },
            Formula::ExpBound { op, threshold, temporal, inner } => Formula::ExpBound {
                op: *op,
                threshold: *threshold,
                temporal: *temporal,
                inner: Box::new(inner.map_bottom_up(f)),
            },
            Formula::GradedModality { grade, inner } => Formula::GradedModality {
                grade: *grade,
                inner: Box::new(inner.map_bottom_up(f)),
            },
            Formula::FixedPoint { is_least, variable, body } => Formula::FixedPoint {
                is_least: *is_least,
                variable: variable.clone(),
                body: Box::new(body.map_bottom_up(f)),
            },
        };
        f(mapped)
    }

    /// Check if the formula is in positive normal form (negations only on atoms).
    pub fn is_positive_normal_form(&self) -> bool {
        match self {
            Formula::Not(inner) => matches!(inner.as_ref(), Formula::Atom(_)),
            Formula::Bool(_) | Formula::Atom(_) | Formula::QVal(_) | Formula::Var(_) => true,
            Formula::BoolBin { lhs, rhs, .. } => {
                lhs.is_positive_normal_form() && rhs.is_positive_normal_form()
            }
            Formula::Next { inner, .. } | Formula::Globally { inner, .. }
            | Formula::Finally { inner, .. } | Formula::ProbBound { inner, .. }
            | Formula::ExpBound { inner, .. } | Formula::GradedModality { inner, .. } => {
                inner.is_positive_normal_form()
            }
            Formula::Until { hold, goal, .. } | Formula::Release { trigger: hold, invariant: goal, .. }
            | Formula::BoundedUntil { hold, goal, .. } => {
                hold.is_positive_normal_form() && goal.is_positive_normal_form()
            }
            Formula::FixedPoint { body, .. } => body.is_positive_normal_form(),
        }
    }

    /// Check whether this is a "flat" (non-nested temporal) formula.
    pub fn is_flat(&self) -> bool {
        match self {
            Formula::Next { inner, .. } | Formula::Globally { inner, .. }
            | Formula::Finally { inner, .. } => inner.info().temporal_count == 0,
            Formula::Until { hold, goal, .. } | Formula::Release { trigger: hold, invariant: goal, .. }
            | Formula::BoundedUntil { hold, goal, .. } => {
                hold.info().temporal_count == 0 && goal.info().temporal_count == 0
            }
            Formula::BoolBin { lhs, rhs, .. } => lhs.is_flat() && rhs.is_flat(),
            Formula::Not(inner) => inner.is_flat(),
            Formula::ProbBound { inner, .. } | Formula::ExpBound { inner, .. }
            | Formula::GradedModality { inner, .. } => inner.is_flat(),
            _ => true,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Hash for Formula (needed for sets/maps keyed by formula)
// ───────────────────────────────────────────────────────────────────────────────

impl Eq for Formula {}

impl std::hash::Hash for Formula {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Discriminant
        std::mem::discriminant(self).hash(state);
        match self {
            Formula::Bool(b) => b.hash(state),
            Formula::Atom(name) => name.hash(state),
            Formula::QVal(v) => v.hash(state),
            Formula::Var(v) => v.hash(state),
            Formula::Not(inner) => inner.hash(state),
            Formula::BoolBin { op, lhs, rhs } => {
                op.hash(state); lhs.hash(state); rhs.hash(state);
            }
            Formula::Next { quantifier, inner } => { quantifier.hash(state); inner.hash(state); }
            Formula::Globally { quantifier, inner } => { quantifier.hash(state); inner.hash(state); }
            Formula::Finally { quantifier, inner } => { quantifier.hash(state); inner.hash(state); }
            Formula::Until { quantifier, hold, goal } => {
                quantifier.hash(state); hold.hash(state); goal.hash(state);
            }
            Formula::Release { quantifier, trigger, invariant } => {
                quantifier.hash(state); trigger.hash(state); invariant.hash(state);
            }
            Formula::BoundedUntil { quantifier, hold, goal, bound } => {
                quantifier.hash(state); hold.hash(state); goal.hash(state); bound.hash(state);
            }
            Formula::ProbBound { op, threshold, inner } => {
                op.hash(state); OrderedFloat(*threshold).hash(state); inner.hash(state);
            }
            Formula::ExpBound { op, threshold, temporal, inner } => {
                op.hash(state); OrderedFloat(*threshold).hash(state); temporal.hash(state); inner.hash(state);
            }
            Formula::GradedModality { grade, inner } => { grade.hash(state); inner.hash(state); }
            Formula::FixedPoint { is_least, variable, body } => {
                is_least.hash(state); variable.hash(state); body.hash(state);
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── constructors ──

    #[test]
    fn test_basic_constructors() {
        let t = Formula::top();
        let f = Formula::bot();
        let p = Formula::atom("safe");
        assert_eq!(t, Formula::Bool(true));
        assert_eq!(f, Formula::Bool(false));
        assert_eq!(p, Formula::Atom("safe".to_string()));
    }

    #[test]
    fn test_temporal_constructors() {
        let phi = Formula::atom("refusal");
        let ax = Formula::ax(phi.clone());
        let ef = Formula::ef(phi.clone());
        assert!(matches!(ax, Formula::Next { quantifier: PathQuantifier::All, .. }));
        assert!(matches!(ef, Formula::Finally { quantifier: PathQuantifier::Exists, .. }));
    }

    #[test]
    fn test_until_constructor() {
        let hold = Formula::atom("safe");
        let goal = Formula::atom("done");
        let au = Formula::au(hold, goal);
        assert!(matches!(au, Formula::Until { quantifier: PathQuantifier::All, .. }));
    }

    #[test]
    fn test_quantitative_constructors() {
        let phi = Formula::atom("refusal");
        let pb = Formula::prob_ge(0.95, phi);
        match &pb {
            Formula::ProbBound { op, threshold, .. } => {
                assert_eq!(*op, ComparisonOp::Ge);
                assert!((threshold - 0.95).abs() < 1e-10);
            }
            _ => panic!("Expected ProbBound"),
        }
    }

    // ── info ──

    #[test]
    fn test_formula_info_atom() {
        let f = Formula::atom("p");
        let info = f.info();
        assert_eq!(info.size, 1);
        assert_eq!(info.depth, 0);
        assert!(info.atoms.contains("p"));
        assert_eq!(info.temporal_count, 0);
    }

    #[test]
    fn test_formula_info_complex() {
        // AG(p ∧ AX q)
        let f = Formula::ag(Formula::and(
            Formula::atom("p"),
            Formula::ax(Formula::atom("q")),
        ));
        let info = f.info();
        assert_eq!(info.size, 5); // AG, AND, p, AX, q
        assert!(info.atoms.contains("p"));
        assert!(info.atoms.contains("q"));
        assert_eq!(info.temporal_count, 2); // AG and AX
    }

    #[test]
    fn test_formula_depth() {
        // ¬(p ∧ q)
        let f = Formula::not(Formula::and(Formula::atom("p"), Formula::atom("q")));
        assert_eq!(f.depth(), 2); // NOT -> AND -> p/q
    }

    // ── subformulas ──

    #[test]
    fn test_subformulas() {
        let f = Formula::and(Formula::atom("p"), Formula::atom("q"));
        let subs = f.subformulas();
        assert_eq!(subs.len(), 3); // AND, p, q
    }

    #[test]
    fn test_subformulas_nested() {
        let f = Formula::ag(Formula::implies(Formula::atom("a"), Formula::ef(Formula::atom("b"))));
        let subs = f.subformulas();
        assert!(subs.len() >= 5);
    }

    // ── simplification ──

    #[test]
    fn test_simplify_double_negation() {
        let f = Formula::not(Formula::not(Formula::atom("p")));
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::atom("p"));
    }

    #[test]
    fn test_simplify_and_true() {
        let f = Formula::and(Formula::atom("p"), Formula::top());
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::atom("p"));
    }

    #[test]
    fn test_simplify_and_false() {
        let f = Formula::and(Formula::atom("p"), Formula::bot());
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::bot());
    }

    #[test]
    fn test_simplify_or_true() {
        let f = Formula::or(Formula::atom("p"), Formula::top());
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::top());
    }

    #[test]
    fn test_simplify_or_false() {
        let f = Formula::or(Formula::bot(), Formula::atom("q"));
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::atom("q"));
    }

    #[test]
    fn test_simplify_idempotent() {
        let f = Formula::and(Formula::atom("p"), Formula::atom("p"));
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::atom("p"));
    }

    #[test]
    fn test_simplify_implies_false() {
        let f = Formula::implies(Formula::bot(), Formula::atom("p"));
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::top());
    }

    #[test]
    fn test_simplify_prob_trivial() {
        let f = Formula::ProbBound {
            op: ComparisonOp::Ge,
            threshold: 0.0,
            inner: Box::new(Formula::atom("p")),
        };
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::top());
    }

    #[test]
    fn test_simplify_globally_true() {
        let f = Formula::ag(Formula::top());
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::top());
    }

    #[test]
    fn test_simplify_finally_false() {
        let f = Formula::ef(Formula::bot());
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::bot());
    }

    #[test]
    fn test_simplify_bounded_until_zero() {
        let f = Formula::bounded_until(PathQuantifier::All, Formula::atom("p"), Formula::atom("q"), 0);
        let s = FormulaSimplifier::new().simplify(&f);
        assert_eq!(s, Formula::atom("q"));
    }

    // ── NNF ──

    #[test]
    fn test_nnf_atom() {
        let f = Formula::not(Formula::atom("p"));
        let nnf = f.to_nnf();
        // ¬p should remain as ¬p (atom negation is NNF)
        assert!(matches!(nnf, Formula::Not(_)));
    }

    #[test]
    fn test_nnf_double_neg() {
        let f = Formula::not(Formula::not(Formula::atom("p")));
        let nnf = f.to_nnf();
        assert_eq!(nnf, Formula::atom("p"));
    }

    #[test]
    fn test_nnf_demorgan() {
        // ¬(p ∧ q) => ¬p ∨ ¬q
        let f = Formula::not(Formula::and(Formula::atom("p"), Formula::atom("q")));
        let nnf = f.to_nnf();
        match nnf {
            Formula::BoolBin { op: BoolOp::Or, .. } => {} // correct
            _ => panic!("Expected disjunction from De Morgan, got {:?}", nnf),
        }
    }

    #[test]
    fn test_nnf_temporal_dual() {
        // ¬AG p => EF ¬p
        let f = Formula::not(Formula::ag(Formula::atom("p")));
        let nnf = f.to_nnf();
        assert!(matches!(nnf, Formula::Finally { quantifier: PathQuantifier::Exists, .. }));
    }

    // ── parsing ──

    #[test]
    fn test_parse_atom() {
        let f = FormulaParser::parse("safe").unwrap();
        assert_eq!(f, Formula::atom("safe"));
    }

    #[test]
    fn test_parse_true_false() {
        assert_eq!(FormulaParser::parse("true").unwrap(), Formula::top());
        assert_eq!(FormulaParser::parse("false").unwrap(), Formula::bot());
    }

    #[test]
    fn test_parse_negation() {
        let f = FormulaParser::parse("!safe").unwrap();
        assert_eq!(f, Formula::not(Formula::atom("safe")));
    }

    #[test]
    fn test_parse_and() {
        let f = FormulaParser::parse("safe && compliant").unwrap();
        assert_eq!(f, Formula::and(Formula::atom("safe"), Formula::atom("compliant")));
    }

    #[test]
    fn test_parse_or() {
        let f = FormulaParser::parse("safe || toxic").unwrap();
        assert_eq!(f, Formula::or(Formula::atom("safe"), Formula::atom("toxic")));
    }

    #[test]
    fn test_parse_implies() {
        let f = FormulaParser::parse("toxic -> refused").unwrap();
        assert_eq!(f, Formula::implies(Formula::atom("toxic"), Formula::atom("refused")));
    }

    #[test]
    fn test_parse_temporal_ax() {
        let f = FormulaParser::parse("AX safe").unwrap();
        assert_eq!(f, Formula::ax(Formula::atom("safe")));
    }

    #[test]
    fn test_parse_temporal_ef() {
        let f = FormulaParser::parse("EF done").unwrap();
        assert_eq!(f, Formula::ef(Formula::atom("done")));
    }

    #[test]
    fn test_parse_temporal_ag() {
        let f = FormulaParser::parse("AG safe").unwrap();
        assert_eq!(f, Formula::ag(Formula::atom("safe")));
    }

    #[test]
    fn test_parse_until() {
        let f = FormulaParser::parse("A[safe U done]").unwrap();
        assert_eq!(f, Formula::au(Formula::atom("safe"), Formula::atom("done")));
    }

    #[test]
    fn test_parse_prob_bound() {
        let f = FormulaParser::parse("P[>=0.95](safe)").unwrap();
        match &f {
            Formula::ProbBound { op, threshold, inner } => {
                assert_eq!(*op, ComparisonOp::Ge);
                assert!((threshold - 0.95).abs() < 1e-10);
                assert_eq!(inner.as_ref(), &Formula::atom("safe"));
            }
            _ => panic!("Expected ProbBound, got {:?}", f),
        }
    }

    #[test]
    fn test_parse_fixedpoint() {
        let f = FormulaParser::parse("mu X. (safe || AX X)").unwrap();
        match &f {
            Formula::FixedPoint { is_least, variable, .. } => {
                assert!(*is_least);
                assert_eq!(variable, "X");
            }
            _ => panic!("Expected FixedPoint"),
        }
    }

    #[test]
    fn test_parse_parenthesized() {
        let f = FormulaParser::parse("(safe && (toxic || refused))").unwrap();
        match f {
            Formula::BoolBin { op: BoolOp::And, rhs, .. } => {
                assert!(matches!(*rhs, Formula::BoolBin { op: BoolOp::Or, .. }));
            }
            _ => panic!("Bad parse"),
        }
    }

    // ── pretty-printing ──

    #[test]
    fn test_display_atom() {
        assert_eq!(Formula::atom("safe").to_string(), "safe");
    }

    #[test]
    fn test_display_and() {
        let f = Formula::and(Formula::atom("a"), Formula::atom("b"));
        assert_eq!(f.to_string(), "(a ∧ b)");
    }

    #[test]
    fn test_display_ag() {
        let f = Formula::ag(Formula::atom("safe"));
        assert_eq!(f.to_string(), "AG safe");
    }

    #[test]
    fn test_display_until() {
        let f = Formula::eu(Formula::atom("a"), Formula::atom("b"));
        assert_eq!(f.to_string(), "E[a U b]");
    }

    #[test]
    fn test_display_prob() {
        let f = Formula::prob_ge(0.9, Formula::atom("safe"));
        let s = f.to_string();
        assert!(s.contains("P["));
        assert!(s.contains("safe"));
    }

    #[test]
    fn test_ascii_printer() {
        let f = Formula::and(Formula::atom("a"), Formula::atom("b"));
        let s = FormulaPrinter::ascii().to_string(&f);
        assert_eq!(s, "(a && b)");
    }

    // ── roundtrip ──

    #[test]
    fn test_parse_roundtrip_simple() {
        let formulas = vec![
            "safe",
            "true",
            "false",
            "!toxic",
            "AX safe",
            "EF done",
            "AG safe",
        ];
        for src in formulas {
            let parsed = FormulaParser::parse(src).unwrap();
            // Re-print with ASCII printer and re-parse
            let printed = FormulaPrinter::ascii().to_string(&parsed);
            let reparsed = FormulaParser::parse(&printed).unwrap();
            assert_eq!(parsed, reparsed, "Roundtrip failed for: {}", src);
        }
    }

    // ── substitution ──

    #[test]
    fn test_substitute_var() {
        let f = Formula::or(Formula::var("X"), Formula::atom("p"));
        let result = f.substitute("X", &Formula::atom("q"));
        assert_eq!(result, Formula::or(Formula::atom("q"), Formula::atom("p")));
    }

    #[test]
    fn test_substitute_shadowed() {
        let f = Formula::mu("X", Formula::or(Formula::var("X"), Formula::var("Y")));
        let result = f.substitute("X", &Formula::atom("replaced"));
        // X is bound by μ, so should NOT be replaced
        match &result {
            Formula::FixedPoint { body, .. } => {
                match body.as_ref() {
                    Formula::BoolBin { lhs, .. } => {
                        assert_eq!(lhs.as_ref(), &Formula::Var("X".to_string()));
                    }
                    _ => panic!("Expected BoolBin"),
                }
            }
            _ => panic!("Expected FixedPoint"),
        }
    }

    // ── to_fixedpoint ──

    #[test]
    fn test_af_to_fixedpoint() {
        let f = Formula::af(Formula::atom("done"));
        let fp = f.to_fixedpoint();
        assert!(matches!(fp, Formula::FixedPoint { is_least: true, .. }));
    }

    #[test]
    fn test_ag_to_fixedpoint() {
        let f = Formula::ag(Formula::atom("safe"));
        let fp = f.to_fixedpoint();
        assert!(matches!(fp, Formula::FixedPoint { is_least: false, .. }));
    }

    // ── positive normal form check ──

    #[test]
    fn test_pnf_positive() {
        let f = Formula::and(Formula::atom("p"), Formula::not(Formula::atom("q")));
        assert!(f.is_positive_normal_form());
    }

    #[test]
    fn test_pnf_negative() {
        let f = Formula::not(Formula::and(Formula::atom("p"), Formula::atom("q")));
        assert!(!f.is_positive_normal_form());
    }

    // ── is_flat ──

    #[test]
    fn test_is_flat_simple() {
        let f = Formula::ag(Formula::atom("safe"));
        assert!(f.is_flat());
    }

    #[test]
    fn test_is_flat_nested() {
        let f = Formula::ag(Formula::ef(Formula::atom("done")));
        assert!(!f.is_flat());
    }

    // ── comparison op ──

    #[test]
    fn test_comparison_evaluate() {
        assert!(ComparisonOp::Ge.evaluate(0.95, 0.9));
        assert!(!ComparisonOp::Gt.evaluate(0.9, 0.9));
        assert!(ComparisonOp::Le.evaluate(0.9, 0.95));
        assert!(ComparisonOp::Eq.evaluate(0.5, 0.5));
    }

    #[test]
    fn test_comparison_negate() {
        assert_eq!(ComparisonOp::Ge.negate(), ComparisonOp::Lt);
        assert_eq!(ComparisonOp::Lt.negate(), ComparisonOp::Ge);
    }

    // ── map_bottom_up ──

    #[test]
    fn test_map_bottom_up_rename() {
        let f = Formula::and(Formula::atom("old"), Formula::atom("other"));
        let result = f.map_bottom_up(&|node| match node {
            Formula::Atom(name) if name == "old" => Formula::atom("new"),
            other => other,
        });
        assert_eq!(result, Formula::and(Formula::atom("new"), Formula::atom("other")));
    }

    // ── graded modality ──

    #[test]
    fn test_graded_modality() {
        let f = Formula::graded(3, Formula::atom("safe"));
        match &f {
            Formula::GradedModality { grade, inner } => {
                assert_eq!(*grade, 3);
                assert_eq!(inner.as_ref(), &Formula::atom("safe"));
            }
            _ => panic!("Expected GradedModality"),
        }
    }

    // ── qval ──

    #[test]
    fn test_qval_clamp() {
        let f = Formula::qval(1.5);
        match f {
            Formula::QVal(v) => assert!((v.into_inner() - 1.0).abs() < 1e-10),
            _ => panic!("Expected QVal"),
        }
        let f2 = Formula::qval(-0.5);
        match f2 {
            Formula::QVal(v) => assert!((v.into_inner()).abs() < 1e-10),
            _ => panic!("Expected QVal"),
        }
    }

    // ── free variables ──

    #[test]
    fn test_free_vars() {
        let f = Formula::or(Formula::var("X"), Formula::var("Y"));
        let info = f.info();
        assert!(info.free_vars.contains("X"));
        assert!(info.free_vars.contains("Y"));
    }

    #[test]
    fn test_bound_var_not_free() {
        let f = Formula::mu("X", Formula::or(Formula::var("X"), Formula::atom("p")));
        let info = f.info();
        assert!(!info.free_vars.contains("X"));
    }
}
