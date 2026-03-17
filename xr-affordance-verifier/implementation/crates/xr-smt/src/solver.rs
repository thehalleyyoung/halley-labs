//! SMT solver module: internal solving engine and external solver interface.
//!
//! Provides [`InternalSolver`], a lightweight solver that handles linear
//! feasibility via bound propagation, and [`ExternalSolverInterface`] for
//! generating SMT-LIB2 scripts to be consumed by external solvers such as Z3.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use xr_types::VerifierError;

use crate::expr::{SmtDecl, SmtExpr, SmtSort};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Variable assignments from a satisfying model.
pub type Model = HashMap<String, f64>;

// ---------------------------------------------------------------------------
// SolverResult
// ---------------------------------------------------------------------------

/// Result of an SMT satisfiability check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverResult {
    /// Satisfiable — a model (variable assignment) is available.
    Sat(Model),
    /// Unsatisfiable — no assignment can satisfy the assertions.
    Unsat,
    /// The solver could not determine satisfiability; reason is provided.
    Unknown(String),
    /// The solver timed out before reaching a conclusion.
    Timeout,
}

impl SolverResult {
    /// Returns `true` when the result is [`SolverResult::Sat`].
    pub fn is_sat(&self) -> bool {
        matches!(self, SolverResult::Sat(_))
    }

    /// Returns `true` when the result is [`SolverResult::Unsat`].
    pub fn is_unsat(&self) -> bool {
        matches!(self, SolverResult::Unsat)
    }

    /// Returns a reference to the satisfying model, if available.
    pub fn model(&self) -> Option<&Model> {
        match self {
            SolverResult::Sat(m) => Some(m),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SmtSolver trait
// ---------------------------------------------------------------------------

/// Trait that every solver backend must implement.
pub trait SmtSolver {
    /// Check satisfiability of the given assertions.
    fn check_sat(&mut self, assertions: &[SmtExpr]) -> Result<SolverResult, VerifierError>;

    /// Check satisfiability with an explicit timeout (seconds).
    fn check_sat_with_timeout(
        &mut self,
        assertions: &[SmtExpr],
        timeout_secs: f64,
    ) -> Result<SolverResult, VerifierError>;

    /// Push a new assertion scope onto the solver stack.
    fn push(&mut self);

    /// Pop the most recent assertion scope from the solver stack.
    fn pop(&mut self);

    /// Reset the solver to its initial state.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// SolverStatistics
// ---------------------------------------------------------------------------

/// Accumulated statistics for solver invocations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStatistics {
    /// Total number of `check_sat` calls.
    pub num_checks: u64,
    /// Number of SAT results.
    pub num_sat: u64,
    /// Number of UNSAT results.
    pub num_unsat: u64,
    /// Number of UNKNOWN results.
    pub num_unknown: u64,
    /// Number of TIMEOUT results.
    pub num_timeout: u64,
    /// Cumulative wall-clock time spent in checks (seconds).
    pub total_time_secs: f64,
    /// Maximum wall-clock time of any single check (seconds).
    pub max_check_time_secs: f64,
}

impl SolverStatistics {
    /// Record the outcome and elapsed time of a single check.
    pub fn record_check(&mut self, result: &SolverResult, elapsed_secs: f64) {
        self.num_checks += 1;
        self.total_time_secs += elapsed_secs;
        if elapsed_secs > self.max_check_time_secs {
            self.max_check_time_secs = elapsed_secs;
        }
        match result {
            SolverResult::Sat(_) => self.num_sat += 1,
            SolverResult::Unsat => self.num_unsat += 1,
            SolverResult::Unknown(_) => self.num_unknown += 1,
            SolverResult::Timeout => self.num_timeout += 1,
        }
    }

    /// Ratio of SAT results to total checks (0.0 when no checks performed).
    pub fn sat_ratio(&self) -> f64 {
        if self.num_checks == 0 {
            0.0
        } else {
            self.num_sat as f64 / self.num_checks as f64
        }
    }

    /// Average wall-clock time per check (0.0 when no checks performed).
    pub fn avg_check_time(&self) -> f64 {
        if self.num_checks == 0 {
            0.0
        } else {
            self.total_time_secs / self.num_checks as f64
        }
    }
}

// ---------------------------------------------------------------------------
// InternalSolver
// ---------------------------------------------------------------------------

/// A lightweight, built-in solver that handles linear feasibility problems via
/// bound propagation and midpoint assignment heuristics.
///
/// For problems outside its capabilities it returns `Unknown`.
pub struct InternalSolver {
    /// Variable declarations.
    declarations: Vec<SmtDecl>,
    /// Stack of assertion scopes (push/pop).
    assertion_stack: Vec<Vec<SmtExpr>>,
    /// Accumulated solver statistics.
    statistics: SolverStatistics,
    /// Default timeout for [`SmtSolver::check_sat`] (seconds).
    timeout_secs: f64,
    /// Maximum number of bound-propagation iterations.
    max_iterations: usize,
}

impl InternalSolver {
    /// Create a new solver with sensible defaults.
    pub fn new() -> Self {
        Self {
            declarations: Vec::new(),
            assertion_stack: vec![Vec::new()],
            statistics: SolverStatistics::default(),
            timeout_secs: 30.0,
            max_iterations: 1000,
        }
    }

    /// Builder: override the default timeout (seconds).
    pub fn with_timeout(mut self, secs: f64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Builder: override the maximum number of propagation iterations.
    pub fn with_max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }

    /// Register a variable declaration with the solver.
    pub fn declare(&mut self, decl: SmtDecl) {
        self.declarations.push(decl);
    }

    /// Read-only access to accumulated statistics.
    pub fn statistics(&self) -> &SolverStatistics {
        &self.statistics
    }

    // -- internal helpers ---------------------------------------------------

    /// Collect all assertions from every active scope.
    fn all_assertions(&self) -> Vec<SmtExpr> {
        self.assertion_stack
            .iter()
            .flat_map(|s| s.iter().cloned())
            .collect()
    }

    /// Flatten top-level `And` nodes so individual conjuncts can be analysed.
    fn flatten_assertions(assertions: &[SmtExpr]) -> Vec<&SmtExpr> {
        let mut flat: Vec<&SmtExpr> = Vec::new();
        for a in assertions {
            match a {
                SmtExpr::And(children) => {
                    for c in children {
                        flat.push(c);
                    }
                }
                other => flat.push(other),
            }
        }
        flat
    }

    /// Attempt to solve the conjunction of `assertions` using bound propagation
    /// over linear constraints.
    ///
    /// Falls back to `Unknown` for non-linear or structurally complex problems.
    fn solve_linear_feasibility(
        &self,
        assertions: &[SmtExpr],
        deadline: Instant,
    ) -> Result<SolverResult, VerifierError> {
        // Collect all declared variable names.  If there are no variables at
        // all we still want to evaluate constant assertions.
        let var_names: Vec<String> = {
            let mut names: Vec<String> =
                self.declarations.iter().map(|d| d.name.clone()).collect();
            // Also gather free variables from the assertions themselves so that
            // undeclared variables are handled gracefully.
            for a in assertions {
                for v in a.free_variables() {
                    if !names.contains(&v) {
                        names.push(v);
                    }
                }
            }
            names
        };

        // Quick constant-evaluation pass: if every assertion is ground and
        // evaluates to true, return Sat with an empty model.
        if var_names.is_empty() {
            let empty = HashMap::new();
            let all_true = assertions
                .iter()
                .all(|a| a.eval_bool(&empty) == Some(true));
            return if all_true {
                Ok(SolverResult::Sat(HashMap::new()))
            } else {
                Ok(SolverResult::Unsat)
            };
        }

        // Extract initial variable bounds.
        let mut bounds = self.extract_bounds(assertions);

        // Ensure every variable has an entry (default to very wide bounds).
        for v in &var_names {
            bounds.entry(v.clone()).or_insert((-1e18, 1e18));
        }

        // Iterative bound propagation.
        for _ in 0..self.max_iterations {
            if Instant::now() >= deadline {
                return Ok(SolverResult::Timeout);
            }
            if !self.propagate_bounds(&mut bounds, assertions) {
                break;
            }
        }

        // Check for empty intervals (unsat).
        for (var, (lo, hi)) in &bounds {
            if lo > hi {
                return Ok(SolverResult::Unsat);
            }
            if lo.is_nan() || hi.is_nan() {
                return Err(VerifierError::Numeric(format!(
                    "NaN bound for variable {var}"
                )));
            }
        }

        // Try to find a satisfying assignment within the computed bounds.
        match self.find_assignment(&bounds, assertions) {
            Some(model) => Ok(SolverResult::Sat(model)),
            None => Ok(SolverResult::Unknown(
                "bound propagation could not find a satisfying assignment".into(),
            )),
        }
    }

    /// Extract simple variable bounds from assertions of the form
    /// `(Var ≤ Const)`, `(Var ≥ Const)`, etc.
    fn extract_bounds(&self, assertions: &[SmtExpr]) -> HashMap<String, (f64, f64)> {
        let mut bounds: HashMap<String, (f64, f64)> = HashMap::new();

        let flat = Self::flatten_assertions(assertions);
        for expr in flat {
            self.extract_bound_from_expr(expr, &mut bounds);
        }
        bounds
    }

    /// Analyse a single comparison expression and tighten `bounds` if possible.
    fn extract_bound_from_expr(
        &self,
        expr: &SmtExpr,
        bounds: &mut HashMap<String, (f64, f64)>,
    ) {
        match expr {
            // x <= c  or  x < c
            SmtExpr::Le(lhs, rhs) | SmtExpr::Lt(lhs, rhs) => {
                if let (SmtExpr::Var(v), SmtExpr::Const(c)) = (lhs.as_ref(), rhs.as_ref()) {
                    let entry = bounds.entry(v.clone()).or_insert((-1e18, 1e18));
                    let ub = if matches!(expr, SmtExpr::Lt(..)) {
                        *c - f64::EPSILON
                    } else {
                        *c
                    };
                    entry.1 = entry.1.min(ub);
                }
                // c <= x  →  x >= c
                if let (SmtExpr::Const(c), SmtExpr::Var(v)) = (lhs.as_ref(), rhs.as_ref()) {
                    let entry = bounds.entry(v.clone()).or_insert((-1e18, 1e18));
                    let lb = if matches!(expr, SmtExpr::Lt(..)) {
                        *c + f64::EPSILON
                    } else {
                        *c
                    };
                    entry.0 = entry.0.max(lb);
                }
            }
            // x >= c  or  x > c
            SmtExpr::Ge(lhs, rhs) | SmtExpr::Gt(lhs, rhs) => {
                if let (SmtExpr::Var(v), SmtExpr::Const(c)) = (lhs.as_ref(), rhs.as_ref()) {
                    let entry = bounds.entry(v.clone()).or_insert((-1e18, 1e18));
                    let lb = if matches!(expr, SmtExpr::Gt(..)) {
                        *c + f64::EPSILON
                    } else {
                        *c
                    };
                    entry.0 = entry.0.max(lb);
                }
                // c >= x  →  x <= c
                if let (SmtExpr::Const(c), SmtExpr::Var(v)) = (lhs.as_ref(), rhs.as_ref()) {
                    let entry = bounds.entry(v.clone()).or_insert((-1e18, 1e18));
                    let ub = if matches!(expr, SmtExpr::Gt(..)) {
                        *c - f64::EPSILON
                    } else {
                        *c
                    };
                    entry.1 = entry.1.min(ub);
                }
            }
            // x == c
            SmtExpr::Eq(lhs, rhs) => {
                if let (SmtExpr::Var(v), SmtExpr::Const(c)) = (lhs.as_ref(), rhs.as_ref()) {
                    let entry = bounds.entry(v.clone()).or_insert((-1e18, 1e18));
                    entry.0 = entry.0.max(*c);
                    entry.1 = entry.1.min(*c);
                }
                if let (SmtExpr::Const(c), SmtExpr::Var(v)) = (lhs.as_ref(), rhs.as_ref()) {
                    let entry = bounds.entry(v.clone()).or_insert((-1e18, 1e18));
                    entry.0 = entry.0.max(*c);
                    entry.1 = entry.1.min(*c);
                }
            }
            _ => {}
        }
    }

    /// Perform one round of bound propagation over linear constraints of the
    /// form `a*x + b*y + ... <= c`.
    ///
    /// Returns `true` if any bound was tightened during this pass.
    fn propagate_bounds(
        &self,
        bounds: &mut HashMap<String, (f64, f64)>,
        assertions: &[SmtExpr],
    ) -> bool {
        let mut changed = false;
        let flat = Self::flatten_assertions(assertions);

        for expr in &flat {
            // Try to derive tighter bounds from two-variable linear constraints
            // of the form  x + c*y <= d  (or variants).
            if let Some(tightened) = self.try_propagate_linear(expr, bounds) {
                for (var, (lo, hi)) in tightened {
                    let entry = bounds.entry(var).or_insert((-1e18, 1e18));
                    if lo > entry.0 {
                        entry.0 = lo;
                        changed = true;
                    }
                    if hi < entry.1 {
                        entry.1 = hi;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// Attempt to extract tighter bounds from a single linear expression.
    ///
    /// Handles patterns such as:
    /// - `Add(Var(x), Var(y)) <= Const(c)` → upper bound for x given y's bounds
    /// - `Sub(Var(x), Var(y)) <= Const(c)` → x <= c + y_max
    /// - `Mul(Const(a), Var(x)) <= Const(c)` → x <= c/a (or x >= c/a if a < 0)
    fn try_propagate_linear(
        &self,
        expr: &SmtExpr,
        bounds: &HashMap<String, (f64, f64)>,
    ) -> Option<HashMap<String, (f64, f64)>> {
        let mut result: HashMap<String, (f64, f64)> = HashMap::new();

        match expr {
            // a*x <= c  →  x <= c/a (a>0) or x >= c/a (a<0)
            SmtExpr::Le(lhs, rhs) => {
                if let (SmtExpr::Mul(coef_box, var_box), SmtExpr::Const(c)) =
                    (lhs.as_ref(), rhs.as_ref())
                {
                    if let (SmtExpr::Const(a), SmtExpr::Var(v)) =
                        (coef_box.as_ref(), var_box.as_ref())
                    {
                        if *a > 0.0 {
                            let ub = c / a;
                            result.insert(v.clone(), (-1e18, ub));
                        } else if *a < 0.0 {
                            let lb = c / a;
                            result.insert(v.clone(), (lb, 1e18));
                        }
                        return Some(result);
                    }
                }
                // x + y <= c  →  x <= c - y_lo,  y <= c - x_lo
                if let (SmtExpr::Add(a, b), SmtExpr::Const(c)) =
                    (lhs.as_ref(), rhs.as_ref())
                {
                    if let (SmtExpr::Var(vx), SmtExpr::Var(vy)) = (a.as_ref(), b.as_ref()) {
                        if let Some(&(y_lo, _)) = bounds.get(vy.as_str()) {
                            let ub_x = c - y_lo;
                            result.insert(vx.clone(), (-1e18, ub_x));
                        }
                        if let Some(&(x_lo, _)) = bounds.get(vx.as_str()) {
                            let ub_y = c - x_lo;
                            result.insert(vy.clone(), (-1e18, ub_y));
                        }
                        return Some(result);
                    }
                }
                // x - y <= c  →  x <= c + y_max
                if let (SmtExpr::Sub(a, b), SmtExpr::Const(c)) =
                    (lhs.as_ref(), rhs.as_ref())
                {
                    if let (SmtExpr::Var(vx), SmtExpr::Var(vy)) = (a.as_ref(), b.as_ref()) {
                        if let Some(&(_, y_hi)) = bounds.get(vy.as_str()) {
                            let ub_x = c + y_hi;
                            result.insert(vx.clone(), (-1e18, ub_x));
                        }
                        if let Some(&(x_lo, _)) = bounds.get(vx.as_str()) {
                            let lb_y = x_lo - c;
                            result.insert(vy.clone(), (lb_y, 1e18));
                        }
                        return Some(result);
                    }
                }
            }
            // a*x >= c  →  mirror of Le
            SmtExpr::Ge(lhs, rhs) => {
                if let (SmtExpr::Mul(coef_box, var_box), SmtExpr::Const(c)) =
                    (lhs.as_ref(), rhs.as_ref())
                {
                    if let (SmtExpr::Const(a), SmtExpr::Var(v)) =
                        (coef_box.as_ref(), var_box.as_ref())
                    {
                        if *a > 0.0 {
                            let lb = c / a;
                            result.insert(v.clone(), (lb, 1e18));
                        } else if *a < 0.0 {
                            let ub = c / a;
                            result.insert(v.clone(), (-1e18, ub));
                        }
                        return Some(result);
                    }
                }
                // x + y >= c  →  x >= c - y_max,  y >= c - x_max
                if let (SmtExpr::Add(a, b), SmtExpr::Const(c)) =
                    (lhs.as_ref(), rhs.as_ref())
                {
                    if let (SmtExpr::Var(vx), SmtExpr::Var(vy)) = (a.as_ref(), b.as_ref()) {
                        if let Some(&(_, y_hi)) = bounds.get(vy.as_str()) {
                            let lb_x = c - y_hi;
                            result.insert(vx.clone(), (lb_x, 1e18));
                        }
                        if let Some(&(_, x_hi)) = bounds.get(vx.as_str()) {
                            let lb_y = c - x_hi;
                            result.insert(vy.clone(), (lb_y, 1e18));
                        }
                        return Some(result);
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Try the midpoint of each variable's bounds and verify all assertions.
    fn find_assignment(
        &self,
        bounds: &HashMap<String, (f64, f64)>,
        assertions: &[SmtExpr],
    ) -> Option<Model> {
        // First attempt: midpoints.
        let midpoint_model: Model = bounds
            .iter()
            .map(|(v, (lo, hi))| {
                let mid = if lo.is_finite() && hi.is_finite() {
                    (lo + hi) / 2.0
                } else if lo.is_finite() {
                    lo + 1.0
                } else if hi.is_finite() {
                    hi - 1.0
                } else {
                    0.0
                };
                (v.clone(), mid)
            })
            .collect();

        if self.check_assignment(&midpoint_model, assertions) {
            return Some(midpoint_model);
        }

        // Second attempt: lower bounds (clamped).
        let lo_model: Model = bounds
            .iter()
            .map(|(v, (lo, hi))| {
                let val = if lo.is_finite() {
                    let nudge = (hi - lo).abs() * 1e-9;
                    lo + nudge
                } else if hi.is_finite() {
                    hi - 1.0
                } else {
                    0.0
                };
                (v.clone(), val)
            })
            .collect();

        if self.check_assignment(&lo_model, assertions) {
            return Some(lo_model);
        }

        // Third attempt: upper bounds (clamped).
        let hi_model: Model = bounds
            .iter()
            .map(|(v, (lo, hi))| {
                let val = if hi.is_finite() {
                    let nudge = (hi - lo).abs() * 1e-9;
                    hi - nudge
                } else if lo.is_finite() {
                    lo + 1.0
                } else {
                    0.0
                };
                (v.clone(), val)
            })
            .collect();

        if self.check_assignment(&hi_model, assertions) {
            return Some(hi_model);
        }

        // Fourth attempt: quarter and three-quarter points.
        for frac in &[0.25, 0.75] {
            let model: Model = bounds
                .iter()
                .map(|(v, (lo, hi))| {
                    let val = if lo.is_finite() && hi.is_finite() {
                        lo + (hi - lo) * frac
                    } else {
                        0.0
                    };
                    (v.clone(), val)
                })
                .collect();
            if self.check_assignment(&model, assertions) {
                return Some(model);
            }
        }

        None
    }

    /// Check whether every assertion evaluates to `true` under `model`.
    fn check_assignment(&self, model: &Model, assertions: &[SmtExpr]) -> bool {
        assertions
            .iter()
            .all(|a| a.eval_bool(model) == Some(true))
    }
}

impl Default for InternalSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtSolver for InternalSolver {
    fn check_sat(&mut self, assertions: &[SmtExpr]) -> Result<SolverResult, VerifierError> {
        self.check_sat_with_timeout(assertions, self.timeout_secs)
    }

    fn check_sat_with_timeout(
        &mut self,
        assertions: &[SmtExpr],
        timeout_secs: f64,
    ) -> Result<SolverResult, VerifierError> {
        let start = Instant::now();
        let deadline = start + std::time::Duration::from_secs_f64(timeout_secs);

        // Combine scoped assertions with the caller-supplied ones.
        let mut combined = self.all_assertions();
        combined.extend_from_slice(assertions);

        let result = self.solve_linear_feasibility(&combined, deadline)?;

        let elapsed = start.elapsed().as_secs_f64();
        self.statistics.record_check(&result, elapsed);

        if Instant::now() >= deadline && !result.is_sat() && !result.is_unsat() {
            return Err(VerifierError::SmtTimeout {
                seconds: timeout_secs,
            });
        }

        Ok(result)
    }

    fn push(&mut self) {
        self.assertion_stack.push(Vec::new());
    }

    fn pop(&mut self) {
        if self.assertion_stack.len() > 1 {
            self.assertion_stack.pop();
        }
    }

    fn reset(&mut self) {
        self.declarations.clear();
        self.assertion_stack = vec![Vec::new()];
        // Statistics are intentionally preserved across resets so callers can
        // observe cumulative behaviour.
    }
}

// ---------------------------------------------------------------------------
// ExternalSolverInterface
// ---------------------------------------------------------------------------

/// Utility for generating SMT-LIB2 scripts and parsing external solver output.
///
/// This struct does **not** invoke an external process; it merely handles the
/// serialization / deserialization protocol so callers can pipe the generated
/// script to Z3, CVC5, or any SMT-LIB2 compatible solver.
pub struct ExternalSolverInterface {
    /// Logic to declare in the SMT-LIB2 preamble.
    logic: String,
}

impl ExternalSolverInterface {
    /// Create a new interface defaulting to QF_LRA logic.
    pub fn new() -> Self {
        Self {
            logic: "QF_LRA".to_string(),
        }
    }

    /// Generate a complete SMT-LIB2 script from declarations and assertions.
    ///
    /// The output includes `(set-logic ...)`, variable declarations, each
    /// assertion wrapped in `(assert ...)`, a `(check-sat)` command, and
    /// `(get-model)`.
    pub fn generate_smtlib2(
        &self,
        declarations: &[SmtDecl],
        assertions: &[SmtExpr],
    ) -> String {
        let mut script = String::with_capacity(1024);

        // Preamble
        script.push_str(&format!("(set-logic {})\n", self.logic));

        // Declarations
        for decl in declarations {
            script.push_str(&decl.to_smtlib2());
            script.push('\n');
        }

        // Assertions
        for assertion in assertions {
            script.push_str(&format!("(assert {})\n", assertion.to_smtlib2()));
        }

        // Commands
        script.push_str("(check-sat)\n");
        script.push_str("(get-model)\n");
        script.push_str("(exit)\n");

        script
    }

    /// Parse the textual output of an external SMT solver into a
    /// [`SolverResult`].
    ///
    /// Recognises the keywords `sat`, `unsat`, `unknown`, and `timeout` on
    /// their own line.  When the answer is `sat`, a subsequent `(model ...)`
    /// or `(define-fun ...)` block is parsed for variable–value assignments.
    pub fn parse_result(&self, output: &str) -> Result<SolverResult, VerifierError> {
        let trimmed = output.trim();

        if trimmed.is_empty() {
            return Err(VerifierError::SmtSolver(
                "empty output from external solver".into(),
            ));
        }

        // Find the sat/unsat/unknown/timeout answer line.
        let first_line = trimmed
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("");

        match first_line {
            "unsat" => Ok(SolverResult::Unsat),
            "timeout" => Ok(SolverResult::Timeout),
            "unknown" => {
                let reason = trimmed
                    .lines()
                    .skip(1)
                    .map(str::trim)
                    .find(|l| !l.is_empty())
                    .unwrap_or("no reason provided");
                Ok(SolverResult::Unknown(reason.to_string()))
            }
            "sat" => {
                let model = self.parse_model(trimmed)?;
                Ok(SolverResult::Sat(model))
            }
            other => Err(VerifierError::SmtSolver(format!(
                "unexpected solver answer: {other}"
            ))),
        }
    }

    /// Parse a model block from SMT-LIB2 output.
    ///
    /// Handles two common formats:
    /// 1. `(model (define-fun x () Real 3.5) ...)`
    /// 2. Bare `(define-fun x () Real 3.5)` lines
    fn parse_model(&self, output: &str) -> Result<Model, VerifierError> {
        let mut model = HashMap::new();

        // Strategy: find all `(define-fun <name> () <sort> <value>)` fragments.
        let define_fun_prefix = "define-fun";
        for segment in output.split(define_fun_prefix).skip(1) {
            let segment = segment.trim();
            // Expected: "<name> () <sort> <value>)"
            let tokens: Vec<&str> = segment.split_whitespace().collect();
            if tokens.len() < 4 {
                continue;
            }
            let name = tokens[0].to_string();
            // tokens[1] == "()", tokens[2] == sort name
            // tokens[3..] == value (may include closing parens)
            let raw_value = tokens[3].trim_end_matches(')');

            // Handle negative values written as `(- N)`.
            let value = if raw_value == "(-" || raw_value == "(/" {
                // Negative literal: `(- 3.5)` spread across tokens.
                self.parse_numeric_tokens(&tokens[3..])
            } else {
                raw_value.parse::<f64>().ok()
            };

            if let Some(v) = value {
                model.insert(name, v);
            }
        }

        Ok(model)
    }

    /// Parse a potentially parenthesised numeric value from solver output
    /// tokens such as `["(-", "3.5)"]` or `["(/", "1.0", "3.0)"]`.
    fn parse_numeric_tokens(&self, tokens: &[&str]) -> Option<f64> {
        // Join the tokens and strip all parens.
        let joined: String = tokens.iter().copied().collect::<Vec<_>>().join(" ");
        let cleaned = joined.replace(['(', ')'], "");
        let parts: Vec<&str> = cleaned.split_whitespace().collect();

        match parts.as_slice() {
            ["-", num] => num.parse::<f64>().ok().map(|v| -v),
            ["/", num, den] => {
                let n = num.parse::<f64>().ok()?;
                let d = den.parse::<f64>().ok()?;
                if d == 0.0 {
                    None
                } else {
                    Some(n / d)
                }
            }
            ["-", "/", num, den] => {
                let n = num.parse::<f64>().ok()?;
                let d = den.parse::<f64>().ok()?;
                if d == 0.0 {
                    None
                } else {
                    Some(-(n / d))
                }
            }
            [num] => num.parse::<f64>().ok(),
            _ => None,
        }
    }
}

impl Default for ExternalSolverInterface {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{SmtDecl, SmtExpr, SmtSort};

    // -- SolverResult methods -----------------------------------------------

    #[test]
    fn test_solver_result_methods() {
        let model: Model = [("x".into(), 1.0)].into_iter().collect();
        let sat = SolverResult::Sat(model.clone());
        assert!(sat.is_sat());
        assert!(!sat.is_unsat());
        assert_eq!(sat.model().unwrap()["x"], 1.0);

        let unsat = SolverResult::Unsat;
        assert!(!unsat.is_sat());
        assert!(unsat.is_unsat());
        assert!(unsat.model().is_none());

        let unknown = SolverResult::Unknown("reason".into());
        assert!(!unknown.is_sat());
        assert!(!unknown.is_unsat());
        assert!(unknown.model().is_none());

        let timeout = SolverResult::Timeout;
        assert!(!timeout.is_sat());
        assert!(!timeout.is_unsat());
        assert!(timeout.model().is_none());
    }

    // -- InternalSolver: simple SAT -----------------------------------------

    #[test]
    fn test_internal_solver_simple_sat() {
        let mut solver = InternalSolver::new();
        solver.declare(SmtDecl::new("x", SmtSort::Real));

        // x >= 2 AND x <= 5  →  SAT with x in [2, 5]
        let assertions = vec![
            SmtExpr::Ge(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(2.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(5.0)),
            ),
        ];

        let result = solver.check_sat(&assertions).unwrap();
        assert!(result.is_sat(), "expected SAT, got {result:?}");
        let m = result.model().unwrap();
        let x = m["x"];
        assert!(
            (2.0..=5.0).contains(&x),
            "x = {x} is outside [2, 5]"
        );
    }

    // -- InternalSolver: simple UNSAT ---------------------------------------

    #[test]
    fn test_internal_solver_simple_unsat() {
        let mut solver = InternalSolver::new();
        solver.declare(SmtDecl::new("x", SmtSort::Real));

        // x >= 10 AND x <= 3  →  UNSAT
        let assertions = vec![
            SmtExpr::Ge(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(10.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(3.0)),
            ),
        ];

        let result = solver.check_sat(&assertions).unwrap();
        assert!(result.is_unsat(), "expected UNSAT, got {result:?}");
    }

    // -- bound extraction ---------------------------------------------------

    #[test]
    fn test_bound_extraction() {
        let solver = InternalSolver::new();

        let assertions = vec![
            SmtExpr::Ge(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(1.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(10.0)),
            ),
            SmtExpr::Ge(
                Box::new(SmtExpr::Var("y".into())),
                Box::new(SmtExpr::Const(-3.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Var("y".into())),
                Box::new(SmtExpr::Const(7.0)),
            ),
        ];

        let bounds = solver.extract_bounds(&assertions);
        let (x_lo, x_hi) = bounds["x"];
        assert!((x_lo - 1.0).abs() < 1e-12);
        assert!((x_hi - 10.0).abs() < 1e-12);

        let (y_lo, y_hi) = bounds["y"];
        assert!((y_lo - (-3.0)).abs() < 1e-12);
        assert!((y_hi - 7.0).abs() < 1e-12);
    }

    // -- bound propagation --------------------------------------------------

    #[test]
    fn test_bound_propagation() {
        let solver = InternalSolver::new();

        // x in [0, 10], y in [0, 10], x + y <= 8
        let assertions = vec![
            SmtExpr::Ge(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(0.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(10.0)),
            ),
            SmtExpr::Ge(
                Box::new(SmtExpr::Var("y".into())),
                Box::new(SmtExpr::Const(0.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Var("y".into())),
                Box::new(SmtExpr::Const(10.0)),
            ),
            SmtExpr::Le(
                Box::new(SmtExpr::Add(
                    Box::new(SmtExpr::Var("x".into())),
                    Box::new(SmtExpr::Var("y".into())),
                )),
                Box::new(SmtExpr::Const(8.0)),
            ),
        ];

        let mut bounds = solver.extract_bounds(&assertions);
        let changed = solver.propagate_bounds(&mut bounds, &assertions);
        assert!(changed, "propagation should tighten at least one bound");

        // After propagation the upper bounds on x and y must be ≤ 8 (since
        // the other variable has lower bound 0).
        assert!(bounds["x"].1 <= 8.0 + 1e-9, "x upper bound should be ≤ 8");
        assert!(bounds["y"].1 <= 8.0 + 1e-9, "y upper bound should be ≤ 8");
    }

    // -- SMT-LIB2 generation -----------------------------------------------

    #[test]
    fn test_smtlib2_generation() {
        let iface = ExternalSolverInterface::new();
        let decls = vec![
            SmtDecl::new("x", SmtSort::Real),
            SmtDecl::new("y", SmtSort::Real),
        ];
        let assertions = vec![SmtExpr::Le(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Const(5.0)),
        )];

        let script = iface.generate_smtlib2(&decls, &assertions);

        assert!(script.contains("(set-logic QF_LRA)"), "missing set-logic");
        assert!(
            script.contains("(declare-fun x () Real)"),
            "missing x declaration"
        );
        assert!(
            script.contains("(declare-fun y () Real)"),
            "missing y declaration"
        );
        assert!(script.contains("(assert"), "missing assert");
        assert!(script.contains("(check-sat)"), "missing check-sat");
        assert!(script.contains("(get-model)"), "missing get-model");
        assert!(script.contains("(exit)"), "missing exit");
    }

    // -- solver statistics --------------------------------------------------

    #[test]
    fn test_solver_statistics() {
        let mut stats = SolverStatistics::default();
        assert_eq!(stats.num_checks, 0);
        assert_eq!(stats.sat_ratio(), 0.0);
        assert_eq!(stats.avg_check_time(), 0.0);

        let sat_result = SolverResult::Sat(HashMap::new());
        stats.record_check(&sat_result, 0.5);
        assert_eq!(stats.num_checks, 1);
        assert_eq!(stats.num_sat, 1);
        assert!((stats.sat_ratio() - 1.0).abs() < 1e-12);
        assert!((stats.avg_check_time() - 0.5).abs() < 1e-12);
        assert!((stats.max_check_time_secs - 0.5).abs() < 1e-12);

        let unsat_result = SolverResult::Unsat;
        stats.record_check(&unsat_result, 0.3);
        assert_eq!(stats.num_checks, 2);
        assert_eq!(stats.num_unsat, 1);
        assert!((stats.sat_ratio() - 0.5).abs() < 1e-12);
        assert!((stats.avg_check_time() - 0.4).abs() < 1e-12);
        assert!((stats.max_check_time_secs - 0.5).abs() < 1e-12);

        let timeout_result = SolverResult::Timeout;
        stats.record_check(&timeout_result, 30.0);
        assert_eq!(stats.num_timeout, 1);
        assert!((stats.max_check_time_secs - 30.0).abs() < 1e-12);

        let unknown_result = SolverResult::Unknown("reason".into());
        stats.record_check(&unknown_result, 1.0);
        assert_eq!(stats.num_unknown, 1);
    }

    // -- push / pop scoping -------------------------------------------------

    #[test]
    fn test_push_pop_scoping() {
        let mut solver = InternalSolver::new();
        solver.declare(SmtDecl::new("x", SmtSort::Real));

        // Outer scope: x >= 0
        solver.assertion_stack.last_mut().unwrap().push(SmtExpr::Ge(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Const(0.0)),
        ));

        // Push inner scope and add x <= 5
        solver.push();
        solver.assertion_stack.last_mut().unwrap().push(SmtExpr::Le(
            Box::new(SmtExpr::Var("x".into())),
            Box::new(SmtExpr::Const(5.0)),
        ));
        assert_eq!(solver.assertion_stack.len(), 2);

        let result = solver.check_sat(&[]).unwrap();
        assert!(result.is_sat());

        // Pop inner scope — the x <= 5 constraint is gone.
        solver.pop();
        assert_eq!(solver.assertion_stack.len(), 1);

        // Now add x <= 100 externally; should still be SAT (only x >= 0 in scope).
        let result = solver
            .check_sat(&[SmtExpr::Le(
                Box::new(SmtExpr::Var("x".into())),
                Box::new(SmtExpr::Const(100.0)),
            )])
            .unwrap();
        assert!(result.is_sat());

        // Pop on a single-scope stack should be a no-op.
        solver.pop();
        assert_eq!(solver.assertion_stack.len(), 1);
    }

    // -- parse_result -------------------------------------------------------

    #[test]
    fn test_parse_sat_result() {
        let iface = ExternalSolverInterface::new();

        let output = "sat\n(model\n  (define-fun x () Real 3.5)\n  (define-fun y () Real (- 2.0))\n)";
        let result = iface.parse_result(output).unwrap();
        assert!(result.is_sat());
        let m = result.model().unwrap();
        assert!((m["x"] - 3.5).abs() < 1e-12);
        assert!((m["y"] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_parse_unsat_result() {
        let iface = ExternalSolverInterface::new();
        let result = iface.parse_result("unsat\n").unwrap();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_parse_unknown_result() {
        let iface = ExternalSolverInterface::new();
        let result = iface.parse_result("unknown\nincomplete").unwrap();
        assert!(!result.is_sat());
        assert!(!result.is_unsat());
        if let SolverResult::Unknown(reason) = result {
            assert_eq!(reason, "incomplete");
        } else {
            panic!("expected Unknown variant");
        }
    }
}
