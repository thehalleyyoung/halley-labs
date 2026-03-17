//! SMT solver interface with SMTLIB2 output generation.
//!
//! Provides a trait-based abstraction over SMT solvers, a mock implementation
//! for testing, and Z3-compatible SMTLIB2 script generation.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::constraint::SmtExpr;

// ---------------------------------------------------------------------------
// SolverResult
// ---------------------------------------------------------------------------

/// Outcome of an SMT solver check-sat invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverResult {
    /// The formula is satisfiable; an optional model is attached.
    Sat(Option<RawModel>),
    /// The formula is unsatisfiable.
    Unsat,
    /// The solver could not determine satisfiability.
    Unknown(String),
    /// The solver exceeded its time budget.
    Timeout(Duration),
}

impl SolverResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SolverResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SolverResult::Unsat)
    }

    pub fn model(&self) -> Option<&RawModel> {
        match self {
            SolverResult::Sat(Some(m)) => Some(m),
            _ => None,
        }
    }
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverResult::Sat(_) => write!(f, "sat"),
            SolverResult::Unsat => write!(f, "unsat"),
            SolverResult::Unknown(reason) => write!(f, "unknown ({})", reason),
            SolverResult::Timeout(d) => write!(f, "timeout after {:.2}s", d.as_secs_f64()),
        }
    }
}

// ---------------------------------------------------------------------------
// RawModel – variable assignments extracted from SAT
// ---------------------------------------------------------------------------

/// A raw model mapping variable names to string-encoded values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawModel {
    assignments: HashMap<String, ModelValue>,
}

/// A single value inside a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelValue {
    Bool(bool),
    Int(i64),
    BitVec { value: u64, width: u32 },
    Str(String),
}

impl ModelValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ModelValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            ModelValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_bitvec(&self) -> Option<(u64, u32)> {
        match self {
            ModelValue::BitVec { value, width } => Some((*value, *width)),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ModelValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl fmt::Display for ModelValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelValue::Bool(b) => write!(f, "{}", b),
            ModelValue::Int(i) => write!(f, "{}", i),
            ModelValue::BitVec { value, width } => write!(f, "#b{:0>width$b}", value, width = *width as usize),
            ModelValue::Str(s) => write!(f, "\"{}\"", s),
        }
    }
}

impl RawModel {
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, value: ModelValue) {
        self.assignments.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<&ModelValue> {
        self.assignments.get(name)
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.get(name).and_then(|v| v.as_bool())
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.get(name).and_then(|v| v.as_int())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &ModelValue)> {
        self.assignments.iter()
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Parse a SMTLIB2 model response string into a `RawModel`.
    pub fn parse_smtlib2(response: &str) -> IsoSpecResult<Self> {
        let mut model = RawModel::new();
        let trimmed = response.trim();
        if trimmed.is_empty() {
            return Ok(model);
        }

        // Simple S-expression parser for (model (define-fun name () Sort value)...)
        let inner = if trimmed.starts_with("(model") {
            let without_prefix = trimmed.strip_prefix("(model").unwrap_or(trimmed);
            without_prefix.strip_suffix(')').unwrap_or(without_prefix)
        } else if trimmed.starts_with('(') && trimmed.ends_with(')') {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };

        let mut depth: i32 = 0;
        let mut current_define = String::new();
        let mut defines: Vec<String> = Vec::new();

        for ch in inner.chars() {
            match ch {
                '(' => {
                    depth += 1;
                    if depth == 1 {
                        current_define.clear();
                    }
                    current_define.push(ch);
                }
                ')' => {
                    current_define.push(ch);
                    depth -= 1;
                    if depth == 0 {
                        defines.push(current_define.trim().to_string());
                        current_define.clear();
                    }
                }
                _ => {
                    if depth > 0 {
                        current_define.push(ch);
                    }
                }
            }
        }

        for def in &defines {
            if let Some(parsed) = Self::parse_define_fun(def) {
                model.insert(parsed.0, parsed.1);
            }
        }

        Ok(model)
    }

    fn parse_define_fun(sexpr: &str) -> Option<(String, ModelValue)> {
        let inner = sexpr.strip_prefix('(')?.strip_suffix(')')?;
        let tokens: Vec<&str> = inner.split_whitespace().collect();
        if tokens.len() < 4 || tokens[0] != "define-fun" {
            return None;
        }
        let name = tokens[1].to_string();
        // Find the value token (last meaningful token)
        let value_str = *tokens.last()?;
        let value = Self::parse_value(value_str)?;
        Some((name, value))
    }

    fn parse_value(s: &str) -> Option<ModelValue> {
        if s == "true" {
            return Some(ModelValue::Bool(true));
        }
        if s == "false" {
            return Some(ModelValue::Bool(false));
        }
        if let Ok(i) = s.parse::<i64>() {
            return Some(ModelValue::Int(i));
        }
        if let Some(bv) = s.strip_prefix("#b") {
            let width = bv.len() as u32;
            let value = u64::from_str_radix(bv, 2).ok()?;
            return Some(ModelValue::BitVec { value, width });
        }
        if let Some(hex) = s.strip_prefix("#x") {
            let width = (hex.len() as u32) * 4;
            let value = u64::from_str_radix(hex, 16).ok()?;
            return Some(ModelValue::BitVec { value, width });
        }
        Some(ModelValue::Str(s.to_string()))
    }
}

impl Default for RawModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SolverConfig
// ---------------------------------------------------------------------------

/// Configuration for an SMT solver invocation.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum time the solver may spend.
    pub timeout: Duration,
    /// SMT-LIB logic to use (e.g. "QF_LIA", "ALL").
    pub logic: String,
    /// Produce models on SAT.
    pub produce_models: bool,
    /// Produce unsat cores on UNSAT.
    pub produce_unsat_cores: bool,
    /// Incremental mode (push/pop).
    pub incremental: bool,
    /// Extra solver options as key-value pairs.
    pub options: HashMap<String, String>,
    /// Random seed for solver.
    pub seed: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            logic: "QF_LIA".to_string(),
            produce_models: true,
            produce_unsat_cores: false,
            incremental: false,
            options: HashMap::new(),
            seed: 42,
        }
    }
}

impl SolverConfig {
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_logic(mut self, logic: &str) -> Self {
        self.logic = logic.to_string();
        self
    }

    pub fn with_incremental(mut self, inc: bool) -> Self {
        self.incremental = inc;
        self
    }

    pub fn with_unsat_cores(mut self, cores: bool) -> Self {
        self.produce_unsat_cores = cores;
        self
    }

    pub fn with_option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Render the SMTLIB2 preamble for this configuration.
    pub fn to_smtlib2_preamble(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("(set-logic {})", self.logic));
        if self.produce_models {
            lines.push("(set-option :produce-models true)".to_string());
        }
        if self.produce_unsat_cores {
            lines.push("(set-option :produce-unsat-cores true)".to_string());
        }
        let timeout_ms = self.timeout.as_millis();
        lines.push(format!("(set-option :timeout {})", timeout_ms));
        lines.push(format!("(set-option :random-seed {})", self.seed));
        for (k, v) in &self.options {
            lines.push(format!("(set-option :{} {})", k, v));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// SmtSolver trait
// ---------------------------------------------------------------------------

/// Trait abstracting an SMT solver backend.
pub trait SmtSolver: Send + Sync {
    /// Submit a full SMTLIB2 script and obtain a result.
    fn check_sat(&mut self, script: &str) -> IsoSpecResult<SolverResult>;

    /// Push a new assertion context (incremental mode).
    fn push(&mut self) -> IsoSpecResult<()>;

    /// Pop the most recent assertion context.
    fn pop(&mut self) -> IsoSpecResult<()>;

    /// Assert a single formula in the current context.
    fn assert_formula(&mut self, formula: &str) -> IsoSpecResult<()>;

    /// Declare a constant of a given sort.
    fn declare_const(&mut self, name: &str, sort: &str) -> IsoSpecResult<()>;

    /// Retrieve the current configuration.
    fn config(&self) -> &SolverConfig;

    /// Reset the solver to its initial state.
    fn reset(&mut self) -> IsoSpecResult<()>;
}

// ---------------------------------------------------------------------------
// Smtlib2Writer – SMTLIB2 script generation
// ---------------------------------------------------------------------------

/// Builds SMTLIB2 scripts compatible with Z3, CVC5, etc.
pub struct Smtlib2Writer {
    config: SolverConfig,
    declarations: Vec<String>,
    assertions: Vec<String>,
    named_assertions: Vec<(String, String)>,
}

impl Smtlib2Writer {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            declarations: Vec::new(),
            assertions: Vec::new(),
            named_assertions: Vec::new(),
        }
    }

    pub fn declare_const(&mut self, name: &str, sort: &str) {
        self.declarations
            .push(format!("(declare-const {} {})", name, sort));
    }

    pub fn declare_fun(&mut self, name: &str, args: &[&str], ret: &str) {
        let arg_str = if args.is_empty() {
            "()".to_string()
        } else {
            format!("({})", args.join(" "))
        };
        self.declarations
            .push(format!("(declare-fun {} {} {})", name, arg_str, ret));
    }

    pub fn assert_expr(&mut self, expr: &str) {
        self.assertions.push(format!("(assert {})", expr));
    }

    pub fn assert_named(&mut self, name: &str, expr: &str) {
        self.named_assertions
            .push((name.to_string(), expr.to_string()));
    }

    pub fn assert_smt_expr(&mut self, expr: &SmtExpr) {
        let rendered = render_smt_expr(expr);
        self.assert_expr(&rendered);
    }

    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(self.config.to_smtlib2_preamble());
        lines.push(String::new());
        for decl in &self.declarations {
            lines.push(decl.clone());
        }
        if !self.declarations.is_empty() {
            lines.push(String::new());
        }
        for assertion in &self.assertions {
            lines.push(assertion.clone());
        }
        for (name, expr) in &self.named_assertions {
            lines.push(format!("(assert (! {} :named {}))", expr, name));
        }
        lines.push(String::new());
        lines.push("(check-sat)".to_string());
        if self.config.produce_models {
            lines.push("(get-model)".to_string());
        }
        if self.config.produce_unsat_cores {
            lines.push("(get-unsat-core)".to_string());
        }
        lines.push("(exit)".to_string());
        lines.join("\n")
    }

    pub fn declaration_count(&self) -> usize {
        self.declarations.len()
    }

    pub fn assertion_count(&self) -> usize {
        self.assertions.len() + self.named_assertions.len()
    }
}

/// Recursively render an `SmtExpr` to an SMTLIB2 string.
pub fn render_smt_expr(expr: &SmtExpr) -> String {
    match expr {
        SmtExpr::Const(name) => name.clone(),
        SmtExpr::BoolLit(b) => if *b { "true" } else { "false" }.to_string(),
        SmtExpr::IntLit(i) => {
            if *i < 0 {
                format!("(- {})", -i)
            } else {
                i.to_string()
            }
        }
        SmtExpr::Not(inner) => format!("(not {})", render_smt_expr(inner)),
        SmtExpr::And(children) => {
            if children.is_empty() {
                "true".to_string()
            } else if children.len() == 1 {
                render_smt_expr(&children[0])
            } else {
                let parts: Vec<String> = children.iter().map(|c| render_smt_expr(c)).collect();
                format!("(and {})", parts.join(" "))
            }
        }
        SmtExpr::Or(children) => {
            if children.is_empty() {
                "false".to_string()
            } else if children.len() == 1 {
                render_smt_expr(&children[0])
            } else {
                let parts: Vec<String> = children.iter().map(|c| render_smt_expr(c)).collect();
                format!("(or {})", parts.join(" "))
            }
        }
        SmtExpr::Implies(lhs, rhs) => {
            format!("(=> {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Eq(lhs, rhs) => {
            format!("(= {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Lt(lhs, rhs) => {
            format!("(< {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Le(lhs, rhs) => {
            format!("(<= {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Gt(lhs, rhs) => {
            format!("(> {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Ge(lhs, rhs) => {
            format!("(>= {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Add(lhs, rhs) => {
            format!("(+ {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Sub(lhs, rhs) => {
            format!("(- {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Mul(lhs, rhs) => {
            format!("(* {} {})", render_smt_expr(lhs), render_smt_expr(rhs))
        }
        SmtExpr::Ite(cond, t, e) => {
            format!(
                "(ite {} {} {})",
                render_smt_expr(cond),
                render_smt_expr(t),
                render_smt_expr(e)
            )
        }
        SmtExpr::Distinct(children) => {
            let parts: Vec<String> = children.iter().map(|c| render_smt_expr(c)).collect();
            format!("(distinct {})", parts.join(" "))
        }
        SmtExpr::ForAll(vars, body) => {
            let bindings: Vec<String> = vars.iter().map(|(n, s)| format!("({} {})", n, s)).collect();
            format!("(forall ({}) {})", bindings.join(" "), render_smt_expr(body))
        }
        SmtExpr::Exists(vars, body) => {
            let bindings: Vec<String> = vars.iter().map(|(n, s)| format!("({} {})", n, s)).collect();
            format!("(exists ({}) {})", bindings.join(" "), render_smt_expr(body))
        }
        SmtExpr::Select(arr, idx) => {
            format!("(select {} {})", render_smt_expr(arr), render_smt_expr(idx))
        }
        SmtExpr::Store(arr, idx, val) => {
            format!(
                "(store {} {} {})",
                render_smt_expr(arr),
                render_smt_expr(idx),
                render_smt_expr(val)
            )
        }
        SmtExpr::Apply(func, args) => {
            if args.is_empty() {
                func.clone()
            } else {
                let parts: Vec<String> = args.iter().map(|a| render_smt_expr(a)).collect();
                format!("({} {})", func, parts.join(" "))
            }
        }
        SmtExpr::Let(bindings, body) => {
            let bind_strs: Vec<String> = bindings
                .iter()
                .map(|(n, e)| format!("({} {})", n, render_smt_expr(e)))
                .collect();
            format!("(let ({}) {})", bind_strs.join(" "), render_smt_expr(body))
        }
        SmtExpr::BitVecLit(value, width) => {
            format!("(_ bv{} {})", value, width)
        }
        SmtExpr::BvAnd(a, b) => format!("(bvand {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::BvOr(a, b) => format!("(bvor {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::BvAdd(a, b) => format!("(bvadd {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::BvSub(a, b) => format!("(bvsub {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::BvMul(a, b) => format!("(bvmul {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::BvUlt(a, b) => format!("(bvult {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::BvSlt(a, b) => format!("(bvslt {} {})", render_smt_expr(a), render_smt_expr(b)),
        SmtExpr::Extract(high, low, e) => {
            format!("((_ extract {} {}) {})", high, low, render_smt_expr(e))
        }
        SmtExpr::ZeroExtend(extra, e) => {
            format!("((_ zero_extend {}) {})", extra, render_smt_expr(e))
        }
        SmtExpr::SignExtend(extra, e) => {
            format!("((_ sign_extend {}) {})", extra, render_smt_expr(e))
        }
        SmtExpr::Concat(a, b) => {
            format!("(concat {} {})", render_smt_expr(a), render_smt_expr(b))
        }
        SmtExpr::Comment(text) => format!("; {}", text),
        _ => format!("; unsupported expr: {:?}", expr),
    }
}

// ---------------------------------------------------------------------------
// MockSmtSolver
// ---------------------------------------------------------------------------

/// A mock SMT solver for unit testing.
///
/// Returns pre-configured results for check-sat queries and records all
/// operations for inspection.
pub struct MockSmtSolver {
    config: SolverConfig,
    /// Queue of results to return on successive check-sat calls.
    result_queue: Vec<SolverResult>,
    /// Index into result_queue for the next check-sat.
    next_result: usize,
    /// Recorded assertions.
    pub recorded_assertions: Vec<String>,
    /// Recorded declarations.
    pub recorded_declarations: Vec<(String, String)>,
    /// Current context depth (push/pop).
    context_depth: usize,
    /// Recorded push/pop operations.
    pub context_ops: Vec<ContextOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextOp {
    Push,
    Pop,
    Reset,
}

impl MockSmtSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            result_queue: Vec::new(),
            next_result: 0,
            recorded_assertions: Vec::new(),
            recorded_declarations: Vec::new(),
            context_depth: 0,
            context_ops: Vec::new(),
        }
    }

    /// Queue a result to be returned by the next `check_sat` call.
    pub fn enqueue_result(&mut self, result: SolverResult) {
        self.result_queue.push(result);
    }

    /// Create a mock that always returns SAT with an empty model.
    pub fn always_sat() -> Self {
        let mut s = Self::new(SolverConfig::default());
        for _ in 0..100 {
            s.enqueue_result(SolverResult::Sat(Some(RawModel::new())));
        }
        s
    }

    /// Create a mock that always returns UNSAT.
    pub fn always_unsat() -> Self {
        let mut s = Self::new(SolverConfig::default());
        for _ in 0..100 {
            s.enqueue_result(SolverResult::Unsat);
        }
        s
    }
}

impl SmtSolver for MockSmtSolver {
    fn check_sat(&mut self, _script: &str) -> IsoSpecResult<SolverResult> {
        if self.next_result < self.result_queue.len() {
            let result = self.result_queue[self.next_result].clone();
            self.next_result += 1;
            Ok(result)
        } else {
            Ok(SolverResult::Unknown("no more queued results".to_string()))
        }
    }

    fn push(&mut self) -> IsoSpecResult<()> {
        self.context_depth += 1;
        self.context_ops.push(ContextOp::Push);
        Ok(())
    }

    fn pop(&mut self) -> IsoSpecResult<()> {
        if self.context_depth == 0 {
            return Err(IsoSpecError::SmtSolver { msg: "pop on empty context stack".into() });
        }
        self.context_depth -= 1;
        self.context_ops.push(ContextOp::Pop);
        Ok(())
    }

    fn assert_formula(&mut self, formula: &str) -> IsoSpecResult<()> {
        self.recorded_assertions.push(formula.to_string());
        Ok(())
    }

    fn declare_const(&mut self, name: &str, sort: &str) -> IsoSpecResult<()> {
        self.recorded_declarations
            .push((name.to_string(), sort.to_string()));
        Ok(())
    }

    fn config(&self) -> &SolverConfig {
        &self.config
    }

    fn reset(&mut self) -> IsoSpecResult<()> {
        self.recorded_assertions.clear();
        self.recorded_declarations.clear();
        self.context_depth = 0;
        self.context_ops.push(ContextOp::Reset);
        self.next_result = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SolverStats – statistics for a solving session
// ---------------------------------------------------------------------------

/// Accumulated statistics from one or more solver invocations.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub total_checks: u64,
    pub sat_count: u64,
    pub unsat_count: u64,
    pub unknown_count: u64,
    pub timeout_count: u64,
    pub total_time: Duration,
    pub max_time: Duration,
    pub total_assertions: u64,
    pub total_declarations: u64,
}

impl SolverStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_check(&mut self, result: &SolverResult, elapsed: Duration) {
        self.total_checks += 1;
        self.total_time += elapsed;
        if elapsed > self.max_time {
            self.max_time = elapsed;
        }
        match result {
            SolverResult::Sat(_) => self.sat_count += 1,
            SolverResult::Unsat => self.unsat_count += 1,
            SolverResult::Unknown(_) => self.unknown_count += 1,
            SolverResult::Timeout(_) => self.timeout_count += 1,
        }
    }

    pub fn average_time(&self) -> Duration {
        if self.total_checks == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.total_checks as u32
        }
    }

    pub fn merge(&mut self, other: &SolverStats) {
        self.total_checks += other.total_checks;
        self.sat_count += other.sat_count;
        self.unsat_count += other.unsat_count;
        self.unknown_count += other.unknown_count;
        self.timeout_count += other.timeout_count;
        self.total_time += other.total_time;
        if other.max_time > self.max_time {
            self.max_time = other.max_time;
        }
        self.total_assertions += other.total_assertions;
        self.total_declarations += other.total_declarations;
    }
}

impl fmt::Display for SolverStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checks={} (sat={}, unsat={}, unknown={}, timeout={}) time={:.2}s avg={:.3}s max={:.3}s",
            self.total_checks,
            self.sat_count,
            self.unsat_count,
            self.unknown_count,
            self.timeout_count,
            self.total_time.as_secs_f64(),
            self.average_time().as_secs_f64(),
            self.max_time.as_secs_f64(),
        )
    }
}

// ---------------------------------------------------------------------------
// TimedSolver wrapper
// ---------------------------------------------------------------------------

/// Wraps another solver and collects timing statistics.
pub struct TimedSolver<S: SmtSolver> {
    inner: S,
    pub stats: SolverStats,
}

impl<S: SmtSolver> TimedSolver<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: SolverStats::new(),
        }
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: SmtSolver> SmtSolver for TimedSolver<S> {
    fn check_sat(&mut self, script: &str) -> IsoSpecResult<SolverResult> {
        let start = Instant::now();
        let result = self.inner.check_sat(script)?;
        let elapsed = start.elapsed();
        self.stats.record_check(&result, elapsed);
        Ok(result)
    }

    fn push(&mut self) -> IsoSpecResult<()> {
        self.inner.push()
    }

    fn pop(&mut self) -> IsoSpecResult<()> {
        self.inner.pop()
    }

    fn assert_formula(&mut self, formula: &str) -> IsoSpecResult<()> {
        self.stats.total_assertions += 1;
        self.inner.assert_formula(formula)
    }

    fn declare_const(&mut self, name: &str, sort: &str) -> IsoSpecResult<()> {
        self.stats.total_declarations += 1;
        self.inner.declare_const(name, sort)
    }

    fn config(&self) -> &SolverConfig {
        self.inner.config()
    }

    fn reset(&mut self) -> IsoSpecResult<()> {
        self.inner.reset()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_result_display() {
        assert_eq!(format!("{}", SolverResult::Sat(None)), "sat");
        assert_eq!(format!("{}", SolverResult::Unsat), "unsat");
        assert_eq!(
            format!("{}", SolverResult::Unknown("incomplete".into())),
            "unknown (incomplete)"
        );
    }

    #[test]
    fn test_solver_result_predicates() {
        assert!(SolverResult::Sat(None).is_sat());
        assert!(!SolverResult::Sat(None).is_unsat());
        assert!(SolverResult::Unsat.is_unsat());
        assert!(!SolverResult::Unsat.is_sat());
    }

    #[test]
    fn test_raw_model_basic() {
        let mut model = RawModel::new();
        model.insert("x".into(), ModelValue::Int(42));
        model.insert("y".into(), ModelValue::Bool(true));
        model.insert("z".into(), ModelValue::Str("hello".into()));

        assert_eq!(model.get_int("x"), Some(42));
        assert_eq!(model.get_bool("y"), Some(true));
        assert_eq!(model.get("z").unwrap().as_str(), Some("hello"));
        assert_eq!(model.len(), 3);
        assert!(!model.is_empty());
    }

    #[test]
    fn test_raw_model_parse_smtlib2() {
        let response = r#"(model
  (define-fun x () Int 42)
  (define-fun flag () Bool true)
)"#;
        let model = RawModel::parse_smtlib2(response).unwrap();
        assert_eq!(model.get_int("x"), Some(42));
        assert_eq!(model.get_bool("flag"), Some(true));
    }

    #[test]
    fn test_solver_config_preamble() {
        let config = SolverConfig::default()
            .with_logic("QF_LIA")
            .with_timeout(Duration::from_secs(10));
        let preamble = config.to_smtlib2_preamble();
        assert!(preamble.contains("(set-logic QF_LIA)"));
        assert!(preamble.contains(":produce-models true"));
        assert!(preamble.contains(":timeout 10000"));
    }

    #[test]
    fn test_smtlib2_writer() {
        let config = SolverConfig::default();
        let mut writer = Smtlib2Writer::new(config);
        writer.declare_const("x", "Int");
        writer.declare_const("y", "Int");
        writer.assert_expr("(< x y)");
        writer.assert_expr("(> y 0)");

        let script = writer.render();
        assert!(script.contains("(declare-const x Int)"));
        assert!(script.contains("(declare-const y Int)"));
        assert!(script.contains("(assert (< x y))"));
        assert!(script.contains("(check-sat)"));
        assert_eq!(writer.declaration_count(), 2);
        assert_eq!(writer.assertion_count(), 2);
    }

    #[test]
    fn test_mock_solver_queue() {
        let mut solver = MockSmtSolver::new(SolverConfig::default());
        solver.enqueue_result(SolverResult::Sat(None));
        solver.enqueue_result(SolverResult::Unsat);

        let r1 = solver.check_sat("(check-sat)").unwrap();
        assert!(r1.is_sat());
        let r2 = solver.check_sat("(check-sat)").unwrap();
        assert!(r2.is_unsat());
        let r3 = solver.check_sat("(check-sat)").unwrap();
        assert!(matches!(r3, SolverResult::Unknown(_)));
    }

    #[test]
    fn test_mock_solver_push_pop() {
        let mut solver = MockSmtSolver::new(SolverConfig::default());
        solver.push().unwrap();
        solver.push().unwrap();
        solver.pop().unwrap();
        solver.pop().unwrap();
        assert!(solver.pop().is_err());

        assert_eq!(
            solver.context_ops,
            vec![ContextOp::Push, ContextOp::Push, ContextOp::Pop, ContextOp::Pop]
        );
    }

    #[test]
    fn test_timed_solver_stats() {
        let mut mock = MockSmtSolver::new(SolverConfig::default());
        mock.enqueue_result(SolverResult::Sat(None));
        mock.enqueue_result(SolverResult::Unsat);

        let mut timed = TimedSolver::new(mock);
        timed.check_sat("").unwrap();
        timed.check_sat("").unwrap();
        timed.assert_formula("(> x 0)").unwrap();
        timed.declare_const("x", "Int").unwrap();

        assert_eq!(timed.stats.total_checks, 2);
        assert_eq!(timed.stats.sat_count, 1);
        assert_eq!(timed.stats.unsat_count, 1);
        assert_eq!(timed.stats.total_assertions, 1);
        assert_eq!(timed.stats.total_declarations, 1);
    }

    #[test]
    fn test_render_smt_expr() {
        let expr = SmtExpr::And(vec![
            SmtExpr::Lt(
                Box::new(SmtExpr::Const("x".into())),
                Box::new(SmtExpr::Const("y".into())),
            ),
            SmtExpr::Gt(
                Box::new(SmtExpr::Const("y".into())),
                Box::new(SmtExpr::IntLit(0)),
            ),
        ]);
        let rendered = render_smt_expr(&expr);
        assert_eq!(rendered, "(and (< x y) (> y 0))");
    }

    #[test]
    fn test_model_value_display() {
        assert_eq!(format!("{}", ModelValue::Bool(true)), "true");
        assert_eq!(format!("{}", ModelValue::Int(-5)), "-5");
        assert_eq!(
            format!("{}", ModelValue::BitVec { value: 0b1010, width: 4 }),
            "#b1010"
        );
        assert_eq!(format!("{}", ModelValue::Str("abc".into())), "\"abc\"");
    }

    #[test]
    fn test_solver_stats_merge() {
        let mut a = SolverStats::new();
        a.record_check(&SolverResult::Sat(None), Duration::from_millis(100));
        let mut b = SolverStats::new();
        b.record_check(&SolverResult::Unsat, Duration::from_millis(200));
        a.merge(&b);
        assert_eq!(a.total_checks, 2);
        assert_eq!(a.sat_count, 1);
        assert_eq!(a.unsat_count, 1);
        assert_eq!(a.max_time, Duration::from_millis(200));
    }
}
