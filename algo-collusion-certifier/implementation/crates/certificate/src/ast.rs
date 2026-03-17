//! Certificate DSL abstract syntax tree.
//!
//! Defines the full structure of a CollusionProof certificate, including proof
//! steps, expressions, literals, and operators.

use serde::{Deserialize, Serialize};
use shared_types::{
    ConfidenceInterval, ConfidenceLevel, OracleAccessLevel, PlayerId, PValue, SignificanceLevel,
};
use std::fmt;

// ── Top-level certificate ────────────────────────────────────────────────────

/// A complete certificate consisting of a header and body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateAST {
    pub header: CertificateHeader,
    pub body: CertificateBody,
}

impl CertificateAST {
    pub fn new(header: CertificateHeader, body: CertificateBody) -> Self {
        Self { header, body }
    }

    /// Number of proof steps in the certificate.
    pub fn step_count(&self) -> usize {
        self.body.steps.len()
    }

    /// Return all reference identifiers declared in proof steps.
    pub fn declared_refs(&self) -> Vec<String> {
        self.body
            .steps
            .iter()
            .filter_map(|s| s.declared_ref())
            .collect()
    }

    /// Pretty-print the certificate to a multi-line string.
    pub fn pretty_print(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Certificate v{}\n", self.header.version));
        out.push_str(&format!("  Timestamp : {}\n", self.header.timestamp));
        out.push_str(&format!("  Scenario  : {}\n", self.header.scenario));
        out.push_str(&format!(
            "  Oracle    : {:?}\n",
            self.header.oracle_level
        ));
        out.push_str(&format!("  Alpha     : {:.6}\n", self.header.alpha.value()));
        out.push_str("Steps:\n");
        for (i, step) in self.body.steps.iter().enumerate() {
            out.push_str(&format!("  [{}] {}\n", i, step.summary()));
        }
        out
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for CertificateAST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pretty_print())
    }
}

// ── Certificate header ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateHeader {
    pub version: String,
    pub timestamp: String,
    pub scenario: String,
    pub oracle_level: OracleAccessLevel,
    pub alpha: SignificanceLevel,
}

impl CertificateHeader {
    pub fn new(
        scenario: &str,
        oracle_level: OracleAccessLevel,
        alpha: f64,
    ) -> Self {
        Self {
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            scenario: scenario.to_string(),
            oracle_level,
            alpha: SignificanceLevel::new(alpha).expect("alpha must be in (0, 1)"),
        }
    }
}

// ── Certificate body ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateBody {
    pub steps: Vec<ProofStep>,
}

impl CertificateBody {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn push(&mut self, step: ProofStep) {
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for CertificateBody {
    fn default() -> Self {
        Self::new()
    }
}

// ── Proof step enum ──────────────────────────────────────────────────────────

/// A single step in a certificate proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofStep {
    DataDeclaration(TrajectoryRef, SegmentSpec),
    StatisticalTest(TestRef, TestType, Statistic, PValueWrapper),
    EquilibriumClaim(EquilibriumRef, GameSpec, NashProfile),
    DeviationBound(DeviationRef, PlayerId, Bound, ConfidenceLevel),
    PunishmentEvidence(PunishmentRef, PlayerId, PayoffDrop, PValueWrapper),
    CollusionPremium(CPRef, Value, CIWrapper),
    Inference(InferenceRef, Rule, Premises, Conclusion),
    Verdict(VerdictType, Confidence, SupportingRefs),
}

impl ProofStep {
    /// Return the reference string declared by this step, if any.
    pub fn declared_ref(&self) -> Option<String> {
        match self {
            ProofStep::DataDeclaration(r, _) => Some(r.0.clone()),
            ProofStep::StatisticalTest(r, _, _, _) => Some(r.0.clone()),
            ProofStep::EquilibriumClaim(r, _, _) => Some(r.0.clone()),
            ProofStep::DeviationBound(r, _, _, _) => Some(r.0.clone()),
            ProofStep::PunishmentEvidence(r, _, _, _) => Some(r.0.clone()),
            ProofStep::CollusionPremium(r, _, _) => Some(r.0.clone()),
            ProofStep::Inference(r, _, _, _) => Some(r.0.clone()),
            ProofStep::Verdict(_, _, _) => None,
        }
    }

    /// One-line summary of this proof step.
    pub fn summary(&self) -> String {
        match self {
            ProofStep::DataDeclaration(r, seg) => {
                format!("DataDecl({}, segment={})", r.0, seg.segment_type)
            }
            ProofStep::StatisticalTest(r, tt, stat, pv) => {
                format!(
                    "StatTest({}, type={}, stat={:.4}, p={:.6})",
                    r.0, tt.name, stat.value, pv.0
                )
            }
            ProofStep::EquilibriumClaim(r, _, _) => {
                format!("EqClaim({})", r.0)
            }
            ProofStep::DeviationBound(r, player, bound, cl) => {
                format!(
                    "DevBound({}, player={}, bound={:.4}, conf={:.4})",
                    r.0,
                    player,
                    bound.value,
                    cl.value()
                )
            }
            ProofStep::PunishmentEvidence(r, player, drop, pv) => {
                format!(
                    "Punishment({}, player={}, drop={:.4}, p={:.6})",
                    r.0, player, drop.value, pv.0
                )
            }
            ProofStep::CollusionPremium(r, val, _ci) => {
                format!("CP({}, value={:.4})", r.0, val.0)
            }
            ProofStep::Inference(r, rule, premises, _conclusion) => {
                format!(
                    "Infer({}, rule={}, from=[{}])",
                    r.0,
                    rule.name,
                    premises.refs.join(", ")
                )
            }
            ProofStep::Verdict(vt, conf, refs) => {
                format!(
                    "Verdict({:?}, conf={:.4}, refs=[{}])",
                    vt,
                    conf.0,
                    refs.refs.join(", ")
                )
            }
        }
    }

    /// Return the step kind as a string.
    pub fn kind(&self) -> &'static str {
        match self {
            ProofStep::DataDeclaration(..) => "DataDeclaration",
            ProofStep::StatisticalTest(..) => "StatisticalTest",
            ProofStep::EquilibriumClaim(..) => "EquilibriumClaim",
            ProofStep::DeviationBound(..) => "DeviationBound",
            ProofStep::PunishmentEvidence(..) => "PunishmentEvidence",
            ProofStep::CollusionPremium(..) => "CollusionPremium",
            ProofStep::Inference(..) => "Inference",
            ProofStep::Verdict(..) => "Verdict",
        }
    }

    /// Return all reference IDs this step depends on.
    pub fn dependency_refs(&self) -> Vec<String> {
        match self {
            ProofStep::Inference(_, _, premises, _) => premises.refs.clone(),
            ProofStep::Verdict(_, _, refs) => refs.refs.clone(),
            _ => Vec::new(),
        }
    }
}

impl fmt::Display for ProofStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ── Wrapper types for proof step fields ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TrajectoryRef(pub String);

impl TrajectoryRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentSpec {
    pub segment_type: String,
    pub start_round: usize,
    pub end_round: usize,
    pub trajectory_hash: String,
    pub num_players: usize,
}

impl SegmentSpec {
    pub fn new(
        segment_type: &str,
        start: usize,
        end: usize,
        hash: &str,
        num_players: usize,
    ) -> Self {
        Self {
            segment_type: segment_type.to_string(),
            start_round: start,
            end_round: end,
            trajectory_hash: hash.to_string(),
            num_players,
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.end_round.saturating_sub(self.start_round)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TestRef(pub String);

impl TestRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestType {
    pub name: String,
    pub category: String,
}

impl TestType {
    pub fn new(name: &str, category: &str) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Statistic {
    pub value: f64,
    pub name_idx: u16,
}

impl Statistic {
    pub fn new(value: f64) -> Self {
        Self { value, name_idx: 0 }
    }

    pub fn named(value: f64, name_idx: u16) -> Self {
        Self { value, name_idx }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PValueWrapper(pub f64);

impl PValueWrapper {
    pub fn new(p: f64) -> Self {
        Self(p.clamp(0.0, 1.0))
    }

    pub fn is_significant(&self, alpha: f64) -> bool {
        self.0 < alpha
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EquilibriumRef(pub String);

impl EquilibriumRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameSpec {
    pub num_players: usize,
    pub market_type: String,
    pub demand_params: Vec<f64>,
    pub marginal_costs: Vec<f64>,
    pub discount_factor: f64,
}

impl GameSpec {
    pub fn new(num_players: usize, market_type: &str) -> Self {
        Self {
            num_players,
            market_type: market_type.to_string(),
            demand_params: Vec::new(),
            marginal_costs: Vec::new(),
            discount_factor: 0.95,
        }
    }

    pub fn from_game_config(config: &shared_types::GameConfig) -> Self {
        let demand_params = match &config.demand_system {
            shared_types::DemandSystem::Linear {
                max_quantity,
                slope,
            } => vec![*max_quantity, *slope],
            shared_types::DemandSystem::Logit { temperature, outside_option_value, market_size } => {
                vec![*temperature, *outside_option_value, *market_size]
            }
            shared_types::DemandSystem::CES { elasticity_of_substitution, market_size, quality_indices } => {
                let mut params = vec![*elasticity_of_substitution, *market_size];
                params.extend(quality_indices.iter().copied());
                params
            }
        };
        let market_type = match &config.market_type {
            shared_types::MarketType::Bertrand => "Bertrand",
            shared_types::MarketType::Cournot => "Cournot",
        };
        Self {
            num_players: config.num_players,
            market_type: market_type.to_string(),
            demand_params,
            marginal_costs: config.marginal_costs.iter().map(|c| c.0).collect(),
            discount_factor: config.discount_factor,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashProfile {
    pub prices: Vec<f64>,
    pub profits: Vec<f64>,
    pub is_symmetric: bool,
}

impl NashProfile {
    pub fn new(prices: Vec<f64>, profits: Vec<f64>) -> Self {
        let is_symmetric = if prices.len() > 1 {
            let first = prices[0];
            prices.iter().all(|p| (p - first).abs() < 1e-10)
        } else {
            true
        };
        Self {
            prices,
            profits,
            is_symmetric,
        }
    }

    pub fn from_nash_eq(eq: &game_theory::NashEquilibrium) -> Self {
        Self {
            prices: eq.strategy_profile.iter().map(|&s| s as f64).collect(),
            profits: eq.payoffs.clone(),
            is_symmetric: eq.is_symmetric,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DeviationRef(pub String);

impl DeviationRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bound {
    pub value: f64,
    pub bound_type: BoundType,
}

impl Bound {
    pub fn upper(value: f64) -> Self {
        Self {
            value,
            bound_type: BoundType::Upper,
        }
    }

    pub fn lower(value: f64) -> Self {
        Self {
            value,
            bound_type: BoundType::Lower,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BoundType {
    Upper,
    Lower,
    Exact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PunishmentRef(pub String);

impl PunishmentRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PayoffDrop {
    pub value: f64,
    pub relative: f64,
}

impl PayoffDrop {
    pub fn new(value: f64, relative: f64) -> Self {
        Self { value, relative }
    }

    pub fn from_absolute(value: f64, baseline: f64) -> Self {
        let relative = if baseline.abs() > 1e-12 {
            value / baseline
        } else {
            0.0
        };
        Self { value, relative }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CPRef(pub String);

impl CPRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Value(pub f64);

impl Value {
    pub fn new(v: f64) -> Self {
        Self(v)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CIWrapper {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

impl CIWrapper {
    pub fn new(lower: f64, upper: f64, level: f64) -> Self {
        Self { lower, upper, level }
    }

    pub fn from_ci(ci: &ConfidenceInterval) -> Self {
        Self {
            lower: ci.lower,
            upper: ci.upper,
            level: ci.level,
        }
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct InferenceRef(pub String);

impl InferenceRef {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    pub params: Vec<f64>,
}

impl Rule {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            params: Vec::new(),
        }
    }

    pub fn with_params(name: &str, params: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            params,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Premises {
    pub refs: Vec<String>,
}

impl Premises {
    pub fn new(refs: Vec<String>) -> Self {
        Self { refs }
    }

    pub fn single(r: &str) -> Self {
        Self {
            refs: vec![r.to_string()],
        }
    }

    pub fn pair(a: &str, b: &str) -> Self {
        Self {
            refs: vec![a.to_string(), b.to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conclusion {
    pub statement: String,
    pub expression: Option<Expression>,
}

impl Conclusion {
    pub fn new(statement: &str) -> Self {
        Self {
            statement: statement.to_string(),
            expression: None,
        }
    }

    pub fn with_expr(statement: &str, expr: Expression) -> Self {
        Self {
            statement: statement.to_string(),
            expression: Some(expr),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerdictType {
    Collusive,
    Competitive,
    Inconclusive,
}

impl fmt::Display for VerdictType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerdictType::Collusive => write!(f, "COLLUSIVE"),
            VerdictType::Competitive => write!(f, "COMPETITIVE"),
            VerdictType::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Confidence(pub f64);

impl Confidence {
    pub fn new(c: f64) -> Self {
        Self(c.clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportingRefs {
    pub refs: Vec<String>,
}

impl SupportingRefs {
    pub fn new(refs: Vec<String>) -> Self {
        Self { refs }
    }
}

// ── Expression enum for proof terms ──────────────────────────────────────────

/// Typed expressions in the proof term language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    Literal(LiteralValue),
    Variable(String),
    BinaryExpr {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    UnaryExpr {
        op: UnaryOp,
        operand: Box<Expression>,
    },
    Comparison {
        op: ComparisonOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    Conditional {
        condition: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>,
    },
    IntervalExpr(Box<Expression>, Box<Expression>),
    Annotated {
        expr: Box<Expression>,
        annotation: TypeAnnotation,
    },
}

impl Expression {
    pub fn rational(numer: i64, denom: i64) -> Self {
        Expression::Literal(LiteralValue::Rational(RationalLiteral { numer, denom }))
    }

    pub fn float(v: f64) -> Self {
        Expression::Literal(LiteralValue::Float(v))
    }

    pub fn boolean(b: bool) -> Self {
        Expression::Literal(LiteralValue::Bool(BoolLiteral(b)))
    }

    pub fn interval(lo: f64, hi: f64) -> Self {
        Expression::Literal(LiteralValue::Interval(IntervalLiteral { lo, hi }))
    }

    pub fn var(name: &str) -> Self {
        Expression::Variable(name.to_string())
    }

    pub fn add(left: Expression, right: Expression) -> Self {
        Expression::BinaryExpr {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn sub(left: Expression, right: Expression) -> Self {
        Expression::BinaryExpr {
            op: BinaryOp::Sub,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn mul(left: Expression, right: Expression) -> Self {
        Expression::BinaryExpr {
            op: BinaryOp::Mul,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn div(left: Expression, right: Expression) -> Self {
        Expression::BinaryExpr {
            op: BinaryOp::Div,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn lt(left: Expression, right: Expression) -> Self {
        Expression::Comparison {
            op: ComparisonOp::Lt,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn le(left: Expression, right: Expression) -> Self {
        Expression::Comparison {
            op: ComparisonOp::Le,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn gt(left: Expression, right: Expression) -> Self {
        Expression::Comparison {
            op: ComparisonOp::Gt,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn ge(left: Expression, right: Expression) -> Self {
        Expression::Comparison {
            op: ComparisonOp::Ge,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn eq(left: Expression, right: Expression) -> Self {
        Expression::Comparison {
            op: ComparisonOp::Eq,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Evaluate a simple expression to an f64, returning None for non-numeric leaves.
    pub fn try_eval_f64(&self) -> Option<f64> {
        match self {
            Expression::Literal(LiteralValue::Float(v)) => Some(*v),
            Expression::Literal(LiteralValue::Rational(r)) => {
                if r.denom == 0 {
                    None
                } else {
                    Some(r.numer as f64 / r.denom as f64)
                }
            }
            Expression::BinaryExpr { op, left, right } => {
                let l = left.try_eval_f64()?;
                let r = right.try_eval_f64()?;
                Some(match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => {
                        if r.abs() < 1e-300 {
                            return None;
                        }
                        l / r
                    }
                    BinaryOp::Pow => l.powf(r),
                    BinaryOp::Min => l.min(r),
                    BinaryOp::Max => l.max(r),
                })
            }
            Expression::UnaryExpr { op, operand } => {
                let v = operand.try_eval_f64()?;
                Some(match op {
                    UnaryOp::Neg => -v,
                    UnaryOp::Abs => v.abs(),
                    UnaryOp::Sqrt => {
                        if v < 0.0 {
                            return None;
                        }
                        v.sqrt()
                    }
                    UnaryOp::Log => {
                        if v <= 0.0 {
                            return None;
                        }
                        v.ln()
                    }
                })
            }
            _ => None,
        }
    }

    /// Count AST nodes.
    pub fn node_count(&self) -> usize {
        match self {
            Expression::Literal(_) | Expression::Variable(_) => 1,
            Expression::BinaryExpr { left, right, .. }
            | Expression::Comparison { left, right, .. }
            | Expression::IntervalExpr(left, right) => {
                1 + left.node_count() + right.node_count()
            }
            Expression::UnaryExpr { operand, .. } => 1 + operand.node_count(),
            Expression::FunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.node_count()).sum::<usize>()
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                1 + condition.node_count()
                    + then_expr.node_count()
                    + else_expr.node_count()
            }
            Expression::Annotated { expr, .. } => 1 + expr.node_count(),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Literal(lit) => write!(f, "{}", lit),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::BinaryExpr { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
            Expression::UnaryExpr { op, operand } => write!(f, "{}({})", op, operand),
            Expression::Comparison { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
            Expression::FunctionCall { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
                write!(f, "{}({})", name, arg_strs.join(", "))
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                write!(f, "if {} then {} else {}", condition, then_expr, else_expr)
            }
            Expression::IntervalExpr(lo, hi) => write!(f, "[{}, {}]", lo, hi),
            Expression::Annotated { expr, annotation } => {
                write!(f, "({} : {:?})", expr, annotation)
            }
        }
    }
}

// ── Literal types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiteralValue {
    Float(f64),
    Rational(RationalLiteral),
    Interval(IntervalLiteral),
    Bool(BoolLiteral),
    String(String),
    Integer(i64),
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiteralValue::Float(v) => write!(f, "{:.6}", v),
            LiteralValue::Rational(r) => write!(f, "{}/{}", r.numer, r.denom),
            LiteralValue::Interval(i) => write!(f, "[{:.6}, {:.6}]", i.lo, i.hi),
            LiteralValue::Bool(b) => write!(f, "{}", b.0),
            LiteralValue::String(s) => write!(f, "\"{}\"", s),
            LiteralValue::Integer(i) => write!(f, "{}", i),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RationalLiteral {
    pub numer: i64,
    pub denom: i64,
}

impl RationalLiteral {
    pub fn new(numer: i64, denom: i64) -> Self {
        assert!(denom != 0, "denominator must be nonzero");
        Self { numer, denom }
    }

    pub fn to_f64(&self) -> f64 {
        self.numer as f64 / self.denom as f64
    }

    pub fn is_zero(&self) -> bool {
        self.numer == 0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntervalLiteral {
    pub lo: f64,
    pub hi: f64,
}

impl IntervalLiteral {
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoolLiteral(pub bool);

// ── Operator enums ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Pow => write!(f, "^"),
            BinaryOp::Min => write!(f, "min"),
            BinaryOp::Max => write!(f, "max"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Log,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Abs => write!(f, "abs"),
            UnaryOp::Sqrt => write!(f, "sqrt"),
            UnaryOp::Log => write!(f, "log"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl ComparisonOp {
    /// Evaluate a comparison on two f64 values.
    pub fn eval_f64(&self, a: f64, b: f64) -> bool {
        match self {
            ComparisonOp::Lt => a < b,
            ComparisonOp::Le => a <= b,
            ComparisonOp::Gt => a > b,
            ComparisonOp::Ge => a >= b,
            ComparisonOp::Eq => (a - b).abs() < 1e-12,
            ComparisonOp::Ne => (a - b).abs() >= 1e-12,
        }
    }

    /// Return the flipped comparison (swap operands).
    pub fn flip(&self) -> Self {
        match self {
            ComparisonOp::Lt => ComparisonOp::Gt,
            ComparisonOp::Le => ComparisonOp::Ge,
            ComparisonOp::Gt => ComparisonOp::Lt,
            ComparisonOp::Ge => ComparisonOp::Le,
            ComparisonOp::Eq => ComparisonOp::Eq,
            ComparisonOp::Ne => ComparisonOp::Ne,
        }
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonOp::Lt => write!(f, "<"),
            ComparisonOp::Le => write!(f, "<="),
            ComparisonOp::Gt => write!(f, ">"),
            ComparisonOp::Ge => write!(f, ">="),
            ComparisonOp::Eq => write!(f, "=="),
            ComparisonOp::Ne => write!(f, "!="),
        }
    }
}

// ── Type annotation ──────────────────────────────────────────────────────────

/// Phantom-type segment annotation for type-safety in proof checking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TypeAnnotation {
    TrainingData,
    TestingData,
    ValidationData,
    HoldoutData,
    AggregatedResult,
    RationalVerified,
    IntervalBounded,
    Custom(String),
}

impl TypeAnnotation {
    /// Returns true if this annotation represents raw data segments.
    pub fn is_data_segment(&self) -> bool {
        matches!(
            self,
            TypeAnnotation::TrainingData
                | TypeAnnotation::TestingData
                | TypeAnnotation::ValidationData
                | TypeAnnotation::HoldoutData
        )
    }

    /// Returns true if two annotations are compatible for mixing in a proof step.
    pub fn compatible_with(&self, other: &TypeAnnotation) -> bool {
        if self == other {
            return true;
        }
        // Non-data types can be combined freely
        if !self.is_data_segment() || !other.is_data_segment() {
            return true;
        }
        // Different data segments cannot be mixed
        false
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::OracleAccessLevel;

    #[test]
    fn test_certificate_ast_creation() {
        let header = CertificateHeader::new("test_scenario", OracleAccessLevel::Layer0, 0.05);
        let body = CertificateBody::new();
        let cert = CertificateAST::new(header, body);
        assert_eq!(cert.step_count(), 0);
        assert!(cert.declared_refs().is_empty());
    }

    #[test]
    fn test_certificate_header() {
        let header = CertificateHeader::new("scn1", OracleAccessLevel::Layer2, 0.01);
        assert_eq!(header.scenario, "scn1");
        assert_eq!(header.oracle_level, OracleAccessLevel::Layer2);
        assert!((header.alpha.value() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_proof_step_data_declaration() {
        let step = ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 100, 500, "abc123", 2),
        );
        assert_eq!(step.declared_ref(), Some("traj_0".to_string()));
        assert_eq!(step.kind(), "DataDeclaration");
        assert!(step.dependency_refs().is_empty());
    }

    #[test]
    fn test_proof_step_statistical_test() {
        let step = ProofStep::StatisticalTest(
            TestRef::new("test_corr"),
            TestType::new("PriceCorrelation", "layer0"),
            Statistic::new(3.14),
            PValueWrapper::new(0.001),
        );
        assert_eq!(step.declared_ref(), Some("test_corr".to_string()));
        assert!(step.summary().contains("3.14"));
    }

    #[test]
    fn test_proof_step_verdict() {
        let step = ProofStep::Verdict(
            VerdictType::Collusive,
            Confidence::new(0.95),
            SupportingRefs::new(vec!["ref1".into(), "ref2".into()]),
        );
        assert_eq!(step.declared_ref(), None);
        assert_eq!(step.dependency_refs(), vec!["ref1", "ref2"]);
    }

    #[test]
    fn test_expression_arithmetic() {
        let expr = Expression::add(Expression::float(2.0), Expression::float(3.0));
        assert!((expr.try_eval_f64().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_expression_rational() {
        let expr = Expression::rational(1, 3);
        let val = expr.try_eval_f64().unwrap();
        assert!((val - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_expression_nested() {
        // (2 + 3) * 4 = 20
        let inner = Expression::add(Expression::float(2.0), Expression::float(3.0));
        let expr = Expression::mul(inner, Expression::float(4.0));
        assert!((expr.try_eval_f64().unwrap() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_expression_div_by_zero() {
        let expr = Expression::div(Expression::float(1.0), Expression::float(0.0));
        assert!(expr.try_eval_f64().is_none());
    }

    #[test]
    fn test_expression_node_count() {
        let expr = Expression::add(
            Expression::mul(Expression::float(2.0), Expression::var("x")),
            Expression::float(1.0),
        );
        assert_eq!(expr.node_count(), 5);
    }

    #[test]
    fn test_expression_display() {
        let expr = Expression::lt(Expression::var("p"), Expression::float(0.05));
        let s = format!("{}", expr);
        assert!(s.contains("<"));
        assert!(s.contains("p"));
    }

    #[test]
    fn test_comparison_op_eval() {
        assert!(ComparisonOp::Lt.eval_f64(1.0, 2.0));
        assert!(!ComparisonOp::Lt.eval_f64(2.0, 1.0));
        assert!(ComparisonOp::Ge.eval_f64(2.0, 2.0));
        assert!(ComparisonOp::Ne.eval_f64(1.0, 2.0));
    }

    #[test]
    fn test_comparison_op_flip() {
        assert_eq!(ComparisonOp::Lt.flip(), ComparisonOp::Gt);
        assert_eq!(ComparisonOp::Le.flip(), ComparisonOp::Ge);
        assert_eq!(ComparisonOp::Eq.flip(), ComparisonOp::Eq);
    }

    #[test]
    fn test_type_annotation_compatibility() {
        assert!(TypeAnnotation::TrainingData.compatible_with(&TypeAnnotation::TrainingData));
        assert!(!TypeAnnotation::TrainingData.compatible_with(&TypeAnnotation::TestingData));
        assert!(TypeAnnotation::RationalVerified.compatible_with(&TypeAnnotation::IntervalBounded));
        assert!(TypeAnnotation::TrainingData.compatible_with(&TypeAnnotation::RationalVerified));
    }

    #[test]
    fn test_segment_spec() {
        let spec = SegmentSpec::new("testing", 100, 500, "deadbeef", 2);
        assert_eq!(spec.num_rounds(), 400);
        assert_eq!(spec.num_players, 2);
    }

    #[test]
    fn test_pvalue_wrapper() {
        let pv = PValueWrapper::new(0.03);
        assert!(pv.is_significant(0.05));
        assert!(!pv.is_significant(0.01));
    }

    #[test]
    fn test_certificate_pretty_print() {
        let header = CertificateHeader::new("scenario_1", OracleAccessLevel::Layer1, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "hash", 2),
        ));
        let cert = CertificateAST::new(header, body);
        let pp = cert.pretty_print();
        assert!(pp.contains("Certificate v1.0.0"));
        assert!(pp.contains("scenario_1"));
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        let header = CertificateHeader::new("s1", OracleAccessLevel::Layer0, 0.05);
        let body = CertificateBody::new();
        let cert = CertificateAST::new(header, body);
        let json = cert.to_json().unwrap();
        let cert2 = CertificateAST::from_json(&json).unwrap();
        assert_eq!(cert2.header.scenario, "s1");
    }

    #[test]
    fn test_nash_profile_from_eq() {
        let eq = game_theory::NashEquilibrium {
            prices: vec![3.0, 3.0],
            profits: vec![4.0, 4.0],
            is_symmetric: true,
        };
        let profile = NashProfile::from_nash_eq(&eq);
        assert!(profile.is_symmetric);
        assert_eq!(profile.prices, vec![3.0, 3.0]);
    }

    #[test]
    fn test_game_spec_from_config() {
        let config = shared_types::GameConfig::default();
        let spec = GameSpec::from_game_config(&config);
        assert_eq!(spec.num_players, 2);
        assert_eq!(spec.market_type, "Bertrand");
    }

    #[test]
    fn test_rational_literal() {
        let r = RationalLiteral::new(1, 4);
        assert!((r.to_f64() - 0.25).abs() < 1e-12);
        assert!(!r.is_zero());
        assert!(RationalLiteral::new(0, 5).is_zero());
    }

    #[test]
    fn test_interval_literal() {
        let i = IntervalLiteral::new(1.0, 3.0);
        assert!((i.width() - 2.0).abs() < 1e-12);
        assert!(i.contains(2.0));
        assert!(!i.contains(4.0));
        assert!((i.midpoint() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_verdict_type_display() {
        assert_eq!(format!("{}", VerdictType::Collusive), "COLLUSIVE");
        assert_eq!(format!("{}", VerdictType::Competitive), "COMPETITIVE");
        assert_eq!(format!("{}", VerdictType::Inconclusive), "INCONCLUSIVE");
    }

    #[test]
    fn test_ci_wrapper() {
        let ci = CIWrapper::new(0.1, 0.5, 0.95);
        assert!((ci.width() - 0.4).abs() < 1e-12);
        assert!(ci.contains(0.3));
        assert!(!ci.contains(0.6));
    }
}
