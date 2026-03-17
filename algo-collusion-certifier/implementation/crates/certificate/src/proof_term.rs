//! Proof term language for CollusionProof certificates.
//!
//! Defines the core proof language including proof terms, axiom schemas,
//! inference rules, and instantiation mechanisms.

use crate::ast::{ComparisonOp, Expression};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Core proof term enum ─────────────────────────────────────────────────────

/// A proof term in the CollusionProof certification language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofTerm {
    /// An axiom schema instantiated with specific values.
    Axiom(AxiomSchema, Instantiation),
    /// Modus ponens: from (P → Q) and P, derive Q.
    ModusPonens(Box<ProofTerm>, Box<ProofTerm>),
    /// Conjunction introduction: from P and Q, derive P ∧ Q.
    Conjunction(Box<ProofTerm>, Box<ProofTerm>),
    /// Disjunction: P ∨ Q.
    Disjunction(Box<ProofTerm>, Box<ProofTerm>),
    /// Universal instantiation: from ∀x.P(x) and value v, derive P(v).
    UniversalInstantiation(Box<ProofTerm>, ProofValue),
    /// A statistical bound: distribution satisfies bound at confidence level.
    StatisticalBound(Distribution, BoundSpec, f64),
    /// An arithmetic fact: expression relation expression.
    ArithmeticFact(Expression, Relation, Expression),
    /// Interval containment: value lies within interval.
    IntervalContainment(ProofValue, IntervalSpec),
    /// Transitivity chain: a sequence of proof terms forming a chain.
    TransitivityChain(Vec<ProofTerm>),
    /// Reference to a previously established fact.
    Reference(String),
    /// Inference rule application.
    RuleApplication(InferenceRule, Vec<ProofTerm>),
    /// An assumption (hypothesis) introduced into context.
    Assumption(String, Proposition),
    /// Let-binding for intermediate results.
    LetBinding(String, Box<ProofTerm>, Box<ProofTerm>),
}

impl ProofTerm {
    /// Count the total nodes in this proof term tree.
    pub fn node_count(&self) -> usize {
        match self {
            ProofTerm::Axiom(..) | ProofTerm::Reference(_) => 1,
            ProofTerm::ModusPonens(a, b)
            | ProofTerm::Conjunction(a, b)
            | ProofTerm::Disjunction(a, b) => 1 + a.node_count() + b.node_count(),
            ProofTerm::UniversalInstantiation(p, _) => 1 + p.node_count(),
            ProofTerm::StatisticalBound(..)
            | ProofTerm::ArithmeticFact(..)
            | ProofTerm::IntervalContainment(..)
            | ProofTerm::Assumption(..) => 1,
            ProofTerm::TransitivityChain(chain) => {
                1 + chain.iter().map(|t| t.node_count()).sum::<usize>()
            }
            ProofTerm::RuleApplication(_, premises) => {
                1 + premises.iter().map(|t| t.node_count()).sum::<usize>()
            }
            ProofTerm::LetBinding(_, bound, body) => {
                1 + bound.node_count() + body.node_count()
            }
        }
    }

    /// Depth of the proof term tree.
    pub fn depth(&self) -> usize {
        match self {
            ProofTerm::Axiom(..)
            | ProofTerm::Reference(_)
            | ProofTerm::StatisticalBound(..)
            | ProofTerm::ArithmeticFact(..)
            | ProofTerm::IntervalContainment(..)
            | ProofTerm::Assumption(..) => 1,
            ProofTerm::ModusPonens(a, b)
            | ProofTerm::Conjunction(a, b)
            | ProofTerm::Disjunction(a, b) => 1 + a.depth().max(b.depth()),
            ProofTerm::UniversalInstantiation(p, _) => 1 + p.depth(),
            ProofTerm::TransitivityChain(chain) => {
                1 + chain.iter().map(|t| t.depth()).max().unwrap_or(0)
            }
            ProofTerm::RuleApplication(_, premises) => {
                1 + premises.iter().map(|t| t.depth()).max().unwrap_or(0)
            }
            ProofTerm::LetBinding(_, bound, body) => {
                1 + bound.depth().max(body.depth())
            }
        }
    }

    /// Collect all references (by name) used in this proof.
    pub fn collect_references(&self) -> Vec<String> {
        let mut refs = Vec::new();
        self.collect_refs_inner(&mut refs);
        refs.sort();
        refs.dedup();
        refs
    }

    fn collect_refs_inner(&self, refs: &mut Vec<String>) {
        match self {
            ProofTerm::Reference(r) => refs.push(r.clone()),
            ProofTerm::ModusPonens(a, b)
            | ProofTerm::Conjunction(a, b)
            | ProofTerm::Disjunction(a, b) => {
                a.collect_refs_inner(refs);
                b.collect_refs_inner(refs);
            }
            ProofTerm::UniversalInstantiation(p, _) => p.collect_refs_inner(refs),
            ProofTerm::TransitivityChain(chain) => {
                for t in chain {
                    t.collect_refs_inner(refs);
                }
            }
            ProofTerm::RuleApplication(_, premises) => {
                for t in premises {
                    t.collect_refs_inner(refs);
                }
            }
            ProofTerm::LetBinding(_, bound, body) => {
                bound.collect_refs_inner(refs);
                body.collect_refs_inner(refs);
            }
            _ => {}
        }
    }

    /// Summary string for display.
    pub fn summary(&self) -> String {
        match self {
            ProofTerm::Axiom(schema, _) => format!("Axiom({:?})", schema),
            ProofTerm::ModusPonens(_, _) => "ModusPonens(..)".to_string(),
            ProofTerm::Conjunction(_, _) => "Conjunction(..)".to_string(),
            ProofTerm::Disjunction(_, _) => "Disjunction(..)".to_string(),
            ProofTerm::UniversalInstantiation(_, v) => {
                format!("∀Elim(value={:?})", v)
            }
            ProofTerm::StatisticalBound(dist, bound, conf) => {
                format!("StatBound({:?}, {:?}, conf={:.4})", dist, bound, conf)
            }
            ProofTerm::ArithmeticFact(l, rel, r) => {
                format!("Arith({} {} {})", l, rel, r)
            }
            ProofTerm::IntervalContainment(val, interval) => {
                format!("IntvContain({:?} ∈ {:?})", val, interval)
            }
            ProofTerm::TransitivityChain(chain) => {
                format!("Trans(len={})", chain.len())
            }
            ProofTerm::Reference(r) => format!("Ref({})", r),
            ProofTerm::RuleApplication(rule, premises) => {
                format!("Rule({:?}, premises={})", rule, premises.len())
            }
            ProofTerm::Assumption(name, _) => format!("Assume({})", name),
            ProofTerm::LetBinding(name, _, _) => format!("Let({})", name),
        }
    }
}

impl fmt::Display for ProofTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ── Axiom schemas ────────────────────────────────────────────────────────────

/// Axiom schemas for the CollusionProof certification system.
/// Each schema encodes a verified domain-semantic fact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AxiomSchema {
    /// Definition of competitive null hypothesis H0.
    CompetitiveNullDef,
    /// If a test is α-sound, rejection implies evidence at level α.
    TestSoundness,
    /// Maximum correlation under competitive null.
    CorrelationBound,
    /// C3' theorem: deviation existence criterion.
    DeviationExistence,
    /// Punishment produces observable payoff drop.
    PunishmentDetectability,
    /// Collusion premium definition and bounds.
    CollusionPremiumDef,
    /// Family-wise error rate control: FWER ≤ α.
    FWERControl,
    /// Interval arithmetic computation rules.
    IntervalArithmetic,
    /// Rational arithmetic preserves orderings.
    RationalExactness,
    /// Profit is monotone in own price (under conditions).
    MonotonicityOfProfit,
    /// Nash equilibrium characterization.
    NashEquilibriumDef,
    /// Minimax lower bound on individual rationality.
    IndividualRationalityBound,
    /// Decompose observed payoff into components.
    PayoffDecomposition,
    /// Phantom-type segments are independent.
    SegmentIndependence,
    /// Berry-Esseen bound for finite-sample CLT correction.
    BerryEsseenBound,
}

impl AxiomSchema {
    /// Return the required number of instantiation parameters.
    pub fn required_params(&self) -> usize {
        match self {
            AxiomSchema::CompetitiveNullDef => 2,
            AxiomSchema::TestSoundness => 3,
            AxiomSchema::CorrelationBound => 2,
            AxiomSchema::DeviationExistence => 4,
            AxiomSchema::PunishmentDetectability => 3,
            AxiomSchema::CollusionPremiumDef => 3,
            AxiomSchema::FWERControl => 2,
            AxiomSchema::IntervalArithmetic => 4,
            AxiomSchema::RationalExactness => 2,
            AxiomSchema::MonotonicityOfProfit => 3,
            AxiomSchema::NashEquilibriumDef => 2,
            AxiomSchema::IndividualRationalityBound => 2,
            AxiomSchema::PayoffDecomposition => 3,
            AxiomSchema::SegmentIndependence => 2,
            AxiomSchema::BerryEsseenBound => 3,
        }
    }

    /// Human-readable description of the axiom.
    pub fn description(&self) -> &'static str {
        match self {
            AxiomSchema::CompetitiveNullDef => {
                "Under H0, prices are generated by independent competitive algorithms"
            }
            AxiomSchema::TestSoundness => {
                "An α-sound test rejecting H0 provides evidence at significance α"
            }
            AxiomSchema::CorrelationBound => {
                "Under competitive play, price correlation is bounded by market structure"
            }
            AxiomSchema::DeviationExistence => {
                "C3' theorem: if CP > 0, profitable deviations exist"
            }
            AxiomSchema::PunishmentDetectability => {
                "Punishment by other players causes observable payoff drop"
            }
            AxiomSchema::CollusionPremiumDef => {
                "CP = (observed_profit - nash_profit) / (collusive_profit - nash_profit)"
            }
            AxiomSchema::FWERControl => "FWER ≤ α under Holm-Bonferroni correction",
            AxiomSchema::IntervalArithmetic => {
                "Interval arithmetic rules preserve containment"
            }
            AxiomSchema::RationalExactness => {
                "Rational arithmetic preserves comparison orderings"
            }
            AxiomSchema::MonotonicityOfProfit => {
                "Profit is monotone in own price below optimal"
            }
            AxiomSchema::NashEquilibriumDef => {
                "At Nash equilibrium, no player can profitably deviate"
            }
            AxiomSchema::IndividualRationalityBound => {
                "Each player achieves at least minimax payoff"
            }
            AxiomSchema::PayoffDecomposition => {
                "Observed payoff = Nash + premium component"
            }
            AxiomSchema::SegmentIndependence => {
                "Differently-typed trajectory segments are statistically independent"
            }
            AxiomSchema::BerryEsseenBound => {
                "CLT error ≤ C * E[|X|³] / (σ³ √n)"
            }
        }
    }
}

// ── Inference rules ──────────────────────────────────────────────────────────

/// Inference rules for the proof term language.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InferenceRule {
    AndIntro,
    AndElim,
    OrIntro,
    OrElim,
    ImplicationElim,
    UniversalElim,
    ExistentialIntro,
    ChainRule,
    ContraPositive,
    AlphaSpending,
    IntervalRefine,
    BootstrapCI,
    CompositeRejection,
    MonotonePriceComparison,
    ConfidenceComposition,
    ErrorPropagation,
    NumericalVerification,
    SegmentSplit,
    TestComposition,
    HolmCorrection,
    PayoffComparison,
    DeviationInference,
    PunishmentInference,
    VerdictDerivation,
    WeakeningRule,
}

impl InferenceRule {
    /// Minimum number of premises required by this rule.
    pub fn min_premises(&self) -> usize {
        match self {
            InferenceRule::AndIntro => 2,
            InferenceRule::AndElim => 1,
            InferenceRule::OrIntro => 1,
            InferenceRule::OrElim => 3,
            InferenceRule::ImplicationElim => 2,
            InferenceRule::UniversalElim => 1,
            InferenceRule::ExistentialIntro => 1,
            InferenceRule::ChainRule => 2,
            InferenceRule::ContraPositive => 1,
            InferenceRule::AlphaSpending => 1,
            InferenceRule::IntervalRefine => 1,
            InferenceRule::BootstrapCI => 1,
            InferenceRule::CompositeRejection => 2,
            InferenceRule::MonotonePriceComparison => 2,
            InferenceRule::ConfidenceComposition => 2,
            InferenceRule::ErrorPropagation => 1,
            InferenceRule::NumericalVerification => 1,
            InferenceRule::SegmentSplit => 1,
            InferenceRule::TestComposition => 2,
            InferenceRule::HolmCorrection => 1,
            InferenceRule::PayoffComparison => 2,
            InferenceRule::DeviationInference => 1,
            InferenceRule::PunishmentInference => 1,
            InferenceRule::VerdictDerivation => 1,
            InferenceRule::WeakeningRule => 1,
        }
    }

    /// Human-readable description of the rule.
    pub fn description(&self) -> &'static str {
        match self {
            InferenceRule::AndIntro => "From P and Q, derive P ∧ Q",
            InferenceRule::AndElim => "From P ∧ Q, derive P (or Q)",
            InferenceRule::OrIntro => "From P, derive P ∨ Q",
            InferenceRule::OrElim => "From P ∨ Q and (P→R) and (Q→R), derive R",
            InferenceRule::ImplicationElim => "Modus ponens: from P→Q and P, derive Q",
            InferenceRule::UniversalElim => "From ∀x.P(x), derive P(c) for any c",
            InferenceRule::ExistentialIntro => "From P(c), derive ∃x.P(x)",
            InferenceRule::ChainRule => "Transitivity of implications",
            InferenceRule::ContraPositive => "From P→Q, derive ¬Q→¬P",
            InferenceRule::AlphaSpending => "Consume alpha budget for a test",
            InferenceRule::IntervalRefine => "Narrow interval bounds using new evidence",
            InferenceRule::BootstrapCI => "Derive confidence interval from bootstrap",
            InferenceRule::CompositeRejection => "Combine sub-test rejections",
            InferenceRule::MonotonePriceComparison => "Compare prices via monotonicity",
            InferenceRule::ConfidenceComposition => "Compose confidence levels",
            InferenceRule::ErrorPropagation => "Propagate error bounds through computation",
            InferenceRule::NumericalVerification => {
                "Verify f64 claim via rational arithmetic"
            }
            InferenceRule::SegmentSplit => "Split trajectory into typed segments",
            InferenceRule::TestComposition => "Compose independent test results",
            InferenceRule::HolmCorrection => "Apply Holm-Bonferroni correction",
            InferenceRule::PayoffComparison => "Compare payoffs with certified bounds",
            InferenceRule::DeviationInference => "Infer deviation existence from C3'",
            InferenceRule::PunishmentInference => "Infer punishment from test result",
            InferenceRule::VerdictDerivation => "Derive final verdict from evidence chain",
            InferenceRule::WeakeningRule => "Weaken a conclusion (adds slack)",
        }
    }

    /// Whether this rule consumes alpha budget.
    pub fn consumes_alpha(&self) -> bool {
        matches!(
            self,
            InferenceRule::AlphaSpending
                | InferenceRule::HolmCorrection
                | InferenceRule::TestComposition
                | InferenceRule::CompositeRejection
        )
    }
}

// ── Supporting types ─────────────────────────────────────────────────────────

/// Instantiation bindings for an axiom schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instantiation {
    pub bindings: Vec<(String, ProofValue)>,
}

impl Instantiation {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    pub fn bind(mut self, name: &str, value: ProofValue) -> Self {
        self.bindings.push((name.to_string(), value));
        self
    }

    pub fn get(&self, name: &str) -> Option<&ProofValue> {
        self.bindings
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl Default for Instantiation {
    fn default() -> Self {
        Self::new()
    }
}

/// A value that can appear in proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofValue {
    Float(f64),
    Rational(i64, i64),
    Integer(i64),
    Bool(bool),
    Interval(f64, f64),
    String(String),
    Player(usize),
    Vector(Vec<f64>),
}

impl ProofValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ProofValue::Float(v) => Some(*v),
            ProofValue::Rational(n, d) => {
                if *d == 0 {
                    None
                } else {
                    Some(*n as f64 / *d as f64)
                }
            }
            ProofValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ProofValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_interval(&self) -> Option<(f64, f64)> {
        match self {
            ProofValue::Interval(lo, hi) => Some((*lo, *hi)),
            _ => None,
        }
    }

    pub fn as_player(&self) -> Option<usize> {
        match self {
            ProofValue::Player(p) => Some(*p),
            ProofValue::Integer(i) if *i >= 0 => Some(*i as usize),
            _ => None,
        }
    }
}

/// Specification of a probability distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    Normal { mean: f64, variance: f64 },
    ChiSquared { df: usize },
    StudentT { df: usize },
    Empirical { sample_size: usize },
    Bootstrap { num_resamples: usize },
    Unknown,
}

impl Distribution {
    pub fn is_known(&self) -> bool {
        !matches!(self, Distribution::Unknown)
    }
}

/// Specification of a bound on a distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundSpec {
    pub value: f64,
    pub direction: BoundDirection,
}

impl BoundSpec {
    pub fn upper(value: f64) -> Self {
        Self {
            value,
            direction: BoundDirection::Upper,
        }
    }

    pub fn lower(value: f64) -> Self {
        Self {
            value,
            direction: BoundDirection::Lower,
        }
    }

    pub fn two_sided(value: f64) -> Self {
        Self {
            value,
            direction: BoundDirection::TwoSided,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BoundDirection {
    Upper,
    Lower,
    TwoSided,
}

/// A mathematical relation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Relation {
    LessThan,
    LessOrEqual,
    Equal,
    GreaterOrEqual,
    GreaterThan,
    NotEqual,
}

impl Relation {
    pub fn eval_f64(&self, a: f64, b: f64) -> bool {
        match self {
            Relation::LessThan => a < b,
            Relation::LessOrEqual => a <= b,
            Relation::Equal => (a - b).abs() < 1e-12,
            Relation::GreaterOrEqual => a >= b,
            Relation::GreaterThan => a > b,
            Relation::NotEqual => (a - b).abs() >= 1e-12,
        }
    }

    pub fn from_comparison_op(op: ComparisonOp) -> Self {
        match op {
            ComparisonOp::Lt => Relation::LessThan,
            ComparisonOp::Le => Relation::LessOrEqual,
            ComparisonOp::Gt => Relation::GreaterThan,
            ComparisonOp::Ge => Relation::GreaterOrEqual,
            ComparisonOp::Eq => Relation::Equal,
            ComparisonOp::Ne => Relation::NotEqual,
        }
    }

    pub fn flip(&self) -> Self {
        match self {
            Relation::LessThan => Relation::GreaterThan,
            Relation::LessOrEqual => Relation::GreaterOrEqual,
            Relation::Equal => Relation::Equal,
            Relation::GreaterOrEqual => Relation::LessOrEqual,
            Relation::GreaterThan => Relation::LessThan,
            Relation::NotEqual => Relation::NotEqual,
        }
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Relation::LessThan => write!(f, "<"),
            Relation::LessOrEqual => write!(f, "≤"),
            Relation::Equal => write!(f, "="),
            Relation::GreaterOrEqual => write!(f, "≥"),
            Relation::GreaterThan => write!(f, ">"),
            Relation::NotEqual => write!(f, "≠"),
        }
    }
}

/// An interval specification in proof terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalSpec {
    pub lo: f64,
    pub hi: f64,
    pub closed_lo: bool,
    pub closed_hi: bool,
}

impl IntervalSpec {
    pub fn closed(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            closed_lo: true,
            closed_hi: true,
        }
    }

    pub fn open(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            closed_lo: false,
            closed_hi: false,
        }
    }

    pub fn contains(&self, v: f64) -> bool {
        let lo_ok = if self.closed_lo { v >= self.lo } else { v > self.lo };
        let hi_ok = if self.closed_hi { v <= self.hi } else { v < self.hi };
        lo_ok && hi_ok
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

/// A proposition (logical statement) in the proof system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Proposition {
    Atomic(String),
    And(Box<Proposition>, Box<Proposition>),
    Or(Box<Proposition>, Box<Proposition>),
    Implies(Box<Proposition>, Box<Proposition>),
    Not(Box<Proposition>),
    ForAll(String, Box<Proposition>),
    Exists(String, Box<Proposition>),
    Comparison(Expression, ComparisonOp, Expression),
}

impl Proposition {
    pub fn atomic(s: &str) -> Self {
        Proposition::Atomic(s.to_string())
    }

    pub fn and(p: Proposition, q: Proposition) -> Self {
        Proposition::And(Box::new(p), Box::new(q))
    }

    pub fn or(p: Proposition, q: Proposition) -> Self {
        Proposition::Or(Box::new(p), Box::new(q))
    }

    pub fn implies(p: Proposition, q: Proposition) -> Self {
        Proposition::Implies(Box::new(p), Box::new(q))
    }

    pub fn not(p: Proposition) -> Self {
        Proposition::Not(Box::new(p))
    }
}

impl fmt::Display for Proposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Proposition::Atomic(s) => write!(f, "{}", s),
            Proposition::And(p, q) => write!(f, "({} ∧ {})", p, q),
            Proposition::Or(p, q) => write!(f, "({} ∨ {})", p, q),
            Proposition::Implies(p, q) => write!(f, "({} → {})", p, q),
            Proposition::Not(p) => write!(f, "¬{}", p),
            Proposition::ForAll(x, p) => write!(f, "∀{}.{}", x, p),
            Proposition::Exists(x, p) => write!(f, "∃{}.{}", x, p),
            Proposition::Comparison(l, op, r) => write!(f, "({} {} {})", l, op, r),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expression;

    #[test]
    fn test_proof_term_axiom() {
        let inst = Instantiation::new()
            .bind("alpha", ProofValue::Float(0.05))
            .bind("n", ProofValue::Integer(100));
        let term = ProofTerm::Axiom(AxiomSchema::TestSoundness, inst);
        assert_eq!(term.node_count(), 1);
        assert_eq!(term.depth(), 1);
    }

    #[test]
    fn test_proof_term_modus_ponens() {
        let p = ProofTerm::Reference("P".to_string());
        let pq = ProofTerm::Reference("P_implies_Q".to_string());
        let mp = ProofTerm::ModusPonens(Box::new(pq), Box::new(p));
        assert_eq!(mp.node_count(), 3);
        assert_eq!(mp.depth(), 2);
    }

    #[test]
    fn test_proof_term_conjunction() {
        let a = ProofTerm::Reference("A".to_string());
        let b = ProofTerm::Reference("B".to_string());
        let conj = ProofTerm::Conjunction(Box::new(a), Box::new(b));
        assert_eq!(conj.node_count(), 3);
    }

    #[test]
    fn test_proof_term_transitivity_chain() {
        let chain = ProofTerm::TransitivityChain(vec![
            ProofTerm::Reference("step1".to_string()),
            ProofTerm::Reference("step2".to_string()),
            ProofTerm::Reference("step3".to_string()),
        ]);
        assert_eq!(chain.node_count(), 4);
        assert_eq!(chain.depth(), 2);
    }

    #[test]
    fn test_collect_references() {
        let term = ProofTerm::Conjunction(
            Box::new(ProofTerm::Reference("ref_a".to_string())),
            Box::new(ProofTerm::ModusPonens(
                Box::new(ProofTerm::Reference("ref_b".to_string())),
                Box::new(ProofTerm::Reference("ref_a".to_string())),
            )),
        );
        let refs = term.collect_references();
        assert_eq!(refs, vec!["ref_a", "ref_b"]);
    }

    #[test]
    fn test_axiom_schema_params() {
        assert_eq!(AxiomSchema::CompetitiveNullDef.required_params(), 2);
        assert_eq!(AxiomSchema::DeviationExistence.required_params(), 4);
        assert_eq!(AxiomSchema::BerryEsseenBound.required_params(), 3);
    }

    #[test]
    fn test_inference_rule_premises() {
        assert_eq!(InferenceRule::AndIntro.min_premises(), 2);
        assert_eq!(InferenceRule::AndElim.min_premises(), 1);
        assert_eq!(InferenceRule::OrElim.min_premises(), 3);
    }

    #[test]
    fn test_inference_rule_consumes_alpha() {
        assert!(InferenceRule::AlphaSpending.consumes_alpha());
        assert!(InferenceRule::HolmCorrection.consumes_alpha());
        assert!(!InferenceRule::AndIntro.consumes_alpha());
    }

    #[test]
    fn test_instantiation() {
        let inst = Instantiation::new()
            .bind("x", ProofValue::Float(1.0))
            .bind("y", ProofValue::Integer(2));
        assert_eq!(inst.len(), 2);
        assert!(inst.get("x").is_some());
        assert!(inst.get("z").is_none());
    }

    #[test]
    fn test_proof_value_conversions() {
        assert!((ProofValue::Float(3.14).as_f64().unwrap() - 3.14).abs() < 1e-12);
        assert!((ProofValue::Rational(1, 2).as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(ProofValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ProofValue::Interval(1.0, 2.0).as_interval(), Some((1.0, 2.0)));
        assert_eq!(ProofValue::Player(0).as_player(), Some(0));
    }

    #[test]
    fn test_relation_eval() {
        assert!(Relation::LessThan.eval_f64(1.0, 2.0));
        assert!(!Relation::LessThan.eval_f64(2.0, 1.0));
        assert!(Relation::Equal.eval_f64(1.0, 1.0));
        assert!(Relation::NotEqual.eval_f64(1.0, 2.0));
    }

    #[test]
    fn test_relation_flip() {
        assert_eq!(Relation::LessThan.flip(), Relation::GreaterThan);
        assert_eq!(Relation::LessOrEqual.flip(), Relation::GreaterOrEqual);
        assert_eq!(Relation::Equal.flip(), Relation::Equal);
    }

    #[test]
    fn test_interval_spec() {
        let closed = IntervalSpec::closed(0.0, 1.0);
        assert!(closed.contains(0.0));
        assert!(closed.contains(1.0));
        assert!(closed.contains(0.5));
        assert!(!closed.contains(1.1));

        let open = IntervalSpec::open(0.0, 1.0);
        assert!(!open.contains(0.0));
        assert!(!open.contains(1.0));
        assert!(open.contains(0.5));
    }

    #[test]
    fn test_bound_spec() {
        let ub = BoundSpec::upper(0.05);
        assert_eq!(ub.direction, BoundDirection::Upper);
        let lb = BoundSpec::lower(0.01);
        assert_eq!(lb.direction, BoundDirection::Lower);
    }

    #[test]
    fn test_distribution() {
        assert!(Distribution::Normal { mean: 0.0, variance: 1.0 }.is_known());
        assert!(!Distribution::Unknown.is_known());
    }

    #[test]
    fn test_proposition_display() {
        let p = Proposition::implies(
            Proposition::atomic("P"),
            Proposition::atomic("Q"),
        );
        let s = format!("{}", p);
        assert!(s.contains("→"));
    }

    #[test]
    fn test_proof_term_arithmetic_fact() {
        let term = ProofTerm::ArithmeticFact(
            Expression::float(0.03),
            Relation::LessThan,
            Expression::float(0.05),
        );
        assert_eq!(term.node_count(), 1);
        assert!(term.summary().contains("Arith"));
    }

    #[test]
    fn test_proof_term_statistical_bound() {
        let term = ProofTerm::StatisticalBound(
            Distribution::Normal {
                mean: 0.0,
                variance: 1.0,
            },
            BoundSpec::upper(1.96),
            0.95,
        );
        assert!(term.summary().contains("StatBound"));
    }

    #[test]
    fn test_proof_term_rule_application() {
        let term = ProofTerm::RuleApplication(
            InferenceRule::AndIntro,
            vec![
                ProofTerm::Reference("A".to_string()),
                ProofTerm::Reference("B".to_string()),
            ],
        );
        assert_eq!(term.node_count(), 3);
    }

    #[test]
    fn test_proof_term_let_binding() {
        let term = ProofTerm::LetBinding(
            "x".to_string(),
            Box::new(ProofTerm::Reference("val".to_string())),
            Box::new(ProofTerm::Reference("body".to_string())),
        );
        assert_eq!(term.node_count(), 3);
        assert_eq!(term.depth(), 2);
    }

    #[test]
    fn test_proof_value_rational_zero_denom() {
        assert!(ProofValue::Rational(1, 0).as_f64().is_none());
    }
}
