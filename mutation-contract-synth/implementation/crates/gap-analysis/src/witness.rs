//! # witness
//!
//! Gap witness generation for the MutSpec pipeline.
//!
//! A **gap witness** is a concrete piece of evidence that a surviving,
//! non-equivalent mutant is *not* distinguished by the current contract.
//! Each witness carries:
//!
//! - A **distinguishing input** that causes the mutant and original program to
//!   produce different outputs.
//! - The expected output of the original program on that input.
//! - The actual output of the mutant on that input.
//! - Metadata linking the witness back to the mutation site.
//!
//! Witnesses are the actionable artefacts of gap analysis: they can be
//! converted directly into test cases or used to strengthen the contract.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use shared_types::formula::{Formula, Predicate, Relation, Term};
use shared_types::operators::{MutantId, MutationOperator};
use shared_types::types::QfLiaType;

use crate::analyzer::SurvivingMutant;

// ---------------------------------------------------------------------------
// Distinguishing input
// ---------------------------------------------------------------------------

/// A single concrete input value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputValue {
    /// Variable name.
    pub name: String,

    /// The QF-LIA type of the variable.
    pub ty: QfLiaType,

    /// The concrete integer value.
    pub value: i64,
}

impl InputValue {
    /// Create a new input value.
    pub fn new(name: impl Into<String>, ty: QfLiaType, value: i64) -> Self {
        Self {
            name: name.into(),
            ty,
            value,
        }
    }

    /// Create an integer input.
    pub fn int(name: impl Into<String>, value: i64) -> Self {
        Self::new(name, QfLiaType::Int, value)
    }

    /// Create a boolean input (0 or 1).
    pub fn boolean(name: impl Into<String>, value: bool) -> Self {
        Self::new(name, QfLiaType::Boolean, if value { 1 } else { 0 })
    }
}

impl fmt::Display for InputValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} = {}", self.name, self.ty, self.value)
    }
}

/// A complete distinguishing input vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistinguishingInput {
    /// Unique identifier for this input.
    pub id: Uuid,

    /// Input variable assignments.
    pub values: Vec<InputValue>,

    /// Expected output of the original program.
    pub expected_output: Option<i64>,

    /// Actual output of the mutant.
    pub mutant_output: Option<i64>,

    /// The specific postcondition that is violated.
    pub violated_postcondition: Option<Formula>,

    /// How this input was generated.
    pub generation_method: InputGenerationMethod,

    /// Confidence that this input truly distinguishes the mutant.
    pub confidence: f64,
}

impl DistinguishingInput {
    /// Create a new distinguishing input.
    pub fn new(values: Vec<InputValue>, method: InputGenerationMethod) -> Self {
        Self {
            id: Uuid::new_v4(),
            values,
            expected_output: None,
            mutant_output: None,
            violated_postcondition: None,
            generation_method: method,
            confidence: 1.0,
        }
    }

    /// Attach expected/actual outputs.
    pub fn with_outputs(mut self, expected: i64, actual: i64) -> Self {
        self.expected_output = Some(expected);
        self.mutant_output = Some(actual);
        self
    }

    /// Attach the violated postcondition.
    pub fn with_violated_postcondition(mut self, formula: Formula) -> Self {
        self.violated_postcondition = Some(formula);
        self
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Returns `true` if both outputs are known and they differ.
    pub fn outputs_differ(&self) -> bool {
        match (self.expected_output, self.mutant_output) {
            (Some(e), Some(a)) => e != a,
            _ => false,
        }
    }

    /// Returns the output delta, if both are known.
    pub fn output_delta(&self) -> Option<i64> {
        match (self.expected_output, self.mutant_output) {
            (Some(e), Some(a)) => Some(a - e),
            _ => None,
        }
    }

    /// Number of input variables.
    pub fn arity(&self) -> usize {
        self.values.len()
    }

    /// Look up a specific variable's value.
    pub fn get(&self, name: &str) -> Option<i64> {
        self.values.iter().find(|v| v.name == name).map(|v| v.value)
    }

    /// Convert to a name→value map.
    pub fn as_map(&self) -> IndexMap<String, i64> {
        self.values
            .iter()
            .map(|v| (v.name.clone(), v.value))
            .collect()
    }
}

impl fmt::Display for DistinguishingInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, val) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", val.name, val.value)?;
        }
        write!(f, ")")?;
        if let (Some(e), Some(a)) = (self.expected_output, self.mutant_output) {
            write!(f, " → expected={e}, actual={a}")?;
        }
        Ok(())
    }
}

/// How a distinguishing input was generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputGenerationMethod {
    /// Extracted from an SMT model.
    SmtModel,
    /// Generated via boundary-value analysis.
    BoundaryValue,
    /// Generated via random sampling.
    Random,
    /// Derived from a constraint propagation analysis.
    ConstraintPropagation,
    /// Manually supplied.
    Manual,
}

impl fmt::Display for InputGenerationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SmtModel => write!(f, "SMT model"),
            Self::BoundaryValue => write!(f, "boundary value"),
            Self::Random => write!(f, "random"),
            Self::ConstraintPropagation => write!(f, "constraint propagation"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

// ---------------------------------------------------------------------------
// Gap witness
// ---------------------------------------------------------------------------

/// A gap witness: evidence that a surviving mutant exposes a specification gap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapWitness {
    /// Unique identifier for this witness.
    pub id: Uuid,

    /// The mutant that the witness pertains to.
    pub mutant_id: MutantId,

    /// Function containing the mutation.
    pub function_name: String,

    /// The mutation operator used.
    pub operator: MutationOperator,

    /// Original source fragment.
    pub original_fragment: String,

    /// Mutated source fragment.
    pub mutated_fragment: String,

    /// Concrete distinguishing inputs.
    pub inputs: Vec<DistinguishingInput>,

    /// The contract clauses that were checked and failed to cover.
    pub uncovered_clauses: Vec<String>,

    /// Suggested strengthening of the contract.
    pub suggested_clause: Option<Formula>,

    /// Timestamp of witness creation.
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Free-form explanation of the gap.
    pub explanation: String,
}

impl GapWitness {
    /// Create a new gap witness.
    pub fn new(
        mutant_id: MutantId,
        function_name: String,
        operator: MutationOperator,
        original_fragment: String,
        mutated_fragment: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            mutant_id,
            function_name,
            operator,
            original_fragment,
            mutated_fragment,
            inputs: Vec::new(),
            uncovered_clauses: Vec::new(),
            suggested_clause: None,
            created_at: chrono::Utc::now(),
            explanation: String::new(),
        }
    }

    /// Add a distinguishing input.
    pub fn add_input(&mut self, input: DistinguishingInput) {
        self.inputs.push(input);
    }

    /// Set the explanation.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    /// Set the suggested clause.
    pub fn with_suggested_clause(mut self, clause: Formula) -> Self {
        self.suggested_clause = Some(clause);
        self
    }

    /// Number of distinguishing inputs.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Whether any input demonstrates a concrete output difference.
    pub fn has_concrete_divergence(&self) -> bool {
        self.inputs.iter().any(|i| i.outputs_differ())
    }

    /// Average confidence across all inputs.
    pub fn average_confidence(&self) -> f64 {
        if self.inputs.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.inputs.iter().map(|i| i.confidence).sum();
        sum / self.inputs.len() as f64
    }

    /// Maximum output delta across all inputs.
    pub fn max_output_delta(&self) -> Option<i64> {
        self.inputs
            .iter()
            .filter_map(|i| i.output_delta())
            .map(|d| d.abs())
            .max()
    }
}

impl fmt::Display for GapWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Gap Witness [{}] in {}::{} ({})",
            self.id, self.function_name, self.operator, self.mutant_id,
        )?;
        writeln!(f, "  Original: {}", self.original_fragment)?;
        writeln!(f, "  Mutated:  {}", self.mutated_fragment)?;
        if !self.inputs.is_empty() {
            writeln!(f, "  Inputs ({}):", self.inputs.len())?;
            for input in &self.inputs {
                writeln!(f, "    {input}")?;
            }
        }
        if !self.explanation.is_empty() {
            writeln!(f, "  Explanation: {}", self.explanation)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Witness generator
// ---------------------------------------------------------------------------

/// Generates gap witnesses with concrete distinguishing inputs for
/// non-equivalent surviving mutants.
pub struct WitnessGenerator {
    /// Maximum number of inputs to generate per witness.
    max_inputs: usize,
}

impl WitnessGenerator {
    /// Create a new witness generator.
    pub fn new(max_inputs: usize) -> Self {
        Self { max_inputs }
    }

    /// Generate witnesses for a single surviving mutant.
    pub fn generate(&self, survivor: &SurvivingMutant) -> anyhow::Result<Vec<GapWitness>> {
        let mut witness = GapWitness::new(
            survivor.id.clone(),
            survivor.function_name.clone(),
            survivor.operator.clone(),
            survivor.original_fragment.clone(),
            survivor.mutated_fragment.clone(),
        );

        // Strategy 1: derive inputs from WP formulas.
        if let (Some(orig_wp), Some(mut_wp)) = (&survivor.original_wp, &survivor.mutant_wp) {
            let wp_inputs = self.inputs_from_wp(orig_wp, mut_wp);
            for input in wp_inputs {
                if witness.inputs.len() >= self.max_inputs {
                    break;
                }
                witness.add_input(input);
            }
        }

        // Strategy 2: boundary value inputs.
        let boundary_inputs = self.boundary_value_inputs(survivor);
        for input in boundary_inputs {
            if witness.inputs.len() >= self.max_inputs {
                break;
            }
            witness.add_input(input);
        }

        // Strategy 3: operator-specific heuristic inputs.
        let heuristic_inputs = self.operator_heuristic_inputs(survivor);
        for input in heuristic_inputs {
            if witness.inputs.len() >= self.max_inputs {
                break;
            }
            witness.add_input(input);
        }

        // Build explanation.
        witness.explanation = self.build_explanation(survivor, &witness);

        // Suggest a clause that would cover this mutant.
        witness.suggested_clause = self.suggest_clause(survivor);

        Ok(vec![witness])
    }

    /// Generate witnesses for multiple survivors.
    pub fn generate_all(
        &self,
        survivors: &[SurvivingMutant],
    ) -> Vec<(MutantId, anyhow::Result<Vec<GapWitness>>)> {
        survivors
            .iter()
            .map(|s| (s.id.clone(), self.generate(s)))
            .collect()
    }

    // -- Strategy 1: WP-derived inputs --------------------------------------

    /// Extract distinguishing inputs from the difference of WP formulas.
    ///
    /// The distinguishing condition is:
    ///   original_wp(input) ∧ ¬mutant_wp(input)
    ///
    /// We extract variable bounds from the formula and construct concrete
    /// inputs satisfying those bounds.
    fn inputs_from_wp(
        &self,
        original_wp: &Formula,
        mutant_wp: &Formula,
    ) -> Vec<DistinguishingInput> {
        // Build the distinguishing formula.
        let diff = Formula::And(vec![
            original_wp.clone(),
            Formula::Not(Box::new(mutant_wp.clone())),
        ]);

        // Extract variable bounds.
        let bounds = extract_variable_bounds(&diff);
        if bounds.is_empty() {
            return Vec::new();
        }

        // Construct an input from the mid-point of each bound.
        let values: Vec<InputValue> = bounds
            .iter()
            .map(|(name, (lo, hi))| {
                let mid = lo.saturating_add(*hi) / 2;
                InputValue::int(name.clone(), mid)
            })
            .collect();

        if values.is_empty() {
            return Vec::new();
        }

        let input =
            DistinguishingInput::new(values, InputGenerationMethod::SmtModel).with_confidence(0.9);

        vec![input]
    }

    // -- Strategy 2: boundary value inputs ----------------------------------

    /// Generate boundary-value inputs based on constants appearing in the
    /// mutation fragments.
    fn boundary_value_inputs(&self, survivor: &SurvivingMutant) -> Vec<DistinguishingInput> {
        let constants = extract_constants_from_fragments(
            &survivor.original_fragment,
            &survivor.mutated_fragment,
        );

        let variables = extract_variable_names_from_fragments(
            &survivor.original_fragment,
            &survivor.mutated_fragment,
        );

        if variables.is_empty() {
            return Vec::new();
        }

        let mut inputs = Vec::new();

        // Boundary values: 0, 1, -1, and constants ± 1.
        let mut boundary_values: Vec<i64> = vec![0, 1, -1, i64::MAX, i64::MIN];
        for c in &constants {
            boundary_values.push(*c);
            boundary_values.push(c.saturating_add(1));
            boundary_values.push(c.saturating_sub(1));
        }
        boundary_values.sort_unstable();
        boundary_values.dedup();

        // Generate inputs using boundary values.
        let vars_vec: Vec<&String> = variables.iter().collect();
        for &bv in boundary_values.iter().take(self.max_inputs) {
            let values: Vec<InputValue> = vars_vec
                .iter()
                .map(|name| InputValue::int((*name).clone(), bv))
                .collect();
            inputs.push(
                DistinguishingInput::new(values, InputGenerationMethod::BoundaryValue)
                    .with_confidence(0.5),
            );
        }

        inputs
    }

    // -- Strategy 3: operator heuristic inputs ------------------------------

    /// Generate inputs tailored to the specific mutation operator.
    fn operator_heuristic_inputs(&self, survivor: &SurvivingMutant) -> Vec<DistinguishingInput> {
        let variables = extract_variable_names_from_fragments(
            &survivor.original_fragment,
            &survivor.mutated_fragment,
        );

        if variables.is_empty() {
            return Vec::new();
        }

        let vars_vec: Vec<&String> = variables.iter().collect();

        let test_values: Vec<Vec<i64>> = match survivor.operator {
            MutationOperator::Aor => {
                // Arithmetic operator replacement: values where + ≠ - ≠ * ≠ /
                vec![vec![2, 3], vec![5, 7], vec![-1, 2], vec![0, 1]]
            }
            MutationOperator::Ror => {
                // Relational operator replacement: boundary values.
                vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![-1, 0]]
            }
            MutationOperator::Lcr => {
                // Logical connector replacement.
                vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
            }
            MutationOperator::Uoi => {
                // Unary operator insertion.
                vec![vec![0], vec![1], vec![-1], vec![42]]
            }
            MutationOperator::Abs => {
                // Absolute value insertion.
                vec![vec![-1], vec![-100], vec![0], vec![1]]
            }
            MutationOperator::Rvr => {
                // Return value replacement.
                vec![vec![0], vec![1], vec![-1]]
            }
            _ => {
                // Default: small values.
                vec![vec![0, 0], vec![1, 1], vec![-1, -1]]
            }
        };

        let mut inputs = Vec::new();

        for vals in test_values {
            let values: Vec<InputValue> = vars_vec
                .iter()
                .zip(vals.iter().cycle())
                .map(|(name, &v)| InputValue::int((*name).clone(), v))
                .collect();

            inputs.push(
                DistinguishingInput::new(values, InputGenerationMethod::ConstraintPropagation)
                    .with_confidence(0.3),
            );
        }

        inputs
    }

    // -- Explanation & clause suggestion ------------------------------------

    /// Build a human-readable explanation of the gap.
    fn build_explanation(&self, survivor: &SurvivingMutant, witness: &GapWitness) -> String {
        let operator_desc = operator_description(&survivor.operator);
        let input_summary = if witness.inputs.is_empty() {
            "no concrete distinguishing inputs could be generated".to_string()
        } else {
            let methods: Vec<String> = witness
                .inputs
                .iter()
                .map(|i| format!("{}", i.generation_method))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            format!(
                "{} distinguishing input(s) generated via {}",
                witness.inputs.len(),
                methods.join(", ")
            )
        };

        format!(
            "Mutation operator {op} changed `{orig}` to `{mut_}` in function `{func}`. \
             The current contract does not distinguish this mutant from the original. \
             {inputs}.",
            op = operator_desc,
            orig = survivor.original_fragment,
            mut_ = survivor.mutated_fragment,
            func = survivor.function_name,
            inputs = input_summary,
        )
    }

    /// Suggest a contract clause that would cover this mutant.
    fn suggest_clause(&self, survivor: &SurvivingMutant) -> Option<Formula> {
        // If WP formulas are available, suggest the difference.
        if let (Some(orig), Some(mutant)) = (&survivor.original_wp, &survivor.mutant_wp) {
            // The distinguishing clause: original ∧ ¬mutant
            let clause = Formula::And(vec![orig.clone(), Formula::Not(Box::new(mutant.clone()))]);
            return Some(clause);
        }

        // Operator-specific suggestions.
        match survivor.operator {
            MutationOperator::Ror => {
                // For relational operator replacement, suggest a boundary check.
                let vars = extract_variable_names_from_fragments(
                    &survivor.original_fragment,
                    &survivor.mutated_fragment,
                );
                let vars_vec: Vec<&String> = vars.iter().collect();
                if vars_vec.len() >= 2 {
                    let pred = Predicate::new(
                        Relation::Ne,
                        Term::Var(vars_vec[0].clone()),
                        Term::Var(vars_vec[1].clone()),
                    );
                    return Some(Formula::Atom(pred));
                }
            }
            MutationOperator::Aor => {
                // For arithmetic operator replacement, suggest non-zero check.
                let vars = extract_variable_names_from_fragments(
                    &survivor.original_fragment,
                    &survivor.mutated_fragment,
                );
                if let Some(var) = vars.iter().next() {
                    let pred = Predicate::new(Relation::Ne, Term::Var(var.clone()), Term::Const(0));
                    return Some(Formula::Atom(pred));
                }
            }
            _ => {}
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract variable bounds from a formula.
///
/// For each variable `x` referenced in the formula, attempts to find
/// lower and upper bounds from atomic predicates of the form `x rel c`.
fn extract_variable_bounds(formula: &Formula) -> HashMap<String, (i64, i64)> {
    let mut bounds: HashMap<String, (i64, i64)> = HashMap::new();

    let predicates = collect_predicates(formula);

    for pred in &predicates {
        if let (Term::Var(name), Term::Const(c)) = (&pred.left, &pred.right) {
            let entry = bounds.entry(name.clone()).or_insert((i64::MIN, i64::MAX));
            match pred.relation {
                Relation::Eq => {
                    entry.0 = *c;
                    entry.1 = *c;
                }
                Relation::Lt => {
                    entry.1 = std::cmp::min(entry.1, c - 1);
                }
                Relation::Le => {
                    entry.1 = std::cmp::min(entry.1, *c);
                }
                Relation::Gt => {
                    entry.0 = std::cmp::max(entry.0, c + 1);
                }
                Relation::Ge => {
                    entry.0 = std::cmp::max(entry.0, *c);
                }
                Relation::Ne => {
                    // Cannot derive tight bounds from ≠ alone.
                }
            }
        }

        // Symmetric case: `c rel x`.
        if let (Term::Const(c), Term::Var(name)) = (&pred.left, &pred.right) {
            let entry = bounds.entry(name.clone()).or_insert((i64::MIN, i64::MAX));
            match pred.relation {
                Relation::Eq => {
                    entry.0 = *c;
                    entry.1 = *c;
                }
                Relation::Lt => {
                    entry.0 = std::cmp::max(entry.0, c + 1);
                }
                Relation::Le => {
                    entry.0 = std::cmp::max(entry.0, *c);
                }
                Relation::Gt => {
                    entry.1 = std::cmp::min(entry.1, c - 1);
                }
                Relation::Ge => {
                    entry.1 = std::cmp::min(entry.1, *c);
                }
                Relation::Ne => {}
            }
        }
    }

    // Prune infeasible bounds.
    bounds.retain(|_, (lo, hi)| lo <= hi);

    bounds
}

/// Collect all atomic predicates from a formula (recursively).
fn collect_predicates(formula: &Formula) -> Vec<Predicate> {
    let mut predicates = Vec::new();
    collect_predicates_rec(formula, &mut predicates);
    predicates
}

fn collect_predicates_rec(formula: &Formula, out: &mut Vec<Predicate>) {
    match formula {
        Formula::Atom(p) => out.push(p.clone()),
        Formula::Not(inner) => collect_predicates_rec(inner, out),
        Formula::And(children) | Formula::Or(children) => {
            for child in children {
                collect_predicates_rec(child, out);
            }
        }
        Formula::Implies(lhs, rhs) => {
            collect_predicates_rec(lhs, out);
            collect_predicates_rec(rhs, out);
        }
        Formula::True | Formula::False => {}
        Formula::Iff(a, b) => {
            collect_predicates_rec(a, out);
            collect_predicates_rec(b, out);
        }
        Formula::Forall(_, body) | Formula::Exists(_, body) => {
            collect_predicates_rec(body, out);
        }
    }
}

/// Extract integer constants from source fragments.
fn extract_constants_from_fragments(original: &str, mutated: &str) -> Vec<i64> {
    let mut constants = Vec::new();

    let extract = |s: &str, out: &mut Vec<i64>| {
        for token in s.split(|c: char| !c.is_ascii_digit() && c != '-') {
            if let Ok(val) = token.parse::<i64>() {
                out.push(val);
            }
        }
    };

    extract(original, &mut constants);
    extract(mutated, &mut constants);

    constants.sort_unstable();
    constants.dedup();
    constants
}

/// Extract variable names from source fragments.
fn extract_variable_names_from_fragments(
    original: &str,
    mutated: &str,
) -> indexmap::IndexSet<String> {
    let mut names = indexmap::IndexSet::new();

    let extract = |s: &str, out: &mut indexmap::IndexSet<String>| {
        for token in s.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if !token.is_empty()
                && token
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_alphabetic() || c == '_')
                && !is_keyword(token)
            {
                out.insert(token.to_string());
            }
        }
    };

    extract(original, &mut names);
    extract(mutated, &mut names);

    names
}

/// Check if a token is a language keyword.
fn is_keyword(token: &str) -> bool {
    matches!(
        token,
        "if" | "else"
            | "while"
            | "for"
            | "return"
            | "true"
            | "false"
            | "int"
            | "long"
            | "boolean"
            | "void"
            | "null"
            | "new"
            | "var"
            | "let"
            | "const"
            | "fn"
            | "func"
    )
}

/// Human-readable description of a mutation operator.
fn operator_description(op: &MutationOperator) -> &'static str {
    match op {
        MutationOperator::Aor => "AOR (arithmetic operator replacement)",
        MutationOperator::Ror => "ROR (relational operator replacement)",
        MutationOperator::Lcr => "LCR (logical connector replacement)",
        MutationOperator::Uoi => "UOI (unary operator insertion)",
        MutationOperator::Abs => "ABS (absolute value insertion)",
        MutationOperator::Cor => "COR (conditional operator replacement)",
        MutationOperator::Sdl => "SDL (statement deletion)",
        MutationOperator::Rvr => "RVR (return value replacement)",
        MutationOperator::Crc => "CRC (constant replacement with constant)",
        MutationOperator::Air => "AIR (array index replacement)",
        MutationOperator::Osw => "OSW (operator-specific widening)",
        MutationOperator::Bcn => "BCN (branch condition negation)",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::formula::{Predicate, Relation, Term};

    #[test]
    fn input_value_display() {
        let iv = InputValue::int("x", 42);
        let s = format!("{iv}");
        assert!(s.contains("x"));
        assert!(s.contains("42"));
    }

    #[test]
    fn distinguishing_input_outputs_differ() {
        let di = DistinguishingInput::new(
            vec![InputValue::int("x", 5)],
            InputGenerationMethod::SmtModel,
        )
        .with_outputs(10, 20);

        assert!(di.outputs_differ());
        assert_eq!(di.output_delta(), Some(10));
    }

    #[test]
    fn distinguishing_input_no_outputs() {
        let di = DistinguishingInput::new(vec![], InputGenerationMethod::Random);
        assert!(!di.outputs_differ());
        assert_eq!(di.output_delta(), None);
    }

    #[test]
    fn distinguishing_input_as_map() {
        let di = DistinguishingInput::new(
            vec![InputValue::int("x", 1), InputValue::int("y", 2)],
            InputGenerationMethod::BoundaryValue,
        );
        let map = di.as_map();
        assert_eq!(map.get("x"), Some(&1));
        assert_eq!(map.get("y"), Some(&2));
    }

    #[test]
    fn gap_witness_average_confidence() {
        let mut w = GapWitness::new(
            MutantId::new(),
            "foo".into(),
            MutationOperator::Aor,
            "x + y".into(),
            "x - y".into(),
        );
        w.add_input(
            DistinguishingInput::new(vec![], InputGenerationMethod::SmtModel).with_confidence(0.8),
        );
        w.add_input(
            DistinguishingInput::new(vec![], InputGenerationMethod::Random).with_confidence(0.4),
        );
        assert!((w.average_confidence() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_constants() {
        let constants = extract_constants_from_fragments("x + 3", "x + 5");
        assert!(constants.contains(&3));
        assert!(constants.contains(&5));
    }

    #[test]
    fn extract_variable_names() {
        let names = extract_variable_names_from_fragments("x + y", "x - z");
        assert!(names.contains("x"));
        assert!(names.contains("y"));
        assert!(names.contains("z"));
    }

    #[test]
    fn keyword_detection() {
        assert!(is_keyword("if"));
        assert!(is_keyword("return"));
        assert!(!is_keyword("myVar"));
    }

    #[test]
    fn extract_bounds_from_formula() {
        // x >= 0 ∧ x <= 10
        let f = Formula::And(vec![
            Formula::Atom(Predicate::new(
                Relation::Ge,
                Term::Var("x".into()),
                Term::Const(0),
            )),
            Formula::Atom(Predicate::new(
                Relation::Le,
                Term::Var("x".into()),
                Term::Const(10),
            )),
        ]);
        let bounds = extract_variable_bounds(&f);
        assert_eq!(bounds.get("x"), Some(&(0, 10)));
    }

    #[test]
    fn extract_bounds_infeasible() {
        // x >= 10 ∧ x <= 5  → empty (infeasible)
        let f = Formula::And(vec![
            Formula::Atom(Predicate::new(
                Relation::Ge,
                Term::Var("x".into()),
                Term::Const(10),
            )),
            Formula::Atom(Predicate::new(
                Relation::Le,
                Term::Var("x".into()),
                Term::Const(5),
            )),
        ]);
        let bounds = extract_variable_bounds(&f);
        assert!(bounds.is_empty());
    }

    #[test]
    fn witness_display() {
        let w = GapWitness::new(
            MutantId::new(),
            "bar".into(),
            MutationOperator::Ror,
            "x < y".into(),
            "x <= y".into(),
        );
        let s = format!("{w}");
        assert!(s.contains("bar"));
        assert!(s.contains("x < y"));
    }

    #[test]
    fn input_generation_method_display() {
        assert_eq!(format!("{}", InputGenerationMethod::SmtModel), "SMT model");
        assert_eq!(
            format!("{}", InputGenerationMethod::BoundaryValue),
            "boundary value"
        );
    }
}
