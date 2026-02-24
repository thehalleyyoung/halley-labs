//! Memory model DSL parser and compiler.
//!
//! Provides a DSL for defining memory models with relation expressions,
//! constraint reordering for early termination, model validation,
//! a standard model library, and model diff computation.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{BitMatrix, ExecutionGraph};
use crate::checker::memory_model::{
    MemoryModel, RelationExpr, PredicateExpr, Constraint,
    BuiltinModel, DerivedRelation, RelationDef,
};

// ---------------------------------------------------------------------------
// ModelParser — parse model DSL
// ---------------------------------------------------------------------------

/// Parse memory model definitions from the DSL.
pub struct ModelParser;

impl ModelParser {
    /// Parse a model definition string.
    pub fn parse(input: &str) -> Result<MemoryModel, ModelParseError> {
        let mut model = MemoryModel::new("unnamed");
        let lines: Vec<&str> = input.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .collect();

        for line in &lines {
            if let Some(name) = line.strip_prefix("model ") {
                model.name = name.trim_matches('"').to_string();
            } else if let Some(rest) = line.strip_prefix("let ") {
                Self::parse_derived(&mut model, rest)?;
            } else if let Some(rest) = line.strip_prefix("acyclic ") {
                let expr = Self::parse_relation_expr(rest.trim())?;
                model.add_acyclic(expr);
            } else if let Some(rest) = line.strip_prefix("irreflexive ") {
                let expr = Self::parse_relation_expr(rest.trim())?;
                model.add_irreflexive(expr);
            } else if let Some(rest) = line.strip_prefix("empty ") {
                let expr = Self::parse_relation_expr(rest.trim())?;
                model.add_empty(expr);
            }
        }

        Ok(model)
    }

    fn parse_derived(model: &mut MemoryModel, input: &str) -> Result<(), ModelParseError> {
        let parts: Vec<&str> = input.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(ModelParseError::SyntaxError(
                format!("Expected 'name = expr', got: {}", input)
            ));
        }
        let name = parts[0].trim();
        let expr = Self::parse_relation_expr(parts[1].trim())?;
        model.add_derived(name, expr, "");
        Ok(())
    }

    /// Parse a simple relation expression.
    pub fn parse_relation_expr(input: &str) -> Result<RelationExpr, ModelParseError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ModelParseError::SyntaxError("empty expression".to_string()));
        }

        // Handle union (|)
        if let Some(pos) = Self::find_top_level_op(input, '|') {
            let left = Self::parse_relation_expr(&input[..pos])?;
            let right = Self::parse_relation_expr(&input[pos + 1..])?;
            return Ok(RelationExpr::union(left, right));
        }

        // Handle intersection (&)
        if let Some(pos) = Self::find_top_level_op(input, '&') {
            let left = Self::parse_relation_expr(&input[..pos])?;
            let right = Self::parse_relation_expr(&input[pos + 1..])?;
            return Ok(RelationExpr::inter(left, right));
        }

        // Handle sequence (;)
        if let Some(pos) = Self::find_top_level_op(input, ';') {
            let left = Self::parse_relation_expr(&input[..pos])?;
            let right = Self::parse_relation_expr(&input[pos + 1..])?;
            return Ok(RelationExpr::seq(left, right));
        }

        // Handle suffixes
        if input.ends_with('+') {
            let inner = Self::parse_relation_expr(&input[..input.len() - 1])?;
            return Ok(RelationExpr::plus(inner));
        }
        if input.ends_with('*') {
            let inner = Self::parse_relation_expr(&input[..input.len() - 1])?;
            return Ok(RelationExpr::star(inner));
        }
        if input.ends_with('?') {
            let inner = Self::parse_relation_expr(&input[..input.len() - 1])?;
            return Ok(RelationExpr::optional(inner));
        }
        if input.ends_with("^-1") {
            let inner = Self::parse_relation_expr(&input[..input.len() - 3])?;
            return Ok(RelationExpr::inverse(inner));
        }

        // Handle parentheses
        if input.starts_with('(') && input.ends_with(')') {
            return Self::parse_relation_expr(&input[1..input.len() - 1]);
        }

        // Handle filters
        if input.starts_with('[') && input.ends_with(']') {
            let pred_str = &input[1..input.len() - 1];
            let pred = Self::parse_predicate(pred_str)?;
            return Ok(RelationExpr::filter(pred));
        }

        // Base case: identifiers
        match input {
            "id" => Ok(RelationExpr::Identity),
            "0" => Ok(RelationExpr::Empty),
            _ => Ok(RelationExpr::base(input)),
        }
    }

    fn parse_predicate(input: &str) -> Result<PredicateExpr, ModelParseError> {
        let input = input.trim();
        match input {
            "R" | "Read" => Ok(PredicateExpr::IsRead),
            "W" | "Write" => Ok(PredicateExpr::IsWrite),
            "F" | "Fence" => Ok(PredicateExpr::IsFence),
            "RMW" => Ok(PredicateExpr::IsRMW),
            "true" => Ok(PredicateExpr::True),
            _ => Ok(PredicateExpr::True),
        }
    }

    fn find_top_level_op(input: &str, op: char) -> Option<usize> {
        let mut depth = 0;
        for (i, c) in input.char_indices() {
            match c {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                c if c == op && depth == 0 => return Some(i),
                _ => {}
            }
        }
        None
    }
}

/// Model parse error.
#[derive(Debug, Clone)]
pub enum ModelParseError {
    SyntaxError(String),
    UnknownRelation(String),
    ValidationError(Vec<String>),
}

impl fmt::Display for ModelParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SyntaxError(s) => write!(f, "Syntax error: {}", s),
            Self::UnknownRelation(s) => write!(f, "Unknown relation: {}", s),
            Self::ValidationError(errs) => {
                write!(f, "Validation errors: {}", errs.join(", "))
            }
        }
    }
}

impl std::error::Error for ModelParseError {}

// ---------------------------------------------------------------------------
// Relation expression optimization
// ---------------------------------------------------------------------------

/// Optimize a relation expression for efficient evaluation.
pub fn optimize_expr(expr: &RelationExpr) -> RelationExpr {
    match expr {
        // Union with empty.
        RelationExpr::Union(a, b) => {
            let oa = optimize_expr(a);
            let ob = optimize_expr(b);
            match (&oa, &ob) {
                (RelationExpr::Empty, _) => ob,
                (_, RelationExpr::Empty) => oa,
                _ if oa == ob => oa,
                _ => RelationExpr::union(oa, ob),
            }
        }
        RelationExpr::Inter(a, b) => {
            let oa = optimize_expr(a);
            let ob = optimize_expr(b);
            match (&oa, &ob) {
                (RelationExpr::Empty, _) | (_, RelationExpr::Empty) => RelationExpr::Empty,
                _ if oa == ob => oa,
                _ => RelationExpr::inter(oa, ob),
            }
        }
        RelationExpr::Seq(a, b) => {
            let oa = optimize_expr(a);
            let ob = optimize_expr(b);
            match (&oa, &ob) {
                (RelationExpr::Identity, _) => ob,
                (_, RelationExpr::Identity) => oa,
                (RelationExpr::Empty, _) | (_, RelationExpr::Empty) => RelationExpr::Empty,
                _ => RelationExpr::seq(oa, ob),
            }
        }
        RelationExpr::Diff(a, b) => {
            let oa = optimize_expr(a);
            let ob = optimize_expr(b);
            match (&oa, &ob) {
                (RelationExpr::Empty, _) => RelationExpr::Empty,
                (_, RelationExpr::Empty) => oa,
                _ => RelationExpr::diff(oa, ob),
            }
        }
        RelationExpr::Inverse(a) => {
            let oa = optimize_expr(a);
            match oa {
                RelationExpr::Inverse(inner) => *inner,
                _ => RelationExpr::inverse(oa),
            }
        }
        RelationExpr::Plus(a) => {
            let oa = optimize_expr(a);
            match &oa {
                RelationExpr::Empty => RelationExpr::Empty,
                _ => RelationExpr::plus(oa),
            }
        }
        RelationExpr::Star(a) => {
            let oa = optimize_expr(a);
            match &oa {
                RelationExpr::Empty => RelationExpr::Identity,
                _ => RelationExpr::star(oa),
            }
        }
        RelationExpr::Optional(a) => {
            let oa = optimize_expr(a);
            RelationExpr::optional(oa)
        }
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Constraint reordering for early termination
// ---------------------------------------------------------------------------

/// Reorder constraints for early termination.
/// Cheaper constraints (fewer relation references) are checked first.
pub fn reorder_constraints(constraints: &[Constraint]) -> Vec<Constraint> {
    let mut scored: Vec<(usize, &Constraint)> = constraints.iter()
        .enumerate()
        .map(|(i, c)| {
            let complexity = c.expr().referenced_bases().len();
            (complexity, c)
        })
        .collect();
    scored.sort_by_key(|&(complexity, _)| complexity);
    scored.into_iter().map(|(_, c)| c.clone()).collect()
}

// ---------------------------------------------------------------------------
// Model validation
// ---------------------------------------------------------------------------

/// Validate a memory model for well-formedness.
pub fn validate_model(model: &MemoryModel) -> Result<(), Vec<String>> {
    model.validate()
}

/// Check if a model is well-formed (all referenced relations exist).
pub fn check_well_formedness(model: &MemoryModel) -> Vec<String> {
    match model.validate() {
        Ok(()) => Vec::new(),
        Err(errors) => errors,
    }
}

// ---------------------------------------------------------------------------
// Standard model library
// ---------------------------------------------------------------------------

/// Get a standard model by name.
pub fn standard_model(name: &str) -> Option<MemoryModel> {
    match name.to_uppercase().as_str() {
        "SC" => Some(BuiltinModel::SC.build()),
        "TSO" => Some(BuiltinModel::TSO.build()),
        "PSO" => Some(BuiltinModel::PSO.build()),
        "ARM" | "ARMV8" => Some(BuiltinModel::ARM.build()),
        "RISCV" | "RISC-V" => Some(BuiltinModel::RISCV.build()),
        "PTX" => Some(BuiltinModel::PTX.build()),
        "WEBGPU" => Some(BuiltinModel::WebGPU.build()),
        "VULKAN" => Some(build_vulkan()),
        _ => None,
    }
}

/// List all standard model names.
pub fn standard_model_names() -> Vec<&'static str> {
    vec!["SC", "TSO", "PSO", "ARM", "RISC-V", "PTX", "WebGPU", "Vulkan"]
}

/// Build a Vulkan memory model.
fn build_vulkan() -> MemoryModel {
    let mut m = MemoryModel::new("Vulkan");
    m.add_derived("com",
        RelationExpr::union_many(vec![
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
            RelationExpr::base("fr"),
        ]),
        "communication",
    );
    // Vulkan is similar to WebGPU: acyclic(po ∪ com).
    m.add_acyclic(
        RelationExpr::union(RelationExpr::base("po"), RelationExpr::base("com")),
    );
    m
}

// ---------------------------------------------------------------------------
// Model diff — what behaviors differ between two models
// ---------------------------------------------------------------------------

/// Result of diffing two memory models.
#[derive(Debug, Clone)]
pub struct ModelDiff {
    pub model_a: String,
    pub model_b: String,
    pub only_in_a: Vec<String>,
    pub only_in_b: Vec<String>,
    pub common_constraints: Vec<String>,
    pub constraint_diff: Vec<ConstraintDiff>,
}

/// Difference in a single constraint.
#[derive(Debug, Clone)]
pub struct ConstraintDiff {
    pub constraint_a: Option<String>,
    pub constraint_b: Option<String>,
    pub description: String,
}

impl ModelDiff {
    /// Compare two memory models.
    pub fn diff(a: &MemoryModel, b: &MemoryModel) -> Self {
        let a_rels: Vec<String> = a.derived_relations.iter()
            .map(|d| d.name.clone()).collect();
        let b_rels: Vec<String> = b.derived_relations.iter()
            .map(|d| d.name.clone()).collect();

        let only_in_a: Vec<String> = a_rels.iter()
            .filter(|r| !b_rels.contains(r))
            .cloned().collect();
        let only_in_b: Vec<String> = b_rels.iter()
            .filter(|r| !a_rels.contains(r))
            .cloned().collect();
        let common: Vec<String> = a_rels.iter()
            .filter(|r| b_rels.contains(r))
            .cloned().collect();

        let mut constraint_diff = Vec::new();
        let a_constraints: Vec<String> = a.constraints.iter()
            .map(|c| format!("{}", c)).collect();
        let b_constraints: Vec<String> = b.constraints.iter()
            .map(|c| format!("{}", c)).collect();

        for ac in &a_constraints {
            if !b_constraints.contains(ac) {
                constraint_diff.push(ConstraintDiff {
                    constraint_a: Some(ac.clone()),
                    constraint_b: None,
                    description: format!("Only in {}", a.name),
                });
            }
        }
        for bc in &b_constraints {
            if !a_constraints.contains(bc) {
                constraint_diff.push(ConstraintDiff {
                    constraint_a: None,
                    constraint_b: Some(bc.clone()),
                    description: format!("Only in {}", b.name),
                });
            }
        }

        Self {
            model_a: a.name.clone(),
            model_b: b.name.clone(),
            only_in_a,
            only_in_b,
            common_constraints: common,
            constraint_diff,
        }
    }
}

impl fmt::Display for ModelDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model diff: {} vs {}", self.model_a, self.model_b)?;
        if !self.only_in_a.is_empty() {
            writeln!(f, "  Only in {}: {:?}", self.model_a, self.only_in_a)?;
        }
        if !self.only_in_b.is_empty() {
            writeln!(f, "  Only in {}: {:?}", self.model_b, self.only_in_b)?;
        }
        for cd in &self.constraint_diff {
            writeln!(f, "  {}", cd.description)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OutputFormatter
// ---------------------------------------------------------------------------

/// Format verification output.
pub struct OutputFormatter;

impl OutputFormatter {
    /// Format a memory model as a string.
    pub fn format_model(model: &MemoryModel) -> String {
        format!("{}", model)
    }

    /// Format a constraint list.
    pub fn format_constraints(constraints: &[Constraint]) -> String {
        constraints.iter()
            .map(|c| format!("  {}", c))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_model() {
        let input = r#"
            model "TestSC"
            acyclic po | rf | co | fr
        "#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.name, "TestSC");
        assert_eq!(model.constraints.len(), 1);
    }

    #[test]
    fn test_parse_with_derived() {
        let input = r#"
            model "TestTSO"
            let com = rf | co | fr
            acyclic po | com
        "#;
        let model = ModelParser::parse(input).unwrap();
        assert_eq!(model.name, "TestTSO");
        assert_eq!(model.derived_relations.len(), 1);
        assert_eq!(model.constraints.len(), 1);
    }

    #[test]
    fn test_parse_relation_expr_base() {
        let expr = ModelParser::parse_relation_expr("po").unwrap();
        assert_eq!(expr, RelationExpr::base("po"));
    }

    #[test]
    fn test_parse_relation_expr_union() {
        let expr = ModelParser::parse_relation_expr("po | rf").unwrap();
        match expr {
            RelationExpr::Union(_, _) => {}
            _ => panic!("Expected union"),
        }
    }

    #[test]
    fn test_parse_relation_expr_sequence() {
        let expr = ModelParser::parse_relation_expr("po ; rf").unwrap();
        match expr {
            RelationExpr::Seq(_, _) => {}
            _ => panic!("Expected sequence"),
        }
    }

    #[test]
    fn test_parse_relation_expr_closure() {
        let expr = ModelParser::parse_relation_expr("po+").unwrap();
        match expr {
            RelationExpr::Plus(_) => {}
            _ => panic!("Expected plus"),
        }
    }

    #[test]
    fn test_parse_relation_expr_star() {
        let expr = ModelParser::parse_relation_expr("po*").unwrap();
        match expr {
            RelationExpr::Star(_) => {}
            _ => panic!("Expected star"),
        }
    }

    #[test]
    fn test_parse_relation_expr_inverse() {
        let expr = ModelParser::parse_relation_expr("rf^-1").unwrap();
        match expr {
            RelationExpr::Inverse(_) => {}
            _ => panic!("Expected inverse"),
        }
    }

    #[test]
    fn test_parse_relation_expr_identity() {
        let expr = ModelParser::parse_relation_expr("id").unwrap();
        assert_eq!(expr, RelationExpr::Identity);
    }

    #[test]
    fn test_parse_relation_expr_empty() {
        let expr = ModelParser::parse_relation_expr("0").unwrap();
        assert_eq!(expr, RelationExpr::Empty);
    }

    #[test]
    fn test_parse_filter() {
        let expr = ModelParser::parse_relation_expr("[R]").unwrap();
        match expr {
            RelationExpr::Filter(PredicateExpr::IsRead) => {}
            _ => panic!("Expected read filter"),
        }
    }

    #[test]
    fn test_parse_error_empty() {
        let result = ModelParser::parse_relation_expr("");
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize_union_empty() {
        let expr = RelationExpr::union(RelationExpr::base("po"), RelationExpr::Empty);
        let opt = optimize_expr(&expr);
        assert_eq!(opt, RelationExpr::base("po"));
    }

    #[test]
    fn test_optimize_seq_identity() {
        let expr = RelationExpr::seq(RelationExpr::Identity, RelationExpr::base("rf"));
        let opt = optimize_expr(&expr);
        assert_eq!(opt, RelationExpr::base("rf"));
    }

    #[test]
    fn test_optimize_double_inverse() {
        let expr = RelationExpr::inverse(RelationExpr::inverse(RelationExpr::base("po")));
        let opt = optimize_expr(&expr);
        assert_eq!(opt, RelationExpr::base("po"));
    }

    #[test]
    fn test_reorder_constraints() {
        let c1 = Constraint::acyclic(RelationExpr::union(
            RelationExpr::base("po"),
            RelationExpr::union(RelationExpr::base("rf"), RelationExpr::base("co")),
        ));
        let c2 = Constraint::irreflexive(RelationExpr::base("fr"));
        let reordered = reorder_constraints(&[c1, c2]);
        // Simpler constraint (fewer refs) should come first.
        assert_eq!(reordered.len(), 2);
    }

    #[test]
    fn test_standard_model_sc() {
        let model = standard_model("SC").unwrap();
        assert_eq!(model.name, "SC");
    }

    #[test]
    fn test_standard_model_tso() {
        let model = standard_model("TSO").unwrap();
        assert_eq!(model.name, "TSO");
    }

    #[test]
    fn test_standard_model_ptx() {
        let model = standard_model("PTX").unwrap();
        assert_eq!(model.name, "PTX");
    }

    #[test]
    fn test_standard_model_vulkan() {
        let model = standard_model("Vulkan").unwrap();
        assert_eq!(model.name, "Vulkan");
    }

    #[test]
    fn test_standard_model_unknown() {
        assert!(standard_model("UNKNOWN").is_none());
    }

    #[test]
    fn test_standard_model_names() {
        let names = standard_model_names();
        assert!(names.contains(&"SC"));
        assert!(names.contains(&"TSO"));
        assert!(names.contains(&"Vulkan"));
    }

    #[test]
    fn test_model_diff() {
        let sc = BuiltinModel::SC.build();
        let tso = BuiltinModel::TSO.build();
        let diff = ModelDiff::diff(&sc, &tso);
        assert_eq!(diff.model_a, "SC");
        assert_eq!(diff.model_b, "TSO");
    }

    #[test]
    fn test_model_diff_display() {
        let sc = BuiltinModel::SC.build();
        let tso = BuiltinModel::TSO.build();
        let diff = ModelDiff::diff(&sc, &tso);
        let s = format!("{}", diff);
        assert!(s.contains("Model diff"));
    }

    #[test]
    fn test_validate_model() {
        let sc = BuiltinModel::SC.build();
        assert!(validate_model(&sc).is_ok());
    }

    #[test]
    fn test_check_well_formedness() {
        let sc = BuiltinModel::SC.build();
        let errors = check_well_formedness(&sc);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_output_formatter() {
        let sc = BuiltinModel::SC.build();
        let s = OutputFormatter::format_model(&sc);
        assert!(s.contains("SC"));
    }

    #[test]
    fn test_model_parse_error_display() {
        let e = ModelParseError::SyntaxError("test error".to_string());
        let s = format!("{}", e);
        assert!(s.contains("test error"));
    }
}
