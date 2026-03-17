use crate::error::ElaborationError;
use crate::ir::*;
use crate::source_map::Span;
use regsynth_types::{CompositionOp, FormalizabilityGrade, TemporalInterval};
use std::collections::HashMap;

/// Elaborator: expands compositions, resolves references, computes effective
/// temporal bounds, and propagates formalizability grades.
pub struct Elaborator {
    /// Map from obligation name to its Stage2 representation
    obligation_map: HashMap<String, Stage2Obligation>,
    /// Map from obligation name to its formalizability grade
    formalizability_map: HashMap<String, FormalizabilityGrade>,
    /// Map from obligation name to effective temporal interval
    temporal_map: HashMap<String, TemporalInterval>,
    /// Generated constraint variables
    variables: Vec<ConstraintVariable>,
    errors: Vec<ElaborationError>,
    next_var_id: u64,
}

#[derive(Debug, Clone)]
pub struct ConstraintVariable {
    pub id: String,
    pub name: String,
    pub source: String,
}

impl Elaborator {
    pub fn new() -> Self {
        Self {
            obligation_map: HashMap::new(),
            formalizability_map: HashMap::new(),
            temporal_map: HashMap::new(),
            variables: Vec::new(),
            errors: Vec::new(),
            next_var_id: 0,
        }
    }

    /// Run elaboration on a Stage2 program, producing an elaborated Stage2 program.
    pub fn elaborate(&mut self, program: &Stage2Program) -> (Stage2Program, Vec<ElaborationError>) {
        // Phase 1: Index all obligations
        for obl in &program.obligations {
            self.obligation_map.insert(obl.name.clone(), obl.clone());
            if let Some(grade) = obl.formalizability {
                self.formalizability_map.insert(obl.name.clone(), grade);
            }
            let interval = TemporalInterval::new(
                obl.temporal_start
                    .as_ref()
                    .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()),
                obl.temporal_end
                    .as_ref()
                    .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()),
            );
            self.temporal_map.insert(obl.name.clone(), interval);
        }

        // Phase 2: Resolve compositions and propagate
        let mut elaborated_obligations = Vec::new();
        for obl in &program.obligations {
            let elaborated = self.elaborate_obligation(obl);
            elaborated_obligations.push(elaborated);
        }

        let result = Stage2Program {
            obligations: elaborated_obligations,
            strategies: program.strategies.clone(),
            mappings: program.mappings.clone(),
        };

        (result, std::mem::take(&mut self.errors))
    }

    fn elaborate_obligation(&mut self, obl: &Stage2Obligation) -> Stage2Obligation {
        let mut elaborated = obl.clone();

        // Resolve cross-jurisdictional references in compositions
        let mut resolved_compositions = Vec::new();
        for comp in &obl.composed_with {
            match self.resolve_composition(obl, comp) {
                Ok(resolved) => resolved_compositions.push(resolved),
                Err(e) => self.errors.push(e),
            }
        }
        elaborated.composed_with = resolved_compositions;

        // Compute effective temporal bounds through compositions
        let effective_temporal = self.compute_effective_temporal(obl);
        if let Some(start) = effective_temporal.start {
            elaborated.temporal_start = Some(start.format("%Y-%m-%d").to_string());
        }
        if let Some(end) = effective_temporal.end {
            elaborated.temporal_end = Some(end.format("%Y-%m-%d").to_string());
        }

        // Propagate formalizability grade through compositions
        let effective_grade = self.compute_effective_formalizability(obl);
        elaborated.formalizability = Some(effective_grade);

        // Generate constraint variables
        let var = self.generate_variable(&obl.name, &obl.id);
        self.variables.push(var);

        elaborated
    }

    fn resolve_composition(
        &self,
        source: &Stage2Obligation,
        comp: &Stage2ComposedRef,
    ) -> Result<Stage2ComposedRef, ElaborationError> {
        let span: Span = source.span.clone().into();

        // Look up target obligation
        let target = self.obligation_map.get(&comp.target_name).ok_or_else(|| {
            ElaborationError::unresolved_reference(span, &comp.target_name)
        })?;

        // Validate composition compatibility
        self.validate_composition_compatibility(source, target, comp.op, span)?;

        Ok(Stage2ComposedRef {
            op: comp.op,
            target_id: target.id.clone(),
            target_name: comp.target_name.clone(),
        })
    }

    fn validate_composition_compatibility(
        &self,
        source: &Stage2Obligation,
        target: &Stage2Obligation,
        op: CompositionOp,
        span: Span,
    ) -> Result<(), ElaborationError> {
        // For override, source jurisdiction should be a child of target jurisdiction
        if op == CompositionOp::Override {
            if let (Some(sj), Some(tj)) = (&source.jurisdiction, &target.jurisdiction) {
                if sj == tj {
                    // Same jurisdiction is fine for override
                } else if !sj.starts_with(tj.as_str()) {
                    self.report_jurisdiction_warning(span, sj, tj);
                }
            }
        }

        // Check temporal overlap for conjunction/disjunction
        if matches!(op, CompositionOp::Conjunction | CompositionOp::Disjunction) {
            let src_temporal = self
                .temporal_map
                .get(&source.name)
                .cloned()
                .unwrap_or_else(TemporalInterval::unbounded);
            let tgt_temporal = self
                .temporal_map
                .get(&target.name)
                .cloned()
                .unwrap_or_else(TemporalInterval::unbounded);

            if !src_temporal.overlaps(&tgt_temporal) {
                return Err(ElaborationError::temporal_conflict(
                    span,
                    &format!(
                        "obligations '{}' and '{}' have non-overlapping temporal bounds",
                        source.name, target.name
                    ),
                ));
            }
        }

        Ok(())
    }

    fn report_jurisdiction_warning(&self, _span: Span, _source_jur: &str, _target_jur: &str) {
        // In a real implementation, this would emit a warning
        log::debug!(
            "jurisdiction override: {} overrides {}",
            _source_jur,
            _target_jur
        );
    }

    /// Compute effective temporal bounds by intersecting with composed obligations.
    fn compute_effective_temporal(&self, obl: &Stage2Obligation) -> TemporalInterval {
        let mut effective = self
            .temporal_map
            .get(&obl.name)
            .cloned()
            .unwrap_or_else(TemporalInterval::unbounded);

        for comp in &obl.composed_with {
            if let Some(target_interval) = self.temporal_map.get(&comp.target_name) {
                match comp.op {
                    CompositionOp::Conjunction => {
                        // Conjunction: intersection of intervals
                        if let Some(intersection) = effective.intersection(target_interval) {
                            effective = intersection;
                        }
                    }
                    CompositionOp::Disjunction => {
                        // Disjunction: union (use the broader interval)
                        let start = match (effective.start, target_interval.start) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            _ => None,
                        };
                        let end = match (effective.end, target_interval.end) {
                            (Some(a), Some(b)) => Some(a.max(b)),
                            _ => None,
                        };
                        effective = TemporalInterval::new(start, end);
                    }
                    CompositionOp::Override => {
                        // Override: use source's temporal bounds (already set)
                    }
                    CompositionOp::Exception => {
                        // Exception: keep source's bounds
                    }
                }
            }
        }

        effective
    }

    /// Compute effective formalizability by composing grades.
    fn compute_effective_formalizability(&self, obl: &Stage2Obligation) -> FormalizabilityGrade {
        let base = obl
            .formalizability
            .unwrap_or(FormalizabilityGrade::F1);

        let mut effective = base;
        for comp in &obl.composed_with {
            if let Some(target_grade) = self.formalizability_map.get(&comp.target_name) {
                // Compose: take the worse (higher number) grade
                effective = effective.compose(*target_grade);
            }
        }

        effective
    }

    fn generate_variable(&mut self, name: &str, source_id: &str) -> ConstraintVariable {
        self.next_var_id += 1;
        ConstraintVariable {
            id: format!("v_{}", self.next_var_id),
            name: name.to_string(),
            source: source_id.to_string(),
        }
    }

    /// Get all generated constraint variables.
    pub fn variables(&self) -> &[ConstraintVariable] {
        &self.variables
    }
}

impl Default for Elaborator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: elaborate a Stage2 program.
pub fn elaborate(program: &Stage2Program) -> (Stage2Program, Vec<ElaborationError>) {
    let mut elaborator = Elaborator::new();
    elaborator.elaborate(program)
}

/// Full elaboration from AST.
pub fn elaborate_from_ast(
    ast_program: &crate::ast::Program,
) -> (Stage2Program, Vec<ElaborationError>) {
    let s1 = crate::ir::ast_to_stage1(ast_program);
    let s2 = crate::ir::stage1_to_stage2(&s1);
    elaborate(&s2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn elaborate_src(src: &str) -> (Stage2Program, Vec<ElaborationError>) {
        let (tokens, _) = lex(src);
        let (program, _) = parse(tokens);
        elaborate_from_ast(&program)
    }

    #[test]
    fn test_simple_elaboration() {
        let (result, errors) = elaborate_src(r#"
            obligation test {
                requires: true;
                formalizability: 2;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        assert_eq!(result.obligations.len(), 1);
        assert_eq!(
            result.obligations[0].formalizability,
            Some(FormalizabilityGrade::F2)
        );
    }

    #[test]
    fn test_formalizability_propagation() {
        let (result, errors) = elaborate_src(r#"
            obligation obl_a {
                requires: true;
                formalizability: 2;
            }
            obligation obl_b {
                requires: true;
                formalizability: 4;
                compose: ⊗ obl_a;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        // obl_b should have effective formalizability F4 (max of F4, F2)
        let obl_b = result
            .obligations
            .iter()
            .find(|o| o.name == "obl_b")
            .unwrap();
        assert_eq!(obl_b.formalizability, Some(FormalizabilityGrade::F4));
    }

    #[test]
    fn test_temporal_intersection() {
        let (result, errors) = elaborate_src(r#"
            obligation obl_a {
                requires: true;
                temporal: #2024-01-01 -> #2024-12-31;
            }
            obligation obl_b {
                requires: true;
                temporal: #2024-06-01 -> #2025-06-30;
                compose: ⊗ obl_a;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let obl_b = result
            .obligations
            .iter()
            .find(|o| o.name == "obl_b")
            .unwrap();
        // Conjunction should yield intersection: 2024-06-01 to 2024-12-31
        assert_eq!(obl_b.temporal_start.as_deref(), Some("2024-06-01"));
        assert_eq!(obl_b.temporal_end.as_deref(), Some("2024-12-31"));
    }

    #[test]
    fn test_non_overlapping_temporal_error() {
        let (_, errors) = elaborate_src(r#"
            obligation obl_a {
                requires: true;
                temporal: #2020-01-01 -> #2020-12-31;
            }
            obligation obl_b {
                requires: true;
                temporal: #2024-01-01 -> #2024-12-31;
                compose: ⊗ obl_a;
            }
        "#);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("non-overlapping"));
    }

    #[test]
    fn test_unresolved_composition_ref() {
        let (_, errors) = elaborate_src(r#"
            obligation test {
                requires: true;
                compose: ⊗ nonexistent;
            }
        "#);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("unresolved"));
    }

    #[test]
    fn test_disjunction_temporal_union() {
        let (result, errors) = elaborate_src(r#"
            obligation obl_a {
                requires: true;
                temporal: #2024-01-01 -> #2024-06-30;
            }
            obligation obl_b {
                requires: true;
                temporal: #2024-03-01 -> #2024-12-31;
                compose: ⊕ obl_a;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let obl_b = result.obligations.iter().find(|o| o.name == "obl_b").unwrap();
        // Disjunction yields union: 2024-01-01 to 2024-12-31
        assert_eq!(obl_b.temporal_start.as_deref(), Some("2024-01-01"));
        assert_eq!(obl_b.temporal_end.as_deref(), Some("2024-12-31"));
    }
}
