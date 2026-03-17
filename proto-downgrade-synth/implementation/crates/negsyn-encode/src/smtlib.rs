//! SMT-LIB2 output generation.
//!
//! Generates standard SMT-LIB2 format output including sort/function
//! declarations, assertions, check-sat/get-model commands, and
//! incremental push/pop for CEGAR.

use crate::{SmtConstraint, SmtDeclaration, SmtExpr, SmtFormula, SmtSort};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Write;

// ─── Writer configuration ───────────────────────────────────────────────

/// Configuration for SMT-LIB2 output generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriterConfig {
    pub logic: String,
    pub produce_models: bool,
    pub produce_unsat_cores: bool,
    pub incremental: bool,
    pub indent_width: usize,
    pub max_line_width: usize,
    pub emit_comments: bool,
    pub emit_statistics: bool,
    pub named_assertions: bool,
    pub timeout_ms: Option<u64>,
}

impl Default for WriterConfig {
    fn default() -> Self {
        WriterConfig {
            logic: "QF_AUFBV".to_string(),
            produce_models: true,
            produce_unsat_cores: false,
            incremental: false,
            indent_width: 2,
            max_line_width: 120,
            emit_comments: true,
            emit_statistics: false,
            named_assertions: false,
            timeout_ms: None,
        }
    }
}

// ─── Pretty printer ─────────────────────────────────────────────────────

/// Indentation-aware pretty printer for S-expressions.
#[derive(Debug, Clone)]
struct PrettyPrinter {
    indent: usize,
    indent_width: usize,
    max_width: usize,
}

impl PrettyPrinter {
    fn new(indent_width: usize, max_width: usize) -> Self {
        PrettyPrinter {
            indent: 0,
            indent_width,
            max_width,
        }
    }

    fn indent_str(&self) -> String {
        " ".repeat(self.indent * self.indent_width)
    }

    fn push(&mut self) {
        self.indent += 1;
    }

    fn pop(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }

    /// Format an SMT expression with pretty-printing.
    fn format_expr(&self, expr: &SmtExpr) -> String {
        let simple = format!("{}", expr);
        if simple.len() <= self.max_width {
            return simple;
        }
        self.format_expr_multiline(expr, self.indent)
    }

    fn format_expr_multiline(&self, expr: &SmtExpr, depth: usize) -> String {
        let indent = " ".repeat(depth * self.indent_width);
        let inner_indent = " ".repeat((depth + 1) * self.indent_width);

        match expr {
            SmtExpr::And(es) if es.len() > 2 => {
                let mut parts = vec![format!("{}(and", indent)];
                for e in es {
                    let s = self.format_expr_multiline(e, depth + 1);
                    parts.push(format!("{}{}", inner_indent, s.trim_start()));
                }
                parts.push(format!("{})", indent));
                parts.join("\n")
            }
            SmtExpr::Or(es) if es.len() > 2 => {
                let mut parts = vec![format!("{}(or", indent)];
                for e in es {
                    let s = self.format_expr_multiline(e, depth + 1);
                    parts.push(format!("{}{}", inner_indent, s.trim_start()));
                }
                parts.push(format!("{})", indent));
                parts.join("\n")
            }
            SmtExpr::Ite(c, t, e) => {
                let cs = self.format_expr_multiline(c, depth + 1);
                let ts = self.format_expr_multiline(t, depth + 1);
                let es = self.format_expr_multiline(e, depth + 1);
                format!(
                    "{}(ite\n{}{}\n{}{}\n{}{})",
                    indent,
                    inner_indent,
                    cs.trim_start(),
                    inner_indent,
                    ts.trim_start(),
                    inner_indent,
                    es.trim_start()
                )
            }
            SmtExpr::Let(bindings, body) => {
                let mut parts = vec![format!("{}(let (", indent)];
                for (name, val) in bindings {
                    let vs = self.format_expr_multiline(val, depth + 2);
                    parts.push(format!("{}({} {})", inner_indent, name, vs.trim_start()));
                }
                let bs = self.format_expr_multiline(body, depth + 1);
                parts.push(format!("{})", inner_indent));
                parts.push(format!("{}{})", inner_indent, bs.trim_start()));
                parts.join("\n")
            }
            SmtExpr::ForAll(vars, body) | SmtExpr::Exists(vars, body) => {
                let quantifier = match expr {
                    SmtExpr::ForAll(..) => "forall",
                    _ => "exists",
                };
                let var_list: String = vars
                    .iter()
                    .map(|(n, s)| format!("({} {})", n, s))
                    .collect::<Vec<_>>()
                    .join(" ");
                let bs = self.format_expr_multiline(body, depth + 1);
                format!(
                    "{}({} ({})\n{}{})",
                    indent,
                    quantifier,
                    var_list,
                    inner_indent,
                    bs.trim_start()
                )
            }
            _ => format!("{}{}", indent, expr),
        }
    }
}

// ─── SmtLib2Writer ──────────────────────────────────────────────────────

/// Generates SMT-LIB2 format output from an SmtFormula.
#[derive(Debug, Clone)]
pub struct SmtLib2Writer {
    config: WriterConfig,
    printer: PrettyPrinter,
}

impl SmtLib2Writer {
    pub fn new(config: WriterConfig) -> Self {
        let printer = PrettyPrinter::new(config.indent_width, config.max_line_width);
        SmtLib2Writer { config, printer }
    }

    pub fn with_defaults() -> Self {
        Self::new(WriterConfig::default())
    }

    /// Write the complete SMT-LIB2 output for a formula.
    pub fn write_formula(&self, formula: &SmtFormula) -> String {
        let mut output = String::with_capacity(formula.total_nodes() * 20);

        // Header
        self.write_header(&mut output, formula);

        // Logic declaration
        self.write_set_logic(&mut output);

        // Options
        self.write_options(&mut output);

        // Sort declarations
        self.write_declarations(&mut output, &formula.declarations);

        // Assertions
        self.write_assertions(&mut output, &formula.constraints);

        // Check-sat and get-model
        self.write_check_sat(&mut output);

        // Footer
        self.write_footer(&mut output);

        output
    }

    /// Write to a writer stream.
    pub fn write_to<W: Write>(&self, formula: &SmtFormula, writer: &mut W) -> std::io::Result<()> {
        let output = self.write_formula(formula);
        writer.write_all(output.as_bytes())
    }

    fn write_header(&self, output: &mut String, formula: &SmtFormula) {
        if self.config.emit_comments {
            output.push_str("; ═══════════════════════════════════════════════════════════\n");
            output.push_str("; NegSynth DYENCODE SMT-LIB2 output\n");
            output.push_str(&format!(
                "; Library: {} v{}\n",
                if formula.library_name.is_empty() {
                    "negsyn-encode"
                } else {
                    &formula.library_name
                },
                if formula.library_version.is_empty() {
                    "0.1.0"
                } else {
                    &formula.library_version
                }
            ));
            output.push_str(&format!("; Depth bound: {}\n", formula.depth_bound));
            output.push_str(&format!("; Adversary budget: {}\n", formula.adversary_budget));
            output.push_str(&format!("; Constraints: {}\n", formula.constraint_count()));
            output.push_str(&format!("; Declarations: {}\n", formula.declaration_count()));
            output.push_str(&format!("; Total nodes: {}\n", formula.total_nodes()));
            output.push_str("; ═══════════════════════════════════════════════════════════\n\n");
        }
    }

    fn write_set_logic(&self, output: &mut String) {
        output.push_str(&format!("(set-logic {})\n", self.config.logic));
    }

    fn write_options(&self, output: &mut String) {
        if self.config.produce_models {
            output.push_str("(set-option :produce-models true)\n");
        }
        if self.config.produce_unsat_cores {
            output.push_str("(set-option :produce-unsat-cores true)\n");
        }
        if let Some(timeout) = self.config.timeout_ms {
            output.push_str(&format!("(set-option :timeout {})\n", timeout));
        }
        output.push('\n');
    }

    fn write_declarations(&self, output: &mut String, declarations: &[SmtDeclaration]) {
        if self.config.emit_comments && !declarations.is_empty() {
            output.push_str("; ─── Declarations ─────────────────────────────────────────\n");
        }

        for decl in declarations {
            match decl {
                SmtDeclaration::DeclareSort { name, arity } => {
                    output.push_str(&format!("(declare-sort {} {})\n", name, arity));
                }
                SmtDeclaration::DeclareFun { name, args, ret } => {
                    output.push_str(&format!("(declare-fun {} (", name));
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            output.push(' ');
                        }
                        output.push_str(&format!("{}", arg));
                    }
                    output.push_str(&format!(") {})\n", ret));
                }
                SmtDeclaration::DefineFun { name, args, ret, body } => {
                    output.push_str(&format!("(define-fun {} (", name));
                    for (i, (n, s)) in args.iter().enumerate() {
                        if i > 0 {
                            output.push(' ');
                        }
                        output.push_str(&format!("({} {})", n, s));
                    }
                    let body_str = self.printer.format_expr(body);
                    output.push_str(&format!(") {} {})\n", ret, body_str));
                }
                SmtDeclaration::DeclareConst { name, sort } => {
                    output.push_str(&format!("(declare-const {} {})\n", name, sort));
                }
            }
        }
        output.push('\n');
    }

    fn write_assertions(&self, output: &mut String, constraints: &[SmtConstraint]) {
        if self.config.emit_comments && !constraints.is_empty() {
            output.push_str("; ─── Assertions ───────────────────────────────────────────\n");
        }

        // Group constraints by origin for readability
        let mut current_origin: Option<String> = None;

        for constraint in constraints {
            let origin_label = self.origin_section_label(&constraint.origin);
            if self.config.emit_comments {
                if current_origin.as_ref() != Some(&origin_label) {
                    output.push_str(&format!("\n; --- {} ---\n", origin_label));
                    current_origin = Some(origin_label);
                }
                output.push_str(&format!("; {}\n", constraint.label));
            }

            if self.config.named_assertions && self.config.produce_unsat_cores {
                let sanitized = constraint.label.replace(' ', "_").replace('/', "_");
                output.push_str(&format!(
                    "(assert (! {} :named {}))\n",
                    self.printer.format_expr(&constraint.formula),
                    sanitized
                ));
            } else {
                output.push_str(&format!(
                    "(assert {})\n",
                    self.printer.format_expr(&constraint.formula)
                ));
            }
        }
        output.push('\n');
    }

    fn write_check_sat(&self, output: &mut String) {
        if self.config.emit_comments {
            output.push_str("; ─── Check satisfiability ─────────────────────────────────\n");
        }
        output.push_str("(check-sat)\n");
        if self.config.produce_models {
            output.push_str("(get-model)\n");
        }
        if self.config.produce_unsat_cores {
            output.push_str("(get-unsat-core)\n");
        }
    }

    fn write_footer(&self, output: &mut String) {
        output.push_str("(exit)\n");
    }

    fn origin_section_label(&self, origin: &crate::ConstraintOrigin) -> String {
        match origin {
            crate::ConstraintOrigin::InitialState => "Initial State".to_string(),
            crate::ConstraintOrigin::Transition { step, .. } => {
                format!("Transitions (step {})", step)
            }
            crate::ConstraintOrigin::AdversaryAction { step, .. } => {
                format!("Adversary Actions (step {})", step)
            }
            crate::ConstraintOrigin::KnowledgeAccumulation { step } => {
                format!("Knowledge (step {})", step)
            }
            crate::ConstraintOrigin::PropertyNegation => "Property Negation".to_string(),
            crate::ConstraintOrigin::FrameCondition { step } => {
                format!("Frame Conditions (step {})", step)
            }
            crate::ConstraintOrigin::BudgetBound => "Budget Bounds".to_string(),
            crate::ConstraintOrigin::DepthBound => "Depth Bounds".to_string(),
            crate::ConstraintOrigin::SymmetryBreaking => "Symmetry Breaking".to_string(),
        }
    }

    /// Write incremental push command.
    pub fn write_push(&self) -> String {
        "(push 1)\n".to_string()
    }

    /// Write incremental pop command.
    pub fn write_pop(&self) -> String {
        "(pop 1)\n".to_string()
    }

    /// Write push/assert/check-sat/pop for incremental solving.
    pub fn write_incremental_check(
        &self,
        additional_constraints: &[SmtConstraint],
    ) -> String {
        let mut output = String::new();
        output.push_str("(push 1)\n");

        for c in additional_constraints {
            if self.config.emit_comments {
                output.push_str(&format!("; {}\n", c.label));
            }
            output.push_str(&format!("(assert {})\n", self.printer.format_expr(&c.formula)));
        }

        output.push_str("(check-sat)\n");
        if self.config.produce_models {
            output.push_str("(get-model)\n");
        }
        output.push_str("(pop 1)\n");
        output
    }

    /// Write a single assertion.
    pub fn write_assert(&self, expr: &SmtExpr) -> String {
        format!("(assert {})\n", self.printer.format_expr(expr))
    }

    /// Write a single declaration.
    pub fn write_declaration(&self, decl: &SmtDeclaration) -> String {
        let mut output = String::new();
        self.write_declarations(&mut output, &[decl.clone()]);
        output
    }

    /// Write just the check-sat portion.
    pub fn write_check_sat_only(&self) -> String {
        let mut output = String::new();
        self.write_check_sat(&mut output);
        output
    }

    pub fn config(&self) -> &WriterConfig {
        &self.config
    }
}

// ─── Convenience functions ──────────────────────────────────────────────

/// Quick conversion of a formula to SMT-LIB2 string with default settings.
pub fn to_smtlib2(formula: &SmtFormula) -> String {
    let writer = SmtLib2Writer::with_defaults();
    writer.write_formula(formula)
}

/// Convert with custom logic setting.
pub fn to_smtlib2_with_logic(formula: &SmtFormula, logic: &str) -> String {
    let config = WriterConfig {
        logic: logic.to_string(),
        ..Default::default()
    };
    let writer = SmtLib2Writer::new(config);
    writer.write_formula(formula)
}

/// Generate minimal SMT-LIB2 (no comments, compact).
pub fn to_smtlib2_compact(formula: &SmtFormula) -> String {
    let config = WriterConfig {
        emit_comments: false,
        named_assertions: false,
        ..Default::default()
    };
    let writer = SmtLib2Writer::new(config);
    writer.write_formula(formula)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintOrigin, SmtConstraint, SmtFormula};

    fn make_test_formula() -> SmtFormula {
        let mut formula = SmtFormula::new(10, 3);
        formula.library_name = "negsyn-encode".to_string();
        formula.library_version = "0.1.0".to_string();

        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "x".to_string(),
            sort: SmtSort::BitVec(16),
        });
        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "y".to_string(),
            sort: SmtSort::BitVec(16),
        });
        formula.add_declaration(SmtDeclaration::DeclareConst {
            name: "active".to_string(),
            sort: SmtSort::Bool,
        });

        formula.add_constraint(SmtConstraint::new(
            SmtExpr::bv_ule(SmtExpr::var("x"), SmtExpr::bv_lit(100, 16)),
            ConstraintOrigin::InitialState,
            "x_bound",
        ));
        formula.add_constraint(SmtConstraint::new(
            SmtExpr::implies(
                SmtExpr::var("active"),
                SmtExpr::bv_ult(SmtExpr::var("x"), SmtExpr::var("y")),
            ),
            ConstraintOrigin::PropertyNegation,
            "downgrade_condition",
        ));

        formula
    }

    #[test]
    fn test_basic_output() {
        let formula = make_test_formula();
        let output = to_smtlib2(&formula);

        assert!(output.contains("(set-logic QF_AUFBV)"));
        assert!(output.contains("(declare-const x (_ BitVec 16))"));
        assert!(output.contains("(declare-const y (_ BitVec 16))"));
        assert!(output.contains("(assert"));
        assert!(output.contains("(check-sat)"));
        assert!(output.contains("(get-model)"));
        assert!(output.contains("(exit)"));
    }

    #[test]
    fn test_comments_present() {
        let formula = make_test_formula();
        let output = to_smtlib2(&formula);

        assert!(output.contains("; NegSynth DYENCODE"));
        assert!(output.contains("; Depth bound: 10"));
        assert!(output.contains("; Adversary budget: 3"));
        assert!(output.contains("; x_bound"));
        assert!(output.contains("; downgrade_condition"));
    }

    #[test]
    fn test_compact_output() {
        let formula = make_test_formula();
        let output = to_smtlib2_compact(&formula);

        assert!(!output.contains("; NegSynth"));
        assert!(output.contains("(set-logic QF_AUFBV)"));
        assert!(output.contains("(check-sat)"));
    }

    #[test]
    fn test_custom_logic() {
        let formula = make_test_formula();
        let output = to_smtlib2_with_logic(&formula, "QF_BV");

        assert!(output.contains("(set-logic QF_BV)"));
    }

    #[test]
    fn test_incremental_check() {
        let writer = SmtLib2Writer::with_defaults();
        let constraints = vec![SmtConstraint::new(
            SmtExpr::var("extra"),
            ConstraintOrigin::PropertyNegation,
            "extra_constraint",
        )];
        let output = writer.write_incremental_check(&constraints);

        assert!(output.contains("(push 1)"));
        assert!(output.contains("(assert extra)"));
        assert!(output.contains("(check-sat)"));
        assert!(output.contains("(pop 1)"));
    }

    #[test]
    fn test_push_pop() {
        let writer = SmtLib2Writer::with_defaults();
        assert_eq!(writer.write_push(), "(push 1)\n");
        assert_eq!(writer.write_pop(), "(pop 1)\n");
    }

    #[test]
    fn test_write_assert() {
        let writer = SmtLib2Writer::with_defaults();
        let expr = SmtExpr::eq(SmtExpr::var("x"), SmtExpr::bv_lit(42, 16));
        let output = writer.write_assert(&expr);
        assert!(output.contains("(assert (= x (_ bv42 16)))"));
    }

    #[test]
    fn test_named_assertions() {
        let config = WriterConfig {
            produce_unsat_cores: true,
            named_assertions: true,
            ..Default::default()
        };
        let writer = SmtLib2Writer::new(config);

        let formula = make_test_formula();
        let output = writer.write_formula(&formula);
        assert!(output.contains(":named"));
        assert!(output.contains("(get-unsat-core)"));
    }

    #[test]
    fn test_timeout_option() {
        let config = WriterConfig {
            timeout_ms: Some(60000),
            ..Default::default()
        };
        let writer = SmtLib2Writer::new(config);
        let formula = make_test_formula();
        let output = writer.write_formula(&formula);
        assert!(output.contains("(set-option :timeout 60000)"));
    }

    #[test]
    fn test_declare_fun() {
        let formula = {
            let mut f = SmtFormula::new(1, 1);
            f.add_declaration(SmtDeclaration::DeclareFun {
                name: "score".to_string(),
                args: vec![SmtSort::BitVec(16)],
                ret: SmtSort::BitVec(16),
            });
            f
        };
        let output = to_smtlib2(&formula);
        assert!(output.contains("(declare-fun score ((_ BitVec 16)) (_ BitVec 16))"));
    }

    #[test]
    fn test_define_fun() {
        let formula = {
            let mut f = SmtFormula::new(1, 1);
            f.add_declaration(SmtDeclaration::DefineFun {
                name: "is_high".to_string(),
                args: vec![("x".to_string(), SmtSort::BitVec(4))],
                ret: SmtSort::Bool,
                body: SmtExpr::bv_ule(SmtExpr::bv_lit(3, 4), SmtExpr::var("x")),
            });
            f
        };
        let output = to_smtlib2(&formula);
        assert!(output.contains("define-fun is_high"));
    }

    #[test]
    fn test_declare_sort() {
        let formula = {
            let mut f = SmtFormula::new(1, 1);
            f.add_declaration(SmtDeclaration::DeclareSort {
                name: "DYTerm".to_string(),
                arity: 0,
            });
            f
        };
        let output = to_smtlib2(&formula);
        assert!(output.contains("(declare-sort DYTerm 0)"));
    }

    #[test]
    fn test_origin_labels() {
        let writer = SmtLib2Writer::with_defaults();
        assert_eq!(
            writer.origin_section_label(&ConstraintOrigin::InitialState),
            "Initial State"
        );
        assert_eq!(
            writer.origin_section_label(&ConstraintOrigin::BudgetBound),
            "Budget Bounds"
        );
    }

    #[test]
    fn test_pretty_printing_large_and() {
        let printer = PrettyPrinter::new(2, 40);
        let expr = SmtExpr::and(vec![
            SmtExpr::var("a"),
            SmtExpr::var("b"),
            SmtExpr::var("c"),
            SmtExpr::var("d"),
        ]);
        let formatted = printer.format_expr(&expr);
        // Should either be single line or multi-line
        assert!(formatted.contains("and"));
    }

    #[test]
    fn test_write_to_buffer() {
        let formula = make_test_formula();
        let writer = SmtLib2Writer::with_defaults();
        let mut buf = Vec::new();
        writer.write_to(&formula, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("(check-sat)"));
    }

    #[test]
    fn test_write_declaration() {
        let writer = SmtLib2Writer::with_defaults();
        let decl = SmtDeclaration::DeclareConst {
            name: "test".to_string(),
            sort: SmtSort::Bool,
        };
        let output = writer.write_declaration(&decl);
        assert!(output.contains("(declare-const test Bool)"));
    }

    #[test]
    fn test_empty_formula() {
        let formula = SmtFormula::new(0, 0);
        let output = to_smtlib2(&formula);
        assert!(output.contains("(set-logic"));
        assert!(output.contains("(check-sat)"));
        assert!(output.contains("(exit)"));
    }
}
