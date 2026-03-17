pub mod ast;
pub mod elaborator;
pub mod error;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod pretty_printer;
pub mod source_map;
pub mod token;
pub mod type_checker;
pub mod types;

// ─── Re-exports ─────────────────────────────────────────────────

pub use ast::{Program, Declaration, Expression, ObligationBody};
pub use error::DslError;
pub use ir::{Stage1Program, Stage2Program, Stage3Program};
pub use source_map::{SourceMap, Span};
pub use types::{DslType, TypeEnv};

// ─── High-level API ─────────────────────────────────────────────

/// Parse a DSL source string into an AST program.
/// Returns the program and any errors encountered.
pub fn parse_source(source: &str) -> Result<Program, Vec<DslError>> {
    let (tokens, lex_errors) = lexer::lex(source);
    if !lex_errors.is_empty() {
        return Err(lex_errors.into_iter().map(DslError::Lex).collect());
    }
    let (program, parse_errors) = parser::parse(tokens);
    if !parse_errors.is_empty() {
        return Err(parse_errors.into_iter().map(DslError::Parse).collect());
    }
    Ok(program)
}

/// Parse and type-check a DSL source string.
/// Returns the program and any errors encountered.
pub fn check_source(source: &str) -> Result<Program, Vec<DslError>> {
    let program = parse_source(source)?;
    let mut checker = type_checker::TypeChecker::new();
    let type_errors = checker.check_program(&program);
    if !type_errors.is_empty() {
        return Err(type_errors.into_iter().map(DslError::Type).collect());
    }
    Ok(program)
}

/// Full compilation pipeline: parse → type-check → elaborate → lower to Stage3 IR.
pub fn compile(source: &str) -> Result<Stage3Program, Vec<DslError>> {
    let program = check_source(source)?;

    let s1 = ir::ast_to_stage1(&program);
    let s2 = ir::stage1_to_stage2(&s1);

    let mut elaborator_inst = elaborator::Elaborator::new();
    let (elaborated, elab_errors) = elaborator_inst.elaborate(&s2);
    if !elab_errors.is_empty() {
        return Err(elab_errors.into_iter().map(DslError::Elaboration).collect());
    }

    Ok(ir::stage2_to_stage3(&elaborated))
}

/// Pretty-print a program back to source code.
pub fn format_source(source: &str) -> Result<String, Vec<DslError>> {
    let program = parse_source(source)?;
    Ok(pretty_printer::pretty_print(&program))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_source() {
        let result = parse_source(r#"
            obligation transparency {
                requires: true;
                risk_level: high;
            }
        "#);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.declarations.len(), 1);
    }

    #[test]
    fn test_parse_source_error() {
        let result = parse_source("obligation { }");
        assert!(result.is_err());
    }

    #[test]
    fn test_check_source() {
        let result = check_source(r#"
            obligation test {
                requires: true and false;
                formalizability: 3;
            }
        "#);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_source_type_error() {
        let result = check_source(r#"
            obligation test {
                requires: 42;
            }
        "#);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile() {
        let result = compile(r#"
            obligation obl_a {
                requires: true;
                risk_level: high;
                formalizability: 2;
            }
        "#);
        assert!(result.is_ok());
        let s3 = result.unwrap();
        assert!(!s3.constraints.is_empty());
        assert!(!s3.variables.is_empty());
    }

    #[test]
    fn test_format_source() {
        let result = format_source(r#"
            obligation test { requires: true; risk_level: high; }
        "#);
        assert!(result.is_ok());
        let formatted = result.unwrap();
        assert!(formatted.contains("obligation test"));
        assert!(formatted.contains("risk_level: high"));
    }

    #[test]
    fn test_full_pipeline_complex() {
        let result = compile(r#"
            jurisdiction "EU" {
                obligation transparency {
                    risk_level: high;
                    formalizability: 2;
                    requires: true;
                    domain: "healthcare";
                    temporal: #2024-01-01 -> #2025-12-31;
                }
                permission research_use {
                    requires: true;
                    formalizability: 1;
                }
            }
            prohibition social_scoring {
                risk_level: unacceptable;
                requires: true;
                formalizability: 4;
            }
        "#);
        assert!(result.is_ok(), "compile errors: {:?}", result.err());
        let s3 = result.unwrap();
        assert!(s3.variables.len() >= 3);
    }

    #[test]
    fn test_roundtrip_format() {
        let src = r#"obligation test {
    risk_level: high;
    formalizability: 3;
    requires: true;
}
"#;
        let formatted1 = format_source(src).unwrap();
        let formatted2 = format_source(&formatted1).unwrap();
        assert_eq!(formatted1, formatted2, "format is not idempotent");
    }
}
