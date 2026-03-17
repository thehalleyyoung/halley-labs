//! SMT-LIB2 script generation and response parsing.
//!
//! Provides types for constructing SMT-LIB2 commands, assembling scripts,
//! and interpreting solver responses.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::expr::{Expr, ExprId, ExprPool, Sort, SortedVar};

// ---------------------------------------------------------------------------
// SmtCommand
// ---------------------------------------------------------------------------

/// A single SMT-LIB2 command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtCommand {
    /// `(set-logic <symbol>)`
    SetLogic(String),
    /// `(declare-fun <name> (<sorts>) <sort>)`
    DeclareFun {
        name: String,
        arg_sorts: Vec<Sort>,
        return_sort: Sort,
    },
    /// `(declare-const <name> <sort>)`
    DeclareConst { name: String, sort: Sort },
    /// `(define-fun <name> (<sorted-vars>) <sort> <body>)`
    DefineFun {
        name: String,
        params: Vec<SortedVar>,
        return_sort: Sort,
        body: ExprId,
    },
    /// `(assert <expr>)`
    Assert(ExprId),
    /// `(check-sat)`
    CheckSat,
    /// `(get-model)`
    GetModel,
    /// `(get-value (<exprs>))`
    GetValue(Vec<ExprId>),
    /// `(push <n>)`
    Push(u32),
    /// `(pop <n>)`
    Pop(u32),
    /// `(reset)`
    Reset,
    /// `(exit)`
    Exit,
    /// Raw SMT-LIB2 text passthrough.
    Raw(String),
}

// ---------------------------------------------------------------------------
// SmtResponse
// ---------------------------------------------------------------------------

/// Parsed response from an SMT solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtResponse {
    Sat,
    Unsat,
    Unknown,
    Error(String),
    Model(String),
    Values(Vec<(String, String)>),
    Unsupported,
}

impl SmtResponse {
    /// Returns `true` when the solver reported satisfiable.
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResponse::Sat)
    }

    /// Returns `true` when the solver reported unsatisfiable.
    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResponse::Unsat)
    }
}

impl fmt::Display for SmtResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtResponse::Sat => write!(f, "sat"),
            SmtResponse::Unsat => write!(f, "unsat"),
            SmtResponse::Unknown => write!(f, "unknown"),
            SmtResponse::Error(e) => write!(f, "(error \"{}\")", e),
            SmtResponse::Model(m) => write!(f, "{}", m),
            SmtResponse::Values(vs) => {
                write!(f, "(")?;
                for (i, (k, v)) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", k, v)?;
                }
                write!(f, ")")
            }
            SmtResponse::Unsupported => write!(f, "unsupported"),
        }
    }
}

// ---------------------------------------------------------------------------
// Script
// ---------------------------------------------------------------------------

/// An ordered sequence of SMT-LIB2 commands forming a complete query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Script {
    /// The commands comprising this script, in order.
    pub commands: Vec<SmtCommand>,
}

impl Script {
    /// Create an empty script.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Append a command to the script.
    pub fn push(&mut self, cmd: SmtCommand) {
        self.commands.push(cmd);
    }

    /// Number of commands in the script.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the script has no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

impl Default for Script {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SmtLib2Writer
// ---------------------------------------------------------------------------

/// Serialises [`Script`] and individual [`SmtCommand`]s to SMT-LIB2 text.
#[derive(Debug)]
pub struct SmtLib2Writer<'a> {
    pool: &'a ExprPool,
    /// Indentation level for pretty-printing.
    indent: usize,
}

impl<'a> SmtLib2Writer<'a> {
    /// Create a new writer bound to an expression pool.
    pub fn new(pool: &'a ExprPool) -> Self {
        Self { pool, indent: 0 }
    }

    /// Render a complete script to an SMT-LIB2 string.
    pub fn render_script(&self, script: &Script) -> String {
        let mut out = String::new();
        for cmd in &script.commands {
            out.push_str(&self.render_command(cmd));
            out.push('\n');
        }
        out
    }

    /// Render a single command to an SMT-LIB2 string.
    pub fn render_command(&self, cmd: &SmtCommand) -> String {
        match cmd {
            SmtCommand::SetLogic(logic) => format!("(set-logic {})", logic),
            SmtCommand::DeclareFun {
                name,
                arg_sorts,
                return_sort,
            } => {
                let args: Vec<String> = arg_sorts.iter().map(|s| format!("{}", s)).collect();
                format!("(declare-fun {} ({}) {})", name, args.join(" "), return_sort)
            }
            SmtCommand::DeclareConst { name, sort } => {
                format!("(declare-const {} {})", name, sort)
            }
            SmtCommand::DefineFun { name, params, return_sort, body } => {
                let ps: Vec<String> = params.iter().map(|p| format!("{}", p)).collect();
                let body_str = format!("{}", crate::expr::ExprDisplay { pool: self.pool, id: *body });
                format!("(define-fun {} ({}) {} {})", name, ps.join(" "), return_sort, body_str)
            }
            SmtCommand::Assert(id) => {
                let e = format!("{}", crate::expr::ExprDisplay { pool: self.pool, id: *id });
                format!("(assert {})", e)
            }
            SmtCommand::CheckSat => "(check-sat)".to_string(),
            SmtCommand::GetModel => "(get-model)".to_string(),
            SmtCommand::GetValue(ids) => {
                let exprs: Vec<String> = ids
                    .iter()
                    .map(|id| format!("{}", crate::expr::ExprDisplay { pool: self.pool, id: *id }))
                    .collect();
                format!("(get-value ({}))", exprs.join(" "))
            }
            SmtCommand::Push(n) => format!("(push {})", n),
            SmtCommand::Pop(n) => format!("(pop {})", n),
            SmtCommand::Reset => "(reset)".to_string(),
            SmtCommand::Exit => "(exit)".to_string(),
            SmtCommand::Raw(s) => s.clone(),
        }
    }

    /// Parse raw solver output text into an [`SmtResponse`].
    pub fn parse_response(text: &str) -> SmtResponse {
        let trimmed = text.trim();
        match trimmed {
            "sat" => SmtResponse::Sat,
            "unsat" => SmtResponse::Unsat,
            "unknown" => SmtResponse::Unknown,
            "unsupported" => SmtResponse::Unsupported,
            other => {
                if other.starts_with("(error") {
                    SmtResponse::Error(other.to_string())
                } else if other.starts_with("(model") || other.starts_with("((") {
                    SmtResponse::Model(other.to_string())
                } else {
                    SmtResponse::Error(format!("unexpected solver output: {}", other))
                }
            }
        }
    }
}
