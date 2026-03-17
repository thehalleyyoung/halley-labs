//! FPBench (FPCore) format support.
//!
//! Implements parsing and emission of the FPBench standard format
//! (see <https://fpbench.org>), the community benchmark format for
//! floating-point expressions.
//!
//! # FPCore Syntax (subset)
//!
//! ```text
//! (FPCore (x y z)
//!   :name "example"
//!   :pre (> x 0)
//!   (+ x (* y z)))
//! ```

use crate::expression::{ExprBuilder, ExprNode, ExprTree, FpOp, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// An FPCore benchmark expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpCore {
    /// Benchmark name.
    pub name: Option<String>,
    /// Input variable names.
    pub inputs: Vec<String>,
    /// Precondition on inputs (human-readable string).
    pub precondition: Option<String>,
    /// The expression tree.
    pub body: ExprTree,
    /// Additional properties (`:cite`, `:description`, etc.).
    pub properties: HashMap<String, String>,
}

/// Errors during FPCore parsing.
#[derive(Debug, Clone)]
pub enum FpBenchError {
    UnexpectedEof,
    ExpectedToken(String),
    UnknownOperator(String),
    MalformedNumber(String),
    MissingInputs,
}

impl fmt::Display for FpBenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::ExpectedToken(t) => write!(f, "expected token: {}", t),
            Self::UnknownOperator(op) => write!(f, "unknown operator: {}", op),
            Self::MalformedNumber(s) => write!(f, "malformed number: {}", s),
            Self::MissingInputs => write!(f, "missing input variable list"),
        }
    }
}

impl std::error::Error for FpBenchError {}

/// Tokenizer for S-expression-based FPCore.
struct Tokenizer {
    tokens: Vec<String>,
    pos: usize,
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        let mut tokens = Vec::new();
        let padded = input.replace('(', " ( ").replace(')', " ) ");
        for tok in padded.split_whitespace() {
            tokens.push(tok.to_string());
        }
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Result<&str, FpBenchError> {
        if self.pos < self.tokens.len() {
            let tok = &self.tokens[self.pos];
            self.pos += 1;
            Ok(tok)
        } else {
            Err(FpBenchError::UnexpectedEof)
        }
    }

    fn expect(&mut self, expected: &str) -> Result<(), FpBenchError> {
        let tok = self.next()?;
        if tok == expected {
            Ok(())
        } else {
            Err(FpBenchError::ExpectedToken(expected.to_string()))
        }
    }
}

/// Parse an FPCore string into an [`FpCore`].
///
/// # Example
///
/// ```
/// use fpdiag_types::fpbench::parse_fpcore;
///
/// let src = "(FPCore (x) :name \"square\" (* x x))";
/// let core = parse_fpcore(src).unwrap();
/// assert_eq!(core.name.as_deref(), Some("square"));
/// assert_eq!(core.inputs, vec!["x"]);
/// ```
pub fn parse_fpcore(input: &str) -> Result<FpCore, FpBenchError> {
    let mut tok = Tokenizer::new(input);
    tok.expect("(")?;
    tok.expect("FPCore")?;

    // Parse input list
    tok.expect("(")?;
    let mut inputs = Vec::new();
    loop {
        match tok.peek() {
            Some(")") => {
                tok.next()?;
                break;
            }
            Some(_) => {
                let name = tok.next()?.to_string();
                inputs.push(name);
            }
            None => return Err(FpBenchError::UnexpectedEof),
        }
    }

    // Parse properties
    let mut name = None;
    let mut precondition = None;
    let mut properties = HashMap::new();

    while let Some(t) = tok.peek() {
        if t.starts_with(':') {
            let key = tok.next()?.to_string();
            let val = tok.next()?.to_string();
            let val = val.trim_matches('"').to_string();
            match key.as_str() {
                ":name" => name = Some(val),
                ":pre" => precondition = Some(val),
                _ => {
                    properties.insert(key, val);
                }
            }
        } else {
            break;
        }
    }

    // Parse body expression
    let mut builder = ExprBuilder::new();
    let mut var_map: HashMap<String, NodeId> = HashMap::new();
    for inp in &inputs {
        let nid = builder.variable(inp.clone());
        var_map.insert(inp.clone(), nid);
    }

    let root = parse_expr(&mut tok, &mut builder, &var_map)?;
    let body = builder.build(root);

    // Consume trailing )
    let _ = tok.next();

    Ok(FpCore {
        name,
        inputs,
        precondition,
        body,
        properties,
    })
}

fn parse_expr(
    tok: &mut Tokenizer,
    builder: &mut ExprBuilder,
    vars: &HashMap<String, NodeId>,
) -> Result<NodeId, FpBenchError> {
    let t = tok.next()?;
    if t == "(" {
        let op_str = tok.next()?;
        let op = match op_str {
            "+" => FpOp::Add,
            "-" => FpOp::Sub,
            "*" => FpOp::Mul,
            "/" => FpOp::Div,
            "neg" => FpOp::Neg,
            "abs" | "fabs" => FpOp::Abs,
            "sqrt" => FpOp::Sqrt,
            "fma" => FpOp::Fma,
            "exp" => FpOp::Exp,
            "exp2" => FpOp::Exp2,
            "log" => FpOp::Log,
            "log2" => FpOp::Log2,
            "log10" => FpOp::Log10,
            "pow" => FpOp::Pow,
            "sin" => FpOp::Sin,
            "cos" => FpOp::Cos,
            "tan" => FpOp::Tan,
            "asin" => FpOp::Asin,
            "acos" => FpOp::Acos,
            "atan" => FpOp::Atan,
            "atan2" => FpOp::Atan2,
            "sinh" => FpOp::Sinh,
            "cosh" => FpOp::Cosh,
            "tanh" => FpOp::Tanh,
            "floor" => FpOp::Floor,
            "ceil" => FpOp::Ceil,
            "round" => FpOp::Round,
            "trunc" | "truncate" => FpOp::Trunc,
            "fmin" | "min" => FpOp::Min,
            "fmax" | "max" => FpOp::Max,
            other => return Err(FpBenchError::UnknownOperator(other.to_string())),
        };

        let mut children = Vec::new();
        loop {
            if tok.peek() == Some(")") {
                tok.next()?;
                break;
            }
            children.push(parse_expr(tok, builder, vars)?);
        }

        Ok(builder.op(op, children))
    } else if let Some(&nid) = vars.get(t) {
        Ok(nid)
    } else if let Ok(val) = t.parse::<f64>() {
        Ok(builder.constant(val))
    } else {
        // Treat as a new variable reference
        Ok(builder.variable(t.to_string()))
    }
}

/// Emit an [`ExprTree`] as an FPCore S-expression string.
pub fn emit_fpcore(core: &FpCore) -> String {
    let mut out = String::new();
    out.push_str("(FPCore (");
    out.push_str(&core.inputs.join(" "));
    out.push(')');

    if let Some(name) = &core.name {
        out.push_str(&format!("\n  :name \"{}\"", name));
    }
    if let Some(pre) = &core.precondition {
        out.push_str(&format!("\n  :pre {}", pre));
    }
    for (k, v) in &core.properties {
        out.push_str(&format!("\n  {} \"{}\"", k, v));
    }

    out.push_str("\n  ");
    if let Some(root) = core.body.root {
        emit_node(&core.body, root, &mut out);
    }
    out.push(')');
    out
}

fn emit_node(tree: &ExprTree, id: NodeId, out: &mut String) {
    match tree.get(id) {
        Some(ExprNode::Constant(v)) => {
            if *v == v.floor() && v.abs() < 1e15 {
                out.push_str(&format!("{:.1}", v));
            } else {
                out.push_str(&format!("{:e}", v));
            }
        }
        Some(ExprNode::Variable { name, .. }) => {
            out.push_str(name);
        }
        Some(ExprNode::Operation { op, children }) => {
            out.push('(');
            out.push_str(op.symbol());
            for &child in children {
                out.push(' ');
                emit_node(tree, child, out);
            }
            out.push(')');
        }
        Some(ExprNode::LibraryCall { function, children }) => {
            out.push('(');
            out.push_str(function);
            for &child in children {
                out.push(' ');
                emit_node(tree, child, out);
            }
            out.push(')');
        }
        None => out.push_str("???"),
    }
}

/// Emit an SMT-LIB2 representation of an expression tree.
///
/// Uses the QF_FP logic for floating-point reasoning.
pub fn emit_smtlib(tree: &ExprTree, variables: &[String]) -> String {
    let mut out = String::new();
    out.push_str("; SMT-LIB2 encoding generated by Penumbra\n");
    out.push_str("(set-logic QF_FP)\n");
    out.push_str("(set-info :source |Penumbra FP diagnosis engine|)\n\n");

    for var in variables {
        out.push_str(&format!(
            "(declare-const {} (_ FloatingPoint 11 53))\n",
            var
        ));
    }
    out.push('\n');

    if let Some(root) = tree.root {
        out.push_str("(define-fun result () (_ FloatingPoint 11 53)\n  ");
        emit_smtlib_node(tree, root, &mut out);
        out.push_str(")\n\n");
    }

    out.push_str("(check-sat)\n");
    out.push_str("(exit)\n");
    out
}

fn emit_smtlib_node(tree: &ExprTree, id: NodeId, out: &mut String) {
    match tree.get(id) {
        Some(ExprNode::Constant(v)) => {
            let bits = v.to_bits();
            let sign = if *v < 0.0 { "1" } else { "0" };
            let exp = (bits >> 52) & 0x7FF;
            let frac = bits & 0xFFFFFFFFFFFFF;
            out.push_str(&format!("(fp #b{} #b{:011b} #b{:052b})", sign, exp, frac));
        }
        Some(ExprNode::Variable { name, .. }) => {
            out.push_str(name);
        }
        Some(ExprNode::Operation { op, children }) => {
            let smt_op = match op {
                FpOp::Add => "fp.add RNE",
                FpOp::Sub => "fp.sub RNE",
                FpOp::Mul => "fp.mul RNE",
                FpOp::Div => "fp.div RNE",
                FpOp::Neg => "fp.neg",
                FpOp::Abs => "fp.abs",
                FpOp::Sqrt => "fp.sqrt RNE",
                FpOp::Fma => "fp.fma RNE",
                FpOp::Min => "fp.min",
                FpOp::Max => "fp.max",
                _ => "fp.add RNE", // fallback for unsupported transcendentals
            };
            out.push('(');
            out.push_str(smt_op);
            for &child in children {
                out.push(' ');
                emit_smtlib_node(tree, child, out);
            }
            out.push(')');
        }
        Some(ExprNode::LibraryCall { function, children }) => {
            out.push_str(&format!("(|{}|", function));
            for &child in children {
                out.push(' ');
                emit_smtlib_node(tree, child, out);
            }
            out.push(')');
        }
        None => out.push_str("(_ NaN 11 53)"),
    }
}

/// Parse multiple FPCore expressions from a file.
pub fn parse_fpbench_file(content: &str) -> Vec<Result<FpCore, FpBenchError>> {
    let mut results = Vec::new();
    let mut depth = 0i32;
    let mut start = None;

    for (i, ch) in content.char_indices() {
        match ch {
            '(' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        results.push(parse_fpcore(&content[s..=i]));
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_fpcore() {
        let src = r#"(FPCore (x) :name "square" (* x x))"#;
        let core = parse_fpcore(src).unwrap();
        assert_eq!(core.name.as_deref(), Some("square"));
        assert_eq!(core.inputs, vec!["x"]);
        assert_eq!(core.body.len(), 2); // x (variable) and * (op referencing x twice)
    }

    #[test]
    fn parse_multi_input() {
        let src = r#"(FPCore (x y z) :name "fma-test" (fma x y z))"#;
        let core = parse_fpcore(src).unwrap();
        assert_eq!(core.inputs.len(), 3);
    }

    #[test]
    fn roundtrip_emit() {
        let src = r#"(FPCore (x y) :name "add" (+ x y))"#;
        let core = parse_fpcore(src).unwrap();
        let emitted = emit_fpcore(&core);
        assert!(emitted.contains("FPCore"));
        assert!(emitted.contains("+ x y"));
    }

    #[test]
    fn emit_smtlib_basic() {
        let src = r#"(FPCore (x y) (+ x y))"#;
        let core = parse_fpcore(src).unwrap();
        let smt = emit_smtlib(&core.body, &core.inputs);
        assert!(smt.contains("QF_FP"));
        assert!(smt.contains("declare-const x"));
        assert!(smt.contains("fp.add RNE"));
    }

    #[test]
    fn parse_nested_expr() {
        let src = r#"(FPCore (a b c) :name "quadratic"
            (/ (+ (neg b) (sqrt (- (* b b) (* 4.0 (* a c))))) (* 2.0 a)))"#;
        let core = parse_fpcore(src).unwrap();
        assert_eq!(core.name.as_deref(), Some("quadratic"));
        assert!(core.body.len() > 5);
    }

    #[test]
    fn parse_multiple_benchmarks() {
        let content = r#"
            (FPCore (x) :name "exp-1" (- (exp x) 1.0))
            (FPCore (x) :name "log1p" (log (+ 1.0 x)))
        "#;
        let results = parse_fpbench_file(content);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
