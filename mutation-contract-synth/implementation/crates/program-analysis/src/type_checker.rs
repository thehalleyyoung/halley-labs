//! Type checking for the loop-free imperative language.
//!
//! Validates types, enforces QF-LIA fragment constraints (no nonlinear arithmetic),
//! and checks for loop-free property (no loops or recursion).

use std::collections::{HashMap, HashSet};

use shared_types::{
    AnalysisError, AnalysisResult, ArithOp, Expression, Function, LogicOp, Parameter, Program,
    RelOp, Statement, SymbolTable, Type, UnaryOp,
};

// ---------------------------------------------------------------------------
// TypeChecker
// ---------------------------------------------------------------------------

/// Type checker with type environment management.
pub struct TypeChecker {
    symbols: SymbolTable,
    function_sigs: HashMap<String, FunctionSig>,
    errors: Vec<AnalysisError>,
    /// Track function call graph for recursion detection.
    call_graph: HashMap<String, HashSet<String>>,
    /// Currently type-checking function name.
    current_function: Option<String>,
}

/// A function signature for type checking.
#[derive(Debug, Clone)]
struct FunctionSig {
    params: Vec<(String, Type)>,
    return_type: Type,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            symbols: SymbolTable::new(),
            function_sigs: HashMap::new(),
            errors: Vec::new(),
            call_graph: HashMap::new(),
            current_function: None,
        }
    }

    /// Type check an entire program.
    pub fn check_program(&mut self, program: &Program) -> AnalysisResult<()> {
        // First pass: collect function signatures
        for func in &program.functions {
            let sig = FunctionSig {
                params: func
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), p.ty.clone()))
                    .collect(),
                return_type: func.return_type.clone(),
            };
            if self.function_sigs.insert(func.name.clone(), sig).is_some() {
                self.errors.push(AnalysisError::TypeError {
                    message: format!("Duplicate function definition: '{}'", func.name),
                    location: Some(func.span.start),
                });
            }
        }

        // Second pass: type check each function body
        for func in &program.functions {
            self.check_function(func)?;
        }

        if !self.errors.is_empty() {
            return Err(self.errors[0].clone());
        }
        Ok(())
    }

    /// Type check a single function.
    pub fn check_function(&mut self, func: &Function) -> AnalysisResult<()> {
        self.symbols = SymbolTable::new();
        self.current_function = Some(func.name.clone());
        self.call_graph.entry(func.name.clone()).or_default();

        // Register parameters
        for param in &func.params {
            self.symbols.define(param.name.clone(), param.ty.clone());
        }

        // Check body
        for stmt in &func.body {
            if let Err(e) = self.check_statement(stmt, &func.return_type) {
                self.errors.push(e);
            }
        }

        self.current_function = None;
        Ok(())
    }

    // -- Statement type checking --------------------------------------------

    fn check_statement(&mut self, stmt: &Statement, ret_type: &Type) -> AnalysisResult<()> {
        match stmt {
            Statement::VarDecl {
                name,
                ty,
                init,
                span,
            } => {
                if self.symbols.is_defined(name) {
                    self.errors.push(AnalysisError::TypeError {
                        message: format!("Variable '{}' already declared in this scope", name),
                        location: Some(span.start),
                    });
                }
                self.symbols.define(name.clone(), ty.clone());
                if let Some(init_expr) = init {
                    let init_ty = self.infer_type(init_expr)?;
                    self.check_type_compatible(ty, &init_ty, &span.start)?;
                }
                Ok(())
            }
            Statement::Assign {
                target,
                value,
                span,
            } => {
                let target_ty = self.symbols.lookup(target).cloned();
                match target_ty {
                    Some(ty) => {
                        let val_ty = self.infer_type(value)?;
                        self.check_type_compatible(&ty, &val_ty, &span.start)
                    }
                    None if target == "_" => {
                        // Discard result (function call as statement)
                        let _ = self.infer_type(value)?;
                        Ok(())
                    }
                    None => Err(AnalysisError::TypeError {
                        message: format!("Undefined variable '{}'", target),
                        location: Some(span.start),
                    }),
                }
            }
            Statement::ArrayAssign {
                array,
                index,
                value,
                span,
            } => {
                let arr_ty = self.symbols.lookup(array).cloned().ok_or_else(|| {
                    AnalysisError::TypeError {
                        message: format!("Undefined array '{}'", array),
                        location: Some(span.start),
                    }
                })?;
                match &arr_ty {
                    Type::Array(elem_ty) => {
                        let idx_ty = self.infer_type(index)?;
                        self.check_type_compatible(&Type::Int, &idx_ty, &span.start)?;
                        let val_ty = self.infer_type(value)?;
                        self.check_type_compatible(elem_ty, &val_ty, &span.start)
                    }
                    _ => Err(AnalysisError::TypeError {
                        message: format!("'{}' is not an array", array),
                        location: Some(span.start),
                    }),
                }
            }
            Statement::If {
                condition,
                then_block,
                else_block,
                span,
            } => {
                let cond_ty = self.infer_type(condition)?;
                self.check_type_compatible(&Type::Bool, &cond_ty, &span.start)?;
                self.symbols.push_scope();
                for s in then_block {
                    self.check_statement(s, ret_type)?;
                }
                self.symbols.pop_scope();
                if let Some(else_stmts) = else_block {
                    self.symbols.push_scope();
                    for s in else_stmts {
                        self.check_statement(s, ret_type)?;
                    }
                    self.symbols.pop_scope();
                }
                Ok(())
            }
            Statement::Return { value, span } => match (value, ret_type) {
                (None, Type::Void) => Ok(()),
                (None, _) => Err(AnalysisError::TypeError {
                    message: format!("Expected return value of type {}", ret_type),
                    location: Some(span.start),
                }),
                (Some(expr), Type::Void) => Err(AnalysisError::TypeError {
                    message: "Cannot return a value from void function".into(),
                    location: Some(span.start),
                }),
                (Some(expr), expected) => {
                    let actual = self.infer_type(expr)?;
                    self.check_type_compatible(expected, &actual, &span.start)
                }
            },
            Statement::Assert { condition, span } => {
                let cond_ty = self.infer_type(condition)?;
                self.check_type_compatible(&Type::Bool, &cond_ty, &span.start)
            }
            Statement::Block { stmts, .. } => {
                self.symbols.push_scope();
                for s in stmts {
                    self.check_statement(s, ret_type)?;
                }
                self.symbols.pop_scope();
                Ok(())
            }
        }
    }

    // -- Expression type inference ------------------------------------------

    /// Infer the type of an expression.
    pub fn infer_type(&mut self, expr: &Expression) -> AnalysisResult<Type> {
        match expr {
            Expression::IntLiteral { .. } => Ok(Type::Int),
            Expression::BoolLiteral { .. } => Ok(Type::Bool),
            Expression::Variable { name, span } => {
                self.symbols
                    .lookup(name)
                    .cloned()
                    .ok_or_else(|| AnalysisError::TypeError {
                        message: format!("Undefined variable '{}'", name),
                        location: Some(span.start),
                    })
            }
            Expression::BinaryArith {
                left,
                op,
                right,
                span,
            } => {
                let lt = self.infer_type(left)?;
                let rt = self.infer_type(right)?;
                self.check_type_compatible(&Type::Int, &lt, &span.start)?;
                self.check_type_compatible(&Type::Int, &rt, &span.start)?;
                Ok(Type::Int)
            }
            Expression::BinaryRel {
                left,
                op,
                right,
                span,
            } => {
                let lt = self.infer_type(left)?;
                let rt = self.infer_type(right)?;
                self.check_type_compatible(&lt, &rt, &span.start)?;
                Ok(Type::Bool)
            }
            Expression::BinaryLogic {
                left,
                op,
                right,
                span,
            } => {
                let lt = self.infer_type(left)?;
                let rt = self.infer_type(right)?;
                self.check_type_compatible(&Type::Bool, &lt, &span.start)?;
                self.check_type_compatible(&Type::Bool, &rt, &span.start)?;
                Ok(Type::Bool)
            }
            Expression::Unary { op, operand, span } => {
                let inner = self.infer_type(operand)?;
                match op {
                    UnaryOp::Neg => {
                        self.check_type_compatible(&Type::Int, &inner, &span.start)?;
                        Ok(Type::Int)
                    }
                    UnaryOp::Not => {
                        self.check_type_compatible(&Type::Bool, &inner, &span.start)?;
                        Ok(Type::Bool)
                    }
                    UnaryOp::BitwiseNot => {
                        self.check_type_compatible(&Type::Int, &inner, &span.start)?;
                        Ok(Type::Int)
                    }
                }
            }
            Expression::ArrayAccess { array, index, span } => {
                let arr_ty = self.infer_type(array)?;
                let idx_ty = self.infer_type(index)?;
                self.check_type_compatible(&Type::Int, &idx_ty, &span.start)?;
                match arr_ty {
                    Type::Array(elem) => Ok(*elem),
                    _ => Err(AnalysisError::TypeError {
                        message: "Cannot index non-array type".into(),
                        location: Some(span.start),
                    }),
                }
            }
            Expression::FunctionCall { name, args, span } => {
                // Record in call graph
                if let Some(ref caller) = self.current_function {
                    self.call_graph
                        .entry(caller.clone())
                        .or_default()
                        .insert(name.clone());
                }

                let sig = self.function_sigs.get(name).cloned().ok_or_else(|| {
                    AnalysisError::TypeError {
                        message: format!("Undefined function '{}'", name),
                        location: Some(span.start),
                    }
                })?;

                if args.len() != sig.params.len() {
                    return Err(AnalysisError::TypeError {
                        message: format!(
                            "Function '{}' expects {} arguments, got {}",
                            name,
                            sig.params.len(),
                            args.len()
                        ),
                        location: Some(span.start),
                    });
                }

                for (arg, (_, param_ty)) in args.iter().zip(sig.params.iter()) {
                    let arg_ty = self.infer_type(arg)?;
                    self.check_type_compatible(param_ty, &arg_ty, &span.start)?;
                }

                Ok(sig.return_type)
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                span,
            } => {
                let cond_ty = self.infer_type(condition)?;
                self.check_type_compatible(&Type::Bool, &cond_ty, &span.start)?;
                let then_ty = self.infer_type(then_expr)?;
                let else_ty = self.infer_type(else_expr)?;
                self.check_type_compatible(&then_ty, &else_ty, &span.start)?;
                Ok(then_ty)
            }
        }
    }

    fn check_type_compatible(
        &self,
        expected: &Type,
        actual: &Type,
        loc: &shared_types::SourceLocation,
    ) -> AnalysisResult<()> {
        if expected == actual
            || matches!(expected, Type::Unknown)
            || matches!(actual, Type::Unknown)
        {
            Ok(())
        } else {
            Err(AnalysisError::TypeError {
                message: format!("Type mismatch: expected {}, got {}", expected, actual),
                location: Some(*loc),
            })
        }
    }

    // -- QF-LIA validation --------------------------------------------------

    /// Validate that the program is in the QF-LIA (quantifier-free linear integer arithmetic)
    /// fragment: no nonlinear multiplication (both operands non-constant).
    pub fn validate_qflia(&self, program: &Program) -> AnalysisResult<()> {
        for func in &program.functions {
            for stmt in &func.body {
                self.check_qflia_statement(stmt)?;
            }
        }
        Ok(())
    }

    fn check_qflia_statement(&self, stmt: &Statement) -> AnalysisResult<()> {
        match stmt {
            Statement::VarDecl { init: Some(e), .. } => self.check_qflia_expr(e),
            Statement::Assign { value, .. } => self.check_qflia_expr(value),
            Statement::ArrayAssign { index, value, .. } => {
                self.check_qflia_expr(index)?;
                self.check_qflia_expr(value)
            }
            Statement::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                self.check_qflia_expr(condition)?;
                for s in then_block {
                    self.check_qflia_statement(s)?;
                }
                if let Some(stmts) = else_block {
                    for s in stmts {
                        self.check_qflia_statement(s)?;
                    }
                }
                Ok(())
            }
            Statement::Return { value: Some(e), .. } => self.check_qflia_expr(e),
            Statement::Assert { condition, .. } => self.check_qflia_expr(condition),
            Statement::Block { stmts, .. } => {
                for s in stmts {
                    self.check_qflia_statement(s)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn check_qflia_expr(&self, expr: &Expression) -> AnalysisResult<()> {
        match expr {
            Expression::BinaryArith {
                left,
                op: ArithOp::Mul,
                right,
                span,
            } => {
                // At least one operand must be a constant for linear arithmetic
                if !left.is_literal() && !right.is_literal() {
                    return Err(AnalysisError::ValidationError {
                        message: format!(
                            "Nonlinear multiplication detected: both operands are non-constant"
                        ),
                    });
                }
                self.check_qflia_expr(left)?;
                self.check_qflia_expr(right)
            }
            Expression::BinaryArith { left, right, .. }
            | Expression::BinaryRel { left, right, .. }
            | Expression::BinaryLogic { left, right, .. } => {
                self.check_qflia_expr(left)?;
                self.check_qflia_expr(right)
            }
            Expression::Unary { operand, .. } => self.check_qflia_expr(operand),
            Expression::ArrayAccess { array, index, .. } => {
                self.check_qflia_expr(array)?;
                self.check_qflia_expr(index)
            }
            Expression::FunctionCall { args, .. } => {
                for a in args {
                    self.check_qflia_expr(a)?;
                }
                Ok(())
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                self.check_qflia_expr(condition)?;
                self.check_qflia_expr(then_expr)?;
                self.check_qflia_expr(else_expr)
            }
            _ => Ok(()),
        }
    }

    // -- Loop-free validation -----------------------------------------------

    /// Validate that the program contains no loops or recursion.
    pub fn validate_loop_free(&self, program: &Program) -> AnalysisResult<()> {
        // Check for loops in statements
        for func in &program.functions {
            self.check_no_loops(&func.body)?;
        }

        // Check for recursion in call graph
        self.check_no_recursion()?;

        Ok(())
    }

    fn check_no_loops(&self, stmts: &[Statement]) -> AnalysisResult<()> {
        for stmt in stmts {
            match stmt {
                Statement::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    self.check_no_loops(then_block)?;
                    if let Some(stmts) = else_block {
                        self.check_no_loops(stmts)?;
                    }
                }
                Statement::Block { stmts, .. } => {
                    self.check_no_loops(stmts)?;
                }
                _ => {} // No loop constructs in our language
            }
        }
        Ok(())
    }

    fn check_no_recursion(&self) -> AnalysisResult<()> {
        // Detect cycles in the call graph using DFS
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for func_name in self.call_graph.keys() {
            if !visited.contains(func_name) {
                if self.has_cycle(func_name, &mut visited, &mut in_stack) {
                    return Err(AnalysisError::ValidationError {
                        message: format!("Recursion detected involving function '{}'", func_name),
                    });
                }
            }
        }
        Ok(())
    }

    fn has_cycle(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        in_stack.insert(node.to_string());

        if let Some(callees) = self.call_graph.get(node) {
            for callee in callees {
                if !visited.contains(callee) {
                    if self.has_cycle(callee, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(callee) {
                    return true;
                }
            }
        }

        in_stack.remove(node);
        false
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn check(src: &str) -> AnalysisResult<()> {
        let prog = Parser::parse_source(src)?;
        let mut checker = TypeChecker::new();
        checker.check_program(&prog)
    }

    fn check_ok(src: &str) {
        check(src).expect("Type check should succeed");
    }

    fn check_err(src: &str) -> AnalysisError {
        check(src).expect_err("Type check should fail")
    }

    #[test]
    fn test_type_simple() {
        check_ok("fn f(x: int) -> int { return x; }");
    }

    #[test]
    fn test_type_arithmetic() {
        check_ok("fn f(x: int, y: int) -> int { return x + y; }");
    }

    #[test]
    fn test_type_comparison() {
        check_ok("fn f(x: int, y: int) -> bool { return x > y; }");
    }

    #[test]
    fn test_type_logic() {
        check_ok("fn f(a: bool, b: bool) -> bool { return a && b; }");
    }

    #[test]
    fn test_type_if() {
        check_ok("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
    }

    #[test]
    fn test_type_assert() {
        check_ok("fn f(x: int) -> void { assert(x > 0); }");
    }

    #[test]
    fn test_type_var_decl() {
        check_ok("fn f() -> int { let x: int = 42; return x; }");
    }

    #[test]
    fn test_type_array() {
        check_ok("fn f(a: array<int>) -> int { return a[0]; }");
    }

    #[test]
    fn test_type_function_call() {
        check_ok("fn add(x: int, y: int) -> int { return x + y; } fn main() -> int { return add(1, 2); }");
    }

    #[test]
    fn test_type_ternary() {
        check_ok("fn f(x: int) -> int { return x > 0 ? x : -x; }");
    }

    #[test]
    fn test_type_negation() {
        check_ok("fn f(x: int) -> int { return -x; }");
    }

    #[test]
    fn test_type_not() {
        check_ok("fn f(b: bool) -> bool { return !b; }");
    }

    #[test]
    fn test_type_void_return() {
        check_ok("fn f() -> void { return; }");
    }

    #[test]
    fn test_type_error_return_mismatch() {
        let err = check_err("fn f() -> int { return true; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_arith_on_bool() {
        let err = check_err("fn f(x: bool) -> int { return x + 1; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_logic_on_int() {
        let err = check_err("fn f(x: int, y: int) -> bool { return x && y; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_undefined_var() {
        let err = check_err("fn f() -> int { return z; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_undefined_func() {
        let err = check_err("fn f() -> int { return unknown(1); }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_wrong_arg_count() {
        let err = check_err(
            "fn add(x: int, y: int) -> int { return x + y; } fn f() -> int { return add(1); }",
        );
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_cond_not_bool() {
        let err = check_err("fn f(x: int) -> int { if (x) { return 1; } else { return 0; } }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_void_return_value() {
        let err = check_err("fn f() -> void { return 42; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_index_non_array() {
        let err = check_err("fn f(x: int) -> int { return x[0]; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_neg_bool() {
        let err = check_err("fn f(b: bool) -> int { return -b; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_not_int() {
        let err = check_err("fn f(x: int) -> bool { return !x; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    // -- QF-LIA validation tests --

    #[test]
    fn test_qflia_linear() {
        let prog = Parser::parse_source("fn f(x: int) -> int { return x * 2; }").unwrap();
        let checker = TypeChecker::new();
        assert!(checker.validate_qflia(&prog).is_ok());
    }

    #[test]
    fn test_qflia_nonlinear() {
        let prog = Parser::parse_source("fn f(x: int, y: int) -> int { return x * y; }").unwrap();
        let checker = TypeChecker::new();
        assert!(checker.validate_qflia(&prog).is_err());
    }

    // -- Loop-free validation tests --

    #[test]
    fn test_loop_free_valid() {
        let prog = Parser::parse_source(
            "fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }",
        )
        .unwrap();
        let mut checker = TypeChecker::new();
        checker.check_program(&prog).unwrap();
        assert!(checker.validate_loop_free(&prog).is_ok());
    }

    #[test]
    fn test_complex_type_check() {
        check_ok(
            r#"
            fn abs(x: int) -> int {
                if (x >= 0) { return x; } else { return -x; }
            }
            fn max(a: int, b: int) -> int {
                if (a >= b) { return a; } else { return b; }
            }
            fn clamp(x: int, lo: int, hi: int) -> int {
                if (x < lo) { return lo; }
                else if (x > hi) { return hi; }
                else { return x; }
            }
        "#,
        );
    }

    #[test]
    fn test_type_check_with_blocks() {
        check_ok(
            r#"
            fn f(x: int) -> int {
                let y: int = 0;
                {
                    let z: int = x + 1;
                    y = z * 2;
                }
                return y;
            }
        "#,
        );
    }

    #[test]
    fn test_type_array_assign() {
        check_ok("fn f(a: array<int>) -> void { a[0] = 42; }");
    }

    #[test]
    fn test_type_error_array_assign_wrong_type() {
        let err = check_err("fn f(a: array<int>) -> void { a[0] = true; }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_error_assert_non_bool() {
        let err = check_err("fn f(x: int) -> void { assert(x); }");
        assert!(matches!(err, AnalysisError::TypeError { .. }));
    }

    #[test]
    fn test_type_multiple_functions() {
        check_ok(
            r#"
            fn double(x: int) -> int { return x * 2; }
            fn triple(x: int) -> int { return x + double(x); }
        "#,
        );
    }
}
