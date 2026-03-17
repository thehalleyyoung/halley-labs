//! AST to IR lowering pass.
//!
//! Transforms the source AST into a basic-block IR suitable for analysis.
//! - Expression flattening (complex exprs -> sequences of simple assignments)
//! - Control flow linearization (if-else -> basic blocks with branches)
//! - Temporary variable generation
//! - Type annotation propagation and scope handling

use std::collections::HashMap;

use shared_types::{
    BasicBlock, Expression, Function, IrExpr, IrFunction, IrProgram, IrStatement, Program,
    QfLiaType, SsaVar, Statement, Terminator,
};

// ---------------------------------------------------------------------------
// Lowering context
// ---------------------------------------------------------------------------

/// Tracks state during IR lowering of a single function.
struct LoweringCtx {
    blocks: Vec<BasicBlock>,
    current_block: usize,
    next_block_id: usize,
    temp_counter: usize,
    scopes: Vec<HashMap<String, QfLiaType>>,
    var_types: HashMap<String, QfLiaType>,
    return_type: QfLiaType,
}

impl LoweringCtx {
    fn new(return_type: QfLiaType) -> Self {
        let entry = BasicBlock::new(0, Terminator::Unreachable).with_label("entry");
        LoweringCtx {
            blocks: vec![entry],
            current_block: 0,
            next_block_id: 1,
            temp_counter: 0,
            scopes: vec![HashMap::new()],
            var_types: HashMap::new(),
            return_type,
        }
    }

    fn fresh_temp(&mut self) -> String {
        let name = format!("__t{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn fresh_block(&mut self, label: &str) -> usize {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.push(
            BasicBlock::new(id, Terminator::Unreachable).with_label(format!("{}_{}", label, id)),
        );
        id
    }

    fn current_block_mut(&mut self) -> &mut BasicBlock {
        let bid = self.current_block;
        self.blocks.iter_mut().find(|b| b.id == bid).unwrap()
    }

    fn emit_stmt(&mut self, stmt: IrStatement) {
        let bid = self.current_block;
        self.blocks
            .iter_mut()
            .find(|b| b.id == bid)
            .unwrap()
            .statements
            .push(stmt);
    }

    fn set_terminator(&mut self, term: Terminator) {
        let bid = self.current_block;
        self.blocks
            .iter_mut()
            .find(|b| b.id == bid)
            .unwrap()
            .terminator = term;
    }

    fn switch_to(&mut self, block: usize) {
        self.current_block = block;
    }

    fn declare_var(&mut self, name: &str, ty: QfLiaType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
        self.var_types.insert(name.to_string(), ty);
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn lookup_type(&self, name: &str) -> QfLiaType {
        self.var_types.get(name).copied().unwrap_or(QfLiaType::Int)
    }

    fn emit_assign(&mut self, name: &str, value: IrExpr) {
        let ty = self.lookup_type(name);
        self.emit_stmt(IrStatement::Assign {
            target: SsaVar::base(name, ty),
            value,
        });
    }
}

// ---------------------------------------------------------------------------
// IrLowering
// ---------------------------------------------------------------------------

/// Lowers a parsed AST program to the IR representation.
pub struct IrLowering;

impl IrLowering {
    pub fn new() -> Self {
        IrLowering
    }

    /// Lower an entire program.
    pub fn lower_program(&self, program: &Program) -> shared_types::Result<IrProgram> {
        let mut functions = Vec::new();
        for func in &program.functions {
            functions.push(self.lower_function(func)?);
        }
        Ok(IrProgram::new(functions))
    }

    /// Lower a single function.
    pub fn lower_function(&self, func: &Function) -> shared_types::Result<IrFunction> {
        let mut ctx = LoweringCtx::new(func.return_type);

        let mut params = Vec::new();
        for p in &func.params {
            ctx.declare_var(&p.name, p.ty);
            params.push(SsaVar::base(p.name.clone(), p.ty));
        }

        // func.body is a single Statement; extract inner statements
        let body_stmts = Self::extract_stmts(&func.body);
        for stmt in body_stmts {
            self.lower_statement(&mut ctx, stmt)?;
        }

        // Ensure all blocks have terminators
        for block in &mut ctx.blocks {
            if matches!(block.terminator, Terminator::Unreachable) {
                block.terminator = Terminator::Return { value: None };
            }
        }

        Ok(IrFunction {
            name: func.name.clone(),
            params,
            return_type: func.return_type,
            blocks: ctx.blocks,
        })
    }

    /// Extract a slice of statements from a body Statement.
    fn extract_stmts(stmt: &Statement) -> Vec<&Statement> {
        match stmt {
            Statement::Sequence(stmts) | Statement::Block(stmts) => stmts.iter().collect(),
            other => vec![other],
        }
    }

    // -- Statements ---------------------------------------------------------

    fn lower_statement(&self, ctx: &mut LoweringCtx, stmt: &Statement) -> shared_types::Result<()> {
        match stmt {
            Statement::VarDecl { var, init, .. } => {
                ctx.declare_var(&var.name, var.ty);
                if let Some(init_expr) = init {
                    let ir_val = self.lower_expr(ctx, init_expr)?;
                    ctx.emit_assign(&var.name, ir_val);
                }
                Ok(())
            }
            Statement::Assign { target, value, .. } => {
                let ir_val = self.lower_expr(ctx, value)?;
                ctx.emit_assign(target, ir_val);
                Ok(())
            }
            Statement::IfElse {
                condition,
                then_branch,
                else_branch,
                ..
            } => self.lower_if(ctx, condition, then_branch, else_branch.as_deref()),
            Statement::Return { value, .. } => {
                let ir_val = match value {
                    Some(e) => Some(self.lower_expr(ctx, e)?),
                    None => None,
                };
                ctx.set_terminator(Terminator::Return { value: ir_val });
                let dead = ctx.fresh_block("post_return");
                ctx.switch_to(dead);
                Ok(())
            }
            Statement::Assert {
                condition, message, ..
            } => {
                let ir_cond = self.lower_expr(ctx, condition)?;
                ctx.emit_stmt(IrStatement::Assert {
                    condition: ir_cond,
                    message: message.clone(),
                });
                Ok(())
            }
            Statement::Block(stmts) => {
                ctx.push_scope();
                for s in stmts {
                    self.lower_statement(ctx, s)?;
                }
                ctx.pop_scope();
                Ok(())
            }
            Statement::Sequence(stmts) => {
                for s in stmts {
                    self.lower_statement(ctx, s)?;
                }
                Ok(())
            }
        }
    }

    fn lower_if(
        &self,
        ctx: &mut LoweringCtx,
        condition: &Expression,
        then_branch: &Statement,
        else_branch: Option<&Statement>,
    ) -> shared_types::Result<()> {
        let cond_ir = self.lower_expr(ctx, condition)?;

        let then_block = ctx.fresh_block("then");
        let else_block = ctx.fresh_block("else");
        let merge_block = ctx.fresh_block("merge");

        ctx.set_terminator(Terminator::ConditionalBranch {
            condition: cond_ir,
            true_target: then_block,
            false_target: else_block,
        });

        ctx.switch_to(then_block);
        self.lower_statement(ctx, then_branch)?;
        if matches!(ctx.current_block_mut().terminator, Terminator::Unreachable) {
            ctx.set_terminator(Terminator::Branch {
                target: merge_block,
            });
        }

        ctx.switch_to(else_block);
        if let Some(eb) = else_branch {
            self.lower_statement(ctx, eb)?;
        }
        if matches!(ctx.current_block_mut().terminator, Terminator::Unreachable) {
            ctx.set_terminator(Terminator::Branch {
                target: merge_block,
            });
        }

        ctx.switch_to(merge_block);
        Ok(())
    }

    // -- Expressions --------------------------------------------------------

    fn lower_expr(&self, ctx: &mut LoweringCtx, expr: &Expression) -> shared_types::Result<IrExpr> {
        match expr {
            Expression::IntLiteral(value) => Ok(IrExpr::Const(*value)),
            Expression::BoolLiteral(value) => Ok(IrExpr::BoolConst(*value)),
            Expression::Var(name) => {
                let ty = ctx.lookup_type(name);
                Ok(IrExpr::Var(SsaVar::base(name.clone(), ty)))
            }

            Expression::BinaryArith { op, lhs, rhs } => {
                let l = self.lower_expr_to_simple(ctx, lhs)?;
                let r = self.lower_expr_to_simple(ctx, rhs)?;
                Ok(IrExpr::BinArith {
                    op: *op,
                    lhs: Box::new(l),
                    rhs: Box::new(r),
                })
            }
            Expression::Relational { op, lhs, rhs } => {
                let l = self.lower_expr_to_simple(ctx, lhs)?;
                let r = self.lower_expr_to_simple(ctx, rhs)?;
                Ok(IrExpr::Rel {
                    op: *op,
                    lhs: Box::new(l),
                    rhs: Box::new(r),
                })
            }
            Expression::LogicalAnd(left, right) => self.lower_short_circuit_and(ctx, left, right),
            Expression::LogicalOr(left, right) => self.lower_short_circuit_or(ctx, left, right),
            Expression::LogicalNot(operand) => {
                let inner = self.lower_expr_to_simple(ctx, operand)?;
                Ok(IrExpr::Not(Box::new(inner)))
            }
            Expression::UnaryArith(operand) => {
                let inner = self.lower_expr_to_simple(ctx, operand)?;
                Ok(IrExpr::Neg(Box::new(inner)))
            }
            Expression::ArrayAccess { array, index } => {
                let arr_ir = self.lower_expr(ctx, array)?;
                let idx_ir = self.lower_expr_to_simple(ctx, index)?;
                Ok(IrExpr::Select {
                    array: Box::new(arr_ir),
                    index: Box::new(idx_ir),
                })
            }
            Expression::FunctionCall { name, args } => {
                let mut ir_args = Vec::new();
                for arg in args {
                    ir_args.push(self.lower_expr_to_simple(ctx, arg)?);
                }
                Ok(IrExpr::Call {
                    name: name.clone(),
                    args: ir_args,
                })
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let result_var = ctx.fresh_temp();
                ctx.declare_var(&result_var, QfLiaType::Int);
                let cond_ir = self.lower_expr(ctx, condition)?;

                let then_block = ctx.fresh_block("tern_then");
                let else_block = ctx.fresh_block("tern_else");
                let merge_block = ctx.fresh_block("tern_merge");

                ctx.set_terminator(Terminator::ConditionalBranch {
                    condition: cond_ir,
                    true_target: then_block,
                    false_target: else_block,
                });

                ctx.switch_to(then_block);
                let then_val = self.lower_expr(ctx, then_expr)?;
                ctx.emit_assign(&result_var, then_val);
                ctx.set_terminator(Terminator::Branch {
                    target: merge_block,
                });

                ctx.switch_to(else_block);
                let else_val = self.lower_expr(ctx, else_expr)?;
                ctx.emit_assign(&result_var, else_val);
                ctx.set_terminator(Terminator::Branch {
                    target: merge_block,
                });

                ctx.switch_to(merge_block);
                let ty = ctx.lookup_type(&result_var);
                Ok(IrExpr::Var(SsaVar::base(result_var, ty)))
            }
        }
    }

    /// Lower an expression to a simple form (variable or constant).
    fn lower_expr_to_simple(
        &self,
        ctx: &mut LoweringCtx,
        expr: &Expression,
    ) -> shared_types::Result<IrExpr> {
        let ir = self.lower_expr(ctx, expr)?;
        match &ir {
            IrExpr::Var(_) | IrExpr::Const(_) | IrExpr::BoolConst(_) => Ok(ir),
            _ => {
                let tmp = ctx.fresh_temp();
                ctx.declare_var(&tmp, QfLiaType::Int);
                ctx.emit_assign(&tmp, ir);
                let ty = ctx.lookup_type(&tmp);
                Ok(IrExpr::Var(SsaVar::base(tmp, ty)))
            }
        }
    }

    fn lower_short_circuit_and(
        &self,
        ctx: &mut LoweringCtx,
        left: &Expression,
        right: &Expression,
    ) -> shared_types::Result<IrExpr> {
        let result = ctx.fresh_temp();
        ctx.declare_var(&result, QfLiaType::Boolean);
        let lhs = self.lower_expr(ctx, left)?;

        let right_block = ctx.fresh_block("and_rhs");
        let merge_block = ctx.fresh_block("and_merge");

        ctx.emit_assign(&result, IrExpr::BoolConst(false));
        ctx.set_terminator(Terminator::ConditionalBranch {
            condition: lhs,
            true_target: right_block,
            false_target: merge_block,
        });

        ctx.switch_to(right_block);
        let rhs = self.lower_expr(ctx, right)?;
        ctx.emit_assign(&result, rhs);
        ctx.set_terminator(Terminator::Branch {
            target: merge_block,
        });

        ctx.switch_to(merge_block);
        let ty = ctx.lookup_type(&result);
        Ok(IrExpr::Var(SsaVar::base(result, ty)))
    }

    fn lower_short_circuit_or(
        &self,
        ctx: &mut LoweringCtx,
        left: &Expression,
        right: &Expression,
    ) -> shared_types::Result<IrExpr> {
        let result = ctx.fresh_temp();
        ctx.declare_var(&result, QfLiaType::Boolean);
        let lhs = self.lower_expr(ctx, left)?;

        let right_block = ctx.fresh_block("or_rhs");
        let merge_block = ctx.fresh_block("or_merge");

        ctx.emit_assign(&result, IrExpr::BoolConst(true));
        ctx.set_terminator(Terminator::ConditionalBranch {
            condition: lhs,
            true_target: merge_block,
            false_target: right_block,
        });

        ctx.switch_to(right_block);
        let rhs = self.lower_expr(ctx, right)?;
        ctx.emit_assign(&result, rhs);
        ctx.set_terminator(Terminator::Branch {
            target: merge_block,
        });

        ctx.switch_to(merge_block);
        let ty = ctx.lookup_type(&result);
        Ok(IrExpr::Var(SsaVar::base(result, ty)))
    }
}

impl Default for IrLowering {
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
    use shared_types::ArithOp;

    fn lower(src: &str) -> IrProgram {
        let prog = Parser::parse_source(src).unwrap();
        IrLowering::new().lower_program(&prog).unwrap()
    }

    fn lower_fn(src: &str) -> IrFunction {
        lower(src).functions.into_iter().next().unwrap()
    }

    #[test]
    fn test_lower_empty_function() {
        let f = lower_fn("fn f() -> void { }");
        assert_eq!(f.name, "f");
        assert_eq!(f.blocks.len(), 1);
    }

    #[test]
    fn test_lower_return_const() {
        let f = lower_fn("fn f() -> int { return 42; }");
        assert!(matches!(
            &f.blocks[0].terminator,
            Terminator::Return {
                value: Some(IrExpr::Const(42))
            }
        ));
    }

    #[test]
    fn test_lower_var_decl_and_return() {
        let f = lower_fn("fn f() -> int { let x: int = 5; return x; }");
        assert_eq!(f.blocks[0].statements.len(), 1);
        match &f.blocks[0].statements[0] {
            IrStatement::Assign { target, value } => {
                assert_eq!(target.name, "x");
                assert_eq!(*value, IrExpr::Const(5));
            }
            _ => panic!("expected assign"),
        }
    }

    #[test]
    fn test_lower_arithmetic() {
        let f = lower_fn("fn f(x: int, y: int) -> int { return x + y; }");
        match &f.blocks[0].terminator {
            Terminator::Return {
                value:
                    Some(IrExpr::BinArith {
                        op: ArithOp::Add, ..
                    }),
            } => {}
            other => panic!("expected return of add, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_if_else() {
        let f = lower_fn("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        assert!(f.blocks.len() >= 3);
        assert!(matches!(
            &f.blocks[0].terminator,
            Terminator::ConditionalBranch { .. }
        ));
    }

    #[test]
    fn test_lower_nested_if() {
        let f = lower_fn(
            r#"
            fn f(x: int) -> int {
                if (x > 0) { return 1; }
                else if (x < 0) { return -1; }
                else { return 0; }
            }
        "#,
        );
        let branches = f
            .blocks
            .iter()
            .filter(|b| matches!(&b.terminator, Terminator::ConditionalBranch { .. }))
            .count();
        assert!(branches >= 2);
    }

    #[test]
    fn test_lower_assert() {
        let f = lower_fn("fn f(x: int) -> void { assert(x > 0); }");
        assert!(f.blocks[0]
            .statements
            .iter()
            .any(|s| matches!(s, IrStatement::Assert { .. })));
    }

    #[test]
    fn test_lower_array_access() {
        let f = lower_fn("fn f(a: array<int>) -> int { return a[0]; }");
        // Array access should produce a Select in the IR
        assert!(!f.blocks.is_empty());
    }

    #[test]
    fn test_lower_function_call() {
        let f = lower_fn("fn f(x: int) -> int { return add(x, 1); }");
        match &f.blocks[0].terminator {
            Terminator::Return {
                value: Some(IrExpr::Call { name, args }),
            } => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected return of call"),
        }
    }

    #[test]
    fn test_lower_complex_expr_flattening() {
        let f = lower_fn("fn f(x: int, y: int) -> int { return (x + y) * (x - y); }");
        assert!(!f.blocks[0].statements.is_empty());
    }

    #[test]
    fn test_lower_multiple_assignments() {
        let f =
            lower_fn("fn f(x: int) -> int { let y: int = x + 1; let z: int = y * 2; return z; }");
        assert!(f.blocks[0].statements.len() >= 2);
    }

    #[test]
    fn test_lower_block_scoping() {
        let f =
            lower_fn("fn f() -> int { let x: int = 1; { let y: int = 2; x = x + y; } return x; }");
        assert!(!f.blocks.is_empty());
    }

    #[test]
    fn test_lower_ternary() {
        let f = lower_fn("fn f(x: int) -> int { return x > 0 ? x : 0; }");
        assert!(f.blocks.len() >= 3);
    }

    #[test]
    fn test_lower_whole_program() {
        let ir = lower("fn abs(x: int) -> int { if (x >= 0) { return x; } else { return -x; } } fn max(a: int, b: int) -> int { if (a >= b) { return a; } else { return b; } }");
        assert_eq!(ir.functions.len(), 2);
    }

    #[test]
    fn test_entry_block_zero() {
        let f = lower_fn("fn f() -> void { }");
        assert_eq!(f.blocks[0].id, 0);
    }

    #[test]
    fn test_all_blocks_terminated() {
        let f = lower_fn("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
        for b in &f.blocks {
            assert!(
                !matches!(b.terminator, Terminator::Unreachable),
                "block {:?} unreachable",
                b.label
            );
        }
    }

    #[test]
    fn test_lower_params() {
        let f = lower_fn("fn f(a: int, b: bool) -> int { return a; }");
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.params[0].name, "a");
        assert_eq!(f.params[0].ty, QfLiaType::Int);
        assert_eq!(f.params[1].name, "b");
        assert_eq!(f.params[1].ty, QfLiaType::Boolean);
    }

    #[test]
    fn test_lower_short_circuit_and() {
        let f = lower_fn("fn f(a: bool, b: bool) -> bool { let r: bool = a && b; return r; }");
        assert!(f.blocks.len() >= 3);
    }

    #[test]
    fn test_lower_short_circuit_or() {
        let f = lower_fn("fn f(a: bool, b: bool) -> bool { let r: bool = a || b; return r; }");
        assert!(f.blocks.len() >= 3);
    }
}
