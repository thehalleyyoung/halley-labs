use crate::ast::*;
use crate::error::TypeError;
use crate::source_map::Span;
use crate::types::{DslType, ObligationTypeKind, TypeEnv};


/// Type checker for the regulatory DSL.
pub struct TypeChecker {
    env: TypeEnv,
    errors: Vec<TypeError>,
    /// Track which jurisdiction scope we're currently inside.
    current_jurisdiction: Option<String>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            errors: Vec::new(),
            current_jurisdiction: None,
        }
    }

    pub fn with_env(env: TypeEnv) -> Self {
        Self {
            env,
            errors: Vec::new(),
            current_jurisdiction: None,
        }
    }

    /// Type check a complete program. Returns the type environment and errors.
    pub fn check_program(&mut self, program: &Program) -> Vec<TypeError> {
        // First pass: register all jurisdiction names
        for decl in &program.declarations {
            self.register_declarations(decl);
        }
        // Second pass: type check
        for decl in &program.declarations {
            self.check_declaration(decl);
        }
        std::mem::take(&mut self.errors)
    }

    /// Pre-register declarations for forward references.
    fn register_declarations(&mut self, decl: &Declaration) {
        match &decl.kind {
            DeclarationKind::Jurisdiction(j) => {
                self.env
                    .register_jurisdiction(j.name.clone(), j.parent.clone());
                for child in &j.body {
                    self.register_declarations(child);
                }
            }
            DeclarationKind::Obligation(o) => {
                let jur = o.jurisdiction.clone().or_else(|| self.current_jurisdiction.clone());
                self.env.bind(
                    o.name.clone(),
                    DslType::ObligationType {
                        kind: ObligationTypeKind::Obligation,
                        jurisdiction: jur,
                    },
                );
            }
            DeclarationKind::Permission(p) => {
                let jur = p.jurisdiction.clone().or_else(|| self.current_jurisdiction.clone());
                self.env.bind(
                    p.name.clone(),
                    DslType::ObligationType {
                        kind: ObligationTypeKind::Permission,
                        jurisdiction: jur,
                    },
                );
            }
            DeclarationKind::Prohibition(p) => {
                let jur = p.jurisdiction.clone().or_else(|| self.current_jurisdiction.clone());
                self.env.bind(
                    p.name.clone(),
                    DslType::ObligationType {
                        kind: ObligationTypeKind::Prohibition,
                        jurisdiction: jur,
                    },
                );
            }
            DeclarationKind::Framework(f) => {
                for child in &f.body {
                    self.register_declarations(child);
                }
            }
            DeclarationKind::Strategy(s) => {
                self.env.bind(s.name.clone(), DslType::Strategy);
            }
            DeclarationKind::Cost(c) => {
                self.env.bind(c.name.clone(), DslType::Cost);
            }
            DeclarationKind::Temporal(t) => {
                self.env.bind(t.name.clone(), DslType::Temporal);
            }
            DeclarationKind::Mapping(m) => {
                self.env.bind(
                    m.name.clone(),
                    DslType::Function {
                        params: vec![DslType::Str],
                        ret: Box::new(DslType::Str),
                    },
                );
            }
        }
    }

    /// Check a single declaration.
    pub fn check_declaration(&mut self, decl: &Declaration) {
        match &decl.kind {
            DeclarationKind::Jurisdiction(j) => self.check_jurisdiction(j),
            DeclarationKind::Obligation(o) => self.check_obligation(o, ObligationTypeKind::Obligation),
            DeclarationKind::Permission(p) => {
                self.check_obligation_like(&p.name, &p.jurisdiction, &p.body, p.span, ObligationTypeKind::Permission);
            }
            DeclarationKind::Prohibition(p) => {
                self.check_obligation_like(&p.name, &p.jurisdiction, &p.body, p.span, ObligationTypeKind::Prohibition);
            }
            DeclarationKind::Framework(f) => self.check_framework(f),
            DeclarationKind::Strategy(s) => self.check_strategy(s),
            DeclarationKind::Cost(c) => self.check_cost(c),
            DeclarationKind::Temporal(t) => self.check_temporal(t),
            DeclarationKind::Mapping(m) => self.check_mapping(m),
        }
    }

    fn check_jurisdiction(&mut self, j: &JurisdictionDecl) {
        // Check parent exists if specified
        if let Some(parent) = &j.parent {
            if !self.env.has_jurisdiction(parent) {
                self.errors.push(TypeError::undefined_jurisdiction(
                    j.span,
                    parent,
                ));
            }
        }

        // Check if already defined in current scope
        if self.env.is_bound_in_current_scope(&j.name) {
            self.errors
                .push(TypeError::duplicate_definition(j.span, &j.name));
        }

        let prev_jur = self.current_jurisdiction.take();
        self.current_jurisdiction = Some(j.name.clone());
        self.env.push_scope();

        for child in &j.body {
            self.check_declaration(child);
        }

        self.env.pop_scope();
        self.current_jurisdiction = prev_jur;
    }

    fn check_obligation(&mut self, o: &ObligationDecl, kind: ObligationTypeKind) {
        self.check_obligation_like(&o.name, &o.jurisdiction, &o.body, o.span, kind);
    }

    fn check_obligation_like(
        &mut self,
        name: &str,
        jurisdiction: &Option<String>,
        body: &ObligationBody,
        span: Span,
        kind: ObligationTypeKind,
    ) {
        // Validate jurisdiction if specified
        if let Some(j) = jurisdiction {
            if !self.env.has_jurisdiction(j) && self.current_jurisdiction.as_deref() != Some(j) {
                self.errors
                    .push(TypeError::undefined_jurisdiction(span, j));
            }
        }

        self.check_obligation_body(body);

        // Bind the name
        let jur = jurisdiction
            .clone()
            .or_else(|| self.current_jurisdiction.clone());
        self.env.bind(
            name.to_string(),
            DslType::ObligationType {
                kind,
                jurisdiction: jur,
            },
        );
    }

    fn check_framework(&mut self, f: &FrameworkDecl) {
        let prev_jur = self.current_jurisdiction.take();
        self.current_jurisdiction = f.jurisdiction.clone().or(prev_jur.clone());
        self.env.push_scope();
        for child in &f.body {
            self.check_declaration(child);
        }
        self.env.pop_scope();
        self.current_jurisdiction = prev_jur;
    }

    fn check_strategy(&mut self, s: &StrategyDecl) {
        for item in &s.body {
            self.check_expression(&item.value);
        }
    }

    fn check_cost(&mut self, c: &CostDecl) {
        let ty = self.check_expression(&c.amount);
        if !ty.is_numeric() && ty != DslType::Error {
            self.errors.push(TypeError::type_mismatch(
                c.span,
                "numeric",
                &ty.to_string(),
            ));
        }
    }

    fn check_temporal(&mut self, t: &TemporalDecl) {
        self.check_temporal_consistency(t.start.as_deref(), t.end.as_deref(), t.span);
    }

    fn check_mapping(&mut self, m: &MappingDecl) {
        for entry in &m.entries {
            self.check_expression(&entry.from);
            self.check_expression(&entry.to);
        }
    }

    pub fn check_obligation_body(&mut self, body: &ObligationBody) {
        // Check conditions
        for cond in &body.conditions {
            let ty = self.check_expression(cond);
            if ty != DslType::Bool && ty != DslType::Error {
                self.errors.push(TypeError::type_mismatch(
                    cond.span,
                    "Bool",
                    &ty.to_string(),
                ));
            }
        }

        // Check exemptions
        for exempt in &body.exemptions {
            let ty = self.check_expression(exempt);
            if ty != DslType::Bool && ty != DslType::Error {
                self.errors.push(TypeError::type_mismatch(
                    exempt.span,
                    "Bool",
                    &ty.to_string(),
                ));
            }
        }

        // Validate formalizability grade
        if let Some(grade) = &body.formalizability {
            if grade.as_u8() < 1 || grade.as_u8() > 5 {
                self.errors.push(TypeError::new(
                    body.span,
                    format!("invalid formalizability grade: {}", grade),
                ));
            }
        }

        // Validate temporal consistency
        self.check_temporal_consistency(
            body.temporal_start.as_deref(),
            body.temporal_end.as_deref(),
            body.span,
        );

        // Check compositions reference valid obligations
        for comp in &body.compositions {
            if self.env.lookup(&comp.operand).is_none() {
                self.errors.push(TypeError::new(
                    comp.span,
                    format!("composition references undefined obligation '{}'", comp.operand),
                ));
            } else if let Some(ty) = self.env.lookup(&comp.operand) {
                if !ty.is_obligation() {
                    self.errors.push(TypeError::invalid_composition(
                        comp.span,
                        &format!("'{}' is not an obligation type", comp.operand),
                    ));
                }
            }
        }

        // Check extra fields
        for field in &body.extra_fields {
            self.check_expression(&field.value);
        }
    }

    fn check_temporal_consistency(
        &mut self,
        start: Option<&str>,
        end: Option<&str>,
        span: Span,
    ) {
        if let (Some(s), Some(e)) = (start, end) {
            if let (Ok(sd), Ok(ed)) = (
                chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d"),
                chrono::NaiveDate::parse_from_str(e, "%Y-%m-%d"),
            ) {
                if sd > ed {
                    self.errors.push(TypeError::new(
                        span,
                        format!("temporal start date {} is after end date {}", s, e),
                    ));
                }
            }
        }
    }

    /// Type check an expression and return its type.
    pub fn check_expression(&mut self, expr: &Expression) -> DslType {
        match &expr.kind {
            ExpressionKind::Literal(lit) => self.check_literal(lit),
            ExpressionKind::Variable(name) => {
                match self.env.lookup(name) {
                    Some(ty) => ty.clone(),
                    None => {
                        self.errors
                            .push(TypeError::undefined_variable(expr.span, name));
                        DslType::Error
                    }
                }
            }
            ExpressionKind::BinaryOp { op, left, right } => {
                let lt = self.check_expression(left);
                let rt = self.check_expression(right);
                self.check_binary_op(*op, &lt, &rt, expr.span)
            }
            ExpressionKind::UnaryOp { op, operand } => {
                let t = self.check_expression(operand);
                self.check_unary_op(*op, &t, expr.span)
            }
            ExpressionKind::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let ct = self.check_expression(condition);
                if ct != DslType::Bool && ct != DslType::Error {
                    self.errors.push(TypeError::type_mismatch(
                        condition.span,
                        "Bool",
                        &ct.to_string(),
                    ));
                }
                let tt = self.check_expression(then_branch);
                let et = self.check_expression(else_branch);
                match self.env.type_join(&tt, &et) {
                    Some(joined) => joined,
                    None => {
                        self.errors.push(TypeError::type_mismatch(
                            expr.span,
                            &tt.to_string(),
                            &et.to_string(),
                        ));
                        DslType::Error
                    }
                }
            }
            ExpressionKind::Quantifier {
                kind: _,
                variable,
                domain,
                body,
            } => {
                self.env.push_scope();
                let var_type = if let Some(d) = domain {
                    self.check_expression(d)
                } else {
                    self.env.fresh_type_var()
                };
                self.env.bind(variable.clone(), var_type);
                let body_type = self.check_expression(body);
                self.env.pop_scope();
                if body_type != DslType::Bool && body_type != DslType::Error {
                    self.errors.push(TypeError::type_mismatch(
                        body.span,
                        "Bool",
                        &body_type.to_string(),
                    ));
                }
                DslType::Bool
            }
            ExpressionKind::FunctionCall { function, args } => {
                let fn_type = match self.env.lookup(function) {
                    Some(ty) => ty.clone(),
                    None => {
                        self.errors
                            .push(TypeError::undefined_variable(expr.span, function));
                        return DslType::Error;
                    }
                };
                match fn_type {
                    DslType::Function { params, ret } => {
                        if args.len() != params.len() {
                            self.errors.push(TypeError::new(
                                expr.span,
                                format!(
                                    "function '{}' expects {} arguments, got {}",
                                    function,
                                    params.len(),
                                    args.len()
                                ),
                            ));
                        }
                        for (arg, param) in args.iter().zip(params.iter()) {
                            let at = self.check_expression(arg);
                            if !self.env.is_subtype(&at, param) && at != DslType::Error {
                                self.errors.push(TypeError::type_mismatch(
                                    arg.span,
                                    &param.to_string(),
                                    &at.to_string(),
                                ));
                            }
                        }
                        *ret
                    }
                    _ => {
                        self.errors.push(TypeError::new(
                            expr.span,
                            format!("'{}' is not a function", function),
                        ));
                        DslType::Error
                    }
                }
            }
            ExpressionKind::ArticleRef { .. } => DslType::ArticleRef,
            ExpressionKind::FieldAccess { object, field } => {
                let obj_ty = self.check_expression(object);
                self.check_field_access(&obj_ty, field, expr.span)
            }
            ExpressionKind::Composition { op, left, right } => {
                let lt = self.check_expression(left);
                let rt = self.check_expression(right);
                self.check_composition(*op, &lt, &rt, expr.span)
            }
        }
    }

    fn check_literal(&self, lit: &Literal) -> DslType {
        match lit {
            Literal::Bool(_) => DslType::Bool,
            Literal::Int(_) => DslType::Int,
            Literal::Float(_) => DslType::Float,
            Literal::Str(_) => DslType::Str,
            Literal::Date(_) => DslType::Temporal,
        }
    }

    fn check_binary_op(&mut self, op: BinOp, lt: &DslType, rt: &DslType, span: Span) -> DslType {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                match lt.arithmetic_result(rt) {
                    Some(t) => t,
                    None => {
                        if *lt != DslType::Error && *rt != DslType::Error {
                            self.errors.push(TypeError::type_mismatch(
                                span,
                                "numeric types",
                                &format!("{} and {}", lt, rt),
                            ));
                        }
                        DslType::Error
                    }
                }
            }
            BinOp::Eq | BinOp::NotEq => {
                if lt != rt && !self.env.is_subtype(lt, rt) && !self.env.is_subtype(rt, lt) {
                    if *lt != DslType::Error && *rt != DslType::Error {
                        self.errors.push(TypeError::type_mismatch(
                            span,
                            &lt.to_string(),
                            &rt.to_string(),
                        ));
                    }
                }
                DslType::Bool
            }
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                if !lt.is_comparable() || !rt.is_comparable() {
                    if *lt != DslType::Error && *rt != DslType::Error {
                        self.errors.push(TypeError::type_mismatch(
                            span,
                            "comparable types",
                            &format!("{} and {}", lt, rt),
                        ));
                    }
                }
                DslType::Bool
            }
            BinOp::And | BinOp::Or | BinOp::Implies => {
                if *lt != DslType::Bool && *lt != DslType::Error {
                    self.errors
                        .push(TypeError::type_mismatch(span, "Bool", &lt.to_string()));
                }
                if *rt != DslType::Bool && *rt != DslType::Error {
                    self.errors
                        .push(TypeError::type_mismatch(span, "Bool", &rt.to_string()));
                }
                DslType::Bool
            }
        }
    }

    fn check_unary_op(&mut self, op: UnOp, t: &DslType, span: Span) -> DslType {
        match op {
            UnOp::Not => {
                if *t != DslType::Bool && *t != DslType::Error {
                    self.errors
                        .push(TypeError::type_mismatch(span, "Bool", &t.to_string()));
                }
                DslType::Bool
            }
            UnOp::Neg => {
                if !t.is_numeric() && *t != DslType::Error {
                    self.errors
                        .push(TypeError::type_mismatch(span, "numeric", &t.to_string()));
                }
                t.clone()
            }
        }
    }

    fn check_field_access(&mut self, obj_ty: &DslType, field: &str, span: Span) -> DslType {
        match obj_ty {
            DslType::ObligationType { .. } => {
                // Obligation types have known fields
                match field {
                    "risk_level" => DslType::RiskLevel,
                    "formalizability" => DslType::FormalizabilityGrade,
                    "domain" => DslType::Domain,
                    "temporal" => DslType::Temporal,
                    "jurisdiction" => DslType::Str,
                    "conditions" => DslType::Set(Box::new(DslType::Bool)),
                    _ => {
                        self.errors.push(TypeError::new(
                            span,
                            format!("unknown field '{}' on obligation type", field),
                        ));
                        DslType::Error
                    }
                }
            }
            DslType::Strategy => match field {
                "approach" | "name" => DslType::Str,
                "threshold" => DslType::Float,
                _ => {
                    // Strategies have dynamic fields
                    self.env.fresh_type_var()
                }
            },
            DslType::Error => DslType::Error,
            _ => {
                self.errors.push(TypeError::new(
                    span,
                    format!("cannot access field '{}' on type {}", field, obj_ty),
                ));
                DslType::Error
            }
        }
    }

    fn check_composition(
        &mut self,
        op: regsynth_types::CompositionOp,
        lt: &DslType,
        rt: &DslType,
        span: Span,
    ) -> DslType {
        // Both operands must be obligation types
        match (lt, rt) {
            (
                DslType::ObligationType {
                    kind: k1,
                    jurisdiction: j1,
                },
                DslType::ObligationType {
                    kind: _k2,
                    jurisdiction: j2,
                },
            ) => {
                // For override (▷), the result takes the left jurisdiction
                let result_jur = match op {
                    regsynth_types::CompositionOp::Override => j1.clone(),
                    _ => {
                        // For conjunction/disjunction/exception, jurisdictions should be compatible
                        match (j1, j2) {
                            (Some(a), Some(b)) if a == b => Some(a.clone()),
                            (Some(a), Some(b)) => {
                                // Check if one is a parent of the other
                                if self.env.is_subtype(lt, rt) {
                                    j2.clone()
                                } else if self.env.is_subtype(rt, lt) {
                                    j1.clone()
                                } else {
                                    self.errors.push(TypeError::invalid_composition(
                                        span,
                                        &format!(
                                            "incompatible jurisdictions '{}' and '{}'",
                                            a, b
                                        ),
                                    ));
                                    j1.clone()
                                }
                            }
                            (j, None) | (None, j) => j.clone(),
                        }
                    }
                };

                DslType::ObligationType {
                    kind: *k1,
                    jurisdiction: result_jur,
                }
            }
            (DslType::Error, _) | (_, DslType::Error) => DslType::Error,
            _ => {
                self.errors.push(TypeError::invalid_composition(
                    span,
                    &format!(
                        "expected obligation types, got {} and {}",
                        lt, rt
                    ),
                ));
                DslType::Error
            }
        }
    }

    /// Get the current type environment.
    pub fn env(&self) -> &TypeEnv {
        &self.env
    }

    /// Get mutable access to the type environment.
    pub fn env_mut(&mut self) -> &mut TypeEnv {
        &mut self.env
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn check_src(src: &str) -> Vec<TypeError> {
        let (tokens, lex_errors) = lex(src);
        assert!(lex_errors.is_empty(), "lex errors: {:?}", lex_errors);
        let (program, parse_errors) = parse(tokens);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let mut checker = TypeChecker::new();
        checker.check_program(&program)
    }

    #[test]
    fn test_valid_obligation() {
        let errors = check_src(r#"
            jurisdiction "EU" { }
            obligation transparency: "EU" {
                risk_level: high;
                formalizability: 2;
                requires: true;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_condition_must_be_bool() {
        let errors = check_src(r#"
            obligation test {
                requires: 42;
            }
        "#);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("type mismatch"));
    }

    #[test]
    fn test_undefined_variable() {
        let errors = check_src(r#"
            obligation test {
                requires: undefined_var;
            }
        "#);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("undefined variable"));
    }

    #[test]
    fn test_temporal_consistency() {
        let errors = check_src(r#"
            obligation test {
                temporal: #2025-12-31 -> #2024-01-01;
            }
        "#);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("after end date"));
    }

    #[test]
    fn test_composition_type_check() {
        let errors = check_src(r#"
            jurisdiction "EU" { }
            obligation obl_a: "EU" {
                requires: true;
            }
            obligation obl_b: "EU" {
                requires: true;
                compose: ⊗ obl_a;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_composition_undefined_ref() {
        let errors = check_src(r#"
            obligation test {
                compose: ⊗ nonexistent;
            }
        "#);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_nested_jurisdiction_scope() {
        let errors = check_src(r#"
            jurisdiction "EU" {
                obligation inner_obl {
                    requires: true;
                }
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_if_then_else_type_check() {
        let errors = check_src(r#"
            obligation test {
                requires: if true then true else false;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_binary_op_type_check() {
        let errors = check_src(r#"
            obligation test {
                requires: true and false;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_permission_and_prohibition() {
        let errors = check_src(r#"
            permission data_use {
                requires: true;
            }
            prohibition scoring {
                requires: true;
            }
        "#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }
}
