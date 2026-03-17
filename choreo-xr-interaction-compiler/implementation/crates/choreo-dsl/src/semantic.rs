//! Semantic analysis pass for the Choreo DSL.
//!
//! Performs name resolution, scope checking, duplicate/undefined/unused
//! detection, cycle detection among region parents, and pattern/action
//! validity verification.

use crate::ast::*;
use choreo_types::{Diagnostic, Severity, Span};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Symbol kinds and info
// ---------------------------------------------------------------------------

/// The kind of symbol a name resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Region,
    Entity,
    Zone,
    Interaction,
    Scene,
    LetBinding,
    Parameter,
    Import,
    Timer,
}

/// The scope kind where a symbol was declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Global,
    Scene,
    Interaction,
    Zone,
}

/// Information recorded for every symbol.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub kind: SymbolKind,
    pub scope: ScopeKind,
    pub span: Span,
    pub used: bool,
}

// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Scope {
    kind: ScopeKind,
    symbols: HashMap<String, SymbolInfo>,
}

impl Scope {
    fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
            symbols: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SemanticAnalyzer
// ---------------------------------------------------------------------------

/// Performs semantic analysis on a Choreo AST.
pub struct SemanticAnalyzer {
    scopes: Vec<Scope>,
    diagnostics: Vec<Diagnostic>,
    /// All names referenced (name -> first span).
    references: Vec<(String, Span)>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new(ScopeKind::Global)],
            diagnostics: Vec::new(),
            references: Vec::new(),
        }
    }

    /// Run full semantic analysis and return diagnostics.
    pub fn analyze(&mut self, program: &Program) -> Vec<Diagnostic> {
        self.build_symbol_table(program);
        self.check_duplicate_definitions(program);
        self.check_undefined_references(program);
        self.check_unused_definitions();
        self.check_cycle_detection(program);
        self.check_all_patterns(program);
        self.check_all_actions(program);
        self.diagnostics.clone()
    }

    // -- scope helpers --

    fn push_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(Scope::new(kind));
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn current_scope_kind(&self) -> ScopeKind {
        self.scopes.last().map_or(ScopeKind::Global, |s| s.kind)
    }

    fn define(&mut self, name: &str, kind: SymbolKind, span: &Span) {
        let scope_kind = self.current_scope_kind();
        let info = SymbolInfo {
            kind,
            scope: scope_kind,
            span: span.clone(),
            used: false,
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope.symbols.insert(name.to_string(), info);
        }
    }

    fn lookup(&self, name: &str) -> Option<&SymbolInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.symbols.get(name) {
                return Some(info);
            }
        }
        None
    }

    fn mark_used(&mut self, name: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(info) = scope.symbols.get_mut(name) {
                info.used = true;
                return;
            }
        }
    }

    // -- symbol table building --

    fn build_symbol_table(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Region(r) => self.define(&r.name, SymbolKind::Region, &r.span),
                Declaration::Entity(e) => self.define(&e.name, SymbolKind::Entity, &e.span),
                Declaration::Zone(z) => self.define(&z.name, SymbolKind::Zone, &z.span),
                Declaration::Interaction(i) => {
                    self.define(&i.name, SymbolKind::Interaction, &i.span)
                }
                Declaration::Scene(s) => self.define(&s.name, SymbolKind::Scene, &s.span),
                Declaration::LetBinding(l) => self.define(&l.name, SymbolKind::LetBinding, &l.span),
                Declaration::Import(imp) => {
                    if let Some(items) = &imp.items {
                        for item in items {
                            self.define(item, SymbolKind::Import, &imp.span);
                        }
                    }
                }
            }
        }
    }

    // -- duplicate definitions --

    fn check_duplicate_definitions(&mut self, program: &Program) {
        let mut seen: HashMap<String, Span> = HashMap::new();
        for decl in &program.declarations {
            let (name, span) = match decl {
                Declaration::Region(r) => (r.name.clone(), r.span.clone()),
                Declaration::Entity(e) => (e.name.clone(), e.span.clone()),
                Declaration::Zone(z) => (z.name.clone(), z.span.clone()),
                Declaration::Interaction(i) => (i.name.clone(), i.span.clone()),
                Declaration::Scene(s) => (s.name.clone(), s.span.clone()),
                Declaration::LetBinding(l) => (l.name.clone(), l.span.clone()),
                Declaration::Import(_) => continue,
            };
            if let Some(prev_span) = seen.get(&name) {
                self.diagnostics.push(Diagnostic {
                    severity: Severity::Error,
                    message: format!(
                        "duplicate definition of '{}' (previously defined at offset {})",
                        name, prev_span.start
                    ),
                    span: Some(span),
                });
            } else {
                seen.insert(name, span);
            }
        }

        // Check for duplicates within scene entity/region/interaction refs
        for decl in &program.declarations {
            if let Declaration::Scene(scene) = decl {
                self.check_scene_duplicates(scene);
            }
        }
    }

    fn check_scene_duplicates(&mut self, scene: &SceneDecl) {
        let mut seen: HashSet<String> = HashSet::new();
        for eref in &scene.entities {
            let name = match eref {
                EntityRef::Named(n, _) => n.clone(),
                EntityRef::Inline(e) => e.name.clone(),
            };
            if !seen.insert(name.clone()) {
                self.diagnostics.push(Diagnostic {
                    severity: Severity::Error,
                    message: format!(
                        "duplicate entity reference '{}' in scene '{}'",
                        name, scene.name
                    ),
                    span: Some(scene.span.clone()),
                });
            }
        }
    }

    // -- undefined references --

    fn check_undefined_references(&mut self, program: &Program) {
        self.references.clear();
        self.collect_references(program);

        let mut undefined = Vec::new();
        let mut used = Vec::new();

        for (name, span) in &self.references {
            if self.lookup(name).is_none() {
                undefined.push((name.clone(), span.clone()));
            } else {
                used.push(name.clone());
            }
        }

        for (name, span) in undefined {
            self.diagnostics.push(Diagnostic {
                severity: Severity::Error,
                message: format!("undefined reference to '{}'", name),
                span: Some(span),
            });
        }

        for name in &used {
            self.mark_used(name);
        }
    }

    fn collect_references(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Region(r) => {
                    if let Some(parent) = &r.parent {
                        self.references.push((parent.clone(), r.span.clone()));
                    }
                    self.collect_geometry_refs(&r.geometry);
                    for c in &r.constraints {
                        self.collect_constraint_refs(c);
                    }
                }
                Declaration::Entity(e) => {
                    if let Some(pos) = &e.initial_position {
                        self.collect_expr_refs(pos);
                    }
                    if let Some(rot) = &e.initial_rotation {
                        self.collect_expr_refs(rot);
                    }
                    if let Some(bv) = &e.bounding_volume {
                        self.collect_geometry_refs(bv);
                    }
                    for (_, expr) in &e.properties {
                        self.collect_expr_refs(expr);
                    }
                }
                Declaration::Zone(z) => {
                    for region_name in &z.regions {
                        self.references.push((region_name.clone(), z.span.clone()));
                    }
                    for rule in &z.rules {
                        self.collect_pattern_refs(&rule.pattern);
                        for action in &rule.actions {
                            self.collect_action_refs(action);
                        }
                    }
                }
                Declaration::Interaction(i) => {
                    self.push_scope(ScopeKind::Interaction);
                    for param in &i.parameters {
                        self.define(&param.name, SymbolKind::Parameter, &param.span);
                    }
                    if let Some(pat) = &i.pattern {
                        self.collect_pattern_refs(pat);
                    }
                    match &i.body {
                        InteractionBody::Simple {
                            guard, actions, ..
                        } => {
                            self.collect_pattern_refs(guard);
                            for action in actions {
                                self.collect_action_refs(action);
                            }
                        }
                        InteractionBody::Choreography(choreo) => {
                            self.collect_choreography_refs(choreo);
                        }
                    }
                    self.pop_scope();
                }
                Declaration::Scene(s) => {
                    for eref in &s.entities {
                        if let EntityRef::Named(name, span) = eref {
                            self.references.push((name.clone(), span.clone()));
                        }
                    }
                    for rref in &s.regions {
                        if let RegionRef::Named(name, span) = rref {
                            self.references.push((name.clone(), span.clone()));
                        }
                    }
                    for iref in &s.interactions {
                        if let InteractionRef::Named(name, args, span) = iref {
                            self.references.push((name.clone(), span.clone()));
                            for arg in args {
                                self.collect_expr_refs(arg);
                            }
                        }
                    }
                    for c in &s.constraints {
                        self.collect_constraint_refs(c);
                    }
                }
                Declaration::LetBinding(l) => {
                    self.collect_expr_refs(&l.value);
                }
                Declaration::Import(_) => {}
            }
        }
    }

    fn collect_expr_refs(&mut self, expr: &Expr) {
        match expr {
            Expr::Identifier(name, span) => {
                self.references.push((name.clone(), span.clone()));
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_expr_refs(left);
                self.collect_expr_refs(right);
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_expr_refs(operand);
            }
            Expr::FunctionCall {
                function, args, ..
            } => {
                self.collect_expr_refs(function);
                for arg in args {
                    self.collect_expr_refs(arg);
                }
            }
            Expr::FieldAccess { object, .. } => {
                self.collect_expr_refs(object);
            }
            Expr::Index { object, index, .. } => {
                self.collect_expr_refs(object);
                self.collect_expr_refs(index);
            }
            Expr::Tuple { elements, .. } => {
                for e in elements {
                    self.collect_expr_refs(e);
                }
            }
            Expr::Vector3Literal { x, y, z, .. } => {
                self.collect_expr_refs(x);
                self.collect_expr_refs(y);
                self.collect_expr_refs(z);
            }
            Expr::QuaternionLiteral { w, x, y, z, .. } => {
                self.collect_expr_refs(w);
                self.collect_expr_refs(x);
                self.collect_expr_refs(y);
                self.collect_expr_refs(z);
            }
            Expr::IfExpr {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_expr_refs(condition);
                self.collect_expr_refs(then_branch);
                if let Some(eb) = else_branch {
                    self.collect_expr_refs(eb);
                }
            }
            Expr::LetExpr {
                value, body, ..
            } => {
                self.collect_expr_refs(value);
                self.collect_expr_refs(body);
            }
            Expr::Literal(_, _) => {}
        }
    }

    fn collect_geometry_refs(&mut self, geom: &GeometryExpr) {
        match geom {
            GeometryExpr::Box {
                center,
                half_extents,
                ..
            } => {
                self.collect_expr_refs(center);
                self.collect_expr_refs(half_extents);
            }
            GeometryExpr::Sphere {
                center, radius, ..
            } => {
                self.collect_expr_refs(center);
                self.collect_expr_refs(radius);
            }
            GeometryExpr::Capsule {
                start,
                end,
                radius,
                ..
            } => {
                self.collect_expr_refs(start);
                self.collect_expr_refs(end);
                self.collect_expr_refs(radius);
            }
            GeometryExpr::Cylinder {
                center,
                radius,
                height,
                ..
            } => {
                self.collect_expr_refs(center);
                self.collect_expr_refs(radius);
                self.collect_expr_refs(height);
            }
            GeometryExpr::ConvexHull { points, .. } => {
                for p in points {
                    self.collect_expr_refs(p);
                }
            }
            GeometryExpr::CSGUnion { a, b, .. }
            | GeometryExpr::CSGIntersection { a, b, .. }
            | GeometryExpr::CSGDifference { a, b, .. } => {
                self.collect_geometry_refs(a);
                self.collect_geometry_refs(b);
            }
            GeometryExpr::Transform {
                geometry,
                transform,
                ..
            } => {
                self.collect_geometry_refs(geometry);
                self.collect_expr_refs(transform);
            }
            GeometryExpr::Reference { name, span } => {
                self.references.push((name.clone(), span.clone()));
            }
        }
    }

    fn collect_pattern_refs(&mut self, pattern: &PatternExpr) {
        match pattern {
            PatternExpr::Gaze {
                entity,
                target,
                angle,
                ..
            } => {
                self.collect_expr_refs(entity);
                self.collect_expr_refs(target);
                if let Some(a) = angle {
                    self.collect_expr_refs(a);
                }
            }
            PatternExpr::Reach {
                entity,
                target,
                distance,
                ..
            } => {
                self.collect_expr_refs(entity);
                self.collect_expr_refs(target);
                if let Some(d) = distance {
                    self.collect_expr_refs(d);
                }
            }
            PatternExpr::Grab { target, .. } | PatternExpr::Release { target, .. } => {
                self.collect_expr_refs(target);
            }
            PatternExpr::Proximity {
                entity_a,
                entity_b,
                distance,
                ..
            } => {
                self.collect_expr_refs(entity_a);
                self.collect_expr_refs(entity_b);
                self.collect_expr_refs(distance);
            }
            PatternExpr::Inside {
                entity, region, ..
            } => {
                self.collect_expr_refs(entity);
                self.collect_expr_refs(region);
            }
            PatternExpr::Touch {
                entity_a,
                entity_b,
                ..
            } => {
                self.collect_expr_refs(entity_a);
                self.collect_expr_refs(entity_b);
            }
            PatternExpr::Conjunction { patterns, .. }
            | PatternExpr::Disjunction { patterns, .. }
            | PatternExpr::Sequence { patterns, .. } => {
                for p in patterns {
                    self.collect_pattern_refs(p);
                }
            }
            PatternExpr::Negation { pattern, .. } => {
                self.collect_pattern_refs(pattern);
            }
            PatternExpr::Timeout { duration, .. } => {
                self.collect_expr_refs(duration);
            }
            PatternExpr::TimedPattern { pattern, .. } => {
                self.collect_pattern_refs(pattern);
            }
            PatternExpr::Custom { name, args, span } => {
                self.references.push((name.clone(), span.clone()));
                for a in args {
                    self.collect_expr_refs(a);
                }
            }
        }
    }

    fn collect_action_refs(&mut self, action: &ActionExpr) {
        match action {
            ActionExpr::Activate { target, .. } | ActionExpr::Deactivate { target, .. } => {
                self.collect_expr_refs(target);
            }
            ActionExpr::Emit { args, .. } => {
                for a in args {
                    self.collect_expr_refs(a);
                }
            }
            ActionExpr::SetTimer { duration, .. } => {
                self.collect_expr_refs(duration);
            }
            ActionExpr::CancelTimer { .. } => {}
            ActionExpr::UpdatePosition {
                entity, position, ..
            } => {
                self.collect_expr_refs(entity);
                self.collect_expr_refs(position);
            }
            ActionExpr::Spawn { .. } => {}
            ActionExpr::Destroy { entity, .. } => {
                self.collect_expr_refs(entity);
            }
            ActionExpr::Custom { name, args, span } => {
                self.references.push((name.clone(), span.clone()));
                for a in args {
                    self.collect_expr_refs(a);
                }
            }
        }
    }

    fn collect_constraint_refs(&mut self, constraint: &SpatialConstraintExpr) {
        match constraint {
            SpatialConstraintExpr::NonOverlapping(names, span) => {
                for n in names {
                    self.references.push((n.clone(), span.clone()));
                }
            }
            SpatialConstraintExpr::ContainedIn(inner, outer, span) => {
                self.references.push((inner.clone(), span.clone()));
                self.references.push((outer.clone(), span.clone()));
            }
            SpatialConstraintExpr::MinDistance(a, b, expr, span)
            | SpatialConstraintExpr::MaxDistance(a, b, expr, span) => {
                self.references.push((a.clone(), span.clone()));
                self.references.push((b.clone(), span.clone()));
                self.collect_expr_refs(expr);
            }
            SpatialConstraintExpr::WithinRegion(entity, region, span) => {
                self.references.push((entity.clone(), span.clone()));
                self.references.push((region.clone(), span.clone()));
            }
            SpatialConstraintExpr::Custom(name, args, span) => {
                self.references.push((name.clone(), span.clone()));
                for a in args {
                    self.collect_expr_refs(a);
                }
            }
        }
    }

    fn collect_choreography_refs(&mut self, choreo: &ChoreographyExpr) {
        match choreo {
            ChoreographyExpr::Sequential { steps, .. } => {
                for s in steps {
                    self.collect_choreography_refs(s);
                }
            }
            ChoreographyExpr::Parallel { branches, .. } => {
                for b in branches {
                    self.collect_choreography_refs(b);
                }
            }
            ChoreographyExpr::Choice { options, .. } => {
                for o in options {
                    self.collect_choreography_refs(o);
                }
            }
            ChoreographyExpr::Loop { body, bound, .. } => {
                self.collect_choreography_refs(body);
                if let Some(b) = bound {
                    self.collect_expr_refs(b);
                }
            }
            ChoreographyExpr::Guarded { guard, body, .. } => {
                self.collect_pattern_refs(guard);
                self.collect_choreography_refs(body);
            }
            ChoreographyExpr::Action { actions, .. } => {
                for a in actions {
                    self.collect_action_refs(a);
                }
            }
            ChoreographyExpr::Reference { name, args, span } => {
                self.references.push((name.clone(), span.clone()));
                for a in args {
                    self.collect_expr_refs(a);
                }
            }
            ChoreographyExpr::Conditional {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_expr_refs(condition);
                self.collect_choreography_refs(then_branch);
                if let Some(eb) = else_branch {
                    self.collect_choreography_refs(eb);
                }
            }
            ChoreographyExpr::LetIn {
                value, body, ..
            } => {
                self.collect_expr_refs(value);
                self.collect_choreography_refs(body);
            }
        }
    }

    // -- unused definitions --

    fn check_unused_definitions(&mut self) {
        // Only check the global scope for unused warnings
        if let Some(global) = self.scopes.first() {
            for (name, info) in &global.symbols {
                if !info.used && info.kind != SymbolKind::Scene && info.kind != SymbolKind::Import {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: format!("'{}' is defined but never used", name),
                        span: Some(info.span.clone()),
                    });
                }
            }
        }
    }

    // -- cycle detection in region parent references --

    fn check_cycle_detection(&mut self, program: &Program) {
        let mut parent_map: HashMap<String, String> = HashMap::new();
        for decl in &program.declarations {
            if let Declaration::Region(r) = decl {
                if let Some(parent) = &r.parent {
                    parent_map.insert(r.name.clone(), parent.clone());
                }
            }
        }

        for start in parent_map.keys() {
            let mut visited: HashSet<String> = HashSet::new();
            let mut current = start.clone();
            loop {
                if !visited.insert(current.clone()) {
                    // Found a cycle
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: format!(
                            "cycle detected in region parent chain involving '{}'",
                            start
                        ),
                        span: None,
                    });
                    break;
                }
                match parent_map.get(&current) {
                    Some(parent) => current = parent.clone(),
                    None => break,
                }
            }
        }
    }

    // -- pattern validity --

    fn check_all_patterns(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Interaction(i) => {
                    if let Some(pat) = &i.pattern {
                        self.check_pattern_validity(pat);
                    }
                    match &i.body {
                        InteractionBody::Simple { guard, .. } => {
                            self.check_pattern_validity(guard);
                        }
                        InteractionBody::Choreography(choreo) => {
                            self.check_choreography_patterns(choreo);
                        }
                    }
                }
                Declaration::Zone(z) => {
                    for rule in &z.rules {
                        self.check_pattern_validity(&rule.pattern);
                    }
                }
                _ => {}
            }
        }
    }

    fn check_choreography_patterns(&mut self, choreo: &ChoreographyExpr) {
        match choreo {
            ChoreographyExpr::Sequential { steps, .. } => {
                for s in steps {
                    self.check_choreography_patterns(s);
                }
            }
            ChoreographyExpr::Parallel { branches, .. } => {
                for b in branches {
                    self.check_choreography_patterns(b);
                }
            }
            ChoreographyExpr::Choice { options, .. } => {
                for o in options {
                    self.check_choreography_patterns(o);
                }
            }
            ChoreographyExpr::Loop { body, .. } => {
                self.check_choreography_patterns(body);
            }
            ChoreographyExpr::Guarded { guard, body, .. } => {
                self.check_pattern_validity(guard);
                self.check_choreography_patterns(body);
            }
            ChoreographyExpr::Conditional {
                then_branch,
                else_branch,
                ..
            } => {
                self.check_choreography_patterns(then_branch);
                if let Some(eb) = else_branch {
                    self.check_choreography_patterns(eb);
                }
            }
            ChoreographyExpr::LetIn { body, .. } => {
                self.check_choreography_patterns(body);
            }
            _ => {}
        }
    }

    /// Validate a pattern expression: check that sequences have ≥2 elements,
    /// conjunctions/disjunctions have ≥2 branches, and nested patterns are valid.
    pub fn check_pattern_validity(&mut self, pattern: &PatternExpr) {
        match pattern {
            PatternExpr::Conjunction { patterns, span } => {
                if patterns.len() < 2 {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "conjunction pattern with fewer than 2 branches".to_string(),
                        span: Some(span.clone()),
                    });
                }
                for p in patterns {
                    self.check_pattern_validity(p);
                }
            }
            PatternExpr::Disjunction { patterns, span } => {
                if patterns.len() < 2 {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "disjunction pattern with fewer than 2 branches".to_string(),
                        span: Some(span.clone()),
                    });
                }
                for p in patterns {
                    self.check_pattern_validity(p);
                }
            }
            PatternExpr::Sequence { patterns, span } => {
                if patterns.len() < 2 {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "sequence pattern with fewer than 2 steps".to_string(),
                        span: Some(span.clone()),
                    });
                }
                for p in patterns {
                    self.check_pattern_validity(p);
                }
            }
            PatternExpr::Negation { pattern, .. } => {
                self.check_pattern_validity(pattern);
                // Negation of a negation is suspicious
                if matches!(**pattern, PatternExpr::Negation { .. }) {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "double negation in pattern".to_string(),
                        span: Some(pattern.span().clone()),
                    });
                }
            }
            PatternExpr::TimedPattern {
                pattern,
                constraint,
                span,
            } => {
                self.check_pattern_validity(pattern);
                self.check_temporal_constraint_validity(constraint, span);
            }
            _ => {}
        }
    }

    fn check_temporal_constraint_validity(
        &mut self,
        constraint: &TemporalConstraintExpr,
        _span: &Span,
    ) {
        match constraint {
            TemporalConstraintExpr::Within(expr, span) | TemporalConstraintExpr::After(expr, span) => {
                // Warn if the constraint uses a non-literal (might be fine but worth noting)
                if let Expr::Literal(Literal::Duration(val, _), _) = expr {
                    if *val <= 0.0 {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: "temporal constraint duration must be positive".to_string(),
                            span: Some(span.clone()),
                        });
                    }
                }
            }
            TemporalConstraintExpr::Between(lo, hi, span) => {
                if let (
                    Expr::Literal(Literal::Duration(lo_val, lo_unit), _),
                    Expr::Literal(Literal::Duration(hi_val, hi_unit), _),
                ) = (lo, hi)
                {
                    let lo_sec = lo_unit.to_seconds(*lo_val);
                    let hi_sec = hi_unit.to_seconds(*hi_val);
                    if lo_sec >= hi_sec {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message:
                                "between constraint: lower bound must be less than upper bound"
                                    .to_string(),
                            span: Some(span.clone()),
                        });
                    }
                }
            }
        }
    }

    // -- action validity --

    fn check_all_actions(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Interaction(i) => match &i.body {
                    InteractionBody::Simple { actions, .. } => {
                        for action in actions {
                            self.check_action_validity(action);
                        }
                    }
                    InteractionBody::Choreography(choreo) => {
                        self.check_choreography_actions(choreo);
                    }
                },
                Declaration::Zone(z) => {
                    for rule in &z.rules {
                        for action in &rule.actions {
                            self.check_action_validity(action);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn check_choreography_actions(&mut self, choreo: &ChoreographyExpr) {
        match choreo {
            ChoreographyExpr::Action { actions, .. } => {
                for a in actions {
                    self.check_action_validity(a);
                }
            }
            ChoreographyExpr::Sequential { steps, .. } => {
                for s in steps {
                    self.check_choreography_actions(s);
                }
            }
            ChoreographyExpr::Parallel { branches, .. } => {
                for b in branches {
                    self.check_choreography_actions(b);
                }
            }
            ChoreographyExpr::Choice { options, .. } => {
                for o in options {
                    self.check_choreography_actions(o);
                }
            }
            ChoreographyExpr::Loop { body, .. } => {
                self.check_choreography_actions(body);
            }
            ChoreographyExpr::Guarded { body, .. } => {
                self.check_choreography_actions(body);
            }
            ChoreographyExpr::Conditional {
                then_branch,
                else_branch,
                ..
            } => {
                self.check_choreography_actions(then_branch);
                if let Some(eb) = else_branch {
                    self.check_choreography_actions(eb);
                }
            }
            ChoreographyExpr::LetIn { body, .. } => {
                self.check_choreography_actions(body);
            }
            _ => {}
        }
    }

    /// Validate an action expression.
    pub fn check_action_validity(&mut self, action: &ActionExpr) {
        match action {
            ActionExpr::SetTimer { name, duration, span } => {
                if name.is_empty() {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: "timer name cannot be empty".to_string(),
                        span: Some(span.clone()),
                    });
                }
                if let Expr::Literal(Literal::Duration(val, _), _) = duration.as_ref() {
                    if *val <= 0.0 {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: "timer duration must be positive".to_string(),
                            span: Some(span.clone()),
                        });
                    }
                }
            }
            ActionExpr::CancelTimer { name, span } => {
                if name.is_empty() {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: "timer name cannot be empty".to_string(),
                        span: Some(span.clone()),
                    });
                }
            }
            ActionExpr::Emit {
                event_name, span, ..
            } => {
                if event_name.is_empty() {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: "event name cannot be empty".to_string(),
                        span: Some(span.clone()),
                    });
                }
            }
            _ => {}
        }
    }
}

impl Default for SemanticAnalyzer {
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
    use crate::ast::*;
    use crate::token::DurationUnit;

    fn dummy_span() -> Span {
        Span {
            start: 0,
            end: 0,
            file: None,
        }
    }

    fn make_id(name: &str) -> Expr {
        Expr::Identifier(name.to_string(), dummy_span())
    }

    fn make_region(name: &str, parent: Option<&str>) -> Declaration {
        Declaration::Region(RegionDecl {
            name: name.to_string(),
            geometry: GeometryExpr::Reference {
                name: "base".into(),
                span: dummy_span(),
            },
            parent: parent.map(|s| s.to_string()),
            constraints: vec![],
            span: dummy_span(),
        })
    }

    fn make_entity(name: &str) -> Declaration {
        Declaration::Entity(EntityDecl {
            name: name.to_string(),
            entity_type: EntityTypeExpr::Object,
            initial_position: None,
            initial_rotation: None,
            bounding_volume: None,
            properties: vec![],
            span: dummy_span(),
        })
    }

    #[test]
    fn test_duplicate_definition() {
        let program = Program {
            declarations: vec![
                make_region("r1", None),
                make_region("r1", None),
            ],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        assert!(
            diags.iter().any(|d| d.message.contains("duplicate")),
            "expected duplicate definition error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_undefined_reference() {
        let program = Program {
            declarations: vec![Declaration::Zone(ZoneDecl {
                name: "z1".to_string(),
                regions: vec!["nonexistent_region".to_string()],
                rules: vec![],
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        assert!(
            diags.iter().any(|d| d.message.contains("undefined")),
            "expected undefined reference error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_unused_definition() {
        let program = Program {
            declarations: vec![
                make_region("r1", None),
                make_entity("e1"),
            ],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        let warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Warning && d.message.contains("never used"))
            .collect();
        // Both r1 and e1 are unused
        assert!(warnings.len() >= 2, "expected unused warnings, got: {:?}", warnings);
    }

    #[test]
    fn test_cycle_detection() {
        let program = Program {
            declarations: vec![
                make_region("a", Some("b")),
                make_region("b", Some("c")),
                make_region("c", Some("a")),
            ],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        assert!(
            diags.iter().any(|d| d.message.contains("cycle")),
            "expected cycle detection error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_pattern_validity_single_branch_conjunction() {
        let program = Program {
            declarations: vec![Declaration::Interaction(InteractionDecl {
                name: "test_interaction".to_string(),
                parameters: vec![],
                pattern: Some(PatternExpr::Conjunction {
                    patterns: vec![PatternExpr::Grab {
                        target: Box::new(make_id("obj")),
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                }),
                body: InteractionBody::Simple {
                    guard: PatternExpr::Grab {
                        target: Box::new(make_id("obj")),
                        span: dummy_span(),
                    },
                    actions: vec![],
                    span: dummy_span(),
                },
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("fewer than 2")),
            "expected warning about single-branch conjunction, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_negative_duration_in_timer() {
        let program = Program {
            declarations: vec![Declaration::Interaction(InteractionDecl {
                name: "timer_test".to_string(),
                parameters: vec![],
                pattern: None,
                body: InteractionBody::Simple {
                    guard: PatternExpr::Grab {
                        target: Box::new(make_id("x")),
                        span: dummy_span(),
                    },
                    actions: vec![ActionExpr::SetTimer {
                        name: "t".to_string(),
                        duration: Box::new(Expr::Literal(
                            Literal::Duration(-1.0, DurationUnit::S),
                            dummy_span(),
                        )),
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("timer duration must be positive")),
            "expected positive duration error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_no_errors_on_valid_program() {
        let program = Program {
            declarations: vec![
                make_region("r1", None),
                make_entity("player"),
                Declaration::Zone(ZoneDecl {
                    name: "z1".to_string(),
                    regions: vec!["r1".to_string()],
                    rules: vec![ZoneRule {
                        pattern: PatternExpr::Inside {
                            entity: Box::new(make_id("player")),
                            region: Box::new(make_id("r1")),
                            span: dummy_span(),
                        },
                        actions: vec![ActionExpr::Emit {
                            event_name: "entered".to_string(),
                            args: vec![],
                            span: dummy_span(),
                        }],
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_empty_event_name() {
        let program = Program {
            declarations: vec![Declaration::Interaction(InteractionDecl {
                name: "emit_test".to_string(),
                parameters: vec![],
                pattern: None,
                body: InteractionBody::Simple {
                    guard: PatternExpr::Grab {
                        target: Box::new(make_id("x")),
                        span: dummy_span(),
                    },
                    actions: vec![ActionExpr::Emit {
                        event_name: "".to_string(),
                        args: vec![],
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut analyzer = SemanticAnalyzer::new();
        let diags = analyzer.analyze(&program);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("event name cannot be empty")),
            "expected empty event name error, got: {:?}",
            diags
        );
    }
}
