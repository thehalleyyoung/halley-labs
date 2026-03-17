//! Desugaring pass for the Choreo DSL.
//!
//! Expands syntactic sugar to core representations: flattens nested
//! sequential/parallel choreographies, expands let bindings by substitution,
//! normalises pattern and geometry expressions, and canonicalises temporal
//! constraints.

use crate::ast::*;
use choreo_types::Span;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SubstitutionEnv
// ---------------------------------------------------------------------------

/// Environment for tracking let-binding substitutions.
#[derive(Debug, Clone)]
pub struct SubstitutionEnv {
    bindings: HashMap<String, Expr>,
}

impl SubstitutionEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: String, value: Expr) {
        self.bindings.insert(name, value);
    }

    pub fn lookup(&self, name: &str) -> Option<&Expr> {
        self.bindings.get(name)
    }

    pub fn with_binding(&self, name: String, value: Expr) -> Self {
        let mut new_env = self.clone();
        new_env.bind(name, value);
        new_env
    }
}

impl Default for SubstitutionEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Desugarer
// ---------------------------------------------------------------------------

/// The desugaring pass transforms the AST into a simpler canonical form.
pub struct Desugarer {
    env: SubstitutionEnv,
}

impl Desugarer {
    pub fn new() -> Self {
        Self {
            env: SubstitutionEnv::new(),
        }
    }

    /// Desugar an entire program.
    pub fn desugar_program(&mut self, program: &Program) -> Program {
        // First pass: collect top-level let bindings into the env
        for decl in &program.declarations {
            if let Declaration::LetBinding(lb) = decl {
                let desugared_val = self.desugar_expr(&lb.value);
                self.env.bind(lb.name.clone(), desugared_val);
            }
        }

        // Second pass: desugar all declarations (let bindings are removed)
        let declarations: Vec<Declaration> = program
            .declarations
            .iter()
            .filter(|d| !matches!(d, Declaration::LetBinding(_)))
            .map(|d| self.desugar_declaration(d))
            .collect();

        Program {
            declarations,
            span: program.span.clone(),
        }
    }

    fn desugar_declaration(&mut self, decl: &Declaration) -> Declaration {
        match decl {
            Declaration::Region(r) => Declaration::Region(self.desugar_region(r)),
            Declaration::Entity(e) => Declaration::Entity(self.desugar_entity(e)),
            Declaration::Zone(z) => Declaration::Zone(self.desugar_zone(z)),
            Declaration::Interaction(i) => Declaration::Interaction(self.desugar_interaction(i)),
            Declaration::Scene(s) => Declaration::Scene(self.desugar_scene(s)),
            Declaration::LetBinding(l) => Declaration::LetBinding(l.clone()),
            Declaration::Import(i) => Declaration::Import(i.clone()),
        }
    }

    fn desugar_region(&mut self, r: &RegionDecl) -> RegionDecl {
        RegionDecl {
            name: r.name.clone(),
            geometry: self.desugar_geometry(&r.geometry),
            parent: r.parent.clone(),
            constraints: r
                .constraints
                .iter()
                .map(|c| self.desugar_constraint(c))
                .collect(),
            span: r.span.clone(),
        }
    }

    fn desugar_entity(&mut self, e: &EntityDecl) -> EntityDecl {
        EntityDecl {
            name: e.name.clone(),
            entity_type: e.entity_type.clone(),
            initial_position: e.initial_position.as_ref().map(|p| self.desugar_expr(p)),
            initial_rotation: e.initial_rotation.as_ref().map(|r| self.desugar_expr(r)),
            bounding_volume: e.bounding_volume.as_ref().map(|g| self.desugar_geometry(g)),
            properties: e
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), self.desugar_expr(v)))
                .collect(),
            span: e.span.clone(),
        }
    }

    fn desugar_zone(&mut self, z: &ZoneDecl) -> ZoneDecl {
        ZoneDecl {
            name: z.name.clone(),
            regions: z.regions.clone(),
            rules: z
                .rules
                .iter()
                .map(|rule| ZoneRule {
                    pattern: self.desugar_pattern(&rule.pattern),
                    actions: rule
                        .actions
                        .iter()
                        .map(|a| self.desugar_action(a))
                        .collect(),
                    span: rule.span.clone(),
                })
                .collect(),
            span: z.span.clone(),
        }
    }

    fn desugar_interaction(&mut self, i: &InteractionDecl) -> InteractionDecl {
        InteractionDecl {
            name: i.name.clone(),
            parameters: i.parameters.clone(),
            pattern: i.pattern.as_ref().map(|p| self.desugar_pattern(p)),
            body: self.desugar_body(&i.body),
            span: i.span.clone(),
        }
    }

    fn desugar_body(&mut self, body: &InteractionBody) -> InteractionBody {
        match body {
            InteractionBody::Simple {
                guard,
                actions,
                span,
            } => InteractionBody::Simple {
                guard: self.desugar_pattern(guard),
                actions: actions.iter().map(|a| self.desugar_action(a)).collect(),
                span: span.clone(),
            },
            InteractionBody::Choreography(choreo) => {
                InteractionBody::Choreography(self.desugar_choreography(choreo))
            }
        }
    }

    fn desugar_scene(&mut self, s: &SceneDecl) -> SceneDecl {
        SceneDecl {
            name: s.name.clone(),
            entities: s.entities.clone(),
            regions: s.regions.clone(),
            interactions: s
                .interactions
                .iter()
                .map(|iref| match iref {
                    InteractionRef::Named(name, args, span) => InteractionRef::Named(
                        name.clone(),
                        args.iter().map(|a| self.desugar_expr(a)).collect(),
                        span.clone(),
                    ),
                    InteractionRef::Inline(decl) => {
                        InteractionRef::Inline(self.desugar_interaction(decl))
                    }
                })
                .collect(),
            constraints: s
                .constraints
                .iter()
                .map(|c| self.desugar_constraint(c))
                .collect(),
            span: s.span.clone(),
        }
    }

    fn desugar_constraint(&mut self, c: &SpatialConstraintExpr) -> SpatialConstraintExpr {
        match c {
            SpatialConstraintExpr::MinDistance(a, b, expr, span) => {
                SpatialConstraintExpr::MinDistance(
                    a.clone(),
                    b.clone(),
                    self.desugar_expr(expr),
                    span.clone(),
                )
            }
            SpatialConstraintExpr::MaxDistance(a, b, expr, span) => {
                SpatialConstraintExpr::MaxDistance(
                    a.clone(),
                    b.clone(),
                    self.desugar_expr(expr),
                    span.clone(),
                )
            }
            SpatialConstraintExpr::Custom(name, args, span) => SpatialConstraintExpr::Custom(
                name.clone(),
                args.iter().map(|a| self.desugar_expr(a)).collect(),
                span.clone(),
            ),
            other => other.clone(),
        }
    }

    // -- geometry desugaring --

    /// Desugar a geometry expression: expand CSG to canonical form by
    /// left-associating nested unions/intersections and flattening transforms.
    pub fn desugar_geometry(&mut self, geo: &GeometryExpr) -> GeometryExpr {
        match geo {
            GeometryExpr::Box {
                center,
                half_extents,
                span,
            } => GeometryExpr::Box {
                center: Box::new(self.desugar_expr(center)),
                half_extents: Box::new(self.desugar_expr(half_extents)),
                span: span.clone(),
            },
            GeometryExpr::Sphere {
                center,
                radius,
                span,
            } => GeometryExpr::Sphere {
                center: Box::new(self.desugar_expr(center)),
                radius: Box::new(self.desugar_expr(radius)),
                span: span.clone(),
            },
            GeometryExpr::Capsule {
                start,
                end,
                radius,
                span,
            } => GeometryExpr::Capsule {
                start: Box::new(self.desugar_expr(start)),
                end: Box::new(self.desugar_expr(end)),
                radius: Box::new(self.desugar_expr(radius)),
                span: span.clone(),
            },
            GeometryExpr::Cylinder {
                center,
                radius,
                height,
                span,
            } => GeometryExpr::Cylinder {
                center: Box::new(self.desugar_expr(center)),
                radius: Box::new(self.desugar_expr(radius)),
                height: Box::new(self.desugar_expr(height)),
                span: span.clone(),
            },
            GeometryExpr::ConvexHull { points, span } => GeometryExpr::ConvexHull {
                points: points.iter().map(|p| self.desugar_expr(p)).collect(),
                span: span.clone(),
            },
            GeometryExpr::CSGUnion { a, b, span } => {
                let da = self.desugar_geometry(a);
                let db = self.desugar_geometry(b);
                // Flatten nested unions: union(union(x,y), z) -> keep as-is
                // but flatten union(x, union(y, z)) into a canonical left-assoc form
                self.flatten_csg_union(da, db, span)
            }
            GeometryExpr::CSGIntersection { a, b, span } => {
                let da = self.desugar_geometry(a);
                let db = self.desugar_geometry(b);
                self.flatten_csg_intersection(da, db, span)
            }
            GeometryExpr::CSGDifference { a, b, span } => GeometryExpr::CSGDifference {
                a: Box::new(self.desugar_geometry(a)),
                b: Box::new(self.desugar_geometry(b)),
                span: span.clone(),
            },
            GeometryExpr::Transform {
                geometry,
                transform,
                span,
            } => {
                let dg = self.desugar_geometry(geometry);
                let dt = self.desugar_expr(transform);
                // Flatten nested transforms: transform(transform(g, t1), t2)
                // is kept as-is (composing transforms needs runtime)
                GeometryExpr::Transform {
                    geometry: Box::new(dg),
                    transform: Box::new(dt),
                    span: span.clone(),
                }
            }
            GeometryExpr::Reference { name, span } => {
                // If the name refers to a let-binding that is a geometry, keep reference
                GeometryExpr::Reference {
                    name: name.clone(),
                    span: span.clone(),
                }
            }
        }
    }

    /// Flatten a CSG union into left-associative canonical form.
    fn flatten_csg_union(
        &self,
        a: GeometryExpr,
        b: GeometryExpr,
        span: &Span,
    ) -> GeometryExpr {
        let mut leaves = Vec::new();
        Self::collect_union_leaves(&a, &mut leaves);
        Self::collect_union_leaves(&b, &mut leaves);

        if leaves.len() <= 1 {
            return leaves.into_iter().next().unwrap_or(a);
        }

        let mut result = leaves.remove(0);
        for leaf in leaves {
            result = GeometryExpr::CSGUnion {
                a: Box::new(result),
                b: Box::new(leaf),
                span: span.clone(),
            };
        }
        result
    }

    fn collect_union_leaves(geom: &GeometryExpr, out: &mut Vec<GeometryExpr>) {
        if let GeometryExpr::CSGUnion { a, b, .. } = geom {
            Self::collect_union_leaves(a, out);
            Self::collect_union_leaves(b, out);
        } else {
            out.push(geom.clone());
        }
    }

    /// Flatten a CSG intersection into left-associative canonical form.
    fn flatten_csg_intersection(
        &self,
        a: GeometryExpr,
        b: GeometryExpr,
        span: &Span,
    ) -> GeometryExpr {
        let mut leaves = Vec::new();
        Self::collect_intersection_leaves(&a, &mut leaves);
        Self::collect_intersection_leaves(&b, &mut leaves);

        if leaves.len() <= 1 {
            return leaves.into_iter().next().unwrap_or(a);
        }

        let mut result = leaves.remove(0);
        for leaf in leaves {
            result = GeometryExpr::CSGIntersection {
                a: Box::new(result),
                b: Box::new(leaf),
                span: span.clone(),
            };
        }
        result
    }

    fn collect_intersection_leaves(geom: &GeometryExpr, out: &mut Vec<GeometryExpr>) {
        if let GeometryExpr::CSGIntersection { a, b, .. } = geom {
            Self::collect_intersection_leaves(a, out);
            Self::collect_intersection_leaves(b, out);
        } else {
            out.push(geom.clone());
        }
    }

    // -- pattern desugaring --

    /// Desugar a pattern expression.
    pub fn desugar_pattern(&mut self, pat: &PatternExpr) -> PatternExpr {
        match pat {
            PatternExpr::Gaze {
                entity,
                target,
                angle,
                span,
            } => PatternExpr::Gaze {
                entity: Box::new(self.desugar_expr(entity)),
                target: Box::new(self.desugar_expr(target)),
                angle: angle.as_ref().map(|a| Box::new(self.desugar_expr(a))),
                span: span.clone(),
            },
            PatternExpr::Reach {
                entity,
                target,
                distance,
                span,
            } => PatternExpr::Reach {
                entity: Box::new(self.desugar_expr(entity)),
                target: Box::new(self.desugar_expr(target)),
                distance: distance.as_ref().map(|d| Box::new(self.desugar_expr(d))),
                span: span.clone(),
            },
            PatternExpr::Grab { target, span } => PatternExpr::Grab {
                target: Box::new(self.desugar_expr(target)),
                span: span.clone(),
            },
            PatternExpr::Release { target, span } => PatternExpr::Release {
                target: Box::new(self.desugar_expr(target)),
                span: span.clone(),
            },
            PatternExpr::Proximity {
                entity_a,
                entity_b,
                distance,
                span,
            } => PatternExpr::Proximity {
                entity_a: Box::new(self.desugar_expr(entity_a)),
                entity_b: Box::new(self.desugar_expr(entity_b)),
                distance: Box::new(self.desugar_expr(distance)),
                span: span.clone(),
            },
            PatternExpr::Inside {
                entity,
                region,
                span,
            } => PatternExpr::Inside {
                entity: Box::new(self.desugar_expr(entity)),
                region: Box::new(self.desugar_expr(region)),
                span: span.clone(),
            },
            PatternExpr::Touch {
                entity_a,
                entity_b,
                span,
            } => PatternExpr::Touch {
                entity_a: Box::new(self.desugar_expr(entity_a)),
                entity_b: Box::new(self.desugar_expr(entity_b)),
                span: span.clone(),
            },
            PatternExpr::Conjunction { patterns, span } => {
                let desugared: Vec<PatternExpr> = patterns
                    .iter()
                    .flat_map(|p| {
                        let dp = self.desugar_pattern(p);
                        // Flatten nested conjunctions
                        if let PatternExpr::Conjunction {
                            patterns: inner, ..
                        } = dp
                        {
                            inner
                        } else {
                            vec![dp]
                        }
                    })
                    .collect();
                if desugared.len() == 1 {
                    desugared.into_iter().next().unwrap()
                } else {
                    PatternExpr::Conjunction {
                        patterns: desugared,
                        span: span.clone(),
                    }
                }
            }
            PatternExpr::Disjunction { patterns, span } => {
                let desugared: Vec<PatternExpr> = patterns
                    .iter()
                    .flat_map(|p| {
                        let dp = self.desugar_pattern(p);
                        // Flatten nested disjunctions
                        if let PatternExpr::Disjunction {
                            patterns: inner, ..
                        } = dp
                        {
                            inner
                        } else {
                            vec![dp]
                        }
                    })
                    .collect();
                if desugared.len() == 1 {
                    desugared.into_iter().next().unwrap()
                } else {
                    PatternExpr::Disjunction {
                        patterns: desugared,
                        span: span.clone(),
                    }
                }
            }
            PatternExpr::Sequence { patterns, span } => {
                let desugared: Vec<PatternExpr> = patterns
                    .iter()
                    .flat_map(|p| {
                        let dp = self.desugar_pattern(p);
                        // Flatten nested sequences
                        if let PatternExpr::Sequence {
                            patterns: inner, ..
                        } = dp
                        {
                            inner
                        } else {
                            vec![dp]
                        }
                    })
                    .collect();
                if desugared.len() == 1 {
                    desugared.into_iter().next().unwrap()
                } else {
                    PatternExpr::Sequence {
                        patterns: desugared,
                        span: span.clone(),
                    }
                }
            }
            PatternExpr::Negation { pattern, span } => {
                let dp = self.desugar_pattern(pattern);
                // Double negation elimination: not(not(p)) -> p
                if let PatternExpr::Negation { pattern: inner, .. } = dp {
                    *inner
                } else {
                    PatternExpr::Negation {
                        pattern: Box::new(dp),
                        span: span.clone(),
                    }
                }
            }
            PatternExpr::Timeout { duration, span } => PatternExpr::Timeout {
                duration: Box::new(self.desugar_expr(duration)),
                span: span.clone(),
            },
            PatternExpr::TimedPattern {
                pattern,
                constraint,
                span,
            } => PatternExpr::TimedPattern {
                pattern: Box::new(self.desugar_pattern(pattern)),
                constraint: self.normalize_temporal_constraint(constraint),
                span: span.clone(),
            },
            PatternExpr::Custom { name, args, span } => PatternExpr::Custom {
                name: name.clone(),
                args: args.iter().map(|a| self.desugar_expr(a)).collect(),
                span: span.clone(),
            },
        }
    }

    /// Normalize a temporal constraint to canonical form (in seconds).
    fn normalize_temporal_constraint(
        &mut self,
        constraint: &TemporalConstraintExpr,
    ) -> TemporalConstraintExpr {
        match constraint {
            TemporalConstraintExpr::Within(expr, span) => {
                TemporalConstraintExpr::Within(self.normalize_duration_expr(expr), span.clone())
            }
            TemporalConstraintExpr::After(expr, span) => {
                TemporalConstraintExpr::After(self.normalize_duration_expr(expr), span.clone())
            }
            TemporalConstraintExpr::Between(lo, hi, span) => TemporalConstraintExpr::Between(
                self.normalize_duration_expr(lo),
                self.normalize_duration_expr(hi),
                span.clone(),
            ),
        }
    }

    /// Convert duration literals to seconds.
    fn normalize_duration_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Literal(Literal::Duration(val, unit), span) => {
                let seconds = unit.to_seconds(*val);
                Expr::Literal(
                    Literal::Duration(seconds, crate::token::DurationUnit::S),
                    span.clone(),
                )
            }
            other => self.desugar_expr(other),
        }
    }

    // -- choreography desugaring --

    /// Desugar a choreography expression.
    pub fn desugar_choreography(&mut self, choreo: &ChoreographyExpr) -> ChoreographyExpr {
        match choreo {
            ChoreographyExpr::Sequential { steps, span } => {
                let desugared: Vec<ChoreographyExpr> = steps
                    .iter()
                    .flat_map(|s| {
                        let ds = self.desugar_choreography(s);
                        // Flatten nested sequentials
                        if let ChoreographyExpr::Sequential {
                            steps: inner, ..
                        } = ds
                        {
                            inner
                        } else {
                            vec![ds]
                        }
                    })
                    .collect();
                if desugared.len() == 1 {
                    desugared.into_iter().next().unwrap()
                } else {
                    ChoreographyExpr::Sequential {
                        steps: desugared,
                        span: span.clone(),
                    }
                }
            }
            ChoreographyExpr::Parallel { branches, span } => {
                let desugared: Vec<ChoreographyExpr> = branches
                    .iter()
                    .flat_map(|b| {
                        let db = self.desugar_choreography(b);
                        // Flatten nested parallels
                        if let ChoreographyExpr::Parallel {
                            branches: inner, ..
                        } = db
                        {
                            inner
                        } else {
                            vec![db]
                        }
                    })
                    .collect();
                if desugared.len() == 1 {
                    desugared.into_iter().next().unwrap()
                } else {
                    ChoreographyExpr::Parallel {
                        branches: desugared,
                        span: span.clone(),
                    }
                }
            }
            ChoreographyExpr::Choice { options, span } => {
                let desugared: Vec<ChoreographyExpr> = options
                    .iter()
                    .map(|o| self.desugar_choreography(o))
                    .collect();
                if desugared.len() == 1 {
                    desugared.into_iter().next().unwrap()
                } else {
                    ChoreographyExpr::Choice {
                        options: desugared,
                        span: span.clone(),
                    }
                }
            }
            ChoreographyExpr::Loop { body, bound, span } => ChoreographyExpr::Loop {
                body: Box::new(self.desugar_choreography(body)),
                bound: bound.as_ref().map(|b| Box::new(self.desugar_expr(b))),
                span: span.clone(),
            },
            ChoreographyExpr::Guarded { guard, body, span } => ChoreographyExpr::Guarded {
                guard: self.desugar_pattern(guard),
                body: Box::new(self.desugar_choreography(body)),
                span: span.clone(),
            },
            ChoreographyExpr::Action { actions, span } => ChoreographyExpr::Action {
                actions: actions.iter().map(|a| self.desugar_action(a)).collect(),
                span: span.clone(),
            },
            ChoreographyExpr::Reference { name, args, span } => ChoreographyExpr::Reference {
                name: name.clone(),
                args: args.iter().map(|a| self.desugar_expr(a)).collect(),
                span: span.clone(),
            },
            ChoreographyExpr::Conditional {
                condition,
                then_branch,
                else_branch,
                span,
            } => ChoreographyExpr::Conditional {
                condition: Box::new(self.desugar_expr(condition)),
                then_branch: Box::new(self.desugar_choreography(then_branch)),
                else_branch: else_branch
                    .as_ref()
                    .map(|eb| Box::new(self.desugar_choreography(eb))),
                span: span.clone(),
            },
            ChoreographyExpr::LetIn {
                name,
                value,
                body,
                span,
            } => {
                // Expand let-in by substitution: evaluate value, bind, desugar body
                let desugared_val = self.desugar_expr(value);
                let old_env = self.env.clone();
                self.env.bind(name.clone(), desugared_val);
                let desugared_body = self.desugar_choreography(body);
                self.env = old_env;
                // After substitution the LetIn is eliminated if the body no
                // longer references the name. We always inline for simplicity.
                desugared_body
            }
        }
    }

    // -- action desugaring --

    fn desugar_action(&mut self, action: &ActionExpr) -> ActionExpr {
        match action {
            ActionExpr::Activate { target, span } => ActionExpr::Activate {
                target: Box::new(self.desugar_expr(target)),
                span: span.clone(),
            },
            ActionExpr::Deactivate { target, span } => ActionExpr::Deactivate {
                target: Box::new(self.desugar_expr(target)),
                span: span.clone(),
            },
            ActionExpr::Emit {
                event_name,
                args,
                span,
            } => ActionExpr::Emit {
                event_name: event_name.clone(),
                args: args.iter().map(|a| self.desugar_expr(a)).collect(),
                span: span.clone(),
            },
            ActionExpr::SetTimer {
                name,
                duration,
                span,
            } => ActionExpr::SetTimer {
                name: name.clone(),
                duration: Box::new(self.desugar_expr(duration)),
                span: span.clone(),
            },
            ActionExpr::CancelTimer { name, span } => ActionExpr::CancelTimer {
                name: name.clone(),
                span: span.clone(),
            },
            ActionExpr::UpdatePosition {
                entity,
                position,
                span,
            } => ActionExpr::UpdatePosition {
                entity: Box::new(self.desugar_expr(entity)),
                position: Box::new(self.desugar_expr(position)),
                span: span.clone(),
            },
            ActionExpr::Spawn { entity, span } => ActionExpr::Spawn {
                entity: entity.clone(),
                span: span.clone(),
            },
            ActionExpr::Destroy { entity, span } => ActionExpr::Destroy {
                entity: Box::new(self.desugar_expr(entity)),
                span: span.clone(),
            },
            ActionExpr::Custom { name, args, span } => ActionExpr::Custom {
                name: name.clone(),
                args: args.iter().map(|a| self.desugar_expr(a)).collect(),
                span: span.clone(),
            },
        }
    }

    // -- expression desugaring --

    /// Desugar an expression: substitute let-bound identifiers.
    pub fn desugar_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Identifier(name, _span) => {
                if let Some(val) = self.env.lookup(name) {
                    val.clone()
                } else {
                    expr.clone()
                }
            }
            Expr::BinaryOp {
                op,
                left,
                right,
                span,
            } => Expr::BinaryOp {
                op: *op,
                left: Box::new(self.desugar_expr(left)),
                right: Box::new(self.desugar_expr(right)),
                span: span.clone(),
            },
            Expr::UnaryOp { op, operand, span } => {
                let d = self.desugar_expr(operand);
                // Constant fold double negation: --x -> x
                if *op == UnOp::Neg {
                    if let Expr::UnaryOp {
                        op: UnOp::Neg,
                        operand: inner,
                        ..
                    } = &d
                    {
                        return *inner.clone();
                    }
                }
                // Boolean not-not elimination
                if *op == UnOp::Not {
                    if let Expr::UnaryOp {
                        op: UnOp::Not,
                        operand: inner,
                        ..
                    } = &d
                    {
                        return *inner.clone();
                    }
                }
                Expr::UnaryOp {
                    op: *op,
                    operand: Box::new(d),
                    span: span.clone(),
                }
            }
            Expr::FunctionCall {
                function,
                args,
                span,
            } => Expr::FunctionCall {
                function: Box::new(self.desugar_expr(function)),
                args: args.iter().map(|a| self.desugar_expr(a)).collect(),
                span: span.clone(),
            },
            Expr::FieldAccess {
                object,
                field,
                span,
            } => Expr::FieldAccess {
                object: Box::new(self.desugar_expr(object)),
                field: field.clone(),
                span: span.clone(),
            },
            Expr::Index {
                object,
                index,
                span,
            } => Expr::Index {
                object: Box::new(self.desugar_expr(object)),
                index: Box::new(self.desugar_expr(index)),
                span: span.clone(),
            },
            Expr::Tuple { elements, span } => Expr::Tuple {
                elements: elements.iter().map(|e| self.desugar_expr(e)).collect(),
                span: span.clone(),
            },
            Expr::Vector3Literal { x, y, z, span } => Expr::Vector3Literal {
                x: Box::new(self.desugar_expr(x)),
                y: Box::new(self.desugar_expr(y)),
                z: Box::new(self.desugar_expr(z)),
                span: span.clone(),
            },
            Expr::QuaternionLiteral { w, x, y, z, span } => Expr::QuaternionLiteral {
                w: Box::new(self.desugar_expr(w)),
                x: Box::new(self.desugar_expr(x)),
                y: Box::new(self.desugar_expr(y)),
                z: Box::new(self.desugar_expr(z)),
                span: span.clone(),
            },
            Expr::IfExpr {
                condition,
                then_branch,
                else_branch,
                span,
            } => Expr::IfExpr {
                condition: Box::new(self.desugar_expr(condition)),
                then_branch: Box::new(self.desugar_expr(then_branch)),
                else_branch: else_branch
                    .as_ref()
                    .map(|eb| Box::new(self.desugar_expr(eb))),
                span: span.clone(),
            },
            Expr::LetExpr {
                name,
                value,
                body,
                span,
            } => {
                // Inline the let binding
                let desugared_val = self.desugar_expr(value);
                let old_env = self.env.clone();
                self.env.bind(name.clone(), desugared_val.clone());
                let desugared_body = self.desugar_expr(body);
                self.env = old_env;
                // Check if the binding is still referenced
                if expr_references_name(&desugared_body, name) {
                    Expr::LetExpr {
                        name: name.clone(),
                        value: Box::new(desugared_val),
                        body: Box::new(desugared_body),
                        span: span.clone(),
                    }
                } else {
                    desugared_body
                }
            }
            Expr::Literal(_, _) => expr.clone(),
        }
    }
}

impl Default for Desugarer {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if an expression still textually references a name (used after
/// substitution to decide whether to keep a let-expr wrapper).
fn expr_references_name(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Identifier(n, _) => n == name,
        Expr::BinaryOp { left, right, .. } => {
            expr_references_name(left, name) || expr_references_name(right, name)
        }
        Expr::UnaryOp { operand, .. } => expr_references_name(operand, name),
        Expr::FunctionCall {
            function, args, ..
        } => {
            expr_references_name(function, name)
                || args.iter().any(|a| expr_references_name(a, name))
        }
        Expr::FieldAccess { object, .. } => expr_references_name(object, name),
        Expr::Index { object, index, .. } => {
            expr_references_name(object, name) || expr_references_name(index, name)
        }
        Expr::Tuple { elements, .. } => elements.iter().any(|e| expr_references_name(e, name)),
        Expr::Vector3Literal { x, y, z, .. } => {
            expr_references_name(x, name)
                || expr_references_name(y, name)
                || expr_references_name(z, name)
        }
        Expr::QuaternionLiteral { w, x, y, z, .. } => {
            expr_references_name(w, name)
                || expr_references_name(x, name)
                || expr_references_name(y, name)
                || expr_references_name(z, name)
        }
        Expr::IfExpr {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            expr_references_name(condition, name)
                || expr_references_name(then_branch, name)
                || else_branch
                    .as_ref()
                    .map_or(false, |eb| expr_references_name(eb, name))
        }
        Expr::LetExpr {
            name: bound,
            value,
            body,
            ..
        } => {
            expr_references_name(value, name)
                || (bound != name && expr_references_name(body, name))
        }
        Expr::Literal(_, _) => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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

    fn make_int(v: i64) -> Expr {
        Expr::Literal(Literal::Int(v), dummy_span())
    }

    #[test]
    fn test_let_binding_expansion() {
        let program = Program {
            declarations: vec![
                Declaration::LetBinding(LetBindingDecl {
                    name: "threshold".to_string(),
                    type_annotation: None,
                    value: Expr::Literal(Literal::Float(1.5), dummy_span()),
                    span: dummy_span(),
                }),
                Declaration::Region(RegionDecl {
                    name: "r1".to_string(),
                    geometry: GeometryExpr::Sphere {
                        center: Box::new(make_int(0)),
                        radius: Box::new(make_id("threshold")),
                        span: dummy_span(),
                    },
                    parent: None,
                    constraints: vec![],
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_program(&program);
        // Let bindings should be removed
        assert!(result.declarations.len() == 1);
        // The identifier "threshold" should be replaced
        if let Declaration::Region(r) = &result.declarations[0] {
            if let GeometryExpr::Sphere { radius, .. } = &r.geometry {
                assert!(
                    matches!(radius.as_ref(), Expr::Literal(Literal::Float(f), _) if (*f - 1.5).abs() < 1e-10),
                    "expected threshold to be substituted with 1.5"
                );
            } else {
                panic!("expected sphere geometry");
            }
        } else {
            panic!("expected region declaration");
        }
    }

    #[test]
    fn test_flatten_nested_conjunction() {
        let inner = PatternExpr::Conjunction {
            patterns: vec![
                PatternExpr::Grab {
                    target: Box::new(make_id("a")),
                    span: dummy_span(),
                },
                PatternExpr::Grab {
                    target: Box::new(make_id("b")),
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let outer = PatternExpr::Conjunction {
            patterns: vec![
                inner,
                PatternExpr::Grab {
                    target: Box::new(make_id("c")),
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_pattern(&outer);
        if let PatternExpr::Conjunction { patterns, .. } = &result {
            assert_eq!(patterns.len(), 3, "expected flattened conjunction with 3 patterns");
        } else {
            panic!("expected conjunction");
        }
    }

    #[test]
    fn test_flatten_nested_sequential_choreography() {
        let inner = ChoreographyExpr::Sequential {
            steps: vec![
                ChoreographyExpr::Action {
                    actions: vec![],
                    span: dummy_span(),
                },
                ChoreographyExpr::Action {
                    actions: vec![],
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let outer = ChoreographyExpr::Sequential {
            steps: vec![
                inner,
                ChoreographyExpr::Action {
                    actions: vec![],
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_choreography(&outer);
        if let ChoreographyExpr::Sequential { steps, .. } = &result {
            assert_eq!(steps.len(), 3, "expected flattened sequential with 3 steps");
        } else {
            panic!("expected sequential");
        }
    }

    #[test]
    fn test_double_negation_elimination_pattern() {
        let inner = PatternExpr::Negation {
            pattern: Box::new(PatternExpr::Grab {
                target: Box::new(make_id("x")),
                span: dummy_span(),
            }),
            span: dummy_span(),
        };
        let outer = PatternExpr::Negation {
            pattern: Box::new(inner),
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_pattern(&outer);
        assert!(
            matches!(result, PatternExpr::Grab { .. }),
            "expected double negation to be eliminated"
        );
    }

    #[test]
    fn test_temporal_normalization() {
        let pattern = PatternExpr::TimedPattern {
            pattern: Box::new(PatternExpr::Grab {
                target: Box::new(make_id("x")),
                span: dummy_span(),
            }),
            constraint: TemporalConstraintExpr::Within(
                Expr::Literal(Literal::Duration(500.0, DurationUnit::Ms), dummy_span()),
                dummy_span(),
            ),
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_pattern(&pattern);
        if let PatternExpr::TimedPattern { constraint, .. } = &result {
            if let TemporalConstraintExpr::Within(expr, _) = constraint {
                if let Expr::Literal(Literal::Duration(val, unit), _) = expr {
                    assert!(
                        (*val - 0.5).abs() < 1e-10,
                        "expected 500ms normalized to 0.5s"
                    );
                    assert_eq!(*unit, DurationUnit::S);
                } else {
                    panic!("expected duration literal");
                }
            } else {
                panic!("expected Within constraint");
            }
        } else {
            panic!("expected timed pattern");
        }
    }

    #[test]
    fn test_csg_union_flattening() {
        let geo = GeometryExpr::CSGUnion {
            a: Box::new(GeometryExpr::CSGUnion {
                a: Box::new(GeometryExpr::Reference {
                    name: "a".into(),
                    span: dummy_span(),
                }),
                b: Box::new(GeometryExpr::Reference {
                    name: "b".into(),
                    span: dummy_span(),
                }),
                span: dummy_span(),
            }),
            b: Box::new(GeometryExpr::Reference {
                name: "c".into(),
                span: dummy_span(),
            }),
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_geometry(&geo);
        // Should be left-associated: union(union(a, b), c)
        if let GeometryExpr::CSGUnion { a, b, .. } = &result {
            assert!(matches!(b.as_ref(), GeometryExpr::Reference { name, .. } if name == "c"));
            assert!(matches!(a.as_ref(), GeometryExpr::CSGUnion { .. }));
        } else {
            panic!("expected CSGUnion");
        }
    }

    #[test]
    fn test_double_neg_expr_elimination() {
        let expr = Expr::UnaryOp {
            op: UnOp::Neg,
            operand: Box::new(Expr::UnaryOp {
                op: UnOp::Neg,
                operand: Box::new(make_int(42)),
                span: dummy_span(),
            }),
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_expr(&expr);
        assert!(
            matches!(result, Expr::Literal(Literal::Int(42), _)),
            "expected --42 to simplify to 42"
        );
    }

    #[test]
    fn test_choreography_let_in_elimination() {
        let choreo = ChoreographyExpr::LetIn {
            name: "x".to_string(),
            value: Box::new(make_int(10)),
            body: Box::new(ChoreographyExpr::Action {
                actions: vec![ActionExpr::Emit {
                    event_name: "test".to_string(),
                    args: vec![make_id("x")],
                    span: dummy_span(),
                }],
                span: dummy_span(),
            }),
            span: dummy_span(),
        };
        let mut desugarer = Desugarer::new();
        let result = desugarer.desugar_choreography(&choreo);
        // LetIn should be eliminated; "x" replaced with 10
        assert!(
            matches!(result, ChoreographyExpr::Action { .. }),
            "expected LetIn to be eliminated"
        );
        if let ChoreographyExpr::Action { actions, .. } = &result {
            if let ActionExpr::Emit { args, .. } = &actions[0] {
                assert!(
                    matches!(&args[0], Expr::Literal(Literal::Int(10), _)),
                    "expected x to be substituted with 10"
                );
            }
        }
    }
}
