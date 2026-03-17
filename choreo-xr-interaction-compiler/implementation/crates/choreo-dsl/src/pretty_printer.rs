//! Pretty printer for the Choreo DSL AST.
//!
//! Produces formatted, human-readable source text from an AST. Handles
//! operator precedence, indentation, and configurable style.

use crate::ast::*;
use crate::token::{DistanceUnit, DurationUnit};

// ---------------------------------------------------------------------------
// PrintStyle configuration
// ---------------------------------------------------------------------------

/// Configuration for the pretty printer.
#[derive(Debug, Clone)]
pub struct PrintStyle {
    /// Number of spaces per indentation level.
    pub indent_width: usize,
    /// Maximum desired line width (soft limit for wrapping hints).
    pub max_line_width: usize,
    /// Whether to use ANSI color codes in output.
    pub use_colors: bool,
}

impl Default for PrintStyle {
    fn default() -> Self {
        Self {
            indent_width: 4,
            max_line_width: 100,
            use_colors: false,
        }
    }
}

// ---------------------------------------------------------------------------
// PrettyPrinter
// ---------------------------------------------------------------------------

/// Pretty-prints a Choreo AST back to formatted source text.
pub struct PrettyPrinter {
    indent_level: usize,
    buffer: String,
    style: PrintStyle,
    current_line_len: usize,
}

impl PrettyPrinter {
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            buffer: String::new(),
            style: PrintStyle::default(),
            current_line_len: 0,
        }
    }

    pub fn with_style(style: PrintStyle) -> Self {
        Self {
            indent_level: 0,
            buffer: String::new(),
            style,
            current_line_len: 0,
        }
    }

    // -- helpers --

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    fn write(&mut self, s: &str) {
        self.buffer.push_str(s);
        self.current_line_len += s.len();
    }

    fn writeln(&mut self, s: &str) {
        self.buffer.push_str(s);
        self.buffer.push('\n');
        self.current_line_len = 0;
    }

    fn write_indent(&mut self) {
        let spaces = " ".repeat(self.indent_level * self.style.indent_width);
        self.buffer.push_str(&spaces);
        self.current_line_len = spaces.len();
    }

    fn write_indented_line(&mut self, s: &str) {
        self.write_indent();
        self.writeln(s);
    }

    fn keyword(&self, kw: &str) -> String {
        if self.style.use_colors {
            format!("\x1b[1;34m{}\x1b[0m", kw)
        } else {
            kw.to_string()
        }
    }

    fn type_name(&self, name: &str) -> String {
        if self.style.use_colors {
            format!("\x1b[1;32m{}\x1b[0m", name)
        } else {
            name.to_string()
        }
    }

    fn string_lit(&self, s: &str) -> String {
        if self.style.use_colors {
            format!("\x1b[33m\"{}\"\x1b[0m", s)
        } else {
            format!("\"{}\"", s)
        }
    }

    fn number_lit(&self, n: &str) -> String {
        if self.style.use_colors {
            format!("\x1b[36m{}\x1b[0m", n)
        } else {
            n.to_string()
        }
    }

    // -- public API --

    /// Print a program to a formatted string.
    pub fn print_program(&mut self, program: &Program) -> String {
        self.buffer.clear();
        self.indent_level = 0;
        self.current_line_len = 0;

        for (i, decl) in program.declarations.iter().enumerate() {
            if i > 0 {
                self.writeln("");
            }
            self.print_declaration(decl);
        }
        self.buffer.clone()
    }

    /// Print a single declaration.
    pub fn print_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Region(r) => self.print_region_decl(r),
            Declaration::Entity(e) => self.print_entity_decl(e),
            Declaration::Zone(z) => self.print_zone_decl(z),
            Declaration::Interaction(i) => self.print_interaction_decl(i),
            Declaration::Scene(s) => self.print_scene_decl(s),
            Declaration::LetBinding(l) => self.print_let_binding_decl(l),
            Declaration::Import(i) => self.print_import_decl(i),
        }
    }

    fn print_region_decl(&mut self, r: &RegionDecl) {
        self.write_indent();
        self.write(&format!("{} {} = ", self.keyword("region"), r.name));
        self.print_geometry_expr(&r.geometry);
        if let Some(parent) = &r.parent {
            self.write(&format!(" : {}", parent));
        }
        if !r.constraints.is_empty() {
            self.writeln(" {");
            self.indent();
            for c in &r.constraints {
                self.write_indent();
                self.print_constraint(c);
                self.writeln(";");
            }
            self.dedent();
            self.write_indented_line("}");
        } else {
            self.writeln(";");
        }
    }

    fn print_entity_decl(&mut self, e: &EntityDecl) {
        self.write_indent();
        self.write(&format!(
            "{} {} : {} ",
            self.keyword("entity"),
            e.name,
            self.format_entity_type(&e.entity_type)
        ));
        let has_details = e.initial_position.is_some()
            || e.initial_rotation.is_some()
            || e.bounding_volume.is_some()
            || !e.properties.is_empty();
        if has_details {
            self.writeln("{");
            self.indent();
            if let Some(pos) = &e.initial_position {
                self.write_indent();
                self.write("position: ");
                self.print_expr(pos);
                self.writeln(";");
            }
            if let Some(rot) = &e.initial_rotation {
                self.write_indent();
                self.write("rotation: ");
                self.print_expr(rot);
                self.writeln(";");
            }
            if let Some(bv) = &e.bounding_volume {
                self.write_indent();
                self.write("bounds: ");
                self.print_geometry_expr(bv);
                self.writeln(";");
            }
            for (key, val) in &e.properties {
                self.write_indent();
                self.write(&format!("{}: ", key));
                self.print_expr(val);
                self.writeln(";");
            }
            self.dedent();
            self.write_indented_line("}");
        } else {
            self.writeln("{}");
        }
    }

    fn format_entity_type(&self, et: &EntityTypeExpr) -> String {
        match et {
            EntityTypeExpr::User => self.type_name("User"),
            EntityTypeExpr::Hand(HandSideExpr::Left) => self.type_name("Hand(Left)"),
            EntityTypeExpr::Hand(HandSideExpr::Right) => self.type_name("Hand(Right)"),
            EntityTypeExpr::Head => self.type_name("Head"),
            EntityTypeExpr::Object => self.type_name("Object"),
            EntityTypeExpr::Controller(HandSideExpr::Left) => {
                self.type_name("Controller(Left)")
            }
            EntityTypeExpr::Controller(HandSideExpr::Right) => {
                self.type_name("Controller(Right)")
            }
            EntityTypeExpr::Anchor => self.type_name("Anchor"),
            EntityTypeExpr::Custom(name) => self.type_name(name),
        }
    }

    fn print_zone_decl(&mut self, z: &ZoneDecl) {
        self.write_indent();
        self.write(&format!(
            "{} {}({}) ",
            self.keyword("zone"),
            z.name,
            z.regions.join(", ")
        ));
        self.writeln("{");
        self.indent();
        for rule in &z.rules {
            self.write_indent();
            self.write(&format!("{} ", self.keyword("when")));
            self.print_pattern_expr(&rule.pattern);
            self.write(&format!(" {} ", self.keyword("then")));
            self.writeln("{");
            self.indent();
            for action in &rule.actions {
                self.write_indent();
                self.print_action_expr(action);
                self.writeln(";");
            }
            self.dedent();
            self.write_indented_line("}");
        }
        self.dedent();
        self.write_indented_line("}");
    }

    fn print_interaction_decl(&mut self, i: &InteractionDecl) {
        self.write_indent();
        self.write(&format!("{} {}", self.keyword("interaction"), i.name));
        if !i.parameters.is_empty() {
            self.write("(");
            for (idx, param) in i.parameters.iter().enumerate() {
                if idx > 0 {
                    self.write(", ");
                }
                self.write(&param.name);
                if let Some(ty) = &param.type_annotation {
                    self.write(": ");
                    self.print_type_annotation(ty);
                }
            }
            self.write(")");
        }
        if let Some(pat) = &i.pattern {
            self.write(&format!(" {} ", self.keyword("when")));
            self.print_pattern_expr(pat);
        }
        self.write(" ");
        match &i.body {
            InteractionBody::Simple {
                guard, actions, ..
            } => {
                self.writeln("{");
                self.indent();
                self.write_indent();
                self.write(&format!("{} ", self.keyword("when")));
                self.print_pattern_expr(guard);
                self.write(&format!(" {} ", self.keyword("then")));
                self.writeln("{");
                self.indent();
                for action in actions {
                    self.write_indent();
                    self.print_action_expr(action);
                    self.writeln(";");
                }
                self.dedent();
                self.write_indented_line("}");
                self.dedent();
                self.write_indented_line("}");
            }
            InteractionBody::Choreography(choreo) => {
                self.writeln("{");
                self.indent();
                self.write_indent();
                self.print_choreography(choreo);
                self.writeln("");
                self.dedent();
                self.write_indented_line("}");
            }
        }
    }

    fn print_scene_decl(&mut self, s: &SceneDecl) {
        self.write_indent();
        self.write(&format!("{} {} ", self.keyword("scene"), s.name));
        self.writeln("{");
        self.indent();
        for eref in &s.entities {
            self.write_indent();
            match eref {
                EntityRef::Named(name, _) => self.writeln(&format!("entity {};", name)),
                EntityRef::Inline(e) => self.print_entity_decl(e),
            }
        }
        for rref in &s.regions {
            self.write_indent();
            match rref {
                RegionRef::Named(name, _) => self.writeln(&format!("region {};", name)),
                RegionRef::Inline(r) => self.print_region_decl(r),
            }
        }
        for iref in &s.interactions {
            self.write_indent();
            match iref {
                InteractionRef::Named(name, args, _) => {
                    if args.is_empty() {
                        self.writeln(&format!("interaction {};", name));
                    } else {
                        self.write(&format!("interaction {}(", name));
                        for (idx, arg) in args.iter().enumerate() {
                            if idx > 0 {
                                self.write(", ");
                            }
                            self.print_expr(arg);
                        }
                        self.writeln(");");
                    }
                }
                InteractionRef::Inline(i) => self.print_interaction_decl(i),
            }
        }
        for c in &s.constraints {
            self.write_indent();
            self.print_constraint(c);
            self.writeln(";");
        }
        self.dedent();
        self.write_indented_line("}");
    }

    fn print_let_binding_decl(&mut self, l: &LetBindingDecl) {
        self.write_indent();
        self.write(&format!("{} {}", self.keyword("let"), l.name));
        if let Some(ty) = &l.type_annotation {
            self.write(": ");
            self.print_type_annotation(ty);
        }
        self.write(" = ");
        self.print_expr(&l.value);
        self.writeln(";");
    }

    fn print_import_decl(&mut self, i: &ImportDecl) {
        self.write_indent();
        self.write(&format!(
            "{} {}",
            self.keyword("import"),
            self.string_lit(&i.path)
        ));
        if let Some(items) = &i.items {
            self.write(&format!(" {{ {} }}", items.join(", ")));
        }
        self.writeln(";");
    }

    fn print_type_annotation(&mut self, ty: &TypeAnnotation) {
        match ty {
            TypeAnnotation::Named(name) => self.write(name),
            TypeAnnotation::Region => self.write(&self.type_name("Region").clone()),
            TypeAnnotation::Entity => self.write(&self.type_name("Entity").clone()),
            TypeAnnotation::Duration => self.write(&self.type_name("Duration").clone()),
            TypeAnnotation::Distance => self.write(&self.type_name("Distance").clone()),
            TypeAnnotation::Angle => self.write(&self.type_name("Angle").clone()),
            TypeAnnotation::Bool => self.write(&self.type_name("Bool").clone()),
            TypeAnnotation::Int => self.write(&self.type_name("Int").clone()),
            TypeAnnotation::Float => self.write(&self.type_name("Float").clone()),
            TypeAnnotation::String_ => self.write(&self.type_name("String").clone()),
            TypeAnnotation::Vector3 => self.write(&self.type_name("Vector3").clone()),
            TypeAnnotation::Quaternion => self.write(&self.type_name("Quaternion").clone()),
            TypeAnnotation::Tuple(elems) => {
                self.write("(");
                for (idx, elem) in elems.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_type_annotation(elem);
                }
                self.write(")");
            }
            TypeAnnotation::Function(params, ret) => {
                self.write("(");
                for (idx, param) in params.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_type_annotation(param);
                }
                self.write(") -> ");
                self.print_type_annotation(ret);
            }
            TypeAnnotation::Span(_) => self.write("<span>"),
        }
    }

    fn print_constraint(&mut self, c: &SpatialConstraintExpr) {
        match c {
            SpatialConstraintExpr::NonOverlapping(names, _) => {
                self.write(&format!("non_overlapping({})", names.join(", ")));
            }
            SpatialConstraintExpr::ContainedIn(inner, outer, _) => {
                self.write(&format!("contained_in({}, {})", inner, outer));
            }
            SpatialConstraintExpr::MinDistance(a, b, expr, _) => {
                self.write(&format!("min_distance({}, {}, ", a, b));
                self.print_expr(expr);
                self.write(")");
            }
            SpatialConstraintExpr::MaxDistance(a, b, expr, _) => {
                self.write(&format!("max_distance({}, {}, ", a, b));
                self.print_expr(expr);
                self.write(")");
            }
            SpatialConstraintExpr::WithinRegion(entity, region, _) => {
                self.write(&format!("within_region({}, {})", entity, region));
            }
            SpatialConstraintExpr::Custom(name, args, _) => {
                self.write(&format!("{}(", name));
                for (idx, arg) in args.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_expr(arg);
                }
                self.write(")");
            }
        }
    }

    // -- geometry expression printing --

    /// Print a geometry expression.
    pub fn print_geometry_expr(&mut self, expr: &GeometryExpr) {
        match expr {
            GeometryExpr::Box {
                center,
                half_extents,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("box")));
                self.print_expr(center);
                self.write(", ");
                self.print_expr(half_extents);
                self.write(")");
            }
            GeometryExpr::Sphere {
                center, radius, ..
            } => {
                self.write(&format!("{}(", self.keyword("sphere")));
                self.print_expr(center);
                self.write(", ");
                self.print_expr(radius);
                self.write(")");
            }
            GeometryExpr::Capsule {
                start,
                end,
                radius,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("capsule")));
                self.print_expr(start);
                self.write(", ");
                self.print_expr(end);
                self.write(", ");
                self.print_expr(radius);
                self.write(")");
            }
            GeometryExpr::Cylinder {
                center,
                radius,
                height,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("cylinder")));
                self.print_expr(center);
                self.write(", ");
                self.print_expr(radius);
                self.write(", ");
                self.print_expr(height);
                self.write(")");
            }
            GeometryExpr::ConvexHull { points, .. } => {
                self.write(&format!("{}(", self.keyword("convex_hull")));
                for (idx, p) in points.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_expr(p);
                }
                self.write(")");
            }
            GeometryExpr::CSGUnion { a, b, .. } => {
                self.write(&format!("{}(", self.keyword("union")));
                self.print_geometry_expr(a);
                self.write(", ");
                self.print_geometry_expr(b);
                self.write(")");
            }
            GeometryExpr::CSGIntersection { a, b, .. } => {
                self.write(&format!("{}(", self.keyword("intersection")));
                self.print_geometry_expr(a);
                self.write(", ");
                self.print_geometry_expr(b);
                self.write(")");
            }
            GeometryExpr::CSGDifference { a, b, .. } => {
                self.write(&format!("{}(", self.keyword("difference")));
                self.print_geometry_expr(a);
                self.write(", ");
                self.print_geometry_expr(b);
                self.write(")");
            }
            GeometryExpr::Transform {
                geometry,
                transform,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("transform")));
                self.print_geometry_expr(geometry);
                self.write(", ");
                self.print_expr(transform);
                self.write(")");
            }
            GeometryExpr::Reference { name, .. } => {
                self.write(name);
            }
        }
    }

    // -- pattern expression printing --

    /// Print a pattern expression with proper precedence.
    pub fn print_pattern_expr(&mut self, pat: &PatternExpr) {
        self.print_pattern_inner(pat, 0);
    }

    fn pattern_precedence(pat: &PatternExpr) -> u8 {
        match pat {
            PatternExpr::Disjunction { .. } => 1,
            PatternExpr::Conjunction { .. } => 2,
            PatternExpr::Sequence { .. } => 3,
            PatternExpr::Negation { .. } => 4,
            PatternExpr::TimedPattern { .. } => 5,
            _ => 10, // atoms
        }
    }

    fn print_pattern_inner(&mut self, pat: &PatternExpr, parent_prec: u8) {
        let my_prec = Self::pattern_precedence(pat);
        let needs_parens = my_prec < parent_prec;
        if needs_parens {
            self.write("(");
        }

        match pat {
            PatternExpr::Gaze {
                entity,
                target,
                angle,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("gaze")));
                self.print_expr(entity);
                self.write(", ");
                self.print_expr(target);
                if let Some(a) = angle {
                    self.write(", ");
                    self.print_expr(a);
                }
                self.write(")");
            }
            PatternExpr::Reach {
                entity,
                target,
                distance,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("reach")));
                self.print_expr(entity);
                self.write(", ");
                self.print_expr(target);
                if let Some(d) = distance {
                    self.write(", ");
                    self.print_expr(d);
                }
                self.write(")");
            }
            PatternExpr::Grab { target, .. } => {
                self.write(&format!("{}(", self.keyword("grab")));
                self.print_expr(target);
                self.write(")");
            }
            PatternExpr::Release { target, .. } => {
                self.write(&format!("{}(", self.keyword("release")));
                self.print_expr(target);
                self.write(")");
            }
            PatternExpr::Proximity {
                entity_a,
                entity_b,
                distance,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("proximity")));
                self.print_expr(entity_a);
                self.write(", ");
                self.print_expr(entity_b);
                self.write(", ");
                self.print_expr(distance);
                self.write(")");
            }
            PatternExpr::Inside {
                entity, region, ..
            } => {
                self.write(&format!("{}(", self.keyword("inside")));
                self.print_expr(entity);
                self.write(", ");
                self.print_expr(region);
                self.write(")");
            }
            PatternExpr::Touch {
                entity_a,
                entity_b,
                ..
            } => {
                self.write(&format!("{}(", self.keyword("touch")));
                self.print_expr(entity_a);
                self.write(", ");
                self.print_expr(entity_b);
                self.write(")");
            }
            PatternExpr::Conjunction { patterns, .. } => {
                for (idx, p) in patterns.iter().enumerate() {
                    if idx > 0 {
                        self.write(&format!(" {} ", self.keyword("and")));
                    }
                    self.print_pattern_inner(p, my_prec + 1);
                }
            }
            PatternExpr::Disjunction { patterns, .. } => {
                for (idx, p) in patterns.iter().enumerate() {
                    if idx > 0 {
                        self.write(&format!(" {} ", self.keyword("or")));
                    }
                    self.print_pattern_inner(p, my_prec + 1);
                }
            }
            PatternExpr::Sequence { patterns, .. } => {
                for (idx, p) in patterns.iter().enumerate() {
                    if idx > 0 {
                        self.write("; ");
                    }
                    self.print_pattern_inner(p, my_prec + 1);
                }
            }
            PatternExpr::Negation { pattern, .. } => {
                self.write(&format!("{} ", self.keyword("not")));
                self.print_pattern_inner(pattern, my_prec);
            }
            PatternExpr::Timeout { duration, .. } => {
                self.write(&format!("{}(", self.keyword("timeout")));
                self.print_expr(duration);
                self.write(")");
            }
            PatternExpr::TimedPattern {
                pattern,
                constraint,
                ..
            } => {
                self.print_pattern_inner(pattern, my_prec + 1);
                match constraint {
                    TemporalConstraintExpr::Within(expr, _) => {
                        self.write(&format!(" {} ", self.keyword("within")));
                        self.print_expr(expr);
                    }
                    TemporalConstraintExpr::After(expr, _) => {
                        self.write(&format!(" {} ", self.keyword("after")));
                        self.print_expr(expr);
                    }
                    TemporalConstraintExpr::Between(lo, hi, _) => {
                        self.write(" between ");
                        self.print_expr(lo);
                        self.write(" and ");
                        self.print_expr(hi);
                    }
                }
            }
            PatternExpr::Custom { name, args, .. } => {
                self.write(&format!("{}(", name));
                for (idx, arg) in args.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_expr(arg);
                }
                self.write(")");
            }
        }

        if needs_parens {
            self.write(")");
        }
    }

    // -- action expression printing --

    /// Print an action expression.
    pub fn print_action_expr(&mut self, act: &ActionExpr) {
        match act {
            ActionExpr::Activate { target, .. } => {
                self.write(&format!("{}(", self.keyword("activate")));
                self.print_expr(target);
                self.write(")");
            }
            ActionExpr::Deactivate { target, .. } => {
                self.write(&format!("{}(", self.keyword("deactivate")));
                self.print_expr(target);
                self.write(")");
            }
            ActionExpr::Emit {
                event_name, args, ..
            } => {
                self.write(&format!("{}({}", self.keyword("emit"), event_name));
                if !args.is_empty() {
                    self.write(", ");
                    for (idx, arg) in args.iter().enumerate() {
                        if idx > 0 {
                            self.write(", ");
                        }
                        self.print_expr(arg);
                    }
                }
                self.write(")");
            }
            ActionExpr::SetTimer {
                name, duration, ..
            } => {
                self.write(&format!("{}({}, ", self.keyword("set_timer"), name));
                self.print_expr(duration);
                self.write(")");
            }
            ActionExpr::CancelTimer { name, .. } => {
                self.write(&format!("{}({})", self.keyword("cancel_timer"), name));
            }
            ActionExpr::UpdatePosition {
                entity, position, ..
            } => {
                self.write("update_position(");
                self.print_expr(entity);
                self.write(", ");
                self.print_expr(position);
                self.write(")");
            }
            ActionExpr::Spawn { entity, .. } => {
                self.write(&format!(
                    "{}({})",
                    self.keyword("spawn"),
                    entity.name
                ));
            }
            ActionExpr::Destroy { entity, .. } => {
                self.write(&format!("{}(", self.keyword("destroy")));
                self.print_expr(entity);
                self.write(")");
            }
            ActionExpr::Custom { name, args, .. } => {
                self.write(&format!("{}(", name));
                for (idx, arg) in args.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_expr(arg);
                }
                self.write(")");
            }
        }
    }

    // -- choreography printing --

    /// Print a choreography expression with proper indentation.
    pub fn print_choreography(&mut self, choreo: &ChoreographyExpr) {
        match choreo {
            ChoreographyExpr::Sequential { steps, .. } => {
                self.write(&format!("{} ", self.keyword("seq")));
                self.writeln("{");
                self.indent();
                for step in steps {
                    self.write_indent();
                    self.print_choreography(step);
                    self.writeln(";");
                }
                self.dedent();
                self.write_indent();
                self.write("}");
            }
            ChoreographyExpr::Parallel { branches, .. } => {
                self.write(&format!("{} ", self.keyword("par")));
                self.writeln("{");
                self.indent();
                for (idx, branch) in branches.iter().enumerate() {
                    if idx > 0 {
                        self.write_indented_line("|");
                    }
                    self.write_indent();
                    self.print_choreography(branch);
                    self.writeln("");
                }
                self.dedent();
                self.write_indent();
                self.write("}");
            }
            ChoreographyExpr::Choice { options, .. } => {
                self.write(&format!("{} ", self.keyword("choice")));
                self.writeln("{");
                self.indent();
                for (idx, option) in options.iter().enumerate() {
                    if idx > 0 {
                        self.write_indented_line("|");
                    }
                    self.write_indent();
                    self.print_choreography(option);
                    self.writeln("");
                }
                self.dedent();
                self.write_indent();
                self.write("}");
            }
            ChoreographyExpr::Loop { body, bound, .. } => {
                self.write(&self.keyword("loop").clone());
                if let Some(b) = bound {
                    self.write("(");
                    self.print_expr(b);
                    self.write(")");
                }
                self.write(" ");
                self.writeln("{");
                self.indent();
                self.write_indent();
                self.print_choreography(body);
                self.writeln("");
                self.dedent();
                self.write_indent();
                self.write("}");
            }
            ChoreographyExpr::Guarded { guard, body, .. } => {
                self.write(&format!("{} ", self.keyword("when")));
                self.print_pattern_expr(guard);
                self.write(" => ");
                self.print_choreography(body);
            }
            ChoreographyExpr::Action { actions, .. } => {
                if actions.len() == 1 {
                    self.print_action_expr(&actions[0]);
                } else {
                    self.writeln("{");
                    self.indent();
                    for action in actions {
                        self.write_indent();
                        self.print_action_expr(action);
                        self.writeln(";");
                    }
                    self.dedent();
                    self.write_indent();
                    self.write("}");
                }
            }
            ChoreographyExpr::Reference { name, args, .. } => {
                self.write(name);
                if !args.is_empty() {
                    self.write("(");
                    for (idx, arg) in args.iter().enumerate() {
                        if idx > 0 {
                            self.write(", ");
                        }
                        self.print_expr(arg);
                    }
                    self.write(")");
                }
            }
            ChoreographyExpr::Conditional {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.write(&format!("{} ", self.keyword("if")));
                self.print_expr(condition);
                self.write(" ");
                self.writeln("{");
                self.indent();
                self.write_indent();
                self.print_choreography(then_branch);
                self.writeln("");
                self.dedent();
                self.write_indent();
                self.write("}");
                if let Some(eb) = else_branch {
                    self.write(&format!(" {} ", self.keyword("else")));
                    self.writeln("{");
                    self.indent();
                    self.write_indent();
                    self.print_choreography(eb);
                    self.writeln("");
                    self.dedent();
                    self.write_indent();
                    self.write("}");
                }
            }
            ChoreographyExpr::LetIn {
                name, value, body, ..
            } => {
                self.write(&format!("{} {} = ", self.keyword("let"), name));
                self.print_expr(value);
                self.write(&format!(" {} ", self.keyword("in")));
                self.print_choreography(body);
            }
        }
    }

    // -- expression printing --

    /// Print an expression with operator precedence.
    pub fn print_expr(&mut self, expr: &Expr) {
        self.print_expr_inner(expr, 0);
    }

    fn print_expr_inner(&mut self, expr: &Expr, parent_prec: u8) {
        match expr {
            Expr::Literal(lit, _) => self.print_literal(lit),
            Expr::Identifier(name, _) => self.write(name),
            Expr::BinaryOp {
                op, left, right, ..
            } => {
                let (left_bp, _right_bp) = op.binding_power();
                let needs_parens = left_bp < parent_prec;
                if needs_parens {
                    self.write("(");
                }
                self.print_expr_inner(left, left_bp);
                self.write(&format!(" {} ", op));
                self.print_expr_inner(right, left_bp + 1);
                if needs_parens {
                    self.write(")");
                }
            }
            Expr::UnaryOp { op, operand, .. } => {
                self.write(&format!("{}", op));
                if *op == UnOp::Not {
                    self.write(" ");
                }
                self.print_expr_inner(operand, 13);
            }
            Expr::FunctionCall {
                function, args, ..
            } => {
                self.print_expr_inner(function, 0);
                self.write("(");
                for (idx, arg) in args.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_expr_inner(arg, 0);
                }
                self.write(")");
            }
            Expr::FieldAccess { object, field, .. } => {
                self.print_expr_inner(object, 14);
                self.write(&format!(".{}", field));
            }
            Expr::Index { object, index, .. } => {
                self.print_expr_inner(object, 14);
                self.write("[");
                self.print_expr_inner(index, 0);
                self.write("]");
            }
            Expr::Tuple { elements, .. } => {
                self.write("(");
                for (idx, elem) in elements.iter().enumerate() {
                    if idx > 0 {
                        self.write(", ");
                    }
                    self.print_expr_inner(elem, 0);
                }
                self.write(")");
            }
            Expr::Vector3Literal { x, y, z, .. } => {
                self.write("vec3(");
                self.print_expr_inner(x, 0);
                self.write(", ");
                self.print_expr_inner(y, 0);
                self.write(", ");
                self.print_expr_inner(z, 0);
                self.write(")");
            }
            Expr::QuaternionLiteral { w, x, y, z, .. } => {
                self.write("quat(");
                self.print_expr_inner(w, 0);
                self.write(", ");
                self.print_expr_inner(x, 0);
                self.write(", ");
                self.print_expr_inner(y, 0);
                self.write(", ");
                self.print_expr_inner(z, 0);
                self.write(")");
            }
            Expr::IfExpr {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.write(&format!("{} ", self.keyword("if")));
                self.print_expr_inner(condition, 0);
                self.write(" { ");
                self.print_expr_inner(then_branch, 0);
                self.write(" }");
                if let Some(eb) = else_branch {
                    self.write(&format!(" {} {{ ", self.keyword("else")));
                    self.print_expr_inner(eb, 0);
                    self.write(" }");
                }
            }
            Expr::LetExpr {
                name, value, body, ..
            } => {
                self.write(&format!("{} {} = ", self.keyword("let"), name));
                self.print_expr_inner(value, 0);
                self.write(&format!(" {} ", self.keyword("in")));
                self.print_expr_inner(body, 0);
            }
        }
    }

    fn print_literal(&mut self, lit: &Literal) {
        match lit {
            Literal::Int(i) => self.write(&self.number_lit(&i.to_string()).clone()),
            Literal::Float(f) => {
                let s = if f.fract() == 0.0 {
                    format!("{:.1}", f)
                } else {
                    format!("{}", f)
                };
                self.write(&self.number_lit(&s).clone());
            }
            Literal::String(s) => self.write(&self.string_lit(s).clone()),
            Literal::Bool(b) => {
                let kw = if *b { "true" } else { "false" };
                self.write(&self.keyword(kw).clone());
            }
            Literal::Duration(val, unit) => {
                let s = format!("{}{}", val, unit);
                self.write(&self.number_lit(&s).clone());
            }
            Literal::Distance(val, unit) => {
                let s = format!("{}{}", val, self.format_distance_unit(unit));
                self.write(&self.number_lit(&s).clone());
            }
            Literal::Angle(val) => {
                let s = format!("{}deg", val);
                self.write(&self.number_lit(&s).clone());
            }
        }
    }

    fn format_distance_unit(&self, unit: &DistanceUnit) -> &'static str {
        match unit {
            DistanceUnit::Mm => "mm",
            DistanceUnit::Cm => "cm",
            DistanceUnit::M => "m",
        }
    }
}

impl Default for PrettyPrinter {
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
    use crate::token::DurationUnit;

    fn dummy_span() -> Span {
        choreo_types::Span {
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

    fn make_float(v: f64) -> Expr {
        Expr::Literal(Literal::Float(v), dummy_span())
    }

    #[test]
    fn test_print_simple_let_binding() {
        let program = Program {
            declarations: vec![Declaration::LetBinding(LetBindingDecl {
                name: "threshold".to_string(),
                type_annotation: Some(TypeAnnotation::Float),
                value: Expr::Literal(Literal::Float(1.5), dummy_span()),
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        let output = pp.print_program(&program);
        assert!(output.contains("let threshold"), "output: {}", output);
        assert!(output.contains("Float"), "output: {}", output);
        assert!(output.contains("1.5"), "output: {}", output);
    }

    #[test]
    fn test_print_region() {
        let program = Program {
            declarations: vec![Declaration::Region(RegionDecl {
                name: "lobby".to_string(),
                geometry: GeometryExpr::Sphere {
                    center: Box::new(Expr::Vector3Literal {
                        x: Box::new(make_float(0.0)),
                        y: Box::new(make_float(0.0)),
                        z: Box::new(make_float(0.0)),
                        span: dummy_span(),
                    }),
                    radius: Box::new(make_float(5.0)),
                    span: dummy_span(),
                },
                parent: None,
                constraints: vec![],
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        let output = pp.print_program(&program);
        assert!(output.contains("region lobby"), "output: {}", output);
        assert!(output.contains("sphere("), "output: {}", output);
    }

    #[test]
    fn test_print_binary_op_precedence() {
        // (a + b) * c  should be printed as  (a + b) * c
        let expr = Expr::BinaryOp {
            op: BinOp::Mul,
            left: Box::new(Expr::BinaryOp {
                op: BinOp::Add,
                left: Box::new(make_id("a")),
                right: Box::new(make_id("b")),
                span: dummy_span(),
            }),
            right: Box::new(make_id("c")),
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        pp.print_expr(&expr);
        let output = pp.buffer.clone();
        assert!(
            output.contains("(a + b) * c"),
            "expected precedence-aware output, got: {}",
            output
        );
    }

    #[test]
    fn test_print_pattern_conjunction() {
        let pat = PatternExpr::Conjunction {
            patterns: vec![
                PatternExpr::Grab {
                    target: Box::new(make_id("obj")),
                    span: dummy_span(),
                },
                PatternExpr::Inside {
                    entity: Box::new(make_id("user")),
                    region: Box::new(make_id("zone")),
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        pp.print_pattern_expr(&pat);
        let output = pp.buffer.clone();
        assert!(output.contains("grab(obj)"), "output: {}", output);
        assert!(output.contains("and"), "output: {}", output);
        assert!(output.contains("inside(user, zone)"), "output: {}", output);
    }

    #[test]
    fn test_print_choreography_sequential() {
        let choreo = ChoreographyExpr::Sequential {
            steps: vec![
                ChoreographyExpr::Action {
                    actions: vec![ActionExpr::Activate {
                        target: Box::new(make_id("obj")),
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
                ChoreographyExpr::Action {
                    actions: vec![ActionExpr::Deactivate {
                        target: Box::new(make_id("obj")),
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        pp.print_choreography(&choreo);
        let output = pp.buffer.clone();
        assert!(output.contains("seq"), "output: {}", output);
        assert!(output.contains("activate(obj)"), "output: {}", output);
        assert!(output.contains("deactivate(obj)"), "output: {}", output);
    }

    #[test]
    fn test_print_import() {
        let program = Program {
            declarations: vec![Declaration::Import(ImportDecl {
                path: "std.spatial".to_string(),
                items: Some(vec!["Box".to_string(), "Sphere".to_string()]),
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        let output = pp.print_program(&program);
        assert!(output.contains("import"), "output: {}", output);
        assert!(output.contains("std.spatial"), "output: {}", output);
        assert!(output.contains("Box"), "output: {}", output);
    }

    #[test]
    fn test_print_duration_literal() {
        let expr = Expr::Literal(Literal::Duration(2.5, DurationUnit::S), dummy_span());
        let mut pp = PrettyPrinter::new();
        pp.print_expr(&expr);
        let output = pp.buffer.clone();
        assert!(output.contains("2.5s"), "output: {}", output);
    }

    #[test]
    fn test_print_entity_decl() {
        let program = Program {
            declarations: vec![Declaration::Entity(EntityDecl {
                name: "player".to_string(),
                entity_type: EntityTypeExpr::User,
                initial_position: Some(Expr::Vector3Literal {
                    x: Box::new(make_float(0.0)),
                    y: Box::new(make_float(1.0)),
                    z: Box::new(make_float(0.0)),
                    span: dummy_span(),
                }),
                initial_rotation: None,
                bounding_volume: None,
                properties: vec![],
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let mut pp = PrettyPrinter::new();
        let output = pp.print_program(&program);
        assert!(output.contains("entity player"), "output: {}", output);
        assert!(output.contains("User"), "output: {}", output);
        assert!(output.contains("position"), "output: {}", output);
    }
}
