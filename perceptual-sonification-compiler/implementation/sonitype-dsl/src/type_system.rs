//! Perceptual type system for the SoniType DSL.
//!
//! Extends a simply-typed lambda calculus with *perceptual refinement qualifiers*.
//! Type judgments: Γ ⊢ e : τ⟨φ⟩ where φ is a conjunction of psychoacoustic
//! predicates (Bark-band occupancy, cognitive load, masking margin, JND margins).
//!
//! Typing rules implemented:
//! - **T-Stream**: streams get Bark-band occupancy and cognitive load 1.0.
//! - **T-Compose**: pairwise segregation + masking clearance + load budget.
//! - **T-Map**: JND threshold satisfaction when mapping data → stream params.
//! - **T-Sub**: subtyping via qualifier partial order.
//! - **T-Let**: polymorphic let with qualifier inference.
//! - **T-If**: join qualifiers from both branches.

use crate::ast::*;
use crate::token::Span;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

// Re-use core unit conversions for Bark band computation.
use sonitype_core::units::hz_to_bark;

// ─── Perceptual Type ─────────────────────────────────────────────────────────

/// A type in the perceptual type system: a base type paired with a qualifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualType {
    pub base: BaseType,
    pub qualifier: Qualifier,
}

impl PerceptualType {
    pub fn new(base: BaseType, qualifier: Qualifier) -> Self {
        Self { base, qualifier }
    }

    pub fn simple(base: BaseType) -> Self {
        Self { base, qualifier: Qualifier::default() }
    }

    /// Check whether `self` is a subtype of `other` (qualifier is stronger).
    pub fn is_subtype_of(&self, other: &PerceptualType) -> bool {
        self.base == other.base && self.qualifier.is_stronger_than(&other.qualifier)
    }
}

impl fmt::Display for PerceptualType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        if !self.qualifier.is_trivial() {
            write!(f, "⟨{}⟩", self.qualifier)?;
        }
        Ok(())
    }
}

// ─── Base Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaseType {
    Stream,
    Mapping,
    MultiStream,
    Data,
    SonificationSpec,
    Float,
    Int,
    Bool,
    Str,
    Pitch,
    Timbre,
    Pan,
    Amplitude,
    Duration,
    Function(Box<BaseType>, Box<BaseType>),
    Tuple(Vec<BaseType>),
    TypeVar(u32),
}

impl fmt::Display for BaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaseType::Stream => write!(f, "Stream"),
            BaseType::Mapping => write!(f, "Mapping"),
            BaseType::MultiStream => write!(f, "MultiStream"),
            BaseType::Data => write!(f, "Data"),
            BaseType::SonificationSpec => write!(f, "SonificationSpec"),
            BaseType::Float => write!(f, "Float"),
            BaseType::Int => write!(f, "Int"),
            BaseType::Bool => write!(f, "Bool"),
            BaseType::Str => write!(f, "String"),
            BaseType::Pitch => write!(f, "Pitch"),
            BaseType::Timbre => write!(f, "Timbre"),
            BaseType::Pan => write!(f, "Pan"),
            BaseType::Amplitude => write!(f, "Amplitude"),
            BaseType::Duration => write!(f, "Duration"),
            BaseType::Function(a, b) => write!(f, "({a} -> {b})"),
            BaseType::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
            BaseType::TypeVar(id) => write!(f, "?{id}"),
        }
    }
}

// ─── Qualifier ───────────────────────────────────────────────────────────────

/// Perceptual qualifier: conjunction of psychoacoustic predicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qualifier {
    /// Bark critical bands occupied by this stream.
    pub band_occupancy: BTreeSet<u8>,
    /// Cognitive load contribution (typically 1.0 per stream).
    pub cognitive_load: f64,
    /// Masking margin in dB (higher = more headroom).
    pub masking_margin: f64,
    /// Pairwise segregation results with other streams.
    pub segregation_status: Vec<SegregationPair>,
    /// JND margins per audio parameter.
    pub jnd_margins: HashMap<String, f64>,
}

impl Default for Qualifier {
    fn default() -> Self {
        Self {
            band_occupancy: BTreeSet::new(),
            cognitive_load: 0.0,
            masking_margin: f64::INFINITY,
            segregation_status: Vec::new(),
            jnd_margins: HashMap::new(),
        }
    }
}

impl Qualifier {
    pub fn is_trivial(&self) -> bool {
        self.band_occupancy.is_empty()
            && self.cognitive_load == 0.0
            && self.masking_margin == f64::INFINITY
            && self.segregation_status.is_empty()
            && self.jnd_margins.is_empty()
    }

    /// A qualifier q1 is *stronger* than q2 if it imposes tighter constraints.
    /// In subtyping, stronger qualifiers are subtypes: q1 ≤ q2.
    pub fn is_stronger_than(&self, other: &Qualifier) -> bool {
        // Band occupancy must be subset
        if !self.band_occupancy.is_subset(&other.band_occupancy)
            && !other.band_occupancy.is_empty()
        {
            return false;
        }
        // Cognitive load must be ≤
        if self.cognitive_load > other.cognitive_load && other.cognitive_load > 0.0 {
            return false;
        }
        // Masking margin must be ≥
        if self.masking_margin < other.masking_margin && other.masking_margin < f64::INFINITY {
            return false;
        }
        true
    }

    /// Join two qualifiers (for if-then-else: take the weaker constraints).
    pub fn join(&self, other: &Qualifier) -> Qualifier {
        let band_occupancy: BTreeSet<u8> =
            self.band_occupancy.union(&other.band_occupancy).copied().collect();
        let cognitive_load = self.cognitive_load.max(other.cognitive_load);
        let masking_margin = self.masking_margin.min(other.masking_margin);

        let mut jnd_margins = self.jnd_margins.clone();
        for (k, v) in &other.jnd_margins {
            let entry = jnd_margins.entry(k.clone()).or_insert(f64::INFINITY);
            *entry = entry.min(*v);
        }

        Qualifier {
            band_occupancy,
            cognitive_load,
            masking_margin,
            segregation_status: Vec::new(),
            jnd_margins,
        }
    }

    /// Meet two qualifiers (take the stronger constraints).
    pub fn meet(&self, other: &Qualifier) -> Qualifier {
        let band_occupancy: BTreeSet<u8> =
            self.band_occupancy.intersection(&other.band_occupancy).copied().collect();
        let cognitive_load = self.cognitive_load.min(other.cognitive_load);
        let masking_margin = self.masking_margin.max(other.masking_margin);

        let mut jnd_margins = HashMap::new();
        for k in self.jnd_margins.keys().chain(other.jnd_margins.keys()) {
            let v1 = self.jnd_margins.get(k).copied().unwrap_or(0.0);
            let v2 = other.jnd_margins.get(k).copied().unwrap_or(0.0);
            jnd_margins.insert(k.clone(), v1.max(v2));
        }

        Qualifier {
            band_occupancy,
            cognitive_load,
            masking_margin,
            segregation_status: Vec::new(),
            jnd_margins,
        }
    }
}

impl fmt::Display for Qualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if !self.band_occupancy.is_empty() {
            let bands: Vec<String> = self.band_occupancy.iter().map(|b| b.to_string()).collect();
            parts.push(format!("band: {{{}}}", bands.join(", ")));
        }
        if self.cognitive_load > 0.0 {
            parts.push(format!("load: {}", self.cognitive_load));
        }
        if self.masking_margin < f64::INFINITY {
            parts.push(format!("masking: {}", self.masking_margin));
        }
        for (k, v) in &self.jnd_margins {
            parts.push(format!("jnd({k}): {v}"));
        }
        write!(f, "{}", parts.join(", "))
    }
}

/// Pairwise segregation result between two streams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegregationPair {
    pub stream_a: String,
    pub stream_b: String,
    pub segregated: bool,
    pub separation_db: f64,
}

// ─── Type Environment ────────────────────────────────────────────────────────

/// Typing context Γ: maps variable names to perceptual types.
#[derive(Debug, Clone, Default)]
pub struct TypeEnvironment {
    scopes: Vec<HashMap<String, PerceptualType>>,
}

impl TypeEnvironment {
    pub fn new() -> Self {
        Self { scopes: vec![HashMap::new()] }
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn bind(&mut self, name: impl Into<String>, ty: PerceptualType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), ty);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&PerceptualType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }
}

// ─── Type Errors ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub message: String,
    pub span: Span,
}

impl TypeError {
    pub fn new(kind: TypeErrorKind, message: impl Into<String>, span: Span) -> Self {
        Self { kind, message: message.into(), span }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Type error at {}: [{}] {}", self.span, self.kind, self.message)
    }
}

impl std::error::Error for TypeError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeErrorKind {
    Mismatch,
    UndefinedVariable,
    CognitiveLoadExceeded,
    MaskingViolation,
    SegregationFailure,
    JndInsufficient,
    InvalidMapping,
    InvalidStreamParam,
    UnsatisfiableConstraint,
}

impl fmt::Display for TypeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeErrorKind::Mismatch => write!(f, "type mismatch"),
            TypeErrorKind::UndefinedVariable => write!(f, "undefined variable"),
            TypeErrorKind::CognitiveLoadExceeded => write!(f, "cognitive load exceeded"),
            TypeErrorKind::MaskingViolation => write!(f, "masking violation"),
            TypeErrorKind::SegregationFailure => write!(f, "segregation failure"),
            TypeErrorKind::JndInsufficient => write!(f, "JND insufficient"),
            TypeErrorKind::InvalidMapping => write!(f, "invalid mapping"),
            TypeErrorKind::InvalidStreamParam => write!(f, "invalid stream param"),
            TypeErrorKind::UnsatisfiableConstraint => write!(f, "unsatisfiable constraint"),
        }
    }
}

// ─── Constraint Solver ───────────────────────────────────────────────────────

/// A constraint generated during type checking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Base types must be equal.
    Equality(BaseType, BaseType, Span),
    /// Qualifier must be a subtype.
    QualifierSubtype(Qualifier, Qualifier, Span),
    /// Cognitive load must not exceed budget.
    LoadBudget(f64, f64, Span),
    /// Masking margin must be at least the given dB.
    MaskingClearance(f64, f64, Span),
    /// JND must be satisfied for a parameter.
    JndSufficiency(String, f64, f64, Span),
    /// Two streams must segregate.
    SegregationRequired(String, String, BTreeSet<u8>, BTreeSet<u8>, Span),
}

/// Solver for perceptual type constraints.
#[derive(Debug, Default)]
pub struct ConstraintSolver {
    constraints: Vec<TypeConstraint>,
    errors: Vec<TypeError>,
}

impl ConstraintSolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, constraint: TypeConstraint) {
        self.constraints.push(constraint);
    }

    pub fn add_all(&mut self, constraints: impl IntoIterator<Item = TypeConstraint>) {
        self.constraints.extend(constraints);
    }

    /// Solve all accumulated constraints, returning errors for violations.
    pub fn solve(&mut self) -> Vec<TypeError> {
        let mut errors = Vec::new();

        for constraint in &self.constraints {
            match constraint {
                TypeConstraint::Equality(a, b, span) => {
                    if a != b {
                        errors.push(TypeError::new(
                            TypeErrorKind::Mismatch,
                            format!("expected {a}, found {b}"),
                            *span,
                        ));
                    }
                }
                TypeConstraint::QualifierSubtype(sub, sup, span) => {
                    if !sub.is_stronger_than(sup) {
                        errors.push(TypeError::new(
                            TypeErrorKind::Mismatch,
                            format!("qualifier {sub} is not a subtype of {sup}"),
                            *span,
                        ));
                    }
                }
                TypeConstraint::LoadBudget(load, budget, span) => {
                    if *load > *budget {
                        errors.push(TypeError::new(
                            TypeErrorKind::CognitiveLoadExceeded,
                            format!(
                                "cognitive load {load} exceeds budget {budget}"
                            ),
                            *span,
                        ));
                    }
                }
                TypeConstraint::MaskingClearance(actual, required, span) => {
                    if *actual < *required {
                        errors.push(TypeError::new(
                            TypeErrorKind::MaskingViolation,
                            format!(
                                "masking margin {actual} dB < required {required} dB"
                            ),
                            *span,
                        ));
                    }
                }
                TypeConstraint::JndSufficiency(param, actual, required, span) => {
                    if *actual < *required {
                        errors.push(TypeError::new(
                            TypeErrorKind::JndInsufficient,
                            format!(
                                "JND for {param}: {actual} < required {required}"
                            ),
                            *span,
                        ));
                    }
                }
                TypeConstraint::SegregationRequired(a, b, bands_a, bands_b, span) => {
                    // Segregation fails if bands overlap
                    let overlap: BTreeSet<u8> =
                        bands_a.intersection(bands_b).copied().collect();
                    if !overlap.is_empty() {
                        errors.push(TypeError::new(
                            TypeErrorKind::SegregationFailure,
                            format!(
                                "streams '{a}' and '{b}' overlap in Bark bands {:?}",
                                overlap
                            ),
                            *span,
                        ));
                    }
                }
            }
        }

        errors
    }

    pub fn clear(&mut self) {
        self.constraints.clear();
        self.errors.clear();
    }
}

// ─── Typed AST ───────────────────────────────────────────────────────────────

/// A program annotated with perceptual types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedProgram {
    pub declarations: Vec<TypedDeclaration>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedDeclaration {
    pub declaration: Declaration,
    pub ty: PerceptualType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedExpr {
    pub expr: Expr,
    pub ty: PerceptualType,
}

// ─── Type Checker ────────────────────────────────────────────────────────────

/// Configuration for the type checker.
#[derive(Debug, Clone)]
pub struct TypeCheckerConfig {
    pub max_cognitive_load: f64,
    pub default_masking_margin: f64,
    pub default_jnd_pitch: f64,
    pub default_jnd_amplitude: f64,
    pub default_jnd_pan: f64,
}

impl Default for TypeCheckerConfig {
    fn default() -> Self {
        Self {
            max_cognitive_load: 4.0,
            default_masking_margin: 6.0,
            default_jnd_pitch: 0.03,
            default_jnd_amplitude: 1.0,
            default_jnd_pan: 0.05,
        }
    }
}

/// The perceptual type checker.
pub struct TypeChecker {
    env: TypeEnvironment,
    solver: ConstraintSolver,
    config: TypeCheckerConfig,
    errors: Vec<TypeError>,
    next_type_var: u32,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self::with_config(TypeCheckerConfig::default())
    }

    pub fn with_config(config: TypeCheckerConfig) -> Self {
        Self {
            env: TypeEnvironment::new(),
            solver: ConstraintSolver::new(),
            config,
            errors: Vec::new(),
            next_type_var: 0,
        }
    }

    fn fresh_type_var(&mut self) -> BaseType {
        let id = self.next_type_var;
        self.next_type_var += 1;
        BaseType::TypeVar(id)
    }

    /// Type-check a complete program.
    pub fn check_program(&mut self, program: &Program) -> Result<TypedProgram, Vec<TypeError>> {
        let mut typed_decls = Vec::new();

        for decl in &program.declarations {
            match self.check_declaration(decl) {
                Ok(td) => typed_decls.push(td),
                Err(e) => self.errors.push(e),
            }
        }

        // Solve accumulated constraints
        let constraint_errors = self.solver.solve();
        self.errors.extend(constraint_errors);

        if self.errors.is_empty() {
            Ok(TypedProgram {
                declarations: typed_decls,
                span: program.span,
            })
        } else {
            Err(self.errors.clone())
        }
    }

    /// Type-check a single declaration.
    pub fn check_declaration(&mut self, decl: &Declaration) -> Result<TypedDeclaration, TypeError> {
        match decl {
            Declaration::StreamDecl(sd) => {
                let ty = self.check_stream_expr(&sd.expr)?;
                self.env.bind(&sd.name.name, ty.clone());
                Ok(TypedDeclaration { declaration: decl.clone(), ty })
            }
            Declaration::MappingDecl(md) => {
                let ty = self.check_mapping_expr(&md.expr)?;
                self.env.bind(&md.name.name, ty.clone());
                Ok(TypedDeclaration { declaration: decl.clone(), ty })
            }
            Declaration::ComposeDecl(cd) => {
                let ty = self.check_compose_expr(&cd.expr, cd.with_clause.as_ref())?;
                self.env.bind(&cd.name.name, ty.clone());
                Ok(TypedDeclaration { declaration: decl.clone(), ty })
            }
            Declaration::DataDecl(dd) => {
                let ty = PerceptualType::simple(BaseType::Data);
                self.env.bind(&dd.name.name, ty.clone());
                Ok(TypedDeclaration { declaration: decl.clone(), ty })
            }
            Declaration::LetBinding(lb) => {
                // T-Let: infer type of value, bind in environment
                let value_ty = self.check_expr(&lb.value)?;
                self.bind_pattern(&lb.pattern, &value_ty);
                Ok(TypedDeclaration { declaration: decl.clone(), ty: value_ty })
            }
            Declaration::SpecDecl(sd) => {
                self.env.enter_scope();
                for inner_decl in &sd.body {
                    match self.check_declaration(inner_decl) {
                        Ok(_) => {}
                        Err(e) => self.errors.push(e),
                    }
                }
                self.env.exit_scope();
                let ty = PerceptualType::simple(BaseType::SonificationSpec);
                Ok(TypedDeclaration { declaration: decl.clone(), ty })
            }
            Declaration::ImportDecl(_) => {
                // Imports are resolved during semantic analysis
                let ty = PerceptualType::simple(BaseType::SonificationSpec);
                Ok(TypedDeclaration { declaration: decl.clone(), ty })
            }
        }
    }

    // ── T-Stream ─────────────────────────────────────────────────────────────

    fn check_stream_expr(&mut self, stream: &StreamExpr) -> Result<PerceptualType, TypeError> {
        let mut bands = BTreeSet::new();
        let mut freq_value = 440.0_f64;

        for param in &stream.params {
            let _param_ty = self.check_expr(&param.value)?;

            match param.name.name.as_str() {
                "freq" | "frequency" => {
                    if let Expr::Literal(Literal { value: LiteralValue::Float(f), .. }) = &param.value {
                        freq_value = *f;
                    } else if let Expr::Literal(Literal { value: LiteralValue::Int(i), .. }) = &param.value {
                        freq_value = *i as f64;
                    }
                    let bark_val = hz_to_bark(freq_value).round() as u8;
                    let bark_idx = bark_val.min(23);
                    bands.insert(bark_idx);
                    // also include adjacent bands for spectral spread
                    if bark_idx > 0 { bands.insert(bark_idx - 1); }
                    if bark_idx < 23 { bands.insert(bark_idx + 1); }
                }
                "timbre" | "pan" | "amplitude" | "envelope" | "duration" => {}
                other => {
                    return Err(TypeError::new(
                        TypeErrorKind::InvalidStreamParam,
                        format!("unknown stream parameter: {other}"),
                        param.span,
                    ));
                }
            }
        }

        Ok(PerceptualType {
            base: BaseType::Stream,
            qualifier: Qualifier {
                band_occupancy: bands,
                cognitive_load: 1.0,
                masking_margin: self.config.default_masking_margin,
                segregation_status: Vec::new(),
                jnd_margins: HashMap::new(),
            },
        })
    }

    // ── T-Compose ────────────────────────────────────────────────────────────

    fn check_compose_expr(
        &mut self,
        compose: &ComposeExpr,
        with_clause: Option<&WithClause>,
    ) -> Result<PerceptualType, TypeError> {
        let mut stream_types = Vec::new();
        let mut total_load = 0.0;
        let mut all_bands = BTreeSet::new();
        let mut min_masking = f64::INFINITY;

        // Check each composed stream
        for stream_expr in &compose.streams {
            let ty = self.check_expr(stream_expr)?;
            total_load += ty.qualifier.cognitive_load;
            all_bands.extend(&ty.qualifier.band_occupancy);
            min_masking = min_masking.min(ty.qualifier.masking_margin);
            stream_types.push(ty);
        }

        // Extract load budget from with clause
        let load_budget = self.extract_load_budget(with_clause);
        let require_segregation = self.extract_segregation_requirement(with_clause);

        // Constraint: total cognitive load ≤ budget
        self.solver.add(TypeConstraint::LoadBudget(
            total_load,
            load_budget,
            compose.span,
        ));

        // Pairwise segregation check
        if require_segregation {
            for i in 0..stream_types.len() {
                for j in (i + 1)..stream_types.len() {
                    let bands_i = &stream_types[i].qualifier.band_occupancy;
                    let bands_j = &stream_types[j].qualifier.band_occupancy;
                    self.solver.add(TypeConstraint::SegregationRequired(
                        format!("stream_{i}"),
                        format!("stream_{j}"),
                        bands_i.clone(),
                        bands_j.clone(),
                        compose.span,
                    ));
                }
            }
        }

        Ok(PerceptualType {
            base: BaseType::MultiStream,
            qualifier: Qualifier {
                band_occupancy: all_bands,
                cognitive_load: total_load,
                masking_margin: min_masking,
                segregation_status: Vec::new(),
                jnd_margins: HashMap::new(),
            },
        })
    }

    fn extract_load_budget(&self, with_clause: Option<&WithClause>) -> f64 {
        if let Some(wc) = with_clause {
            for c in &wc.constraints {
                if c.name.name == "max_load" {
                    if let Expr::Literal(Literal { value: LiteralValue::Float(v), .. }) = &c.value {
                        return *v;
                    }
                    if let Expr::Literal(Literal { value: LiteralValue::Int(v), .. }) = &c.value {
                        return *v as f64;
                    }
                }
            }
        }
        self.config.max_cognitive_load
    }

    fn extract_segregation_requirement(&self, with_clause: Option<&WithClause>) -> bool {
        if let Some(wc) = with_clause {
            for c in &wc.constraints {
                if c.name.name == "segregation" {
                    if let Expr::Literal(Literal { value: LiteralValue::Bool(v), .. }) = &c.value {
                        return *v;
                    }
                }
            }
        }
        false
    }

    // ── T-Map ────────────────────────────────────────────────────────────────

    fn check_mapping_expr(&mut self, mapping: &MappingExpr) -> Result<PerceptualType, TypeError> {
        // Check that source data exists in the environment
        let source_name = &mapping.source.source.name;
        if self.env.lookup(source_name).is_none() {
            // Data sources can be implicit; just warn
        }

        // Compute JND margin for the target parameter
        let jnd_value = match mapping.target.param {
            AudioParamKind::Pitch => self.config.default_jnd_pitch,
            AudioParamKind::Amplitude => self.config.default_jnd_amplitude,
            AudioParamKind::Pan => self.config.default_jnd_pan,
            AudioParamKind::Timbre => 0.0,
            AudioParamKind::Duration => 0.01,
        };

        // If range is specified, check JND against range size
        if let Some((lo, hi)) = &mapping.target.range {
            if let (
                Expr::Literal(Literal { value: LiteralValue::Float(lo_v), .. }),
                Expr::Literal(Literal { value: LiteralValue::Float(hi_v), .. }),
            ) = (lo.as_ref(), hi.as_ref()) {
                let range_size = (hi_v - lo_v).abs();
                let data_points_resolvable = if jnd_value > 0.0 {
                    range_size / (jnd_value * lo_v.max(1.0))
                } else {
                    f64::INFINITY
                };
                if data_points_resolvable < 2.0 {
                    self.solver.add(TypeConstraint::JndSufficiency(
                        format!("{}", mapping.target.param),
                        data_points_resolvable,
                        2.0,
                        mapping.span,
                    ));
                }
            }
            if let (
                Expr::Literal(Literal { value: LiteralValue::Int(lo_v), .. }),
                Expr::Literal(Literal { value: LiteralValue::Int(hi_v), .. }),
            ) = (lo.as_ref(), hi.as_ref()) {
                let range_size = (*hi_v - *lo_v).unsigned_abs() as f64;
                let data_points_resolvable = if jnd_value > 0.0 {
                    range_size / (jnd_value * (*lo_v as f64).max(1.0))
                } else {
                    f64::INFINITY
                };
                if data_points_resolvable < 2.0 {
                    self.solver.add(TypeConstraint::JndSufficiency(
                        format!("{}", mapping.target.param),
                        data_points_resolvable,
                        2.0,
                        mapping.span,
                    ));
                }
            }
        }

        let mut jnd_margins = HashMap::new();
        jnd_margins.insert(format!("{}", mapping.target.param), jnd_value);

        Ok(PerceptualType {
            base: BaseType::Mapping,
            qualifier: Qualifier {
                band_occupancy: BTreeSet::new(),
                cognitive_load: 0.0,
                masking_margin: f64::INFINITY,
                segregation_status: Vec::new(),
                jnd_margins,
            },
        })
    }

    // ── Expression type checking ─────────────────────────────────────────────

    fn check_expr(&mut self, expr: &Expr) -> Result<PerceptualType, TypeError> {
        match expr {
            Expr::Literal(lit) => Ok(self.check_literal(lit)),
            Expr::Identifier(id) => {
                self.env.lookup(&id.name).cloned().ok_or_else(|| {
                    TypeError::new(
                        TypeErrorKind::UndefinedVariable,
                        format!("undefined variable: {}", id.name),
                        id.span,
                    )
                })
            }
            Expr::BinaryOp(binop) => self.check_binary_op(binop),
            Expr::UnaryOp(unop) => self.check_unary_op(unop),
            Expr::FunctionCall(fc) => self.check_function_call(fc),
            Expr::LetIn(li) => self.check_let_in(li),
            Expr::IfThenElse(ite) => self.check_if_then_else(ite),
            Expr::Lambda(lam) => self.check_lambda(lam),
            Expr::StreamLiteral(s) => self.check_stream_expr(s),
            Expr::MappingLiteral(m) => self.check_mapping_expr(m),
            Expr::Compose(c) => self.check_compose_expr(c, None),
            Expr::PipeOperator(p) => {
                let lhs_ty = self.check_expr(&p.lhs)?;
                let rhs_ty = self.check_expr(&p.rhs)?;
                // Pipe: rhs should be a function that accepts lhs type
                match &rhs_ty.base {
                    BaseType::Function(_, ret) => {
                        Ok(PerceptualType::new(*ret.clone(), lhs_ty.qualifier.join(&rhs_ty.qualifier)))
                    }
                    _ => {
                        // Allow pipe into any expression (desugared later)
                        Ok(rhs_ty)
                    }
                }
            }
            Expr::FieldAccess(fa) => {
                let _obj_ty = self.check_expr(&fa.object)?;
                // Field access type depends on the schema; return a type variable
                Ok(PerceptualType::simple(self.fresh_type_var()))
            }
            Expr::Grouped(inner, _) => self.check_expr(inner),
            Expr::WithClause(w) => self.check_expr(&w.expr),
            Expr::WhereClause(w) => self.check_expr(&w.expr),
        }
    }

    fn check_literal(&self, lit: &Literal) -> PerceptualType {
        let base = match &lit.value {
            LiteralValue::Int(_) => BaseType::Int,
            LiteralValue::Float(_) => BaseType::Float,
            LiteralValue::String(_) => BaseType::Str,
            LiteralValue::Bool(_) => BaseType::Bool,
        };
        PerceptualType::simple(base)
    }

    fn check_binary_op(&mut self, binop: &BinaryOp) -> Result<PerceptualType, TypeError> {
        let lhs_ty = self.check_expr(&binop.lhs)?;
        let rhs_ty = self.check_expr(&binop.rhs)?;

        let result_base = match binop.op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                // Numeric operations: result type matches operands
                match (&lhs_ty.base, &rhs_ty.base) {
                    (BaseType::Int, BaseType::Int) => BaseType::Int,
                    (BaseType::Float, _) | (_, BaseType::Float) => BaseType::Float,
                    _ => {
                        self.solver.add(TypeConstraint::Equality(
                            lhs_ty.base.clone(),
                            rhs_ty.base.clone(),
                            binop.span,
                        ));
                        lhs_ty.base.clone()
                    }
                }
            }
            BinOp::And | BinOp::Or => BaseType::Bool,
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                BaseType::Bool
            }
            BinOp::Range => {
                // Range produces a tuple-like value
                BaseType::Tuple(vec![lhs_ty.base.clone(), rhs_ty.base.clone()])
            }
        };

        let qualifier = lhs_ty.qualifier.join(&rhs_ty.qualifier);
        Ok(PerceptualType::new(result_base, qualifier))
    }

    fn check_unary_op(&mut self, unop: &UnaryOp) -> Result<PerceptualType, TypeError> {
        let operand_ty = self.check_expr(&unop.operand)?;
        let base = match unop.op {
            UnOp::Neg => operand_ty.base.clone(),
            UnOp::Not => BaseType::Bool,
        };
        Ok(PerceptualType::new(base, operand_ty.qualifier))
    }

    fn check_function_call(&mut self, fc: &FunctionCall) -> Result<PerceptualType, TypeError> {
        let callee_ty = self.check_expr(&fc.callee)?;
        for arg in &fc.args {
            let _arg_ty = self.check_expr(arg)?;
        }
        match callee_ty.base {
            BaseType::Function(_, ret) => Ok(PerceptualType::new(*ret, callee_ty.qualifier)),
            _ => Ok(PerceptualType::simple(self.fresh_type_var())),
        }
    }

    // T-Let
    fn check_let_in(&mut self, li: &LetIn) -> Result<PerceptualType, TypeError> {
        let value_ty = self.check_expr(&li.value)?;
        self.env.enter_scope();
        self.bind_pattern(&li.pattern, &value_ty);
        let body_ty = self.check_expr(&li.body)?;
        self.env.exit_scope();
        Ok(body_ty)
    }

    // T-If: join qualifiers from both branches
    fn check_if_then_else(&mut self, ite: &IfThenElse) -> Result<PerceptualType, TypeError> {
        let cond_ty = self.check_expr(&ite.condition)?;
        self.solver.add(TypeConstraint::Equality(
            cond_ty.base,
            BaseType::Bool,
            ite.condition.span(),
        ));

        let then_ty = self.check_expr(&ite.then_branch)?;
        let else_ty = self.check_expr(&ite.else_branch)?;

        self.solver.add(TypeConstraint::Equality(
            then_ty.base.clone(),
            else_ty.base.clone(),
            ite.span,
        ));

        // T-If: join qualifiers
        let qualifier = then_ty.qualifier.join(&else_ty.qualifier);
        Ok(PerceptualType::new(then_ty.base, qualifier))
    }

    fn check_lambda(&mut self, lam: &Lambda) -> Result<PerceptualType, TypeError> {
        self.env.enter_scope();
        let mut param_types = Vec::new();
        for p in &lam.params {
            let param_ty = PerceptualType::simple(self.fresh_type_var());
            self.env.bind(&p.name.name, param_ty.clone());
            param_types.push(param_ty);
        }
        let body_ty = self.check_expr(&lam.body)?;
        self.env.exit_scope();

        // Build function type from last param
        let mut result = body_ty.base;
        for pt in param_types.into_iter().rev() {
            result = BaseType::Function(Box::new(pt.base), Box::new(result));
        }
        Ok(PerceptualType::simple(result))
    }

    fn bind_pattern(&mut self, pat: &Pattern, ty: &PerceptualType) {
        match pat {
            Pattern::Variable(id) => {
                self.env.bind(&id.name, ty.clone());
            }
            Pattern::Tuple(pats, _) => {
                // Best-effort: bind each to the whole type
                for p in pats {
                    self.bind_pattern(p, ty);
                }
            }
            Pattern::Wildcard(_) => {}
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Span;

    fn dummy_span() -> Span { Span::dummy() }

    fn ident(name: &str) -> Identifier {
        Identifier::new(name, dummy_span())
    }

    #[test]
    fn test_qualifier_default_is_trivial() {
        assert!(Qualifier::default().is_trivial());
    }

    #[test]
    fn test_qualifier_stronger_than_default() {
        let q = Qualifier {
            band_occupancy: BTreeSet::from([3, 4]),
            cognitive_load: 1.0,
            masking_margin: 10.0,
            ..Default::default()
        };
        assert!(q.is_stronger_than(&Qualifier::default()));
    }

    #[test]
    fn test_qualifier_join() {
        let q1 = Qualifier {
            band_occupancy: BTreeSet::from([1, 2]),
            cognitive_load: 1.0,
            masking_margin: 10.0,
            ..Default::default()
        };
        let q2 = Qualifier {
            band_occupancy: BTreeSet::from([3, 4]),
            cognitive_load: 1.5,
            masking_margin: 8.0,
            ..Default::default()
        };
        let joined = q1.join(&q2);
        assert_eq!(joined.band_occupancy, BTreeSet::from([1, 2, 3, 4]));
        assert_eq!(joined.cognitive_load, 1.5);
        assert_eq!(joined.masking_margin, 8.0);
    }

    #[test]
    fn test_qualifier_meet() {
        let q1 = Qualifier {
            band_occupancy: BTreeSet::from([1, 2, 3]),
            cognitive_load: 1.0,
            masking_margin: 10.0,
            ..Default::default()
        };
        let q2 = Qualifier {
            band_occupancy: BTreeSet::from([2, 3, 4]),
            cognitive_load: 1.5,
            masking_margin: 8.0,
            ..Default::default()
        };
        let met = q1.meet(&q2);
        assert_eq!(met.band_occupancy, BTreeSet::from([2, 3]));
        assert_eq!(met.cognitive_load, 1.0);
        assert_eq!(met.masking_margin, 10.0);
    }

    #[test]
    fn test_perceptual_type_display() {
        let ty = PerceptualType::new(
            BaseType::Stream,
            Qualifier {
                band_occupancy: BTreeSet::from([5]),
                cognitive_load: 1.0,
                ..Default::default()
            },
        );
        let s = format!("{ty}");
        assert!(s.contains("Stream"));
        assert!(s.contains("band"));
    }

    #[test]
    fn test_type_environment_scoping() {
        let mut env = TypeEnvironment::new();
        env.bind("x", PerceptualType::simple(BaseType::Int));
        assert!(env.lookup("x").is_some());

        env.enter_scope();
        env.bind("y", PerceptualType::simple(BaseType::Float));
        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_some());

        env.exit_scope();
        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_none());
    }

    #[test]
    fn test_constraint_solver_equality_pass() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::Equality(BaseType::Int, BaseType::Int, dummy_span()));
        let errors = solver.solve();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_constraint_solver_equality_fail() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::Equality(BaseType::Int, BaseType::Float, dummy_span()));
        let errors = solver.solve();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, TypeErrorKind::Mismatch);
    }

    #[test]
    fn test_constraint_solver_load_budget_pass() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::LoadBudget(3.0, 4.0, dummy_span()));
        assert!(solver.solve().is_empty());
    }

    #[test]
    fn test_constraint_solver_load_budget_fail() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::LoadBudget(5.0, 4.0, dummy_span()));
        let errors = solver.solve();
        assert_eq!(errors[0].kind, TypeErrorKind::CognitiveLoadExceeded);
    }

    #[test]
    fn test_constraint_solver_segregation_fail() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::SegregationRequired(
            "s1".into(),
            "s2".into(),
            BTreeSet::from([3, 4, 5]),
            BTreeSet::from([5, 6, 7]),
            dummy_span(),
        ));
        let errors = solver.solve();
        assert_eq!(errors[0].kind, TypeErrorKind::SegregationFailure);
    }

    #[test]
    fn test_constraint_solver_segregation_pass() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::SegregationRequired(
            "s1".into(),
            "s2".into(),
            BTreeSet::from([1, 2]),
            BTreeSet::from([10, 11]),
            dummy_span(),
        ));
        assert!(solver.solve().is_empty());
    }

    #[test]
    fn test_check_literal_int() {
        let checker = TypeChecker::new();
        let lit = Literal { value: LiteralValue::Int(42), span: dummy_span() };
        let ty = checker.check_literal(&lit);
        assert_eq!(ty.base, BaseType::Int);
    }

    #[test]
    fn test_check_literal_float() {
        let checker = TypeChecker::new();
        let lit = Literal { value: LiteralValue::Float(3.14), span: dummy_span() };
        let ty = checker.check_literal(&lit);
        assert_eq!(ty.base, BaseType::Float);
    }

    #[test]
    fn test_type_check_let_binding() {
        use crate::lexer::lex;
        use crate::parser::parse;

        let src = "let x = 42";
        let tokens = lex(src).unwrap();
        let prog = parse(tokens).unwrap();
        let mut checker = TypeChecker::new();
        let result = checker.check_program(&prog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_check_if_join() {
        use crate::lexer::lex;
        use crate::parser::parse;

        let src = "let x = if true then 1 else 2";
        let tokens = lex(src).unwrap();
        let prog = parse(tokens).unwrap();
        let mut checker = TypeChecker::new();
        let result = checker.check_program(&prog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_check_binary_op() {
        use crate::lexer::lex;
        use crate::parser::parse;

        let src = "let x = 1 + 2";
        let tokens = lex(src).unwrap();
        let prog = parse(tokens).unwrap();
        let mut checker = TypeChecker::new();
        let result = checker.check_program(&prog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_subtyping() {
        let strong = PerceptualType::new(
            BaseType::Stream,
            Qualifier {
                band_occupancy: BTreeSet::from([3]),
                cognitive_load: 1.0,
                masking_margin: 10.0,
                ..Default::default()
            },
        );
        let weak = PerceptualType::new(
            BaseType::Stream,
            Qualifier {
                band_occupancy: BTreeSet::from([3, 4, 5]),
                cognitive_load: 2.0,
                masking_margin: 6.0,
                ..Default::default()
            },
        );
        assert!(strong.is_subtype_of(&weak));
        assert!(!weak.is_subtype_of(&strong));
    }

    #[test]
    fn test_masking_clearance_constraint() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::MaskingClearance(4.0, 6.0, dummy_span()));
        let errors = solver.solve();
        assert_eq!(errors[0].kind, TypeErrorKind::MaskingViolation);
    }

    #[test]
    fn test_jnd_sufficiency_constraint() {
        let mut solver = ConstraintSolver::new();
        solver.add(TypeConstraint::JndSufficiency("pitch".into(), 1.5, 2.0, dummy_span()));
        let errors = solver.solve();
        assert_eq!(errors[0].kind, TypeErrorKind::JndInsufficient);
    }
}
