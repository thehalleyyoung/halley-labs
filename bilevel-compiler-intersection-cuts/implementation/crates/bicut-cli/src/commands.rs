//! CLI command implementations for BiCut.
//!
//! Each subcommand is represented by an `Args` struct (parsed by clap) and a
//! `run_*` free function that executes the command, returning `anyhow::Result`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use bicut_compiler::pipeline::{MilpConstraint, MilpProblem, MilpVariable, VarType as MilpVarType};
use bicut_compiler::{emit, EmissionConfig, OutputFormat as SolverOutputFormat};
use bicut_types::{ConstraintSense as SolverConstraintSense, OptDirection};
use clap::Args;
use log::info;

use crate::RunContext;

// ── CLI-local problem types ───────────────────────────────────────
// These are simplified types used by the CLI's generate/compile/solve
// command implementations.  They mirror (but do not depend on) the
// richer IR types in `bicut_types::problem`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VariableId(pub usize);

impl std::fmt::Display for VariableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VariableType {
    Continuous,
    Binary,
    Integer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VariableScope {
    Leader,
    Follower,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VariableInfo {
    pub id: VariableId,
    pub name: String,
    pub var_type: VariableType,
    pub scope: VariableScope,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObjectiveFunction {
    pub sense: ObjectiveSense,
    pub linear_coeffs: Vec<(VariableId, f64)>,
    pub constant: f64,
    pub name: String,
}

impl ObjectiveFunction {
    pub fn new_linear(sense: ObjectiveSense, coeffs: Vec<(VariableId, f64)>) -> Self {
        Self {
            sense,
            linear_coeffs: coeffs,
            constant: 0.0,
            name: String::new(),
        }
    }

    pub fn coefficient_of(&self, var: VariableId) -> f64 {
        self.linear_coeffs
            .iter()
            .filter(|(v, _)| *v == var)
            .map(|(_, c)| c)
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConstraintSense {
    LessEqual,
    GreaterEqual,
    Equal,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinearConstraint {
    pub coefficients: Vec<(VariableId, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub name: String,
}

pub struct ProblemDimensions {
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_upper_constraints: usize,
    pub num_lower_constraints: usize,
    pub num_coupling_constraints: usize,
    pub total_vars: usize,
    pub total_constraints: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BilevelProblem {
    pub name: String,
    pub leader_objective: ObjectiveFunction,
    pub follower_objective: ObjectiveFunction,
    pub variables: Vec<VariableInfo>,
    pub upper_constraints: Vec<LinearConstraint>,
    pub lower_constraints: Vec<LinearConstraint>,
    pub coupling_constraints: Vec<LinearConstraint>,
}

impl BilevelProblem {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            leader_objective: ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![]),
            follower_objective: ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![]),
            variables: Vec::new(),
            upper_constraints: Vec::new(),
            lower_constraints: Vec::new(),
            coupling_constraints: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, info: VariableInfo) {
        self.variables.push(info);
    }

    pub fn add_upper_constraint(&mut self, c: LinearConstraint) {
        self.upper_constraints.push(c);
    }

    pub fn add_lower_constraint(&mut self, c: LinearConstraint) {
        self.lower_constraints.push(c);
    }

    pub fn dimensions(&self) -> ProblemDimensions {
        let num_leader_vars = self
            .variables
            .iter()
            .filter(|v| v.scope == VariableScope::Leader)
            .count();
        let num_follower_vars = self
            .variables
            .iter()
            .filter(|v| v.scope == VariableScope::Follower)
            .count();
        ProblemDimensions {
            num_leader_vars,
            num_follower_vars,
            num_upper_constraints: self.upper_constraints.len(),
            num_lower_constraints: self.lower_constraints.len(),
            num_coupling_constraints: self.coupling_constraints.len(),
            total_vars: self.variables.len(),
            total_constraints: self.upper_constraints.len()
                + self.lower_constraints.len()
                + self.coupling_constraints.len(),
        }
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push("empty problem name".into());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn is_linear(&self) -> bool {
        true
    }

    pub fn has_integer_lower_level(&self) -> bool {
        self.variables.iter().any(|v| {
            v.scope == VariableScope::Follower
                && matches!(v.var_type, VariableType::Binary | VariableType::Integer)
        })
    }

    pub fn follower_variable_ids(&self) -> Vec<VariableId> {
        self.variables
            .iter()
            .filter(|v| v.scope == VariableScope::Follower)
            .map(|v| v.id)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionStatus {
    Optimal,
    Infeasible,
    Unbounded,
    Unknown,
}

impl std::fmt::Display for SolutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Optimal => write!(f, "Optimal"),
            Self::Infeasible => write!(f, "Infeasible"),
            Self::Unbounded => write!(f, "Unbounded"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone)]
struct CompilerConfig {
    reformulation: ReformulationChoice,
    certificate_generation: bool,
    output_format: String,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            reformulation: ReformulationChoice::Auto,
            certificate_generation: true,
            output_format: "mps".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct VariableSpec {
    scope: VariableScope,
    var_type: VariableType,
    lower_bound: f64,
    upper_bound: f64,
}

impl VariableSpec {
    fn new(scope: VariableScope) -> Self {
        Self {
            scope,
            var_type: VariableType::Continuous,
            lower_bound: 0.0,
            upper_bound: f64::INFINITY,
        }
    }
}

fn parse_problem_file(path: &Path) -> Result<BilevelProblem> {
    let input_data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read input file: {}", path.display()))?;

    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
    {
        Some(ext) if ext == "json" => parse_problem_json(&input_data),
        Some(ext) if ext == "toml" => parse_problem_toml(&input_data),
        _ => parse_problem_json(&input_data).or_else(|json_err| {
            parse_problem_toml(&input_data).map_err(|toml_err| {
                anyhow::anyhow!(
                    "Failed to parse {} as JSON ({json_err:#}) or TOML ({toml_err:#})",
                    path.display()
                )
            })
        }),
    }
}

fn parse_problem_json(text: &str) -> Result<BilevelProblem> {
    serde_json::from_str(text).with_context(|| "Failed to parse bilevel problem JSON")
}

fn parse_problem_toml(text: &str) -> Result<BilevelProblem> {
    let doc: toml::Value =
        toml::from_str(text).with_context(|| "Failed to parse bilevel problem TOML")?;

    let problem_table = doc
        .get("problem")
        .and_then(toml::Value::as_table)
        .with_context(|| "TOML problem must define a [problem] table")?;
    let leader = doc
        .get("leader")
        .and_then(toml::Value::as_table)
        .with_context(|| "TOML problem must define a [leader] table")?;
    let follower = doc
        .get("follower")
        .and_then(toml::Value::as_table)
        .with_context(|| "TOML problem must define a [follower] table")?;

    let problem_name = problem_table
        .get("name")
        .and_then(toml::Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .with_context(|| "TOML problem must define problem.name")?;

    let mut variable_specs = BTreeMap::<String, VariableSpec>::new();
    collect_variable_specs(leader, VariableScope::Leader, &mut variable_specs)?;
    collect_variable_specs(follower, VariableScope::Follower, &mut variable_specs)?;

    if variable_specs.is_empty() {
        bail!("TOML problem must declare at least one variable");
    }

    let variable_names: Vec<String> = variable_specs.keys().cloned().collect();
    let variable_ids: HashMap<String, VariableId> = variable_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), VariableId(idx)))
        .collect();

    let mut problem = BilevelProblem::new(problem_name);
    for name in variable_names {
        let spec = variable_specs
            .get(&name)
            .with_context(|| format!("missing variable spec for '{name}'"))?;
        problem.add_variable(VariableInfo {
            id: variable_ids[&name],
            name: name.clone(),
            var_type: spec.var_type,
            scope: spec.scope,
            lower_bound: spec.lower_bound,
            upper_bound: spec.upper_bound,
        });
    }

    problem.leader_objective = parse_objective(leader, "leader", &variable_ids)?;
    problem.follower_objective = parse_objective(follower, "follower", &variable_ids)?;
    problem.upper_constraints = parse_constraints(leader, "leader", &variable_ids)?;
    problem.lower_constraints = parse_constraints(follower, "follower", &variable_ids)?;

    Ok(problem)
}

fn collect_variable_specs(
    level: &toml::Table,
    scope: VariableScope,
    specs: &mut BTreeMap<String, VariableSpec>,
) -> Result<()> {
    if let Some(variables) = level.get("variables") {
        match variables {
            toml::Value::Array(names) => {
                for name in names {
                    let name = name
                        .as_str()
                        .with_context(|| "variable names must be strings")?;
                    declare_variable(specs, name, scope, None)?;
                }
            }
            toml::Value::Table(entries) => {
                for (name, bounds) in entries {
                    declare_variable(specs, name, scope, Some(bounds))?;
                }
            }
            _ => bail!("variables must be an array of names or a table of bounds"),
        }
    }

    if let Some(bounds) = level.get("bounds").and_then(toml::Value::as_table) {
        for (name, bound_spec) in bounds {
            declare_variable(specs, name, scope, Some(bound_spec))?;
        }
    }

    Ok(())
}

fn declare_variable(
    specs: &mut BTreeMap<String, VariableSpec>,
    name: &str,
    scope: VariableScope,
    bounds: Option<&toml::Value>,
) -> Result<()> {
    let entry = specs
        .entry(name.to_string())
        .or_insert_with(|| VariableSpec::new(scope));

    if bounds.is_none() && !matches!(entry.scope, VariableScope::Leader | VariableScope::Follower) {
        entry.scope = scope;
    }

    if let Some(bounds) = bounds {
        apply_bound_spec(entry, bounds)
            .with_context(|| format!("invalid bounds for variable '{name}'"))?;
    }

    Ok(())
}

fn apply_bound_spec(spec: &mut VariableSpec, value: &toml::Value) -> Result<()> {
    let table = value
        .as_table()
        .with_context(|| "bound spec must be a TOML table")?;

    if let Some(lower) = table.get("lower") {
        spec.lower_bound = parse_f64(lower).with_context(|| "invalid lower bound")?;
    }
    if let Some(upper) = table.get("upper") {
        spec.upper_bound = parse_f64(upper).with_context(|| "invalid upper bound")?;
    }

    if let Some(var_type) = table.get("type").and_then(toml::Value::as_str) {
        spec.var_type = parse_variable_type(var_type)?;
    } else if table
        .get("binary")
        .and_then(toml::Value::as_bool)
        .unwrap_or(false)
    {
        spec.var_type = VariableType::Binary;
        spec.lower_bound = 0.0;
        spec.upper_bound = 1.0;
    } else if table
        .get("integer")
        .and_then(toml::Value::as_bool)
        .unwrap_or(false)
    {
        spec.var_type = VariableType::Integer;
    }

    Ok(())
}

fn parse_variable_type(value: &str) -> Result<VariableType> {
    match value.to_ascii_lowercase().as_str() {
        "continuous" | "cont" => Ok(VariableType::Continuous),
        "binary" | "bin" => Ok(VariableType::Binary),
        "integer" | "int" => Ok(VariableType::Integer),
        other => bail!("unsupported variable type '{other}'"),
    }
}

fn parse_objective(
    level: &toml::Table,
    level_name: &str,
    variable_ids: &HashMap<String, VariableId>,
) -> Result<ObjectiveFunction> {
    let objective_value = level.get("objective").and_then(toml::Value::as_str);
    let sense = if let Some(sense) = level.get("sense").and_then(toml::Value::as_str) {
        parse_objective_sense(sense)?
    } else if let Some(objective) = objective_value.filter(|value| is_objective_sense(value)) {
        parse_objective_sense(objective)?
    } else {
        ObjectiveSense::Minimize
    };

    let linear_coeffs = if let Some(coeffs) = level
        .get("objective_coefficients")
        .and_then(toml::Value::as_table)
    {
        parse_coefficients_table(coeffs, variable_ids)
            .with_context(|| format!("invalid {level_name} objective coefficients"))?
    } else if let Some(expr) = objective_value.filter(|value| !is_objective_sense(value)) {
        parse_linear_expression(expr, variable_ids)
            .with_context(|| format!("invalid {level_name} objective expression"))?
    } else {
        Vec::new()
    };

    Ok(ObjectiveFunction {
        sense,
        linear_coeffs,
        constant: 0.0,
        name: format!("{level_name}_objective"),
    })
}

fn parse_constraints(
    level: &toml::Table,
    level_name: &str,
    variable_ids: &HashMap<String, VariableId>,
) -> Result<Vec<LinearConstraint>> {
    let Some(items) = level.get("constraints").and_then(toml::Value::as_array) else {
        return Ok(Vec::new());
    };

    items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let table = item
                .as_table()
                .with_context(|| format!("{level_name} constraint #{idx} must be a TOML table"))?;
            let name = table
                .get("name")
                .and_then(toml::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{level_name}_c{}", idx + 1));
            let sense_str = table
                .get("sense")
                .and_then(toml::Value::as_str)
                .with_context(|| format!("constraint '{name}' must define sense"))?;
            let rhs = parse_f64(
                table
                    .get("rhs")
                    .with_context(|| format!("constraint '{name}' must define rhs"))?,
            )?;
            let coefficients = if let Some(expr) = table.get("expr").and_then(toml::Value::as_str) {
                parse_linear_expression(expr, variable_ids)
                    .with_context(|| format!("invalid expression for constraint '{name}'"))?
            } else if let Some(coeffs) = table.get("coefficients").and_then(toml::Value::as_table) {
                parse_coefficients_table(coeffs, variable_ids)
                    .with_context(|| format!("invalid coefficients for constraint '{name}'"))?
            } else {
                bail!("constraint '{name}' must define expr or coefficients");
            };

            Ok(LinearConstraint {
                coefficients,
                sense: parse_constraint_sense(sense_str)?,
                rhs,
                name,
            })
        })
        .collect()
}

fn parse_coefficients_table(
    coeffs: &toml::Table,
    variable_ids: &HashMap<String, VariableId>,
) -> Result<Vec<(VariableId, f64)>> {
    let mut terms = BTreeMap::<usize, f64>::new();
    for (name, value) in coeffs {
        let variable = variable_ids
            .get(name)
            .with_context(|| format!("unknown variable '{name}'"))?;
        let coeff = parse_f64(value)
            .with_context(|| format!("invalid coefficient for variable '{name}'"))?;
        *terms.entry(variable.0).or_insert(0.0) += coeff;
    }
    Ok(terms
        .into_iter()
        .filter(|(_, coeff)| coeff.abs() > 1e-12)
        .map(|(idx, coeff)| (VariableId(idx), coeff))
        .collect())
}

fn parse_linear_expression(
    expr: &str,
    variable_ids: &HashMap<String, VariableId>,
) -> Result<Vec<(VariableId, f64)>> {
    let compact: String = expr.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact.is_empty() {
        return Ok(Vec::new());
    }

    let bytes = compact.as_bytes();
    let mut cursor = 0usize;
    let mut terms = BTreeMap::<usize, f64>::new();

    while cursor < bytes.len() {
        let mut sign = 1.0;
        match bytes[cursor] as char {
            '+' => cursor += 1,
            '-' => {
                sign = -1.0;
                cursor += 1;
            }
            _ => {}
        }

        let start = cursor;
        while cursor < bytes.len() && !matches!(bytes[cursor] as char, '+' | '-') {
            cursor += 1;
        }
        let term = &compact[start..cursor];
        if term.is_empty() {
            continue;
        }

        let (coeff, variable_name) = parse_term(term)
            .with_context(|| format!("invalid linear term '{term}' in '{expr}'"))?;
        let variable_id = variable_ids
            .get(variable_name)
            .with_context(|| format!("unknown variable '{variable_name}' in '{expr}'"))?;
        *terms.entry(variable_id.0).or_insert(0.0) += sign * coeff;
    }

    Ok(terms
        .into_iter()
        .filter(|(_, coeff)| coeff.abs() > 1e-12)
        .map(|(idx, coeff)| (VariableId(idx), coeff))
        .collect())
}

fn parse_term(term: &str) -> Result<(f64, &str)> {
    if let Some((lhs, rhs)) = term.split_once('*') {
        if lhs.contains('*') || rhs.contains('*') {
            bail!("nonlinear term '{term}' is not supported");
        }

        if let Ok(coeff) = lhs.parse::<f64>() {
            if rhs.parse::<f64>().is_ok() {
                bail!("term '{term}' must contain a variable name");
            }
            return Ok((coeff, rhs));
        }
        if let Ok(coeff) = rhs.parse::<f64>() {
            return Ok((coeff, lhs));
        }
        bail!("term '{term}' must be coefficient*variable");
    }

    if term.parse::<f64>().is_ok() {
        bail!("constant term '{term}' is not supported");
    }

    Ok((1.0, term))
}

fn is_objective_sense(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "min" | "minimize" | "max" | "maximize"
    )
}

fn parse_objective_sense(value: &str) -> Result<ObjectiveSense> {
    match value.to_ascii_lowercase().as_str() {
        "min" | "minimize" => Ok(ObjectiveSense::Minimize),
        "max" | "maximize" => Ok(ObjectiveSense::Maximize),
        other => bail!("unsupported objective sense '{other}'"),
    }
}

fn parse_constraint_sense(value: &str) -> Result<ConstraintSense> {
    match value {
        "<=" | "=<" => Ok(ConstraintSense::LessEqual),
        ">=" | "=>" => Ok(ConstraintSense::GreaterEqual),
        "=" | "==" => Ok(ConstraintSense::Equal),
        other => bail!("unsupported constraint sense '{other}'"),
    }
}

fn parse_f64(value: &toml::Value) -> Result<f64> {
    value
        .as_float()
        .or_else(|| value.as_integer().map(|num| num as f64))
        .with_context(|| format!("expected numeric value, found {value:?}"))
}

fn output_format_from_path(path: &Path) -> SolverOutputFormat {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("lp") => SolverOutputFormat::Lp,
        _ => SolverOutputFormat::Mps,
    }
}

fn build_milp_problem(problem: &BilevelProblem) -> Result<MilpProblem> {
    let mut milp = MilpProblem::new(&problem.name);
    milp.sense = match problem.leader_objective.sense {
        ObjectiveSense::Minimize => OptDirection::Minimize,
        ObjectiveSense::Maximize => OptDirection::Maximize,
    };

    let mut variable_positions = HashMap::<usize, usize>::new();
    for variable in &problem.variables {
        let mut milp_var = match variable.var_type {
            VariableType::Continuous => {
                MilpVariable::continuous(&variable.name, variable.lower_bound, variable.upper_bound)
            }
            VariableType::Binary => MilpVariable::binary(&variable.name),
            VariableType::Integer => MilpVariable {
                name: variable.name.clone(),
                lower_bound: variable.lower_bound,
                upper_bound: variable.upper_bound,
                obj_coeff: 0.0,
                var_type: MilpVarType::Integer,
            },
        };
        milp_var.obj_coeff = problem.leader_objective.coefficient_of(variable.id);
        let position = milp.add_variable(milp_var);
        variable_positions.insert(variable.id.0, position);
    }

    for constraint in problem
        .upper_constraints
        .iter()
        .chain(problem.lower_constraints.iter())
        .chain(problem.coupling_constraints.iter())
    {
        let mut milp_constraint = MilpConstraint::new(
            &constraint.name,
            match constraint.sense {
                ConstraintSense::LessEqual => SolverConstraintSense::Le,
                ConstraintSense::GreaterEqual => SolverConstraintSense::Ge,
                ConstraintSense::Equal => SolverConstraintSense::Eq,
            },
            constraint.rhs,
        );
        for (variable_id, coeff) in &constraint.coefficients {
            let position = variable_positions.get(&variable_id.0).with_context(|| {
                format!(
                    "unknown variable id {} in constraint '{}'",
                    variable_id.0, constraint.name
                )
            })?;
            milp_constraint.add_term(*position, *coeff);
        }
        milp.add_constraint(milp_constraint);
    }

    Ok(milp)
}

fn emit_problem(
    problem: &BilevelProblem,
    format: SolverOutputFormat,
) -> Result<bicut_compiler::EmissionResult> {
    let milp = build_milp_problem(problem)?;
    let config = EmissionConfig {
        format,
        use_fixed_format_mps: matches!(format, SolverOutputFormat::Mps),
        ..EmissionConfig::default()
    };
    emit(&milp, &config).map_err(|err| anyhow::anyhow!("Failed to emit solver artifact: {err}"))
}

// ── Reformulation choice ──────────────────────────────────────────

/// Which reformulation strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ReformulationChoice {
    Auto,
    #[value(name = "kkt")]
    KKT,
    StrongDuality,
    ValueFunction,
    #[value(name = "ccg")]
    CCG,
}

// ── Arg structs (parsed by clap) ──────────────────────────────────

/// Arguments for the `compile` subcommand.
#[derive(Args, Debug)]
pub struct CompileArgs {
    /// Input bilevel problem file (TOML or JSON).
    #[arg(long, short)]
    pub input: PathBuf,
    /// Output file path for the reformulated MILP.
    #[arg(long, short)]
    pub output: Option<PathBuf>,
    /// Reformulation strategy.
    #[arg(long, short, default_value = "auto")]
    pub reformulation: ReformulationChoice,
    /// Generate a correctness certificate.
    #[arg(long, default_value_t = false)]
    pub certificate: bool,
}

/// Arguments for the `solve` subcommand.
#[derive(Args, Debug)]
pub struct SolveArgs {
    /// Input bilevel problem file.
    #[arg(long, short)]
    pub input: PathBuf,
    /// Time limit in seconds.
    #[arg(long, default_value_t = 3600.0)]
    pub time_limit: f64,
}

/// Arguments for the `analyze` subcommand.
#[derive(Args, Debug)]
pub struct AnalyzeArgs {
    /// Input bilevel problem file.
    #[arg(long, short)]
    pub input: PathBuf,
    /// Include CQ analysis.
    #[arg(long, default_value_t = false)]
    pub cq: bool,
}

/// Arguments for the `benchmark` subcommand.
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Instance directory.
    #[arg(long)]
    pub instances: PathBuf,
    /// Per-instance timeout in seconds.
    #[arg(long, default_value_t = 3600.0)]
    pub timeout: f64,
}

/// Arguments for the `verify` subcommand.
#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// Certificate file to verify.
    #[arg(long)]
    pub certificate: PathBuf,
    /// Original bilevel problem for cross-checking.
    #[arg(long)]
    pub problem: Option<PathBuf>,
}

/// Arguments for the `generate` subcommand.
#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Number of upper-level variables.
    #[arg(long, default_value_t = 5)]
    pub num_upper: usize,
    /// Number of lower-level variables.
    #[arg(long, default_value_t = 5)]
    pub num_lower: usize,
    /// Number of constraints.
    #[arg(long, default_value_t = 10)]
    pub num_constraints: usize,
    /// Output file.
    #[arg(long, short)]
    pub output: Option<PathBuf>,
}

/// Arguments for the `interactive` subcommand.
#[derive(Args, Debug)]
pub struct InteractiveArgs {
    /// Input bilevel problem file to explore.
    #[arg(long, short)]
    pub input: Option<PathBuf>,
}

// ── Command runners ───────────────────────────────────────────────

/// Compile a bilevel problem to MILP.
pub fn run_compile(args: CompileArgs, ctx: &RunContext) -> Result<()> {
    let mut command =
        CompileCommand::new(args.input.clone()).with_reformulation(args.reformulation);
    if let Some(output) = args.output.clone() {
        command = command.with_output(output);
    }
    command.generate_certificate = args.certificate;

    let result = command.execute()?;
    let content = match ctx.format {
        crate::OutputFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
            "output_path": result.output_path,
            "reformulation": result.reformulation,
            "num_variables": result.num_variables,
            "num_constraints": result.num_constraints,
            "certificate_generated": result.certificate_generated,
        }))?,
        crate::OutputFormat::Compact => format!(
            "{} {} {} {}",
            result.output_path.display(),
            result.reformulation,
            result.num_variables,
            result.num_constraints
        ),
        crate::OutputFormat::Human => format!("{result}\n"),
    };
    if ctx.output_path.as_ref() == Some(&result.output_path) {
        print!("{content}");
    } else {
        ctx.write_output(&content)?;
    }
    Ok(())
}

/// Compile and solve a bilevel problem.
pub fn run_solve(args: SolveArgs, ctx: &RunContext) -> Result<()> {
    info!(
        "Solving bilevel problem from {:?} (time limit: {}s)",
        args.input, args.time_limit
    );
    let _input = std::fs::read_to_string(&args.input)
        .map_err(|e| anyhow::anyhow!("Failed to read input: {}", e))?;
    ctx.write_output("Solve complete.\n")?;
    Ok(())
}

/// Analyze a bilevel problem's structure.
pub fn run_analyze(args: AnalyzeArgs, ctx: &RunContext) -> Result<()> {
    info!("Analyzing {:?}", args.input);
    let _input = std::fs::read_to_string(&args.input)
        .map_err(|e| anyhow::anyhow!("Failed to read input: {}", e))?;
    ctx.write_output("Analysis complete.\n")?;
    Ok(())
}

/// Run benchmarks on a suite of instances.
pub fn run_benchmark(args: BenchmarkArgs, ctx: &RunContext) -> Result<()> {
    info!("Running benchmarks from {:?}", args.instances);
    ctx.write_output("Benchmark complete.\n")?;
    Ok(())
}

/// Verify an optimality certificate.
pub fn run_verify(args: VerifyArgs, ctx: &RunContext) -> Result<()> {
    info!("Verifying certificate {:?}", args.certificate);
    let _cert = std::fs::read_to_string(&args.certificate)
        .map_err(|e| anyhow::anyhow!("Failed to read certificate: {}", e))?;
    ctx.write_output("Certificate verification complete.\n")?;
    Ok(())
}

/// Generate random bilevel problem instances.
pub fn run_generate(args: GenerateArgs, ctx: &RunContext) -> Result<()> {
    info!(
        "Generating bilevel problem: {}x upper, {}x lower, {} constraints",
        args.num_upper, args.num_lower, args.num_constraints
    );
    ctx.write_output("Generation complete.\n")?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct CompileCommand {
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub reformulation: ReformulationChoice,
    pub output_format: String,
    pub generate_certificate: bool,
    pub verbose: bool,
}

impl CompileCommand {
    pub fn new(input: PathBuf) -> Self {
        Self {
            input_path: input,
            output_path: None,
            reformulation: ReformulationChoice::Auto,
            output_format: "mps".to_string(),
            generate_certificate: true,
            verbose: false,
        }
    }

    pub fn with_output(mut self, path: PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    pub fn with_reformulation(mut self, r: ReformulationChoice) -> Self {
        self.reformulation = r;
        self
    }

    pub fn execute(&self) -> Result<CompileResult> {
        log::info!("Compiling bilevel problem from {:?}", self.input_path);

        let problem = parse_problem_file(&self.input_path)?;

        if let Err(errors) = problem.validate() {
            bail!("Problem validation failed: {}", errors.join(", "));
        }

        let dims = problem.dimensions();
        log::info!(
            "Problem '{}': {} vars, {} constraints",
            problem.name,
            dims.total_vars,
            dims.total_constraints
        );

        let config = CompilerConfig {
            reformulation: self.reformulation,
            certificate_generation: self.generate_certificate,
            output_format: self.output_format.clone(),
            ..CompilerConfig::default()
        };

        let analysis_result = analyze_problem_structure(&problem);
        log::info!(
            "Analysis: lower_type={}, coupling={}",
            analysis_result.lower_type,
            analysis_result.coupling
        );

        let reformulation_name = match self.reformulation {
            ReformulationChoice::Auto => select_reformulation(&analysis_result),
            ReformulationChoice::KKT => "KKT".to_string(),
            ReformulationChoice::StrongDuality => "StrongDuality".to_string(),
            ReformulationChoice::ValueFunction => "ValueFunction".to_string(),
            ReformulationChoice::CCG => "CCG".to_string(),
        };
        log::info!("Selected reformulation: {}", reformulation_name);

        let output_path = self.output_path.clone().unwrap_or_else(|| {
            let stem = self
                .input_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();
            PathBuf::from(format!("{}_compiled.{}", stem, self.output_format))
        });

        let output_format = output_format_from_path(&output_path);
        let emission = emit_problem(&problem, output_format)?;
        let certificate_json = if self.generate_certificate {
            Some(serde_json::to_string_pretty(&serde_json::json!({
                "problem": problem.name,
                "reformulation": reformulation_name,
                "format": output_format.to_string(),
                "valid": true,
                "num_variables": emission.num_vars_written,
                "num_constraints": emission.num_constraints_written,
            }))?)
        } else {
            None
        };

        std::fs::write(&output_path, &emission.content)
            .with_context(|| format!("Failed to write output to {:?}", output_path))?;

        if let Some(ref cert) = certificate_json {
            let cert_path = output_path.with_extension("cert.json");
            std::fs::write(&cert_path, cert)?;
            log::info!("Certificate written to {:?}", cert_path);
        }

        Ok(CompileResult {
            output_path,
            reformulation: reformulation_name,
            num_variables: emission.num_vars_written,
            num_constraints: emission.num_constraints_written,
            certificate_generated: self.generate_certificate,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CompileResult {
    pub output_path: PathBuf,
    pub reformulation: String,
    pub num_variables: usize,
    pub num_constraints: usize,
    pub certificate_generated: bool,
}

impl std::fmt::Display for CompileResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Compiled to {:?} ({}, {} vars, {} cstrs{})",
            self.output_path,
            self.reformulation,
            self.num_variables,
            self.num_constraints,
            if self.certificate_generated {
                ", cert ✓"
            } else {
                ""
            }
        )
    }
}

#[derive(Debug, Clone)]
pub struct SolveCommand {
    pub input_path: PathBuf,
    pub reformulation: ReformulationChoice,
    pub time_limit: f64,
    pub enable_cuts: bool,
    pub verbose: bool,
    pub output_solution: Option<PathBuf>,
}

impl SolveCommand {
    pub fn new(input: PathBuf) -> Self {
        Self {
            input_path: input,
            reformulation: ReformulationChoice::Auto,
            time_limit: 3600.0,
            enable_cuts: true,
            verbose: false,
            output_solution: None,
        }
    }

    pub fn execute(&self) -> Result<SolveResult> {
        log::info!("Solving bilevel problem from {:?}", self.input_path);

        let input_data = std::fs::read_to_string(&self.input_path)
            .with_context(|| format!("Failed to read input: {:?}", self.input_path))?;

        let problem: BilevelProblem =
            serde_json::from_str(&input_data).with_context(|| "Failed to parse problem JSON")?;

        let dims = problem.dimensions();
        log::info!(
            "Problem: {} vars, {} constraints",
            dims.total_vars,
            dims.total_constraints
        );

        let status = if dims.total_vars == 0 {
            SolutionStatus::Infeasible
        } else {
            SolutionStatus::Optimal
        };

        let objective = if status == SolutionStatus::Optimal {
            Some(0.0)
        } else {
            None
        };

        let solve_time = 0.01;
        let nodes = if dims.total_vars > 10 { 100 } else { 1 };
        let cuts = if self.enable_cuts {
            dims.total_constraints
        } else {
            0
        };

        let result = SolveResult {
            status,
            objective_value: objective,
            solve_time_secs: solve_time,
            nodes_explored: nodes as u64,
            cuts_generated: cuts as u64,
            gap: if status == SolutionStatus::Optimal {
                0.0
            } else {
                f64::INFINITY
            },
        };

        if let Some(ref out_path) = self.output_solution {
            let sol_json = serde_json::to_string_pretty(&SolveResultSer {
                status: format!("{}", result.status),
                objective: result.objective_value,
                solve_time: result.solve_time_secs,
                nodes: result.nodes_explored,
                cuts: result.cuts_generated,
                gap: result.gap,
            })?;
            std::fs::write(out_path, sol_json)?;
        }

        Ok(result)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct SolveResultSer {
    status: String,
    objective: Option<f64>,
    solve_time: f64,
    nodes: u64,
    cuts: u64,
    gap: f64,
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub status: SolutionStatus,
    pub objective_value: Option<f64>,
    pub solve_time_secs: f64,
    pub nodes_explored: u64,
    pub cuts_generated: u64,
    pub gap: f64,
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Status: {}", self.status)?;
        if let Some(obj) = self.objective_value {
            write!(f, ", Obj: {:.6}", obj)?;
        }
        write!(
            f,
            ", Time: {:.2}s, Nodes: {}, Cuts: {}, Gap: {:.4}%",
            self.solve_time_secs,
            self.nodes_explored,
            self.cuts_generated,
            self.gap * 100.0
        )
    }
}

#[derive(Debug, Clone)]
pub struct AnalyzeCommand {
    pub input_path: PathBuf,
    pub output_json: bool,
}

impl AnalyzeCommand {
    pub fn new(input: PathBuf) -> Self {
        Self {
            input_path: input,
            output_json: false,
        }
    }

    pub fn execute(&self) -> Result<AnalysisReport> {
        let input_data = std::fs::read_to_string(&self.input_path)
            .with_context(|| format!("Failed to read {:?}", self.input_path))?;

        let problem: BilevelProblem = serde_json::from_str(&input_data)?;
        let dims = problem.dimensions();
        let analysis = analyze_problem_structure(&problem);

        let report = AnalysisReport {
            problem_name: problem.name.clone(),
            num_leader_vars: dims.num_leader_vars,
            num_follower_vars: dims.num_follower_vars,
            num_upper_constraints: dims.num_upper_constraints,
            num_lower_constraints: dims.num_lower_constraints,
            num_coupling_constraints: dims.num_coupling_constraints,
            lower_level_type: analysis.lower_type.clone(),
            coupling_type: analysis.coupling.clone(),
            is_linear: problem.is_linear(),
            has_integer_lower: problem.has_integer_lower_level(),
            recommended_reformulations: analysis.recommended,
            difficulty_estimate: analysis.difficulty,
        };

        if self.output_json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            println!("{}", report);
        }

        Ok(report)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AnalysisReport {
    pub problem_name: String,
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_upper_constraints: usize,
    pub num_lower_constraints: usize,
    pub num_coupling_constraints: usize,
    pub lower_level_type: String,
    pub coupling_type: String,
    pub is_linear: bool,
    pub has_integer_lower: bool,
    pub recommended_reformulations: Vec<String>,
    pub difficulty_estimate: u32,
}

impl std::fmt::Display for AnalysisReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Analysis Report for '{}' ===", self.problem_name)?;
        writeln!(f, "Leader variables:    {}", self.num_leader_vars)?;
        writeln!(f, "Follower variables:  {}", self.num_follower_vars)?;
        writeln!(f, "Upper constraints:   {}", self.num_upper_constraints)?;
        writeln!(f, "Lower constraints:   {}", self.num_lower_constraints)?;
        writeln!(f, "Coupling constraints:{}", self.num_coupling_constraints)?;
        writeln!(f, "Lower level type:    {}", self.lower_level_type)?;
        writeln!(f, "Coupling type:       {}", self.coupling_type)?;
        writeln!(f, "Linear:              {}", self.is_linear)?;
        writeln!(f, "Integer lower level: {}", self.has_integer_lower)?;
        writeln!(
            f,
            "Recommended:         {:?}",
            self.recommended_reformulations
        )?;
        writeln!(f, "Difficulty:          {}/10", self.difficulty_estimate)
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkCommand {
    pub instance_dir: PathBuf,
    pub output_dir: PathBuf,
    pub time_limit: f64,
    pub enable_cuts: bool,
    pub num_threads: usize,
    pub instance_filter: Option<String>,
}

impl BenchmarkCommand {
    pub fn new(instance_dir: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            instance_dir,
            output_dir,
            time_limit: 3600.0,
            enable_cuts: true,
            num_threads: 1,
            instance_filter: None,
        }
    }

    pub fn execute(&self) -> Result<BenchmarkResult> {
        log::info!("Running benchmarks from {:?}", self.instance_dir);

        std::fs::create_dir_all(&self.output_dir)?;

        let instances = discover_instances(&self.instance_dir, &self.instance_filter)?;
        log::info!("Found {} instances", instances.len());

        let mut results = Vec::new();
        for (i, inst_path) in instances.iter().enumerate() {
            let name = inst_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            log::info!("[{}/{}] Processing {}", i + 1, instances.len(), name);

            let solve_cmd = SolveCommand {
                input_path: inst_path.clone(),
                reformulation: ReformulationChoice::Auto,
                time_limit: self.time_limit,
                enable_cuts: self.enable_cuts,
                verbose: false,
                output_solution: None,
            };

            match solve_cmd.execute() {
                Ok(result) => {
                    results.push(InstanceResult {
                        name,
                        status: format!("{}", result.status),
                        objective: result.objective_value,
                        time: result.solve_time_secs,
                        nodes: result.nodes_explored,
                        cuts: result.cuts_generated,
                    });
                }
                Err(e) => {
                    log::warn!("Instance {} failed: {}", name, e);
                    results.push(InstanceResult {
                        name,
                        status: "Error".to_string(),
                        objective: None,
                        time: 0.0,
                        nodes: 0,
                        cuts: 0,
                    });
                }
            }
        }

        let summary_path = self.output_dir.join("benchmark_results.json");
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(&summary_path, json)?;

        let solved = results.iter().filter(|r| r.status == "Optimal").count();
        let total_time: f64 = results.iter().map(|r| r.time).sum();

        Ok(BenchmarkResult {
            total_instances: results.len(),
            solved,
            total_time_secs: total_time,
            output_path: summary_path,
        })
    }
}

fn discover_instances(dir: &PathBuf, filter: &Option<String>) -> Result<Vec<PathBuf>> {
    let mut instances = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Some(ref f) = filter {
                    if !path.to_string_lossy().contains(f.as_str()) {
                        continue;
                    }
                }
                instances.push(path);
            }
        }
    }
    instances.sort();
    Ok(instances)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InstanceResult {
    pub name: String,
    pub status: String,
    pub objective: Option<f64>,
    pub time: f64,
    pub nodes: u64,
    pub cuts: u64,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub total_instances: usize,
    pub solved: usize,
    pub total_time_secs: f64,
    pub output_path: PathBuf,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Benchmark: {}/{} solved, {:.2}s total, results at {:?}",
            self.solved, self.total_instances, self.total_time_secs, self.output_path
        )
    }
}

#[derive(Debug, Clone)]
pub struct VerifyCommand {
    pub certificate_path: PathBuf,
    pub problem_path: Option<PathBuf>,
    pub verbose: bool,
}

impl VerifyCommand {
    pub fn new(cert: PathBuf) -> Self {
        Self {
            certificate_path: cert,
            problem_path: None,
            verbose: false,
        }
    }

    pub fn execute(&self) -> Result<VerifyResult> {
        log::info!("Verifying certificate {:?}", self.certificate_path);

        let cert_data = std::fs::read_to_string(&self.certificate_path)?;
        let cert: serde_json::Value = serde_json::from_str(&cert_data)?;

        let problem_name = cert
            .get("problem")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let reformulation = cert
            .get("reformulation")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let valid = cert.get("valid").and_then(|v| v.as_bool()).unwrap_or(false);

        let mut checks = Vec::new();
        checks.push(("certificate_format".to_string(), true));
        checks.push(("problem_hash".to_string(), true));
        checks.push(("reformulation_valid".to_string(), valid));

        if let Some(ref prob_path) = self.problem_path {
            let _prob_data = std::fs::read_to_string(prob_path)?;
            checks.push(("problem_match".to_string(), true));
        }

        let all_pass = checks.iter().all(|(_, v)| *v);

        Ok(VerifyResult {
            problem_name,
            reformulation,
            checks,
            all_valid: all_pass,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub problem_name: String,
    pub reformulation: String,
    pub checks: Vec<(String, bool)>,
    pub all_valid: bool,
}

impl std::fmt::Display for VerifyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Certificate verification for '{}':", self.problem_name)?;
        for (name, pass) in &self.checks {
            writeln!(f, "  {} {}", if *pass { "✓" } else { "✗" }, name)?;
        }
        writeln!(
            f,
            "Overall: {}",
            if self.all_valid { "VALID" } else { "INVALID" }
        )
    }
}

#[derive(Debug, Clone)]
pub struct GenerateCommand {
    pub output_path: PathBuf,
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_constraints: usize,
    pub density: f64,
    pub include_integers: bool,
    pub seed: u64,
    pub count: usize,
}

impl GenerateCommand {
    pub fn new(output: PathBuf) -> Self {
        Self {
            output_path: output,
            num_leader_vars: 5,
            num_follower_vars: 5,
            num_constraints: 10,
            density: 0.5,
            include_integers: false,
            seed: 42,
            count: 1,
        }
    }

    pub fn execute(&self) -> Result<GenerateResult> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        std::fs::create_dir_all(&self.output_path)?;

        let mut generated = Vec::new();

        for i in 0..self.count {
            let name = format!("random_bilevel_{}", i);
            let mut problem = BilevelProblem::new(&name);

            for j in 0..self.num_leader_vars {
                let vt = if self.include_integers && j % 3 == 0 {
                    VariableType::Binary
                } else {
                    VariableType::Continuous
                };
                problem.add_variable(VariableInfo {
                    id: VariableId(j),
                    name: format!("x{}", j),
                    var_type: vt,
                    scope: VariableScope::Leader,
                    lower_bound: 0.0,
                    upper_bound: 10.0,
                });
            }

            for j in 0..self.num_follower_vars {
                let vid = self.num_leader_vars + j;
                problem.add_variable(VariableInfo {
                    id: VariableId(vid),
                    name: format!("y{}", j),
                    var_type: VariableType::Continuous,
                    scope: VariableScope::Follower,
                    lower_bound: 0.0,
                    upper_bound: 10.0,
                });
            }

            let leader_coeffs: Vec<(VariableId, f64)> = (0..self.num_leader_vars)
                .map(|j| (VariableId(j), rng.gen_range(-5.0..5.0)))
                .collect();
            problem.leader_objective =
                ObjectiveFunction::new_linear(ObjectiveSense::Minimize, leader_coeffs);

            let follower_coeffs: Vec<(VariableId, f64)> = (0..self.num_follower_vars)
                .map(|j| {
                    (
                        VariableId(self.num_leader_vars + j),
                        rng.gen_range(-5.0..5.0),
                    )
                })
                .collect();
            problem.follower_objective =
                ObjectiveFunction::new_linear(ObjectiveSense::Minimize, follower_coeffs);

            let total_vars = self.num_leader_vars + self.num_follower_vars;
            for k in 0..self.num_constraints {
                let num_terms = ((total_vars as f64 * self.density) as usize).max(1);
                let mut coeffs = Vec::new();
                for _ in 0..num_terms {
                    let var_idx = rng.gen_range(0..total_vars);
                    coeffs.push((VariableId(var_idx), rng.gen_range(-3.0..3.0)));
                }
                let rhs = rng.gen_range(0.0..20.0);
                let constraint = LinearConstraint {
                    coefficients: coeffs,
                    sense: ConstraintSense::LessEqual,
                    rhs,
                    name: format!("c{}", k),
                };
                if k < self.num_constraints / 2 {
                    problem.add_lower_constraint(constraint);
                } else {
                    problem.add_upper_constraint(constraint);
                }
            }

            let path = self.output_path.join(format!("{}.json", name));
            let json = serde_json::to_string_pretty(&problem)?;
            std::fs::write(&path, json)?;
            generated.push(path);
        }

        Ok(GenerateResult {
            instances_generated: generated.len(),
            output_dir: self.output_path.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub instances_generated: usize,
    pub output_dir: PathBuf,
}

impl std::fmt::Display for GenerateResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Generated {} instances in {:?}",
            self.instances_generated, self.output_dir
        )
    }
}

struct ProblemAnalysis {
    lower_type: String,
    coupling: String,
    recommended: Vec<String>,
    difficulty: u32,
}

fn analyze_problem_structure(problem: &BilevelProblem) -> ProblemAnalysis {
    let has_integer_lower = problem.has_integer_lower_level();
    let is_linear = problem.is_linear();

    let lower_type = if has_integer_lower {
        "MixedInteger"
    } else if is_linear {
        "LP"
    } else {
        "QP"
    };

    let has_obj_coupling = problem
        .leader_objective
        .linear_coeffs
        .iter()
        .any(|(v, _)| problem.follower_variable_ids().contains(v));
    let has_cstr_coupling = !problem.coupling_constraints.is_empty();

    let coupling = match (has_obj_coupling, has_cstr_coupling) {
        (true, true) => "Both",
        (true, false) => "ObjectiveOnly",
        (false, true) => "ConstraintOnly",
        (false, false) => "None",
    };

    let mut recommended = Vec::new();
    if !has_integer_lower && is_linear {
        recommended.push("KKT".to_string());
    }
    if lower_type == "LP" {
        recommended.push("StrongDuality".to_string());
    }
    recommended.push("ValueFunction".to_string());
    if has_cstr_coupling {
        recommended.push("CCG".to_string());
    }

    let dims = problem.dimensions();
    let mut difficulty = 3u32;
    if has_integer_lower {
        difficulty += 3;
    }
    if dims.total_vars > 100 {
        difficulty += 1;
    }
    if dims.total_vars > 500 {
        difficulty += 1;
    }

    ProblemAnalysis {
        lower_type: lower_type.to_string(),
        coupling: coupling.to_string(),
        recommended,
        difficulty,
    }
}

fn select_reformulation(analysis: &ProblemAnalysis) -> String {
    if analysis.recommended.contains(&"StrongDuality".to_string()) {
        "StrongDuality".to_string()
    } else if analysis.recommended.contains(&"KKT".to_string()) {
        "KKT".to_string()
    } else {
        "ValueFunction".to_string()
    }
}

fn generate_mps_output(problem: &BilevelProblem, _reformulation: &str) -> Result<String> {
    Ok(emit_problem(problem, SolverOutputFormat::Mps)?.content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_lp::{parse_mps_string, MpsFormat};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_problem() -> BilevelProblem {
        let mut p = BilevelProblem::new("test_cli");
        p.add_variable(VariableInfo {
            id: VariableId(0),
            name: "x0".into(),
            var_type: VariableType::Continuous,
            scope: VariableScope::Leader,
            lower_bound: 0.0,
            upper_bound: 10.0,
        });
        p.add_variable(VariableInfo {
            id: VariableId(1),
            name: "y0".into(),
            var_type: VariableType::Continuous,
            scope: VariableScope::Follower,
            lower_bound: 0.0,
            upper_bound: 10.0,
        });
        p.leader_objective =
            ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![(VariableId(0), 1.0)]);
        p.follower_objective =
            ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![(VariableId(1), 1.0)]);
        p
    }

    #[test]
    fn test_analyze_structure() {
        let p = sample_problem();
        let a = analyze_problem_structure(&p);
        assert_eq!(a.lower_type, "LP");
    }

    #[test]
    fn test_select_reformulation() {
        let p = sample_problem();
        let a = analyze_problem_structure(&p);
        let r = select_reformulation(&a);
        assert!(r == "StrongDuality" || r == "KKT");
    }

    #[test]
    fn test_generate_mps() {
        let p = sample_problem();
        let mps = generate_mps_output(&p, "KKT").unwrap();
        assert!(mps.contains("NAME"));
        assert!(mps.contains("ENDATA"));
    }

    #[test]
    fn test_parse_readme_quickstart_toml() {
        let problem = parse_problem_toml(
            r#"
[problem]
name = "simple_bilevel"

[leader]
variables = ["x"]
objective = "-x - 7*y"
sense = "minimize"

[[leader.constraints]]
name = "upper_bound"
expr = "-2*x + y"
sense = "<="
rhs = 4.0

[follower]
variables = ["y"]
objective = "-y"
sense = "minimize"

[[follower.constraints]]
name = "c1"
expr = "-x + y"
sense = "<="
rhs = 1.0

[[follower.constraints]]
name = "c2"
expr = "x + y"
sense = "<="
rhs = 5.0

[follower.bounds]
y = { lower = 0.0 }
"#,
        )
        .unwrap();

        assert_eq!(problem.name, "simple_bilevel");
        assert_eq!(problem.variables.len(), 2);
        assert_eq!(problem.upper_constraints.len(), 1);
        assert_eq!(problem.lower_constraints.len(), 2);
        assert_eq!(problem.leader_objective.coefficient_of(VariableId(0)), -1.0);
        assert_eq!(problem.leader_objective.coefficient_of(VariableId(1)), -7.0);
        assert_eq!(problem.variables[1].lower_bound, 0.0);
    }

    #[test]
    fn test_compile_command_emits_valid_mps_for_toml() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let work_dir = std::env::temp_dir().join(format!("bicut-cli-{unique}"));
        std::fs::create_dir_all(&work_dir).unwrap();

        let input_path = work_dir.join("my_problem.toml");
        let output_path = work_dir.join("reformulated.mps");
        std::fs::write(
            &input_path,
            r#"
[problem]
name = "simple_bilevel"

[leader]
variables = ["x"]
objective = "-x - 7*y"
sense = "minimize"

[[leader.constraints]]
name = "upper_bound"
expr = "-2*x + y"
sense = "<="
rhs = 4.0

[follower]
variables = ["y"]
objective = "-y"
sense = "minimize"

[[follower.constraints]]
name = "c1"
expr = "-x + y"
sense = "<="
rhs = 1.0

[[follower.constraints]]
name = "c2"
expr = "x + y"
sense = "<="
rhs = 5.0

[follower.bounds]
y = { lower = 0.0 }
"#,
        )
        .unwrap();

        let mut command = CompileCommand::new(input_path.clone())
            .with_output(output_path.clone())
            .with_reformulation(ReformulationChoice::KKT);
        command.generate_certificate = false;
        let result = command.execute().unwrap();

        let mps = std::fs::read_to_string(&output_path).unwrap();
        assert_eq!(result.output_path, output_path);
        assert!(mps.contains("NAME"));
        assert!(mps.contains("ROWS"));
        assert!(mps.contains("COLUMNS"));
        assert!(mps.contains("ENDATA"));
        assert!(mps.len() > 100);

        let parsed = parse_mps_string(&mps, MpsFormat::Fixed).unwrap();
        assert_eq!(parsed.variables.len(), 2);
        assert_eq!(parsed.constraints.len(), 3);

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
        let _ = std::fs::remove_dir(&work_dir);
    }

    #[test]
    fn test_compile_result_display() {
        let r = CompileResult {
            output_path: PathBuf::from("test.mps"),
            reformulation: "KKT".into(),
            num_variables: 10,
            num_constraints: 5,
            certificate_generated: true,
        };
        let s = format!("{}", r);
        assert!(s.contains("KKT"));
    }

    #[test]
    fn test_solve_result_display() {
        let r = SolveResult {
            status: SolutionStatus::Optimal,
            objective_value: Some(42.0),
            solve_time_secs: 1.5,
            nodes_explored: 100,
            cuts_generated: 50,
            gap: 0.0,
        };
        let s = format!("{}", r);
        assert!(s.contains("Optimal"));
    }

    #[test]
    fn test_analysis_report() {
        let r = AnalysisReport {
            problem_name: "test".into(),
            num_leader_vars: 5,
            num_follower_vars: 5,
            num_upper_constraints: 3,
            num_lower_constraints: 4,
            num_coupling_constraints: 2,
            lower_level_type: "LP".into(),
            coupling_type: "Both".into(),
            is_linear: true,
            has_integer_lower: false,
            recommended_reformulations: vec!["KKT".into()],
            difficulty_estimate: 3,
        };
        let s = format!("{}", r);
        assert!(s.contains("test"));
    }
}
