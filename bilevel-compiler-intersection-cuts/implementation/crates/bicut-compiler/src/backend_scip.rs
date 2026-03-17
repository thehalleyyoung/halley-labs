//! SCIP-specific emission backend.
//!
//! Generates CIP-format files, SCIP parameter settings (`.set`), branching
//! priorities, and constraint-handler configuration for the SCIP solver.
//!
//! # CIP format
//!
//! The SCIP CIP (Constraint Integer Program) format consists of sections:
//!
//! ```text
//! STATISTICS
//!   Problem name: ...
//!   Variables: N (B binary, I integer, C continuous)
//!   Constraints: M
//! OBJECTIVE
//!   Sense: minimize
//!   obj: coeff1*var1 + coeff2*var2 + ...
//! VARIABLES
//!   var_name [type] [lb,ub] obj:coeff
//! CONSTRAINTS
//!   constraint_name: lhs <= expr <= rhs
//! END
//! ```

use std::fmt;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use crate::emission::{EmissionConfig, EmissionResult, OutputFormat};
use crate::pipeline::{
    IndicatorConstraint, MilpConstraint, MilpProblem, MilpVariable, Sos1Set, VarType,
};
use crate::CompilerError;
use bicut_types::{ConstraintSense, OptDirection, VarBound};

// ════════════════════════════════════════════════════════════════════════════
// SCIP emphasis
// ════════════════════════════════════════════════════════════════════════════

/// SCIP solver emphasis setting controlling the trade-off between finding
/// feasible solutions and proving optimality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScipEmphasis {
    /// Balanced default behaviour.
    Default,
    /// Favour finding feasible solutions quickly.
    Feasibility,
    /// Favour proving optimality (stronger cuts, less heuristics).
    Optimality,
    /// Count the number of feasible solutions.
    Counter,
    /// CP-style optimality (aggressive cuts and heuristics).
    CPOptimal,
}

impl ScipEmphasis {
    fn scip_value(&self) -> &'static str {
        match self {
            ScipEmphasis::Default => "DEFAULT",
            ScipEmphasis::Feasibility => "FEASIBILITY",
            ScipEmphasis::Optimality => "OPTIMALITY",
            ScipEmphasis::Counter => "COUNTER",
            ScipEmphasis::CPOptimal => "CPOPTIMAL",
        }
    }
}

impl fmt::Display for ScipEmphasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.scip_value())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SCIP configuration
// ════════════════════════════════════════════════════════════════════════════

/// Configuration for the SCIP emission backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScipConfig {
    /// Activate the bilevel-specific constraint handler.
    pub use_constraint_handler: bool,
    /// Branching priority assigned to binary variables.
    pub branching_priority_binary: i32,
    /// Branching priority assigned to general integer variables.
    pub branching_priority_integer: i32,
    /// Maximum number of branch-and-bound nodes (u64::MAX = unlimited).
    pub node_limit: u64,
    /// Wall-clock time limit in seconds (>= 1e20 = unlimited).
    pub time_limit_sec: f64,
    /// Relative MIP gap tolerance (0.0 = prove optimal).
    pub gap_limit: f64,
    /// Solver emphasis setting.
    pub emphasis: ScipEmphasis,
    /// Frequency of heuristic calls (-1 = disable).
    pub heuristics_freq: i32,
    /// Frequency of separation rounds.
    pub separating_freq: i32,
    /// Maximum presolving rounds (-1 = automatic).
    pub presolving_rounds: i32,
    /// Frequency of information display lines.
    pub display_freq: i32,
}

impl Default for ScipConfig {
    fn default() -> Self {
        Self {
            use_constraint_handler: false,
            branching_priority_binary: 100,
            branching_priority_integer: 50,
            node_limit: u64::MAX,
            time_limit_sec: 1e+20,
            gap_limit: 0.0,
            emphasis: ScipEmphasis::Default,
            heuristics_freq: 1,
            separating_freq: 10,
            presolving_rounds: -1,
            display_freq: 100,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SCIP parameter types
// ════════════════════════════════════════════════════════════════════════════

/// A single SCIP parameter entry for a `.set` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScipParam {
    /// Hierarchical parameter path (e.g. `"limits/time"`).
    pub path: String,
    /// The parameter value.
    pub value: ScipParamValue,
}

impl fmt::Display for ScipParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.path, self.value)
    }
}

/// Typed value for a SCIP parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScipParamValue {
    Bool(bool),
    Int(i64),
    Real(f64),
    Str(String),
}

impl fmt::Display for ScipParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScipParamValue::Bool(b) => {
                write!(f, "{}", if *b { "TRUE" } else { "FALSE" })
            }
            ScipParamValue::Int(i) => write!(f, "{}", i),
            ScipParamValue::Real(r) => {
                if *r == r.floor() && r.abs() < 1e15 {
                    write!(f, "{:.1}", r)
                } else {
                    write!(f, "{}", r)
                }
            }
            ScipParamValue::Str(s) => write!(f, "\"{}\"", s),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SCIP output bundle
// ════════════════════════════════════════════════════════════════════════════

/// Output produced by the SCIP emission backend.
#[derive(Debug, Clone)]
pub struct ScipOutput {
    /// Full CIP-format file content.
    pub cip_content: String,
    /// Content of the `.set` parameter file.
    pub parameter_content: String,
    /// Variable-name → branching-priority pairs.
    pub branching_priorities: Vec<(String, i32)>,
    /// Constraint-handler configuration block.
    pub constraint_handler_settings: String,
}

// ════════════════════════════════════════════════════════════════════════════
// ScipEmitter
// ════════════════════════════════════════════════════════════════════════════

/// Emitter that converts a [`MilpProblem`] into SCIP-specific artifacts.
pub struct ScipEmitter {
    config: ScipConfig,
}

impl ScipEmitter {
    /// Create a new emitter with the given configuration.
    pub fn new(config: ScipConfig) -> Self {
        Self { config }
    }

    // ── Top-level entry point ──────────────────────────────────────────

    /// Emit all SCIP artifacts for the given MILP.
    pub fn emit(&self, milp: &MilpProblem) -> Result<ScipOutput, CompilerError> {
        let cip_content = self.emit_cip_format(milp)?;
        let parameter_content = self.generate_parameter_file();
        let raw_priorities = self.assign_branching_priorities(milp);
        let branching_priorities: Vec<(String, i32)> = raw_priorities
            .iter()
            .map(|&(idx, prio)| (milp.variables[idx].name.clone(), prio))
            .collect();
        let constraint_handler_settings = self.write_constraint_handler_config();

        Ok(ScipOutput {
            cip_content,
            parameter_content,
            branching_priorities,
            constraint_handler_settings,
        })
    }

    // ── CIP format ─────────────────────────────────────────────────────

    /// Produce the full CIP-format string for `milp`.
    pub fn emit_cip_format(&self, milp: &MilpProblem) -> Result<String, CompilerError> {
        if milp.variables.is_empty() {
            return Err(CompilerError::InvalidProblem(
                "MILP has no variables".into(),
            ));
        }
        for v in &milp.variables {
            if v.lower_bound.is_nan() || v.upper_bound.is_nan() {
                return Err(CompilerError::Numerical(format!(
                    "variable '{}' has NaN bound",
                    v.name
                )));
            }
        }

        let mut out = String::with_capacity(8192);
        out.push_str(&self.write_cip_statistics(milp));
        out.push_str(&self.write_cip_objective(milp));
        out.push_str(&self.write_cip_variables(milp));
        out.push_str(&self.write_cip_constraints(milp));
        out.push_str("END\n");
        Ok(out)
    }

    // ── STATISTICS section ─────────────────────────────────────────────

    /// Write the CIP `STATISTICS` section.
    pub fn write_cip_statistics(&self, milp: &MilpProblem) -> String {
        let n_binary = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .count();
        let n_integer = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Integer)
            .count();
        let n_continuous = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Continuous)
            .count();

        let mut parts: Vec<String> = Vec::new();
        if n_binary > 0 {
            parts.push(format!("{} binary", n_binary));
        }
        if n_integer > 0 {
            parts.push(format!("{} integer", n_integer));
        }
        if n_continuous > 0 {
            parts.push(format!("{} continuous", n_continuous));
        }

        let var_detail = if parts.is_empty() {
            "0".to_string()
        } else {
            format!("{} ({})", milp.variables.len(), parts.join(", "))
        };

        let total_constraints =
            milp.constraints.len() + milp.sos1_sets.len() + milp.indicator_constraints.len();

        format!(
            "STATISTICS\n\
             \x20 Problem name: {}\n\
             \x20 Variables: {}\n\
             \x20 Constraints: {}\n",
            milp.name, var_detail, total_constraints
        )
    }

    // ── OBJECTIVE section ──────────────────────────────────────────────

    /// Write the CIP `OBJECTIVE` section.
    pub fn write_cip_objective(&self, milp: &MilpProblem) -> String {
        let sense_str = match milp.sense {
            OptDirection::Minimize => "minimize",
            OptDirection::Maximize => "maximize",
        };

        let terms: Vec<String> = milp
            .variables
            .iter()
            .filter(|v| v.obj_coeff.abs() > 1e-15)
            .map(|v| format_term(v.obj_coeff, &v.name))
            .collect();

        let obj_expr = if terms.is_empty() {
            "0".to_string()
        } else {
            join_signed_terms(&terms)
        };

        format!(
            "OBJECTIVE\n\
             \x20 Sense: {}\n\
             \x20 obj: {}\n",
            sense_str, obj_expr
        )
    }

    // ── VARIABLES section ──────────────────────────────────────────────

    /// Write the CIP `VARIABLES` section with types and bounds.
    pub fn write_cip_variables(&self, milp: &MilpProblem) -> String {
        let mut out = String::from("VARIABLES\n");
        for v in &milp.variables {
            let type_str = match v.var_type {
                VarType::Binary => "binary",
                VarType::Integer => "integer",
                VarType::Continuous => "continuous",
            };
            let lb = format_bound(v.lower_bound);
            let ub = format_bound(v.upper_bound);
            let _ = writeln!(
                out,
                "  {} [{}] [{},{}] obj:{}",
                v.name,
                type_str,
                lb,
                ub,
                format_coeff(v.obj_coeff),
            );
        }
        out
    }

    // ── CONSTRAINTS section ────────────────────────────────────────────

    /// Write the CIP `CONSTRAINTS` section (linear, SOS1, indicator).
    pub fn write_cip_constraints(&self, milp: &MilpProblem) -> String {
        let mut out = String::from("CONSTRAINTS\n");

        // Linear constraints — expressed as  lhs <= expr <= rhs.
        for c in &milp.constraints {
            let expr = format_linear_expr(&c.coeffs, &milp.variables);
            let (lhs_bound, rhs_bound) = sense_to_double_bound(c.sense, c.rhs);
            let _ = writeln!(
                out,
                "  {}: {} <= {} <= {}",
                c.name,
                format_bound(lhs_bound),
                expr,
                format_bound(rhs_bound),
            );
        }

        // SOS1 sets
        for sos in &milp.sos1_sets {
            let members: Vec<String> = sos
                .members
                .iter()
                .zip(sos.weights.iter())
                .map(|(&vi, &w)| format!("{}:{}", milp.variables[vi].name, format_coeff(w)))
                .collect();
            let _ = writeln!(out, "  {} [SOS1]: {{{}}}", sos.name, members.join(", "));
        }

        // Indicator constraints
        for ic in &milp.indicator_constraints {
            let val = if ic.active_value { 1 } else { 0 };
            let expr = format_linear_expr(&ic.coeffs, &milp.variables);
            let sense_str = match ic.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "==",
            };
            let _ = writeln!(
                out,
                "  {} [indicator]: if {} = {} then {} {} {}",
                ic.name,
                milp.variables[ic.binary_var].name,
                val,
                expr,
                sense_str,
                format_coeff(ic.rhs),
            );
        }

        out
    }

    // ── Branching priorities ───────────────────────────────────────────

    /// Write a CIP-style branching-priority section.
    pub fn write_branching_priorities(&self, milp: &MilpProblem) -> String {
        let priorities = self.assign_branching_priorities(milp);
        if priorities.is_empty() {
            return String::new();
        }
        let mut out = String::from("BRANCHINGPRIORITIES\n");
        for &(idx, prio) in &priorities {
            let _ = writeln!(out, "  {} {}", milp.variables[idx].name, prio);
        }
        out
    }

    /// Assign branching priorities to variables based on their type.
    pub fn assign_branching_priorities(&self, milp: &MilpProblem) -> Vec<(usize, i32)> {
        let mut priorities = Vec::new();
        for (i, v) in milp.variables.iter().enumerate() {
            let prio = match v.var_type {
                VarType::Binary => self.config.branching_priority_binary,
                VarType::Integer => self.config.branching_priority_integer,
                VarType::Continuous => 0,
            };
            if prio != 0 {
                priorities.push((i, prio));
            }
        }
        priorities
    }

    // ── Parameter file (.set) ──────────────────────────────────────────

    /// Generate a SCIP `.set` parameter file from the current configuration.
    pub fn generate_parameter_file(&self) -> String {
        let mut lines: Vec<String> = Vec::with_capacity(32);
        lines.push("# SCIP parameter file generated by bicut-compiler".into());
        lines.push(String::new());

        // ── Limits ──
        if self.config.node_limit != u64::MAX {
            lines.push(format!("limits/nodes = {}", self.config.node_limit));
        }
        if self.config.time_limit_sec < 1e+19 {
            lines.push(format!("limits/time = {}", self.config.time_limit_sec));
        }
        if self.config.gap_limit > 0.0 {
            lines.push(format!("limits/gap = {}", self.config.gap_limit));
        }

        // ── Display ──
        lines.push(format!("display/freq = {}", self.config.display_freq));

        // ── Emphasis ──
        if self.config.emphasis != ScipEmphasis::Default {
            lines.push(format!("# emphasis: {}", self.config.emphasis.scip_value()));
            match self.config.emphasis {
                ScipEmphasis::Feasibility => {
                    lines.push("heuristics/emphasis = aggressive".into());
                    lines.push("separating/emphasis = fast".into());
                }
                ScipEmphasis::Optimality => {
                    lines.push("separating/emphasis = aggressive".into());
                    lines.push("heuristics/emphasis = off".into());
                }
                ScipEmphasis::Counter => {
                    lines.push("constraints/countsols/active = TRUE".into());
                }
                ScipEmphasis::CPOptimal => {
                    lines.push("separating/emphasis = aggressive".into());
                    lines.push("heuristics/emphasis = aggressive".into());
                }
                ScipEmphasis::Default => {}
            }
        }

        // ── Heuristics ──
        if self.config.heuristics_freq != 1 {
            lines.push(format!("heuristics/freq = {}", self.config.heuristics_freq));
        }

        // ── Separating ──
        if self.config.separating_freq != 10 {
            lines.push(format!("separating/freq = {}", self.config.separating_freq));
        }

        // ── Presolving ──
        if self.config.presolving_rounds != -1 {
            lines.push(format!(
                "presolving/maxrounds = {}",
                self.config.presolving_rounds
            ));
        }

        lines.push(String::new());
        lines.join("\n")
    }

    // ── Constraint handler config ──────────────────────────────────────

    /// Generate SCIP settings for the bilevel constraint handler plugin.
    pub fn write_constraint_handler_config(&self) -> String {
        if !self.config.use_constraint_handler {
            return "# Constraint handler disabled\n".into();
        }

        let mut out = String::with_capacity(512);
        out.push_str("# Bilevel constraint handler settings\n");
        out.push_str("constraints/bilevel/active = TRUE\n");
        out.push_str("constraints/bilevel/sepafreq = 1\n");
        out.push_str("constraints/bilevel/propfreq = 1\n");
        out.push_str("constraints/bilevel/eagerfreq = 100\n");
        out.push_str("constraints/bilevel/maxprerounds = -1\n");
        out.push_str("constraints/bilevel/delaysepa = FALSE\n");
        out.push_str("constraints/bilevel/delayprop = FALSE\n");
        out.push_str("constraints/bilevel/checkaliways = TRUE\n");
        out
    }

    // ── Auto-configuration ─────────────────────────────────────────────

    /// Suggest SCIP parameters based on problem structure.
    pub fn suggest_parameters(&self, milp: &MilpProblem) -> Vec<ScipParam> {
        let mut params = Vec::new();
        let n = milp.variables.len();
        let m = milp.constraints.len();
        if n == 0 {
            return params;
        }

        let n_binary = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .count();
        let n_integer = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Integer)
            .count();
        let n_discrete = n_binary + n_integer;

        // Node limit heuristic: scale with problem size.
        if n + m > 1000 {
            let limit = ((n + m) as u64).saturating_mul(500);
            params.push(ScipParam {
                path: "limits/nodes".into(),
                value: ScipParamValue::Int(limit as i64),
            });
        }

        // Time limit for large problems.
        if n + m > 5000 {
            params.push(ScipParam {
                path: "limits/time".into(),
                value: ScipParamValue::Real(3600.0),
            });
        }

        // If mostly binary, prioritise pseudo-cost branching.
        if n_discrete > 0 && (n_binary as f64 / n as f64) > 0.5 {
            params.push(ScipParam {
                path: "branching/pscost/priority".into(),
                value: ScipParamValue::Int(200),
            });
        }

        // Presolving benefit estimation.
        let presolving_benefit = Self::estimate_presolving_benefit(milp);
        if presolving_benefit > 0.7 {
            params.push(ScipParam {
                path: "presolving/maxrounds".into(),
                value: ScipParamValue::Int(-1),
            });
        } else if presolving_benefit < 0.3 {
            params.push(ScipParam {
                path: "presolving/maxrounds".into(),
                value: ScipParamValue::Int(0),
            });
        }

        // SOS1 branching when SOS1 sets are present.
        if !milp.sos1_sets.is_empty() {
            params.push(ScipParam {
                path: "constraints/SOS1/branchsos".into(),
                value: ScipParamValue::Bool(true),
            });
        }

        // Upgrade linear constraints that look like indicators.
        if !milp.indicator_constraints.is_empty() {
            params.push(ScipParam {
                path: "constraints/indicator/upgradelinear".into(),
                value: ScipParamValue::Bool(true),
            });
        }

        // RINS heuristic frequency based on integrality ratio.
        if n_discrete > 0 {
            let ratio = n_discrete as f64 / n as f64;
            let freq = if ratio > 0.8 {
                5
            } else if ratio > 0.3 {
                10
            } else {
                20
            };
            params.push(ScipParam {
                path: "heuristics/rins/freq".into(),
                value: ScipParamValue::Int(freq),
            });
        }

        // Aggressive Gomory and CMIR cuts for integer-heavy problems.
        if n_integer > 0 && (n_integer as f64 / n as f64) > 0.3 {
            params.push(ScipParam {
                path: "separating/gomory/freq".into(),
                value: ScipParamValue::Int(1),
            });
            params.push(ScipParam {
                path: "separating/cmir/freq".into(),
                value: ScipParamValue::Int(5),
            });
        }

        params
    }

    /// Estimate the benefit of presolving on a [0, 1] scale.
    ///
    /// Higher values indicate the problem is likely to benefit significantly
    /// from presolving (many equalities, dense constraints, binary vars).
    pub fn estimate_presolving_benefit(milp: &MilpProblem) -> f64 {
        let n = milp.variables.len();
        let m = milp.constraints.len();
        if n == 0 || m == 0 {
            return 0.0;
        }

        let mut score: f64 = 0.5;

        // Equality constraints create substitution opportunities.
        let eq_count = milp
            .constraints
            .iter()
            .filter(|c| c.sense == ConstraintSense::Eq)
            .count();
        let eq_frac = eq_count as f64 / m as f64;
        score += 0.2 * eq_frac;

        // Constraint density — higher density means more probing payoff.
        let total_nonzeros: usize = milp.constraints.iter().map(|c| c.coeffs.len()).sum();
        let density = total_nonzeros as f64 / (n as f64 * m as f64);
        if density > 0.1 {
            score += 0.15;
        }

        // Binary variables benefit from domain propagation.
        let binary_count = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .count();
        let binary_frac = binary_count as f64 / n as f64;
        score += 0.15 * binary_frac;

        score.clamp(0.0, 1.0)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Formatting helpers
// ════════════════════════════════════════════════════════════════════════════

/// Format a floating-point coefficient, printing integers without decimals.
fn format_coeff(c: f64) -> String {
    if c == 0.0 {
        return "0".to_string();
    }
    if c == c.floor() && c.abs() < 1e15 {
        format!("{}", c as i64)
    } else {
        format!("{}", c)
    }
}

/// Format a bound value, using `+inf` / `-inf` for infinities.
fn format_bound(b: f64) -> String {
    if b >= f64::MAX || b == f64::INFINITY {
        "+inf".to_string()
    } else if b <= f64::MIN || b == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format_coeff(b)
    }
}

/// Format a single term `coeff * var_name`.
fn format_term(coeff: f64, var_name: &str) -> String {
    if (coeff - 1.0).abs() < 1e-15 {
        var_name.to_string()
    } else if (coeff + 1.0).abs() < 1e-15 {
        format!("-{}", var_name)
    } else {
        format!("{}*{}", format_coeff(coeff), var_name)
    }
}

/// Format a linear expression `Σ coeff_j * var_j` from sparse `(index, coeff)` pairs.
fn format_linear_expr(coeffs: &[(usize, f64)], variables: &[MilpVariable]) -> String {
    let terms: Vec<String> = coeffs
        .iter()
        .filter(|(_, c)| c.abs() > 1e-15)
        .map(|&(vi, c)| format_term(c, &variables[vi].name))
        .collect();
    if terms.is_empty() {
        "0".to_string()
    } else {
        join_signed_terms(&terms)
    }
}

/// Join already-formatted terms with `+` / `-` signs.
fn join_signed_terms(terms: &[String]) -> String {
    if terms.is_empty() {
        return "0".to_string();
    }
    let mut out = terms[0].clone();
    for t in &terms[1..] {
        if t.starts_with('-') {
            let _ = write!(out, " - {}", &t[1..]);
        } else {
            let _ = write!(out, " + {}", t);
        }
    }
    out
}

/// Convert a single-sided `ConstraintSense` + rhs into the double-sided
/// `(lhs_bound, rhs_bound)` representation used by CIP.
fn sense_to_double_bound(sense: ConstraintSense, rhs: f64) -> (f64, f64) {
    match sense {
        ConstraintSense::Le => (f64::NEG_INFINITY, rhs),
        ConstraintSense::Ge => (rhs, f64::INFINITY),
        ConstraintSense::Eq => (rhs, rhs),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small test MILP with one binary, one continuous, and one
    /// integer variable, two linear constraints, one SOS1 set, and one
    /// indicator constraint.
    fn small_milp() -> MilpProblem {
        let mut milp = MilpProblem::new("bilevel_reformulation");
        milp.sense = OptDirection::Minimize;

        let x = milp.add_variable(MilpVariable {
            name: "x".into(),
            lower_bound: 0.0,
            upper_bound: 1.0,
            obj_coeff: 1.0,
            var_type: VarType::Binary,
        });
        let y = milp.add_variable(MilpVariable {
            name: "y".into(),
            lower_bound: 0.0,
            upper_bound: f64::INFINITY,
            obj_coeff: 2.5,
            var_type: VarType::Continuous,
        });
        let z = milp.add_variable(MilpVariable {
            name: "z".into(),
            lower_bound: 0.0,
            upper_bound: 10.0,
            obj_coeff: -1.0,
            var_type: VarType::Integer,
        });

        let mut c1 = MilpConstraint::new("c1", ConstraintSense::Le, 10.0);
        c1.add_term(x, 1.0);
        c1.add_term(y, 3.0);
        milp.add_constraint(c1);

        let mut c2 = MilpConstraint::new("c2", ConstraintSense::Eq, 0.0);
        c2.add_term(y, 1.0);
        c2.add_term(z, -2.0);
        milp.add_constraint(c2);

        milp.add_sos1_set(Sos1Set {
            name: "sos1".into(),
            sos_type: 1,
            members: vec![x, z],
            weights: vec![1.0, 2.0],
        });

        milp.add_indicator_constraint(IndicatorConstraint {
            name: "ind1".into(),
            binary_var: x,
            active_value: true,
            coeffs: vec![(y, 1.0)],
            sense: ConstraintSense::Le,
            rhs: 5.0,
        });

        milp
    }

    #[test]
    fn test_cip_format_generation() {
        let emitter = ScipEmitter::new(ScipConfig::default());
        let milp = small_milp();
        let cip = emitter.emit_cip_format(&milp).unwrap();

        assert!(cip.starts_with("STATISTICS"));
        assert!(cip.contains("OBJECTIVE"));
        assert!(cip.contains("VARIABLES"));
        assert!(cip.contains("CONSTRAINTS"));
        assert!(cip.trim_end().ends_with("END"));
        assert!(cip.contains("bilevel_reformulation"));
    }

    #[test]
    fn test_variable_section_with_types() {
        let emitter = ScipEmitter::new(ScipConfig::default());
        let milp = small_milp();
        let section = emitter.write_cip_variables(&milp);

        assert!(section.contains("x [binary] [0,1]"));
        assert!(section.contains("y [continuous] [0,+inf]"));
        assert!(section.contains("z [integer] [0,10]"));
        assert!(section.contains("obj:1"));
        assert!(section.contains("obj:2.5"));
        assert!(section.contains("obj:-1"));
    }

    #[test]
    fn test_constraint_section_formatting() {
        let emitter = ScipEmitter::new(ScipConfig::default());
        let milp = small_milp();
        let section = emitter.write_cip_constraints(&milp);

        assert!(section.contains("c1:"));
        assert!(section.contains("c2:"));
        assert!(section.contains("sos1 [SOS1]"));
        assert!(section.contains("ind1 [indicator]"));
        assert!(section.contains("if x = 1 then"));
    }

    #[test]
    fn test_branching_priority_assignment() {
        let emitter = ScipEmitter::new(ScipConfig {
            branching_priority_binary: 200,
            branching_priority_integer: 75,
            ..ScipConfig::default()
        });
        let milp = small_milp();
        let prios = emitter.assign_branching_priorities(&milp);

        // x = binary → 200, z = integer → 75, y = continuous → excluded
        assert_eq!(prios.len(), 2);
        assert_eq!(prios[0], (0, 200));
        assert_eq!(prios[1], (2, 75));
    }

    #[test]
    fn test_parameter_file_generation() {
        let config = ScipConfig {
            node_limit: 10_000,
            time_limit_sec: 600.0,
            gap_limit: 0.01,
            display_freq: 50,
            presolving_rounds: 3,
            ..ScipConfig::default()
        };
        let emitter = ScipEmitter::new(config);
        let params = emitter.generate_parameter_file();

        assert!(params.contains("limits/nodes = 10000"));
        assert!(params.contains("limits/time = 600"));
        assert!(params.contains("limits/gap = 0.01"));
        assert!(params.contains("display/freq = 50"));
        assert!(params.contains("presolving/maxrounds = 3"));
    }

    #[test]
    fn test_scip_emphasis_settings() {
        let emitter_feas = ScipEmitter::new(ScipConfig {
            emphasis: ScipEmphasis::Feasibility,
            ..ScipConfig::default()
        });
        let params_feas = emitter_feas.generate_parameter_file();
        assert!(params_feas.contains("FEASIBILITY"));
        assert!(params_feas.contains("heuristics/emphasis = aggressive"));

        let emitter_ctr = ScipEmitter::new(ScipConfig {
            emphasis: ScipEmphasis::Counter,
            ..ScipConfig::default()
        });
        let params_ctr = emitter_ctr.generate_parameter_file();
        assert!(params_ctr.contains("countsols"));

        let emitter_opt = ScipEmitter::new(ScipConfig {
            emphasis: ScipEmphasis::Optimality,
            ..ScipConfig::default()
        });
        let params_opt = emitter_opt.generate_parameter_file();
        assert!(params_opt.contains("heuristics/emphasis = off"));
        assert!(params_opt.contains("separating/emphasis = aggressive"));
    }

    #[test]
    fn test_full_emit_small_problem() {
        let emitter = ScipEmitter::new(ScipConfig {
            use_constraint_handler: true,
            ..ScipConfig::default()
        });
        let milp = small_milp();
        let output = emitter.emit(&milp).unwrap();

        assert!(!output.cip_content.is_empty());
        assert!(output.cip_content.contains("STATISTICS"));
        assert!(output.cip_content.contains("END"));

        assert!(!output.parameter_content.is_empty());
        assert!(output.parameter_content.contains("bicut-compiler"));

        assert!(!output.branching_priorities.is_empty());
        let names: Vec<&str> = output
            .branching_priorities
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"z"));

        assert!(output.constraint_handler_settings.contains("bilevel"));
        assert!(output.constraint_handler_settings.contains("TRUE"));
    }
}

// ── Type alias for crate-level re-export ───────────────────────────

/// The SCIP backend (alias for [`ScipEmitter`]).
pub type ScipBackend = ScipEmitter;
