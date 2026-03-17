//! Gurobi-specific MILP emission backend.
//!
//! Supports indicator constraints, SOS1/SOS2 sets, lazy constraints,
//! and Gurobi's `.lp` / `.prm` file formats with solver-specific extensions.

use crate::pipeline::{
    IndicatorConstraint, MilpConstraint, MilpProblem, MilpVariable, Sos1Set, VarType,
};
use crate::CompilerError;
use bicut_types::{ConstraintSense, OptDirection};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;

// ════════════════════════════════════════════════════════════════════════════
// Configuration types
// ════════════════════════════════════════════════════════════════════════════

/// Gurobi presolve aggressiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GurobiPresolve {
    Auto,
    Off,
    Conservative,
    Aggressive,
}

impl GurobiPresolve {
    fn param_value(&self) -> i64 {
        match self {
            Self::Auto => -1,
            Self::Off => 0,
            Self::Conservative => 1,
            Self::Aggressive => 2,
        }
    }
}

/// Gurobi LP method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GurobiMethod {
    Auto,
    PrimalSimplex,
    DualSimplex,
    Barrier,
    Concurrent,
}

impl GurobiMethod {
    fn param_value(&self) -> i64 {
        match self {
            Self::Auto => -1,
            Self::PrimalSimplex => 0,
            Self::DualSimplex => 1,
            Self::Barrier => 2,
            Self::Concurrent => 3,
        }
    }
}

/// Full configuration for the Gurobi emission backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GurobiConfig {
    pub use_indicator_constraints: bool,
    pub use_lazy_constraints: bool,
    pub use_sos1: bool,
    pub mip_gap: f64,
    pub time_limit_sec: f64,
    pub threads: usize,
    pub presolve: GurobiPresolve,
    pub method: GurobiMethod,
    pub feasibility_tol: f64,
    pub integrality_tol: f64,
    pub output_flag: bool,
}

impl Default for GurobiConfig {
    fn default() -> Self {
        Self {
            use_indicator_constraints: true,
            use_lazy_constraints: false,
            use_sos1: true,
            mip_gap: 1e-4,
            time_limit_sec: 3600.0,
            threads: 0,
            presolve: GurobiPresolve::Auto,
            method: GurobiMethod::Auto,
            feasibility_tol: 1e-6,
            integrality_tol: 1e-5,
            output_flag: true,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Parameter types
// ════════════════════════════════════════════════════════════════════════════

/// Typed value for a Gurobi parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GurobiParamValue {
    Int(i64),
    Float(f64),
    Str(String),
}

/// A single Gurobi solver parameter (name + value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GurobiParam {
    pub name: String,
    pub value: GurobiParamValue,
}

impl GurobiParam {
    pub fn int(name: &str, v: i64) -> Self {
        Self {
            name: name.to_string(),
            value: GurobiParamValue::Int(v),
        }
    }

    pub fn float(name: &str, v: f64) -> Self {
        Self {
            name: name.to_string(),
            value: GurobiParamValue::Float(v),
        }
    }

    pub fn string(name: &str, v: &str) -> Self {
        Self {
            name: name.to_string(),
            value: GurobiParamValue::Str(v.to_string()),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Model complexity / difficulty
// ════════════════════════════════════════════════════════════════════════════

/// Coarse difficulty classification for auto-tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    VeryHard,
}

/// Structural statistics of a MILP instance used for parameter tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComplexity {
    pub num_binary: usize,
    pub num_integer: usize,
    pub num_continuous: usize,
    pub num_constraints: usize,
    pub num_nonzeros: usize,
    pub density: f64,
    pub estimated_difficulty: DifficultyLevel,
}

// ════════════════════════════════════════════════════════════════════════════
// Emission output
// ════════════════════════════════════════════════════════════════════════════

/// The result bundle produced by [`GurobiEmitter::emit`].
#[derive(Debug, Clone)]
pub struct GurobiOutput {
    pub model_content: String,
    pub parameter_content: String,
    pub format: String,
    pub indicators_count: usize,
    pub lazy_count: usize,
    pub sos_count: usize,
}

// ════════════════════════════════════════════════════════════════════════════
// Emitter
// ════════════════════════════════════════════════════════════════════════════

/// Gurobi-specific MILP emitter.
///
/// Converts a [`MilpProblem`] into Gurobi's extended LP format, MPS with
/// indicator/lazy extensions, and generates matching `.prm` files.
pub struct GurobiEmitter {
    config: GurobiConfig,
}

impl GurobiEmitter {
    /// Create a new emitter with the given configuration.
    pub fn new(config: GurobiConfig) -> Self {
        Self { config }
    }

    // ── top-level emit ─────────────────────────────────────────────

    /// Emit the full Gurobi output bundle (model + parameters).
    pub fn emit(&self, milp: &MilpProblem) -> Result<GurobiOutput, CompilerError> {
        if milp.variables.is_empty() {
            return Err(CompilerError::Emission(
                "Cannot emit an empty model with no variables".into(),
            ));
        }
        let model_content = self.emit_grb_format(milp)?;
        let parameter_content = self.generate_parameter_file();

        let indicators_count = if self.config.use_indicator_constraints {
            milp.indicator_constraints.len()
        } else {
            0
        };
        let lazy_count = if self.config.use_lazy_constraints {
            milp.constraints.len()
        } else {
            0
        };
        let sos_count = if self.config.use_sos1 {
            milp.sos1_sets.len()
        } else {
            0
        };

        Ok(GurobiOutput {
            model_content,
            parameter_content,
            format: "grb_lp".to_string(),
            indicators_count,
            lazy_count,
            sos_count,
        })
    }

    // ── Gurobi LP format (.lp with extensions) ────────────────────

    /// Emit the model in Gurobi's extended `.lp` format.
    pub fn emit_grb_format(&self, milp: &MilpProblem) -> Result<String, CompilerError> {
        let mut out = String::with_capacity(4096);

        writeln!(out, "\\ Gurobi LP format — generated by bicut-compiler").unwrap();
        writeln!(out, "\\ Model: {}", milp.name).unwrap();
        writeln!(out).unwrap();

        out.push_str(&self.write_objective_section(milp));
        out.push_str(&self.write_constraint_section(milp));

        if self.config.use_indicator_constraints && !milp.indicator_constraints.is_empty() {
            out.push_str(&self.write_indicator_constraints(milp));
        }
        if self.config.use_lazy_constraints && !milp.constraints.is_empty() {
            out.push_str(&self.write_lazy_constraints(milp));
        }
        if self.config.use_sos1 && !milp.sos1_sets.is_empty() {
            out.push_str(&self.write_sos_sets(milp));
        }

        out.push_str(&self.write_bounds_section(milp));
        out.push_str(&self.write_variable_types(milp));

        writeln!(out, "End").unwrap();
        Ok(out)
    }

    // ── MPS with Gurobi extensions ────────────────────────────────

    /// Emit MPS with Gurobi-specific indicator and lazy constraint sections.
    pub fn emit_mps_with_extensions(&self, milp: &MilpProblem) -> Result<String, CompilerError> {
        if milp.variables.is_empty() {
            return Err(CompilerError::Emission(
                "Cannot emit MPS for model with no variables".into(),
            ));
        }

        let mut out = String::with_capacity(8192);

        writeln!(out, "NAME          {}", milp.name).unwrap();

        // ROWS section
        writeln!(out, "ROWS").unwrap();
        let obj_row_name = "OBJ";
        writeln!(out, " N  {obj_row_name}").unwrap();
        for (i, con) in milp.constraints.iter().enumerate() {
            let sense_char = match con.sense {
                ConstraintSense::Le => 'L',
                ConstraintSense::Ge => 'G',
                ConstraintSense::Eq => 'E',
            };
            let row_name = if con.name.is_empty() {
                format!("R{i}")
            } else {
                con.name.clone()
            };
            writeln!(out, " {sense_char}  {row_name}").unwrap();
        }

        // COLUMNS section
        writeln!(out, "COLUMNS").unwrap();
        for (j, var) in milp.variables.iter().enumerate() {
            let vname = if var.name.is_empty() {
                format!("x{j}")
            } else {
                var.name.clone()
            };

            if var.var_type == VarType::Binary || var.var_type == VarType::Integer {
                writeln!(out, "    INT{j}  'MARKER'  'INTORG'").unwrap();
            }

            if var.obj_coeff.abs() > 1e-20 {
                writeln!(out, "    {vname}  {obj_row_name}  {:.12e}", var.obj_coeff).unwrap();
            }
            for (i, con) in milp.constraints.iter().enumerate() {
                for &(col, coeff) in &con.coeffs {
                    if col == j && coeff.abs() > 1e-20 {
                        let rname = if con.name.is_empty() {
                            format!("R{i}")
                        } else {
                            con.name.clone()
                        };
                        writeln!(out, "    {vname}  {rname}  {coeff:.12e}").unwrap();
                    }
                }
            }

            if var.var_type == VarType::Binary || var.var_type == VarType::Integer {
                writeln!(out, "    INT{j}  'MARKER'  'INTEND'").unwrap();
            }
        }

        // RHS section
        writeln!(out, "RHS").unwrap();
        for (i, con) in milp.constraints.iter().enumerate() {
            let rname = if con.name.is_empty() {
                format!("R{i}")
            } else {
                con.name.clone()
            };
            if con.rhs.abs() > 1e-20 {
                writeln!(out, "    RHS  {rname}  {:.12e}", con.rhs).unwrap();
            }
        }

        // BOUNDS section
        writeln!(out, "BOUNDS").unwrap();
        for (j, var) in milp.variables.iter().enumerate() {
            let vname = if var.name.is_empty() {
                format!("x{j}")
            } else {
                var.name.clone()
            };
            match var.var_type {
                VarType::Binary => {
                    writeln!(out, " BV BND  {vname}").unwrap();
                }
                _ => {
                    if var.lower_bound != 0.0 || var.lower_bound == f64::NEG_INFINITY {
                        if var.lower_bound == f64::NEG_INFINITY {
                            writeln!(out, " MI BND  {vname}").unwrap();
                        } else {
                            writeln!(out, " LO BND  {vname}  {:.12e}", var.lower_bound).unwrap();
                        }
                    }
                    if var.upper_bound != f64::INFINITY {
                        writeln!(out, " UP BND  {vname}  {:.12e}", var.upper_bound).unwrap();
                    }
                }
            }
        }

        // Gurobi indicator extension
        if self.config.use_indicator_constraints && !milp.indicator_constraints.is_empty() {
            writeln!(out, "INDICATORS").unwrap();
            for ind in &milp.indicator_constraints {
                let bin_name = milp
                    .variables
                    .get(ind.binary_var)
                    .map(|v| v.name.as_str())
                    .unwrap_or("?");
                let active_val: u8 = if ind.active_value { 1 } else { 0 };
                let ind_name = if ind.name.is_empty() {
                    "IND"
                } else {
                    &ind.name
                };
                writeln!(out, " IF {ind_name} {bin_name} {active_val}").unwrap();
            }
        }

        // Gurobi lazy constraint extension
        if self.config.use_lazy_constraints && !milp.constraints.is_empty() {
            writeln!(out, "LAZYCONS").unwrap();
            for (i, con) in milp.constraints.iter().enumerate() {
                let rname = if con.name.is_empty() {
                    format!("R{i}")
                } else {
                    con.name.clone()
                };
                writeln!(out, " LC {rname} 1").unwrap();
            }
        }

        writeln!(out, "ENDATA").unwrap();
        Ok(out)
    }

    // ── Section writers ───────────────────────────────────────────

    /// Objective section in Gurobi LP format.
    pub fn write_objective_section(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();
        let dir = match milp.sense {
            OptDirection::Minimize => "Minimize",
            OptDirection::Maximize => "Maximize",
        };
        writeln!(sec, "{dir}").unwrap();
        write!(sec, "  obj:").unwrap();

        let mut first = true;
        for (j, var) in milp.variables.iter().enumerate() {
            if var.obj_coeff.abs() < 1e-20 {
                continue;
            }
            let name = if var.name.is_empty() {
                format!("x{j}")
            } else {
                var.name.clone()
            };
            if first {
                write!(sec, " {:.6} {name}", var.obj_coeff).unwrap();
                first = false;
            } else if var.obj_coeff >= 0.0 {
                write!(sec, " + {:.6} {name}", var.obj_coeff).unwrap();
            } else {
                write!(sec, " - {:.6} {name}", -var.obj_coeff).unwrap();
            }
        }
        if first {
            write!(sec, " 0").unwrap();
        }
        writeln!(sec).unwrap();
        sec
    }

    /// Standard constraints section in Gurobi LP format.
    pub fn write_constraint_section(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();
        writeln!(sec, "Subject To").unwrap();
        for (i, con) in milp.constraints.iter().enumerate() {
            let cname = if con.name.is_empty() {
                format!("c{i}")
            } else {
                con.name.clone()
            };
            write!(sec, "  {cname}:").unwrap();

            let mut first = true;
            for &(col, coeff) in &con.coeffs {
                let vname = milp
                    .variables
                    .get(col)
                    .map(|v| v.name.as_str())
                    .unwrap_or("?");
                if first {
                    write!(sec, " {coeff:.6} {vname}").unwrap();
                    first = false;
                } else if coeff >= 0.0 {
                    write!(sec, " + {coeff:.6} {vname}").unwrap();
                } else {
                    write!(sec, " - {:.6} {vname}", -coeff).unwrap();
                }
            }
            if first {
                write!(sec, " 0").unwrap();
            }

            let sense_str = match con.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "=",
            };
            writeln!(sec, " {sense_str} {:.6}", con.rhs).unwrap();
        }
        sec
    }

    /// Gurobi indicator constraint syntax:
    /// `name: binary_var = val -> linear_expr sense rhs`
    pub fn write_indicator_constraints(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();
        writeln!(sec, "General Constraints").unwrap();
        for (i, ind) in milp.indicator_constraints.iter().enumerate() {
            let ind_name = if ind.name.is_empty() {
                format!("ind{i}")
            } else {
                ind.name.clone()
            };
            let bin_name = milp
                .variables
                .get(ind.binary_var)
                .map(|v| v.name.as_str())
                .unwrap_or("?");
            let active_val: u8 = if ind.active_value { 1 } else { 0 };

            write!(sec, "  {ind_name}: {bin_name} = {active_val} ->").unwrap();

            let mut first = true;
            for &(col, coeff) in &ind.coeffs {
                let vname = milp
                    .variables
                    .get(col)
                    .map(|v| v.name.as_str())
                    .unwrap_or("?");
                if first {
                    write!(sec, " {coeff:.6} {vname}").unwrap();
                    first = false;
                } else if coeff >= 0.0 {
                    write!(sec, " + {coeff:.6} {vname}").unwrap();
                } else {
                    write!(sec, " - {:.6} {vname}", -coeff).unwrap();
                }
            }
            if first {
                write!(sec, " 0").unwrap();
            }

            let sense_str = match ind.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "=",
            };
            writeln!(sec, " {sense_str} {:.6}", ind.rhs).unwrap();
        }
        sec
    }

    /// Lazy constraint annotations (Gurobi `_lazy` attribute lines).
    pub fn write_lazy_constraints(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();
        writeln!(sec, "Lazy Constraints").unwrap();
        for (i, con) in milp.constraints.iter().enumerate() {
            let cname = if con.name.is_empty() {
                format!("lazy{i}")
            } else {
                format!("{}_lazy", con.name)
            };
            write!(sec, "  {cname}:").unwrap();

            let mut first = true;
            for &(col, coeff) in &con.coeffs {
                let vname = milp
                    .variables
                    .get(col)
                    .map(|v| v.name.as_str())
                    .unwrap_or("?");
                if first {
                    write!(sec, " {coeff:.6} {vname}").unwrap();
                    first = false;
                } else if coeff >= 0.0 {
                    write!(sec, " + {coeff:.6} {vname}").unwrap();
                } else {
                    write!(sec, " - {:.6} {vname}", -coeff).unwrap();
                }
            }
            if first {
                write!(sec, " 0").unwrap();
            }

            let sense_str = match con.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "=",
            };
            writeln!(sec, " {sense_str} {:.6}", con.rhs).unwrap();
        }
        sec
    }

    /// SOS1 / SOS2 set section in Gurobi LP format.
    pub fn write_sos_sets(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();
        writeln!(sec, "SOS").unwrap();
        for (i, sos) in milp.sos1_sets.iter().enumerate() {
            let sname = if sos.name.is_empty() {
                format!("sos{i}")
            } else {
                sos.name.clone()
            };
            writeln!(sec, "  {sname}: S1 ::").unwrap();
            for (k, &member) in sos.members.iter().enumerate() {
                let vname = milp
                    .variables
                    .get(member)
                    .map(|v| v.name.as_str())
                    .unwrap_or("?");
                let weight = sos.weights.get(k).copied().unwrap_or((k + 1) as f64);
                writeln!(sec, "    {vname}: {weight:.6}").unwrap();
            }
        }
        sec
    }

    /// Bounds section in Gurobi LP format.
    pub fn write_bounds_section(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();
        writeln!(sec, "Bounds").unwrap();
        for (j, var) in milp.variables.iter().enumerate() {
            let name = if var.name.is_empty() {
                format!("x{j}")
            } else {
                var.name.clone()
            };
            let lb = var.lower_bound;
            let ub = var.upper_bound;
            if lb == f64::NEG_INFINITY && ub == f64::INFINITY {
                writeln!(sec, "  {name} free").unwrap();
            } else if lb == f64::NEG_INFINITY {
                writeln!(sec, "  -Inf <= {name} <= {ub:.6}").unwrap();
            } else if ub == f64::INFINITY {
                writeln!(sec, "  {lb:.6} <= {name} <= +Inf").unwrap();
            } else {
                writeln!(sec, "  {lb:.6} <= {name} <= {ub:.6}").unwrap();
            }
        }
        sec
    }

    /// Variable type declarations (Binary, General, Semi-Continuous).
    pub fn write_variable_types(&self, milp: &MilpProblem) -> String {
        let mut sec = String::new();

        let binaries: Vec<&str> = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .map(|v| v.name.as_str())
            .collect();
        if !binaries.is_empty() {
            writeln!(sec, "Binary").unwrap();
            for name in &binaries {
                writeln!(sec, "  {name}").unwrap();
            }
        }

        let integers: Vec<&str> = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Integer)
            .map(|v| v.name.as_str())
            .collect();
        if !integers.is_empty() {
            writeln!(sec, "General").unwrap();
            for name in &integers {
                writeln!(sec, "  {name}").unwrap();
            }
        }

        sec
    }

    // ── Parameter file (.prm) ─────────────────────────────────────

    /// Generate a Gurobi `.prm` parameter file from the current configuration.
    pub fn generate_parameter_file(&self) -> String {
        let mut prm = String::new();
        writeln!(prm, "# Gurobi parameter file — generated by bicut-compiler").unwrap();

        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::float("MIPGap", self.config.mip_gap),)
        )
        .unwrap();
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::float("TimeLimit", self.config.time_limit_sec),)
        )
        .unwrap();
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::int("Threads", self.config.threads as i64),)
        )
        .unwrap();
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::int(
                "Presolve",
                self.config.presolve.param_value()
            ),)
        )
        .unwrap();
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::int(
                "Method",
                self.config.method.param_value()
            ),)
        )
        .unwrap();
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::float(
                "FeasibilityTol",
                self.config.feasibility_tol
            ),)
        )
        .unwrap();
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::float(
                "IntFeasTol",
                self.config.integrality_tol
            ),)
        )
        .unwrap();
        let flag_val: i64 = if self.config.output_flag { 1 } else { 0 };
        writeln!(
            prm,
            "{}",
            Self::format_gurobi_param(&GurobiParam::int("OutputFlag", flag_val),)
        )
        .unwrap();

        prm
    }

    /// Format a single Gurobi parameter as a `.prm` file line.
    pub fn format_gurobi_param(param: &GurobiParam) -> String {
        match &param.value {
            GurobiParamValue::Int(v) => format!("{} {v}", param.name),
            GurobiParamValue::Float(v) => format!("{} {v:.6e}", param.name),
            GurobiParamValue::Str(v) => format!("{} {v}", param.name),
        }
    }

    // ── Complexity estimation & auto-tuning ───────────────────────

    /// Estimate the structural complexity of a MILP instance.
    pub fn estimate_model_complexity(milp: &MilpProblem) -> ModelComplexity {
        let num_binary = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .count();
        let num_integer = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Integer)
            .count();
        let num_continuous = milp
            .variables
            .iter()
            .filter(|v| v.var_type == VarType::Continuous)
            .count();

        let num_constraints =
            milp.constraints.len() + milp.indicator_constraints.len() + milp.sos1_sets.len();

        let num_nonzeros: usize = milp
            .constraints
            .iter()
            .map(|c| c.coeffs.len())
            .sum::<usize>()
            + milp
                .indicator_constraints
                .iter()
                .map(|ic| ic.coeffs.len())
                .sum::<usize>();

        let total_vars = milp.variables.len();
        let max_entries = if total_vars > 0 && num_constraints > 0 {
            total_vars * num_constraints
        } else {
            1
        };
        let density = num_nonzeros as f64 / max_entries as f64;

        let discrete_count = num_binary + num_integer;
        let estimated_difficulty = if discrete_count == 0 && num_constraints < 500 {
            DifficultyLevel::Easy
        } else if discrete_count < 50 && num_constraints < 2000 {
            DifficultyLevel::Medium
        } else if discrete_count < 500 && num_constraints < 10_000 {
            DifficultyLevel::Hard
        } else {
            DifficultyLevel::VeryHard
        };

        ModelComplexity {
            num_binary,
            num_integer,
            num_continuous,
            num_constraints,
            num_nonzeros,
            density,
            estimated_difficulty,
        }
    }

    /// Suggest Gurobi parameters given a [`ModelComplexity`] estimate.
    pub fn suggest_parameters(complexity: &ModelComplexity) -> Vec<GurobiParam> {
        let mut params = Vec::new();

        match complexity.estimated_difficulty {
            DifficultyLevel::Easy => {
                params.push(GurobiParam::int(
                    "Presolve",
                    GurobiPresolve::Auto.param_value(),
                ));
                params.push(GurobiParam::float("MIPGap", 1e-4));
                params.push(GurobiParam::int(
                    "Method",
                    GurobiMethod::DualSimplex.param_value(),
                ));
            }
            DifficultyLevel::Medium => {
                params.push(GurobiParam::int(
                    "Presolve",
                    GurobiPresolve::Conservative.param_value(),
                ));
                params.push(GurobiParam::float("MIPGap", 1e-4));
                params.push(GurobiParam::int("Method", GurobiMethod::Auto.param_value()));
                params.push(GurobiParam::int("Cuts", 2));
            }
            DifficultyLevel::Hard => {
                params.push(GurobiParam::int(
                    "Presolve",
                    GurobiPresolve::Aggressive.param_value(),
                ));
                params.push(GurobiParam::float("MIPGap", 1e-3));
                params.push(GurobiParam::int(
                    "Method",
                    GurobiMethod::Barrier.param_value(),
                ));
                params.push(GurobiParam::int("Cuts", 3));
                params.push(GurobiParam::int("MIPFocus", 1));
            }
            DifficultyLevel::VeryHard => {
                params.push(GurobiParam::int(
                    "Presolve",
                    GurobiPresolve::Aggressive.param_value(),
                ));
                params.push(GurobiParam::float("MIPGap", 5e-3));
                params.push(GurobiParam::int(
                    "Method",
                    GurobiMethod::Barrier.param_value(),
                ));
                params.push(GurobiParam::int("Cuts", 3));
                params.push(GurobiParam::int("MIPFocus", 2));
                params.push(GurobiParam::int("Heuristics", 20));
                params.push(GurobiParam::float("ImproveStartTime", 60.0));
            }
        }

        if complexity.density > 0.3 {
            params.push(GurobiParam::int("AggFill", 0));
        }

        if complexity.num_binary > 200 {
            params.push(GurobiParam::int("Symmetry", 2));
        }

        params
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{MilpConstraint, MilpVariable};

    /// Build a small test model used across several tests.
    fn sample_milp() -> MilpProblem {
        let mut m = MilpProblem::new("test_model");
        m.sense = OptDirection::Minimize;

        let x0 = m.add_variable(MilpVariable {
            name: "x".to_string(),
            lower_bound: 0.0,
            upper_bound: 10.0,
            obj_coeff: 1.0,
            var_type: VarType::Continuous,
        });
        let x1 = m.add_variable(MilpVariable::binary("y"));
        m.set_obj_coeff(x1, 2.0);

        let mut c = MilpConstraint::new("budget", ConstraintSense::Le, 5.0);
        c.add_term(x0, 1.0);
        c.add_term(x1, 3.0);
        m.add_constraint(c);

        let mut c2 = MilpConstraint::new("link", ConstraintSense::Ge, 1.0);
        c2.add_term(x0, 1.0);
        c2.add_term(x1, -1.0);
        m.add_constraint(c2);

        m.add_sos1_set(Sos1Set {
            name: "sos_xy".to_string(),
            sos_type: 1,
            members: vec![x0, x1],
            weights: vec![1.0, 2.0],
        });

        m.add_indicator_constraint(IndicatorConstraint {
            name: "ind_link".to_string(),
            binary_var: x1,
            active_value: true,
            coeffs: vec![(x0, 1.0)],
            sense: ConstraintSense::Le,
            rhs: 4.0,
        });

        m
    }

    #[test]
    fn test_parameter_file_generation() {
        let cfg = GurobiConfig {
            mip_gap: 1e-3,
            time_limit_sec: 120.0,
            threads: 4,
            presolve: GurobiPresolve::Aggressive,
            method: GurobiMethod::Barrier,
            feasibility_tol: 1e-7,
            integrality_tol: 1e-6,
            output_flag: false,
            ..GurobiConfig::default()
        };
        let emitter = GurobiEmitter::new(cfg);
        let prm = emitter.generate_parameter_file();

        assert!(prm.contains("MIPGap"));
        assert!(prm.contains("TimeLimit"));
        assert!(prm.contains("Threads 4"));
        assert!(prm.contains("Presolve 2"));
        assert!(prm.contains("Method 2"));
        assert!(prm.contains("OutputFlag 0"));
    }

    #[test]
    fn test_indicator_constraint_format() {
        let milp = sample_milp();
        let emitter = GurobiEmitter::new(GurobiConfig::default());
        let section = emitter.write_indicator_constraints(&milp);

        assert!(section.contains("General Constraints"));
        assert!(section.contains("ind_link:"));
        assert!(section.contains("y = 1 ->"));
        assert!(section.contains("<= 4"));
    }

    #[test]
    fn test_lazy_constraint_annotation() {
        let milp = sample_milp();
        let emitter = GurobiEmitter::new(GurobiConfig {
            use_lazy_constraints: true,
            ..Default::default()
        });
        let section = emitter.write_lazy_constraints(&milp);

        assert!(section.contains("Lazy Constraints"));
        assert!(section.contains("budget_lazy:"));
        assert!(section.contains("link_lazy:"));
        assert!(section.contains("<= 5"));
        assert!(section.contains(">= 1"));
    }

    #[test]
    fn test_sos_set_writing() {
        let milp = sample_milp();
        let emitter = GurobiEmitter::new(GurobiConfig::default());
        let section = emitter.write_sos_sets(&milp);

        assert!(section.contains("SOS"));
        assert!(section.contains("sos_xy: S1 ::"));
        assert!(section.contains("x: 1"));
        assert!(section.contains("y: 2"));
    }

    #[test]
    fn test_model_complexity_estimation() {
        let milp = sample_milp();
        let cx = GurobiEmitter::estimate_model_complexity(&milp);

        assert_eq!(cx.num_binary, 1);
        assert_eq!(cx.num_continuous, 1);
        assert_eq!(cx.num_integer, 0);
        assert_eq!(cx.num_constraints, 4); // 2 linear + 1 sos + 1 indicator
        assert!(cx.density > 0.0);
        assert!(matches!(
            cx.estimated_difficulty,
            DifficultyLevel::Easy | DifficultyLevel::Medium
        ));
    }

    #[test]
    fn test_parameter_suggestions() {
        let easy = ModelComplexity {
            num_binary: 0,
            num_integer: 0,
            num_continuous: 50,
            num_constraints: 30,
            num_nonzeros: 100,
            density: 0.05,
            estimated_difficulty: DifficultyLevel::Easy,
        };
        let params = GurobiEmitter::suggest_parameters(&easy);
        assert!(!params.is_empty());
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"Presolve"));
        assert!(names.contains(&"MIPGap"));

        let hard = ModelComplexity {
            num_binary: 300,
            num_integer: 50,
            num_continuous: 1000,
            num_constraints: 5000,
            num_nonzeros: 20000,
            density: 0.4,
            estimated_difficulty: DifficultyLevel::Hard,
        };
        let hp = GurobiEmitter::suggest_parameters(&hard);
        let hnames: Vec<&str> = hp.iter().map(|p| p.name.as_str()).collect();
        assert!(hnames.contains(&"MIPFocus"));
        assert!(hnames.contains(&"AggFill"));
        assert!(hnames.contains(&"Symmetry"));
    }

    #[test]
    fn test_full_gurobi_emit() {
        let milp = sample_milp();
        let cfg = GurobiConfig {
            use_indicator_constraints: true,
            use_lazy_constraints: true,
            use_sos1: true,
            ..GurobiConfig::default()
        };
        let emitter = GurobiEmitter::new(cfg);
        let output = emitter.emit(&milp).expect("emit should succeed");

        assert!(output.model_content.contains("Minimize"));
        assert!(output.model_content.contains("Subject To"));
        assert!(output.model_content.contains("General Constraints"));
        assert!(output.model_content.contains("Lazy Constraints"));
        assert!(output.model_content.contains("SOS"));
        assert!(output.model_content.contains("Bounds"));
        assert!(output.model_content.contains("Binary"));
        assert!(output.model_content.contains("End"));

        assert_eq!(output.indicators_count, 1);
        assert_eq!(output.lazy_count, 2);
        assert_eq!(output.sos_count, 1);
        assert_eq!(output.format, "grb_lp");

        assert!(output.parameter_content.contains("MIPGap"));
    }
}
