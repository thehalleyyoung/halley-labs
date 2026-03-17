//! HiGHS-specific backend emission for bilevel MILP compilation.
//!
//! Generates MPS/LP model files, HiGHS options files, and row-generation
//! metadata suitable for the HiGHS open-source LP/MIP solver.

use crate::CompilerError;
use bicut_types::{ConstraintSense, OptDirection};
use serde::{Deserialize, Serialize};
use std::fmt;

// -- Pipeline types (local until crate::pipeline is created) ----------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilpVariable {
    pub name: String,
    pub var_type: VarType,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub obj_coeff: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilpConstraint {
    pub name: String,
    pub coefficients: Vec<(usize, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sos1Set {
    pub name: String,
    pub members: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConstraint {
    pub name: String,
    pub binary_var: usize,
    pub active_value: bool,
    pub coefficients: Vec<(usize, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilpProblem {
    pub name: String,
    pub direction: OptDirection,
    pub variables: Vec<MilpVariable>,
    pub constraints: Vec<MilpConstraint>,
    pub sos1_sets: Vec<Sos1Set>,
    pub indicators: Vec<IndicatorConstraint>,
}

// -- HiGHS enums ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HighsSolverType {
    Simplex,
    Ipm,
    MipSolver,
}

impl fmt::Display for HighsSolverType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Simplex => "simplex",
            Self::Ipm => "ipm",
            Self::MipSolver => "mip",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HighsPresolve {
    Off,
    Choose,
    On,
}

impl fmt::Display for HighsPresolve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Choose => "choose",
            Self::On => "on",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HighsParallel {
    Off,
    Choose,
    On,
}

impl fmt::Display for HighsParallel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Choose => "choose",
            Self::On => "on",
        })
    }
}

// -- Parameter types --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HighsParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl fmt::Display for HighsParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", if *v { "true" } else { "false" }),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighsParam {
    pub name: String,
    pub value: HighsParamValue,
}

// -- Configuration ----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighsConfig {
    pub solver_type: HighsSolverType,
    pub time_limit_sec: f64,
    pub mip_rel_gap: f64,
    pub presolve: HighsPresolve,
    pub simplex_strategy: i32,
    pub parallel: HighsParallel,
    pub output_flag: bool,
    pub ranging: bool,
    pub ipm_iteration_limit: i32,
    pub mip_max_nodes: i64,
    pub primal_feasibility_tol: f64,
    pub dual_feasibility_tol: f64,
    pub allowed_matrix_scale_factor: i32,
}

impl Default for HighsConfig {
    fn default() -> Self {
        Self {
            solver_type: HighsSolverType::MipSolver,
            time_limit_sec: 3600.0,
            mip_rel_gap: 1e-4,
            presolve: HighsPresolve::Choose,
            simplex_strategy: 1,
            parallel: HighsParallel::Choose,
            output_flag: true,
            ranging: false,
            ipm_iteration_limit: 10_000,
            mip_max_nodes: i64::MAX,
            primal_feasibility_tol: 1e-7,
            dual_feasibility_tol: 1e-7,
            allowed_matrix_scale_factor: 20,
        }
    }
}

// -- Output / classification types ------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowGenerationEntry {
    pub row_index: usize,
    pub coefficients: Vec<(usize, f64)>,
    pub rhs: f64,
    pub sense: ConstraintSense,
    pub is_lazy: bool,
    pub priority: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintClassification {
    pub core_rows: Vec<usize>,
    pub cut_rows: Vec<usize>,
    pub lazy_rows: Vec<usize>,
    pub num_dense_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighsOutput {
    pub model_content: String,
    pub options_content: String,
    pub format: String,
    pub row_pool: Vec<RowGenerationEntry>,
}

// -- Emitter ----------------------------------------------------------------

const DENSE_FRAC: f64 = 0.25;
const LARGE_COEFF: f64 = 1e6;

pub struct HighsEmitter {
    config: HighsConfig,
}

impl HighsEmitter {
    pub fn new(config: HighsConfig) -> Self {
        Self { config }
    }

    /// Emit a complete HiGHS bundle (model + options + row pool).
    pub fn emit(&self, milp: &MilpProblem) -> Result<HighsOutput, CompilerError> {
        let model_content = self.emit_mps(milp)?;
        let options_content = self.generate_options_file();
        let row_pool = self.generate_row_pool(milp);
        Ok(HighsOutput {
            model_content,
            options_content,
            format: "mps".into(),
            row_pool,
        })
    }

    /// Generate free-MPS with HiGHS-compatible extensions.
    pub fn emit_mps(&self, milp: &MilpProblem) -> Result<String, CompilerError> {
        if milp.variables.is_empty() {
            return Err(CompilerError::InvalidProblem(
                "MILP has no variables".into(),
            ));
        }
        let vname = |j: usize| {
            let v = &milp.variables[j];
            if v.name.is_empty() {
                format!("x{j}")
            } else {
                v.name.clone()
            }
        };
        let rname = |i: usize| {
            let c = &milp.constraints[i];
            if c.name.is_empty() {
                format!("R{i}")
            } else {
                c.name.clone()
            }
        };
        let mut b = String::with_capacity(milp.variables.len() * 80);
        b.push_str(&format!("NAME          {}\nROWS\n N  OBJ\n", milp.name));
        for (i, c) in milp.constraints.iter().enumerate() {
            let ch = match c.sense {
                ConstraintSense::Le => 'L',
                ConstraintSense::Ge => 'G',
                ConstraintSense::Eq => 'E',
            };
            b.push_str(&format!(" {ch}  {}\n", rname(i)));
        }
        b.push_str("COLUMNS\n");
        let mut in_int = false;
        for (j, var) in milp.variables.iter().enumerate() {
            let is_int = matches!(var.var_type, VarType::Integer | VarType::Binary);
            if is_int && !in_int {
                b.push_str("    MARKER  'MARKER'  'INTORG'\n");
                in_int = true;
            } else if !is_int && in_int {
                b.push_str("    MARKER  'MARKER'  'INTEND'\n");
                in_int = false;
            }
            if var.obj_coeff.abs() > f64::EPSILON {
                b.push_str(&format!("    {}  OBJ  {:.12}\n", vname(j), var.obj_coeff));
            }
            for (i, c) in milp.constraints.iter().enumerate() {
                for &(col, coeff) in &c.coefficients {
                    if col == j && coeff.abs() > f64::EPSILON {
                        b.push_str(&format!("    {}  {}  {coeff:.12}\n", vname(j), rname(i)));
                    }
                }
            }
        }
        if in_int {
            b.push_str("    MARKER  'MARKER'  'INTEND'\n");
        }
        b.push_str("RHS\n");
        for (i, c) in milp.constraints.iter().enumerate() {
            if c.rhs.abs() > f64::EPSILON {
                b.push_str(&format!("    RHS  {}  {:.12}\n", rname(i), c.rhs));
            }
        }
        b.push_str("BOUNDS\n");
        for (j, var) in milp.variables.iter().enumerate() {
            match var.var_type {
                VarType::Binary => b.push_str(&format!(" BV BOUND  {}\n", vname(j))),
                _ => {
                    if var.lower_bound != 0.0 {
                        b.push_str(&format!(
                            " LO BOUND  {}  {:.12}\n",
                            vname(j),
                            var.lower_bound
                        ));
                    }
                    if var.upper_bound < 1e20 {
                        b.push_str(&format!(
                            " UP BOUND  {}  {:.12}\n",
                            vname(j),
                            var.upper_bound
                        ));
                    }
                }
            }
        }
        if !milp.sos1_sets.is_empty() {
            b.push_str("SOS\n");
            for (k, sos) in milp.sos1_sets.iter().enumerate() {
                let sn = if sos.name.is_empty() {
                    format!("S{k}")
                } else {
                    sos.name.clone()
                };
                b.push_str(&format!(" S1 {sn}\n"));
                for &(col, wt) in &sos.members {
                    b.push_str(&format!("    {}  {wt:.6}\n", vname(col)));
                }
            }
        }
        b.push_str("ENDATA\n");
        Ok(b)
    }

    /// Generate LP-format string compatible with HiGHS.
    pub fn emit_lp(&self, milp: &MilpProblem) -> Result<String, CompilerError> {
        if milp.variables.is_empty() {
            return Err(CompilerError::InvalidProblem(
                "MILP has no variables".into(),
            ));
        }
        let vn = |j: usize| {
            let v = &milp.variables[j];
            if v.name.is_empty() {
                format!("x{j}")
            } else {
                v.name.clone()
            }
        };
        let mut b = String::with_capacity(milp.variables.len() * 60);
        let dir = match milp.direction {
            OptDirection::Minimize => "Minimize",
            OptDirection::Maximize => "Maximize",
        };
        b.push_str(&format!("\\ Problem: {}\n{dir}\n obj: ", milp.name));
        let mut first = true;
        for (j, var) in milp.variables.iter().enumerate() {
            if var.obj_coeff.abs() <= f64::EPSILON {
                continue;
            }
            if first {
                b.push_str(&format!("{} {}", var.obj_coeff, vn(j)));
                first = false;
            } else if var.obj_coeff > 0.0 {
                b.push_str(&format!(" + {} {}", var.obj_coeff, vn(j)));
            } else {
                b.push_str(&format!(" - {} {}", var.obj_coeff.abs(), vn(j)));
            }
        }
        if first {
            b.push('0');
        }
        b.push_str("\nSubject To\n");
        for (i, c) in milp.constraints.iter().enumerate() {
            let cn = if c.name.is_empty() {
                format!("R{i}")
            } else {
                c.name.clone()
            };
            b.push_str(&format!(" {cn}: "));
            let mut cf = true;
            for &(col, coeff) in &c.coefficients {
                if coeff.abs() <= f64::EPSILON {
                    continue;
                }
                if cf {
                    b.push_str(&format!("{coeff} {}", vn(col)));
                    cf = false;
                } else if coeff > 0.0 {
                    b.push_str(&format!(" + {coeff} {}", vn(col)));
                } else {
                    b.push_str(&format!(" - {} {}", coeff.abs(), vn(col)));
                }
            }
            let ss = match c.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "=",
            };
            b.push_str(&format!(" {ss} {}\n", c.rhs));
        }
        b.push_str("Bounds\n");
        for (j, var) in milp.variables.iter().enumerate() {
            if var.var_type == VarType::Binary {
                continue;
            }
            if var.lower_bound == 0.0 && var.upper_bound >= 1e20 {
                continue;
            }
            if var.upper_bound >= 1e20 {
                b.push_str(&format!(" {} <= {}\n", var.lower_bound, vn(j)));
            } else {
                b.push_str(&format!(
                    " {} <= {} <= {}\n",
                    var.lower_bound,
                    vn(j),
                    var.upper_bound
                ));
            }
        }
        let ints: Vec<_> = milp
            .variables
            .iter()
            .enumerate()
            .filter(|(_, v)| v.var_type == VarType::Integer)
            .map(|(j, _)| vn(j))
            .collect();
        if !ints.is_empty() {
            b.push_str("General\n");
            for n in &ints {
                b.push_str(&format!(" {n}\n"));
            }
        }
        let bins: Vec<_> = milp
            .variables
            .iter()
            .enumerate()
            .filter(|(_, v)| v.var_type == VarType::Binary)
            .map(|(j, _)| vn(j))
            .collect();
        if !bins.is_empty() {
            b.push_str("Binary\n");
            for n in &bins {
                b.push_str(&format!(" {n}\n"));
            }
        }
        b.push_str("End\n");
        Ok(b)
    }

    /// Generate HiGHS options file content from the current configuration.
    pub fn generate_options_file(&self) -> String {
        let c = &self.config;
        [
            format!("solver = {}", c.solver_type),
            format!("time_limit = {}", c.time_limit_sec),
            format!("mip_rel_gap = {}", c.mip_rel_gap),
            format!("presolve = {}", c.presolve),
            format!("simplex_strategy = {}", c.simplex_strategy),
            format!("parallel = {}", c.parallel),
            format!(
                "output_flag = {}",
                if c.output_flag { "true" } else { "false" }
            ),
            format!("ranging = {}", if c.ranging { "on" } else { "off" }),
            format!("ipm_iteration_limit = {}", c.ipm_iteration_limit),
            format!("mip_max_nodes = {}", c.mip_max_nodes),
            format!(
                "primal_feasibility_tolerance = {}",
                c.primal_feasibility_tol
            ),
            format!("dual_feasibility_tolerance = {}", c.dual_feasibility_tol),
            format!(
                "allowed_matrix_scale_factor = {}",
                c.allowed_matrix_scale_factor
            ),
        ]
        .join("\n")
            + "\n"
    }

    /// Build a pool of rows suitable for lazy generation or cut callbacks.
    pub fn generate_row_pool(&self, milp: &MilpProblem) -> Vec<RowGenerationEntry> {
        let cc = self.classify_constraints(milp);
        let mut pool: Vec<RowGenerationEntry> = cc
            .lazy_rows
            .iter()
            .map(|&i| {
                let c = &milp.constraints[i];
                RowGenerationEntry {
                    row_index: i,
                    coefficients: c.coefficients.clone(),
                    rhs: c.rhs,
                    sense: c.sense,
                    is_lazy: true,
                    priority: 1,
                }
            })
            .chain(cc.cut_rows.iter().map(|&i| {
                let c = &milp.constraints[i];
                RowGenerationEntry {
                    row_index: i,
                    coefficients: c.coefficients.clone(),
                    rhs: c.rhs,
                    sense: c.sense,
                    is_lazy: false,
                    priority: 0,
                }
            }))
            .collect();
        pool.sort_by(|a, b| b.priority.cmp(&a.priority));
        pool
    }

    /// Suggest HiGHS parameters tuned to the given problem's structure.
    pub fn suggest_parameters(&self, milp: &MilpProblem) -> Vec<HighsParam> {
        let nv = milp.variables.len();
        let nc = milp.constraints.len();
        let ni = milp
            .variables
            .iter()
            .filter(|v| matches!(v.var_type, VarType::Integer | VarType::Binary))
            .count();
        let mut p = Vec::new();
        if nv + nc > 5000 {
            p.push(HighsParam {
                name: "presolve".into(),
                value: HighsParamValue::Str("on".into()),
            });
        }
        if nv > 1000 {
            p.push(HighsParam {
                name: "parallel".into(),
                value: HighsParamValue::Str("on".into()),
            });
        }
        if ni > 0 && ni <= 50 {
            p.push(HighsParam {
                name: "mip_rel_gap".into(),
                value: HighsParamValue::Float(1e-6),
            });
        }
        if nc > 0 && nv as f64 / nc as f64 > 5.0 {
            p.push(HighsParam {
                name: "simplex_strategy".into(),
                value: HighsParamValue::Int(4),
            });
        }
        let int_frac = if nv > 0 { ni as f64 / nv as f64 } else { 0.0 };
        if int_frac > 0.5 && nv > 200 {
            p.push(HighsParam {
                name: "mip_max_nodes".into(),
                value: HighsParamValue::Int(500_000),
            });
        }
        let big = milp
            .constraints
            .iter()
            .any(|c| c.coefficients.iter().any(|&(_, v)| v.abs() > LARGE_COEFF));
        if big {
            p.push(HighsParam {
                name: "primal_feasibility_tolerance".into(),
                value: HighsParamValue::Float(1e-9),
            });
            p.push(HighsParam {
                name: "dual_feasibility_tolerance".into(),
                value: HighsParamValue::Float(1e-9),
            });
        }
        p
    }

    /// Classify constraints into core, cut, and lazy buckets.
    pub fn classify_constraints(&self, milp: &MilpProblem) -> ConstraintClassification {
        let nv = milp.variables.len();
        let thresh = (nv as f64 * DENSE_FRAC).max(1.0) as usize;
        let mut cc = ConstraintClassification::default();
        for (i, c) in milp.constraints.iter().enumerate() {
            let dense = c.coefficients.len() >= thresh;
            if dense {
                cc.num_dense_rows += 1;
            }
            let big = c.coefficients.iter().any(|&(_, v)| v.abs() > LARGE_COEFF);
            if dense && big {
                cc.lazy_rows.push(i);
            } else if dense || big {
                cc.cut_rows.push(i);
            } else {
                cc.core_rows.push(i);
            }
        }
        cc
    }

    /// Rough heuristic solver-time estimate in seconds.
    pub fn estimate_solver_time(milp: &MilpProblem) -> f64 {
        let n = milp.variables.len() as f64;
        let m = milp.constraints.len() as f64;
        let ni = milp
            .variables
            .iter()
            .filter(|v| matches!(v.var_type, VarType::Integer | VarType::Binary))
            .count() as f64;
        let lp_base = (m * n).sqrt() * 1e-4;
        let int_frac = if n > 0.0 { ni / n } else { 0.0 };
        let mip_factor = if ni > 0.0 {
            (1.0 + int_frac).powf(2.0) * ni.ln().max(1.0)
        } else {
            1.0
        };
        let nnz: usize = milp.constraints.iter().map(|c| c.coefficients.len()).sum();
        let sp = 1.0 + (nnz as f64 / (n * m).max(1.0)).min(1.0);
        (lp_base * mip_factor * sp).max(0.01)
    }

    /// Format a single HiGHS option line.
    pub fn format_highs_option(name: &str, value: &HighsParamValue) -> String {
        format!("{name} = {value}")
    }
}

// == Tests ==================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_milp() -> MilpProblem {
        MilpProblem {
            name: "test".into(),
            direction: OptDirection::Minimize,
            variables: vec![
                MilpVariable {
                    name: "x0".into(),
                    var_type: VarType::Continuous,
                    lower_bound: 0.0,
                    upper_bound: 10.0,
                    obj_coeff: 1.0,
                },
                MilpVariable {
                    name: "x1".into(),
                    var_type: VarType::Binary,
                    lower_bound: 0.0,
                    upper_bound: 1.0,
                    obj_coeff: 2.0,
                },
                MilpVariable {
                    name: "x2".into(),
                    var_type: VarType::Integer,
                    lower_bound: 0.0,
                    upper_bound: 5.0,
                    obj_coeff: -1.0,
                },
            ],
            constraints: vec![
                MilpConstraint {
                    name: "c0".into(),
                    coefficients: vec![(0, 1.0), (1, 3.0)],
                    sense: ConstraintSense::Le,
                    rhs: 10.0,
                },
                MilpConstraint {
                    name: "c1".into(),
                    coefficients: vec![(0, 2.0), (2, -1.0)],
                    sense: ConstraintSense::Ge,
                    rhs: 0.0,
                },
                MilpConstraint {
                    name: "c2".into(),
                    coefficients: vec![(0, 1.0), (1, 1.0), (2, 1.0)],
                    sense: ConstraintSense::Eq,
                    rhs: 5.0,
                },
            ],
            sos1_sets: vec![],
            indicators: vec![],
        }
    }

    #[test]
    fn test_mps_emission() {
        let e = HighsEmitter::new(HighsConfig::default());
        let mps = e.emit_mps(&small_milp()).unwrap();
        assert!(mps.contains("NAME"));
        assert!(mps.contains("ROWS"));
        assert!(mps.contains("COLUMNS"));
        assert!(mps.contains("RHS"));
        assert!(mps.contains("BOUNDS"));
        assert!(mps.contains("ENDATA"));
        assert!(mps.contains(" L  c0"));
        assert!(mps.contains(" G  c1"));
        assert!(mps.contains(" E  c2"));
        assert!(mps.contains("BV BOUND  x1"));
    }

    #[test]
    fn test_lp_emission() {
        let e = HighsEmitter::new(HighsConfig::default());
        let lp = e.emit_lp(&small_milp()).unwrap();
        assert!(lp.starts_with("\\"));
        assert!(lp.contains("Minimize"));
        assert!(lp.contains("Subject To"));
        assert!(lp.contains("Bounds"));
        assert!(lp.contains("Binary"));
        assert!(lp.contains("General"));
        assert!(lp.contains("End"));
        assert!(lp.contains("c0:"));
    }

    #[test]
    fn test_options_file() {
        let cfg = HighsConfig {
            solver_type: HighsSolverType::Simplex,
            time_limit_sec: 120.0,
            mip_rel_gap: 1e-3,
            presolve: HighsPresolve::On,
            simplex_strategy: 2,
            parallel: HighsParallel::Off,
            output_flag: false,
            ranging: true,
            ipm_iteration_limit: 5000,
            mip_max_nodes: 100_000,
            primal_feasibility_tol: 1e-8,
            dual_feasibility_tol: 1e-8,
            allowed_matrix_scale_factor: 10,
        };
        let opts = HighsEmitter::new(cfg).generate_options_file();
        assert!(opts.contains("solver = simplex"));
        assert!(opts.contains("time_limit = 120"));
        assert!(opts.contains("presolve = on"));
        assert!(opts.contains("parallel = off"));
        assert!(opts.contains("output_flag = false"));
        assert!(opts.contains("ranging = on"));
    }

    #[test]
    fn test_row_classification() {
        // Need enough variables so DENSE_FRAC threshold is > 1
        // With 12 vars, thresh = (12*0.25).max(1) = 3
        let mut vars: Vec<MilpVariable> = (0..12)
            .map(|i| MilpVariable {
                name: format!("x{i}"),
                var_type: VarType::Continuous,
                lower_bound: 0.0,
                upper_bound: 10.0,
                obj_coeff: 1.0,
            })
            .collect();
        vars[1].var_type = VarType::Binary;
        vars[1].upper_bound = 1.0;
        let mut milp = MilpProblem {
            name: "classify_test".into(),
            direction: OptDirection::Minimize,
            variables: vars,
            constraints: vec![
                // Sparse, small coeffs → core
                MilpConstraint {
                    name: "sparse".into(),
                    coefficients: vec![(0, 1.0), (1, 3.0)],
                    sense: ConstraintSense::Le,
                    rhs: 10.0,
                },
                // Sparse, small coeffs → core
                MilpConstraint {
                    name: "sparse2".into(),
                    coefficients: vec![(2, 2.0)],
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                },
            ],
            sos1_sets: vec![],
            indicators: vec![],
        };
        // Add a dense row with a large coefficient → should be lazy or cut
        milp.constraints.push(MilpConstraint {
            name: "dense_big".into(),
            coefficients: vec![(0, 2e6), (1, 1.0), (2, -3.0)],
            sense: ConstraintSense::Le,
            rhs: 100.0,
        });
        let cc = HighsEmitter::new(HighsConfig::default()).classify_constraints(&milp);
        assert!(!cc.core_rows.is_empty());
        let di = milp.constraints.len() - 1;
        assert!(
            cc.cut_rows.contains(&di) || cc.lazy_rows.contains(&di),
            "dense row with large coeff must not be core"
        );
    }

    #[test]
    fn test_parameter_suggestions() {
        let params = HighsEmitter::new(HighsConfig::default()).suggest_parameters(&small_milp());
        let gap = params.iter().find(|p| p.name == "mip_rel_gap");
        assert!(
            gap.is_some(),
            "should suggest mip_rel_gap for small integer problem"
        );
        assert_eq!(gap.unwrap().value, HighsParamValue::Float(1e-6));
    }

    #[test]
    fn test_time_estimation() {
        let t = HighsEmitter::estimate_solver_time(&small_milp());
        assert!(t > 0.0, "time estimate must be positive");
        assert!(t < 10.0, "small problem should have small estimate");
        let trivial = MilpProblem {
            name: "t".into(),
            direction: OptDirection::Minimize,
            variables: vec![MilpVariable {
                name: "y".into(),
                var_type: VarType::Continuous,
                lower_bound: 0.0,
                upper_bound: 1.0,
                obj_coeff: 1.0,
            }],
            constraints: vec![],
            sos1_sets: vec![],
            indicators: vec![],
        };
        assert!(HighsEmitter::estimate_solver_time(&trivial) >= 0.01);
    }

    #[test]
    fn test_full_emit() {
        let out = HighsEmitter::new(HighsConfig::default())
            .emit(&small_milp())
            .unwrap();
        assert_eq!(out.format, "mps");
        assert!(out.model_content.contains("ENDATA"));
        assert!(out.options_content.contains("solver = mip"));
        assert!(out.row_pool.is_empty() || out.row_pool.iter().all(|r| r.coefficients.len() <= 3));
    }
}
