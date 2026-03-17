//! Compiler pipeline types for bilevel-to-MILP reformulation.
//!
//! Provides `MilpProblem` — the mixed-integer linear program produced by
//! KKT / strong-duality reformulations — together with supporting constraint
//! types (SOS-1, indicator), and the top-level pipeline orchestration types.

use bicut_types::{BilevelProblem, ConstraintSense, LpProblem, OptDirection};
use serde::{Deserialize, Serialize};

use crate::bigm::{BigMComputer, BigMConfig};
use crate::certificate_gen::CompilerCertificate;
use crate::ReformulationType;
use crate::{CompilerConfig, CompilerError, CompilerResult};

// ── Variable type ──────────────────────────────────────────────────

/// Variable type in the reformulated MILP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

impl Default for VarType {
    fn default() -> Self {
        VarType::Continuous
    }
}

// ── MILP variable ──────────────────────────────────────────────────

/// A single variable in the MILP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilpVariable {
    pub name: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub obj_coeff: f64,
    pub var_type: VarType,
}

impl MilpVariable {
    pub fn continuous(name: &str, lb: f64, ub: f64) -> Self {
        Self {
            name: name.to_string(),
            lower_bound: lb,
            upper_bound: ub,
            obj_coeff: 0.0,
            var_type: VarType::Continuous,
        }
    }

    pub fn binary(name: &str) -> Self {
        Self {
            name: name.to_string(),
            lower_bound: 0.0,
            upper_bound: 1.0,
            obj_coeff: 0.0,
            var_type: VarType::Binary,
        }
    }
}

// ── MILP constraint ────────────────────────────────────────────────

/// A linear constraint in the MILP: Σ coeff_j · x_j  sense  rhs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilpConstraint {
    pub name: String,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub coeffs: Vec<(usize, f64)>,
    /// Optional range value for MPS range rows (rarely used).
    #[serde(default)]
    pub range: Option<f64>,
}

impl MilpConstraint {
    pub fn new(name: &str, sense: ConstraintSense, rhs: f64) -> Self {
        Self {
            name: name.to_string(),
            sense,
            rhs,
            coeffs: Vec::new(),
            range: None,
        }
    }

    pub fn add_term(&mut self, var_idx: usize, coeff: f64) {
        if coeff.abs() > 1e-20 {
            self.coeffs.push((var_idx, coeff));
        }
    }
}

// ── SOS-1 set ──────────────────────────────────────────────────────

/// An SOS-1 (Special Ordered Set type 1) constraint: at most one member
/// may be non-zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sos1Set {
    pub name: String,
    pub sos_type: u8,
    pub members: Vec<usize>,
    pub weights: Vec<f64>,
}

// ── Indicator constraint ───────────────────────────────────────────

/// Indicator constraint: `binary_var == active_value` ⟹ linear constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConstraint {
    pub name: String,
    pub binary_var: usize,
    pub active_value: bool,
    pub coeffs: Vec<(usize, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

// ── MilpProblem ────────────────────────────────────────────────────

/// A complete mixed-integer linear program produced by bilevel reformulation.
///
/// Uses an object-oriented representation: each variable and constraint is
/// stored as its own struct, making the model easy to query from backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilpProblem {
    pub name: String,
    pub sense: OptDirection,
    pub variables: Vec<MilpVariable>,
    pub constraints: Vec<MilpConstraint>,
    pub sos1_sets: Vec<Sos1Set>,
    pub indicator_constraints: Vec<IndicatorConstraint>,
}

/// Kept for backward-compat with reformulation passes that build via the OOP pattern.
#[allow(dead_code)]
impl MilpProblem {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sense: OptDirection::Minimize,
            variables: Vec::new(),
            constraints: Vec::new(),
            sos1_sets: Vec::new(),
            indicator_constraints: Vec::new(),
        }
    }

    /// Add a variable from a [`MilpVariable`] descriptor, returning its index.
    pub fn add_variable(&mut self, var: MilpVariable) -> usize {
        let idx = self.variables.len();
        self.variables.push(var);
        idx
    }

    /// Add a constraint from a [`MilpConstraint`] descriptor, returning its index.
    pub fn add_constraint(&mut self, con: MilpConstraint) -> usize {
        let idx = self.constraints.len();
        self.constraints.push(con);
        idx
    }

    pub fn add_sos1_set(&mut self, sos: Sos1Set) {
        self.sos1_sets.push(sos);
    }

    pub fn add_indicator_constraint(&mut self, ind: IndicatorConstraint) {
        self.indicator_constraints.push(ind);
    }

    pub fn set_obj_coeff(&mut self, var_idx: usize, coeff: f64) {
        if var_idx < self.variables.len() {
            self.variables[var_idx].obj_coeff = coeff;
        }
    }

    pub fn num_vars(&self) -> usize {
        self.variables.len()
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn num_binary_vars(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.var_type == VarType::Binary)
            .count()
    }
}

// ── Pipeline stage tracking ────────────────────────────────────────

/// Identifies a stage within the compiler pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    Validation,
    Preprocessing,
    Reformulation,
    ComplementarityEncoding,
    Emission,
    CertificateGeneration,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation => f.write_str("Validation"),
            Self::Preprocessing => f.write_str("Preprocessing"),
            Self::Reformulation => f.write_str("Reformulation"),
            Self::ComplementarityEncoding => f.write_str("ComplementarityEncoding"),
            Self::Emission => f.write_str("Emission"),
            Self::CertificateGeneration => f.write_str("CertificateGeneration"),
        }
    }
}

// ── Pipeline result ────────────────────────────────────────────────

/// Result returned by [`CompilerPipeline::run`].
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub reformulated_problem: LpProblem,
    pub certificate: Option<CompilerCertificate>,
    pub iterations: Option<usize>,
    pub num_big_m_constraints: usize,
}

// ── Compiler pipeline ──────────────────────────────────────────────

/// Top-level pipeline that orchestrates validation → reformulation → emission.
pub struct CompilerPipeline {
    config: CompilerConfig,
}

impl CompilerPipeline {
    pub fn new(config: CompilerConfig) -> Self {
        Self { config }
    }

    pub fn run(&mut self, problem: &BilevelProblem) -> CompilerResult<PipelineResult> {
        // Validate
        if let Err(msg) = bicut_core::validate::quick_validate(problem) {
            return Err(CompilerError::Validation(msg));
        }
        if problem.num_lower_vars == 0 {
            return Err(CompilerError::Validation(
                "lower level has no variables".into(),
            ));
        }

        let reformulation = self.config.reformulation;

        // Transform using the selected pass
        let (milp, iterations) = match reformulation {
            ReformulationType::KKT => {
                let bigm_set = if self.config.complementarity_encoding.needs_big_m() {
                    let comp = BigMComputer::new(BigMConfig::default());
                    Some(comp.compute_all_bigms(problem))
                } else {
                    None
                };
                let kkt_config = crate::kkt_pass::KktConfig {
                    encoding: self.config.complementarity_encoding,
                    tolerance: self.config.tolerance,
                    ..Default::default()
                };
                let pass = crate::kkt_pass::KktPass::new(kkt_config);
                let r = pass.apply(problem)?;
                (r.milp, None)
            }
            ReformulationType::StrongDuality => {
                let config = crate::strong_duality_pass::StrongDualityConfig {
                    tolerance: self.config.tolerance,
                    ..Default::default()
                };
                let pass = crate::strong_duality_pass::StrongDualityPass::new(config);
                let r = pass.apply(problem)?;
                // Convert strong_duality_pass::MilpProblem into pipeline::MilpProblem
                let milp = convert_strong_duality_milp(r.milp);
                (milp, None)
            }
            ReformulationType::ValueFunction => {
                let config = crate::value_function_pass::ValueFunctionConfig {
                    tolerance: self.config.tolerance,
                    ..Default::default()
                };
                let pass = crate::value_function_pass::ValueFunctionPass::new(config);
                let r = pass.apply(problem)?;
                (r.milp, None)
            }
            ReformulationType::CCG => {
                let config = crate::ccg_pass::CcgConfig {
                    max_iterations: self.config.max_iterations,
                    convergence_tol: self.config.tolerance,
                    ..Default::default()
                };
                let pass = crate::ccg_pass::CcgPass::new(config);
                let r = pass.apply(problem)?;
                let num_iters = r.iterations.len();
                (r.milp, Some(num_iters))
            }
        };

        let reformulated = milp_to_lp(&milp);
        let num_big_m = milp.num_binary_vars();

        Ok(PipelineResult {
            reformulated_problem: reformulated,
            certificate: None,
            iterations,
            num_big_m_constraints: num_big_m,
        })
    }
}

/// Convert a `strong_duality_pass::MilpProblem` into a `pipeline::MilpProblem`.
fn convert_strong_duality_milp(sd: crate::strong_duality_pass::MilpProblem) -> MilpProblem {
    let n = sd.num_vars;
    let mut milp = MilpProblem::new(&format!("sd_{}", n));
    milp.sense = sd.direction;
    for i in 0..n {
        let var = MilpVariable {
            name: sd
                .var_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("v{}", i)),
            lower_bound: sd.var_lower.get(i).copied().unwrap_or(0.0),
            upper_bound: sd.var_upper.get(i).copied().unwrap_or(f64::INFINITY),
            obj_coeff: sd.obj_coeffs.get(i).copied().unwrap_or(0.0),
            var_type: match sd
                .var_types
                .get(i)
                .copied()
                .unwrap_or(crate::strong_duality_pass::VarType::Continuous)
            {
                crate::strong_duality_pass::VarType::Continuous => VarType::Continuous,
                crate::strong_duality_pass::VarType::Integer => VarType::Integer,
                crate::strong_duality_pass::VarType::Binary => VarType::Binary,
            },
        };
        milp.add_variable(var);
    }
    // Build constraints from sparse matrix
    let m = sd.num_constraints;
    for i in 0..m {
        let sense = sd
            .constraint_senses
            .get(i)
            .copied()
            .unwrap_or(ConstraintSense::Le);
        let rhs = sd.constraint_rhs.get(i).copied().unwrap_or(0.0);
        let name = sd
            .constraint_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("c{}", i));
        let mut con = MilpConstraint::new(&name, sense, rhs);
        for entry in &sd.constraint_matrix.entries {
            if entry.row == i {
                con.add_term(entry.col, entry.value);
            }
        }
        milp.add_constraint(con);
    }
    milp
}

/// Convert a MilpProblem into a standard LpProblem.
pub fn milp_to_lp(milp: &MilpProblem) -> LpProblem {
    let n = milp.variables.len();
    let m = milp.constraints.len();
    let mut lp = LpProblem::new(n, m);
    lp.direction = milp.sense;
    lp.c = milp.variables.iter().map(|v| v.obj_coeff).collect();
    lp.var_bounds = milp
        .variables
        .iter()
        .map(|v| bicut_types::VarBound {
            lower: v.lower_bound,
            upper: v.upper_bound,
        })
        .collect();
    let mut a = bicut_types::SparseMatrix::new(m, n);
    for (i, con) in milp.constraints.iter().enumerate() {
        for &(j, coeff) in &con.coeffs {
            a.add_entry(i, j, coeff);
        }
    }
    lp.a_matrix = a;
    lp.b_rhs = milp.constraints.iter().map(|c| c.rhs).collect();
    lp.senses = milp.constraints.iter().map(|c| c.sense).collect();
    lp.num_vars = n;
    lp.num_constraints = m;
    lp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_milp_construction() {
        let mut milp = MilpProblem::new("test");
        milp.add_variable(MilpVariable::continuous("x", 0.0, 10.0));
        milp.add_variable(MilpVariable::binary("z"));
        assert_eq!(milp.num_vars(), 2);
        assert_eq!(milp.num_binary_vars(), 1);
    }

    #[test]
    fn test_milp_to_lp() {
        let mut milp = MilpProblem::new("test");
        milp.add_variable(MilpVariable::continuous("x", 0.0, 10.0));
        milp.set_obj_coeff(0, 1.0);
        let mut con = MilpConstraint::new("c1", ConstraintSense::Le, 5.0);
        con.add_term(0, 2.0);
        milp.add_constraint(con);
        let lp = milp_to_lp(&milp);
        assert_eq!(lp.num_vars, 1);
        assert_eq!(lp.num_constraints, 1);
        assert_eq!(lp.c[0], 1.0);
    }

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(PipelineStage::Validation.to_string(), "Validation");
    }

    #[test]
    fn test_pipeline_creation() {
        let config = CompilerConfig::default();
        let _pipeline = CompilerPipeline::new(config);
    }

    #[test]
    fn test_var_type_default() {
        assert_eq!(VarType::default(), VarType::Continuous);
    }
}
