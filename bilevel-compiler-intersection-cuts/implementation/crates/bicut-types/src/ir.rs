use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::constraint::{ConstraintSense, LinearConstraint};
use crate::variable::{VariableId, VariableScope, VariableType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IrNodeId(pub usize);

impl fmt::Display for IrNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ir#{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrVariable {
    pub id: VariableId,
    pub name: String,
    pub var_type: VariableType,
    pub scope: VariableScope,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub annotations: HashMap<String, String>,
}

impl IrVariable {
    pub fn new(id: VariableId, name: &str, vt: VariableType, scope: VariableScope) -> Self {
        Self {
            id,
            name: name.to_string(),
            var_type: vt,
            scope,
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
            annotations: HashMap::new(),
        }
    }
    pub fn with_bounds(mut self, lb: f64, ub: f64) -> Self {
        self.lower_bound = lb;
        self.upper_bound = ub;
        self
    }
    pub fn annotate(mut self, key: &str, val: &str) -> Self {
        self.annotations.insert(key.into(), val.into());
        self
    }
    pub fn is_bounded(&self) -> bool {
        self.lower_bound.is_finite() && self.upper_bound.is_finite()
    }
    pub fn range(&self) -> f64 {
        if self.is_bounded() {
            self.upper_bound - self.lower_bound
        } else {
            f64::INFINITY
        }
    }
    pub fn is_binary(&self) -> bool {
        self.var_type == VariableType::Binary
    }
    pub fn is_integer(&self) -> bool {
        matches!(self.var_type, VariableType::Integer | VariableType::Binary)
    }
    pub fn is_leader(&self) -> bool {
        matches!(self.scope, VariableScope::Leader | VariableScope::Shared)
    }
    pub fn is_follower(&self) -> bool {
        matches!(self.scope, VariableScope::Follower | VariableScope::Shared)
    }
}

impl fmt::Display for IrVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{:?}({:?})[{},{}]",
            self.name, self.var_type, self.scope, self.lower_bound, self.upper_bound
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrConstraint {
    pub id: usize,
    pub name: String,
    pub coefficients: Vec<(VariableId, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub level: IrConstraintLevel,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrConstraintLevel {
    Upper,
    Lower,
    Coupling,
    KKT,
    StrongDuality,
    ValueFunction,
    Cut,
}

impl fmt::Display for IrConstraintLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upper => write!(f, "Upper"),
            Self::Lower => write!(f, "Lower"),
            Self::Coupling => write!(f, "Coupling"),
            Self::KKT => write!(f, "KKT"),
            Self::StrongDuality => write!(f, "SD"),
            Self::ValueFunction => write!(f, "VF"),
            Self::Cut => write!(f, "Cut"),
        }
    }
}

impl IrConstraint {
    pub fn new(
        id: usize,
        name: &str,
        coeffs: Vec<(VariableId, f64)>,
        sense: ConstraintSense,
        rhs: f64,
        level: IrConstraintLevel,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            coefficients: coeffs,
            sense,
            rhs,
            level,
            annotations: HashMap::new(),
        }
    }
    pub fn annotate(mut self, key: &str, val: &str) -> Self {
        self.annotations.insert(key.into(), val.into());
        self
    }
    pub fn num_nonzeros(&self) -> usize {
        self.coefficients.len()
    }
    pub fn density(&self, n_vars: usize) -> f64 {
        if n_vars == 0 {
            0.0
        } else {
            self.num_nonzeros() as f64 / n_vars as f64
        }
    }
    /// Convert this IR constraint to a [`LinearConstraint`].
    pub fn to_linear_constraint(&self) -> LinearConstraint {
        use crate::constraint::ConstraintId;
        let expr = crate::expression::LinearExpr::from_terms(0.0, self.coefficients.clone());
        LinearConstraint::new(
            ConstraintId::new(self.id),
            &self.name,
            expr,
            self.sense,
            self.rhs,
        )
    }
    pub fn max_abs_coeff(&self) -> f64 {
        self.coefficients
            .iter()
            .map(|(_, c)| c.abs())
            .fold(0.0f64, f64::max)
    }
    pub fn involves_variable(&self, vid: VariableId) -> bool {
        self.coefficients.iter().any(|(v, _)| *v == vid)
    }
}

impl fmt::Display for IrConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({} terms {} {:.4})",
            self.level,
            self.name,
            self.coefficients.len(),
            match self.sense {
                ConstraintSense::Le => "<=",
                ConstraintSense::Ge => ">=",
                ConstraintSense::Eq => "=",
            },
            self.rhs
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrObjectiveSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrObjective {
    pub sense: IrObjectiveSense,
    pub linear_terms: Vec<(VariableId, f64)>,
    pub quadratic_terms: Vec<(VariableId, VariableId, f64)>,
    pub constant: f64,
    pub name: String,
}

impl IrObjective {
    pub fn linear(sense: IrObjectiveSense, terms: Vec<(VariableId, f64)>) -> Self {
        Self {
            sense,
            linear_terms: terms,
            quadratic_terms: Vec::new(),
            constant: 0.0,
            name: "obj".into(),
        }
    }
    pub fn is_linear(&self) -> bool {
        self.quadratic_terms.is_empty()
    }
    pub fn num_terms(&self) -> usize {
        self.linear_terms.len() + self.quadratic_terms.len()
    }
    pub fn coefficient_of(&self, vid: VariableId) -> f64 {
        self.linear_terms
            .iter()
            .filter(|(v, _)| *v == vid)
            .map(|(_, c)| *c)
            .sum()
    }
}

impl fmt::Display for IrObjective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} terms",
            match self.sense {
                IrObjectiveSense::Minimize => "min",
                IrObjectiveSense::Maximize => "max",
            },
            self.num_terms()
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BilevelAnnotationType {
    LeaderVariable,
    FollowerVariable,
    DualVariable,
    SlackVariable,
    AuxiliaryVariable,
    ComplementarityConstraint,
    StrongDualityConstraint,
    ValueFunctionConstraint,
    BilevelCut,
    OriginalConstraint,
}

impl fmt::Display for BilevelAnnotationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeaderVariable => write!(f, "LeaderVar"),
            Self::FollowerVariable => write!(f, "FollowerVar"),
            Self::DualVariable => write!(f, "DualVar"),
            Self::SlackVariable => write!(f, "SlackVar"),
            Self::AuxiliaryVariable => write!(f, "AuxVar"),
            Self::ComplementarityConstraint => write!(f, "CompCstr"),
            Self::StrongDualityConstraint => write!(f, "SDCstr"),
            Self::ValueFunctionConstraint => write!(f, "VFCstr"),
            Self::BilevelCut => write!(f, "Cut"),
            Self::OriginalConstraint => write!(f, "OrigCstr"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilevelAnnotation {
    pub annotation_type: BilevelAnnotationType,
    pub source_variable: Option<VariableId>,
    pub source_constraint: Option<usize>,
    pub metadata: HashMap<String, String>,
}

impl BilevelAnnotation {
    pub fn new(at: BilevelAnnotationType) -> Self {
        Self {
            annotation_type: at,
            source_variable: None,
            source_constraint: None,
            metadata: HashMap::new(),
        }
    }
    pub fn with_source_var(mut self, vid: VariableId) -> Self {
        self.source_variable = Some(vid);
        self
    }
    pub fn with_source_constraint(mut self, cid: usize) -> Self {
        self.source_constraint = Some(cid);
        self
    }
    pub fn with_meta(mut self, key: &str, val: &str) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
    pub fn is_original(&self) -> bool {
        matches!(
            self.annotation_type,
            BilevelAnnotationType::OriginalConstraint
                | BilevelAnnotationType::LeaderVariable
                | BilevelAnnotationType::FollowerVariable
        )
    }
    pub fn is_reformulation_artifact(&self) -> bool {
        !self.is_original()
    }
}

impl fmt::Display for BilevelAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Annot({})", self.annotation_type)?;
        if let Some(v) = self.source_variable {
            write!(f, " src_var={}", v.0)?;
        }
        if let Some(c) = self.source_constraint {
            write!(f, " src_cstr={}", c)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrProgram {
    pub name: String,
    pub variables: Vec<IrVariable>,
    pub constraints: Vec<IrConstraint>,
    pub objective: IrObjective,
    pub variable_annotations: HashMap<VariableId, BilevelAnnotation>,
    pub constraint_annotations: HashMap<usize, BilevelAnnotation>,
    pub metadata: HashMap<String, String>,
}

impl IrProgram {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: IrObjective::linear(IrObjectiveSense::Minimize, vec![]),
            variable_annotations: HashMap::new(),
            constraint_annotations: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_variable(&mut self, var: IrVariable) -> VariableId {
        let id = var.id;
        self.variables.push(var);
        id
    }

    pub fn add_constraint(&mut self, cstr: IrConstraint) -> usize {
        let id = cstr.id;
        self.constraints.push(cstr);
        id
    }

    pub fn annotate_variable(&mut self, vid: VariableId, ann: BilevelAnnotation) {
        self.variable_annotations.insert(vid, ann);
    }

    pub fn annotate_constraint(&mut self, cid: usize, ann: BilevelAnnotation) {
        self.constraint_annotations.insert(cid, ann);
    }

    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
    pub fn num_integer_variables(&self) -> usize {
        self.variables.iter().filter(|v| v.is_integer()).count()
    }
    pub fn num_binary_variables(&self) -> usize {
        self.variables.iter().filter(|v| v.is_binary()).count()
    }
    pub fn num_continuous_variables(&self) -> usize {
        self.variables.iter().filter(|v| !v.is_integer()).count()
    }

    pub fn total_nonzeros(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| c.num_nonzeros())
            .sum::<usize>()
            + self.objective.num_terms()
    }

    pub fn density(&self) -> f64 {
        let n = self.num_variables();
        let m = self.num_constraints();
        if n == 0 || m == 0 {
            return 0.0;
        }
        self.total_nonzeros() as f64 / (n * m) as f64
    }

    pub fn variable_by_id(&self, vid: VariableId) -> Option<&IrVariable> {
        self.variables.iter().find(|v| v.id == vid)
    }

    pub fn constraints_involving(&self, vid: VariableId) -> Vec<&IrConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.involves_variable(vid))
            .collect()
    }

    pub fn constraints_at_level(&self, level: IrConstraintLevel) -> Vec<&IrConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.level == level)
            .collect()
    }

    pub fn leader_variables(&self) -> Vec<&IrVariable> {
        self.variables.iter().filter(|v| v.is_leader()).collect()
    }
    pub fn follower_variables(&self) -> Vec<&IrVariable> {
        self.variables.iter().filter(|v| v.is_follower()).collect()
    }

    pub fn reformulation_variables(&self) -> Vec<&IrVariable> {
        self.variables
            .iter()
            .filter(|v| {
                self.variable_annotations
                    .get(&v.id)
                    .map_or(false, |a| a.is_reformulation_artifact())
            })
            .collect()
    }

    pub fn original_variables(&self) -> Vec<&IrVariable> {
        self.variables
            .iter()
            .filter(|v| {
                self.variable_annotations
                    .get(&v.id)
                    .map_or(true, |a| a.is_original())
            })
            .collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "IR '{}': {} vars ({} int), {} constrs, {} nnz",
            self.name,
            self.num_variables(),
            self.num_integer_variables(),
            self.num_constraints(),
            self.total_nonzeros()
        )
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let var_ids: std::collections::HashSet<_> = self.variables.iter().map(|v| v.id).collect();
        for c in &self.constraints {
            for (vid, _) in &c.coefficients {
                if !var_ids.contains(vid) {
                    errors.push(format!(
                        "Constraint '{}' references unknown variable {}",
                        c.name, vid.0
                    ));
                }
            }
        }
        for (vid, _) in &self.objective.linear_terms {
            if !var_ids.contains(vid) {
                errors.push(format!("Objective references unknown variable {}", vid.0));
            }
        }
        for v in &self.variables {
            if v.lower_bound > v.upper_bound {
                errors.push(format!("Variable {} has lb > ub", v.name));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn clone_with_name(&self, name: &str) -> Self {
        let mut p = self.clone();
        p.name = name.to_string();
        p
    }

    pub fn next_variable_id(&self) -> VariableId {
        VariableId(self.variables.iter().map(|v| v.id.0).max().unwrap_or(0) + 1)
    }

    pub fn next_constraint_id(&self) -> usize {
        self.constraints.iter().map(|c| c.id).max().unwrap_or(0) + 1
    }
}

impl fmt::Display for IrProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IR Program: {}", self.name)?;
        writeln!(f, "  Objective: {}", self.objective)?;
        writeln!(
            f,
            "  Variables: {} ({} integer)",
            self.num_variables(),
            self.num_integer_variables()
        )?;
        writeln!(f, "  Constraints: {}", self.num_constraints())?;
        writeln!(
            f,
            "  Annotations: {} vars, {} cstrs",
            self.variable_annotations.len(),
            self.constraint_annotations.len()
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedIr {
    pub program: IrProgram,
    pub reformulation_type: Option<String>,
    pub source_problem: Option<String>,
    pub pass_history: Vec<String>,
}

impl TypedIr {
    pub fn new(program: IrProgram) -> Self {
        Self {
            program,
            reformulation_type: None,
            source_problem: None,
            pass_history: Vec::new(),
        }
    }
    pub fn with_reformulation(mut self, rt: &str) -> Self {
        self.reformulation_type = Some(rt.into());
        self
    }
    pub fn with_source(mut self, src: &str) -> Self {
        self.source_problem = Some(src.into());
        self
    }
    pub fn record_pass(&mut self, pass: &str) {
        self.pass_history.push(pass.to_string());
    }
    pub fn num_passes(&self) -> usize {
        self.pass_history.len()
    }
}

impl fmt::Display for TypedIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypedIR({}", self.program.name)?;
        if let Some(ref rt) = self.reformulation_type {
            write!(f, ", ref={}", rt)?;
        }
        write!(f, ", {} passes)", self.num_passes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ir() -> IrProgram {
        let mut ir = IrProgram::new("test");
        let v0 = IrVariable::new(
            VariableId(0),
            "x",
            VariableType::Continuous,
            VariableScope::Leader,
        )
        .with_bounds(0.0, 10.0);
        let v1 = IrVariable::new(
            VariableId(1),
            "y",
            VariableType::Integer,
            VariableScope::Follower,
        )
        .with_bounds(0.0, 5.0);
        ir.add_variable(v0);
        ir.add_variable(v1);
        ir.objective = IrObjective::linear(
            IrObjectiveSense::Minimize,
            vec![(VariableId(0), 1.0), (VariableId(1), -1.0)],
        );
        ir.add_constraint(IrConstraint::new(
            0,
            "c1",
            vec![(VariableId(0), 1.0), (VariableId(1), 2.0)],
            ConstraintSense::Le,
            10.0,
            IrConstraintLevel::Upper,
        ));
        ir
    }

    #[test]
    fn test_ir_creation() {
        let ir = make_ir();
        assert_eq!(ir.num_variables(), 2);
    }
    #[test]
    fn test_ir_integer() {
        let ir = make_ir();
        assert_eq!(ir.num_integer_variables(), 1);
    }
    #[test]
    fn test_ir_validate() {
        let ir = make_ir();
        assert!(ir.validate().is_ok());
    }
    #[test]
    fn test_ir_annotations() {
        let mut ir = make_ir();
        ir.annotate_variable(
            VariableId(0),
            BilevelAnnotation::new(BilevelAnnotationType::LeaderVariable),
        );
        assert!(ir.variable_annotations.contains_key(&VariableId(0)));
    }
    #[test]
    fn test_typed_ir() {
        let ir = make_ir();
        let mut tir = TypedIr::new(ir);
        tir.record_pass("kkt");
        assert_eq!(tir.num_passes(), 1);
    }
    #[test]
    fn test_constraint_level() {
        let ir = make_ir();
        assert_eq!(ir.constraints_at_level(IrConstraintLevel::Upper).len(), 1);
    }
    #[test]
    fn test_next_ids() {
        let ir = make_ir();
        assert_eq!(ir.next_variable_id(), VariableId(2));
        assert_eq!(ir.next_constraint_id(), 1);
    }
}
