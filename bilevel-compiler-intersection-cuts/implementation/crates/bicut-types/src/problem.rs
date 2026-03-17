use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::constraint::{ConstraintId, ConstraintSense, LinearConstraint};
use crate::expression::LinearExpr;
use crate::matrix::SparseMatrixCsr;
use crate::variable::{VariableId, VariableInfo, VariableScope, VariableSet, VariableType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemLevel {
    Leader,
    Follower,
}

impl fmt::Display for ProblemLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProblemLevel::Leader => write!(f, "Leader"),
            ProblemLevel::Follower => write!(f, "Follower"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

impl fmt::Display for ObjectiveSense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveSense::Minimize => write!(f, "min"),
            ObjectiveSense::Maximize => write!(f, "max"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveFunction {
    pub sense: ObjectiveSense,
    pub linear_coeffs: Vec<(VariableId, f64)>,
    pub quadratic_coeffs: Vec<(VariableId, VariableId, f64)>,
    pub constant: f64,
    pub name: String,
}

impl ObjectiveFunction {
    pub fn new_linear(sense: ObjectiveSense, coeffs: Vec<(VariableId, f64)>) -> Self {
        Self {
            sense,
            linear_coeffs: coeffs,
            quadratic_coeffs: Vec::new(),
            constant: 0.0,
            name: String::from("obj"),
        }
    }

    pub fn new_quadratic(
        sense: ObjectiveSense,
        linear: Vec<(VariableId, f64)>,
        quadratic: Vec<(VariableId, VariableId, f64)>,
    ) -> Self {
        Self {
            sense,
            linear_coeffs: linear,
            quadratic_coeffs: quadratic,
            constant: 0.0,
            name: String::from("obj"),
        }
    }

    pub fn is_linear(&self) -> bool {
        self.quadratic_coeffs.is_empty()
    }

    pub fn is_quadratic(&self) -> bool {
        !self.quadratic_coeffs.is_empty()
    }

    pub fn evaluate(&self, values: &HashMap<VariableId, f64>) -> f64 {
        let mut val = self.constant;
        for (vid, coeff) in &self.linear_coeffs {
            if let Some(&x) = values.get(vid) {
                val += coeff * x;
            }
        }
        for (vi, vj, coeff) in &self.quadratic_coeffs {
            let xi = values.get(vi).copied().unwrap_or(0.0);
            let xj = values.get(vj).copied().unwrap_or(0.0);
            val += coeff * xi * xj;
        }
        val
    }

    pub fn num_terms(&self) -> usize {
        self.linear_coeffs.len() + self.quadratic_coeffs.len()
    }

    pub fn variable_ids(&self) -> Vec<VariableId> {
        let mut ids: Vec<VariableId> = self.linear_coeffs.iter().map(|(v, _)| *v).collect();
        for (vi, vj, _) in &self.quadratic_coeffs {
            ids.push(*vi);
            ids.push(*vj);
        }
        ids.sort();
        ids.dedup();
        ids
    }

    pub fn to_linear_expr(&self) -> Option<LinearExpr> {
        if !self.is_linear() {
            return None;
        }
        let mut expr = LinearExpr::constant(self.constant);
        for (vid, coeff) in &self.linear_coeffs {
            expr.add_term(*vid, *coeff);
        }
        Some(expr)
    }

    pub fn negate(&self) -> Self {
        Self {
            sense: match self.sense {
                ObjectiveSense::Minimize => ObjectiveSense::Maximize,
                ObjectiveSense::Maximize => ObjectiveSense::Minimize,
            },
            linear_coeffs: self.linear_coeffs.iter().map(|(v, c)| (*v, -c)).collect(),
            quadratic_coeffs: self
                .quadratic_coeffs
                .iter()
                .map(|(vi, vj, c)| (*vi, *vj, -c))
                .collect(),
            constant: -self.constant,
            name: self.name.clone(),
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            sense: self.sense,
            linear_coeffs: self
                .linear_coeffs
                .iter()
                .map(|(v, c)| (*v, c * factor))
                .collect(),
            quadratic_coeffs: self
                .quadratic_coeffs
                .iter()
                .map(|(vi, vj, c)| (*vi, *vj, c * factor))
                .collect(),
            constant: self.constant * factor,
            name: self.name.clone(),
        }
    }

    pub fn coefficient_of(&self, var: VariableId) -> f64 {
        self.linear_coeffs
            .iter()
            .filter(|(v, _)| *v == var)
            .map(|(_, c)| *c)
            .sum()
    }

    pub fn max_abs_coefficient(&self) -> f64 {
        let lin_max = self
            .linear_coeffs
            .iter()
            .map(|(_, c)| c.abs())
            .fold(0.0f64, f64::max);
        let quad_max = self
            .quadratic_coeffs
            .iter()
            .map(|(_, _, c)| c.abs())
            .fold(0.0f64, f64::max);
        lin_max.max(quad_max)
    }
}

impl fmt::Display for ObjectiveFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ", self.sense)?;
        let mut first = true;
        if self.constant.abs() > 1e-15 {
            write!(f, "{:.4}", self.constant)?;
            first = false;
        }
        for (vid, coeff) in &self.linear_coeffs {
            if !first {
                if *coeff >= 0.0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
                write!(f, "{:.4}*x{}", coeff.abs(), vid.0)?;
            } else {
                write!(f, "{:.4}*x{}", coeff, vid.0)?;
                first = false;
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDimensions {
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_leader_continuous: usize,
    pub num_leader_integer: usize,
    pub num_leader_binary: usize,
    pub num_follower_continuous: usize,
    pub num_follower_integer: usize,
    pub num_follower_binary: usize,
    pub num_upper_constraints: usize,
    pub num_lower_constraints: usize,
    pub num_coupling_constraints: usize,
    pub total_vars: usize,
    pub total_constraints: usize,
}

impl ProblemDimensions {
    pub fn compute(problem: &BilevelProblem) -> Self {
        let mut dims = ProblemDimensions {
            num_leader_vars: 0,
            num_follower_vars: 0,
            num_leader_continuous: 0,
            num_leader_integer: 0,
            num_leader_binary: 0,
            num_follower_continuous: 0,
            num_follower_integer: 0,
            num_follower_binary: 0,
            num_upper_constraints: problem.upper_constraints.len(),
            num_lower_constraints: problem.lower_constraints.len(),
            num_coupling_constraints: problem.coupling_constraints.len(),
            total_vars: 0,
            total_constraints: 0,
        };
        for var in problem.variables.iter() {
            match var.scope {
                VariableScope::Leader => {
                    dims.num_leader_vars += 1;
                    match var.var_type {
                        VariableType::Continuous => dims.num_leader_continuous += 1,
                        VariableType::Integer => dims.num_leader_integer += 1,
                        VariableType::Binary => dims.num_leader_binary += 1,
                        VariableType::SemiContinuous => dims.num_leader_continuous += 1,
                    }
                }
                VariableScope::Follower => {
                    dims.num_follower_vars += 1;
                    match var.var_type {
                        VariableType::Continuous => dims.num_follower_continuous += 1,
                        VariableType::Integer => dims.num_follower_integer += 1,
                        VariableType::Binary => dims.num_follower_binary += 1,
                        VariableType::SemiContinuous => dims.num_follower_continuous += 1,
                    }
                }
                VariableScope::Shared => {
                    dims.num_leader_vars += 1;
                    dims.num_follower_vars += 1;
                }
            }
        }
        dims.total_vars = problem.variables.len();
        dims.total_constraints =
            dims.num_upper_constraints + dims.num_lower_constraints + dims.num_coupling_constraints;
        dims
    }

    pub fn has_integer_leader(&self) -> bool {
        self.num_leader_integer > 0 || self.num_leader_binary > 0
    }

    pub fn has_integer_follower(&self) -> bool {
        self.num_follower_integer > 0 || self.num_follower_binary > 0
    }

    pub fn is_continuous(&self) -> bool {
        !self.has_integer_leader() && !self.has_integer_follower()
    }

    pub fn density(&self) -> f64 {
        if self.total_vars == 0 || self.total_constraints == 0 {
            return 0.0;
        }
        let max_nz = self.total_vars * self.total_constraints;
        let est_nz = (self.total_vars + self.total_constraints) * 3;
        (est_nz as f64) / (max_nz as f64)
    }
}

impl fmt::Display for ProblemDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dims(vars:{}={}L+{}F, cstr:{}={}U+{}L+{}C)",
            self.total_vars,
            self.num_leader_vars,
            self.num_follower_vars,
            self.total_constraints,
            self.num_upper_constraints,
            self.num_lower_constraints,
            self.num_coupling_constraints
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilevelProblem {
    pub name: String,
    pub leader_objective: ObjectiveFunction,
    pub follower_objective: ObjectiveFunction,
    pub variables: VariableSet,
    pub upper_constraints: Vec<LinearConstraint>,
    pub lower_constraints: Vec<LinearConstraint>,
    pub coupling_constraints: Vec<LinearConstraint>,
    pub metadata: HashMap<String, String>,
}

impl BilevelProblem {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            leader_objective: ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![]),
            follower_objective: ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![]),
            variables: VariableSet::new(),
            upper_constraints: Vec::new(),
            lower_constraints: Vec::new(),
            coupling_constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn dimensions(&self) -> ProblemDimensions {
        ProblemDimensions::compute(self)
    }

    pub fn leader_variable_ids(&self) -> Vec<VariableId> {
        self.variables
            .iter()
            .filter(|v| matches!(v.scope, VariableScope::Leader | VariableScope::Shared))
            .map(|v| v.id)
            .collect()
    }

    pub fn follower_variable_ids(&self) -> Vec<VariableId> {
        self.variables
            .iter()
            .filter(|v| matches!(v.scope, VariableScope::Follower | VariableScope::Shared))
            .map(|v| v.id)
            .collect()
    }

    pub fn is_linear(&self) -> bool {
        self.leader_objective.is_linear() && self.follower_objective.is_linear()
    }

    pub fn has_integer_lower_level(&self) -> bool {
        self.variables.iter().any(|v| {
            matches!(v.scope, VariableScope::Follower)
                && !matches!(v.var_type, VariableType::Continuous)
        })
    }

    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    pub fn num_constraints(&self) -> usize {
        self.upper_constraints.len()
            + self.lower_constraints.len()
            + self.coupling_constraints.len()
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.variables.is_empty() {
            errors.push("Problem has no variables".to_string());
        }
        let all_ids: std::collections::HashSet<_> = self.variables.iter().map(|v| v.id).collect();
        for (vid, _) in &self.leader_objective.linear_coeffs {
            if !all_ids.contains(vid) {
                errors.push(format!(
                    "Leader objective references unknown variable {:?}",
                    vid
                ));
            }
        }
        for (vid, _) in &self.follower_objective.linear_coeffs {
            if !all_ids.contains(vid) {
                errors.push(format!(
                    "Follower objective references unknown variable {:?}",
                    vid
                ));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn add_variable(&mut self, info: VariableInfo) -> VariableId {
        self.variables.add(info).expect("failed to add variable")
    }

    pub fn add_upper_constraint(&mut self, c: LinearConstraint) {
        self.upper_constraints.push(c);
    }
    pub fn add_lower_constraint(&mut self, c: LinearConstraint) {
        self.lower_constraints.push(c);
    }
    pub fn add_coupling_constraint(&mut self, c: LinearConstraint) {
        self.coupling_constraints.push(c);
    }

    pub fn constraint_matrix_lower(&self) -> SparseMatrixCsr {
        let n = self.variables.len();
        let m = self.lower_constraints.len();
        let mut row_offsets = vec![0usize; m + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (i, c) in self.lower_constraints.iter().enumerate() {
            for (vid, coeff) in &c.expr.terms {
                col_indices.push(vid.0);
                values.push(*coeff);
            }
            row_offsets[i + 1] = col_indices.len();
        }
        SparseMatrixCsr {
            nrows: m,
            ncols: n,
            row_offsets,
            col_indices,
            values,
        }
    }

    pub fn rhs_lower(&self) -> Vec<f64> {
        self.lower_constraints.iter().map(|c| c.rhs).collect()
    }

    pub fn follower_cost_vector(&self) -> Vec<f64> {
        let n = self.variables.len();
        let mut c = vec![0.0; n];
        for (vid, coeff) in &self.follower_objective.linear_coeffs {
            if vid.0 < n {
                c[vid.0] = *coeff;
            }
        }
        c
    }

    pub fn leader_cost_vector(&self) -> Vec<f64> {
        let n = self.variables.len();
        let mut c = vec![0.0; n];
        for (vid, coeff) in &self.leader_objective.linear_coeffs {
            if vid.0 < n {
                c[vid.0] = *coeff;
            }
        }
        c
    }

    pub fn summary(&self) -> String {
        let dims = self.dimensions();
        format!(
            "BilevelProblem '{}': {} vars, {} constraints",
            self.name, dims.total_vars, dims.total_constraints
        )
    }

    pub fn clone_with_name(&self, name: &str) -> Self {
        let mut p = self.clone();
        p.name = name.to_string();
        p
    }

    pub fn all_constraint_count(&self) -> usize {
        self.upper_constraints.len()
            + self.lower_constraints.len()
            + self.coupling_constraints.len()
    }

    pub fn variable_by_name(&self, name: &str) -> Option<&VariableInfo> {
        self.variables.iter().find(|v| v.name == name)
    }
}

impl fmt::Display for BilevelProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bilevel Problem: {}", self.name)?;
        writeln!(f, "  Leader: {}", self.leader_objective)?;
        writeln!(f, "  Follower: {}", self.follower_objective)?;
        writeln!(f, "  Variables: {}", self.variables.len())?;
        writeln!(
            f,
            "  Upper: {} Lower: {} Coupling: {}",
            self.upper_constraints.len(),
            self.lower_constraints.len(),
            self.coupling_constraints.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_problem() -> BilevelProblem {
        let mut p = BilevelProblem::new("test");
        let x = p.add_variable(VariableInfo {
            id: VariableId(0),
            name: "x".into(),
            var_type: VariableType::Continuous,
            scope: VariableScope::Leader,
            lower_bound: 0.0,
            upper_bound: 10.0,
            obj_coeff: 0.0,
            fixed_value: None,
        });
        let y = p.add_variable(VariableInfo {
            id: VariableId(1),
            name: "y".into(),
            var_type: VariableType::Continuous,
            scope: VariableScope::Follower,
            lower_bound: 0.0,
            upper_bound: 10.0,
            obj_coeff: 0.0,
            fixed_value: None,
        });
        p.leader_objective =
            ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![(x, 1.0), (y, -1.0)]);
        p.follower_objective =
            ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![(y, 1.0)]);
        p.add_lower_constraint(LinearConstraint::new(
            ConstraintId::new(0),
            "lc1",
            LinearExpr::from_terms(0.0, vec![(y, 1.0)]),
            ConstraintSense::Le,
            5.0,
        ));
        p
    }

    #[test]
    fn test_creation() {
        let p = make_problem();
        assert_eq!(p.num_variables(), 2);
    }

    #[test]
    fn test_dimensions() {
        let p = make_problem();
        let d = p.dimensions();
        assert_eq!(d.num_leader_vars, 1);
        assert_eq!(d.num_follower_vars, 1);
    }

    #[test]
    fn test_validate() {
        assert!(make_problem().validate().is_ok());
    }

    #[test]
    fn test_obj_eval() {
        let obj =
            ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![(VariableId(0), 2.0)]);
        let mut v = HashMap::new();
        v.insert(VariableId(0), 3.0);
        assert!((obj.evaluate(&v) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_negate() {
        let obj =
            ObjectiveFunction::new_linear(ObjectiveSense::Minimize, vec![(VariableId(0), 2.0)]);
        let n = obj.negate();
        assert_eq!(n.sense, ObjectiveSense::Maximize);
    }

    #[test]
    fn test_cost_vector() {
        let p = make_problem();
        let c = p.follower_cost_vector();
        assert!((c[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let p = make_problem();
        let s = format!("{}", p);
        assert!(s.contains("test"));
    }
}
