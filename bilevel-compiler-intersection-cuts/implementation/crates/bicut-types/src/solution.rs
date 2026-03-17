use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::variable::VariableId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolutionStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    InfeasibleOrUnbounded,
    TimeLimit,
    IterationLimit,
    NodeLimit,
    Unknown,
    Error,
}

impl fmt::Display for SolutionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Optimal => write!(f, "Optimal"),
            Self::Feasible => write!(f, "Feasible"),
            Self::Infeasible => write!(f, "Infeasible"),
            Self::Unbounded => write!(f, "Unbounded"),
            Self::InfeasibleOrUnbounded => write!(f, "InfeasibleOrUnbounded"),
            Self::TimeLimit => write!(f, "TimeLimit"),
            Self::IterationLimit => write!(f, "IterationLimit"),
            Self::NodeLimit => write!(f, "NodeLimit"),
            Self::Unknown => write!(f, "Unknown"),
            Self::Error => write!(f, "Error"),
        }
    }
}

impl SolutionStatus {
    pub fn is_optimal(&self) -> bool {
        matches!(self, Self::Optimal)
    }
    pub fn is_feasible(&self) -> bool {
        matches!(self, Self::Optimal | Self::Feasible)
    }
    pub fn is_infeasible(&self) -> bool {
        matches!(self, Self::Infeasible)
    }
    pub fn is_unbounded(&self) -> bool {
        matches!(self, Self::Unbounded)
    }
    pub fn is_terminated(&self) -> bool {
        matches!(
            self,
            Self::TimeLimit | Self::IterationLimit | Self::NodeLimit
        )
    }
    pub fn has_solution(&self) -> bool {
        self.is_optimal() || self.is_feasible()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveValue {
    pub primal: f64,
    pub dual_bound: f64,
    pub gap: f64,
}

impl ObjectiveValue {
    pub fn new(primal: f64, dual: f64) -> Self {
        let gap = if primal.abs() < 1e-10 {
            if dual.abs() < 1e-10 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            ((primal - dual) / primal.abs()).abs()
        };
        Self {
            primal,
            dual_bound: dual,
            gap,
        }
    }
    pub fn optimal(value: f64) -> Self {
        Self {
            primal: value,
            dual_bound: value,
            gap: 0.0,
        }
    }
    pub fn is_proven_optimal(&self, tol: f64) -> bool {
        self.gap <= tol
    }
}

impl fmt::Display for ObjectiveValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "obj={:.6} (dual={:.6}, gap={:.4}%)",
            self.primal,
            self.dual_bound,
            self.gap * 100.0
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalSolution {
    pub values: HashMap<VariableId, f64>,
    pub objective: f64,
}

impl PrimalSolution {
    pub fn new(values: HashMap<VariableId, f64>, obj: f64) -> Self {
        Self {
            values,
            objective: obj,
        }
    }
    pub fn empty() -> Self {
        Self {
            values: HashMap::new(),
            objective: f64::INFINITY,
        }
    }
    pub fn get(&self, var: VariableId) -> f64 {
        self.values.get(&var).copied().unwrap_or(0.0)
    }
    pub fn set(&mut self, var: VariableId, val: f64) {
        self.values.insert(var, val);
    }
    pub fn num_nonzeros(&self) -> usize {
        self.values.values().filter(|v| v.abs() > 1e-10).count()
    }
    pub fn to_vec(&self, n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        for (vid, val) in &self.values {
            if vid.0 < n {
                v[vid.0] = *val;
            }
        }
        v
    }
    pub fn from_vec(vals: &[f64], obj: f64) -> Self {
        let values = vals
            .iter()
            .enumerate()
            .map(|(i, &v)| (VariableId(i), v))
            .collect();
        Self {
            values,
            objective: obj,
        }
    }
    pub fn distance_to(&self, other: &PrimalSolution) -> f64 {
        let mut sum_sq = 0.0;
        let all_keys: std::collections::HashSet<_> =
            self.values.keys().chain(other.values.keys()).collect();
        for &k in &all_keys {
            let a = self.values.get(k).copied().unwrap_or(0.0);
            let b = other.values.get(k).copied().unwrap_or(0.0);
            sum_sq += (a - b) * (a - b);
        }
        sum_sq.sqrt()
    }
    pub fn is_integer_feasible(&self, integer_vars: &[VariableId], tol: f64) -> bool {
        integer_vars.iter().all(|vid| {
            let val = self.get(*vid);
            (val - val.round()).abs() <= tol
        })
    }
    pub fn fractionality(&self, integer_vars: &[VariableId]) -> f64 {
        if integer_vars.is_empty() {
            return 0.0;
        }
        integer_vars
            .iter()
            .map(|vid| {
                let v = self.get(*vid);
                let frac = v - v.floor();
                frac.min(1.0 - frac)
            })
            .sum::<f64>()
            / integer_vars.len() as f64
    }
}

impl fmt::Display for PrimalSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Primal(obj={:.6}, {} vars, {} nnz)",
            self.objective,
            self.values.len(),
            self.num_nonzeros()
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualSolution {
    pub constraint_duals: Vec<f64>,
    pub reduced_costs: HashMap<VariableId, f64>,
    pub objective: f64,
}

impl DualSolution {
    pub fn new(duals: Vec<f64>, obj: f64) -> Self {
        Self {
            constraint_duals: duals,
            reduced_costs: HashMap::new(),
            objective: obj,
        }
    }
    pub fn empty() -> Self {
        Self {
            constraint_duals: Vec::new(),
            reduced_costs: HashMap::new(),
            objective: 0.0,
        }
    }
    pub fn get_dual(&self, idx: usize) -> f64 {
        self.constraint_duals.get(idx).copied().unwrap_or(0.0)
    }
    pub fn get_reduced_cost(&self, var: VariableId) -> f64 {
        self.reduced_costs.get(&var).copied().unwrap_or(0.0)
    }
    pub fn set_reduced_cost(&mut self, var: VariableId, rc: f64) {
        self.reduced_costs.insert(var, rc);
    }
    pub fn is_dual_feasible(&self, tol: f64) -> bool {
        self.constraint_duals.iter().all(|&d| d >= -tol)
    }
    pub fn num_active_constraints(&self, tol: f64) -> usize {
        self.constraint_duals
            .iter()
            .filter(|&&d| d.abs() > tol)
            .count()
    }
}

impl fmt::Display for DualSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dual(obj={:.6}, {} duals)",
            self.objective,
            self.constraint_duals.len()
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilevelSolution {
    pub status: SolutionStatus,
    pub leader_solution: Option<PrimalSolution>,
    pub follower_solution: Option<PrimalSolution>,
    pub dual_solution: Option<DualSolution>,
    pub objective_value: Option<ObjectiveValue>,
    pub solve_time_secs: f64,
    pub nodes_explored: u64,
    pub iterations: u64,
    pub cuts_generated: u64,
    pub metadata: HashMap<String, String>,
}

impl BilevelSolution {
    pub fn new(status: SolutionStatus) -> Self {
        Self {
            status,
            leader_solution: None,
            follower_solution: None,
            dual_solution: None,
            objective_value: None,
            solve_time_secs: 0.0,
            nodes_explored: 0,
            iterations: 0,
            cuts_generated: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn optimal(leader: PrimalSolution, follower: PrimalSolution, obj: f64) -> Self {
        Self {
            status: SolutionStatus::Optimal,
            leader_solution: Some(leader),
            follower_solution: Some(follower),
            dual_solution: None,
            objective_value: Some(ObjectiveValue::optimal(obj)),
            solve_time_secs: 0.0,
            nodes_explored: 0,
            iterations: 0,
            cuts_generated: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn infeasible() -> Self {
        Self::new(SolutionStatus::Infeasible)
    }
    pub fn has_solution(&self) -> bool {
        self.status.has_solution()
    }
    pub fn primal_bound(&self) -> f64 {
        self.objective_value
            .as_ref()
            .map(|o| o.primal)
            .unwrap_or(f64::INFINITY)
    }
    pub fn dual_bound(&self) -> f64 {
        self.objective_value
            .as_ref()
            .map(|o| o.dual_bound)
            .unwrap_or(f64::NEG_INFINITY)
    }
    pub fn gap(&self) -> f64 {
        self.objective_value
            .as_ref()
            .map(|o| o.gap)
            .unwrap_or(f64::INFINITY)
    }
}

impl fmt::Display for BilevelSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BilevelSol({}", self.status)?;
        if let Some(ref obj) = self.objective_value {
            write!(f, ", {}", obj)?;
        }
        write!(
            f,
            ", {:.2}s, {} nodes)",
            self.solve_time_secs, self.nodes_explored
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionPool {
    solutions: Vec<BilevelSolution>,
    max_size: usize,
}

impl SolutionPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            solutions: Vec::new(),
            max_size,
        }
    }
    pub fn add(&mut self, sol: BilevelSolution) {
        if self.solutions.len() >= self.max_size {
            if let Some(worst_idx) = self
                .solutions
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.primal_bound()
                        .partial_cmp(&b.primal_bound())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if sol.primal_bound() < self.solutions[worst_idx].primal_bound() {
                    self.solutions[worst_idx] = sol;
                }
            }
        } else {
            self.solutions.push(sol);
        }
    }
    pub fn best(&self) -> Option<&BilevelSolution> {
        self.solutions.iter().min_by(|a, b| {
            a.primal_bound()
                .partial_cmp(&b.primal_bound())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
    pub fn len(&self) -> usize {
        self.solutions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }
    pub fn best_objective(&self) -> f64 {
        self.best()
            .map(|s| s.primal_bound())
            .unwrap_or(f64::INFINITY)
    }
    pub fn iter(&self) -> impl Iterator<Item = &BilevelSolution> {
        self.solutions.iter()
    }
    pub fn clear(&mut self) {
        self.solutions.clear();
    }
    pub fn diversity(&self) -> f64 {
        if self.solutions.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..self.solutions.len() {
            for j in (i + 1)..self.solutions.len() {
                let diff =
                    (self.solutions[i].primal_bound() - self.solutions[j].primal_bound()).abs();
                total += diff;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }
}

impl fmt::Display for SolutionPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool({} sols, best={:.6})",
            self.len(),
            self.best_objective()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status() {
        assert!(SolutionStatus::Optimal.is_optimal());
        assert!(SolutionStatus::Feasible.has_solution());
        assert!(!SolutionStatus::Infeasible.has_solution());
    }

    #[test]
    fn test_primal() {
        let mut sol = PrimalSolution::empty();
        sol.set(VariableId(0), 1.5);
        assert!((sol.get(VariableId(0)) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_integer_feasible() {
        let sol = PrimalSolution::from_vec(&[1.0, 2.0, 3.0], 0.0);
        assert!(sol.is_integer_feasible(&[VariableId(0), VariableId(1)], 1e-6));
    }

    #[test]
    fn test_dual() {
        let dual = DualSolution::new(vec![1.0, 0.0, 2.0], 5.0);
        assert!((dual.get_dual(0) - 1.0).abs() < 1e-10);
        assert_eq!(dual.num_active_constraints(1e-6), 2);
    }

    #[test]
    fn test_bilevel_sol() {
        let sol = BilevelSolution::infeasible();
        assert!(!sol.has_solution());
    }

    #[test]
    fn test_pool() {
        let mut pool = SolutionPool::new(3);
        pool.add(BilevelSolution::new(SolutionStatus::Optimal));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_objective_value() {
        let ov = ObjectiveValue::new(100.0, 90.0);
        assert!((ov.gap - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let s1 = PrimalSolution::from_vec(&[1.0, 0.0], 0.0);
        let s2 = PrimalSolution::from_vec(&[0.0, 0.0], 0.0);
        assert!((s1.distance_to(&s2) - 1.0).abs() < 1e-10);
    }
}
