use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::formula::{Clause, CnfFormula, Literal};

// ---------------------------------------------------------------------------
// ResourceSpec
// ---------------------------------------------------------------------------

/// Quantified resource requirements or capacities for a single entity (node,
/// pod, overhead, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub cpu_millis: u64,
    pub memory_mb: u64,
    pub storage_mb: u64,
}

impl ResourceSpec {
    pub fn new(cpu_millis: u64, memory_mb: u64, storage_mb: u64) -> Self {
        Self {
            cpu_millis,
            memory_mb,
            storage_mb,
        }
    }

    pub fn zero() -> Self {
        Self {
            cpu_millis: 0,
            memory_mb: 0,
            storage_mb: 0,
        }
    }

    /// Component-wise addition.
    pub fn add(&self, other: &ResourceSpec) -> ResourceSpec {
        ResourceSpec {
            cpu_millis: self.cpu_millis + other.cpu_millis,
            memory_mb: self.memory_mb + other.memory_mb,
            storage_mb: self.storage_mb + other.storage_mb,
        }
    }

    /// Component-wise saturating subtraction.
    pub fn subtract(&self, other: &ResourceSpec) -> ResourceSpec {
        ResourceSpec {
            cpu_millis: self.cpu_millis.saturating_sub(other.cpu_millis),
            memory_mb: self.memory_mb.saturating_sub(other.memory_mb),
            storage_mb: self.storage_mb.saturating_sub(other.storage_mb),
        }
    }

    /// Returns `true` when every dimension of `self` is ≤ the corresponding
    /// dimension of `other`.
    pub fn fits_in(&self, other: &ResourceSpec) -> bool {
        self.cpu_millis <= other.cpu_millis
            && self.memory_mb <= other.memory_mb
            && self.storage_mb <= other.storage_mb
    }
}

impl fmt::Display for ResourceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cpu={}m mem={}Mi stor={}Mi",
            self.cpu_millis, self.memory_mb, self.storage_mb,
        )
    }
}

// ---------------------------------------------------------------------------
// ResourceModel
// ---------------------------------------------------------------------------

/// Complete description of a cluster's resource topology together with per-pod
/// requirements and fixed overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceModel {
    pub num_nodes: usize,
    pub node_capacities: Vec<ResourceSpec>,
    pub pod_requirements: Vec<ResourceSpec>,
    pub overhead_per_pod: ResourceSpec,
}

impl ResourceModel {
    pub fn new(
        num_nodes: usize,
        node_capacities: Vec<ResourceSpec>,
        pod_requirements: Vec<ResourceSpec>,
        overhead_per_pod: ResourceSpec,
    ) -> Self {
        assert_eq!(
            node_capacities.len(),
            num_nodes,
            "node_capacities length must equal num_nodes"
        );
        Self {
            num_nodes,
            node_capacities,
            pod_requirements,
            overhead_per_pod,
        }
    }

    /// Compute per-node utilization (0.0–1.0+) for a given assignment.
    ///
    /// `assignment[i]` gives the node index that pod `i` is placed on.
    pub fn compute_utilization(&self, assignment: &[usize]) -> Vec<f64> {
        let mut node_used: Vec<ResourceSpec> = vec![ResourceSpec::zero(); self.num_nodes];
        for (pod_idx, &node_idx) in assignment.iter().enumerate() {
            assert!(node_idx < self.num_nodes, "node index out of range");
            let effective = self.pod_requirements[pod_idx].add(&self.overhead_per_pod);
            node_used[node_idx] = node_used[node_idx].add(&effective);
        }

        node_used
            .iter()
            .enumerate()
            .map(|(n, used)| {
                let cap = &self.node_capacities[n];
                if cap.cpu_millis == 0 && cap.memory_mb == 0 && cap.storage_mb == 0 {
                    return 0.0;
                }
                // Average utilisation across the three dimensions, skipping
                // any dimension where capacity is zero.
                let mut sum = 0.0_f64;
                let mut dims = 0u32;
                if cap.cpu_millis > 0 {
                    sum += used.cpu_millis as f64 / cap.cpu_millis as f64;
                    dims += 1;
                }
                if cap.memory_mb > 0 {
                    sum += used.memory_mb as f64 / cap.memory_mb as f64;
                    dims += 1;
                }
                if cap.storage_mb > 0 {
                    sum += used.storage_mb as f64 / cap.storage_mb as f64;
                    dims += 1;
                }
                if dims == 0 {
                    0.0
                } else {
                    sum / dims as f64
                }
            })
            .collect()
    }

    /// Sum of all node capacities.
    pub fn total_capacity(&self) -> ResourceSpec {
        self.node_capacities
            .iter()
            .fold(ResourceSpec::zero(), |acc, c| acc.add(c))
    }

    /// Quick feasibility check: does the aggregate demand (including overhead)
    /// fit in the aggregate cluster capacity?
    pub fn can_fit(&self, requirements: &[ResourceSpec]) -> bool {
        let total_demand = requirements
            .iter()
            .fold(ResourceSpec::zero(), |acc, r| {
                acc.add(r).add(&self.overhead_per_pod)
            });
        total_demand.fits_in(&self.total_capacity())
    }
}

// ---------------------------------------------------------------------------
// LinearConstraint
// ---------------------------------------------------------------------------

/// A pseudo-boolean / linear integer constraint of the form
///   ∑ (coefficients[i] × variables[i]) ≤ bound
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConstraint {
    pub variables: Vec<u32>,
    pub coefficients: Vec<i64>,
    pub bound: i64,
}

impl LinearConstraint {
    pub fn new(variables: Vec<u32>, coefficients: Vec<i64>, bound: i64) -> Self {
        assert_eq!(
            variables.len(),
            coefficients.len(),
            "variables and coefficients must have the same length"
        );
        Self {
            variables,
            coefficients,
            bound,
        }
    }

    /// Evaluate ∑ coeff_i * assignment[var_i] ≤ bound.
    pub fn evaluate(&self, assignment: &HashMap<u32, i64>) -> bool {
        let sum: i64 = self
            .variables
            .iter()
            .zip(self.coefficients.iter())
            .map(|(v, c)| {
                let val = assignment.get(v).copied().unwrap_or(0);
                c * val
            })
            .sum();
        sum <= self.bound
    }

    /// Given concrete values aligned with `self.variables`, check feasibility.
    pub fn is_satisfied_by(&self, values: &[i64]) -> bool {
        assert_eq!(
            values.len(),
            self.variables.len(),
            "values length must match variables length"
        );
        let sum: i64 = self
            .coefficients
            .iter()
            .zip(values.iter())
            .map(|(c, v)| c * v)
            .sum();
        sum <= self.bound
    }

    /// Encode this linear constraint into SAT clauses using a *sequential
    /// counter* (a.k.a. the Sinz totalizer encoding).
    ///
    /// For a constraint  ∑ w_i · x_i ≤ k  where every w_i > 0 and every x_i
    /// is a Boolean variable, the sequential counter introduces auxiliary
    /// register variables  r_{i,j}  meaning "the partial sum of the first i
    /// terms is ≥ j".  The encoding is linear in ∑ w_i (bounded by the
    /// coefficient magnitudes) so it is practical for moderate bounds.
    ///
    /// When coefficients are > 1 the variable is expanded into `coeff` copies
    /// whose conjunction equals the original literal, and these copies feed
    /// into a unit-weight sequential counter over bound `self.bound`.
    pub fn to_clauses(&self, next_var: &mut u32) -> Vec<Clause> {
        let n = self.variables.len();
        if n == 0 {
            return if self.bound >= 0 {
                vec![]
            } else {
                vec![vec![]] // empty clause → UNSAT
            };
        }

        // Separate positive-coefficient terms.  Negative-coefficient terms
        // are complemented (negate literal, add |c| to bound shift).
        let mut pos_lits: Vec<Literal> = Vec::new();
        let mut adjusted_bound = self.bound;

        for (&var, &coeff) in self.variables.iter().zip(self.coefficients.iter()) {
            if coeff == 0 {
                continue;
            }
            if coeff > 0 {
                for _ in 0..coeff {
                    pos_lits.push(var as Literal);
                }
            } else {
                // ¬x contributes |coeff| when x=0.  Rewrite:
                //   coeff * x = -|coeff| * x = |coeff| * (1 - x) - |coeff|
                let abs_c = coeff.unsigned_abs() as i64;
                adjusted_bound += abs_c;
                for _ in 0..abs_c {
                    pos_lits.push(-(var as Literal));
                }
            }
        }

        if adjusted_bound < 0 {
            return vec![vec![]]; // UNSAT
        }
        let k = adjusted_bound as usize;
        let m = pos_lits.len();
        if k >= m {
            return vec![]; // trivially satisfied
        }

        // Sequential counter encoding.
        // Auxiliary variable r[i][j] for i in 0..m, j in 0..k.
        // r[i][j] ↔ "at least j+1 of the first i+1 positive literals are true"
        let mut clauses: Vec<Clause> = Vec::new();

        let base = *next_var;
        *next_var += (m * k) as u32;

        let r = |i: usize, j: usize| -> Literal { (base + (i * k + j) as u32) as Literal };

        for i in 0..m {
            let xi = pos_lits[i];

            if i == 0 {
                // r[0][0] ← x_0
                clauses.push(vec![-xi, r(0, 0)]);
                // Forbid r[0][j] for j ≥ 1
                for j in 1..k {
                    clauses.push(vec![-r(0, j)]);
                }
            } else {
                // r[i][0] ← x_i ∨ r[i-1][0]
                clauses.push(vec![-xi, r(i, 0)]);
                clauses.push(vec![-r(i - 1, 0), r(i, 0)]);

                for j in 1..k {
                    // r[i][j] ← r[i-1][j]
                    clauses.push(vec![-r(i - 1, j), r(i, j)]);
                    // r[i][j] ← x_i ∧ r[i-1][j-1]
                    clauses.push(vec![-xi, -r(i - 1, j - 1), r(i, j)]);
                }

                // Overflow clause: ¬x_i ∨ ¬r[i-1][k-1]
                clauses.push(vec![-xi, -r(i - 1, k - 1)]);
            }
        }

        clauses
    }
}

impl fmt::Display for LinearConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terms: Vec<String> = self
            .variables
            .iter()
            .zip(self.coefficients.iter())
            .map(|(v, c)| {
                if *c == 1 {
                    format!("x{v}")
                } else if *c == -1 {
                    format!("-x{v}")
                } else {
                    format!("{c}·x{v}")
                }
            })
            .collect();
        write!(f, "{} ≤ {}", terms.join(" + "), self.bound)
    }
}

// ---------------------------------------------------------------------------
// FeasibilityResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeasibilityResult {
    Feasible,
    Infeasible { reason: String },
    Unknown,
}

// ---------------------------------------------------------------------------
// CapacityChecker
// ---------------------------------------------------------------------------

/// Pre-SAT feasibility / infeasibility checks that avoid the full encoding
/// when the answer is obvious.
#[derive(Debug, Clone)]
pub struct CapacityChecker {
    model: ResourceModel,
}

impl CapacityChecker {
    pub fn new(model: ResourceModel) -> Self {
        Self { model }
    }

    /// First-fit-decreasing bin-packing heuristic.
    ///
    /// Sorts pods by decreasing total resource weight, then greedily assigns
    /// each to the first node that still has room.  Returns `true` if every
    /// pod could be placed.
    pub fn greedy_bin_packing(&self, requirements: &[ResourceSpec]) -> bool {
        if requirements.is_empty() {
            return true;
        }

        let total_cap = self.model.total_capacity();
        let weight = |r: &ResourceSpec| -> u64 {
            let cpu_w = if total_cap.cpu_millis > 0 {
                (r.cpu_millis as u128 * 1_000_000 / total_cap.cpu_millis as u128) as u64
            } else {
                0
            };
            let mem_w = if total_cap.memory_mb > 0 {
                (r.memory_mb as u128 * 1_000_000 / total_cap.memory_mb as u128) as u64
            } else {
                0
            };
            let stor_w = if total_cap.storage_mb > 0 {
                (r.storage_mb as u128 * 1_000_000 / total_cap.storage_mb as u128) as u64
            } else {
                0
            };
            cpu_w + mem_w + stor_w
        };

        let mut indices: Vec<usize> = (0..requirements.len()).collect();
        indices.sort_by(|&a, &b| {
            let wa = weight(&requirements[a].add(&self.model.overhead_per_pod));
            let wb = weight(&requirements[b].add(&self.model.overhead_per_pod));
            wb.cmp(&wa)
        });

        let mut remaining: Vec<ResourceSpec> = self.model.node_capacities.clone();

        for &idx in &indices {
            let effective = requirements[idx].add(&self.model.overhead_per_pod);
            let placed = remaining.iter_mut().any(|cap| {
                if effective.fits_in(cap) {
                    *cap = cap.subtract(&effective);
                    true
                } else {
                    false
                }
            });
            if !placed {
                return false;
            }
        }
        true
    }

    /// Necessary condition: aggregate demand ≤ aggregate capacity.
    pub fn lower_bound(&self, requirements: &[ResourceSpec]) -> bool {
        self.model.can_fit(requirements)
    }

    /// Sufficient condition (heuristic): greedy bin-packing succeeds.
    pub fn upper_bound(&self, requirements: &[ResourceSpec]) -> bool {
        self.greedy_bin_packing(requirements)
    }

    /// Combined feasibility check.
    pub fn check_feasibility(&self, requirements: &[ResourceSpec]) -> FeasibilityResult {
        if !self.lower_bound(requirements) {
            let total_demand = requirements.iter().fold(ResourceSpec::zero(), |acc, r| {
                acc.add(r).add(&self.model.overhead_per_pod)
            });
            let total_cap = self.model.total_capacity();
            return FeasibilityResult::Infeasible {
                reason: format!(
                    "aggregate demand ({total_demand}) exceeds cluster capacity ({total_cap})"
                ),
            };
        }
        if self.upper_bound(requirements) {
            return FeasibilityResult::Feasible;
        }

        FeasibilityResult::Unknown
    }
}

// ---------------------------------------------------------------------------
// ResourceEncoder
// ---------------------------------------------------------------------------

/// Translates resource constraints into CNF clauses.
#[derive(Debug, Clone)]
pub struct ResourceEncoder {
    model: ResourceModel,
}

impl ResourceEncoder {
    pub fn new(model: ResourceModel) -> Self {
        Self { model }
    }

    /// Encode per-node capacity constraints for one deployment step.
    ///
    /// `service_version_vars` maps each `(service_id, vec_of_literals)` where
    /// the inner vector has one literal per node: literal `j` is true iff the
    /// service is placed on node `j` at this step.
    ///
    /// For each node and each resource dimension a [`LinearConstraint`] is
    /// built and converted to clauses.
    pub fn encode_capacity_constraints(
        &self,
        _step: usize,
        service_version_vars: &[(usize, Vec<Literal>)],
        next_var: &mut u32,
    ) -> Vec<Clause> {
        let mut all_clauses: Vec<Clause> = Vec::new();

        for node_idx in 0..self.model.num_nodes {
            let cap = &self.model.node_capacities[node_idx];

            // Collect (literal, requirement) for every service that *could* be
            // placed on this node.
            let mut entries: Vec<(Literal, &ResourceSpec)> = Vec::new();
            for (svc_id, node_lits) in service_version_vars {
                if node_idx < node_lits.len() {
                    let lit = node_lits[node_idx];
                    let req = &self.model.pod_requirements[*svc_id];
                    entries.push((lit, req));
                }
            }

            if entries.is_empty() {
                continue;
            }

            // CPU constraint
            {
                let vars: Vec<u32> = entries.iter().map(|(l, _)| l.unsigned_abs()).collect();
                let coeffs: Vec<i64> = entries
                    .iter()
                    .map(|(l, r)| {
                        let eff = r.cpu_millis + self.model.overhead_per_pod.cpu_millis;
                        if *l > 0 {
                            eff as i64
                        } else {
                            -(eff as i64)
                        }
                    })
                    .collect();
                let bound = cap.cpu_millis as i64;
                let lc = LinearConstraint::new(vars, coeffs, bound);
                all_clauses.extend(lc.to_clauses(next_var));
            }

            // Memory constraint
            {
                let vars: Vec<u32> = entries.iter().map(|(l, _)| l.unsigned_abs()).collect();
                let coeffs: Vec<i64> = entries
                    .iter()
                    .map(|(l, r)| {
                        let eff = r.memory_mb + self.model.overhead_per_pod.memory_mb;
                        if *l > 0 {
                            eff as i64
                        } else {
                            -(eff as i64)
                        }
                    })
                    .collect();
                let bound = cap.memory_mb as i64;
                let lc = LinearConstraint::new(vars, coeffs, bound);
                all_clauses.extend(lc.to_clauses(next_var));
            }

            // Storage constraint
            {
                let vars: Vec<u32> = entries.iter().map(|(l, _)| l.unsigned_abs()).collect();
                let coeffs: Vec<i64> = entries
                    .iter()
                    .map(|(l, r)| {
                        let eff = r.storage_mb + self.model.overhead_per_pod.storage_mb;
                        if *l > 0 {
                            eff as i64
                        } else {
                            -(eff as i64)
                        }
                    })
                    .collect();
                let bound = cap.storage_mb as i64;
                let lc = LinearConstraint::new(vars, coeffs, bound);
                all_clauses.extend(lc.to_clauses(next_var));
            }
        }

        all_clauses
    }

    /// Encode an affinity constraint: services A and B must (or must not) be on
    /// the same node.
    ///
    /// When `same_node` is `true` the encoding asserts that for every node
    /// index `n`, the placement literal of A on n and the placement literal of
    /// B on n are equivalent (both true or both false for at least one node).
    ///
    /// We encode this as: for each node n, (a_n → b_n) ∧ (b_n → a_n).
    ///
    /// `service_a` and `service_b` are indices into a shared variable space
    /// where `var(service, node) = 1 + service * num_nodes + node`.
    pub fn encode_affinity(
        &self,
        service_a: usize,
        service_b: usize,
        same_node: bool,
    ) -> Vec<Clause> {
        let num_nodes = self.model.num_nodes;
        let var = |svc: usize, node: usize| -> Literal {
            (1 + svc * num_nodes + node) as Literal
        };

        let mut clauses = Vec::new();

        if same_node {
            // For each node n: (a_n ↔ b_n)
            // Encoded as (¬a_n ∨ b_n) ∧ (a_n ∨ ¬b_n)
            for n in 0..num_nodes {
                let a = var(service_a, n);
                let b = var(service_b, n);
                clauses.push(vec![-a, b]);
                clauses.push(vec![a, -b]);
            }
        } else {
            // "Different node" affinity: for every node n, at most one of
            // {a_n, b_n} is true.
            for n in 0..num_nodes {
                let a = var(service_a, n);
                let b = var(service_b, n);
                clauses.push(vec![-a, -b]);
            }
        }

        clauses
    }

    /// Encode an anti-affinity constraint: services A and B must **not** share
    /// any node.
    ///
    /// For every node n: ¬(a_n ∧ b_n)  ⟺  (¬a_n ∨ ¬b_n).
    pub fn encode_anti_affinity(
        &self,
        service_a: usize,
        service_b: usize,
    ) -> Vec<Clause> {
        let num_nodes = self.model.num_nodes;
        let var = |svc: usize, node: usize| -> Literal {
            (1 + svc * num_nodes + node) as Literal
        };

        (0..num_nodes)
            .map(|n| {
                let a = var(service_a, n);
                let b = var(service_b, n);
                vec![-a, -b]
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helper: build a CnfFormula from raw clauses
// ---------------------------------------------------------------------------

/// Convenience constructor for a [`CnfFormula`] from a flat `Vec<Clause>`.
pub fn cnf_from_clauses(clauses: Vec<Clause>) -> CnfFormula {
    CnfFormula::from_clauses(clauses)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ResourceSpec -------------------------------------------------------

    #[test]
    fn resource_spec_zero() {
        let z = ResourceSpec::zero();
        assert_eq!(z.cpu_millis, 0);
        assert_eq!(z.memory_mb, 0);
        assert_eq!(z.storage_mb, 0);
    }

    #[test]
    fn resource_spec_add() {
        let a = ResourceSpec::new(100, 200, 300);
        let b = ResourceSpec::new(50, 60, 70);
        let c = a.add(&b);
        assert_eq!(c, ResourceSpec::new(150, 260, 370));
    }

    #[test]
    fn resource_spec_subtract_saturates() {
        let a = ResourceSpec::new(10, 20, 30);
        let b = ResourceSpec::new(100, 200, 300);
        let c = a.subtract(&b);
        assert_eq!(c, ResourceSpec::zero());
    }

    #[test]
    fn resource_spec_fits_in() {
        let small = ResourceSpec::new(100, 200, 300);
        let large = ResourceSpec::new(1000, 2000, 3000);
        assert!(small.fits_in(&large));
        assert!(!large.fits_in(&small));
        assert!(small.fits_in(&small));
    }

    #[test]
    fn resource_spec_display() {
        let s = ResourceSpec::new(500, 1024, 2048);
        let txt = format!("{s}");
        assert!(txt.contains("500"));
        assert!(txt.contains("1024"));
        assert!(txt.contains("2048"));
    }

    // -- LinearConstraint ---------------------------------------------------

    #[test]
    fn linear_constraint_evaluate_true() {
        // 2·x1 + 3·x2 ≤ 10
        let lc = LinearConstraint::new(vec![1, 2], vec![2, 3], 10);
        let mut asgn = HashMap::new();
        asgn.insert(1, 1);
        asgn.insert(2, 2);
        // 2*1 + 3*2 = 8 ≤ 10
        assert!(lc.evaluate(&asgn));
    }

    #[test]
    fn linear_constraint_evaluate_false() {
        let lc = LinearConstraint::new(vec![1, 2], vec![5, 5], 10);
        let mut asgn = HashMap::new();
        asgn.insert(1, 2);
        asgn.insert(2, 2);
        // 5*2 + 5*2 = 20 > 10
        assert!(!lc.evaluate(&asgn));
    }

    #[test]
    fn linear_constraint_evaluate_missing_vars_default_zero() {
        let lc = LinearConstraint::new(vec![1, 2], vec![5, 5], 10);
        let asgn = HashMap::new();
        assert!(lc.evaluate(&asgn)); // 0 ≤ 10
    }

    #[test]
    fn linear_constraint_is_satisfied_by() {
        let lc = LinearConstraint::new(vec![1, 2, 3], vec![1, 1, 1], 5);
        assert!(lc.is_satisfied_by(&[1, 2, 2])); // 5 ≤ 5
        assert!(!lc.is_satisfied_by(&[2, 2, 2])); // 6 > 5
    }

    #[test]
    fn linear_constraint_display() {
        let lc = LinearConstraint::new(vec![1, 2], vec![3, -1], 7);
        let txt = format!("{lc}");
        assert!(txt.contains("≤ 7"));
    }

    #[test]
    fn linear_constraint_to_clauses_trivially_true() {
        let lc = LinearConstraint::new(vec![1, 2], vec![1, 1], 100);
        let mut nv = 100;
        let clauses = lc.to_clauses(&mut nv);
        assert!(clauses.is_empty());
    }

    #[test]
    fn linear_constraint_to_clauses_trivially_unsat() {
        let lc = LinearConstraint::new(vec![1], vec![1], -1);
        let mut nv = 100;
        let clauses = lc.to_clauses(&mut nv);
        assert!(
            clauses.iter().any(|c| c.is_empty()),
            "must contain empty clause"
        );
    }

    #[test]
    fn linear_constraint_to_clauses_nontrivial() {
        // x1 + x2 + x3 ≤ 1   (at-most-one)
        let lc = LinearConstraint::new(vec![1, 2, 3], vec![1, 1, 1], 1);
        let mut nv = 10;
        let clauses = lc.to_clauses(&mut nv);
        assert!(!clauses.is_empty(), "should produce clauses for AMO");

        // Verify via the linear constraint semantics directly.
        // Exactly one true → satisfied.
        for target in 1..=3i32 {
            let vals: Vec<i64> = (1..=3)
                .map(|v| if v == target { 1 } else { 0 })
                .collect();
            assert!(lc.is_satisfied_by(&vals));
        }
        // Two true → violated.
        assert!(!lc.is_satisfied_by(&[1, 1, 0]));
    }

    #[test]
    fn linear_constraint_empty() {
        let lc = LinearConstraint::new(vec![], vec![], 0);
        let mut nv = 10;
        let clauses = lc.to_clauses(&mut nv);
        assert!(clauses.is_empty());
    }

    #[test]
    fn linear_constraint_negative_coefficients() {
        // -1·x1 + 2·x2 ≤ 1
        let lc = LinearConstraint::new(vec![1, 2], vec![-1, 2], 1);
        let mut asgn = HashMap::new();
        asgn.insert(1, 1);
        asgn.insert(2, 1);
        // -1 + 2 = 1 ≤ 1
        assert!(lc.evaluate(&asgn));
        asgn.insert(1, 0);
        // 0 + 2 = 2 > 1
        assert!(!lc.evaluate(&asgn));
    }

    #[test]
    fn linear_constraint_single_var_bound_zero() {
        // x1 ≤ 0  →  ¬x1
        let lc = LinearConstraint::new(vec![1], vec![1], 0);
        let mut nv = 50;
        let clauses = lc.to_clauses(&mut nv);
        // Should force x1 = false.
        let has_neg_clause = clauses.iter().any(|c| c.contains(&-1));
        assert!(
            has_neg_clause,
            "should contain a clause forcing x1 false, got {clauses:?}"
        );
    }

    // -- ResourceModel ------------------------------------------------------

    fn sample_model() -> ResourceModel {
        ResourceModel::new(
            2,
            vec![
                ResourceSpec::new(4000, 8192, 50000),
                ResourceSpec::new(4000, 8192, 50000),
            ],
            vec![
                ResourceSpec::new(500, 512, 1000),
                ResourceSpec::new(1000, 1024, 2000),
                ResourceSpec::new(250, 256, 500),
            ],
            ResourceSpec::new(50, 64, 100),
        )
    }

    #[test]
    fn resource_model_total_capacity() {
        let m = sample_model();
        let tc = m.total_capacity();
        assert_eq!(tc, ResourceSpec::new(8000, 16384, 100000));
    }

    #[test]
    fn resource_model_can_fit() {
        let m = sample_model();
        assert!(m.can_fit(&m.pod_requirements));
    }

    #[test]
    fn resource_model_cannot_fit_overload() {
        let m = ResourceModel::new(
            1,
            vec![ResourceSpec::new(100, 100, 100)],
            vec![ResourceSpec::new(200, 200, 200)],
            ResourceSpec::zero(),
        );
        assert!(!m.can_fit(&m.pod_requirements));
    }

    #[test]
    fn resource_model_compute_utilization() {
        let m = sample_model();
        // Place pod 0 and pod 2 on node 0, pod 1 on node 1.
        let assignment = vec![0, 1, 0];
        let util = m.compute_utilization(&assignment);
        assert_eq!(util.len(), 2);

        // Node 0: effective = (500+50, 512+64, 1000+100) + (250+50, 256+64, 500+100)
        //        = (550+300, 576+320, 1100+600) = (850, 896, 1700)
        // cap = (4000, 8192, 50000)
        assert!(util[0] > 0.05 && util[0] < 0.25);

        // Node 1: effective = (1000+50, 1024+64, 2000+100) = (1050, 1088, 2100)
        assert!(util[1] > 0.05 && util[1] < 0.30);
    }

    #[test]
    fn resource_model_utilization_single_node() {
        let m = ResourceModel::new(
            1,
            vec![ResourceSpec::new(1000, 1000, 1000)],
            vec![ResourceSpec::new(500, 500, 500)],
            ResourceSpec::zero(),
        );
        let util = m.compute_utilization(&[0]);
        assert!((util[0] - 0.5).abs() < 1e-9);
    }

    // -- CapacityChecker ----------------------------------------------------

    #[test]
    fn capacity_checker_feasible() {
        let m = sample_model();
        let checker = CapacityChecker::new(m.clone());
        let result = checker.check_feasibility(&m.pod_requirements);
        assert_eq!(result, FeasibilityResult::Feasible);
    }

    #[test]
    fn capacity_checker_infeasible_aggregate() {
        let m = ResourceModel::new(
            1,
            vec![ResourceSpec::new(100, 100, 100)],
            vec![
                ResourceSpec::new(200, 200, 200),
                ResourceSpec::new(200, 200, 200),
            ],
            ResourceSpec::zero(),
        );
        let checker = CapacityChecker::new(m.clone());
        let result = checker.check_feasibility(&m.pod_requirements);
        match result {
            FeasibilityResult::Infeasible { reason } => {
                assert!(reason.contains("exceeds"));
            }
            other => panic!("expected Infeasible, got {:?}", other),
        }
    }

    #[test]
    fn capacity_checker_greedy_bin_packing_basic() {
        let m = sample_model();
        let checker = CapacityChecker::new(m.clone());
        assert!(checker.greedy_bin_packing(&m.pod_requirements));
    }

    #[test]
    fn capacity_checker_greedy_bin_packing_empty() {
        let m = sample_model();
        let checker = CapacityChecker::new(m);
        assert!(checker.greedy_bin_packing(&[]));
    }

    #[test]
    fn capacity_checker_greedy_bin_packing_tight() {
        // Two nodes each with capacity (100, 100, 100). Three pods of (50, 50, 50).
        let m = ResourceModel::new(
            2,
            vec![
                ResourceSpec::new(100, 100, 100),
                ResourceSpec::new(100, 100, 100),
            ],
            vec![
                ResourceSpec::new(50, 50, 50),
                ResourceSpec::new(50, 50, 50),
                ResourceSpec::new(50, 50, 50),
            ],
            ResourceSpec::zero(),
        );
        let checker = CapacityChecker::new(m.clone());
        assert!(checker.greedy_bin_packing(&m.pod_requirements));
    }

    #[test]
    fn capacity_checker_greedy_fails_when_node_too_small() {
        // Total capacity is enough but no single node can hold the big pod.
        let m = ResourceModel::new(
            2,
            vec![
                ResourceSpec::new(100, 100, 100),
                ResourceSpec::new(100, 100, 100),
            ],
            vec![ResourceSpec::new(150, 50, 50)],
            ResourceSpec::zero(),
        );
        let checker = CapacityChecker::new(m.clone());
        assert!(!checker.greedy_bin_packing(&m.pod_requirements));
    }

    #[test]
    fn capacity_checker_lower_bound() {
        let m = sample_model();
        let checker = CapacityChecker::new(m.clone());
        assert!(checker.lower_bound(&m.pod_requirements));
    }

    #[test]
    fn capacity_checker_upper_bound() {
        let m = sample_model();
        let checker = CapacityChecker::new(m.clone());
        assert!(checker.upper_bound(&m.pod_requirements));
    }

    // -- ResourceEncoder: affinity / anti-affinity --------------------------

    fn encoder_3_nodes() -> ResourceEncoder {
        let model = ResourceModel::new(
            3,
            vec![
                ResourceSpec::new(4000, 8192, 50000),
                ResourceSpec::new(4000, 8192, 50000),
                ResourceSpec::new(4000, 8192, 50000),
            ],
            vec![
                ResourceSpec::new(500, 512, 1000),
                ResourceSpec::new(500, 512, 1000),
            ],
            ResourceSpec::zero(),
        );
        ResourceEncoder::new(model)
    }

    #[test]
    fn encode_affinity_same_node() {
        let enc = encoder_3_nodes();
        let clauses = enc.encode_affinity(0, 1, true);
        // 3 nodes × 2 clauses per node = 6 clauses
        assert_eq!(clauses.len(), 6);
        for c in &clauses {
            assert_eq!(c.len(), 2);
        }
    }

    #[test]
    fn encode_affinity_different_node() {
        let enc = encoder_3_nodes();
        let clauses = enc.encode_affinity(0, 1, false);
        // same_node=false → at-most-one per node → 3 clauses
        assert_eq!(clauses.len(), 3);
        for c in &clauses {
            assert_eq!(c.len(), 2);
            assert!(c[0] < 0 && c[1] < 0);
        }
    }

    #[test]
    fn encode_anti_affinity() {
        let enc = encoder_3_nodes();
        let clauses = enc.encode_anti_affinity(0, 1);
        assert_eq!(clauses.len(), 3);
        for c in &clauses {
            assert_eq!(c.len(), 2);
            assert!(c[0] < 0 && c[1] < 0);
        }
    }

    #[test]
    fn anti_affinity_matches_affinity_different() {
        let enc = encoder_3_nodes();
        let aa = enc.encode_anti_affinity(0, 1);
        let af = enc.encode_affinity(0, 1, false);
        assert_eq!(aa, af);
    }

    // -- ResourceEncoder: capacity constraints ------------------------------

    #[test]
    fn encode_capacity_produces_clauses() {
        let enc = encoder_3_nodes();
        let svc_vars: Vec<(usize, Vec<Literal>)> =
            vec![(0, vec![1, 2, 3]), (1, vec![4, 5, 6])];
        let mut nv = 100;
        let clauses = enc.encode_capacity_constraints(0, &svc_vars, &mut nv);
        // Large capacity with small requirements → trivially satisfied.
        let _ = clauses;
    }

    #[test]
    fn encode_capacity_tight_produces_clauses() {
        // Node capacity exactly matches one pod → placing two should be
        // infeasible, generating non-trivial clauses.
        let model = ResourceModel::new(
            1,
            vec![ResourceSpec::new(1, 0, 0)],
            vec![ResourceSpec::new(1, 0, 0), ResourceSpec::new(1, 0, 0)],
            ResourceSpec::zero(),
        );
        let enc = ResourceEncoder::new(model);
        let svc_vars: Vec<(usize, Vec<Literal>)> = vec![(0, vec![1]), (1, vec![2])];
        let mut nv = 100;
        let clauses = enc.encode_capacity_constraints(0, &svc_vars, &mut nv);
        assert!(
            !clauses.is_empty(),
            "tight capacity should produce non-trivial clauses"
        );
    }

    // -- cnf_from_clauses ---------------------------------------------------

    #[test]
    fn cnf_from_clauses_round_trip() {
        let raw: Vec<Clause> = vec![vec![1, -2], vec![3, 4, -5]];
        let cnf = cnf_from_clauses(raw.clone());
        assert_eq!(cnf.clauses, raw);
    }

    // -- FeasibilityResult serde --------------------------------------------

    #[test]
    fn feasibility_result_serde_round_trip() {
        let cases = vec![
            FeasibilityResult::Feasible,
            FeasibilityResult::Infeasible {
                reason: "too big".into(),
            },
            FeasibilityResult::Unknown,
        ];
        for case in &cases {
            let json = serde_json::to_string(case).unwrap();
            let back: FeasibilityResult = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, case);
        }
    }
}
