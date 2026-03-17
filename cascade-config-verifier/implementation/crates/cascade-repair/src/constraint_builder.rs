//! Constraint construction for repair synthesis.
//!
//! Translates graph structure and risk information into mathematical
//! constraints (amplification products, timeout sums, consistency buffers)
//! that the repair synthesizer can solve against.

use serde::{Deserialize, Serialize};

use super::synthesizer::ParameterBounds;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A decision variable representing a tuneable edge parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairVariable {
    pub id: usize,
    pub edge: (String, String),
    pub parameter: String,
    pub current_value: f64,
    pub min_value: f64,
    pub max_value: f64,
}

impl RepairVariable {
    /// Returns the allowed range width.
    pub fn range(&self) -> f64 {
        self.max_value - self.min_value
    }
}

/// Weights controlling the trade-off between retry and timeout deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairWeights {
    pub retry_weight: f64,
    pub timeout_weight: f64,
}

impl Default for RepairWeights {
    fn default() -> Self {
        Self {
            retry_weight: 1.0,
            timeout_weight: 1.0,
        }
    }
}

/// Constraint: the product of `(1 + retry)` factors along a path must not
/// exceed `max_product`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplificationConstraint {
    pub path: Vec<String>,
    pub max_product: f64,
    pub variable_indices: Vec<usize>,
}

impl AmplificationConstraint {
    /// Check if the given variable assignments satisfy this constraint.
    pub fn is_satisfied(&self, vars: &[RepairVariable]) -> bool {
        let product: f64 = self
            .variable_indices
            .iter()
            .filter_map(|&i| vars.get(i))
            .filter(|v| v.parameter == "retry")
            .map(|v| 1.0 + v.current_value)
            .product();
        product <= self.max_product
    }
}

/// Constraint: the sum of timeouts along a path must not exceed `max_sum`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConstraint {
    pub path: Vec<String>,
    pub max_sum: u64,
    pub variable_indices: Vec<usize>,
}

impl TimeoutConstraint {
    pub fn is_satisfied(&self, vars: &[RepairVariable]) -> bool {
        let sum: f64 = self
            .variable_indices
            .iter()
            .filter_map(|&i| vars.get(i))
            .filter(|v| v.parameter == "timeout")
            .map(|v| v.current_value)
            .sum();
        (sum as u64) <= self.max_sum
    }
}

/// Constraint: the upstream timeout must exceed the downstream timeout by at
/// least `min_buffer_ms`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConstraint {
    pub upstream_var: usize,
    pub downstream_var: usize,
    pub min_buffer_ms: u64,
}

impl ConsistencyConstraint {
    pub fn is_satisfied(&self, vars: &[RepairVariable]) -> bool {
        let up = vars.get(self.upstream_var).map(|v| v.current_value).unwrap_or(0.0);
        let down = vars.get(self.downstream_var).map(|v| v.current_value).unwrap_or(0.0);
        up >= down + self.min_buffer_ms as f64
    }
}

/// Simple box constraint on a single variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundConstraint {
    pub variable: usize,
    pub min: f64,
    pub max: f64,
}

impl BoundConstraint {
    pub fn is_satisfied(&self, vars: &[RepairVariable]) -> bool {
        vars.get(self.variable)
            .map(|v| v.current_value >= self.min && v.current_value <= self.max)
            .unwrap_or(false)
    }
}

/// Objective term: penalise deviation of a variable from its current value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationObjective {
    pub variable: usize,
    pub current: f64,
    pub weight: f64,
}

impl DeviationObjective {
    pub fn cost(&self, vars: &[RepairVariable]) -> f64 {
        vars.get(self.variable)
            .map(|v| self.weight * (v.current_value - self.current).abs())
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// ConstraintBuilder
// ---------------------------------------------------------------------------

/// Builds constraints and variables from the graph adjacency for the repair
/// synthesis engine.
#[derive(Debug, Clone, Default)]
pub struct ConstraintBuilder;

impl ConstraintBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Create one retry variable and one timeout variable for each edge.
    pub fn create_variables(
        &self,
        adj: &[(String, String, u32, u64)],
        bounds: &ParameterBounds,
    ) -> Vec<RepairVariable> {
        let mut vars = Vec::new();
        let mut id = 0usize;
        for (src, tgt, retries, timeout) in adj {
            vars.push(RepairVariable {
                id,
                edge: (src.clone(), tgt.clone()),
                parameter: "retry".to_string(),
                current_value: *retries as f64,
                min_value: bounds.min_retry as f64,
                max_value: bounds.max_retry as f64,
            });
            id += 1;
            vars.push(RepairVariable {
                id,
                edge: (src.clone(), tgt.clone()),
                parameter: "timeout".to_string(),
                current_value: *timeout as f64,
                min_value: bounds.min_timeout_ms as f64,
                max_value: bounds.max_timeout_ms as f64,
            });
            id += 1;
        }
        vars
    }

    /// Build an amplification constraint for a specific path.
    pub fn build_amplification_constraint(
        &self,
        path: &[String],
        adj: &[(String, String, u32, u64)],
        threshold: f64,
        vars: &[RepairVariable],
    ) -> AmplificationConstraint {
        let mut indices = Vec::new();
        for w in path.windows(2) {
            // Find the retry variable for this edge.
            for v in vars.iter() {
                if v.edge.0 == w[0] && v.edge.1 == w[1] && v.parameter == "retry" {
                    indices.push(v.id);
                }
            }
        }
        // Also include timeout vars that are on the path (for completeness).
        for w in path.windows(2) {
            for v in vars.iter() {
                if v.edge.0 == w[0] && v.edge.1 == w[1] && v.parameter == "timeout" {
                    indices.push(v.id);
                }
            }
        }
        let _ = adj; // adj used indirectly through vars
        AmplificationConstraint {
            path: path.to_vec(),
            max_product: threshold,
            variable_indices: indices,
        }
    }

    /// Build a timeout constraint for a specific path.
    ///
    /// The adjacency here is `(source, target, timeout_ms)`.
    pub fn build_timeout_constraint(
        &self,
        path: &[String],
        adj: &[(String, String, u64)],
        deadline: u64,
        vars: &[RepairVariable],
    ) -> TimeoutConstraint {
        let mut indices = Vec::new();
        for w in path.windows(2) {
            for v in vars.iter() {
                if v.edge.0 == w[0] && v.edge.1 == w[1] && v.parameter == "timeout" {
                    indices.push(v.id);
                }
            }
        }
        let _ = adj; // adj used indirectly through vars
        TimeoutConstraint {
            path: path.to_vec(),
            max_sum: deadline,
            variable_indices: indices,
        }
    }

    /// Build bound constraints for all variables.
    pub fn build_bound_constraints(&self, vars: &[RepairVariable]) -> Vec<BoundConstraint> {
        vars.iter()
            .map(|v| BoundConstraint {
                variable: v.id,
                min: v.min_value,
                max: v.max_value,
            })
            .collect()
    }

    /// Build deviation objectives for all variables.
    pub fn build_deviation_objectives(
        &self,
        vars: &[RepairVariable],
        weights: &RepairWeights,
    ) -> Vec<DeviationObjective> {
        vars.iter()
            .map(|v| {
                let w = match v.parameter.as_str() {
                    "retry" => weights.retry_weight,
                    "timeout" => weights.timeout_weight,
                    _ => 1.0,
                };
                DeviationObjective {
                    variable: v.id,
                    current: v.current_value,
                    weight: w,
                }
            })
            .collect()
    }

    /// Build consistency constraints: for every pair of edges
    /// `(A->B, B->C)` on a path, the upstream timeout should exceed the
    /// downstream timeout by `min_buffer_ms`.
    pub fn build_consistency_constraints(
        &self,
        path: &[String],
        vars: &[RepairVariable],
        min_buffer_ms: u64,
    ) -> Vec<ConsistencyConstraint> {
        if path.len() < 3 {
            return Vec::new();
        }
        let mut constraints = Vec::new();
        for triple in path.windows(3) {
            let upstream = vars
                .iter()
                .find(|v| v.edge.0 == triple[0] && v.edge.1 == triple[1] && v.parameter == "timeout");
            let downstream = vars
                .iter()
                .find(|v| v.edge.0 == triple[1] && v.edge.1 == triple[2] && v.parameter == "timeout");
            if let (Some(up), Some(down)) = (upstream, downstream) {
                constraints.push(ConsistencyConstraint {
                    upstream_var: up.id,
                    downstream_var: down.id,
                    min_buffer_ms,
                });
            }
        }
        constraints
    }

    /// Compute total deviation cost for a set of variables given objectives.
    pub fn total_deviation_cost(
        vars: &[RepairVariable],
        objectives: &[DeviationObjective],
    ) -> f64 {
        objectives.iter().map(|o| o.cost(vars)).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_adj() -> Vec<(String, String, u32, u64)> {
        vec![
            ("A".into(), "B".into(), 3, 2000),
            ("B".into(), "C".into(), 4, 3000),
            ("C".into(), "D".into(), 2, 1500),
        ]
    }

    fn sample_bounds() -> ParameterBounds {
        ParameterBounds {
            min_retry: 0,
            max_retry: 10,
            min_timeout_ms: 100,
            max_timeout_ms: 30_000,
        }
    }

    #[test]
    fn test_create_variables_count() {
        let cb = ConstraintBuilder::new();
        let vars = cb.create_variables(&sample_adj(), &sample_bounds());
        // 3 edges × 2 vars each = 6
        assert_eq!(vars.len(), 6);
    }

    #[test]
    fn test_create_variables_ids_unique() {
        let cb = ConstraintBuilder::new();
        let vars = cb.create_variables(&sample_adj(), &sample_bounds());
        let ids: Vec<usize> = vars.iter().map(|v| v.id).collect();
        let unique: std::collections::HashSet<usize> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_variable_bounds() {
        let cb = ConstraintBuilder::new();
        let bounds = sample_bounds();
        let vars = cb.create_variables(&sample_adj(), &bounds);
        for v in &vars {
            if v.parameter == "retry" {
                assert_eq!(v.min_value, bounds.min_retry as f64);
                assert_eq!(v.max_value, bounds.max_retry as f64);
            } else {
                assert_eq!(v.min_value, bounds.min_timeout_ms as f64);
                assert_eq!(v.max_value, bounds.max_timeout_ms as f64);
            }
        }
    }

    #[test]
    fn test_build_amplification_constraint() {
        let cb = ConstraintBuilder::new();
        let adj = sample_adj();
        let vars = cb.create_variables(&adj, &sample_bounds());
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let ac = cb.build_amplification_constraint(&path, &adj, 20.0, &vars);
        assert_eq!(ac.path.len(), 3);
        assert!((ac.max_product - 20.0).abs() < 1e-9);
        // Should contain indices for A->B and B->C (retry + timeout).
        assert!(!ac.variable_indices.is_empty());
    }

    #[test]
    fn test_amplification_constraint_satisfied() {
        let cb = ConstraintBuilder::new();
        let adj = vec![
            ("A".to_string(), "B".to_string(), 2u32, 1000u64),
            ("B".to_string(), "C".to_string(), 1u32, 1000u64),
        ];
        let vars = cb.create_variables(&adj, &sample_bounds());
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let ac = cb.build_amplification_constraint(&path, &adj, 10.0, &vars);
        // (1+2) * (1+1) = 6 <= 10 → satisfied
        assert!(ac.is_satisfied(&vars));
    }

    #[test]
    fn test_amplification_constraint_violated() {
        let cb = ConstraintBuilder::new();
        let adj = vec![
            ("A".to_string(), "B".to_string(), 5u32, 1000u64),
            ("B".to_string(), "C".to_string(), 4u32, 1000u64),
        ];
        let vars = cb.create_variables(&adj, &sample_bounds());
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let ac = cb.build_amplification_constraint(&path, &adj, 10.0, &vars);
        // (1+5) * (1+4) = 30 > 10 → violated
        assert!(!ac.is_satisfied(&vars));
    }

    #[test]
    fn test_build_timeout_constraint() {
        let cb = ConstraintBuilder::new();
        let adj4 = sample_adj();
        let vars = cb.create_variables(&adj4, &sample_bounds());
        let adj3: Vec<(String, String, u64)> = adj4
            .iter()
            .map(|(s, t, _, to)| (s.clone(), t.clone(), *to))
            .collect();
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let tc = cb.build_timeout_constraint(&path, &adj3, 4000, &vars);
        assert_eq!(tc.max_sum, 4000);
        // 2000 + 3000 = 5000 > 4000 → not satisfied
        assert!(!tc.is_satisfied(&vars));
    }

    #[test]
    fn test_build_bound_constraints() {
        let cb = ConstraintBuilder::new();
        let vars = cb.create_variables(&sample_adj(), &sample_bounds());
        let bounds = cb.build_bound_constraints(&vars);
        assert_eq!(bounds.len(), vars.len());
        for bc in &bounds {
            assert!(bc.is_satisfied(&vars));
        }
    }

    #[test]
    fn test_build_deviation_objectives() {
        let cb = ConstraintBuilder::new();
        let vars = cb.create_variables(&sample_adj(), &sample_bounds());
        let weights = RepairWeights {
            retry_weight: 2.0,
            timeout_weight: 0.5,
        };
        let objs = cb.build_deviation_objectives(&vars, &weights);
        assert_eq!(objs.len(), vars.len());
        // Cost should be 0 when values haven't changed.
        let total = ConstraintBuilder::total_deviation_cost(&vars, &objs);
        assert!((total - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_consistency_constraints() {
        let cb = ConstraintBuilder::new();
        let adj = sample_adj();
        let vars = cb.create_variables(&adj, &sample_bounds());
        let path: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let cc = cb.build_consistency_constraints(&path, &vars, 500);
        // path of 4 nodes → windows of 3 → 2 consistency constraints
        assert_eq!(cc.len(), 2);
    }

    #[test]
    fn test_deviation_cost_nonzero() {
        let cb = ConstraintBuilder::new();
        let adj = vec![("X".to_string(), "Y".to_string(), 5u32, 3000u64)];
        let mut vars = cb.create_variables(&adj, &sample_bounds());
        let objs = cb.build_deviation_objectives(&vars, &RepairWeights::default());
        // Modify the retry var.
        vars[0].current_value = 2.0;
        let cost = ConstraintBuilder::total_deviation_cost(&vars, &objs);
        // |2 - 5| * 1.0 = 3.0
        assert!((cost - 3.0).abs() < 1e-9);
    }
}
