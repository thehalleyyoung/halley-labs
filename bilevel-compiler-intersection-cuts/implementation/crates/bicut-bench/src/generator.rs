//! Random instance generation: generate random bilevel programs with controlled
//! properties (size, density, integrality, coupling strength), ensure
//! well-posedness (feasible, bounded), Knapsack interdiction instances,
//! and network interdiction instances.

use crate::instance::{
    BenchmarkInstance, DifficultyClass, InstanceMetadata, InstanceSet, InstanceType,
};
use bicut_types::{BilevelProblem, SparseMatrix, DEFAULT_TOLERANCE};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Density profile for random matrix generation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DensityProfile {
    /// Uniform density: each entry is nonzero with probability `p`.
    Uniform(f64),
    /// Sparse block diagonal structure.
    BlockDiagonal { block_size: usize, density: f64 },
    /// Band matrix with given bandwidth.
    Banded { bandwidth: usize },
    /// Fully dense.
    Dense,
}

impl DensityProfile {
    /// Nominal density value.
    pub fn density_value(&self) -> f64 {
        match self {
            DensityProfile::Uniform(p) => *p,
            DensityProfile::BlockDiagonal { density, .. } => *density,
            DensityProfile::Banded { bandwidth } => (*bandwidth as f64 * 2.0 + 1.0) / 100.0,
            DensityProfile::Dense => 1.0,
        }
    }
}

/// Configuration for random bilevel program generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Number of upper-level variables.
    pub num_upper_vars: usize,
    /// Number of lower-level variables.
    pub num_lower_vars: usize,
    /// Number of upper-level constraints.
    pub num_upper_constraints: usize,
    /// Number of lower-level constraints.
    pub num_lower_constraints: usize,
    /// Density profile for the lower constraint matrix A.
    pub lower_density: DensityProfile,
    /// Density profile for the linking matrix B.
    pub linking_density: DensityProfile,
    /// Density profile for the upper constraint matrix.
    pub upper_density: DensityProfile,
    /// Range [min, max] for objective coefficients.
    pub obj_coeff_range: (f64, f64),
    /// Range [min, max] for constraint matrix entries.
    pub matrix_coeff_range: (f64, f64),
    /// Range [min, max] for RHS values.
    pub rhs_range: (f64, f64),
    /// Coupling strength: scales the linking matrix coefficients.
    pub coupling_strength: f64,
    /// Whether to ensure feasibility of the generated instance.
    pub ensure_feasible: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig {
            num_upper_vars: 5,
            num_lower_vars: 10,
            num_upper_constraints: 3,
            num_lower_constraints: 8,
            lower_density: DensityProfile::Uniform(0.5),
            linking_density: DensityProfile::Uniform(0.3),
            upper_density: DensityProfile::Uniform(0.4),
            obj_coeff_range: (-10.0, 10.0),
            matrix_coeff_range: (0.0, 10.0),
            rhs_range: (1.0, 100.0),
            coupling_strength: 1.0,
            ensure_feasible: true,
            seed: 42,
        }
    }
}

impl GeneratorConfig {
    /// Builder: set dimensions.
    pub fn with_dimensions(
        mut self,
        n_upper: usize,
        n_lower: usize,
        m_upper: usize,
        m_lower: usize,
    ) -> Self {
        self.num_upper_vars = n_upper;
        self.num_lower_vars = n_lower;
        self.num_upper_constraints = m_upper;
        self.num_lower_constraints = m_lower;
        self
    }

    /// Builder: set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder: set coupling strength.
    pub fn with_coupling(mut self, strength: f64) -> Self {
        self.coupling_strength = strength;
        self
    }

    /// Builder: set lower density.
    pub fn with_lower_density(mut self, density: DensityProfile) -> Self {
        self.lower_density = density;
        self
    }

    /// Builder: set linking density.
    pub fn with_linking_density(mut self, density: DensityProfile) -> Self {
        self.linking_density = density;
        self
    }

    /// Builder: small preset.
    pub fn small() -> Self {
        Self::default().with_dimensions(3, 5, 2, 4)
    }

    /// Builder: medium preset.
    pub fn medium() -> Self {
        Self::default().with_dimensions(10, 20, 8, 15)
    }

    /// Builder: large preset.
    pub fn large() -> Self {
        Self::default().with_dimensions(50, 100, 30, 80)
    }
}

/// Configuration for knapsack interdiction instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnapsackInterdictionConfig {
    /// Number of items.
    pub num_items: usize,
    /// Knapsack capacity (fraction of total weight).
    pub capacity_ratio: f64,
    /// Interdiction budget (fraction of total interdiction cost).
    pub budget_ratio: f64,
    /// Range for item profits.
    pub profit_range: (u64, u64),
    /// Range for item weights.
    pub weight_range: (u64, u64),
    /// Random seed.
    pub seed: u64,
}

impl Default for KnapsackInterdictionConfig {
    fn default() -> Self {
        KnapsackInterdictionConfig {
            num_items: 10,
            capacity_ratio: 0.5,
            budget_ratio: 0.3,
            profit_range: (1, 100),
            weight_range: (1, 50),
            seed: 42,
        }
    }
}

impl KnapsackInterdictionConfig {
    /// Builder: set number of items.
    pub fn with_items(mut self, n: usize) -> Self {
        self.num_items = n;
        self
    }

    /// Builder: set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Configuration for network interdiction instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterdictionConfig {
    /// Number of nodes.
    pub num_nodes: usize,
    /// Probability of an edge existing.
    pub edge_probability: f64,
    /// Interdiction budget (number of edges that can be removed).
    pub interdiction_budget: usize,
    /// Range for arc capacities.
    pub capacity_range: (f64, f64),
    /// Random seed.
    pub seed: u64,
}

impl Default for NetworkInterdictionConfig {
    fn default() -> Self {
        NetworkInterdictionConfig {
            num_nodes: 8,
            edge_probability: 0.4,
            interdiction_budget: 3,
            capacity_range: (1.0, 20.0),
            seed: 42,
        }
    }
}

impl NetworkInterdictionConfig {
    /// Builder: set number of nodes.
    pub fn with_nodes(mut self, n: usize) -> Self {
        self.num_nodes = n;
        self
    }

    /// Builder: set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

// ---------------------------------------------------------------------------
// Instance generator
// ---------------------------------------------------------------------------

/// Random instance generator.
pub struct InstanceGenerator {
    rng: StdRng,
}

impl InstanceGenerator {
    /// Create a generator with a given seed.
    pub fn new(seed: u64) -> Self {
        InstanceGenerator {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate a random bilevel program from the configuration.
    pub fn generate(&mut self, name: &str, config: &GeneratorConfig) -> BenchmarkInstance {
        let n_upper = config.num_upper_vars;
        let n_lower = config.num_lower_vars;
        let m_upper = config.num_upper_constraints;
        let m_lower = config.num_lower_constraints;

        // Generate objectives.
        let upper_obj_x = self.random_vec(n_upper, config.obj_coeff_range);
        let upper_obj_y = self.random_vec(n_lower, config.obj_coeff_range);
        let lower_obj_c = self.random_vec(n_lower, config.obj_coeff_range);

        // Generate lower constraint matrix A.
        let lower_a = self.random_sparse_matrix(
            m_lower,
            n_lower,
            &config.lower_density,
            config.matrix_coeff_range,
        );

        // Generate linking matrix B (scaled by coupling strength).
        let linking_raw = self.random_sparse_matrix(
            m_lower,
            n_upper,
            &config.linking_density,
            config.matrix_coeff_range,
        );
        let mut linking_b = SparseMatrix::new(m_lower, n_upper);
        for entry in &linking_raw.entries {
            linking_b.add_entry(entry.row, entry.col, entry.value * config.coupling_strength);
        }

        // Generate RHS for lower level.
        let lower_b = if config.ensure_feasible {
            // Ensure feasibility by making RHS large enough for y = 0, x = 0.
            // A * 0 = 0 <= b, so any positive b works.
            self.random_vec_positive(m_lower, config.rhs_range)
        } else {
            self.random_vec(m_lower, config.rhs_range)
        };

        // Generate upper constraints.
        let upper_a = self.random_sparse_matrix(
            m_upper,
            n_upper + n_lower,
            &config.upper_density,
            config.matrix_coeff_range,
        );
        let upper_b = self.random_vec_positive(m_upper, config.rhs_range);

        let problem = BilevelProblem {
            upper_obj_c_x: upper_obj_x,
            upper_obj_c_y: upper_obj_y,
            lower_obj_c: lower_obj_c,
            lower_a,
            lower_b,
            lower_linking_b: linking_b,
            upper_constraints_a: upper_a,
            upper_constraints_b: upper_b,
            num_upper_vars: n_upper,
            num_lower_vars: n_lower,
            num_lower_constraints: m_lower,
            num_upper_constraints: m_upper,
        };

        let mut inst = BenchmarkInstance::new(name, problem);
        inst.set_instance_type(InstanceType::BLP);
        inst.add_tag("generated");
        inst
    }

    /// Generate multiple random instances with varying seeds.
    pub fn generate_set(
        &mut self,
        prefix: &str,
        config: &GeneratorConfig,
        count: usize,
    ) -> InstanceSet {
        let mut instances = Vec::with_capacity(count);
        for i in 0..count {
            let name = format!("{}_{:04}", prefix, i);
            let mut cfg = config.clone();
            cfg.seed = config.seed.wrapping_add(i as u64);
            let inst = self.generate(&name, &cfg);
            instances.push(inst);
        }
        InstanceSet::from_instances(&format!("{}_set", prefix), instances)
    }

    /// Generate a knapsack interdiction instance.
    ///
    /// Upper level: leader selects which items to interdict (x_i ∈ {0,1}).
    /// Lower level: follower solves a knapsack over non-interdicted items.
    ///
    /// We model this as a continuous relaxation bilevel program:
    ///   max_x  -p^T y
    ///   s.t.   1^T x <= budget
    ///          0 <= x_i <= 1
    ///          y ∈ argmin_{y'} { -p^T y' : w^T y' <= C, y'_i <= 1 - x_i }
    pub fn generate_knapsack_interdiction(
        &mut self,
        name: &str,
        config: &KnapsackInterdictionConfig,
    ) -> BenchmarkInstance {
        let n = config.num_items;
        let profit_dist = Uniform::new_inclusive(config.profit_range.0, config.profit_range.1);
        let weight_dist = Uniform::new_inclusive(config.weight_range.0, config.weight_range.1);

        let profits: Vec<f64> = (0..n)
            .map(|_| profit_dist.sample(&mut self.rng) as f64)
            .collect();
        let weights: Vec<f64> = (0..n)
            .map(|_| weight_dist.sample(&mut self.rng) as f64)
            .collect();

        let total_weight: f64 = weights.iter().sum();
        let capacity = (total_weight * config.capacity_ratio).ceil();
        let budget = (n as f64 * config.budget_ratio).ceil();

        // Upper level: max -p^T y ↔ min p^T y as the upper objective.
        // Upper vars (x): n items, representing interdiction.
        // Lower vars (y): n items, representing knapsack selection.
        let upper_obj_x = vec![0.0; n];
        let upper_obj_y: Vec<f64> = profits.iter().map(|p| -*p).collect(); // min -profit

        // Lower level: min -p^T y (follower maximizes profit).
        let lower_obj_c: Vec<f64> = profits.iter().map(|p| -*p).collect();

        // Lower constraints:
        // 1) w^T y <= C (knapsack capacity)
        // 2) y_i + x_i <= 1 for each item (linking: interdicted items can't be selected)
        let m_lower = 1 + n;
        let mut lower_a = SparseMatrix::new(m_lower, n);
        // Row 0: knapsack constraint.
        for (j, &w) in weights.iter().enumerate() {
            lower_a.add_entry(0, j, w);
        }
        // Rows 1..n+1: y_i <= 1 - x_i → y_i + x_i <= 1 → y_i <= 1 (in A) with linking.
        for i in 0..n {
            lower_a.add_entry(1 + i, i, 1.0);
        }

        let mut lower_b = vec![0.0; m_lower];
        lower_b[0] = capacity;
        for i in 0..n {
            lower_b[1 + i] = 1.0; // base RHS for y_i <= 1
        }

        // Linking matrix B: for constraints y_i + x_i <= 1 → y_i <= 1 - x_i.
        // A y <= b + B x, where B x adds -x_i to row 1+i.
        let mut linking_b = SparseMatrix::new(m_lower, n);
        for i in 0..n {
            linking_b.add_entry(1 + i, i, -1.0);
        }

        // Upper constraints: 1^T x <= budget, 0 <= x_i <= 1.
        // We model the budget constraint as a single upper constraint.
        let m_upper = 1;
        let mut upper_a = SparseMatrix::new(m_upper, 2 * n);
        for j in 0..n {
            upper_a.add_entry(0, j, 1.0);
        }
        let upper_b = vec![budget];

        let problem = BilevelProblem {
            upper_obj_c_x: upper_obj_x,
            upper_obj_c_y: upper_obj_y,
            lower_obj_c,
            lower_a,
            lower_b,
            lower_linking_b: linking_b,
            upper_constraints_a: upper_a,
            upper_constraints_b: upper_b,
            num_upper_vars: n,
            num_lower_vars: n,
            num_lower_constraints: m_lower,
            num_upper_constraints: m_upper,
        };

        let mut inst = BenchmarkInstance::new(name, problem);
        inst.set_instance_type(InstanceType::KnapsackInterdiction);
        inst.add_tag("knapsack_interdiction");
        inst.add_tag("generated");
        inst
    }

    /// Generate a network interdiction instance (maximum flow).
    ///
    /// Upper: leader removes edges to minimize max flow.
    /// Lower: follower routes max flow from source to sink.
    ///
    /// We generate a random directed graph, then model as a bilevel LP.
    pub fn generate_network_interdiction(
        &mut self,
        name: &str,
        config: &NetworkInterdictionConfig,
    ) -> BenchmarkInstance {
        let n_nodes = config.num_nodes.max(2);
        let source = 0;
        let sink = n_nodes - 1;

        // Generate random edges.
        let cap_dist = Uniform::new(config.capacity_range.0, config.capacity_range.1);
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i == j {
                    continue;
                }
                if self.rng.gen_bool(config.edge_probability.clamp(0.0, 1.0)) {
                    let cap = cap_dist.sample(&mut self.rng);
                    edges.push((i, j, cap));
                }
            }
        }

        // Ensure at least one path from source to sink.
        if edges.is_empty() || !has_path(&edges, source, sink, n_nodes) {
            // Add a direct edge.
            let cap = cap_dist.sample(&mut self.rng);
            edges.push((source, sink, cap));
        }

        let n_edges = edges.len();

        // Upper vars: x_e ∈ [0,1] for each edge (interdiction).
        // Lower vars: f_e >= 0 for each edge (flow).
        let n_upper = n_edges;
        let n_lower = n_edges;

        // Upper objective: minimize the flow value.
        // The flow value is the net flow out of source.
        // For minimization: upper_obj_y[e] = 1 if e leaves source, -1 if e enters source.
        let upper_obj_x = vec![0.0; n_upper];
        let mut upper_obj_y = vec![0.0; n_lower];
        for (e, &(from, to, _)) in edges.iter().enumerate() {
            if from == source {
                upper_obj_y[e] = 1.0; // minimize flow out of source (min -maxflow → we want min)
            }
            if to == source {
                upper_obj_y[e] = -1.0;
            }
        }

        // Lower objective: maximize flow = min -flow value.
        let mut lower_obj = vec![0.0; n_lower];
        for (e, &(from, to, _)) in edges.iter().enumerate() {
            if from == source {
                lower_obj[e] = -1.0; // minimize negative flow = maximize flow
            }
            if to == source {
                lower_obj[e] = 1.0;
            }
        }

        // Lower constraints:
        // 1) Flow conservation: for each non-source, non-sink node.
        //    sum_{e in delta+(v)} f_e - sum_{e in delta-(v)} f_e = 0
        //    We model as ≤ 0 and ≥ 0, i.e., two constraints per node.
        //    For simplicity, use one ≤ constraint and one ≥ constraint (as ≤ with negation).
        // 2) Capacity: f_e ≤ cap_e * (1 - x_e)  → f_e ≤ cap_e + (-cap_e)*x_e.
        let interior_nodes: Vec<usize> =
            (0..n_nodes).filter(|&v| v != source && v != sink).collect();
        let n_flow_cons = 2 * interior_nodes.len();
        let n_cap_cons = n_edges;
        let m_lower = n_flow_cons + n_cap_cons;

        let mut lower_a = SparseMatrix::new(m_lower, n_lower);
        let mut lower_b = vec![0.0; m_lower];
        let mut linking_b = SparseMatrix::new(m_lower, n_upper);

        // Flow conservation.
        for (idx, &v) in interior_nodes.iter().enumerate() {
            let row_pos = 2 * idx; // sum_out - sum_in <= 0
            let row_neg = 2 * idx + 1; // -(sum_out - sum_in) <= 0

            for (e, &(from, to, _)) in edges.iter().enumerate() {
                if from == v {
                    lower_a.add_entry(row_pos, e, 1.0);
                    lower_a.add_entry(row_neg, e, -1.0);
                }
                if to == v {
                    lower_a.add_entry(row_pos, e, -1.0);
                    lower_a.add_entry(row_neg, e, 1.0);
                }
            }
        }

        // Capacity constraints: f_e <= cap_e - cap_e * x_e.
        for (e, &(_, _, cap)) in edges.iter().enumerate() {
            let row = n_flow_cons + e;
            lower_a.add_entry(row, e, 1.0);
            lower_b[row] = cap;
            linking_b.add_entry(row, e, -cap); // b + B x: b=cap, B[row,e]=-cap
        }

        // Upper constraint: budget on interdiction.
        let m_upper = 1;
        let mut upper_a = SparseMatrix::new(m_upper, n_upper + n_lower);
        for e in 0..n_edges {
            upper_a.add_entry(0, e, 1.0);
        }
        let upper_b = vec![config.interdiction_budget as f64];

        let problem = BilevelProblem {
            upper_obj_c_x: upper_obj_x,
            upper_obj_c_y: upper_obj_y,
            lower_obj_c: lower_obj,
            lower_a,
            lower_b,
            lower_linking_b: linking_b,
            upper_constraints_a: upper_a,
            upper_constraints_b: upper_b,
            num_upper_vars: n_upper,
            num_lower_vars: n_lower,
            num_lower_constraints: m_lower,
            num_upper_constraints: m_upper,
        };

        let mut inst = BenchmarkInstance::new(name, problem);
        inst.set_instance_type(InstanceType::NetworkInterdiction);
        inst.add_tag("network_interdiction");
        inst.add_tag("generated");
        inst
    }

    /// Generate a set of instances with increasing size for scalability tests.
    pub fn generate_scalability_set(
        &mut self,
        prefix: &str,
        base_config: &GeneratorConfig,
        sizes: &[usize],
    ) -> InstanceSet {
        let mut instances = Vec::with_capacity(sizes.len());
        for (idx, &size) in sizes.iter().enumerate() {
            let name = format!("{}_n{}", prefix, size);
            let config = GeneratorConfig {
                num_upper_vars: size,
                num_lower_vars: size * 2,
                num_upper_constraints: size,
                num_lower_constraints: size * 2,
                seed: base_config.seed.wrapping_add(idx as u64),
                ..base_config.clone()
            };
            instances.push(self.generate(&name, &config));
        }
        InstanceSet::from_instances(&format!("{}_scalability", prefix), instances)
    }

    /// Generate instances with varying coupling strength.
    pub fn generate_coupling_sweep(
        &mut self,
        prefix: &str,
        base_config: &GeneratorConfig,
        strengths: &[f64],
    ) -> InstanceSet {
        let mut instances = Vec::with_capacity(strengths.len());
        for (idx, &s) in strengths.iter().enumerate() {
            let name = format!("{}_coupling_{:.1}", prefix, s);
            let config = GeneratorConfig {
                coupling_strength: s,
                seed: base_config.seed.wrapping_add(idx as u64),
                ..base_config.clone()
            };
            instances.push(self.generate(&name, &config));
        }
        InstanceSet::from_instances(&format!("{}_coupling", prefix), instances)
    }

    // -- Internal helpers --

    fn random_vec(&mut self, n: usize, range: (f64, f64)) -> Vec<f64> {
        let dist = Uniform::new(range.0, range.1);
        (0..n).map(|_| dist.sample(&mut self.rng)).collect()
    }

    fn random_vec_positive(&mut self, n: usize, range: (f64, f64)) -> Vec<f64> {
        let lo = range.0.max(0.01);
        let hi = range.1.max(lo + 0.01);
        let dist = Uniform::new(lo, hi);
        (0..n).map(|_| dist.sample(&mut self.rng)).collect()
    }

    fn random_sparse_matrix(
        &mut self,
        rows: usize,
        cols: usize,
        profile: &DensityProfile,
        coeff_range: (f64, f64),
    ) -> SparseMatrix {
        let mut mat = SparseMatrix::new(rows, cols);
        let coeff_dist = Uniform::new(coeff_range.0, coeff_range.1);

        match profile {
            DensityProfile::Uniform(p) => {
                for i in 0..rows {
                    for j in 0..cols {
                        if self.rng.gen_bool((*p).clamp(0.0, 1.0)) {
                            let v = coeff_dist.sample(&mut self.rng);
                            if v.abs() > DEFAULT_TOLERANCE {
                                mat.add_entry(i, j, v);
                            }
                        }
                    }
                }
            }
            DensityProfile::BlockDiagonal {
                block_size,
                density,
            } => {
                let bs = (*block_size).max(1);
                for i in 0..rows {
                    let block = i / bs;
                    for j in 0..cols {
                        let j_block = j / bs;
                        if block == j_block && self.rng.gen_bool((*density).clamp(0.0, 1.0)) {
                            let v = coeff_dist.sample(&mut self.rng);
                            if v.abs() > DEFAULT_TOLERANCE {
                                mat.add_entry(i, j, v);
                            }
                        }
                    }
                }
            }
            DensityProfile::Banded { bandwidth } => {
                let bw = *bandwidth;
                for i in 0..rows {
                    let j_min = if i >= bw { i - bw } else { 0 };
                    let j_max = (i + bw + 1).min(cols);
                    for j in j_min..j_max {
                        let v = coeff_dist.sample(&mut self.rng);
                        if v.abs() > DEFAULT_TOLERANCE {
                            mat.add_entry(i, j, v);
                        }
                    }
                }
            }
            DensityProfile::Dense => {
                for i in 0..rows {
                    for j in 0..cols {
                        let v = coeff_dist.sample(&mut self.rng);
                        if v.abs() > DEFAULT_TOLERANCE {
                            mat.add_entry(i, j, v);
                        }
                    }
                }
            }
        }

        // Ensure at least one entry per row for non-triviality.
        for i in 0..rows {
            let has_entry = mat.entries.iter().any(|e| e.row == i);
            if !has_entry && cols > 0 {
                let j = self.rng.gen_range(0..cols);
                let v = coeff_dist.sample(&mut self.rng);
                mat.add_entry(i, j, if v.abs() < DEFAULT_TOLERANCE { 1.0 } else { v });
            }
        }

        mat
    }
}

/// BFS-based reachability check.
fn has_path(edges: &[(usize, usize, f64)], source: usize, sink: usize, n_nodes: usize) -> bool {
    let mut visited = vec![false; n_nodes];
    let mut queue = std::collections::VecDeque::new();
    visited[source] = true;
    queue.push_back(source);
    while let Some(v) = queue.pop_front() {
        if v == sink {
            return true;
        }
        for &(from, to, _) in edges {
            if from == v && !visited[to] {
                visited[to] = true;
                queue.push_back(to);
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_default() {
        let mut gen = InstanceGenerator::new(123);
        let config = GeneratorConfig::default();
        let inst = gen.generate("test_default", &config);
        assert_eq!(inst.problem.num_upper_vars, 5);
        assert_eq!(inst.problem.num_lower_vars, 10);
        assert!(inst.validate().is_ok());
    }

    #[test]
    fn test_generate_small() {
        let mut gen = InstanceGenerator::new(42);
        let config = GeneratorConfig::small();
        let inst = gen.generate("small", &config);
        assert_eq!(inst.problem.num_upper_vars, 3);
        assert!(inst.validate().is_ok());
    }

    #[test]
    fn test_generate_set() {
        let mut gen = InstanceGenerator::new(0);
        let config = GeneratorConfig::small();
        let set = gen.generate_set("batch", &config, 5);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_generate_deterministic() {
        let config = GeneratorConfig::default().with_seed(999);
        let mut gen1 = InstanceGenerator::new(999);
        let inst1 = gen1.generate("a", &config);
        let mut gen2 = InstanceGenerator::new(999);
        let inst2 = gen2.generate("a", &config);
        assert_eq!(inst1.problem.lower_obj_c, inst2.problem.lower_obj_c);
    }

    #[test]
    fn test_knapsack_interdiction() {
        let mut gen = InstanceGenerator::new(42);
        let config = KnapsackInterdictionConfig::default();
        let inst = gen.generate_knapsack_interdiction("ki_test", &config);
        assert_eq!(inst.instance_type(), InstanceType::KnapsackInterdiction);
        assert!(inst.validate().is_ok());
        // n items → n upper vars, n lower vars.
        assert_eq!(inst.problem.num_upper_vars, config.num_items);
        assert_eq!(inst.problem.num_lower_vars, config.num_items);
    }

    #[test]
    fn test_network_interdiction() {
        let mut gen = InstanceGenerator::new(42);
        let config = NetworkInterdictionConfig::default();
        let inst = gen.generate_network_interdiction("ni_test", &config);
        assert_eq!(inst.instance_type(), InstanceType::NetworkInterdiction);
        assert!(inst.validate().is_ok());
    }

    #[test]
    fn test_scalability_set() {
        let mut gen = InstanceGenerator::new(0);
        let config = GeneratorConfig::default();
        let set = gen.generate_scalability_set("scale", &config, &[2, 5, 10]);
        assert_eq!(set.len(), 3);
        // Sizes should increase.
        assert!(set.instances[0].problem.num_upper_vars < set.instances[2].problem.num_upper_vars);
    }

    #[test]
    fn test_coupling_sweep() {
        let mut gen = InstanceGenerator::new(0);
        let config = GeneratorConfig::small();
        let set = gen.generate_coupling_sweep("coup", &config, &[0.0, 0.5, 1.0, 2.0]);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_density_profiles() {
        let mut gen = InstanceGenerator::new(42);
        let range = (0.0, 10.0);
        let mat_uniform = gen.random_sparse_matrix(5, 5, &DensityProfile::Uniform(0.5), range);
        assert!(!mat_uniform.entries.is_empty());
        let mat_dense = gen.random_sparse_matrix(3, 3, &DensityProfile::Dense, range);
        assert!(mat_dense.entries.len() >= 3);
        let mat_band =
            gen.random_sparse_matrix(5, 5, &DensityProfile::Banded { bandwidth: 1 }, range);
        assert!(!mat_band.entries.is_empty());
    }

    #[test]
    fn test_has_path() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        assert!(has_path(&edges, 0, 2, 3));
        assert!(!has_path(&edges, 2, 0, 3));
    }
}
