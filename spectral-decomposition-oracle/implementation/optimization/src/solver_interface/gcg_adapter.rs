//! GCG solver adapter (mock/emulation for standalone use).
//!
//! Emulates GCG-like Dantzig-Wolfe decomposition behavior for testing
//! without requiring actual GCG/SCIP C bindings.

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution, SolverStatus};
use crate::solver_interface::{SolverConfig, SolverInterface};
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;

/// GCG decomposition format (analogous to .dec files).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecFormat {
    pub num_blocks: usize,
    pub block_constraints: Vec<Vec<usize>>,
    pub linking_constraints: Vec<usize>,
    pub block_variables: Vec<Vec<usize>>,
}

/// Statistics from a GCG emulation solve.
#[derive(Debug, Clone, Default)]
pub struct GcgStats {
    pub master_lp_iterations: usize,
    pub pricing_rounds: usize,
    pub columns_generated: usize,
    pub solve_time: f64,
}

/// GCG solver mock adapter.
pub struct GcgAdapter {
    config: SolverConfig,
    problem: Option<LpProblem>,
    last_solution: Option<LpSolution>,
    status: SolverStatus,
    dec_format: Option<DecFormat>,
    stats: GcgStats,
}

impl GcgAdapter {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            problem: None,
            last_solution: None,
            status: SolverStatus::NumericalError,
            dec_format: None,
            stats: GcgStats::default(),
        }
    }

    /// Generate a DecFormat from a variable partition.
    pub fn generate_dec_format(
        &self,
        problem: &LpProblem,
        partition: &[usize],
    ) -> OptResult<DecFormat> {
        if partition.len() != problem.num_vars {
            return Err(OptError::invalid_problem(format!(
                "Partition length {} != num_vars {}",
                partition.len(),
                problem.num_vars
            )));
        }

        let num_blocks = partition.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut block_variables: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for (v, &b) in partition.iter().enumerate() {
            block_variables[b].push(v);
        }

        let mut block_constraints: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        let mut linking_constraints = Vec::new();

        for i in 0..problem.num_constraints {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };

            let mut blocks_touched = HashSet::new();
            for idx in row_start..row_end {
                if idx < problem.col_indices.len() {
                    blocks_touched.insert(partition[problem.col_indices[idx]]);
                }
            }

            if blocks_touched.len() > 1 {
                linking_constraints.push(i);
            } else if let Some(&b) = blocks_touched.iter().next() {
                block_constraints[b].push(i);
            }
        }

        Ok(DecFormat {
            num_blocks,
            block_constraints,
            linking_constraints,
            block_variables,
        })
    }

    /// Serialize DecFormat to .dec file format string.
    pub fn format_dec_string(&self, dec: &DecFormat) -> String {
        let mut lines = Vec::new();
        lines.push("PRESOLVED".to_string());
        lines.push("0".to_string());
        lines.push(format!("NBLOCKS"));
        lines.push(format!("{}", dec.num_blocks));

        for (b, constraints) in dec.block_constraints.iter().enumerate() {
            lines.push(format!("BLOCK {}", b + 1));
            for &ci in constraints {
                lines.push(format!("  c{}", ci));
            }
        }

        lines.push("MASTERCONSS".to_string());
        for &ci in &dec.linking_constraints {
            lines.push(format!("  c{}", ci));
        }

        lines.join("\n")
    }

    /// Parse a .dec format string into DecFormat.
    pub fn parse_dec_string(dec_str: &str) -> OptResult<DecFormat> {
        let mut num_blocks = 0;
        let mut block_constraints: Vec<Vec<usize>> = Vec::new();
        let mut linking_constraints = Vec::new();
        let mut current_block: Option<usize> = None;
        let mut in_master = false;

        for line in dec_str.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "NBLOCKS" {
                continue;
            }

            if trimmed == "PRESOLVED" {
                continue;
            }

            if trimmed == "MASTERCONSS" {
                in_master = true;
                current_block = None;
                continue;
            }

            if trimmed.starts_with("BLOCK ") {
                in_master = false;
                let block_num: usize = trimmed[6..]
                    .trim()
                    .parse()
                    .map_err(|_| OptError::invalid_problem("Invalid block number in dec"))?;
                let block_idx = block_num.saturating_sub(1);
                while block_constraints.len() <= block_idx {
                    block_constraints.push(Vec::new());
                }
                current_block = Some(block_idx);
                num_blocks = num_blocks.max(block_num);
                continue;
            }

            if let Some(ci_str) = trimmed.strip_prefix('c') {
                if let Ok(ci) = ci_str.parse::<usize>() {
                    if in_master {
                        linking_constraints.push(ci);
                    } else if let Some(b) = current_block {
                        if b < block_constraints.len() {
                            block_constraints[b].push(ci);
                        }
                    }
                }
            } else if let Ok(n) = trimmed.parse::<usize>() {
                if num_blocks == 0 {
                    num_blocks = n;
                    block_constraints = vec![Vec::new(); n];
                }
            }
        }

        if num_blocks == 0 {
            num_blocks = block_constraints.len().max(1);
        }

        Ok(DecFormat {
            num_blocks,
            block_constraints,
            linking_constraints,
            block_variables: vec![Vec::new(); num_blocks],
        })
    }

    /// Set the decomposition structure.
    pub fn set_decomposition(&mut self, dec: DecFormat) {
        self.dec_format = Some(dec);
    }

    /// Simple structure detection by analyzing constraint matrix sparsity.
    pub fn detect_structure(&self, problem: &LpProblem) -> OptResult<DecFormat> {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        if n == 0 || m == 0 {
            return Ok(DecFormat {
                num_blocks: 1,
                block_constraints: vec![(0..m).collect()],
                linking_constraints: Vec::new(),
                block_variables: vec![(0..n).collect()],
            });
        }

        // Build adjacency: two variables are connected if they appear in the same constraint
        let mut var_constraints: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..m {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };
            for idx in row_start..row_end {
                if idx < problem.col_indices.len() {
                    let v = problem.col_indices[idx];
                    if v < n {
                        var_constraints[v].push(i);
                    }
                }
            }
        }

        // Simple connected components via union-find
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let ri = find(parent, i);
            let rj = find(parent, j);
            if ri != rj {
                parent[ri] = rj;
            }
        }

        // Connect variables that share constraints (but limit to find natural blocks)
        for i in 0..m {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };

            let vars_in_row: Vec<usize> = (row_start..row_end)
                .filter_map(|idx| {
                    if idx < problem.col_indices.len() {
                        let v = problem.col_indices[idx];
                        if v < n {
                            Some(v)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            // Only connect within "small" constraints (heuristic for structure detection)
            if vars_in_row.len() <= n / 3 + 1 {
                for w in vars_in_row.windows(2) {
                    union(&mut parent, w[0], w[1]);
                }
            }
        }

        // Build blocks from components
        let mut component_map: indexmap::IndexMap<usize, Vec<usize>> = indexmap::IndexMap::new();
        for v in 0..n {
            let root = find(&mut parent, v);
            component_map.entry(root).or_default().push(v);
        }

        let num_blocks = component_map.len().max(1);
        let mut partition = vec![0usize; n];
        let mut block_variables = Vec::new();

        for (block_idx, (_root, vars)) in component_map.iter().enumerate() {
            for &v in vars {
                partition[v] = block_idx;
            }
            block_variables.push(vars.clone());
        }

        // Classify constraints
        let mut block_constraints: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        let mut linking_constraints = Vec::new();

        for i in 0..m {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };

            let mut blocks_touched = HashSet::new();
            for idx in row_start..row_end {
                if idx < problem.col_indices.len() {
                    let v = problem.col_indices[idx];
                    if v < n {
                        blocks_touched.insert(partition[v]);
                    }
                }
            }

            if blocks_touched.len() > 1 {
                linking_constraints.push(i);
            } else if let Some(&b) = blocks_touched.iter().next() {
                block_constraints[b].push(i);
            }
        }

        info!(
            "GCG: detected {} blocks, {} linking constraints",
            num_blocks,
            linking_constraints.len()
        );

        Ok(DecFormat {
            num_blocks,
            block_constraints,
            linking_constraints,
            block_variables,
        })
    }

    /// Emulate GCG Dantzig-Wolfe decomposition.
    pub fn emulate_dw(&mut self, problem: &LpProblem) -> OptResult<LpSolution> {
        info!("GCG: emulating Dantzig-Wolfe decomposition");

        let dec = if let Some(ref dec) = self.dec_format {
            dec.clone()
        } else {
            self.detect_structure(problem)?
        };

        // Build partition from dec
        let mut partition = vec![0usize; problem.num_vars];
        for (b, vars) in dec.block_variables.iter().enumerate() {
            for &v in vars {
                if v < partition.len() {
                    partition[v] = b;
                }
            }
        }

        let config = crate::dw::DWConfig::default();
        let mut dw = crate::dw::decomposition::DWDecomposition::new(problem, &partition, config)?;
        let result = dw.solve()?;

        self.stats.pricing_rounds = result.iterations;
        self.stats.columns_generated = result.num_columns_generated;

        Ok(LpSolution {
            status: match result.status {
                crate::dw::DWStatus::Optimal | crate::dw::DWStatus::GapClosed => {
                    SolverStatus::Optimal
                }
                crate::dw::DWStatus::Infeasible => SolverStatus::Infeasible,
                crate::dw::DWStatus::TimeLimit => SolverStatus::TimeLimit,
                crate::dw::DWStatus::IterationLimit => SolverStatus::IterationLimit,
                crate::dw::DWStatus::NumericalError => SolverStatus::NumericalError,
            },
            objective_value: result.lower_bound,
            primal_values: result.master_solution,
            dual_values: Vec::new(),
            reduced_costs: Vec::new(),
            basis_status: Vec::new(),
            iterations: result.iterations,
            time_seconds: result.time_seconds,
        })
    }

    /// Get solve statistics.
    pub fn get_stats(&self) -> &GcgStats {
        &self.stats
    }

    /// Internal LP solve.
    fn solve_internal(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let mut solver =
            crate::solver_interface::unified::UnifiedSolver::new(self.config.clone());
        solver.solve_lp(problem)
    }
}

impl SolverInterface for GcgAdapter {
    fn solve_lp(&mut self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        self.problem = Some(problem.clone());

        let result = self.solve_internal(problem)?;

        self.stats.solve_time = start.elapsed().as_secs_f64();
        self.stats.master_lp_iterations = result.iterations;

        self.status = result.status;
        self.last_solution = Some(result.clone());

        info!(
            "GCG: solved in {:.4}s, {} iterations",
            self.stats.solve_time, self.stats.master_lp_iterations
        );

        Ok(result)
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn get_dual_values(&self) -> OptResult<Vec<f64>> {
        self.last_solution
            .as_ref()
            .map(|s| s.dual_values.clone())
            .ok_or_else(|| OptError::solver("No solution available"))
    }

    fn get_basis(&self) -> OptResult<Vec<BasisStatus>> {
        self.last_solution
            .as_ref()
            .map(|s| s.basis_status.clone())
            .ok_or_else(|| OptError::solver("No solution available"))
    }

    fn add_constraint(
        &mut self,
        coeffs: &[(usize, f64)],
        ctype: ConstraintType,
        rhs: f64,
    ) -> OptResult<usize> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        let idx = problem.num_constraints;
        let (indices, vals): (Vec<usize>, Vec<f64>) = coeffs.iter().copied().unzip();
        problem.add_constraint(&indices, &vals, ctype, rhs)?;
        Ok(idx)
    }

    fn add_variable(&mut self, obj: f64, lb: f64, ub: f64) -> OptResult<usize> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        let idx = problem.num_vars;
        problem.add_variable(obj, lb, ub, None);
        Ok(idx)
    }

    fn set_objective(&mut self, coeffs: &[(usize, f64)]) -> OptResult<()> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        for &(i, val) in coeffs {
            if i < problem.obj_coeffs.len() {
                problem.obj_coeffs[i] = val;
            }
        }
        Ok(())
    }

    fn set_time_limit(&mut self, seconds: f64) {
        self.config.time_limit = seconds;
    }

    fn name(&self) -> &str {
        "GCG-Emulation"
    }

    fn reset(&mut self) {
        self.problem = None;
        self.last_solution = None;
        self.status = SolverStatus::NumericalError;
        self.stats = GcgStats::default();
        self.dec_format = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lp() -> LpProblem {
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 2.0, 3.0, 4.0];
        lp.lower_bounds = vec![0.0; 4];
        lp.upper_bounds = vec![10.0; 4];
        lp.row_starts = vec![0, 2, 4, 6];
        lp.col_indices = vec![0, 1, 2, 3, 0, 2];
        lp.values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le];
        lp.rhs = vec![5.0, 6.0, 8.0];
        lp.num_constraints = 3;
        lp
    }

    #[test]
    fn test_gcg_creation() {
        let config = SolverConfig::default().with_type(SolverType::GcgEmulation);
        let adapter = GcgAdapter::new(config);
        assert_eq!(adapter.name(), "GCG-Emulation");
    }

    #[test]
    fn test_gcg_solve() {
        let config = SolverConfig::default();
        let mut adapter = GcgAdapter::new(config);
        let lp = make_test_lp();
        let result = adapter.solve_lp(&lp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gcg_generate_dec() {
        let config = SolverConfig::default();
        let adapter = GcgAdapter::new(config);
        let lp = make_test_lp();
        let partition = vec![0, 0, 1, 1];
        let dec = adapter.generate_dec_format(&lp, &partition);
        assert!(dec.is_ok());
        let dec = dec.unwrap();
        assert_eq!(dec.num_blocks, 2);
    }

    #[test]
    fn test_gcg_format_dec_string() {
        let config = SolverConfig::default();
        let adapter = GcgAdapter::new(config);
        let dec = DecFormat {
            num_blocks: 2,
            block_constraints: vec![vec![0, 1], vec![2]],
            linking_constraints: vec![3],
            block_variables: vec![vec![0, 1], vec![2, 3]],
        };
        let s = adapter.format_dec_string(&dec);
        assert!(s.contains("NBLOCKS"));
        assert!(s.contains("BLOCK 1"));
        assert!(s.contains("MASTERCONSS"));
    }

    #[test]
    fn test_gcg_parse_dec_string() {
        let dec_str = "PRESOLVED\n0\nNBLOCKS\n2\nBLOCK 1\n  c0\n  c1\nBLOCK 2\n  c2\nMASTERCONSS\n  c3";
        let dec = GcgAdapter::parse_dec_string(dec_str);
        assert!(dec.is_ok());
        let dec = dec.unwrap();
        assert_eq!(dec.num_blocks, 2);
        assert_eq!(dec.block_constraints[0], vec![0, 1]);
        assert_eq!(dec.linking_constraints, vec![3]);
    }

    #[test]
    fn test_gcg_detect_structure() {
        let config = SolverConfig::default();
        let adapter = GcgAdapter::new(config);
        let lp = make_test_lp();
        let dec = adapter.detect_structure(&lp);
        assert!(dec.is_ok());
        let dec = dec.unwrap();
        assert!(dec.num_blocks >= 1);
    }

    #[test]
    fn test_gcg_set_decomposition() {
        let config = SolverConfig::default();
        let mut adapter = GcgAdapter::new(config);
        let dec = DecFormat {
            num_blocks: 2,
            block_constraints: vec![vec![0], vec![1]],
            linking_constraints: vec![2],
            block_variables: vec![vec![0, 1], vec![2, 3]],
        };
        adapter.set_decomposition(dec);
        assert!(adapter.dec_format.is_some());
    }

    #[test]
    fn test_gcg_stats() {
        let config = SolverConfig::default();
        let mut adapter = GcgAdapter::new(config);
        let lp = make_test_lp();
        adapter.solve_lp(&lp).unwrap();
        let stats = adapter.get_stats();
        assert!(stats.solve_time >= 0.0);
    }

    #[test]
    fn test_gcg_reset() {
        let config = SolverConfig::default();
        let mut adapter = GcgAdapter::new(config);
        let lp = make_test_lp();
        adapter.solve_lp(&lp).unwrap();
        adapter.reset();
        assert!(adapter.get_dual_values().is_err());
        assert!(adapter.dec_format.is_none());
    }

    #[test]
    fn test_gcg_invalid_partition() {
        let config = SolverConfig::default();
        let adapter = GcgAdapter::new(config);
        let lp = make_test_lp();
        let partition = vec![0, 1]; // wrong length
        assert!(adapter.generate_dec_format(&lp, &partition).is_err());
    }

    #[test]
    fn test_gcg_roundtrip_dec() {
        let config = SolverConfig::default();
        let adapter = GcgAdapter::new(config);
        let dec = DecFormat {
            num_blocks: 3,
            block_constraints: vec![vec![0], vec![1, 2], vec![3]],
            linking_constraints: vec![4, 5],
            block_variables: vec![],
        };
        let s = adapter.format_dec_string(&dec);
        let parsed = GcgAdapter::parse_dec_string(&s).unwrap();
        assert_eq!(parsed.num_blocks, 3);
        assert_eq!(parsed.linking_constraints, vec![4, 5]);
    }
}
