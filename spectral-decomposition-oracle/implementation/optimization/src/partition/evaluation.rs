//! Partition evaluation: quality metrics, L3 bound estimates, Pareto analysis.

use crate::error::{OptError, OptResult};
use crate::partition::{AdjacencyInfo, Partition};
use log::debug;
use serde::{Deserialize, Serialize};

/// Configuration for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub use_dual_weights: bool,
    pub use_reduced_costs: bool,
    pub num_objectives: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            use_dual_weights: false,
            use_reduced_costs: false,
            num_objectives: 3,
        }
    }
}

/// Quality metrics for a single block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockQuality {
    pub block_id: usize,
    pub size: usize,
    pub internal_weight: f64,
    pub external_weight: f64,
    pub density: f64,
}

/// Full evaluation of a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionEvaluation {
    pub crossing_weight: f64,
    pub dual_weighted_crossing: f64,
    pub balance_ratio: f64,
    pub num_blocks: usize,
    pub total_crossing_edges: usize,
    pub l3_bound_estimate: f64,
    pub pareto_rank: usize,
}

/// Partition evaluator engine.
pub struct PartitionEvaluator;

impl PartitionEvaluator {
    /// Full evaluation of a partition.
    pub fn evaluate(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        dual_weights: Option<&[f64]>,
    ) -> OptResult<PartitionEvaluation> {
        if !partition.is_valid() {
            return Err(OptError::InvalidProblem {
                reason: "Partition is invalid".into(),
            });
        }
        if partition.num_elements != adjacency.num_vertices {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "Partition has {} elements but adjacency has {} vertices",
                    partition.num_elements, adjacency.num_vertices
                ),
            });
        }

        let cw = Self::crossing_weight(partition, adjacency);
        let dwc = match dual_weights {
            Some(dw) => Self::dual_weighted_crossing(partition, adjacency, dw),
            None => 0.0,
        };
        let br = partition.balance_ratio();
        let nb = partition.num_blocks();
        let tce = Self::total_crossing_edges(partition, adjacency);
        let l3e = Self::l3_bound_estimate(partition, adjacency, dual_weights);

        debug!(
            "Evaluation: cw={:.4}, dwc={:.4}, br={:.4}, blocks={}, crossing_edges={}, l3={:.4}",
            cw, dwc, br, nb, tce, l3e
        );

        Ok(PartitionEvaluation {
            crossing_weight: cw,
            dual_weighted_crossing: dwc,
            balance_ratio: br,
            num_blocks: nb,
            total_crossing_edges: tce,
            l3_bound_estimate: l3e,
            pareto_rank: 0,
        })
    }

    /// Sum of edge weights where endpoints are in different blocks.
    pub fn crossing_weight(partition: &Partition, adjacency: &AdjacencyInfo) -> f64 {
        let block_map = Self::build_block_map(partition);
        let mut total = 0.0;
        for &(u, v, w) in &adjacency.edges {
            if u < block_map.len() && v < block_map.len() && block_map[u] != block_map[v] {
                total += w;
            }
        }
        total
    }

    /// Dual-weighted crossing: sum of |y*_e| for crossing edges.
    pub fn dual_weighted_crossing(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        dual_weights: &[f64],
    ) -> f64 {
        let block_map = Self::build_block_map(partition);
        let mut total = 0.0;
        for (idx, &(u, v, _w)) in adjacency.edges.iter().enumerate() {
            if u < block_map.len() && v < block_map.len() && block_map[u] != block_map[v] {
                let dw = if idx < dual_weights.len() {
                    dual_weights[idx].abs()
                } else {
                    0.0
                };
                total += dw;
            }
        }
        total
    }

    /// Estimate L3 bound quality. Lower crossing_weight / total_weight → tighter bound.
    pub fn l3_bound_estimate(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        dual_weights: Option<&[f64]>,
    ) -> f64 {
        let total_w = adjacency.total_weight();
        if total_w < 1e-15 {
            return 0.0;
        }

        let cw = Self::crossing_weight(partition, adjacency);
        let base_ratio = cw / total_w;

        // Incorporate dual weights if available: the dual-weighted crossing
        // gives a more informative bound estimate.
        let dual_contribution = match dual_weights {
            Some(dw) => {
                let dwc = Self::dual_weighted_crossing(partition, adjacency, dw);
                let dw_sum: f64 = dw.iter().map(|d| d.abs()).sum();
                if dw_sum > 1e-15 {
                    dwc / dw_sum
                } else {
                    0.0
                }
            }
            None => 0.0,
        };

        // Blend: use dual contribution if available, otherwise just the base ratio.
        if dual_contribution > 0.0 {
            0.5 * base_ratio + 0.5 * dual_contribution
        } else {
            base_ratio
        }
    }

    /// Compare two partition evaluations by l3_bound_estimate (lower is better).
    pub fn compare_partitions(
        a: &PartitionEvaluation,
        b: &PartitionEvaluation,
    ) -> std::cmp::Ordering {
        a.l3_bound_estimate
            .partial_cmp(&b.l3_bound_estimate)
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Find Pareto-optimal partitions considering (crossing_weight, 1-balance_ratio, num_blocks).
    /// Returns indices of non-dominated evaluations.
    pub fn pareto_frontier(evaluations: &[PartitionEvaluation]) -> Vec<usize> {
        if evaluations.is_empty() {
            return Vec::new();
        }

        let objectives: Vec<[f64; 3]> = evaluations
            .iter()
            .map(|e| {
                [
                    e.crossing_weight,
                    1.0 - e.balance_ratio, // lower is better
                    e.num_blocks as f64,    // fewer blocks can be better
                ]
            })
            .collect();

        let n = objectives.len();
        let mut dominated = vec![false; n];

        for i in 0..n {
            if dominated[i] {
                continue;
            }
            for j in 0..n {
                if i == j || dominated[j] {
                    continue;
                }
                if Self::dominates(&objectives[j], &objectives[i]) {
                    dominated[i] = true;
                    break;
                }
            }
        }

        let frontier: Vec<usize> = (0..n).filter(|&i| !dominated[i]).collect();
        debug!("Pareto frontier: {} of {} partitions", frontier.len(), n);
        frontier
    }

    /// L3-C Benders specialization: weight crossings by reduced costs.
    pub fn benders_specialization(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        reduced_costs: &[f64],
    ) -> f64 {
        let block_map = Self::build_block_map(partition);
        let mut total = 0.0;
        for (idx, &(u, v, _w)) in adjacency.edges.iter().enumerate() {
            if u < block_map.len() && v < block_map.len() && block_map[u] != block_map[v] {
                let rc = if idx < reduced_costs.len() {
                    reduced_costs[idx].abs()
                } else {
                    0.0
                };
                total += rc;
            }
        }
        total
    }

    /// L3-C Dantzig-Wolfe specialization: weight crossings by linking constraint duals.
    pub fn dw_specialization(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        linking_duals: &[f64],
    ) -> f64 {
        let block_map = Self::build_block_map(partition);
        let mut total = 0.0;
        for (idx, &(u, v, _w)) in adjacency.edges.iter().enumerate() {
            if u < block_map.len() && v < block_map.len() && block_map[u] != block_map[v] {
                let ld = if idx < linking_duals.len() {
                    linking_duals[idx].abs()
                } else {
                    0.0
                };
                total += ld;
            }
        }
        total
    }

    /// Per-block quality breakdown.
    pub fn block_quality_breakdown(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
    ) -> Vec<BlockQuality> {
        let block_map = Self::build_block_map(partition);
        let num_blocks = partition.num_blocks();
        let mut internal_weights = vec![0.0f64; num_blocks];
        let mut external_weights = vec![0.0f64; num_blocks];
        let mut internal_edge_counts = vec![0usize; num_blocks];

        for &(u, v, w) in &adjacency.edges {
            if u >= block_map.len() || v >= block_map.len() {
                continue;
            }
            let bu = block_map[u];
            let bv = block_map[v];
            if bu == bv {
                internal_weights[bu] += w;
                internal_edge_counts[bu] += 1;
            } else {
                external_weights[bu] += w;
                external_weights[bv] += w;
            }
        }

        partition
            .blocks
            .iter()
            .enumerate()
            .map(|(bi, block)| {
                let size = block.len();
                let max_edges = if size > 1 {
                    size * (size - 1) / 2
                } else {
                    1
                };
                let density = internal_edge_counts[bi] as f64 / max_edges as f64;
                BlockQuality {
                    block_id: bi,
                    size,
                    internal_weight: internal_weights[bi],
                    external_weight: external_weights[bi],
                    density: density.min(1.0),
                }
            })
            .collect()
    }

    // --- helpers ---

    fn build_block_map(partition: &Partition) -> Vec<usize> {
        let mut map = vec![0usize; partition.num_elements];
        for (bi, block) in partition.blocks.iter().enumerate() {
            for &e in block {
                if e < map.len() {
                    map[e] = bi;
                }
            }
        }
        map
    }

    fn total_crossing_edges(partition: &Partition, adjacency: &AdjacencyInfo) -> usize {
        let block_map = Self::build_block_map(partition);
        adjacency
            .edges
            .iter()
            .filter(|&&(u, v, _)| {
                u < block_map.len() && v < block_map.len() && block_map[u] != block_map[v]
            })
            .count()
    }

    /// Check if objective vector `a` dominates `b` (all objectives ≤ and at least one <).
    fn dominates(a: &[f64; 3], b: &[f64; 3]) -> bool {
        let all_leq = a.iter().zip(b.iter()).all(|(ai, bi)| *ai <= *bi + 1e-12);
        let any_lt = a.iter().zip(b.iter()).any(|(ai, bi)| *ai < *bi - 1e-12);
        all_leq && any_lt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_graph() -> (Partition, AdjacencyInfo) {
        let adj = AdjacencyInfo::from_edges(
            6,
            vec![
                (0, 1, 5.0),
                (1, 2, 5.0),
                (0, 2, 5.0),
                (3, 4, 5.0),
                (4, 5, 5.0),
                (3, 5, 5.0),
                (2, 3, 1.0),
            ],
        );
        let p = Partition::from_blocks(vec![vec![0, 1, 2], vec![3, 4, 5]], 6).unwrap();
        (p, adj)
    }

    #[test]
    fn test_crossing_weight() {
        let (p, adj) = two_cluster_graph();
        let cw = PartitionEvaluator::crossing_weight(&p, &adj);
        assert!((cw - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_crossing_weight_single_block() {
        let adj = AdjacencyInfo::from_edges(3, vec![(0, 1, 2.0), (1, 2, 3.0)]);
        let p = Partition::new(3);
        let cw = PartitionEvaluator::crossing_weight(&p, &adj);
        assert!((cw - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dual_weighted_crossing() {
        let (p, adj) = two_cluster_graph();
        // 7 edges; edge 6 is (2,3) which crosses.
        let dual_weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let dwc = PartitionEvaluator::dual_weighted_crossing(&p, &adj, &dual_weights);
        // Only edge 6 crosses; dual weight = 0.7
        assert!((dwc - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_l3_bound_estimate_no_duals() {
        let (p, adj) = two_cluster_graph();
        let l3 = PartitionEvaluator::l3_bound_estimate(&p, &adj, None);
        let total_w = adj.total_weight(); // 31.0
        let cw = 1.0;
        assert!((l3 - cw / total_w).abs() < 1e-9);
    }

    #[test]
    fn test_l3_bound_estimate_with_duals() {
        let (p, adj) = two_cluster_graph();
        let duals = vec![1.0; 7];
        let l3 = PartitionEvaluator::l3_bound_estimate(&p, &adj, Some(&duals));
        // base_ratio = 1.0/31.0, dual_contribution = 1.0/7.0
        let expected = 0.5 * (1.0 / 31.0) + 0.5 * (1.0 / 7.0);
        assert!((l3 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_full() {
        let (p, adj) = two_cluster_graph();
        let eval = PartitionEvaluator::evaluate(&p, &adj, None).unwrap();
        assert!((eval.crossing_weight - 1.0).abs() < 1e-9);
        assert_eq!(eval.num_blocks, 2);
        assert!((eval.balance_ratio - 1.0).abs() < 1e-9);
        assert_eq!(eval.total_crossing_edges, 1);
    }

    #[test]
    fn test_evaluate_invalid_partition() {
        let adj = AdjacencyInfo::new(3);
        let p = Partition {
            blocks: vec![vec![0, 0]],
            num_elements: 3,
        };
        let result = PartitionEvaluator::evaluate(&p, &adj, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_partitions() {
        let a = PartitionEvaluation {
            crossing_weight: 1.0,
            dual_weighted_crossing: 0.0,
            balance_ratio: 1.0,
            num_blocks: 2,
            total_crossing_edges: 1,
            l3_bound_estimate: 0.1,
            pareto_rank: 0,
        };
        let b = PartitionEvaluation {
            l3_bound_estimate: 0.5,
            ..a.clone()
        };
        assert_eq!(
            PartitionEvaluator::compare_partitions(&a, &b),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_pareto_frontier() {
        let evals = vec![
            PartitionEvaluation {
                crossing_weight: 1.0,
                dual_weighted_crossing: 0.0,
                balance_ratio: 0.5,
                num_blocks: 2,
                total_crossing_edges: 1,
                l3_bound_estimate: 0.1,
                pareto_rank: 0,
            },
            PartitionEvaluation {
                crossing_weight: 2.0,
                dual_weighted_crossing: 0.0,
                balance_ratio: 1.0,
                num_blocks: 2,
                total_crossing_edges: 2,
                l3_bound_estimate: 0.2,
                pareto_rank: 0,
            },
            PartitionEvaluation {
                crossing_weight: 3.0,
                dual_weighted_crossing: 0.0,
                balance_ratio: 0.3,
                num_blocks: 3,
                total_crossing_edges: 3,
                l3_bound_estimate: 0.3,
                pareto_rank: 0,
            },
        ];
        let frontier = PartitionEvaluator::pareto_frontier(&evals);
        // eval[0] dominates eval[2] (1<3, 0.5>0.3 → 1-0.5=0.5<1-0.3=0.7, 2<3)
        // eval[0] and eval[1] are non-dominated (better cw but worse balance)
        assert!(frontier.contains(&0));
        assert!(frontier.contains(&1));
    }

    #[test]
    fn test_benders_specialization() {
        let (p, adj) = two_cluster_graph();
        let reduced_costs = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0];
        let val = PartitionEvaluator::benders_specialization(&p, &adj, &reduced_costs);
        // Only edge 6 crosses, RC=3.0
        assert!((val - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_dw_specialization() {
        let (p, adj) = two_cluster_graph();
        let linking = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5];
        let val = PartitionEvaluator::dw_specialization(&p, &adj, &linking);
        assert!((val - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_block_quality_breakdown() {
        let (p, adj) = two_cluster_graph();
        let bq = PartitionEvaluator::block_quality_breakdown(&p, &adj);
        assert_eq!(bq.len(), 2);
        // Block 0 has edges (0,1), (1,2), (0,2) = 15.0 internal
        assert!((bq[0].internal_weight - 15.0).abs() < 1e-9);
        assert_eq!(bq[0].size, 3);
        // Block 0 external: edge (2,3) contributes 1.0
        assert!((bq[0].external_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_block_quality_density() {
        // Triangle in block 0: 3 edges / C(3,2) = 3/3 = 1.0
        let (p, adj) = two_cluster_graph();
        let bq = PartitionEvaluator::block_quality_breakdown(&p, &adj);
        assert!((bq[0].density - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pareto_frontier_empty() {
        let frontier = PartitionEvaluator::pareto_frontier(&[]);
        assert!(frontier.is_empty());
    }

    #[test]
    fn test_pareto_frontier_single() {
        let evals = vec![PartitionEvaluation {
            crossing_weight: 1.0,
            dual_weighted_crossing: 0.0,
            balance_ratio: 1.0,
            num_blocks: 2,
            total_crossing_edges: 1,
            l3_bound_estimate: 0.1,
            pareto_rank: 0,
        }];
        let frontier = PartitionEvaluator::pareto_frontier(&evals);
        assert_eq!(frontier, vec![0]);
    }
}
