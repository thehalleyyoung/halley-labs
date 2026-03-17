//! Partition refinement algorithms: local search, Kernighan-Lin, simulated annealing, multi-start.

use crate::error::{OptError, OptResult};
use crate::partition::{AdjacencyInfo, Partition};
use log::{debug, info};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Configuration for partition refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementConfig {
    pub max_iterations: usize,
    pub improvement_threshold: f64,
    pub use_local_search: bool,
    pub use_kernighan_lin: bool,
    pub use_simulated_annealing: bool,
    pub sa_initial_temp: f64,
    pub sa_cooling_rate: f64,
    pub sa_iterations_per_temp: usize,
    pub multi_start_count: usize,
    pub seed: u64,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            improvement_threshold: 1e-6,
            use_local_search: true,
            use_kernighan_lin: true,
            use_simulated_annealing: false,
            sa_initial_temp: 1.0,
            sa_cooling_rate: 0.95,
            sa_iterations_per_temp: 100,
            multi_start_count: 1,
            seed: 42,
        }
    }
}

/// Result of a refinement run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementResult {
    pub refined_partition: Partition,
    pub initial_quality: f64,
    pub final_quality: f64,
    pub iterations: usize,
    pub history: Vec<f64>,
    pub method_used: String,
}

/// Partition refinement engine.
pub struct PartitionRefiner;

impl PartitionRefiner {
    /// Run the refinement pipeline on `partition` according to `config`.
    pub fn refine(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        config: &RefinementConfig,
    ) -> OptResult<RefinementResult> {
        if !partition.is_valid() {
            return Err(OptError::InvalidProblem {
                reason: "Input partition is invalid".into(),
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

        let mut current = partition.clone();
        let initial_quality = Self::compute_crossing_weight(&current, adjacency);
        let mut best_quality = initial_quality;
        let mut history = vec![initial_quality];
        let mut methods = Vec::new();
        let mut total_iterations = 0usize;

        for iter in 0..config.max_iterations {
            let prev_quality = best_quality;

            if config.use_local_search {
                let q = Self::local_search(&mut current, adjacency, config.max_iterations);
                if q < best_quality {
                    best_quality = q;
                }
            }

            if config.use_kernighan_lin && current.num_blocks() >= 2 {
                let q = Self::kernighan_lin(&mut current, adjacency, config.max_iterations);
                if q < best_quality {
                    best_quality = q;
                }
            }

            if config.use_simulated_annealing {
                let q = Self::simulated_annealing(&mut current, adjacency, config);
                if q < best_quality {
                    best_quality = q;
                }
            }

            history.push(best_quality);
            total_iterations = iter + 1;

            if (prev_quality - best_quality).abs() < config.improvement_threshold {
                debug!(
                    "Refinement converged at iteration {} (delta={:.2e})",
                    iter,
                    prev_quality - best_quality
                );
                break;
            }
        }

        if config.use_local_search {
            methods.push("local_search");
        }
        if config.use_kernighan_lin {
            methods.push("kernighan_lin");
        }
        if config.use_simulated_annealing {
            methods.push("simulated_annealing");
        }

        info!(
            "Refinement complete: {:.4} -> {:.4} in {} iterations",
            initial_quality, best_quality, total_iterations
        );

        Ok(RefinementResult {
            refined_partition: current,
            initial_quality,
            final_quality: best_quality,
            iterations: total_iterations,
            history,
            method_used: methods.join("+"),
        })
    }

    /// Greedy local search: for each element try all block moves; accept best improvement.
    pub fn local_search(
        partition: &mut Partition,
        adjacency: &AdjacencyInfo,
        max_iter: usize,
    ) -> f64 {
        let mut current_cw = Self::compute_crossing_weight(partition, adjacency);

        for _iter in 0..max_iter {
            let mut improved = false;

            for elem in 0..partition.num_elements {
                let from_block = match partition.block_of(elem) {
                    Some(b) => b,
                    None => continue,
                };

                let mut best_delta = 0.0f64;
                let mut best_to = from_block;

                for to_block in 0..partition.num_blocks() {
                    if to_block == from_block {
                        continue;
                    }
                    let delta = Self::compute_move_delta(partition, adjacency, elem, to_block);
                    if delta < best_delta {
                        best_delta = delta;
                        best_to = to_block;
                    }
                }

                if best_delta < -1e-12 {
                    partition.move_element(elem, from_block, best_to);
                    current_cw += best_delta;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        current_cw
    }

    /// Kernighan-Lin style swap optimization between pairs of blocks.
    pub fn kernighan_lin(
        partition: &mut Partition,
        adjacency: &AdjacencyInfo,
        max_iter: usize,
    ) -> f64 {
        let mut current_cw = Self::compute_crossing_weight(partition, adjacency);

        for _outer in 0..max_iter {
            let mut improved = false;
            let num_blocks = partition.num_blocks();

            for bi in 0..num_blocks {
                for bj in (bi + 1)..num_blocks {
                    let block_i: Vec<usize> = partition.blocks[bi].clone();
                    let block_j: Vec<usize> = partition.blocks[bj].clone();

                    if block_i.is_empty() || block_j.is_empty() {
                        continue;
                    }

                    // Find best single swap between these two blocks.
                    let mut best_gain = 0.0f64;
                    let mut best_swap: Option<(usize, usize)> = None;

                    for &ei in &block_i {
                        for &ej in &block_j {
                            let gain = Self::compute_swap_gain(partition, adjacency, ei, ej, bi, bj);
                            if gain > best_gain {
                                best_gain = gain;
                                best_swap = Some((ei, ej));
                            }
                        }
                    }

                    if let Some((ei, ej)) = best_swap {
                        if best_gain > 1e-12 {
                            // Build a sequence of swaps, tracking the best prefix.
                            partition.move_element(ei, bi, bj);
                            partition.move_element(ej, bj, bi);
                            current_cw -= best_gain;
                            improved = true;

                            // Try a second pass: continue swapping in this block pair.
                            let mut cumulative_gain = best_gain;
                            let mut locked_i = vec![ei];
                            let mut locked_j = vec![ej];

                            for _pass in 0..3 {
                                let cur_bi: Vec<usize> = partition.blocks[bi]
                                    .iter()
                                    .copied()
                                    .filter(|e| !locked_i.contains(e))
                                    .collect();
                                let cur_bj: Vec<usize> = partition.blocks[bj]
                                    .iter()
                                    .copied()
                                    .filter(|e| !locked_j.contains(e))
                                    .collect();

                                let mut pass_best_gain = 0.0f64;
                                let mut pass_best: Option<(usize, usize)> = None;

                                for &ei2 in &cur_bi {
                                    for &ej2 in &cur_bj {
                                        let g = Self::compute_swap_gain(
                                            partition, adjacency, ei2, ej2, bi, bj,
                                        );
                                        if g > pass_best_gain {
                                            pass_best_gain = g;
                                            pass_best = Some((ei2, ej2));
                                        }
                                    }
                                }

                                if let Some((ei2, ej2)) = pass_best {
                                    if pass_best_gain > 1e-12 {
                                        partition.move_element(ei2, bi, bj);
                                        partition.move_element(ej2, bj, bi);
                                        current_cw -= pass_best_gain;
                                        cumulative_gain += pass_best_gain;
                                        locked_i.push(ei2);
                                        locked_j.push(ej2);
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }

                            debug!(
                                "KL swap between blocks {} and {}: cumulative gain {:.6}",
                                bi, bj, cumulative_gain
                            );
                        }
                    }
                }
            }

            if !improved {
                break;
            }
        }

        current_cw
    }

    /// Simulated annealing: random moves with Boltzmann acceptance.
    pub fn simulated_annealing(
        partition: &mut Partition,
        adjacency: &AdjacencyInfo,
        config: &RefinementConfig,
    ) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut current_cw = Self::compute_crossing_weight(partition, adjacency);
        let mut best_cw = current_cw;
        let mut best_partition = partition.clone();
        let mut temp = config.sa_initial_temp;
        let num_blocks = partition.num_blocks();

        if num_blocks < 2 || partition.num_elements == 0 {
            return current_cw;
        }

        let max_temp_steps = ((config.sa_initial_temp / 0.001f64).ln()
            / (1.0 / config.sa_cooling_rate).ln())
        .ceil() as usize
            + 1;

        for _temp_step in 0..max_temp_steps {
            if temp < 1e-10 {
                break;
            }

            for _ in 0..config.sa_iterations_per_temp {
                let elem = rng.gen_range(0..partition.num_elements);
                let from_block = match partition.block_of(elem) {
                    Some(b) => b,
                    None => continue,
                };

                // Pick a random different block.
                let to_block = loop {
                    let b = rng.gen_range(0..num_blocks);
                    if b != from_block {
                        break b;
                    }
                };

                let delta = Self::compute_move_delta(partition, adjacency, elem, to_block);

                let accept = if delta < 0.0 {
                    true
                } else if temp > 1e-15 {
                    let prob = (-delta / temp).exp();
                    rng.gen::<f64>() < prob
                } else {
                    false
                };

                if accept {
                    partition.move_element(elem, from_block, to_block);
                    current_cw += delta;
                    if current_cw < best_cw {
                        best_cw = current_cw;
                        best_partition = partition.clone();
                    }
                }
            }

            temp *= config.sa_cooling_rate;
        }

        // Restore best found.
        *partition = best_partition;
        best_cw
    }

    /// Generate multiple random initial partitions, refine each, return the best.
    pub fn multi_start_refine(
        num_elements: usize,
        num_blocks: usize,
        adjacency: &AdjacencyInfo,
        config: &RefinementConfig,
    ) -> OptResult<RefinementResult> {
        if num_elements == 0 {
            return Err(OptError::InvalidProblem {
                reason: "Cannot partition 0 elements".into(),
            });
        }
        if num_blocks == 0 || num_blocks > num_elements {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "Invalid num_blocks={} for num_elements={}",
                    num_blocks, num_elements
                ),
            });
        }

        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut best_result: Option<RefinementResult> = None;

        for start in 0..config.multi_start_count.max(1) {
            // Generate random initial partition.
            let mut blocks: Vec<Vec<usize>> = (0..num_blocks).map(|_| Vec::new()).collect();
            for e in 0..num_elements {
                let b = rng.gen_range(0..num_blocks);
                blocks[b].push(e);
            }
            // Ensure all blocks non-empty: move one element from the largest block.
            for b in 0..num_blocks {
                if blocks[b].is_empty() {
                    let largest = blocks
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, bl)| bl.len())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if let Some(e) = blocks[largest].pop() {
                        blocks[b].push(e);
                    }
                }
            }

            let partition = Partition {
                blocks,
                num_elements,
            };

            let result = Self::refine(&partition, adjacency, config)?;
            debug!(
                "Multi-start {}: quality {:.6} -> {:.6}",
                start, result.initial_quality, result.final_quality
            );

            let replace = match &best_result {
                None => true,
                Some(prev) => result.final_quality < prev.final_quality,
            };
            if replace {
                best_result = Some(result);
            }
        }

        best_result.ok_or_else(|| OptError::SolverError {
            message: "Multi-start produced no results".into(),
        })
    }

    /// Crossing weight: sum of edge weights where endpoints are in different blocks.
    pub fn compute_crossing_weight(partition: &Partition, adjacency: &AdjacencyInfo) -> f64 {
        let mut block_map = vec![0usize; partition.num_elements];
        for (bi, block) in partition.blocks.iter().enumerate() {
            for &e in block {
                block_map[e] = bi;
            }
        }
        let mut total = 0.0;
        for &(u, v, w) in &adjacency.edges {
            if u < partition.num_elements && v < partition.num_elements && block_map[u] != block_map[v]
            {
                total += w;
            }
        }
        total
    }

    /// Delta in crossing weight from moving `element` to `to_block`.
    /// Negative means improvement (less crossing).
    pub fn compute_move_delta(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        element: usize,
        to_block: usize,
    ) -> f64 {
        let from_block = match partition.block_of(element) {
            Some(b) => b,
            None => return 0.0,
        };
        if from_block == to_block {
            return 0.0;
        }

        let mut delta = 0.0;
        for &(nbr, weight) in adjacency.neighbors(element) {
            let nbr_block = match partition.block_of(nbr) {
                Some(b) => b,
                None => continue,
            };
            if nbr_block == from_block {
                // Was internal, becomes crossing.
                delta += weight;
            } else if nbr_block == to_block {
                // Was crossing, becomes internal.
                delta -= weight;
            }
            // Neighbor in another block: stays crossing either way.
        }
        delta
    }

    /// Gain from swapping `ei` (in `bi`) with `ej` (in `bj`).
    /// Positive gain means crossing weight decreases.
    fn compute_swap_gain(
        partition: &Partition,
        adjacency: &AdjacencyInfo,
        ei: usize,
        ej: usize,
        bi: usize,
        bj: usize,
    ) -> f64 {
        // external_i = sum of weights of ei's edges to vertices NOT in bi
        // internal_i = sum of weights of ei's edges to vertices in bi (excluding ej)
        let mut ext_i = 0.0;
        let mut int_i = 0.0;
        let mut edge_ij = 0.0;

        for &(nbr, w) in adjacency.neighbors(ei) {
            if nbr == ej {
                edge_ij += w;
            } else {
                let nb = partition.block_of(nbr).unwrap_or(usize::MAX);
                if nb == bi {
                    int_i += w;
                } else {
                    ext_i += w;
                }
            }
        }

        let mut ext_j = 0.0;
        let mut int_j = 0.0;

        for &(nbr, w) in adjacency.neighbors(ej) {
            if nbr == ei {
                // Already counted.
            } else {
                let nb = partition.block_of(nbr).unwrap_or(usize::MAX);
                if nb == bj {
                    int_j += w;
                } else {
                    ext_j += w;
                }
            }
        }

        // After swap: ei goes to bj, ej goes to bi.
        // The gain is: (ext_i - int_i) + (ext_j - int_j) - 2 * edge_ij
        // This is the classic KL gain formula.
        (ext_i - int_i) + (ext_j - int_j) - 2.0 * edge_ij
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> (Partition, AdjacencyInfo) {
        // 4-vertex path: 0-1-2-3
        let adj = AdjacencyInfo::from_edges(
            4,
            vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
        );
        let p = Partition::from_blocks(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        (p, adj)
    }

    fn triangle_graph() -> (Partition, AdjacencyInfo) {
        let adj = AdjacencyInfo::from_edges(
            3,
            vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
        );
        let p = Partition::from_blocks(vec![vec![0], vec![1], vec![2]], 3).unwrap();
        (p, adj)
    }

    #[test]
    fn test_compute_crossing_weight_path() {
        let (p, adj) = simple_graph();
        // Edge 1-2 crosses: weight = 1.0
        let cw = PartitionRefiner::compute_crossing_weight(&p, &adj);
        assert!((cw - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_crossing_weight_triangle() {
        let (p, adj) = triangle_graph();
        // All edges cross: 3.0
        let cw = PartitionRefiner::compute_crossing_weight(&p, &adj);
        assert!((cw - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_move_delta() {
        let (p, adj) = simple_graph();
        // Move element 2 from block 1 to block 0.
        // 2's neighbors: 1 (block 0) and 3 (block 1).
        // nbr 1 in from_block? No (from_block=1, nbr_block=0) -> 0
        // nbr 1 in to_block? Yes (to_block=0, nbr_block=0) -> -1.0
        // nbr 3 in from_block? Yes (from_block=1) -> +1.0
        // nbr 3 in to_block? No -> 0
        // delta = 0.0
        let delta = PartitionRefiner::compute_move_delta(&p, &adj, 2, 0);
        assert!((delta - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_search_no_change_on_optimal() {
        // Disconnected graph: no edges. Already optimal (crossing = 0).
        let adj = AdjacencyInfo::new(4);
        let mut p = Partition::from_blocks(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        let cw = PartitionRefiner::local_search(&mut p, &adj, 10);
        assert!((cw - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_local_search_improves() {
        // 0-1 (w=10), 2-3 (w=10), 0-2 (w=1)
        // Start: {0,2}, {1,3} -> crossing: 0-1(10) + 2-3(10) = 20
        // Optimal: {0,1}, {2,3} -> crossing: 0-2(1) = 1
        let adj = AdjacencyInfo::from_edges(
            4,
            vec![(0, 1, 10.0), (2, 3, 10.0), (0, 2, 1.0)],
        );
        let mut p = Partition::from_blocks(vec![vec![0, 2], vec![1, 3]], 4).unwrap();
        let cw = PartitionRefiner::local_search(&mut p, &adj, 20);
        assert!(cw <= 1.0 + 1e-9);
    }

    #[test]
    fn test_kernighan_lin_basic() {
        let adj = AdjacencyInfo::from_edges(
            4,
            vec![(0, 1, 10.0), (2, 3, 10.0), (0, 2, 1.0)],
        );
        let mut p = Partition::from_blocks(vec![vec![0, 2], vec![1, 3]], 4).unwrap();
        let cw = PartitionRefiner::kernighan_lin(&mut p, &adj, 10);
        assert!(cw <= 1.0 + 1e-9);
    }

    #[test]
    fn test_simulated_annealing_basic() {
        let adj = AdjacencyInfo::from_edges(
            4,
            vec![(0, 1, 10.0), (2, 3, 10.0), (1, 3, 0.1)],
        );
        let mut p = Partition::from_blocks(vec![vec![0, 2], vec![1, 3]], 4).unwrap();
        let config = RefinementConfig {
            sa_initial_temp: 10.0,
            sa_cooling_rate: 0.8,
            sa_iterations_per_temp: 200,
            seed: 123,
            ..Default::default()
        };
        let cw = PartitionRefiner::simulated_annealing(&mut p, &adj, &config);
        // Should find something better than initial 20.0.
        assert!(cw < 20.0 + 1e-9);
    }

    #[test]
    fn test_refine_pipeline() {
        let adj = AdjacencyInfo::from_edges(
            6,
            vec![
                (0, 1, 5.0),
                (1, 2, 5.0),
                (3, 4, 5.0),
                (4, 5, 5.0),
                (2, 3, 1.0),
            ],
        );
        let p = Partition::from_blocks(vec![vec![0, 1, 2], vec![3, 4, 5]], 6).unwrap();
        let config = RefinementConfig::default();
        let result = PartitionRefiner::refine(&p, &adj, &config).unwrap();
        assert!(result.final_quality <= result.initial_quality + 1e-9);
    }

    #[test]
    fn test_multi_start() {
        let adj = AdjacencyInfo::from_edges(
            6,
            vec![
                (0, 1, 5.0),
                (1, 2, 5.0),
                (3, 4, 5.0),
                (4, 5, 5.0),
                (2, 3, 0.5),
            ],
        );
        let config = RefinementConfig {
            multi_start_count: 5,
            max_iterations: 5,
            ..Default::default()
        };
        let result =
            PartitionRefiner::multi_start_refine(6, 2, &adj, &config).unwrap();
        assert!(result.final_quality <= result.initial_quality + 1e-9);
        assert!(result.refined_partition.is_valid());
    }

    #[test]
    fn test_refine_invalid_partition() {
        let adj = AdjacencyInfo::new(3);
        let p = Partition {
            blocks: vec![vec![0, 1, 1]],
            num_elements: 3,
        };
        let config = RefinementConfig::default();
        let result = PartitionRefiner::refine(&p, &adj, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_refine_mismatched_sizes() {
        let adj = AdjacencyInfo::new(5);
        let p = Partition::from_blocks(vec![vec![0, 1, 2]], 3).unwrap();
        let config = RefinementConfig::default();
        let result = PartitionRefiner::refine(&p, &adj, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_start_invalid_num_blocks() {
        let adj = AdjacencyInfo::new(4);
        let config = RefinementConfig::default();
        let result = PartitionRefiner::multi_start_refine(4, 0, &adj, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_crossing_weight_single_block() {
        let adj = AdjacencyInfo::from_edges(3, vec![(0, 1, 2.0), (1, 2, 3.0)]);
        let p = Partition::new(3); // single block
        let cw = PartitionRefiner::compute_crossing_weight(&p, &adj);
        assert!((cw - 0.0).abs() < 1e-9);
    }
}
