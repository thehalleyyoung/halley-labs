//! Greedy partition construction algorithms.

use crate::error::{OptError, OptResult};
use crate::partition::{AdjacencyInfo, Partition};
use log::{debug, info};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Variable ordering heuristics for greedy assignment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VariableOrdering {
    /// Assign highest-degree vertices first.
    HighestDegree,
    /// Assign most constrained (most already-assigned neighbors) first.
    MostConstrained,
    /// Random permutation.
    Random,
    /// Natural order 0, 1, 2, ...
    Natural,
}

/// Configuration for greedy partition construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreedyConfig {
    pub num_blocks: usize,
    pub ordering: VariableOrdering,
    pub balance_tolerance: f64,
    pub max_block_capacity: Option<usize>,
    pub num_heuristics: usize,
    pub seed: u64,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            num_blocks: 2,
            ordering: VariableOrdering::HighestDegree,
            balance_tolerance: 0.0,
            max_block_capacity: None,
            num_heuristics: 3,
            seed: 42,
        }
    }
}

/// Greedy partition builder.
pub struct GreedyBuilder;

impl GreedyBuilder {
    /// Main entry: run multiple heuristics and return the best partition.
    pub fn build(
        num_elements: usize,
        adjacency: &AdjacencyInfo,
        config: &GreedyConfig,
    ) -> OptResult<Partition> {
        if num_elements == 0 {
            return Err(OptError::InvalidProblem {
                reason: "Cannot partition 0 elements".into(),
            });
        }
        if config.num_blocks == 0 || config.num_blocks > num_elements {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "Invalid num_blocks={} for num_elements={}",
                    config.num_blocks, num_elements
                ),
            });
        }
        if adjacency.num_vertices != num_elements {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "Adjacency has {} vertices but num_elements={}",
                    adjacency.num_vertices, num_elements
                ),
            });
        }

        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let orderings = Self::heuristic_orderings(config);

        let mut best_partition: Option<Partition> = None;
        let mut best_cw = f64::INFINITY;

        for (idx, ordering) in orderings.iter().enumerate() {
            let result = Self::build_single(
                num_elements,
                adjacency,
                ordering,
                config.num_blocks,
                config.balance_tolerance,
                config.max_block_capacity,
                &mut rng,
            );

            match result {
                Ok(p) => {
                    let cw = Self::compute_crossing_weight(&p, adjacency);
                    debug!("Heuristic {} ({:?}): crossing_weight = {:.4}", idx, ordering, cw);
                    if cw < best_cw {
                        best_cw = cw;
                        best_partition = Some(p);
                    }
                }
                Err(e) => {
                    debug!("Heuristic {} ({:?}) failed: {}", idx, ordering, e);
                }
            }
        }

        match best_partition {
            Some(p) => {
                info!(
                    "Greedy build: best crossing_weight = {:.4} with {} blocks",
                    best_cw,
                    p.num_blocks()
                );
                Ok(p)
            }
            None => Err(OptError::SolverError {
                message: "All greedy heuristics failed".into(),
            }),
        }
    }

    /// Single greedy construction pass.
    pub fn build_single(
        num_elements: usize,
        adjacency: &AdjacencyInfo,
        ordering: &VariableOrdering,
        num_blocks: usize,
        balance_tol: f64,
        max_cap: Option<usize>,
        rng: &mut impl Rng,
    ) -> OptResult<Partition> {
        let order = Self::variable_ordering(num_elements, adjacency, ordering, rng);
        let mut blocks: Vec<Vec<usize>> = (0..num_blocks).map(|_| Vec::new()).collect();
        let mut assignment = vec![usize::MAX; num_elements];

        for &elem in &order {
            let best_block = Self::select_best_block_with_assignment(
                elem,
                adjacency,
                &assignment,
                &blocks,
                num_blocks,
                balance_tol,
                max_cap,
                num_elements,
            );
            blocks[best_block].push(elem);
            assignment[elem] = best_block;
        }

        // Ensure every block is non-empty by redistributing from largest blocks.
        for b in 0..num_blocks {
            if blocks[b].is_empty() {
                let largest = blocks
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != b)
                    .max_by_key(|(_, bl)| bl.len())
                    .map(|(i, _)| i);
                if let Some(li) = largest {
                    if blocks[li].len() > 1 {
                        let e = blocks[li].pop().unwrap();
                        assignment[e] = b;
                        blocks[b].push(e);
                    }
                }
            }
        }

        Ok(Partition {
            blocks,
            num_elements,
        })
    }

    /// Determine the order in which to assign variables.
    pub fn variable_ordering(
        num_elements: usize,
        adjacency: &AdjacencyInfo,
        ordering: &VariableOrdering,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let mut order: Vec<usize> = (0..num_elements).collect();
        match ordering {
            VariableOrdering::HighestDegree => {
                order.sort_by(|&a, &b| adjacency.degree(b).cmp(&adjacency.degree(a)));
            }
            VariableOrdering::MostConstrained => {
                // Sort by degree as a proxy; in the actual assignment loop the
                // "most constrained" aspect is handled dynamically, but for
                // ordering we use degree as a tie-breaker heuristic.
                order.sort_by(|&a, &b| {
                    let da = adjacency.degree(a);
                    let db = adjacency.degree(b);
                    db.cmp(&da)
                });
            }
            VariableOrdering::Random => {
                order.shuffle(rng);
            }
            VariableOrdering::Natural => {
                // Already in natural order.
            }
        }
        order
    }

    /// For a given element, pick the block that minimizes crossing weight
    /// increase while respecting balance constraints.
    pub fn select_best_block(
        element: usize,
        adjacency: &AdjacencyInfo,
        current_partition: &Partition,
        num_blocks: usize,
        balance_tol: f64,
        max_cap: Option<usize>,
    ) -> usize {
        let mut best_block = 0;
        let mut best_score = f64::INFINITY;

        let total_assigned: usize = current_partition.blocks.iter().map(|b| b.len()).sum();
        let ideal_size = if num_blocks > 0 {
            (total_assigned + 1) as f64 / num_blocks as f64
        } else {
            f64::INFINITY
        };

        for b in 0..num_blocks {
            let block_size = current_partition.blocks[b].len();

            // Check capacity constraint.
            if let Some(cap) = max_cap {
                if block_size >= cap {
                    continue;
                }
            }

            // Check balance constraint.
            if balance_tol < 1.0 {
                let max_allowed = (ideal_size * (1.0 + balance_tol)).ceil() as usize;
                if block_size >= max_allowed && max_allowed > 0 {
                    continue;
                }
            }

            let cw = Self::crossing_weight_if_assigned(element, b, adjacency, current_partition);
            if cw < best_score {
                best_score = cw;
                best_block = b;
            }
        }

        best_block
    }

    /// Internal selection using the assignment vector for efficiency.
    fn select_best_block_with_assignment(
        element: usize,
        adjacency: &AdjacencyInfo,
        assignment: &[usize],
        blocks: &[Vec<usize>],
        num_blocks: usize,
        balance_tol: f64,
        max_cap: Option<usize>,
        num_elements: usize,
    ) -> usize {
        let mut best_block = 0;
        let mut best_score = f64::INFINITY;

        let total_assigned: usize = blocks.iter().map(|b| b.len()).sum();
        let ideal_size = if num_blocks > 0 {
            (total_assigned + 1) as f64 / num_blocks as f64
        } else {
            f64::INFINITY
        };

        for b in 0..num_blocks {
            let block_size = blocks[b].len();

            if let Some(cap) = max_cap {
                if block_size >= cap {
                    continue;
                }
            }

            if balance_tol < 1.0 {
                let max_allowed = (ideal_size * (1.0 + balance_tol)).ceil() as usize;
                if block_size >= max_allowed && max_allowed > 0 {
                    continue;
                }
            }

            // Compute how many crossing edges this element would contribute in block b.
            let mut crossing = 0.0;
            for &(nbr, weight) in adjacency.neighbors(element) {
                if nbr < num_elements && assignment[nbr] != usize::MAX {
                    if assignment[nbr] != b {
                        crossing += weight;
                    }
                }
            }

            // Tie-break by block size for balance.
            let balance_penalty = block_size as f64 * 0.001;
            let score = crossing + balance_penalty;

            if score < best_score {
                best_score = score;
                best_block = b;
            }
        }

        best_block
    }

    /// Compute what crossing weight would result if `element` were assigned to `block`.
    pub fn crossing_weight_if_assigned(
        element: usize,
        block: usize,
        adjacency: &AdjacencyInfo,
        partition: &Partition,
    ) -> f64 {
        let mut crossing = 0.0;
        for &(nbr, weight) in adjacency.neighbors(element) {
            let nbr_block = partition.block_of(nbr);
            match nbr_block {
                Some(nb) if nb != block => {
                    crossing += weight;
                }
                None => {
                    // Neighbor not yet assigned; does not contribute.
                }
                _ => {}
            }
        }
        crossing
    }

    /// Check if a partition is balanced within tolerance.
    pub fn is_balanced(partition: &Partition, balance_tol: f64) -> bool {
        if partition.num_blocks() <= 1 {
            return true;
        }
        if balance_tol >= 1.0 {
            return true;
        }
        let sizes = partition.block_sizes();
        let min_s = *sizes.iter().min().unwrap_or(&0) as f64;
        let max_s = *sizes.iter().max().unwrap_or(&1) as f64;
        if max_s == 0.0 {
            return true;
        }
        let ratio = min_s / max_s;
        // balance_tol = 0 means ratio must be 1.0 (perfect)
        // balance_tol = 1 means any ratio is fine
        ratio >= 1.0 - balance_tol
    }

    /// Build a uniformly random partition.
    pub fn build_random(num_elements: usize, num_blocks: usize, rng: &mut impl Rng) -> Partition {
        let mut blocks: Vec<Vec<usize>> = (0..num_blocks).map(|_| Vec::new()).collect();
        for e in 0..num_elements {
            let b = rng.gen_range(0..num_blocks.max(1));
            blocks[b].push(e);
        }
        // Ensure non-empty blocks.
        for b in 0..num_blocks {
            if blocks[b].is_empty() {
                let largest = blocks
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != b)
                    .max_by_key(|(_, bl)| bl.len())
                    .map(|(i, _)| i);
                if let Some(li) = largest {
                    if blocks[li].len() > 1 {
                        let e = blocks[li].pop().unwrap();
                        blocks[b].push(e);
                    }
                }
            }
        }
        Partition {
            blocks,
            num_elements,
        }
    }

    /// Round-robin assignment: cycle through blocks in given order.
    pub fn build_round_robin(
        num_elements: usize,
        num_blocks: usize,
        order: &[usize],
    ) -> Partition {
        let k = num_blocks.max(1);
        let mut blocks: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
        for (i, &elem) in order.iter().enumerate() {
            blocks[i % k].push(elem);
        }
        // Handle case where order doesn't cover all elements.
        let mut seen: Vec<bool> = vec![false; num_elements];
        for &e in order {
            if e < num_elements {
                seen[e] = true;
            }
        }
        let mut block_idx = 0;
        for e in 0..num_elements {
            if !seen[e] {
                blocks[block_idx % k].push(e);
                block_idx += 1;
            }
        }
        Partition {
            blocks,
            num_elements,
        }
    }

    /// Produce the list of orderings to try based on config.
    fn heuristic_orderings(config: &GreedyConfig) -> Vec<VariableOrdering> {
        let mut orderings = Vec::new();
        orderings.push(config.ordering.clone());

        let all = [
            VariableOrdering::HighestDegree,
            VariableOrdering::MostConstrained,
            VariableOrdering::Random,
            VariableOrdering::Natural,
        ];

        for o in &all {
            if *o != config.ordering && orderings.len() < config.num_heuristics {
                orderings.push(o.clone());
            }
        }

        // Fill remaining slots with Random (different seeds handled by rng state).
        while orderings.len() < config.num_heuristics {
            orderings.push(VariableOrdering::Random);
        }

        orderings.truncate(config.num_heuristics);
        orderings
    }

    fn compute_crossing_weight(partition: &Partition, adjacency: &AdjacencyInfo) -> f64 {
        let mut block_map = vec![0usize; partition.num_elements];
        for (bi, block) in partition.blocks.iter().enumerate() {
            for &e in block {
                block_map[e] = bi;
            }
        }
        let mut total = 0.0;
        for &(u, v, w) in &adjacency.edges {
            if u < partition.num_elements
                && v < partition.num_elements
                && block_map[u] != block_map[v]
            {
                total += w;
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_graph(n: usize) -> AdjacencyInfo {
        let edges: Vec<(usize, usize, f64)> = (0..n.saturating_sub(1))
            .map(|i| (i, i + 1, 1.0))
            .collect();
        AdjacencyInfo::from_edges(n, edges)
    }

    fn cluster_graph() -> AdjacencyInfo {
        // Two clusters: {0,1,2} and {3,4,5} with heavy internal edges, light bridge.
        AdjacencyInfo::from_edges(
            6,
            vec![
                (0, 1, 10.0),
                (0, 2, 10.0),
                (1, 2, 10.0),
                (3, 4, 10.0),
                (3, 5, 10.0),
                (4, 5, 10.0),
                (2, 3, 0.5),
            ],
        )
    }

    #[test]
    fn test_build_basic() {
        let adj = line_graph(6);
        let config = GreedyConfig {
            num_blocks: 2,
            ..Default::default()
        };
        let p = GreedyBuilder::build(6, &adj, &config).unwrap();
        assert!(p.is_valid());
        assert_eq!(p.num_blocks(), 2);
    }

    #[test]
    fn test_build_cluster() {
        let adj = cluster_graph();
        let config = GreedyConfig {
            num_blocks: 2,
            num_heuristics: 4,
            ..Default::default()
        };
        let p = GreedyBuilder::build(6, &adj, &config).unwrap();
        assert!(p.is_valid());
        let cw = GreedyBuilder::compute_crossing_weight(&p, &adj);
        // Should find the natural cluster split with crossing = 0.5.
        assert!(cw <= 0.5 + 1e-9, "crossing weight = {}", cw);
    }

    #[test]
    fn test_build_single_natural() {
        let adj = line_graph(4);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let p = GreedyBuilder::build_single(
            4,
            &adj,
            &VariableOrdering::Natural,
            2,
            1.0,
            None,
            &mut rng,
        )
        .unwrap();
        assert!(p.is_valid());
    }

    #[test]
    fn test_variable_ordering_highest_degree() {
        let adj = AdjacencyInfo::from_edges(
            4,
            vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
        );
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let order = GreedyBuilder::variable_ordering(
            4,
            &adj,
            &VariableOrdering::HighestDegree,
            &mut rng,
        );
        assert_eq!(order[0], 0); // Vertex 0 has degree 3.
    }

    #[test]
    fn test_variable_ordering_random() {
        let adj = AdjacencyInfo::new(10);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let order =
            GreedyBuilder::variable_ordering(10, &adj, &VariableOrdering::Random, &mut rng);
        assert_eq!(order.len(), 10);
        // Check it's a permutation.
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_is_balanced() {
        let p = Partition::from_blocks(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        assert!(GreedyBuilder::is_balanced(&p, 0.0));

        let p2 = Partition::from_blocks(vec![vec![0, 1, 2], vec![3]], 4).unwrap();
        assert!(!GreedyBuilder::is_balanced(&p2, 0.0));
        assert!(GreedyBuilder::is_balanced(&p2, 1.0));
    }

    #[test]
    fn test_build_random_partition() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p = GreedyBuilder::build_random(10, 3, &mut rng);
        assert_eq!(p.num_elements, 10);
        // All elements accounted for.
        let total: usize = p.blocks.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_build_round_robin() {
        let order: Vec<usize> = (0..6).collect();
        let p = GreedyBuilder::build_round_robin(6, 3, &order);
        assert_eq!(p.blocks[0], vec![0, 3]);
        assert_eq!(p.blocks[1], vec![1, 4]);
        assert_eq!(p.blocks[2], vec![2, 5]);
    }

    #[test]
    fn test_build_zero_elements() {
        let adj = AdjacencyInfo::new(0);
        let config = GreedyConfig {
            num_blocks: 1,
            ..Default::default()
        };
        let result = GreedyBuilder::build(0, &adj, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_with_capacity() {
        let adj = line_graph(8);
        let config = GreedyConfig {
            num_blocks: 2,
            max_block_capacity: Some(5),
            balance_tolerance: 1.0,
            ..Default::default()
        };
        let p = GreedyBuilder::build(8, &adj, &config).unwrap();
        assert!(p.is_valid());
        for block in &p.blocks {
            assert!(block.len() <= 5);
        }
    }

    #[test]
    fn test_crossing_weight_if_assigned() {
        let adj = AdjacencyInfo::from_edges(4, vec![(0, 1, 3.0), (0, 2, 5.0)]);
        let p = Partition::from_blocks(vec![vec![1], vec![2]], 4).unwrap();
        // Element 0 not yet in partition; if assigned to block 0 (with 1):
        // nbr 1 is in block 0 -> same -> 0
        // nbr 2 is in block 1 -> diff -> 5.0
        let cw = GreedyBuilder::crossing_weight_if_assigned(0, 0, &adj, &p);
        assert!((cw - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_three_blocks() {
        let adj = AdjacencyInfo::from_edges(
            9,
            vec![
                (0, 1, 10.0),
                (1, 2, 10.0),
                (0, 2, 10.0),
                (3, 4, 10.0),
                (4, 5, 10.0),
                (3, 5, 10.0),
                (6, 7, 10.0),
                (7, 8, 10.0),
                (6, 8, 10.0),
                (2, 3, 0.1),
                (5, 6, 0.1),
            ],
        );
        let config = GreedyConfig {
            num_blocks: 3,
            num_heuristics: 4,
            ..Default::default()
        };
        let p = GreedyBuilder::build(9, &adj, &config).unwrap();
        assert!(p.is_valid());
        let cw = GreedyBuilder::compute_crossing_weight(&p, &adj);
        // Optimal split has crossing = 0.2
        assert!(cw <= 0.2 + 1e-6, "crossing weight = {}", cw);
    }
}
