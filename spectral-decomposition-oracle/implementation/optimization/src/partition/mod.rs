//! Partition module: shared types and submodule declarations for partition optimization.

pub mod evaluation;
pub mod greedy;
pub mod refinement;

pub use evaluation::{
    BlockQuality, EvaluationConfig, PartitionEvaluation, PartitionEvaluator,
};
pub use greedy::{GreedyBuilder, GreedyConfig, VariableOrdering};
pub use refinement::{PartitionRefiner, RefinementConfig, RefinementResult};

use crate::error::{OptError, OptResult};
use serde::{Deserialize, Serialize};

/// A partition of `num_elements` elements into disjoint blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    pub blocks: Vec<Vec<usize>>,
    pub num_elements: usize,
}

impl Partition {
    /// Create a trivial partition: all elements in one block.
    pub fn new(num_elements: usize) -> Self {
        let block: Vec<usize> = (0..num_elements).collect();
        Self {
            blocks: vec![block],
            num_elements,
        }
    }

    /// Create a partition from pre-built blocks.
    pub fn from_blocks(blocks: Vec<Vec<usize>>, num_elements: usize) -> OptResult<Self> {
        let p = Self {
            blocks,
            num_elements,
        };
        if !p.is_valid() {
            return Err(OptError::InvalidProblem {
                reason: "Partition is invalid: not every element appears in exactly one block"
                    .into(),
            });
        }
        Ok(p)
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Return the block index containing `element`, or `None`.
    pub fn block_of(&self, element: usize) -> Option<usize> {
        self.blocks
            .iter()
            .position(|block| block.contains(&element))
    }

    /// Sizes of each block.
    pub fn block_sizes(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.len()).collect()
    }

    /// Check that every element in `0..num_elements` appears in exactly one block.
    pub fn is_valid(&self) -> bool {
        let mut counts = vec![0usize; self.num_elements];
        for block in &self.blocks {
            for &e in block {
                if e >= self.num_elements {
                    return false;
                }
                counts[e] += 1;
            }
        }
        counts.iter().all(|&c| c == 1)
    }

    /// Ratio of smallest block size to largest (0..=1). 1 means perfectly balanced.
    pub fn balance_ratio(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        let sizes = self.block_sizes();
        let min_s = sizes.iter().copied().min().unwrap_or(0) as f64;
        let max_s = sizes.iter().copied().max().unwrap_or(1).max(1) as f64;
        min_s / max_s
    }

    /// Merge block `b` into block `a`, removing block `b`.
    pub fn merge_blocks(&mut self, a: usize, b: usize) {
        assert!(a != b && a < self.blocks.len() && b < self.blocks.len());
        let taken = self.blocks[b].clone();
        self.blocks[a].extend(taken);
        self.blocks.remove(b);
    }

    /// Split a block at `block_idx` using `partition_fn` which returns `true` for the
    /// first new block and `false` for the second. Returns the index of the new block.
    pub fn split_block<F: Fn(usize) -> bool>(
        &mut self,
        block_idx: usize,
        partition_fn: F,
    ) -> usize {
        let old = std::mem::take(&mut self.blocks[block_idx]);
        let (left, right): (Vec<usize>, Vec<usize>) =
            old.into_iter().partition(|&e| partition_fn(e));
        self.blocks[block_idx] = left;
        self.blocks.push(right);
        self.blocks.len() - 1
    }

    /// Move `element` from `from_block` to `to_block`.
    pub fn move_element(&mut self, element: usize, from_block: usize, to_block: usize) {
        if let Some(pos) = self.blocks[from_block].iter().position(|&e| e == element) {
            self.blocks[from_block].remove(pos);
            self.blocks[to_block].push(element);
        }
    }

    /// Count of neighbors of `element` that are in a different block.
    pub fn crossing_elements(&self, element: usize, adjacency: &AdjacencyInfo) -> usize {
        let my_block = match self.block_of(element) {
            Some(b) => b,
            None => return 0,
        };
        adjacency
            .neighbors(element)
            .iter()
            .filter(|&&(nbr, _)| self.block_of(nbr) != Some(my_block))
            .count()
    }
}

/// Weighted adjacency-list graph representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacencyInfo {
    pub edges: Vec<(usize, usize, f64)>,
    adj: Vec<Vec<(usize, f64)>>,
    pub num_vertices: usize,
    pub num_edges: usize,
}

impl AdjacencyInfo {
    /// Create an empty graph with `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
            num_vertices: n,
            num_edges: 0,
        }
    }

    /// Build from an edge list.
    pub fn from_edges(n: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        let mut info = Self::new(n);
        for (u, v, w) in edges {
            info.add_edge(u, v, w);
        }
        info
    }

    /// Add an undirected weighted edge.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.edges.push((u, v, weight));
        if u < self.num_vertices {
            self.adj[u].push((v, weight));
        }
        if v < self.num_vertices {
            self.adj[v].push((u, weight));
        }
        self.num_edges += 1;
    }

    pub fn neighbors(&self, v: usize) -> &[(usize, f64)] {
        if v < self.num_vertices {
            &self.adj[v]
        } else {
            &[]
        }
    }

    pub fn degree(&self, v: usize) -> usize {
        if v < self.num_vertices {
            self.adj[v].len()
        } else {
            0
        }
    }

    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|&(_, _, w)| w).sum()
    }
}

/// Summary quality metrics for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionQuality {
    pub crossing_weight: f64,
    pub balance_ratio: f64,
    pub num_blocks: usize,
    pub total_elements: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_new() {
        let p = Partition::new(5);
        assert_eq!(p.num_blocks(), 1);
        assert_eq!(p.num_elements, 5);
        assert!(p.is_valid());
    }

    #[test]
    fn test_partition_from_blocks() {
        let p = Partition::from_blocks(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        assert_eq!(p.num_blocks(), 2);
        assert!(p.is_valid());
    }

    #[test]
    fn test_partition_invalid() {
        let res = Partition::from_blocks(vec![vec![0, 1], vec![1, 2]], 3);
        assert!(res.is_err());
    }

    #[test]
    fn test_block_of() {
        let p = Partition::from_blocks(vec![vec![0, 2], vec![1, 3]], 4).unwrap();
        assert_eq!(p.block_of(2), Some(0));
        assert_eq!(p.block_of(3), Some(1));
        assert_eq!(p.block_of(4), None);
    }

    #[test]
    fn test_balance_ratio() {
        let p = Partition::from_blocks(vec![vec![0, 1, 2], vec![3]], 4).unwrap();
        assert!((p.balance_ratio() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_merge_blocks() {
        let mut p = Partition::from_blocks(vec![vec![0], vec![1], vec![2]], 3).unwrap();
        p.merge_blocks(0, 2);
        assert_eq!(p.num_blocks(), 2);
        assert!(p.blocks[0].contains(&2));
    }

    #[test]
    fn test_split_block() {
        let mut p = Partition::new(4);
        let new_idx = p.split_block(0, |e| e < 2);
        assert_eq!(p.num_blocks(), 2);
        assert_eq!(p.blocks[0], vec![0, 1]);
        assert_eq!(p.blocks[new_idx], vec![2, 3]);
    }

    #[test]
    fn test_move_element() {
        let mut p = Partition::from_blocks(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        p.move_element(1, 0, 1);
        assert_eq!(p.blocks[0], vec![0]);
        assert!(p.blocks[1].contains(&1));
    }

    #[test]
    fn test_adjacency_info() {
        let adj = AdjacencyInfo::from_edges(3, vec![(0, 1, 1.0), (1, 2, 2.0)]);
        assert_eq!(adj.num_edges, 2);
        assert_eq!(adj.degree(1), 2);
        assert!((adj.total_weight() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_crossing_elements() {
        let adj = AdjacencyInfo::from_edges(4, vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
        let p = Partition::from_blocks(vec![vec![0, 1], vec![2, 3]], 4).unwrap();
        assert_eq!(p.crossing_elements(0, &adj), 2);
    }
}
