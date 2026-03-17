//! Partition types for variable/constraint partitioning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{self, SpectralError, ValidationError};

/// A partition assigning each item (variable/constraint) to a block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// assignment[i] = block index for item i.
    pub assignment: Vec<usize>,
    pub num_blocks: usize,
}

impl Partition {
    pub fn new(assignment: Vec<usize>) -> Self {
        let num_blocks = assignment.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        Self { assignment, num_blocks }
    }

    pub fn uniform(n: usize, k: usize) -> Self {
        let assignment: Vec<usize> = (0..n).map(|i| i % k).collect();
        Self { assignment, num_blocks: k }
    }

    pub fn singleton(n: usize) -> Self {
        Self { assignment: (0..n).collect(), num_blocks: n }
    }

    pub fn single_block(n: usize) -> Self {
        Self { assignment: vec![0; n], num_blocks: 1 }
    }

    pub fn len(&self) -> usize { self.assignment.len() }
    pub fn is_empty(&self) -> bool { self.assignment.is_empty() }

    pub fn block_of(&self, item: usize) -> Option<usize> {
        self.assignment.get(item).copied()
    }

    pub fn block_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.num_blocks];
        for &b in &self.assignment { if b < self.num_blocks { sizes[b] += 1; } }
        sizes
    }

    pub fn block_members(&self, block: usize) -> Vec<usize> {
        self.assignment.iter().enumerate()
            .filter(|(_, &b)| b == block)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn balance_ratio(&self) -> f64 {
        let sizes = self.block_sizes();
        if sizes.is_empty() { return 1.0; }
        let max = *sizes.iter().max().unwrap_or(&1) as f64;
        let min = *sizes.iter().min().unwrap_or(&0) as f64;
        if max < 1e-15 { 1.0 } else { min / max }
    }

    /// Move item to a new block. Returns old block.
    pub fn move_item(&mut self, item: usize, new_block: usize) -> Option<usize> {
        if item >= self.assignment.len() { return None; }
        let old = self.assignment[item];
        self.assignment[item] = new_block;
        if new_block >= self.num_blocks { self.num_blocks = new_block + 1; }
        Some(old)
    }

    /// Merge block `from` into block `to`.
    pub fn merge_blocks(&mut self, from: usize, to: usize) {
        for b in &mut self.assignment {
            if *b == from { *b = to; }
        }
        self.reindex();
    }

    /// Split a block: items matching predicate go to a new block.
    pub fn split_block(&mut self, block: usize, predicate: impl Fn(usize) -> bool) {
        let new_block = self.num_blocks;
        let mut changed = false;
        for (i, b) in self.assignment.iter_mut().enumerate() {
            if *b == block && predicate(i) {
                *b = new_block;
                changed = true;
            }
        }
        if changed { self.num_blocks += 1; }
    }

    /// Re-index blocks to be contiguous 0..k.
    pub fn reindex(&mut self) {
        let mut map = HashMap::new();
        let mut next = 0;
        for b in &mut self.assignment {
            let new = *map.entry(*b).or_insert_with(|| { let v = next; next += 1; v });
            *b = new;
        }
        self.num_blocks = next;
    }

    /// Compute cut weight given an edge list.
    pub fn cut_weight(&self, edges: &[(usize, usize, f64)]) -> f64 {
        edges.iter()
            .filter(|&&(u, v, _)| self.assignment.get(u) != self.assignment.get(v))
            .map(|&(_, _, w)| w)
            .sum()
    }

    /// Normalized cut: sum over blocks of (cut(S, V\S) / vol(S)).
    pub fn normalized_cut(&self, edges: &[(usize, usize, f64)]) -> f64 {
        // Compute volume per block
        let mut vol = vec![0.0; self.num_blocks];
        for &(u, v, w) in edges {
            if let Some(&bu) = self.assignment.get(u) { vol[bu] += w; }
            if let Some(&bv) = self.assignment.get(v) { vol[bv] += w; }
        }

        let mut ncut = 0.0;
        for &(u, v, w) in edges {
            let bu = self.assignment.get(u).copied();
            let bv = self.assignment.get(v).copied();
            if bu != bv {
                if let Some(bu) = bu { if vol[bu] > 1e-15 { ncut += w / vol[bu]; } }
                if let Some(bv) = bv { if vol[bv] > 1e-15 { ncut += w / vol[bv]; } }
            }
        }
        ncut
    }

    /// Serialize to GCG .dec format string.
    pub fn to_dec_format(&self, num_constraints: usize) -> String {
        let mut lines = Vec::new();
        lines.push(format!("PRESOLVED"));
        lines.push(format!("NBLOCKS"));
        lines.push(format!("{}", self.num_blocks));
        for b in 0..self.num_blocks {
            lines.push(format!("BLOCK {}", b + 1));
            for (i, &block) in self.assignment.iter().enumerate() {
                if block == b && i < num_constraints {
                    lines.push(format!("  cons{}", i));
                }
            }
        }
        lines.push("MASTERCONSS".to_string());
        lines.join("\n")
    }

    /// Parse from .dec format string.
    pub fn from_dec_format(text: &str, num_items: usize) -> error::Result<Self> {
        let mut assignment = vec![0usize; num_items];
        let mut current_block = 0usize;
        let mut num_blocks = 0usize;
        let mut in_block = false;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("BLOCK") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if let Some(b) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
                    current_block = b.saturating_sub(1);
                    if current_block >= num_blocks { num_blocks = current_block + 1; }
                    in_block = true;
                }
            } else if trimmed.starts_with("MASTERCONSS") || trimmed.starts_with("NBLOCKS") || trimmed.starts_with("PRESOLVED") {
                in_block = false;
            } else if in_block && trimmed.starts_with("cons") {
                if let Ok(idx) = trimmed.trim_start_matches("cons").trim().parse::<usize>() {
                    if idx < num_items { assignment[idx] = current_block; }
                }
            }
        }

        if num_blocks == 0 {
            return Err(SpectralError::Io(crate::error::IoError::InvalidDecFormat {
                reason: "No blocks found".into(),
            }));
        }

        Ok(Self { assignment, num_blocks })
    }

    pub fn validate(&self) -> error::Result<()> {
        if self.assignment.is_empty() {
            return Err(SpectralError::Validation(ValidationError::EmptyInput {
                context: "partition".into(),
            }));
        }
        for (i, &b) in self.assignment.iter().enumerate() {
            if b >= self.num_blocks {
                return Err(SpectralError::Validation(ValidationError::InvariantViolated {
                    invariant: format!("item {} assigned to block {} >= num_blocks {}", i, b, self.num_blocks),
                }));
            }
        }
        Ok(())
    }
}

/// Statistics about a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionStats {
    pub num_items: usize,
    pub num_blocks: usize,
    pub block_sizes: Vec<usize>,
    pub min_block_size: usize,
    pub max_block_size: usize,
    pub avg_block_size: f64,
    pub balance_ratio: f64,
    pub cut_weight: f64,
    pub normalized_cut: f64,
}

impl PartitionStats {
    pub fn compute(partition: &Partition, edges: &[(usize, usize, f64)]) -> Self {
        let sizes = partition.block_sizes();
        let min_bs = sizes.iter().copied().min().unwrap_or(0);
        let max_bs = sizes.iter().copied().max().unwrap_or(0);
        let avg_bs = if sizes.is_empty() { 0.0 } else { sizes.iter().sum::<usize>() as f64 / sizes.len() as f64 };
        Self {
            num_items: partition.len(),
            num_blocks: partition.num_blocks,
            block_sizes: sizes,
            min_block_size: min_bs,
            max_block_size: max_bs,
            avg_block_size: avg_bs,
            balance_ratio: partition.balance_ratio(),
            cut_weight: partition.cut_weight(edges),
            normalized_cut: partition.normalized_cut(edges),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_partition_new() {
        let p = Partition::new(vec![0, 0, 1, 1, 2]);
        assert_eq!(p.num_blocks, 3); assert_eq!(p.len(), 5);
    }

    #[test] fn test_uniform() {
        let p = Partition::uniform(10, 3);
        assert_eq!(p.num_blocks, 3); assert_eq!(p.len(), 10);
    }

    #[test] fn test_block_sizes() {
        let p = Partition::new(vec![0, 0, 1, 1, 1]);
        assert_eq!(p.block_sizes(), vec![2, 3]);
    }

    #[test] fn test_block_members() {
        let p = Partition::new(vec![0, 1, 0, 1]);
        assert_eq!(p.block_members(0), vec![0, 2]);
    }

    #[test] fn test_balance_ratio() {
        let p = Partition::new(vec![0, 0, 1, 1]);
        assert!((p.balance_ratio() - 1.0).abs() < 1e-10);
    }

    #[test] fn test_move_item() {
        let mut p = Partition::new(vec![0, 0, 1]);
        let old = p.move_item(0, 1);
        assert_eq!(old, Some(0));
        assert_eq!(p.assignment, vec![1, 0, 1]);
    }

    #[test] fn test_merge_blocks() {
        let mut p = Partition::new(vec![0, 1, 2]);
        p.merge_blocks(2, 0);
        assert_eq!(p.block_of(2), Some(0));
    }

    #[test] fn test_split_block() {
        let mut p = Partition::new(vec![0, 0, 0, 1]);
        p.split_block(0, |i| i >= 2);
        assert_eq!(p.block_of(2), Some(2));
        assert_eq!(p.block_of(0), Some(0));
    }

    #[test] fn test_cut_weight() {
        let p = Partition::new(vec![0, 0, 1, 1]);
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0)];
        assert!((p.cut_weight(&edges) - 2.0).abs() < 1e-10);
    }

    #[test] fn test_dec_roundtrip() {
        let p = Partition::new(vec![0, 0, 1, 1, 1]);
        let dec = p.to_dec_format(5);
        let p2 = Partition::from_dec_format(&dec, 5).unwrap();
        assert_eq!(p2.num_blocks, 2);
        assert_eq!(p2.assignment[0], p2.assignment[1]);
    }

    #[test] fn test_validate() {
        let p = Partition::new(vec![0, 1, 0]);
        assert!(p.validate().is_ok());
    }

    #[test] fn test_validate_empty() {
        let p = Partition { assignment: vec![], num_blocks: 0 };
        assert!(p.validate().is_err());
    }

    #[test] fn test_partition_stats() {
        let p = Partition::new(vec![0, 0, 1, 1]);
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0)];
        let stats = PartitionStats::compute(&p, &edges);
        assert_eq!(stats.num_blocks, 2);
        assert!((stats.cut_weight - 2.0).abs() < 1e-10);
    }

    #[test] fn test_reindex() {
        let mut p = Partition { assignment: vec![0, 5, 5, 10], num_blocks: 11 };
        p.reindex();
        assert_eq!(p.num_blocks, 3);
        assert!(p.assignment.iter().all(|&b| b < 3));
    }

    #[test] fn test_singleton() {
        let p = Partition::singleton(3);
        assert_eq!(p.num_blocks, 3);
        assert_eq!(p.assignment, vec![0, 1, 2]);
    }
}
