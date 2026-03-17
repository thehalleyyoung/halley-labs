//! Delta debugging baseline and helper algorithms.

use serde::{Deserialize, Serialize};
use shared_types::Result;
use std::time::Instant;

/// Result of delta debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDResult {
    pub original_size: usize,
    pub reduced_size: usize,
    pub reduction_ratio: f64,
    pub iterations: usize,
    pub is_minimal: bool,
    pub time_ms: u64,
}

impl DDResult {
    pub fn reduction_percentage(&self) -> f64 {
        if self.original_size == 0 { return 0.0; }
        100.0 * (1.0 - self.reduced_size as f64 / self.original_size as f64)
    }
}

/// Token-level delta debugger (grammar-unaware baseline).
pub struct TokenLevelDD {
    pub min_granularity: usize,
    pub max_iterations: usize,
}

impl Default for TokenLevelDD {
    fn default() -> Self {
        Self { min_granularity: 1, max_iterations: 500 }
    }
}

impl TokenLevelDD {
    pub fn new() -> Self { Self::default() }

    /// Classic ddmin on a token sequence.
    pub fn ddmin<F>(&self, tokens: &[String], test_fn: &F) -> (Vec<String>, DDResult)
    where
        F: Fn(&[String]) -> bool,
    {
        let start = Instant::now();
        let original_size = tokens.len();
        let mut current = tokens.to_vec();
        let mut n = 2usize;
        let mut iterations = 0;

        while current.len() >= 2 && iterations < self.max_iterations {
            iterations += 1;
            let chunk_size = (current.len() + n - 1) / n;
            let chunks: Vec<Vec<String>> = current
                .chunks(chunk_size.max(1))
                .map(|c| c.to_vec())
                .collect();

            let mut reduced = false;

            // Try each subset
            for (i, chunk) in chunks.iter().enumerate() {
                if test_fn(chunk) {
                    current = chunk.clone();
                    n = 2;
                    reduced = true;
                    break;
                }

                // Try complement
                let complement: Vec<String> = chunks
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .flat_map(|(_, c)| c.iter().cloned())
                    .collect();

                if !complement.is_empty() && test_fn(&complement) {
                    current = complement;
                    n = n.saturating_sub(1).max(2);
                    reduced = true;
                    break;
                }
            }

            if !reduced {
                if n >= current.len() {
                    break;
                }
                n = (2 * n).min(current.len());
            }
        }

        let reduced_size = current.len();
        let result = DDResult {
            original_size,
            reduced_size,
            reduction_ratio: if original_size > 0 { reduced_size as f64 / original_size as f64 } else { 1.0 },
            iterations,
            is_minimal: n >= current.len(),
            time_ms: start.elapsed().as_millis() as u64,
        };
        (current, result)
    }
}

/// Hierarchical delta debugger that operates on tree levels.
pub struct HierarchicalDD {
    pub max_iterations: usize,
}

impl Default for HierarchicalDD {
    fn default() -> Self {
        Self { max_iterations: 200 }
    }
}

impl HierarchicalDD {
    pub fn new() -> Self { Self::default() }

    /// Process a tree level-by-level from top to bottom.
    /// `nodes_by_level` is a vector of levels, each containing node indices.
    /// `test_fn` checks if removing a set of nodes still preserves the property.
    pub fn level_by_level<F>(
        &self,
        nodes_by_level: &[Vec<usize>],
        test_fn: &F,
    ) -> (Vec<usize>, DDResult)
    where
        F: Fn(&[usize]) -> bool,
    {
        let start = Instant::now();
        let all_nodes: Vec<usize> = nodes_by_level.iter().flatten().copied().collect();
        let original_size = all_nodes.len();
        let mut kept_nodes = all_nodes.clone();
        let mut iterations = 0;

        for level in nodes_by_level {
            if iterations >= self.max_iterations { break; }

            for &node in level {
                iterations += 1;
                if iterations >= self.max_iterations { break; }

                let without: Vec<usize> = kept_nodes.iter().copied().filter(|&n| n != node).collect();
                if test_fn(&without) {
                    kept_nodes = without;
                }
            }
        }

        let reduced_size = kept_nodes.len();
        let result = DDResult {
            original_size,
            reduced_size,
            reduction_ratio: if original_size > 0 { reduced_size as f64 / original_size as f64 } else { 1.0 },
            iterations,
            is_minimal: true,
            time_ms: start.elapsed().as_millis() as u64,
        };
        (kept_nodes, result)
    }
}

/// Tree-level delta debugger (respects tree structure but not grammar).
pub struct TreeLevelDD {
    pub max_iterations: usize,
}

impl Default for TreeLevelDD {
    fn default() -> Self { Self { max_iterations: 300 } }
}

impl TreeLevelDD {
    pub fn new() -> Self { Self::default() }

    /// Delta debug on tree nodes, trying to remove subtrees.
    pub fn reduce<F>(
        &self,
        subtree_roots: &[usize],
        test_fn: &F,
    ) -> (Vec<usize>, DDResult)
    where
        F: Fn(&[usize]) -> bool,
    {
        let start = Instant::now();
        let original_size = subtree_roots.len();
        let mut current: Vec<usize> = subtree_roots.to_vec();
        let mut iterations = 0;

        // Try removing each subtree root one at a time
        let mut i = 0;
        while i < current.len() && iterations < self.max_iterations {
            iterations += 1;
            let mut without = current.clone();
            without.remove(i);

            if !without.is_empty() && test_fn(&without) {
                current = without;
                // Don't increment i since we removed the element
            } else {
                i += 1;
            }
        }

        let reduced_size = current.len();
        let result = DDResult {
            original_size,
            reduced_size,
            reduction_ratio: if original_size > 0 { reduced_size as f64 / original_size as f64 } else { 1.0 },
            iterations,
            is_minimal: i >= current.len(),
            time_ms: start.elapsed().as_millis() as u64,
        };
        (current, result)
    }
}

/// Compare GCHDD and token-level DD results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDComparison {
    pub gchdd_size: usize,
    pub token_dd_size: usize,
    pub gchdd_time_ms: u64,
    pub token_dd_time_ms: u64,
    pub gchdd_is_grammatical: bool,
    pub token_dd_is_grammatical: bool,
    pub size_improvement: f64,
}

impl DDComparison {
    pub fn new(
        gchdd_size: usize,
        token_dd_size: usize,
        gchdd_time_ms: u64,
        token_dd_time_ms: u64,
        gchdd_grammatical: bool,
        token_dd_grammatical: bool,
    ) -> Self {
        let improvement = if token_dd_size > 0 {
            1.0 - gchdd_size as f64 / token_dd_size as f64
        } else {
            0.0
        };
        Self {
            gchdd_size,
            token_dd_size,
            gchdd_time_ms,
            token_dd_time_ms,
            gchdd_is_grammatical: gchdd_grammatical,
            token_dd_is_grammatical: token_dd_grammatical,
            size_improvement: improvement,
        }
    }

    pub fn gchdd_better(&self) -> bool {
        self.gchdd_is_grammatical && (!self.token_dd_is_grammatical || self.gchdd_size <= self.token_dd_size)
    }
}

/// Partition a sequence into n chunks.
pub fn partition<T: Clone>(items: &[T], n: usize) -> Vec<Vec<T>> {
    if n == 0 || items.is_empty() {
        return vec![items.to_vec()];
    }
    let chunk_size = (items.len() + n - 1) / n;
    items.chunks(chunk_size.max(1)).map(|c| c.to_vec()).collect()
}

/// Compute the complement of a chunk within the full sequence.
pub fn complement<T: Clone + PartialEq>(full: &[T], subset: &[T]) -> Vec<T> {
    full.iter().filter(|item| !subset.contains(item)).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_level_ddmin() {
        let tokens: Vec<String> = "The big fluffy cat sat on the soft mat"
            .split_whitespace()
            .map(String::from)
            .collect();

        let dd = TokenLevelDD::new();
        let (result, stats) = dd.ddmin(&tokens, &|subset: &[String]| {
            subset.iter().any(|t| t == "cat")
        });
        assert!(result.contains(&"cat".to_string()));
        assert!(result.len() <= tokens.len());
    }

    #[test]
    fn test_token_level_ddmin_single() {
        let tokens: Vec<String> = vec!["hello".into()];
        let dd = TokenLevelDD::new();
        let (result, _) = dd.ddmin(&tokens, &|_| true);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_hierarchical_dd() {
        let levels = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6, 7, 8, 9]];
        let hdd = HierarchicalDD::new();
        let (kept, stats) = hdd.level_by_level(&levels, &|nodes: &[usize]| {
            nodes.contains(&3) && nodes.contains(&7)
        });
        assert!(kept.contains(&3));
        assert!(kept.contains(&7));
    }

    #[test]
    fn test_tree_level_dd() {
        let roots = vec![0, 1, 2, 3, 4];
        let tdd = TreeLevelDD::new();
        let (kept, stats) = tdd.reduce(&roots, &|nodes: &[usize]| {
            nodes.contains(&2) && nodes.contains(&4)
        });
        assert!(kept.contains(&2));
        assert!(kept.contains(&4));
    }

    #[test]
    fn test_partition() {
        let items = vec![1, 2, 3, 4, 5];
        let parts = partition(&items, 2);
        assert_eq!(parts.len(), 2);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_complement() {
        let full = vec![1, 2, 3, 4, 5];
        let subset = vec![2, 4];
        let comp = complement(&full, &subset);
        assert_eq!(comp, vec![1, 3, 5]);
    }

    #[test]
    fn test_dd_result_percentage() {
        let r = DDResult {
            original_size: 10,
            reduced_size: 3,
            reduction_ratio: 0.3,
            iterations: 5,
            is_minimal: true,
            time_ms: 100,
        };
        assert!((r.reduction_percentage() - 70.0).abs() < 0.01);
    }

    #[test]
    fn test_dd_comparison() {
        let comp = DDComparison::new(5, 8, 100, 50, true, false);
        assert!(comp.gchdd_better());
        assert!(comp.size_improvement > 0.0);
    }

    #[test]
    fn test_ddmin_empty() {
        let tokens: Vec<String> = Vec::new();
        let dd = TokenLevelDD::new();
        let (result, _) = dd.ddmin(&tokens, &|_: &[String]| true);
        assert!(result.is_empty());
    }
}
