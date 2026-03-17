//! Cut pool management for bilevel intersection cuts.
//!
//! Stores generated cuts, provides scoring (efficacy, orthogonality, sparsity),
//! aging and purging policies, parallel cut selection, cut strengthening hooks,
//! rank tracking, and pool statistics.

use crate::{cut_orthogonality, BilevelCut, CutError, CutResult, MIN_EFFICACY, TOLERANCE};
use bicut_types::ConstraintSense;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the cut pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutPoolConfig {
    /// Maximum number of cuts in the pool.
    pub max_size: usize,
    /// Maximum age (rounds without being binding) before purging.
    pub max_age: usize,
    /// Minimum efficacy to keep a cut in the pool.
    pub min_efficacy: f64,
    /// Weight for efficacy in the scoring function.
    pub efficacy_weight: f64,
    /// Weight for orthogonality in the scoring function.
    pub orthogonality_weight: f64,
    /// Weight for sparsity in the scoring function.
    pub sparsity_weight: f64,
    /// Whether to remove near-parallel cuts.
    pub remove_parallel: bool,
    /// Cosine similarity threshold for parallel detection.
    pub parallel_threshold: f64,
    /// Purge fraction: what fraction of cuts to remove when pool is full.
    pub purge_fraction: f64,
}

impl Default for CutPoolConfig {
    fn default() -> Self {
        Self {
            max_size: 5000,
            max_age: 10,
            min_efficacy: MIN_EFFICACY,
            efficacy_weight: 1.0,
            orthogonality_weight: 0.1,
            sparsity_weight: 0.01,
            remove_parallel: true,
            parallel_threshold: 0.99,
            purge_fraction: 0.3,
        }
    }
}

/// An entry in the cut pool with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutEntry {
    pub cut: BilevelCut,
    pub id: u64,
    pub age: usize,
    pub times_binding: usize,
    pub times_violated: usize,
    pub last_efficacy: f64,
    pub creation_round: usize,
    pub last_active_round: usize,
    pub score: f64,
    pub rank: u32,
    pub active: bool,
}

impl CutEntry {
    pub fn new(cut: BilevelCut, id: u64, round: usize) -> Self {
        let rank = cut.rank;
        Self {
            cut,
            id,
            age: 0,
            times_binding: 0,
            times_violated: 0,
            last_efficacy: 0.0,
            creation_round: round,
            last_active_round: round,
            score: 0.0,
            rank,
            active: true,
        }
    }
}

/// Pool statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolStatistics {
    pub total_added: usize,
    pub total_purged: usize,
    pub total_duplicates_rejected: usize,
    pub current_size: usize,
    pub active_cuts: usize,
    pub avg_age: f64,
    pub avg_efficacy: f64,
    pub max_rank: u32,
    pub rank_distribution: HashMap<u32, usize>,
    pub purge_rounds: usize,
}

/// Scoring information for a cut.
#[derive(Debug, Clone)]
pub struct CutScore {
    pub efficacy: f64,
    pub orthogonality: f64,
    pub sparsity: f64,
    pub age_penalty: f64,
    pub combined: f64,
}

/// The cut pool.
#[derive(Debug)]
pub struct CutPool {
    pub config: CutPoolConfig,
    entries: Vec<CutEntry>,
    next_id: u64,
    current_round: usize,
    stats: PoolStatistics,
    /// Hash set for duplicate detection.
    hashes: HashMap<Vec<OrderedFloat<f64>>, u64>,
}

impl CutPool {
    pub fn new(config: CutPoolConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            next_id: 0,
            current_round: 0,
            stats: PoolStatistics::default(),
            hashes: HashMap::new(),
        }
    }

    /// Add a cut to the pool.
    pub fn add_cut(&mut self, cut: BilevelCut, point: &[f64]) -> Option<u64> {
        // Check for duplicates.
        let hash = self.compute_hash(&cut);
        if self.hashes.contains_key(&hash) {
            self.stats.total_duplicates_rejected += 1;
            return None;
        }

        // Check parallel cuts.
        if self.config.remove_parallel {
            let dominated = self.entries.iter().any(|e| {
                if !e.active {
                    return false;
                }
                let cos = crate::cut_cosine_similarity(&cut.coeffs, &e.cut.coeffs);
                cos.abs() > self.config.parallel_threshold
            });
            if dominated {
                self.stats.total_duplicates_rejected += 1;
                return None;
            }
        }

        // Purge if full.
        if self.entries.len() >= self.config.max_size {
            self.purge();
        }

        let id = self.next_id;
        self.next_id += 1;

        let mut entry = CutEntry::new(cut, id, self.current_round);
        entry.last_efficacy = entry.cut.compute_efficacy(point);
        entry.score = self.compute_score_for_entry(&entry);

        self.hashes.insert(hash, id);
        self.entries.push(entry);
        self.stats.total_added += 1;
        self.stats.current_size = self.entries.len();

        Some(id)
    }

    /// Get violated cuts for a point, sorted by efficacy descending.
    pub fn get_violated_cuts(
        &self,
        point: &[f64],
        min_violation: f64,
        max_count: usize,
    ) -> Vec<&CutEntry> {
        let mut violated: Vec<(&CutEntry, f64)> = self
            .entries
            .iter()
            .filter(|e| e.active && e.cut.is_violated(point, min_violation))
            .map(|e| (e, e.cut.compute_efficacy(point)))
            .collect();
        violated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        violated
            .into_iter()
            .take(max_count)
            .map(|(e, _)| e)
            .collect()
    }

    /// Select a diverse set of cuts using a greedy orthogonality-based procedure.
    pub fn select_diverse_cuts(
        &self,
        point: &[f64],
        max_count: usize,
        dim: usize,
    ) -> Vec<&CutEntry> {
        let mut violated: Vec<(&CutEntry, f64)> = self
            .entries
            .iter()
            .filter(|e| e.active && e.cut.is_violated(point, self.config.min_efficacy))
            .map(|e| (e, e.cut.compute_efficacy(point)))
            .collect();

        if violated.is_empty() {
            return Vec::new();
        }

        violated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<&CutEntry> = Vec::new();
        for (entry, _) in &violated {
            if selected.len() >= max_count {
                break;
            }

            let is_diverse = selected.iter().all(|s| {
                let orth = cut_orthogonality(&entry.cut.coeffs, &s.cut.coeffs);
                orth > 0.1
            });

            if is_diverse || selected.is_empty() {
                selected.push(entry);
            }
        }
        selected
    }

    /// Advance the round counter and age all cuts.
    pub fn advance_round(&mut self) {
        self.current_round += 1;
        for entry in &mut self.entries {
            if entry.active {
                entry.age += 1;
            }
        }
    }

    /// Mark cuts as binding (active in the LP).
    pub fn mark_binding(&mut self, cut_ids: &[u64]) {
        for entry in &mut self.entries {
            if cut_ids.contains(&entry.id) {
                entry.times_binding += 1;
                entry.age = 0;
                entry.last_active_round = self.current_round;
            }
        }
    }

    /// Mark cuts as violated at the current solution.
    pub fn mark_violated(&mut self, cut_ids: &[u64]) {
        for entry in &mut self.entries {
            if cut_ids.contains(&entry.id) {
                entry.times_violated += 1;
            }
        }
    }

    /// Purge old, low-quality cuts.
    pub fn purge(&mut self) {
        let purge_count = (self.entries.len() as f64 * self.config.purge_fraction) as usize;
        if purge_count == 0 {
            return;
        }

        // Score all entries.
        for entry in &mut self.entries {
            entry.score = self.compute_score_value(
                entry.last_efficacy,
                entry.age,
                entry.times_binding,
                entry.cut.nnz,
            );
        }

        // Sort by score ascending (worst first).
        self.entries.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove the worst entries.
        let to_remove: Vec<u64> = self
            .entries
            .iter()
            .take(purge_count)
            .map(|e| e.id)
            .collect();

        self.entries.retain(|e| !to_remove.contains(&e.id));

        // Clean up hashes.
        self.hashes.retain(|_, id| !to_remove.contains(id));

        self.stats.total_purged += purge_count;
        self.stats.current_size = self.entries.len();
        self.stats.purge_rounds += 1;
    }

    /// Remove cuts that have been inactive for too long.
    pub fn purge_aged(&mut self) {
        let max_age = self.config.max_age;
        let before = self.entries.len();
        let removed_ids: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| e.age > max_age)
            .map(|e| e.id)
            .collect();
        self.entries.retain(|e| e.age <= max_age);
        self.hashes.retain(|_, id| !removed_ids.contains(id));
        let purged = before - self.entries.len();
        self.stats.total_purged += purged;
        self.stats.current_size = self.entries.len();
    }

    /// Compute the combined score for a cut entry.
    fn compute_score_for_entry(&self, entry: &CutEntry) -> f64 {
        self.compute_score_value(
            entry.last_efficacy,
            entry.age,
            entry.times_binding,
            entry.cut.nnz,
        )
    }

    /// Compute the combined score from components.
    fn compute_score_value(
        &self,
        efficacy: f64,
        age: usize,
        times_binding: usize,
        nnz: usize,
    ) -> f64 {
        let eff_score = self.config.efficacy_weight * efficacy;
        let sparse_score = self.config.sparsity_weight / (nnz as f64 + 1.0);
        let age_penalty = 0.1 * age as f64;
        let binding_bonus = 0.05 * times_binding as f64;
        eff_score + sparse_score - age_penalty + binding_bonus
    }

    /// Compute a hash for duplicate detection.
    fn compute_hash(&self, cut: &BilevelCut) -> Vec<OrderedFloat<f64>> {
        let precision = 1e6;
        let mut hash: Vec<OrderedFloat<f64>> = cut
            .coeffs
            .iter()
            .map(|&(j, a)| OrderedFloat((j as f64 * precision + (a * 1e4).round())))
            .collect();
        hash.push(OrderedFloat((cut.rhs * 1e4).round()));
        hash
    }

    /// Get pool statistics.
    pub fn statistics(&self) -> PoolStatistics {
        let mut stats = self.stats.clone();
        stats.current_size = self.entries.len();
        stats.active_cuts = self.entries.iter().filter(|e| e.active).count();

        if !self.entries.is_empty() {
            stats.avg_age =
                self.entries.iter().map(|e| e.age as f64).sum::<f64>() / self.entries.len() as f64;
            stats.avg_efficacy = self.entries.iter().map(|e| e.last_efficacy).sum::<f64>()
                / self.entries.len() as f64;
        }

        let mut rank_dist = HashMap::new();
        for entry in &self.entries {
            *rank_dist.entry(entry.rank).or_insert(0) += 1;
        }
        stats.rank_distribution = rank_dist;
        stats.max_rank = self.entries.iter().map(|e| e.rank).max().unwrap_or(0);
        stats
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hashes.clear();
        self.stats = PoolStatistics::default();
    }

    /// Get all entries (read-only).
    pub fn entries(&self) -> &[CutEntry] {
        &self.entries
    }

    /// Get a cut by ID.
    pub fn get_by_id(&self, id: u64) -> Option<&CutEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Deactivate a cut (keep in pool but don't use).
    pub fn deactivate(&mut self, id: u64) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.active = false;
        }
    }

    /// Reactivate a cut.
    pub fn reactivate(&mut self, id: u64) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.active = true;
        }
    }

    /// Get the current round.
    pub fn current_round(&self) -> usize {
        self.current_round
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> CutPool {
        CutPool::new(CutPoolConfig {
            max_size: 10,
            ..CutPoolConfig::default()
        })
    }

    fn make_cut(idx: usize, coeff: f64) -> BilevelCut {
        BilevelCut::new(vec![(idx, coeff)], 1.0, ConstraintSense::Ge)
    }

    #[test]
    fn test_pool_creation() {
        let pool = make_pool();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_add_cut() {
        let mut pool = make_pool();
        let id = pool.add_cut(make_cut(0, 1.0), &[0.0]);
        assert!(id.is_some());
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_duplicate_rejection() {
        let mut pool = make_pool();
        pool.add_cut(make_cut(0, 1.0), &[0.0]);
        let id2 = pool.add_cut(make_cut(0, 1.0), &[0.0]);
        assert!(id2.is_none());
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_get_violated_cuts() {
        let mut pool = make_pool();
        pool.add_cut(make_cut(0, 1.0), &[0.0]);
        let violated = pool.get_violated_cuts(&[0.0], 1e-8, 10);
        assert_eq!(violated.len(), 1);
    }

    #[test]
    fn test_advance_round() {
        let mut pool = make_pool();
        pool.add_cut(make_cut(0, 1.0), &[0.0]);
        pool.advance_round();
        assert_eq!(pool.entries[0].age, 1);
    }

    #[test]
    fn test_purge_aged() {
        let mut pool = CutPool::new(CutPoolConfig {
            max_age: 2,
            ..CutPoolConfig::default()
        });
        pool.add_cut(make_cut(0, 1.0), &[0.0]);
        pool.advance_round();
        pool.advance_round();
        pool.advance_round();
        pool.purge_aged();
        assert!(pool.is_empty());
    }

    #[test]
    fn test_mark_binding() {
        let mut pool = make_pool();
        let id = pool.add_cut(make_cut(0, 1.0), &[0.0]).unwrap();
        pool.advance_round();
        pool.mark_binding(&[id]);
        assert_eq!(pool.entries[0].times_binding, 1);
        assert_eq!(pool.entries[0].age, 0);
    }

    #[test]
    fn test_statistics() {
        let mut pool = make_pool();
        pool.add_cut(make_cut(0, 1.0), &[0.0]);
        pool.add_cut(make_cut(1, 2.0), &[0.0]);
        let stats = pool.statistics();
        assert_eq!(stats.current_size, 2);
        assert_eq!(stats.active_cuts, 2);
    }

    #[test]
    fn test_deactivate_reactivate() {
        let mut pool = make_pool();
        let id = pool.add_cut(make_cut(0, 1.0), &[0.0]).unwrap();
        pool.deactivate(id);
        let stats = pool.statistics();
        assert_eq!(stats.active_cuts, 0);
        pool.reactivate(id);
        let stats = pool.statistics();
        assert_eq!(stats.active_cuts, 1);
    }

    #[test]
    fn test_clear() {
        let mut pool = make_pool();
        pool.add_cut(make_cut(0, 1.0), &[0.0]);
        pool.clear();
        assert!(pool.is_empty());
    }
}
