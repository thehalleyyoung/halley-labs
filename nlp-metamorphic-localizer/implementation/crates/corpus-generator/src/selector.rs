//! Seed selection strategies for coverage-guided test generation.

use crate::coverage::CoverageTracker;
use crate::seed_corpus::{AnnotatedSeed, SeedCorpus};
use serde::{Deserialize, Serialize};
use rand::Rng;

/// Strategy for selecting seeds from the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Random selection.
    Random,
    /// Select seeds that cover the most-needed transformation.
    CoverageGuided,
    /// Round-robin through transformations.
    RoundRobin,
    /// Prioritize rare constructions.
    RareConstructionFirst,
    /// Weighted random selection favoring underrepresented transformations.
    WeightedRandom,
}

/// Result of a seed selection.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub seed_index: usize,
    pub transformation: String,
    pub reason: String,
}

/// Selects seeds from a corpus according to a strategy.
pub struct SeedSelector {
    strategy: SelectionStrategy,
    round_robin_index: usize,
    transformation_order: Vec<String>,
}

impl SeedSelector {
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            strategy,
            round_robin_index: 0,
            transformation_order: Vec::new(),
        }
    }

    pub fn set_transformation_order(&mut self, order: Vec<String>) {
        self.transformation_order = order;
    }

    /// Select the next seed and transformation from the corpus.
    pub fn select(
        &mut self,
        corpus: &SeedCorpus,
        tracker: &CoverageTracker,
    ) -> Option<SelectionResult> {
        if corpus.is_empty() {
            return None;
        }

        match &self.strategy {
            SelectionStrategy::Random => self.select_random(corpus),
            SelectionStrategy::CoverageGuided => self.select_coverage_guided(corpus, tracker),
            SelectionStrategy::RoundRobin => self.select_round_robin(corpus),
            SelectionStrategy::RareConstructionFirst => self.select_rare_first(corpus),
            SelectionStrategy::WeightedRandom => self.select_weighted(corpus, tracker),
        }
    }

    fn select_random(&self, corpus: &SeedCorpus) -> Option<SelectionResult> {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..corpus.len());
        let seed = &corpus.seeds[idx];
        if seed.applicable_transformations.is_empty() {
            return None;
        }
        let t_idx = rng.gen_range(0..seed.applicable_transformations.len());
        Some(SelectionResult {
            seed_index: idx,
            transformation: seed.applicable_transformations[t_idx].clone(),
            reason: "random selection".to_string(),
        })
    }

    fn select_coverage_guided(
        &self,
        corpus: &SeedCorpus,
        tracker: &CoverageTracker,
    ) -> Option<SelectionResult> {
        let needed = tracker.most_needed_transformation()?;
        let candidates = corpus.seeds_for_transformation(&needed);
        if candidates.is_empty() {
            return self.select_random(corpus);
        }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..candidates.len());
        let seed_idx = corpus
            .seeds
            .iter()
            .position(|s| std::ptr::eq(s, candidates[idx]))
            .unwrap_or(0);
        Some(SelectionResult {
            seed_index: seed_idx,
            transformation: needed,
            reason: "coverage-guided (most needed)".to_string(),
        })
    }

    fn select_round_robin(&mut self, corpus: &SeedCorpus) -> Option<SelectionResult> {
        if self.transformation_order.is_empty() {
            return self.select_random(corpus);
        }
        let transform = &self.transformation_order[self.round_robin_index % self.transformation_order.len()];
        self.round_robin_index += 1;

        let candidates = corpus.seeds_for_transformation(transform);
        if candidates.is_empty() {
            return self.select_random(corpus);
        }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..candidates.len());
        let seed_idx = corpus
            .seeds
            .iter()
            .position(|s| std::ptr::eq(s, candidates[idx]))
            .unwrap_or(0);
        Some(SelectionResult {
            seed_index: seed_idx,
            transformation: transform.clone(),
            reason: "round-robin".to_string(),
        })
    }

    fn select_rare_first(&self, corpus: &SeedCorpus) -> Option<SelectionResult> {
        // Find seeds with the fewest applicable transformations (rare constructions).
        let mut sorted: Vec<(usize, &AnnotatedSeed)> =
            corpus.seeds.iter().enumerate().collect();
        sorted.sort_by_key(|(_, s)| s.transformation_count());

        for (idx, seed) in sorted {
            if !seed.applicable_transformations.is_empty() {
                let mut rng = rand::thread_rng();
                let t_idx = rng.gen_range(0..seed.applicable_transformations.len());
                return Some(SelectionResult {
                    seed_index: idx,
                    transformation: seed.applicable_transformations[t_idx].clone(),
                    reason: "rare construction first".to_string(),
                });
            }
        }
        None
    }

    fn select_weighted(
        &self,
        corpus: &SeedCorpus,
        tracker: &CoverageTracker,
    ) -> Option<SelectionResult> {
        // Weight based on coverage deficit.
        let report = tracker.report();
        let deficits: Vec<(String, usize)> = report
            .iter()
            .filter(|r| r.deficit > 0)
            .map(|r| (r.transformation_name.clone(), r.deficit))
            .collect();

        if deficits.is_empty() {
            return self.select_random(corpus);
        }

        let total_deficit: usize = deficits.iter().map(|(_, d)| d).sum();
        let mut rng = rand::thread_rng();
        let pick = rng.gen_range(0..total_deficit);

        let mut cumulative = 0;
        let mut chosen_transform = deficits[0].0.clone();
        for (t, d) in &deficits {
            cumulative += d;
            if pick < cumulative {
                chosen_transform = t.clone();
                break;
            }
        }

        let candidates = corpus.seeds_for_transformation(&chosen_transform);
        if candidates.is_empty() {
            return self.select_random(corpus);
        }
        let idx = rng.gen_range(0..candidates.len());
        let seed_idx = corpus
            .seeds
            .iter()
            .position(|s| std::ptr::eq(s, candidates[idx]))
            .unwrap_or(0);

        Some(SelectionResult {
            seed_index: seed_idx,
            transformation: chosen_transform,
            reason: "weighted random by deficit".to_string(),
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::CoverageGoal;

    #[test]
    fn test_random_selection() {
        let corpus = SeedCorpus::standard_handcrafted();
        let goal = CoverageGoal::uniform(
            &["passivization".to_string(), "clefting".to_string()],
            5,
        );
        let tracker = CoverageTracker::new(goal);
        let mut selector = SeedSelector::new(SelectionStrategy::Random);
        let result = selector.select(&corpus, &tracker);
        assert!(result.is_some());
    }

    #[test]
    fn test_coverage_guided_selection() {
        let corpus = SeedCorpus::standard_handcrafted();
        let goal = CoverageGoal::uniform(
            &["passivization".to_string(), "clefting".to_string()],
            5,
        );
        let mut tracker = CoverageTracker::new(goal);
        // Satisfy passivization entirely.
        for _ in 0..5 {
            tracker.record("passivization");
        }
        let mut selector = SeedSelector::new(SelectionStrategy::CoverageGuided);
        let result = selector.select(&corpus, &tracker);
        assert!(result.is_some());
        // Should pick clefting since passivization is satisfied.
        assert_eq!(result.unwrap().transformation, "clefting");
    }

    #[test]
    fn test_round_robin_selection() {
        let corpus = SeedCorpus::standard_handcrafted();
        let goal = CoverageGoal::uniform(
            &["passivization".to_string()],
            1,
        );
        let tracker = CoverageTracker::new(goal);
        let mut selector = SeedSelector::new(SelectionStrategy::RoundRobin);
        selector.set_transformation_order(vec![
            "passivization".to_string(),
            "clefting".to_string(),
        ]);
        let r1 = selector.select(&corpus, &tracker);
        assert_eq!(r1.unwrap().transformation, "passivization");
        let r2 = selector.select(&corpus, &tracker);
        assert_eq!(r2.unwrap().transformation, "clefting");
    }
}
