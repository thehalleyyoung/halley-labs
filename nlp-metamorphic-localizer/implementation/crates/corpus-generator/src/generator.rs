//! Test case generator combining seed corpus, coverage tracking, and selection.

use crate::coverage::{CoverageGoal, CoverageTracker};
use crate::seed_corpus::SeedCorpus;
use crate::selector::{SeedSelector, SelectionStrategy};
use serde::{Deserialize, Serialize};

/// Configuration for the test generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    pub strategy: SelectionStrategy,
    pub max_tests: usize,
    pub coverage_goal: Option<CoverageGoal>,
    pub max_word_count: usize,
    pub min_word_count: usize,
    pub seed: Option<u64>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::CoverageGuided,
            max_tests: 1000,
            coverage_goal: None,
            max_word_count: 50,
            min_word_count: 3,
            seed: None,
        }
    }
}

/// A generated test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTestCase {
    pub id: String,
    pub seed_id: String,
    pub input_text: String,
    pub transformation: String,
    pub selection_reason: String,
}

/// The main test case generator.
pub struct TestGenerator {
    config: GeneratorConfig,
    corpus: SeedCorpus,
    selector: SeedSelector,
    tracker: CoverageTracker,
    generated_count: usize,
}

impl TestGenerator {
    pub fn new(corpus: SeedCorpus, config: GeneratorConfig) -> Self {
        let all_transforms: Vec<String> = {
            let mut t: Vec<String> = corpus
                .seeds
                .iter()
                .flat_map(|s| s.applicable_transformations.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            t.sort();
            t
        };

        let goal = config.coverage_goal.clone().unwrap_or_else(|| {
            let per_transform = config.max_tests / all_transforms.len().max(1);
            CoverageGoal::uniform(&all_transforms, per_transform.max(1))
        });

        let mut selector = SeedSelector::new(config.strategy.clone());
        selector.set_transformation_order(all_transforms);
        let tracker = CoverageTracker::new(goal);

        Self {
            config,
            corpus,
            selector,
            tracker,
            generated_count: 0,
        }
    }

    /// Generate the next test case.
    pub fn next(&mut self) -> Option<GeneratedTestCase> {
        if self.generated_count >= self.config.max_tests {
            return None;
        }
        if self.tracker.is_satisfied() && self.generated_count > 0 {
            return None;
        }

        let selection = self.selector.select(&self.corpus, &self.tracker)?;
        let seed = &self.corpus.seeds[selection.seed_index];

        // Filter by word count.
        if seed.sentence.word_count < self.config.min_word_count
            || seed.sentence.word_count > self.config.max_word_count
        {
            // Skip and try again (simplified: just increment and return).
            self.generated_count += 1;
            return self.next();
        }

        self.generated_count += 1;
        self.tracker.record(&selection.transformation);

        Some(GeneratedTestCase {
            id: format!("gen_{:05}", self.generated_count),
            seed_id: seed.sentence.id.clone(),
            input_text: seed.sentence.text.clone(),
            transformation: selection.transformation,
            selection_reason: selection.reason,
        })
    }

    /// Generate all test cases up to the configured maximum.
    pub fn generate_all(&mut self) -> Vec<GeneratedTestCase> {
        let mut cases = Vec::new();
        while let Some(case) = self.next() {
            cases.push(case);
        }
        cases
    }

    /// Get the current coverage tracker.
    pub fn coverage(&self) -> &CoverageTracker {
        &self.tracker
    }

    /// Get how many tests have been generated.
    pub fn generated_count(&self) -> usize {
        self.generated_count
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_basic() {
        let corpus = SeedCorpus::standard_handcrafted();
        let config = GeneratorConfig {
            max_tests: 10,
            strategy: SelectionStrategy::Random,
            ..Default::default()
        };
        let mut gen = TestGenerator::new(corpus, config);
        let cases = gen.generate_all();
        assert!(cases.len() <= 10);
        assert!(!cases.is_empty());
        for case in &cases {
            assert!(!case.input_text.is_empty());
            assert!(!case.transformation.is_empty());
        }
    }

    #[test]
    fn test_generator_coverage_guided() {
        let corpus = SeedCorpus::standard_handcrafted();
        let transforms = vec!["passivization".to_string(), "clefting".to_string()];
        let goal = CoverageGoal::uniform(&transforms, 3);
        let config = GeneratorConfig {
            max_tests: 50,
            strategy: SelectionStrategy::CoverageGuided,
            coverage_goal: Some(goal),
            ..Default::default()
        };
        let mut gen = TestGenerator::new(corpus, config);
        let cases = gen.generate_all();
        assert!(!cases.is_empty());

        // Check that both transformations are covered.
        let pass_count = cases.iter().filter(|c| c.transformation == "passivization").count();
        let cleft_count = cases.iter().filter(|c| c.transformation == "clefting").count();
        assert!(pass_count >= 3);
        assert!(cleft_count >= 3);
    }

    #[test]
    fn test_generator_ids() {
        let corpus = SeedCorpus::standard_handcrafted();
        let config = GeneratorConfig {
            max_tests: 5,
            strategy: SelectionStrategy::Random,
            ..Default::default()
        };
        let mut gen = TestGenerator::new(corpus, config);
        let cases = gen.generate_all();
        // All IDs should be unique.
        let ids: std::collections::HashSet<_> = cases.iter().map(|c| &c.id).collect();
        assert_eq!(ids.len(), cases.len());
    }
}
