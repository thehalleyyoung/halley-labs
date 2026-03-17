//! Mutant filtering strategies.
//!
//! Provides a composable pipeline of filters that reduce the mutant set to the
//! most interesting or informative mutants before further analysis.

use crate::dominator::DominatorSet;
use crate::{CoverageError, KillMatrix, MutantDescriptor, MutantId, MutationOperator, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// FilterKind
// ---------------------------------------------------------------------------

/// The kind of filter to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterKind {
    /// Keep only mutants from the specified operators.
    ByOperator(Vec<MutationOperator>),
    /// Keep only mutants in the specified functions.
    ByFunction(Vec<String>),
    /// Keep only mutants in the specified files.
    ByFile(Vec<String>),
    /// Remove mutants that are always detected (trivially easy).
    RemoveAlwaysKilled { min_killing_tests: usize },
    /// Remove mutants that are never detected (potentially equivalent).
    RemoveNeverKilled,
    /// Keep only dominator mutants.
    KeepDominators,
    /// Remove mutants by ID.
    ExcludeIds(BTreeSet<MutantId>),
    /// Keep only mutants by ID.
    IncludeIds(BTreeSet<MutantId>),
    /// Remove mutants that contribute less than the threshold to the score.
    ByScoreContribution { min_contribution: f64 },
    /// Keep the top N mutants by some criterion.
    TopN { n: usize, criterion: RankCriterion },
    /// Custom filter with a label.
    Custom {
        label: String,
        ids_to_keep: BTreeSet<MutantId>,
    },
}

/// Criterion for ranking mutants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RankCriterion {
    /// Fewest tests that detect this mutant (hardest to detect).
    FewestKillingTests,
    /// Most tests that detect this mutant (easiest / most detectable).
    MostKillingTests,
}

impl fmt::Display for FilterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByOperator(ops) => {
                let names: Vec<&str> = ops.iter().map(|o| o.short_name()).collect();
                write!(f, "ByOperator({})", names.join(", "))
            }
            Self::ByFunction(funcs) => write!(f, "ByFunction({})", funcs.join(", ")),
            Self::ByFile(files) => write!(f, "ByFile({})", files.join(", ")),
            Self::RemoveAlwaysKilled { min_killing_tests } => {
                write!(f, "RemoveAlwaysKilled(min={})", min_killing_tests)
            }
            Self::RemoveNeverKilled => write!(f, "RemoveNeverKilled"),
            Self::KeepDominators => write!(f, "KeepDominators"),
            Self::ExcludeIds(ids) => write!(f, "ExcludeIds({})", ids.len()),
            Self::IncludeIds(ids) => write!(f, "IncludeIds({})", ids.len()),
            Self::ByScoreContribution { min_contribution } => {
                write!(f, "ByScoreContribution(min={:.4})", min_contribution)
            }
            Self::TopN { n, .. } => write!(f, "TopN({})", n),
            Self::Custom { label, .. } => write!(f, "Custom({})", label),
        }
    }
}

// ---------------------------------------------------------------------------
// FilterResult
// ---------------------------------------------------------------------------

/// Result of applying a filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    /// Filter that was applied.
    pub filter_description: String,
    /// Mutant indices that passed the filter.
    pub kept_indices: BTreeSet<usize>,
    /// Mutant indices that were removed.
    pub removed_indices: BTreeSet<usize>,
    /// Number kept.
    pub kept_count: usize,
    /// Number removed.
    pub removed_count: usize,
}

impl FilterResult {
    /// Fraction of mutants retained by this filter.
    pub fn retention_rate(&self) -> f64 {
        let total = self.kept_count + self.removed_count;
        if total == 0 {
            1.0
        } else {
            self.kept_count as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// MutantFilterPipeline
// ---------------------------------------------------------------------------

/// A composable pipeline of mutant filters.
pub struct MutantFilterPipeline {
    filters: Vec<FilterKind>,
    descriptors: HashMap<MutantId, MutantDescriptor>,
    dominator_set: Option<DominatorSet>,
}

impl MutantFilterPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            descriptors: HashMap::new(),
            dominator_set: None,
        }
    }

    /// Add a filter to the end of the pipeline.
    pub fn add_filter(&mut self, filter: FilterKind) -> &mut Self {
        self.filters.push(filter);
        self
    }

    /// Builder-style: chain a filter.
    pub fn with_filter(mut self, filter: FilterKind) -> Self {
        self.filters.push(filter);
        self
    }

    /// Register mutant descriptors for operator/function filters.
    pub fn register_descriptors(&mut self, descs: Vec<MutantDescriptor>) {
        for d in descs {
            self.descriptors.insert(d.id.clone(), d);
        }
    }

    /// Set the dominator set for the KeepDominators filter.
    pub fn set_dominator_set(&mut self, dom: DominatorSet) {
        self.dominator_set = Some(dom);
    }

    /// Apply all filters in sequence. Returns per-step results and final set.
    pub fn apply(&self, km: &KillMatrix) -> (Vec<FilterResult>, BTreeSet<usize>) {
        let mut current: BTreeSet<usize> = (0..km.num_mutants()).collect();
        let mut results = Vec::new();

        for filter in &self.filters {
            let passed = self.apply_single(filter, &current, km);
            let removed: BTreeSet<usize> = current.difference(&passed).copied().collect();
            results.push(FilterResult {
                filter_description: format!("{}", filter),
                kept_indices: passed.clone(),
                removed_indices: removed.clone(),
                kept_count: passed.len(),
                removed_count: removed.len(),
            });
            current = passed;
        }

        (results, current)
    }

    /// Apply a single filter to the current set of mutant indices.
    fn apply_single(
        &self,
        filter: &FilterKind,
        current: &BTreeSet<usize>,
        km: &KillMatrix,
    ) -> BTreeSet<usize> {
        match filter {
            FilterKind::ByOperator(ops) => {
                let op_set: HashSet<MutationOperator> = ops.iter().copied().collect();
                current
                    .iter()
                    .copied()
                    .filter(|&i| {
                        self.descriptors
                            .get(&km.mutants[i])
                            .map_or(false, |d| op_set.contains(&d.operator))
                    })
                    .collect()
            }

            FilterKind::ByFunction(funcs) => {
                let func_set: HashSet<&String> = funcs.iter().collect();
                current
                    .iter()
                    .copied()
                    .filter(|&i| {
                        self.descriptors
                            .get(&km.mutants[i])
                            .map_or(false, |d| func_set.contains(&d.site.function_name))
                    })
                    .collect()
            }

            FilterKind::ByFile(files) => {
                let file_set: HashSet<&String> = files.iter().collect();
                current
                    .iter()
                    .copied()
                    .filter(|&i| {
                        self.descriptors
                            .get(&km.mutants[i])
                            .map_or(false, |d| file_set.contains(&d.site.file))
                    })
                    .collect()
            }

            FilterKind::RemoveAlwaysKilled { min_killing_tests } => current
                .iter()
                .copied()
                .filter(|&i| km.killing_tests(i).len() < *min_killing_tests)
                .collect(),

            FilterKind::RemoveNeverKilled => current
                .iter()
                .copied()
                .filter(|&i| km.is_killed(i))
                .collect(),

            FilterKind::KeepDominators => match &self.dominator_set {
                Some(dom) => {
                    let dom_ids: HashSet<&MutantId> = dom.members.iter().collect();
                    current
                        .iter()
                        .copied()
                        .filter(|&i| dom_ids.contains(&km.mutants[i]))
                        .collect()
                }
                None => current.clone(),
            },

            FilterKind::ExcludeIds(ids) => current
                .iter()
                .copied()
                .filter(|&i| !ids.contains(&km.mutants[i]))
                .collect(),

            FilterKind::IncludeIds(ids) => current
                .iter()
                .copied()
                .filter(|&i| ids.contains(&km.mutants[i]))
                .collect(),

            FilterKind::ByScoreContribution { min_contribution } => current
                .iter()
                .copied()
                .filter(|&i| {
                    if !km.is_killed(i) {
                        return false;
                    }
                    let ks = km.killing_tests(i);
                    let contribution = 1.0 / ks.len().max(1) as f64;
                    contribution >= *min_contribution
                })
                .collect(),

            FilterKind::TopN { n, criterion } => {
                let mut scored: Vec<(usize, usize)> = current
                    .iter()
                    .map(|&i| (i, km.killing_tests(i).len()))
                    .collect();
                match criterion {
                    RankCriterion::FewestKillingTests => {
                        scored.sort_by_key(|&(_, k)| k);
                    }
                    RankCriterion::MostKillingTests => {
                        scored.sort_by_key(|&(_, k)| std::cmp::Reverse(k));
                    }
                }
                scored.into_iter().take(*n).map(|(i, _)| i).collect()
            }

            FilterKind::Custom { ids_to_keep, .. } => current
                .iter()
                .copied()
                .filter(|&i| ids_to_keep.contains(&km.mutants[i]))
                .collect(),
        }
    }

    /// Apply and return a restricted kill matrix.
    pub fn apply_to_matrix(&self, km: &KillMatrix) -> (KillMatrix, Vec<FilterResult>) {
        let (results, kept) = self.apply(km);
        let restricted = km.restrict_mutants(&kept);
        (restricted, results)
    }

    /// Human-readable pipeline summary.
    pub fn summary(&self) -> String {
        let names: Vec<String> = self.filters.iter().map(|f| format!("{}", f)).collect();
        format!("Pipeline[{}]: {}", self.filters.len(), names.join(" -> "))
    }

    /// Number of filters in the pipeline.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl Default for MutantFilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a standard filter pipeline that removes trivial mutants and keeps
/// only those from the specified operators.
pub fn standard_pipeline(operators: Vec<MutationOperator>) -> MutantFilterPipeline {
    MutantFilterPipeline::new()
        .with_filter(FilterKind::RemoveNeverKilled)
        .with_filter(FilterKind::RemoveAlwaysKilled {
            min_killing_tests: usize::MAX,
        })
        .with_filter(FilterKind::ByOperator(operators))
}

/// Create a pipeline that keeps only dominator mutants.
pub fn dominator_pipeline(dom: DominatorSet) -> MutantFilterPipeline {
    let mut pipe = MutantFilterPipeline::new();
    pipe.set_dominator_set(dom);
    pipe.add_filter(FilterKind::KeepDominators);
    pipe
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        make_test_kill_matrix, make_test_mutant, MutantId, MutationOperator, MutationSite,
    };

    fn km_and_descs() -> (KillMatrix, Vec<MutantDescriptor>) {
        //       m0  m1  m2  m3  m4  m5
        // t0: [  K   K   K   .   .   . ]
        // t1: [  K   .   K   K   .   . ]
        // t2: [  .   .   .   .   K   . ]
        let km = make_test_kill_matrix(
            3,
            6,
            &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 4)],
        );
        let ops = [
            MutationOperator::AOR,
            MutationOperator::ROR,
            MutationOperator::AOR,
            MutationOperator::COR,
            MutationOperator::ROR,
            MutationOperator::SDL,
        ];
        let funcs = ["func_a", "func_a", "func_b", "func_b", "func_c", "func_c"];
        let descs: Vec<MutantDescriptor> = km
            .mutants
            .iter()
            .enumerate()
            .map(|(i, id)| {
                MutantDescriptor::new(
                    id.clone(),
                    ops[i],
                    MutationSite::new("test.c", funcs[i], i + 1, 1),
                    "orig",
                    "repl",
                )
            })
            .collect();
        (km, descs)
    }

    #[test]
    fn test_filter_by_operator() {
        let (km, descs) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.register_descriptors(descs);
        pipe.add_filter(FilterKind::ByOperator(vec![MutationOperator::AOR]));
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2); // m0, m2
        assert!(kept.contains(&0) && kept.contains(&2));
    }

    #[test]
    fn test_filter_by_function() {
        let (km, descs) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.register_descriptors(descs);
        pipe.add_filter(FilterKind::ByFunction(vec!["func_a".into()]));
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2); // m0, m1
    }

    #[test]
    fn test_filter_by_file() {
        let (km, descs) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.register_descriptors(descs);
        pipe.add_filter(FilterKind::ByFile(vec!["test.c".into()]));
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 6);
    }

    #[test]
    fn test_remove_always_killed() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::RemoveAlwaysKilled {
            min_killing_tests: 2,
        });
        let (_, kept) = pipe.apply(&km);
        assert!(!kept.contains(&0));
        assert!(!kept.contains(&2));
    }

    #[test]
    fn test_remove_never_killed() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::RemoveNeverKilled);
        let (_, kept) = pipe.apply(&km);
        assert!(!kept.contains(&5));
        assert_eq!(kept.len(), 5);
    }

    #[test]
    fn test_exclude_ids() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::ExcludeIds(BTreeSet::from([
            MutantId::new("m0"),
            MutantId::new("m1"),
        ])));
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 4);
        assert!(!kept.contains(&0));
        assert!(!kept.contains(&1));
    }

    #[test]
    fn test_include_ids() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::IncludeIds(BTreeSet::from([
            MutantId::new("m0"),
            MutantId::new("m3"),
        ])));
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_top_n_fewest() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::TopN {
            n: 2,
            criterion: RankCriterion::FewestKillingTests,
        });
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_pipeline_composition() {
        let (km, descs) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.register_descriptors(descs);
        pipe.add_filter(FilterKind::RemoveNeverKilled);
        pipe.add_filter(FilterKind::ByOperator(vec![
            MutationOperator::AOR,
            MutationOperator::ROR,
        ]));
        let (results, kept) = pipe.apply(&km);
        assert_eq!(results.len(), 2);
        assert!(kept.len() <= 4);
    }

    #[test]
    fn test_apply_to_matrix() {
        let (km, descs) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.register_descriptors(descs);
        pipe.add_filter(FilterKind::ByOperator(vec![MutationOperator::AOR]));
        let (restricted, _) = pipe.apply_to_matrix(&km);
        assert_eq!(restricted.num_mutants(), 2);
        assert_eq!(restricted.num_tests(), 3);
    }

    #[test]
    fn test_filter_result_retention() {
        let result = FilterResult {
            filter_description: "test".into(),
            kept_indices: BTreeSet::from([0, 1, 2]),
            removed_indices: BTreeSet::from([3, 4]),
            kept_count: 3,
            removed_count: 2,
        };
        assert!((result.retention_rate() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_keep_dominators() {
        let (km, _) = km_and_descs();
        let dom = DominatorSet {
            members: vec![MutantId::new("m0"), MutantId::new("m3")],
            covered_tests: BTreeSet::new(),
            representation: BTreeMap::new(),
            total_killed: 5,
            algorithm: crate::dominator::DominatorAlgorithm::Greedy,
        };
        let mut pipe = MutantFilterPipeline::new();
        pipe.set_dominator_set(dom);
        pipe.add_filter(FilterKind::KeepDominators);
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&0) && kept.contains(&3));
    }

    #[test]
    fn test_custom_filter() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::Custom {
            label: "my_filter".into(),
            ids_to_keep: BTreeSet::from([MutantId::new("m1"), MutantId::new("m4")]),
        });
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_summary() {
        let pipe = MutantFilterPipeline::new()
            .with_filter(FilterKind::RemoveNeverKilled)
            .with_filter(FilterKind::ByOperator(vec![MutationOperator::AOR]));
        let s = pipe.summary();
        assert!(s.contains("Pipeline[2]"));
        assert!(s.contains("RemoveNeverKilled"));
        assert!(s.contains("ByOperator"));
    }

    #[test]
    fn test_empty_pipeline() {
        let km = make_test_kill_matrix(2, 3, &[(0, 0), (1, 1)]);
        let pipe = MutantFilterPipeline::new();
        let (results, kept) = pipe.apply(&km);
        assert!(results.is_empty());
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn test_score_contribution_filter() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::ByScoreContribution {
            min_contribution: 0.5,
        });
        let (_, kept) = pipe.apply(&km);
        assert!(kept.len() <= 6);
    }

    #[test]
    fn test_filter_kind_display() {
        let f = FilterKind::ByOperator(vec![MutationOperator::AOR, MutationOperator::ROR]);
        let s = format!("{}", f);
        assert!(s.contains("AOR"));
        assert!(s.contains("ROR"));
    }

    #[test]
    fn test_dominators_no_set() {
        let km = make_test_kill_matrix(2, 2, &[(0, 0), (1, 1)]);
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::KeepDominators);
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_standard_pipeline() {
        let pipe = standard_pipeline(vec![MutationOperator::AOR]);
        assert_eq!(pipe.len(), 3);
    }

    #[test]
    fn test_dominator_pipeline() {
        let dom = DominatorSet {
            members: vec![MutantId::new("m0")],
            covered_tests: BTreeSet::new(),
            representation: BTreeMap::new(),
            total_killed: 1,
            algorithm: crate::dominator::DominatorAlgorithm::Greedy,
        };
        let pipe = dominator_pipeline(dom);
        assert_eq!(pipe.len(), 1);
    }

    #[test]
    fn test_pipeline_len_and_empty() {
        let pipe = MutantFilterPipeline::new();
        assert!(pipe.is_empty());
        assert_eq!(pipe.len(), 0);

        let pipe = pipe.with_filter(FilterKind::RemoveNeverKilled);
        assert!(!pipe.is_empty());
        assert_eq!(pipe.len(), 1);
    }

    #[test]
    fn test_top_n_most_killed() {
        let (km, _) = km_and_descs();
        let mut pipe = MutantFilterPipeline::new();
        pipe.add_filter(FilterKind::TopN {
            n: 3,
            criterion: RankCriterion::MostKillingTests,
        });
        let (_, kept) = pipe.apply(&km);
        assert_eq!(kept.len(), 3);
    }
}
