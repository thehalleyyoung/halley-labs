//! Coverage metrics: state, path, and transition coverage analysis.

use crate::{Lts, LtsState, LtsTransition};

use log::{debug, info};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

/// State coverage: percentage of reachable negotiation states explored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCoverage {
    pub total_states: usize,
    pub reachable_states: usize,
    pub explored_states: usize,
    pub coverage_pct: f64,
    pub unexplored_states: Vec<u32>,
    pub phase_coverage: BTreeMap<String, f64>,
}

impl StateCoverage {
    pub fn compute(lts: &Lts, explored: &BTreeSet<u32>) -> Self {
        let reachable = lts.reachable_states();
        let reachable_set: BTreeSet<u32> = reachable.iter().copied().collect();
        let explored_reachable: BTreeSet<u32> =
            explored.intersection(&reachable_set).copied().collect();

        let coverage_pct = if reachable_set.is_empty() {
            0.0
        } else {
            explored_reachable.len() as f64 / reachable_set.len() as f64 * 100.0
        };

        let unexplored: Vec<u32> = reachable_set
            .difference(&explored_reachable)
            .copied()
            .collect();

        let mut phase_totals: BTreeMap<String, usize> = BTreeMap::new();
        let mut phase_explored: BTreeMap<String, usize> = BTreeMap::new();

        for &sid in &reachable_set {
            if let Some(state) = lts.get_state(sid) {
                let key = format!("{:?}", state.phase);
                *phase_totals.entry(key.clone()).or_insert(0) += 1;
                if explored_reachable.contains(&sid) {
                    *phase_explored.entry(key).or_insert(0) += 1;
                }
            }
        }

        let phase_coverage: BTreeMap<String, f64> = phase_totals
            .iter()
            .map(|(phase, &total)| {
                let exp = phase_explored.get(phase).copied().unwrap_or(0);
                let pct = if total > 0 {
                    exp as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                (phase.clone(), pct)
            })
            .collect();

        Self {
            total_states: lts.state_count(),
            reachable_states: reachable_set.len(),
            explored_states: explored_reachable.len(),
            coverage_pct,
            unexplored_states: unexplored,
            phase_coverage,
        }
    }

    pub fn meets_threshold(&self, threshold_pct: f64) -> bool {
        self.coverage_pct >= threshold_pct
    }
}

/// Path coverage: percentage of execution paths explored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCoverage {
    pub total_paths: usize,
    pub explored_paths: usize,
    pub coverage_pct: f64,
    pub max_path_length: usize,
    pub avg_path_length: f64,
    pub path_length_distribution: Vec<(usize, usize)>,
}

impl PathCoverage {
    pub fn compute(lts: &Lts, explored_paths: &[Vec<u32>], max_depth: usize) -> Self {
        let total_paths = count_paths(lts, max_depth);
        let unique_explored: BTreeSet<Vec<u32>> = explored_paths.iter().cloned().collect();
        let explored = unique_explored.len();

        let coverage_pct = if total_paths > 0 {
            (explored as f64 / total_paths as f64 * 100.0).min(100.0)
        } else {
            0.0
        };

        let max_len = explored_paths
            .iter()
            .map(|p| p.len())
            .max()
            .unwrap_or(0);

        let avg_len = if explored_paths.is_empty() {
            0.0
        } else {
            explored_paths.iter().map(|p| p.len()).sum::<usize>() as f64
                / explored_paths.len() as f64
        };

        let mut length_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for path in explored_paths {
            *length_counts.entry(path.len()).or_insert(0) += 1;
        }

        Self {
            total_paths,
            explored_paths: explored,
            coverage_pct,
            max_path_length: max_len,
            avg_path_length: avg_len,
            path_length_distribution: length_counts.into_iter().collect(),
        }
    }

    pub fn meets_threshold(&self, threshold_pct: f64) -> bool {
        self.coverage_pct >= threshold_pct
    }
}

fn count_paths(lts: &Lts, max_depth: usize) -> usize {
    if lts.states.is_empty() {
        return 0;
    }

    let mut count = 0usize;
    let mut stack: Vec<(u32, usize, BTreeSet<u32>)> = Vec::new();
    let mut visited = BTreeSet::new();
    visited.insert(lts.initial_state);
    stack.push((lts.initial_state, 0, visited));

    while let Some((current, depth, path_visited)) = stack.pop() {
        if depth >= max_depth {
            count += 1;
            continue;
        }

        let successors: Vec<u32> = lts
            .transitions_from(current)
            .iter()
            .map(|t| t.target)
            .filter(|t| !path_visited.contains(t))
            .collect();

        if successors.is_empty() {
            count += 1;
        } else {
            for succ in successors {
                let mut new_visited = path_visited.clone();
                new_visited.insert(succ);
                stack.push((succ, depth + 1, new_visited));
                if count > 100_000 {
                    return count;
                }
            }
        }
    }

    count.max(1)
}

/// Transition coverage: percentage of LTS transitions exercised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionCoverage {
    pub total_transitions: usize,
    pub exercised_transitions: usize,
    pub coverage_pct: f64,
    pub uncovered_transitions: Vec<u32>,
    pub downgrade_transitions_covered: usize,
    pub downgrade_transitions_total: usize,
}

impl TransitionCoverage {
    pub fn compute(lts: &Lts, exercised: &BTreeSet<u32>) -> Self {
        let total = lts.transition_count();
        let all_ids: BTreeSet<u32> = lts.transitions.iter().map(|t| t.id).collect();
        let exercised_valid: BTreeSet<u32> = exercised.intersection(&all_ids).copied().collect();

        let coverage_pct = if total > 0 {
            exercised_valid.len() as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        let uncovered: Vec<u32> = all_ids.difference(&exercised_valid).copied().collect();

        let downgrade_total = lts.transitions.iter().filter(|t| t.is_downgrade).count();
        let downgrade_covered = lts
            .transitions
            .iter()
            .filter(|t| t.is_downgrade && exercised_valid.contains(&t.id))
            .count();

        Self {
            total_transitions: total,
            exercised_transitions: exercised_valid.len(),
            coverage_pct,
            uncovered_transitions: uncovered,
            downgrade_transitions_covered: downgrade_covered,
            downgrade_transitions_total: downgrade_total,
        }
    }

    pub fn meets_threshold(&self, threshold_pct: f64) -> bool {
        self.coverage_pct >= threshold_pct
    }

    pub fn all_downgrade_covered(&self) -> bool {
        self.downgrade_transitions_covered == self.downgrade_transitions_total
    }
}

/// Validates that coverage meets required bounds.
pub struct CoverageBoundValidator {
    state_threshold: f64,
    path_threshold: f64,
    transition_threshold: f64,
    require_all_downgrade: bool,
}

impl CoverageBoundValidator {
    pub fn new(
        state_threshold: f64,
        path_threshold: f64,
        transition_threshold: f64,
    ) -> Self {
        Self {
            state_threshold,
            path_threshold,
            transition_threshold,
            require_all_downgrade: true,
        }
    }

    pub fn strict() -> Self {
        Self::new(99.0, 99.0, 99.0)
    }

    pub fn relaxed() -> Self {
        Self::new(90.0, 80.0, 85.0)
    }

    pub fn with_downgrade_requirement(mut self, require: bool) -> Self {
        self.require_all_downgrade = require;
        self
    }

    pub fn validate(
        &self,
        state_cov: &StateCoverage,
        path_cov: &PathCoverage,
        trans_cov: &TransitionCoverage,
    ) -> CoverageValidationResult {
        let mut violations = Vec::new();

        if !state_cov.meets_threshold(self.state_threshold) {
            violations.push(CoverageViolation {
                metric: "state_coverage".into(),
                threshold: self.state_threshold,
                actual: state_cov.coverage_pct,
                gap: self.state_threshold - state_cov.coverage_pct,
                details: format!(
                    "{} of {} reachable states unexplored",
                    state_cov.unexplored_states.len(),
                    state_cov.reachable_states
                ),
            });
        }

        if !path_cov.meets_threshold(self.path_threshold) {
            violations.push(CoverageViolation {
                metric: "path_coverage".into(),
                threshold: self.path_threshold,
                actual: path_cov.coverage_pct,
                gap: self.path_threshold - path_cov.coverage_pct,
                details: format!(
                    "{} of {} paths explored",
                    path_cov.explored_paths, path_cov.total_paths
                ),
            });
        }

        if !trans_cov.meets_threshold(self.transition_threshold) {
            violations.push(CoverageViolation {
                metric: "transition_coverage".into(),
                threshold: self.transition_threshold,
                actual: trans_cov.coverage_pct,
                gap: self.transition_threshold - trans_cov.coverage_pct,
                details: format!(
                    "{} of {} transitions uncovered",
                    trans_cov.uncovered_transitions.len(),
                    trans_cov.total_transitions
                ),
            });
        }

        if self.require_all_downgrade && !trans_cov.all_downgrade_covered() {
            violations.push(CoverageViolation {
                metric: "downgrade_coverage".into(),
                threshold: 100.0,
                actual: if trans_cov.downgrade_transitions_total > 0 {
                    trans_cov.downgrade_transitions_covered as f64
                        / trans_cov.downgrade_transitions_total as f64
                        * 100.0
                } else {
                    100.0
                },
                gap: 0.0,
                details: format!(
                    "{} of {} downgrade transitions covered",
                    trans_cov.downgrade_transitions_covered,
                    trans_cov.downgrade_transitions_total
                ),
            });
        }

        CoverageValidationResult {
            passed: violations.is_empty(),
            violations,
        }
    }
}

impl Default for CoverageBoundValidator {
    fn default() -> Self {
        Self::strict()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageViolation {
    pub metric: String,
    pub threshold: f64,
    pub actual: f64,
    pub gap: f64,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageValidationResult {
    pub passed: bool,
    pub violations: Vec<CoverageViolation>,
}

/// The main coverage analyzer.
pub struct CoverageAnalyzer {
    lts: Lts,
    explored_states: BTreeSet<u32>,
    explored_paths: Vec<Vec<u32>>,
    exercised_transitions: BTreeSet<u32>,
    max_path_depth: usize,
}

impl CoverageAnalyzer {
    pub fn new(lts: Lts) -> Self {
        Self {
            lts,
            explored_states: BTreeSet::new(),
            explored_paths: Vec::new(),
            exercised_transitions: BTreeSet::new(),
            max_path_depth: 20,
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_path_depth = depth;
        self
    }

    /// Simulate full exploration of the LTS.
    pub fn explore_full(&mut self) {
        let reachable = self.lts.reachable_states();
        for &sid in &reachable {
            self.explored_states.insert(sid);
        }
        for trans in &self.lts.transitions {
            if reachable.contains(&trans.source) && reachable.contains(&trans.target) {
                self.exercised_transitions.insert(trans.id);
            }
        }
        self.explored_paths = self.enumerate_paths(self.max_path_depth);
    }

    /// Simulate random exploration with a fixed number of walks.
    pub fn explore_random(&mut self, num_walks: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..num_walks {
            let mut current = self.lts.initial_state;
            let mut path = vec![current];
            self.explored_states.insert(current);

            for _ in 0..self.max_path_depth {
                let transitions = self.lts.transitions_from(current);
                if transitions.is_empty() {
                    break;
                }

                if let Some(&&ref trans) = transitions.choose(&mut rng) {
                    self.exercised_transitions.insert(trans.id);
                    current = trans.target;
                    self.explored_states.insert(current);
                    path.push(current);
                } else {
                    break;
                }
            }

            self.explored_paths.push(path);
        }
    }

    pub fn state_coverage(&self) -> StateCoverage {
        StateCoverage::compute(&self.lts, &self.explored_states)
    }

    pub fn path_coverage(&self) -> PathCoverage {
        PathCoverage::compute(&self.lts, &self.explored_paths, self.max_path_depth)
    }

    pub fn transition_coverage(&self) -> TransitionCoverage {
        TransitionCoverage::compute(&self.lts, &self.exercised_transitions)
    }

    /// Perform a coverage gap analysis: identify what's missing and how to cover it.
    pub fn gap_analysis(&self) -> CoverageGapAnalysis {
        let state_cov = self.state_coverage();
        let trans_cov = self.transition_coverage();

        let mut uncovered_state_info: Vec<GapEntry> = Vec::new();
        for &sid in &state_cov.unexplored_states {
            if let Some(state) = self.lts.get_state(sid) {
                let incoming: Vec<u32> = self
                    .lts
                    .transitions_to(sid)
                    .iter()
                    .map(|t| t.source)
                    .collect();
                uncovered_state_info.push(GapEntry {
                    id: sid,
                    label: state.label.clone(),
                    phase: format!("{:?}", state.phase),
                    reachable_from: incoming,
                    suggestion: format!(
                        "Add test covering {:?} phase via states {:?}",
                        state.phase,
                        self.lts.transitions_to(sid).iter().map(|t| t.source).collect::<Vec<_>>()
                    ),
                });
            }
        }

        let mut uncovered_transition_info: Vec<TransitionGapEntry> = Vec::new();
        for &tid in &trans_cov.uncovered_transitions {
            if let Some(trans) = self.lts.transitions.iter().find(|t| t.id == tid) {
                uncovered_transition_info.push(TransitionGapEntry {
                    id: tid,
                    source: trans.source,
                    target: trans.target,
                    label: trans.label.clone(),
                    is_downgrade: trans.is_downgrade,
                    suggestion: format!(
                        "Exercise transition {} → {} ({})",
                        trans.source, trans.target, trans.label
                    ),
                });
            }
        }

        CoverageGapAnalysis {
            uncovered_states: uncovered_state_info,
            uncovered_transitions: uncovered_transition_info,
            state_coverage_pct: state_cov.coverage_pct,
            transition_coverage_pct: trans_cov.coverage_pct,
        }
    }

    /// Compare coverage achieved by systematic vs random exploration.
    pub fn compare_with_random(&self, num_walks: usize, seed: u64) -> RandomComparisonResult {
        let systematic_state = self.state_coverage();
        let systematic_trans = self.transition_coverage();

        let mut random_analyzer = CoverageAnalyzer::new(self.lts.clone());
        random_analyzer.max_path_depth = self.max_path_depth;
        random_analyzer.explore_random(num_walks, seed);

        let random_state = random_analyzer.state_coverage();
        let random_trans = random_analyzer.transition_coverage();

        RandomComparisonResult {
            systematic_state_coverage: systematic_state.coverage_pct,
            random_state_coverage: random_state.coverage_pct,
            state_coverage_advantage: systematic_state.coverage_pct - random_state.coverage_pct,
            systematic_transition_coverage: systematic_trans.coverage_pct,
            random_transition_coverage: random_trans.coverage_pct,
            transition_coverage_advantage: systematic_trans.coverage_pct
                - random_trans.coverage_pct,
            random_walks: num_walks,
        }
    }

    fn enumerate_paths(&self, max_depth: usize) -> Vec<Vec<u32>> {
        let mut paths = Vec::new();
        let mut stack: Vec<(u32, Vec<u32>, BTreeSet<u32>)> = Vec::new();
        let mut initial_visited = BTreeSet::new();
        initial_visited.insert(self.lts.initial_state);
        stack.push((
            self.lts.initial_state,
            vec![self.lts.initial_state],
            initial_visited,
        ));

        while let Some((current, path, visited)) = stack.pop() {
            if path.len() > max_depth || paths.len() > 10_000 {
                paths.push(path);
                continue;
            }

            let successors: Vec<u32> = self
                .lts
                .transitions_from(current)
                .iter()
                .map(|t| t.target)
                .filter(|t| !visited.contains(t))
                .collect();

            if successors.is_empty() {
                paths.push(path);
            } else {
                for succ in successors {
                    let mut new_path = path.clone();
                    new_path.push(succ);
                    let mut new_visited = visited.clone();
                    new_visited.insert(succ);
                    stack.push((succ, new_path, new_visited));
                }
            }
        }

        paths
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapEntry {
    pub id: u32,
    pub label: String,
    pub phase: String,
    pub reachable_from: Vec<u32>,
    pub suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionGapEntry {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub label: String,
    pub is_downgrade: bool,
    pub suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGapAnalysis {
    pub uncovered_states: Vec<GapEntry>,
    pub uncovered_transitions: Vec<TransitionGapEntry>,
    pub state_coverage_pct: f64,
    pub transition_coverage_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomComparisonResult {
    pub systematic_state_coverage: f64,
    pub random_state_coverage: f64,
    pub state_coverage_advantage: f64,
    pub systematic_transition_coverage: f64,
    pub random_transition_coverage: f64,
    pub transition_coverage_advantage: f64,
    pub random_walks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::HandshakePhase;

    fn make_test_lts() -> Lts {
        let mut lts = Lts::new("test");
        lts.add_state(LtsState::new(0, "init", HandshakePhase::Init));
        let mut s1 = LtsState::new(1, "hello", HandshakePhase::ClientHelloSent);
        lts.add_state(s1);
        let mut s2 = LtsState::new(2, "server", HandshakePhase::ServerHelloReceived);
        s2.is_accepting = true;
        lts.add_state(s2);
        let mut s3 = LtsState::new(3, "error", HandshakePhase::Abort);
        s3.is_error = true;
        lts.add_state(s3);

        lts.add_transition(LtsTransition::new(0, 0, 1, "start"));
        let mut t1 = LtsTransition::new(1, 1, 2, "negotiate");
        t1.cipher_suite_id = Some(0x002F);
        lts.add_transition(t1);
        let mut t2 = LtsTransition::new(2, 1, 3, "fail");
        t2.is_downgrade = true;
        t2.cipher_suite_id = Some(0x0003);
        lts.add_transition(t2);

        lts
    }

    #[test]
    fn test_state_coverage_full() {
        let lts = make_test_lts();
        let explored: BTreeSet<u32> = [0, 1, 2, 3].iter().copied().collect();
        let cov = StateCoverage::compute(&lts, &explored);
        assert!((cov.coverage_pct - 100.0).abs() < 0.01);
        assert!(cov.unexplored_states.is_empty());
    }

    #[test]
    fn test_state_coverage_partial() {
        let lts = make_test_lts();
        let explored: BTreeSet<u32> = [0, 1].iter().copied().collect();
        let cov = StateCoverage::compute(&lts, &explored);
        assert!(cov.coverage_pct > 0.0 && cov.coverage_pct < 100.0);
        assert!(!cov.unexplored_states.is_empty());
    }

    #[test]
    fn test_state_coverage_phase_breakdown() {
        let lts = make_test_lts();
        let explored: BTreeSet<u32> = [0, 1, 2].iter().copied().collect();
        let cov = StateCoverage::compute(&lts, &explored);
        assert!(cov.phase_coverage.contains_key("Initial"));
        assert!(cov.phase_coverage.contains_key("ClientHello"));
    }

    #[test]
    fn test_transition_coverage_full() {
        let lts = make_test_lts();
        let exercised: BTreeSet<u32> = [0, 1, 2].iter().copied().collect();
        let cov = TransitionCoverage::compute(&lts, &exercised);
        assert!((cov.coverage_pct - 100.0).abs() < 0.01);
        assert!(cov.uncovered_transitions.is_empty());
    }

    #[test]
    fn test_transition_coverage_partial() {
        let lts = make_test_lts();
        let exercised: BTreeSet<u32> = [0].iter().copied().collect();
        let cov = TransitionCoverage::compute(&lts, &exercised);
        assert!(cov.coverage_pct < 100.0);
        assert!(!cov.uncovered_transitions.is_empty());
    }

    #[test]
    fn test_transition_downgrade_coverage() {
        let lts = make_test_lts();
        let exercised: BTreeSet<u32> = [0, 1].iter().copied().collect();
        let cov = TransitionCoverage::compute(&lts, &exercised);
        assert_eq!(cov.downgrade_transitions_total, 1);
        assert_eq!(cov.downgrade_transitions_covered, 0);
        assert!(!cov.all_downgrade_covered());

        let exercised2: BTreeSet<u32> = [0, 1, 2].iter().copied().collect();
        let cov2 = TransitionCoverage::compute(&lts, &exercised2);
        assert!(cov2.all_downgrade_covered());
    }

    #[test]
    fn test_path_coverage() {
        let lts = make_test_lts();
        let paths = vec![vec![0, 1, 2], vec![0, 1, 3]];
        let cov = PathCoverage::compute(&lts, &paths, 10);
        assert!(cov.explored_paths > 0);
        assert!(cov.coverage_pct > 0.0);
        assert_eq!(cov.max_path_length, 3);
    }

    #[test]
    fn test_coverage_bound_validator_strict() {
        let lts = make_test_lts();
        let all_states: BTreeSet<u32> = [0, 1, 2, 3].iter().copied().collect();
        let all_trans: BTreeSet<u32> = [0, 1, 2].iter().copied().collect();
        let all_paths = vec![vec![0, 1, 2], vec![0, 1, 3]];

        let state_cov = StateCoverage::compute(&lts, &all_states);
        let path_cov = PathCoverage::compute(&lts, &all_paths, 10);
        let trans_cov = TransitionCoverage::compute(&lts, &all_trans);

        let validator = CoverageBoundValidator::strict();
        let result = validator.validate(&state_cov, &path_cov, &trans_cov);
        assert!(state_cov.meets_threshold(99.0));
        assert!(trans_cov.meets_threshold(99.0));
    }

    #[test]
    fn test_coverage_bound_validator_fails() {
        let lts = make_test_lts();
        let partial_states: BTreeSet<u32> = [0].iter().copied().collect();
        let state_cov = StateCoverage::compute(&lts, &partial_states);
        let path_cov = PathCoverage::compute(&lts, &[], 10);
        let trans_cov = TransitionCoverage::compute(&lts, &BTreeSet::new());

        let validator = CoverageBoundValidator::strict();
        let result = validator.validate(&state_cov, &path_cov, &trans_cov);
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_coverage_analyzer_full() {
        let lts = make_test_lts();
        let mut analyzer = CoverageAnalyzer::new(lts);
        analyzer.explore_full();

        let state_cov = analyzer.state_coverage();
        assert!((state_cov.coverage_pct - 100.0).abs() < 0.01);

        let trans_cov = analyzer.transition_coverage();
        assert!((trans_cov.coverage_pct - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_coverage_analyzer_random() {
        let lts = make_test_lts();
        let mut analyzer = CoverageAnalyzer::new(lts);
        analyzer.explore_random(100, 42);

        let state_cov = analyzer.state_coverage();
        assert!(state_cov.coverage_pct > 0.0);

        let trans_cov = analyzer.transition_coverage();
        assert!(trans_cov.coverage_pct > 0.0);
    }

    #[test]
    fn test_gap_analysis() {
        let lts = make_test_lts();
        let mut analyzer = CoverageAnalyzer::new(lts);
        analyzer.explored_states.insert(0);
        analyzer.explored_states.insert(1);

        let gaps = analyzer.gap_analysis();
        assert!(!gaps.uncovered_states.is_empty());
    }

    #[test]
    fn test_random_comparison() {
        let lts = make_test_lts();
        let mut analyzer = CoverageAnalyzer::new(lts);
        analyzer.explore_full();

        let comparison = analyzer.compare_with_random(10, 42);
        assert!(comparison.systematic_state_coverage >= comparison.random_state_coverage);
    }

    #[test]
    fn test_empty_lts_coverage() {
        let lts = Lts::new("empty");
        let cov = StateCoverage::compute(&lts, &BTreeSet::new());
        assert_eq!(cov.total_states, 0);
        assert_eq!(cov.coverage_pct, 0.0);
    }

    #[test]
    fn test_path_coverage_empty() {
        let lts = Lts::new("empty");
        let cov = PathCoverage::compute(&lts, &[], 10);
        assert_eq!(cov.explored_paths, 0);
    }

    #[test]
    fn test_coverage_threshold_check() {
        let lts = make_test_lts();
        let full: BTreeSet<u32> = [0, 1, 2, 3].iter().copied().collect();
        let cov = StateCoverage::compute(&lts, &full);
        assert!(cov.meets_threshold(99.0));
        assert!(cov.meets_threshold(100.0));
        assert!(!StateCoverage::compute(&lts, &BTreeSet::from([0])).meets_threshold(99.0));
    }
}
