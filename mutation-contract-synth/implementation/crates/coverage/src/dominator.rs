//! Dominator set computation (Theorem T2).
//!
//! The **dominator set** D is a minimal subset of the killed mutants such that
//! the specification inferred from D is the same as the specification inferred
//! from all killed mutants: `phi_M(D) = phi_M(MKill)`.
//!
//! This achieves a reduction from `|MKill| = O(nk)` to `|D| = O(n)`.

use crate::subsumption::{SubsumptionConfidence, SubsumptionGraph};
use crate::{CoverageError, KillMatrix, MutantDescriptor, MutantId, MutationOperator, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// DominatorSet
// ---------------------------------------------------------------------------

/// A dominator set: minimal mutants whose combined coverage equals full MKill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominatorSet {
    /// Mutant IDs in the dominator set, in selection order.
    pub members: Vec<MutantId>,
    /// The set of all test indices covered by the dominator set.
    pub covered_tests: BTreeSet<usize>,
    /// For each dominator, the set of non-dominator mutants it represents.
    pub representation: BTreeMap<MutantId, BTreeSet<MutantId>>,
    /// Total number of killed mutants in the original set.
    pub total_killed: usize,
    /// Algorithm used to compute this set.
    pub algorithm: DominatorAlgorithm,
}

/// Which algorithm produced the dominator set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DominatorAlgorithm {
    Greedy,
    Optimal,
    SubsumptionBased,
}

impl fmt::Display for DominatorAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greedy => write!(f, "greedy"),
            Self::Optimal => write!(f, "optimal"),
            Self::SubsumptionBased => write!(f, "subsumption-based"),
        }
    }
}

impl DominatorSet {
    /// Number of dominators.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Check membership.
    pub fn contains(&self, id: &MutantId) -> bool {
        self.members.contains(id)
    }

    /// Reduction factor: `|MKill| / |D|`.
    pub fn reduction_factor(&self) -> f64 {
        if self.members.is_empty() {
            0.0
        } else {
            self.total_killed as f64 / self.members.len() as f64
        }
    }

    /// Percentage of killed mutants retained.
    pub fn retention_rate(&self) -> f64 {
        if self.total_killed == 0 {
            0.0
        } else {
            self.members.len() as f64 / self.total_killed as f64 * 100.0
        }
    }

    /// Get the mutants represented by a dominator.
    pub fn represented_by(&self, id: &MutantId) -> Option<&BTreeSet<MutantId>> {
        self.representation.get(id)
    }

    /// Find which dominator represents a given mutant.
    pub fn dominator_of(&self, mutant: &MutantId) -> Option<&MutantId> {
        for (dom, represented) in &self.representation {
            if represented.contains(mutant) || dom == mutant {
                return Some(dom);
            }
        }
        None
    }

    /// Every test that detects any killed mutant also detects a dominator.
    pub fn validate(&self, matrix: &KillMatrix) -> bool {
        let killed = matrix.killed_set();
        let dom_indices: BTreeSet<usize> = self
            .members
            .iter()
            .filter_map(|id| matrix.mutant_index(id))
            .collect();
        for t in 0..matrix.num_tests() {
            let detects_any = killed.iter().any(|&m| matrix.get(t, m).is_kill());
            if detects_any {
                let detects_dom = dom_indices.iter().any(|&m| matrix.get(t, m).is_kill());
                if !detects_dom {
                    return false;
                }
            }
        }
        true
    }

    /// Stronger: same test specification as full killed set.
    pub fn validate_specification_equivalence(&self, matrix: &KillMatrix) -> bool {
        let killed = matrix.killed_set();
        let full_spec: BTreeSet<usize> = killed
            .iter()
            .flat_map(|&m| matrix.killing_tests(m))
            .collect();
        let dom_spec: BTreeSet<usize> = self
            .members
            .iter()
            .filter_map(|id| matrix.mutant_index(id))
            .flat_map(|m| matrix.killing_tests(m))
            .collect();
        full_spec == dom_spec
    }
}

// ---------------------------------------------------------------------------
// Quality & Statistics
// ---------------------------------------------------------------------------

/// Quality metrics for a dominator set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominatorQuality {
    pub dominator_size: usize,
    pub total_killed: usize,
    pub reduction_factor: f64,
    pub is_valid: bool,
    pub spec_equivalent: bool,
    pub test_coverage: f64,
    pub avg_representation: f64,
    pub max_representation: usize,
    pub min_representation: usize,
}

/// Statistics about dominator set computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominatorStats {
    pub algorithm: String,
    pub input_killed: usize,
    pub dominator_size: usize,
    pub reduction_factor: f64,
    pub computation_time_ms: u64,
    pub per_operator: BTreeMap<String, OperatorDominatorStats>,
    pub quality: DominatorQuality,
}

/// Per-operator dominator statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperatorDominatorStats {
    pub total_killed: usize,
    pub dominators: usize,
    pub reduction_factor: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominatorConfig {
    pub algorithm: DominatorAlgorithm,
    pub optimal_threshold: usize,
    pub validate: bool,
    pub use_subsumption: bool,
    pub per_operator_stats: bool,
}

impl Default for DominatorConfig {
    fn default() -> Self {
        Self {
            algorithm: DominatorAlgorithm::Greedy,
            optimal_threshold: 20,
            validate: true,
            use_subsumption: false,
            per_operator_stats: true,
        }
    }
}

// ---------------------------------------------------------------------------
// DominatorSetComputer
// ---------------------------------------------------------------------------

/// Computes dominator sets from a kill matrix and optional subsumption graph.
pub struct DominatorSetComputer {
    config: DominatorConfig,
    subsumption_graph: Option<SubsumptionGraph>,
    descriptors: HashMap<MutantId, MutantDescriptor>,
}

impl DominatorSetComputer {
    pub fn new() -> Self {
        Self {
            config: DominatorConfig::default(),
            subsumption_graph: None,
            descriptors: HashMap::new(),
        }
    }

    pub fn with_config(config: DominatorConfig) -> Self {
        Self {
            config,
            subsumption_graph: None,
            descriptors: HashMap::new(),
        }
    }

    pub fn set_subsumption_graph(&mut self, graph: SubsumptionGraph) {
        self.subsumption_graph = Some(graph);
    }

    pub fn register_descriptors(&mut self, descs: Vec<MutantDescriptor>) {
        for d in descs {
            self.descriptors.insert(d.id.clone(), d);
        }
    }

    /// Main entry point: compute the dominator set.
    pub fn compute(&self, km: &KillMatrix) -> Result<DominatorSet> {
        let killed = km.killed_set();
        if killed.is_empty() {
            return Ok(DominatorSet {
                members: Vec::new(),
                covered_tests: BTreeSet::new(),
                representation: BTreeMap::new(),
                total_killed: 0,
                algorithm: self.config.algorithm,
            });
        }
        if self.config.use_subsumption && self.subsumption_graph.is_some() {
            self.compute_from_subsumption(km)
        } else {
            match self.config.algorithm {
                DominatorAlgorithm::Optimal if killed.len() <= self.config.optimal_threshold => {
                    self.compute_optimal(km)
                }
                _ => self.compute_greedy(km),
            }
        }
    }

    /// Compute dominator set with detailed statistics.
    pub fn compute_with_stats(&self, km: &KillMatrix) -> Result<(DominatorSet, DominatorStats)> {
        let start = std::time::Instant::now();
        let dom = self.compute(km)?;
        let elapsed = start.elapsed().as_millis() as u64;
        let quality = self.assess_quality(&dom, km);
        let per_operator = if self.config.per_operator_stats {
            self.compute_operator_stats(&dom, km)
        } else {
            BTreeMap::new()
        };
        let stats = DominatorStats {
            algorithm: format!("{}", dom.algorithm),
            input_killed: dom.total_killed,
            dominator_size: dom.size(),
            reduction_factor: dom.reduction_factor(),
            computation_time_ms: elapsed,
            per_operator,
            quality,
        };
        Ok((dom, stats))
    }

    // -- Greedy set-cover -------------------------------------------------

    fn compute_greedy(&self, km: &KillMatrix) -> Result<DominatorSet> {
        let killed: Vec<usize> = km.killed_set().into_iter().collect();
        let ks: Vec<BTreeSet<usize>> = killed.iter().map(|&m| km.killing_tests(m)).collect();
        let universe: BTreeSet<usize> = ks.iter().flat_map(|s| s.iter().copied()).collect();
        let mut uncovered = universe.clone();
        let mut sel = Vec::new();
        let mut sel_set = HashSet::new();

        while !uncovered.is_empty() {
            let best = killed
                .iter()
                .enumerate()
                .filter(|(i, _)| !sel_set.contains(i))
                .max_by_key(|(i, _)| ks[*i].intersection(&uncovered).count());
            match best {
                Some((idx, _)) => {
                    sel.push(idx);
                    sel_set.insert(idx);
                    for t in &ks[idx] {
                        uncovered.remove(t);
                    }
                }
                None => break,
            }
        }

        let members: Vec<MutantId> = sel.iter().map(|&i| km.mutants[killed[i]].clone()).collect();
        let representation = self.build_representation(&members, &killed, km);
        Ok(DominatorSet {
            members,
            covered_tests: universe,
            representation,
            total_killed: killed.len(),
            algorithm: DominatorAlgorithm::Greedy,
        })
    }

    // -- Optimal (brute-force) --------------------------------------------

    fn compute_optimal(&self, km: &KillMatrix) -> Result<DominatorSet> {
        let killed: Vec<usize> = km.killed_set().into_iter().collect();
        let ks: Vec<BTreeSet<usize>> = killed.iter().map(|&m| km.killing_tests(m)).collect();
        let universe: BTreeSet<usize> = ks.iter().flat_map(|s| s.iter().copied()).collect();
        let n = killed.len();
        if n == 0 {
            return Ok(DominatorSet {
                members: Vec::new(),
                covered_tests: BTreeSet::new(),
                representation: BTreeMap::new(),
                total_killed: 0,
                algorithm: DominatorAlgorithm::Optimal,
            });
        }
        let mut best_subset: Option<Vec<usize>> = None;
        'outer: for size in 1..=n {
            for combo in combinations(n, size) {
                let covered: BTreeSet<usize> =
                    combo.iter().flat_map(|&i| ks[i].iter().copied()).collect();
                if covered == universe {
                    best_subset = Some(combo);
                    break 'outer;
                }
            }
        }
        let selected = best_subset.unwrap_or_else(|| (0..n).collect());
        let members: Vec<MutantId> = selected
            .iter()
            .map(|&i| km.mutants[killed[i]].clone())
            .collect();
        let representation = self.build_representation(&members, &killed, km);
        Ok(DominatorSet {
            members,
            covered_tests: universe,
            representation,
            total_killed: killed.len(),
            algorithm: DominatorAlgorithm::Optimal,
        })
    }

    // -- Subsumption-based ------------------------------------------------

    fn compute_from_subsumption(&self, km: &KillMatrix) -> Result<DominatorSet> {
        let graph = self
            .subsumption_graph
            .as_ref()
            .ok_or_else(|| CoverageError::ComputationError("no subsumption graph".into()))?;
        let killed = km.killed_set();
        let killed_ids: BTreeSet<MutantId> =
            killed.iter().map(|&i| km.mutants[i].clone()).collect();

        let mut roots: Vec<MutantId> = graph
            .roots()
            .into_iter()
            .filter(|id| killed_ids.contains(id))
            .collect();
        for id in &killed_ids {
            if graph.node_index(id).is_none() && !roots.contains(id) {
                roots.push(id.clone());
            }
        }

        let universe: BTreeSet<usize> = killed.iter().flat_map(|&m| km.killing_tests(m)).collect();
        let mut covered: BTreeSet<usize> = roots
            .iter()
            .filter_map(|id| km.mutant_index(id))
            .flat_map(|m| km.killing_tests(m))
            .collect();

        if covered != universe {
            let kv: Vec<usize> = killed.iter().copied().collect();
            let ks: Vec<BTreeSet<usize>> = kv.iter().map(|&m| km.killing_tests(m)).collect();
            let root_set: HashSet<usize> =
                roots.iter().filter_map(|id| km.mutant_index(id)).collect();
            let mut unc: BTreeSet<usize> = universe.difference(&covered).copied().collect();
            while !unc.is_empty() {
                let best = kv
                    .iter()
                    .enumerate()
                    .filter(|(_, &m)| !root_set.contains(&m))
                    .max_by_key(|(i, _)| ks[*i].intersection(&unc).count());
                match best {
                    Some((idx, &m)) => {
                        roots.push(km.mutants[m].clone());
                        for t in &ks[idx] {
                            unc.remove(t);
                            covered.insert(*t);
                        }
                    }
                    None => break,
                }
            }
        }

        let kv: Vec<usize> = killed.iter().copied().collect();
        let representation = self.build_representation(&roots, &kv, km);
        Ok(DominatorSet {
            members: roots,
            covered_tests: universe,
            representation,
            total_killed: killed.len(),
            algorithm: DominatorAlgorithm::SubsumptionBased,
        })
    }

    // -- Quality ----------------------------------------------------------

    /// Assess quality of a dominator set.
    pub fn assess_quality(&self, dom: &DominatorSet, km: &KillMatrix) -> DominatorQuality {
        let is_valid = dom.validate(km);
        let spec_equivalent = dom.validate_specification_equivalence(km);
        let test_coverage = if km.num_tests() == 0 {
            0.0
        } else {
            dom.covered_tests.len() as f64 / km.num_tests() as f64
        };
        let rs: Vec<usize> = dom.representation.values().map(|s| s.len()).collect();
        let avg = if rs.is_empty() {
            0.0
        } else {
            rs.iter().sum::<usize>() as f64 / rs.len() as f64
        };
        DominatorQuality {
            dominator_size: dom.size(),
            total_killed: dom.total_killed,
            reduction_factor: dom.reduction_factor(),
            is_valid,
            spec_equivalent,
            test_coverage,
            avg_representation: avg,
            max_representation: rs.iter().copied().max().unwrap_or(0),
            min_representation: rs.iter().copied().min().unwrap_or(0),
        }
    }

    // -- Per-operator stats -----------------------------------------------

    fn compute_operator_stats(
        &self,
        dom: &DominatorSet,
        km: &KillMatrix,
    ) -> BTreeMap<String, OperatorDominatorStats> {
        let mut stats: BTreeMap<String, OperatorDominatorStats> = BTreeMap::new();
        for &m in &km.killed_set() {
            if let Some(desc) = self.descriptors.get(&km.mutants[m]) {
                stats
                    .entry(desc.operator.short_name().to_string())
                    .or_default()
                    .total_killed += 1;
            }
        }
        for id in &dom.members {
            if let Some(desc) = self.descriptors.get(id) {
                stats
                    .entry(desc.operator.short_name().to_string())
                    .or_default()
                    .dominators += 1;
            }
        }
        for s in stats.values_mut() {
            s.reduction_factor = if s.dominators == 0 {
                0.0
            } else {
                s.total_killed as f64 / s.dominators as f64
            };
        }
        stats
    }

    // -- Representation ---------------------------------------------------

    fn build_representation(
        &self,
        members: &[MutantId],
        killed_indices: &[usize],
        km: &KillMatrix,
    ) -> BTreeMap<MutantId, BTreeSet<MutantId>> {
        let member_set: HashSet<&MutantId> = members.iter().collect();
        let mut repr: BTreeMap<MutantId, BTreeSet<MutantId>> = BTreeMap::new();
        for id in members {
            repr.insert(id.clone(), BTreeSet::new());
        }
        for &m_idx in killed_indices {
            let m_id = &km.mutants[m_idx];
            if member_set.contains(m_id) {
                continue;
            }
            let m_ks = km.killing_tests(m_idx);
            let best = members.iter().max_by_key(|did| {
                km.mutant_index(did)
                    .map(|di| {
                        let dk = km.killing_tests(di);
                        m_ks.intersection(&dk).count()
                    })
                    .unwrap_or(0)
            });
            if let Some(dom) = best {
                repr.entry(dom.clone()).or_default().insert(m_id.clone());
            }
        }
        repr
    }

    /// Check if update is needed.
    pub fn needs_update(&self, existing: &DominatorSet, km: &KillMatrix) -> bool {
        !existing.validate_specification_equivalence(km)
    }

    /// Incrementally update a dominator set.
    pub fn update_incremental(
        &self,
        existing: &DominatorSet,
        km: &KillMatrix,
    ) -> Result<DominatorSet> {
        if !self.needs_update(existing, km) {
            return Ok(existing.clone());
        }
        self.compute(km)
    }

    /// Compare two dominator sets.
    pub fn compare(a: &DominatorSet, b: &DominatorSet) -> DominatorComparison {
        let a_set: BTreeSet<MutantId> = a.members.iter().cloned().collect();
        let b_set: BTreeSet<MutantId> = b.members.iter().cloned().collect();
        let common: BTreeSet<MutantId> = a_set.intersection(&b_set).cloned().collect();
        let only_a: BTreeSet<MutantId> = a_set.difference(&b_set).cloned().collect();
        let only_b: BTreeSet<MutantId> = b_set.difference(&a_set).cloned().collect();
        let jaccard = if a_set.is_empty() && b_set.is_empty() {
            1.0
        } else {
            let u = a_set.union(&b_set).count();
            common.len() as f64 / u as f64
        };
        DominatorComparison {
            size_a: a.size(),
            size_b: b.size(),
            common,
            only_in_a: only_a,
            only_in_b: only_b,
            jaccard_similarity: jaccard,
        }
    }
}

impl Default for DominatorSetComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison between two dominator sets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominatorComparison {
    pub size_a: usize,
    pub size_b: usize,
    pub common: BTreeSet<MutantId>,
    pub only_in_a: BTreeSet<MutantId>,
    pub only_in_b: BTreeSet<MutantId>,
    pub jaccard_similarity: f64,
}

// -- Combination helper ---------------------------------------------------

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }
    let mut result = Vec::new();
    let mut cur = Vec::with_capacity(k);
    comb_rec(n, k, 0, &mut cur, &mut result);
    result
}

fn comb_rec(n: usize, k: usize, start: usize, cur: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if cur.len() == k {
        result.push(cur.clone());
        return;
    }
    let rem = k - cur.len();
    for i in start..=(n - rem) {
        cur.push(i);
        comb_rec(n, k, i + 1, cur, result);
        cur.pop();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_kill_matrix, make_test_mutant, MutantId, MutationOperator};

    fn standard_km() -> KillMatrix {
        //       m0  m1  m2  m3  m4
        // t0: [  K   K   .   .   . ]
        // t1: [  K   .   K   .   . ]
        // t2: [  .   K   .   K   . ]
        // t3: [  .   .   K   K   K ]
        make_test_kill_matrix(
            4,
            5,
            &[
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (3, 3),
                (3, 4),
            ],
        )
    }

    #[test]
    fn test_greedy_basic() {
        let km = standard_km();
        let dom = DominatorSetComputer::new().compute(&km).unwrap();
        assert!(dom.size() > 0 && dom.size() <= km.num_mutants());
        assert!(dom.validate(&km));
        assert!(dom.validate_specification_equivalence(&km));
    }

    #[test]
    fn test_greedy_covers_all() {
        let km = standard_km();
        let dom = DominatorSetComputer::new().compute(&km).unwrap();
        let all: BTreeSet<usize> = (0..km.num_tests())
            .filter(|&t| (0..km.num_mutants()).any(|m| km.get(t, m).is_kill()))
            .collect();
        assert_eq!(dom.covered_tests, all);
    }

    #[test]
    fn test_optimal_small() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]);
        let cfg = DominatorConfig {
            algorithm: DominatorAlgorithm::Optimal,
            optimal_threshold: 10,
            ..Default::default()
        };
        let dom = DominatorSetComputer::with_config(cfg).compute(&km).unwrap();
        assert!(dom.validate_specification_equivalence(&km));
    }

    #[test]
    fn test_reduction_factor() {
        let dom = DominatorSetComputer::new().compute(&standard_km()).unwrap();
        assert!(dom.reduction_factor() >= 1.0);
    }

    #[test]
    fn test_empty() {
        assert_eq!(
            DominatorSetComputer::new()
                .compute(&make_test_kill_matrix(0, 0, &[]))
                .unwrap()
                .size(),
            0
        );
    }

    #[test]
    fn test_no_kills_result() {
        assert_eq!(
            DominatorSetComputer::new()
                .compute(&make_test_kill_matrix(3, 3, &[]))
                .unwrap()
                .size(),
            0
        );
    }

    #[test]
    fn test_single_mutant() {
        assert_eq!(
            DominatorSetComputer::new()
                .compute(&make_test_kill_matrix(2, 1, &[(0, 0), (1, 0)]))
                .unwrap()
                .size(),
            1
        );
    }

    #[test]
    fn test_identical_sets() {
        let km = make_test_kill_matrix(
            2,
            4,
            &[
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ],
        );
        let dom = DominatorSetComputer::new().compute(&km).unwrap();
        assert_eq!(dom.size(), 1);
        assert!(dom.validate_specification_equivalence(&km));
    }

    #[test]
    fn test_disjoint() {
        assert_eq!(
            DominatorSetComputer::new()
                .compute(&make_test_kill_matrix(3, 3, &[(0, 0), (1, 1), (2, 2)]))
                .unwrap()
                .size(),
            3
        );
    }

    #[test]
    fn test_dominator_of() {
        let km = standard_km();
        let dom = DominatorSetComputer::new().compute(&km).unwrap();
        for m in km.killed_set() {
            assert!(dom.dominator_of(&km.mutants[m]).is_some());
        }
    }

    #[test]
    fn test_quality() {
        let km = standard_km();
        let c = DominatorSetComputer::new();
        let dom = c.compute(&km).unwrap();
        let q = c.assess_quality(&dom, &km);
        assert!(q.is_valid && q.spec_equivalent);
    }

    #[test]
    fn test_stats() {
        let (dom, stats) = DominatorSetComputer::new()
            .compute_with_stats(&standard_km())
            .unwrap();
        assert_eq!(stats.dominator_size, dom.size());
    }

    #[test]
    fn test_compare_sets() {
        let km = standard_km();
        let a = DominatorSetComputer::new().compute(&km).unwrap();
        let cfg = DominatorConfig {
            algorithm: DominatorAlgorithm::Optimal,
            optimal_threshold: 10,
            ..Default::default()
        };
        let b = DominatorSetComputer::with_config(cfg).compute(&km).unwrap();
        let c = DominatorSetComputer::compare(&a, &b);
        assert!(c.jaccard_similarity >= 0.0 && c.jaccard_similarity <= 1.0);
    }

    #[test]
    fn test_needs_update_false() {
        let km = standard_km();
        let c = DominatorSetComputer::new();
        let dom = c.compute(&km).unwrap();
        assert!(!c.needs_update(&dom, &km));
    }

    #[test]
    fn test_subsumption_based() {
        let km = standard_km();
        let mut g = SubsumptionGraph::with_mutants(&km.mutants);
        g.add_edge(
            MutantId::new("m0"),
            MutantId::new("m4"),
            SubsumptionConfidence::Dynamic,
        );
        let cfg = DominatorConfig {
            use_subsumption: true,
            ..Default::default()
        };
        let mut c = DominatorSetComputer::with_config(cfg);
        c.set_subsumption_graph(g);
        assert!(c
            .compute(&km)
            .unwrap()
            .validate_specification_equivalence(&km));
    }

    #[test]
    fn test_combinations_fn() {
        assert_eq!(combinations(4, 2).len(), 6);
        assert_eq!(combinations(5, 0), vec![vec![]]);
        assert_eq!(combinations(3, 4), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn test_validation_failure() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        let fake = DominatorSet {
            members: vec![MutantId::new("m0")],
            covered_tests: BTreeSet::from([0]),
            representation: BTreeMap::new(),
            total_killed: 3,
            algorithm: DominatorAlgorithm::Greedy,
        };
        assert!(!fake.validate(&km));
    }

    #[test]
    fn test_retention() {
        let r = DominatorSetComputer::new()
            .compute(&standard_km())
            .unwrap()
            .retention_rate();
        assert!(r > 0.0 && r <= 100.0);
    }

    #[test]
    fn test_op_stats() {
        let km = standard_km();
        let mut c = DominatorSetComputer::new();
        for (i, id) in km.mutants.iter().enumerate() {
            let op = if i % 2 == 0 {
                MutationOperator::AOR
            } else {
                MutationOperator::ROR
            };
            c.descriptors
                .insert(id.clone(), make_test_mutant(id.as_str(), op));
        }
        let (_, st) = c.compute_with_stats(&km).unwrap();
        assert!(!st.per_operator.is_empty());
    }

    #[test]
    fn test_incremental_noop() {
        let km = standard_km();
        let c = DominatorSetComputer::new();
        let dom = c.compute(&km).unwrap();
        let u = c.update_incremental(&dom, &km).unwrap();
        assert_eq!(dom.members, u.members);
    }
}
