use crate::transition_system::{RegulatoryTransitionSystem, Transition, TransitionLabel};
use crate::{RegulatoryState, StateId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};

/// A bisimulation relation represented as a partition of state IDs into equivalence classes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisimulationRelation {
    partition: Vec<BTreeSet<StateId>>,
}

impl BisimulationRelation {
    /// Create an empty relation with no blocks.
    pub fn empty() -> Self {
        Self {
            partition: Vec::new(),
        }
    }

    /// Create a relation from an explicit partition.
    pub fn from_partition(partition: Vec<BTreeSet<StateId>>) -> Self {
        Self { partition }
    }

    /// Merge the blocks containing states `a` and `b`.
    pub fn add_pair(&mut self, a: StateId, b: StateId) {
        let block_a = self.partition.iter().position(|blk| blk.contains(&a));
        let block_b = self.partition.iter().position(|blk| blk.contains(&b));

        match (block_a, block_b) {
            (Some(ia), Some(ib)) if ia == ib => {}
            (Some(ia), Some(ib)) => {
                let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
                let hi_block = self.partition.remove(hi);
                self.partition[lo].extend(hi_block);
            }
            (Some(ia), None) => {
                self.partition[ia].insert(b);
            }
            (None, Some(ib)) => {
                self.partition[ib].insert(a);
            }
            (None, None) => {
                let mut blk = BTreeSet::new();
                blk.insert(a);
                blk.insert(b);
                self.partition.push(blk);
            }
        }
    }

    /// Check if two states are in the same equivalence class.
    pub fn are_bisimilar(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        self.partition.iter().any(|blk| blk.contains(a) && blk.contains(b))
    }

    /// Return the index of the block containing this state, if any.
    pub fn block_of(&self, state: &str) -> Option<usize> {
        self.partition.iter().position(|blk| blk.contains(state))
    }

    /// Return the partition.
    pub fn partition(&self) -> &[BTreeSet<StateId>] {
        &self.partition
    }

    /// Number of equivalence classes.
    pub fn num_blocks(&self) -> usize {
        self.partition.len()
    }
}

/// Compute the coarsest bisimulation relation on a regulatory transition system
/// using partition refinement.
pub fn compute_bisimulation(sys: &RegulatoryTransitionSystem) -> BisimulationRelation {
    if sys.states.is_empty() {
        return BisimulationRelation::empty();
    }

    // Initial partition: group by obligation set
    let mut obl_to_block: HashMap<BTreeSet<String>, Vec<StateId>> = HashMap::new();
    for (sid, state) in &sys.states {
        obl_to_block
            .entry(state.obligations.clone())
            .or_default()
            .push(sid.clone());
    }
    let mut partition: Vec<BTreeSet<StateId>> = obl_to_block
        .into_values()
        .map(|v| v.into_iter().collect())
        .collect();

    // Iteratively refine until stable
    loop {
        let new_partition = refine_partition(&partition, sys);
        if new_partition.len() == partition.len() {
            break;
        }
        partition = new_partition;
    }

    BisimulationRelation::from_partition(partition)
}

/// One step of partition refinement.
pub fn refine_partition(
    partition: &[BTreeSet<StateId>],
    sys: &RegulatoryTransitionSystem,
) -> Vec<BTreeSet<StateId>> {
    let mut state_to_block: HashMap<&str, usize> = HashMap::new();
    for (idx, blk) in partition.iter().enumerate() {
        for s in blk {
            state_to_block.insert(s.as_str(), idx);
        }
    }

    let mut new_partition = Vec::new();

    for blk in partition {
        let mut sig_map: HashMap<Vec<(String, Vec<usize>)>, BTreeSet<StateId>> = HashMap::new();

        for state_id in blk {
            let mut sig: HashMap<String, BTreeSet<usize>> = HashMap::new();
            for t in sys.successors(state_id) {
                let label = format!("{:?}", TransitionLabel::from_event(&t.event));
                if let Some(&target_block) = state_to_block.get(t.to.as_str()) {
                    sig.entry(label).or_default().insert(target_block);
                }
            }
            let mut sig_vec: Vec<(String, Vec<usize>)> = sig
                .into_iter()
                .map(|(l, bs)| (l, bs.into_iter().collect()))
                .collect();
            sig_vec.sort();

            sig_map
                .entry(sig_vec)
                .or_insert_with(BTreeSet::new)
                .insert(state_id.clone());
        }

        for (_, sub_block) in sig_map {
            new_partition.push(sub_block);
        }
    }

    new_partition
}

/// Build a quotient system by collapsing bisimilar states.
pub fn quotient_system(
    sys: &RegulatoryTransitionSystem,
    relation: &BisimulationRelation,
) -> RegulatoryTransitionSystem {
    let mut quotient = RegulatoryTransitionSystem::new();

    let mut state_to_block: HashMap<String, usize> = HashMap::new();
    let mut block_ids: Vec<String> = Vec::new();

    for (idx, blk) in relation.partition().iter().enumerate() {
        let block_id = format!("block-{}", idx);
        block_ids.push(block_id.clone());

        let mut obligations = BTreeSet::new();
        for sid in blk {
            state_to_block.insert(sid.clone(), idx);
            if let Some(state) = sys.states.get(sid) {
                obligations.extend(state.obligations.iter().cloned());
            }
        }

        quotient.add_state(RegulatoryState::with_obligations(&block_id, obligations));
    }

    if let Some(ref init) = sys.initial_state {
        if let Some(&block_idx) = state_to_block.get(init) {
            let _ = quotient.set_initial_state(&block_ids[block_idx]);
        }
    }

    let mut seen_transitions: HashSet<(usize, usize, String)> = HashSet::new();
    for t in &sys.transitions {
        if let (Some(&from_block), Some(&to_block)) = (
            state_to_block.get(&t.from),
            state_to_block.get(&t.to),
        ) {
            let label = format!("{:?}", TransitionLabel::from_event(&t.event));
            if seen_transitions.insert((from_block, to_block, label)) {
                quotient.add_transition_unchecked(Transition {
                    from: block_ids[from_block].clone(),
                    to: block_ids[to_block].clone(),
                    event: t.event.clone(),
                });
            }
        }
    }

    quotient
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RegulatoryEvent, ymd};

    fn make_system() -> RegulatoryTransitionSystem {
        let mut sys = RegulatoryTransitionSystem::new();

        let s0 = RegulatoryState::with_obligations(
            "s0",
            ["A".to_string()].into_iter().collect(),
        );
        let s1 = RegulatoryState::with_obligations(
            "s1",
            ["A".to_string()].into_iter().collect(),
        );
        let s2 = RegulatoryState::with_obligations(
            "s2",
            ["A".to_string(), "B".to_string()].into_iter().collect(),
        );
        let s3 = RegulatoryState::with_obligations(
            "s3",
            ["B".to_string()].into_iter().collect(),
        );

        sys.add_state(s0);
        sys.add_state(s1);
        sys.add_state(s2);
        sys.add_state(s3);

        sys.add_transition_unchecked(Transition {
            from: "s0".into(),
            to: "s2".into(),
            event: RegulatoryEvent::PhaseIn {
                milestone: "M1".into(),
                date: ymd(2025, 1, 1),
            },
        });
        sys.add_transition_unchecked(Transition {
            from: "s1".into(),
            to: "s2".into(),
            event: RegulatoryEvent::PhaseIn {
                milestone: "M1".into(),
                date: ymd(2025, 1, 1),
            },
        });
        sys.add_transition_unchecked(Transition {
            from: "s2".into(),
            to: "s3".into(),
            event: RegulatoryEvent::Sunset {
                description: "sunset A".into(),
                date: ymd(2026, 1, 1),
            },
        });

        sys
    }

    #[test]
    fn test_bisimulation_empty_system() {
        let sys = RegulatoryTransitionSystem::new();
        let rel = compute_bisimulation(&sys);
        assert_eq!(rel.num_blocks(), 0);
    }

    #[test]
    fn test_bisimulation_groups_equivalent_states() {
        let sys = make_system();
        let rel = compute_bisimulation(&sys);
        assert!(rel.are_bisimilar("s0", "s1"));
        assert!(!rel.are_bisimilar("s0", "s2"));
        assert!(!rel.are_bisimilar("s2", "s3"));
    }

    #[test]
    fn test_bisimulation_num_blocks() {
        let sys = make_system();
        let rel = compute_bisimulation(&sys);
        assert_eq!(rel.num_blocks(), 3);
    }

    #[test]
    fn test_block_of() {
        let sys = make_system();
        let rel = compute_bisimulation(&sys);
        let b0 = rel.block_of("s0").unwrap();
        let b1 = rel.block_of("s1").unwrap();
        assert_eq!(b0, b1);
    }

    #[test]
    fn test_quotient_system() {
        let sys = make_system();
        let rel = compute_bisimulation(&sys);
        let q = quotient_system(&sys, &rel);
        assert_eq!(q.state_count(), 3);
        assert_eq!(q.transition_count(), 2);
    }

    #[test]
    fn test_add_pair() {
        let mut rel = BisimulationRelation::empty();
        rel.add_pair("a".to_string(), "b".to_string());
        assert!(rel.are_bisimilar("a", "b"));
        assert_eq!(rel.num_blocks(), 1);

        rel.add_pair("c".to_string(), "d".to_string());
        assert_eq!(rel.num_blocks(), 2);

        rel.add_pair("b".to_string(), "c".to_string());
        assert!(rel.are_bisimilar("a", "d"));
        assert_eq!(rel.num_blocks(), 1);
    }

    #[test]
    fn test_are_bisimilar_self() {
        let rel = BisimulationRelation::empty();
        assert!(rel.are_bisimilar("x", "x"));
    }

    #[test]
    fn test_refine_partition_no_change() {
        let sys = make_system();
        let rel = compute_bisimulation(&sys);
        let refined = refine_partition(rel.partition(), &sys);
        assert_eq!(refined.len(), rel.num_blocks());
    }

    #[test]
    fn test_quotient_preserves_reachability() {
        let sys = make_system();
        let rel = compute_bisimulation(&sys);
        let q = quotient_system(&sys, &rel);
        let reachable = q.compute_reachable_states();
        assert_eq!(reachable.len(), q.state_count());
    }

    #[test]
    fn test_all_distinct_no_merge() {
        let mut sys = RegulatoryTransitionSystem::new();
        sys.add_state(RegulatoryState::with_obligations(
            "s0",
            ["X".to_string()].into_iter().collect(),
        ));
        sys.add_state(RegulatoryState::with_obligations(
            "s1",
            ["Y".to_string()].into_iter().collect(),
        ));
        sys.add_state(RegulatoryState::with_obligations(
            "s2",
            ["Z".to_string()].into_iter().collect(),
        ));

        let rel = compute_bisimulation(&sys);
        assert_eq!(rel.num_blocks(), 3);
        assert!(!rel.are_bisimilar("s0", "s1"));
    }
}
