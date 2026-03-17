use crate::{ObligationId, RegulatoryState};
use regsynth_types::{Constraint, ConstraintExpr, ConstraintSet, VarId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// Tracks how constraints change between two regulatory states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintEvolution {
    pub added: Vec<ObligationId>,
    pub removed: Vec<ObligationId>,
    pub modified: Vec<ObligationId>,
}

impl ConstraintEvolution {
    pub fn empty() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
            modified: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    /// Compute the evolution between two regulatory states.
    pub fn diff(before: &RegulatoryState, after: &RegulatoryState) -> Self {
        let added: Vec<ObligationId> = after
            .obligations
            .difference(&before.obligations)
            .cloned()
            .collect();
        let removed: Vec<ObligationId> = before
            .obligations
            .difference(&after.obligations)
            .cloned()
            .collect();
        Self {
            added,
            removed,
            modified: Vec::new(),
        }
    }

    /// Total number of changes (added + removed + modified).
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }
}

/// A timeline of constraint sets indexed by discrete timestep.
#[derive(Debug, Clone)]
pub struct ConstraintTimeline {
    steps: BTreeMap<usize, ConstraintSet>,
}

impl ConstraintTimeline {
    pub fn new() -> Self {
        Self {
            steps: BTreeMap::new(),
        }
    }

    /// Add or replace the constraint set at a given timestep.
    pub fn add_step(&mut self, timestep: usize, constraints: ConstraintSet) {
        self.steps.insert(timestep, constraints);
    }

    /// Unroll the timeline up to `horizon` timesteps, filling gaps with the
    /// most recent constraint set (or empty if none precedes).
    pub fn unroll(&self, horizon: usize) -> Vec<ConstraintSet> {
        let mut result = Vec::with_capacity(horizon);
        let mut current = ConstraintSet::new();
        for t in 0..horizon {
            if let Some(cs) = self.steps.get(&t) {
                current = cs.clone();
            }
            result.push(current.clone());
        }
        result
    }

    /// Generate budget constraints for each timestep.
    pub fn compute_budget_constraints(&self, budget_per_step: f64) -> Vec<Constraint> {
        let mut constraints = Vec::new();
        for (&timestep, _cs) in &self.steps {
            let var_name = format!("cost@{}", timestep);
            let c = Constraint::soft(
                &format!("budget-t{}", timestep),
                ConstraintExpr::Compare(
                    regsynth_types::CompareOp::Le,
                    Box::new(regsynth_types::ArithExpr::Var(VarId::new(&var_name))),
                    Box::new(regsynth_types::ArithExpr::Const(budget_per_step)),
                ),
                1.0,
            )
            .with_description(&format!(
                "Budget constraint at timestep {}: cost <= {}",
                timestep, budget_per_step
            ));
            constraints.push(c);
        }
        constraints
    }

    /// Detect temporal conflicts: obligations that appear and disappear
    /// across consecutive timesteps, suggesting instability.
    pub fn detect_temporal_conflicts(&self) -> Vec<String> {
        let mut conflicts = Vec::new();
        let sorted_steps: Vec<usize> = self.steps.keys().cloned().collect();

        for window in sorted_steps.windows(2) {
            let t1 = window[0];
            let t2 = window[1];
            let cs1 = &self.steps[&t1];
            let cs2 = &self.steps[&t2];

            let sources1: HashSet<String> = cs1
                .all()
                .iter()
                .filter_map(|c| c.source_obligation.clone())
                .collect();
            let sources2: HashSet<String> = cs2
                .all()
                .iter()
                .filter_map(|c| c.source_obligation.clone())
                .collect();

            for removed in sources1.difference(&sources2) {
                conflicts.push(format!(
                    "Obligation '{}' present at t={} but absent at t={}",
                    removed, t1, t2
                ));
            }
        }
        conflicts
    }
}

impl Default for ConstraintTimeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RegulatoryState;

    fn state(id: &str, obls: &[&str]) -> RegulatoryState {
        let obligations = obls.iter().map(|s| s.to_string()).collect();
        RegulatoryState::with_obligations(id, obligations)
    }

    #[test]
    fn test_evolution_empty() {
        let e = ConstraintEvolution::empty();
        assert!(e.is_empty());
        assert_eq!(e.total_changes(), 0);
    }

    #[test]
    fn test_evolution_diff() {
        let before = state("s1", &["a", "b", "c"]);
        let after = state("s2", &["b", "c", "d", "e"]);
        let evo = ConstraintEvolution::diff(&before, &after);
        assert_eq!(evo.added.len(), 2);
        assert_eq!(evo.removed.len(), 1);
        assert_eq!(evo.total_changes(), 3);
    }

    #[test]
    fn test_evolution_diff_identical() {
        let s = state("s1", &["a", "b"]);
        let evo = ConstraintEvolution::diff(&s, &s);
        assert!(evo.is_empty());
    }

    #[test]
    fn test_timeline_unroll_empty() {
        let tl = ConstraintTimeline::new();
        let unrolled = tl.unroll(3);
        assert_eq!(unrolled.len(), 3);
        for cs in &unrolled {
            assert_eq!(cs.len(), 0);
        }
    }

    #[test]
    fn test_timeline_unroll_with_steps() {
        let mut tl = ConstraintTimeline::new();
        let mut cs0 = ConstraintSet::new();
        cs0.add(Constraint::hard("c1", ConstraintExpr::bool_const(true)));
        tl.add_step(0, cs0);

        let mut cs2 = ConstraintSet::new();
        cs2.add(Constraint::hard("c2", ConstraintExpr::bool_const(true)));
        cs2.add(Constraint::hard("c3", ConstraintExpr::bool_const(false)));
        tl.add_step(2, cs2);

        let unrolled = tl.unroll(4);
        assert_eq!(unrolled[0].len(), 1);
        assert_eq!(unrolled[1].len(), 1);
        assert_eq!(unrolled[2].len(), 2);
        assert_eq!(unrolled[3].len(), 2);
    }

    #[test]
    fn test_budget_constraints() {
        let mut tl = ConstraintTimeline::new();
        let cs = ConstraintSet::new();
        tl.add_step(0, cs.clone());
        tl.add_step(3, cs);

        let budgets = tl.compute_budget_constraints(100.0);
        assert_eq!(budgets.len(), 2);
        assert!(budgets[0].description.contains("timestep 0"));
        assert!(budgets[1].description.contains("timestep 3"));
    }

    #[test]
    fn test_detect_temporal_conflicts() {
        let mut tl = ConstraintTimeline::new();

        let mut cs0 = ConstraintSet::new();
        cs0.add(
            Constraint::hard("c1", ConstraintExpr::bool_const(true))
                .with_source("obl-A", "EU"),
        );
        cs0.add(
            Constraint::hard("c2", ConstraintExpr::bool_const(true))
                .with_source("obl-B", "EU"),
        );
        tl.add_step(0, cs0);

        let mut cs1 = ConstraintSet::new();
        cs1.add(
            Constraint::hard("c1", ConstraintExpr::bool_const(true))
                .with_source("obl-A", "EU"),
        );
        tl.add_step(1, cs1);

        let conflicts = tl.detect_temporal_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].contains("obl-B"));
    }

    #[test]
    fn test_detect_no_conflicts() {
        let mut tl = ConstraintTimeline::new();
        let mut cs = ConstraintSet::new();
        cs.add(
            Constraint::hard("c1", ConstraintExpr::bool_const(true))
                .with_source("obl-A", "EU"),
        );
        tl.add_step(0, cs.clone());
        tl.add_step(1, cs);

        let conflicts = tl.detect_temporal_conflicts();
        assert!(conflicts.is_empty());
    }
}
